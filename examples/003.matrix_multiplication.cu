#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_pipeline_primitives.h>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "util/error.h"
#include "util/util.cuh"

__global__ void naive_mm(const float* A, const float* B, float* C, int M, int N, int K)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  int m = blockIdx.y * blockDim.y + threadIdx.y;

  if (n >= N || m >= M) {
    return;
  }

  A += m * K;
  B += n;
  float sum = 0.0;
#pragma unroll
  for (int k = 0; k < K; ++k) {
    sum += A[k] * B[k * N];
  }
  C[m * N + n] = sum;
}
void launch_naive_mm(const float* A, const float* B, float* C, int M, int N, int K)
{
  dim3 threads_per_block(16, 16);
  dim3 blocks_per_grid((N + threads_per_block.x - 1) / threads_per_block.x,
                       (M + threads_per_block.y - 1) / threads_per_block.y);

  naive_mm<<<blocks_per_grid, threads_per_block>>>(A, B, C, M, N, K);
  CHECK_CUDA_ERROR();
}

template<int BLOCK_TILE_M, int BLOCK_TILE_N, int THREAD_TILE_M, int THREAD_TILE_N, int TILE_K>
__global__ void llmmm(const float* A, const float* B, float* C, int M, int N, int K)
{
  constexpr int THREAD_COUNT =
    device_thread_count_calculator<BLOCK_TILE_M, BLOCK_TILE_N, THREAD_TILE_M, THREAD_TILE_N>();

  constexpr int A_LDG_REG_COUNT = BLOCK_TILE_M * TILE_K / THREAD_COUNT;
  constexpr int B_LDG_REG_COUNT = BLOCK_TILE_N * TILE_K / THREAD_COUNT;
  // The following checks are to ensure that data can be loaded via float4.
  static_assert(BLOCK_TILE_M * TILE_K % THREAD_COUNT == 0);
  static_assert(BLOCK_TILE_N * TILE_K % THREAD_COUNT == 0);
  static_assert(A_LDG_REG_COUNT % 4 == 0);
  static_assert(B_LDG_REG_COUNT % 4 == 0);
  constexpr int A_LDG_LOOP_COUNT = A_LDG_REG_COUNT / 4;
  constexpr int B_LDG_LOOP_COUNT = B_LDG_REG_COUNT / 4;

  // For loading
  float A_ldg_reg[A_LDG_REG_COUNT];
  float B_ldg_reg[B_LDG_REG_COUNT];

  constexpr int    SM_BUFFER_COUNT = 1;
  __shared__ float A_sm[SM_BUFFER_COUNT][TILE_K][BLOCK_TILE_M];
  __shared__ float B_sm[SM_BUFFER_COUNT][TILE_K][BLOCK_TILE_N];

  // For computing
  float A_comp_reg[THREAD_TILE_M];
  float B_comp_reg[THREAD_TILE_N];
  float C_reg[THREAD_TILE_M][THREAD_TILE_N] = {0};
  // The following checks are to ensure that data can be loaded via float4.
  static_assert(THREAD_TILE_M % 4 == 0);
  static_assert(THREAD_TILE_N % 4 == 0);

  const int block_m_offset = blockIdx.y * BLOCK_TILE_M;
  const int block_n_offset = blockIdx.x * BLOCK_TILE_N;

  constexpr int CAL_THREADS_ON_M_AXIS = BLOCK_TILE_M / THREAD_TILE_M;
  constexpr int CAL_THREADS_ON_N_AXIS = BLOCK_TILE_N / THREAD_TILE_N;
  static_assert(CAL_THREADS_ON_M_AXIS * CAL_THREADS_ON_N_AXIS == THREAD_COUNT);
  constexpr int CAL_THREAD_M_LOAD_COUNT = THREAD_TILE_M / 4;
  constexpr int CAL_THREAD_N_LOAD_COUNT = THREAD_TILE_N / 4;
  const int     comp_thread_m_offset    = threadIdx.x / CAL_THREADS_ON_N_AXIS * THREAD_TILE_M;
  constexpr int CAL_THREAD_M_STRIDE     = 4;
  const int     comp_thread_n_offset    = threadIdx.x % CAL_THREADS_ON_N_AXIS * 4;
  constexpr int CAL_THREAD_N_STRIDE     = CAL_THREADS_ON_N_AXIS * 4;
  const int     k_iter_count            = K / TILE_K;
  for (int k_iter = 0; k_iter < k_iter_count; ++k_iter) {
    __syncthreads();
    const int k_iter_offset = k_iter * TILE_K;
    // Load A, global -> register
#pragma unroll
    for (int loop = 0; loop < A_LDG_LOOP_COUNT; ++loop) {
      const int m = (loop * THREAD_COUNT + threadIdx.x) * 4 / TILE_K;
      const int k = (loop * THREAD_COUNT + threadIdx.x) * 4 % TILE_K;
      FETCH_FLOAT4(A_ldg_reg[loop * 4], A[OFFSET(block_m_offset + m, k_iter_offset + k, K)]);
    }
    // Load B, global -> register
#pragma unroll
    for (int loop = 0; loop < B_LDG_LOOP_COUNT; ++loop) {
      const int k = (loop * THREAD_COUNT + threadIdx.x) * 4 / BLOCK_TILE_N;
      const int n = (loop * THREAD_COUNT + threadIdx.x) * 4 % BLOCK_TILE_N;
      FETCH_FLOAT4(B_ldg_reg[loop * 4], B[OFFSET(k_iter_offset + k, block_n_offset + n, N)]);
    }

    // Store A, register -> shared
#pragma unroll
    for (int loop = 0; loop < A_LDG_LOOP_COUNT; ++loop) {
      const int m       = (loop * THREAD_COUNT + threadIdx.x) * 4 / TILE_K;
      const int k       = (loop * THREAD_COUNT + threadIdx.x) * 4 % TILE_K;
      A_sm[0][k + 0][m] = A_ldg_reg[loop * 4 + 0];
      A_sm[0][k + 1][m] = A_ldg_reg[loop * 4 + 1];
      A_sm[0][k + 2][m] = A_ldg_reg[loop * 4 + 2];
      A_sm[0][k + 3][m] = A_ldg_reg[loop * 4 + 3];
    }
    // Store B, register -> shared
#pragma unroll
    for (int loop = 0; loop < B_LDG_LOOP_COUNT; ++loop) {
      const int k = (loop * THREAD_COUNT + threadIdx.x) * 4 / BLOCK_TILE_N;
      const int n = (loop * THREAD_COUNT + threadIdx.x) * 4 % BLOCK_TILE_N;
      STORE_FLOAT4(B_sm[0][k][n], B_ldg_reg[loop * 4]);
    }

    __syncthreads();

    // Computing
#pragma unroll
    for (int k = 0; k < TILE_K; ++k) {
      // Load A from shared to register
#pragma unroll
      for (int i = 0; i < CAL_THREAD_M_LOAD_COUNT; ++i) {
        FETCH_FLOAT4(A_comp_reg[i * 4], A_sm[0][k][comp_thread_m_offset + i * CAL_THREAD_M_STRIDE]);
      }
      // Load B from shared to register
#pragma unroll
      for (int i = 0; i < CAL_THREAD_N_LOAD_COUNT; ++i) {
        FETCH_FLOAT4(B_comp_reg[i * 4], B_sm[0][k][comp_thread_n_offset + i * CAL_THREAD_N_STRIDE]);
      }
#pragma unroll
      for (int i = 0; i < THREAD_TILE_M; ++i) {
#pragma unroll
        for (int j = 0; j < THREAD_TILE_N; ++j) {
          C_reg[i][j] += A_comp_reg[i] * B_comp_reg[j];
        }
      }
    }
  }

#pragma unroll
  for (int i = 0; i < THREAD_TILE_M; ++i) {
    const int m = block_m_offset + comp_thread_m_offset + i;
#pragma unroll
    for (int j = 0; j < THREAD_TILE_N; j += 4) {
      const int n = block_n_offset + comp_thread_n_offset + j / 4 * CAL_THREAD_N_STRIDE;
      STORE_FLOAT4(C[OFFSET(m, n, N)], C_reg[i][j]);
    }
  }
}

template<int BLOCK_TILE_M, int BLOCK_TILE_N, int THREAD_TILE_M, int THREAD_TILE_N, int TILE_K>
__global__ void llmmm__overlap_sm2reg(const float* A, const float* B, float* C, int M, int N, int K)
{
  constexpr int THREAD_COUNT =
    device_thread_count_calculator<BLOCK_TILE_M, BLOCK_TILE_N, THREAD_TILE_M, THREAD_TILE_N>();

  constexpr int A_LDG_REG_COUNT = BLOCK_TILE_M * TILE_K / THREAD_COUNT;
  constexpr int B_LDG_REG_COUNT = BLOCK_TILE_N * TILE_K / THREAD_COUNT;
  // The following checks are to ensure that data can be loaded via float4.
  static_assert(BLOCK_TILE_M * TILE_K % THREAD_COUNT == 0);
  static_assert(BLOCK_TILE_N * TILE_K % THREAD_COUNT == 0);
  static_assert(A_LDG_REG_COUNT % 4 == 0);
  static_assert(B_LDG_REG_COUNT % 4 == 0);
  constexpr int A_LDG_LOOP_COUNT = A_LDG_REG_COUNT / 4;
  constexpr int B_LDG_LOOP_COUNT = B_LDG_REG_COUNT / 4;

  // For loading
  float A_ldg_reg[A_LDG_REG_COUNT];
  float B_ldg_reg[B_LDG_REG_COUNT];

  constexpr int    SM_BUFFER_COUNT = 1;
  __shared__ float A_sm[SM_BUFFER_COUNT][TILE_K][BLOCK_TILE_M];
  __shared__ float B_sm[SM_BUFFER_COUNT][TILE_K][BLOCK_TILE_N];

  // For computing
  constexpr int CAL_REG_BUFFER_COUNT = 2;
  float         A_comp_reg[CAL_REG_BUFFER_COUNT][THREAD_TILE_M];
  float         B_comp_reg[CAL_REG_BUFFER_COUNT][THREAD_TILE_N];
  float         C_reg[THREAD_TILE_M][THREAD_TILE_N] = {0};
  // The following checks are to ensure that data can be loaded via float4.
  static_assert(THREAD_TILE_M % 4 == 0);
  static_assert(THREAD_TILE_N % 4 == 0);

  const int block_m_offset = blockIdx.y * BLOCK_TILE_M;
  const int block_n_offset = blockIdx.x * BLOCK_TILE_N;

  constexpr int CAL_THREADS_ON_M_AXIS = BLOCK_TILE_M / THREAD_TILE_M;
  constexpr int CAL_THREADS_ON_N_AXIS = BLOCK_TILE_N / THREAD_TILE_N;
  static_assert(CAL_THREADS_ON_M_AXIS * CAL_THREADS_ON_N_AXIS == THREAD_COUNT);
  constexpr int CAL_THREAD_M_LOAD_COUNT = THREAD_TILE_M / 4;
  constexpr int CAL_THREAD_N_LOAD_COUNT = THREAD_TILE_N / 4;
  const int     comp_thread_m_offset    = threadIdx.x / CAL_THREADS_ON_N_AXIS * THREAD_TILE_M;
  constexpr int CAL_THREAD_M_STRIDE     = 4;
  const int     comp_thread_n_offset    = threadIdx.x % CAL_THREADS_ON_N_AXIS * 4;
  constexpr int CAL_THREAD_N_STRIDE     = CAL_THREADS_ON_N_AXIS * 4;
  const int     k_iter_count            = K / TILE_K;
  for (int k_iter = 0; k_iter < k_iter_count; ++k_iter) {
    __syncthreads();
    const int k_iter_offset = k_iter * TILE_K;
    // Load A, global -> register
#pragma unroll
    for (int loop = 0; loop < A_LDG_LOOP_COUNT; ++loop) {
      const int m = (loop * THREAD_COUNT + threadIdx.x) * 4 / TILE_K;
      const int k = (loop * THREAD_COUNT + threadIdx.x) * 4 % TILE_K;
      FETCH_FLOAT4(A_ldg_reg[loop * 4], A[OFFSET(block_m_offset + m, k_iter_offset + k, K)]);
    }
    // Load B, global -> register
#pragma unroll
    for (int loop = 0; loop < B_LDG_LOOP_COUNT; ++loop) {
      const int k = (loop * THREAD_COUNT + threadIdx.x) * 4 / BLOCK_TILE_N;
      const int n = (loop * THREAD_COUNT + threadIdx.x) * 4 % BLOCK_TILE_N;
      FETCH_FLOAT4(B_ldg_reg[loop * 4], B[OFFSET(k_iter_offset + k, block_n_offset + n, N)]);
    }

    // Store A, register -> shared
#pragma unroll
    for (int loop = 0; loop < A_LDG_LOOP_COUNT; ++loop) {
      const int m       = (loop * THREAD_COUNT + threadIdx.x) * 4 / TILE_K;
      const int k       = (loop * THREAD_COUNT + threadIdx.x) * 4 % TILE_K;
      A_sm[0][k + 0][m] = A_ldg_reg[loop * 4 + 0];
      A_sm[0][k + 1][m] = A_ldg_reg[loop * 4 + 1];
      A_sm[0][k + 2][m] = A_ldg_reg[loop * 4 + 2];
      A_sm[0][k + 3][m] = A_ldg_reg[loop * 4 + 3];
    }
    // Store B, register -> shared
#pragma unroll
    for (int loop = 0; loop < B_LDG_LOOP_COUNT; ++loop) {
      const int k = (loop * THREAD_COUNT + threadIdx.x) * 4 / BLOCK_TILE_N;
      const int n = (loop * THREAD_COUNT + threadIdx.x) * 4 % BLOCK_TILE_N;
      STORE_FLOAT4(B_sm[0][k][n], B_ldg_reg[loop * 4]);
    }

    __syncthreads();

#pragma unroll
    for (int i = 0; i < CAL_THREAD_M_LOAD_COUNT; ++i) {
      FETCH_FLOAT4(A_comp_reg[0][i * 4], A_sm[0][0][comp_thread_m_offset + i * CAL_THREAD_M_STRIDE]);
    }
#pragma unroll
    for (int i = 0; i < CAL_THREAD_N_LOAD_COUNT; ++i) {
      FETCH_FLOAT4(B_comp_reg[0][i * 4], B_sm[0][0][comp_thread_n_offset + i * CAL_THREAD_N_STRIDE]);
    }

    // Computing
#pragma unroll
    for (int k = 0; k < TILE_K - 1; ++k) {
#pragma unroll
      for (int i = 0; i < CAL_THREAD_M_LOAD_COUNT; ++i) {
        FETCH_FLOAT4(A_comp_reg[(k + 1) & 1][i * 4], A_sm[0][k + 1][comp_thread_m_offset + i * CAL_THREAD_M_STRIDE]);
      }
#pragma unroll
      for (int i = 0; i < CAL_THREAD_N_LOAD_COUNT; ++i) {
        FETCH_FLOAT4(B_comp_reg[(k + 1) & 1][i * 4], B_sm[0][k + 1][comp_thread_n_offset + i * CAL_THREAD_N_STRIDE]);
      }
#pragma unroll
      for (int i = 0; i < THREAD_TILE_M; ++i) {
#pragma unroll
        for (int j = 0; j < THREAD_TILE_N; ++j) {
          C_reg[i][j] += A_comp_reg[k & 1][i] * B_comp_reg[k & 1][j];
        }
      }
    }
#pragma unroll
    for (int i = 0; i < THREAD_TILE_M; ++i) {
#pragma unroll
      for (int j = 0; j < THREAD_TILE_N; ++j) {
        C_reg[i][j] += A_comp_reg[1][i] * B_comp_reg[1][j];
      }
    }
  }

#pragma unroll
  for (int i = 0; i < THREAD_TILE_M; ++i) {
    const int m = block_m_offset + comp_thread_m_offset + i;
#pragma unroll
    for (int j = 0; j < THREAD_TILE_N; j += 4) {
      const int n = block_n_offset + comp_thread_n_offset + j / 4 * CAL_THREAD_N_STRIDE;
      STORE_FLOAT4(C[OFFSET(m, n, N)], C_reg[i][j]);
    }
  }
}

template<int BLOCK_TILE_M, int BLOCK_TILE_N, int THREAD_TILE_M, int THREAD_TILE_N, int TILE_K>
__global__ void llmmm__overlap_global2sm(const float* A, const float* B, float* C, int M, int N, int K)
{
  constexpr int THREAD_COUNT =
    device_thread_count_calculator<BLOCK_TILE_M, BLOCK_TILE_N, THREAD_TILE_M, THREAD_TILE_N>();

  constexpr int A_LDG_REG_COUNT = BLOCK_TILE_M * TILE_K / THREAD_COUNT;
  constexpr int B_LDG_REG_COUNT = BLOCK_TILE_N * TILE_K / THREAD_COUNT;
  // The following checks are to ensure that data can be loaded via float4.
  static_assert(BLOCK_TILE_M * TILE_K % THREAD_COUNT == 0);
  static_assert(BLOCK_TILE_N * TILE_K % THREAD_COUNT == 0);
  static_assert(A_LDG_REG_COUNT % 4 == 0);
  static_assert(B_LDG_REG_COUNT % 4 == 0);
  constexpr int A_LDG_LOOP_COUNT = A_LDG_REG_COUNT / 4;
  constexpr int B_LDG_LOOP_COUNT = B_LDG_REG_COUNT / 4;

  // For loading
  float A_ldg_reg[A_LDG_REG_COUNT];
  float B_ldg_reg[B_LDG_REG_COUNT];

  constexpr int    SM_BUFFER_COUNT = 2;
  __shared__ float A_sm[SM_BUFFER_COUNT][TILE_K][BLOCK_TILE_M];
  __shared__ float B_sm[SM_BUFFER_COUNT][TILE_K][BLOCK_TILE_N];

  // For computing
  float A_comp_reg[THREAD_TILE_M];
  float B_comp_reg[THREAD_TILE_N];
  float C_reg[THREAD_TILE_M][THREAD_TILE_N] = {0};
  // The following checks are to ensure that data can be loaded via float4.
  static_assert(THREAD_TILE_M % 4 == 0);
  static_assert(THREAD_TILE_N % 4 == 0);

  const int block_m_offset = blockIdx.y * BLOCK_TILE_M;
  const int block_n_offset = blockIdx.x * BLOCK_TILE_N;

  constexpr int CAL_THREADS_ON_M_AXIS = BLOCK_TILE_M / THREAD_TILE_M;
  constexpr int CAL_THREADS_ON_N_AXIS = BLOCK_TILE_N / THREAD_TILE_N;
  static_assert(CAL_THREADS_ON_M_AXIS * CAL_THREADS_ON_N_AXIS == THREAD_COUNT);
  constexpr int CAL_THREAD_M_LOAD_COUNT = THREAD_TILE_M / 4;
  constexpr int CAL_THREAD_N_LOAD_COUNT = THREAD_TILE_N / 4;
  const int     comp_thread_m_offset    = threadIdx.x / CAL_THREADS_ON_N_AXIS * THREAD_TILE_M;
  constexpr int CAL_THREAD_M_STRIDE     = 4;
  const int     comp_thread_n_offset    = threadIdx.x % CAL_THREADS_ON_N_AXIS * 4;
  constexpr int CAL_THREAD_N_STRIDE     = CAL_THREADS_ON_N_AXIS * 4;
  const int     k_iter_count            = K / TILE_K;
#define global_2_ldg_reg()                                                                                             \
  {                                                                                                                    \
    /* Load A, global -> register */                                                                                   \
    _Pragma("unroll") for (int loop = 0; loop < A_LDG_LOOP_COUNT; ++loop)                                              \
    {                                                                                                                  \
      const int m = (loop * THREAD_COUNT + threadIdx.x) * 4 / TILE_K;                                                  \
      const int k = (loop * THREAD_COUNT + threadIdx.x) * 4 % TILE_K;                                                  \
      FETCH_FLOAT4(A_ldg_reg[loop * 4], A[OFFSET(block_m_offset + m, k_iter_offset + k, K)]);                          \
    }                                                                                                                  \
    /* Load B, global->register */                                                                                     \
    _Pragma("unroll") for (int loop = 0; loop < B_LDG_LOOP_COUNT; ++loop)                                              \
    {                                                                                                                  \
      const int k = (loop * THREAD_COUNT + threadIdx.x) * 4 / BLOCK_TILE_N;                                            \
      const int n = (loop * THREAD_COUNT + threadIdx.x) * 4 % BLOCK_TILE_N;                                            \
      FETCH_FLOAT4(B_ldg_reg[loop * 4], B[OFFSET(k_iter_offset + k, block_n_offset + n, N)]);                          \
    }                                                                                                                  \
  }

#define ldg_reg_2_sm()                                                                                                 \
  /* Store A, register -> shared */                                                                                    \
  _Pragma("unroll") for (int loop = 0; loop < A_LDG_LOOP_COUNT; ++loop)                                                \
  {                                                                                                                    \
    const int m                       = (loop * THREAD_COUNT + threadIdx.x) * 4 / TILE_K;                              \
    const int k                       = (loop * THREAD_COUNT + threadIdx.x) * 4 % TILE_K;                              \
    A_sm[SM_LOADING_BUFFER][k + 0][m] = A_ldg_reg[loop * 4 + 0];                                                       \
    A_sm[SM_LOADING_BUFFER][k + 1][m] = A_ldg_reg[loop * 4 + 1];                                                       \
    A_sm[SM_LOADING_BUFFER][k + 2][m] = A_ldg_reg[loop * 4 + 2];                                                       \
    A_sm[SM_LOADING_BUFFER][k + 3][m] = A_ldg_reg[loop * 4 + 3];                                                       \
  }                                                                                                                    \
  /* Store B, register -> shared */                                                                                    \
  _Pragma("unroll") for (int loop = 0; loop < B_LDG_LOOP_COUNT; ++loop)                                                \
  {                                                                                                                    \
    const int k = (loop * THREAD_COUNT + threadIdx.x) * 4 / BLOCK_TILE_N;                                              \
    const int n = (loop * THREAD_COUNT + threadIdx.x) * 4 % BLOCK_TILE_N;                                              \
    STORE_FLOAT4(B_sm[SM_LOADING_BUFFER][k][n], B_ldg_reg[loop * 4]);                                                  \
  }

  {
    const int     k_iter_offset     = 0;
    constexpr int SM_LOADING_BUFFER = 0;
    global_2_ldg_reg();
    ldg_reg_2_sm();
  }

  for (int k_iter = 0; k_iter < k_iter_count - 1; ++k_iter) {
    const int SM_LOADING_BUFFER     = (k_iter + 1) & 1;
    const int SM_CALCULATING_BUFFER = k_iter & 1;
    /* if (k_iter + 1 < k_iter_count) */ {
      const int k_iter_offset = (k_iter + 1) * TILE_K;
      global_2_ldg_reg();
    }
    __syncthreads();

    // Computing
#pragma unroll
    for (int k = 0; k < TILE_K; ++k) {
      // Load A from shared to register
#pragma unroll
      for (int i = 0; i < CAL_THREAD_M_LOAD_COUNT; ++i) {
        FETCH_FLOAT4(A_comp_reg[i * 4], A_sm[SM_CALCULATING_BUFFER][k][comp_thread_m_offset + i * CAL_THREAD_M_STRIDE]);
      }
      // Load B from shared to register
#pragma unroll
      for (int i = 0; i < CAL_THREAD_N_LOAD_COUNT; ++i) {
        FETCH_FLOAT4(B_comp_reg[i * 4], B_sm[SM_CALCULATING_BUFFER][k][comp_thread_n_offset + i * CAL_THREAD_N_STRIDE]);
      }
#pragma unroll
      for (int i = 0; i < THREAD_TILE_M; ++i) {
#pragma unroll
        for (int j = 0; j < THREAD_TILE_N; ++j) {
          C_reg[i][j] += A_comp_reg[i] * B_comp_reg[j];
        }
      }
    }
    /* if (k_iter + 1 < k_iter_count) */ {
      ldg_reg_2_sm();
    }
  }

  __syncthreads();

  {
    const int k_iter                = k_iter_count - 1;
    const int SM_CALCULATING_BUFFER = k_iter & 1;

    // Computing
#pragma unroll
    for (int k = 0; k < TILE_K; ++k) {
      // Load A from shared to register
#pragma unroll
      for (int i = 0; i < CAL_THREAD_M_LOAD_COUNT; ++i) {
        FETCH_FLOAT4(A_comp_reg[i * 4], A_sm[SM_CALCULATING_BUFFER][k][comp_thread_m_offset + i * CAL_THREAD_M_STRIDE]);
      }
      // Load B from shared to register
#pragma unroll
      for (int i = 0; i < CAL_THREAD_N_LOAD_COUNT; ++i) {
        FETCH_FLOAT4(B_comp_reg[i * 4], B_sm[SM_CALCULATING_BUFFER][k][comp_thread_n_offset + i * CAL_THREAD_N_STRIDE]);
      }
#pragma unroll
      for (int i = 0; i < THREAD_TILE_M; ++i) {
#pragma unroll
        for (int j = 0; j < THREAD_TILE_N; ++j) {
          C_reg[i][j] += A_comp_reg[i] * B_comp_reg[j];
        }
      }
    }
  }

#pragma unroll
  for (int i = 0; i < THREAD_TILE_M; ++i) {
    const int m = block_m_offset + comp_thread_m_offset + i;
#pragma unroll
    for (int j = 0; j < THREAD_TILE_N; j += 4) {
      const int n = block_n_offset + comp_thread_n_offset + j / 4 * CAL_THREAD_N_STRIDE;
      STORE_FLOAT4(C[OFFSET(m, n, N)], C_reg[i][j]);
    }
  }
}

template<int BLOCK_TILE_M, int BLOCK_TILE_N, int THREAD_TILE_M, int THREAD_TILE_N, int TILE_K>
__global__ void llmmm__overlap_global2sm2reg__using_if(const float* A, const float* B, float* C, int M, int N, int K)
{
  constexpr int THREAD_COUNT =
    device_thread_count_calculator<BLOCK_TILE_M, BLOCK_TILE_N, THREAD_TILE_M, THREAD_TILE_N>();

  constexpr int A_LDG_REG_COUNT = BLOCK_TILE_M * TILE_K / THREAD_COUNT;
  constexpr int B_LDG_REG_COUNT = BLOCK_TILE_N * TILE_K / THREAD_COUNT;
  // The following checks are to ensure that data can be loaded via float4.
  static_assert(BLOCK_TILE_M * TILE_K % THREAD_COUNT == 0);
  static_assert(BLOCK_TILE_N * TILE_K % THREAD_COUNT == 0);
  static_assert(A_LDG_REG_COUNT % 4 == 0);
  static_assert(B_LDG_REG_COUNT % 4 == 0);
  constexpr int A_LDG_LOOP_COUNT = A_LDG_REG_COUNT / 4;
  constexpr int B_LDG_LOOP_COUNT = B_LDG_REG_COUNT / 4;

  // For loading
  float A_ldg_reg[A_LDG_REG_COUNT];
  float B_ldg_reg[B_LDG_REG_COUNT];

  constexpr int    SM_BUFFER_COUNT = 2;
  __shared__ float A_sm[SM_BUFFER_COUNT][TILE_K][BLOCK_TILE_M];
  __shared__ float B_sm[SM_BUFFER_COUNT][TILE_K][BLOCK_TILE_N];

  // For computing
  constexpr int CAL_REG_BUFFER_COUNT = 2;
  float         A_comp_reg[CAL_REG_BUFFER_COUNT][THREAD_TILE_M];
  float         B_comp_reg[CAL_REG_BUFFER_COUNT][THREAD_TILE_N];
  float         C_reg[THREAD_TILE_M][THREAD_TILE_N] = {0};
  // The following checks are to ensure that data can be loaded via float4.
  static_assert(THREAD_TILE_M % 4 == 0);
  static_assert(THREAD_TILE_N % 4 == 0);

  const int block_m_offset = blockIdx.y * BLOCK_TILE_M;
  const int block_n_offset = blockIdx.x * BLOCK_TILE_N;

  constexpr int CAL_THREADS_ON_M_AXIS = BLOCK_TILE_M / THREAD_TILE_M;
  constexpr int CAL_THREADS_ON_N_AXIS = BLOCK_TILE_N / THREAD_TILE_N;
  static_assert(CAL_THREADS_ON_M_AXIS * CAL_THREADS_ON_N_AXIS == THREAD_COUNT);
  constexpr int CAL_THREAD_M_LOAD_COUNT = THREAD_TILE_M / 4;
  constexpr int CAL_THREAD_N_LOAD_COUNT = THREAD_TILE_N / 4;
  const int     comp_thread_m_offset    = threadIdx.x / CAL_THREADS_ON_N_AXIS * THREAD_TILE_M;
  constexpr int CAL_THREAD_M_STRIDE     = 4;
  const int     comp_thread_n_offset    = threadIdx.x % CAL_THREADS_ON_N_AXIS * 4;
  constexpr int CAL_THREAD_N_STRIDE     = CAL_THREADS_ON_N_AXIS * 4;
  const int     k_iter_count            = K / TILE_K;
#define global_2_ldg_reg()                                                                                             \
  {                                                                                                                    \
    /* Load A, global -> register */                                                                                   \
    _Pragma("unroll") for (int loop = 0; loop < A_LDG_LOOP_COUNT; ++loop)                                              \
    {                                                                                                                  \
      const int m = (loop * THREAD_COUNT + threadIdx.x) * 4 / TILE_K;                                                  \
      const int k = (loop * THREAD_COUNT + threadIdx.x) * 4 % TILE_K;                                                  \
      FETCH_FLOAT4(A_ldg_reg[loop * 4], A[OFFSET(block_m_offset + m, k_iter_offset + k, K)]);                          \
    }                                                                                                                  \
    /* Load B, global->register */                                                                                     \
    _Pragma("unroll") for (int loop = 0; loop < B_LDG_LOOP_COUNT; ++loop)                                              \
    {                                                                                                                  \
      const int k = (loop * THREAD_COUNT + threadIdx.x) * 4 / BLOCK_TILE_N;                                            \
      const int n = (loop * THREAD_COUNT + threadIdx.x) * 4 % BLOCK_TILE_N;                                            \
      FETCH_FLOAT4(B_ldg_reg[loop * 4], B[OFFSET(k_iter_offset + k, block_n_offset + n, N)]);                          \
    }                                                                                                                  \
  }

#define ldg_reg_2_sm()                                                                                                 \
  /* Store A, register -> shared */                                                                                    \
  _Pragma("unroll") for (int loop = 0; loop < A_LDG_LOOP_COUNT; ++loop)                                                \
  {                                                                                                                    \
    const int m                       = (loop * THREAD_COUNT + threadIdx.x) * 4 / TILE_K;                              \
    const int k                       = (loop * THREAD_COUNT + threadIdx.x) * 4 % TILE_K;                              \
    A_sm[SM_LOADING_BUFFER][k + 0][m] = A_ldg_reg[loop * 4 + 0];                                                       \
    A_sm[SM_LOADING_BUFFER][k + 1][m] = A_ldg_reg[loop * 4 + 1];                                                       \
    A_sm[SM_LOADING_BUFFER][k + 2][m] = A_ldg_reg[loop * 4 + 2];                                                       \
    A_sm[SM_LOADING_BUFFER][k + 3][m] = A_ldg_reg[loop * 4 + 3];                                                       \
  }                                                                                                                    \
  /* Store B, register -> shared */                                                                                    \
  _Pragma("unroll") for (int loop = 0; loop < B_LDG_LOOP_COUNT; ++loop)                                                \
  {                                                                                                                    \
    const int k = (loop * THREAD_COUNT + threadIdx.x) * 4 / BLOCK_TILE_N;                                              \
    const int n = (loop * THREAD_COUNT + threadIdx.x) * 4 % BLOCK_TILE_N;                                              \
    STORE_FLOAT4(B_sm[SM_LOADING_BUFFER][k][n], B_ldg_reg[loop * 4]);                                                  \
  }

#define sm_2_comp_reg()                                                                                                \
  _Pragma("unroll") for (int i = 0; i < CAL_THREAD_M_LOAD_COUNT; ++i)                                                  \
  {                                                                                                                    \
    FETCH_FLOAT4(A_comp_reg[CAL_REG_LOADING_BUFFER][i * 4],                                                            \
                 A_sm[SM_CALCULATING_BUFFER][k][comp_thread_m_offset + i * CAL_THREAD_M_STRIDE]);                      \
  }                                                                                                                    \
  _Pragma("unroll") for (int i = 0; i < CAL_THREAD_N_LOAD_COUNT; ++i)                                                  \
  {                                                                                                                    \
    FETCH_FLOAT4(B_comp_reg[CAL_REG_LOADING_BUFFER][i * 4],                                                            \
                 B_sm[SM_CALCULATING_BUFFER][k][comp_thread_n_offset + i * CAL_THREAD_N_STRIDE]);                      \
  }

#define compute()                                                                                                      \
  _Pragma("unroll") for (int i = 0; i < THREAD_TILE_M; ++i)                                                            \
  {                                                                                                                    \
    _Pragma("unroll") for (int j = 0; j < THREAD_TILE_N; ++j)                                                          \
    {                                                                                                                  \
      C_reg[i][j] += A_comp_reg[CAL_REG_CALCULATING_BUFFER][i] * B_comp_reg[CAL_REG_CALCULATING_BUFFER][j];            \
    }                                                                                                                  \
  }

  {
    const int     k_iter_offset     = 0;
    constexpr int SM_LOADING_BUFFER = 0;
    global_2_ldg_reg();
    ldg_reg_2_sm();
  }

  __syncthreads();
  {
    constexpr int k_iter_offset = TILE_K;
    if (k_iter_offset < K) {
      global_2_ldg_reg();
    }
  }

  {
    constexpr int k                      = 0;
    constexpr int CAL_REG_LOADING_BUFFER = 0;
    constexpr int SM_CALCULATING_BUFFER  = 0;
    sm_2_comp_reg();
  }

  int SM_LOADING_BUFFER     = 1;
  int SM_CALCULATING_BUFFER = 0;
  for (int k_iter = 0; k_iter < k_iter_count; ++k_iter) {
    // Computing
#pragma unroll
    for (int k = 1; k < TILE_K; ++k) {
      const int CAL_REG_LOADING_BUFFER = k & 1;
      sm_2_comp_reg();
      const int CAL_REG_CALCULATING_BUFFER = (k + 1) & 1;
      compute()
    }
    if (k_iter + 1 < k_iter_count) {
      ldg_reg_2_sm();
      __syncthreads();
      if (k_iter + 2 < k_iter_count) {
        const int k_iter_offset = (k_iter + 2) * TILE_K;
        global_2_ldg_reg();
      }
      SM_CALCULATING_BUFFER ^= 1;
      SM_LOADING_BUFFER ^= 1;
      {
        constexpr int k                      = 0;
        constexpr int CAL_REG_LOADING_BUFFER = 0;
        sm_2_comp_reg();
      }
    }
    {
      constexpr int CAL_REG_CALCULATING_BUFFER = 1;
      compute();
    }
  }

#pragma unroll
  for (int i = 0; i < THREAD_TILE_M; ++i) {
    const int m = block_m_offset + comp_thread_m_offset + i;
#pragma unroll
    for (int j = 0; j < THREAD_TILE_N; j += 4) {
      const int n = block_n_offset + comp_thread_n_offset + j / 4 * CAL_THREAD_N_STRIDE;
      STORE_FLOAT4(C[OFFSET(m, n, N)], C_reg[i][j]);
    }
  }
#undef global_2_ldg_reg
#undef ldg_reg_2_sm
#undef sm_2_comp_reg
#undef compute
}

template<int BLOCK_TILE_M, int BLOCK_TILE_N, int THREAD_TILE_M, int THREAD_TILE_N, int TILE_K>
__global__ void llmmm__overlap_global2sm2reg(const float* A, const float* B, float* C, int M, int N, int K)
{
  constexpr int THREAD_COUNT =
    device_thread_count_calculator<BLOCK_TILE_M, BLOCK_TILE_N, THREAD_TILE_M, THREAD_TILE_N>();

  constexpr int A_LDG_REG_COUNT = BLOCK_TILE_M * TILE_K / THREAD_COUNT;
  constexpr int B_LDG_REG_COUNT = BLOCK_TILE_N * TILE_K / THREAD_COUNT;
  // The following checks are to ensure that data can be loaded via float4.
  static_assert(BLOCK_TILE_M * TILE_K % THREAD_COUNT == 0);
  static_assert(BLOCK_TILE_N * TILE_K % THREAD_COUNT == 0);
  static_assert(A_LDG_REG_COUNT % 4 == 0);
  static_assert(B_LDG_REG_COUNT % 4 == 0);
  constexpr int A_LDG_LOOP_COUNT = A_LDG_REG_COUNT / 4;
  constexpr int B_LDG_LOOP_COUNT = B_LDG_REG_COUNT / 4;

  // For loading
  float A_ldg_reg[A_LDG_REG_COUNT];
  float B_ldg_reg[B_LDG_REG_COUNT];

  constexpr int    SM_BUFFER_COUNT = 2;
  __shared__ float A_sm[SM_BUFFER_COUNT][TILE_K][BLOCK_TILE_M];
  __shared__ float B_sm[SM_BUFFER_COUNT][TILE_K][BLOCK_TILE_N];

  // For computing
  constexpr int CAL_REG_BUFFER_COUNT = 2;
  float         A_comp_reg[CAL_REG_BUFFER_COUNT][THREAD_TILE_M];
  float         B_comp_reg[CAL_REG_BUFFER_COUNT][THREAD_TILE_N];
  float         C_reg[THREAD_TILE_M][THREAD_TILE_N] = {0};
  // The following checks are to ensure that data can be loaded via float4.
  static_assert(THREAD_TILE_M % 4 == 0);
  static_assert(THREAD_TILE_N % 4 == 0);

  const int block_m_offset = blockIdx.y * BLOCK_TILE_M;
  const int block_n_offset = blockIdx.x * BLOCK_TILE_N;

  constexpr int CAL_THREADS_ON_M_AXIS = BLOCK_TILE_M / THREAD_TILE_M;
  constexpr int CAL_THREADS_ON_N_AXIS = BLOCK_TILE_N / THREAD_TILE_N;
  static_assert(CAL_THREADS_ON_M_AXIS * CAL_THREADS_ON_N_AXIS == THREAD_COUNT);
  constexpr int CAL_THREAD_M_LOAD_COUNT = THREAD_TILE_M / 4;
  constexpr int CAL_THREAD_N_LOAD_COUNT = THREAD_TILE_N / 4;
  const int     comp_thread_m_offset    = threadIdx.x / CAL_THREADS_ON_N_AXIS * THREAD_TILE_M;
  constexpr int CAL_THREAD_M_STRIDE     = 4;
  const int     comp_thread_n_offset    = threadIdx.x % CAL_THREADS_ON_N_AXIS * 4;
  constexpr int CAL_THREAD_N_STRIDE     = CAL_THREADS_ON_N_AXIS * 4;
  const int     k_iter_count            = K / TILE_K;
#define global_2_ldg_reg()                                                                                             \
  {                                                                                                                    \
    /* Load A, global -> register */                                                                                   \
    _Pragma("unroll") for (int loop = 0; loop < A_LDG_LOOP_COUNT; ++loop)                                              \
    {                                                                                                                  \
      const int m = (loop * THREAD_COUNT + threadIdx.x) * 4 / TILE_K;                                                  \
      const int k = (loop * THREAD_COUNT + threadIdx.x) * 4 % TILE_K;                                                  \
      FETCH_FLOAT4(A_ldg_reg[loop * 4], A[OFFSET(block_m_offset + m, k_iter_offset + k, K)]);                          \
    }                                                                                                                  \
    /* Load B, global->register */                                                                                     \
    _Pragma("unroll") for (int loop = 0; loop < B_LDG_LOOP_COUNT; ++loop)                                              \
    {                                                                                                                  \
      const int k = (loop * THREAD_COUNT + threadIdx.x) * 4 / BLOCK_TILE_N;                                            \
      const int n = (loop * THREAD_COUNT + threadIdx.x) * 4 % BLOCK_TILE_N;                                            \
      FETCH_FLOAT4(B_ldg_reg[loop * 4], B[OFFSET(k_iter_offset + k, block_n_offset + n, N)]);                          \
    }                                                                                                                  \
  }

#define ldg_reg_2_sm()                                                                                                 \
  /* Store A, register -> shared */                                                                                    \
  _Pragma("unroll") for (int loop = 0; loop < A_LDG_LOOP_COUNT; ++loop)                                                \
  {                                                                                                                    \
    const int m                       = (loop * THREAD_COUNT + threadIdx.x) * 4 / TILE_K;                              \
    const int k                       = (loop * THREAD_COUNT + threadIdx.x) * 4 % TILE_K;                              \
    A_sm[SM_LOADING_BUFFER][k + 0][m] = A_ldg_reg[loop * 4 + 0];                                                       \
    A_sm[SM_LOADING_BUFFER][k + 1][m] = A_ldg_reg[loop * 4 + 1];                                                       \
    A_sm[SM_LOADING_BUFFER][k + 2][m] = A_ldg_reg[loop * 4 + 2];                                                       \
    A_sm[SM_LOADING_BUFFER][k + 3][m] = A_ldg_reg[loop * 4 + 3];                                                       \
  }                                                                                                                    \
  /* Store B, register -> shared */                                                                                    \
  _Pragma("unroll") for (int loop = 0; loop < B_LDG_LOOP_COUNT; ++loop)                                                \
  {                                                                                                                    \
    const int k = (loop * THREAD_COUNT + threadIdx.x) * 4 / BLOCK_TILE_N;                                              \
    const int n = (loop * THREAD_COUNT + threadIdx.x) * 4 % BLOCK_TILE_N;                                              \
    STORE_FLOAT4(B_sm[SM_LOADING_BUFFER][k][n], B_ldg_reg[loop * 4]);                                                  \
  }

#define sm_2_comp_reg()                                                                                                \
  _Pragma("unroll") for (int i = 0; i < CAL_THREAD_M_LOAD_COUNT; ++i)                                                  \
  {                                                                                                                    \
    FETCH_FLOAT4(A_comp_reg[CAL_REG_LOADING_BUFFER][i * 4],                                                            \
                 A_sm[SM_CALCULATING_BUFFER][k][comp_thread_m_offset + i * CAL_THREAD_M_STRIDE]);                      \
  }                                                                                                                    \
  _Pragma("unroll") for (int i = 0; i < CAL_THREAD_N_LOAD_COUNT; ++i)                                                  \
  {                                                                                                                    \
    FETCH_FLOAT4(B_comp_reg[CAL_REG_LOADING_BUFFER][i * 4],                                                            \
                 B_sm[SM_CALCULATING_BUFFER][k][comp_thread_n_offset + i * CAL_THREAD_N_STRIDE]);                      \
  }

#define compute()                                                                                                      \
  _Pragma("unroll") for (int i = 0; i < THREAD_TILE_M; ++i)                                                            \
  {                                                                                                                    \
    _Pragma("unroll") for (int j = 0; j < THREAD_TILE_N; ++j)                                                          \
    {                                                                                                                  \
      C_reg[i][j] += A_comp_reg[CAL_REG_CALCULATING_BUFFER][i] * B_comp_reg[CAL_REG_CALCULATING_BUFFER][j];            \
    }                                                                                                                  \
  }

  {
    const int     k_iter_offset     = 0;
    constexpr int SM_LOADING_BUFFER = 0;
    global_2_ldg_reg();
    ldg_reg_2_sm();
  }

  __syncthreads();
  {
    constexpr int k_iter_offset = TILE_K;
    if (k_iter_offset < K) {
      global_2_ldg_reg();
    }
  }

  {
    constexpr int k                      = 0;
    constexpr int CAL_REG_LOADING_BUFFER = 0;
    constexpr int SM_CALCULATING_BUFFER  = 0;
    sm_2_comp_reg();
  }

  int SM_LOADING_BUFFER     = 1;
  int SM_CALCULATING_BUFFER = 0;
  for (int k_iter = 0; k_iter < k_iter_count - 2; ++k_iter) {
    // Computing
#pragma unroll
    for (int k = 1; k < TILE_K; ++k) {
      const int CAL_REG_LOADING_BUFFER = k & 1;
      sm_2_comp_reg();
      const int CAL_REG_CALCULATING_BUFFER = (k + 1) & 1;
      compute()
    }
    /* if (k_iter + 1 < k_iter_count) */ {
      ldg_reg_2_sm();
      __syncthreads();
      /* if (k_iter + 2 < k_iter_count) */ {
        const int k_iter_offset = (k_iter + 2) * TILE_K;
        global_2_ldg_reg();
      }
      SM_CALCULATING_BUFFER ^= 1;
      SM_LOADING_BUFFER ^= 1;
      {
        constexpr int k                      = 0;
        constexpr int CAL_REG_LOADING_BUFFER = 0;
        sm_2_comp_reg();
      }
    }
    {
      constexpr int CAL_REG_CALCULATING_BUFFER = 1;
      compute();
    }
  }
  {
    /* const int k_iter = k_iter_count - 2; */
    // Computing
#pragma unroll
    for (int k = 1; k < TILE_K; ++k) {
      const int CAL_REG_LOADING_BUFFER = k & 1;
      sm_2_comp_reg();
      const int CAL_REG_CALCULATING_BUFFER = (k + 1) & 1;
      compute()
    }
    /* if (k_iter + 1 < k_iter_count) */ {
      ldg_reg_2_sm();
      __syncthreads();
      SM_CALCULATING_BUFFER ^= 1;
      SM_LOADING_BUFFER ^= 1;
      {
        constexpr int k                      = 0;
        constexpr int CAL_REG_LOADING_BUFFER = 0;
        sm_2_comp_reg();
      }
    }
    {
      constexpr int CAL_REG_CALCULATING_BUFFER = 1;
      compute();
    }
  }

  {
    /* const int k_iter = k_iter_count - 1; */
    // Computing
#pragma unroll
    for (int k = 1; k < TILE_K; ++k) {
      const int CAL_REG_LOADING_BUFFER = k & 1;
      sm_2_comp_reg();
      const int CAL_REG_CALCULATING_BUFFER = (k + 1) & 1;
      compute()
    }
    {
      constexpr int CAL_REG_CALCULATING_BUFFER = 1;
      compute();
    }
  }

#pragma unroll
  for (int i = 0; i < THREAD_TILE_M; ++i) {
    const int m = block_m_offset + comp_thread_m_offset + i;
#pragma unroll
    for (int j = 0; j < THREAD_TILE_N; j += 4) {
      const int n = block_n_offset + comp_thread_n_offset + j / 4 * CAL_THREAD_N_STRIDE;
      STORE_FLOAT4(C[OFFSET(m, n, N)], C_reg[i][j]);
    }
  }
#undef global_2_ldg_reg
#undef ldg_reg_2_sm
#undef sm_2_comp_reg
#undef compute
}

template<int BLOCK_TILE_M, int BLOCK_TILE_N, int THREAD_TILE_M, int THREAD_TILE_N, int TILE_K>
__global__ void
llmmm__overlap_global2sm2reg__quardra_buffer(const float* A, const float* B, float* C, int M, int N, int K)
{
  constexpr int THREAD_COUNT =
    device_thread_count_calculator<BLOCK_TILE_M, BLOCK_TILE_N, THREAD_TILE_M, THREAD_TILE_N>();

  constexpr int A_LDG_REG_COUNT = BLOCK_TILE_M * TILE_K / THREAD_COUNT;
  constexpr int B_LDG_REG_COUNT = BLOCK_TILE_N * TILE_K / THREAD_COUNT;
  // The following checks are to ensure that data can be loaded via float4.
  static_assert(BLOCK_TILE_M * TILE_K % THREAD_COUNT == 0);
  static_assert(BLOCK_TILE_N * TILE_K % THREAD_COUNT == 0);
  static_assert(A_LDG_REG_COUNT % 4 == 0);
  static_assert(B_LDG_REG_COUNT % 4 == 0);
  constexpr int A_LDG_LOOP_COUNT = A_LDG_REG_COUNT / 4;
  constexpr int B_LDG_LOOP_COUNT = B_LDG_REG_COUNT / 4;

  // For loading.
  float A_ldg_reg[A_LDG_REG_COUNT];
  float B_ldg_reg[B_LDG_REG_COUNT];

  // computing, pending, reg2sm ing, global2reg ing
  constexpr int    SM_BUFFER_COUNT = 4;
  __shared__ float A_sm[SM_BUFFER_COUNT][TILE_K][BLOCK_TILE_M];
  __shared__ float B_sm[SM_BUFFER_COUNT][TILE_K][BLOCK_TILE_N];

  // For computing. sm_2_comp_reg and computing.
  constexpr int CAL_REG_BUFFER_COUNT = 2;
  float         A_comp_reg[CAL_REG_BUFFER_COUNT][THREAD_TILE_M];
  float         B_comp_reg[CAL_REG_BUFFER_COUNT][THREAD_TILE_N];
  float         C_reg[THREAD_TILE_M][THREAD_TILE_N] = {0};
  // The following checks are to ensure that data can be loaded via float4.
  static_assert(THREAD_TILE_M % 4 == 0);
  static_assert(THREAD_TILE_N % 4 == 0);

  const int block_m_offset = blockIdx.y * BLOCK_TILE_M;
  const int block_n_offset = blockIdx.x * BLOCK_TILE_N;

  constexpr int CAL_THREADS_ON_M_AXIS = BLOCK_TILE_M / THREAD_TILE_M;
  constexpr int CAL_THREADS_ON_N_AXIS = BLOCK_TILE_N / THREAD_TILE_N;
  static_assert(CAL_THREADS_ON_M_AXIS * CAL_THREADS_ON_N_AXIS == THREAD_COUNT);
  constexpr int CAL_THREAD_M_LOAD_COUNT = THREAD_TILE_M / 4;
  constexpr int CAL_THREAD_N_LOAD_COUNT = THREAD_TILE_N / 4;
  const int     comp_thread_m_offset    = threadIdx.x / CAL_THREADS_ON_N_AXIS * THREAD_TILE_M;
  constexpr int CAL_THREAD_M_STRIDE     = 4;
  const int     comp_thread_n_offset    = threadIdx.x % CAL_THREADS_ON_N_AXIS * 4;
  constexpr int CAL_THREAD_N_STRIDE     = CAL_THREADS_ON_N_AXIS * 4;
  const int     k_iter_count            = K / TILE_K;
#define global_2_ldg_reg()                                                                                             \
  {                                                                                                                    \
    /* Load A, global -> register */                                                                                   \
    _Pragma("unroll") for (int loop = 0; loop < A_LDG_LOOP_COUNT; ++loop)                                              \
    {                                                                                                                  \
      const int m = (loop * THREAD_COUNT + threadIdx.x) * 4 / TILE_K;                                                  \
      const int k = (loop * THREAD_COUNT + threadIdx.x) * 4 % TILE_K;                                                  \
      FETCH_FLOAT4(A_ldg_reg[loop * 4], A[OFFSET(block_m_offset + m, k_iter_offset + k, K)]);                          \
    }                                                                                                                  \
    /* Load B, global->register */                                                                                     \
    _Pragma("unroll") for (int loop = 0; loop < B_LDG_LOOP_COUNT; ++loop)                                              \
    {                                                                                                                  \
      const int k = (loop * THREAD_COUNT + threadIdx.x) * 4 / BLOCK_TILE_N;                                            \
      const int n = (loop * THREAD_COUNT + threadIdx.x) * 4 % BLOCK_TILE_N;                                            \
      FETCH_FLOAT4(B_ldg_reg[loop * 4], B[OFFSET(k_iter_offset + k, block_n_offset + n, N)]);                          \
    }                                                                                                                  \
  }

#define ldg_reg_2_sm()                                                                                                 \
  /* Store A, register -> shared */                                                                                    \
  _Pragma("unroll") for (int loop = 0; loop < A_LDG_LOOP_COUNT; ++loop)                                                \
  {                                                                                                                    \
    const int m               = (loop * THREAD_COUNT + threadIdx.x) * 4 / TILE_K;                                      \
    const int k               = (loop * THREAD_COUNT + threadIdx.x) * 4 % TILE_K;                                      \
    A_sm[SM_REG2SM][k + 0][m] = A_ldg_reg[loop * 4 + 0];                                                               \
    A_sm[SM_REG2SM][k + 1][m] = A_ldg_reg[loop * 4 + 1];                                                               \
    A_sm[SM_REG2SM][k + 2][m] = A_ldg_reg[loop * 4 + 2];                                                               \
    A_sm[SM_REG2SM][k + 3][m] = A_ldg_reg[loop * 4 + 3];                                                               \
  }                                                                                                                    \
  /* Store B, register -> shared */                                                                                    \
  _Pragma("unroll") for (int loop = 0; loop < B_LDG_LOOP_COUNT; ++loop)                                                \
  {                                                                                                                    \
    const int k = (loop * THREAD_COUNT + threadIdx.x) * 4 / BLOCK_TILE_N;                                              \
    const int n = (loop * THREAD_COUNT + threadIdx.x) * 4 % BLOCK_TILE_N;                                              \
    STORE_FLOAT4(B_sm[SM_REG2SM][k][n], B_ldg_reg[loop * 4]);                                                          \
  }

#define sm_2_comp_reg()                                                                                                \
  _Pragma("unroll") for (int i = 0; i < CAL_THREAD_M_LOAD_COUNT; ++i)                                                  \
  {                                                                                                                    \
    FETCH_FLOAT4(A_comp_reg[CAL_REG_SM2REG][i * 4],                                                                    \
                 A_sm[SM_COMPUTING][k][comp_thread_m_offset + i * CAL_THREAD_M_STRIDE]);                               \
  }                                                                                                                    \
  _Pragma("unroll") for (int i = 0; i < CAL_THREAD_N_LOAD_COUNT; ++i)                                                  \
  {                                                                                                                    \
    FETCH_FLOAT4(B_comp_reg[CAL_REG_SM2REG][i * 4],                                                                    \
                 B_sm[SM_COMPUTING][k][comp_thread_n_offset + i * CAL_THREAD_N_STRIDE]);                               \
  }

#define compute()                                                                                                      \
  _Pragma("unroll") for (int i = 0; i < THREAD_TILE_M; ++i)                                                            \
  {                                                                                                                    \
    _Pragma("unroll") for (int j = 0; j < THREAD_TILE_N; ++j)                                                          \
    {                                                                                                                  \
      C_reg[i][j] += A_comp_reg[CAL_REG_COMPUTING][i] * B_comp_reg[CAL_REG_COMPUTING][j];                              \
    }                                                                                                                  \
  }

  {
    // Load the first tile in k direction from global to register.
    constexpr int k_iter_offset = 0;
    global_2_ldg_reg();
  }

  {
    // Store the first tile in k direction from register to shared.
    constexpr int SM_REG2SM = 0;
    ldg_reg_2_sm();
  }

  {
    // Load the second tile in k direction from global to register.
    constexpr int k_iter_offset = TILE_K;
    global_2_ldg_reg();
  }

  __syncthreads();

  {
    // Load the first line in the first tile from shared to computing register.
    constexpr int k              = 0;
    constexpr int SM_COMPUTING   = 0;
    constexpr int CAL_REG_SM2REG = 0;
    sm_2_comp_reg();
  }

  {
    // Store the second tile in k direction from register to shared.
    constexpr int SM_REG2SM = 1;
    ldg_reg_2_sm();
  }

  {
    // Load the third tile in k direction from global to register.
    constexpr int k_iter_offset = TILE_K * 2;
    global_2_ldg_reg();
  }

  int SM_COMPUTING = 0;
  int SM_REG2SM    = 2;
  for (int k_iter = 0; k_iter < k_iter_count - 3; ++k_iter) {
    // Computing
#pragma unroll
    for (int k = 1; k < TILE_K; ++k) {
      const int CAL_REG_SM2REG = k & 1;
      sm_2_comp_reg();
      const int CAL_REG_COMPUTING = (k - 1) & 1;
      compute();
    }
    /* if (k_iter + 1 < k_iter_count)*/ {
      /* if (k_iter + 2 < k_iter_count) */ {
        ldg_reg_2_sm();
        __syncthreads();
      }
      SM_COMPUTING = (SM_COMPUTING + 1) % 4;
      SM_REG2SM    = (SM_REG2SM + 1) % 4;
      {
        constexpr int k              = 0;
        constexpr int CAL_REG_SM2REG = 0;
        sm_2_comp_reg();
      }
      /* if (k_iter + 3 < k_iter_count) */ {
        const int k_iter_offset = (k_iter + 3) * TILE_K;
        global_2_ldg_reg();
      }
    }
    {
      constexpr int CAL_REG_COMPUTING = 1;
      compute();
    }
  }
  {
    /* const int k_iter = k_iter_count - 3; */
    // Computing
#pragma unroll
    for (int k = 1; k < TILE_K; ++k) {
      const int CAL_REG_SM2REG = k & 1;
      sm_2_comp_reg();
      const int CAL_REG_COMPUTING = (k - 1) & 1;
      compute();
    }
    /* if (k_iter + 1 < k_iter_count)*/ {
      /* if (k_iter + 2 < k_iter_count) */ {
        ldg_reg_2_sm();
        __syncthreads();
      }
      SM_COMPUTING = (SM_COMPUTING + 1) % 4;
      SM_REG2SM    = (SM_REG2SM + 1) % 4;
      {
        constexpr int k              = 0;
        constexpr int CAL_REG_SM2REG = 0;
        sm_2_comp_reg();
      }
    }
    {
      constexpr int CAL_REG_COMPUTING = 1;
      compute();
    }
  }
  {
    /* const int k_iter = k_iter_count - 2; */
    // Computing
#pragma unroll
    for (int k = 1; k < TILE_K; ++k) {
      const int CAL_REG_SM2REG = k & 1;
      sm_2_comp_reg();
      const int CAL_REG_COMPUTING = (k - 1) & 1;
      compute();
    }
    /* if (k_iter + 1 < k_iter_count)*/ {
      SM_COMPUTING = (SM_COMPUTING + 1) % 4;
      SM_REG2SM    = (SM_REG2SM + 1) % 4;
      {
        constexpr int k              = 0;
        constexpr int CAL_REG_SM2REG = 0;
        sm_2_comp_reg();
      }
    }
    {
      constexpr int CAL_REG_COMPUTING = 1;
      compute();
    }
  }
  {
    /* const int k_iter = k_iter_count - 1; */
    // Computing
#pragma unroll
    for (int k = 1; k < TILE_K; ++k) {
      const int CAL_REG_SM2REG = k & 1;
      sm_2_comp_reg();
      const int CAL_REG_COMPUTING = (k - 1) & 1;
      compute();
    }
    {
      constexpr int CAL_REG_COMPUTING = 1;
      compute();
    }
  }

#pragma unroll
  for (int i = 0; i < THREAD_TILE_M; ++i) {
    const int m = block_m_offset + comp_thread_m_offset + i;
#pragma unroll
    for (int j = 0; j < THREAD_TILE_N; j += 4) {
      const int n = block_n_offset + comp_thread_n_offset + j / 4 * CAL_THREAD_N_STRIDE;
      STORE_FLOAT4(C[OFFSET(m, n, N)], C_reg[i][j]);
    }
  }
#undef global_2_ldg_reg
#undef ldg_reg_2_sm
#undef sm_2_comp_reg
#undef compute
}

template<int BLOCK_TILE_M, int BLOCK_TILE_N, int THREAD_TILE_M, int THREAD_TILE_N, int TILE_K>
__global__ void llmmm__overlap_global2sm2reg__quardra_buffer__double_ldg_reg(
  const float* A, const float* B, float* C, int M, int N, int K)
{
  constexpr int THREAD_COUNT =
    device_thread_count_calculator<BLOCK_TILE_M, BLOCK_TILE_N, THREAD_TILE_M, THREAD_TILE_N>();

  constexpr int A_LDG_REG_COUNT = BLOCK_TILE_M * TILE_K / THREAD_COUNT;
  constexpr int B_LDG_REG_COUNT = BLOCK_TILE_N * TILE_K / THREAD_COUNT;
  // The following checks are to ensure that data can be loaded via float4.
  static_assert(BLOCK_TILE_M * TILE_K % THREAD_COUNT == 0);
  static_assert(BLOCK_TILE_N * TILE_K % THREAD_COUNT == 0);
  static_assert(A_LDG_REG_COUNT % 4 == 0);
  static_assert(B_LDG_REG_COUNT % 4 == 0);
  constexpr int A_LDG_LOOP_COUNT = A_LDG_REG_COUNT / 4;
  constexpr int B_LDG_LOOP_COUNT = B_LDG_REG_COUNT / 4;

  // For loading.
  constexpr int LDG_REG_BUFFER_COUNT = 2;
  constexpr int LDG_REG_BUFFER_0     = 0;
  constexpr int LDG_REG_BUFFER_1     = 1;
  float         A_ldg_reg[LDG_REG_BUFFER_COUNT][A_LDG_REG_COUNT];
  float         B_ldg_reg[LDG_REG_BUFFER_COUNT][B_LDG_REG_COUNT];

  // computing, pending, reg2sm ing, global2reg ing
  constexpr int    SM_BUFFER_COUNT = 4;
  __shared__ float A_sm[SM_BUFFER_COUNT][TILE_K][BLOCK_TILE_M];
  __shared__ float B_sm[SM_BUFFER_COUNT][TILE_K][BLOCK_TILE_N];

  // For computing. sm_2_comp_reg and computing.
  constexpr int CAL_REG_BUFFER_COUNT = 2;
  float         A_comp_reg[CAL_REG_BUFFER_COUNT][THREAD_TILE_M];
  float         B_comp_reg[CAL_REG_BUFFER_COUNT][THREAD_TILE_N];
  float         C_reg[THREAD_TILE_M][THREAD_TILE_N] = {0};
  // The following checks are to ensure that data can be loaded via float4.
  static_assert(THREAD_TILE_M % 4 == 0);
  static_assert(THREAD_TILE_N % 4 == 0);

  const int block_m_offset = blockIdx.y * BLOCK_TILE_M;
  const int block_n_offset = blockIdx.x * BLOCK_TILE_N;

  constexpr int CAL_THREADS_ON_M_AXIS = BLOCK_TILE_M / THREAD_TILE_M;
  constexpr int CAL_THREADS_ON_N_AXIS = BLOCK_TILE_N / THREAD_TILE_N;
  static_assert(CAL_THREADS_ON_M_AXIS * CAL_THREADS_ON_N_AXIS == THREAD_COUNT);
  constexpr int CAL_THREAD_M_LOAD_COUNT = THREAD_TILE_M / 4;
  constexpr int CAL_THREAD_N_LOAD_COUNT = THREAD_TILE_N / 4;
  const int     comp_thread_m_offset    = threadIdx.x / CAL_THREADS_ON_N_AXIS * THREAD_TILE_M;
  constexpr int CAL_THREAD_M_STRIDE     = 4;
  const int     comp_thread_n_offset    = threadIdx.x % CAL_THREADS_ON_N_AXIS * 4;
  constexpr int CAL_THREAD_N_STRIDE     = CAL_THREADS_ON_N_AXIS * 4;
  const int     k_iter_count            = K / TILE_K;
#define global_2_ldg_reg(LDG_REG_INDEX)                                                                                \
  {                                                                                                                    \
    /* Load A, global -> register */                                                                                   \
    _Pragma("unroll") for (int loop = 0; loop < A_LDG_LOOP_COUNT; ++loop)                                              \
    {                                                                                                                  \
      const int m = (loop * THREAD_COUNT + threadIdx.x) * 4 / TILE_K;                                                  \
      const int k = (loop * THREAD_COUNT + threadIdx.x) * 4 % TILE_K;                                                  \
      FETCH_FLOAT4(A_ldg_reg[LDG_REG_INDEX][loop * 4], A[OFFSET(block_m_offset + m, k_iter_offset + k, K)]);           \
    }                                                                                                                  \
    /* Load B, global->register */                                                                                     \
    _Pragma("unroll") for (int loop = 0; loop < B_LDG_LOOP_COUNT; ++loop)                                              \
    {                                                                                                                  \
      const int k = (loop * THREAD_COUNT + threadIdx.x) * 4 / BLOCK_TILE_N;                                            \
      const int n = (loop * THREAD_COUNT + threadIdx.x) * 4 % BLOCK_TILE_N;                                            \
      FETCH_FLOAT4(B_ldg_reg[LDG_REG_INDEX][loop * 4], B[OFFSET(k_iter_offset + k, block_n_offset + n, N)]);           \
    }                                                                                                                  \
  }

#define ldg_reg_2_sm(LDG_REG_INDEX)                                                                                    \
  /* Store A, register -> shared */                                                                                    \
  _Pragma("unroll") for (int loop = 0; loop < A_LDG_LOOP_COUNT; ++loop)                                                \
  {                                                                                                                    \
    const int m               = (loop * THREAD_COUNT + threadIdx.x) * 4 / TILE_K;                                      \
    const int k               = (loop * THREAD_COUNT + threadIdx.x) * 4 % TILE_K;                                      \
    A_sm[SM_REG2SM][k + 0][m] = A_ldg_reg[LDG_REG_INDEX][loop * 4 + 0];                                                \
    A_sm[SM_REG2SM][k + 1][m] = A_ldg_reg[LDG_REG_INDEX][loop * 4 + 1];                                                \
    A_sm[SM_REG2SM][k + 2][m] = A_ldg_reg[LDG_REG_INDEX][loop * 4 + 2];                                                \
    A_sm[SM_REG2SM][k + 3][m] = A_ldg_reg[LDG_REG_INDEX][loop * 4 + 3];                                                \
  }                                                                                                                    \
  /* Store B, register -> shared */                                                                                    \
  _Pragma("unroll") for (int loop = 0; loop < B_LDG_LOOP_COUNT; ++loop)                                                \
  {                                                                                                                    \
    const int k = (loop * THREAD_COUNT + threadIdx.x) * 4 / BLOCK_TILE_N;                                              \
    const int n = (loop * THREAD_COUNT + threadIdx.x) * 4 % BLOCK_TILE_N;                                              \
    STORE_FLOAT4(B_sm[SM_REG2SM][k][n], B_ldg_reg[LDG_REG_INDEX][loop * 4]);                                           \
  }

#define sm_2_comp_reg()                                                                                                \
  _Pragma("unroll") for (int i = 0; i < CAL_THREAD_M_LOAD_COUNT; ++i)                                                  \
  {                                                                                                                    \
    FETCH_FLOAT4(A_comp_reg[CAL_REG_SM2REG][i * 4],                                                                    \
                 A_sm[SM_COMPUTING][k][comp_thread_m_offset + i * CAL_THREAD_M_STRIDE]);                               \
  }                                                                                                                    \
  _Pragma("unroll") for (int i = 0; i < CAL_THREAD_N_LOAD_COUNT; ++i)                                                  \
  {                                                                                                                    \
    FETCH_FLOAT4(B_comp_reg[CAL_REG_SM2REG][i * 4],                                                                    \
                 B_sm[SM_COMPUTING][k][comp_thread_n_offset + i * CAL_THREAD_N_STRIDE]);                               \
  }

#define compute()                                                                                                      \
  _Pragma("unroll") for (int i = 0; i < THREAD_TILE_M; ++i)                                                            \
  {                                                                                                                    \
    _Pragma("unroll") for (int j = 0; j < THREAD_TILE_N; ++j)                                                          \
    {                                                                                                                  \
      C_reg[i][j] += A_comp_reg[CAL_REG_COMPUTING][i] * B_comp_reg[CAL_REG_COMPUTING][j];                              \
    }                                                                                                                  \
  }

  {
    // Load the first tile in k direction from global to register.
    constexpr int k_iter_offset = 0;
    global_2_ldg_reg(LDG_REG_BUFFER_0);
  }

  {
    // Load the second tile in k direction from global to register.
    constexpr int k_iter_offset = TILE_K;
    global_2_ldg_reg(LDG_REG_BUFFER_1);
  }

  {
    // Store the first tile in k direction from register to shared.
    constexpr int SM_REG2SM = 0;
    ldg_reg_2_sm(LDG_REG_BUFFER_0);
  }

  {
    // Store the second tile in k direction from register to shared.
    constexpr int SM_REG2SM = 1;
    ldg_reg_2_sm(LDG_REG_BUFFER_1);
  }

  __syncthreads();

  {
    // Load the first line in the first tile from shared to computing register.
    constexpr int k              = 0;
    constexpr int SM_COMPUTING   = 0;
    constexpr int CAL_REG_SM2REG = 0;
    sm_2_comp_reg();
  }

  {
    // Load the third tile in k direction from global to register.
    constexpr int k_iter_offset = TILE_K * 2;
    global_2_ldg_reg(LDG_REG_BUFFER_0);
  }

  {
    // Load the third tile in k direction from global to register.
    constexpr int k_iter_offset = TILE_K * 3;
    global_2_ldg_reg(LDG_REG_BUFFER_1);
  }

  int SM_COMPUTING_OFFSET = 0;
  int SM_REG2SM_OFFSET    = 2;
  // k_iter_count must be an even number.
  for (int k_iter = 0; k_iter < k_iter_count - 4; k_iter += 2) {
#pragma unroll
    for (int k = 1; k < TILE_K; ++k) {
      const int CAL_REG_SM2REG = k & 1;
      const int SM_COMPUTING   = SM_COMPUTING_OFFSET;
      sm_2_comp_reg();
      const int CAL_REG_COMPUTING = (k - 1) & 1;
      compute();
    }
    {
      constexpr int k              = 0;
      constexpr int CAL_REG_SM2REG = 0;
      const int     SM_COMPUTING   = SM_COMPUTING_OFFSET + 1;
      sm_2_comp_reg();
    }
    {
      const int SM_REG2SM = SM_REG2SM_OFFSET;
      ldg_reg_2_sm(LDG_REG_BUFFER_0);
    }
    {
      const int SM_REG2SM = SM_REG2SM_OFFSET + 1;
      ldg_reg_2_sm(LDG_REG_BUFFER_1);
    }
    {
      const int k_iter_offset = (k_iter + 4) * TILE_K;
      global_2_ldg_reg(LDG_REG_BUFFER_0);
    }
    {
      SM_REG2SM_OFFSET ^= 2;
      const int k_iter_offset = (k_iter + 5) * TILE_K;
      global_2_ldg_reg(LDG_REG_BUFFER_1);
    }
    {
      constexpr int CAL_REG_COMPUTING = 1;
      compute();
    }
#pragma unroll
    for (int k = 1; k < TILE_K; ++k) {
      const int CAL_REG_SM2REG = k & 1;
      const int SM_COMPUTING   = SM_COMPUTING_OFFSET + 1;
      sm_2_comp_reg();
      const int CAL_REG_COMPUTING = (k - 1) & 1;
      compute();
    }
    __syncthreads();
    SM_COMPUTING_OFFSET ^= 2;
    {
      constexpr int k              = 0;
      constexpr int CAL_REG_SM2REG = 0;
      const int     SM_COMPUTING   = SM_COMPUTING_OFFSET;
      sm_2_comp_reg();
    }
    {
      constexpr int CAL_REG_COMPUTING = 1;
      compute();
    }
  }
  {
    // k_iter == k_iter_count - 4
#pragma unroll
    for (int k = 1; k < TILE_K; ++k) {
      const int CAL_REG_SM2REG = k & 1;
      const int SM_COMPUTING   = SM_COMPUTING_OFFSET;
      sm_2_comp_reg();
      const int CAL_REG_COMPUTING = (k - 1) & 1;
      compute();
    }
    {
      const int SM_REG2SM = SM_REG2SM_OFFSET;
      ldg_reg_2_sm(LDG_REG_BUFFER_0);
    }
    {
      const int SM_REG2SM = SM_REG2SM_OFFSET + 1;
      ldg_reg_2_sm(LDG_REG_BUFFER_1);
      SM_REG2SM_OFFSET ^= 2;
    }
    {
      constexpr int k              = 0;
      constexpr int CAL_REG_SM2REG = 0;
      const int     SM_COMPUTING   = SM_COMPUTING_OFFSET + 1;
      sm_2_comp_reg();
    }
    {
      constexpr int CAL_REG_COMPUTING = 1;
      compute();
    }
#pragma unroll
    for (int k = 1; k < TILE_K; ++k) {
      const int CAL_REG_SM2REG = k & 1;
      const int SM_COMPUTING   = SM_COMPUTING_OFFSET + 1;
      sm_2_comp_reg();
      const int CAL_REG_COMPUTING = (k - 1) & 1;
      compute();
    }
    __syncthreads();
    SM_COMPUTING_OFFSET ^= 2;
    {
      constexpr int k              = 0;
      constexpr int CAL_REG_SM2REG = 0;
      const int     SM_COMPUTING   = SM_COMPUTING_OFFSET;
      sm_2_comp_reg();
    }
    {
      constexpr int CAL_REG_COMPUTING = 1;
      compute();
    }
  }

  {
    // k_iter == k_iter_count - 2
#pragma unroll
    for (int k = 1; k < TILE_K; ++k) {
      const int CAL_REG_SM2REG = k & 1;
      const int SM_COMPUTING   = SM_COMPUTING_OFFSET;
      sm_2_comp_reg();
      const int CAL_REG_COMPUTING = (k - 1) & 1;
      compute();
    }
    {
      constexpr int k              = 0;
      constexpr int CAL_REG_SM2REG = 0;
      const int     SM_COMPUTING   = SM_COMPUTING_OFFSET + 1;
      sm_2_comp_reg();
    }
    {
      constexpr int CAL_REG_COMPUTING = 1;
      compute();
    }
#pragma unroll
    for (int k = 1; k < TILE_K; ++k) {
      const int CAL_REG_SM2REG = k & 1;
      const int SM_COMPUTING   = SM_COMPUTING_OFFSET + 1;
      sm_2_comp_reg();
      const int CAL_REG_COMPUTING = (k - 1) & 1;
      compute();
    }
    {
      constexpr int CAL_REG_COMPUTING = 1;
      compute();
    }
  }

#pragma unroll
  for (int i = 0; i < THREAD_TILE_M; ++i) {
    const int m = block_m_offset + comp_thread_m_offset + i;
#pragma unroll
    for (int j = 0; j < THREAD_TILE_N; j += 4) {
      const int n = block_n_offset + comp_thread_n_offset + j / 4 * CAL_THREAD_N_STRIDE;
      STORE_FLOAT4(C[OFFSET(m, n, N)], C_reg[i][j]);
    }
  }
#undef global_2_ldg_reg
#undef ldg_reg_2_sm
#undef sm_2_comp_reg
#undef compute
}

template<int BLOCK_TILE_M, int BLOCK_TILE_N, int THREAD_TILE_M, int THREAD_TILE_N, int TILE_K>
__global__ void llmmm__overlap_global2sm2reg__quardra_buffer__double_ldg_reg__st_global_wt(
  const float* A, const float* B, float* C, int M, int N, int K)
{
  constexpr int THREAD_COUNT =
    device_thread_count_calculator<BLOCK_TILE_M, BLOCK_TILE_N, THREAD_TILE_M, THREAD_TILE_N>();

  constexpr int A_LDG_REG_COUNT = BLOCK_TILE_M * TILE_K / THREAD_COUNT;
  constexpr int B_LDG_REG_COUNT = BLOCK_TILE_N * TILE_K / THREAD_COUNT;
  // The following checks are to ensure that data can be loaded via float4.
  static_assert(BLOCK_TILE_M * TILE_K % THREAD_COUNT == 0);
  static_assert(BLOCK_TILE_N * TILE_K % THREAD_COUNT == 0);
  static_assert(A_LDG_REG_COUNT % 4 == 0);
  static_assert(B_LDG_REG_COUNT % 4 == 0);
  constexpr int A_LDG_LOOP_COUNT = A_LDG_REG_COUNT / 4;
  constexpr int B_LDG_LOOP_COUNT = B_LDG_REG_COUNT / 4;

  // For loading.
  constexpr int LDG_REG_BUFFER_COUNT = 2;
  constexpr int LDG_REG_BUFFER_0     = 0;
  constexpr int LDG_REG_BUFFER_1     = 1;
  float         A_ldg_reg[LDG_REG_BUFFER_COUNT][A_LDG_REG_COUNT];
  float         B_ldg_reg[LDG_REG_BUFFER_COUNT][B_LDG_REG_COUNT];

  // computing, pending, reg2sm ing, global2reg ing
  constexpr int    SM_BUFFER_COUNT = 4;
  __shared__ float A_sm[SM_BUFFER_COUNT][TILE_K][BLOCK_TILE_M];
  __shared__ float B_sm[SM_BUFFER_COUNT][TILE_K][BLOCK_TILE_N];

  // For computing. sm_2_comp_reg and computing.
  constexpr int CAL_REG_BUFFER_COUNT = 2;
  float         A_comp_reg[CAL_REG_BUFFER_COUNT][THREAD_TILE_M];
  float         B_comp_reg[CAL_REG_BUFFER_COUNT][THREAD_TILE_N];
  float         C_reg[THREAD_TILE_M][THREAD_TILE_N] = {0};
  // The following checks are to ensure that data can be loaded via float4.
  static_assert(THREAD_TILE_M % 4 == 0);
  static_assert(THREAD_TILE_N % 4 == 0);

  const int block_m_offset = blockIdx.y * BLOCK_TILE_M;
  const int block_n_offset = blockIdx.x * BLOCK_TILE_N;

  constexpr int CAL_THREADS_ON_M_AXIS = BLOCK_TILE_M / THREAD_TILE_M;
  constexpr int CAL_THREADS_ON_N_AXIS = BLOCK_TILE_N / THREAD_TILE_N;
  static_assert(CAL_THREADS_ON_M_AXIS * CAL_THREADS_ON_N_AXIS == THREAD_COUNT);
  constexpr int CAL_THREAD_M_LOAD_COUNT = THREAD_TILE_M / 4;
  constexpr int CAL_THREAD_N_LOAD_COUNT = THREAD_TILE_N / 4;
  const int     comp_thread_m_offset    = threadIdx.x / CAL_THREADS_ON_N_AXIS * THREAD_TILE_M;
  constexpr int CAL_THREAD_M_STRIDE     = 4;
  const int     comp_thread_n_offset    = threadIdx.x % CAL_THREADS_ON_N_AXIS * 4;
  constexpr int CAL_THREAD_N_STRIDE     = CAL_THREADS_ON_N_AXIS * 4;
  const int     k_iter_count            = K / TILE_K;
#define global_2_ldg_reg(LDG_REG_INDEX)                                                                                \
  {                                                                                                                    \
    /* Load A, global -> register */                                                                                   \
    _Pragma("unroll") for (int loop = 0; loop < A_LDG_LOOP_COUNT; ++loop)                                              \
    {                                                                                                                  \
      const int m = (loop * THREAD_COUNT + threadIdx.x) * 4 / TILE_K;                                                  \
      const int k = (loop * THREAD_COUNT + threadIdx.x) * 4 % TILE_K;                                                  \
      FETCH_FLOAT4(A_ldg_reg[LDG_REG_INDEX][loop * 4], A[OFFSET(block_m_offset + m, k_iter_offset + k, K)]);           \
    }                                                                                                                  \
    /* Load B, global->register */                                                                                     \
    _Pragma("unroll") for (int loop = 0; loop < B_LDG_LOOP_COUNT; ++loop)                                              \
    {                                                                                                                  \
      const int k = (loop * THREAD_COUNT + threadIdx.x) * 4 / BLOCK_TILE_N;                                            \
      const int n = (loop * THREAD_COUNT + threadIdx.x) * 4 % BLOCK_TILE_N;                                            \
      FETCH_FLOAT4(B_ldg_reg[LDG_REG_INDEX][loop * 4], B[OFFSET(k_iter_offset + k, block_n_offset + n, N)]);           \
    }                                                                                                                  \
  }

#define ldg_reg_2_sm(LDG_REG_INDEX)                                                                                    \
  /* Store A, register -> shared */                                                                                    \
  _Pragma("unroll") for (int loop = 0; loop < A_LDG_LOOP_COUNT; ++loop)                                                \
  {                                                                                                                    \
    const int m               = (loop * THREAD_COUNT + threadIdx.x) * 4 / TILE_K;                                      \
    const int k               = (loop * THREAD_COUNT + threadIdx.x) * 4 % TILE_K;                                      \
    A_sm[SM_REG2SM][k + 0][m] = A_ldg_reg[LDG_REG_INDEX][loop * 4 + 0];                                                \
    A_sm[SM_REG2SM][k + 1][m] = A_ldg_reg[LDG_REG_INDEX][loop * 4 + 1];                                                \
    A_sm[SM_REG2SM][k + 2][m] = A_ldg_reg[LDG_REG_INDEX][loop * 4 + 2];                                                \
    A_sm[SM_REG2SM][k + 3][m] = A_ldg_reg[LDG_REG_INDEX][loop * 4 + 3];                                                \
  }                                                                                                                    \
  /* Store B, register -> shared */                                                                                    \
  _Pragma("unroll") for (int loop = 0; loop < B_LDG_LOOP_COUNT; ++loop)                                                \
  {                                                                                                                    \
    const int k = (loop * THREAD_COUNT + threadIdx.x) * 4 / BLOCK_TILE_N;                                              \
    const int n = (loop * THREAD_COUNT + threadIdx.x) * 4 % BLOCK_TILE_N;                                              \
    STORE_FLOAT4(B_sm[SM_REG2SM][k][n], B_ldg_reg[LDG_REG_INDEX][loop * 4]);                                           \
  }

#define sm_2_comp_reg()                                                                                                \
  _Pragma("unroll") for (int i = 0; i < CAL_THREAD_M_LOAD_COUNT; ++i)                                                  \
  {                                                                                                                    \
    FETCH_FLOAT4(A_comp_reg[CAL_REG_SM2REG][i * 4],                                                                    \
                 A_sm[SM_COMPUTING][k][comp_thread_m_offset + i * CAL_THREAD_M_STRIDE]);                               \
  }                                                                                                                    \
  _Pragma("unroll") for (int i = 0; i < CAL_THREAD_N_LOAD_COUNT; ++i)                                                  \
  {                                                                                                                    \
    FETCH_FLOAT4(B_comp_reg[CAL_REG_SM2REG][i * 4],                                                                    \
                 B_sm[SM_COMPUTING][k][comp_thread_n_offset + i * CAL_THREAD_N_STRIDE]);                               \
  }

#define compute()                                                                                                      \
  _Pragma("unroll") for (int i = 0; i < THREAD_TILE_M; ++i)                                                            \
  {                                                                                                                    \
    _Pragma("unroll") for (int j = 0; j < THREAD_TILE_N; ++j)                                                          \
    {                                                                                                                  \
      C_reg[i][j] += A_comp_reg[CAL_REG_COMPUTING][i] * B_comp_reg[CAL_REG_COMPUTING][j];                              \
    }                                                                                                                  \
  }

  {
    // Load the first tile in k direction from global to register.
    constexpr int k_iter_offset = 0;
    global_2_ldg_reg(LDG_REG_BUFFER_0);
  }

  {
    // Load the second tile in k direction from global to register.
    constexpr int k_iter_offset = TILE_K;
    global_2_ldg_reg(LDG_REG_BUFFER_1);
  }

  {
    // Store the first tile in k direction from register to shared.
    constexpr int SM_REG2SM = 0;
    ldg_reg_2_sm(LDG_REG_BUFFER_0);
  }

  {
    // Store the second tile in k direction from register to shared.
    constexpr int SM_REG2SM = 1;
    ldg_reg_2_sm(LDG_REG_BUFFER_1);
  }

  __syncthreads();

  {
    // Load the first line in the first tile from shared to computing register.
    constexpr int k              = 0;
    constexpr int SM_COMPUTING   = 0;
    constexpr int CAL_REG_SM2REG = 0;
    sm_2_comp_reg();
  }

  {
    // Load the third tile in k direction from global to register.
    constexpr int k_iter_offset = TILE_K * 2;
    global_2_ldg_reg(LDG_REG_BUFFER_0);
  }

  {
    // Load the third tile in k direction from global to register.
    constexpr int k_iter_offset = TILE_K * 3;
    global_2_ldg_reg(LDG_REG_BUFFER_1);
  }

  int SM_COMPUTING_OFFSET = 0;
  int SM_REG2SM_OFFSET    = 2;
  // k_iter_count must be an even number.
  for (int k_iter = 0; k_iter < k_iter_count - 4; k_iter += 2) {
#pragma unroll
    for (int k = 1; k < TILE_K; ++k) {
      const int CAL_REG_SM2REG = k & 1;
      const int SM_COMPUTING   = SM_COMPUTING_OFFSET;
      sm_2_comp_reg();
      const int CAL_REG_COMPUTING = (k - 1) & 1;
      compute();
    }
    {
      constexpr int k              = 0;
      constexpr int CAL_REG_SM2REG = 0;
      const int     SM_COMPUTING   = SM_COMPUTING_OFFSET + 1;
      sm_2_comp_reg();
    }
    {
      const int SM_REG2SM = SM_REG2SM_OFFSET;
      ldg_reg_2_sm(LDG_REG_BUFFER_0);
    }
    {
      const int SM_REG2SM = SM_REG2SM_OFFSET + 1;
      ldg_reg_2_sm(LDG_REG_BUFFER_1);
    }
    {
      const int k_iter_offset = (k_iter + 4) * TILE_K;
      global_2_ldg_reg(LDG_REG_BUFFER_0);
    }
    {
      SM_REG2SM_OFFSET ^= 2;
      const int k_iter_offset = (k_iter + 5) * TILE_K;
      global_2_ldg_reg(LDG_REG_BUFFER_1);
    }
    {
      constexpr int CAL_REG_COMPUTING = 1;
      compute();
    }
#pragma unroll
    for (int k = 1; k < TILE_K; ++k) {
      const int CAL_REG_SM2REG = k & 1;
      const int SM_COMPUTING   = SM_COMPUTING_OFFSET + 1;
      sm_2_comp_reg();
      const int CAL_REG_COMPUTING = (k - 1) & 1;
      compute();
    }
    __syncthreads();
    SM_COMPUTING_OFFSET ^= 2;
    {
      constexpr int k              = 0;
      constexpr int CAL_REG_SM2REG = 0;
      const int     SM_COMPUTING   = SM_COMPUTING_OFFSET;
      sm_2_comp_reg();
    }
    {
      constexpr int CAL_REG_COMPUTING = 1;
      compute();
    }
  }
  {
    // k_iter == k_iter_count - 4
#pragma unroll
    for (int k = 1; k < TILE_K; ++k) {
      const int CAL_REG_SM2REG = k & 1;
      const int SM_COMPUTING   = SM_COMPUTING_OFFSET;
      sm_2_comp_reg();
      const int CAL_REG_COMPUTING = (k - 1) & 1;
      compute();
    }
    {
      const int SM_REG2SM = SM_REG2SM_OFFSET;
      ldg_reg_2_sm(LDG_REG_BUFFER_0);
    }
    {
      const int SM_REG2SM = SM_REG2SM_OFFSET + 1;
      ldg_reg_2_sm(LDG_REG_BUFFER_1);
      SM_REG2SM_OFFSET ^= 2;
    }
    {
      constexpr int k              = 0;
      constexpr int CAL_REG_SM2REG = 0;
      const int     SM_COMPUTING   = SM_COMPUTING_OFFSET + 1;
      sm_2_comp_reg();
    }
    {
      constexpr int CAL_REG_COMPUTING = 1;
      compute();
    }
#pragma unroll
    for (int k = 1; k < TILE_K; ++k) {
      const int CAL_REG_SM2REG = k & 1;
      const int SM_COMPUTING   = SM_COMPUTING_OFFSET + 1;
      sm_2_comp_reg();
      const int CAL_REG_COMPUTING = (k - 1) & 1;
      compute();
    }
    __syncthreads();
    SM_COMPUTING_OFFSET ^= 2;
    {
      constexpr int k              = 0;
      constexpr int CAL_REG_SM2REG = 0;
      const int     SM_COMPUTING   = SM_COMPUTING_OFFSET;
      sm_2_comp_reg();
    }
    {
      constexpr int CAL_REG_COMPUTING = 1;
      compute();
    }
  }

  {
    // k_iter == k_iter_count - 2
#pragma unroll
    for (int k = 1; k < TILE_K; ++k) {
      const int CAL_REG_SM2REG = k & 1;
      const int SM_COMPUTING   = SM_COMPUTING_OFFSET;
      sm_2_comp_reg();
      const int CAL_REG_COMPUTING = (k - 1) & 1;
      compute();
    }
    {
      constexpr int k              = 0;
      constexpr int CAL_REG_SM2REG = 0;
      const int     SM_COMPUTING   = SM_COMPUTING_OFFSET + 1;
      sm_2_comp_reg();
    }
    {
      constexpr int CAL_REG_COMPUTING = 1;
      compute();
    }
#pragma unroll
    for (int k = 1; k < TILE_K; ++k) {
      const int CAL_REG_SM2REG = k & 1;
      const int SM_COMPUTING   = SM_COMPUTING_OFFSET + 1;
      sm_2_comp_reg();
      const int CAL_REG_COMPUTING = (k - 1) & 1;
      compute();
    }
    {
      constexpr int CAL_REG_COMPUTING = 1;
      compute();
    }
  }

#pragma unroll
  for (int i = 0; i < THREAD_TILE_M; ++i) {
    const int m = block_m_offset + comp_thread_m_offset + i;
#pragma unroll
    for (int j = 0; j < THREAD_TILE_N; j += 4) {
      const int n = block_n_offset + comp_thread_n_offset + j / 4 * CAL_THREAD_N_STRIDE;
      asm volatile(
        "st.global.wt.v4.f32 [%0], {%1, %2, %3, %4};"
        :
        : "l"(&C[OFFSET(m, n, N)]), "f"(C_reg[i][j]), "f"(C_reg[i][j + 1]), "f"(C_reg[i][j + 2]), "f"(C_reg[i][j + 3])
        : "memory");
    }
  }
#undef global_2_ldg_reg
#undef ldg_reg_2_sm
#undef sm_2_comp_reg
#undef compute
}

template<int BLOCK_TILE_M, int BLOCK_TILE_N, int THREAD_TILE_M, int THREAD_TILE_N, int TILE_K>
__global__ void llmmm__overlap_global2sm2reg__quardra_buffer__double_ldg_reg__st_global_wt__pipeline(
  const float* A, const float* B, float* C, int M, int N, int K)
{
  constexpr int THREAD_COUNT =
    device_thread_count_calculator<BLOCK_TILE_M, BLOCK_TILE_N, THREAD_TILE_M, THREAD_TILE_N>();

  // The following checks are to ensure that data can be loaded via float4.
  static_assert(BLOCK_TILE_M * TILE_K % THREAD_COUNT == 0);
  static_assert(BLOCK_TILE_N * TILE_K % THREAD_COUNT == 0);
  constexpr int A_LDGSTS_LOOP_COUNT = BLOCK_TILE_M * TILE_K / THREAD_COUNT;
  constexpr int B_LDGSTS_LOOP_COUNT = BLOCK_TILE_N * TILE_K / THREAD_COUNT;

  // computing, pending, reg2sm ing, global2reg ing
  constexpr int    SM_BUFFER_COUNT = 4;
  __shared__ float A_sm[SM_BUFFER_COUNT][TILE_K][BLOCK_TILE_M];
  __shared__ float B_sm[SM_BUFFER_COUNT][TILE_K][BLOCK_TILE_N];

  // For computing. sm_2_comp_reg and computing.
  constexpr int CAL_REG_BUFFER_COUNT = 2;
  float         A_comp_reg[CAL_REG_BUFFER_COUNT][THREAD_TILE_M];
  float         B_comp_reg[CAL_REG_BUFFER_COUNT][THREAD_TILE_N];
  float         C_reg[THREAD_TILE_M][THREAD_TILE_N] = {0};
  // The following checks are to ensure that data can be loaded via float4.
  static_assert(THREAD_TILE_M % 4 == 0);
  static_assert(THREAD_TILE_N % 4 == 0);

  const int block_m_offset = blockIdx.y * BLOCK_TILE_M;
  const int block_n_offset = blockIdx.x * BLOCK_TILE_N;

  constexpr int CAL_THREADS_ON_M_AXIS = BLOCK_TILE_M / THREAD_TILE_M;
  constexpr int CAL_THREADS_ON_N_AXIS = BLOCK_TILE_N / THREAD_TILE_N;
  static_assert(CAL_THREADS_ON_M_AXIS * CAL_THREADS_ON_N_AXIS == THREAD_COUNT);
  constexpr int CAL_THREAD_M_LOAD_COUNT = THREAD_TILE_M / 4;
  constexpr int CAL_THREAD_N_LOAD_COUNT = THREAD_TILE_N / 4;
  const int     comp_thread_m_offset    = threadIdx.x / CAL_THREADS_ON_N_AXIS * THREAD_TILE_M;
  constexpr int CAL_THREAD_M_STRIDE     = 4;
  const int     comp_thread_n_offset    = threadIdx.x % CAL_THREADS_ON_N_AXIS * 4;
  constexpr int CAL_THREAD_N_STRIDE     = CAL_THREADS_ON_N_AXIS * 4;
  const int     k_iter_count            = K / TILE_K;

#define global_2_sm()                                                                                                  \
  {                                                                                                                    \
    /* Load A, global -> register */                                                                                   \
    _Pragma("unroll") for (int loop = 0; loop < A_LDGSTS_LOOP_COUNT; ++loop)                                           \
    {                                                                                                                  \
      const int m = (loop * THREAD_COUNT + threadIdx.x) / TILE_K;                                                      \
      const int k = (loop * THREAD_COUNT + threadIdx.x) % TILE_K;                                                      \
      __pipeline_memcpy_async(                                                                                         \
        &A_sm[SM_LOADING_BUFFER][k][m], &A[OFFSET(block_m_offset + m, k_iter_offset + k, K)], sizeof(float), 0);       \
    }                                                                                                                  \
    /* Load B, global->register */                                                                                     \
    _Pragma("unroll") for (int loop = 0; loop < B_LDGSTS_LOOP_COUNT; ++loop)                                           \
    {                                                                                                                  \
      const int k = (loop * THREAD_COUNT + threadIdx.x) / BLOCK_TILE_N;                                                \
      const int n = (loop * THREAD_COUNT + threadIdx.x) % BLOCK_TILE_N;                                                \
      __pipeline_memcpy_async(                                                                                         \
        &B_sm[SM_LOADING_BUFFER][k][n], &B[OFFSET(k_iter_offset + k, block_n_offset + n, N)], sizeof(float), 0);       \
    }                                                                                                                  \
  }

#define sm_2_comp_reg()                                                                                                \
  _Pragma("unroll") for (int i = 0; i < CAL_THREAD_M_LOAD_COUNT; ++i)                                                  \
  {                                                                                                                    \
    FETCH_FLOAT4(A_comp_reg[CAL_REG_SM2REG][i * 4],                                                                    \
                 A_sm[SM_COMPUTING][k][comp_thread_m_offset + i * CAL_THREAD_M_STRIDE]);                               \
  }                                                                                                                    \
  _Pragma("unroll") for (int i = 0; i < CAL_THREAD_N_LOAD_COUNT; ++i)                                                  \
  {                                                                                                                    \
    FETCH_FLOAT4(B_comp_reg[CAL_REG_SM2REG][i * 4],                                                                    \
                 B_sm[SM_COMPUTING][k][comp_thread_n_offset + i * CAL_THREAD_N_STRIDE]);                               \
  }

#define compute()                                                                                                      \
  _Pragma("unroll") for (int i = 0; i < THREAD_TILE_M; ++i)                                                            \
  {                                                                                                                    \
    _Pragma("unroll") for (int j = 0; j < THREAD_TILE_N; ++j)                                                          \
    {                                                                                                                  \
      C_reg[i][j] += A_comp_reg[CAL_REG_COMPUTING][i] * B_comp_reg[CAL_REG_COMPUTING][j];                              \
    }                                                                                                                  \
  }

  {
    // Load the first tile in k direction from global to register.
    constexpr int k_iter_offset     = 0;
    constexpr int SM_LOADING_BUFFER = 0;
    global_2_sm();
  }

  {
    // Load the second tile in k direction from global to register.
    constexpr int k_iter_offset     = TILE_K;
    constexpr int SM_LOADING_BUFFER = 1;
    global_2_sm();
  }

  __pipeline_commit();

  __pipeline_wait_prior(0);

  __syncthreads();

  {
    // Load the first line in the first tile from shared to computing register.
    constexpr int k              = 0;
    constexpr int SM_COMPUTING   = 0;
    constexpr int CAL_REG_SM2REG = 0;
    sm_2_comp_reg();
  }

  {
    // Load the third tile in k direction from global to register.
    constexpr int k_iter_offset     = TILE_K * 2;
    constexpr int SM_LOADING_BUFFER = 2;
    global_2_sm();
  }

  {
    // Load the third tile in k direction from global to register.
    constexpr int k_iter_offset     = TILE_K * 3;
    constexpr int SM_LOADING_BUFFER = 3;
    global_2_sm();
  }

  __pipeline_commit();

  int SM_COMPUTING_OFFSET = 0;
  int SM_LOADING_OFFSET   = 2;
  // k_iter_count must be an even number.
  for (int k_iter = 0; k_iter < k_iter_count - 4; k_iter += 2) {
#pragma unroll
    for (int k = 1; k < TILE_K; ++k) {
      const int CAL_REG_SM2REG = k & 1;
      const int SM_COMPUTING   = SM_COMPUTING_OFFSET;
      sm_2_comp_reg();
      const int CAL_REG_COMPUTING = (k - 1) & 1;
      compute();
    }
    {
      constexpr int k              = 0;
      constexpr int CAL_REG_SM2REG = 0;
      const int     SM_COMPUTING   = SM_COMPUTING_OFFSET + 1;
      sm_2_comp_reg();
    }
    {
      constexpr int CAL_REG_COMPUTING = 1;
      compute();
    }
#pragma unroll
    for (int k = 1; k < TILE_K; ++k) {
      const int CAL_REG_SM2REG = k & 1;
      const int SM_COMPUTING   = SM_COMPUTING_OFFSET + 1;
      sm_2_comp_reg();
      const int CAL_REG_COMPUTING = (k - 1) & 1;
      compute();
    }
    __pipeline_wait_prior(0);
    __syncthreads();
    SM_COMPUTING_OFFSET ^= 2;
    {
      constexpr int k              = 0;
      constexpr int CAL_REG_SM2REG = 0;
      const int     SM_COMPUTING   = SM_COMPUTING_OFFSET;
      sm_2_comp_reg();
    }
    {
      constexpr int CAL_REG_COMPUTING = 1;
      compute();
    }
    SM_LOADING_OFFSET ^= 2;
    {
      const int k_iter_offset     = (k_iter + 4) * TILE_K;
      const int SM_LOADING_BUFFER = SM_LOADING_OFFSET;
      global_2_sm();
    }
    {
      const int k_iter_offset     = (k_iter + 5) * TILE_K;
      const int SM_LOADING_BUFFER = SM_LOADING_OFFSET + 1;
      global_2_sm();
    }
    __pipeline_commit();
  }
  {
    // k_iter == k_iter_count - 4
#pragma unroll
    for (int k = 1; k < TILE_K; ++k) {
      const int CAL_REG_SM2REG = k & 1;
      const int SM_COMPUTING   = SM_COMPUTING_OFFSET;
      sm_2_comp_reg();
      const int CAL_REG_COMPUTING = (k - 1) & 1;
      compute();
    }
    {
      constexpr int k              = 0;
      constexpr int CAL_REG_SM2REG = 0;
      const int     SM_COMPUTING   = SM_COMPUTING_OFFSET + 1;
      sm_2_comp_reg();
    }
    {
      constexpr int CAL_REG_COMPUTING = 1;
      compute();
    }
#pragma unroll
    for (int k = 1; k < TILE_K; ++k) {
      const int CAL_REG_SM2REG = k & 1;
      const int SM_COMPUTING   = SM_COMPUTING_OFFSET + 1;
      sm_2_comp_reg();
      const int CAL_REG_COMPUTING = (k - 1) & 1;
      compute();
    }
    __pipeline_wait_prior(0);
    __syncthreads();
    SM_COMPUTING_OFFSET ^= 2;
    {
      constexpr int k              = 0;
      constexpr int CAL_REG_SM2REG = 0;
      const int     SM_COMPUTING   = SM_COMPUTING_OFFSET;
      sm_2_comp_reg();
    }
    {
      constexpr int CAL_REG_COMPUTING = 1;
      compute();
    }
  }

  {
    // k_iter == k_iter_count - 2
#pragma unroll
    for (int k = 1; k < TILE_K; ++k) {
      const int CAL_REG_SM2REG = k & 1;
      const int SM_COMPUTING   = SM_COMPUTING_OFFSET;
      sm_2_comp_reg();
      const int CAL_REG_COMPUTING = (k - 1) & 1;
      compute();
    }
    {
      constexpr int k              = 0;
      constexpr int CAL_REG_SM2REG = 0;
      const int     SM_COMPUTING   = SM_COMPUTING_OFFSET + 1;
      sm_2_comp_reg();
    }
    {
      constexpr int CAL_REG_COMPUTING = 1;
      compute();
    }
#pragma unroll
    for (int k = 1; k < TILE_K; ++k) {
      const int CAL_REG_SM2REG = k & 1;
      const int SM_COMPUTING   = SM_COMPUTING_OFFSET + 1;
      sm_2_comp_reg();
      const int CAL_REG_COMPUTING = (k - 1) & 1;
      compute();
    }
    {
      constexpr int CAL_REG_COMPUTING = 1;
      compute();
    }
  }

#pragma unroll
  for (int i = 0; i < THREAD_TILE_M; ++i) {
    const int m = block_m_offset + comp_thread_m_offset + i;
#pragma unroll
    for (int j = 0; j < THREAD_TILE_N; j += 4) {
      const int n = block_n_offset + comp_thread_n_offset + j / 4 * CAL_THREAD_N_STRIDE;
      asm volatile(
        "st.global.wt.v4.f32 [%0], {%1, %2, %3, %4};"
        :
        : "l"(&C[OFFSET(m, n, N)]), "f"(C_reg[i][j]), "f"(C_reg[i][j + 1]), "f"(C_reg[i][j + 2]), "f"(C_reg[i][j + 3])
        : "memory");
    }
  }
#undef global_2_sm
#undef sm_2_comp_reg
#undef compute
}

template<int BLOCK_TILE_M, int BLOCK_TILE_N, int THREAD_TILE_M, int THREAD_TILE_N, int TILE_K>
__global__ void llmmm__overlap_global2sm2reg__quardra_buffer__double_ldg_reg__st_global_wt__pipeline_PTX(
  const float* A, const float* B, float* C, int M, int N, int K)
{
  constexpr int THREAD_COUNT =
    device_thread_count_calculator<BLOCK_TILE_M, BLOCK_TILE_N, THREAD_TILE_M, THREAD_TILE_N>();

  // The following checks are to ensure that data can be loaded via float4.
  static_assert(BLOCK_TILE_M * TILE_K % THREAD_COUNT == 0);
  static_assert(BLOCK_TILE_N * TILE_K % THREAD_COUNT == 0);
  constexpr int A_LDGSTS_LOOP_COUNT = BLOCK_TILE_M * TILE_K / THREAD_COUNT;
  constexpr int B_LDGSTS_LOOP_COUNT = BLOCK_TILE_N * TILE_K / THREAD_COUNT;

  // computing, pending, reg2sm ing, global2reg ing
  constexpr int    SM_BUFFER_COUNT = 4;
  __shared__ float A_sm[SM_BUFFER_COUNT][TILE_K][BLOCK_TILE_M];
  __shared__ float B_sm[SM_BUFFER_COUNT][TILE_K][BLOCK_TILE_N];

  // For computing. sm_2_comp_reg and computing.
  constexpr int CAL_REG_BUFFER_COUNT = 2;
  float         A_comp_reg[CAL_REG_BUFFER_COUNT][THREAD_TILE_M];
  float         B_comp_reg[CAL_REG_BUFFER_COUNT][THREAD_TILE_N];
  float         C_reg[THREAD_TILE_M][THREAD_TILE_N] = {0};
  // The following checks are to ensure that data can be loaded via float4.
  static_assert(THREAD_TILE_M % 4 == 0);
  static_assert(THREAD_TILE_N % 4 == 0);

  const int block_m_offset = blockIdx.y * BLOCK_TILE_M;
  const int block_n_offset = blockIdx.x * BLOCK_TILE_N;

  constexpr int CAL_THREADS_ON_M_AXIS = BLOCK_TILE_M / THREAD_TILE_M;
  constexpr int CAL_THREADS_ON_N_AXIS = BLOCK_TILE_N / THREAD_TILE_N;
  static_assert(CAL_THREADS_ON_M_AXIS * CAL_THREADS_ON_N_AXIS == THREAD_COUNT);
  constexpr int CAL_THREAD_M_LOAD_COUNT = THREAD_TILE_M / 4;
  constexpr int CAL_THREAD_N_LOAD_COUNT = THREAD_TILE_N / 4;
  const int     comp_thread_m_offset    = threadIdx.x / CAL_THREADS_ON_N_AXIS * THREAD_TILE_M;
  constexpr int CAL_THREAD_M_STRIDE     = 4;
  const int     comp_thread_n_offset    = threadIdx.x % CAL_THREADS_ON_N_AXIS * 4;
  constexpr int CAL_THREAD_N_STRIDE     = CAL_THREADS_ON_N_AXIS * 4;
  const int     k_iter_count            = K / TILE_K;

#define global_2_sm()                                                                                                  \
  {                                                                                                                    \
    /* Load A, global -> register */                                                                                   \
    _Pragma("unroll") for (int loop = 0; loop < A_LDGSTS_LOOP_COUNT; ++loop)                                           \
    {                                                                                                                  \
      const int m       = (loop * THREAD_COUNT + threadIdx.x) / TILE_K;                                                \
      const int k       = (loop * THREAD_COUNT + threadIdx.x) % TILE_K;                                                \
      uint32_t  dst_ptr = __cvta_generic_to_shared(&A_sm[SM_LOADING_BUFFER][k][m]);                                    \
      asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], 4, 4;\n" ::"r"(dst_ptr),                            \
                   "l"(&A[OFFSET(block_m_offset + m, k_iter_offset + k, K)]));                                         \
    }                                                                                                                  \
    /* Load B, global->register */                                                                                     \
    _Pragma("unroll") for (int loop = 0; loop < B_LDGSTS_LOOP_COUNT; ++loop)                                           \
    {                                                                                                                  \
      const int k       = (loop * THREAD_COUNT + threadIdx.x) / BLOCK_TILE_N;                                          \
      const int n       = (loop * THREAD_COUNT + threadIdx.x) % BLOCK_TILE_N;                                          \
      uint32_t  dst_ptr = __cvta_generic_to_shared(&B_sm[SM_LOADING_BUFFER][k][n]);                                    \
      asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], 4, 4;\n" ::"r"(dst_ptr),                            \
                   "l"(&B[OFFSET(k_iter_offset + k, block_n_offset + n, N)]));                                         \
    }                                                                                                                  \
  }

#define commit()                                                                                                       \
  {                                                                                                                    \
    asm volatile("cp.async.commit_group;");                                                                            \
  }

#define wait()                                                                                                         \
  {                                                                                                                    \
    asm volatile("cp.async.wait_group 0;");                                                                            \
  }

#define sm_2_comp_reg()                                                                                                \
  _Pragma("unroll") for (int i = 0; i < CAL_THREAD_M_LOAD_COUNT; ++i)                                                  \
  {                                                                                                                    \
    FETCH_FLOAT4(A_comp_reg[CAL_REG_SM2REG][i * 4],                                                                    \
                 A_sm[SM_COMPUTING][k][comp_thread_m_offset + i * CAL_THREAD_M_STRIDE]);                               \
  }                                                                                                                    \
  _Pragma("unroll") for (int i = 0; i < CAL_THREAD_N_LOAD_COUNT; ++i)                                                  \
  {                                                                                                                    \
    FETCH_FLOAT4(B_comp_reg[CAL_REG_SM2REG][i * 4],                                                                    \
                 B_sm[SM_COMPUTING][k][comp_thread_n_offset + i * CAL_THREAD_N_STRIDE]);                               \
  }

#define compute()                                                                                                      \
  _Pragma("unroll") for (int i = 0; i < THREAD_TILE_M; ++i)                                                            \
  {                                                                                                                    \
    _Pragma("unroll") for (int j = 0; j < THREAD_TILE_N; ++j)                                                          \
    {                                                                                                                  \
      C_reg[i][j] += A_comp_reg[CAL_REG_COMPUTING][i] * B_comp_reg[CAL_REG_COMPUTING][j];                              \
    }                                                                                                                  \
  }

  {
    // Load the first tile in k direction from global to register.
    constexpr int k_iter_offset     = 0;
    constexpr int SM_LOADING_BUFFER = 0;
    global_2_sm();
  }

  {
    // Load the second tile in k direction from global to register.
    constexpr int k_iter_offset     = TILE_K;
    constexpr int SM_LOADING_BUFFER = 1;
    global_2_sm();
  }

  commit();

  wait();

  __syncthreads();

  {
    // Load the first line in the first tile from shared to computing register.
    constexpr int k              = 0;
    constexpr int SM_COMPUTING   = 0;
    constexpr int CAL_REG_SM2REG = 0;
    sm_2_comp_reg();
  }

  {
    // Load the third tile in k direction from global to register.
    constexpr int k_iter_offset     = TILE_K * 2;
    constexpr int SM_LOADING_BUFFER = 2;
    global_2_sm();
  }

  {
    // Load the third tile in k direction from global to register.
    constexpr int k_iter_offset     = TILE_K * 3;
    constexpr int SM_LOADING_BUFFER = 3;
    global_2_sm();
  }

  commit();

  int SM_COMPUTING_OFFSET = 0;
  int SM_LOADING_OFFSET   = 2;
  // k_iter_count must be an even number.
  for (int k_iter = 0; k_iter < k_iter_count - 4; k_iter += 2) {
#pragma unroll
    for (int k = 1; k < TILE_K; ++k) {
      const int CAL_REG_SM2REG = k & 1;
      const int SM_COMPUTING   = SM_COMPUTING_OFFSET;
      sm_2_comp_reg();
      const int CAL_REG_COMPUTING = (k - 1) & 1;
      compute();
    }
    {
      constexpr int k              = 0;
      constexpr int CAL_REG_SM2REG = 0;
      const int     SM_COMPUTING   = SM_COMPUTING_OFFSET + 1;
      sm_2_comp_reg();
    }
    {
      constexpr int CAL_REG_COMPUTING = 1;
      compute();
    }
#pragma unroll
    for (int k = 1; k < TILE_K; ++k) {
      const int CAL_REG_SM2REG = k & 1;
      const int SM_COMPUTING   = SM_COMPUTING_OFFSET + 1;
      sm_2_comp_reg();
      const int CAL_REG_COMPUTING = (k - 1) & 1;
      compute();
    }
    wait();
    __syncthreads();
    SM_COMPUTING_OFFSET ^= 2;
    {
      constexpr int k              = 0;
      constexpr int CAL_REG_SM2REG = 0;
      const int     SM_COMPUTING   = SM_COMPUTING_OFFSET;
      sm_2_comp_reg();
    }
    {
      constexpr int CAL_REG_COMPUTING = 1;
      compute();
    }
    SM_LOADING_OFFSET ^= 2;
    {
      const int k_iter_offset     = (k_iter + 4) * TILE_K;
      const int SM_LOADING_BUFFER = SM_LOADING_OFFSET;
      global_2_sm();
    }
    {
      const int k_iter_offset     = (k_iter + 5) * TILE_K;
      const int SM_LOADING_BUFFER = SM_LOADING_OFFSET + 1;
      global_2_sm();
    }
    commit();
  }
  {
    // k_iter == k_iter_count - 4
#pragma unroll
    for (int k = 1; k < TILE_K; ++k) {
      const int CAL_REG_SM2REG = k & 1;
      const int SM_COMPUTING   = SM_COMPUTING_OFFSET;
      sm_2_comp_reg();
      const int CAL_REG_COMPUTING = (k - 1) & 1;
      compute();
    }
    {
      constexpr int k              = 0;
      constexpr int CAL_REG_SM2REG = 0;
      const int     SM_COMPUTING   = SM_COMPUTING_OFFSET + 1;
      sm_2_comp_reg();
    }
    {
      constexpr int CAL_REG_COMPUTING = 1;
      compute();
    }
#pragma unroll
    for (int k = 1; k < TILE_K; ++k) {
      const int CAL_REG_SM2REG = k & 1;
      const int SM_COMPUTING   = SM_COMPUTING_OFFSET + 1;
      sm_2_comp_reg();
      const int CAL_REG_COMPUTING = (k - 1) & 1;
      compute();
    }
    wait();
    __syncthreads();
    SM_COMPUTING_OFFSET ^= 2;
    {
      constexpr int k              = 0;
      constexpr int CAL_REG_SM2REG = 0;
      const int     SM_COMPUTING   = SM_COMPUTING_OFFSET;
      sm_2_comp_reg();
    }
    {
      constexpr int CAL_REG_COMPUTING = 1;
      compute();
    }
  }

  {
    // k_iter == k_iter_count - 2
#pragma unroll
    for (int k = 1; k < TILE_K; ++k) {
      const int CAL_REG_SM2REG = k & 1;
      const int SM_COMPUTING   = SM_COMPUTING_OFFSET;
      sm_2_comp_reg();
      const int CAL_REG_COMPUTING = (k - 1) & 1;
      compute();
    }
    {
      constexpr int k              = 0;
      constexpr int CAL_REG_SM2REG = 0;
      const int     SM_COMPUTING   = SM_COMPUTING_OFFSET + 1;
      sm_2_comp_reg();
    }
    {
      constexpr int CAL_REG_COMPUTING = 1;
      compute();
    }
#pragma unroll
    for (int k = 1; k < TILE_K; ++k) {
      const int CAL_REG_SM2REG = k & 1;
      const int SM_COMPUTING   = SM_COMPUTING_OFFSET + 1;
      sm_2_comp_reg();
      const int CAL_REG_COMPUTING = (k - 1) & 1;
      compute();
    }
    {
      constexpr int CAL_REG_COMPUTING = 1;
      compute();
    }
  }

#pragma unroll
  for (int i = 0; i < THREAD_TILE_M; ++i) {
    const int m = block_m_offset + comp_thread_m_offset + i;
#pragma unroll
    for (int j = 0; j < THREAD_TILE_N; j += 4) {
      const int n = block_n_offset + comp_thread_n_offset + j / 4 * CAL_THREAD_N_STRIDE;
      asm volatile(
        "st.global.wt.v4.f32 [%0], {%1, %2, %3, %4};"
        :
        : "l"(&C[OFFSET(m, n, N)]), "f"(C_reg[i][j]), "f"(C_reg[i][j + 1]), "f"(C_reg[i][j + 2]), "f"(C_reg[i][j + 3])
        : "memory");
    }
  }
#undef global_2_sm
#undef commit
#undef wait
#undef sm_2_comp_reg
#undef compute
}

#define test_and_launch_function(function_name)                                                                        \
  template<int BLOCK_TILE_M, int BLOCK_TILE_N, int THREAD_TILE_M, int THREAD_TILE_N, int TILE_K>                       \
  void launch_##function_name(const float* A, const float* B, float* C, int M, int N, int K)                           \
  {                                                                                                                    \
    if (N % 128 != 0 || N < 1024) {                                                                                    \
      throw std::runtime_error("Not support N = " + std::to_string(N));                                                \
    }                                                                                                                  \
    if (K % 128 != 0 || K < TILE_K * 4 || K / TILE_K % 2 != 0) {                                                       \
      throw std::runtime_error("Not support K = " + std::to_string(K));                                                \
    }                                                                                                                  \
    static_assert(8 <= BLOCK_TILE_N && BLOCK_TILE_N <= 256 && ((BLOCK_TILE_N & (BLOCK_TILE_N - 1)) == 0));             \
    static_assert(8 <= TILE_K && TILE_K <= 128 && ((TILE_K & (TILE_K - 1)) == 0));                                     \
    /* This check is for reading data using float4.*/                                                                  \
    static_assert(TILE_K * BLOCK_TILE_N >= 128);                                                                       \
    static_assert(BLOCK_TILE_M == 128);                                                                                \
    static_assert(BLOCK_TILE_M % THREAD_TILE_M == 0);                                                                  \
    static_assert(BLOCK_TILE_N % THREAD_TILE_N == 0);                                                                  \
    const int aligned_M = M / BLOCK_TILE_M * BLOCK_TILE_M;                                                             \
    dim3      grid(N / BLOCK_TILE_N, aligned_M / BLOCK_TILE_M);                                                        \
    dim3      block(host_thread_count_calculator<BLOCK_TILE_M, BLOCK_TILE_N, THREAD_TILE_M, THREAD_TILE_N>());         \
    auto      kernel_func = &function_name<BLOCK_TILE_M, BLOCK_TILE_N, THREAD_TILE_M, THREAD_TILE_N, TILE_K>;          \
    auto      kSmemSize   = 0;                                                                                         \
    auto      err         = cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize); \
    if (err) {                                                                                                         \
      std::stringstream log_prefix;                                                                                    \
      log_prefix << "function_name=" << #function_name << ", " << BLOCK_TILE_M << "x" << BLOCK_TILE_N << "x" << TILE_K \
                 << ", " << THREAD_TILE_M << "x" << THREAD_TILE_N;                                                     \
      std::cerr << log_prefix.str() << ", " << cudaGetErrorString(err) << std::endl;                                   \
      return;                                                                                                          \
    }                                                                                                                  \
    function_name<BLOCK_TILE_M, BLOCK_TILE_N, THREAD_TILE_M, THREAD_TILE_N, TILE_K>                                    \
      <<<grid, block>>>(A, B, C, M, N, K);                                                                             \
    const int unaligned_M = M - aligned_M;                                                                             \
    if (unaligned_M) {                                                                                                 \
      throw std::runtime_error("Not implemented");                                                                     \
    }                                                                                                                  \
  }                                                                                                                    \
                                                                                                                       \
  template<int BLOCK_TILE_M, int BLOCK_TILE_N, int THREAD_TILE_M, int THREAD_TILE_N, int TILE_K>                       \
  void test_##function_name(const float*              A,                                                               \
                            const float*              B,                                                               \
                            float*                    C,                                                               \
                            int                       M,                                                               \
                            int                       N,                                                               \
                            int                       K,                                                               \
                            const std::vector<float>& groundtruth,                                                     \
                            std::vector<float>&       host_C)                                                          \
  {                                                                                                                    \
    std::stringstream log_prefix;                                                                                      \
    log_prefix << "function_name=" << #function_name << ", " << BLOCK_TILE_M << "x" << BLOCK_TILE_N << "x" << TILE_K   \
               << ", " << THREAD_TILE_M << "x" << THREAD_TILE_N;                                                       \
    cudaMemset(C, 0, M* N * sizeof(float));                                                                            \
    CHECK_CUDA_ERROR_WITH_INFO(log_prefix.str().c_str());                                                              \
    launch_##function_name<BLOCK_TILE_M, BLOCK_TILE_N, THREAD_TILE_M, THREAD_TILE_N, TILE_K>(A, B, C, M, N, K);        \
    CHECK_CUDA_ERROR_WITH_INFO(log_prefix.str().c_str());                                                              \
    cudaMemcpy(host_C.data(), C, sizeof(float) * host_C.size(), cudaMemcpyDefault);                                    \
    CHECK_CUDA_ERROR_WITH_INFO(log_prefix.str().c_str());                                                              \
                                                                                                                       \
    const float(*host_result_ptr)[N]   = reinterpret_cast<const float(*)[N]>(groundtruth.data());                      \
    const float(*device_result_ptr)[N] = reinterpret_cast<const float(*)[N]>(host_C.data());                           \
                                                                                                                       \
    constexpr float EPS = 1e-1;                                                                                        \
                                                                                                                       \
    for (int i = 0; i < M; ++i) {                                                                                      \
      for (int j = 0; j < N; ++j) {                                                                                    \
        if (isnan(device_result_ptr[i][j]) || fabs(host_result_ptr[i][j] - device_result_ptr[i][j]) > EPS) {           \
          printf("%.7f, %.7f\n", host_result_ptr[i][j], device_result_ptr[i][j]);                                      \
          std::stringstream ss;                                                                                        \
          ss << #function_name << ", " << BLOCK_TILE_M << "x" << BLOCK_TILE_N << ", " << THREAD_TILE_M << "x"          \
             << THREAD_TILE_N << ", invalid result, m=" << i << ", n=" << j << ", expected " << host_result_ptr[i][j]  \
             << ", got " << device_result_ptr[i][j];                                                                   \
          throw std::runtime_error(ss.str());                                                                          \
        }                                                                                                              \
      }                                                                                                                \
    }                                                                                                                  \
    std::cout << "success, " << #function_name << ",  BLOCK_TILE_M=" << BLOCK_TILE_M                                   \
              << ", BLOCK_TILE_N=" << BLOCK_TILE_N << ", TILE_K=" << TILE_K << ", THREAD_TILE_M=" << THREAD_TILE_M     \
              << ", THREAD_TILE_N=" << THREAD_TILE_N << std::endl;                                                     \
  }

test_and_launch_function(llmmm);
test_and_launch_function(llmmm__overlap_sm2reg);
test_and_launch_function(llmmm__overlap_global2sm);
test_and_launch_function(llmmm__overlap_global2sm2reg);
test_and_launch_function(llmmm__overlap_global2sm2reg__using_if);
test_and_launch_function(llmmm__overlap_global2sm2reg__quardra_buffer);
test_and_launch_function(llmmm__overlap_global2sm2reg__quardra_buffer__double_ldg_reg);
test_and_launch_function(llmmm__overlap_global2sm2reg__quardra_buffer__double_ldg_reg__st_global_wt);
test_and_launch_function(llmmm__overlap_global2sm2reg__quardra_buffer__double_ldg_reg__st_global_wt__pipeline);
test_and_launch_function(llmmm__overlap_global2sm2reg__quardra_buffer__double_ldg_reg__st_global_wt__pipeline_PTX);

int main()
{
  static const int M = (1 << 12), N = (1 << 12), K = (1 << 12);
  // static const int M = 128, N = 128, K = 128;

  std::vector<float>                    host_A(M * K), host_B(K * N), host_C(M * N), host_result(M * N);
  std::random_device                    rd;
  std::mt19937                          gen(rd());
  std::uniform_real_distribution<float> dis(-5, 5);
  for (auto& vec : {&host_A, &host_B}) {
#if 1
    for (auto& data : *vec) {
      data = dis(gen);
    }
#else
    if (vec == &host_A) {
      for (size_t i = 0; i < vec->size(); ++i) {
        int row = i / K;
        int col = i % K;
        if (row < 128 && col < 128) {
          vec->operator[](i) = row * 128 + col;
          vec->operator[](i) = (row == col);
        }
        else {
          vec->operator[](i) = 0;
        }
      }
    }
    if (vec == &host_B) {
      for (size_t i = 0; i < vec->size(); ++i) {
        int row = i / N;
        int col = i % N;
        if (row < 128 && col < 128) {
          vec->operator[](i) = row * 128 + col;
        }
        else {
          vec->operator[](i) = 0;
        }
      }
    }
#endif
  }

  float *A, *B, *C;
  for (auto& pair : {std::make_pair(host_A, &A), std::make_pair(host_B, &B), std::make_pair(host_C, &C)}) {
    const std::vector<float>& host   = pair.first;
    float*&                   device = *pair.second;
    cudaMalloc(&device, sizeof(float) * host.size());
    cudaMemcpy(device, host.data(), sizeof(float) * host.size(), cudaMemcpyDefault);
    CHECK_CUDA_ERROR();
  }

  {
    cudaMemset(C, 0, M * N * sizeof(float));
    launch_naive_mm(A, B, C, M, N, K);
    cudaMemcpy(host_result.data(), C, sizeof(float) * host_C.size(), cudaMemcpyDefault);
    CHECK_CUDA_ERROR();
  }

  /* clang-format off */
  test_llmmm<128, 32, 8, 4, 16>(A, B, C, M, N, K, host_result, host_C);
  test_llmmm<128, 64, 8, 4, 16>(A, B, C, M, N, K, host_result, host_C);
  test_llmmm<128, 32, 8, 8, 16>(A, B, C, M, N, K, host_result, host_C);
  test_llmmm<128, 64, 8, 8, 16>(A, B, C, M, N, K, host_result, host_C);
  test_llmmm<128, 128, 8, 8, 8>(A, B, C, M, N, K, host_result, host_C);
  test_llmmm<128, 256, 8, 16, 8>(A, B, C, M, N, K, host_result, host_C);
  test_llmmm<128, 256, 8, 16, 16>(A, B, C, M, N, K, host_result, host_C);
  test_llmmm<128, 256, 8, 16, 8>(A, B, C, M, N, K, host_result, host_C);
  test_llmmm<128, 128, 16, 8, 8>(A, B, C, M, N, K, host_result, host_C);

  test_llmmm__overlap_sm2reg<128, 32, 8, 4, 16>(A, B, C, M, N, K, host_result, host_C);
  test_llmmm__overlap_sm2reg<128, 64, 8, 4, 16>(A, B, C, M, N, K, host_result, host_C);
  test_llmmm__overlap_sm2reg<128, 32, 8, 8, 16>(A, B, C, M, N, K, host_result, host_C);
  test_llmmm__overlap_sm2reg<128, 64, 8, 8, 16>(A, B, C, M, N, K, host_result, host_C);
  test_llmmm__overlap_sm2reg<128, 128, 8, 8, 8>(A, B, C, M, N, K, host_result, host_C);
  test_llmmm__overlap_sm2reg<128, 256, 8, 16, 8>(A, B, C, M, N, K, host_result, host_C);
  test_llmmm__overlap_sm2reg<128, 256, 8, 16, 16>(A, B, C, M, N, K, host_result, host_C);
  test_llmmm__overlap_sm2reg<128, 256, 8, 16, 8>(A, B, C, M, N, K, host_result, host_C);

  test_llmmm__overlap_global2sm<128, 32, 8, 4, 16>(A, B, C, M, N, K, host_result, host_C);
  test_llmmm__overlap_global2sm<128, 64, 8, 4, 16>(A, B, C, M, N, K, host_result, host_C);
  test_llmmm__overlap_global2sm<128, 32, 8, 8, 16>(A, B, C, M, N, K, host_result, host_C);
  test_llmmm__overlap_global2sm<128, 64, 8, 8, 16>(A, B, C, M, N, K, host_result, host_C);
  test_llmmm__overlap_global2sm<128, 128, 8, 8, 8>(A, B, C, M, N, K, host_result, host_C);
  test_llmmm__overlap_global2sm<128, 256, 8, 16, 8>(A, B, C, M, N, K, host_result, host_C);
  test_llmmm__overlap_global2sm<128, 256, 8, 16, 16>(A, B, C, M, N, K, host_result, host_C);
  test_llmmm__overlap_global2sm<128, 256, 8, 16, 8>(A, B, C, M, N, K, host_result, host_C);

  test_llmmm__overlap_global2sm2reg<128, 32, 8, 4, 16>(A, B, C, M, N, K, host_result, host_C);
  test_llmmm__overlap_global2sm2reg<128, 64, 8, 4, 16>(A, B, C, M, N, K, host_result, host_C);
  test_llmmm__overlap_global2sm2reg<128, 32, 8, 8, 16>(A, B, C, M, N, K, host_result, host_C);
  test_llmmm__overlap_global2sm2reg<128, 64, 8, 8, 16>(A, B, C, M, N, K, host_result, host_C);
  test_llmmm__overlap_global2sm2reg<128, 128, 8, 8, 8>(A, B, C, M, N, K, host_result, host_C);
  test_llmmm__overlap_global2sm2reg<128, 256, 8, 16, 8>(A, B, C, M, N, K, host_result, host_C);
  test_llmmm__overlap_global2sm2reg<128, 256, 8, 16, 16>(A, B, C, M, N, K, host_result, host_C);

  test_llmmm__overlap_global2sm2reg__using_if<128, 32, 8, 4, 16>(A, B, C, M, N, K, host_result, host_C);
  test_llmmm__overlap_global2sm2reg__using_if<128, 64, 8, 4, 16>(A, B, C, M, N, K, host_result, host_C);
  test_llmmm__overlap_global2sm2reg__using_if<128, 32, 8, 8, 16>(A, B, C, M, N, K, host_result, host_C);
  test_llmmm__overlap_global2sm2reg__using_if<128, 64, 8, 8, 16>(A, B, C, M, N, K, host_result, host_C);
  test_llmmm__overlap_global2sm2reg__using_if<128, 128, 8, 8, 8>(A, B, C, M, N, K, host_result, host_C);
  test_llmmm__overlap_global2sm2reg__using_if<128, 256, 8, 16, 8>(A, B, C, M, N, K, host_result, host_C);
  test_llmmm__overlap_global2sm2reg__using_if<128, 256, 8, 16, 16>(A, B, C, M, N, K, host_result, host_C);
  test_llmmm__overlap_global2sm2reg__using_if<128, 256, 8, 16, 8>(A, B, C, M, N, K, host_result, host_C);

  test_llmmm__overlap_global2sm2reg__quardra_buffer<128, 32, 8, 4, 16>(A, B, C, M, N, K, host_result, host_C);
  test_llmmm__overlap_global2sm2reg__quardra_buffer<128, 64, 8, 4, 16>(A, B, C, M, N, K, host_result, host_C);
  test_llmmm__overlap_global2sm2reg__quardra_buffer<128, 32, 8, 8, 16>(A, B, C, M, N, K, host_result, host_C);
  test_llmmm__overlap_global2sm2reg__quardra_buffer<128, 64, 8, 8, 16>(A, B, C, M, N, K, host_result, host_C);
  test_llmmm__overlap_global2sm2reg__quardra_buffer<128, 128, 8, 8, 8>(A, B, C, M, N, K, host_result, host_C);
  test_llmmm__overlap_global2sm2reg__quardra_buffer<128, 256, 8, 16, 8>(A, B, C, M, N, K, host_result, host_C);

  test_llmmm__overlap_global2sm2reg__quardra_buffer__double_ldg_reg<128, 32, 8, 4, 16>(A, B, C, M, N, K, host_result, host_C);
  test_llmmm__overlap_global2sm2reg__quardra_buffer__double_ldg_reg<128, 64, 8, 4, 16>(A, B, C, M, N, K, host_result, host_C);
  test_llmmm__overlap_global2sm2reg__quardra_buffer__double_ldg_reg<128, 32, 8, 8, 16>(A, B, C, M, N, K, host_result, host_C);
  test_llmmm__overlap_global2sm2reg__quardra_buffer__double_ldg_reg<128, 64, 8, 8, 16>(A, B, C, M, N, K, host_result, host_C);
  test_llmmm__overlap_global2sm2reg__quardra_buffer__double_ldg_reg<128, 128, 8, 8, 8>(A, B, C, M, N, K, host_result, host_C);
  test_llmmm__overlap_global2sm2reg__quardra_buffer__double_ldg_reg<128, 256, 8, 16, 8>(A, B, C, M, N, K, host_result, host_C);

  test_llmmm__overlap_global2sm2reg__quardra_buffer__double_ldg_reg__st_global_wt<128, 32, 8, 4, 16>(A, B, C, M, N, K, host_result, host_C);
  test_llmmm__overlap_global2sm2reg__quardra_buffer__double_ldg_reg__st_global_wt<128, 64, 8, 4, 16>(A, B, C, M, N, K, host_result, host_C);
  test_llmmm__overlap_global2sm2reg__quardra_buffer__double_ldg_reg__st_global_wt<128, 32, 8, 8, 16>(A, B, C, M, N, K, host_result, host_C);
  test_llmmm__overlap_global2sm2reg__quardra_buffer__double_ldg_reg__st_global_wt<128, 64, 8, 8, 16>(A, B, C, M, N, K, host_result, host_C);
  test_llmmm__overlap_global2sm2reg__quardra_buffer__double_ldg_reg__st_global_wt<128, 128, 8, 8, 8>(A, B, C, M, N, K, host_result, host_C);
  test_llmmm__overlap_global2sm2reg__quardra_buffer__double_ldg_reg__st_global_wt<128, 128, 16, 8, 8>(A, B, C, M, N, K, host_result, host_C);
  test_llmmm__overlap_global2sm2reg__quardra_buffer__double_ldg_reg__st_global_wt<128, 256, 8, 16, 8>(A, B, C, M, N, K, host_result, host_C);

  test_llmmm__overlap_global2sm2reg__quardra_buffer__double_ldg_reg__st_global_wt__pipeline<128, 32, 8, 4, 16>(A, B, C, M, N, K, host_result, host_C);
  test_llmmm__overlap_global2sm2reg__quardra_buffer__double_ldg_reg__st_global_wt__pipeline<128, 64, 8, 4, 16>(A, B, C, M, N, K, host_result, host_C);
  test_llmmm__overlap_global2sm2reg__quardra_buffer__double_ldg_reg__st_global_wt__pipeline<128, 32, 8, 8, 16>(A, B, C, M, N, K, host_result, host_C);
  test_llmmm__overlap_global2sm2reg__quardra_buffer__double_ldg_reg__st_global_wt__pipeline<128, 64, 8, 8, 16>(A, B, C, M, N, K, host_result, host_C);
  test_llmmm__overlap_global2sm2reg__quardra_buffer__double_ldg_reg__st_global_wt__pipeline<128, 128, 8, 8, 8>(A, B, C, M, N, K, host_result, host_C);
  test_llmmm__overlap_global2sm2reg__quardra_buffer__double_ldg_reg__st_global_wt__pipeline<128, 256, 8, 16, 8>(A, B, C, M, N, K, host_result, host_C);

  test_llmmm__overlap_global2sm2reg__quardra_buffer__double_ldg_reg__st_global_wt__pipeline_PTX<128, 32, 8, 4, 16>(A, B, C, M, N, K, host_result, host_C);
  test_llmmm__overlap_global2sm2reg__quardra_buffer__double_ldg_reg__st_global_wt__pipeline_PTX<128, 64, 8, 4, 16>(A, B, C, M, N, K, host_result, host_C);
  test_llmmm__overlap_global2sm2reg__quardra_buffer__double_ldg_reg__st_global_wt__pipeline_PTX<128, 32, 8, 8, 16>(A, B, C, M, N, K, host_result, host_C);
  test_llmmm__overlap_global2sm2reg__quardra_buffer__double_ldg_reg__st_global_wt__pipeline_PTX<128, 64, 8, 8, 16>(A, B, C, M, N, K, host_result, host_C);
  test_llmmm__overlap_global2sm2reg__quardra_buffer__double_ldg_reg__st_global_wt__pipeline_PTX<128, 128, 8, 8, 8>(A, B, C, M, N, K, host_result, host_C);
  test_llmmm__overlap_global2sm2reg__quardra_buffer__double_ldg_reg__st_global_wt__pipeline_PTX<128, 256, 8, 16, 8>(A, B, C, M, N, K, host_result, host_C);
  /* clang-format on */

  {
    CHECK_CUDA_RETURN(cudaMemset(C, 0, sizeof(float) * host_C.size()));
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    float alpha  = 1.0;
    float beta   = 0;
    auto  result = cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, &alpha, A, K, B, N, &beta, C, N);
    // auto  result = cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K, &beta, C, N);
    if (result != 0) {
      throw std::runtime_error("failed to launch cublasSgemm");
    }
    CHECK_CUDA_RETURN(cudaDeviceSynchronize());
    CHECK_CUDA_RETURN(cudaMemcpy(host_C.data(), C, sizeof(float) * host_C.size(), cudaMemcpyDefault));
    // for (int i = 0; i < host_C.size(); i++) {
    //   if ((host_C[i] - host_result[i]) > 1e-1) {
    //     throw std::runtime_error("invalid result");
    //   }
    // }
  }

  return 0;
}
