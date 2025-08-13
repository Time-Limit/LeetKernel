#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda/barrier>
#include <cuda_pipeline_primitives.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "util/error.h"
#include "util/fp16.mm.cuh"
#include "util/util.cuh"

const int limit = 16;

__global__ void fp32_naive_mm(const float* A, const float* B, float* C, int M, int N, int K)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  int m = blockIdx.y * blockDim.y + threadIdx.y;

  if (n >= N || m >= M) {
    return;
  }

  A += m * K;
  B += n;
  double sum = 0.0;
#pragma unroll
  for (int k = 0; k < K; ++k) {
    sum += A[k] * B[k * N];
  }
  C[m * N + n] = sum;
}

void launch_fp32_naive_mm(const float* A, const float* B, float* C, int M, int N, int K)
{
  dim3 threads_per_block(16, 16);
  dim3 blocks_per_grid((N + threads_per_block.x - 1) / threads_per_block.x,
                       (M + threads_per_block.y - 1) / threads_per_block.y);

  fp32_naive_mm<<<blocks_per_grid, threads_per_block>>>(A, B, C, M, N, K);
  CHECK_CUDA_ERROR();
}

template<typename T, int BLOCK_TILE_M, int BLOCK_TILE_N, int WARP_TILE_M, int WARP_TILE_N>
__global__ void fp16_mma_m16n8k16_ldmatrix(const T* A, const T* B, T* C, int M, int N, int K)
{
  constexpr int WARP_COUNT   = BLOCK_TILE_M / WARP_TILE_M * BLOCK_TILE_N / WARP_TILE_N;
  constexpr int THREAD_COUNT = WARP_COUNT * 32;

  constexpr int LOOP_TILE_K = 16;
  // The 64 elements of type T in each 8x8 matrix are stored consecutively in a single layer.
  __shared__ T A_sm[2][BLOCK_TILE_M][LOOP_TILE_K / 2];
  __shared__ T B_sm[2][BLOCK_TILE_N][LOOP_TILE_K / 2];

  static_assert(BLOCK_TILE_M * LOOP_TILE_K % THREAD_COUNT == 0);
  static_assert(BLOCK_TILE_M * LOOP_TILE_K / THREAD_COUNT % 4 == 0);
  constexpr int A_LDG_COUNT_PER_THREAD = BLOCK_TILE_M * LOOP_TILE_K / THREAD_COUNT;
  constexpr int A_LDG_LOOP_COUNT       = A_LDG_COUNT_PER_THREAD / 4;
  // clang-format off
  // This is the thread layout of the same warp that loads matrix A, where each thread reads M1xK4 elements of type T at a
  // loop iteration.
  // T0 T8  .... T24
  // T1 T9       T25
  // T2 T10      T26
  // T3 T11      T27
  // T4 T12      T28
  // T5 T13      T29
  // T6 T14      T30
  // T7 T15      T31
  // clang-format on
  T A_ldg_reg[A_LDG_LOOP_COUNT][4];

  static_assert(BLOCK_TILE_N * LOOP_TILE_K % THREAD_COUNT == 0);
  static_assert(BLOCK_TILE_N * LOOP_TILE_K / THREAD_COUNT % 4 == 0);
  constexpr int B_LDG_COUNT_PER_THREAD = BLOCK_TILE_N * LOOP_TILE_K / THREAD_COUNT;
  constexpr int B_LDG_LOOP_COUNT       = B_LDG_COUNT_PER_THREAD / 4;
  // clang-format off
  // This is the thread layout of the same warp that loads matrix B, where each thread reads K2xN2 elements of type T at a
  // loop iteration.
  // T0 T4 ... T28
  // T1 T5     T29
  // T2 T6     T30
  // T3 T7     T31
  // clang-format on
  T B_ldg_reg[B_LDG_LOOP_COUNT][2][2];

  const int m_block_offset = blockIdx.y * BLOCK_TILE_M;
  const int n_block_offset = blockIdx.x * BLOCK_TILE_N;

  const int warp_id = threadIdx.x / 32;
  const int lane_id = threadIdx.x % 32;

  constexpr int M_MMA_WARP_COUNT       = BLOCK_TILE_M / WARP_TILE_M;
  constexpr int M_GROUP_COUNT_PER_WARP = WARP_TILE_M / 8;
  constexpr int N_GROUP_COUNT_PER_WARP = WARP_TILE_N / 16;

  T     A_mma_reg[M_GROUP_COUNT_PER_WARP][4];
  T     B_mma_reg[N_GROUP_COUNT_PER_WARP][8];
  float C_mma_reg[M_GROUP_COUNT_PER_WARP][N_GROUP_COUNT_PER_WARP][4] = {0};

  const int m_warp_offset = warp_id % M_MMA_WARP_COUNT * WARP_TILE_M;
  const int n_warp_offset = warp_id / M_MMA_WARP_COUNT * WARP_TILE_N;

  for (int k_loop_offset = 0; k_loop_offset < K; k_loop_offset += LOOP_TILE_K) {
    for (int loop = 0; loop < A_LDG_LOOP_COUNT; ++loop) {
      const int m = (loop * WARP_COUNT + warp_id) * 8 + lane_id % 8;
      const int k = lane_id / 8 * 4;
      FETCH_FLOAT2(A_ldg_reg[loop][0], A[OFFSET(m_block_offset + m, k_loop_offset + k, K)]);
    }
    for (int loop = 0; loop < B_LDG_LOOP_COUNT; ++loop) {
      const int k = (loop * WARP_COUNT + warp_id) % 2 * 8 + lane_id % 4 * 2;
      const int n = (loop * WARP_COUNT + warp_id) / 2 * 16 + lane_id / 4 * 2;
      FETCH_FLOAT(B_ldg_reg[loop][0][0], B[OFFSET(k_loop_offset + k, n_block_offset + n, K)]);
      FETCH_FLOAT(B_ldg_reg[loop][1][0], B[OFFSET(k_loop_offset + k + 1, n_block_offset + n, K)]);
    }
    for (int loop = 0; loop < A_LDG_LOOP_COUNT; ++loop) {
      const int m = (loop * WARP_COUNT + warp_id) * 8 + lane_id % 8;
      const int k = lane_id / 8 * 4;
      STORE_FLOAT2(A_sm[k / 8][m][k % 8], A_ldg_reg[loop][0]);
    }
    for (int loop = 0; loop < B_LDG_LOOP_COUNT; ++loop) {
      const int k = (loop * WARP_COUNT + warp_id) % 2 * 8 + lane_id % 4 * 2;
      const int n = (loop * WARP_COUNT + warp_id) / 2 * 16 + lane_id / 4 * 2;
      {
        T transpose[2] = {B_ldg_reg[loop][0][0], B_ldg_reg[loop][1][0]};
        STORE_FLOAT(B_sm[k / 8][n][k % 8], transpose[0]);
      }
      {
        T transpose[2] = {B_ldg_reg[loop][0][1], B_ldg_reg[loop][1][1]};
        STORE_FLOAT(B_sm[k / 8][n + 1][k % 8], transpose[0]);
      }
    }
    __syncthreads();
    // if (k_loop_offset == -1 && this_thread_can_log(0)) {
    //   const T* data = &A_sm[0][0][0];
    //   for (int i = 0; i < BLOCK_TILE_M * LOOP_TILE_K; ++i) {
    //     if (i % 64 == 0) {
    //       printf("\nA_sm, layer = %03d, ", i / 64);
    //     }
    //     printf("m%03dk%03d ", int(data[i]) / limit, int(data[i]) % limit);
    //   }
    //   data = &B_sm[0][0][0];
    //   for (int i = 0; i < BLOCK_TILE_N * LOOP_TILE_K; ++i) {
    //     if (i % 64 == 0) {
    //       printf("\nB_sm, layer = %03d, ", i / 64);
    //     }
    //     printf("n%03dk%03d ", int(data[i]) % limit, int(data[i]) / limit);
    //   }
    // }
    // The shape of A group is m8xk16
    for (int group = 0; group < M_GROUP_COUNT_PER_WARP; ++group) {
      uint32_t src = __cvta_generic_to_shared(&A_sm[lane_id / 8][m_warp_offset + group * 8 + lane_id % 8][0]);
      asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
                   : "=r"(*(uint32_t*)&A_mma_reg[group][0]), "=r"(*(uint32_t*)&A_mma_reg[group][2])
                   : "r"(src));
    }
    // The shape of B group is n16xk16
    for (int group = 0; group < N_GROUP_COUNT_PER_WARP; ++group) {
      uint32_t src = __cvta_generic_to_shared(&B_sm[lane_id / 16][n_warp_offset + group * 16 + lane_id % 16][0]);
      asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                   : "=r"(*(uint32_t*)&B_mma_reg[group][0]),
                     "=r"(*(uint32_t*)&B_mma_reg[group][2]),
                     "=r"(*(uint32_t*)&B_mma_reg[group][4]),
                     "=r"(*(uint32_t*)&B_mma_reg[group][6])
                   : "r"(src));
    }
    // for (int i = 0; k_loop_offset == -1 && i < 256; ++i) {
    //   if (this_thread_can_log(i)) {
    //     for (int group = 0; group < M_GROUP_COUNT_PER_WARP; ++group) {
    //       printf("\nA_mma_reg, thread = %03d, group = %03d, ", threadIdx.x, group);
    //       for (int r = 0; r < 4; ++r) {
    //         printf(" m%03dk%03d", int(A_mma_reg[group][r]) / limit, int(A_mma_reg[group][r]) % limit);
    //       }
    //     }
    //   }
    //   __syncthreads();
    // }
    // for (int i = 0; k_loop_offset == -1 && i < 256; ++i) {
    //   if (this_thread_can_log(i)) {
    //     for (int group = 0; group < N_GROUP_COUNT_PER_WARP; ++group) {
    //       printf("\nB_mma_reg, thread = %03d, group = %03d, ", threadIdx.x, group);
    //       for (int r = 0; r < 8; ++r) {
    //         printf(" n%03dk%03d", int(B_mma_reg[group][r]) % limit, int(B_mma_reg[group][r]) / limit);
    //       }
    //     }
    //   }
    //   __syncthreads();
    // }
    for (int mg = 0; mg < M_GROUP_COUNT_PER_WARP; ++mg) {
      for (int ng = 0; ng < N_GROUP_COUNT_PER_WARP; ++ng) {
        mma_m16n8k16_row_col(C_mma_reg[mg][ng], B_mma_reg[ng], A_mma_reg[mg], C_mma_reg[mg][ng]);
      }
    }
    __syncthreads();
  }

  for (int mg = 0; mg < M_GROUP_COUNT_PER_WARP; ++mg) {
    for (int ng = 0; ng < N_GROUP_COUNT_PER_WARP; ++ng) {
      // int m = m_block_offset + m_warp_offset + mg * 8;
      // int n = n_block_offset + n_warp_offset + ng * 16;
      T casted[4] = {
        C_mma_reg[mg][ng][0],
        C_mma_reg[mg][ng][1],
        C_mma_reg[mg][ng][2],
        C_mma_reg[mg][ng][3],
      };
      shfl_23_and_01(casted, 0x4, lane_id);
      T                store[4]           = {casted[0], casted[2], casted[1], casted[3]};
      static const int lane_2_n_offset[8] = {0, 8, 2, 10, 4, 12, 6, 14};
      // int              m                  = m_block_offset + m_loop * LOOP_TILE_M + lane_id % 4 * 2;
      // int              n                  = n_block_offset + n_warp_offset + lane_2_n_offset[lane_id / 4];
      int m = m_block_offset + m_warp_offset + mg * 8 + lane_id % 4 * 2;
      int n = n_block_offset + n_warp_offset + ng * 16 + lane_2_n_offset[lane_id / 4];
      STORE_FLOAT(C[OFFSET(m, n, N)], store[0]);
      STORE_FLOAT(C[OFFSET(m + 1, n, N)], store[2]);
    }
  }
}

template<typename T, int BLOCK_TILE_M, int BLOCK_TILE_N, int WARP_TILE_M, int WARP_TILE_N>
__global__ void fp16_mma_m16n8k16_ldmatrix_trans(const T* A, const T* B, T* C, int M, int N, int K)
{
  constexpr int WARP_COUNT   = BLOCK_TILE_M / WARP_TILE_M * BLOCK_TILE_N / WARP_TILE_N;
  constexpr int THREAD_COUNT = WARP_COUNT * 32;

  constexpr int LOOP_TILE_K = 16;
  // The 64 elements of type T in each 8x8 matrix are stored consecutively in a single layer.
  __shared__ T A_sm[2][BLOCK_TILE_M][LOOP_TILE_K / 2];
  __shared__ T B_sm[LOOP_TILE_K / 8][BLOCK_TILE_N / 8][64];

  static_assert(BLOCK_TILE_M * LOOP_TILE_K % THREAD_COUNT == 0);
  static_assert(BLOCK_TILE_M * LOOP_TILE_K / THREAD_COUNT % 8 == 0);
  constexpr int A_LDG_COUNT_PER_THREAD = BLOCK_TILE_M * LOOP_TILE_K / THREAD_COUNT;
  constexpr int A_LDG_LOOP_COUNT       = A_LDG_COUNT_PER_THREAD / 8;
  // clang-format off
  // This is the thread layout of the same warp that loads matrix A, where each thread reads M1xK8 elements of type T at a
  // loop iteration.
  // T0  T16
  // T1  T17
  // T2  T18
  // ... ...
  // T14 T30
  // T15 T31
  // clang-format on
  float A_ldg_reg[A_LDG_LOOP_COUNT][4];

  static_assert(BLOCK_TILE_N * LOOP_TILE_K % THREAD_COUNT == 0);
  static_assert(BLOCK_TILE_N * LOOP_TILE_K / THREAD_COUNT % 8 == 0);
  constexpr int B_LDG_COUNT_PER_THREAD = BLOCK_TILE_N * LOOP_TILE_K / THREAD_COUNT;
  constexpr int B_LDG_LOOP_COUNT       = B_LDG_COUNT_PER_THREAD / 8;
  // clang-format off
  // This is the thread layout of the same warp that loads matrix B, where each thread reads K1xN8 elements of type T at a
  // loop iteration.
  // T0  T16
  // T1  T17
  // T2  T18
  // ... ...
  // T14 T30
  // T15 T31
  // clang-format on
  float B_ldg_reg[B_LDG_LOOP_COUNT][4];

  const int m_block_offset = blockIdx.y * BLOCK_TILE_M;
  const int n_block_offset = blockIdx.x * BLOCK_TILE_N;

  const int warp_id = threadIdx.x / 32;
  const int lane_id = threadIdx.x % 32;

  constexpr int M_MMA_WARP_COUNT       = BLOCK_TILE_M / WARP_TILE_M;
  constexpr int M_GROUP_COUNT_PER_WARP = WARP_TILE_M / 8;
  constexpr int N_GROUP_COUNT_PER_WARP = WARP_TILE_N / 16;

  T     A_mma_reg[M_GROUP_COUNT_PER_WARP][4];
  T     B_mma_reg[N_GROUP_COUNT_PER_WARP][8];
  float C_mma_reg[M_GROUP_COUNT_PER_WARP][N_GROUP_COUNT_PER_WARP][4] = {0};

  const int m_warp_offset = warp_id % M_MMA_WARP_COUNT * WARP_TILE_M;
  const int n_warp_offset = warp_id / M_MMA_WARP_COUNT * WARP_TILE_N;

  for (int k_loop_offset = 0; k_loop_offset < K; k_loop_offset += LOOP_TILE_K) {
#pragma unroll
    for (int loop = 0; loop < A_LDG_LOOP_COUNT; ++loop) {
      const int m = (loop * WARP_COUNT + warp_id) * 16 + lane_id % 16;
      const int k = lane_id / 16 * 8;
      FETCH_FLOAT4(A_ldg_reg[loop][0], A[OFFSET(m_block_offset + m, k_loop_offset + k, K)]);
    }
#pragma unroll
    for (int loop = 0; loop < B_LDG_LOOP_COUNT; ++loop) {
      const int k = lane_id % 16;
      const int n = (loop * WARP_COUNT + warp_id) * 16 + lane_id / 16 * 8;
      FETCH_FLOAT4(B_ldg_reg[loop][0], B[OFFSET(k_loop_offset + k, n_block_offset + n, K)]);
    }
#pragma unroll
    for (int loop = 0; loop < A_LDG_LOOP_COUNT; ++loop) {
      const int m = (loop * WARP_COUNT + warp_id) * 16 + lane_id % 16;
      const int k = lane_id / 16 * 8;
      STORE_FLOAT4(A_sm[k / 8][m][0], A_ldg_reg[loop][0]);
    }
#pragma unroll
    for (int loop = 0; loop < B_LDG_LOOP_COUNT; ++loop) {
      const int k = lane_id % 16;
      const int n = (loop * WARP_COUNT + warp_id) * 16 + lane_id / 16 * 8;
      STORE_FLOAT4(B_sm[k / 8][n / 8][k % 8 * 8], B_ldg_reg[loop][0]);
    }
    __syncthreads();
    // The shape of A group is m8xk16
#pragma unroll
    for (int group = 0; group < M_GROUP_COUNT_PER_WARP; ++group) {
      uint32_t src = __cvta_generic_to_shared(&A_sm[lane_id / 8][m_warp_offset + group * 8 + lane_id % 8][0]);
      asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
                   : "=r"(*(uint32_t*)&A_mma_reg[group][0]), "=r"(*(uint32_t*)&A_mma_reg[group][2])
                   : "r"(src));
    }
    // The shape of B group is n16xk16
#pragma unroll
    for (int group = 0; group < N_GROUP_COUNT_PER_WARP; ++group) {
      // uint32_t src = __cvta_generic_to_shared(&B_sm[lane_id / 16][n_warp_offset + group * 16 + lane_id % 16][0]);
      uint32_t src = __cvta_generic_to_shared(
        &B_sm[lane_id % 16 / 8][group * 2 + (n_warp_offset + lane_id / 16 * 8) / 8][lane_id % 8 * 8]);
      asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
                   : "=r"(*(uint32_t*)&B_mma_reg[group][0]),
                     "=r"(*(uint32_t*)&B_mma_reg[group][4]),
                     "=r"(*(uint32_t*)&B_mma_reg[group][2]),
                     "=r"(*(uint32_t*)&B_mma_reg[group][6])
                   : "r"(src));
    }
#pragma unroll
    for (int mg = 0; mg < M_GROUP_COUNT_PER_WARP; ++mg) {
#pragma unroll
      for (int ng = 0; ng < N_GROUP_COUNT_PER_WARP; ++ng) {
        mma_m16n8k16_row_col(C_mma_reg[mg][ng], B_mma_reg[ng], A_mma_reg[mg], C_mma_reg[mg][ng]);
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int mg = 0; mg < M_GROUP_COUNT_PER_WARP; ++mg) {
#pragma unroll
    for (int ng = 0; ng < N_GROUP_COUNT_PER_WARP; ++ng) {
      // int m = m_block_offset + m_warp_offset + mg * 8;
      // int n = n_block_offset + n_warp_offset + ng * 16;
      T casted[4] = {
        C_mma_reg[mg][ng][0],
        C_mma_reg[mg][ng][1],
        C_mma_reg[mg][ng][2],
        C_mma_reg[mg][ng][3],
      };
      shfl_23_and_01(casted, 0x4, lane_id);
      T                store[4]           = {casted[0], casted[2], casted[1], casted[3]};
      static const int lane_2_n_offset[8] = {0, 8, 2, 10, 4, 12, 6, 14};
      int              m                  = m_block_offset + m_warp_offset + mg * 8 + lane_id % 4 * 2;
      int              n                  = n_block_offset + n_warp_offset + ng * 16 + lane_2_n_offset[lane_id / 4];
      STORE_FLOAT(C[OFFSET(m, n, N)], store[0]);
      STORE_FLOAT(C[OFFSET(m + 1, n, N)], store[2]);
    }
  }
}

template<typename T, int BLOCK_TILE_M, int BLOCK_TILE_N, int WARP_TILE_M, int WARP_TILE_N>
__global__ void fp16_mma_m16n8k16_ldmatrix_trans__overlap_global_2_sm(const T* A, const T* B, T* C, int M, int N, int K)
{
  constexpr int WARP_COUNT   = BLOCK_TILE_M / WARP_TILE_M * BLOCK_TILE_N / WARP_TILE_N;
  constexpr int THREAD_COUNT = WARP_COUNT * 32;

  constexpr int LOOP_TILE_K     = 16;
  constexpr int LDG_BUFFER_SIZE = 2;
  // The 64 elements of type T in each 8x8 matrix are stored consecutively in a single layer.
  __shared__ T A_sm[LDG_BUFFER_SIZE][2][BLOCK_TILE_M][LOOP_TILE_K / 2];
  __shared__ T B_sm[LDG_BUFFER_SIZE][LOOP_TILE_K / 8][BLOCK_TILE_N / 8][8][8];

  static_assert(BLOCK_TILE_M * LOOP_TILE_K % THREAD_COUNT == 0);
  static_assert(BLOCK_TILE_M * LOOP_TILE_K / THREAD_COUNT % 8 == 0);
  constexpr int A_LDG_COUNT_PER_THREAD = BLOCK_TILE_M * LOOP_TILE_K / THREAD_COUNT;
  constexpr int A_LDG_LOOP_COUNT       = A_LDG_COUNT_PER_THREAD / 8;
  // clang-format off
  // This is the thread layout of the same warp that loads matrix A, where each thread reads M1xK8 elements of type T at a
  // loop iteration.
  // T0  T16
  // T1  T17
  // T2  T18
  // ... ...
  // T14 T30
  // T15 T31
  // clang-format on
  float A_ldg_reg[A_LDG_LOOP_COUNT][4];

  static_assert(BLOCK_TILE_N * LOOP_TILE_K % THREAD_COUNT == 0);
  static_assert(BLOCK_TILE_N * LOOP_TILE_K / THREAD_COUNT % 8 == 0);
  constexpr int B_LDG_COUNT_PER_THREAD = BLOCK_TILE_N * LOOP_TILE_K / THREAD_COUNT;
  constexpr int B_LDG_LOOP_COUNT       = B_LDG_COUNT_PER_THREAD / 8;
  // clang-format off
  // This is the thread layout of the same warp that loads matrix B, where each thread reads K1xN8 elements of type T at a
  // loop iteration.
  // T0  T16
  // T1  T17
  // T2  T18
  // ... ...
  // T14 T30
  // T15 T31
  // clang-format on
  float B_ldg_reg[B_LDG_LOOP_COUNT][4];

  const int m_block_offset = blockIdx.y * BLOCK_TILE_M;
  const int n_block_offset = blockIdx.x * BLOCK_TILE_N;

  const int warp_id = threadIdx.x / 32;
  const int lane_id = threadIdx.x % 32;

  constexpr int M_MMA_WARP_COUNT       = BLOCK_TILE_M / WARP_TILE_M;
  constexpr int M_GROUP_COUNT_PER_WARP = WARP_TILE_M / 8;
  constexpr int N_GROUP_COUNT_PER_WARP = WARP_TILE_N / 16;

  T     A_mma_reg[M_GROUP_COUNT_PER_WARP][4];
  T     B_mma_reg[N_GROUP_COUNT_PER_WARP][8];
  float C_mma_reg[M_GROUP_COUNT_PER_WARP][N_GROUP_COUNT_PER_WARP][4] = {0};

  const int m_warp_offset = warp_id % M_MMA_WARP_COUNT * WARP_TILE_M;
  const int n_warp_offset = warp_id / M_MMA_WARP_COUNT * WARP_TILE_N;

#define global_2_ldg_reg()                                                                                             \
  {                                                                                                                    \
    for (int loop = 0; loop < A_LDG_LOOP_COUNT; ++loop) {                                                              \
      const int m = (loop * WARP_COUNT + warp_id) * 16 + lane_id % 16;                                                 \
      const int k = lane_id / 16 * 8;                                                                                  \
      FETCH_FLOAT4(A_ldg_reg[loop][0], A[OFFSET(m_block_offset + m, k_loop_offset + k, K)]);                           \
    }                                                                                                                  \
    for (int loop = 0; loop < B_LDG_LOOP_COUNT; ++loop) {                                                              \
      const int k = lane_id % 16;                                                                                      \
      const int n = (loop * WARP_COUNT + warp_id) * 16 + lane_id / 16 * 8;                                             \
      FETCH_FLOAT4(B_ldg_reg[loop][0], B[OFFSET(k_loop_offset + k, n_block_offset + n, K)]);                           \
    }                                                                                                                  \
  }

#define ldg_reg_2_sm()                                                                                                 \
  {                                                                                                                    \
    for (int loop = 0; loop < A_LDG_LOOP_COUNT; ++loop) {                                                              \
      const int m = (loop * WARP_COUNT + warp_id) * 16 + lane_id % 16;                                                 \
      const int k = lane_id / 16 * 8;                                                                                  \
      STORE_FLOAT4(A_sm[LDG_BUFFER_INDEX][k / 8][m][0], A_ldg_reg[loop][0]);                                           \
    }                                                                                                                  \
    for (int loop = 0; loop < B_LDG_LOOP_COUNT; ++loop) {                                                              \
      const int k = lane_id % 16;                                                                                      \
      const int n = (loop * WARP_COUNT + warp_id) * 16 + lane_id / 16 * 8;                                             \
      STORE_FLOAT4(B_sm[LDG_BUFFER_INDEX][k / 8][n / 8][k % 8][0], B_ldg_reg[loop][0]);                                \
    }                                                                                                                  \
  }

  int LDG_BUFFER_INDEX = 0;
  int k_loop_offset    = 0;
  global_2_ldg_reg();
  ldg_reg_2_sm();
  __syncthreads();

  k_loop_offset = LOOP_TILE_K;
  while (k_loop_offset < K) {
    global_2_ldg_reg();
    // The shape of A group is m8xk16
    for (int group = 0; group < M_GROUP_COUNT_PER_WARP; ++group) {
      uint32_t src =
        __cvta_generic_to_shared(&A_sm[LDG_BUFFER_INDEX][lane_id / 8][m_warp_offset + group * 8 + lane_id % 8][0]);
      asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
                   : "=r"(*(uint32_t*)&A_mma_reg[group][0]), "=r"(*(uint32_t*)&A_mma_reg[group][2])
                   : "r"(src));
    }
    // The shape of B group is n16xk16
    for (int group = 0; group < N_GROUP_COUNT_PER_WARP; ++group) {
      uint32_t src = __cvta_generic_to_shared(
        &B_sm[LDG_BUFFER_INDEX][lane_id % 16 / 8][(n_warp_offset + group * 16 + lane_id / 16 * 8) / 8][lane_id % 8][0]);
      asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
                   : "=r"(*(uint32_t*)&B_mma_reg[group][0]),
                     "=r"(*(uint32_t*)&B_mma_reg[group][4]),
                     "=r"(*(uint32_t*)&B_mma_reg[group][2]),
                     "=r"(*(uint32_t*)&B_mma_reg[group][6])
                   : "r"(src));
    }
    for (int mg = 0; mg < M_GROUP_COUNT_PER_WARP; ++mg) {
      for (int ng = 0; ng < N_GROUP_COUNT_PER_WARP; ++ng) {
        mma_m16n8k16_row_col(C_mma_reg[mg][ng], B_mma_reg[ng], A_mma_reg[mg], C_mma_reg[mg][ng]);
      }
    }
    LDG_BUFFER_INDEX ^= 1;
    ldg_reg_2_sm();

    __syncthreads();
    k_loop_offset += LOOP_TILE_K;
  }
  {
    for (int group = 0; group < M_GROUP_COUNT_PER_WARP; ++group) {
      uint32_t src =
        __cvta_generic_to_shared(&A_sm[LDG_BUFFER_INDEX][lane_id / 8][m_warp_offset + group * 8 + lane_id % 8][0]);
      asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
                   : "=r"(*(uint32_t*)&A_mma_reg[group][0]), "=r"(*(uint32_t*)&A_mma_reg[group][2])
                   : "r"(src));
    }
    // The shape of B group is n16xk16
    for (int group = 0; group < N_GROUP_COUNT_PER_WARP; ++group) {
      uint32_t src = __cvta_generic_to_shared(
        &B_sm[LDG_BUFFER_INDEX][lane_id % 16 / 8][(n_warp_offset + group * 16 + lane_id / 16 * 8) / 8][lane_id % 8][0]);
      asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
                   : "=r"(*(uint32_t*)&B_mma_reg[group][0]),
                     "=r"(*(uint32_t*)&B_mma_reg[group][4]),
                     "=r"(*(uint32_t*)&B_mma_reg[group][2]),
                     "=r"(*(uint32_t*)&B_mma_reg[group][6])
                   : "r"(src));
    }
    for (int mg = 0; mg < M_GROUP_COUNT_PER_WARP; ++mg) {
      for (int ng = 0; ng < N_GROUP_COUNT_PER_WARP; ++ng) {
        mma_m16n8k16_row_col(C_mma_reg[mg][ng], B_mma_reg[ng], A_mma_reg[mg], C_mma_reg[mg][ng]);
      }
    }
  }

  for (int mg = 0; mg < M_GROUP_COUNT_PER_WARP; ++mg) {
    for (int ng = 0; ng < N_GROUP_COUNT_PER_WARP; ++ng) {
      // int m = m_block_offset + m_warp_offset + mg * 8;
      // int n = n_block_offset + n_warp_offset + ng * 16;
      T casted[4] = {
        C_mma_reg[mg][ng][0],
        C_mma_reg[mg][ng][1],
        C_mma_reg[mg][ng][2],
        C_mma_reg[mg][ng][3],
      };
      shfl_23_and_01(casted, 0x4, lane_id);
      T                store[4]           = {casted[0], casted[2], casted[1], casted[3]};
      static const int lane_2_n_offset[8] = {0, 8, 2, 10, 4, 12, 6, 14};
      int              m                  = m_block_offset + m_warp_offset + mg * 8 + lane_id % 4 * 2;
      int              n                  = n_block_offset + n_warp_offset + ng * 16 + lane_2_n_offset[lane_id / 4];
      STORE_FLOAT(C[OFFSET(m, n, N)], store[0]);
      STORE_FLOAT(C[OFFSET(m + 1, n, N)], store[2]);
    }
  }
#undef global_2_ldg_reg
#undef ldg_reg_2_sm
}

template<typename T, int BLOCK_TILE_M, int BLOCK_TILE_N, int WARP_TILE_M, int WARP_TILE_N>
__global__ void
fp16_mma_m16n8k16_ldmatrix_trans__overlap_global_2_sm__quadra_buffer(const T* A, const T* B, T* C, int M, int N, int K)
{
  constexpr int WARP_COUNT   = BLOCK_TILE_M / WARP_TILE_M * BLOCK_TILE_N / WARP_TILE_N;
  constexpr int THREAD_COUNT = WARP_COUNT * 32;

  constexpr int LOOP_TILE_K            = 16;
  constexpr int LDG_SM_BUFFER_SIZE     = 4;
  constexpr int LDG_REG_BUFFER_SIZE    = 2;
  constexpr int LDG_REG_BUFFER_INDEX_0 = 0;
  constexpr int LDG_REG_BUFFER_INDEX_1 = 1;
  // The 64 elements of type T in each 8x8 matrix are stored consecutively in a single layer.
  __shared__ T A_sm[LDG_SM_BUFFER_SIZE][2][BLOCK_TILE_M][LOOP_TILE_K / 2];
  __shared__ T B_sm[LDG_SM_BUFFER_SIZE][LOOP_TILE_K / 8][BLOCK_TILE_N / 8][8][8];

  static_assert(BLOCK_TILE_M * LOOP_TILE_K % THREAD_COUNT == 0);
  static_assert(BLOCK_TILE_M * LOOP_TILE_K / THREAD_COUNT % 8 == 0);
  constexpr int A_LDG_COUNT_PER_THREAD = BLOCK_TILE_M * LOOP_TILE_K / THREAD_COUNT;
  constexpr int A_LDG_LOOP_COUNT       = A_LDG_COUNT_PER_THREAD / 8;
  // clang-format off
  // This is the thread layout of the same warp that loads matrix A, where each thread reads M1xK8 elements of type T at a
  // loop iteration.
  // T0  T16
  // T1  T17
  // T2  T18
  // ... ...
  // T14 T30
  // T15 T31
  // clang-format on
  float A_ldg_reg[LDG_REG_BUFFER_SIZE][A_LDG_LOOP_COUNT][4];

  static_assert(BLOCK_TILE_N * LOOP_TILE_K % THREAD_COUNT == 0);
  static_assert(BLOCK_TILE_N * LOOP_TILE_K / THREAD_COUNT % 8 == 0);
  constexpr int B_LDG_COUNT_PER_THREAD = BLOCK_TILE_N * LOOP_TILE_K / THREAD_COUNT;
  constexpr int B_LDG_LOOP_COUNT       = B_LDG_COUNT_PER_THREAD / 8;
  // clang-format off
  // This is the thread layout of the same warp that loads matrix B, where each thread reads K1xN8 elements of type T at a
  // loop iteration.
  // T0  T16
  // T1  T17
  // T2  T18
  // ... ...
  // T14 T30
  // T15 T31
  // clang-format on
  float B_ldg_reg[LDG_REG_BUFFER_SIZE][B_LDG_LOOP_COUNT][4];

  const int m_block_offset = blockIdx.y * BLOCK_TILE_M;
  const int n_block_offset = blockIdx.x * BLOCK_TILE_N;

  const int warp_id                 = threadIdx.x / 32;
  const int lane_id                 = threadIdx.x % 32;
  const int transposed_lane_id_mask = (lane_id / 8 == 0 || lane_id / 8 == 3) ? 0x00 : 0x18;
  const int transposed_lane_id      = lane_id ^ transposed_lane_id_mask;

  constexpr int M_MMA_WARP_COUNT       = BLOCK_TILE_M / WARP_TILE_M;
  constexpr int M_GROUP_COUNT_PER_WARP = WARP_TILE_M / 8;
  constexpr int N_GROUP_COUNT_PER_WARP = WARP_TILE_N / 16;

  constexpr int MMA_REG_BUFFER_SIZE    = 2;
  constexpr int MMA_REG_BUFFER_INDEX_0 = 0;
  constexpr int MMA_REG_BUFFER_INDEX_1 = 1;
  T             A_mma_reg[MMA_REG_BUFFER_SIZE][M_GROUP_COUNT_PER_WARP][4];
  T             B_mma_reg[MMA_REG_BUFFER_SIZE][N_GROUP_COUNT_PER_WARP][8];
  float         C_mma_reg[M_GROUP_COUNT_PER_WARP][N_GROUP_COUNT_PER_WARP][4] = {0};

  const int m_warp_offset = warp_id % M_MMA_WARP_COUNT * WARP_TILE_M;
  const int n_warp_offset = warp_id / M_MMA_WARP_COUNT * WARP_TILE_N;

#define global_2_ldg_reg(k_loop_offset, ldg_reg_buffer_index)                                                          \
  {                                                                                                                    \
    for (int loop = 0; loop < A_LDG_LOOP_COUNT; ++loop) {                                                              \
      const int m = (loop * WARP_COUNT + warp_id) * 16 + lane_id % 16;                                                 \
      const int k = lane_id / 16 * 8;                                                                                  \
      FETCH_FLOAT4(A_ldg_reg[ldg_reg_buffer_index][loop][0], A[OFFSET(m_block_offset + m, (k_loop_offset) + k, K)]);   \
    }                                                                                                                  \
    for (int loop = 0; loop < B_LDG_LOOP_COUNT; ++loop) {                                                              \
      const int k = lane_id % 16;                                                                                      \
      const int n = (loop * WARP_COUNT + warp_id) * 16 + lane_id / 16 * 8;                                             \
      FETCH_FLOAT4(B_ldg_reg[ldg_reg_buffer_index][loop][0], B[OFFSET((k_loop_offset) + k, n_block_offset + n, K)]);   \
    }                                                                                                                  \
  }

#define ldg_reg_2_sm(ldg_sm_buffer_index, ldg_reg_buffer_index)                                                        \
  {                                                                                                                    \
    for (int loop = 0; loop < A_LDG_LOOP_COUNT; ++loop) {                                                              \
      const int m = (loop * WARP_COUNT + warp_id) * 16 + lane_id % 16;                                                 \
      const int k = lane_id / 16 * 8;                                                                                  \
      STORE_FLOAT4(A_sm[ldg_sm_buffer_index][k / 8][m][0], A_ldg_reg[ldg_reg_buffer_index][loop][0]);                  \
    }                                                                                                                  \
    for (int loop = 0; loop < B_LDG_LOOP_COUNT; ++loop) {                                                              \
      const int k = lane_id % 16;                                                                                      \
      const int n = (loop * WARP_COUNT + warp_id) * 16 + lane_id / 16 * 8;                                             \
      STORE_FLOAT4(B_sm[ldg_sm_buffer_index][k / 8][n / 8][k % 8][0], B_ldg_reg[ldg_reg_buffer_index][loop][0]);       \
    }                                                                                                                  \
  }

#define sm_2_A_mma_reg(ldg_sm_buffer_index, mma_reg_buffer_index, group)                                               \
  /* for (int group = 0; group < M_GROUP_COUNT_PER_WARP; ++group) */ {                                                 \
    uint32_t src =                                                                                                     \
      __cvta_generic_to_shared(&A_sm[ldg_sm_buffer_index][lane_id / 8][m_warp_offset + (group) * 8 + lane_id % 8][0]); \
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"                                            \
                 : "=r"(*(uint32_t*)&A_mma_reg[mma_reg_buffer_index][group][0]),                                       \
                   "=r"(*(uint32_t*)&A_mma_reg[mma_reg_buffer_index][group][2])                                        \
                 : "r"(src));                                                                                          \
  }

#define sm_2_B_mma_reg(ldg_sm_buffer_index, mma_reg_buffer_index, group)                                               \
  /*for (int group = 0; group < N_GROUP_COUNT_PER_WARP; ++group) */ {                                                  \
    uint32_t src = __cvta_generic_to_shared(                                                                           \
      &B_sm[ldg_sm_buffer_index][transposed_lane_id % 16 / 8]                                                          \
           [(n_warp_offset + (group) * 16 + transposed_lane_id / 16 * 8) / 8][transposed_lane_id % 8][0]);             \
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"                              \
                 : "=r"(*(uint32_t*)&B_mma_reg[mma_reg_buffer_index][group][0]),                                       \
                   "=r"(*(uint32_t*)&B_mma_reg[mma_reg_buffer_index][group][2]),                                       \
                   "=r"(*(uint32_t*)&B_mma_reg[mma_reg_buffer_index][group][4]),                                       \
                   "=r"(*(uint32_t*)&B_mma_reg[mma_reg_buffer_index][group][6])                                        \
                 : "r"(src));                                                                                          \
  }

#define mma_m16n8k16_row_col_macro(d, a, b, c)                                                                         \
  uint32_t const* A = reinterpret_cast<uint32_t const*>(&a);                                                           \
  uint32_t const* B = reinterpret_cast<uint32_t const*>(&b);                                                           \
  float const*    C = reinterpret_cast<float const*>(&c);                                                              \
  float*          D = reinterpret_cast<float*>(&d);                                                                    \
  if constexpr (std::is_same<T, half>::value) {                                                                        \
    asm volatile(                                                                                                      \
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32  {%0,%1,%2,%3}, "                                             \
      "{%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"                                                                   \
      : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])                                                                 \
      : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3])); \
  }                                                                                                                    \
  else if constexpr (std::is_same<T, __nv_bfloat16>::value) {                                                          \
    asm volatile(                                                                                                      \
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32  {%0,%1,%2,%3}, "                                           \
      "{%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"                                                                   \
      : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])                                                                 \
      : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3])); \
  }                                                                                                                    \
  else {                                                                                                               \
    static_assert(std::is_same<T, half>::value == false && std::is_same<T, __nv_bfloat16>::value == false);            \
  }

  global_2_ldg_reg(0, LDG_REG_BUFFER_INDEX_0);
  global_2_ldg_reg(LOOP_TILE_K, LDG_REG_BUFFER_INDEX_1);
  ldg_reg_2_sm(0, LDG_REG_BUFFER_INDEX_0);
  ldg_reg_2_sm(1, LDG_REG_BUFFER_INDEX_1);
  __syncthreads();

  int LDG_SM_BUFFER_INDEX = 0;
  int k_loop_offset       = LOOP_TILE_K * 2;

  for (int mg = 0; mg < M_GROUP_COUNT_PER_WARP; ++mg) {
    sm_2_A_mma_reg(LDG_SM_BUFFER_INDEX, MMA_REG_BUFFER_INDEX_0, mg);
  }
  for (int ng = 0; ng < N_GROUP_COUNT_PER_WARP; ++ng) {
    sm_2_B_mma_reg(LDG_SM_BUFFER_INDEX, MMA_REG_BUFFER_INDEX_0, ng);
  }

  global_2_ldg_reg(k_loop_offset, LDG_REG_BUFFER_INDEX_0);
  global_2_ldg_reg(k_loop_offset + LOOP_TILE_K, LDG_REG_BUFFER_INDEX_1);

  while (k_loop_offset + LOOP_TILE_K * 2 < K) {
    static_assert(M_GROUP_COUNT_PER_WARP == N_GROUP_COUNT_PER_WARP);
    for (int mg = 0; mg < M_GROUP_COUNT_PER_WARP; ++mg) {
      for (int ng = 0; ng < M_GROUP_COUNT_PER_WARP; ++ng) {
        mma_m16n8k16_row_col_macro(C_mma_reg[mg][ng],
                                   B_mma_reg[MMA_REG_BUFFER_INDEX_0][ng],
                                   A_mma_reg[MMA_REG_BUFFER_INDEX_0][mg],
                                   C_mma_reg[mg][ng]);
      }
      sm_2_A_mma_reg(LDG_SM_BUFFER_INDEX + 1, MMA_REG_BUFFER_INDEX_1, mg);
      sm_2_B_mma_reg(LDG_SM_BUFFER_INDEX + 1, MMA_REG_BUFFER_INDEX_1, mg);
    }
    for (int ng = M_GROUP_COUNT_PER_WARP; ng < N_GROUP_COUNT_PER_WARP; ++ng) {
      for (int mg = 0; mg < M_GROUP_COUNT_PER_WARP; ++mg) {
        mma_m16n8k16_row_col_macro(C_mma_reg[mg][ng],
                                   B_mma_reg[MMA_REG_BUFFER_INDEX_0][ng],
                                   A_mma_reg[MMA_REG_BUFFER_INDEX_0][mg],
                                   C_mma_reg[mg][ng]);
      }
      sm_2_A_mma_reg(LDG_SM_BUFFER_INDEX + 1, MMA_REG_BUFFER_INDEX_1, ng);
      sm_2_B_mma_reg(LDG_SM_BUFFER_INDEX + 1, MMA_REG_BUFFER_INDEX_1, ng);
    }

    LDG_SM_BUFFER_INDEX ^= 2;
    k_loop_offset += LOOP_TILE_K * 2;
    ldg_reg_2_sm(LDG_SM_BUFFER_INDEX, LDG_REG_BUFFER_INDEX_0);
    global_2_ldg_reg(k_loop_offset, LDG_REG_BUFFER_INDEX_0);
    ldg_reg_2_sm(LDG_SM_BUFFER_INDEX + 1, LDG_REG_BUFFER_INDEX_1);
    global_2_ldg_reg(k_loop_offset + LOOP_TILE_K, LDG_REG_BUFFER_INDEX_1);

    __syncthreads();

    static_assert(M_GROUP_COUNT_PER_WARP == N_GROUP_COUNT_PER_WARP);
    for (int mg = 0; mg < M_GROUP_COUNT_PER_WARP; ++mg) {
      for (int ng = 0; ng < M_GROUP_COUNT_PER_WARP; ++ng) {
        mma_m16n8k16_row_col_macro(C_mma_reg[mg][ng],
                                   B_mma_reg[MMA_REG_BUFFER_INDEX_1][ng],
                                   A_mma_reg[MMA_REG_BUFFER_INDEX_1][mg],
                                   C_mma_reg[mg][ng]);
      }
      sm_2_A_mma_reg(LDG_SM_BUFFER_INDEX, MMA_REG_BUFFER_INDEX_0, mg);
      sm_2_B_mma_reg(LDG_SM_BUFFER_INDEX, MMA_REG_BUFFER_INDEX_0, mg);
    }
    for (int ng = M_GROUP_COUNT_PER_WARP; ng < N_GROUP_COUNT_PER_WARP; ++ng) {
      for (int mg = 0; mg < M_GROUP_COUNT_PER_WARP; ++mg) {
        mma_m16n8k16_row_col_macro(C_mma_reg[mg][ng],
                                   B_mma_reg[MMA_REG_BUFFER_INDEX_1][ng],
                                   A_mma_reg[MMA_REG_BUFFER_INDEX_1][mg],
                                   C_mma_reg[mg][ng]);
      }
      sm_2_A_mma_reg(LDG_SM_BUFFER_INDEX, MMA_REG_BUFFER_INDEX_0, ng);
      sm_2_B_mma_reg(LDG_SM_BUFFER_INDEX, MMA_REG_BUFFER_INDEX_0, ng);
    }
  }
  {
    static_assert(M_GROUP_COUNT_PER_WARP == N_GROUP_COUNT_PER_WARP);
    for (int mg = 0; mg < M_GROUP_COUNT_PER_WARP; ++mg) {
      for (int ng = 0; ng < M_GROUP_COUNT_PER_WARP; ++ng) {
        mma_m16n8k16_row_col_macro(C_mma_reg[mg][ng],
                                   B_mma_reg[MMA_REG_BUFFER_INDEX_0][ng],
                                   A_mma_reg[MMA_REG_BUFFER_INDEX_0][mg],
                                   C_mma_reg[mg][ng]);
      }
      sm_2_A_mma_reg(LDG_SM_BUFFER_INDEX + 1, MMA_REG_BUFFER_INDEX_1, mg);
      sm_2_B_mma_reg(LDG_SM_BUFFER_INDEX + 1, MMA_REG_BUFFER_INDEX_1, mg);
    }
    for (int ng = M_GROUP_COUNT_PER_WARP; ng < N_GROUP_COUNT_PER_WARP; ++ng) {
      for (int mg = 0; mg < M_GROUP_COUNT_PER_WARP; ++mg) {
        mma_m16n8k16_row_col_macro(C_mma_reg[mg][ng],
                                   B_mma_reg[MMA_REG_BUFFER_INDEX_0][ng],
                                   A_mma_reg[MMA_REG_BUFFER_INDEX_0][mg],
                                   C_mma_reg[mg][ng]);
      }
      sm_2_A_mma_reg(LDG_SM_BUFFER_INDEX + 1, MMA_REG_BUFFER_INDEX_1, ng);
      sm_2_B_mma_reg(LDG_SM_BUFFER_INDEX + 1, MMA_REG_BUFFER_INDEX_1, ng);
    }

    LDG_SM_BUFFER_INDEX ^= 2;
    k_loop_offset += LOOP_TILE_K * 2;
    ldg_reg_2_sm(LDG_SM_BUFFER_INDEX, LDG_REG_BUFFER_INDEX_0);
    ldg_reg_2_sm(LDG_SM_BUFFER_INDEX + 1, LDG_REG_BUFFER_INDEX_1);

    __syncthreads();

    static_assert(M_GROUP_COUNT_PER_WARP == N_GROUP_COUNT_PER_WARP);
    for (int mg = 0; mg < M_GROUP_COUNT_PER_WARP; ++mg) {
      for (int ng = 0; ng < M_GROUP_COUNT_PER_WARP; ++ng) {
        mma_m16n8k16_row_col_macro(C_mma_reg[mg][ng],
                                   B_mma_reg[MMA_REG_BUFFER_INDEX_1][ng],
                                   A_mma_reg[MMA_REG_BUFFER_INDEX_1][mg],
                                   C_mma_reg[mg][ng]);
      }
      sm_2_A_mma_reg(LDG_SM_BUFFER_INDEX, MMA_REG_BUFFER_INDEX_0, mg);
      sm_2_B_mma_reg(LDG_SM_BUFFER_INDEX, MMA_REG_BUFFER_INDEX_0, mg);
    }
    for (int ng = M_GROUP_COUNT_PER_WARP; ng < N_GROUP_COUNT_PER_WARP; ++ng) {
      for (int mg = 0; mg < M_GROUP_COUNT_PER_WARP; ++mg) {
        mma_m16n8k16_row_col_macro(C_mma_reg[mg][ng],
                                   B_mma_reg[MMA_REG_BUFFER_INDEX_1][ng],
                                   A_mma_reg[MMA_REG_BUFFER_INDEX_1][mg],
                                   C_mma_reg[mg][ng]);
      }
      sm_2_A_mma_reg(LDG_SM_BUFFER_INDEX, MMA_REG_BUFFER_INDEX_0, ng);
      sm_2_B_mma_reg(LDG_SM_BUFFER_INDEX, MMA_REG_BUFFER_INDEX_0, ng);
    }
  }
  {
    static_assert(M_GROUP_COUNT_PER_WARP == N_GROUP_COUNT_PER_WARP);
    for (int mg = 0; mg < M_GROUP_COUNT_PER_WARP; ++mg) {
      for (int ng = 0; ng < N_GROUP_COUNT_PER_WARP; ++ng) {
        mma_m16n8k16_row_col_macro(C_mma_reg[mg][ng],
                                   B_mma_reg[MMA_REG_BUFFER_INDEX_0][ng],
                                   A_mma_reg[MMA_REG_BUFFER_INDEX_0][mg],
                                   C_mma_reg[mg][ng]);
      }
      sm_2_A_mma_reg(LDG_SM_BUFFER_INDEX + 1, MMA_REG_BUFFER_INDEX_1, mg);
      sm_2_B_mma_reg(LDG_SM_BUFFER_INDEX + 1, MMA_REG_BUFFER_INDEX_1, mg);
    }

    for (int mg = 0; mg < M_GROUP_COUNT_PER_WARP; ++mg) {
      for (int ng = 0; ng < N_GROUP_COUNT_PER_WARP; ++ng) {
        mma_m16n8k16_row_col_macro(C_mma_reg[mg][ng],
                                   B_mma_reg[MMA_REG_BUFFER_INDEX_1][ng],
                                   A_mma_reg[MMA_REG_BUFFER_INDEX_1][mg],
                                   C_mma_reg[mg][ng]);
      }
    }
  }

  for (int mg = 0; mg < M_GROUP_COUNT_PER_WARP; ++mg) {
    for (int ng = 0; ng < N_GROUP_COUNT_PER_WARP; ++ng) {
      // int m = m_block_offset + m_warp_offset + mg * 8;
      // int n = n_block_offset + n_warp_offset + ng * 16;
      T casted[4] = {
        C_mma_reg[mg][ng][0],
        C_mma_reg[mg][ng][1],
        C_mma_reg[mg][ng][2],
        C_mma_reg[mg][ng][3],
      };
      shfl_23_and_01(casted, 0x4, lane_id);
      T                store[4]           = {casted[0], casted[2], casted[1], casted[3]};
      static const int lane_2_n_offset[8] = {0, 8, 2, 10, 4, 12, 6, 14};
      int              m                  = m_block_offset + m_warp_offset + mg * 8 + lane_id % 4 * 2;
      int              n                  = n_block_offset + n_warp_offset + ng * 16 + lane_2_n_offset[lane_id / 4];
      STORE_FLOAT(C[OFFSET(m, n, N)], store[0]);
      STORE_FLOAT(C[OFFSET(m + 1, n, N)], store[2]);
    }
  }
#undef global_2_ldg_reg
#undef ldg_reg_2_sm
#undef sm_2_A_mma_reg
#undef sm_2_B_mma_reg
#undef mma_m16n8k16_row_col_macro
}

template<typename T, int BLOCK_TILE_M, int BLOCK_TILE_N, int WARP_TILE_M, int WARP_TILE_N>
__global__ void fp16_mma_m16n8k16_ldmatrix_trans__overlap_global_2_sm__quadra_buffer__reduce_instructions(
  const T* A, const T* B, T* C, int M, int N, int K)
{
  constexpr int WARP_COUNT   = BLOCK_TILE_M / WARP_TILE_M * BLOCK_TILE_N / WARP_TILE_N;
  constexpr int THREAD_COUNT = WARP_COUNT * 32;

  constexpr int LOOP_TILE_K            = 16;
  constexpr int LDG_SM_BUFFER_SIZE     = 4;
  constexpr int LDG_REG_BUFFER_SIZE    = 2;
  constexpr int LDG_REG_BUFFER_INDEX_0 = 0;
  constexpr int LDG_REG_BUFFER_INDEX_1 = 1;

  constexpr int A_sm_dim0 = LDG_SM_BUFFER_SIZE;
  constexpr int A_sm_dim1 = 2;
  constexpr int A_sm_dim2 = BLOCK_TILE_M;
  constexpr int A_sm_dim3 = LOOP_TILE_K / 2;
  constexpr int B_sm_dim0 = LDG_SM_BUFFER_SIZE;
  constexpr int B_sm_dim1 = LOOP_TILE_K / 8;
  constexpr int B_sm_dim2 = BLOCK_TILE_N / 8;
  constexpr int B_sm_dim3 = 64;

  // The 64 elements of type T in each 8x8 matrix are stored consecutively in a single layer of shared memory.
  __shared__ T A_sm[A_sm_dim0 * A_sm_dim1 * A_sm_dim2 * A_sm_dim3];
  __shared__ T B_sm[B_sm_dim0 * B_sm_dim1 * B_sm_dim2 * B_sm_dim3];

  static_assert(BLOCK_TILE_M * LOOP_TILE_K % THREAD_COUNT == 0);
  static_assert(BLOCK_TILE_M * LOOP_TILE_K / THREAD_COUNT % 8 == 0);
  constexpr int A_LDG_COUNT_PER_THREAD = BLOCK_TILE_M * LOOP_TILE_K / THREAD_COUNT;
  constexpr int A_LDG_LOOP_COUNT       = A_LDG_COUNT_PER_THREAD / 8;
  // clang-format off
  // This is the thread layout of the same warp that loads matrix A, where each thread reads M1xK8 elements of type T at a
  // loop iteration.
  // T0  T16
  // T1  T17
  // T2  T18
  // ... ...
  // T14 T30
  // T15 T31
  // clang-format on
  float A_ldg_reg[LDG_REG_BUFFER_SIZE][A_LDG_LOOP_COUNT][4];

  static_assert(BLOCK_TILE_N * LOOP_TILE_K % THREAD_COUNT == 0);
  static_assert(BLOCK_TILE_N * LOOP_TILE_K / THREAD_COUNT % 8 == 0);
  constexpr int B_LDG_COUNT_PER_THREAD = BLOCK_TILE_N * LOOP_TILE_K / THREAD_COUNT;
  constexpr int B_LDG_LOOP_COUNT       = B_LDG_COUNT_PER_THREAD / 8;
  // clang-format off
  // This is the thread layout of the same warp that loads matrix B, where each thread reads K1xN8 elements of type T at a
  // loop iteration.
  // T0  T16
  // T1  T17
  // T2  T18
  // ... ...
  // T14 T30
  // T15 T31
  // clang-format on
  float B_ldg_reg[LDG_REG_BUFFER_SIZE][B_LDG_LOOP_COUNT][4];

  const int m_block_offset = blockIdx.y * BLOCK_TILE_M;
  const int n_block_offset = blockIdx.x * BLOCK_TILE_N;

  const int warp_id                 = threadIdx.x / 32;
  const int lane_id                 = threadIdx.x % 32;
  const int transposed_lane_id_mask = (lane_id / 8 == 0 || lane_id / 8 == 3) ? 0x00 : 0x18;
  const int transposed_lane_id      = lane_id ^ transposed_lane_id_mask;

  constexpr int M_MMA_WARP_COUNT       = BLOCK_TILE_M / WARP_TILE_M;
  constexpr int M_GROUP_COUNT_PER_WARP = WARP_TILE_M / 8;
  constexpr int N_GROUP_COUNT_PER_WARP = WARP_TILE_N / 16;

  constexpr int MMA_REG_BUFFER_SIZE    = 2;
  constexpr int MMA_REG_BUFFER_INDEX_0 = 0;
  constexpr int MMA_REG_BUFFER_INDEX_1 = 1;
  T             A_mma_reg[MMA_REG_BUFFER_SIZE][M_GROUP_COUNT_PER_WARP][4];
  T             B_mma_reg[MMA_REG_BUFFER_SIZE][N_GROUP_COUNT_PER_WARP][8];
  float         C_mma_reg[M_GROUP_COUNT_PER_WARP][N_GROUP_COUNT_PER_WARP][4] = {0};

  const int m_warp_offset = warp_id % M_MMA_WARP_COUNT * WARP_TILE_M;
  const int n_warp_offset = warp_id / M_MMA_WARP_COUNT * WARP_TILE_N;

  const int A_ldg_reg_2_A_sm_partial_offset =
    lane_id / 16 * A_sm_dim2 * A_sm_dim3 + (warp_id * 16 + lane_id % 16) * A_sm_dim3;

  const int B_ldg_reg_2_B_sm_partial_offset =
    (lane_id % 16) / 8 * B_sm_dim2 * B_sm_dim3 + (lane_id % 16) % 8 * 8 + (warp_id * 2 + lane_id / 16) * B_sm_dim3;

  const int A_global_partial_offset = (m_block_offset + warp_id * 16 + lane_id % 16) * K + lane_id / 16 * 8;
  const int B_global_partial_offset = lane_id % 16 * N + n_block_offset + warp_id * 16 + lane_id / 16 * 8;

  const T* A_global_ptr_for_ldg[A_LDG_LOOP_COUNT];
  const T* B_global_ptr_for_ldg[B_LDG_LOOP_COUNT];
  for (int loop = 0; loop < A_LDG_LOOP_COUNT; ++loop) {
    A_global_ptr_for_ldg[loop] = &A[A_global_partial_offset + loop * WARP_COUNT * 16 * K];
  }
  for (int loop = 0; loop < B_LDG_LOOP_COUNT; ++loop) {
    B_global_ptr_for_ldg[loop] = &B[B_global_partial_offset + loop * WARP_COUNT * 16];
  }

  const T* A_sm_ptr_for_ldg[A_LDG_LOOP_COUNT];
  const T* B_sm_ptr_for_ldg[B_LDG_LOOP_COUNT];
  for (int loop = 0; loop < A_LDG_LOOP_COUNT; ++loop) {
    A_sm_ptr_for_ldg[loop] = &A_sm[A_ldg_reg_2_A_sm_partial_offset + loop * WARP_COUNT * 16 * A_sm_dim3];
  }
  for (int loop = 0; loop < B_LDG_LOOP_COUNT; ++loop) {
    B_sm_ptr_for_ldg[loop] = &B_sm[B_ldg_reg_2_B_sm_partial_offset + loop * WARP_COUNT * 2 * B_sm_dim3];
  }

  const int A_sm_2_A_mma_reg_partial_offset =
    lane_id % 16 / 8 * A_sm_dim2 * A_sm_dim3 + (m_warp_offset + lane_id % 8) * A_sm_dim3;

  const int B_sm_2_B_mma_reg_partial_offset = transposed_lane_id % 16 / 8 * B_sm_dim2 * B_sm_dim3
                                              + (n_warp_offset + transposed_lane_id / 16 * 8) / 8 * B_sm_dim3
                                              + transposed_lane_id % 8 * 8;

  const T* A_sm_ptr_for_mma[M_GROUP_COUNT_PER_WARP];
  const T* B_sm_ptr_for_mma[N_GROUP_COUNT_PER_WARP];

  for (int group = 0; group < M_GROUP_COUNT_PER_WARP; ++group) {
    A_sm_ptr_for_mma[group] = &A_sm[A_sm_2_A_mma_reg_partial_offset + (group + lane_id / 16) * 8 * A_sm_dim3];
  }
  for (int group = 0; group < N_GROUP_COUNT_PER_WARP; ++group) {
    B_sm_ptr_for_mma[group] = &B_sm[B_sm_2_B_mma_reg_partial_offset + (group * 2 * B_sm_dim3)];
  }

#define global_2_ldg_reg(k_loop_offset, ldg_reg_buffer_index)                                                          \
  {                                                                                                                    \
    _Pragma("unroll") for (int loop = 0; loop < A_LDG_LOOP_COUNT; ++loop)                                              \
    {                                                                                                                  \
      /* const int m = (loop * WARP_COUNT + warp_id) * 16 + lane_id % 16; */                                           \
      /* const int k = lane_id / 16 * 8; */                                                                            \
      FETCH_FLOAT4_WITH_PTR(&A_ldg_reg[ldg_reg_buffer_index][loop][0], A_global_ptr_for_ldg[loop] + k_loop_offset);    \
    }                                                                                                                  \
    _Pragma("unroll") for (int loop = 0; loop < B_LDG_LOOP_COUNT; ++loop)                                              \
    {                                                                                                                  \
      /* const int k = lane_id % 16;                                           */                                      \
      /* const int n = (loop * WARP_COUNT + warp_id) * 16 + lane_id / 16 * 8;  */                                      \
      FETCH_FLOAT4_WITH_PTR(&B_ldg_reg[ldg_reg_buffer_index][loop][0],                                                 \
                            B_global_ptr_for_ldg[loop] + (k_loop_offset) * N);                                         \
    }                                                                                                                  \
  }

#define ldg_reg_2_sm(ldg_sm_buffer_index, ldg_reg_buffer_index)                                                        \
  {                                                                                                                    \
    _Pragma("unroll") for (int loop = 0; loop < A_LDG_LOOP_COUNT; ++loop)                                              \
    {                                                                                                                  \
      /* const int m = (loop * WARP_COUNT + warp_id) * 16 + lane_id % 16; */                                           \
      /* const int k = lane_id / 16 * 8;  */                                                                           \
      STORE_FLOAT4_WITH_PTR(A_sm_ptr_for_ldg[loop] + (ldg_sm_buffer_index) * A_sm_dim1 * A_sm_dim2 * A_sm_dim3,        \
                            &A_ldg_reg[ldg_reg_buffer_index][loop][0]);                                                \
    }                                                                                                                  \
    _Pragma("unroll") for (int loop = 0; loop < B_LDG_LOOP_COUNT; ++loop)                                              \
    {                                                                                                                  \
      /*const int k = lane_id % 16; */                                                                                 \
      /*const int n = (loop * WARP_COUNT + warp_id) * 16 + lane_id / 16 * 8;*/                                         \
      STORE_FLOAT4_WITH_PTR(B_sm_ptr_for_ldg[loop] + (ldg_sm_buffer_index) * B_sm_dim1 * B_sm_dim2 * B_sm_dim3,        \
                            &B_ldg_reg[ldg_reg_buffer_index][loop][0]);                                                \
    }                                                                                                                  \
  }

#define sm_2_A_mma_reg(ldg_sm_buffer_index, mma_reg_buffer_index)                                                      \
  if constexpr (M_GROUP_COUNT_PER_WARP == 1) {                                                                         \
    for (int group = 0; group < M_GROUP_COUNT_PER_WARP; ++group) {                                                     \
      uint32_t src =                                                                                                   \
        __cvta_generic_to_shared(A_sm_ptr_for_mma[group] + (ldg_sm_buffer_index) * A_sm_dim1 * A_sm_dim2 * A_sm_dim3); \
      asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"                                          \
                   : "=r"(*(uint32_t*)&A_mma_reg[mma_reg_buffer_index][group][0]),                                     \
                     "=r"(*(uint32_t*)&A_mma_reg[mma_reg_buffer_index][group][2])                                      \
                   : "r"(src));                                                                                        \
    }                                                                                                                  \
  }                                                                                                                    \
  else if constexpr (M_GROUP_COUNT_PER_WARP % 2 == 0) {                                                                \
    for (int group = 0; group < M_GROUP_COUNT_PER_WARP; group += 2) {                                                  \
      uint32_t src =                                                                                                   \
        __cvta_generic_to_shared(A_sm_ptr_for_mma[group] + (ldg_sm_buffer_index) * A_sm_dim1 * A_sm_dim2 * A_sm_dim3); \
      asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"                                  \
                   : "=r"(*(uint32_t*)&A_mma_reg[mma_reg_buffer_index][group][0]),                                     \
                     "=r"(*(uint32_t*)&A_mma_reg[mma_reg_buffer_index][group][2]),                                     \
                     "=r"(*(uint32_t*)&A_mma_reg[mma_reg_buffer_index][group + 1][0]),                                 \
                     "=r"(*(uint32_t*)&A_mma_reg[mma_reg_buffer_index][group + 1][2])                                  \
                   : "r"(src));                                                                                        \
    }                                                                                                                  \
  }                                                                                                                    \
  else {                                                                                                               \
    static_assert(M_GROUP_COUNT_PER_WARP == 1 || M_GROUP_COUNT_PER_WARP % 2 == 0);                                     \
  }

#define sm_2_B_mma_reg(ldg_sm_buffer_index, mma_reg_buffer_index)                                                      \
  for (int group = 0; group < N_GROUP_COUNT_PER_WARP; ++group) {                                                       \
    uint32_t src =                                                                                                     \
      __cvta_generic_to_shared(B_sm_ptr_for_mma[group] + (ldg_sm_buffer_index) * B_sm_dim1 * B_sm_dim2 * B_sm_dim3);   \
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"                              \
                 : "=r"(*(uint32_t*)&B_mma_reg[mma_reg_buffer_index][group][0]),                                       \
                   "=r"(*(uint32_t*)&B_mma_reg[mma_reg_buffer_index][group][2]),                                       \
                   "=r"(*(uint32_t*)&B_mma_reg[mma_reg_buffer_index][group][4]),                                       \
                   "=r"(*(uint32_t*)&B_mma_reg[mma_reg_buffer_index][group][6])                                        \
                 : "r"(src));                                                                                          \
  }

  global_2_ldg_reg(0, LDG_REG_BUFFER_INDEX_0);
  global_2_ldg_reg(LOOP_TILE_K, LDG_REG_BUFFER_INDEX_1);
  ldg_reg_2_sm(0, LDG_REG_BUFFER_INDEX_0);
  ldg_reg_2_sm(1, LDG_REG_BUFFER_INDEX_1);
  __syncthreads();

  int LDG_SM_BUFFER_INDEX = 0;
  int k_loop_offset       = LOOP_TILE_K * 2;

  sm_2_A_mma_reg(LDG_SM_BUFFER_INDEX, MMA_REG_BUFFER_INDEX_0);
  sm_2_B_mma_reg(LDG_SM_BUFFER_INDEX, MMA_REG_BUFFER_INDEX_0);

  global_2_ldg_reg(k_loop_offset, LDG_REG_BUFFER_INDEX_0);
  global_2_ldg_reg(k_loop_offset + LOOP_TILE_K, LDG_REG_BUFFER_INDEX_1);

  while (k_loop_offset + LOOP_TILE_K * 2 < K) {
    sm_2_A_mma_reg(LDG_SM_BUFFER_INDEX + 1, MMA_REG_BUFFER_INDEX_1);
    sm_2_B_mma_reg(LDG_SM_BUFFER_INDEX + 1, MMA_REG_BUFFER_INDEX_1);
    for (int mg = 0; mg < M_GROUP_COUNT_PER_WARP; ++mg) {
      for (int ng = 0; ng < N_GROUP_COUNT_PER_WARP; ++ng) {
        mma_m16n8k16_row_col(C_mma_reg[mg][ng],
                             B_mma_reg[MMA_REG_BUFFER_INDEX_0][ng],
                             A_mma_reg[MMA_REG_BUFFER_INDEX_0][mg],
                             C_mma_reg[mg][ng]);
      }
    }

    LDG_SM_BUFFER_INDEX ^= 2;
    k_loop_offset += LOOP_TILE_K * 2;
    ldg_reg_2_sm(LDG_SM_BUFFER_INDEX, LDG_REG_BUFFER_INDEX_0);
    ldg_reg_2_sm(LDG_SM_BUFFER_INDEX + 1, LDG_REG_BUFFER_INDEX_1);

    __syncthreads();

    global_2_ldg_reg(k_loop_offset, LDG_REG_BUFFER_INDEX_0);
    global_2_ldg_reg(k_loop_offset + LOOP_TILE_K, LDG_REG_BUFFER_INDEX_1);

    sm_2_A_mma_reg(LDG_SM_BUFFER_INDEX, MMA_REG_BUFFER_INDEX_0);
    sm_2_B_mma_reg(LDG_SM_BUFFER_INDEX, MMA_REG_BUFFER_INDEX_0);

    for (int mg = 0; mg < M_GROUP_COUNT_PER_WARP; ++mg) {
      for (int ng = 0; ng < N_GROUP_COUNT_PER_WARP; ++ng) {
        mma_m16n8k16_row_col(C_mma_reg[mg][ng],
                             B_mma_reg[MMA_REG_BUFFER_INDEX_1][ng],
                             A_mma_reg[MMA_REG_BUFFER_INDEX_1][mg],
                             C_mma_reg[mg][ng]);
      }
    }
  }
  {
    sm_2_A_mma_reg(LDG_SM_BUFFER_INDEX + 1, MMA_REG_BUFFER_INDEX_1);
    sm_2_B_mma_reg(LDG_SM_BUFFER_INDEX + 1, MMA_REG_BUFFER_INDEX_1);
    for (int mg = 0; mg < M_GROUP_COUNT_PER_WARP; ++mg) {
      for (int ng = 0; ng < N_GROUP_COUNT_PER_WARP; ++ng) {
        mma_m16n8k16_row_col(C_mma_reg[mg][ng],
                             B_mma_reg[MMA_REG_BUFFER_INDEX_0][ng],
                             A_mma_reg[MMA_REG_BUFFER_INDEX_0][mg],
                             C_mma_reg[mg][ng]);
      }
    }

    LDG_SM_BUFFER_INDEX ^= 2;
    k_loop_offset += LOOP_TILE_K * 2;
    ldg_reg_2_sm(LDG_SM_BUFFER_INDEX, LDG_REG_BUFFER_INDEX_0);
    ldg_reg_2_sm(LDG_SM_BUFFER_INDEX + 1, LDG_REG_BUFFER_INDEX_1);

    __syncthreads();
    sm_2_A_mma_reg(LDG_SM_BUFFER_INDEX, MMA_REG_BUFFER_INDEX_0);
    sm_2_B_mma_reg(LDG_SM_BUFFER_INDEX, MMA_REG_BUFFER_INDEX_0);
    for (int mg = 0; mg < M_GROUP_COUNT_PER_WARP; ++mg) {
      for (int ng = 0; ng < M_GROUP_COUNT_PER_WARP; ++ng) {
        mma_m16n8k16_row_col(C_mma_reg[mg][ng],
                             B_mma_reg[MMA_REG_BUFFER_INDEX_1][ng],
                             A_mma_reg[MMA_REG_BUFFER_INDEX_1][mg],
                             C_mma_reg[mg][ng]);
      }
    }
  }
  {
    sm_2_A_mma_reg(LDG_SM_BUFFER_INDEX + 1, MMA_REG_BUFFER_INDEX_1);
    sm_2_B_mma_reg(LDG_SM_BUFFER_INDEX + 1, MMA_REG_BUFFER_INDEX_1);
    for (int mg = 0; mg < M_GROUP_COUNT_PER_WARP; ++mg) {
      for (int ng = 0; ng < N_GROUP_COUNT_PER_WARP; ++ng) {
        mma_m16n8k16_row_col(C_mma_reg[mg][ng],
                             B_mma_reg[MMA_REG_BUFFER_INDEX_0][ng],
                             A_mma_reg[MMA_REG_BUFFER_INDEX_0][mg],
                             C_mma_reg[mg][ng]);
      }
    }

    for (int mg = 0; mg < M_GROUP_COUNT_PER_WARP; ++mg) {
      for (int ng = 0; ng < N_GROUP_COUNT_PER_WARP; ++ng) {
        mma_m16n8k16_row_col(C_mma_reg[mg][ng],
                             B_mma_reg[MMA_REG_BUFFER_INDEX_1][ng],
                             A_mma_reg[MMA_REG_BUFFER_INDEX_1][mg],
                             C_mma_reg[mg][ng]);
      }
    }
  }

  for (int mg = 0; mg < M_GROUP_COUNT_PER_WARP; ++mg) {
    for (int ng = 0; ng < N_GROUP_COUNT_PER_WARP; ++ng) {
      // int m = m_block_offset + m_warp_offset + mg * 8;
      // int n = n_block_offset + n_warp_offset + ng * 16;
      T casted[4] = {
        C_mma_reg[mg][ng][0],
        C_mma_reg[mg][ng][1],
        C_mma_reg[mg][ng][2],
        C_mma_reg[mg][ng][3],
      };
      shfl_23_and_01(casted, 0x4, lane_id);
      T                store[4]           = {casted[0], casted[2], casted[1], casted[3]};
      static const int lane_2_n_offset[8] = {0, 8, 2, 10, 4, 12, 6, 14};
      int              m                  = m_block_offset + m_warp_offset + mg * 8 + lane_id % 4 * 2;
      int              n                  = n_block_offset + n_warp_offset + ng * 16 + lane_2_n_offset[lane_id / 4];
      STORE_FLOAT(C[OFFSET(m, n, N)], store[0]);
      STORE_FLOAT(C[OFFSET(m + 1, n, N)], store[2]);
    }
  }
#undef global_2_ldg_reg
#undef ldg_reg_2_sm
#undef sm_2_A_mma_reg
#undef sm_2_B_mma_reg
}

template<typename T, int BLOCK_TILE_M, int BLOCK_TILE_N, int WARP_TILE_M, int WARP_TILE_N>
__global__ void
fp16_mma_m16n8k16_ldmatrix_trans__overlap_global_2_sm__quadra_buffer__reduce_instructions__reorder_instructions(
  const T* A, const T* B, T* C, int M, int N, int K)
{
  constexpr int WARP_COUNT   = BLOCK_TILE_M / WARP_TILE_M * BLOCK_TILE_N / WARP_TILE_N;
  constexpr int THREAD_COUNT = WARP_COUNT * 32;

  constexpr int LOOP_TILE_K            = 16;
  constexpr int LDG_SM_BUFFER_SIZE     = 4;
  constexpr int LDG_REG_BUFFER_SIZE    = 2;
  constexpr int LDG_REG_BUFFER_INDEX_0 = 0;
  constexpr int LDG_REG_BUFFER_INDEX_1 = 1;

  constexpr int A_sm_dim0 = LDG_SM_BUFFER_SIZE;
  constexpr int A_sm_dim1 = 2;
  constexpr int A_sm_dim2 = BLOCK_TILE_M;
  constexpr int A_sm_dim3 = LOOP_TILE_K / 2;
  constexpr int B_sm_dim0 = LDG_SM_BUFFER_SIZE;
  constexpr int B_sm_dim1 = LOOP_TILE_K / 8;
  constexpr int B_sm_dim2 = BLOCK_TILE_N / 8;
  constexpr int B_sm_dim3 = 64;

  // The 64 elements of type T in each 8x8 matrix are stored consecutively in a single layer of shared memory.
  __shared__ T A_sm[A_sm_dim0 * A_sm_dim1 * A_sm_dim2 * A_sm_dim3];
  __shared__ T B_sm[B_sm_dim0 * B_sm_dim1 * B_sm_dim2 * B_sm_dim3];

  static_assert(BLOCK_TILE_M * LOOP_TILE_K % THREAD_COUNT == 0);
  static_assert(BLOCK_TILE_M * LOOP_TILE_K / THREAD_COUNT % 8 == 0);
  constexpr int A_LDG_COUNT_PER_THREAD = BLOCK_TILE_M * LOOP_TILE_K / THREAD_COUNT;
  constexpr int A_LDG_LOOP_COUNT       = A_LDG_COUNT_PER_THREAD / 8;
  // clang-format off
  // This is the thread layout of the same warp that loads matrix A, where each thread reads M1xK8 elements of type T at a
  // loop iteration.
  // T0  T16
  // T1  T17
  // T2  T18
  // ... ...
  // T14 T30
  // T15 T31
  // clang-format on
  float A_ldg_reg[LDG_REG_BUFFER_SIZE][A_LDG_LOOP_COUNT][4];

  static_assert(BLOCK_TILE_N * LOOP_TILE_K % THREAD_COUNT == 0);
  static_assert(BLOCK_TILE_N * LOOP_TILE_K / THREAD_COUNT % 8 == 0);
  constexpr int B_LDG_COUNT_PER_THREAD = BLOCK_TILE_N * LOOP_TILE_K / THREAD_COUNT;
  constexpr int B_LDG_LOOP_COUNT       = B_LDG_COUNT_PER_THREAD / 8;
  // clang-format off
  // This is the thread layout of the same warp that loads matrix B, where each thread reads K1xN8 elements of type T at a
  // loop iteration.
  // T0  T16
  // T1  T17
  // T2  T18
  // ... ...
  // T14 T30
  // T15 T31
  // clang-format on
  float B_ldg_reg[LDG_REG_BUFFER_SIZE][B_LDG_LOOP_COUNT][4];

  const int m_block_offset = blockIdx.y * BLOCK_TILE_M;
  const int n_block_offset = blockIdx.x * BLOCK_TILE_N;

  const int warp_id                 = threadIdx.x / 32;
  const int lane_id                 = threadIdx.x % 32;
  const int transposed_lane_id_mask = (lane_id / 8 == 0 || lane_id / 8 == 3) ? 0x00 : 0x18;
  const int transposed_lane_id      = lane_id ^ transposed_lane_id_mask;

  constexpr int M_MMA_WARP_COUNT       = BLOCK_TILE_M / WARP_TILE_M;
  constexpr int M_GROUP_COUNT_PER_WARP = WARP_TILE_M / 8;
  constexpr int N_GROUP_COUNT_PER_WARP = WARP_TILE_N / 16;

  constexpr int MMA_REG_BUFFER_SIZE    = 2;
  constexpr int MMA_REG_BUFFER_INDEX_0 = 0;
  constexpr int MMA_REG_BUFFER_INDEX_1 = 1;
  T             A_mma_reg[MMA_REG_BUFFER_SIZE][M_GROUP_COUNT_PER_WARP][4];
  T             B_mma_reg[MMA_REG_BUFFER_SIZE][N_GROUP_COUNT_PER_WARP][8];
  float         C_mma_reg[M_GROUP_COUNT_PER_WARP][N_GROUP_COUNT_PER_WARP][4] = {0};

  const int m_warp_offset = warp_id % M_MMA_WARP_COUNT * WARP_TILE_M;
  const int n_warp_offset = warp_id / M_MMA_WARP_COUNT * WARP_TILE_N;

  const int A_ldg_reg_2_A_sm_partial_offset =
    lane_id / 16 * A_sm_dim2 * A_sm_dim3 + (warp_id * 16 + lane_id % 16) * A_sm_dim3;

  const int B_ldg_reg_2_B_sm_partial_offset =
    (lane_id % 16) / 8 * B_sm_dim2 * B_sm_dim3 + (lane_id % 16) % 8 * 8 + (warp_id * 2 + lane_id / 16) * B_sm_dim3;

  const int A_global_partial_offset = (m_block_offset + warp_id * 16 + lane_id % 16) * K + lane_id / 16 * 8;
  const int B_global_partial_offset = lane_id % 16 * N + n_block_offset + warp_id * 16 + lane_id / 16 * 8;

  const T* A_global_ptr_for_ldg[A_LDG_LOOP_COUNT];
  const T* B_global_ptr_for_ldg[B_LDG_LOOP_COUNT];
  for (int loop = 0; loop < A_LDG_LOOP_COUNT; ++loop) {
    A_global_ptr_for_ldg[loop] = &A[A_global_partial_offset + loop * WARP_COUNT * 16 * K];
  }
  for (int loop = 0; loop < B_LDG_LOOP_COUNT; ++loop) {
    B_global_ptr_for_ldg[loop] = &B[B_global_partial_offset + loop * WARP_COUNT * 16];
  }

  const T* A_sm_ptr_for_ldg[A_LDG_LOOP_COUNT];
  const T* B_sm_ptr_for_ldg[B_LDG_LOOP_COUNT];
  for (int loop = 0; loop < A_LDG_LOOP_COUNT; ++loop) {
    A_sm_ptr_for_ldg[loop] = &A_sm[A_ldg_reg_2_A_sm_partial_offset + loop * WARP_COUNT * 16 * A_sm_dim3];
  }
  for (int loop = 0; loop < B_LDG_LOOP_COUNT; ++loop) {
    B_sm_ptr_for_ldg[loop] = &B_sm[B_ldg_reg_2_B_sm_partial_offset + loop * WARP_COUNT * 2 * B_sm_dim3];
  }

  const int A_sm_2_A_mma_reg_partial_offset =
    lane_id % 16 / 8 * A_sm_dim2 * A_sm_dim3 + (m_warp_offset + lane_id % 8) * A_sm_dim3;

  const int B_sm_2_B_mma_reg_partial_offset = transposed_lane_id % 16 / 8 * B_sm_dim2 * B_sm_dim3
                                              + (n_warp_offset + transposed_lane_id / 16 * 8) / 8 * B_sm_dim3
                                              + transposed_lane_id % 8 * 8;

  const T* A_sm_ptr_for_mma[M_GROUP_COUNT_PER_WARP];
  const T* B_sm_ptr_for_mma[N_GROUP_COUNT_PER_WARP];

  for (int group = 0; group < M_GROUP_COUNT_PER_WARP; ++group) {
    A_sm_ptr_for_mma[group] = &A_sm[A_sm_2_A_mma_reg_partial_offset + (group + lane_id / 16) * 8 * A_sm_dim3];
  }
  for (int group = 0; group < N_GROUP_COUNT_PER_WARP; ++group) {
    B_sm_ptr_for_mma[group] = &B_sm[B_sm_2_B_mma_reg_partial_offset + (group * 2 * B_sm_dim3)];
  }

#define global_2_ldg_reg(k_loop_offset, ldg_reg_buffer_index)                                                          \
  {                                                                                                                    \
    _Pragma("unroll") for (int loop = 0; loop < A_LDG_LOOP_COUNT; ++loop)                                              \
    {                                                                                                                  \
      /* const int m = (loop * WARP_COUNT + warp_id) * 16 + lane_id % 16; */                                           \
      /* const int k = lane_id / 16 * 8; */                                                                            \
      FETCH_FLOAT4_WITH_PTR(&A_ldg_reg[ldg_reg_buffer_index][loop][0], A_global_ptr_for_ldg[loop] + k_loop_offset);    \
    }                                                                                                                  \
    _Pragma("unroll") for (int loop = 0; loop < B_LDG_LOOP_COUNT; ++loop)                                              \
    {                                                                                                                  \
      /* const int k = lane_id % 16;                                           */                                      \
      /* const int n = (loop * WARP_COUNT + warp_id) * 16 + lane_id / 16 * 8;  */                                      \
      FETCH_FLOAT4_WITH_PTR(&B_ldg_reg[ldg_reg_buffer_index][loop][0],                                                 \
                            B_global_ptr_for_ldg[loop] + (k_loop_offset) * N);                                         \
    }                                                                                                                  \
  }

#define ldg_reg_2_sm(ldg_sm_buffer_index, ldg_reg_buffer_index)                                                        \
  {                                                                                                                    \
    _Pragma("unroll") for (int loop = 0; loop < A_LDG_LOOP_COUNT; ++loop)                                              \
    {                                                                                                                  \
      /* const int m = (loop * WARP_COUNT + warp_id) * 16 + lane_id % 16; */                                           \
      /* const int k = lane_id / 16 * 8;  */                                                                           \
      STORE_FLOAT4_WITH_PTR(A_sm_ptr_for_ldg[loop] + (ldg_sm_buffer_index) * A_sm_dim1 * A_sm_dim2 * A_sm_dim3,        \
                            &A_ldg_reg[ldg_reg_buffer_index][loop][0]);                                                \
    }                                                                                                                  \
    _Pragma("unroll") for (int loop = 0; loop < B_LDG_LOOP_COUNT; ++loop)                                              \
    {                                                                                                                  \
      /*const int k = lane_id % 16; */                                                                                 \
      /*const int n = (loop * WARP_COUNT + warp_id) * 16 + lane_id / 16 * 8;*/                                         \
      STORE_FLOAT4_WITH_PTR(B_sm_ptr_for_ldg[loop] + (ldg_sm_buffer_index) * B_sm_dim1 * B_sm_dim2 * B_sm_dim3,        \
                            &B_ldg_reg[ldg_reg_buffer_index][loop][0]);                                                \
    }                                                                                                                  \
  }

#define sm_2_A_mma_reg(ldg_sm_buffer_index, mma_reg_buffer_index, group)                                               \
  if constexpr (M_GROUP_COUNT_PER_WARP == 1) {                                                                         \
    /* for (int group = 0; group < M_GROUP_COUNT_PER_WARP; ++group) */ {                                               \
      uint32_t src =                                                                                                   \
        __cvta_generic_to_shared(A_sm_ptr_for_mma[group] + (ldg_sm_buffer_index) * A_sm_dim1 * A_sm_dim2 * A_sm_dim3); \
      asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"                                          \
                   : "=r"(*(uint32_t*)&A_mma_reg[mma_reg_buffer_index][group][0]),                                     \
                     "=r"(*(uint32_t*)&A_mma_reg[mma_reg_buffer_index][group][2])                                      \
                   : "r"(src));                                                                                        \
    }                                                                                                                  \
  }                                                                                                                    \
  else if constexpr (M_GROUP_COUNT_PER_WARP % 2 == 0) {                                                                \
    /*for (int group = 0; group < M_GROUP_COUNT_PER_WARP; group += 2) */ {                                             \
      uint32_t src =                                                                                                   \
        __cvta_generic_to_shared(A_sm_ptr_for_mma[group] + (ldg_sm_buffer_index) * A_sm_dim1 * A_sm_dim2 * A_sm_dim3); \
      asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"                                  \
                   : "=r"(*(uint32_t*)&A_mma_reg[mma_reg_buffer_index][group][0]),                                     \
                     "=r"(*(uint32_t*)&A_mma_reg[mma_reg_buffer_index][group][2]),                                     \
                     "=r"(*(uint32_t*)&A_mma_reg[mma_reg_buffer_index][group + 1][0]),                                 \
                     "=r"(*(uint32_t*)&A_mma_reg[mma_reg_buffer_index][group + 1][2])                                  \
                   : "r"(src));                                                                                        \
    }                                                                                                                  \
  }                                                                                                                    \
  else {                                                                                                               \
    static_assert(M_GROUP_COUNT_PER_WARP == 1 || M_GROUP_COUNT_PER_WARP % 2 == 0);                                     \
  }

#define sm_2_B_mma_reg(ldg_sm_buffer_index, mma_reg_buffer_index, group)                                               \
  /* for (int group = 0; group < N_GROUP_COUNT_PER_WARP; ++group) */ {                                                 \
    uint32_t src =                                                                                                     \
      __cvta_generic_to_shared(B_sm_ptr_for_mma[group] + (ldg_sm_buffer_index) * B_sm_dim1 * B_sm_dim2 * B_sm_dim3);   \
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"                              \
                 : "=r"(*(uint32_t*)&B_mma_reg[mma_reg_buffer_index][group][0]),                                       \
                   "=r"(*(uint32_t*)&B_mma_reg[mma_reg_buffer_index][group][2]),                                       \
                   "=r"(*(uint32_t*)&B_mma_reg[mma_reg_buffer_index][group][4]),                                       \
                   "=r"(*(uint32_t*)&B_mma_reg[mma_reg_buffer_index][group][6])                                        \
                 : "r"(src));                                                                                          \
  }

#define mma_and_ldm(ldm_sm_buffer_index, ldm_reg_buffer_index, mma_reg_buffer_index, rank, ldm_switch, mma_switch)     \
  {                                                                                                                    \
    if constexpr (mma_switch && rank < M_GROUP_COUNT_PER_WARP * N_GROUP_COUNT_PER_WARP) {                              \
      constexpr int mg = rank % M_GROUP_COUNT_PER_WARP;                                                                \
      constexpr int ng = rank / M_GROUP_COUNT_PER_WARP;                                                                \
      mma_m16n8k16_row_col(C_mma_reg[mg][ng],                                                                          \
                           B_mma_reg[mma_reg_buffer_index][ng],                                                        \
                           A_mma_reg[mma_reg_buffer_index][mg],                                                        \
                           C_mma_reg[mg][ng]);                                                                         \
    }                                                                                                                  \
    static_assert(M_GROUP_COUNT_PER_WARP == 1 || M_GROUP_COUNT_PER_WARP % 2 == 0);                                     \
    if constexpr (ldm_switch && rank < N_GROUP_COUNT_PER_WARP) {                                                       \
      sm_2_B_mma_reg(ldm_sm_buffer_index, ldm_reg_buffer_index, rank);                                                 \
    }                                                                                                                  \
    if constexpr (ldm_switch && N_GROUP_COUNT_PER_WARP <= rank                                                         \
                  && rank < N_GROUP_COUNT_PER_WARP + (M_GROUP_COUNT_PER_WARP + 1) / 2) {                               \
      sm_2_A_mma_reg(ldm_sm_buffer_index, ldm_reg_buffer_index, (rank - N_GROUP_COUNT_PER_WARP) * 2);                  \
    }                                                                                                                  \
  }

// FIXME This code is really stupid. Please find a way to optimize it as soon as possible.
#define alternate_mma_and_ldm(ldm_sm_buffer_index, ldm_reg_buffer_index, mma_reg_buffer_index, ldm_switch, mma_switch) \
  static_assert(M_GROUP_COUNT_PER_WARP * N_GROUP_COUNT_PER_WARP <= 32);                                                \
  mma_and_ldm(ldm_sm_buffer_index, ldm_reg_buffer_index, mma_reg_buffer_index, 0, ldm_switch, mma_switch);             \
  mma_and_ldm(ldm_sm_buffer_index, ldm_reg_buffer_index, mma_reg_buffer_index, 1, ldm_switch, mma_switch);             \
  mma_and_ldm(ldm_sm_buffer_index, ldm_reg_buffer_index, mma_reg_buffer_index, 2, ldm_switch, mma_switch);             \
  mma_and_ldm(ldm_sm_buffer_index, ldm_reg_buffer_index, mma_reg_buffer_index, 3, ldm_switch, mma_switch);             \
  mma_and_ldm(ldm_sm_buffer_index, ldm_reg_buffer_index, mma_reg_buffer_index, 4, ldm_switch, mma_switch);             \
  mma_and_ldm(ldm_sm_buffer_index, ldm_reg_buffer_index, mma_reg_buffer_index, 5, ldm_switch, mma_switch);             \
  mma_and_ldm(ldm_sm_buffer_index, ldm_reg_buffer_index, mma_reg_buffer_index, 6, ldm_switch, mma_switch);             \
  mma_and_ldm(ldm_sm_buffer_index, ldm_reg_buffer_index, mma_reg_buffer_index, 7, ldm_switch, mma_switch);             \
  mma_and_ldm(ldm_sm_buffer_index, ldm_reg_buffer_index, mma_reg_buffer_index, 8, ldm_switch, mma_switch);             \
  mma_and_ldm(ldm_sm_buffer_index, ldm_reg_buffer_index, mma_reg_buffer_index, 9, ldm_switch, mma_switch);             \
  mma_and_ldm(ldm_sm_buffer_index, ldm_reg_buffer_index, mma_reg_buffer_index, 10, ldm_switch, mma_switch);            \
  mma_and_ldm(ldm_sm_buffer_index, ldm_reg_buffer_index, mma_reg_buffer_index, 11, ldm_switch, mma_switch);            \
  mma_and_ldm(ldm_sm_buffer_index, ldm_reg_buffer_index, mma_reg_buffer_index, 12, ldm_switch, mma_switch);            \
  mma_and_ldm(ldm_sm_buffer_index, ldm_reg_buffer_index, mma_reg_buffer_index, 13, ldm_switch, mma_switch);            \
  mma_and_ldm(ldm_sm_buffer_index, ldm_reg_buffer_index, mma_reg_buffer_index, 14, ldm_switch, mma_switch);            \
  mma_and_ldm(ldm_sm_buffer_index, ldm_reg_buffer_index, mma_reg_buffer_index, 15, ldm_switch, mma_switch);            \
  mma_and_ldm(ldm_sm_buffer_index, ldm_reg_buffer_index, mma_reg_buffer_index, 16, ldm_switch, mma_switch);            \
  mma_and_ldm(ldm_sm_buffer_index, ldm_reg_buffer_index, mma_reg_buffer_index, 17, ldm_switch, mma_switch);            \
  mma_and_ldm(ldm_sm_buffer_index, ldm_reg_buffer_index, mma_reg_buffer_index, 18, ldm_switch, mma_switch);            \
  mma_and_ldm(ldm_sm_buffer_index, ldm_reg_buffer_index, mma_reg_buffer_index, 19, ldm_switch, mma_switch);            \
  mma_and_ldm(ldm_sm_buffer_index, ldm_reg_buffer_index, mma_reg_buffer_index, 20, ldm_switch, mma_switch);            \
  mma_and_ldm(ldm_sm_buffer_index, ldm_reg_buffer_index, mma_reg_buffer_index, 21, ldm_switch, mma_switch);            \
  mma_and_ldm(ldm_sm_buffer_index, ldm_reg_buffer_index, mma_reg_buffer_index, 22, ldm_switch, mma_switch);            \
  mma_and_ldm(ldm_sm_buffer_index, ldm_reg_buffer_index, mma_reg_buffer_index, 23, ldm_switch, mma_switch);            \
  mma_and_ldm(ldm_sm_buffer_index, ldm_reg_buffer_index, mma_reg_buffer_index, 24, ldm_switch, mma_switch);            \
  mma_and_ldm(ldm_sm_buffer_index, ldm_reg_buffer_index, mma_reg_buffer_index, 25, ldm_switch, mma_switch);            \
  mma_and_ldm(ldm_sm_buffer_index, ldm_reg_buffer_index, mma_reg_buffer_index, 26, ldm_switch, mma_switch);            \
  mma_and_ldm(ldm_sm_buffer_index, ldm_reg_buffer_index, mma_reg_buffer_index, 27, ldm_switch, mma_switch);            \
  mma_and_ldm(ldm_sm_buffer_index, ldm_reg_buffer_index, mma_reg_buffer_index, 28, ldm_switch, mma_switch);            \
  mma_and_ldm(ldm_sm_buffer_index, ldm_reg_buffer_index, mma_reg_buffer_index, 29, ldm_switch, mma_switch);            \
  mma_and_ldm(ldm_sm_buffer_index, ldm_reg_buffer_index, mma_reg_buffer_index, 30, ldm_switch, mma_switch);            \
  mma_and_ldm(ldm_sm_buffer_index, ldm_reg_buffer_index, mma_reg_buffer_index, 31, ldm_switch, mma_switch);

  global_2_ldg_reg(0, LDG_REG_BUFFER_INDEX_0);
  global_2_ldg_reg(LOOP_TILE_K, LDG_REG_BUFFER_INDEX_1);
  ldg_reg_2_sm(0, LDG_REG_BUFFER_INDEX_0);
  ldg_reg_2_sm(1, LDG_REG_BUFFER_INDEX_1);
  __syncthreads();

  int LDG_SM_BUFFER_INDEX = 0;
  int k_loop_offset       = LOOP_TILE_K * 2;

  alternate_mma_and_ldm(LDG_SM_BUFFER_INDEX, MMA_REG_BUFFER_INDEX_0, 0, true, false);

  global_2_ldg_reg(k_loop_offset, LDG_REG_BUFFER_INDEX_0);
  global_2_ldg_reg(k_loop_offset + LOOP_TILE_K, LDG_REG_BUFFER_INDEX_1);

  while (k_loop_offset + LOOP_TILE_K * 2 < K) {

    alternate_mma_and_ldm(LDG_SM_BUFFER_INDEX + 1, MMA_REG_BUFFER_INDEX_1, MMA_REG_BUFFER_INDEX_0, true, true);

    LDG_SM_BUFFER_INDEX ^= 2;
    k_loop_offset += LOOP_TILE_K * 2;
    ldg_reg_2_sm(LDG_SM_BUFFER_INDEX, LDG_REG_BUFFER_INDEX_0);
    ldg_reg_2_sm(LDG_SM_BUFFER_INDEX + 1, LDG_REG_BUFFER_INDEX_1);

    __syncthreads();

    global_2_ldg_reg(k_loop_offset, LDG_REG_BUFFER_INDEX_0);
    global_2_ldg_reg(k_loop_offset + LOOP_TILE_K, LDG_REG_BUFFER_INDEX_1);

    alternate_mma_and_ldm(LDG_SM_BUFFER_INDEX, MMA_REG_BUFFER_INDEX_0, MMA_REG_BUFFER_INDEX_1, true, true);
  }
  {
    alternate_mma_and_ldm(LDG_SM_BUFFER_INDEX + 1, MMA_REG_BUFFER_INDEX_1, MMA_REG_BUFFER_INDEX_0, true, true);

    LDG_SM_BUFFER_INDEX ^= 2;
    k_loop_offset += LOOP_TILE_K * 2;
    ldg_reg_2_sm(LDG_SM_BUFFER_INDEX, LDG_REG_BUFFER_INDEX_0);
    ldg_reg_2_sm(LDG_SM_BUFFER_INDEX + 1, LDG_REG_BUFFER_INDEX_1);

    __syncthreads();

    alternate_mma_and_ldm(LDG_SM_BUFFER_INDEX, MMA_REG_BUFFER_INDEX_0, MMA_REG_BUFFER_INDEX_1, true, true);
  }

  alternate_mma_and_ldm(LDG_SM_BUFFER_INDEX + 1, MMA_REG_BUFFER_INDEX_1, MMA_REG_BUFFER_INDEX_0, true, true);

  alternate_mma_and_ldm(LDG_SM_BUFFER_INDEX, MMA_REG_BUFFER_INDEX_0, MMA_REG_BUFFER_INDEX_1, false, true);

  for (int mg = 0; mg < M_GROUP_COUNT_PER_WARP; ++mg) {
    for (int ng = 0; ng < N_GROUP_COUNT_PER_WARP; ++ng) {
      // int m = m_block_offset + m_warp_offset + mg * 8;
      // int n = n_block_offset + n_warp_offset + ng * 16;
      T casted[4] = {
        C_mma_reg[mg][ng][0],
        C_mma_reg[mg][ng][1],
        C_mma_reg[mg][ng][2],
        C_mma_reg[mg][ng][3],
      };
      shfl_23_and_01(casted, 0x4, lane_id);
      T                store[4]           = {casted[0], casted[2], casted[1], casted[3]};
      static const int lane_2_n_offset[8] = {0, 8, 2, 10, 4, 12, 6, 14};
      int              m                  = m_block_offset + m_warp_offset + mg * 8 + lane_id % 4 * 2;
      int              n                  = n_block_offset + n_warp_offset + ng * 16 + lane_2_n_offset[lane_id / 4];
      STORE_FLOAT(C[OFFSET(m, n, N)], store[0]);
      STORE_FLOAT(C[OFFSET(m + 1, n, N)], store[2]);
    }
  }
#undef global_2_ldg_reg
#undef ldg_reg_2_sm
#undef sm_2_A_mma_reg
#undef sm_2_B_mma_reg
#undef mma_and_ldm
#undef alternate_mma_and_ldm
}

template<typename T, int BLOCK_TILE_M, int BLOCK_TILE_N, int WARP_TILE_M, int WARP_TILE_N>
__global__ void
fp16_mma_m16n8k16_ldmatrix_trans__overlap_global_2_sm__quadra_buffer__reduce_instructions__reorder_instructions__overlap_reg_2_global(
  const T* A, const T* B, T* C, int M, int N, int K)
{
  constexpr int WARP_COUNT   = BLOCK_TILE_M / WARP_TILE_M * BLOCK_TILE_N / WARP_TILE_N;
  constexpr int THREAD_COUNT = WARP_COUNT * 32;

  constexpr int LOOP_TILE_K            = 16;
  constexpr int LDG_SM_BUFFER_SIZE     = 4;
  constexpr int LDG_REG_BUFFER_SIZE    = 2;
  constexpr int LDG_REG_BUFFER_INDEX_0 = 0;
  constexpr int LDG_REG_BUFFER_INDEX_1 = 1;

  constexpr int A_sm_dim0 = LDG_SM_BUFFER_SIZE;
  constexpr int A_sm_dim1 = 2;
  constexpr int A_sm_dim2 = BLOCK_TILE_M;
  constexpr int A_sm_dim3 = LOOP_TILE_K / 2;
  constexpr int B_sm_dim0 = LDG_SM_BUFFER_SIZE;
  constexpr int B_sm_dim1 = LOOP_TILE_K / 8;
  constexpr int B_sm_dim2 = BLOCK_TILE_N / 8;
  constexpr int B_sm_dim3 = 64;

  // The 64 elements of type T in each 8x8 matrix are stored consecutively in a single layer of shared memory.
  __shared__ union {
    struct {
      T A_sm[A_sm_dim0 * A_sm_dim1 * A_sm_dim2 * A_sm_dim3];
      T B_sm[B_sm_dim0 * B_sm_dim1 * B_sm_dim2 * B_sm_dim3];
    } mma;
    static_assert(WARP_TILE_N % 16 == 0);
    T result[WARP_COUNT][WARP_TILE_N / 16][WARP_TILE_M][16];
  } data;

  static_assert(BLOCK_TILE_M * LOOP_TILE_K % THREAD_COUNT == 0);
  static_assert(BLOCK_TILE_M * LOOP_TILE_K / THREAD_COUNT % 8 == 0);
  constexpr int A_LDG_COUNT_PER_THREAD = BLOCK_TILE_M * LOOP_TILE_K / THREAD_COUNT;
  constexpr int A_LDG_LOOP_COUNT       = A_LDG_COUNT_PER_THREAD / 8;
  // clang-format off
  // This is the thread layout of the same warp that loads matrix A, where each thread reads M1xK8 elements of type T at a
  // loop iteration.
  // T0  T16
  // T1  T17
  // T2  T18
  // ... ...
  // T14 T30
  // T15 T31
  // clang-format on
  float A_ldg_reg[LDG_REG_BUFFER_SIZE][A_LDG_LOOP_COUNT][4];

  static_assert(BLOCK_TILE_N * LOOP_TILE_K % THREAD_COUNT == 0);
  static_assert(BLOCK_TILE_N * LOOP_TILE_K / THREAD_COUNT % 8 == 0);
  constexpr int B_LDG_COUNT_PER_THREAD = BLOCK_TILE_N * LOOP_TILE_K / THREAD_COUNT;
  constexpr int B_LDG_LOOP_COUNT       = B_LDG_COUNT_PER_THREAD / 8;
  // clang-format off
  // This is the thread layout of the same warp that loads matrix B, where each thread reads K1xN8 elements of type T at a
  // loop iteration.
  // T0  T16
  // T1  T17
  // T2  T18
  // ... ...
  // T14 T30
  // T15 T31
  // clang-format on
  float B_ldg_reg[LDG_REG_BUFFER_SIZE][B_LDG_LOOP_COUNT][4];

  const int m_block_offset = blockIdx.y * BLOCK_TILE_M;
  const int n_block_offset = blockIdx.x * BLOCK_TILE_N;

  const int warp_id                 = threadIdx.x / 32;
  const int lane_id                 = threadIdx.x % 32;
  const int transposed_lane_id_mask = (lane_id / 8 == 0 || lane_id / 8 == 3) ? 0x00 : 0x18;
  const int transposed_lane_id      = lane_id ^ transposed_lane_id_mask;

  constexpr int M_MMA_WARP_COUNT       = BLOCK_TILE_M / WARP_TILE_M;
  constexpr int M_GROUP_COUNT_PER_WARP = WARP_TILE_M / 8;
  constexpr int N_GROUP_COUNT_PER_WARP = WARP_TILE_N / 16;

  constexpr int MMA_REG_BUFFER_SIZE    = 2;
  constexpr int MMA_REG_BUFFER_INDEX_0 = 0;
  constexpr int MMA_REG_BUFFER_INDEX_1 = 1;
  T             A_mma_reg[MMA_REG_BUFFER_SIZE][M_GROUP_COUNT_PER_WARP][4];
  T             B_mma_reg[MMA_REG_BUFFER_SIZE][N_GROUP_COUNT_PER_WARP][8];
  float         C_mma_reg[M_GROUP_COUNT_PER_WARP][N_GROUP_COUNT_PER_WARP][4] = {0};
  T             C_transposed[M_GROUP_COUNT_PER_WARP][N_GROUP_COUNT_PER_WARP][4];

  const int m_warp_offset = warp_id % M_MMA_WARP_COUNT * WARP_TILE_M;
  const int n_warp_offset = warp_id / M_MMA_WARP_COUNT * WARP_TILE_N;

  const int A_ldg_reg_2_A_sm_partial_offset =
    lane_id / 16 * A_sm_dim2 * A_sm_dim3 + (warp_id * 16 + lane_id % 16) * A_sm_dim3;

  const int B_ldg_reg_2_B_sm_partial_offset =
    (lane_id % 16) / 8 * B_sm_dim2 * B_sm_dim3 + (lane_id % 16) % 8 * 8 + (warp_id * 2 + lane_id / 16) * B_sm_dim3;

  const int A_global_partial_offset = (m_block_offset + warp_id * 16 + lane_id % 16) * K + lane_id / 16 * 8;
  const int B_global_partial_offset = lane_id % 16 * N + n_block_offset + warp_id * 16 + lane_id / 16 * 8;

  const T* A_global_ptr_for_ldg[A_LDG_LOOP_COUNT];
  const T* B_global_ptr_for_ldg[B_LDG_LOOP_COUNT];
  for (int loop = 0; loop < A_LDG_LOOP_COUNT; ++loop) {
    A_global_ptr_for_ldg[loop] = &A[A_global_partial_offset + loop * WARP_COUNT * 16 * K];
  }
  for (int loop = 0; loop < B_LDG_LOOP_COUNT; ++loop) {
    B_global_ptr_for_ldg[loop] = &B[B_global_partial_offset + loop * WARP_COUNT * 16];
  }

  const T* A_sm_ptr_for_ldg[A_LDG_LOOP_COUNT];
  const T* B_sm_ptr_for_ldg[B_LDG_LOOP_COUNT];
  for (int loop = 0; loop < A_LDG_LOOP_COUNT; ++loop) {
    A_sm_ptr_for_ldg[loop] = &data.mma.A_sm[A_ldg_reg_2_A_sm_partial_offset + loop * WARP_COUNT * 16 * A_sm_dim3];
  }
  for (int loop = 0; loop < B_LDG_LOOP_COUNT; ++loop) {
    B_sm_ptr_for_ldg[loop] = &data.mma.B_sm[B_ldg_reg_2_B_sm_partial_offset + loop * WARP_COUNT * 2 * B_sm_dim3];
  }

  const int A_sm_2_A_mma_reg_partial_offset =
    lane_id % 16 / 8 * A_sm_dim2 * A_sm_dim3 + (m_warp_offset + lane_id % 8) * A_sm_dim3;

  const int B_sm_2_B_mma_reg_partial_offset = transposed_lane_id % 16 / 8 * B_sm_dim2 * B_sm_dim3
                                              + (n_warp_offset + transposed_lane_id / 16 * 8) / 8 * B_sm_dim3
                                              + transposed_lane_id % 8 * 8;

  const T* A_sm_ptr_for_mma[M_GROUP_COUNT_PER_WARP];
  const T* B_sm_ptr_for_mma[N_GROUP_COUNT_PER_WARP];

  for (int group = 0; group < M_GROUP_COUNT_PER_WARP; ++group) {
    A_sm_ptr_for_mma[group] = &data.mma.A_sm[A_sm_2_A_mma_reg_partial_offset + (group + lane_id / 16) * 8 * A_sm_dim3];
  }
  for (int group = 0; group < N_GROUP_COUNT_PER_WARP; ++group) {
    B_sm_ptr_for_mma[group] = &data.mma.B_sm[B_sm_2_B_mma_reg_partial_offset + (group * 2 * B_sm_dim3)];
  }

#define global_2_ldg_reg(k_loop_offset, ldg_reg_buffer_index)                                                          \
  {                                                                                                                    \
    _Pragma("unroll") for (int loop = 0; loop < A_LDG_LOOP_COUNT; ++loop)                                              \
    {                                                                                                                  \
      /* const int m = (loop * WARP_COUNT + warp_id) * 16 + lane_id % 16; */                                           \
      /* const int k = lane_id / 16 * 8; */                                                                            \
      FETCH_FLOAT4_WITH_PTR(&A_ldg_reg[ldg_reg_buffer_index][loop][0], A_global_ptr_for_ldg[loop] + k_loop_offset);    \
    }                                                                                                                  \
    _Pragma("unroll") for (int loop = 0; loop < B_LDG_LOOP_COUNT; ++loop)                                              \
    {                                                                                                                  \
      /* const int k = lane_id % 16;                                           */                                      \
      /* const int n = (loop * WARP_COUNT + warp_id) * 16 + lane_id / 16 * 8;  */                                      \
      FETCH_FLOAT4_WITH_PTR(&B_ldg_reg[ldg_reg_buffer_index][loop][0],                                                 \
                            B_global_ptr_for_ldg[loop] + (k_loop_offset) * N);                                         \
    }                                                                                                                  \
  }

#define ldg_reg_2_sm(ldg_sm_buffer_index, ldg_reg_buffer_index)                                                        \
  {                                                                                                                    \
    _Pragma("unroll") for (int loop = 0; loop < A_LDG_LOOP_COUNT; ++loop)                                              \
    {                                                                                                                  \
      /* const int m = (loop * WARP_COUNT + warp_id) * 16 + lane_id % 16; */                                           \
      /* const int k = lane_id / 16 * 8;  */                                                                           \
      STORE_FLOAT4_WITH_PTR(A_sm_ptr_for_ldg[loop] + (ldg_sm_buffer_index) * A_sm_dim1 * A_sm_dim2 * A_sm_dim3,        \
                            &A_ldg_reg[ldg_reg_buffer_index][loop][0]);                                                \
    }                                                                                                                  \
    _Pragma("unroll") for (int loop = 0; loop < B_LDG_LOOP_COUNT; ++loop)                                              \
    {                                                                                                                  \
      /*const int k = lane_id % 16; */                                                                                 \
      /*const int n = (loop * WARP_COUNT + warp_id) * 16 + lane_id / 16 * 8;*/                                         \
      STORE_FLOAT4_WITH_PTR(B_sm_ptr_for_ldg[loop] + (ldg_sm_buffer_index) * B_sm_dim1 * B_sm_dim2 * B_sm_dim3,        \
                            &B_ldg_reg[ldg_reg_buffer_index][loop][0]);                                                \
    }                                                                                                                  \
  }

#define sm_2_A_mma_reg(ldg_sm_buffer_index, mma_reg_buffer_index, group)                                               \
  if constexpr (M_GROUP_COUNT_PER_WARP == 1) {                                                                         \
    /* for (int group = 0; group < M_GROUP_COUNT_PER_WARP; ++group) */ {                                               \
      uint32_t src =                                                                                                   \
        __cvta_generic_to_shared(A_sm_ptr_for_mma[group] + (ldg_sm_buffer_index) * A_sm_dim1 * A_sm_dim2 * A_sm_dim3); \
      asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"                                          \
                   : "=r"(*(uint32_t*)&A_mma_reg[mma_reg_buffer_index][group][0]),                                     \
                     "=r"(*(uint32_t*)&A_mma_reg[mma_reg_buffer_index][group][2])                                      \
                   : "r"(src));                                                                                        \
    }                                                                                                                  \
  }                                                                                                                    \
  else if constexpr (M_GROUP_COUNT_PER_WARP % 2 == 0) {                                                                \
    /*for (int group = 0; group < M_GROUP_COUNT_PER_WARP; group += 2) */ {                                             \
      uint32_t src =                                                                                                   \
        __cvta_generic_to_shared(A_sm_ptr_for_mma[group] + (ldg_sm_buffer_index) * A_sm_dim1 * A_sm_dim2 * A_sm_dim3); \
      asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"                                  \
                   : "=r"(*(uint32_t*)&A_mma_reg[mma_reg_buffer_index][group][0]),                                     \
                     "=r"(*(uint32_t*)&A_mma_reg[mma_reg_buffer_index][group][2]),                                     \
                     "=r"(*(uint32_t*)&A_mma_reg[mma_reg_buffer_index][group + 1][0]),                                 \
                     "=r"(*(uint32_t*)&A_mma_reg[mma_reg_buffer_index][group + 1][2])                                  \
                   : "r"(src));                                                                                        \
    }                                                                                                                  \
  }                                                                                                                    \
  else {                                                                                                               \
    static_assert(M_GROUP_COUNT_PER_WARP == 1 || M_GROUP_COUNT_PER_WARP % 2 == 0);                                     \
  }

#define sm_2_B_mma_reg(ldg_sm_buffer_index, mma_reg_buffer_index, group)                                               \
  /* for (int group = 0; group < N_GROUP_COUNT_PER_WARP; ++group) */ {                                                 \
    uint32_t src =                                                                                                     \
      __cvta_generic_to_shared(B_sm_ptr_for_mma[group] + (ldg_sm_buffer_index) * B_sm_dim1 * B_sm_dim2 * B_sm_dim3);   \
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"                              \
                 : "=r"(*(uint32_t*)&B_mma_reg[mma_reg_buffer_index][group][0]),                                       \
                   "=r"(*(uint32_t*)&B_mma_reg[mma_reg_buffer_index][group][2]),                                       \
                   "=r"(*(uint32_t*)&B_mma_reg[mma_reg_buffer_index][group][4]),                                       \
                   "=r"(*(uint32_t*)&B_mma_reg[mma_reg_buffer_index][group][6])                                        \
                 : "r"(src));                                                                                          \
  }

#define mma_ldm_stg(                                                                                                   \
  ldm_sm_buffer_index, ldm_reg_buffer_index, mma_reg_buffer_index, rank, ldm_switch, mma_switch, stg_switch)           \
  {                                                                                                                    \
    if constexpr (mma_switch && rank < M_GROUP_COUNT_PER_WARP * N_GROUP_COUNT_PER_WARP) {                              \
      constexpr int mg = rank % M_GROUP_COUNT_PER_WARP;                                                                \
      constexpr int ng = rank / M_GROUP_COUNT_PER_WARP;                                                                \
      mma_m16n8k16_row_col(C_mma_reg[mg][ng],                                                                          \
                           B_mma_reg[mma_reg_buffer_index][ng],                                                        \
                           A_mma_reg[mma_reg_buffer_index][mg],                                                        \
                           C_mma_reg[mg][ng]);                                                                         \
    }                                                                                                                  \
    static_assert(M_GROUP_COUNT_PER_WARP == 1 || M_GROUP_COUNT_PER_WARP % 2 == 0);                                     \
    if constexpr (ldm_switch && rank < (M_GROUP_COUNT_PER_WARP + 1) / 2) {                                             \
      sm_2_A_mma_reg(ldm_sm_buffer_index, ldm_reg_buffer_index, rank * 2);                                             \
    }                                                                                                                  \
    if constexpr (ldm_switch && (M_GROUP_COUNT_PER_WARP + 1) / 2 <= rank                                               \
                  && rank < (M_GROUP_COUNT_PER_WARP + 1) / 2 + N_GROUP_COUNT_PER_WARP) {                               \
      sm_2_B_mma_reg(ldm_sm_buffer_index, ldm_reg_buffer_index, rank - (M_GROUP_COUNT_PER_WARP + 1) / 2);              \
    }                                                                                                                  \
    if constexpr (stg_switch && rank < M_GROUP_COUNT_PER_WARP * N_GROUP_COUNT_PER_WARP) {                              \
      constexpr int mg = rank % M_GROUP_COUNT_PER_WARP;                                                                \
      constexpr int ng = rank / M_GROUP_COUNT_PER_WARP;                                                                \
      T casted[4]      = {C_mma_reg[mg][ng][0], C_mma_reg[mg][ng][1], C_mma_reg[mg][ng][2], C_mma_reg[mg][ng][3]};     \
      asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n"                                                   \
                   : "=r"(*(uint32_t*)&C_transposed[mg][ng][0])                                                        \
                   : "r"(*(uint32_t*)&casted[0]));                                                                     \
      asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n"                                                   \
                   : "=r"(*(uint32_t*)&C_transposed[mg][ng][2])                                                        \
                   : "r"(*(uint32_t*)&casted[2]));                                                                     \
      int m = m_block_offset + m_warp_offset + mg * 8 + lane_id / 4;                                                   \
      int n = n_block_offset + n_warp_offset + ng * 16 + lane_id % 4 * 2;                                              \
      STORE_FLOAT(C[OFFSET(m, n, N)], C_transposed[mg][ng][0]);                                                        \
      STORE_FLOAT(C[OFFSET(m, n + 8, N)], C_transposed[mg][ng][2]);                                                    \
    }                                                                                                                  \
  }

// FIXME This code is really stupid. Please find a way to optimize it as soon as possible.
#define alternate_mma_ldm_stg(                                                                                         \
  ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, ldm_switch, mma_switch, stg_switch)                          \
  static_assert(M_GROUP_COUNT_PER_WARP * N_GROUP_COUNT_PER_WARP <= 32);                                                \
  mma_ldm_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, 0, ldm_switch, mma_switch, stg_switch);          \
  mma_ldm_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, 1, ldm_switch, mma_switch, stg_switch);          \
  mma_ldm_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, 2, ldm_switch, mma_switch, stg_switch);          \
  mma_ldm_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, 3, ldm_switch, mma_switch, stg_switch);          \
  mma_ldm_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, 4, ldm_switch, mma_switch, stg_switch);          \
  mma_ldm_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, 5, ldm_switch, mma_switch, stg_switch);          \
  mma_ldm_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, 6, ldm_switch, mma_switch, stg_switch);          \
  mma_ldm_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, 7, ldm_switch, mma_switch, stg_switch);          \
  mma_ldm_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, 8, ldm_switch, mma_switch, stg_switch);          \
  mma_ldm_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, 9, ldm_switch, mma_switch, stg_switch);          \
  mma_ldm_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, 10, ldm_switch, mma_switch, stg_switch);         \
  mma_ldm_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, 11, ldm_switch, mma_switch, stg_switch);         \
  mma_ldm_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, 12, ldm_switch, mma_switch, stg_switch);         \
  mma_ldm_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, 13, ldm_switch, mma_switch, stg_switch);         \
  mma_ldm_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, 14, ldm_switch, mma_switch, stg_switch);         \
  mma_ldm_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, 15, ldm_switch, mma_switch, stg_switch);         \
  mma_ldm_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, 16, ldm_switch, mma_switch, stg_switch);         \
  mma_ldm_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, 17, ldm_switch, mma_switch, stg_switch);         \
  mma_ldm_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, 18, ldm_switch, mma_switch, stg_switch);         \
  mma_ldm_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, 19, ldm_switch, mma_switch, stg_switch);         \
  mma_ldm_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, 20, ldm_switch, mma_switch, stg_switch);         \
  mma_ldm_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, 21, ldm_switch, mma_switch, stg_switch);         \
  mma_ldm_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, 22, ldm_switch, mma_switch, stg_switch);         \
  mma_ldm_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, 23, ldm_switch, mma_switch, stg_switch);         \
  mma_ldm_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, 24, ldm_switch, mma_switch, stg_switch);         \
  mma_ldm_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, 25, ldm_switch, mma_switch, stg_switch);         \
  mma_ldm_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, 26, ldm_switch, mma_switch, stg_switch);         \
  mma_ldm_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, 27, ldm_switch, mma_switch, stg_switch);         \
  mma_ldm_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, 28, ldm_switch, mma_switch, stg_switch);         \
  mma_ldm_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, 29, ldm_switch, mma_switch, stg_switch);         \
  mma_ldm_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, 30, ldm_switch, mma_switch, stg_switch);         \
  mma_ldm_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, 31, ldm_switch, mma_switch, stg_switch);

  global_2_ldg_reg(0, LDG_REG_BUFFER_INDEX_0);
  global_2_ldg_reg(LOOP_TILE_K, LDG_REG_BUFFER_INDEX_1);
  ldg_reg_2_sm(0, LDG_REG_BUFFER_INDEX_0);
  ldg_reg_2_sm(1, LDG_REG_BUFFER_INDEX_1);
  __syncthreads();

  int LDG_SM_BUFFER_INDEX = 0;
  int k_loop_offset       = LOOP_TILE_K * 2;

  alternate_mma_ldm_stg(LDG_SM_BUFFER_INDEX, MMA_REG_BUFFER_INDEX_0, 0, true, false, false);

  global_2_ldg_reg(k_loop_offset, LDG_REG_BUFFER_INDEX_0);
  global_2_ldg_reg(k_loop_offset + LOOP_TILE_K, LDG_REG_BUFFER_INDEX_1);

  while (k_loop_offset + LOOP_TILE_K * 2 < K) {

    alternate_mma_ldm_stg(LDG_SM_BUFFER_INDEX + 1, MMA_REG_BUFFER_INDEX_1, MMA_REG_BUFFER_INDEX_0, true, true, false);

    LDG_SM_BUFFER_INDEX ^= 2;
    k_loop_offset += LOOP_TILE_K * 2;
    ldg_reg_2_sm(LDG_SM_BUFFER_INDEX, LDG_REG_BUFFER_INDEX_0);
    ldg_reg_2_sm(LDG_SM_BUFFER_INDEX + 1, LDG_REG_BUFFER_INDEX_1);

    __syncthreads();

    global_2_ldg_reg(k_loop_offset, LDG_REG_BUFFER_INDEX_0);
    global_2_ldg_reg(k_loop_offset + LOOP_TILE_K, LDG_REG_BUFFER_INDEX_1);

    alternate_mma_ldm_stg(LDG_SM_BUFFER_INDEX, MMA_REG_BUFFER_INDEX_0, MMA_REG_BUFFER_INDEX_1, true, true, false);
  }
  {
    alternate_mma_ldm_stg(LDG_SM_BUFFER_INDEX + 1, MMA_REG_BUFFER_INDEX_1, MMA_REG_BUFFER_INDEX_0, true, true, false);

    LDG_SM_BUFFER_INDEX ^= 2;
    k_loop_offset += LOOP_TILE_K * 2;
    ldg_reg_2_sm(LDG_SM_BUFFER_INDEX, LDG_REG_BUFFER_INDEX_0);
    ldg_reg_2_sm(LDG_SM_BUFFER_INDEX + 1, LDG_REG_BUFFER_INDEX_1);

    __syncthreads();

    alternate_mma_ldm_stg(LDG_SM_BUFFER_INDEX, MMA_REG_BUFFER_INDEX_0, MMA_REG_BUFFER_INDEX_1, true, true, false);
  }

  {
    alternate_mma_ldm_stg(LDG_SM_BUFFER_INDEX + 1, MMA_REG_BUFFER_INDEX_1, MMA_REG_BUFFER_INDEX_0, true, true, false);

    alternate_mma_ldm_stg(LDG_SM_BUFFER_INDEX, MMA_REG_BUFFER_INDEX_0, MMA_REG_BUFFER_INDEX_1, false, true, true);
  }

#undef global_2_ldg_reg
#undef ldg_reg_2_sm
#undef sm_2_A_mma_reg
#undef sm_2_B_mma_reg
#undef mma_ldm_stg
#undef alternate_mma_ldm_stg
}

template<typename T, int BLOCK_TILE_M, int BLOCK_TILE_N, int WARP_TILE_M, int WARP_TILE_N>
__global__ void
fp16_mma_m16n8k16_ldmatrix_trans__overlap_global_2_sm__quadra_buffer__reduce_instructions__reorder_instructions__overlap_reg_2_global__stg_memory_coalecesing(
  const T* A, const T* B, T* C, int M, int N, int K)
{
  constexpr int WARP_COUNT   = BLOCK_TILE_M / WARP_TILE_M * BLOCK_TILE_N / WARP_TILE_N;
  constexpr int THREAD_COUNT = WARP_COUNT * 32;

  constexpr int LOOP_TILE_K            = 16;
  constexpr int LDG_SM_BUFFER_SIZE     = 4;
  constexpr int LDG_REG_BUFFER_SIZE    = 2;
  constexpr int LDG_REG_BUFFER_INDEX_0 = 0;
  constexpr int LDG_REG_BUFFER_INDEX_1 = 1;

  constexpr int A_sm_dim0 = LDG_SM_BUFFER_SIZE;
  constexpr int A_sm_dim1 = 2;
  constexpr int A_sm_dim2 = BLOCK_TILE_M;
  constexpr int A_sm_dim3 = LOOP_TILE_K / 2;
  constexpr int B_sm_dim0 = LDG_SM_BUFFER_SIZE;
  constexpr int B_sm_dim1 = LOOP_TILE_K / 8;
  constexpr int B_sm_dim2 = BLOCK_TILE_N / 8;
  constexpr int B_sm_dim3 = 64;

  // The 64 elements of type T in each 8x8 matrix are stored consecutively in a single layer of shared memory.
  __shared__ union {
    struct {
      T A_sm[A_sm_dim0 * A_sm_dim1 * A_sm_dim2 * A_sm_dim3];
      T B_sm[B_sm_dim0 * B_sm_dim1 * B_sm_dim2 * B_sm_dim3];
    } mma;
    static_assert(WARP_TILE_N % 16 == 0);
    T result[WARP_COUNT][WARP_TILE_N / 16][WARP_TILE_M][16];
  } data;

  static_assert(BLOCK_TILE_M * LOOP_TILE_K % THREAD_COUNT == 0);
  static_assert(BLOCK_TILE_M * LOOP_TILE_K / THREAD_COUNT % 8 == 0);
  constexpr int A_LDG_COUNT_PER_THREAD = BLOCK_TILE_M * LOOP_TILE_K / THREAD_COUNT;
  constexpr int A_LDG_LOOP_COUNT       = A_LDG_COUNT_PER_THREAD / 8;
  // clang-format off
  // This is the thread layout of the same warp that loads matrix A, where each thread reads M1xK8 elements of type T at a
  // loop iteration.
  // T0  T16
  // T1  T17
  // T2  T18
  // ... ...
  // T14 T30
  // T15 T31
  // clang-format on
  float A_ldg_reg[LDG_REG_BUFFER_SIZE][A_LDG_LOOP_COUNT][4];

  static_assert(BLOCK_TILE_N * LOOP_TILE_K % THREAD_COUNT == 0);
  static_assert(BLOCK_TILE_N * LOOP_TILE_K / THREAD_COUNT % 8 == 0);
  constexpr int B_LDG_COUNT_PER_THREAD = BLOCK_TILE_N * LOOP_TILE_K / THREAD_COUNT;
  constexpr int B_LDG_LOOP_COUNT       = B_LDG_COUNT_PER_THREAD / 8;
  // clang-format off
  // This is the thread layout of the same warp that loads matrix B, where each thread reads K1xN8 elements of type T at a
  // loop iteration.
  // T0  T16
  // T1  T17
  // T2  T18
  // ... ...
  // T14 T30
  // T15 T31
  // clang-format on
  float B_ldg_reg[LDG_REG_BUFFER_SIZE][B_LDG_LOOP_COUNT][4];

  const int m_block_offset = blockIdx.y * BLOCK_TILE_M;
  const int n_block_offset = blockIdx.x * BLOCK_TILE_N;

  const int warp_id                 = threadIdx.x / 32;
  const int lane_id                 = threadIdx.x % 32;
  const int transposed_lane_id_mask = (lane_id / 8 == 0 || lane_id / 8 == 3) ? 0x00 : 0x18;
  const int transposed_lane_id      = lane_id ^ transposed_lane_id_mask;

  constexpr int M_MMA_WARP_COUNT       = BLOCK_TILE_M / WARP_TILE_M;
  constexpr int M_GROUP_COUNT_PER_WARP = WARP_TILE_M / 8;
  constexpr int N_GROUP_COUNT_PER_WARP = WARP_TILE_N / 16;

  constexpr int MMA_REG_BUFFER_SIZE    = 2;
  constexpr int MMA_REG_BUFFER_INDEX_0 = 0;
  constexpr int MMA_REG_BUFFER_INDEX_1 = 1;
  T             A_mma_reg[MMA_REG_BUFFER_SIZE][M_GROUP_COUNT_PER_WARP][4];
  T             B_mma_reg[MMA_REG_BUFFER_SIZE][N_GROUP_COUNT_PER_WARP][8];
  float         C_mma_reg[M_GROUP_COUNT_PER_WARP][N_GROUP_COUNT_PER_WARP][4] = {0};
  static_assert(N_GROUP_COUNT_PER_WARP % 2 == 0);
  union _2x4_or_1x8 {
    T _2x4[2][4];
    T _1x8[8];
  };
  _2x4_or_1x8 C_transposed[M_GROUP_COUNT_PER_WARP][N_GROUP_COUNT_PER_WARP / 2];

  const int m_warp_offset = warp_id % M_MMA_WARP_COUNT * WARP_TILE_M;
  const int n_warp_offset = warp_id / M_MMA_WARP_COUNT * WARP_TILE_N;

  const int A_ldg_reg_2_A_sm_partial_offset =
    lane_id / 16 * A_sm_dim2 * A_sm_dim3 + (warp_id * 16 + lane_id % 16) * A_sm_dim3;

  const int B_ldg_reg_2_B_sm_partial_offset =
    (lane_id % 16) / 8 * B_sm_dim2 * B_sm_dim3 + (lane_id % 16) % 8 * 8 + (warp_id * 2 + lane_id / 16) * B_sm_dim3;

  const int A_global_partial_offset = (m_block_offset + warp_id * 16 + lane_id % 16) * K + lane_id / 16 * 8;
  const int B_global_partial_offset = lane_id % 16 * N + n_block_offset + warp_id * 16 + lane_id / 16 * 8;

  const T* A_global_ptr_for_ldg[A_LDG_LOOP_COUNT];
  const T* B_global_ptr_for_ldg[B_LDG_LOOP_COUNT];
  for (int loop = 0; loop < A_LDG_LOOP_COUNT; ++loop) {
    A_global_ptr_for_ldg[loop] = &A[A_global_partial_offset + loop * WARP_COUNT * 16 * K];
  }
  for (int loop = 0; loop < B_LDG_LOOP_COUNT; ++loop) {
    B_global_ptr_for_ldg[loop] = &B[B_global_partial_offset + loop * WARP_COUNT * 16];
  }

  const T* A_sm_ptr_for_ldg[A_LDG_LOOP_COUNT];
  const T* B_sm_ptr_for_ldg[B_LDG_LOOP_COUNT];
  for (int loop = 0; loop < A_LDG_LOOP_COUNT; ++loop) {
    A_sm_ptr_for_ldg[loop] = &data.mma.A_sm[A_ldg_reg_2_A_sm_partial_offset + loop * WARP_COUNT * 16 * A_sm_dim3];
  }
  for (int loop = 0; loop < B_LDG_LOOP_COUNT; ++loop) {
    B_sm_ptr_for_ldg[loop] = &data.mma.B_sm[B_ldg_reg_2_B_sm_partial_offset + loop * WARP_COUNT * 2 * B_sm_dim3];
  }

  const int A_sm_2_A_mma_reg_partial_offset =
    lane_id % 16 / 8 * A_sm_dim2 * A_sm_dim3 + (m_warp_offset + lane_id % 8) * A_sm_dim3;

  const int B_sm_2_B_mma_reg_partial_offset = transposed_lane_id % 16 / 8 * B_sm_dim2 * B_sm_dim3
                                              + (n_warp_offset + transposed_lane_id / 16 * 8) / 8 * B_sm_dim3
                                              + transposed_lane_id % 8 * 8;

  const T* A_sm_ptr_for_mma[M_GROUP_COUNT_PER_WARP];
  const T* B_sm_ptr_for_mma[N_GROUP_COUNT_PER_WARP];

  for (int group = 0; group < M_GROUP_COUNT_PER_WARP; ++group) {
    A_sm_ptr_for_mma[group] = &data.mma.A_sm[A_sm_2_A_mma_reg_partial_offset + (group + lane_id / 16) * 8 * A_sm_dim3];
  }
  for (int group = 0; group < N_GROUP_COUNT_PER_WARP; ++group) {
    B_sm_ptr_for_mma[group] = &data.mma.B_sm[B_sm_2_B_mma_reg_partial_offset + (group * 2 * B_sm_dim3)];
  }

#define global_2_ldg_reg(k_loop_offset, ldg_reg_buffer_index)                                                          \
  {                                                                                                                    \
    _Pragma("unroll") for (int loop = 0; loop < A_LDG_LOOP_COUNT; ++loop)                                              \
    {                                                                                                                  \
      /* const int m = (loop * WARP_COUNT + warp_id) * 16 + lane_id % 16; */                                           \
      /* const int k = lane_id / 16 * 8; */                                                                            \
      FETCH_FLOAT4_WITH_PTR(&A_ldg_reg[ldg_reg_buffer_index][loop][0], A_global_ptr_for_ldg[loop] + k_loop_offset);    \
    }                                                                                                                  \
    _Pragma("unroll") for (int loop = 0; loop < B_LDG_LOOP_COUNT; ++loop)                                              \
    {                                                                                                                  \
      /* const int k = lane_id % 16;                                           */                                      \
      /* const int n = (loop * WARP_COUNT + warp_id) * 16 + lane_id / 16 * 8;  */                                      \
      FETCH_FLOAT4_WITH_PTR(&B_ldg_reg[ldg_reg_buffer_index][loop][0],                                                 \
                            B_global_ptr_for_ldg[loop] + (k_loop_offset) * N);                                         \
    }                                                                                                                  \
  }

#define ldg_reg_2_sm(ldg_sm_buffer_index, ldg_reg_buffer_index)                                                        \
  {                                                                                                                    \
    _Pragma("unroll") for (int loop = 0; loop < A_LDG_LOOP_COUNT; ++loop)                                              \
    {                                                                                                                  \
      /* const int m = (loop * WARP_COUNT + warp_id) * 16 + lane_id % 16; */                                           \
      /* const int k = lane_id / 16 * 8;  */                                                                           \
      STORE_FLOAT4_WITH_PTR(A_sm_ptr_for_ldg[loop] + (ldg_sm_buffer_index) * A_sm_dim1 * A_sm_dim2 * A_sm_dim3,        \
                            &A_ldg_reg[ldg_reg_buffer_index][loop][0]);                                                \
    }                                                                                                                  \
    _Pragma("unroll") for (int loop = 0; loop < B_LDG_LOOP_COUNT; ++loop)                                              \
    {                                                                                                                  \
      /*const int k = lane_id % 16; */                                                                                 \
      /*const int n = (loop * WARP_COUNT + warp_id) * 16 + lane_id / 16 * 8;*/                                         \
      STORE_FLOAT4_WITH_PTR(B_sm_ptr_for_ldg[loop] + (ldg_sm_buffer_index) * B_sm_dim1 * B_sm_dim2 * B_sm_dim3,        \
                            &B_ldg_reg[ldg_reg_buffer_index][loop][0]);                                                \
    }                                                                                                                  \
  }

#define sm_2_A_mma_reg(ldg_sm_buffer_index, mma_reg_buffer_index, group)                                               \
  if constexpr (M_GROUP_COUNT_PER_WARP == 1) {                                                                         \
    /* for (int group = 0; group < M_GROUP_COUNT_PER_WARP; ++group) */ {                                               \
      uint32_t src =                                                                                                   \
        __cvta_generic_to_shared(A_sm_ptr_for_mma[group] + (ldg_sm_buffer_index) * A_sm_dim1 * A_sm_dim2 * A_sm_dim3); \
      asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"                                          \
                   : "=r"(*(uint32_t*)&A_mma_reg[mma_reg_buffer_index][group][0]),                                     \
                     "=r"(*(uint32_t*)&A_mma_reg[mma_reg_buffer_index][group][2])                                      \
                   : "r"(src));                                                                                        \
    }                                                                                                                  \
  }                                                                                                                    \
  else if constexpr (M_GROUP_COUNT_PER_WARP % 2 == 0) {                                                                \
    /*for (int group = 0; group < M_GROUP_COUNT_PER_WARP; group += 2) */ {                                             \
      uint32_t src =                                                                                                   \
        __cvta_generic_to_shared(A_sm_ptr_for_mma[group] + (ldg_sm_buffer_index) * A_sm_dim1 * A_sm_dim2 * A_sm_dim3); \
      asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"                                  \
                   : "=r"(*(uint32_t*)&A_mma_reg[mma_reg_buffer_index][group][0]),                                     \
                     "=r"(*(uint32_t*)&A_mma_reg[mma_reg_buffer_index][group][2]),                                     \
                     "=r"(*(uint32_t*)&A_mma_reg[mma_reg_buffer_index][group + 1][0]),                                 \
                     "=r"(*(uint32_t*)&A_mma_reg[mma_reg_buffer_index][group + 1][2])                                  \
                   : "r"(src));                                                                                        \
    }                                                                                                                  \
  }                                                                                                                    \
  else {                                                                                                               \
    static_assert(M_GROUP_COUNT_PER_WARP == 1 || M_GROUP_COUNT_PER_WARP % 2 == 0);                                     \
  }

#define sm_2_B_mma_reg(ldg_sm_buffer_index, mma_reg_buffer_index, group)                                               \
  /* for (int group = 0; group < N_GROUP_COUNT_PER_WARP; ++group) */ {                                                 \
    uint32_t src =                                                                                                     \
      __cvta_generic_to_shared(B_sm_ptr_for_mma[group] + (ldg_sm_buffer_index) * B_sm_dim1 * B_sm_dim2 * B_sm_dim3);   \
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"                              \
                 : "=r"(*(uint32_t*)&B_mma_reg[mma_reg_buffer_index][group][0]),                                       \
                   "=r"(*(uint32_t*)&B_mma_reg[mma_reg_buffer_index][group][2]),                                       \
                   "=r"(*(uint32_t*)&B_mma_reg[mma_reg_buffer_index][group][4]),                                       \
                   "=r"(*(uint32_t*)&B_mma_reg[mma_reg_buffer_index][group][6])                                        \
                 : "r"(src));                                                                                          \
  }

  T* C_ptr = &C[(m_block_offset + m_warp_offset + lane_id / 4) * N + n_block_offset + n_warp_offset + lane_id % 4 * 8];

#define mma_ldm_stg(                                                                                                   \
  ldm_sm_buffer_index, ldm_reg_buffer_index, mma_reg_buffer_index, rank, ldm_switch, mma_switch, stg_switch)           \
  {                                                                                                                    \
    if constexpr (mma_switch && rank < M_GROUP_COUNT_PER_WARP * N_GROUP_COUNT_PER_WARP) {                              \
      constexpr int mg = rank % M_GROUP_COUNT_PER_WARP;                                                                \
      constexpr int ng = rank / M_GROUP_COUNT_PER_WARP;                                                                \
      mma_m16n8k16_row_col(C_mma_reg[mg][ng],                                                                          \
                           B_mma_reg[mma_reg_buffer_index][ng],                                                        \
                           A_mma_reg[mma_reg_buffer_index][mg],                                                        \
                           C_mma_reg[mg][ng]);                                                                         \
    }                                                                                                                  \
    static_assert(M_GROUP_COUNT_PER_WARP == 1 || M_GROUP_COUNT_PER_WARP % 2 == 0);                                     \
    if constexpr (ldm_switch && rank < (M_GROUP_COUNT_PER_WARP + 1) / 2) {                                             \
      sm_2_A_mma_reg(ldm_sm_buffer_index, ldm_reg_buffer_index, rank * 2);                                             \
    }                                                                                                                  \
    if constexpr (ldm_switch && (M_GROUP_COUNT_PER_WARP + 1) / 2 <= rank                                               \
                  && rank < (M_GROUP_COUNT_PER_WARP + 1) / 2 + N_GROUP_COUNT_PER_WARP) {                               \
      sm_2_B_mma_reg(ldm_sm_buffer_index, ldm_reg_buffer_index, rank - (M_GROUP_COUNT_PER_WARP + 1) / 2);              \
    }                                                                                                                  \
    if constexpr (stg_switch && rank < M_GROUP_COUNT_PER_WARP * N_GROUP_COUNT_PER_WARP) {                              \
      constexpr int mg = rank % M_GROUP_COUNT_PER_WARP;                                                                \
      constexpr int ng = rank / M_GROUP_COUNT_PER_WARP;                                                                \
      T casted[4]      = {C_mma_reg[mg][ng][0], C_mma_reg[mg][ng][1], C_mma_reg[mg][ng][2], C_mma_reg[mg][ng][3]};     \
      asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n"                                                   \
                   : "=r"(*(uint32_t*)&C_transposed[mg][ng / 2]._2x4[ng % 2][0])                                       \
                   : "r"(*(uint32_t*)&casted[0]));                                                                     \
      asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n"                                                   \
                   : "=r"(*(uint32_t*)&C_transposed[mg][ng / 2]._2x4[ng % 2][2])                                       \
                   : "r"(*(uint32_t*)&casted[2]));                                                                     \
      shfl_23_and_01(C_transposed[mg][ng / 2]._2x4[ng % 2], 0x1, lane_id);                                             \
      if constexpr ((ng + 1) % 2 == 0) {                                                                               \
        shfl_4567_and_0123(C_transposed[mg][ng / 2]._1x8, 0x2, lane_id);                                               \
        STORE_FLOAT4_WITH_PTR(C_ptr + mg * 8 * N + (ng - 1) * 16, &C_transposed[mg][ng / 2]._1x8[0]);                  \
      }                                                                                                                \
    }                                                                                                                  \
  }

// FIXME This code is really stupid. Please find a way to optimize it as soon as possible.
#define alternate_mma_ldm_stg(                                                                                         \
  ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, ldm_switch, mma_switch, stg_switch)                          \
  static_assert(M_GROUP_COUNT_PER_WARP * N_GROUP_COUNT_PER_WARP <= 32);                                                \
  mma_ldm_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, 0, ldm_switch, mma_switch, stg_switch);          \
  mma_ldm_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, 1, ldm_switch, mma_switch, stg_switch);          \
  mma_ldm_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, 2, ldm_switch, mma_switch, stg_switch);          \
  mma_ldm_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, 3, ldm_switch, mma_switch, stg_switch);          \
  mma_ldm_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, 4, ldm_switch, mma_switch, stg_switch);          \
  mma_ldm_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, 5, ldm_switch, mma_switch, stg_switch);          \
  mma_ldm_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, 6, ldm_switch, mma_switch, stg_switch);          \
  mma_ldm_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, 7, ldm_switch, mma_switch, stg_switch);          \
  mma_ldm_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, 8, ldm_switch, mma_switch, stg_switch);          \
  mma_ldm_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, 9, ldm_switch, mma_switch, stg_switch);          \
  mma_ldm_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, 10, ldm_switch, mma_switch, stg_switch);         \
  mma_ldm_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, 11, ldm_switch, mma_switch, stg_switch);         \
  mma_ldm_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, 12, ldm_switch, mma_switch, stg_switch);         \
  mma_ldm_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, 13, ldm_switch, mma_switch, stg_switch);         \
  mma_ldm_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, 14, ldm_switch, mma_switch, stg_switch);         \
  mma_ldm_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, 15, ldm_switch, mma_switch, stg_switch);         \
  mma_ldm_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, 16, ldm_switch, mma_switch, stg_switch);         \
  mma_ldm_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, 17, ldm_switch, mma_switch, stg_switch);         \
  mma_ldm_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, 18, ldm_switch, mma_switch, stg_switch);         \
  mma_ldm_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, 19, ldm_switch, mma_switch, stg_switch);         \
  mma_ldm_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, 20, ldm_switch, mma_switch, stg_switch);         \
  mma_ldm_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, 21, ldm_switch, mma_switch, stg_switch);         \
  mma_ldm_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, 22, ldm_switch, mma_switch, stg_switch);         \
  mma_ldm_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, 23, ldm_switch, mma_switch, stg_switch);         \
  mma_ldm_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, 24, ldm_switch, mma_switch, stg_switch);         \
  mma_ldm_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, 25, ldm_switch, mma_switch, stg_switch);         \
  mma_ldm_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, 26, ldm_switch, mma_switch, stg_switch);         \
  mma_ldm_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, 27, ldm_switch, mma_switch, stg_switch);         \
  mma_ldm_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, 28, ldm_switch, mma_switch, stg_switch);         \
  mma_ldm_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, 29, ldm_switch, mma_switch, stg_switch);         \
  mma_ldm_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, 30, ldm_switch, mma_switch, stg_switch);         \
  mma_ldm_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, 31, ldm_switch, mma_switch, stg_switch);

  global_2_ldg_reg(0, LDG_REG_BUFFER_INDEX_0);
  global_2_ldg_reg(LOOP_TILE_K, LDG_REG_BUFFER_INDEX_1);
  ldg_reg_2_sm(0, LDG_REG_BUFFER_INDEX_0);
  ldg_reg_2_sm(1, LDG_REG_BUFFER_INDEX_1);
  __syncthreads();

  int LDG_SM_BUFFER_INDEX = 0;
  int k_loop_offset       = LOOP_TILE_K * 2;

  alternate_mma_ldm_stg(LDG_SM_BUFFER_INDEX, MMA_REG_BUFFER_INDEX_0, 0, true, false, false);

  global_2_ldg_reg(k_loop_offset, LDG_REG_BUFFER_INDEX_0);
  global_2_ldg_reg(k_loop_offset + LOOP_TILE_K, LDG_REG_BUFFER_INDEX_1);

  while (k_loop_offset + LOOP_TILE_K * 2 < K) {

    alternate_mma_ldm_stg(LDG_SM_BUFFER_INDEX + 1, MMA_REG_BUFFER_INDEX_1, MMA_REG_BUFFER_INDEX_0, true, true, false);

    LDG_SM_BUFFER_INDEX ^= 2;
    k_loop_offset += LOOP_TILE_K * 2;
    ldg_reg_2_sm(LDG_SM_BUFFER_INDEX, LDG_REG_BUFFER_INDEX_0);
    ldg_reg_2_sm(LDG_SM_BUFFER_INDEX + 1, LDG_REG_BUFFER_INDEX_1);

    __syncthreads();

    global_2_ldg_reg(k_loop_offset, LDG_REG_BUFFER_INDEX_0);
    global_2_ldg_reg(k_loop_offset + LOOP_TILE_K, LDG_REG_BUFFER_INDEX_1);

    alternate_mma_ldm_stg(LDG_SM_BUFFER_INDEX, MMA_REG_BUFFER_INDEX_0, MMA_REG_BUFFER_INDEX_1, true, true, false);
  }
  {
    alternate_mma_ldm_stg(LDG_SM_BUFFER_INDEX + 1, MMA_REG_BUFFER_INDEX_1, MMA_REG_BUFFER_INDEX_0, true, true, false);

    LDG_SM_BUFFER_INDEX ^= 2;
    k_loop_offset += LOOP_TILE_K * 2;
    ldg_reg_2_sm(LDG_SM_BUFFER_INDEX, LDG_REG_BUFFER_INDEX_0);
    ldg_reg_2_sm(LDG_SM_BUFFER_INDEX + 1, LDG_REG_BUFFER_INDEX_1);

    __syncthreads();

    alternate_mma_ldm_stg(LDG_SM_BUFFER_INDEX, MMA_REG_BUFFER_INDEX_0, MMA_REG_BUFFER_INDEX_1, true, true, false);
  }

  {
    alternate_mma_ldm_stg(LDG_SM_BUFFER_INDEX + 1, MMA_REG_BUFFER_INDEX_1, MMA_REG_BUFFER_INDEX_0, true, true, false);

    alternate_mma_ldm_stg(LDG_SM_BUFFER_INDEX, MMA_REG_BUFFER_INDEX_0, MMA_REG_BUFFER_INDEX_1, false, true, true);
  }

#undef global_2_ldg_reg
#undef ldg_reg_2_sm
#undef sm_2_A_mma_reg
#undef sm_2_B_mma_reg
#undef mma_ldm_stg
#undef alternate_mma_ldm_stg
}

template<typename T, int BLOCK_TILE_M, int BLOCK_TILE_N, int WARP_TILE_M, int WARP_TILE_N>
__global__ void
fp16_mma_m16n8k16_ldmatrix_trans__overlap_global_2_sm__quadra_buffer__reduce_instructions__reorder_instructions__overlap_reg_2_global__stg_memory_coalecesing__overlap_sts(
  const T* A, const T* B, T* C, int M, int N, int K)
{
  constexpr int WARP_COUNT   = BLOCK_TILE_M / WARP_TILE_M * BLOCK_TILE_N / WARP_TILE_N;
  constexpr int THREAD_COUNT = WARP_COUNT * 32;

  constexpr int LOOP_TILE_K            = 16;
  constexpr int LDG_SM_BUFFER_SIZE     = 4;
  constexpr int LDG_REG_BUFFER_SIZE    = 2;
  constexpr int LDG_REG_BUFFER_INDEX_0 = 0;
  constexpr int LDG_REG_BUFFER_INDEX_1 = 1;

  constexpr int A_sm_dim0 = LDG_SM_BUFFER_SIZE;
  constexpr int A_sm_dim1 = 2;
  constexpr int A_sm_dim2 = BLOCK_TILE_M;
  constexpr int A_sm_dim3 = LOOP_TILE_K / 2;
  constexpr int B_sm_dim0 = LDG_SM_BUFFER_SIZE;
  constexpr int B_sm_dim1 = LOOP_TILE_K / 8;
  constexpr int B_sm_dim2 = BLOCK_TILE_N / 8;
  constexpr int B_sm_dim3 = 64;

  // The 64 elements of type T in each 8x8 matrix are stored consecutively in a single layer of shared memory.
  __shared__ union {
    struct {
      T A_sm[A_sm_dim0 * A_sm_dim1 * A_sm_dim2 * A_sm_dim3];
      T B_sm[B_sm_dim0 * B_sm_dim1 * B_sm_dim2 * B_sm_dim3];
    } mma;
    static_assert(WARP_TILE_N % 16 == 0);
    T result[WARP_COUNT][WARP_TILE_N / 16][WARP_TILE_M][16];
  } data;

  static_assert(BLOCK_TILE_M * LOOP_TILE_K % THREAD_COUNT == 0);
  static_assert(BLOCK_TILE_M * LOOP_TILE_K / THREAD_COUNT % 8 == 0);
  constexpr int A_LDG_COUNT_PER_THREAD = BLOCK_TILE_M * LOOP_TILE_K / THREAD_COUNT;
  constexpr int A_LDG_LOOP_COUNT       = A_LDG_COUNT_PER_THREAD / 8;
  // clang-format off
  // This is the thread layout of the same warp that loads matrix A, where each thread reads M1xK8 elements of type T at a
  // loop iteration.
  // T0  T16
  // T1  T17
  // T2  T18
  // ... ...
  // T14 T30
  // T15 T31
  // clang-format on
  float A_ldg_reg[LDG_REG_BUFFER_SIZE][A_LDG_LOOP_COUNT][4];

  static_assert(BLOCK_TILE_N * LOOP_TILE_K % THREAD_COUNT == 0);
  static_assert(BLOCK_TILE_N * LOOP_TILE_K / THREAD_COUNT % 8 == 0);
  constexpr int B_LDG_COUNT_PER_THREAD = BLOCK_TILE_N * LOOP_TILE_K / THREAD_COUNT;
  constexpr int B_LDG_LOOP_COUNT       = B_LDG_COUNT_PER_THREAD / 8;
  // clang-format off
  // This is the thread layout of the same warp that loads matrix B, where each thread reads K1xN8 elements of type T at a
  // loop iteration.
  // T0  T16
  // T1  T17
  // T2  T18
  // ... ...
  // T14 T30
  // T15 T31
  // clang-format on
  float B_ldg_reg[LDG_REG_BUFFER_SIZE][B_LDG_LOOP_COUNT][4];

  const int m_block_offset = blockIdx.y * BLOCK_TILE_M;
  const int n_block_offset = blockIdx.x * BLOCK_TILE_N;

  const int warp_id                 = threadIdx.x / 32;
  const int lane_id                 = threadIdx.x % 32;
  const int transposed_lane_id_mask = (lane_id / 8 == 0 || lane_id / 8 == 3) ? 0x00 : 0x18;
  const int transposed_lane_id      = lane_id ^ transposed_lane_id_mask;

  constexpr int M_MMA_WARP_COUNT       = BLOCK_TILE_M / WARP_TILE_M;
  constexpr int M_GROUP_COUNT_PER_WARP = WARP_TILE_M / 8;
  constexpr int N_GROUP_COUNT_PER_WARP = WARP_TILE_N / 16;

  constexpr int MMA_REG_BUFFER_SIZE    = 2;
  constexpr int MMA_REG_BUFFER_INDEX_0 = 0;
  constexpr int MMA_REG_BUFFER_INDEX_1 = 1;
  T             A_mma_reg[MMA_REG_BUFFER_SIZE][M_GROUP_COUNT_PER_WARP][4];
  T             B_mma_reg[MMA_REG_BUFFER_SIZE][N_GROUP_COUNT_PER_WARP][8];
  float         C_mma_reg[M_GROUP_COUNT_PER_WARP][N_GROUP_COUNT_PER_WARP][4] = {0};
  static_assert(N_GROUP_COUNT_PER_WARP % 2 == 0);
  union _2x4_or_1x8 {
    T _2x4[2][4];
    T _1x8[8];
  };
  _2x4_or_1x8 C_transposed[M_GROUP_COUNT_PER_WARP][N_GROUP_COUNT_PER_WARP / 2];

  const int m_warp_offset = warp_id % M_MMA_WARP_COUNT * WARP_TILE_M;
  const int n_warp_offset = warp_id / M_MMA_WARP_COUNT * WARP_TILE_N;

  const int A_ldg_reg_2_A_sm_partial_offset =
    lane_id / 16 * A_sm_dim2 * A_sm_dim3 + (warp_id * 16 + lane_id % 16) * A_sm_dim3;

  const int B_ldg_reg_2_B_sm_partial_offset =
    (lane_id % 16) / 8 * B_sm_dim2 * B_sm_dim3 + (lane_id % 16) % 8 * 8 + (warp_id * 2 + lane_id / 16) * B_sm_dim3;

  const int A_global_partial_offset = (m_block_offset + warp_id * 16 + lane_id % 16) * K + lane_id / 16 * 8;
  const int B_global_partial_offset = lane_id % 16 * N + n_block_offset + warp_id * 16 + lane_id / 16 * 8;

  const T* A_global_ptr_for_ldg[A_LDG_LOOP_COUNT];
  const T* B_global_ptr_for_ldg[B_LDG_LOOP_COUNT];
  for (int loop = 0; loop < A_LDG_LOOP_COUNT; ++loop) {
    A_global_ptr_for_ldg[loop] = &A[A_global_partial_offset + loop * WARP_COUNT * 16 * K];
  }
  for (int loop = 0; loop < B_LDG_LOOP_COUNT; ++loop) {
    B_global_ptr_for_ldg[loop] = &B[B_global_partial_offset + loop * WARP_COUNT * 16];
  }

  const T* A_sm_ptr_for_ldg[A_LDG_LOOP_COUNT];
  const T* B_sm_ptr_for_ldg[B_LDG_LOOP_COUNT];
  for (int loop = 0; loop < A_LDG_LOOP_COUNT; ++loop) {
    A_sm_ptr_for_ldg[loop] = &data.mma.A_sm[A_ldg_reg_2_A_sm_partial_offset + loop * WARP_COUNT * 16 * A_sm_dim3];
  }
  for (int loop = 0; loop < B_LDG_LOOP_COUNT; ++loop) {
    B_sm_ptr_for_ldg[loop] = &data.mma.B_sm[B_ldg_reg_2_B_sm_partial_offset + loop * WARP_COUNT * 2 * B_sm_dim3];
  }

  const int A_sm_2_A_mma_reg_partial_offset =
    lane_id % 16 / 8 * A_sm_dim2 * A_sm_dim3 + (m_warp_offset + lane_id % 8) * A_sm_dim3;

  const int B_sm_2_B_mma_reg_partial_offset = transposed_lane_id % 16 / 8 * B_sm_dim2 * B_sm_dim3
                                              + (n_warp_offset + transposed_lane_id / 16 * 8) / 8 * B_sm_dim3
                                              + transposed_lane_id % 8 * 8;

  const T* A_sm_ptr_for_mma[M_GROUP_COUNT_PER_WARP];
  const T* B_sm_ptr_for_mma[N_GROUP_COUNT_PER_WARP];

  for (int group = 0; group < M_GROUP_COUNT_PER_WARP; ++group) {
    A_sm_ptr_for_mma[group] = &data.mma.A_sm[A_sm_2_A_mma_reg_partial_offset + (group + lane_id / 16) * 8 * A_sm_dim3];
  }
  for (int group = 0; group < N_GROUP_COUNT_PER_WARP; ++group) {
    B_sm_ptr_for_mma[group] = &data.mma.B_sm[B_sm_2_B_mma_reg_partial_offset + (group * 2 * B_sm_dim3)];
  }

#define global_2_ldg_reg(k_loop_offset, ldg_reg_buffer_index)                                                          \
  {                                                                                                                    \
    _Pragma("unroll") for (int loop = 0; loop < A_LDG_LOOP_COUNT; ++loop)                                              \
    {                                                                                                                  \
      /* const int m = (loop * WARP_COUNT + warp_id) * 16 + lane_id % 16; */                                           \
      /* const int k = lane_id / 16 * 8; */                                                                            \
      FETCH_FLOAT4_WITH_PTR(&A_ldg_reg[ldg_reg_buffer_index][loop][0], A_global_ptr_for_ldg[loop] + k_loop_offset);    \
    }                                                                                                                  \
    _Pragma("unroll") for (int loop = 0; loop < B_LDG_LOOP_COUNT; ++loop)                                              \
    {                                                                                                                  \
      /* const int k = lane_id % 16;                                           */                                      \
      /* const int n = (loop * WARP_COUNT + warp_id) * 16 + lane_id / 16 * 8;  */                                      \
      FETCH_FLOAT4_WITH_PTR(&B_ldg_reg[ldg_reg_buffer_index][loop][0],                                                 \
                            B_global_ptr_for_ldg[loop] + (k_loop_offset) * N);                                         \
    }                                                                                                                  \
  }

#define ldg_reg_2_sm(ldg_sm_buffer_index, ldg_reg_buffer_index)                                                        \
  {                                                                                                                    \
    _Pragma("unroll") for (int loop = 0; loop < A_LDG_LOOP_COUNT; ++loop)                                              \
    {                                                                                                                  \
      /* const int m = (loop * WARP_COUNT + warp_id) * 16 + lane_id % 16; */                                           \
      /* const int k = lane_id / 16 * 8;  */                                                                           \
      STORE_FLOAT4_WITH_PTR(A_sm_ptr_for_ldg[loop] + (ldg_sm_buffer_index) * A_sm_dim1 * A_sm_dim2 * A_sm_dim3,        \
                            &A_ldg_reg[ldg_reg_buffer_index][loop][0]);                                                \
    }                                                                                                                  \
    _Pragma("unroll") for (int loop = 0; loop < B_LDG_LOOP_COUNT; ++loop)                                              \
    {                                                                                                                  \
      /*const int k = lane_id % 16; */                                                                                 \
      /*const int n = (loop * WARP_COUNT + warp_id) * 16 + lane_id / 16 * 8;*/                                         \
      STORE_FLOAT4_WITH_PTR(B_sm_ptr_for_ldg[loop] + (ldg_sm_buffer_index) * B_sm_dim1 * B_sm_dim2 * B_sm_dim3,        \
                            &B_ldg_reg[ldg_reg_buffer_index][loop][0]);                                                \
    }                                                                                                                  \
  }

#define sm_2_A_mma_reg(ldg_sm_buffer_index, mma_reg_buffer_index, group)                                               \
  if constexpr (M_GROUP_COUNT_PER_WARP == 1) {                                                                         \
    /* for (int group = 0; group < M_GROUP_COUNT_PER_WARP; ++group) */ {                                               \
      uint32_t src =                                                                                                   \
        __cvta_generic_to_shared(A_sm_ptr_for_mma[group] + (ldg_sm_buffer_index) * A_sm_dim1 * A_sm_dim2 * A_sm_dim3); \
      asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"                                          \
                   : "=r"(*(uint32_t*)&A_mma_reg[mma_reg_buffer_index][group][0]),                                     \
                     "=r"(*(uint32_t*)&A_mma_reg[mma_reg_buffer_index][group][2])                                      \
                   : "r"(src));                                                                                        \
    }                                                                                                                  \
  }                                                                                                                    \
  else if constexpr (M_GROUP_COUNT_PER_WARP % 2 == 0) {                                                                \
    /*for (int group = 0; group < M_GROUP_COUNT_PER_WARP; group += 2) */ {                                             \
      uint32_t src =                                                                                                   \
        __cvta_generic_to_shared(A_sm_ptr_for_mma[group] + (ldg_sm_buffer_index) * A_sm_dim1 * A_sm_dim2 * A_sm_dim3); \
      asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"                                  \
                   : "=r"(*(uint32_t*)&A_mma_reg[mma_reg_buffer_index][group][0]),                                     \
                     "=r"(*(uint32_t*)&A_mma_reg[mma_reg_buffer_index][group][2]),                                     \
                     "=r"(*(uint32_t*)&A_mma_reg[mma_reg_buffer_index][group + 1][0]),                                 \
                     "=r"(*(uint32_t*)&A_mma_reg[mma_reg_buffer_index][group + 1][2])                                  \
                   : "r"(src));                                                                                        \
    }                                                                                                                  \
  }                                                                                                                    \
  else {                                                                                                               \
    static_assert(M_GROUP_COUNT_PER_WARP == 1 || M_GROUP_COUNT_PER_WARP % 2 == 0);                                     \
  }

#define sm_2_B_mma_reg(ldg_sm_buffer_index, mma_reg_buffer_index, group)                                               \
  /* for (int group = 0; group < N_GROUP_COUNT_PER_WARP; ++group) */ {                                                 \
    uint32_t src =                                                                                                     \
      __cvta_generic_to_shared(B_sm_ptr_for_mma[group] + (ldg_sm_buffer_index) * B_sm_dim1 * B_sm_dim2 * B_sm_dim3);   \
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"                              \
                 : "=r"(*(uint32_t*)&B_mma_reg[mma_reg_buffer_index][group][0]),                                       \
                   "=r"(*(uint32_t*)&B_mma_reg[mma_reg_buffer_index][group][2]),                                       \
                   "=r"(*(uint32_t*)&B_mma_reg[mma_reg_buffer_index][group][4]),                                       \
                   "=r"(*(uint32_t*)&B_mma_reg[mma_reg_buffer_index][group][6])                                        \
                 : "r"(src));                                                                                          \
  }

  T* C_ptr = &C[(m_block_offset + m_warp_offset + lane_id / 4) * N + n_block_offset + n_warp_offset + lane_id % 4 * 8];

#define ldm_mma_sts_stg(ldm_sm_buffer_index,                                                                           \
                        ldm_reg_buffer_index,                                                                          \
                        mma_reg_buffer_index,                                                                          \
                        sts_sm_base_index,                                                                             \
                        rank,                                                                                          \
                        ldm_switch,                                                                                    \
                        mma_switch,                                                                                    \
                        sts_switch,                                                                                    \
                        stg_switch)                                                                                    \
  {                                                                                                                    \
    static_assert(M_GROUP_COUNT_PER_WARP * N_GROUP_COUNT_PER_WARP >= LDG_REG_BUFFER_SIZE);                             \
    static_assert(LDG_REG_BUFFER_SIZE == 2);                                                                           \
    if constexpr (sts_switch && rank + 2 == M_GROUP_COUNT_PER_WARP * N_GROUP_COUNT_PER_WARP) {                         \
      ldg_reg_2_sm(sts_sm_base_index, LDG_REG_BUFFER_INDEX_0);                                                         \
    }                                                                                                                  \
    if constexpr (sts_switch && rank + 1 == M_GROUP_COUNT_PER_WARP * N_GROUP_COUNT_PER_WARP) {                         \
      ldg_reg_2_sm(sts_sm_base_index + 1, LDG_REG_BUFFER_INDEX_1);                                                     \
    }                                                                                                                  \
    if constexpr (mma_switch && rank < M_GROUP_COUNT_PER_WARP * N_GROUP_COUNT_PER_WARP) {                              \
      constexpr int mg = rank % M_GROUP_COUNT_PER_WARP;                                                                \
      constexpr int ng = rank / M_GROUP_COUNT_PER_WARP;                                                                \
      mma_m16n8k16_row_col(C_mma_reg[mg][ng],                                                                          \
                           B_mma_reg[mma_reg_buffer_index][ng],                                                        \
                           A_mma_reg[mma_reg_buffer_index][mg],                                                        \
                           C_mma_reg[mg][ng]);                                                                         \
    }                                                                                                                  \
    static_assert(M_GROUP_COUNT_PER_WARP == 1 || M_GROUP_COUNT_PER_WARP % 2 == 0);                                     \
    if constexpr (ldm_switch && rank < (M_GROUP_COUNT_PER_WARP + 1) / 2) {                                             \
      sm_2_A_mma_reg(ldm_sm_buffer_index, ldm_reg_buffer_index, rank * 2);                                             \
    }                                                                                                                  \
    if constexpr (ldm_switch && (M_GROUP_COUNT_PER_WARP + 1) / 2 <= rank                                               \
                  && rank < (M_GROUP_COUNT_PER_WARP + 1) / 2 + N_GROUP_COUNT_PER_WARP) {                               \
      sm_2_B_mma_reg(ldm_sm_buffer_index, ldm_reg_buffer_index, rank - (M_GROUP_COUNT_PER_WARP + 1) / 2);              \
    }                                                                                                                  \
    if constexpr (stg_switch && rank < M_GROUP_COUNT_PER_WARP * N_GROUP_COUNT_PER_WARP) {                              \
      constexpr int mg = rank % M_GROUP_COUNT_PER_WARP;                                                                \
      constexpr int ng = rank / M_GROUP_COUNT_PER_WARP;                                                                \
      T casted[4]      = {C_mma_reg[mg][ng][0], C_mma_reg[mg][ng][1], C_mma_reg[mg][ng][2], C_mma_reg[mg][ng][3]};     \
      asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n"                                                   \
                   : "=r"(*(uint32_t*)&C_transposed[mg][ng / 2]._2x4[ng % 2][0])                                       \
                   : "r"(*(uint32_t*)&casted[0]));                                                                     \
      asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n"                                                   \
                   : "=r"(*(uint32_t*)&C_transposed[mg][ng / 2]._2x4[ng % 2][2])                                       \
                   : "r"(*(uint32_t*)&casted[2]));                                                                     \
      shfl_23_and_01(C_transposed[mg][ng / 2]._2x4[ng % 2], 0x1, lane_id);                                             \
      if constexpr ((ng + 1) % 2 == 0) {                                                                               \
        shfl_4567_and_0123(C_transposed[mg][ng / 2]._1x8, 0x2, lane_id);                                               \
        STORE_FLOAT4_WITH_PTR(C_ptr + mg * 8 * N + (ng - 1) * 16, &C_transposed[mg][ng / 2]._1x8[0]);                  \
      }                                                                                                                \
    }                                                                                                                  \
  }

// FIXME This code is really stupid. Please find a way to optimize it as soon as possible.
#define alternate_ldm_mma_sts_stg(ldm_sm_buf_index,                                                                    \
                                  ldm_reg_buf_index,                                                                   \
                                  mma_reg_buf_index,                                                                   \
                                  sts_sm_base_index,                                                                   \
                                  ldm_switch,                                                                          \
                                  mma_switch,                                                                          \
                                  sts_switch,                                                                          \
                                  stg_switch)                                                                          \
  static_assert(M_GROUP_COUNT_PER_WARP * N_GROUP_COUNT_PER_WARP <= 32);                                                \
  /* clang-format off */ \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index,  0, ldm_switch, mma_switch, sts_switch, stg_switch);         \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index,  1, ldm_switch, mma_switch, sts_switch, stg_switch);         \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index,  2, ldm_switch, mma_switch, sts_switch, stg_switch);         \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index,  3, ldm_switch, mma_switch, sts_switch, stg_switch);         \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index,  4, ldm_switch, mma_switch, sts_switch, stg_switch);         \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index,  5, ldm_switch, mma_switch, sts_switch, stg_switch);         \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index,  6, ldm_switch, mma_switch, sts_switch, stg_switch);         \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index,  7, ldm_switch, mma_switch, sts_switch, stg_switch);         \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index,  8, ldm_switch, mma_switch, sts_switch, stg_switch);         \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index,  9, ldm_switch, mma_switch, sts_switch, stg_switch);         \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, 10, ldm_switch, mma_switch, sts_switch, stg_switch);         \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, 11, ldm_switch, mma_switch, sts_switch, stg_switch);         \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, 12, ldm_switch, mma_switch, sts_switch, stg_switch);         \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, 13, ldm_switch, mma_switch, sts_switch, stg_switch);         \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, 14, ldm_switch, mma_switch, sts_switch, stg_switch);         \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, 15, ldm_switch, mma_switch, sts_switch, stg_switch);         \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, 16, ldm_switch, mma_switch, sts_switch, stg_switch);         \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, 17, ldm_switch, mma_switch, sts_switch, stg_switch);         \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, 18, ldm_switch, mma_switch, sts_switch, stg_switch);         \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, 19, ldm_switch, mma_switch, sts_switch, stg_switch);         \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, 20, ldm_switch, mma_switch, sts_switch, stg_switch);         \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, 21, ldm_switch, mma_switch, sts_switch, stg_switch);         \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, 22, ldm_switch, mma_switch, sts_switch, stg_switch);         \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, 23, ldm_switch, mma_switch, sts_switch, stg_switch);         \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, 24, ldm_switch, mma_switch, sts_switch, stg_switch);         \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, 25, ldm_switch, mma_switch, sts_switch, stg_switch);         \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, 26, ldm_switch, mma_switch, sts_switch, stg_switch);         \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, 27, ldm_switch, mma_switch, sts_switch, stg_switch);         \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, 28, ldm_switch, mma_switch, sts_switch, stg_switch);         \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, 29, ldm_switch, mma_switch, sts_switch, stg_switch);         \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, 30, ldm_switch, mma_switch, sts_switch, stg_switch);         \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, 31, ldm_switch, mma_switch, sts_switch, stg_switch);
  /* clang-format on */

  global_2_ldg_reg(0, LDG_REG_BUFFER_INDEX_0);
  global_2_ldg_reg(LOOP_TILE_K, LDG_REG_BUFFER_INDEX_1);
  ldg_reg_2_sm(0, LDG_REG_BUFFER_INDEX_0);
  ldg_reg_2_sm(1, LDG_REG_BUFFER_INDEX_1);
  __syncthreads();

  int LDG_SM_BUFFER_INDEX = 0;
  int k_loop_offset       = LOOP_TILE_K * 2;

  alternate_ldm_mma_sts_stg(LDG_SM_BUFFER_INDEX, MMA_REG_BUFFER_INDEX_0, 0, 0, true, false, false, false);

  global_2_ldg_reg(k_loop_offset, LDG_REG_BUFFER_INDEX_0);
  global_2_ldg_reg(k_loop_offset + LOOP_TILE_K, LDG_REG_BUFFER_INDEX_1);

  while (k_loop_offset + LOOP_TILE_K * 2 < K) {

    alternate_ldm_mma_sts_stg(LDG_SM_BUFFER_INDEX + 1,
                              MMA_REG_BUFFER_INDEX_1,
                              MMA_REG_BUFFER_INDEX_0,
                              (LDG_SM_BUFFER_INDEX ^ 2),
                              true,
                              true,
                              true,
                              false);

    LDG_SM_BUFFER_INDEX ^= 2;
    k_loop_offset += LOOP_TILE_K * 2;

    __syncthreads();

    global_2_ldg_reg(k_loop_offset, LDG_REG_BUFFER_INDEX_0);
    global_2_ldg_reg(k_loop_offset + LOOP_TILE_K, LDG_REG_BUFFER_INDEX_1);

    alternate_ldm_mma_sts_stg(
      LDG_SM_BUFFER_INDEX, MMA_REG_BUFFER_INDEX_0, MMA_REG_BUFFER_INDEX_1, 0, true, true, false, false);
  }

  {
    alternate_ldm_mma_sts_stg(LDG_SM_BUFFER_INDEX + 1,
                              MMA_REG_BUFFER_INDEX_1,
                              MMA_REG_BUFFER_INDEX_0,
                              (LDG_SM_BUFFER_INDEX ^ 2),
                              true,
                              true,
                              true,
                              false);
    LDG_SM_BUFFER_INDEX ^= 2;
    k_loop_offset += LOOP_TILE_K * 2;

    __syncthreads();

    alternate_ldm_mma_sts_stg(
      LDG_SM_BUFFER_INDEX, MMA_REG_BUFFER_INDEX_0, MMA_REG_BUFFER_INDEX_1, 0, true, true, false, false);
  }

  {
    alternate_ldm_mma_sts_stg(
      LDG_SM_BUFFER_INDEX + 1, MMA_REG_BUFFER_INDEX_1, MMA_REG_BUFFER_INDEX_0, 0, true, true, false, false);

    alternate_ldm_mma_sts_stg(
      LDG_SM_BUFFER_INDEX, MMA_REG_BUFFER_INDEX_0, MMA_REG_BUFFER_INDEX_1, 0, false, true, false, true);
  }

#undef global_2_ldg_reg
#undef ldg_reg_2_sm
#undef sm_2_A_mma_reg
#undef sm_2_B_mma_reg
#undef ldm_mma_sts_stg
#undef alternate_ldm_mma_sts_stg
}

template<typename T, int BLOCK_TILE_M, int BLOCK_TILE_N, int WARP_TILE_M, int WARP_TILE_N>
__global__ void
fp16_mma_m16n8k16_ldmatrix_trans__overlap_global_2_sm__quadra_buffer__reduce_instructions__reorder_instructions__overlap_reg_2_global__stg_memory_coalecesing__overlap_sts__stg_wt(
  const T* A, const T* B, T* C, int M, int N, int K)
{
  constexpr int WARP_COUNT   = BLOCK_TILE_M / WARP_TILE_M * BLOCK_TILE_N / WARP_TILE_N;
  constexpr int THREAD_COUNT = WARP_COUNT * 32;

  constexpr int LOOP_TILE_K            = 16;
  constexpr int LDG_SM_BUFFER_SIZE     = 4;
  constexpr int LDG_REG_BUFFER_SIZE    = 2;
  constexpr int LDG_REG_BUFFER_INDEX_0 = 0;
  constexpr int LDG_REG_BUFFER_INDEX_1 = 1;

  constexpr int A_sm_dim0 = LDG_SM_BUFFER_SIZE;
  constexpr int A_sm_dim1 = 2;
  constexpr int A_sm_dim2 = BLOCK_TILE_M;
  constexpr int A_sm_dim3 = LOOP_TILE_K / 2;
  constexpr int B_sm_dim0 = LDG_SM_BUFFER_SIZE;
  constexpr int B_sm_dim1 = LOOP_TILE_K / 8;
  constexpr int B_sm_dim2 = BLOCK_TILE_N / 8;
  constexpr int B_sm_dim3 = 64;

  // The 64 elements of type T in each 8x8 matrix are stored consecutively in a single layer of shared memory.
  __shared__ union {
    struct {
      T A_sm[A_sm_dim0 * A_sm_dim1 * A_sm_dim2 * A_sm_dim3];
      T B_sm[B_sm_dim0 * B_sm_dim1 * B_sm_dim2 * B_sm_dim3];
    } mma;
    static_assert(WARP_TILE_N % 16 == 0);
    T result[WARP_COUNT][WARP_TILE_N / 16][WARP_TILE_M][16];
  } data;

  static_assert(BLOCK_TILE_M * LOOP_TILE_K % THREAD_COUNT == 0);
  static_assert(BLOCK_TILE_M * LOOP_TILE_K / THREAD_COUNT % 8 == 0);
  constexpr int A_LDG_COUNT_PER_THREAD = BLOCK_TILE_M * LOOP_TILE_K / THREAD_COUNT;
  constexpr int A_LDG_LOOP_COUNT       = A_LDG_COUNT_PER_THREAD / 8;
  // clang-format off
  // This is the thread layout of the same warp that loads matrix A, where each thread reads M1xK8 elements of type T at a
  // loop iteration.
  // T0  T16
  // T1  T17
  // T2  T18
  // ... ...
  // T14 T30
  // T15 T31
  // clang-format on
  float A_ldg_reg[LDG_REG_BUFFER_SIZE][A_LDG_LOOP_COUNT][4];

  static_assert(BLOCK_TILE_N * LOOP_TILE_K % THREAD_COUNT == 0);
  static_assert(BLOCK_TILE_N * LOOP_TILE_K / THREAD_COUNT % 8 == 0);
  constexpr int B_LDG_COUNT_PER_THREAD = BLOCK_TILE_N * LOOP_TILE_K / THREAD_COUNT;
  constexpr int B_LDG_LOOP_COUNT       = B_LDG_COUNT_PER_THREAD / 8;
  // clang-format off
  // This is the thread layout of the same warp that loads matrix B, where each thread reads K1xN8 elements of type T at a
  // loop iteration.
  // T0  T16
  // T1  T17
  // T2  T18
  // ... ...
  // T14 T30
  // T15 T31
  // clang-format on
  float B_ldg_reg[LDG_REG_BUFFER_SIZE][B_LDG_LOOP_COUNT][4];

  const int m_block_offset = blockIdx.y * BLOCK_TILE_M;
  const int n_block_offset = blockIdx.x * BLOCK_TILE_N;

  const int warp_id                 = threadIdx.x / 32;
  const int lane_id                 = threadIdx.x % 32;
  const int transposed_lane_id_mask = (lane_id / 8 == 0 || lane_id / 8 == 3) ? 0x00 : 0x18;
  const int transposed_lane_id      = lane_id ^ transposed_lane_id_mask;

  constexpr int M_MMA_WARP_COUNT       = BLOCK_TILE_M / WARP_TILE_M;
  constexpr int M_GROUP_COUNT_PER_WARP = WARP_TILE_M / 8;
  constexpr int N_GROUP_COUNT_PER_WARP = WARP_TILE_N / 16;

  constexpr int MMA_REG_BUFFER_SIZE    = 2;
  constexpr int MMA_REG_BUFFER_INDEX_0 = 0;
  constexpr int MMA_REG_BUFFER_INDEX_1 = 1;
  T             A_mma_reg[MMA_REG_BUFFER_SIZE][M_GROUP_COUNT_PER_WARP][4];
  T             B_mma_reg[MMA_REG_BUFFER_SIZE][N_GROUP_COUNT_PER_WARP][8];
  float         C_mma_reg[M_GROUP_COUNT_PER_WARP][N_GROUP_COUNT_PER_WARP][4] = {0};
  static_assert(N_GROUP_COUNT_PER_WARP % 2 == 0);
  union _2x4_or_1x8 {
    T _2x4[2][4];
    T _1x8[8];
  };
  _2x4_or_1x8 C_transposed[M_GROUP_COUNT_PER_WARP][N_GROUP_COUNT_PER_WARP / 2];

  const int m_warp_offset = warp_id % M_MMA_WARP_COUNT * WARP_TILE_M;
  const int n_warp_offset = warp_id / M_MMA_WARP_COUNT * WARP_TILE_N;

  const int A_ldg_reg_2_A_sm_partial_offset =
    lane_id / 16 * A_sm_dim2 * A_sm_dim3 + (warp_id * 16 + lane_id % 16) * A_sm_dim3;

  const int B_ldg_reg_2_B_sm_partial_offset =
    (lane_id % 16) / 8 * B_sm_dim2 * B_sm_dim3 + (lane_id % 16) % 8 * 8 + (warp_id * 2 + lane_id / 16) * B_sm_dim3;

  const int A_global_partial_offset = (m_block_offset + warp_id * 16 + lane_id % 16) * K + lane_id / 16 * 8;
  const int B_global_partial_offset = lane_id % 16 * N + n_block_offset + warp_id * 16 + lane_id / 16 * 8;

  const T* A_global_ptr_for_ldg[A_LDG_LOOP_COUNT];
  const T* B_global_ptr_for_ldg[B_LDG_LOOP_COUNT];
  for (int loop = 0; loop < A_LDG_LOOP_COUNT; ++loop) {
    A_global_ptr_for_ldg[loop] = &A[A_global_partial_offset + loop * WARP_COUNT * 16 * K];
  }
  for (int loop = 0; loop < B_LDG_LOOP_COUNT; ++loop) {
    B_global_ptr_for_ldg[loop] = &B[B_global_partial_offset + loop * WARP_COUNT * 16];
  }

  const T* A_sm_ptr_for_ldg[A_LDG_LOOP_COUNT];
  const T* B_sm_ptr_for_ldg[B_LDG_LOOP_COUNT];
  for (int loop = 0; loop < A_LDG_LOOP_COUNT; ++loop) {
    A_sm_ptr_for_ldg[loop] = &data.mma.A_sm[A_ldg_reg_2_A_sm_partial_offset + loop * WARP_COUNT * 16 * A_sm_dim3];
  }
  for (int loop = 0; loop < B_LDG_LOOP_COUNT; ++loop) {
    B_sm_ptr_for_ldg[loop] = &data.mma.B_sm[B_ldg_reg_2_B_sm_partial_offset + loop * WARP_COUNT * 2 * B_sm_dim3];
  }

  const int A_sm_2_A_mma_reg_partial_offset =
    lane_id % 16 / 8 * A_sm_dim2 * A_sm_dim3 + (m_warp_offset + lane_id % 8) * A_sm_dim3;

  const int B_sm_2_B_mma_reg_partial_offset = transposed_lane_id % 16 / 8 * B_sm_dim2 * B_sm_dim3
                                              + (n_warp_offset + transposed_lane_id / 16 * 8) / 8 * B_sm_dim3
                                              + transposed_lane_id % 8 * 8;

  const T* A_sm_ptr_for_mma[M_GROUP_COUNT_PER_WARP];
  const T* B_sm_ptr_for_mma[N_GROUP_COUNT_PER_WARP];

  for (int group = 0; group < M_GROUP_COUNT_PER_WARP; ++group) {
    A_sm_ptr_for_mma[group] = &data.mma.A_sm[A_sm_2_A_mma_reg_partial_offset + (group + lane_id / 16) * 8 * A_sm_dim3];
  }
  for (int group = 0; group < N_GROUP_COUNT_PER_WARP; ++group) {
    B_sm_ptr_for_mma[group] = &data.mma.B_sm[B_sm_2_B_mma_reg_partial_offset + (group * 2 * B_sm_dim3)];
  }

#define global_2_ldg_reg(k_loop_offset, ldg_reg_buffer_index)                                                          \
  {                                                                                                                    \
    _Pragma("unroll") for (int loop = 0; loop < A_LDG_LOOP_COUNT; ++loop)                                              \
    {                                                                                                                  \
      /* const int m = (loop * WARP_COUNT + warp_id) * 16 + lane_id % 16; */                                           \
      /* const int k = lane_id / 16 * 8; */                                                                            \
      FETCH_FLOAT4_WITH_PTR(&A_ldg_reg[ldg_reg_buffer_index][loop][0], A_global_ptr_for_ldg[loop] + k_loop_offset);    \
    }                                                                                                                  \
    _Pragma("unroll") for (int loop = 0; loop < B_LDG_LOOP_COUNT; ++loop)                                              \
    {                                                                                                                  \
      /* const int k = lane_id % 16;                                           */                                      \
      /* const int n = (loop * WARP_COUNT + warp_id) * 16 + lane_id / 16 * 8;  */                                      \
      FETCH_FLOAT4_WITH_PTR(&B_ldg_reg[ldg_reg_buffer_index][loop][0],                                                 \
                            B_global_ptr_for_ldg[loop] + (k_loop_offset) * N);                                         \
    }                                                                                                                  \
  }

#define ldg_reg_2_sm(ldg_sm_buffer_index, ldg_reg_buffer_index)                                                        \
  {                                                                                                                    \
    _Pragma("unroll") for (int loop = 0; loop < A_LDG_LOOP_COUNT; ++loop)                                              \
    {                                                                                                                  \
      /* const int m = (loop * WARP_COUNT + warp_id) * 16 + lane_id % 16; */                                           \
      /* const int k = lane_id / 16 * 8;  */                                                                           \
      STORE_FLOAT4_WITH_PTR(A_sm_ptr_for_ldg[loop] + (ldg_sm_buffer_index) * A_sm_dim1 * A_sm_dim2 * A_sm_dim3,        \
                            &A_ldg_reg[ldg_reg_buffer_index][loop][0]);                                                \
    }                                                                                                                  \
    _Pragma("unroll") for (int loop = 0; loop < B_LDG_LOOP_COUNT; ++loop)                                              \
    {                                                                                                                  \
      /*const int k = lane_id % 16; */                                                                                 \
      /*const int n = (loop * WARP_COUNT + warp_id) * 16 + lane_id / 16 * 8;*/                                         \
      STORE_FLOAT4_WITH_PTR(B_sm_ptr_for_ldg[loop] + (ldg_sm_buffer_index) * B_sm_dim1 * B_sm_dim2 * B_sm_dim3,        \
                            &B_ldg_reg[ldg_reg_buffer_index][loop][0]);                                                \
    }                                                                                                                  \
  }

#define sm_2_A_mma_reg(ldg_sm_buffer_index, mma_reg_buffer_index, group)                                               \
  if constexpr (M_GROUP_COUNT_PER_WARP == 1) {                                                                         \
    /* for (int group = 0; group < M_GROUP_COUNT_PER_WARP; ++group) */ {                                               \
      uint32_t src =                                                                                                   \
        __cvta_generic_to_shared(A_sm_ptr_for_mma[group] + (ldg_sm_buffer_index) * A_sm_dim1 * A_sm_dim2 * A_sm_dim3); \
      asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"                                          \
                   : "=r"(*(uint32_t*)&A_mma_reg[mma_reg_buffer_index][group][0]),                                     \
                     "=r"(*(uint32_t*)&A_mma_reg[mma_reg_buffer_index][group][2])                                      \
                   : "r"(src));                                                                                        \
    }                                                                                                                  \
  }                                                                                                                    \
  else if constexpr (M_GROUP_COUNT_PER_WARP % 2 == 0) {                                                                \
    /*for (int group = 0; group < M_GROUP_COUNT_PER_WARP; group += 2) */ {                                             \
      uint32_t src =                                                                                                   \
        __cvta_generic_to_shared(A_sm_ptr_for_mma[group] + (ldg_sm_buffer_index) * A_sm_dim1 * A_sm_dim2 * A_sm_dim3); \
      asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"                                  \
                   : "=r"(*(uint32_t*)&A_mma_reg[mma_reg_buffer_index][group][0]),                                     \
                     "=r"(*(uint32_t*)&A_mma_reg[mma_reg_buffer_index][group][2]),                                     \
                     "=r"(*(uint32_t*)&A_mma_reg[mma_reg_buffer_index][group + 1][0]),                                 \
                     "=r"(*(uint32_t*)&A_mma_reg[mma_reg_buffer_index][group + 1][2])                                  \
                   : "r"(src));                                                                                        \
    }                                                                                                                  \
  }                                                                                                                    \
  else {                                                                                                               \
    static_assert(M_GROUP_COUNT_PER_WARP == 1 || M_GROUP_COUNT_PER_WARP % 2 == 0);                                     \
  }

#define sm_2_B_mma_reg(ldg_sm_buffer_index, mma_reg_buffer_index, group)                                               \
  /* for (int group = 0; group < N_GROUP_COUNT_PER_WARP; ++group) */ {                                                 \
    uint32_t src =                                                                                                     \
      __cvta_generic_to_shared(B_sm_ptr_for_mma[group] + (ldg_sm_buffer_index) * B_sm_dim1 * B_sm_dim2 * B_sm_dim3);   \
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"                              \
                 : "=r"(*(uint32_t*)&B_mma_reg[mma_reg_buffer_index][group][0]),                                       \
                   "=r"(*(uint32_t*)&B_mma_reg[mma_reg_buffer_index][group][2]),                                       \
                   "=r"(*(uint32_t*)&B_mma_reg[mma_reg_buffer_index][group][4]),                                       \
                   "=r"(*(uint32_t*)&B_mma_reg[mma_reg_buffer_index][group][6])                                        \
                 : "r"(src));                                                                                          \
  }

  T* C_ptr = &C[(m_block_offset + m_warp_offset + lane_id / 4) * N + n_block_offset + n_warp_offset + lane_id % 4 * 8];

#define ldm_mma_sts_stg(ldm_sm_buffer_index,                                                                           \
                        ldm_reg_buffer_index,                                                                          \
                        mma_reg_buffer_index,                                                                          \
                        sts_sm_base_index,                                                                             \
                        rank,                                                                                          \
                        ldm_switch,                                                                                    \
                        mma_switch,                                                                                    \
                        sts_switch,                                                                                    \
                        stg_switch)                                                                                    \
  {                                                                                                                    \
    static_assert(M_GROUP_COUNT_PER_WARP * N_GROUP_COUNT_PER_WARP >= LDG_REG_BUFFER_SIZE);                             \
    static_assert(LDG_REG_BUFFER_SIZE == 2);                                                                           \
    if constexpr (sts_switch && rank + 2 == M_GROUP_COUNT_PER_WARP * N_GROUP_COUNT_PER_WARP) {                         \
      ldg_reg_2_sm(sts_sm_base_index, LDG_REG_BUFFER_INDEX_0);                                                         \
    }                                                                                                                  \
    if constexpr (sts_switch && rank + 1 == M_GROUP_COUNT_PER_WARP * N_GROUP_COUNT_PER_WARP) {                         \
      ldg_reg_2_sm(sts_sm_base_index + 1, LDG_REG_BUFFER_INDEX_1);                                                     \
    }                                                                                                                  \
    if constexpr (mma_switch && rank < M_GROUP_COUNT_PER_WARP * N_GROUP_COUNT_PER_WARP) {                              \
      constexpr int mg = rank % M_GROUP_COUNT_PER_WARP;                                                                \
      constexpr int ng = rank / M_GROUP_COUNT_PER_WARP;                                                                \
      mma_m16n8k16_row_col(C_mma_reg[mg][ng],                                                                          \
                           B_mma_reg[mma_reg_buffer_index][ng],                                                        \
                           A_mma_reg[mma_reg_buffer_index][mg],                                                        \
                           C_mma_reg[mg][ng]);                                                                         \
    }                                                                                                                  \
    static_assert(M_GROUP_COUNT_PER_WARP == 1 || M_GROUP_COUNT_PER_WARP % 2 == 0);                                     \
    if constexpr (ldm_switch && rank < (M_GROUP_COUNT_PER_WARP + 1) / 2) {                                             \
      sm_2_A_mma_reg(ldm_sm_buffer_index, ldm_reg_buffer_index, rank * 2);                                             \
    }                                                                                                                  \
    if constexpr (ldm_switch && (M_GROUP_COUNT_PER_WARP + 1) / 2 <= rank                                               \
                  && rank < (M_GROUP_COUNT_PER_WARP + 1) / 2 + N_GROUP_COUNT_PER_WARP) {                               \
      sm_2_B_mma_reg(ldm_sm_buffer_index, ldm_reg_buffer_index, rank - (M_GROUP_COUNT_PER_WARP + 1) / 2);              \
    }                                                                                                                  \
    if constexpr (stg_switch && rank < M_GROUP_COUNT_PER_WARP * N_GROUP_COUNT_PER_WARP) {                              \
      constexpr int mg = rank % M_GROUP_COUNT_PER_WARP;                                                                \
      constexpr int ng = rank / M_GROUP_COUNT_PER_WARP;                                                                \
      T casted[4]      = {C_mma_reg[mg][ng][0], C_mma_reg[mg][ng][1], C_mma_reg[mg][ng][2], C_mma_reg[mg][ng][3]};     \
      asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n"                                                   \
                   : "=r"(*(uint32_t*)&C_transposed[mg][ng / 2]._2x4[ng % 2][0])                                       \
                   : "r"(*(uint32_t*)&casted[0]));                                                                     \
      asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n"                                                   \
                   : "=r"(*(uint32_t*)&C_transposed[mg][ng / 2]._2x4[ng % 2][2])                                       \
                   : "r"(*(uint32_t*)&casted[2]));                                                                     \
      shfl_23_and_01(C_transposed[mg][ng / 2]._2x4[ng % 2], 0x1, lane_id);                                             \
      if constexpr ((ng + 1) % 2 == 0) {                                                                               \
        shfl_4567_and_0123(C_transposed[mg][ng / 2]._1x8, 0x2, lane_id);                                               \
        asm volatile("st.global.wt.v4.f32 [%0], {%1, %2, %3, %4};"                                                     \
                     :                                                                                                 \
                     : "l"(C_ptr + mg * 8 * N + (ng - 1) * 16),                                                        \
                       "f"(*(const float*)&C_transposed[mg][ng / 2]._1x8[0]),                                          \
                       "f"(*(const float*)&C_transposed[mg][ng / 2]._1x8[2]),                                          \
                       "f"(*(const float*)&C_transposed[mg][ng / 2]._1x8[4]),                                          \
                       "f"(*(const float*)&C_transposed[mg][ng / 2]._1x8[6])                                           \
                     : "memory");                                                                                      \
      }                                                                                                                \
    }                                                                                                                  \
  }

// FIXME This code is really stupid. Please find a way to optimize it as soon as possible.
#define alternate_ldm_mma_sts_stg(ldm_sm_buf_index,                                                                    \
                                  ldm_reg_buf_index,                                                                   \
                                  mma_reg_buf_index,                                                                   \
                                  sts_sm_base_index,                                                                   \
                                  ldm_switch,                                                                          \
                                  mma_switch,                                                                          \
                                  sts_switch,                                                                          \
                                  stg_switch)                                                                          \
  static_assert(M_GROUP_COUNT_PER_WARP * N_GROUP_COUNT_PER_WARP <= 32);                                                \
  /* clang-format off */ \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index,  0, ldm_switch, mma_switch, sts_switch, stg_switch);         \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index,  1, ldm_switch, mma_switch, sts_switch, stg_switch);         \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index,  2, ldm_switch, mma_switch, sts_switch, stg_switch);         \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index,  3, ldm_switch, mma_switch, sts_switch, stg_switch);         \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index,  4, ldm_switch, mma_switch, sts_switch, stg_switch);         \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index,  5, ldm_switch, mma_switch, sts_switch, stg_switch);         \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index,  6, ldm_switch, mma_switch, sts_switch, stg_switch);         \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index,  7, ldm_switch, mma_switch, sts_switch, stg_switch);         \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index,  8, ldm_switch, mma_switch, sts_switch, stg_switch);         \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index,  9, ldm_switch, mma_switch, sts_switch, stg_switch);         \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, 10, ldm_switch, mma_switch, sts_switch, stg_switch);         \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, 11, ldm_switch, mma_switch, sts_switch, stg_switch);         \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, 12, ldm_switch, mma_switch, sts_switch, stg_switch);         \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, 13, ldm_switch, mma_switch, sts_switch, stg_switch);         \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, 14, ldm_switch, mma_switch, sts_switch, stg_switch);         \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, 15, ldm_switch, mma_switch, sts_switch, stg_switch);         \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, 16, ldm_switch, mma_switch, sts_switch, stg_switch);         \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, 17, ldm_switch, mma_switch, sts_switch, stg_switch);         \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, 18, ldm_switch, mma_switch, sts_switch, stg_switch);         \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, 19, ldm_switch, mma_switch, sts_switch, stg_switch);         \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, 20, ldm_switch, mma_switch, sts_switch, stg_switch);         \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, 21, ldm_switch, mma_switch, sts_switch, stg_switch);         \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, 22, ldm_switch, mma_switch, sts_switch, stg_switch);         \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, 23, ldm_switch, mma_switch, sts_switch, stg_switch);         \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, 24, ldm_switch, mma_switch, sts_switch, stg_switch);         \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, 25, ldm_switch, mma_switch, sts_switch, stg_switch);         \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, 26, ldm_switch, mma_switch, sts_switch, stg_switch);         \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, 27, ldm_switch, mma_switch, sts_switch, stg_switch);         \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, 28, ldm_switch, mma_switch, sts_switch, stg_switch);         \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, 29, ldm_switch, mma_switch, sts_switch, stg_switch);         \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, 30, ldm_switch, mma_switch, sts_switch, stg_switch);         \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, 31, ldm_switch, mma_switch, sts_switch, stg_switch);
  /* clang-format on */

  global_2_ldg_reg(0, LDG_REG_BUFFER_INDEX_0);
  global_2_ldg_reg(LOOP_TILE_K, LDG_REG_BUFFER_INDEX_1);
  ldg_reg_2_sm(0, LDG_REG_BUFFER_INDEX_0);
  ldg_reg_2_sm(1, LDG_REG_BUFFER_INDEX_1);
  __syncthreads();

  int LDG_SM_BUFFER_INDEX = 0;
  int k_loop_offset       = LOOP_TILE_K * 2;

  alternate_ldm_mma_sts_stg(LDG_SM_BUFFER_INDEX, MMA_REG_BUFFER_INDEX_0, 0, 0, true, false, false, false);

  global_2_ldg_reg(k_loop_offset, LDG_REG_BUFFER_INDEX_0);
  global_2_ldg_reg(k_loop_offset + LOOP_TILE_K, LDG_REG_BUFFER_INDEX_1);

  while (k_loop_offset + LOOP_TILE_K * 2 < K) {

    alternate_ldm_mma_sts_stg(LDG_SM_BUFFER_INDEX + 1,
                              MMA_REG_BUFFER_INDEX_1,
                              MMA_REG_BUFFER_INDEX_0,
                              (LDG_SM_BUFFER_INDEX ^ 2),
                              true,
                              true,
                              true,
                              false);

    LDG_SM_BUFFER_INDEX ^= 2;
    k_loop_offset += LOOP_TILE_K * 2;

    __syncthreads();

    global_2_ldg_reg(k_loop_offset, LDG_REG_BUFFER_INDEX_0);
    global_2_ldg_reg(k_loop_offset + LOOP_TILE_K, LDG_REG_BUFFER_INDEX_1);

    alternate_ldm_mma_sts_stg(
      LDG_SM_BUFFER_INDEX, MMA_REG_BUFFER_INDEX_0, MMA_REG_BUFFER_INDEX_1, 0, true, true, false, false);
  }

  {
    alternate_ldm_mma_sts_stg(LDG_SM_BUFFER_INDEX + 1,
                              MMA_REG_BUFFER_INDEX_1,
                              MMA_REG_BUFFER_INDEX_0,
                              (LDG_SM_BUFFER_INDEX ^ 2),
                              true,
                              true,
                              true,
                              false);
    LDG_SM_BUFFER_INDEX ^= 2;
    k_loop_offset += LOOP_TILE_K * 2;

    __syncthreads();

    alternate_ldm_mma_sts_stg(
      LDG_SM_BUFFER_INDEX, MMA_REG_BUFFER_INDEX_0, MMA_REG_BUFFER_INDEX_1, 0, true, true, false, false);
  }

  {
    alternate_ldm_mma_sts_stg(
      LDG_SM_BUFFER_INDEX + 1, MMA_REG_BUFFER_INDEX_1, MMA_REG_BUFFER_INDEX_0, 0, true, true, false, false);

    alternate_ldm_mma_sts_stg(
      LDG_SM_BUFFER_INDEX, MMA_REG_BUFFER_INDEX_0, MMA_REG_BUFFER_INDEX_1, 0, false, true, false, true);
  }

#undef global_2_ldg_reg
#undef ldg_reg_2_sm
#undef sm_2_A_mma_reg
#undef sm_2_B_mma_reg
#undef ldm_mma_sts_stg
#undef alternate_ldm_mma_sts_stg
}

template<typename T, int BLOCK_TILE_M, int BLOCK_TILE_N, int WARP_TILE_M, int WARP_TILE_N>
__global__ void
fp16_mma_m16n8k16_ldmatrix_trans__overlap_global_2_sm__quadra_buffer__reduce_instructions__reorder_instructions__overlap_reg_2_global__stg_memory_coalecesing__overlap_sts__stg_wt__reduce_IMAD_AND_LEA(
  const T* A, const T* B, T* C, int M, int N, int K)
{
  // if(threadIdx.x == 0) {
  //   printf("%03d %03d %03d\n", blockIdx.x, blockIdx.y, get_smid());
  // }
  constexpr int WARP_COUNT   = BLOCK_TILE_M / WARP_TILE_M * BLOCK_TILE_N / WARP_TILE_N;
  constexpr int THREAD_COUNT = WARP_COUNT * 32;

  constexpr int LOOP_TILE_K         = 16;
  constexpr int LDG_SM_BUFFER_SIZE  = 4;
  constexpr int LDG_REG_BUFFER_SIZE = 2;

  constexpr int A_sm_dim0 = LDG_SM_BUFFER_SIZE;
  constexpr int A_sm_dim1 = 2;
  constexpr int A_sm_dim2 = BLOCK_TILE_M;
  constexpr int A_sm_dim3 = LOOP_TILE_K / 2;
  constexpr int B_sm_dim0 = LDG_SM_BUFFER_SIZE;
  constexpr int B_sm_dim1 = LOOP_TILE_K / 8;
  constexpr int B_sm_dim2 = BLOCK_TILE_N / 8;
  constexpr int B_sm_dim3 = 64;

  // The 64 elements of type T in each 8x8 matrix are stored consecutively in a single layer of shared memory.
  __shared__ union {
    struct {
      T A_sm[A_sm_dim0 * A_sm_dim1 * A_sm_dim2 * A_sm_dim3];
      T B_sm[B_sm_dim0 * B_sm_dim1 * B_sm_dim2 * B_sm_dim3];
    } mma;
    static_assert(WARP_TILE_N % 16 == 0);
    T result[WARP_COUNT][WARP_TILE_N / 16][WARP_TILE_M][16];
  } data;

  static_assert(BLOCK_TILE_M * LOOP_TILE_K % THREAD_COUNT == 0);
  static_assert(BLOCK_TILE_M * LOOP_TILE_K / THREAD_COUNT % 8 == 0);
  constexpr int A_LDG_COUNT_PER_THREAD = BLOCK_TILE_M * LOOP_TILE_K / THREAD_COUNT;
  constexpr int A_LDG_LOOP_COUNT       = A_LDG_COUNT_PER_THREAD / 8;
  // clang-format off
  // This is the thread layout of the same warp that loads matrix A, where each thread reads M1xK8 elements of type T at a
  // loop iteration.
  // T0  T16
  // T1  T17
  // T2  T18
  // ... ...
  // T14 T30
  // T15 T31
  // clang-format on
  float A_ldg_reg[LDG_REG_BUFFER_SIZE][A_LDG_LOOP_COUNT][4];

  static_assert(BLOCK_TILE_N * LOOP_TILE_K % THREAD_COUNT == 0);
  static_assert(BLOCK_TILE_N * LOOP_TILE_K / THREAD_COUNT % 8 == 0);
  constexpr int B_LDG_COUNT_PER_THREAD = BLOCK_TILE_N * LOOP_TILE_K / THREAD_COUNT;
  constexpr int B_LDG_LOOP_COUNT       = B_LDG_COUNT_PER_THREAD / 8;
  // clang-format off
  // This is the thread layout of the same warp that loads matrix B, where each thread reads K1xN8 elements of type T at a
  // loop iteration.
  // T0  T16
  // T1  T17
  // T2  T18
  // ... ...
  // T14 T30
  // T15 T31
  // clang-format on
  float B_ldg_reg[LDG_REG_BUFFER_SIZE][B_LDG_LOOP_COUNT][4];

  const int m_block_offset = blockIdx.y * BLOCK_TILE_M;
  const int n_block_offset = blockIdx.x * BLOCK_TILE_N;

  const int warp_id                 = threadIdx.x / 32;
  const int lane_id                 = threadIdx.x % 32;
  const int transposed_lane_id_mask = (lane_id / 8 == 0 || lane_id / 8 == 3) ? 0x00 : 0x18;
  const int transposed_lane_id      = lane_id ^ transposed_lane_id_mask;

  constexpr int M_MMA_WARP_COUNT       = BLOCK_TILE_M / WARP_TILE_M;
  constexpr int M_GROUP_COUNT_PER_WARP = WARP_TILE_M / 8;
  constexpr int N_GROUP_COUNT_PER_WARP = WARP_TILE_N / 16;

  constexpr int MMA_REG_BUFFER_SIZE    = 2;
  constexpr int MMA_REG_BUFFER_INDEX_0 = 0;
  constexpr int MMA_REG_BUFFER_INDEX_1 = 1;
  T             A_mma_reg[MMA_REG_BUFFER_SIZE][M_GROUP_COUNT_PER_WARP][4];
  T             B_mma_reg[MMA_REG_BUFFER_SIZE][N_GROUP_COUNT_PER_WARP][8];
  float         C_mma_reg[M_GROUP_COUNT_PER_WARP][N_GROUP_COUNT_PER_WARP][4] = {0};
  static_assert(N_GROUP_COUNT_PER_WARP % 2 == 0);
  union _2x4_or_1x8 {
    T _2x4[2][4];
    T _1x8[8];
  };
  _2x4_or_1x8 C_transposed[M_GROUP_COUNT_PER_WARP][N_GROUP_COUNT_PER_WARP / 2];

  const int m_warp_offset = warp_id % M_MMA_WARP_COUNT * WARP_TILE_M;
  const int n_warp_offset = warp_id / M_MMA_WARP_COUNT * WARP_TILE_N;

  const int A_ldg_reg_2_A_sm_partial_offset =
    lane_id / 16 * A_sm_dim2 * A_sm_dim3 + (warp_id * 16 + lane_id % 16) * A_sm_dim3;

  const int B_ldg_reg_2_B_sm_partial_offset =
    (lane_id % 16) / 8 * B_sm_dim2 * B_sm_dim3 + (lane_id % 16) % 8 * 8 + (warp_id * 2 + lane_id / 16) * B_sm_dim3;

  const int A_global_partial_offset = (m_block_offset + warp_id * 16 + lane_id % 16) * K + lane_id / 16 * 8;
  const int B_global_partial_offset = lane_id % 16 * N + n_block_offset + warp_id * 16 + lane_id / 16 * 8;

  const T* A_global_ptr_for_ldg = &A[A_global_partial_offset];
  const T* B_global_ptr_for_ldg = &B[B_global_partial_offset];

  static_assert(A_LDG_LOOP_COUNT <= 4);
  static_assert(B_LDG_LOOP_COUNT <= 4);
  static_assert(LDG_REG_BUFFER_SIZE <= 2);
  const uint64_t A_global_ptr_for_ldg__loop_0__k_0 = (uint64_t)(A_global_ptr_for_ldg + 0 * WARP_COUNT * 16 * K + 0);
  const uint64_t A_global_ptr_for_ldg__loop_1__k_0 = (uint64_t)(A_global_ptr_for_ldg + 1 * WARP_COUNT * 16 * K + 0);
  const uint64_t A_global_ptr_for_ldg__loop_2__k_0 = (uint64_t)(A_global_ptr_for_ldg + 2 * WARP_COUNT * 16 * K + 0);
  const uint64_t A_global_ptr_for_ldg__loop_3__k_0 = (uint64_t)(A_global_ptr_for_ldg + 3 * WARP_COUNT * 16 * K + 0);
  const uint64_t A_global_ptr_for_ldg__loop_0__k_1 =
    (uint64_t)(A_global_ptr_for_ldg + 0 * WARP_COUNT * 16 * K + LOOP_TILE_K);
  const uint64_t A_global_ptr_for_ldg__loop_1__k_1 =
    (uint64_t)(A_global_ptr_for_ldg + 1 * WARP_COUNT * 16 * K + LOOP_TILE_K);
  const uint64_t A_global_ptr_for_ldg__loop_2__k_1 =
    (uint64_t)(A_global_ptr_for_ldg + 2 * WARP_COUNT * 16 * K + LOOP_TILE_K);
  const uint64_t A_global_ptr_for_ldg__loop_3__k_1 =
    (uint64_t)(A_global_ptr_for_ldg + 3 * WARP_COUNT * 16 * K + LOOP_TILE_K);
  const uint64_t B_global_ptr_for_ldg__loop_0__k_0 = (uint64_t)(B_global_ptr_for_ldg + 0 * WARP_COUNT * 16 + 0);
  const uint64_t B_global_ptr_for_ldg__loop_1__k_0 = (uint64_t)(B_global_ptr_for_ldg + 1 * WARP_COUNT * 16 + 0);
  const uint64_t B_global_ptr_for_ldg__loop_2__k_0 = (uint64_t)(B_global_ptr_for_ldg + 2 * WARP_COUNT * 16 + 0);
  const uint64_t B_global_ptr_for_ldg__loop_3__k_0 = (uint64_t)(B_global_ptr_for_ldg + 3 * WARP_COUNT * 16 + 0);
  const uint64_t B_global_ptr_for_ldg__loop_0__k_1 =
    (uint64_t)(B_global_ptr_for_ldg + 0 * WARP_COUNT * 16 + LOOP_TILE_K * N);
  const uint64_t B_global_ptr_for_ldg__loop_1__k_1 =
    (uint64_t)(B_global_ptr_for_ldg + 1 * WARP_COUNT * 16 + LOOP_TILE_K * N);
  const uint64_t B_global_ptr_for_ldg__loop_2__k_1 =
    (uint64_t)(B_global_ptr_for_ldg + 2 * WARP_COUNT * 16 + LOOP_TILE_K * N);
  const uint64_t B_global_ptr_for_ldg__loop_3__k_1 =
    (uint64_t)(B_global_ptr_for_ldg + 3 * WARP_COUNT * 16 + LOOP_TILE_K * N);

  const T* A_sm_ptr_for_ldg = &data.mma.A_sm[A_ldg_reg_2_A_sm_partial_offset];
  const T* B_sm_ptr_for_ldg = &data.mma.B_sm[B_ldg_reg_2_B_sm_partial_offset];

  const int A_sm_2_A_mma_reg_partial_offset =
    lane_id % 16 / 8 * A_sm_dim2 * A_sm_dim3 + (m_warp_offset + lane_id % 8) * A_sm_dim3;

  const int B_sm_2_B_mma_reg_partial_offset = transposed_lane_id % 16 / 8 * B_sm_dim2 * B_sm_dim3
                                              + (n_warp_offset + transposed_lane_id / 16 * 8) / 8 * B_sm_dim3
                                              + transposed_lane_id % 8 * 8;

  const T* A_sm_ptr_for_mma = &data.mma.A_sm[A_sm_2_A_mma_reg_partial_offset + lane_id / 16 * 8 * A_sm_dim3];
  const T* B_sm_ptr_for_mma = &data.mma.B_sm[B_sm_2_B_mma_reg_partial_offset];

  enum {
    LDG_SWITCH_OFF             = 0,
    LDG_SWITCH_ON_EVICT_NORMAL = 1,
    LDG_SWITCH_ON_EVICT_LAST   = 2,
  };

#define A_global_2_ldg_reg(A_global_ptr, ldg_reg_buffer_index, loop, cache_policy)                                     \
  {                                                                                                                    \
    /* const int m = (loop * WARP_COUNT + warp_id) * 16 + lane_id % 16; */                                             \
    /* const int k = lane_id / 16 * 8; */                                                                              \
    if constexpr (cache_policy == LDG_SWITCH_ON_EVICT_LAST) {                                                          \
      FETCH_FLOAT4_EVICT_LAST_AND_PREFETCH_256B_WITH_SRC_PTR(A_ldg_reg[ldg_reg_buffer_index][loop], A_global_ptr);     \
    }                                                                                                                  \
    else {                                                                                                             \
      FETCH_FLOAT4_PREFETCH_256B_WITH_SRC_PTR(A_ldg_reg[ldg_reg_buffer_index][loop], A_global_ptr);                    \
    }                                                                                                                  \
  }

#define B_global_2_ldg_reg(B_global_ptr, ldg_reg_buffer_index, loop, cache_policy)                                     \
  {                                                                                                                    \
    /* const int k = lane_id % 16;                                           */                                        \
    /* const int n = (loop * WARP_COUNT + warp_id) * 16 + lane_id / 16 * 8;  */                                        \
    if constexpr (cache_policy == LDG_SWITCH_ON_EVICT_LAST) {                                                          \
      FETCH_FLOAT4_EVICT_LAST_AND_PREFETCH_256B_WITH_SRC_PTR(B_ldg_reg[ldg_reg_buffer_index][loop], B_global_ptr);     \
    }                                                                                                                  \
    else {                                                                                                             \
      FETCH_FLOAT4_PREFETCH_256B_WITH_SRC_PTR(B_ldg_reg[ldg_reg_buffer_index][loop], B_global_ptr);                    \
    }                                                                                                                  \
  }

#define A_ldg_reg_2_sm(ldg_sm_buffer_index, ldg_reg_buffer_index, loop)                                                \
  {                                                                                                                    \
    /* const int m = (loop * WARP_COUNT + warp_id) * 16 + lane_id % 16; */                                             \
    /* const int k = lane_id / 16 * 8;  */                                                                             \
    STORE_FLOAT4_WITH_PTR(A_sm_ptr_for_ldg + loop * WARP_COUNT * 16 * A_sm_dim3                                        \
                            + (ldg_sm_buffer_index) * A_sm_dim1 * A_sm_dim2 * A_sm_dim3,                               \
                          &A_ldg_reg[ldg_reg_buffer_index][loop][0]);                                                  \
  }

#define B_ldg_reg_2_sm(ldg_sm_buffer_index, ldg_reg_buffer_index, loop)                                                \
  {                                                                                                                    \
    /*const int k = lane_id % 16; */                                                                                   \
    /*const int n = (loop * WARP_COUNT + warp_id) * 16 + lane_id / 16 * 8;*/                                           \
    STORE_FLOAT4_WITH_PTR(B_sm_ptr_for_ldg + (loop) * WARP_COUNT * 2 * B_sm_dim3                                       \
                            + (ldg_sm_buffer_index) * B_sm_dim1 * B_sm_dim2 * B_sm_dim3,                               \
                          &B_ldg_reg[ldg_reg_buffer_index][loop][0]);                                                  \
  }

#define sm_2_A_mma_reg(ldg_sm_buffer_index, mma_reg_buffer_index, group)                                               \
  if constexpr (M_GROUP_COUNT_PER_WARP == 1) {                                                                         \
    /* for (int group = 0; group < M_GROUP_COUNT_PER_WARP; ++group) */ {                                               \
      uint32_t src = __cvta_generic_to_shared(A_sm_ptr_for_mma + (group) * 8 * A_sm_dim3                               \
                                              + (ldg_sm_buffer_index) * A_sm_dim1 * A_sm_dim2 * A_sm_dim3);            \
      asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"                                          \
                   : "=r"(*(uint32_t*)&A_mma_reg[mma_reg_buffer_index][group][0]),                                     \
                     "=r"(*(uint32_t*)&A_mma_reg[mma_reg_buffer_index][group][2])                                      \
                   : "r"(src));                                                                                        \
    }                                                                                                                  \
  }                                                                                                                    \
  else if constexpr (M_GROUP_COUNT_PER_WARP % 2 == 0) {                                                                \
    /*for (int group = 0; group < M_GROUP_COUNT_PER_WARP; group += 2) */ {                                             \
      uint32_t src = __cvta_generic_to_shared(A_sm_ptr_for_mma + (group) * 8 * A_sm_dim3                               \
                                              + (ldg_sm_buffer_index) * A_sm_dim1 * A_sm_dim2 * A_sm_dim3);            \
      asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"                                  \
                   : "=r"(*(uint32_t*)&A_mma_reg[mma_reg_buffer_index][group][0]),                                     \
                     "=r"(*(uint32_t*)&A_mma_reg[mma_reg_buffer_index][group][2]),                                     \
                     "=r"(*(uint32_t*)&A_mma_reg[mma_reg_buffer_index][group + 1][0]),                                 \
                     "=r"(*(uint32_t*)&A_mma_reg[mma_reg_buffer_index][group + 1][2])                                  \
                   : "r"(src));                                                                                        \
    }                                                                                                                  \
  }                                                                                                                    \
  else {                                                                                                               \
    static_assert(M_GROUP_COUNT_PER_WARP == 1 || M_GROUP_COUNT_PER_WARP % 2 == 0);                                     \
  }

#define sm_2_B_mma_reg(ldg_sm_buffer_index, mma_reg_buffer_index, group)                                               \
  /* for (int group = 0; group < N_GROUP_COUNT_PER_WARP; ++group) */ {                                                 \
    uint32_t src = __cvta_generic_to_shared(B_sm_ptr_for_mma + (group) * 2 * B_sm_dim3                                 \
                                            + (ldg_sm_buffer_index) * B_sm_dim1 * B_sm_dim2 * B_sm_dim3);              \
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"                              \
                 : "=r"(*(uint32_t*)&B_mma_reg[mma_reg_buffer_index][group][0]),                                       \
                   "=r"(*(uint32_t*)&B_mma_reg[mma_reg_buffer_index][group][2]),                                       \
                   "=r"(*(uint32_t*)&B_mma_reg[mma_reg_buffer_index][group][4]),                                       \
                   "=r"(*(uint32_t*)&B_mma_reg[mma_reg_buffer_index][group][6])                                        \
                 : "r"(src));                                                                                          \
  }

  T* C_ptr = &C[(m_block_offset + m_warp_offset + lane_id / 4) * N + n_block_offset + n_warp_offset + lane_id % 4 * 8];

#define ldm_mma_sts_stg_ldg(ldm_sm_buffer_index,                                                                       \
                            ldm_reg_buffer_index,                                                                      \
                            mma_reg_buffer_index,                                                                      \
                            sts_sm_base_index,                                                                         \
                            ldg_k_offset_x_2,                                                                          \
                            ldg_k_offset_x_2N,                                                                         \
                            rank,                                                                                      \
                            ldm_switch,                                                                                \
                            mma_switch,                                                                                \
                            sts_switch,                                                                                \
                            stg_switch,                                                                                \
                            ldg_switch)                                                                                \
  {                                                                                                                    \
    /* STS */                                                                                                          \
    constexpr int MxN_GORUP_COUNT = M_GROUP_COUNT_PER_WARP * N_GROUP_COUNT_PER_WARP;                                   \
    constexpr int STS_COUNT       = LDG_REG_BUFFER_SIZE * (A_LDG_LOOP_COUNT + B_LDG_LOOP_COUNT);                       \
    static_assert(STS_COUNT <= MxN_GORUP_COUNT);                                                                       \
    if constexpr (sts_switch && MxN_GORUP_COUNT - STS_COUNT <= rank && rank < MxN_GORUP_COUNT) {                       \
      constexpr int sts_rank        = rank - (MxN_GORUP_COUNT - STS_COUNT);                                            \
      constexpr int sts_addr_offset = sts_rank / (A_LDG_LOOP_COUNT + B_LDG_LOOP_COUNT);                                \
      constexpr int sts_loop        = sts_rank % (A_LDG_LOOP_COUNT + B_LDG_LOOP_COUNT);                                \
      if constexpr (sts_loop < A_LDG_LOOP_COUNT) {                                                                     \
        A_ldg_reg_2_sm(sts_sm_base_index + sts_addr_offset, sts_addr_offset, sts_loop);                                \
      }                                                                                                                \
      if constexpr (A_LDG_LOOP_COUNT <= sts_loop) {                                                                    \
        B_ldg_reg_2_sm(sts_sm_base_index + sts_addr_offset, sts_addr_offset, sts_loop - A_LDG_LOOP_COUNT);             \
      }                                                                                                                \
    }                                                                                                                  \
    /* MMA */                                                                                                          \
    if constexpr (mma_switch && rank < MxN_GORUP_COUNT) {                                                              \
      constexpr int mg = rank % M_GROUP_COUNT_PER_WARP;                                                                \
      constexpr int ng = rank / M_GROUP_COUNT_PER_WARP;                                                                \
      mma_m16n8k16_row_col(C_mma_reg[mg][ng],                                                                          \
                           B_mma_reg[mma_reg_buffer_index][ng],                                                        \
                           A_mma_reg[mma_reg_buffer_index][mg],                                                        \
                           C_mma_reg[mg][ng]);                                                                         \
    }                                                                                                                  \
    /* LDM */                                                                                                          \
    static_assert(M_GROUP_COUNT_PER_WARP == 1 || M_GROUP_COUNT_PER_WARP % 2 == 0);                                     \
    static_assert((M_GROUP_COUNT_PER_WARP + 1) / 2 + N_GROUP_COUNT_PER_WARP < MxN_GORUP_COUNT);                        \
    if constexpr (ldm_switch && rank < (M_GROUP_COUNT_PER_WARP + 1) / 2) {                                             \
      sm_2_A_mma_reg(ldm_sm_buffer_index, ldm_reg_buffer_index, rank * 2);                                             \
    }                                                                                                                  \
    if constexpr (ldm_switch && (M_GROUP_COUNT_PER_WARP + 1) / 2 <= rank                                               \
                  && rank < (M_GROUP_COUNT_PER_WARP + 1) / 2 + N_GROUP_COUNT_PER_WARP) {                               \
      sm_2_B_mma_reg(ldm_sm_buffer_index, ldm_reg_buffer_index, rank - (M_GROUP_COUNT_PER_WARP + 1) / 2);              \
    }                                                                                                                  \
    /* LDG */                                                                                                          \
    if constexpr (ldg_switch && rank < LDG_REG_BUFFER_SIZE * (A_LDG_LOOP_COUNT + B_LDG_LOOP_COUNT)) {                  \
      constexpr int ldg_addr_offset = rank / (A_LDG_LOOP_COUNT + B_LDG_LOOP_COUNT);                                    \
      constexpr int ldg_loop        = rank % (A_LDG_LOOP_COUNT + B_LDG_LOOP_COUNT);                                    \
      static_assert(ldg_addr_offset <= 2);                                                                             \
      if constexpr (ldg_loop < A_LDG_LOOP_COUNT) {                                                                     \
        switch (ldg_loop) {                                                                                            \
          case 0:                                                                                                      \
            switch (ldg_addr_offset) {                                                                                 \
              case 0:                                                                                                  \
                A_global_2_ldg_reg(                                                                                    \
                  A_global_ptr_for_ldg__loop_0__k_0 + ldg_k_offset_x_2, ldg_addr_offset, ldg_loop, ldg_switch);        \
                break;                                                                                                 \
              case 1:                                                                                                  \
                A_global_2_ldg_reg(                                                                                    \
                  A_global_ptr_for_ldg__loop_0__k_1 + ldg_k_offset_x_2, ldg_addr_offset, ldg_loop, ldg_switch);        \
                break;                                                                                                 \
              default:                                                                                                 \
                break;                                                                                                 \
            }                                                                                                          \
            break;                                                                                                     \
          case 1:                                                                                                      \
            switch (ldg_addr_offset) {                                                                                 \
              case 0:                                                                                                  \
                A_global_2_ldg_reg(                                                                                    \
                  A_global_ptr_for_ldg__loop_1__k_0 + ldg_k_offset_x_2, ldg_addr_offset, ldg_loop, ldg_switch);        \
                break;                                                                                                 \
              case 1:                                                                                                  \
                A_global_2_ldg_reg(                                                                                    \
                  A_global_ptr_for_ldg__loop_1__k_1 + ldg_k_offset_x_2, ldg_addr_offset, ldg_loop, ldg_switch);        \
                break;                                                                                                 \
              default:                                                                                                 \
                break;                                                                                                 \
            }                                                                                                          \
            break;                                                                                                     \
          case 2:                                                                                                      \
            switch (ldg_addr_offset) {                                                                                 \
              case 0:                                                                                                  \
                A_global_2_ldg_reg(                                                                                    \
                  A_global_ptr_for_ldg__loop_2__k_0 + ldg_k_offset_x_2, ldg_addr_offset, ldg_loop, ldg_switch);        \
                break;                                                                                                 \
              case 1:                                                                                                  \
                A_global_2_ldg_reg(                                                                                    \
                  A_global_ptr_for_ldg__loop_2__k_1 + ldg_k_offset_x_2, ldg_addr_offset, ldg_loop, ldg_switch);        \
                break;                                                                                                 \
              default:                                                                                                 \
                break;                                                                                                 \
            }                                                                                                          \
            break;                                                                                                     \
          case 3:                                                                                                      \
            switch (ldg_addr_offset) {                                                                                 \
              case 0:                                                                                                  \
                A_global_2_ldg_reg(                                                                                    \
                  A_global_ptr_for_ldg__loop_3__k_0 + ldg_k_offset_x_2, ldg_addr_offset, ldg_loop, ldg_switch);        \
                break;                                                                                                 \
              case 1:                                                                                                  \
                A_global_2_ldg_reg(                                                                                    \
                  A_global_ptr_for_ldg__loop_3__k_1 + ldg_k_offset_x_2, ldg_addr_offset, ldg_loop, ldg_switch);        \
                break;                                                                                                 \
              default:                                                                                                 \
                break;                                                                                                 \
            }                                                                                                          \
            break;                                                                                                     \
          default:                                                                                                     \
            break;                                                                                                     \
        }                                                                                                              \
      }                                                                                                                \
      if constexpr (A_LDG_LOOP_COUNT <= ldg_loop) {                                                                    \
        constexpr int real_ldg_loop = ldg_loop - A_LDG_LOOP_COUNT;                                                     \
        switch (real_ldg_loop) {                                                                                       \
          case 0:                                                                                                      \
            switch (ldg_addr_offset) {                                                                                 \
              case 0:                                                                                                  \
                B_global_2_ldg_reg(                                                                                    \
                  B_global_ptr_for_ldg__loop_0__k_0 + ldg_k_offset_x_2N, ldg_addr_offset, real_ldg_loop, ldg_switch);  \
                break;                                                                                                 \
              case 1:                                                                                                  \
                B_global_2_ldg_reg(                                                                                    \
                  B_global_ptr_for_ldg__loop_0__k_1 + ldg_k_offset_x_2N, ldg_addr_offset, real_ldg_loop, ldg_switch);  \
                break;                                                                                                 \
              default:                                                                                                 \
                break;                                                                                                 \
            }                                                                                                          \
            break;                                                                                                     \
          case 1:                                                                                                      \
            switch (ldg_addr_offset) {                                                                                 \
              case 0:                                                                                                  \
                B_global_2_ldg_reg(                                                                                    \
                  B_global_ptr_for_ldg__loop_1__k_0 + ldg_k_offset_x_2N, ldg_addr_offset, real_ldg_loop, ldg_switch);  \
                break;                                                                                                 \
              case 1:                                                                                                  \
                B_global_2_ldg_reg(                                                                                    \
                  B_global_ptr_for_ldg__loop_1__k_1 + ldg_k_offset_x_2N, ldg_addr_offset, real_ldg_loop, ldg_switch);  \
                break;                                                                                                 \
              default:                                                                                                 \
                break;                                                                                                 \
            }                                                                                                          \
            break;                                                                                                     \
          case 2:                                                                                                      \
            switch (ldg_addr_offset) {                                                                                 \
              case 0:                                                                                                  \
                B_global_2_ldg_reg(                                                                                    \
                  B_global_ptr_for_ldg__loop_2__k_0 + ldg_k_offset_x_2N, ldg_addr_offset, real_ldg_loop, ldg_switch);  \
                break;                                                                                                 \
              case 1:                                                                                                  \
                B_global_2_ldg_reg(                                                                                    \
                  B_global_ptr_for_ldg__loop_2__k_1 + ldg_k_offset_x_2N, ldg_addr_offset, real_ldg_loop, ldg_switch);  \
                break;                                                                                                 \
              default:                                                                                                 \
                break;                                                                                                 \
            }                                                                                                          \
            break;                                                                                                     \
          case 3:                                                                                                      \
            switch (ldg_addr_offset) {                                                                                 \
              case 0:                                                                                                  \
                B_global_2_ldg_reg(                                                                                    \
                  B_global_ptr_for_ldg__loop_3__k_0 + ldg_k_offset_x_2N, ldg_addr_offset, real_ldg_loop, ldg_switch);  \
                break;                                                                                                 \
              case 1:                                                                                                  \
                B_global_2_ldg_reg(                                                                                    \
                  B_global_ptr_for_ldg__loop_3__k_1 + ldg_k_offset_x_2N, ldg_addr_offset, real_ldg_loop, ldg_switch);  \
                break;                                                                                                 \
              default:                                                                                                 \
                break;                                                                                                 \
            }                                                                                                          \
            break;                                                                                                     \
          default:                                                                                                     \
            break;                                                                                                     \
        }                                                                                                              \
      }                                                                                                                \
    }                                                                                                                  \
    /* STG */                                                                                                          \
    if constexpr (stg_switch && rank < MxN_GORUP_COUNT) {                                                              \
      constexpr int mg = rank % M_GROUP_COUNT_PER_WARP;                                                                \
      constexpr int ng = rank / M_GROUP_COUNT_PER_WARP;                                                                \
      T casted[4]      = {C_mma_reg[mg][ng][0], C_mma_reg[mg][ng][1], C_mma_reg[mg][ng][2], C_mma_reg[mg][ng][3]};     \
      asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n"                                                   \
                   : "=r"(*(uint32_t*)&C_transposed[mg][ng / 2]._2x4[ng % 2][0])                                       \
                   : "r"(*(uint32_t*)&casted[0]));                                                                     \
      asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n"                                                   \
                   : "=r"(*(uint32_t*)&C_transposed[mg][ng / 2]._2x4[ng % 2][2])                                       \
                   : "r"(*(uint32_t*)&casted[2]));                                                                     \
      shfl_23_and_01(C_transposed[mg][ng / 2]._2x4[ng % 2], 0x1, lane_id);                                             \
      if constexpr ((ng + 1) % 2 == 0) {                                                                               \
        shfl_4567_and_0123(C_transposed[mg][ng / 2]._1x8, 0x2, lane_id);                                               \
        asm volatile("st.global.wt.v4.f32 [%0], {%1, %2, %3, %4};"                                                     \
                     :                                                                                                 \
                     : "l"(C_ptr + mg * 8 * N + (ng - 1) * 16),                                                        \
                       "f"(*(const float*)&C_transposed[mg][ng / 2]._1x8[0]),                                          \
                       "f"(*(const float*)&C_transposed[mg][ng / 2]._1x8[2]),                                          \
                       "f"(*(const float*)&C_transposed[mg][ng / 2]._1x8[4]),                                          \
                       "f"(*(const float*)&C_transposed[mg][ng / 2]._1x8[6])                                           \
                     : "memory");                                                                                      \
      }                                                                                                                \
    }                                                                                                                  \
  }

// FIXME This code is really stupid. Please find a way to optimize it as soon as possible.
#define alternate_ldm_mma_sts_stg_ldg(ldm_sm_buf_index,                                                                \
                                      ldm_reg_buf_index,                                                               \
                                      mma_reg_buf_index,                                                               \
                                      sts_sm_base_index,                                                               \
                                      ldg_k_offset_x2,                                                                 \
                                      ldg_k_offset_x2N,                                                                \
                                      ldm_switch,                                                                      \
                                      mma_switch,                                                                      \
                                      sts_switch,                                                                      \
                                      stg_switch,                                                                      \
                                      ldg_switch)                                                                      \
  static_assert(M_GROUP_COUNT_PER_WARP * N_GROUP_COUNT_PER_WARP <= 32);                                                \
  /* clang-format off */ \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N,  0, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N,  1, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N,  2, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N,  3, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N,  4, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N,  5, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N,  6, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N,  7, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N,  8, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N,  9, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N, 10, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N, 11, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N, 12, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N, 13, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N, 14, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N, 15, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N, 16, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N, 17, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N, 18, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N, 19, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N, 20, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N, 21, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N, 22, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N, 23, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N, 24, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N, 25, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N, 26, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N, 27, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N, 28, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N, 29, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N, 30, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N, 31, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);
  /* clang-format on */

  alternate_ldm_mma_sts_stg_ldg(0, 0, 0, 0, 0, 0, false, false, false, false, LDG_SWITCH_ON_EVICT_LAST);
  alternate_ldm_mma_sts_stg_ldg(0, 0, 0, 0, 0, 0, false, false, true, false, LDG_SWITCH_OFF);

  __syncthreads();

  int       LDG_SM_BUFFER_INDEX = 0;
  int       k_loop_offset       = LOOP_TILE_K * 2;
  int       k_loop_offset_x2    = k_loop_offset * sizeof(T);
  int       k_loop_offset_x2N   = k_loop_offset_x2 * N;
  const int Nx2xLOOP_TILE_K     = LOOP_TILE_K * 2 * sizeof(T) * N;

  alternate_ldm_mma_sts_stg_ldg(
    LDG_SM_BUFFER_INDEX, MMA_REG_BUFFER_INDEX_0, 0, 0, 0, 0, true, false, false, false, LDG_SWITCH_OFF);

  {
    alternate_ldm_mma_sts_stg_ldg(LDG_SM_BUFFER_INDEX + 1,
                                  MMA_REG_BUFFER_INDEX_1,
                                  MMA_REG_BUFFER_INDEX_0,
                                  (LDG_SM_BUFFER_INDEX ^ 2),
                                  k_loop_offset_x2,
                                  k_loop_offset_x2N,
                                  true,
                                  true,
                                  true,
                                  false,
                                  LDG_SWITCH_ON_EVICT_LAST);

    LDG_SM_BUFFER_INDEX ^= 2;
    k_loop_offset += LOOP_TILE_K * 2;
    k_loop_offset_x2 += LOOP_TILE_K * 2 * sizeof(T);
    k_loop_offset_x2N += Nx2xLOOP_TILE_K;

    __syncthreads();

    alternate_ldm_mma_sts_stg_ldg(LDG_SM_BUFFER_INDEX,
                                  MMA_REG_BUFFER_INDEX_0,
                                  MMA_REG_BUFFER_INDEX_1,
                                  0,
                                  k_loop_offset_x2,
                                  k_loop_offset_x2N,
                                  true,
                                  true,
                                  false,
                                  false,
                                  LDG_SWITCH_ON_EVICT_NORMAL);
  }

  while (k_loop_offset + LOOP_TILE_K * 2 < K) {
    alternate_ldm_mma_sts_stg_ldg(LDG_SM_BUFFER_INDEX + 1,
                                  MMA_REG_BUFFER_INDEX_1,
                                  MMA_REG_BUFFER_INDEX_0,
                                  (LDG_SM_BUFFER_INDEX ^ 2),
                                  0,
                                  0,
                                  true,
                                  true,
                                  true,
                                  false,
                                  LDG_SWITCH_OFF);

    LDG_SM_BUFFER_INDEX ^= 2;
    k_loop_offset += LOOP_TILE_K * 2;
    k_loop_offset_x2 += LOOP_TILE_K * 2 * sizeof(T);
    k_loop_offset_x2N += Nx2xLOOP_TILE_K;

    __syncthreads();

    alternate_ldm_mma_sts_stg_ldg(
      0, 0, 0, 0, k_loop_offset_x2, k_loop_offset_x2N, false, false, false, false, LDG_SWITCH_ON_EVICT_NORMAL);

    alternate_ldm_mma_sts_stg_ldg(LDG_SM_BUFFER_INDEX,
                                  MMA_REG_BUFFER_INDEX_0,
                                  MMA_REG_BUFFER_INDEX_1,
                                  0,
                                  0,
                                  0,
                                  true,
                                  true,
                                  false,
                                  false,
                                  LDG_SWITCH_OFF);
  }

  {
    alternate_ldm_mma_sts_stg_ldg(LDG_SM_BUFFER_INDEX + 1,
                                  MMA_REG_BUFFER_INDEX_1,
                                  MMA_REG_BUFFER_INDEX_0,
                                  (LDG_SM_BUFFER_INDEX ^ 2),
                                  0,
                                  0,
                                  true,
                                  true,
                                  true,
                                  false,
                                  LDG_SWITCH_OFF);
    LDG_SM_BUFFER_INDEX ^= 2;
    k_loop_offset += LOOP_TILE_K * 2;
    k_loop_offset_x2 += LOOP_TILE_K * 2 * sizeof(T);
    k_loop_offset_x2N += Nx2xLOOP_TILE_K;

    __syncthreads();

    alternate_ldm_mma_sts_stg_ldg(LDG_SM_BUFFER_INDEX,
                                  MMA_REG_BUFFER_INDEX_0,
                                  MMA_REG_BUFFER_INDEX_1,
                                  0,
                                  0,
                                  0,
                                  true,
                                  true,
                                  false,
                                  false,
                                  LDG_SWITCH_OFF);
  }

  {
    alternate_ldm_mma_sts_stg_ldg(LDG_SM_BUFFER_INDEX + 1,
                                  MMA_REG_BUFFER_INDEX_1,
                                  MMA_REG_BUFFER_INDEX_0,
                                  0,
                                  0,
                                  0,
                                  true,
                                  true,
                                  false,
                                  false,
                                  LDG_SWITCH_OFF);

    alternate_ldm_mma_sts_stg_ldg(LDG_SM_BUFFER_INDEX,
                                  MMA_REG_BUFFER_INDEX_0,
                                  MMA_REG_BUFFER_INDEX_1,
                                  0,
                                  0,
                                  0,
                                  false,
                                  true,
                                  false,
                                  true,
                                  LDG_SWITCH_OFF);
  }

#undef A_global_2_ldg_reg
#undef A_ldg_reg_2_sm
#undef B_global_2_ldg_reg
#undef B_ldg_reg_2_sm
#undef sm_2_A_mma_reg
#undef sm_2_B_mma_reg
#undef ldm_mma_sts_stg_ldg
#undef alternate_ldm_mma_sts_stg_ldg
}

template<typename T, int BLOCK_TILE_M, int BLOCK_TILE_N, int WARP_TILE_M, int WARP_TILE_N>
__global__ void
fp16_mma_m16n8k16_ldmatrix_trans__overlap_global_2_sm__quadra_buffer__reduce_instructions__reorder_instructions__overlap_reg_2_global__stg_memory_coalecesing__overlap_sts__stg_wt__reduce_IMAD_AND_LEA__reduce_MIO(
  const T* A, const T* B, T* C, int M, int N, int K)
{
  // if(threadIdx.x == 0) {
  //   printf("%03d %03d %03d\n", blockIdx.x, blockIdx.y, get_smid());
  // }
  constexpr int WARP_COUNT   = BLOCK_TILE_M / WARP_TILE_M * BLOCK_TILE_N / WARP_TILE_N;
  constexpr int THREAD_COUNT = WARP_COUNT * 32;

  constexpr int LOOP_TILE_K         = 16;
  constexpr int LDG_SM_BUFFER_SIZE  = 4;
  constexpr int LDG_REG_BUFFER_SIZE = 2;

  constexpr int A_sm_dim0 = LDG_SM_BUFFER_SIZE;
  constexpr int A_sm_dim1 = 2;
  constexpr int A_sm_dim2 = BLOCK_TILE_M;
  constexpr int A_sm_dim3 = LOOP_TILE_K / 2;
  constexpr int B_sm_dim0 = LDG_SM_BUFFER_SIZE;
  constexpr int B_sm_dim1 = LOOP_TILE_K / 8;
  constexpr int B_sm_dim2 = BLOCK_TILE_N / 8;
  constexpr int B_sm_dim3 = 64;

  // The 64 elements of type T in each 8x8 matrix are stored consecutively in a single layer of shared memory.
  __shared__ union {
    struct {
      T A_sm[A_sm_dim0 * A_sm_dim1 * A_sm_dim2 * A_sm_dim3];
      T B_sm[B_sm_dim0 * B_sm_dim1 * B_sm_dim2 * B_sm_dim3];
    } mma;
    static_assert(WARP_TILE_N % 16 == 0);
    T result[WARP_COUNT][WARP_TILE_N / 16][WARP_TILE_M][16];
  } data;

  static_assert(BLOCK_TILE_M * LOOP_TILE_K % THREAD_COUNT == 0);
  static_assert(BLOCK_TILE_M * LOOP_TILE_K / THREAD_COUNT % 8 == 0);
  constexpr int A_LDG_COUNT_PER_THREAD = BLOCK_TILE_M * LOOP_TILE_K / THREAD_COUNT;
  constexpr int A_LDG_LOOP_COUNT       = A_LDG_COUNT_PER_THREAD / 8;
  // clang-format off
  // This is the thread layout of the same warp that loads matrix A, where each thread reads M1xK8 elements of type T at a
  // loop iteration.
  // T0  T16
  // T1  T17
  // T2  T18
  // ... ...
  // T14 T30
  // T15 T31
  // clang-format on
  float A_ldg_reg[LDG_REG_BUFFER_SIZE][A_LDG_LOOP_COUNT][4];

  static_assert(BLOCK_TILE_N * LOOP_TILE_K % THREAD_COUNT == 0);
  static_assert(BLOCK_TILE_N * LOOP_TILE_K / THREAD_COUNT % 8 == 0);
  constexpr int B_LDG_COUNT_PER_THREAD = BLOCK_TILE_N * LOOP_TILE_K / THREAD_COUNT;
  constexpr int B_LDG_LOOP_COUNT       = B_LDG_COUNT_PER_THREAD / 8;
  // clang-format off
  // This is the thread layout of the same warp that loads matrix B, where each thread reads K1xN8 elements of type T at a
  // loop iteration.
  // T0  T16
  // T1  T17
  // T2  T18
  // ... ...
  // T14 T30
  // T15 T31
  // clang-format on
  float B_ldg_reg[LDG_REG_BUFFER_SIZE][B_LDG_LOOP_COUNT][4];

  const int m_block_offset = blockIdx.y * BLOCK_TILE_M;
  const int n_block_offset = blockIdx.x * BLOCK_TILE_N;

  const int warp_id                 = threadIdx.x / 32;
  const int lane_id                 = threadIdx.x % 32;
  const int transposed_lane_id_mask = (lane_id / 8 == 0 || lane_id / 8 == 3) ? 0x00 : 0x18;
  const int transposed_lane_id      = lane_id ^ transposed_lane_id_mask;

  constexpr int M_MMA_WARP_COUNT       = BLOCK_TILE_M / WARP_TILE_M;
  constexpr int M_GROUP_COUNT_PER_WARP = WARP_TILE_M / 8;
  constexpr int N_GROUP_COUNT_PER_WARP = WARP_TILE_N / 16;

  constexpr int MMA_REG_BUFFER_SIZE    = 2;
  constexpr int MMA_REG_BUFFER_INDEX_0 = 0;
  constexpr int MMA_REG_BUFFER_INDEX_1 = 1;
  T             A_mma_reg[MMA_REG_BUFFER_SIZE][M_GROUP_COUNT_PER_WARP][4];
  T             B_mma_reg[MMA_REG_BUFFER_SIZE][N_GROUP_COUNT_PER_WARP][8];
  float         C_mma_reg[M_GROUP_COUNT_PER_WARP][N_GROUP_COUNT_PER_WARP][4] = {0};
  static_assert(N_GROUP_COUNT_PER_WARP % 2 == 0);
  union _2x4_or_1x8 {
    T _2x4[2][4];
    T _1x8[8];
  };
  _2x4_or_1x8 C_transposed[M_GROUP_COUNT_PER_WARP][N_GROUP_COUNT_PER_WARP / 2];

  const int m_warp_offset = warp_id % M_MMA_WARP_COUNT * WARP_TILE_M;
  const int n_warp_offset = warp_id / M_MMA_WARP_COUNT * WARP_TILE_N;

  const int A_ldg_reg_2_A_sm_partial_offset =
    lane_id / 16 * A_sm_dim2 * A_sm_dim3 + (warp_id * 16 + lane_id % 16) * A_sm_dim3;

  const int B_ldg_reg_2_B_sm_partial_offset =
    (lane_id % 16) / 8 * B_sm_dim2 * B_sm_dim3 + (lane_id % 16) % 8 * 8 + (warp_id * 2 + lane_id / 16) * B_sm_dim3;

  const int A_global_partial_offset = (m_block_offset + warp_id * 16 + lane_id % 16) * K + lane_id / 16 * 8;
  const int B_global_partial_offset = lane_id % 16 * N + n_block_offset + warp_id * 16 + lane_id / 16 * 8;

  const T* A_global_ptr_for_ldg = &A[A_global_partial_offset];
  const T* B_global_ptr_for_ldg = &B[B_global_partial_offset];

  static_assert(A_LDG_LOOP_COUNT <= 4);
  static_assert(B_LDG_LOOP_COUNT <= 4);
  static_assert(LDG_REG_BUFFER_SIZE <= 2);
  const uint64_t A_global_ptr_for_ldg__loop_0__k_0 = (uint64_t)(A_global_ptr_for_ldg + 0 * WARP_COUNT * 16 * K + 0);
  const uint64_t A_global_ptr_for_ldg__loop_1__k_0 = (uint64_t)(A_global_ptr_for_ldg + 1 * WARP_COUNT * 16 * K + 0);
  const uint64_t A_global_ptr_for_ldg__loop_2__k_0 = (uint64_t)(A_global_ptr_for_ldg + 2 * WARP_COUNT * 16 * K + 0);
  const uint64_t A_global_ptr_for_ldg__loop_3__k_0 = (uint64_t)(A_global_ptr_for_ldg + 3 * WARP_COUNT * 16 * K + 0);
  const uint64_t A_global_ptr_for_ldg__loop_0__k_1 =
    (uint64_t)(A_global_ptr_for_ldg + 0 * WARP_COUNT * 16 * K + LOOP_TILE_K);
  const uint64_t A_global_ptr_for_ldg__loop_1__k_1 =
    (uint64_t)(A_global_ptr_for_ldg + 1 * WARP_COUNT * 16 * K + LOOP_TILE_K);
  const uint64_t A_global_ptr_for_ldg__loop_2__k_1 =
    (uint64_t)(A_global_ptr_for_ldg + 2 * WARP_COUNT * 16 * K + LOOP_TILE_K);
  const uint64_t A_global_ptr_for_ldg__loop_3__k_1 =
    (uint64_t)(A_global_ptr_for_ldg + 3 * WARP_COUNT * 16 * K + LOOP_TILE_K);
  const uint64_t B_global_ptr_for_ldg__loop_0__k_0 = (uint64_t)(B_global_ptr_for_ldg + 0 * WARP_COUNT * 16 + 0);
  const uint64_t B_global_ptr_for_ldg__loop_1__k_0 = (uint64_t)(B_global_ptr_for_ldg + 1 * WARP_COUNT * 16 + 0);
  const uint64_t B_global_ptr_for_ldg__loop_2__k_0 = (uint64_t)(B_global_ptr_for_ldg + 2 * WARP_COUNT * 16 + 0);
  const uint64_t B_global_ptr_for_ldg__loop_3__k_0 = (uint64_t)(B_global_ptr_for_ldg + 3 * WARP_COUNT * 16 + 0);
  const uint64_t B_global_ptr_for_ldg__loop_0__k_1 =
    (uint64_t)(B_global_ptr_for_ldg + 0 * WARP_COUNT * 16 + LOOP_TILE_K * N);
  const uint64_t B_global_ptr_for_ldg__loop_1__k_1 =
    (uint64_t)(B_global_ptr_for_ldg + 1 * WARP_COUNT * 16 + LOOP_TILE_K * N);
  const uint64_t B_global_ptr_for_ldg__loop_2__k_1 =
    (uint64_t)(B_global_ptr_for_ldg + 2 * WARP_COUNT * 16 + LOOP_TILE_K * N);
  const uint64_t B_global_ptr_for_ldg__loop_3__k_1 =
    (uint64_t)(B_global_ptr_for_ldg + 3 * WARP_COUNT * 16 + LOOP_TILE_K * N);

  const T* A_sm_ptr_for_ldg = &data.mma.A_sm[A_ldg_reg_2_A_sm_partial_offset];
  const T* B_sm_ptr_for_ldg = &data.mma.B_sm[B_ldg_reg_2_B_sm_partial_offset];

  const int A_sm_2_A_mma_reg_partial_offset =
    lane_id % 16 / 8 * A_sm_dim2 * A_sm_dim3 + (m_warp_offset + lane_id % 8) * A_sm_dim3;

  const int B_sm_2_B_mma_reg_partial_offset = transposed_lane_id % 16 / 8 * B_sm_dim2 * B_sm_dim3
                                              + (n_warp_offset + transposed_lane_id / 16 * 8) / 8 * B_sm_dim3
                                              + transposed_lane_id % 8 * 8;

  const T* A_sm_ptr_for_mma = &data.mma.A_sm[A_sm_2_A_mma_reg_partial_offset + lane_id / 16 * 8 * A_sm_dim3];
  const T* B_sm_ptr_for_mma = &data.mma.B_sm[B_sm_2_B_mma_reg_partial_offset];

  enum {
    LDG_SWITCH_OFF             = 0,
    LDG_SWITCH_ON_EVICT_NORMAL = 1,
    LDG_SWITCH_ON_EVICT_LAST   = 2,
  };

#define A_global_2_ldg_reg(A_global_ptr, ldg_reg_buffer_index, loop, cache_policy)                                     \
  {                                                                                                                    \
    /* const int m = (loop * WARP_COUNT + warp_id) * 16 + lane_id % 16; */                                             \
    /* const int k = lane_id / 16 * 8; */                                                                              \
    if constexpr (cache_policy == LDG_SWITCH_ON_EVICT_LAST) {                                                          \
      FETCH_FLOAT4_EVICT_LAST_AND_PREFETCH_256B_WITH_SRC_PTR(A_ldg_reg[ldg_reg_buffer_index][loop], A_global_ptr);     \
    }                                                                                                                  \
    else {                                                                                                             \
      FETCH_FLOAT4_PREFETCH_256B_WITH_SRC_PTR(A_ldg_reg[ldg_reg_buffer_index][loop], A_global_ptr);                    \
    }                                                                                                                  \
  }

#define B_global_2_ldg_reg(B_global_ptr, ldg_reg_buffer_index, loop, cache_policy)                                     \
  {                                                                                                                    \
    /* const int k = lane_id % 16;                                           */                                        \
    /* const int n = (loop * WARP_COUNT + warp_id) * 16 + lane_id / 16 * 8;  */                                        \
    if constexpr (cache_policy == LDG_SWITCH_ON_EVICT_LAST) {                                                          \
      FETCH_FLOAT4_EVICT_LAST_AND_PREFETCH_256B_WITH_SRC_PTR(B_ldg_reg[ldg_reg_buffer_index][loop], B_global_ptr);     \
    }                                                                                                                  \
    else {                                                                                                             \
      FETCH_FLOAT4_PREFETCH_256B_WITH_SRC_PTR(B_ldg_reg[ldg_reg_buffer_index][loop], B_global_ptr);                    \
    }                                                                                                                  \
  }

#define A_ldg_reg_2_sm(ldg_sm_buffer_index, ldg_reg_buffer_index, loop)                                                \
  {                                                                                                                    \
    /* const int m = (loop * WARP_COUNT + warp_id) * 16 + lane_id % 16; */                                             \
    /* const int k = lane_id / 16 * 8;  */                                                                             \
    STORE_FLOAT4_WITH_PTR(A_sm_ptr_for_ldg + loop * WARP_COUNT * 16 * A_sm_dim3                                        \
                            + (ldg_sm_buffer_index) * A_sm_dim1 * A_sm_dim2 * A_sm_dim3,                               \
                          &A_ldg_reg[ldg_reg_buffer_index][loop][0]);                                                  \
  }

#define B_ldg_reg_2_sm(ldg_sm_buffer_index, ldg_reg_buffer_index, loop)                                                \
  {                                                                                                                    \
    /*const int k = lane_id % 16; */                                                                                   \
    /*const int n = (loop * WARP_COUNT + warp_id) * 16 + lane_id / 16 * 8;*/                                           \
    STORE_FLOAT4_WITH_PTR(B_sm_ptr_for_ldg + (loop) * WARP_COUNT * 2 * B_sm_dim3                                       \
                            + (ldg_sm_buffer_index) * B_sm_dim1 * B_sm_dim2 * B_sm_dim3,                               \
                          &B_ldg_reg[ldg_reg_buffer_index][loop][0]);                                                  \
  }

#define sm_2_A_mma_reg(ldg_sm_buffer_index, mma_reg_buffer_index, group)                                               \
  if constexpr (M_GROUP_COUNT_PER_WARP == 1) {                                                                         \
    /* for (int group = 0; group < M_GROUP_COUNT_PER_WARP; ++group) */ {                                               \
      uint32_t src = __cvta_generic_to_shared(A_sm_ptr_for_mma + (group) * 8 * A_sm_dim3                               \
                                              + (ldg_sm_buffer_index) * A_sm_dim1 * A_sm_dim2 * A_sm_dim3);            \
      asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"                                          \
                   : "=r"(*(uint32_t*)&A_mma_reg[mma_reg_buffer_index][group][0]),                                     \
                     "=r"(*(uint32_t*)&A_mma_reg[mma_reg_buffer_index][group][2])                                      \
                   : "r"(src));                                                                                        \
    }                                                                                                                  \
  }                                                                                                                    \
  else if constexpr (M_GROUP_COUNT_PER_WARP % 2 == 0) {                                                                \
    /*for (int group = 0; group < M_GROUP_COUNT_PER_WARP; group += 2) */ {                                             \
      uint32_t src = __cvta_generic_to_shared(A_sm_ptr_for_mma + (group) * 8 * A_sm_dim3                               \
                                              + (ldg_sm_buffer_index) * A_sm_dim1 * A_sm_dim2 * A_sm_dim3);            \
      asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"                                  \
                   : "=r"(*(uint32_t*)&A_mma_reg[mma_reg_buffer_index][group][0]),                                     \
                     "=r"(*(uint32_t*)&A_mma_reg[mma_reg_buffer_index][group][2]),                                     \
                     "=r"(*(uint32_t*)&A_mma_reg[mma_reg_buffer_index][group + 1][0]),                                 \
                     "=r"(*(uint32_t*)&A_mma_reg[mma_reg_buffer_index][group + 1][2])                                  \
                   : "r"(src));                                                                                        \
    }                                                                                                                  \
  }                                                                                                                    \
  else {                                                                                                               \
    static_assert(M_GROUP_COUNT_PER_WARP == 1 || M_GROUP_COUNT_PER_WARP % 2 == 0);                                     \
  }

#define sm_2_B_mma_reg(ldg_sm_buffer_index, mma_reg_buffer_index, group)                                               \
  /* for (int group = 0; group < N_GROUP_COUNT_PER_WARP; ++group) */ {                                                 \
    uint32_t src = __cvta_generic_to_shared(B_sm_ptr_for_mma + (group) * 2 * B_sm_dim3                                 \
                                            + (ldg_sm_buffer_index) * B_sm_dim1 * B_sm_dim2 * B_sm_dim3);              \
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"                              \
                 : "=r"(*(uint32_t*)&B_mma_reg[mma_reg_buffer_index][group][0]),                                       \
                   "=r"(*(uint32_t*)&B_mma_reg[mma_reg_buffer_index][group][2]),                                       \
                   "=r"(*(uint32_t*)&B_mma_reg[mma_reg_buffer_index][group][4]),                                       \
                   "=r"(*(uint32_t*)&B_mma_reg[mma_reg_buffer_index][group][6])                                        \
                 : "r"(src));                                                                                          \
  }

  T* C_ptr = &C[(m_block_offset + m_warp_offset + lane_id / 4) * N + n_block_offset + n_warp_offset + lane_id % 4 * 8];

#define ldm_mma_sts_stg_ldg(ldm_sm_buffer_index,                                                                       \
                            ldm_reg_buffer_index,                                                                      \
                            mma_reg_buffer_index,                                                                      \
                            sts_sm_base_index,                                                                         \
                            ldg_k_offset_x_2,                                                                          \
                            ldg_k_offset_x_2N,                                                                         \
                            rank,                                                                                      \
                            ldm_switch,                                                                                \
                            mma_switch,                                                                                \
                            sts_switch,                                                                                \
                            stg_switch,                                                                                \
                            ldg_switch)                                                                                \
  {                                                                                                                    \
    /* STS */                                                                                                          \
    constexpr int MxN_GORUP_COUNT = M_GROUP_COUNT_PER_WARP * N_GROUP_COUNT_PER_WARP;                                   \
    constexpr int STS_COUNT       = LDG_REG_BUFFER_SIZE * (A_LDG_LOOP_COUNT + B_LDG_LOOP_COUNT);                       \
    static_assert(STS_COUNT <= MxN_GORUP_COUNT);                                                                       \
    if constexpr (sts_switch && MxN_GORUP_COUNT - STS_COUNT <= rank && rank < MxN_GORUP_COUNT) {                       \
      constexpr int sts_rank        = rank - (MxN_GORUP_COUNT - STS_COUNT);                                            \
      constexpr int sts_addr_offset = sts_rank / (A_LDG_LOOP_COUNT + B_LDG_LOOP_COUNT);                                \
      constexpr int sts_loop        = sts_rank % (A_LDG_LOOP_COUNT + B_LDG_LOOP_COUNT);                                \
      if constexpr (sts_loop < A_LDG_LOOP_COUNT) {                                                                     \
        A_ldg_reg_2_sm(sts_sm_base_index + sts_addr_offset, sts_addr_offset, sts_loop);                                \
      }                                                                                                                \
      if constexpr (A_LDG_LOOP_COUNT <= sts_loop) {                                                                    \
        B_ldg_reg_2_sm(sts_sm_base_index + sts_addr_offset, sts_addr_offset, sts_loop - A_LDG_LOOP_COUNT);             \
      }                                                                                                                \
    }                                                                                                                  \
    /* MMA */                                                                                                          \
    if constexpr (mma_switch && rank < MxN_GORUP_COUNT) {                                                              \
      constexpr int mg = rank % M_GROUP_COUNT_PER_WARP;                                                                \
      constexpr int ng = rank / M_GROUP_COUNT_PER_WARP;                                                                \
      mma_m16n8k16_row_col(C_mma_reg[mg][ng],                                                                          \
                           B_mma_reg[mma_reg_buffer_index][ng],                                                        \
                           A_mma_reg[mma_reg_buffer_index][mg],                                                        \
                           C_mma_reg[mg][ng]);                                                                         \
    }                                                                                                                  \
    /* LDM */                                                                                                          \
    static_assert(M_GROUP_COUNT_PER_WARP == 1 || M_GROUP_COUNT_PER_WARP % 2 == 0);                                     \
    constexpr int A_LDM_COUNT = (M_GROUP_COUNT_PER_WARP + 1) / 2;                                                      \
    constexpr int B_LDM_COUNT = N_GROUP_COUNT_PER_WARP;                                                                \
    constexpr int LDM_COUNT   = A_LDM_COUNT + B_LDM_COUNT;                                                             \
    static_assert(LDM_COUNT < MxN_GORUP_COUNT);                                                                        \
    constexpr int LDM_STRIDE = (MxN_GORUP_COUNT - 1) / (LDM_COUNT - 1);                                                \
    static_assert((LDM_COUNT - 1) * LDM_STRIDE + 1 <= MxN_GORUP_COUNT);                                                \
    if constexpr (ldm_switch && rank < (LDM_COUNT - 1) * LDM_STRIDE + 1 && rank % LDM_STRIDE == 0) {                   \
      constexpr int ldm_rank = rank / LDM_STRIDE;                                                                      \
      if constexpr (ldm_switch && ldm_rank < A_LDM_COUNT) {                                                            \
        sm_2_A_mma_reg(ldm_sm_buffer_index, ldm_reg_buffer_index, ldm_rank * 2);                                       \
      }                                                                                                                \
      if constexpr (ldm_switch && A_LDM_COUNT <= ldm_rank && ldm_rank < LDM_COUNT) {                                   \
        sm_2_B_mma_reg(ldm_sm_buffer_index, ldm_reg_buffer_index, ldm_rank - A_LDM_COUNT);                             \
      }                                                                                                                \
    }                                                                                                                  \
    /* LDG */                                                                                                          \
    if constexpr (ldg_switch && rank < LDG_REG_BUFFER_SIZE * (A_LDG_LOOP_COUNT + B_LDG_LOOP_COUNT)) {                  \
      constexpr int ldg_addr_offset = rank / (A_LDG_LOOP_COUNT + B_LDG_LOOP_COUNT);                                    \
      constexpr int ldg_loop        = rank % (A_LDG_LOOP_COUNT + B_LDG_LOOP_COUNT);                                    \
      static_assert(ldg_addr_offset <= 2);                                                                             \
      if constexpr (ldg_loop < A_LDG_LOOP_COUNT) {                                                                     \
        switch (ldg_loop) {                                                                                            \
          case 0:                                                                                                      \
            switch (ldg_addr_offset) {                                                                                 \
              case 0:                                                                                                  \
                A_global_2_ldg_reg(                                                                                    \
                  A_global_ptr_for_ldg__loop_0__k_0 + ldg_k_offset_x_2, ldg_addr_offset, ldg_loop, ldg_switch);        \
                break;                                                                                                 \
              case 1:                                                                                                  \
                A_global_2_ldg_reg(                                                                                    \
                  A_global_ptr_for_ldg__loop_0__k_1 + ldg_k_offset_x_2, ldg_addr_offset, ldg_loop, ldg_switch);        \
                break;                                                                                                 \
              default:                                                                                                 \
                break;                                                                                                 \
            }                                                                                                          \
            break;                                                                                                     \
          case 1:                                                                                                      \
            switch (ldg_addr_offset) {                                                                                 \
              case 0:                                                                                                  \
                A_global_2_ldg_reg(                                                                                    \
                  A_global_ptr_for_ldg__loop_1__k_0 + ldg_k_offset_x_2, ldg_addr_offset, ldg_loop, ldg_switch);        \
                break;                                                                                                 \
              case 1:                                                                                                  \
                A_global_2_ldg_reg(                                                                                    \
                  A_global_ptr_for_ldg__loop_1__k_1 + ldg_k_offset_x_2, ldg_addr_offset, ldg_loop, ldg_switch);        \
                break;                                                                                                 \
              default:                                                                                                 \
                break;                                                                                                 \
            }                                                                                                          \
            break;                                                                                                     \
          case 2:                                                                                                      \
            switch (ldg_addr_offset) {                                                                                 \
              case 0:                                                                                                  \
                A_global_2_ldg_reg(                                                                                    \
                  A_global_ptr_for_ldg__loop_2__k_0 + ldg_k_offset_x_2, ldg_addr_offset, ldg_loop, ldg_switch);        \
                break;                                                                                                 \
              case 1:                                                                                                  \
                A_global_2_ldg_reg(                                                                                    \
                  A_global_ptr_for_ldg__loop_2__k_1 + ldg_k_offset_x_2, ldg_addr_offset, ldg_loop, ldg_switch);        \
                break;                                                                                                 \
              default:                                                                                                 \
                break;                                                                                                 \
            }                                                                                                          \
            break;                                                                                                     \
          case 3:                                                                                                      \
            switch (ldg_addr_offset) {                                                                                 \
              case 0:                                                                                                  \
                A_global_2_ldg_reg(                                                                                    \
                  A_global_ptr_for_ldg__loop_3__k_0 + ldg_k_offset_x_2, ldg_addr_offset, ldg_loop, ldg_switch);        \
                break;                                                                                                 \
              case 1:                                                                                                  \
                A_global_2_ldg_reg(                                                                                    \
                  A_global_ptr_for_ldg__loop_3__k_1 + ldg_k_offset_x_2, ldg_addr_offset, ldg_loop, ldg_switch);        \
                break;                                                                                                 \
              default:                                                                                                 \
                break;                                                                                                 \
            }                                                                                                          \
            break;                                                                                                     \
          default:                                                                                                     \
            break;                                                                                                     \
        }                                                                                                              \
      }                                                                                                                \
      if constexpr (A_LDG_LOOP_COUNT <= ldg_loop) {                                                                    \
        constexpr int real_ldg_loop = ldg_loop - A_LDG_LOOP_COUNT;                                                     \
        switch (real_ldg_loop) {                                                                                       \
          case 0:                                                                                                      \
            switch (ldg_addr_offset) {                                                                                 \
              case 0:                                                                                                  \
                B_global_2_ldg_reg(                                                                                    \
                  B_global_ptr_for_ldg__loop_0__k_0 + ldg_k_offset_x_2N, ldg_addr_offset, real_ldg_loop, ldg_switch);  \
                break;                                                                                                 \
              case 1:                                                                                                  \
                B_global_2_ldg_reg(                                                                                    \
                  B_global_ptr_for_ldg__loop_0__k_1 + ldg_k_offset_x_2N, ldg_addr_offset, real_ldg_loop, ldg_switch);  \
                break;                                                                                                 \
              default:                                                                                                 \
                break;                                                                                                 \
            }                                                                                                          \
            break;                                                                                                     \
          case 1:                                                                                                      \
            switch (ldg_addr_offset) {                                                                                 \
              case 0:                                                                                                  \
                B_global_2_ldg_reg(                                                                                    \
                  B_global_ptr_for_ldg__loop_1__k_0 + ldg_k_offset_x_2N, ldg_addr_offset, real_ldg_loop, ldg_switch);  \
                break;                                                                                                 \
              case 1:                                                                                                  \
                B_global_2_ldg_reg(                                                                                    \
                  B_global_ptr_for_ldg__loop_1__k_1 + ldg_k_offset_x_2N, ldg_addr_offset, real_ldg_loop, ldg_switch);  \
                break;                                                                                                 \
              default:                                                                                                 \
                break;                                                                                                 \
            }                                                                                                          \
            break;                                                                                                     \
          case 2:                                                                                                      \
            switch (ldg_addr_offset) {                                                                                 \
              case 0:                                                                                                  \
                B_global_2_ldg_reg(                                                                                    \
                  B_global_ptr_for_ldg__loop_2__k_0 + ldg_k_offset_x_2N, ldg_addr_offset, real_ldg_loop, ldg_switch);  \
                break;                                                                                                 \
              case 1:                                                                                                  \
                B_global_2_ldg_reg(                                                                                    \
                  B_global_ptr_for_ldg__loop_2__k_1 + ldg_k_offset_x_2N, ldg_addr_offset, real_ldg_loop, ldg_switch);  \
                break;                                                                                                 \
              default:                                                                                                 \
                break;                                                                                                 \
            }                                                                                                          \
            break;                                                                                                     \
          case 3:                                                                                                      \
            switch (ldg_addr_offset) {                                                                                 \
              case 0:                                                                                                  \
                B_global_2_ldg_reg(                                                                                    \
                  B_global_ptr_for_ldg__loop_3__k_0 + ldg_k_offset_x_2N, ldg_addr_offset, real_ldg_loop, ldg_switch);  \
                break;                                                                                                 \
              case 1:                                                                                                  \
                B_global_2_ldg_reg(                                                                                    \
                  B_global_ptr_for_ldg__loop_3__k_1 + ldg_k_offset_x_2N, ldg_addr_offset, real_ldg_loop, ldg_switch);  \
                break;                                                                                                 \
              default:                                                                                                 \
                break;                                                                                                 \
            }                                                                                                          \
            break;                                                                                                     \
          default:                                                                                                     \
            break;                                                                                                     \
        }                                                                                                              \
      }                                                                                                                \
    }                                                                                                                  \
    /* STG */                                                                                                          \
    if constexpr (stg_switch && rank < MxN_GORUP_COUNT) {                                                              \
      constexpr int mg = rank % M_GROUP_COUNT_PER_WARP;                                                                \
      constexpr int ng = rank / M_GROUP_COUNT_PER_WARP;                                                                \
      T casted[4]      = {C_mma_reg[mg][ng][0], C_mma_reg[mg][ng][1], C_mma_reg[mg][ng][2], C_mma_reg[mg][ng][3]};     \
      asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n"                                                   \
                   : "=r"(*(uint32_t*)&C_transposed[mg][ng / 2]._2x4[ng % 2][0])                                       \
                   : "r"(*(uint32_t*)&casted[0]));                                                                     \
      asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n"                                                   \
                   : "=r"(*(uint32_t*)&C_transposed[mg][ng / 2]._2x4[ng % 2][2])                                       \
                   : "r"(*(uint32_t*)&casted[2]));                                                                     \
      shfl_23_and_01(C_transposed[mg][ng / 2]._2x4[ng % 2], 0x1, lane_id);                                             \
      if constexpr ((ng + 1) % 2 == 0) {                                                                               \
        shfl_4567_and_0123(C_transposed[mg][ng / 2]._1x8, 0x2, lane_id);                                               \
        asm volatile("st.global.wt.v4.f32 [%0], {%1, %2, %3, %4};"                                                     \
                     :                                                                                                 \
                     : "l"(C_ptr + mg * 8 * N + (ng - 1) * 16),                                                        \
                       "f"(*(const float*)&C_transposed[mg][ng / 2]._1x8[0]),                                          \
                       "f"(*(const float*)&C_transposed[mg][ng / 2]._1x8[2]),                                          \
                       "f"(*(const float*)&C_transposed[mg][ng / 2]._1x8[4]),                                          \
                       "f"(*(const float*)&C_transposed[mg][ng / 2]._1x8[6])                                           \
                     : "memory");                                                                                      \
      }                                                                                                                \
    }                                                                                                                  \
  }

// FIXME This code is really stupid. Please find a way to optimize it as soon as possible.
#define alternate_ldm_mma_sts_stg_ldg(ldm_sm_buf_index,                                                                \
                                      ldm_reg_buf_index,                                                               \
                                      mma_reg_buf_index,                                                               \
                                      sts_sm_base_index,                                                               \
                                      ldg_k_offset_x2,                                                                 \
                                      ldg_k_offset_x2N,                                                                \
                                      ldm_switch,                                                                      \
                                      mma_switch,                                                                      \
                                      sts_switch,                                                                      \
                                      stg_switch,                                                                      \
                                      ldg_switch)                                                                      \
  static_assert(M_GROUP_COUNT_PER_WARP * N_GROUP_COUNT_PER_WARP <= 32);                                                \
  /* clang-format off */ \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N,  0, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N,  1, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N,  2, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N,  3, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N,  4, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N,  5, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N,  6, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N,  7, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N,  8, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N,  9, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N, 10, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N, 11, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N, 12, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N, 13, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N, 14, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N, 15, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N, 16, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N, 17, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N, 18, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N, 19, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N, 20, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N, 21, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N, 22, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N, 23, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N, 24, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N, 25, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N, 26, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N, 27, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N, 28, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N, 29, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N, 30, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N, 31, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);
  /* clang-format on */

  alternate_ldm_mma_sts_stg_ldg(0, 0, 0, 0, 0, 0, false, false, false, false, LDG_SWITCH_ON_EVICT_LAST);
  alternate_ldm_mma_sts_stg_ldg(0, 0, 0, 0, 0, 0, false, false, true, false, LDG_SWITCH_OFF);

  __syncthreads();

  int       LDG_SM_BUFFER_INDEX = 0;
  int       k_loop_offset       = LOOP_TILE_K * 2;
  int       k_loop_offset_x2    = k_loop_offset * sizeof(T);
  int       k_loop_offset_x2N   = k_loop_offset_x2 * N;
  const int Nx2xLOOP_TILE_K     = LOOP_TILE_K * 2 * sizeof(T) * N;

  alternate_ldm_mma_sts_stg_ldg(
    LDG_SM_BUFFER_INDEX, MMA_REG_BUFFER_INDEX_0, 0, 0, 0, 0, true, false, false, false, LDG_SWITCH_OFF);

  {
    alternate_ldm_mma_sts_stg_ldg(LDG_SM_BUFFER_INDEX + 1,
                                  MMA_REG_BUFFER_INDEX_1,
                                  MMA_REG_BUFFER_INDEX_0,
                                  (LDG_SM_BUFFER_INDEX ^ 2),
                                  k_loop_offset_x2,
                                  k_loop_offset_x2N,
                                  true,
                                  true,
                                  true,
                                  false,
                                  LDG_SWITCH_ON_EVICT_LAST);

    LDG_SM_BUFFER_INDEX ^= 2;
    k_loop_offset += LOOP_TILE_K * 2;
    k_loop_offset_x2 += LOOP_TILE_K * 2 * sizeof(T);
    k_loop_offset_x2N += Nx2xLOOP_TILE_K;

    __syncthreads();

    alternate_ldm_mma_sts_stg_ldg(LDG_SM_BUFFER_INDEX,
                                  MMA_REG_BUFFER_INDEX_0,
                                  MMA_REG_BUFFER_INDEX_1,
                                  0,
                                  k_loop_offset_x2,
                                  k_loop_offset_x2N,
                                  true,
                                  true,
                                  false,
                                  false,
                                  LDG_SWITCH_ON_EVICT_NORMAL);
  }

  while (k_loop_offset + LOOP_TILE_K * 2 < K) {
    alternate_ldm_mma_sts_stg_ldg(LDG_SM_BUFFER_INDEX + 1,
                                  MMA_REG_BUFFER_INDEX_1,
                                  MMA_REG_BUFFER_INDEX_0,
                                  (LDG_SM_BUFFER_INDEX ^ 2),
                                  0,
                                  0,
                                  true,
                                  true,
                                  true,
                                  false,
                                  LDG_SWITCH_OFF);

    LDG_SM_BUFFER_INDEX ^= 2;
    k_loop_offset += LOOP_TILE_K * 2;
    k_loop_offset_x2 += LOOP_TILE_K * 2 * sizeof(T);
    k_loop_offset_x2N += Nx2xLOOP_TILE_K;

    __syncthreads();

    alternate_ldm_mma_sts_stg_ldg(
      0, 0, 0, 0, k_loop_offset_x2, k_loop_offset_x2N, false, false, false, false, LDG_SWITCH_ON_EVICT_NORMAL);

    alternate_ldm_mma_sts_stg_ldg(LDG_SM_BUFFER_INDEX,
                                  MMA_REG_BUFFER_INDEX_0,
                                  MMA_REG_BUFFER_INDEX_1,
                                  0,
                                  0,
                                  0,
                                  true,
                                  true,
                                  false,
                                  false,
                                  LDG_SWITCH_OFF);
  }

  {
    alternate_ldm_mma_sts_stg_ldg(LDG_SM_BUFFER_INDEX + 1,
                                  MMA_REG_BUFFER_INDEX_1,
                                  MMA_REG_BUFFER_INDEX_0,
                                  (LDG_SM_BUFFER_INDEX ^ 2),
                                  0,
                                  0,
                                  true,
                                  true,
                                  true,
                                  false,
                                  LDG_SWITCH_OFF);
    LDG_SM_BUFFER_INDEX ^= 2;
    k_loop_offset += LOOP_TILE_K * 2;
    k_loop_offset_x2 += LOOP_TILE_K * 2 * sizeof(T);
    k_loop_offset_x2N += Nx2xLOOP_TILE_K;

    __syncthreads();

    alternate_ldm_mma_sts_stg_ldg(LDG_SM_BUFFER_INDEX,
                                  MMA_REG_BUFFER_INDEX_0,
                                  MMA_REG_BUFFER_INDEX_1,
                                  0,
                                  0,
                                  0,
                                  true,
                                  true,
                                  false,
                                  false,
                                  LDG_SWITCH_OFF);
  }

  {
    alternate_ldm_mma_sts_stg_ldg(LDG_SM_BUFFER_INDEX + 1,
                                  MMA_REG_BUFFER_INDEX_1,
                                  MMA_REG_BUFFER_INDEX_0,
                                  0,
                                  0,
                                  0,
                                  true,
                                  true,
                                  false,
                                  false,
                                  LDG_SWITCH_OFF);

    alternate_ldm_mma_sts_stg_ldg(LDG_SM_BUFFER_INDEX,
                                  MMA_REG_BUFFER_INDEX_0,
                                  MMA_REG_BUFFER_INDEX_1,
                                  0,
                                  0,
                                  0,
                                  false,
                                  true,
                                  false,
                                  true,
                                  LDG_SWITCH_OFF);
  }

#undef A_global_2_ldg_reg
#undef A_ldg_reg_2_sm
#undef B_global_2_ldg_reg
#undef B_ldg_reg_2_sm
#undef sm_2_A_mma_reg
#undef sm_2_B_mma_reg
#undef ldm_mma_sts_stg_ldg
#undef alternate_ldm_mma_sts_stg_ldg
}

template<typename T, int BLOCK_TILE_M, int BLOCK_TILE_N, int WARP_TILE_M, int WARP_TILE_N>
__global__ void
fp16_mma_m16n8k16_ldmatrix_trans__overlap_global_2_sm__quadra_buffer__reduce_instructions__reorder_instructions__overlap_reg_2_global__stg_memory_coalecesing__overlap_sts__stg_wt__reduce_IMAD_AND_LEA__reduce_MIO__opt_BAR(
  const T* A, const T* B, T* C, int M, int N, int K)
{
  constexpr int WARP_COUNT   = BLOCK_TILE_M / WARP_TILE_M * BLOCK_TILE_N / WARP_TILE_N;
  constexpr int THREAD_COUNT = WARP_COUNT * 32;

  constexpr int LOOP_TILE_K         = 16;
  constexpr int LDG_SM_BUFFER_SIZE  = 4;
  constexpr int LDG_REG_BUFFER_SIZE = 2;

  constexpr int A_sm_dim0 = LDG_SM_BUFFER_SIZE;
  constexpr int A_sm_dim1 = 2;
  constexpr int A_sm_dim2 = BLOCK_TILE_M;
  constexpr int A_sm_dim3 = LOOP_TILE_K / 2;
  constexpr int B_sm_dim0 = LDG_SM_BUFFER_SIZE;
  constexpr int B_sm_dim1 = LOOP_TILE_K / 8;
  constexpr int B_sm_dim2 = BLOCK_TILE_N / 8;
  constexpr int B_sm_dim3 = 64;

  // The 64 elements of type T in each 8x8 matrix are stored consecutively in a single layer of shared memory.
  __shared__ union {
    struct {
      T A_sm[A_sm_dim0 * A_sm_dim1 * A_sm_dim2 * A_sm_dim3];
      T B_sm[B_sm_dim0 * B_sm_dim1 * B_sm_dim2 * B_sm_dim3];
    } mma;
    static_assert(WARP_TILE_N % 16 == 0);
    T result[WARP_COUNT][WARP_TILE_N / 16][WARP_TILE_M][16];
  } data;

  static_assert(BLOCK_TILE_M * LOOP_TILE_K % THREAD_COUNT == 0);
  static_assert(BLOCK_TILE_M * LOOP_TILE_K / THREAD_COUNT % 8 == 0);
  constexpr int A_LDG_COUNT_PER_THREAD = BLOCK_TILE_M * LOOP_TILE_K / THREAD_COUNT;
  constexpr int A_LDG_LOOP_COUNT       = A_LDG_COUNT_PER_THREAD / 8;
  // clang-format off
  // This is the thread layout of the same warp that loads matrix A, where each thread reads M1xK8 elements of type T at a
  // loop iteration.
  // T0  T16
  // T1  T17
  // T2  T18
  // ... ...
  // T14 T30
  // T15 T31
  // clang-format on
  float A_ldg_reg[LDG_REG_BUFFER_SIZE][A_LDG_LOOP_COUNT][4];

  static_assert(BLOCK_TILE_N * LOOP_TILE_K % THREAD_COUNT == 0);
  static_assert(BLOCK_TILE_N * LOOP_TILE_K / THREAD_COUNT % 8 == 0);
  constexpr int B_LDG_COUNT_PER_THREAD = BLOCK_TILE_N * LOOP_TILE_K / THREAD_COUNT;
  constexpr int B_LDG_LOOP_COUNT       = B_LDG_COUNT_PER_THREAD / 8;
  // clang-format off
  // This is the thread layout of the same warp that loads matrix B, where each thread reads K1xN8 elements of type T at a
  // loop iteration.
  // T0  T16
  // T1  T17
  // T2  T18
  // ... ...
  // T14 T30
  // T15 T31
  // clang-format on
  float B_ldg_reg[LDG_REG_BUFFER_SIZE][B_LDG_LOOP_COUNT][4];

  const int m_block_offset = blockIdx.y * BLOCK_TILE_M;
  const int n_block_offset = blockIdx.x * BLOCK_TILE_N;

  const int warp_id                 = threadIdx.x / 32;
  const int lane_id                 = threadIdx.x % 32;
  const int transposed_lane_id_mask = (lane_id / 8 == 0 || lane_id / 8 == 3) ? 0x00 : 0x18;
  const int transposed_lane_id      = lane_id ^ transposed_lane_id_mask;

  constexpr int M_MMA_WARP_COUNT       = BLOCK_TILE_M / WARP_TILE_M;
  constexpr int M_GROUP_COUNT_PER_WARP = WARP_TILE_M / 8;
  constexpr int N_GROUP_COUNT_PER_WARP = WARP_TILE_N / 16;

  constexpr int MMA_REG_BUFFER_SIZE    = 2;
  constexpr int MMA_REG_BUFFER_INDEX_0 = 0;
  constexpr int MMA_REG_BUFFER_INDEX_1 = 1;
  T             A_mma_reg[MMA_REG_BUFFER_SIZE][M_GROUP_COUNT_PER_WARP][4];
  T             B_mma_reg[MMA_REG_BUFFER_SIZE][N_GROUP_COUNT_PER_WARP][8];
  float         C_mma_reg[M_GROUP_COUNT_PER_WARP][N_GROUP_COUNT_PER_WARP][4] = {0};
  static_assert(N_GROUP_COUNT_PER_WARP % 2 == 0);
  union _2x4_or_1x8 {
    T _2x4[2][4];
    T _1x8[8];
  };
  _2x4_or_1x8 C_transposed[M_GROUP_COUNT_PER_WARP][N_GROUP_COUNT_PER_WARP / 2];

  const int m_warp_offset = warp_id % M_MMA_WARP_COUNT * WARP_TILE_M;
  const int n_warp_offset = warp_id / M_MMA_WARP_COUNT * WARP_TILE_N;

  const int A_ldg_reg_2_A_sm_partial_offset =
    lane_id / 16 * A_sm_dim2 * A_sm_dim3 + (warp_id * 16 + lane_id % 16) * A_sm_dim3;

  const int B_ldg_reg_2_B_sm_partial_offset =
    (lane_id % 16) / 8 * B_sm_dim2 * B_sm_dim3 + (lane_id % 16) % 8 * 8 + (warp_id * 2 + lane_id / 16) * B_sm_dim3;

  const int A_global_partial_offset = (m_block_offset + warp_id * 16 + lane_id % 16) * K + lane_id / 16 * 8;
  const int B_global_partial_offset = lane_id % 16 * N + n_block_offset + warp_id * 16 + lane_id / 16 * 8;

  const T* A_global_ptr_for_ldg = &A[A_global_partial_offset];
  const T* B_global_ptr_for_ldg = &B[B_global_partial_offset];

  static_assert(A_LDG_LOOP_COUNT <= 4);
  static_assert(B_LDG_LOOP_COUNT <= 4);
  static_assert(LDG_REG_BUFFER_SIZE <= 2);
  const uint64_t A_global_ptr_for_ldg__loop_0__k_0 = (uint64_t)(A_global_ptr_for_ldg + 0 * WARP_COUNT * 16 * K + 0);
  const uint64_t A_global_ptr_for_ldg__loop_1__k_0 = (uint64_t)(A_global_ptr_for_ldg + 1 * WARP_COUNT * 16 * K + 0);
  const uint64_t A_global_ptr_for_ldg__loop_2__k_0 = (uint64_t)(A_global_ptr_for_ldg + 2 * WARP_COUNT * 16 * K + 0);
  const uint64_t A_global_ptr_for_ldg__loop_3__k_0 = (uint64_t)(A_global_ptr_for_ldg + 3 * WARP_COUNT * 16 * K + 0);
  const uint64_t A_global_ptr_for_ldg__loop_0__k_1 =
    (uint64_t)(A_global_ptr_for_ldg + 0 * WARP_COUNT * 16 * K + LOOP_TILE_K);
  const uint64_t A_global_ptr_for_ldg__loop_1__k_1 =
    (uint64_t)(A_global_ptr_for_ldg + 1 * WARP_COUNT * 16 * K + LOOP_TILE_K);
  const uint64_t A_global_ptr_for_ldg__loop_2__k_1 =
    (uint64_t)(A_global_ptr_for_ldg + 2 * WARP_COUNT * 16 * K + LOOP_TILE_K);
  const uint64_t A_global_ptr_for_ldg__loop_3__k_1 =
    (uint64_t)(A_global_ptr_for_ldg + 3 * WARP_COUNT * 16 * K + LOOP_TILE_K);
  const uint64_t B_global_ptr_for_ldg__loop_0__k_0 = (uint64_t)(B_global_ptr_for_ldg + 0 * WARP_COUNT * 16 + 0);
  const uint64_t B_global_ptr_for_ldg__loop_1__k_0 = (uint64_t)(B_global_ptr_for_ldg + 1 * WARP_COUNT * 16 + 0);
  const uint64_t B_global_ptr_for_ldg__loop_2__k_0 = (uint64_t)(B_global_ptr_for_ldg + 2 * WARP_COUNT * 16 + 0);
  const uint64_t B_global_ptr_for_ldg__loop_3__k_0 = (uint64_t)(B_global_ptr_for_ldg + 3 * WARP_COUNT * 16 + 0);
  const uint64_t B_global_ptr_for_ldg__loop_0__k_1 =
    (uint64_t)(B_global_ptr_for_ldg + 0 * WARP_COUNT * 16 + LOOP_TILE_K * N);
  const uint64_t B_global_ptr_for_ldg__loop_1__k_1 =
    (uint64_t)(B_global_ptr_for_ldg + 1 * WARP_COUNT * 16 + LOOP_TILE_K * N);
  const uint64_t B_global_ptr_for_ldg__loop_2__k_1 =
    (uint64_t)(B_global_ptr_for_ldg + 2 * WARP_COUNT * 16 + LOOP_TILE_K * N);
  const uint64_t B_global_ptr_for_ldg__loop_3__k_1 =
    (uint64_t)(B_global_ptr_for_ldg + 3 * WARP_COUNT * 16 + LOOP_TILE_K * N);

  const T* A_sm_ptr_for_ldg = &data.mma.A_sm[A_ldg_reg_2_A_sm_partial_offset];
  const T* B_sm_ptr_for_ldg = &data.mma.B_sm[B_ldg_reg_2_B_sm_partial_offset];

  const int A_sm_2_A_mma_reg_partial_offset =
    lane_id % 16 / 8 * A_sm_dim2 * A_sm_dim3 + (m_warp_offset + lane_id % 8) * A_sm_dim3;

  const int B_sm_2_B_mma_reg_partial_offset = transposed_lane_id % 16 / 8 * B_sm_dim2 * B_sm_dim3
                                              + (n_warp_offset + transposed_lane_id / 16 * 8) / 8 * B_sm_dim3
                                              + transposed_lane_id % 8 * 8;

  const T* A_sm_ptr_for_mma = &data.mma.A_sm[A_sm_2_A_mma_reg_partial_offset + lane_id / 16 * 8 * A_sm_dim3];
  const T* B_sm_ptr_for_mma = &data.mma.B_sm[B_sm_2_B_mma_reg_partial_offset];

  enum {
    LDG_SWITCH_OFF             = 0,
    LDG_SWITCH_ON_EVICT_NORMAL = 1,
    LDG_SWITCH_ON_EVICT_LAST   = 2,
  };

  using barrier = cuda::barrier<cuda::thread_scope_block>;
  __shared__ char        bar_storage[128];
  barrier*               bar = (barrier*)&bar_storage[0];
  barrier::arrival_token arrival_token;
  if (threadIdx.x == 0) {
    init(bar, THREAD_COUNT);
  }
  __syncthreads();

#define A_global_2_ldg_reg(A_global_ptr, ldg_reg_buffer_index, loop, cache_policy)                                     \
  {                                                                                                                    \
    /* const int m = (loop * WARP_COUNT + warp_id) * 16 + lane_id % 16; */                                             \
    /* const int k = lane_id / 16 * 8; */                                                                              \
    if constexpr (cache_policy == LDG_SWITCH_ON_EVICT_LAST) {                                                          \
      FETCH_FLOAT4_CONST_EVICT_LAST_WITH_SRC_PTR(A_ldg_reg[ldg_reg_buffer_index][loop], A_global_ptr);                 \
    }                                                                                                                  \
    else {                                                                                                             \
      FETCH_FLOAT4_CONST_PREFETCH_256B_WITH_SRC_PTR(A_ldg_reg[ldg_reg_buffer_index][loop], A_global_ptr);              \
    }                                                                                                                  \
  }

#define B_global_2_ldg_reg(B_global_ptr, ldg_reg_buffer_index, loop, cache_policy)                                     \
  {                                                                                                                    \
    /* const int k = lane_id % 16;                                           */                                        \
    /* const int n = (loop * WARP_COUNT + warp_id) * 16 + lane_id / 16 * 8;  */                                        \
    if constexpr (cache_policy == LDG_SWITCH_ON_EVICT_LAST) {                                                          \
      FETCH_FLOAT4_CONST_EVICT_LAST_WITH_SRC_PTR(B_ldg_reg[ldg_reg_buffer_index][loop], B_global_ptr);                 \
    }                                                                                                                  \
    else {                                                                                                             \
      FETCH_FLOAT4_CONST_PREFETCH_256B_WITH_SRC_PTR(B_ldg_reg[ldg_reg_buffer_index][loop], B_global_ptr);              \
    }                                                                                                                  \
  }

#define A_ldg_reg_2_sm(ldg_sm_buffer_index, ldg_reg_buffer_index, loop)                                                \
  {                                                                                                                    \
    /* const int m = (loop * WARP_COUNT + warp_id) * 16 + lane_id % 16; */                                             \
    /* const int k = lane_id / 16 * 8;  */                                                                             \
    STORE_FLOAT4_WITH_PTR(A_sm_ptr_for_ldg + loop * WARP_COUNT * 16 * A_sm_dim3                                        \
                            + (ldg_sm_buffer_index) * A_sm_dim1 * A_sm_dim2 * A_sm_dim3,                               \
                          &A_ldg_reg[ldg_reg_buffer_index][loop][0]);                                                  \
  }

#define B_ldg_reg_2_sm(ldg_sm_buffer_index, ldg_reg_buffer_index, loop)                                                \
  {                                                                                                                    \
    /*const int k = lane_id % 16; */                                                                                   \
    /*const int n = (loop * WARP_COUNT + warp_id) * 16 + lane_id / 16 * 8;*/                                           \
    STORE_FLOAT4_WITH_PTR(B_sm_ptr_for_ldg + (loop) * WARP_COUNT * 2 * B_sm_dim3                                       \
                            + (ldg_sm_buffer_index) * B_sm_dim1 * B_sm_dim2 * B_sm_dim3,                               \
                          &B_ldg_reg[ldg_reg_buffer_index][loop][0]);                                                  \
  }

#define sm_2_A_mma_reg(ldg_sm_buffer_index, mma_reg_buffer_index, group)                                               \
  if constexpr (M_GROUP_COUNT_PER_WARP == 1) {                                                                         \
    /* for (int group = 0; group < M_GROUP_COUNT_PER_WARP; ++group) */ {                                               \
      uint32_t src = __cvta_generic_to_shared(A_sm_ptr_for_mma + (group) * 8 * A_sm_dim3                               \
                                              + (ldg_sm_buffer_index) * A_sm_dim1 * A_sm_dim2 * A_sm_dim3);            \
      asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"                                          \
                   : "=r"(*(uint32_t*)&A_mma_reg[mma_reg_buffer_index][group][0]),                                     \
                     "=r"(*(uint32_t*)&A_mma_reg[mma_reg_buffer_index][group][2])                                      \
                   : "r"(src));                                                                                        \
    }                                                                                                                  \
  }                                                                                                                    \
  else if constexpr (M_GROUP_COUNT_PER_WARP % 2 == 0) {                                                                \
    /*for (int group = 0; group < M_GROUP_COUNT_PER_WARP; group += 2) */ {                                             \
      uint32_t src = __cvta_generic_to_shared(A_sm_ptr_for_mma + (group) * 8 * A_sm_dim3                               \
                                              + (ldg_sm_buffer_index) * A_sm_dim1 * A_sm_dim2 * A_sm_dim3);            \
      asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"                                  \
                   : "=r"(*(uint32_t*)&A_mma_reg[mma_reg_buffer_index][group][0]),                                     \
                     "=r"(*(uint32_t*)&A_mma_reg[mma_reg_buffer_index][group][2]),                                     \
                     "=r"(*(uint32_t*)&A_mma_reg[mma_reg_buffer_index][group + 1][0]),                                 \
                     "=r"(*(uint32_t*)&A_mma_reg[mma_reg_buffer_index][group + 1][2])                                  \
                   : "r"(src));                                                                                        \
    }                                                                                                                  \
  }                                                                                                                    \
  else {                                                                                                               \
    static_assert(M_GROUP_COUNT_PER_WARP == 1 || M_GROUP_COUNT_PER_WARP % 2 == 0);                                     \
  }

#define sm_2_B_mma_reg(ldg_sm_buffer_index, mma_reg_buffer_index, group)                                               \
  /* for (int group = 0; group < N_GROUP_COUNT_PER_WARP; ++group) */ {                                                 \
    uint32_t src = __cvta_generic_to_shared(B_sm_ptr_for_mma + (group) * 2 * B_sm_dim3                                 \
                                            + (ldg_sm_buffer_index) * B_sm_dim1 * B_sm_dim2 * B_sm_dim3);              \
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"                              \
                 : "=r"(*(uint32_t*)&B_mma_reg[mma_reg_buffer_index][group][0]),                                       \
                   "=r"(*(uint32_t*)&B_mma_reg[mma_reg_buffer_index][group][2]),                                       \
                   "=r"(*(uint32_t*)&B_mma_reg[mma_reg_buffer_index][group][4]),                                       \
                   "=r"(*(uint32_t*)&B_mma_reg[mma_reg_buffer_index][group][6])                                        \
                 : "r"(src));                                                                                          \
  }

  T* C_ptr = &C[(m_block_offset + m_warp_offset + lane_id / 4) * N + n_block_offset + n_warp_offset + lane_id % 4 * 8];

#define ldm_mma_sts_stg_ldg(ldm_sm_buffer_index,                                                                       \
                            ldm_reg_buffer_index,                                                                      \
                            mma_reg_buffer_index,                                                                      \
                            sts_sm_base_index,                                                                         \
                            ldg_k_offset_x_2,                                                                          \
                            ldg_k_offset_x_2N,                                                                         \
                            rank,                                                                                      \
                            ldm_switch,                                                                                \
                            mma_switch,                                                                                \
                            sts_switch,                                                                                \
                            stg_switch,                                                                                \
                            ldg_switch,                                                                                \
                            bar_arrive_switch,                                                                         \
                            bar_wait_switch)                                                                           \
  {                                                                                                                    \
    /* STS */                                                                                                          \
    constexpr int MxN_GORUP_COUNT = M_GROUP_COUNT_PER_WARP * N_GROUP_COUNT_PER_WARP;                                   \
    constexpr int STS_COUNT       = LDG_REG_BUFFER_SIZE * (A_LDG_LOOP_COUNT + B_LDG_LOOP_COUNT);                       \
    static_assert(STS_COUNT <= MxN_GORUP_COUNT);                                                                       \
    if constexpr (sts_switch && MxN_GORUP_COUNT - STS_COUNT <= rank && rank < MxN_GORUP_COUNT) {                       \
      constexpr int sts_rank        = rank - (MxN_GORUP_COUNT - STS_COUNT);                                            \
      constexpr int sts_addr_offset = sts_rank / (A_LDG_LOOP_COUNT + B_LDG_LOOP_COUNT);                                \
      constexpr int sts_loop        = sts_rank % (A_LDG_LOOP_COUNT + B_LDG_LOOP_COUNT);                                \
      if constexpr (sts_loop < A_LDG_LOOP_COUNT) {                                                                     \
        A_ldg_reg_2_sm(sts_sm_base_index + sts_addr_offset, sts_addr_offset, sts_loop);                                \
      }                                                                                                                \
      if constexpr (A_LDG_LOOP_COUNT <= sts_loop) {                                                                    \
        B_ldg_reg_2_sm(sts_sm_base_index + sts_addr_offset, sts_addr_offset, sts_loop - A_LDG_LOOP_COUNT);             \
      }                                                                                                                \
      if constexpr (bar_arrive_switch && sts_rank == STS_COUNT - 1) {                                                  \
        arrival_token = bar->arrive();                                                                                 \
      }                                                                                                                \
    }                                                                                                                  \
    /* MMA */                                                                                                          \
    if constexpr (mma_switch && rank < MxN_GORUP_COUNT) {                                                              \
      constexpr int mg = rank % M_GROUP_COUNT_PER_WARP;                                                                \
      constexpr int ng = rank / M_GROUP_COUNT_PER_WARP;                                                                \
      mma_m16n8k16_row_col(C_mma_reg[mg][ng],                                                                          \
                           B_mma_reg[mma_reg_buffer_index][ng],                                                        \
                           A_mma_reg[mma_reg_buffer_index][mg],                                                        \
                           C_mma_reg[mg][ng]);                                                                         \
    }                                                                                                                  \
    /* LDM */                                                                                                          \
    static_assert(M_GROUP_COUNT_PER_WARP == 1 || M_GROUP_COUNT_PER_WARP % 2 == 0);                                     \
    constexpr int A_LDM_COUNT = (M_GROUP_COUNT_PER_WARP + 1) / 2;                                                      \
    constexpr int B_LDM_COUNT = N_GROUP_COUNT_PER_WARP;                                                                \
    constexpr int LDM_COUNT   = A_LDM_COUNT + B_LDM_COUNT;                                                             \
    static_assert(LDM_COUNT < MxN_GORUP_COUNT);                                                                        \
    constexpr int LDM_STRIDE = (MxN_GORUP_COUNT - 1) / (LDM_COUNT - 1);                                                \
    static_assert((LDM_COUNT - 1) * LDM_STRIDE + 1 <= MxN_GORUP_COUNT);                                                \
    if constexpr (ldm_switch && rank < (LDM_COUNT - 1) * LDM_STRIDE + 1 && rank % LDM_STRIDE == 0) {                   \
      constexpr int ldm_rank = rank / LDM_STRIDE;                                                                      \
      if constexpr (false && bar_wait_switch && ldm_rank == 0) {                                                       \
        __syncthreads();                                                                                               \
      }                                                                                                                \
      if constexpr (bar_wait_switch && ldm_rank == 0) {                                                                \
        bar->wait(std::move(arrival_token));                                                                           \
      }                                                                                                                \
      if constexpr (ldm_switch && ldm_rank < A_LDM_COUNT) {                                                            \
        sm_2_A_mma_reg(ldm_sm_buffer_index, ldm_reg_buffer_index, ldm_rank * 2);                                       \
      }                                                                                                                \
      if constexpr (ldm_switch && A_LDM_COUNT <= ldm_rank && ldm_rank < LDM_COUNT) {                                   \
        sm_2_B_mma_reg(ldm_sm_buffer_index, ldm_reg_buffer_index, ldm_rank - A_LDM_COUNT);                             \
      }                                                                                                                \
    }                                                                                                                  \
    /* LDG */                                                                                                          \
    if constexpr (ldg_switch && rank < LDG_REG_BUFFER_SIZE * (A_LDG_LOOP_COUNT + B_LDG_LOOP_COUNT)) {                  \
      constexpr int ldg_addr_offset = rank / (A_LDG_LOOP_COUNT + B_LDG_LOOP_COUNT);                                    \
      constexpr int ldg_loop        = rank % (A_LDG_LOOP_COUNT + B_LDG_LOOP_COUNT);                                    \
      static_assert(ldg_addr_offset <= 2);                                                                             \
      if constexpr (ldg_loop < A_LDG_LOOP_COUNT) {                                                                     \
        switch (ldg_loop) {                                                                                            \
          case 0:                                                                                                      \
            switch (ldg_addr_offset) {                                                                                 \
              case 0:                                                                                                  \
                A_global_2_ldg_reg(                                                                                    \
                  A_global_ptr_for_ldg__loop_0__k_0 + ldg_k_offset_x_2, ldg_addr_offset, ldg_loop, ldg_switch);        \
                break;                                                                                                 \
              case 1:                                                                                                  \
                A_global_2_ldg_reg(                                                                                    \
                  A_global_ptr_for_ldg__loop_0__k_1 + ldg_k_offset_x_2, ldg_addr_offset, ldg_loop, ldg_switch);        \
                break;                                                                                                 \
              default:                                                                                                 \
                break;                                                                                                 \
            }                                                                                                          \
            break;                                                                                                     \
          case 1:                                                                                                      \
            switch (ldg_addr_offset) {                                                                                 \
              case 0:                                                                                                  \
                A_global_2_ldg_reg(                                                                                    \
                  A_global_ptr_for_ldg__loop_1__k_0 + ldg_k_offset_x_2, ldg_addr_offset, ldg_loop, ldg_switch);        \
                break;                                                                                                 \
              case 1:                                                                                                  \
                A_global_2_ldg_reg(                                                                                    \
                  A_global_ptr_for_ldg__loop_1__k_1 + ldg_k_offset_x_2, ldg_addr_offset, ldg_loop, ldg_switch);        \
                break;                                                                                                 \
              default:                                                                                                 \
                break;                                                                                                 \
            }                                                                                                          \
            break;                                                                                                     \
          case 2:                                                                                                      \
            switch (ldg_addr_offset) {                                                                                 \
              case 0:                                                                                                  \
                A_global_2_ldg_reg(                                                                                    \
                  A_global_ptr_for_ldg__loop_2__k_0 + ldg_k_offset_x_2, ldg_addr_offset, ldg_loop, ldg_switch);        \
                break;                                                                                                 \
              case 1:                                                                                                  \
                A_global_2_ldg_reg(                                                                                    \
                  A_global_ptr_for_ldg__loop_2__k_1 + ldg_k_offset_x_2, ldg_addr_offset, ldg_loop, ldg_switch);        \
                break;                                                                                                 \
              default:                                                                                                 \
                break;                                                                                                 \
            }                                                                                                          \
            break;                                                                                                     \
          case 3:                                                                                                      \
            switch (ldg_addr_offset) {                                                                                 \
              case 0:                                                                                                  \
                A_global_2_ldg_reg(                                                                                    \
                  A_global_ptr_for_ldg__loop_3__k_0 + ldg_k_offset_x_2, ldg_addr_offset, ldg_loop, ldg_switch);        \
                break;                                                                                                 \
              case 1:                                                                                                  \
                A_global_2_ldg_reg(                                                                                    \
                  A_global_ptr_for_ldg__loop_3__k_1 + ldg_k_offset_x_2, ldg_addr_offset, ldg_loop, ldg_switch);        \
                break;                                                                                                 \
              default:                                                                                                 \
                break;                                                                                                 \
            }                                                                                                          \
            break;                                                                                                     \
          default:                                                                                                     \
            break;                                                                                                     \
        }                                                                                                              \
      }                                                                                                                \
      if constexpr (A_LDG_LOOP_COUNT <= ldg_loop) {                                                                    \
        constexpr int real_ldg_loop = ldg_loop - A_LDG_LOOP_COUNT;                                                     \
        switch (real_ldg_loop) {                                                                                       \
          case 0:                                                                                                      \
            switch (ldg_addr_offset) {                                                                                 \
              case 0:                                                                                                  \
                B_global_2_ldg_reg(                                                                                    \
                  B_global_ptr_for_ldg__loop_0__k_0 + ldg_k_offset_x_2N, ldg_addr_offset, real_ldg_loop, ldg_switch);  \
                break;                                                                                                 \
              case 1:                                                                                                  \
                B_global_2_ldg_reg(                                                                                    \
                  B_global_ptr_for_ldg__loop_0__k_1 + ldg_k_offset_x_2N, ldg_addr_offset, real_ldg_loop, ldg_switch);  \
                break;                                                                                                 \
              default:                                                                                                 \
                break;                                                                                                 \
            }                                                                                                          \
            break;                                                                                                     \
          case 1:                                                                                                      \
            switch (ldg_addr_offset) {                                                                                 \
              case 0:                                                                                                  \
                B_global_2_ldg_reg(                                                                                    \
                  B_global_ptr_for_ldg__loop_1__k_0 + ldg_k_offset_x_2N, ldg_addr_offset, real_ldg_loop, ldg_switch);  \
                break;                                                                                                 \
              case 1:                                                                                                  \
                B_global_2_ldg_reg(                                                                                    \
                  B_global_ptr_for_ldg__loop_1__k_1 + ldg_k_offset_x_2N, ldg_addr_offset, real_ldg_loop, ldg_switch);  \
                break;                                                                                                 \
              default:                                                                                                 \
                break;                                                                                                 \
            }                                                                                                          \
            break;                                                                                                     \
          case 2:                                                                                                      \
            switch (ldg_addr_offset) {                                                                                 \
              case 0:                                                                                                  \
                B_global_2_ldg_reg(                                                                                    \
                  B_global_ptr_for_ldg__loop_2__k_0 + ldg_k_offset_x_2N, ldg_addr_offset, real_ldg_loop, ldg_switch);  \
                break;                                                                                                 \
              case 1:                                                                                                  \
                B_global_2_ldg_reg(                                                                                    \
                  B_global_ptr_for_ldg__loop_2__k_1 + ldg_k_offset_x_2N, ldg_addr_offset, real_ldg_loop, ldg_switch);  \
                break;                                                                                                 \
              default:                                                                                                 \
                break;                                                                                                 \
            }                                                                                                          \
            break;                                                                                                     \
          case 3:                                                                                                      \
            switch (ldg_addr_offset) {                                                                                 \
              case 0:                                                                                                  \
                B_global_2_ldg_reg(                                                                                    \
                  B_global_ptr_for_ldg__loop_3__k_0 + ldg_k_offset_x_2N, ldg_addr_offset, real_ldg_loop, ldg_switch);  \
                break;                                                                                                 \
              case 1:                                                                                                  \
                B_global_2_ldg_reg(                                                                                    \
                  B_global_ptr_for_ldg__loop_3__k_1 + ldg_k_offset_x_2N, ldg_addr_offset, real_ldg_loop, ldg_switch);  \
                break;                                                                                                 \
              default:                                                                                                 \
                break;                                                                                                 \
            }                                                                                                          \
            break;                                                                                                     \
          default:                                                                                                     \
            break;                                                                                                     \
        }                                                                                                              \
      }                                                                                                                \
    }                                                                                                                  \
    /* STG */                                                                                                          \
    if constexpr (stg_switch && rank < MxN_GORUP_COUNT) {                                                              \
      constexpr int mg = rank % M_GROUP_COUNT_PER_WARP;                                                                \
      constexpr int ng = rank / M_GROUP_COUNT_PER_WARP;                                                                \
      T casted[4]      = {C_mma_reg[mg][ng][0], C_mma_reg[mg][ng][1], C_mma_reg[mg][ng][2], C_mma_reg[mg][ng][3]};     \
      asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n"                                                   \
                   : "=r"(*(uint32_t*)&C_transposed[mg][ng / 2]._2x4[ng % 2][0])                                       \
                   : "r"(*(uint32_t*)&casted[0]));                                                                     \
      asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n"                                                   \
                   : "=r"(*(uint32_t*)&C_transposed[mg][ng / 2]._2x4[ng % 2][2])                                       \
                   : "r"(*(uint32_t*)&casted[2]));                                                                     \
      shfl_23_and_01(C_transposed[mg][ng / 2]._2x4[ng % 2], 0x1, lane_id);                                             \
      if constexpr ((ng + 1) % 2 == 0) {                                                                               \
        shfl_4567_and_0123(C_transposed[mg][ng / 2]._1x8, 0x2, lane_id);                                               \
        asm volatile("st.global.wt.v4.f32 [%0], {%1, %2, %3, %4};"                                                     \
                     :                                                                                                 \
                     : "l"(C_ptr + mg * 8 * N + (ng - 1) * 16),                                                        \
                       "f"(*(const float*)&C_transposed[mg][ng / 2]._1x8[0]),                                          \
                       "f"(*(const float*)&C_transposed[mg][ng / 2]._1x8[2]),                                          \
                       "f"(*(const float*)&C_transposed[mg][ng / 2]._1x8[4]),                                          \
                       "f"(*(const float*)&C_transposed[mg][ng / 2]._1x8[6])                                           \
                     : "memory");                                                                                      \
      }                                                                                                                \
    }                                                                                                                  \
  }

// FIXME This code is really stupid. Please find a way to optimize it as soon as possible.
#define alternate_ldm_mma_sts_stg_ldg(ldm_sm_buf_index,                                                                \
                                      ldm_reg_buf_index,                                                               \
                                      mma_reg_buf_index,                                                               \
                                      sts_sm_base_index,                                                               \
                                      ldg_k_offset_x2,                                                                 \
                                      ldg_k_offset_x2N,                                                                \
                                      ldm_swh,                                                                         \
                                      mma_swh,                                                                         \
                                      sts_swh,                                                                         \
                                      stg_swh,                                                                         \
                                      ldg_swh,                                                                         \
                                      bar_arrive_swh,                                                                  \
                                      bar_wait_swh)                                                                    \
  static_assert(M_GROUP_COUNT_PER_WARP * N_GROUP_COUNT_PER_WARP <= 32);                                                \
  /* clang-format off */ \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N,  0, ldm_swh, mma_swh, sts_swh, stg_swh, ldg_swh, bar_arrive_swh, bar_wait_swh);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N,  1, ldm_swh, mma_swh, sts_swh, stg_swh, ldg_swh, bar_arrive_swh, bar_wait_swh);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N,  2, ldm_swh, mma_swh, sts_swh, stg_swh, ldg_swh, bar_arrive_swh, bar_wait_swh);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N,  3, ldm_swh, mma_swh, sts_swh, stg_swh, ldg_swh, bar_arrive_swh, bar_wait_swh);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N,  4, ldm_swh, mma_swh, sts_swh, stg_swh, ldg_swh, bar_arrive_swh, bar_wait_swh);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N,  5, ldm_swh, mma_swh, sts_swh, stg_swh, ldg_swh, bar_arrive_swh, bar_wait_swh);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N,  6, ldm_swh, mma_swh, sts_swh, stg_swh, ldg_swh, bar_arrive_swh, bar_wait_swh);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N,  7, ldm_swh, mma_swh, sts_swh, stg_swh, ldg_swh, bar_arrive_swh, bar_wait_swh);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N,  8, ldm_swh, mma_swh, sts_swh, stg_swh, ldg_swh, bar_arrive_swh, bar_wait_swh);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N,  9, ldm_swh, mma_swh, sts_swh, stg_swh, ldg_swh, bar_arrive_swh, bar_wait_swh);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N, 10, ldm_swh, mma_swh, sts_swh, stg_swh, ldg_swh, bar_arrive_swh, bar_wait_swh);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N, 11, ldm_swh, mma_swh, sts_swh, stg_swh, ldg_swh, bar_arrive_swh, bar_wait_swh);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N, 12, ldm_swh, mma_swh, sts_swh, stg_swh, ldg_swh, bar_arrive_swh, bar_wait_swh);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N, 13, ldm_swh, mma_swh, sts_swh, stg_swh, ldg_swh, bar_arrive_swh, bar_wait_swh);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N, 14, ldm_swh, mma_swh, sts_swh, stg_swh, ldg_swh, bar_arrive_swh, bar_wait_swh);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N, 15, ldm_swh, mma_swh, sts_swh, stg_swh, ldg_swh, bar_arrive_swh, bar_wait_swh);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N, 16, ldm_swh, mma_swh, sts_swh, stg_swh, ldg_swh, bar_arrive_swh, bar_wait_swh);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N, 17, ldm_swh, mma_swh, sts_swh, stg_swh, ldg_swh, bar_arrive_swh, bar_wait_swh);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N, 18, ldm_swh, mma_swh, sts_swh, stg_swh, ldg_swh, bar_arrive_swh, bar_wait_swh);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N, 19, ldm_swh, mma_swh, sts_swh, stg_swh, ldg_swh, bar_arrive_swh, bar_wait_swh);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N, 20, ldm_swh, mma_swh, sts_swh, stg_swh, ldg_swh, bar_arrive_swh, bar_wait_swh);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N, 21, ldm_swh, mma_swh, sts_swh, stg_swh, ldg_swh, bar_arrive_swh, bar_wait_swh);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N, 22, ldm_swh, mma_swh, sts_swh, stg_swh, ldg_swh, bar_arrive_swh, bar_wait_swh);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N, 23, ldm_swh, mma_swh, sts_swh, stg_swh, ldg_swh, bar_arrive_swh, bar_wait_swh);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N, 24, ldm_swh, mma_swh, sts_swh, stg_swh, ldg_swh, bar_arrive_swh, bar_wait_swh);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N, 25, ldm_swh, mma_swh, sts_swh, stg_swh, ldg_swh, bar_arrive_swh, bar_wait_swh);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N, 26, ldm_swh, mma_swh, sts_swh, stg_swh, ldg_swh, bar_arrive_swh, bar_wait_swh);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N, 27, ldm_swh, mma_swh, sts_swh, stg_swh, ldg_swh, bar_arrive_swh, bar_wait_swh);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N, 28, ldm_swh, mma_swh, sts_swh, stg_swh, ldg_swh, bar_arrive_swh, bar_wait_swh);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N, 29, ldm_swh, mma_swh, sts_swh, stg_swh, ldg_swh, bar_arrive_swh, bar_wait_swh);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N, 30, ldm_swh, mma_swh, sts_swh, stg_swh, ldg_swh, bar_arrive_swh, bar_wait_swh);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset_x2, ldg_k_offset_x2N, 31, ldm_swh, mma_swh, sts_swh, stg_swh, ldg_swh, bar_arrive_swh, bar_wait_swh);
  /* clang-format on */

  alternate_ldm_mma_sts_stg_ldg(0, 0, 0, 0, 0, 0, false, false, false, false, LDG_SWITCH_ON_EVICT_LAST, false, false);
  alternate_ldm_mma_sts_stg_ldg(0, 0, 0, 0, 0, 0, false, false, true, false, LDG_SWITCH_OFF, false, false);

  __syncthreads();

  int       LDG_SM_BUFFER_INDEX = 0;
  int       k_loop_offset       = LOOP_TILE_K * 2;
  int       k_loop_offset_x2    = k_loop_offset * sizeof(T);
  int       k_loop_offset_x2N   = k_loop_offset_x2 * N;
  const int Nx2xLOOP_TILE_K     = LOOP_TILE_K * 2 * sizeof(T) * N;

  alternate_ldm_mma_sts_stg_ldg(
    LDG_SM_BUFFER_INDEX, MMA_REG_BUFFER_INDEX_0, 0, 0, 0, 0, true, false, false, false, LDG_SWITCH_OFF, false, false);

  alternate_ldm_mma_sts_stg_ldg(0,
                                0,
                                0,
                                0,
                                k_loop_offset_x2,
                                k_loop_offset_x2N,
                                false,
                                false,
                                false,
                                false,
                                LDG_SWITCH_ON_EVICT_LAST,
                                false,
                                false);

  while (k_loop_offset + LOOP_TILE_K * 2 < K) {
    alternate_ldm_mma_sts_stg_ldg(LDG_SM_BUFFER_INDEX + 1,
                                  MMA_REG_BUFFER_INDEX_1,
                                  MMA_REG_BUFFER_INDEX_0,
                                  (LDG_SM_BUFFER_INDEX ^ 2),
                                  0,
                                  0,
                                  true,
                                  true,
                                  true,
                                  false,
                                  LDG_SWITCH_OFF,
                                  true,
                                  false);

    LDG_SM_BUFFER_INDEX ^= 2;
    k_loop_offset += LOOP_TILE_K * 2;
    k_loop_offset_x2 += LOOP_TILE_K * 2 * sizeof(T);
    k_loop_offset_x2N += Nx2xLOOP_TILE_K;

    alternate_ldm_mma_sts_stg_ldg(LDG_SM_BUFFER_INDEX,
                                  MMA_REG_BUFFER_INDEX_0,
                                  MMA_REG_BUFFER_INDEX_1,
                                  0,
                                  k_loop_offset_x2,
                                  k_loop_offset_x2N,
                                  true,
                                  true,
                                  false,
                                  false,
                                  LDG_SWITCH_ON_EVICT_NORMAL,
                                  false,
                                  true);
  }

  {
    alternate_ldm_mma_sts_stg_ldg(LDG_SM_BUFFER_INDEX + 1,
                                  MMA_REG_BUFFER_INDEX_1,
                                  MMA_REG_BUFFER_INDEX_0,
                                  (LDG_SM_BUFFER_INDEX ^ 2),
                                  0,
                                  0,
                                  true,
                                  true,
                                  true,
                                  false,
                                  LDG_SWITCH_OFF,
                                  true,
                                  false);

    LDG_SM_BUFFER_INDEX ^= 2;
    k_loop_offset += LOOP_TILE_K * 2;
    k_loop_offset_x2 += LOOP_TILE_K * 2 * sizeof(T);
    k_loop_offset_x2N += Nx2xLOOP_TILE_K;

    alternate_ldm_mma_sts_stg_ldg(LDG_SM_BUFFER_INDEX,
                                  MMA_REG_BUFFER_INDEX_0,
                                  MMA_REG_BUFFER_INDEX_1,
                                  0,
                                  0,
                                  0,
                                  true,
                                  true,
                                  false,
                                  false,
                                  LDG_SWITCH_OFF,
                                  false,
                                  true);
  }

  {
    alternate_ldm_mma_sts_stg_ldg(LDG_SM_BUFFER_INDEX + 1,
                                  MMA_REG_BUFFER_INDEX_1,
                                  MMA_REG_BUFFER_INDEX_0,
                                  0,
                                  0,
                                  0,
                                  true,
                                  true,
                                  false,
                                  false,
                                  LDG_SWITCH_OFF,
                                  false,
                                  false);

    alternate_ldm_mma_sts_stg_ldg(LDG_SM_BUFFER_INDEX,
                                  MMA_REG_BUFFER_INDEX_0,
                                  MMA_REG_BUFFER_INDEX_1,
                                  0,
                                  0,
                                  0,
                                  false,
                                  true,
                                  false,
                                  true,
                                  LDG_SWITCH_OFF,
                                  false,
                                  false);
  }

#undef A_global_2_ldg_reg
#undef A_ldg_reg_2_sm
#undef B_global_2_ldg_reg
#undef B_ldg_reg_2_sm
#undef sm_2_A_mma_reg
#undef sm_2_B_mma_reg
#undef ldm_mma_sts_stg_ldg
#undef alternate_ldm_mma_sts_stg_ldg
}

template<typename T, int BLOCK_TILE_M, int BLOCK_TILE_N, int WARP_TILE_M, int WARP_TILE_N>
__global__ void
fp16_mma_m16n8k16_ldmatrix_trans__overlap_global_2_sm__quadra_buffer__reduce_instructions__reorder_instructions__overlap_reg_2_global__stg_memory_coalecesing__overlap_sts__stg_wt__reduce_IMAD_AND_LEA__m16n8k8(
  const T* A, const T* B, T* C, int M, int N, int K)
{
  constexpr int WARP_COUNT   = BLOCK_TILE_M / WARP_TILE_M * BLOCK_TILE_N / WARP_TILE_N;
  constexpr int THREAD_COUNT = WARP_COUNT * 32;

  constexpr int LOOP_TILE_K         = 16;
  constexpr int LDG_SM_BUFFER_SIZE  = 4;
  constexpr int LDG_REG_BUFFER_SIZE = 2;

  constexpr int A_sm_dim0 = LDG_SM_BUFFER_SIZE;
  constexpr int A_sm_dim1 = 2;
  constexpr int A_sm_dim2 = BLOCK_TILE_M;
  constexpr int A_sm_dim3 = LOOP_TILE_K / 2;
  constexpr int B_sm_dim0 = LDG_SM_BUFFER_SIZE;
  constexpr int B_sm_dim1 = LOOP_TILE_K / 8;
  constexpr int B_sm_dim2 = BLOCK_TILE_N / 8;
  constexpr int B_sm_dim3 = 64;

  // The 64 elements of type T in each 8x8 matrix are stored consecutively in a single layer of shared memory.
  __shared__ union {
    struct {
      T A_sm[A_sm_dim0 * A_sm_dim1 * A_sm_dim2 * A_sm_dim3];
      T B_sm[B_sm_dim0 * B_sm_dim1 * B_sm_dim2 * B_sm_dim3];
    } mma;
    static_assert(WARP_TILE_N % 16 == 0);
    T result[WARP_COUNT][WARP_TILE_N / 16][WARP_TILE_M][16];
  } data;

  static_assert(BLOCK_TILE_M * LOOP_TILE_K % THREAD_COUNT == 0);
  static_assert(BLOCK_TILE_M * LOOP_TILE_K / THREAD_COUNT % 8 == 0);
  constexpr int A_LDG_COUNT_PER_THREAD = BLOCK_TILE_M * LOOP_TILE_K / THREAD_COUNT;
  constexpr int A_LDG_LOOP_COUNT       = A_LDG_COUNT_PER_THREAD / 8;
  // clang-format off
  // This is the thread layout of the same warp that loads matrix A, where each thread reads M1xK8 elements of type T at a
  // loop iteration.
  // T0  T16
  // T1  T17
  // T2  T18
  // ... ...
  // T14 T30
  // T15 T31
  // clang-format on
  float A_ldg_reg[LDG_REG_BUFFER_SIZE][A_LDG_LOOP_COUNT][4];

  static_assert(BLOCK_TILE_N * LOOP_TILE_K % THREAD_COUNT == 0);
  static_assert(BLOCK_TILE_N * LOOP_TILE_K / THREAD_COUNT % 8 == 0);
  constexpr int B_LDG_COUNT_PER_THREAD = BLOCK_TILE_N * LOOP_TILE_K / THREAD_COUNT;
  constexpr int B_LDG_LOOP_COUNT       = B_LDG_COUNT_PER_THREAD / 8;
  // clang-format off
  // This is the thread layout of the same warp that loads matrix B, where each thread reads K1xN8 elements of type T at a
  // loop iteration.
  // T0  T16
  // T1  T17
  // T2  T18
  // ... ...
  // T14 T30
  // T15 T31
  // clang-format on
  float B_ldg_reg[LDG_REG_BUFFER_SIZE][B_LDG_LOOP_COUNT][4];

  const int m_block_offset = blockIdx.y * BLOCK_TILE_M;
  const int n_block_offset = blockIdx.x * BLOCK_TILE_N;

  const int warp_id                 = threadIdx.x / 32;
  const int lane_id                 = threadIdx.x % 32;
  const int transposed_lane_id_mask = (lane_id / 8 == 0 || lane_id / 8 == 3) ? 0x00 : 0x18;
  const int transposed_lane_id      = lane_id ^ transposed_lane_id_mask;

  constexpr int M_MMA_WARP_COUNT       = BLOCK_TILE_M / WARP_TILE_M;
  constexpr int M_GROUP_COUNT_PER_WARP = WARP_TILE_M / 8;
  constexpr int N_GROUP_COUNT_PER_WARP = WARP_TILE_N / 16;

  constexpr int MMA_REG_BUFFER_SIZE    = 2;
  constexpr int MMA_REG_BUFFER_INDEX_0 = 0;
  constexpr int MMA_REG_BUFFER_INDEX_1 = 1;
  T             A_mma_reg[MMA_REG_BUFFER_SIZE][M_GROUP_COUNT_PER_WARP][4];
  T             B_mma_reg[MMA_REG_BUFFER_SIZE][N_GROUP_COUNT_PER_WARP][8];
  float         C_mma_reg[M_GROUP_COUNT_PER_WARP][N_GROUP_COUNT_PER_WARP][4] = {0};
  static_assert(N_GROUP_COUNT_PER_WARP % 2 == 0);
  union _2x4_or_1x8 {
    T _2x4[2][4];
    T _1x8[8];
  };
  _2x4_or_1x8 C_transposed[M_GROUP_COUNT_PER_WARP][N_GROUP_COUNT_PER_WARP / 2];

  const int m_warp_offset = warp_id % M_MMA_WARP_COUNT * WARP_TILE_M;
  const int n_warp_offset = warp_id / M_MMA_WARP_COUNT * WARP_TILE_N;

  const int A_ldg_reg_2_A_sm_partial_offset =
    lane_id / 16 * A_sm_dim2 * A_sm_dim3 + (warp_id * 16 + lane_id % 16) * A_sm_dim3;

  const int B_ldg_reg_2_B_sm_partial_offset =
    (lane_id % 16) / 8 * B_sm_dim2 * B_sm_dim3 + (lane_id % 16) % 8 * 8 + (warp_id * 2 + lane_id / 16) * B_sm_dim3;

  const int A_global_partial_offset = (m_block_offset + warp_id * 16 + lane_id % 16) * K + lane_id / 16 * 8;
  const int B_global_partial_offset = lane_id % 16 * N + n_block_offset + warp_id * 16 + lane_id / 16 * 8;

  const T* A_global_ptr_for_ldg = &A[A_global_partial_offset];
  const T* B_global_ptr_for_ldg = &B[B_global_partial_offset];

  static_assert(A_LDG_LOOP_COUNT <= 4);
  static_assert(B_LDG_LOOP_COUNT <= 4);
  static_assert(LDG_REG_BUFFER_SIZE <= 2);
  const uint64_t A_global_ptr_for_ldg__loop_0__k_0 = (uint64_t)(A_global_ptr_for_ldg + 0 * WARP_COUNT * 16 * K + 0);
  const uint64_t A_global_ptr_for_ldg__loop_1__k_0 = (uint64_t)(A_global_ptr_for_ldg + 1 * WARP_COUNT * 16 * K + 0);
  const uint64_t A_global_ptr_for_ldg__loop_2__k_0 = (uint64_t)(A_global_ptr_for_ldg + 2 * WARP_COUNT * 16 * K + 0);
  const uint64_t A_global_ptr_for_ldg__loop_3__k_0 = (uint64_t)(A_global_ptr_for_ldg + 3 * WARP_COUNT * 16 * K + 0);
  const uint64_t A_global_ptr_for_ldg__loop_0__k_1 =
    (uint64_t)(A_global_ptr_for_ldg + 0 * WARP_COUNT * 16 * K + LOOP_TILE_K);
  const uint64_t A_global_ptr_for_ldg__loop_1__k_1 =
    (uint64_t)(A_global_ptr_for_ldg + 1 * WARP_COUNT * 16 * K + LOOP_TILE_K);
  const uint64_t A_global_ptr_for_ldg__loop_2__k_1 =
    (uint64_t)(A_global_ptr_for_ldg + 2 * WARP_COUNT * 16 * K + LOOP_TILE_K);
  const uint64_t A_global_ptr_for_ldg__loop_3__k_1 =
    (uint64_t)(A_global_ptr_for_ldg + 3 * WARP_COUNT * 16 * K + LOOP_TILE_K);
  const uint64_t B_global_ptr_for_ldg__loop_0__k_0 = (uint64_t)(B_global_ptr_for_ldg + 0 * WARP_COUNT * 16 + 0);
  const uint64_t B_global_ptr_for_ldg__loop_1__k_0 = (uint64_t)(B_global_ptr_for_ldg + 1 * WARP_COUNT * 16 + 0);
  const uint64_t B_global_ptr_for_ldg__loop_2__k_0 = (uint64_t)(B_global_ptr_for_ldg + 2 * WARP_COUNT * 16 + 0);
  const uint64_t B_global_ptr_for_ldg__loop_3__k_0 = (uint64_t)(B_global_ptr_for_ldg + 3 * WARP_COUNT * 16 + 0);
  const uint64_t B_global_ptr_for_ldg__loop_0__k_1 =
    (uint64_t)(B_global_ptr_for_ldg + 0 * WARP_COUNT * 16 + LOOP_TILE_K * N);
  const uint64_t B_global_ptr_for_ldg__loop_1__k_1 =
    (uint64_t)(B_global_ptr_for_ldg + 1 * WARP_COUNT * 16 + LOOP_TILE_K * N);
  const uint64_t B_global_ptr_for_ldg__loop_2__k_1 =
    (uint64_t)(B_global_ptr_for_ldg + 2 * WARP_COUNT * 16 + LOOP_TILE_K * N);
  const uint64_t B_global_ptr_for_ldg__loop_3__k_1 =
    (uint64_t)(B_global_ptr_for_ldg + 3 * WARP_COUNT * 16 + LOOP_TILE_K * N);

  const T* A_sm_ptr_for_ldg = &data.mma.A_sm[A_ldg_reg_2_A_sm_partial_offset];
  const T* B_sm_ptr_for_ldg = &data.mma.B_sm[B_ldg_reg_2_B_sm_partial_offset];

  const int A_sm_2_A_mma_reg_partial_offset =
    lane_id % 16 / 8 * A_sm_dim2 * A_sm_dim3 + (m_warp_offset + lane_id % 8) * A_sm_dim3;

  const int B_sm_2_B_mma_reg_partial_offset = transposed_lane_id % 16 / 8 * B_sm_dim2 * B_sm_dim3
                                              + (n_warp_offset + transposed_lane_id / 16 * 8) / 8 * B_sm_dim3
                                              + transposed_lane_id % 8 * 8;

  const T* A_sm_ptr_for_mma = &data.mma.A_sm[A_sm_2_A_mma_reg_partial_offset + lane_id / 16 * 8 * A_sm_dim3];
  const T* B_sm_ptr_for_mma = &data.mma.B_sm[B_sm_2_B_mma_reg_partial_offset];

  enum {
    LDG_SWITCH_OFF             = 0,
    LDG_SWITCH_ON_EVICT_NORMAL = 1,
    LDG_SWITCH_ON_EVICT_LAST   = 2,
  };

#define A_global_2_ldg_reg(A_global_ptr, ldg_reg_buffer_index, loop, cache_policy)                                     \
  {                                                                                                                    \
    /* const int m = (loop * WARP_COUNT + warp_id) * 16 + lane_id % 16; */                                             \
    /* const int k = lane_id / 16 * 8; */                                                                              \
    if constexpr (cache_policy == LDG_SWITCH_ON_EVICT_LAST) {                                                          \
      FETCH_FLOAT4_EVICT_LAST_AND_PREFETCH_256B_WITH_SRC_PTR(A_ldg_reg[ldg_reg_buffer_index][loop], A_global_ptr);     \
    }                                                                                                                  \
    else {                                                                                                             \
      FETCH_FLOAT4_WITH_PTR(&A_ldg_reg[ldg_reg_buffer_index][loop][0], A_global_ptr);                                  \
    }                                                                                                                  \
  }

#define B_global_2_ldg_reg(B_global_ptr, ldg_reg_buffer_index, loop, cache_policy)                                     \
  {                                                                                                                    \
    /* const int k = lane_id % 16;                                           */                                        \
    /* const int n = (loop * WARP_COUNT + warp_id) * 16 + lane_id / 16 * 8;  */                                        \
    if constexpr (cache_policy == LDG_SWITCH_ON_EVICT_LAST && false) {                                                 \
      FETCH_FLOAT4_EVICT_LAST_AND_PREFETCH_256B_WITH_SRC_PTR(B_ldg_reg[ldg_reg_buffer_index][loop], B_global_ptr);     \
    }                                                                                                                  \
    else {                                                                                                             \
      FETCH_FLOAT4_WITH_PTR(&B_ldg_reg[ldg_reg_buffer_index][loop][0], B_global_ptr);                                  \
    }                                                                                                                  \
  }

#define A_ldg_reg_2_sm(ldg_sm_buffer_index, ldg_reg_buffer_index, loop)                                                \
  {                                                                                                                    \
    /* const int m = (loop * WARP_COUNT + warp_id) * 16 + lane_id % 16; */                                             \
    /* const int k = lane_id / 16 * 8;  */                                                                             \
    STORE_FLOAT4_WITH_PTR(A_sm_ptr_for_ldg + loop * WARP_COUNT * 16 * A_sm_dim3                                        \
                            + (ldg_sm_buffer_index) * A_sm_dim1 * A_sm_dim2 * A_sm_dim3,                               \
                          &A_ldg_reg[ldg_reg_buffer_index][loop][0]);                                                  \
  }

#define B_ldg_reg_2_sm(ldg_sm_buffer_index, ldg_reg_buffer_index, loop)                                                \
  {                                                                                                                    \
    /*const int k = lane_id % 16; */                                                                                   \
    /*const int n = (loop * WARP_COUNT + warp_id) * 16 + lane_id / 16 * 8;*/                                           \
    STORE_FLOAT4_WITH_PTR(B_sm_ptr_for_ldg + (loop) * WARP_COUNT * 2 * B_sm_dim3                                       \
                            + (ldg_sm_buffer_index) * B_sm_dim1 * B_sm_dim2 * B_sm_dim3,                               \
                          &B_ldg_reg[ldg_reg_buffer_index][loop][0]);                                                  \
  }

#define sm_2_A_mma_reg(ldg_sm_buffer_index, mma_reg_buffer_index, group)                                               \
  if constexpr (M_GROUP_COUNT_PER_WARP == 1) {                                                                         \
    /* for (int group = 0; group < M_GROUP_COUNT_PER_WARP; ++group) */ {                                               \
      uint32_t src = __cvta_generic_to_shared(A_sm_ptr_for_mma + (group) * 8 * A_sm_dim3                               \
                                              + (ldg_sm_buffer_index) * A_sm_dim1 * A_sm_dim2 * A_sm_dim3);            \
      asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"                                          \
                   : "=r"(*(uint32_t*)&A_mma_reg[mma_reg_buffer_index][group][0]),                                     \
                     "=r"(*(uint32_t*)&A_mma_reg[mma_reg_buffer_index][group][2])                                      \
                   : "r"(src));                                                                                        \
    }                                                                                                                  \
  }                                                                                                                    \
  else if constexpr (M_GROUP_COUNT_PER_WARP % 2 == 0) {                                                                \
    /*for (int group = 0; group < M_GROUP_COUNT_PER_WARP; group += 2) */ {                                             \
      uint32_t src = __cvta_generic_to_shared(A_sm_ptr_for_mma + (group) * 8 * A_sm_dim3                               \
                                              + (ldg_sm_buffer_index) * A_sm_dim1 * A_sm_dim2 * A_sm_dim3);            \
      asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"                                  \
                   : "=r"(*(uint32_t*)&A_mma_reg[mma_reg_buffer_index][group][0]),                                     \
                     "=r"(*(uint32_t*)&A_mma_reg[mma_reg_buffer_index][group][2]),                                     \
                     "=r"(*(uint32_t*)&A_mma_reg[mma_reg_buffer_index][group + 1][0]),                                 \
                     "=r"(*(uint32_t*)&A_mma_reg[mma_reg_buffer_index][group + 1][2])                                  \
                   : "r"(src));                                                                                        \
    }                                                                                                                  \
  }                                                                                                                    \
  else {                                                                                                               \
    static_assert(M_GROUP_COUNT_PER_WARP == 1 || M_GROUP_COUNT_PER_WARP % 2 == 0);                                     \
  }

#define sm_2_B_mma_reg(ldg_sm_buffer_index, mma_reg_buffer_index, group)                                               \
  /* for (int group = 0; group < N_GROUP_COUNT_PER_WARP; ++group) */ {                                                 \
    uint32_t src = __cvta_generic_to_shared(B_sm_ptr_for_mma + (group) * 2 * B_sm_dim3                                 \
                                            + (ldg_sm_buffer_index) * B_sm_dim1 * B_sm_dim2 * B_sm_dim3);              \
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"                              \
                 : "=r"(*(uint32_t*)&B_mma_reg[mma_reg_buffer_index][group][0]),                                       \
                   "=r"(*(uint32_t*)&B_mma_reg[mma_reg_buffer_index][group][2]),                                       \
                   "=r"(*(uint32_t*)&B_mma_reg[mma_reg_buffer_index][group][4]),                                       \
                   "=r"(*(uint32_t*)&B_mma_reg[mma_reg_buffer_index][group][6])                                        \
                 : "r"(src));                                                                                          \
  }

#define mma_m16n8k8_row_col(d, a, b, c)                                                                                \
  {                                                                                                                    \
    uint32_t const* A = reinterpret_cast<uint32_t const*>(&a);                                                         \
    uint32_t const* B = reinterpret_cast<uint32_t const*>(&b);                                                         \
    float const*    C = reinterpret_cast<float const*>(&c);                                                            \
    float*          D = reinterpret_cast<float*>(&d);                                                                  \
    if constexpr (std::is_same<T, half>::value) {                                                                      \
      asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32  {%0,%1,%2,%3}, "                                 \
                   "{%4,%5}, {%6}, {%7,%8,%9,%10};\n"                                                                  \
                   : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])                                                    \
                   : "r"(A[0]), "r"(A[1]), "r"(B[0]), "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));                     \
    }                                                                                                                  \
    else if constexpr (std::is_same<T, __nv_bfloat16>::value) {                                                        \
      asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32  {%0,%1,%2,%3}, "                               \
                   "{%4,%5}, {%6}, {%7,%8,%9,%10};\n"                                                                  \
                   : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])                                                    \
                   : "r"(A[0]), "r"(A[1]), "r"(B[0]), "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));                     \
    }                                                                                                                  \
    else {                                                                                                             \
      static_assert(std::is_same<T, half>::value == false && std::is_same<T, __nv_bfloat16>::value == false);          \
    }                                                                                                                  \
  }

  T* C_ptr = &C[(m_block_offset + m_warp_offset + lane_id / 4) * N + n_block_offset + n_warp_offset + lane_id % 4 * 8];

#define ldm_mma_sts_stg_ldg(ldm_sm_buffer_index,                                                                       \
                            ldm_reg_buffer_index,                                                                      \
                            mma_reg_buffer_index,                                                                      \
                            sts_sm_base_index,                                                                         \
                            ldg_k_offset,                                                                              \
                            rank,                                                                                      \
                            ldm_switch,                                                                                \
                            mma_switch,                                                                                \
                            sts_switch,                                                                                \
                            stg_switch,                                                                                \
                            ldg_switch)                                                                                \
  {                                                                                                                    \
    /* STS */                                                                                                          \
    constexpr int MxN_GORUP_COUNT   = M_GROUP_COUNT_PER_WARP * N_GROUP_COUNT_PER_WARP;                                 \
    constexpr int MxNx2_GORUP_COUNT = M_GROUP_COUNT_PER_WARP * N_GROUP_COUNT_PER_WARP * 2;                             \
    constexpr int STS_COUNT         = LDG_REG_BUFFER_SIZE * (A_LDG_LOOP_COUNT + B_LDG_LOOP_COUNT);                     \
    static_assert(STS_COUNT <= MxNx2_GORUP_COUNT);                                                                     \
    if constexpr (sts_switch && MxNx2_GORUP_COUNT - STS_COUNT <= rank && rank < MxNx2_GORUP_COUNT) {                   \
      constexpr int sts_rank        = rank - (MxNx2_GORUP_COUNT - STS_COUNT);                                          \
      constexpr int sts_addr_offset = sts_rank / (A_LDG_LOOP_COUNT + B_LDG_LOOP_COUNT);                                \
      constexpr int sts_loop        = sts_rank % (A_LDG_LOOP_COUNT + B_LDG_LOOP_COUNT);                                \
      if constexpr (sts_loop < A_LDG_LOOP_COUNT) {                                                                     \
        A_ldg_reg_2_sm(sts_sm_base_index + sts_addr_offset, sts_addr_offset, sts_loop);                                \
      }                                                                                                                \
      if constexpr (A_LDG_LOOP_COUNT <= sts_loop) {                                                                    \
        B_ldg_reg_2_sm(sts_sm_base_index + sts_addr_offset, sts_addr_offset, sts_loop - A_LDG_LOOP_COUNT);             \
      }                                                                                                                \
    }                                                                                                                  \
    /* MMA */                                                                                                          \
    if constexpr (mma_switch && rank < MxNx2_GORUP_COUNT) {                                                            \
      constexpr int offset = rank / MxN_GORUP_COUNT;                                                                   \
      constexpr int mg     = rank % (M_GROUP_COUNT_PER_WARP * N_GROUP_COUNT_PER_WARP) % M_GROUP_COUNT_PER_WARP;        \
      constexpr int ng     = rank % (M_GROUP_COUNT_PER_WARP * N_GROUP_COUNT_PER_WARP) / M_GROUP_COUNT_PER_WARP;        \
      mma_m16n8k8_row_col(C_mma_reg[mg][ng],                                                                           \
                          B_mma_reg[mma_reg_buffer_index][ng][4 * offset],                                             \
                          A_mma_reg[mma_reg_buffer_index][mg][2 * offset],                                             \
                          C_mma_reg[mg][ng]);                                                                          \
    }                                                                                                                  \
    /* LDM */                                                                                                          \
    static_assert(M_GROUP_COUNT_PER_WARP == 1 || M_GROUP_COUNT_PER_WARP % 2 == 0);                                     \
    static_assert((M_GROUP_COUNT_PER_WARP + 1) / 2 + N_GROUP_COUNT_PER_WARP < MxNx2_GORUP_COUNT);                      \
    if constexpr (ldm_switch && rank < (M_GROUP_COUNT_PER_WARP + 1) / 2) {                                             \
      sm_2_A_mma_reg(ldm_sm_buffer_index, ldm_reg_buffer_index, rank * 2);                                             \
    }                                                                                                                  \
    if constexpr (ldm_switch && (M_GROUP_COUNT_PER_WARP + 1) / 2 <= rank                                               \
                  && rank < (M_GROUP_COUNT_PER_WARP + 1) / 2 + N_GROUP_COUNT_PER_WARP) {                               \
      sm_2_B_mma_reg(ldm_sm_buffer_index, ldm_reg_buffer_index, rank - (M_GROUP_COUNT_PER_WARP + 1) / 2);              \
    }                                                                                                                  \
    /* LDG */                                                                                                          \
    if constexpr (ldg_switch && rank < LDG_REG_BUFFER_SIZE * (A_LDG_LOOP_COUNT + B_LDG_LOOP_COUNT)) {                  \
      constexpr int ldg_addr_offset = rank / (A_LDG_LOOP_COUNT + B_LDG_LOOP_COUNT);                                    \
      constexpr int ldg_loop        = rank % (A_LDG_LOOP_COUNT + B_LDG_LOOP_COUNT);                                    \
      static_assert(ldg_addr_offset <= 2);                                                                             \
      if constexpr (ldg_loop < A_LDG_LOOP_COUNT) {                                                                     \
        switch (ldg_loop) {                                                                                            \
          case 0:                                                                                                      \
            switch (ldg_addr_offset) {                                                                                 \
              case 0:                                                                                                  \
                A_global_2_ldg_reg(A_global_ptr_for_ldg__loop_0__k_0 + ldg_k_offset * sizeof(T),                       \
                                   ldg_addr_offset,                                                                    \
                                   ldg_loop,                                                                           \
                                   ldg_switch);                                                                        \
                break;                                                                                                 \
              case 1:                                                                                                  \
                A_global_2_ldg_reg(A_global_ptr_for_ldg__loop_0__k_1 + ldg_k_offset * sizeof(T),                       \
                                   ldg_addr_offset,                                                                    \
                                   ldg_loop,                                                                           \
                                   ldg_switch);                                                                        \
                break;                                                                                                 \
              default:                                                                                                 \
                break;                                                                                                 \
            }                                                                                                          \
            break;                                                                                                     \
          case 1:                                                                                                      \
            switch (ldg_addr_offset) {                                                                                 \
              case 0:                                                                                                  \
                A_global_2_ldg_reg(A_global_ptr_for_ldg__loop_1__k_0 + ldg_k_offset * sizeof(T),                       \
                                   ldg_addr_offset,                                                                    \
                                   ldg_loop,                                                                           \
                                   ldg_switch);                                                                        \
                break;                                                                                                 \
              case 1:                                                                                                  \
                A_global_2_ldg_reg(A_global_ptr_for_ldg__loop_1__k_1 + ldg_k_offset * sizeof(T),                       \
                                   ldg_addr_offset,                                                                    \
                                   ldg_loop,                                                                           \
                                   ldg_switch);                                                                        \
                break;                                                                                                 \
              default:                                                                                                 \
                break;                                                                                                 \
            }                                                                                                          \
            break;                                                                                                     \
          case 2:                                                                                                      \
            switch (ldg_addr_offset) {                                                                                 \
              case 0:                                                                                                  \
                A_global_2_ldg_reg(A_global_ptr_for_ldg__loop_2__k_0 + ldg_k_offset * sizeof(T),                       \
                                   ldg_addr_offset,                                                                    \
                                   ldg_loop,                                                                           \
                                   ldg_switch);                                                                        \
                break;                                                                                                 \
              case 1:                                                                                                  \
                A_global_2_ldg_reg(A_global_ptr_for_ldg__loop_2__k_1 + ldg_k_offset * sizeof(T),                       \
                                   ldg_addr_offset,                                                                    \
                                   ldg_loop,                                                                           \
                                   ldg_switch);                                                                        \
                break;                                                                                                 \
              default:                                                                                                 \
                break;                                                                                                 \
            }                                                                                                          \
            break;                                                                                                     \
          case 3:                                                                                                      \
            switch (ldg_addr_offset) {                                                                                 \
              case 0:                                                                                                  \
                A_global_2_ldg_reg(A_global_ptr_for_ldg__loop_3__k_0 + ldg_k_offset * sizeof(T),                       \
                                   ldg_addr_offset,                                                                    \
                                   ldg_loop,                                                                           \
                                   ldg_switch);                                                                        \
                break;                                                                                                 \
              case 1:                                                                                                  \
                A_global_2_ldg_reg(A_global_ptr_for_ldg__loop_3__k_1 + ldg_k_offset * sizeof(T),                       \
                                   ldg_addr_offset,                                                                    \
                                   ldg_loop,                                                                           \
                                   ldg_switch);                                                                        \
                break;                                                                                                 \
              default:                                                                                                 \
                break;                                                                                                 \
            }                                                                                                          \
            break;                                                                                                     \
          default:                                                                                                     \
            break;                                                                                                     \
        }                                                                                                              \
      }                                                                                                                \
      if constexpr (A_LDG_LOOP_COUNT <= ldg_loop) {                                                                    \
        constexpr int real_ldg_loop = ldg_loop - A_LDG_LOOP_COUNT;                                                     \
        switch (real_ldg_loop) {                                                                                       \
          case 0:                                                                                                      \
            switch (ldg_addr_offset) {                                                                                 \
              case 0:                                                                                                  \
                B_global_2_ldg_reg(B_global_ptr_for_ldg__loop_0__k_0 + ldg_k_offset * N * sizeof(T),                   \
                                   ldg_addr_offset,                                                                    \
                                   real_ldg_loop,                                                                      \
                                   ldg_switch);                                                                        \
                break;                                                                                                 \
              case 1:                                                                                                  \
                B_global_2_ldg_reg(B_global_ptr_for_ldg__loop_0__k_1 + ldg_k_offset * N * sizeof(T),                   \
                                   ldg_addr_offset,                                                                    \
                                   real_ldg_loop,                                                                      \
                                   ldg_switch);                                                                        \
                break;                                                                                                 \
              default:                                                                                                 \
                break;                                                                                                 \
            }                                                                                                          \
            break;                                                                                                     \
          case 1:                                                                                                      \
            switch (ldg_addr_offset) {                                                                                 \
              case 0:                                                                                                  \
                B_global_2_ldg_reg(B_global_ptr_for_ldg__loop_1__k_0 + ldg_k_offset * N * sizeof(T),                   \
                                   ldg_addr_offset,                                                                    \
                                   real_ldg_loop,                                                                      \
                                   ldg_switch);                                                                        \
                break;                                                                                                 \
              case 1:                                                                                                  \
                B_global_2_ldg_reg(B_global_ptr_for_ldg__loop_1__k_1 + ldg_k_offset * N * sizeof(T),                   \
                                   ldg_addr_offset,                                                                    \
                                   real_ldg_loop,                                                                      \
                                   ldg_switch);                                                                        \
                break;                                                                                                 \
              default:                                                                                                 \
                break;                                                                                                 \
            }                                                                                                          \
            break;                                                                                                     \
          case 2:                                                                                                      \
            switch (ldg_addr_offset) {                                                                                 \
              case 0:                                                                                                  \
                B_global_2_ldg_reg(B_global_ptr_for_ldg__loop_2__k_0 + ldg_k_offset * N * sizeof(T),                   \
                                   ldg_addr_offset,                                                                    \
                                   real_ldg_loop,                                                                      \
                                   ldg_switch);                                                                        \
                break;                                                                                                 \
              case 1:                                                                                                  \
                B_global_2_ldg_reg(B_global_ptr_for_ldg__loop_2__k_1 + ldg_k_offset * N * sizeof(T),                   \
                                   ldg_addr_offset,                                                                    \
                                   real_ldg_loop,                                                                      \
                                   ldg_switch);                                                                        \
                break;                                                                                                 \
              default:                                                                                                 \
                break;                                                                                                 \
            }                                                                                                          \
            break;                                                                                                     \
          case 3:                                                                                                      \
            switch (ldg_addr_offset) {                                                                                 \
              case 0:                                                                                                  \
                B_global_2_ldg_reg(B_global_ptr_for_ldg__loop_3__k_0 + ldg_k_offset * N * sizeof(T),                   \
                                   ldg_addr_offset,                                                                    \
                                   real_ldg_loop,                                                                      \
                                   ldg_switch);                                                                        \
                break;                                                                                                 \
              case 1:                                                                                                  \
                B_global_2_ldg_reg(B_global_ptr_for_ldg__loop_3__k_1 + ldg_k_offset * N * sizeof(T),                   \
                                   ldg_addr_offset,                                                                    \
                                   real_ldg_loop,                                                                      \
                                   ldg_switch);                                                                        \
                break;                                                                                                 \
              default:                                                                                                 \
                break;                                                                                                 \
            }                                                                                                          \
            break;                                                                                                     \
          default:                                                                                                     \
            break;                                                                                                     \
        }                                                                                                              \
      }                                                                                                                \
    }                                                                                                                  \
    /* STG */                                                                                                          \
    if constexpr (stg_switch && rank < MxNx2_GORUP_COUNT) {                                                            \
      constexpr int offset = rank / (M_GROUP_COUNT_PER_WARP * N_GROUP_COUNT_PER_WARP);                                 \
      if constexpr (offset == 1) {                                                                                     \
        constexpr int mg = rank % (M_GROUP_COUNT_PER_WARP * N_GROUP_COUNT_PER_WARP) % M_GROUP_COUNT_PER_WARP;          \
        constexpr int ng = rank % (M_GROUP_COUNT_PER_WARP * N_GROUP_COUNT_PER_WARP) / M_GROUP_COUNT_PER_WARP;          \
        T casted[4]      = {C_mma_reg[mg][ng][0], C_mma_reg[mg][ng][1], C_mma_reg[mg][ng][2], C_mma_reg[mg][ng][3]};   \
        asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n"                                                 \
                     : "=r"(*(uint32_t*)&C_transposed[mg][ng / 2]._2x4[ng % 2][0])                                     \
                     : "r"(*(uint32_t*)&casted[0]));                                                                   \
        asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n"                                                 \
                     : "=r"(*(uint32_t*)&C_transposed[mg][ng / 2]._2x4[ng % 2][2])                                     \
                     : "r"(*(uint32_t*)&casted[2]));                                                                   \
        shfl_23_and_01(C_transposed[mg][ng / 2]._2x4[ng % 2], 0x1, lane_id);                                           \
        if constexpr ((ng + 1) % 2 == 0) {                                                                             \
          shfl_4567_and_0123(C_transposed[mg][ng / 2]._1x8, 0x2, lane_id);                                             \
          asm volatile("st.global.wt.v4.f32 [%0], {%1, %2, %3, %4};"                                                   \
                       :                                                                                               \
                       : "l"(C_ptr + mg * 8 * N + (ng - 1) * 16),                                                      \
                         "f"(*(const float*)&C_transposed[mg][ng / 2]._1x8[0]),                                        \
                         "f"(*(const float*)&C_transposed[mg][ng / 2]._1x8[2]),                                        \
                         "f"(*(const float*)&C_transposed[mg][ng / 2]._1x8[4]),                                        \
                         "f"(*(const float*)&C_transposed[mg][ng / 2]._1x8[6])                                         \
                       : "memory");                                                                                    \
        }                                                                                                              \
      }                                                                                                                \
    }                                                                                                                  \
  }

// FIXME This code is really stupid. Please find a way to optimize it as soon as possible.
#define alternate_ldm_mma_sts_stg_ldg(ldm_sm_buf_index,                                                                \
                                      ldm_reg_buf_index,                                                               \
                                      mma_reg_buf_index,                                                               \
                                      sts_sm_base_index,                                                               \
                                      ldg_k_offset,                                                                    \
                                      ldm_switch,                                                                      \
                                      mma_switch,                                                                      \
                                      sts_switch,                                                                      \
                                      stg_switch,                                                                      \
                                      ldg_switch)                                                                      \
  static_assert(M_GROUP_COUNT_PER_WARP * N_GROUP_COUNT_PER_WARP * 2 <= 64);                                            \
  /* clang-format off */ \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset,  0, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset,  1, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset,  2, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset,  3, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset,  4, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset,  5, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset,  6, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset,  7, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset,  8, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset,  9, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset, 10, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset, 11, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset, 12, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset, 13, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset, 14, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset, 15, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset, 16, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset, 17, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset, 18, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset, 19, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset, 20, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset, 21, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset, 22, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset, 23, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset, 24, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset, 25, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset, 26, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset, 27, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset, 28, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset, 29, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset, 30, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset, 31, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset, 32, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset, 33, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset, 34, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset, 35, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset, 36, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset, 37, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset, 38, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset, 39, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset, 40, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset, 41, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset, 42, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset, 43, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset, 44, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset, 45, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset, 46, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset, 47, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset, 48, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset, 49, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset, 50, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset, 51, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset, 52, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset, 53, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset, 54, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset, 55, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset, 56, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset, 57, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset, 58, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset, 59, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset, 60, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset, 61, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset, 62, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg_ldg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset, 63, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);
  /* clang-format on */

  alternate_ldm_mma_sts_stg_ldg(0, 0, 0, 0, 0, false, false, false, false, LDG_SWITCH_ON_EVICT_LAST);
  alternate_ldm_mma_sts_stg_ldg(0, 0, 0, 0, 0, false, false, true, false, LDG_SWITCH_OFF);

  __syncthreads();

  int LDG_SM_BUFFER_INDEX = 0;
  int k_loop_offset       = LOOP_TILE_K * 2;

  alternate_ldm_mma_sts_stg_ldg(
    LDG_SM_BUFFER_INDEX, MMA_REG_BUFFER_INDEX_0, 0, 0, 0, true, false, false, false, LDG_SWITCH_OFF);

  {
    alternate_ldm_mma_sts_stg_ldg(LDG_SM_BUFFER_INDEX + 1,
                                  MMA_REG_BUFFER_INDEX_1,
                                  MMA_REG_BUFFER_INDEX_0,
                                  (LDG_SM_BUFFER_INDEX ^ 2),
                                  k_loop_offset,
                                  true,
                                  true,
                                  true,
                                  false,
                                  LDG_SWITCH_ON_EVICT_NORMAL);

    LDG_SM_BUFFER_INDEX ^= 2;
    k_loop_offset += LOOP_TILE_K * 2;

    __syncthreads();

    alternate_ldm_mma_sts_stg_ldg(LDG_SM_BUFFER_INDEX,
                                  MMA_REG_BUFFER_INDEX_0,
                                  MMA_REG_BUFFER_INDEX_1,
                                  0,
                                  k_loop_offset,
                                  true,
                                  true,
                                  false,
                                  false,
                                  LDG_SWITCH_ON_EVICT_NORMAL);
  }

  while (k_loop_offset + LOOP_TILE_K * 2 < K) {
    alternate_ldm_mma_sts_stg_ldg(LDG_SM_BUFFER_INDEX + 1,
                                  MMA_REG_BUFFER_INDEX_1,
                                  MMA_REG_BUFFER_INDEX_0,
                                  (LDG_SM_BUFFER_INDEX ^ 2),
                                  0,
                                  true,
                                  true,
                                  true,
                                  false,
                                  LDG_SWITCH_OFF);

    LDG_SM_BUFFER_INDEX ^= 2;
    k_loop_offset += LOOP_TILE_K * 2;

    __syncthreads();

    alternate_ldm_mma_sts_stg_ldg(0, 0, 0, 0, k_loop_offset, false, false, false, false, LDG_SWITCH_ON_EVICT_NORMAL);

    alternate_ldm_mma_sts_stg_ldg(LDG_SM_BUFFER_INDEX,
                                  MMA_REG_BUFFER_INDEX_0,
                                  MMA_REG_BUFFER_INDEX_1,
                                  0,
                                  0,
                                  true,
                                  true,
                                  false,
                                  false,
                                  LDG_SWITCH_OFF);
  }

  {
    alternate_ldm_mma_sts_stg_ldg(LDG_SM_BUFFER_INDEX + 1,
                                  MMA_REG_BUFFER_INDEX_1,
                                  MMA_REG_BUFFER_INDEX_0,
                                  (LDG_SM_BUFFER_INDEX ^ 2),
                                  0,
                                  true,
                                  true,
                                  true,
                                  false,
                                  LDG_SWITCH_OFF);
    LDG_SM_BUFFER_INDEX ^= 2;
    k_loop_offset += LOOP_TILE_K * 2;

    __syncthreads();

    alternate_ldm_mma_sts_stg_ldg(LDG_SM_BUFFER_INDEX,
                                  MMA_REG_BUFFER_INDEX_0,
                                  MMA_REG_BUFFER_INDEX_1,
                                  0,
                                  0,
                                  true,
                                  true,
                                  false,
                                  false,
                                  LDG_SWITCH_OFF);
  }

  {
    alternate_ldm_mma_sts_stg_ldg(LDG_SM_BUFFER_INDEX + 1,
                                  MMA_REG_BUFFER_INDEX_1,
                                  MMA_REG_BUFFER_INDEX_0,
                                  0,
                                  0,
                                  true,
                                  true,
                                  false,
                                  false,
                                  LDG_SWITCH_OFF);

    alternate_ldm_mma_sts_stg_ldg(LDG_SM_BUFFER_INDEX,
                                  MMA_REG_BUFFER_INDEX_0,
                                  MMA_REG_BUFFER_INDEX_1,
                                  0,
                                  0,
                                  false,
                                  true,
                                  false,
                                  true,
                                  LDG_SWITCH_OFF);
  }

#undef A_global_2_ldg_reg
#undef A_ldg_reg_2_sm
#undef B_global_2_ldg_reg
#undef B_ldg_reg_2_sm
#undef sm_2_A_mma_reg
#undef sm_2_B_mma_reg
#undef ldm_mma_sts_stg_ldg
#undef alternate_ldm_mma_sts_stg_ldg
#undef mma_m16n8k8_row_col
}

// template<typename T, int BLOCK_TILE_M, int BLOCK_TILE_N, int WARP_TILE_M, int WARP_TILE_N>
// __global__ void
// fp16_mma_m16n8k16_ldmatrix_trans__overlap_global_2_sm__quadra_buffer__reduce_instructions__reorder_instructions__overlap_reg_2_global__stg_memory_coalecesing__overlap_sts__stg_wt__reduce_loop__backup(
//   const T* A, const T* B, T* C, int M, int N, int K)
// {
//   constexpr int WARP_COUNT   = BLOCK_TILE_M / WARP_TILE_M * BLOCK_TILE_N / WARP_TILE_N;
//   constexpr int THREAD_COUNT = WARP_COUNT * 32;
//
//   constexpr int LOOP_TILE_K         = 16;
//   constexpr int LDG_SM_BUFFER_SIZE  = 4;
//   constexpr int LDG_REG_BUFFER_SIZE = 2;
//
//   constexpr int A_sm_dim0 = LDG_SM_BUFFER_SIZE;
//   constexpr int A_sm_dim1 = 2;
//   constexpr int A_sm_dim2 = BLOCK_TILE_M;
//   constexpr int A_sm_dim3 = LOOP_TILE_K / 2;
//   constexpr int B_sm_dim0 = LDG_SM_BUFFER_SIZE;
//   constexpr int B_sm_dim1 = LOOP_TILE_K / 8;
//   constexpr int B_sm_dim2 = BLOCK_TILE_N / 8;
//   constexpr int B_sm_dim3 = 64;
//
//   // The 64 elements of type T in each 8x8 matrix are stored consecutively in a single layer of shared memory.
//   __shared__ union {
//     struct {
//       T A_sm[A_sm_dim0 * A_sm_dim1 * A_sm_dim2 * A_sm_dim3];
//       T B_sm[B_sm_dim0 * B_sm_dim1 * B_sm_dim2 * B_sm_dim3];
//     } mma;
//     static_assert(WARP_TILE_N % 16 == 0);
//     T result[WARP_COUNT][WARP_TILE_N / 16][WARP_TILE_M][16];
//   } data;
//
//   static_assert(BLOCK_TILE_M * LOOP_TILE_K % THREAD_COUNT == 0);
//   static_assert(BLOCK_TILE_M * LOOP_TILE_K / THREAD_COUNT % 8 == 0);
//   constexpr int A_LDG_COUNT_PER_THREAD = BLOCK_TILE_M * LOOP_TILE_K / THREAD_COUNT;
//   constexpr int A_LDG_LOOP_COUNT       = A_LDG_COUNT_PER_THREAD / 8;
//   // clang-format off
//   // This is the thread layout of the same warp that loads matrix A, where each thread reads M1xK8 elements of type T
//   at a
//   // loop iteration.
//   // T0  T16
//   // T1  T17
//   // T2  T18
//   // ... ...
//   // T14 T30
//   // T15 T31
//   // clang-format on
//   float A_ldg_reg[LDG_REG_BUFFER_SIZE][A_LDG_LOOP_COUNT][4];
//
//   static_assert(BLOCK_TILE_N * LOOP_TILE_K % THREAD_COUNT == 0);
//   static_assert(BLOCK_TILE_N * LOOP_TILE_K / THREAD_COUNT % 8 == 0);
//   constexpr int B_LDG_COUNT_PER_THREAD = BLOCK_TILE_N * LOOP_TILE_K / THREAD_COUNT;
//   constexpr int B_LDG_LOOP_COUNT       = B_LDG_COUNT_PER_THREAD / 8;
//   // clang-format off
//   // This is the thread layout of the same warp that loads matrix B, where each thread reads K1xN8 elements of type T
//   at a
//   // loop iteration.
//   // T0  T16
//   // T1  T17
//   // T2  T18
//   // ... ...
//   // T14 T30
//   // T15 T31
//   // clang-format on
//   float B_ldg_reg[LDG_REG_BUFFER_SIZE][B_LDG_LOOP_COUNT][4];
//
//   const int m_block_offset = blockIdx.y * BLOCK_TILE_M;
//   const int n_block_offset = blockIdx.x * BLOCK_TILE_N;
//
//   const int     warp_id                    = threadIdx.x / 32;
//   const int     lane_id                    = threadIdx.x % 32;
//   const int transposed_lane_id_mask = (lane_id / 8 == 0 || lane_id / 8 == 3) ? 0x00 : 0x18;
//   const int transposed_lane_id      = lane_id ^ transposed_lane_id_mask;
//
//   constexpr int M_MMA_WARP_COUNT       = BLOCK_TILE_M / WARP_TILE_M;
//   constexpr int M_GROUP_COUNT_PER_WARP = WARP_TILE_M / 8;
//   constexpr int N_GROUP_COUNT_PER_WARP = WARP_TILE_N / 16;
//
//   constexpr int MMA_REG_BUFFER_SIZE    = 2;
//   constexpr int MMA_REG_BUFFER_INDEX_0 = 0;
//   constexpr int MMA_REG_BUFFER_INDEX_1 = 1;
//   T             A_mma_reg[MMA_REG_BUFFER_SIZE][M_GROUP_COUNT_PER_WARP][4];
//   T             B_mma_reg[MMA_REG_BUFFER_SIZE][N_GROUP_COUNT_PER_WARP][8];
//   float         C_mma_reg[M_GROUP_COUNT_PER_WARP][N_GROUP_COUNT_PER_WARP][4] = {0};
//   static_assert(N_GROUP_COUNT_PER_WARP % 2 == 0);
//   union _2x4_or_1x8 {
//     T _2x4[2][4];
//     T _1x8[8];
//   };
//   _2x4_or_1x8 C_transposed[M_GROUP_COUNT_PER_WARP][N_GROUP_COUNT_PER_WARP / 2];
//
//   const int m_warp_offset = warp_id % M_MMA_WARP_COUNT * WARP_TILE_M;
//   const int n_warp_offset = warp_id / M_MMA_WARP_COUNT * WARP_TILE_N;
//
//   const int A_ldg_reg_2_A_sm_partial_offset =
//     lane_id / 16 * A_sm_dim2 * A_sm_dim3 + (warp_id * 16 + lane_id % 16) * A_sm_dim3;
//
//   const int B_ldg_reg_2_B_sm_partial_offset =
//     (lane_id % 16) / 8 * B_sm_dim2 * B_sm_dim3 + (lane_id % 16) % 8 * 8 + (warp_id * 2 + lane_id / 16) * B_sm_dim3;
//
//   const int A_global_partial_offset = (m_block_offset + warp_id * 16 + lane_id % 16) * K + lane_id / 16 * 8;
//   const int B_global_partial_offset = lane_id % 16 * N + n_block_offset + warp_id * 16 + lane_id / 16 * 8;
//
//   const T* A_global_ptr_for_ldg = &A[A_global_partial_offset];
//   const T* B_global_ptr_for_ldg = &B[B_global_partial_offset];
//
//   const T* A_sm_ptr_for_ldg = &data.mma.A_sm[A_ldg_reg_2_A_sm_partial_offset];
//   const T* B_sm_ptr_for_ldg = &data.mma.B_sm[B_ldg_reg_2_B_sm_partial_offset];
//
//   const int A_sm_2_A_mma_reg_partial_offset =
//     lane_id % 16 / 8 * A_sm_dim2 * A_sm_dim3 + (m_warp_offset + lane_id % 8) * A_sm_dim3;
//
//   const int B_sm_2_B_mma_reg_partial_offset = transposed_lane_id % 16 / 8 * B_sm_dim2 * B_sm_dim3
//                                               + (n_warp_offset + transposed_lane_id / 16 * 8) / 8 * B_sm_dim3
//                                               + transposed_lane_id % 8 * 8;
//
//   const T* A_sm_ptr_for_mma = &data.mma.A_sm[A_sm_2_A_mma_reg_partial_offset + lane_id / 16 * 8 * A_sm_dim3];
//   const T* B_sm_ptr_for_mma = &data.mma.B_sm[B_sm_2_B_mma_reg_partial_offset];
//
// #define A_global_2_ldg_reg(k_loop_offset_, ldg_reg_buffer_index, loop) \
//   { \
//     /* const int m = (loop * WARP_COUNT + warp_id) * 16 + lane_id % 16; */ \
//     /* const int k = lane_id / 16 * 8; */ \
//     FETCH_FLOAT4_WITH_PTR(&A_ldg_reg[ldg_reg_buffer_index][loop][0], \
//                           A_global_ptr_for_ldg + (loop) * WARP_COUNT * 16 * K + k_loop_offset_); \
//   }
//
// #define B_global_2_ldg_reg(k_loop_offset_, ldg_reg_buffer_index, loop) \
//   { \
//     /* const int k = lane_id % 16;                                           */ \
//     /* const int n = (loop * WARP_COUNT + warp_id) * 16 + lane_id / 16 * 8;  */ \
//     FETCH_FLOAT4_WITH_PTR(&B_ldg_reg[ldg_reg_buffer_index][loop][0], \
//                           B_global_ptr_for_ldg + (loop) * WARP_COUNT * 16 + (k_loop_offset_) * N); \
//   }
//
// #define A_ldg_reg_2_sm(ldg_sm_buf_index, ldg_reg_buffer_index, loop) \
//   { \
//     /* const int m = (loop * WARP_COUNT + warp_id) * 16 + lane_id % 16; */ \
//     /* const int k = lane_id / 16 * 8;  */ \
//     STORE_FLOAT4_WITH_PTR(A_sm_ptr_for_ldg + loop * WARP_COUNT * 16 * A_sm_dim3 \
//                             + (ldg_sm_buf_index) * A_sm_dim1 * A_sm_dim2 * A_sm_dim3, \
//                           &A_ldg_reg[ldg_reg_buffer_index][loop][0]); \
//   }
//
// #define B_ldg_reg_2_sm(ldg_sm_buf_index, ldg_reg_buffer_index, loop) \
//   { \
//     /*const int k = lane_id % 16; */ \
//     /*const int n = (loop * WARP_COUNT + warp_id) * 16 + lane_id / 16 * 8;*/ \
//     STORE_FLOAT4_WITH_PTR(B_sm_ptr_for_ldg + (loop) * WARP_COUNT * 2 * B_sm_dim3 \
//                             + (ldg_sm_buf_index) * B_sm_dim1 * B_sm_dim2 * B_sm_dim3, \
//                           &B_ldg_reg[ldg_reg_buffer_index][loop][0]); \
//   }
//
// #define sm_2_A_mma_reg(ldg_sm_buf_index, mma_reg_buffer_index, group) \
//   if constexpr (M_GROUP_COUNT_PER_WARP == 1) { \
//     /* for (int group = 0; group < M_GROUP_COUNT_PER_WARP; ++group) */ { \
//       uint32_t src = __cvta_generic_to_shared(A_sm_ptr_for_mma + (group) * 8 * A_sm_dim3 \
//                                               + (ldg_sm_buf_index) * A_sm_dim1 * A_sm_dim2 * A_sm_dim3); \
//       asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];" \
//                    : "=r"(*(uint32_t*)&A_mma_reg[mma_reg_buffer_index][group][0]), \
//                      "=r"(*(uint32_t*)&A_mma_reg[mma_reg_buffer_index][group][2]) \
//                    : "r"(src)); \
//     } \
//   } \
//   else if constexpr (M_GROUP_COUNT_PER_WARP % 2 == 0) { \
//     /*for (int group = 0; group < M_GROUP_COUNT_PER_WARP; group += 2) */ { \
//       uint32_t src = __cvta_generic_to_shared(A_sm_ptr_for_mma + (group) * 8 * A_sm_dim3 \
//                                               + (ldg_sm_buf_index) * A_sm_dim1 * A_sm_dim2 * A_sm_dim3); \
//       asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];" \
//                    : "=r"(*(uint32_t*)&A_mma_reg[mma_reg_buffer_index][group][0]), \
//                      "=r"(*(uint32_t*)&A_mma_reg[mma_reg_buffer_index][group][2]), \
//                      "=r"(*(uint32_t*)&A_mma_reg[mma_reg_buffer_index][group + 1][0]), \
//                      "=r"(*(uint32_t*)&A_mma_reg[mma_reg_buffer_index][group + 1][2]) \
//                    : "r"(src)); \
//     } \
//   } \
//   else { \
//     static_assert(M_GROUP_COUNT_PER_WARP == 1 || M_GROUP_COUNT_PER_WARP % 2 == 0); \
//   }
//
// #define sm_2_B_mma_reg(ldg_sm_buf_index, mma_reg_buffer_index, group) \
//   /* for (int group = 0; group < N_GROUP_COUNT_PER_WARP; ++group) */ { \
//     uint32_t src = __cvta_generic_to_shared(B_sm_ptr_for_mma + (group) * 2 * B_sm_dim3 \
//                                             + (ldg_sm_buf_index) * B_sm_dim1 * B_sm_dim2 * B_sm_dim3); \
//     asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];" \
//                  : "=r"(*(uint32_t*)&B_mma_reg[mma_reg_buffer_index][group][0]), \
//                    "=r"(*(uint32_t*)&B_mma_reg[mma_reg_buffer_index][group][2]), \
//                    "=r"(*(uint32_t*)&B_mma_reg[mma_reg_buffer_index][group][4]), \
//                    "=r"(*(uint32_t*)&B_mma_reg[mma_reg_buffer_index][group][6]) \
//                  : "r"(src)); \
//   }
//
//   T* C_ptr = &C[(m_block_offset + m_warp_offset + lane_id / 4) * N + n_block_offset + n_warp_offset + lane_id % 4 *
//   8];
//
// #define ldm_mma_sts_stg(ldm_sm_buffer_index, \
//                         ldm_reg_buffer_index, \
//                         mma_reg_buffer_index, \
//                         sts_sm_base_index, \
//                         ldg_k_offset, \
//                         rank, \
//                         ldm_switch, \
//                         mma_switch, \
//                         sts_switch, \
//                         stg_switch, \
//                         ldg_switch) \
//   { \
//     /* STS */ \
//     constexpr int MxN_GORUP_COUNT = M_GROUP_COUNT_PER_WARP * N_GROUP_COUNT_PER_WARP; \
//     constexpr int STS_COUNT       = LDG_REG_BUFFER_SIZE * (A_LDG_LOOP_COUNT + B_LDG_LOOP_COUNT); \
//     static_assert(STS_COUNT <= MxN_GORUP_COUNT); \
//     if constexpr (sts_switch && MxN_GORUP_COUNT - STS_COUNT <= rank && rank < MxN_GORUP_COUNT) { \
//       constexpr int sts_rank        = rank - (MxN_GORUP_COUNT - STS_COUNT); \
//       constexpr int sts_addr_offset = sts_rank / (A_LDG_LOOP_COUNT + B_LDG_LOOP_COUNT); \
//       constexpr int sts_loop        = sts_rank % (A_LDG_LOOP_COUNT + B_LDG_LOOP_COUNT); \
//       if constexpr (sts_loop < A_LDG_LOOP_COUNT) { \
//         A_ldg_reg_2_sm(sts_sm_base_index + sts_addr_offset, sts_addr_offset, sts_loop); \
//       } \
//       if constexpr (A_LDG_LOOP_COUNT <= sts_loop) { \
//         B_ldg_reg_2_sm(sts_sm_base_index + sts_addr_offset, sts_addr_offset, sts_loop - A_LDG_LOOP_COUNT); \
//       } \
//     } \
//     /* MMA */ \
//     if constexpr (mma_switch && rank < M_GROUP_COUNT_PER_WARP * N_GROUP_COUNT_PER_WARP) { \
//       constexpr int mg = rank % M_GROUP_COUNT_PER_WARP; \
//       constexpr int ng = rank / M_GROUP_COUNT_PER_WARP; \
//       mma_m16n8k16_row_col(C_mma_reg[mg][ng], \
//                            B_mma_reg[mma_reg_buffer_index][ng], \
//                            A_mma_reg[mma_reg_buffer_index][mg], \
//                            C_mma_reg[mg][ng]); \
//     } \
//     /* LDM */ \
//     static_assert(M_GROUP_COUNT_PER_WARP == 1 || M_GROUP_COUNT_PER_WARP % 2 == 0); \
//     static_assert((M_GROUP_COUNT_PER_WARP + 1) / 2 + N_GROUP_COUNT_PER_WARP < MxN_GORUP_COUNT); \
//     if constexpr (ldm_switch && rank < (M_GROUP_COUNT_PER_WARP + 1) / 2) { \
//       sm_2_A_mma_reg(ldm_sm_buffer_index, ldm_reg_buffer_index, rank * 2); \
//     } \
//     if constexpr (ldm_switch && (M_GROUP_COUNT_PER_WARP + 1) / 2 <= rank \
//                   && rank < (M_GROUP_COUNT_PER_WARP + 1) / 2 + N_GROUP_COUNT_PER_WARP) { \
//       sm_2_B_mma_reg(ldm_sm_buffer_index, ldm_reg_buffer_index, rank - (M_GROUP_COUNT_PER_WARP + 1) / 2); \
//     } \
//     /* LDG */ \
//     if constexpr (ldg_switch && rank < LDG_REG_BUFFER_SIZE * (A_LDG_LOOP_COUNT + B_LDG_LOOP_COUNT)) { \
//       constexpr int ldg_addr_offset = rank / (A_LDG_LOOP_COUNT + B_LDG_LOOP_COUNT); \
//       constexpr int k_offset        = ldg_addr_offset * LOOP_TILE_K; \
//       constexpr int ldg_loop        = rank % (A_LDG_LOOP_COUNT + B_LDG_LOOP_COUNT); \
//       if constexpr (ldg_loop < A_LDG_LOOP_COUNT) { \
//         A_global_2_ldg_reg(ldg_k_offset + k_offset, ldg_addr_offset, ldg_loop); \
//       } \
//       if constexpr (A_LDG_LOOP_COUNT <= ldg_loop) { \
//         B_global_2_ldg_reg(ldg_k_offset + k_offset, ldg_addr_offset, ldg_loop - A_LDG_LOOP_COUNT); \
//       } \
//     } \
//     /* STG */ \
//     if constexpr (stg_switch && rank < M_GROUP_COUNT_PER_WARP * N_GROUP_COUNT_PER_WARP) { \
//       constexpr int mg = rank % M_GROUP_COUNT_PER_WARP; \
//       constexpr int ng = rank / M_GROUP_COUNT_PER_WARP; \
//       T casted[4]      = {C_mma_reg[mg][ng][0], C_mma_reg[mg][ng][1], C_mma_reg[mg][ng][2], C_mma_reg[mg][ng][3]}; \
//       asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n" \
//                    : "=r"(*(uint32_t*)&C_transposed[mg][ng / 2]._2x4[ng % 2][0]) \
//                    : "r"(*(uint32_t*)&casted[0])); \
//       asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n" \
//                    : "=r"(*(uint32_t*)&C_transposed[mg][ng / 2]._2x4[ng % 2][2]) \
//                    : "r"(*(uint32_t*)&casted[2])); \
//       shfl_23_and_01(C_transposed[mg][ng / 2]._2x4[ng % 2], 0x1, lane_id); \
//       if constexpr ((ng + 1) % 2 == 0) { \
//         shfl_4567_and_0123(C_transposed[mg][ng / 2]._1x8, 0x2, lane_id); \
//         asm volatile("st.global.wt.v4.f32 [%0], {%1, %2, %3, %4};" \
//                      : \
//                      : "l"(C_ptr + mg * 8 * N + (ng - 1) * 16), \
//                        "f"(*(const float*)&C_transposed[mg][ng / 2]._1x8[0]), \
//                        "f"(*(const float*)&C_transposed[mg][ng / 2]._1x8[2]), \
//                        "f"(*(const float*)&C_transposed[mg][ng / 2]._1x8[4]), \
//                        "f"(*(const float*)&C_transposed[mg][ng / 2]._1x8[6]) \
//                      : "memory"); \
//       } \
//     } \
//   }
//
// // FIXME This code is really stupid. Please find a way to optimize it as soon as possible.
// #define alternate_ldm_mma_sts_stg(ldm_sm_buf_index, \
//                                   ldm_reg_buf_index, \
//                                   mma_reg_buf_index, \
//                                   sts_sm_base_index, \
//                                   ldg_k_offset, \
//                                   ldm_switch, \
//                                   mma_switch, \
//                                   sts_switch, \
//                                   stg_switch, \
//                                   ldg_switch) \
//   static_assert(M_GROUP_COUNT_PER_WARP * N_GROUP_COUNT_PER_WARP <= 32); \
//   /* clang-format off */ \
//   ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset,  0,
//   ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
//   ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset,  1,
//   ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
//   ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset,  2,
//   ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
//   ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset,  3,
//   ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
//   ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset,  4,
//   ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
//   ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset,  5,
//   ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
//   ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset,  6,
//   ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
//   ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset,  7,
//   ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
//   ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset,  8,
//   ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
//   ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset,  9,
//   ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
//   ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset, 10,
//   ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
//   ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset, 11,
//   ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
//   ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset, 12,
//   ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
//   ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset, 13,
//   ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
//   ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset, 14,
//   ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
//   ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset, 15,
//   ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
//   ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset, 16,
//   ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
//   ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset, 17,
//   ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
//   ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset, 18,
//   ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
//   ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset, 19,
//   ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
//   ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset, 20,
//   ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
//   ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset, 21,
//   ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
//   ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset, 22,
//   ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
//   ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset, 23,
//   ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
//   ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset, 24,
//   ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
//   ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset, 25,
//   ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
//   ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset, 26,
//   ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
//   ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset, 27,
//   ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
//   ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset, 28,
//   ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
//   ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset, 29,
//   ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
//   ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset, 30,
//   ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \ ldm_mma_sts_stg(ldm_sm_buf_index,
//   ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, ldg_k_offset, 31, ldm_switch, mma_switch, sts_switch,
//   stg_switch, ldg_switch);
//   /* clang-format on */
//
//   alternate_ldm_mma_sts_stg(0, 0, 0, 0, 0, false, false, false, false, true);
//   alternate_ldm_mma_sts_stg(0, 0, 0, 0, 0, false, false, true, false, false);
//
//   __syncthreads();
//
//   alternate_ldm_mma_sts_stg(0, MMA_REG_BUFFER_INDEX_0, 0, 0, 0, true, false, false, false, false);
//
//   alternate_ldm_mma_sts_stg(0, 0, 0, 0, (LOOP_TILE_K * 2), false, false, false, false, true);
//
// #define main_loop_x_2(k_loop_offset) \
//   { \
//     static_assert((k_loop_offset) % LOOP_TILE_K == 0); \
//     constexpr int ldg_sm_buffer_index = (((k_loop_offset) / LOOP_TILE_K) & 0x2); \
//     alternate_ldm_mma_sts_stg((ldg_sm_buffer_index + 1), \
//                               MMA_REG_BUFFER_INDEX_1, \
//                               MMA_REG_BUFFER_INDEX_0, \
//                               (ldg_sm_buffer_index ^ 2), \
//                               0, \
//                               true, \
//                               true, \
//                               true, \
//                               false, \
//                               false); \
//                                                                                                                        \
//     __syncthreads(); \
//                                                                                                                        \
//     alternate_ldm_mma_sts_stg(0, 0, 0, 0, (k_loop_offset), false, false, false, false, true); \
//                                                                                                                        \
//     alternate_ldm_mma_sts_stg((ldg_sm_buffer_index ^ 2), \
//                               MMA_REG_BUFFER_INDEX_0, \
//                               MMA_REG_BUFFER_INDEX_1, \
//                               0, \
//                               0, \
//                               true, \
//                               true, \
//                               false, \
//                               false, \
//                               false); \
//   }
//
// #define main_loop_x_4(base) \
//   main_loop_x_2(base); \ main_loop_x_2(base + LOOP_TILE_K * 2);
//
// #define main_loop_x_8(base) \
//   main_loop_x_4(base); \ main_loop_x_4(base + LOOP_TILE_K * 4);
//
// #define main_loop_x_16(base) \
//   main_loop_x_8(base); \ main_loop_x_8(base + LOOP_TILE_K * 8);
//
// #define main_loop_x_32(base) \
//   main_loop_x_16(base); \ main_loop_x_16(base + LOOP_TILE_K * 16);
//
// #define main_loop_x_64(base) \
//   main_loop_x_32(base); \ main_loop_x_32(base + LOOP_TILE_K * 32);
//
// #define main_loop_x_128(base) \
//   main_loop_x_64(base); \ main_loop_x_64(base + LOOP_TILE_K * 64);
//
//   main_loop_x_128(LOOP_TILE_K * 4);
//   main_loop_x_64(LOOP_TILE_K * (4 + 128));
//   main_loop_x_32(LOOP_TILE_K * (4 + 128 + 64));
//   main_loop_x_16(LOOP_TILE_K * (4 + 128 + 64 + 32));
//   main_loop_x_8(LOOP_TILE_K * (4 + 128 + 64 + 32 + 16));
//   main_loop_x_4(LOOP_TILE_K * (4 + 128 + 64 + 32 + 16 + 8));
//   main_loop_x_2(LOOP_TILE_K * (4 + 128 + 64 + 32 + 16 + 8 + 4));
//
//   int k_loop_offset       = LOOP_TILE_K * 4 + LOOP_TILE_K * (4 + 128 + 64 + 32 + 16 + 8 - 2);
//   int ldg_sm_buffer_index = ((k_loop_offset / LOOP_TILE_K) & 0x2) ^ 0x2;
//
//   while (k_loop_offset + LOOP_TILE_K * 2 < K) {
//     alternate_ldm_mma_sts_stg(ldg_sm_buffer_index + 1,
//                               MMA_REG_BUFFER_INDEX_1,
//                               MMA_REG_BUFFER_INDEX_0,
//                               (ldg_sm_buffer_index ^ 2),
//                               0,
//                               true,
//                               true,
//                               true,
//                               false,
//                               false);
//
//     ldg_sm_buffer_index ^= 2;
//     k_loop_offset += LOOP_TILE_K * 2;
//
//     __syncthreads();
//
//     alternate_ldm_mma_sts_stg(0, 0, 0, 0, k_loop_offset, false, false, false, false, true);
//
//     alternate_ldm_mma_sts_stg(
//       ldg_sm_buffer_index, MMA_REG_BUFFER_INDEX_0, MMA_REG_BUFFER_INDEX_1, 0, 0, true, true, false, false, false);
//   }
//
//   {
//     alternate_ldm_mma_sts_stg(ldg_sm_buffer_index + 1,
//                               MMA_REG_BUFFER_INDEX_1,
//                               MMA_REG_BUFFER_INDEX_0,
//                               (ldg_sm_buffer_index ^ 2),
//                               0,
//                               true,
//                               true,
//                               true,
//                               false,
//                               false);
//     ldg_sm_buffer_index ^= 2;
//     k_loop_offset += LOOP_TILE_K * 2;
//
//     __syncthreads();
//
//     alternate_ldm_mma_sts_stg(
//       ldg_sm_buffer_index, MMA_REG_BUFFER_INDEX_0, MMA_REG_BUFFER_INDEX_1, 0, 0, true, true, false, false, false);
//   }
//
//   {
//     alternate_ldm_mma_sts_stg(
//       ldg_sm_buffer_index + 1, MMA_REG_BUFFER_INDEX_1, MMA_REG_BUFFER_INDEX_0, 0, 0, true, true, false, false,
//       false);
//
//     alternate_ldm_mma_sts_stg(
//       ldg_sm_buffer_index, MMA_REG_BUFFER_INDEX_0, MMA_REG_BUFFER_INDEX_1, 0, 0, false, true, false, true, false);
//   }
//
// #undef A_global_2_ldg_reg
// #undef A_ldg_reg_2_sm
// #undef B_global_2_ldg_reg
// #undef B_ldg_reg_2_sm
// #undef sm_2_A_mma_reg
// #undef sm_2_B_mma_reg
// #undef ldm_mma_sts_stg
// #undef alternate_ldm_mma_sts_stg
// }

template<typename T, int BLOCK_TILE_M, int BLOCK_TILE_N, int WARP_TILE_M, int WARP_TILE_N>
__global__ void
fp16_mma_m16n8k16_ldmatrix_trans__overlap_global_2_sm__octa_buffer__reduce_instructions__reorder_instructions__overlap_reg_2_global__stg_memory_coalecesing__overlap_sts(
  const T* A, const T* B, T* C, int M, int N, int K)
{
  constexpr int WARP_COUNT   = BLOCK_TILE_M / WARP_TILE_M * BLOCK_TILE_N / WARP_TILE_N;
  constexpr int THREAD_COUNT = WARP_COUNT * 32;

  constexpr int LOOP_TILE_K            = 16;
  constexpr int LDG_SM_BUFFER_SIZE     = 8;
  constexpr int LDG_REG_BUFFER_SIZE    = 4;
  constexpr int LDG_REG_BUFFER_INDEX_0 = 0;
  constexpr int LDG_REG_BUFFER_INDEX_1 = 1;
  constexpr int LDG_REG_BUFFER_INDEX_2 = 2;
  constexpr int LDG_REG_BUFFER_INDEX_3 = 3;

  constexpr int A_sm_dim0 = LDG_SM_BUFFER_SIZE;
  constexpr int A_sm_dim1 = 2;
  constexpr int A_sm_dim2 = BLOCK_TILE_M;
  constexpr int A_sm_dim3 = LOOP_TILE_K / 2;
  constexpr int B_sm_dim0 = LDG_SM_BUFFER_SIZE;
  constexpr int B_sm_dim1 = LOOP_TILE_K / 8;
  constexpr int B_sm_dim2 = BLOCK_TILE_N / 8;
  constexpr int B_sm_dim3 = 64;

  // The 64 elements of type T in each 8x8 matrix are stored consecutively in a single layer of shared memory.
  __shared__ union {
    struct {
      T A_sm[A_sm_dim0 * A_sm_dim1 * A_sm_dim2 * A_sm_dim3];
      T B_sm[B_sm_dim0 * B_sm_dim1 * B_sm_dim2 * B_sm_dim3];
    } mma;
    static_assert(WARP_TILE_N % 16 == 0);
    T result[WARP_COUNT][WARP_TILE_N / 16][WARP_TILE_M][16];
  } data;

  static_assert(BLOCK_TILE_M * LOOP_TILE_K % THREAD_COUNT == 0);
  static_assert(BLOCK_TILE_M * LOOP_TILE_K / THREAD_COUNT % 8 == 0);
  constexpr int A_LDG_COUNT_PER_THREAD = BLOCK_TILE_M * LOOP_TILE_K / THREAD_COUNT;
  constexpr int A_LDG_LOOP_COUNT       = A_LDG_COUNT_PER_THREAD / 8;
  // clang-format off
  // This is the thread layout of the same warp that loads matrix A, where each thread reads M1xK8 elements of type T at a
  // loop iteration.
  // T0  T16
  // T1  T17
  // T2  T18
  // ... ...
  // T14 T30
  // T15 T31
  // clang-format on
  float A_ldg_reg[LDG_REG_BUFFER_SIZE][A_LDG_LOOP_COUNT][4];

  static_assert(BLOCK_TILE_N * LOOP_TILE_K % THREAD_COUNT == 0);
  static_assert(BLOCK_TILE_N * LOOP_TILE_K / THREAD_COUNT % 8 == 0);
  constexpr int B_LDG_COUNT_PER_THREAD = BLOCK_TILE_N * LOOP_TILE_K / THREAD_COUNT;
  constexpr int B_LDG_LOOP_COUNT       = B_LDG_COUNT_PER_THREAD / 8;
  // clang-format off
  // This is the thread layout of the same warp that loads matrix B, where each thread reads K1xN8 elements of type T at a
  // loop iteration.
  // T0  T16
  // T1  T17
  // T2  T18
  // ... ...
  // T14 T30
  // T15 T31
  // clang-format on
  float B_ldg_reg[LDG_REG_BUFFER_SIZE][B_LDG_LOOP_COUNT][4];

  const int m_block_offset = blockIdx.y * BLOCK_TILE_M;
  const int n_block_offset = blockIdx.x * BLOCK_TILE_N;

  const int warp_id = threadIdx.x / 32;
  const int lane_id = threadIdx.x % 32;
  // constexpr int transposed_lane_id_mask[4] = {0x00, 0x18, 0x18, 0x00};
  const int transposed_lane_id_mask = (lane_id / 8 == 0 || lane_id / 8 == 3) ? 0x00 : 0x18;
  const int transposed_lane_id      = lane_id ^ transposed_lane_id_mask;

  constexpr int M_MMA_WARP_COUNT       = BLOCK_TILE_M / WARP_TILE_M;
  constexpr int M_GROUP_COUNT_PER_WARP = WARP_TILE_M / 8;
  constexpr int N_GROUP_COUNT_PER_WARP = WARP_TILE_N / 16;

  constexpr int MMA_REG_BUFFER_SIZE    = 2;
  constexpr int MMA_REG_BUFFER_INDEX_0 = 0;
  constexpr int MMA_REG_BUFFER_INDEX_1 = 1;
  T             A_mma_reg[MMA_REG_BUFFER_SIZE][M_GROUP_COUNT_PER_WARP][4];
  T             B_mma_reg[MMA_REG_BUFFER_SIZE][N_GROUP_COUNT_PER_WARP][8];
  float         C_mma_reg[M_GROUP_COUNT_PER_WARP][N_GROUP_COUNT_PER_WARP][4] = {0};
  static_assert(N_GROUP_COUNT_PER_WARP % 2 == 0);
  union _2x4_or_1x8 {
    T _2x4[2][4];
    T _1x8[8];
  };
  _2x4_or_1x8 C_transposed[M_GROUP_COUNT_PER_WARP][N_GROUP_COUNT_PER_WARP / 2];

  const int m_warp_offset = warp_id % M_MMA_WARP_COUNT * WARP_TILE_M;
  const int n_warp_offset = warp_id / M_MMA_WARP_COUNT * WARP_TILE_N;

  const int A_ldg_reg_2_A_sm_partial_offset =
    lane_id / 16 * A_sm_dim2 * A_sm_dim3 + (warp_id * 16 + lane_id % 16) * A_sm_dim3;

  const int B_ldg_reg_2_B_sm_partial_offset =
    (lane_id % 16) / 8 * B_sm_dim2 * B_sm_dim3 + (lane_id % 16) % 8 * 8 + (warp_id * 2 + lane_id / 16) * B_sm_dim3;

  const int A_global_partial_offset = (m_block_offset + warp_id * 16 + lane_id % 16) * K + lane_id / 16 * 8;
  const int B_global_partial_offset = lane_id % 16 * N + n_block_offset + warp_id * 16 + lane_id / 16 * 8;

  const T* A_global_ptr_for_ldg[A_LDG_LOOP_COUNT];
  const T* B_global_ptr_for_ldg[B_LDG_LOOP_COUNT];
  for (int loop = 0; loop < A_LDG_LOOP_COUNT; ++loop) {
    A_global_ptr_for_ldg[loop] = &A[A_global_partial_offset + loop * WARP_COUNT * 16 * K];
  }
  for (int loop = 0; loop < B_LDG_LOOP_COUNT; ++loop) {
    B_global_ptr_for_ldg[loop] = &B[B_global_partial_offset + loop * WARP_COUNT * 16];
  }

  const T* A_sm_ptr_for_ldg[A_LDG_LOOP_COUNT];
  const T* B_sm_ptr_for_ldg[B_LDG_LOOP_COUNT];
  for (int loop = 0; loop < A_LDG_LOOP_COUNT; ++loop) {
    A_sm_ptr_for_ldg[loop] = &data.mma.A_sm[A_ldg_reg_2_A_sm_partial_offset + loop * WARP_COUNT * 16 * A_sm_dim3];
  }
  for (int loop = 0; loop < B_LDG_LOOP_COUNT; ++loop) {
    B_sm_ptr_for_ldg[loop] = &data.mma.B_sm[B_ldg_reg_2_B_sm_partial_offset + loop * WARP_COUNT * 2 * B_sm_dim3];
  }

  const int A_sm_2_A_mma_reg_partial_offset =
    lane_id % 16 / 8 * A_sm_dim2 * A_sm_dim3 + (m_warp_offset + lane_id % 8) * A_sm_dim3;

  const int B_sm_2_B_mma_reg_partial_offset = transposed_lane_id % 16 / 8 * B_sm_dim2 * B_sm_dim3
                                              + (n_warp_offset + transposed_lane_id / 16 * 8) / 8 * B_sm_dim3
                                              + transposed_lane_id % 8 * 8;

  const T* A_sm_ptr_for_mma[M_GROUP_COUNT_PER_WARP];
  const T* B_sm_ptr_for_mma[N_GROUP_COUNT_PER_WARP];

  for (int group = 0; group < M_GROUP_COUNT_PER_WARP; ++group) {
    A_sm_ptr_for_mma[group] = &data.mma.A_sm[A_sm_2_A_mma_reg_partial_offset + (group + lane_id / 16) * 8 * A_sm_dim3];
  }
  for (int group = 0; group < N_GROUP_COUNT_PER_WARP; ++group) {
    B_sm_ptr_for_mma[group] = &data.mma.B_sm[B_sm_2_B_mma_reg_partial_offset + (group * 2 * B_sm_dim3)];
  }

#define global_2_ldg_reg(k_loop_offset, ldg_reg_buffer_index)                                                          \
  {                                                                                                                    \
    _Pragma("unroll") for (int loop = 0; loop < A_LDG_LOOP_COUNT; ++loop)                                              \
    {                                                                                                                  \
      /* const int m = (loop * WARP_COUNT + warp_id) * 16 + lane_id % 16; */                                           \
      /* const int k = lane_id / 16 * 8; */                                                                            \
      FETCH_FLOAT4_WITH_PTR(&A_ldg_reg[ldg_reg_buffer_index][loop][0], A_global_ptr_for_ldg[loop] + k_loop_offset);    \
    }                                                                                                                  \
    _Pragma("unroll") for (int loop = 0; loop < B_LDG_LOOP_COUNT; ++loop)                                              \
    {                                                                                                                  \
      /* const int k = lane_id % 16;                                           */                                      \
      /* const int n = (loop * WARP_COUNT + warp_id) * 16 + lane_id / 16 * 8;  */                                      \
      FETCH_FLOAT4_WITH_PTR(&B_ldg_reg[ldg_reg_buffer_index][loop][0],                                                 \
                            B_global_ptr_for_ldg[loop] + (k_loop_offset) * N);                                         \
    }                                                                                                                  \
  }

#define ldg_reg_2_sm(ldg_sm_buffer_index, ldg_reg_buffer_index)                                                        \
  {                                                                                                                    \
    _Pragma("unroll") for (int loop = 0; loop < A_LDG_LOOP_COUNT; ++loop)                                              \
    {                                                                                                                  \
      /* const int m = (loop * WARP_COUNT + warp_id) * 16 + lane_id % 16; */                                           \
      /* const int k = lane_id / 16 * 8;  */                                                                           \
      STORE_FLOAT4_WITH_PTR(A_sm_ptr_for_ldg[loop] + (ldg_sm_buffer_index) * A_sm_dim1 * A_sm_dim2 * A_sm_dim3,        \
                            &A_ldg_reg[ldg_reg_buffer_index][loop][0]);                                                \
    }                                                                                                                  \
    _Pragma("unroll") for (int loop = 0; loop < B_LDG_LOOP_COUNT; ++loop)                                              \
    {                                                                                                                  \
      /*const int k = lane_id % 16; */                                                                                 \
      /*const int n = (loop * WARP_COUNT + warp_id) * 16 + lane_id / 16 * 8;*/                                         \
      STORE_FLOAT4_WITH_PTR(B_sm_ptr_for_ldg[loop] + (ldg_sm_buffer_index) * B_sm_dim1 * B_sm_dim2 * B_sm_dim3,        \
                            &B_ldg_reg[ldg_reg_buffer_index][loop][0]);                                                \
    }                                                                                                                  \
  }

#define sm_2_A_mma_reg(ldg_sm_buffer_index, mma_reg_buffer_index, group)                                               \
  if constexpr (M_GROUP_COUNT_PER_WARP == 1) {                                                                         \
    /* for (int group = 0; group < M_GROUP_COUNT_PER_WARP; ++group) */ {                                               \
      uint32_t src =                                                                                                   \
        __cvta_generic_to_shared(A_sm_ptr_for_mma[group] + (ldg_sm_buffer_index) * A_sm_dim1 * A_sm_dim2 * A_sm_dim3); \
      asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"                                          \
                   : "=r"(*(uint32_t*)&A_mma_reg[mma_reg_buffer_index][group][0]),                                     \
                     "=r"(*(uint32_t*)&A_mma_reg[mma_reg_buffer_index][group][2])                                      \
                   : "r"(src));                                                                                        \
    }                                                                                                                  \
  }                                                                                                                    \
  else if constexpr (M_GROUP_COUNT_PER_WARP % 2 == 0) {                                                                \
    /*for (int group = 0; group < M_GROUP_COUNT_PER_WARP; group += 2) */ {                                             \
      uint32_t src =                                                                                                   \
        __cvta_generic_to_shared(A_sm_ptr_for_mma[group] + (ldg_sm_buffer_index) * A_sm_dim1 * A_sm_dim2 * A_sm_dim3); \
      asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"                                  \
                   : "=r"(*(uint32_t*)&A_mma_reg[mma_reg_buffer_index][group][0]),                                     \
                     "=r"(*(uint32_t*)&A_mma_reg[mma_reg_buffer_index][group][2]),                                     \
                     "=r"(*(uint32_t*)&A_mma_reg[mma_reg_buffer_index][group + 1][0]),                                 \
                     "=r"(*(uint32_t*)&A_mma_reg[mma_reg_buffer_index][group + 1][2])                                  \
                   : "r"(src));                                                                                        \
    }                                                                                                                  \
  }                                                                                                                    \
  else {                                                                                                               \
    static_assert(M_GROUP_COUNT_PER_WARP == 1 || M_GROUP_COUNT_PER_WARP % 2 == 0);                                     \
  }

#define sm_2_B_mma_reg(ldg_sm_buffer_index, mma_reg_buffer_index, group)                                               \
  /* for (int group = 0; group < N_GROUP_COUNT_PER_WARP; ++group) */ {                                                 \
    uint32_t src =                                                                                                     \
      __cvta_generic_to_shared(B_sm_ptr_for_mma[group] + (ldg_sm_buffer_index) * B_sm_dim1 * B_sm_dim2 * B_sm_dim3);   \
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"                              \
                 : "=r"(*(uint32_t*)&B_mma_reg[mma_reg_buffer_index][group][0]),                                       \
                   "=r"(*(uint32_t*)&B_mma_reg[mma_reg_buffer_index][group][2]),                                       \
                   "=r"(*(uint32_t*)&B_mma_reg[mma_reg_buffer_index][group][4]),                                       \
                   "=r"(*(uint32_t*)&B_mma_reg[mma_reg_buffer_index][group][6])                                        \
                 : "r"(src));                                                                                          \
  }

  T* C_ptr = &C[(m_block_offset + m_warp_offset + lane_id / 4) * N + n_block_offset + n_warp_offset + lane_id % 4 * 8];

#define ldm_mma_sts_stg(ldm_sm_buffer_index,                                                                           \
                        ldm_reg_buffer_index,                                                                          \
                        mma_reg_buffer_index,                                                                          \
                        sts_sm_base_index,                                                                             \
                        sts_reg_base_index,                                                                            \
                        ldg_k_offset,                                                                                  \
                        rank,                                                                                          \
                        ldm_switch,                                                                                    \
                        mma_switch,                                                                                    \
                        sts_switch,                                                                                    \
                        stg_switch,                                                                                    \
                        ldg_switch)                                                                                    \
  {                                                                                                                    \
    static_assert(M_GROUP_COUNT_PER_WARP * N_GROUP_COUNT_PER_WARP >= LDG_REG_BUFFER_SIZE);                             \
    static_assert(LDG_REG_BUFFER_SIZE == 4);                                                                           \
    if constexpr (ldg_switch && rank == 0) {                                                                           \
      global_2_ldg_reg(ldg_k_offset, LDG_REG_BUFFER_INDEX_0)                                                           \
    }                                                                                                                  \
    if constexpr (ldg_switch && rank == 1) {                                                                           \
      global_2_ldg_reg(ldg_k_offset + LOOP_TILE_K, LDG_REG_BUFFER_INDEX_1)                                             \
    }                                                                                                                  \
    if constexpr (ldg_switch && rank == 2) {                                                                           \
      global_2_ldg_reg(ldg_k_offset + 2 * LOOP_TILE_K, LDG_REG_BUFFER_INDEX_2)                                         \
    }                                                                                                                  \
    if constexpr (ldg_switch && rank == 3) {                                                                           \
      global_2_ldg_reg(ldg_k_offset + 3 * LOOP_TILE_K, LDG_REG_BUFFER_INDEX_3)                                         \
    }                                                                                                                  \
    if constexpr (sts_switch && rank + 4 == M_GROUP_COUNT_PER_WARP * N_GROUP_COUNT_PER_WARP) {                         \
      ldg_reg_2_sm(sts_sm_base_index, sts_reg_base_index);                                                             \
    }                                                                                                                  \
    if constexpr (sts_switch && rank + 3 == M_GROUP_COUNT_PER_WARP * N_GROUP_COUNT_PER_WARP) {                         \
      ldg_reg_2_sm(sts_sm_base_index + 1, sts_reg_base_index + 1);                                                     \
    }                                                                                                                  \
    if constexpr (sts_switch && rank + 2 == M_GROUP_COUNT_PER_WARP * N_GROUP_COUNT_PER_WARP) {                         \
      ldg_reg_2_sm(sts_sm_base_index + 2, sts_reg_base_index + 2);                                                     \
    }                                                                                                                  \
    if constexpr (sts_switch && rank + 1 == M_GROUP_COUNT_PER_WARP * N_GROUP_COUNT_PER_WARP) {                         \
      ldg_reg_2_sm(sts_sm_base_index + 3, sts_reg_base_index + 3);                                                     \
    }                                                                                                                  \
    if constexpr (mma_switch && rank < M_GROUP_COUNT_PER_WARP * N_GROUP_COUNT_PER_WARP) {                              \
      constexpr int mg = rank % M_GROUP_COUNT_PER_WARP;                                                                \
      constexpr int ng = rank / M_GROUP_COUNT_PER_WARP;                                                                \
      mma_m16n8k16_row_col(C_mma_reg[mg][ng],                                                                          \
                           B_mma_reg[mma_reg_buffer_index][ng],                                                        \
                           A_mma_reg[mma_reg_buffer_index][mg],                                                        \
                           C_mma_reg[mg][ng]);                                                                         \
    }                                                                                                                  \
    static_assert(M_GROUP_COUNT_PER_WARP == 1 || M_GROUP_COUNT_PER_WARP % 2 == 0);                                     \
    if constexpr (ldm_switch && rank < (M_GROUP_COUNT_PER_WARP + 1) / 2) {                                             \
      sm_2_A_mma_reg(ldm_sm_buffer_index, ldm_reg_buffer_index, rank * 2);                                             \
    }                                                                                                                  \
    if constexpr (ldm_switch && (M_GROUP_COUNT_PER_WARP + 1) / 2 <= rank                                               \
                  && rank < (M_GROUP_COUNT_PER_WARP + 1) / 2 + N_GROUP_COUNT_PER_WARP) {                               \
      sm_2_B_mma_reg(ldm_sm_buffer_index, ldm_reg_buffer_index, rank - (M_GROUP_COUNT_PER_WARP + 1) / 2);              \
    }                                                                                                                  \
    if constexpr (stg_switch && rank < M_GROUP_COUNT_PER_WARP * N_GROUP_COUNT_PER_WARP) {                              \
      constexpr int mg = rank % M_GROUP_COUNT_PER_WARP;                                                                \
      constexpr int ng = rank / M_GROUP_COUNT_PER_WARP;                                                                \
      T casted[4]      = {C_mma_reg[mg][ng][0], C_mma_reg[mg][ng][1], C_mma_reg[mg][ng][2], C_mma_reg[mg][ng][3]};     \
      asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n"                                                   \
                   : "=r"(*(uint32_t*)&C_transposed[mg][ng / 2]._2x4[ng % 2][0])                                       \
                   : "r"(*(uint32_t*)&casted[0]));                                                                     \
      asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n"                                                   \
                   : "=r"(*(uint32_t*)&C_transposed[mg][ng / 2]._2x4[ng % 2][2])                                       \
                   : "r"(*(uint32_t*)&casted[2]));                                                                     \
      shfl_23_and_01(C_transposed[mg][ng / 2]._2x4[ng % 2], 0x1, lane_id);                                             \
      if constexpr ((ng + 1) % 2 == 0) {                                                                               \
        shfl_4567_and_0123(C_transposed[mg][ng / 2]._1x8, 0x2, lane_id);                                               \
        asm volatile("st.global.wt.v4.f32 [%0], {%1, %2, %3, %4};"                                                     \
                     :                                                                                                 \
                     : "l"(C_ptr + mg * 8 * N + (ng - 1) * 16),                                                        \
                       "f"(*(const float*)&C_transposed[mg][ng / 2]._1x8[0]),                                          \
                       "f"(*(const float*)&C_transposed[mg][ng / 2]._1x8[2]),                                          \
                       "f"(*(const float*)&C_transposed[mg][ng / 2]._1x8[4]),                                          \
                       "f"(*(const float*)&C_transposed[mg][ng / 2]._1x8[6])                                           \
                     : "memory");                                                                                      \
      }                                                                                                                \
    }                                                                                                                  \
  }

// FIXME This code is really stupid. Please find a way to optimize it as soon as possible.
#define alternate_ldm_mma_sts_stg(ldm_sm_buf_index,                                                                    \
                                  ldm_reg_buf_index,                                                                   \
                                  mma_reg_buf_index,                                                                   \
                                  sts_sm_base_index,                                                                   \
                                  sts_reg_base_index,                                                                  \
                                  ldg_k_offset,                                                                        \
                                  ldm_switch,                                                                          \
                                  mma_switch,                                                                          \
                                  sts_switch,                                                                          \
                                  stg_switch,                                                                          \
                                  ldg_switch)                                                                          \
  static_assert(M_GROUP_COUNT_PER_WARP * N_GROUP_COUNT_PER_WARP <= 32);                                                \
  /* clang-format off */ \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, sts_reg_base_index, ldg_k_offset,  0, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);          \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, sts_reg_base_index, ldg_k_offset,  1, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);          \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, sts_reg_base_index, ldg_k_offset,  2, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);          \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, sts_reg_base_index, ldg_k_offset,  3, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);          \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, sts_reg_base_index, ldg_k_offset,  4, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);          \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, sts_reg_base_index, ldg_k_offset,  5, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);          \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, sts_reg_base_index, ldg_k_offset,  6, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);          \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, sts_reg_base_index, ldg_k_offset,  7, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);          \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, sts_reg_base_index, ldg_k_offset,  8, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);          \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, sts_reg_base_index, ldg_k_offset,  9, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);          \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, sts_reg_base_index, ldg_k_offset, 10, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, sts_reg_base_index, ldg_k_offset, 11, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, sts_reg_base_index, ldg_k_offset, 12, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, sts_reg_base_index, ldg_k_offset, 13, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, sts_reg_base_index, ldg_k_offset, 14, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, sts_reg_base_index, ldg_k_offset, 15, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, sts_reg_base_index, ldg_k_offset, 16, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, sts_reg_base_index, ldg_k_offset, 17, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, sts_reg_base_index, ldg_k_offset, 18, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, sts_reg_base_index, ldg_k_offset, 19, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, sts_reg_base_index, ldg_k_offset, 20, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, sts_reg_base_index, ldg_k_offset, 21, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, sts_reg_base_index, ldg_k_offset, 22, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, sts_reg_base_index, ldg_k_offset, 23, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, sts_reg_base_index, ldg_k_offset, 24, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, sts_reg_base_index, ldg_k_offset, 25, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, sts_reg_base_index, ldg_k_offset, 26, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, sts_reg_base_index, ldg_k_offset, 27, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, sts_reg_base_index, ldg_k_offset, 28, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, sts_reg_base_index, ldg_k_offset, 29, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, sts_reg_base_index, ldg_k_offset, 30, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);         \
  ldm_mma_sts_stg(ldm_sm_buf_index, ldm_reg_buf_index, mma_reg_buf_index, sts_sm_base_index, sts_reg_base_index, ldg_k_offset, 31, ldm_switch, mma_switch, sts_switch, stg_switch, ldg_switch);
  /* clang-format on */

  global_2_ldg_reg(0, LDG_REG_BUFFER_INDEX_0);
  global_2_ldg_reg(LOOP_TILE_K, LDG_REG_BUFFER_INDEX_1);
  global_2_ldg_reg(LOOP_TILE_K * 2, LDG_REG_BUFFER_INDEX_2);
  global_2_ldg_reg(LOOP_TILE_K * 3, LDG_REG_BUFFER_INDEX_3);
  ldg_reg_2_sm(0, LDG_REG_BUFFER_INDEX_0);
  ldg_reg_2_sm(1, LDG_REG_BUFFER_INDEX_1);
  ldg_reg_2_sm(2, LDG_REG_BUFFER_INDEX_2);
  ldg_reg_2_sm(3, LDG_REG_BUFFER_INDEX_3);
  __syncthreads();

  int LDG_SM_BUFFER_INDEX = 0;
  int k_loop_offset       = LOOP_TILE_K * 4;

  alternate_ldm_mma_sts_stg(
    LDG_SM_BUFFER_INDEX, MMA_REG_BUFFER_INDEX_0, 0, 0, 0, k_loop_offset, true, false, false, false, true);

  // global_2_ldg_reg(k_loop_offset, LDG_REG_BUFFER_INDEX_0);
  // global_2_ldg_reg(k_loop_offset + LOOP_TILE_K, LDG_REG_BUFFER_INDEX_1);
  // global_2_ldg_reg(k_loop_offset + LOOP_TILE_K * 2, LDG_REG_BUFFER_INDEX_2);
  // global_2_ldg_reg(k_loop_offset + LOOP_TILE_K * 3, LDG_REG_BUFFER_INDEX_3);

  while (k_loop_offset + LOOP_TILE_K * 4 < K) {
    alternate_ldm_mma_sts_stg(LDG_SM_BUFFER_INDEX + 1,
                              MMA_REG_BUFFER_INDEX_1,
                              MMA_REG_BUFFER_INDEX_0,
                              0,
                              0,
                              0,
                              true,
                              true,
                              false,
                              false,
                              false);

    alternate_ldm_mma_sts_stg(LDG_SM_BUFFER_INDEX + 2,
                              MMA_REG_BUFFER_INDEX_0,
                              MMA_REG_BUFFER_INDEX_1,
                              0,
                              0,
                              0,
                              true,
                              true,
                              false,
                              false,
                              false);

    alternate_ldm_mma_sts_stg(LDG_SM_BUFFER_INDEX + 3,
                              MMA_REG_BUFFER_INDEX_1,
                              MMA_REG_BUFFER_INDEX_0,
                              (LDG_SM_BUFFER_INDEX ^ 4),
                              LDG_REG_BUFFER_INDEX_0,
                              0,
                              true,
                              true,
                              true,
                              false,
                              false);

    LDG_SM_BUFFER_INDEX ^= 4;
    k_loop_offset += LOOP_TILE_K * 4;

    __syncthreads();

    alternate_ldm_mma_sts_stg(LDG_SM_BUFFER_INDEX,
                              MMA_REG_BUFFER_INDEX_0,
                              MMA_REG_BUFFER_INDEX_1,
                              0,
                              0,
                              k_loop_offset,
                              true,
                              true,
                              false,
                              false,
                              true);
  }

  {
    alternate_ldm_mma_sts_stg(LDG_SM_BUFFER_INDEX + 1,
                              MMA_REG_BUFFER_INDEX_1,
                              MMA_REG_BUFFER_INDEX_0,
                              0,
                              0,
                              0,
                              true,
                              true,
                              false,
                              false,
                              false);

    alternate_ldm_mma_sts_stg(LDG_SM_BUFFER_INDEX + 2,
                              MMA_REG_BUFFER_INDEX_0,
                              MMA_REG_BUFFER_INDEX_1,
                              0,
                              0,
                              0,
                              true,
                              true,
                              false,
                              false,
                              false);

    alternate_ldm_mma_sts_stg(LDG_SM_BUFFER_INDEX + 3,
                              MMA_REG_BUFFER_INDEX_1,
                              MMA_REG_BUFFER_INDEX_0,
                              (LDG_SM_BUFFER_INDEX ^ 4),
                              LDG_REG_BUFFER_INDEX_0,
                              0,
                              true,
                              true,
                              true,
                              false,
                              false);

    LDG_SM_BUFFER_INDEX ^= 4;
    k_loop_offset += LOOP_TILE_K * 4;

    __syncthreads();

    alternate_ldm_mma_sts_stg(
      LDG_SM_BUFFER_INDEX, MMA_REG_BUFFER_INDEX_0, MMA_REG_BUFFER_INDEX_1, 0, 0, 0, true, true, false, false, false);
  }

  {
    alternate_ldm_mma_sts_stg(LDG_SM_BUFFER_INDEX + 1,
                              MMA_REG_BUFFER_INDEX_1,
                              MMA_REG_BUFFER_INDEX_0,
                              0,
                              0,
                              0,
                              true,
                              true,
                              false,
                              false,
                              false);

    alternate_ldm_mma_sts_stg(LDG_SM_BUFFER_INDEX + 2,
                              MMA_REG_BUFFER_INDEX_0,
                              MMA_REG_BUFFER_INDEX_1,
                              0,
                              0,
                              0,
                              true,
                              true,
                              false,
                              false,
                              false);

    alternate_ldm_mma_sts_stg(LDG_SM_BUFFER_INDEX + 3,
                              MMA_REG_BUFFER_INDEX_1,
                              MMA_REG_BUFFER_INDEX_0,
                              0,
                              0,
                              0,
                              true,
                              true,
                              false,
                              false,
                              false);

    alternate_ldm_mma_sts_stg(
      LDG_SM_BUFFER_INDEX, MMA_REG_BUFFER_INDEX_0, MMA_REG_BUFFER_INDEX_1, 0, 0, 0, false, true, false, true, false);
  }

#undef global_2_ldg_reg
#undef ldg_reg_2_sm
#undef sm_2_A_mma_reg
#undef sm_2_B_mma_reg
#undef ldm_mma_stg
#undef alternate_ldm_mma_stg
}

#define define_check_function(function)                                                                                                \
  template<typename T, int BLOCK_TILE_M, int BLOCK_TILE_N, int WARP_TILE_M, int WARP_TILE_N>                                           \
  void launch_##function(const T* A, const T* B, T* C, int M, int N, int K)                                                            \
  {                                                                                                                                    \
    if (std::is_same<T, half>::value == false && std::is_same<T, __nv_bfloat16>::value == false) {                                     \
      throw std::runtime_error("T is not supported.");                                                                                 \
    }                                                                                                                                  \
    constexpr int LOOP_TILE_K = 16;                                                                                                    \
    if (!(M % BLOCK_TILE_M == 0 && N % BLOCK_TILE_N == 0 && K % LOOP_TILE_K == 0)) {                                                   \
      throw std::runtime_error("M or N or K are not aligned.");                                                                        \
    }                                                                                                                                  \
    auto kSmemSize   = 0;                                                                                                              \
    auto kernel_func = &function<T, BLOCK_TILE_M, BLOCK_TILE_N, WARP_TILE_M, WARP_TILE_N>;                                             \
    CHECK_CUDA_RETURN(cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));                      \
    static_assert(8 <= BLOCK_TILE_M && BLOCK_TILE_M <= 256 && (BLOCK_TILE_M & (BLOCK_TILE_M - 1)) == 0);                               \
    static_assert(16 <= BLOCK_TILE_N && BLOCK_TILE_N <= 256 && (BLOCK_TILE_N & (BLOCK_TILE_N - 1)) == 0);                              \
    static_assert(LOOP_TILE_K == 16);                                                                                                  \
    static_assert(BLOCK_TILE_M % WARP_TILE_M == 0 && BLOCK_TILE_N % WARP_TILE_N == 0);                                                 \
    static_assert(WARP_TILE_N % 16 == 0 && WARP_TILE_M % 8 == 0);                                                                      \
    constexpr int WARP_COUNT = BLOCK_TILE_N / WARP_TILE_N * BLOCK_TILE_M / WARP_TILE_M;                                                \
    static_assert(1 <= WARP_COUNT && WARP_COUNT <= 32 && (WARP_COUNT & (WARP_COUNT - 1)) == 0);                                        \
    dim3 grid(N / BLOCK_TILE_N, M / BLOCK_TILE_M);                                                                                     \
    dim3 block(WARP_COUNT * 32);                                                                                                       \
    function<T, BLOCK_TILE_M, BLOCK_TILE_N, WARP_TILE_M, WARP_TILE_N><<<grid, block>>>(A, B, C, M, N, K);                              \
    CHECK_CUDA_ERROR();                                                                                                                \
  }                                                                                                                                    \
  template<typename T>                                                                                                                 \
  void function##___check_relative_error(                                                                                              \
    const T* A, const T* B, T* C, int M, int N, int K, const std::vector<float>& base)                                                 \
  {                                                                                                                                    \
    std::vector<T> host_C(M* N);                                                                                                       \
    memset(host_C.data(), 0, sizeof(T) * host_C.size());                                                                               \
    cudaMemset(C, 0, sizeof(T) * M * N);                                                                                               \
    launch_##function<T, 128, 128, 64, 64>(A, B, C, M, N, K);                                                                          \
    cudaMemcpy(host_C.data(), C, sizeof(T) * host_C.size(), cudaMemcpyDefault);                                                        \
    float max_error = 0, base_value, current_value;                                                                                    \
    int   position  = 0;                                                                                                               \
    for (int i = 0; i < host_C.size(); ++i) {                                                                                          \
      if (fabs(float(host_C[i]) - base[i]) > max_error) {                                                                              \
        max_error     = fabs(float(host_C[i]) - base[i]);                                                                              \
        base_value    = base[i];                                                                                                       \
        current_value = host_C[i];                                                                                                     \
        position      = i;                                                                                                             \
      }                                                                                                                                \
    }                                                                                                                                  \
    const char* type = std::is_same<T, half>::value ? "half" : "__nv_bfloat16";                                                        \
    const char* name = #function;                                                                                                      \
    printf(                                                                                                                            \
      "max_relative_error = %8.6f, max_absolute_error = %8.3f, base_value = %10.3f, current_value = %10.3f, type=%16s, function=%s\n", \
      fabs(max_error / base_value),                                                                                                    \
      max_error,                                                                                                                       \
      base_value,                                                                                                                      \
      current_value,                                                                                                                   \
      type,                                                                                                                            \
      name);                                                                                                                           \
  }

/* clang-format off */
// define_check_function(fp16_mma_m16n8k16_ldmatrix);
// define_check_function(fp16_mma_m16n8k16_ldmatrix_trans);
// define_check_function(fp16_mma_m16n8k16_ldmatrix_trans__overlap_global_2_sm);
// define_check_function(fp16_mma_m16n8k16_ldmatrix_trans__overlap_global_2_sm__quadra_buffer);
// define_check_function(fp16_mma_m16n8k16_ldmatrix_trans__overlap_global_2_sm__quadra_buffer__reduce_instructions);
// define_check_function(fp16_mma_m16n8k16_ldmatrix_trans__overlap_global_2_sm__quadra_buffer__reduce_instructions__reorder_instructions);
// define_check_function(fp16_mma_m16n8k16_ldmatrix_trans__overlap_global_2_sm__quadra_buffer__reduce_instructions__reorder_instructions__overlap_reg_2_global);
// define_check_function(fp16_mma_m16n8k16_ldmatrix_trans__overlap_global_2_sm__quadra_buffer__reduce_instructions__reorder_instructions__overlap_reg_2_global__stg_memory_coalecesing);
// define_check_function(fp16_mma_m16n8k16_ldmatrix_trans__overlap_global_2_sm__quadra_buffer__reduce_instructions__reorder_instructions__overlap_reg_2_global__stg_memory_coalecesing__overlap_sts);
define_check_function(fp16_mma_m16n8k16_ldmatrix_trans__overlap_global_2_sm__quadra_buffer__reduce_instructions__reorder_instructions__overlap_reg_2_global__stg_memory_coalecesing__overlap_sts__stg_wt);
define_check_function(fp16_mma_m16n8k16_ldmatrix_trans__overlap_global_2_sm__quadra_buffer__reduce_instructions__reorder_instructions__overlap_reg_2_global__stg_memory_coalecesing__overlap_sts__stg_wt__reduce_IMAD_AND_LEA);
define_check_function(fp16_mma_m16n8k16_ldmatrix_trans__overlap_global_2_sm__quadra_buffer__reduce_instructions__reorder_instructions__overlap_reg_2_global__stg_memory_coalecesing__overlap_sts__stg_wt__reduce_IMAD_AND_LEA__reduce_MIO);
define_check_function(fp16_mma_m16n8k16_ldmatrix_trans__overlap_global_2_sm__quadra_buffer__reduce_instructions__reorder_instructions__overlap_reg_2_global__stg_memory_coalecesing__overlap_sts__stg_wt__reduce_IMAD_AND_LEA__reduce_MIO__opt_BAR);
// define_check_function(fp16_mma_m16n8k16_ldmatrix_trans__overlap_global_2_sm__octa_buffer__reduce_instructions__reorder_instructions__overlap_reg_2_global__stg_memory_coalecesing__overlap_sts);
/* clang-format on */

template<typename T, typename = std::enable_if_t<std::is_same<T, half>::value || std::is_same<T, __nv_bfloat16>::value>>
int test(const std::vector<float>& host_A,
         const std::vector<float>& host_B,
         const std::vector<float>& host_C,
         int                       M,
         int                       N,
         int                       K)
{
  std::vector<T> host_fp16_A(M * K), host_fp16_B(K * N), host_fp16_C(M * N);
  for (auto [fp32, fp16] : {std::make_pair(&host_A, &host_fp16_A),
                            std::make_pair(&host_B, &host_fp16_B),
                            std::make_pair(&host_C, &host_fp16_C)}) {
    for (int i = 0; i < fp16->size(); ++i) {
      fp16->at(i) = T(fp32->at(i));
    }
  }

  T *fp16_A, *fp16_B, *fp16_C;
  for (auto& pair : {std::make_pair(host_fp16_A, &fp16_A),
                     std::make_pair(host_fp16_B, &fp16_B),
                     std::make_pair(host_fp16_C, &fp16_C)}) {
    const std::vector<T>& host   = pair.first;
    T*&                   device = *pair.second;
    cudaMalloc(&device, sizeof(T) * host.size());
    cudaMemcpy(device, host.data(), sizeof(T) * host.size(), cudaMemcpyDefault);
    CHECK_CUDA_ERROR();
  }

  /* clang-format off */
  // fp16_mma_m16n8k16_ldmatrix___check_relative_error(fp16_A, fp16_B, fp16_C, M, N, K, host_C);
  // fp16_mma_m16n8k16_ldmatrix_trans___check_relative_error(fp16_A, fp16_B, fp16_C, M, N, K, host_C);
  // fp16_mma_m16n8k16_ldmatrix_trans__overlap_global_2_sm___check_relative_error(fp16_A, fp16_B, fp16_C, M, N, K, host_C);
  // fp16_mma_m16n8k16_ldmatrix_trans__overlap_global_2_sm__quadra_buffer___check_relative_error(fp16_A, fp16_B, fp16_C, M, N, K, host_C);
  // fp16_mma_m16n8k16_ldmatrix_trans__overlap_global_2_sm__quadra_buffer__reduce_instructions___check_relative_error(fp16_A, fp16_B, fp16_C, M, N, K, host_C);
  // fp16_mma_m16n8k16_ldmatrix_trans__overlap_global_2_sm__quadra_buffer__reduce_instructions__reorder_instructions___check_relative_error(fp16_A, fp16_B, fp16_C, M, N, K, host_C);
  // fp16_mma_m16n8k16_ldmatrix_trans__overlap_global_2_sm__quadra_buffer__reduce_instructions__reorder_instructions__overlap_reg_2_global___check_relative_error(fp16_A, fp16_B, fp16_C, M, N, K, host_C);
  // fp16_mma_m16n8k16_ldmatrix_trans__overlap_global_2_sm__quadra_buffer__reduce_instructions__reorder_instructions__overlap_reg_2_global__stg_memory_coalecesing___check_relative_error(fp16_A, fp16_B, fp16_C, M, N, K, host_C);
  fp16_mma_m16n8k16_ldmatrix_trans__overlap_global_2_sm__quadra_buffer__reduce_instructions__reorder_instructions__overlap_reg_2_global__stg_memory_coalecesing__overlap_sts__stg_wt___check_relative_error(fp16_A, fp16_B, fp16_C, M, N, K, host_C);
  fp16_mma_m16n8k16_ldmatrix_trans__overlap_global_2_sm__quadra_buffer__reduce_instructions__reorder_instructions__overlap_reg_2_global__stg_memory_coalecesing__overlap_sts__stg_wt__reduce_IMAD_AND_LEA___check_relative_error(fp16_A, fp16_B, fp16_C, M, N, K, host_C);
  fp16_mma_m16n8k16_ldmatrix_trans__overlap_global_2_sm__quadra_buffer__reduce_instructions__reorder_instructions__overlap_reg_2_global__stg_memory_coalecesing__overlap_sts__stg_wt__reduce_IMAD_AND_LEA__reduce_MIO___check_relative_error(fp16_A, fp16_B, fp16_C, M, N, K, host_C);
  fp16_mma_m16n8k16_ldmatrix_trans__overlap_global_2_sm__quadra_buffer__reduce_instructions__reorder_instructions__overlap_reg_2_global__stg_memory_coalecesing__overlap_sts__stg_wt__reduce_IMAD_AND_LEA__reduce_MIO__opt_BAR___check_relative_error(fp16_A, fp16_B, fp16_C, M, N, K, host_C);
  // fp16_mma_m16n8k16_ldmatrix_trans__overlap_global_2_sm__octa_buffer__reduce_instructions__reorder_instructions__overlap_reg_2_global__stg_memory_coalecesing__overlap_sts___check_relative_error(fp16_A, fp16_B, fp16_C, M, N, K, host_C);
  /* clang-format on */

  CHECK_CUDA_RETURN(cudaFree(fp16_A));
  CHECK_CUDA_RETURN(cudaFree(fp16_B));
  CHECK_CUDA_RETURN(cudaFree(fp16_C));

  return 0;
}

int main()
{
  static const int M = (1 << 12), N = (1 << 12), K = (1 << 12);
  // static const int M = 128, N = 128, K = 128;

  std::vector<float>                    host_A(M * K), host_B(K * N), host_C(M * N);
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
        int row            = i / K;
        int col            = i % K;
        vec->operator[](i) = (row == col);
        if (row < limit && col < limit) {
          vec->operator[](i) = row * limit + col;
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
        if (row < limit && col < limit) {
          vec->operator[](i) = row * limit + col;
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
    launch_fp32_naive_mm(A, B, C, M, N, K);
    cudaMemcpy(host_C.data(), C, sizeof(float) * host_C.size(), cudaMemcpyDefault);
    CHECK_CUDA_ERROR();
  }

  // test<half>(host_A, host_B, host_C, M, N, K);
  test<__nv_bfloat16>(host_A, host_B, host_C, M, N, K);

  CHECK_CUDA_RETURN(cudaFree(A));
  CHECK_CUDA_RETURN(cudaFree(B));
  CHECK_CUDA_RETURN(cudaFree(C));
  return 0;
}
