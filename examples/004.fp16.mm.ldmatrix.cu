#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_pipeline_primitives.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "util/error.h"
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
__global__ void fp16_mma_m16n8k16_ldmatrix(const T* A, const T* B, const T* C, int M, int N, int K)
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
      uint32_t& first  = *(uint32_t*)(&casted[0]);
      uint32_t& second = *(uint32_t*)(&casted[2]);
      uint32_t  swap   = (first ^ second) * (!(lane_id & 0x4));
      first ^= swap;
      second ^= swap;
      first = __shfl_xor_sync(0xffffffff, first, 0x4);
      swap  = (first ^ second) * (!(lane_id & 0x4));
      first ^= swap;
      second ^= swap;
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
__global__ void fp16_mma_m16n8k16_ldmatrix_trans(const T* A, const T* B, const T* C, int M, int N, int K)
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
      uint32_t& first  = *(uint32_t*)(&casted[0]);
      uint32_t& second = *(uint32_t*)(&casted[2]);
      uint32_t  swap   = (first ^ second) * (!(lane_id & 0x4));
      first ^= swap;
      second ^= swap;
      first = __shfl_xor_sync(0xffffffff, first, 0x4);
      swap  = (first ^ second) * (!(lane_id & 0x4));
      first ^= swap;
      second ^= swap;
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
__global__ void fp16_mma_m16n8k16_ldmatrix_trans__overlap_global_2_sm(const T* A, const T* B, const T* C, int M, int N, int K)
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
      uint32_t& first  = *(uint32_t*)(&casted[0]);
      uint32_t& second = *(uint32_t*)(&casted[2]);
      uint32_t  swap   = (first ^ second) * (!(lane_id & 0x4));
      first ^= swap;
      second ^= swap;
      first = __shfl_xor_sync(0xffffffff, first, 0x4);
      swap  = (first ^ second) * (!(lane_id & 0x4));
      first ^= swap;
      second ^= swap;
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
__global__ void fp16_mma_m16n8k16_ldmatrix_trans__overlap_global_2_sm__quadra_buffer(
  const T* A, const T* B, const T* C, int M, int N, int K)
{
  constexpr int WARP_COUNT   = BLOCK_TILE_M / WARP_TILE_M * BLOCK_TILE_N / WARP_TILE_N;
  constexpr int THREAD_COUNT = WARP_COUNT * 32;

  constexpr int LOOP_TILE_K         = 16;
  constexpr int LDG_SM_BUFFER_SIZE  = 4;
  constexpr int LDG_REG_BUFFER_SIZE = 2;
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

  const int     warp_id                    = threadIdx.x / 32;
  const int     lane_id                    = threadIdx.x % 32;
  constexpr int transposed_lane_id_mask[4] = {0x00, 0x18, 0x18, 0x00};
  const int     transposed_lane_id         = lane_id ^ transposed_lane_id_mask[lane_id / 8];

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

#define mma_m16n8k16_row_col_macro(d, a, b, c)                                                                               \
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
  int k_loop_offset = LOOP_TILE_K * 2;

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
      uint32_t& first  = *(uint32_t*)(&casted[0]);
      uint32_t& second = *(uint32_t*)(&casted[2]);
      uint32_t  swap   = (first ^ second) * (!(lane_id & 0x4));
      first ^= swap;
      second ^= swap;
      first = __shfl_xor_sync(0xffffffff, first, 0x4);
      swap  = (first ^ second) * (!(lane_id & 0x4));
      first ^= swap;
      second ^= swap;
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
  const T* A, const T* B, const T* C, int M, int N, int K)
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

  const int     warp_id                    = threadIdx.x / 32;
  const int     lane_id                    = threadIdx.x % 32;
  constexpr int transposed_lane_id_mask[4] = {0x00, 0x18, 0x18, 0x00};
  const int     transposed_lane_id         = lane_id ^ transposed_lane_id_mask[lane_id / 8];

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
    A_sm_ptr_for_ldg[loop] = &A_sm[A_ldg_reg_2_A_sm_partial_offset +loop * WARP_COUNT * 16 * A_sm_dim3];
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
      uint32_t& first  = *(uint32_t*)(&casted[0]);
      uint32_t& second = *(uint32_t*)(&casted[2]);
      uint32_t  swap   = (first ^ second) * (!(lane_id & 0x4));
      first ^= swap;
      second ^= swap;
      first = __shfl_xor_sync(0xffffffff, first, 0x4);
      swap  = (first ^ second) * (!(lane_id & 0x4));
      first ^= swap;
      second ^= swap;
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
__global__ void fp16_mma_m16n8k16_ldmatrix_trans__overlap_global_2_sm__quadra_buffer__reduce_instructions__reorder_instructions(
  const T* A, const T* B, const T* C, int M, int N, int K)
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

  const int     warp_id                    = threadIdx.x / 32;
  const int     lane_id                    = threadIdx.x % 32;
  constexpr int transposed_lane_id_mask[4] = {0x00, 0x18, 0x18, 0x00};
  const int     transposed_lane_id         = lane_id ^ transposed_lane_id_mask[lane_id / 8];

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
    A_sm_ptr_for_ldg[loop] = &A_sm[A_ldg_reg_2_A_sm_partial_offset +loop * WARP_COUNT * 16 * A_sm_dim3];
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
      uint32_t& first  = *(uint32_t*)(&casted[0]);
      uint32_t& second = *(uint32_t*)(&casted[2]);
      uint32_t  swap   = (first ^ second) * (!(lane_id & 0x4));
      first ^= swap;
      second ^= swap;
      first = __shfl_xor_sync(0xffffffff, first, 0x4);
      swap  = (first ^ second) * (!(lane_id & 0x4));
      first ^= swap;
      second ^= swap;
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

#define define_check_function(function)                                                                                                                  \
  template<typename T, int BLOCK_TILE_M, int BLOCK_TILE_N, int WARP_TILE_M, int WARP_TILE_N>                                                             \
  void launch_##function(const T* A, const T* B, T* C, int M, int N, int K)                                                                              \
  {                                                                                                                                                      \
    if (std::is_same<T, half>::value == false && std::is_same<T, __nv_bfloat16>::value == false) {                                                       \
      throw std::runtime_error("T is not supported.");                                                                                                   \
    }                                                                                                                                                    \
    constexpr int LOOP_TILE_K = 16;                                                                                                                      \
    if (!(M % BLOCK_TILE_M == 0 && N % BLOCK_TILE_N == 0 && K % LOOP_TILE_K == 0)) {                                                                     \
      throw std::runtime_error("M or N or K are not aligned.");                                                                                          \
    }                                                                                                                                                    \
    static_assert(8 <= BLOCK_TILE_M && BLOCK_TILE_M <= 256 && (BLOCK_TILE_M & (BLOCK_TILE_M - 1)) == 0);                                                 \
    static_assert(16 <= BLOCK_TILE_N && BLOCK_TILE_N <= 256 && (BLOCK_TILE_N & (BLOCK_TILE_N - 1)) == 0);                                                \
    static_assert(LOOP_TILE_K == 16);                                                                                                                    \
    static_assert(BLOCK_TILE_M % WARP_TILE_M == 0 && BLOCK_TILE_N % WARP_TILE_N == 0);                                                                   \
    static_assert(WARP_TILE_N % 16 == 0 && WARP_TILE_M % 8 == 0 && WARP_TILE_N / 16 == WARP_TILE_M / 8);                                                 \
    constexpr int WARP_COUNT = BLOCK_TILE_N / WARP_TILE_N * BLOCK_TILE_M / WARP_TILE_M;                                                                  \
    static_assert(1 <= WARP_COUNT && WARP_COUNT <= 32 && (WARP_COUNT & (WARP_COUNT - 1)) == 0);                                                          \
    dim3 grid(N / BLOCK_TILE_N, M / BLOCK_TILE_M);                                                                                                       \
    dim3 block(WARP_COUNT * 32);                                                                                                                         \
    function<T, BLOCK_TILE_M, BLOCK_TILE_N, WARP_TILE_M, WARP_TILE_N><<<grid, block>>>(A, B, C, M, N, K);                                                \
    CHECK_CUDA_ERROR();                                                                                                                                  \
  }                                                                                                                                                      \
  template<typename T>                                                                                                                                   \
  void function##___check_relative_error(                                                                                                                \
    const T* A, const T* B, T* C, int M, int N, int K, const std::vector<float>& base)                                                                   \
  {                                                                                                                                                      \
    std::vector<T> host_C(M* N);                                                                                                                         \
    memset(host_C.data(), 0, sizeof(T) * host_C.size());                                                                                                 \
    launch_##function<T, 128, 128, 32, 64>(A, B, C, M, N, K);                                                                                            \
    cudaMemcpy(host_C.data(), C, sizeof(T) * host_C.size(), cudaMemcpyDefault);                                                                          \
    float max_error = 0, base_value, current_value;                                                                                                      \
    int   position  = 0;                                                                                                                                 \
    for (int i = 0; i < host_C.size(); ++i) {                                                                                                            \
      if (fabs(float(host_C[i]) - base[i]) > max_error) {                                                                                                \
        max_error     = fabs(float(host_C[i]) - base[i]);                                                                                                \
        base_value    = base[i];                                                                                                                         \
        current_value = host_C[i];                                                                                                                       \
        position      = i;                                                                                                                               \
      }                                                                                                                                                  \
    }                                                                                                                                                    \
    const char* type = std::is_same<T, half>::value ? "half" : "__nv_bfloat16";                                                                          \
    const char* name = #function;                                                                                                                        \
    printf(                                                                                                                                              \
      "max_relative_error = %8.6f, max_absolute_error = %8.3f, base_value = %10.3f, current_value = %10.3f, type=%16s, function=%s\n", \
      fabs(max_error / base_value),                                                                                                                      \
      max_error,                                                                                                                                         \
      base_value,                                                                                                                                        \
      current_value,                                                                                                                                     \
      type,                                                                                                                                              \
      name);                                                                                                                                             \
  }

/* clang-format off */
define_check_function(fp16_mma_m16n8k16_ldmatrix);
define_check_function(fp16_mma_m16n8k16_ldmatrix_trans);
define_check_function(fp16_mma_m16n8k16_ldmatrix_trans__overlap_global_2_sm);
define_check_function(fp16_mma_m16n8k16_ldmatrix_trans__overlap_global_2_sm__quadra_buffer);
define_check_function(fp16_mma_m16n8k16_ldmatrix_trans__overlap_global_2_sm__quadra_buffer__reduce_instructions);
define_check_function(fp16_mma_m16n8k16_ldmatrix_trans__overlap_global_2_sm__quadra_buffer__reduce_instructions__reorder_instructions);
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
  fp16_mma_m16n8k16_ldmatrix___check_relative_error(fp16_A, fp16_B, fp16_C, M, N, K, host_C);
  fp16_mma_m16n8k16_ldmatrix_trans___check_relative_error(fp16_A, fp16_B, fp16_C, M, N, K, host_C);
  fp16_mma_m16n8k16_ldmatrix_trans__overlap_global_2_sm___check_relative_error(fp16_A, fp16_B, fp16_C, M, N, K, host_C);
  fp16_mma_m16n8k16_ldmatrix_trans__overlap_global_2_sm__quadra_buffer___check_relative_error(fp16_A, fp16_B, fp16_C, M, N, K, host_C);
  fp16_mma_m16n8k16_ldmatrix_trans__overlap_global_2_sm__quadra_buffer__reduce_instructions___check_relative_error(fp16_A, fp16_B, fp16_C, M, N, K, host_C);
  fp16_mma_m16n8k16_ldmatrix_trans__overlap_global_2_sm__quadra_buffer__reduce_instructions__reorder_instructions___check_relative_error(fp16_A, fp16_B, fp16_C, M, N, K, host_C);
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

  test<half>(host_A, host_B, host_C, M, N, K);
  test<__nv_bfloat16>(host_A, host_B, host_C, M, N, K);

  CHECK_CUDA_RETURN(cudaFree(A));
  CHECK_CUDA_RETURN(cudaFree(B));
  CHECK_CUDA_RETURN(cudaFree(C));
  return 0;
}
