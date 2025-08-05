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

const int limit = 128;

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

template<typename T, int BLOCK_TILE_M, int BLOCK_TILE_N, int WARP_TILE_M, int WARP_TILE_N, int TILE_K>
__global__ void llmmm_fp16_mma_m8n8k4(const T* A, const T* B, T* C, int M, int N, int K)
{
  static_assert(std::is_same<T, half>::value);
  const int     warp_id                   = threadIdx.x / 32;
  const int     lane_id                   = threadIdx.x % 32;
  const int     mma_m8n8k4_computation_id = lane_id % 16 / 4;
  const int     mma_m8n8k4_lane_id        = lane_id % 4 + lane_id / 16 * 4;
  const int     mma_m8n8k4_group_id       = lane_id / 16;  // low or high
  constexpr int M_COMPUTATION_COUNT       = WARP_TILE_M / 8;
  constexpr int N_COMPUTATION_COUNT       = WARP_TILE_N / 8;
  static_assert(M_COMPUTATION_COUNT * N_COMPUTATION_COUNT == 4);

  const int m_block_offset = blockIdx.y * BLOCK_TILE_M;
  const int n_block_offset = blockIdx.x * BLOCK_TILE_N;
  const int m_warp_offset  = warp_id * WARP_TILE_M;

  static_assert(TILE_K % 8 == 0);
  static_assert(BLOCK_TILE_N % 8 == 0);
  __shared__ T A_sm[TILE_K / 8][BLOCK_TILE_M][8];
  __shared__ T B_sm[BLOCK_TILE_N / 8][TILE_K][8];

  constexpr int WARP_COUNT   = BLOCK_TILE_M / WARP_TILE_M;
  constexpr int THREAD_COUNT = WARP_COUNT * 32;

  static_assert(TILE_K * BLOCK_TILE_M / 2 % THREAD_COUNT == 0);
  static_assert(TILE_K * BLOCK_TILE_N / 2 % THREAD_COUNT == 0);

  constexpr int A_LDG_REG_COUNT = TILE_K * BLOCK_TILE_M / THREAD_COUNT;
  constexpr int B_LDG_REG_COUNT = TILE_K * BLOCK_TILE_N / THREAD_COUNT;
  T             A_ldg_reg[A_LDG_REG_COUNT];
  T             B_ldg_reg[B_LDG_REG_COUNT];
  static_assert(A_LDG_REG_COUNT % 8 == 0);
  static_assert(B_LDG_REG_COUNT % 8 == 0);
  constexpr int A_LDG_LOOP_COUNT = A_LDG_REG_COUNT / 8;
  constexpr int B_LDG_LOOP_COUNT = B_LDG_REG_COUNT / 8;

  T     A_mma_reg[2][4];
  T     B_mma_reg[4];
  float C_mma_reg[BLOCK_TILE_N / WARP_TILE_N][8] = {0};

  for (int k_offset = 0; k_offset < K; k_offset += TILE_K) {
    for (int loop = 0; loop < A_LDG_LOOP_COUNT; ++loop) {
      const int index = (loop * THREAD_COUNT + threadIdx.x) * 8;
      const int m     = index / TILE_K;
      const int k     = index % TILE_K;
      FETCH_FLOAT4(A_ldg_reg[loop * 8], A[OFFSET(m_block_offset + m, k_offset + k, K)]);
      STORE_FLOAT4(A_sm[k / 8][m][0], A_ldg_reg[loop * 8]);
    }

    for (int loop = 0; loop < B_LDG_LOOP_COUNT; ++loop) {
      const int index = (loop * THREAD_COUNT + threadIdx.x) * 8;
      const int n     = index % BLOCK_TILE_N;
      const int k     = index / BLOCK_TILE_N;
      FETCH_FLOAT4(B_ldg_reg[loop * 8], B[OFFSET(k_offset + k, n_block_offset + n, N)]);
      STORE_FLOAT4(B_sm[n / 8][k][0], B_ldg_reg[loop * 8]);
    }

    __syncthreads();

#pragma unroll
    for (int mma_k_offset = 0; mma_k_offset < TILE_K; mma_k_offset += 8) {
      FETCH_FLOAT4(A_mma_reg[0],
                   A_sm[mma_k_offset / 8]
                       [m_warp_offset + mma_m8n8k4_computation_id / M_COMPUTATION_COUNT * 8 + mma_m8n8k4_lane_id][0]);
#pragma unroll
      for (int mma_n_offset = 0; mma_n_offset < BLOCK_TILE_N; mma_n_offset += WARP_TILE_N) {
        FETCH_FLOAT(B_mma_reg[0],
                    B_sm[mma_n_offset / 8 + mma_m8n8k4_computation_id % M_COMPUTATION_COUNT]
                        [mma_k_offset + mma_m8n8k4_lane_id % 4][mma_m8n8k4_group_id * 2]);
        FETCH_FLOAT(B_mma_reg[2],
                    B_sm[mma_n_offset / 8 + mma_m8n8k4_computation_id % M_COMPUTATION_COUNT]
                        [mma_k_offset + mma_m8n8k4_lane_id % 4][mma_m8n8k4_group_id * 2 + 4]);
        mma_sync_aligned_m8n8k4_row_row_f32_f16_f16_f32<T>(
          C_mma_reg[mma_n_offset / WARP_TILE_N], A_mma_reg[0], B_mma_reg, C_mma_reg[mma_n_offset / WARP_TILE_N]);

        FETCH_FLOAT(B_mma_reg[0],
                    B_sm[mma_n_offset / 8 + mma_m8n8k4_computation_id % M_COMPUTATION_COUNT]
                        [mma_k_offset + mma_m8n8k4_lane_id % 4 + 4][mma_m8n8k4_group_id * 2]);
        FETCH_FLOAT(B_mma_reg[2],
                    B_sm[mma_n_offset / 8 + mma_m8n8k4_computation_id % M_COMPUTATION_COUNT]
                        [mma_k_offset + mma_m8n8k4_lane_id % 4 + 4][mma_m8n8k4_group_id * 2 + 4]);
        mma_sync_aligned_m8n8k4_row_row_f32_f16_f16_f32<T>(
          C_mma_reg[mma_n_offset / WARP_TILE_N], A_mma_reg[1], B_mma_reg, C_mma_reg[mma_n_offset / WARP_TILE_N]);
      }
    }
    __syncthreads();
  }
  const int m_computation_offset = mma_m8n8k4_computation_id / M_COMPUTATION_COUNT * 8;
  const int n_computation_stride = mma_m8n8k4_computation_id % N_COMPUTATION_COUNT * 8;
  int       m_global             = m_block_offset + m_warp_offset + m_computation_offset + (mma_m8n8k4_lane_id & 0xfd);
  T         C_reg[8];
  for (int n_offset = 0; n_offset < BLOCK_TILE_N; n_offset += WARP_TILE_N) {
    for (int i = 0; i < 8; ++i) {
      C_reg[i] = C_mma_reg[n_offset / WARP_TILE_N][i];
    }
    int n_global = n_block_offset + n_offset + (mma_m8n8k4_lane_id & 0x2) * 2 + n_computation_stride;
    STORE_FLOAT2(C[OFFSET(m_global, n_global, N)], C_reg[0]);
    STORE_FLOAT2(C[OFFSET(m_global + 2, n_global, N)], C_reg[4]);
  }
}

template<typename T>
void launch_llmmm_fp16_mma_m8n8k4(const T* A, const T* B, T* C, int M, int N, int K)
{
  constexpr int BLOCK_TILE_M = 128;
  constexpr int BLOCK_TILE_N = 128;
  constexpr int TILE_K       = 16;
  if (!(M % BLOCK_TILE_M == 0 && N % BLOCK_TILE_N == 0 && K % TILE_K == 0)) {
    throw std::runtime_error("M or N or K are not aligned.");
  }
  static_assert(16 <= BLOCK_TILE_M && BLOCK_TILE_M <= 128 && (BLOCK_TILE_M & (BLOCK_TILE_M - 1)) == 0);
  static_assert(16 <= BLOCK_TILE_N && BLOCK_TILE_N <= 256 && (BLOCK_TILE_N & (BLOCK_TILE_N - 1)) == 0);
  static_assert(TILE_K % 4 == 0 && 16 <= TILE_K && TILE_K <= 128 && ((TILE_K & (TILE_K - 1)) == 0));
  constexpr int m_per_warp = 16;
  constexpr int n_per_warp = 16;
  static_assert(BLOCK_TILE_M % m_per_warp == 0 && BLOCK_TILE_N % n_per_warp == 0);
  static_assert(m_per_warp == 8 || m_per_warp == 16 || m_per_warp == 32);
  static_assert(n_per_warp == 8 || n_per_warp == 16 || n_per_warp == 32);
  static_assert(m_per_warp * n_per_warp == 8 * 8 * 4);
  constexpr int warp_count = BLOCK_TILE_M / m_per_warp;
  static_assert(1 <= warp_count && warp_count <= 16 && (warp_count & (warp_count - 1)) == 0);
  dim3 grid(N / BLOCK_TILE_N, M / BLOCK_TILE_M);
  dim3 block(warp_count * 32);
  if constexpr (std::is_same<T, half>::value) {
    llmmm_fp16_mma_m8n8k4<T, BLOCK_TILE_M, BLOCK_TILE_N, m_per_warp, n_per_warp, TILE_K>
      <<<grid, block>>>(A, B, C, M, N, K);
  }
  CHECK_CUDA_ERROR();
}

template<typename T, int BLOCK_TILE_M, int BLOCK_TILE_N, int LOOP_TILE_M, int WARP_TILE_N, int LOOP_TILE_K>
__global__ void llmmm_fp16_mma_m16n8k16(const T* A, const T* B, T* C, int M, int N, int K)
{
  static_assert(std::is_same<T, half>::value || std::is_same<T, __nv_bfloat16>::value);
  static_assert(LOOP_TILE_M == 8);
  static_assert(WARP_TILE_N == 16);
  static_assert(LOOP_TILE_K == 16);
  constexpr int WARP_COUNT   = BLOCK_TILE_N / WARP_TILE_N;
  constexpr int THREAD_COUNT = WARP_COUNT * 32;

  __shared__ T A_sm[2][BLOCK_TILE_M][LOOP_TILE_K / 2];
  __shared__ T B_sm[BLOCK_TILE_N / 8][LOOP_TILE_K / 2][8][2];

  constexpr int A_ELEMENT_COUNT_PER_BLOCK = BLOCK_TILE_M * LOOP_TILE_K;
  constexpr int A_ELEMENT_COUNT_PER_THREAD =
    A_ELEMENT_COUNT_PER_BLOCK / THREAD_COUNT < 4 ? 4 : A_ELEMENT_COUNT_PER_BLOCK / THREAD_COUNT;
  static_assert(A_ELEMENT_COUNT_PER_THREAD % 4 == 0);
  constexpr int A_LDG_LOOP_COUNT = A_ELEMENT_COUNT_PER_THREAD / 4;
  T             A_ldg_reg[A_ELEMENT_COUNT_PER_THREAD];

  constexpr int B_ELEMENT_COUNT_PER_BLOCK  = BLOCK_TILE_N * LOOP_TILE_K;
  constexpr int B_ELEMENT_COUNT_PER_THREAD = B_ELEMENT_COUNT_PER_BLOCK / THREAD_COUNT;
  static_assert(B_ELEMENT_COUNT_PER_THREAD == 8);
  T B_ldg_reg[B_ELEMENT_COUNT_PER_THREAD];

  T     A_mma_reg[4];
  T     B_mma_reg[8];
  float C_mma_reg[BLOCK_TILE_M / LOOP_TILE_M][4] = {0};

  const int warp_id        = threadIdx.x / 32;
  const int lane_id        = threadIdx.x % 32;
  const int m_block_offset = blockIdx.y * BLOCK_TILE_M;
  const int n_block_offset = blockIdx.x * BLOCK_TILE_N;
  const int n_warp_offset  = warp_id * WARP_TILE_N;

  for (int k_loop_offset = 0; k_loop_offset < K; k_loop_offset += LOOP_TILE_K) {
    for (int loop = 0; loop < A_LDG_LOOP_COUNT; ++loop) {
      int m = (loop * WARP_COUNT + warp_id) * 8 + lane_id % 16 / 2;
      int k = lane_id / 16 * 8 + lane_id % 2 * 4;
      FETCH_FLOAT2(A_ldg_reg[loop * 4], A[OFFSET((m_block_offset + m) % M, k_loop_offset + k, K)]);
    }
    {
      int k = ((warp_id & 1) * 8) + (lane_id & 0x6);
      int n = warp_id / 2 * 32 + (lane_id & 0xf8) + (lane_id & 1) * 4;
      FETCH_FLOAT2(B_ldg_reg[0], B[OFFSET(k_loop_offset + k, n_block_offset + n, N)]);
      FETCH_FLOAT2(B_ldg_reg[4], B[OFFSET(k_loop_offset + k + 1, n_block_offset + n, N)]);
    }
    for (int loop = 0; loop < A_LDG_LOOP_COUNT; ++loop) {
      int m = (loop * WARP_COUNT + warp_id) * 8 + lane_id % 16 / 2;
      int k = lane_id / 16 * 8 + lane_id % 2 * 4;
      STORE_FLOAT2(A_sm[k / 8][m][k % 8], A_ldg_reg[loop * 4]);
      // for (int i = 0; false && i < THREAD_COUNT; ++i) {
      //   if (this_thread_can_log(i) && k_loop_offset == 0) {
      //     printf(
      //       "A_ldg_reg, loop = %03d, threadIdx.x = %03d, warp_id = %03d, lane_id = %03d, m = %03d, k = %03d,
      //       m%03dk%03d m%03dk%03d m%03dk%03d m%03dk%03d\n", loop, threadIdx.x, warp_id, lane_id, m, k,
      //       int(A_ldg_reg[loop * 4]) / limit,
      //       int(A_ldg_reg[loop * 4]) % limit,
      //       int(A_ldg_reg[loop * 4 + 1]) / limit,
      //       int(A_ldg_reg[loop * 4 + 1]) % limit,
      //       int(A_ldg_reg[loop * 4 + 2]) / limit,
      //       int(A_ldg_reg[loop * 4 + 2]) % limit,
      //       int(A_ldg_reg[loop * 4 + 3]) / limit,
      //       int(A_ldg_reg[loop * 4 + 3]) % limit);
      //   }
      //   __syncthreads();
      // }
    }
    {
      int k            = ((warp_id & 1) * 8) + (lane_id & 0x6);
      int n            = warp_id / 2 * 32 + (lane_id & 0xf8) + (lane_id & 1) * 4;
      T   transpose[8] = {
        B_ldg_reg[0], B_ldg_reg[4], B_ldg_reg[1], B_ldg_reg[5], B_ldg_reg[2], B_ldg_reg[6], B_ldg_reg[3], B_ldg_reg[7]};
      STORE_FLOAT4(B_sm[n / 8][k / 2][n % 8][0], transpose[0]);
      // for (int i = 0; 0 && i < THREAD_COUNT; ++i) {
      //   if (this_thread_can_log(i) && k_loop_offset == 0) {
      //     printf(
      //       "B_ldg_reg, threadIdx.x = %03d, warp_id = %03d, lane_id = %03d, n = %03d, k = %03d, n%03dk%03d n%03dk%03d
      //       n%03dk%03d n%03dk%03d n%03dk%03d n%03dk%03d n%03dk%03d nn%03dk%03d\n", threadIdx.x, warp_id, lane_id, n,
      //       k,
      //       int(transpose[0]) % limit,
      //       int(transpose[0]) / limit,
      //       int(transpose[1]) % limit,
      //       int(transpose[1]) / limit,
      //       int(transpose[2]) % limit,
      //       int(transpose[2]) / limit,
      //       int(transpose[3]) % limit,
      //       int(transpose[3]) / limit,
      //       int(transpose[4]) % limit,
      //       int(transpose[4]) / limit,
      //       int(transpose[5]) % limit,
      //       int(transpose[5]) / limit,
      //       int(transpose[6]) % limit,
      //       int(transpose[6]) / limit,
      //       int(transpose[7]) % limit,
      //       int(transpose[7]) / limit);
      //   }
      //   __syncthreads();
      // }
    }

    // if (this_thread_can_log() && k_loop_offset == 0) {
    //   const T* output = &A_sm[0][0][0];
    //   for (int i = 0; i < BLOCK_TILE_M * LOOP_TILE_K; ++i) {
    //     if (i % 64 == 0) {
    //       printf("\n A_sm, i = %03d ", i / 64);
    //     }
    //     printf("m%03dk%03d ", int(output[i]) / limit, int(output[i]) % limit);
    //   }
    //   printf("\n");
    // }
    // if (this_thread_can_log() && k_loop_offset == 0) {
    //   const T* output = &data.compute.B[0][0][0][0];
    //   for (int i = 0; i < BLOCK_TILE_N * LOOP_TILE_K; ++i) {
    //     if (i % 64 == 0) {
    //       printf("\n B_sm, i = %03d ", i / 64);
    //     }
    //     printf("n%03dk%03d ", int(output[i]) % limit, int(output[i]) / limit);
    //   }
    //   printf("\n");
    // }

    __syncthreads();

    const int mma_n = n_warp_offset + lane_id / 4;
    const int mma_k = lane_id % 4 * 2;
    FETCH_FLOAT(B_mma_reg[0], B_sm[n_warp_offset / 8][mma_k / 2][mma_n % 8][0]);
    FETCH_FLOAT(B_mma_reg[2], B_sm[(n_warp_offset + 8) / 8][mma_k / 2][mma_n % 8][0]);
    FETCH_FLOAT(B_mma_reg[4], B_sm[n_warp_offset / 8][(mma_k + 8) / 2][mma_n % 8][0]);
    FETCH_FLOAT(B_mma_reg[6], B_sm[(n_warp_offset + 8) / 8][(mma_k + 8) / 2][mma_n % 8][0]);

    for (int m_loop_offset = 0; m_loop_offset < BLOCK_TILE_M; m_loop_offset += LOOP_TILE_M) {
      const int mma_m = m_loop_offset + lane_id / 4;
      const int k     = lane_id % 4 * 2;
      FETCH_FLOAT(A_mma_reg[0], A_sm[0][mma_m][k]);
      FETCH_FLOAT(A_mma_reg[2], A_sm[1][mma_m][k]);
      mma_m16n8k16_row_col(
        C_mma_reg[m_loop_offset / LOOP_TILE_M], B_mma_reg, A_mma_reg, C_mma_reg[m_loop_offset / LOOP_TILE_M]);
      // if (k_loop_offset == 0 && m_loop_offset == 0) {
      //   for (int i = 0; i < 32; ++i) {
      //     if (this_thread_can_log(i)) {
      //       printf(
      //         "threadIdx = %03d, B_mma_reg = %8.3f, %8.3f, %8.3f, %8.3f, %8.3f, %8.3f, %8.3f, %8.3f, A_mma_reg =
      //         %8.3f, %8.3f, %8.3f, %8.3f\n", threadIdx.x, float(B_mma_reg[0]), float(B_mma_reg[1]),
      //         float(B_mma_reg[2]),
      //         float(B_mma_reg[3]),
      //         float(B_mma_reg[4]),
      //         float(B_mma_reg[5]),
      //         float(B_mma_reg[6]),
      //         float(B_mma_reg[7]),
      //         float(A_mma_reg[0]),
      //         float(A_mma_reg[1]),
      //         float(A_mma_reg[2]),
      //         float(A_mma_reg[3]));
      //     }
      //     __syncthreads();
      //   }
      // }
    }
    __syncthreads();
  }

  for (int m_loop = 0; m_loop < BLOCK_TILE_M / LOOP_TILE_M; m_loop++) {
    T casted[4] = {
      C_mma_reg[m_loop][0],
      C_mma_reg[m_loop][1],
      C_mma_reg[m_loop][2],
      C_mma_reg[m_loop][3],
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
    int              m                  = m_block_offset + m_loop * LOOP_TILE_M + lane_id % 4 * 2;
    int              n                  = n_block_offset + n_warp_offset + lane_2_n_offset[lane_id / 4];
    STORE_FLOAT(C[OFFSET(m, n, N)], store[0]);
    STORE_FLOAT(C[OFFSET(m + 1, n, N)], store[2]);
    // for (int i = 0; i < 32; ++i) {
    //   if (this_thread_can_log(i)) {
    //     printf(
    //       "C_mma_reg, threadIdx = %03d, m_loop_offset = %03d, m = %03d, n = %03d, m%03dn%03d m%03dn%03d m%03dn%03d
    //       m%03dn%03d\n", threadIdx.x, m_loop * LOOP_TILE_M, m, n, int(store[0]) / limit, int(store[0]) % limit,
    //       int(store[1]) / limit,
    //       int(store[1]) % limit,
    //       int(store[2]) / limit,
    //       int(store[2]) % limit,
    //       int(store[3]) / limit,
    //       int(store[3]) % limit);
    //   }
    //   __syncthreads();
    // }
  }
}

template<typename T, int BLOCK_TILE_M, int BLOCK_TILE_N, int LOOP_TILE_M, int WARP_TILE_N, int LOOP_TILE_K>
__global__ void llmmm_fp16_mma_m16n8k16__overlap_global_2_sm(const T* A, const T* B, T* C, int M, int N, int K)
{
  static_assert(std::is_same<T, half>::value || std::is_same<T, __nv_bfloat16>::value);
  static_assert(LOOP_TILE_M == 8);
  static_assert(WARP_TILE_N == 16);
  static_assert(LOOP_TILE_K == 16);
  constexpr int WARP_COUNT   = BLOCK_TILE_N / WARP_TILE_N;
  constexpr int THREAD_COUNT = WARP_COUNT * 32;

  __shared__ T A_sm[2][2][BLOCK_TILE_M][LOOP_TILE_K / 2];
  __shared__ T B_sm[2][BLOCK_TILE_N / 8][LOOP_TILE_K / 2][8][2];

  constexpr int A_ELEMENT_COUNT_PER_BLOCK = BLOCK_TILE_M * LOOP_TILE_K;
  constexpr int A_ELEMENT_COUNT_PER_THREAD =
    A_ELEMENT_COUNT_PER_BLOCK / THREAD_COUNT < 4 ? 4 : A_ELEMENT_COUNT_PER_BLOCK / THREAD_COUNT;
  static_assert(A_ELEMENT_COUNT_PER_THREAD % 4 == 0);
  constexpr int A_LDG_LOOP_COUNT = A_ELEMENT_COUNT_PER_THREAD / 4;
  T             A_ldg_reg[A_ELEMENT_COUNT_PER_THREAD];

  constexpr int B_ELEMENT_COUNT_PER_BLOCK  = BLOCK_TILE_N * LOOP_TILE_K;
  constexpr int B_ELEMENT_COUNT_PER_THREAD = B_ELEMENT_COUNT_PER_BLOCK / THREAD_COUNT;
  static_assert(B_ELEMENT_COUNT_PER_THREAD == 8);
  T B_ldg_reg[B_ELEMENT_COUNT_PER_THREAD];

  T     A_mma_reg[4];
  T     B_mma_reg[8];
  float C_mma_reg[BLOCK_TILE_M / LOOP_TILE_M][4] = {0};

  const int warp_id        = threadIdx.x / 32;
  const int lane_id        = threadIdx.x % 32;
  const int m_block_offset = blockIdx.y * BLOCK_TILE_M;
  const int n_block_offset = blockIdx.x * BLOCK_TILE_N;
  const int n_warp_offset  = warp_id * WARP_TILE_N;

#define global_2_ldg_reg()                                                                                             \
  for (int loop = 0; loop < A_LDG_LOOP_COUNT; ++loop) {                                                                \
    int m = (loop * WARP_COUNT + warp_id) * 8 + lane_id % 16 / 2;                                                      \
    int k = lane_id / 16 * 8 + lane_id % 2 * 4;                                                                        \
    FETCH_FLOAT2(A_ldg_reg[loop * 4], A[OFFSET((m_block_offset + m) % M, k_loop_offset + k, K)]);                      \
  }                                                                                                                    \
  {                                                                                                                    \
    int k = ((warp_id & 1) * 8) + (lane_id & 0x6);                                                                     \
    int n = warp_id / 2 * 32 + (lane_id & 0xf8) + (lane_id & 1) * 4;                                                   \
    FETCH_FLOAT2(B_ldg_reg[0], B[OFFSET(k_loop_offset + k, n_block_offset + n, N)]);                                   \
    FETCH_FLOAT2(B_ldg_reg[4], B[OFFSET(k_loop_offset + k + 1, n_block_offset + n, N)]);                               \
  }

#define ldg_reg_2_shared()                                                                                               \
  for (int loop = 0; loop < A_LDG_LOOP_COUNT; ++loop) {                                                                  \
    int m = (loop * WARP_COUNT + warp_id) * 8 + lane_id % 16 / 2;                                                        \
    int k = lane_id / 16 * 8 + lane_id % 2 * 4;                                                                          \
    STORE_FLOAT2(A_sm[SM_LDG_INDEX][k / 8][m][k % 8], A_ldg_reg[loop * 4]);                                              \
  }                                                                                                                      \
  {                                                                                                                      \
    int k            = ((warp_id & 1) * 8) + (lane_id & 0x6);                                                            \
    int n            = warp_id / 2 * 32 + (lane_id & 0xf8) + (lane_id & 1) * 4;                                          \
    T   transpose[8] = {                                                                                                 \
      B_ldg_reg[0], B_ldg_reg[4], B_ldg_reg[1], B_ldg_reg[5], B_ldg_reg[2], B_ldg_reg[6], B_ldg_reg[3], B_ldg_reg[7]}; \
    STORE_FLOAT4(B_sm[SM_LDG_INDEX][n / 8][k / 2][n % 8][0], transpose[0]);                                              \
  }

  {
    constexpr int k_loop_offset = 0;
    global_2_ldg_reg();
  }
  {
    constexpr int SM_LDG_INDEX = 0;
    ldg_reg_2_shared();
  }
  int k_loop_offset = LOOP_TILE_K;
  int SM_COM_INDEX = 0;
  int SM_LDG_INDEX = 1;
  while (k_loop_offset < K) {
    global_2_ldg_reg();
    __syncthreads();

    const int mma_n = n_warp_offset + lane_id / 4;
    const int mma_k = lane_id % 4 * 2;
    FETCH_FLOAT(B_mma_reg[0], B_sm[SM_COM_INDEX][n_warp_offset / 8][mma_k / 2][mma_n % 8][0]);
    FETCH_FLOAT(B_mma_reg[2], B_sm[SM_COM_INDEX][(n_warp_offset + 8) / 8][mma_k / 2][mma_n % 8][0]);
    FETCH_FLOAT(B_mma_reg[4], B_sm[SM_COM_INDEX][n_warp_offset / 8][(mma_k + 8) / 2][mma_n % 8][0]);
    FETCH_FLOAT(B_mma_reg[6], B_sm[SM_COM_INDEX][(n_warp_offset + 8) / 8][(mma_k + 8) / 2][mma_n % 8][0]);

    for (int m_loop_offset = 0; m_loop_offset < BLOCK_TILE_M; m_loop_offset += LOOP_TILE_M) {
      const int mma_m = m_loop_offset + lane_id / 4;
      const int k     = lane_id % 4 * 2;
      FETCH_FLOAT(A_mma_reg[0], A_sm[SM_COM_INDEX][0][mma_m][k]);
      FETCH_FLOAT(A_mma_reg[2], A_sm[SM_COM_INDEX][1][mma_m][k]);
      mma_m16n8k16_row_col(
        C_mma_reg[m_loop_offset / LOOP_TILE_M], B_mma_reg, A_mma_reg, C_mma_reg[m_loop_offset / LOOP_TILE_M]);
    }
    ldg_reg_2_shared();
    k_loop_offset += LOOP_TILE_K;
    SM_COM_INDEX ^= 1;
    SM_LDG_INDEX ^= 1;
  }
  __syncthreads();
  {
    const int mma_n = n_warp_offset + lane_id / 4;
    const int mma_k = lane_id % 4 * 2;
    FETCH_FLOAT(B_mma_reg[0], B_sm[SM_COM_INDEX][n_warp_offset / 8][mma_k / 2][mma_n % 8][0]);
    FETCH_FLOAT(B_mma_reg[2], B_sm[SM_COM_INDEX][(n_warp_offset + 8) / 8][mma_k / 2][mma_n % 8][0]);
    FETCH_FLOAT(B_mma_reg[4], B_sm[SM_COM_INDEX][n_warp_offset / 8][(mma_k + 8) / 2][mma_n % 8][0]);
    FETCH_FLOAT(B_mma_reg[6], B_sm[SM_COM_INDEX][(n_warp_offset + 8) / 8][(mma_k + 8) / 2][mma_n % 8][0]);

    for (int m_loop_offset = 0; m_loop_offset < BLOCK_TILE_M; m_loop_offset += LOOP_TILE_M) {
      const int mma_m = m_loop_offset + lane_id / 4;
      const int k     = lane_id % 4 * 2;
      FETCH_FLOAT(A_mma_reg[0], A_sm[SM_COM_INDEX][0][mma_m][k]);
      FETCH_FLOAT(A_mma_reg[2], A_sm[SM_COM_INDEX][1][mma_m][k]);
      mma_m16n8k16_row_col(
        C_mma_reg[m_loop_offset / LOOP_TILE_M], B_mma_reg, A_mma_reg, C_mma_reg[m_loop_offset / LOOP_TILE_M]);
    }
  }

  for (int m_loop = 0; m_loop < BLOCK_TILE_M / LOOP_TILE_M; m_loop++) {
    T casted[4] = {
      C_mma_reg[m_loop][0],
      C_mma_reg[m_loop][1],
      C_mma_reg[m_loop][2],
      C_mma_reg[m_loop][3],
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
    int              m                  = m_block_offset + m_loop * LOOP_TILE_M + lane_id % 4 * 2;
    int              n                  = n_block_offset + n_warp_offset + lane_2_n_offset[lane_id / 4];
    STORE_FLOAT(C[OFFSET(m, n, N)], store[0]);
    STORE_FLOAT(C[OFFSET(m + 1, n, N)], store[2]);
  }
#undef global_2_ldg_reg
#undef ldg_reg_2_shared
}

template<typename T, int BLOCK_TILE_M, int BLOCK_TILE_N, int LOOP_TILE_M, int WARP_TILE_N, int LOOP_TILE_K>
__global__ void llmmm_fp16_mma_m16n8k16__overlap_global_2_sm__st_global_wt(const T* A, const T* B, T* C, int M, int N, int K)
{
  static_assert(std::is_same<T, half>::value || std::is_same<T, __nv_bfloat16>::value);
  static_assert(LOOP_TILE_M == 8);
  static_assert(WARP_TILE_N == 16);
  static_assert(LOOP_TILE_K == 16);
  constexpr int WARP_COUNT   = BLOCK_TILE_N / WARP_TILE_N;
  constexpr int THREAD_COUNT = WARP_COUNT * 32;

  __shared__ T A_sm[2][2][BLOCK_TILE_M][LOOP_TILE_K / 2];
  __shared__ T B_sm[2][BLOCK_TILE_N / 8][LOOP_TILE_K / 2][8][2];

  constexpr int A_ELEMENT_COUNT_PER_BLOCK = BLOCK_TILE_M * LOOP_TILE_K;
  constexpr int A_ELEMENT_COUNT_PER_THREAD =
    A_ELEMENT_COUNT_PER_BLOCK / THREAD_COUNT < 4 ? 4 : A_ELEMENT_COUNT_PER_BLOCK / THREAD_COUNT;
  static_assert(A_ELEMENT_COUNT_PER_THREAD % 4 == 0);
  constexpr int A_LDG_LOOP_COUNT = A_ELEMENT_COUNT_PER_THREAD / 4;
  T             A_ldg_reg[A_ELEMENT_COUNT_PER_THREAD];

  constexpr int B_ELEMENT_COUNT_PER_BLOCK  = BLOCK_TILE_N * LOOP_TILE_K;
  constexpr int B_ELEMENT_COUNT_PER_THREAD = B_ELEMENT_COUNT_PER_BLOCK / THREAD_COUNT;
  static_assert(B_ELEMENT_COUNT_PER_THREAD == 8);
  T B_ldg_reg[B_ELEMENT_COUNT_PER_THREAD];

  T     A_mma_reg[4];
  T     B_mma_reg[8];
  float C_mma_reg[BLOCK_TILE_M / LOOP_TILE_M][4] = {0};

  const int warp_id        = threadIdx.x / 32;
  const int lane_id        = threadIdx.x % 32;
  const int m_block_offset = blockIdx.y * BLOCK_TILE_M;
  const int n_block_offset = blockIdx.x * BLOCK_TILE_N;
  const int n_warp_offset  = warp_id * WARP_TILE_N;

#define global_2_ldg_reg()                                                                                             \
  for (int loop = 0; loop < A_LDG_LOOP_COUNT; ++loop) {                                                                \
    int m = (loop * WARP_COUNT + warp_id) * 8 + lane_id % 16 / 2;                                                      \
    int k = lane_id / 16 * 8 + lane_id % 2 * 4;                                                                        \
    FETCH_FLOAT2(A_ldg_reg[loop * 4], A[OFFSET((m_block_offset + m) % M, k_loop_offset + k, K)]);                      \
  }                                                                                                                    \
  {                                                                                                                    \
    int k = ((warp_id & 1) * 8) + (lane_id & 0x6);                                                                     \
    int n = warp_id / 2 * 32 + (lane_id & 0xf8) + (lane_id & 1) * 4;                                                   \
    FETCH_FLOAT2(B_ldg_reg[0], B[OFFSET(k_loop_offset + k, n_block_offset + n, N)]);                                   \
    FETCH_FLOAT2(B_ldg_reg[4], B[OFFSET(k_loop_offset + k + 1, n_block_offset + n, N)]);                               \
  }

#define ldg_reg_2_shared()                                                                                               \
  for (int loop = 0; loop < A_LDG_LOOP_COUNT; ++loop) {                                                                  \
    int m = (loop * WARP_COUNT + warp_id) * 8 + lane_id % 16 / 2;                                                        \
    int k = lane_id / 16 * 8 + lane_id % 2 * 4;                                                                          \
    STORE_FLOAT2(A_sm[SM_LDG_INDEX][k / 8][m][k % 8], A_ldg_reg[loop * 4]);                                              \
  }                                                                                                                      \
  {                                                                                                                      \
    int k            = ((warp_id & 1) * 8) + (lane_id & 0x6);                                                            \
    int n            = warp_id / 2 * 32 + (lane_id & 0xf8) + (lane_id & 1) * 4;                                          \
    T   transpose[8] = {                                                                                                 \
      B_ldg_reg[0], B_ldg_reg[4], B_ldg_reg[1], B_ldg_reg[5], B_ldg_reg[2], B_ldg_reg[6], B_ldg_reg[3], B_ldg_reg[7]}; \
    STORE_FLOAT4(B_sm[SM_LDG_INDEX][n / 8][k / 2][n % 8][0], transpose[0]);                                              \
  }

  {
    constexpr int k_loop_offset = 0;
    global_2_ldg_reg();
  }
  {
    constexpr int SM_LDG_INDEX = 0;
    ldg_reg_2_shared();
  }
  int k_loop_offset = LOOP_TILE_K;
  int SM_COM_INDEX  = 0;
  int SM_LDG_INDEX  = 1;
  while (k_loop_offset < K) {
    global_2_ldg_reg();
    __syncthreads();

    const int mma_n = n_warp_offset + lane_id / 4;
    const int mma_k = lane_id % 4 * 2;
    FETCH_FLOAT(B_mma_reg[0], B_sm[SM_COM_INDEX][n_warp_offset / 8][mma_k / 2][mma_n % 8][0]);
    FETCH_FLOAT(B_mma_reg[2], B_sm[SM_COM_INDEX][(n_warp_offset + 8) / 8][mma_k / 2][mma_n % 8][0]);
    FETCH_FLOAT(B_mma_reg[4], B_sm[SM_COM_INDEX][n_warp_offset / 8][(mma_k + 8) / 2][mma_n % 8][0]);
    FETCH_FLOAT(B_mma_reg[6], B_sm[SM_COM_INDEX][(n_warp_offset + 8) / 8][(mma_k + 8) / 2][mma_n % 8][0]);

    for (int m_loop_offset = 0; m_loop_offset < BLOCK_TILE_M; m_loop_offset += LOOP_TILE_M) {
      const int mma_m = m_loop_offset + lane_id / 4;
      const int k     = lane_id % 4 * 2;
      FETCH_FLOAT(A_mma_reg[0], A_sm[SM_COM_INDEX][0][mma_m][k]);
      FETCH_FLOAT(A_mma_reg[2], A_sm[SM_COM_INDEX][1][mma_m][k]);
      mma_m16n8k16_row_col(
        C_mma_reg[m_loop_offset / LOOP_TILE_M], B_mma_reg, A_mma_reg, C_mma_reg[m_loop_offset / LOOP_TILE_M]);
    }
    ldg_reg_2_shared();
    k_loop_offset += LOOP_TILE_K;
    SM_COM_INDEX ^= 1;
    SM_LDG_INDEX ^= 1;
  }
  __syncthreads();
  {
    const int mma_n = n_warp_offset + lane_id / 4;
    const int mma_k = lane_id % 4 * 2;
    FETCH_FLOAT(B_mma_reg[0], B_sm[SM_COM_INDEX][n_warp_offset / 8][mma_k / 2][mma_n % 8][0]);
    FETCH_FLOAT(B_mma_reg[2], B_sm[SM_COM_INDEX][(n_warp_offset + 8) / 8][mma_k / 2][mma_n % 8][0]);
    FETCH_FLOAT(B_mma_reg[4], B_sm[SM_COM_INDEX][n_warp_offset / 8][(mma_k + 8) / 2][mma_n % 8][0]);
    FETCH_FLOAT(B_mma_reg[6], B_sm[SM_COM_INDEX][(n_warp_offset + 8) / 8][(mma_k + 8) / 2][mma_n % 8][0]);

    for (int m_loop_offset = 0; m_loop_offset < BLOCK_TILE_M; m_loop_offset += LOOP_TILE_M) {
      const int mma_m = m_loop_offset + lane_id / 4;
      const int k     = lane_id % 4 * 2;
      FETCH_FLOAT(A_mma_reg[0], A_sm[SM_COM_INDEX][0][mma_m][k]);
      FETCH_FLOAT(A_mma_reg[2], A_sm[SM_COM_INDEX][1][mma_m][k]);
      mma_m16n8k16_row_col(
        C_mma_reg[m_loop_offset / LOOP_TILE_M], B_mma_reg, A_mma_reg, C_mma_reg[m_loop_offset / LOOP_TILE_M]);
    }
  }

  for (int m_loop = 0; m_loop < BLOCK_TILE_M / LOOP_TILE_M; m_loop++) {
    T casted[4] = {
      C_mma_reg[m_loop][0],
      C_mma_reg[m_loop][1],
      C_mma_reg[m_loop][2],
      C_mma_reg[m_loop][3],
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
    int              m                  = m_block_offset + m_loop * LOOP_TILE_M + lane_id % 4 * 2;
    int              n                  = n_block_offset + n_warp_offset + lane_2_n_offset[lane_id / 4];
    // STORE_FLOAT(C[OFFSET(m, n, N)], store[0]);
    // STORE_FLOAT(C[OFFSET(m + 1, n, N)], store[2]);
    /* clang-format off */
    asm volatile("st.global.wt.f32 [%0], %1;" : : "l"(&C[OFFSET(m, n, N)]), "f"(*(const float*)&store[0]) : "memory");
    asm volatile("st.global.wt.f32 [%0], %1;" : : "l"(&C[OFFSET(m + 1, n, N)]), "f"(*(const float*)&store[2]) : "memory");
    /* clang-format on */
  }
#undef global_2_ldg_reg
#undef ldg_reg_2_shared
}

template<typename T, int BLOCK_TILE_M, int BLOCK_TILE_N, int LOOP_TILE_M, int WARP_TILE_N, int LOOP_TILE_K>
__global__ void llmmm_fp16_mma_m16n8k16__overlap_global_2_sm__overlap_sm_2_reg__st_global_wt(
  const T* A, const T* B, T* C, int M, int N, int K)
{
  static_assert(std::is_same<T, half>::value || std::is_same<T, __nv_bfloat16>::value);
  static_assert(LOOP_TILE_M == 8);
  static_assert(WARP_TILE_N == 16);
  static_assert(LOOP_TILE_K == 16);
  constexpr int WARP_COUNT   = BLOCK_TILE_N / WARP_TILE_N;
  constexpr int THREAD_COUNT = WARP_COUNT * 32;

  __shared__ T A_sm[2][2][BLOCK_TILE_M][LOOP_TILE_K / 2];
  __shared__ T B_sm[2][BLOCK_TILE_N / 8][LOOP_TILE_K / 2][8][2];

  constexpr int A_ELEMENT_COUNT_PER_BLOCK = BLOCK_TILE_M * LOOP_TILE_K;
  constexpr int A_ELEMENT_COUNT_PER_THREAD =
    A_ELEMENT_COUNT_PER_BLOCK / THREAD_COUNT < 4 ? 4 : A_ELEMENT_COUNT_PER_BLOCK / THREAD_COUNT;
  static_assert(A_ELEMENT_COUNT_PER_THREAD % 4 == 0);
  constexpr int A_LDG_LOOP_COUNT = A_ELEMENT_COUNT_PER_THREAD / 4;
  T             A_ldg_reg[A_ELEMENT_COUNT_PER_THREAD];

  constexpr int B_ELEMENT_COUNT_PER_BLOCK  = BLOCK_TILE_N * LOOP_TILE_K;
  constexpr int B_ELEMENT_COUNT_PER_THREAD = B_ELEMENT_COUNT_PER_BLOCK / THREAD_COUNT;
  static_assert(B_ELEMENT_COUNT_PER_THREAD == 8);
  T B_ldg_reg[B_ELEMENT_COUNT_PER_THREAD];

  T     A_mma_reg[2][4];
  T     B_mma_reg[2][8];
  float C_mma_reg[BLOCK_TILE_M / LOOP_TILE_M][4] = {0};

  const int warp_id        = threadIdx.x / 32;
  const int lane_id        = threadIdx.x % 32;
  const int m_block_offset = blockIdx.y * BLOCK_TILE_M;
  const int n_block_offset = blockIdx.x * BLOCK_TILE_N;
  const int n_warp_offset  = warp_id * WARP_TILE_N;

#define global_2_ldg_reg()                                                                                             \
  {                                                                                                                    \
    for (int loop = 0; loop < A_LDG_LOOP_COUNT; ++loop) {                                                              \
      int m = (loop * WARP_COUNT + warp_id) * 8 + lane_id % 16 / 2;                                                    \
      int k = lane_id / 16 * 8 + lane_id % 2 * 4;                                                                      \
      FETCH_FLOAT2(A_ldg_reg[loop * 4], A[OFFSET((m_block_offset + m) % M, k_loop_offset + k, K)]);                    \
    }                                                                                                                  \
    {                                                                                                                  \
      int k = ((warp_id & 1) * 8) + (lane_id & 0x6);                                                                   \
      int n = warp_id / 2 * 32 + (lane_id & 0xf8) + (lane_id & 1) * 4;                                                 \
      FETCH_FLOAT2(B_ldg_reg[0], B[OFFSET(k_loop_offset + k, n_block_offset + n, N)]);                                 \
      FETCH_FLOAT2(B_ldg_reg[4], B[OFFSET(k_loop_offset + k + 1, n_block_offset + n, N)]);                             \
    }                                                                                                                  \
  }

#define ldg_reg_2_shared()                                                                                             \
  {                                                                                                                    \
    for (int loop = 0; loop < A_LDG_LOOP_COUNT; ++loop) {                                                              \
      int m = (loop * WARP_COUNT + warp_id) * 8 + lane_id % 16 / 2;                                                    \
      int k = lane_id / 16 * 8 + lane_id % 2 * 4;                                                                      \
      STORE_FLOAT2(A_sm[SM_LDG_INDEX][k / 8][m][k % 8], A_ldg_reg[loop * 4]);                                          \
    }                                                                                                                  \
    {                                                                                                                  \
      int k            = ((warp_id & 1) * 8) + (lane_id & 0x6);                                                        \
      int n            = warp_id / 2 * 32 + (lane_id & 0xf8) + (lane_id & 1) * 4;                                      \
      T   transpose[8] = {B_ldg_reg[0],                                                                                \
                          B_ldg_reg[4],                                                                                \
                          B_ldg_reg[1],                                                                                \
                          B_ldg_reg[5],                                                                                \
                          B_ldg_reg[2],                                                                                \
                          B_ldg_reg[6],                                                                                \
                          B_ldg_reg[3],                                                                                \
                          B_ldg_reg[7]};                                                                               \
      STORE_FLOAT4(B_sm[SM_LDG_INDEX][n / 8][k / 2][n % 8][0], transpose[0]);                                          \
    }                                                                                                                  \
  }

#define sm_2_B_mma_reg()                                                                                               \
  {                                                                                                                    \
    const int mma_n = n_warp_offset + lane_id / 4;                                                                     \
    const int mma_k = lane_id % 4 * 2;                                                                                 \
    FETCH_FLOAT(B_mma_reg[B_MMA_REG_LDS_INDEX][0], B_sm[SM_COM_INDEX][n_warp_offset / 8][mma_k / 2][mma_n % 8][0]);    \
    FETCH_FLOAT(B_mma_reg[B_MMA_REG_LDS_INDEX][2],                                                                     \
                B_sm[SM_COM_INDEX][(n_warp_offset + 8) / 8][mma_k / 2][mma_n % 8][0]);                                 \
    FETCH_FLOAT(B_mma_reg[B_MMA_REG_LDS_INDEX][4],                                                                     \
                B_sm[SM_COM_INDEX][n_warp_offset / 8][(mma_k + 8) / 2][mma_n % 8][0]);                                 \
    FETCH_FLOAT(B_mma_reg[B_MMA_REG_LDS_INDEX][6],                                                                     \
                B_sm[SM_COM_INDEX][(n_warp_offset + 8) / 8][(mma_k + 8) / 2][mma_n % 8][0]);                           \
  }

#define sm_2_A_mma_reg()                                                                                               \
  {                                                                                                                    \
    const int mma_m = m_loop_offset + lane_id / 4;                                                                     \
    const int k     = lane_id % 4 * 2;                                                                                 \
    FETCH_FLOAT(A_mma_reg[A_MMA_REG_LDS_INDEX][0], A_sm[SM_COM_INDEX][0][mma_m][k]);                                   \
    FETCH_FLOAT(A_mma_reg[A_MMA_REG_LDS_INDEX][2], A_sm[SM_COM_INDEX][1][mma_m][k]);                                   \
  }

  {
    constexpr int k_loop_offset = 0;
    global_2_ldg_reg();
  }
  {
    constexpr int SM_LDG_INDEX = 0;
    ldg_reg_2_shared();
  }
  int k_loop_offset = LOOP_TILE_K;
  int SM_COM_INDEX  = 0;
  int SM_LDG_INDEX  = 1;
  __syncthreads();
  int A_MMA_REG_LDS_INDEX = 0;
  int m_loop_offset       = 0;
  sm_2_A_mma_reg();
  int B_MMA_REG_LDS_INDEX = 0;
  sm_2_B_mma_reg();

  while (k_loop_offset < K) {
    global_2_ldg_reg();

    m_loop_offset = LOOP_TILE_M;
    A_MMA_REG_LDS_INDEX ^= 1;
    while (m_loop_offset < BLOCK_TILE_M) {
      sm_2_A_mma_reg();
      A_MMA_REG_LDS_INDEX ^= 1;
      mma_m16n8k16_row_col(C_mma_reg[m_loop_offset / LOOP_TILE_M - 1],
                           B_mma_reg[B_MMA_REG_LDS_INDEX],
                           A_mma_reg[A_MMA_REG_LDS_INDEX],
                           C_mma_reg[m_loop_offset / LOOP_TILE_M - 1]);
      m_loop_offset += LOOP_TILE_M;
    }
    ldg_reg_2_shared();
    __syncthreads();
    k_loop_offset += LOOP_TILE_K;
    SM_COM_INDEX ^= 1;
    SM_LDG_INDEX ^= 1;

    m_loop_offset = 0;
    sm_2_A_mma_reg();
    B_MMA_REG_LDS_INDEX ^= 1;
    sm_2_B_mma_reg();

    mma_m16n8k16_row_col(C_mma_reg[BLOCK_TILE_M / LOOP_TILE_M - 1],
                         B_mma_reg[B_MMA_REG_LDS_INDEX ^ 1],
                         A_mma_reg[A_MMA_REG_LDS_INDEX ^ 1],
                         C_mma_reg[BLOCK_TILE_M / LOOP_TILE_M - 1]);
  }
  {
    m_loop_offset       = LOOP_TILE_M;
    A_MMA_REG_LDS_INDEX = 1;
    while (m_loop_offset < BLOCK_TILE_M) {
      sm_2_A_mma_reg();
      A_MMA_REG_LDS_INDEX ^= 1;
      mma_m16n8k16_row_col(C_mma_reg[m_loop_offset / LOOP_TILE_M - 1],
                           B_mma_reg[B_MMA_REG_LDS_INDEX],
                           A_mma_reg[A_MMA_REG_LDS_INDEX],
                           C_mma_reg[m_loop_offset / LOOP_TILE_M - 1]);
      m_loop_offset += LOOP_TILE_M;
    }
    mma_m16n8k16_row_col(C_mma_reg[m_loop_offset / LOOP_TILE_M - 1],
                         B_mma_reg[B_MMA_REG_LDS_INDEX],
                         A_mma_reg[A_MMA_REG_LDS_INDEX ^ 1],
                         C_mma_reg[m_loop_offset / LOOP_TILE_M - 1]);
  }

  for (int m_loop = 0; m_loop < BLOCK_TILE_M / LOOP_TILE_M; m_loop++) {
    T casted[4] = {
      C_mma_reg[m_loop][0],
      C_mma_reg[m_loop][1],
      C_mma_reg[m_loop][2],
      C_mma_reg[m_loop][3],
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
    int              m                  = m_block_offset + m_loop * LOOP_TILE_M + lane_id % 4 * 2;
    int              n                  = n_block_offset + n_warp_offset + lane_2_n_offset[lane_id / 4];
    /* clang-format off */
    asm volatile("st.global.wt.f32 [%0], %1;" : : "l"(&C[OFFSET(m, n, N)]), "f"(*(const float*)&store[0]) : "memory");
    asm volatile("st.global.wt.f32 [%0], %1;" : : "l"(&C[OFFSET(m + 1, n, N)]), "f"(*(const float*)&store[2]) : "memory");
    /* clang-format on */
  }
#undef global_2_ldg_reg
#undef ldg_reg_2_shared
#undef sm_2_A_mma_reg
#undef sm_2_B_mma_reg
}

template<typename T, int BLOCK_TILE_M, int BLOCK_TILE_N, int LOOP_TILE_M, int WARP_TILE_N, int LOOP_TILE_K>
__global__ void llmmm_fp16_mma_m16n8k16__overlap_global_2_sm__overlap_sm_2_reg__remove_B_sm__st_global_wt(
  const T* A, const T* B, T* C, int M, int N, int K)
{
  static_assert(std::is_same<T, half>::value || std::is_same<T, __nv_bfloat16>::value);
  static_assert(LOOP_TILE_M == 8);
  static_assert(WARP_TILE_N == 16);
  static_assert(LOOP_TILE_K == 16);
  constexpr int WARP_COUNT   = BLOCK_TILE_N / WARP_TILE_N;
  constexpr int THREAD_COUNT = WARP_COUNT * 32;

  __shared__ T A_sm[2][2][BLOCK_TILE_M][LOOP_TILE_K / 2];

  constexpr int A_ELEMENT_COUNT_PER_BLOCK = BLOCK_TILE_M * LOOP_TILE_K;
  constexpr int A_ELEMENT_COUNT_PER_THREAD =
    A_ELEMENT_COUNT_PER_BLOCK / THREAD_COUNT < 4 ? 4 : A_ELEMENT_COUNT_PER_BLOCK / THREAD_COUNT;
  static_assert(A_ELEMENT_COUNT_PER_THREAD % 4 == 0);
  constexpr int A_LDG_LOOP_COUNT = A_ELEMENT_COUNT_PER_THREAD / 4;
  T             A_ldg_reg[A_ELEMENT_COUNT_PER_THREAD];

  T     A_mma_reg[2][4];
  T     B_mma_reg[8];
  T     B_mma_reg_staging[8];
  float C_mma_reg[BLOCK_TILE_M / LOOP_TILE_M][4] = {0};

  const int warp_id        = threadIdx.x / 32;
  const int lane_id        = threadIdx.x % 32;
  const int m_block_offset = blockIdx.y * BLOCK_TILE_M;
  const int n_block_offset = blockIdx.x * BLOCK_TILE_N;
  const int n_warp_offset  = warp_id * WARP_TILE_N;

#define global_2_ldg_reg()                                                                                             \
  {                                                                                                                    \
    for (int loop = 0; loop < A_LDG_LOOP_COUNT; ++loop) {                                                              \
      int m = (loop * WARP_COUNT + warp_id) * 8 + lane_id % 16 / 2;                                                    \
      int k = lane_id / 16 * 8 + lane_id % 2 * 4;                                                                      \
      FETCH_FLOAT2(A_ldg_reg[loop * 4], A[OFFSET((m_block_offset + m) % M, k_loop_offset + k, K)]);                    \
    }                                                                                                                  \
  }

#define global_2_B_mma_reg_staging()                                                                                   \
  {                                                                                                                    \
    const int n = n_block_offset + n_warp_offset + lane_id / 8 * 2 + lane_id % 8 / 4 * 8;                              \
    const int k = k_loop_offset + lane_id % 4 * 2;                                                                     \
    FETCH_FLOAT(B_mma_reg_staging[0], B[OFFSET(k, n, K)]);                                                             \
    FETCH_FLOAT(B_mma_reg_staging[2], B[OFFSET(k + 1, n, K)]);                                                         \
    FETCH_FLOAT(B_mma_reg_staging[4], B[OFFSET(k + 8, n, K)]);                                                         \
    FETCH_FLOAT(B_mma_reg_staging[6], B[OFFSET(k + 9, n, K)]);                                                         \
  }

#define move_staging_2_B_mma_reg_pre()                                                                                 \
  {                                                                                                                    \
    T data[8] = {                                                                                                      \
      B_mma_reg_staging[0],                                                                                            \
      B_mma_reg_staging[2],                                                                                            \
      B_mma_reg_staging[4],                                                                                            \
      B_mma_reg_staging[6],                                                                                            \
      B_mma_reg_staging[1],                                                                                            \
      B_mma_reg_staging[3],                                                                                            \
      B_mma_reg_staging[5],                                                                                            \
      B_mma_reg_staging[7],                                                                                            \
    };                                                                                                                 \
    uint64_t& first  = *(uint64_t*)&data[0];                                                                           \
    uint64_t& second = *(uint64_t*)&data[4];                                                                           \
    uint64_t  swap   = (first ^ second) * (!(lane_id & 0x04));                                                         \
    first ^= swap;                                                                                                     \
    second ^= swap;                                                                                                    \
    first = __shfl_xor_sync(0xffffffff, first, 0x4);                                                                   \
    swap  = (first ^ second) * (!(lane_id & 0x04));                                                                    \
    first ^= swap;                                                                                                     \
    second ^= swap;

#define move_staging_2_B_mma_reg_post()                                                                                \
  B_mma_reg[0] = data[0];                                                                                              \
  B_mma_reg[1] = data[1];                                                                                              \
  B_mma_reg[2] = data[4];                                                                                              \
  B_mma_reg[3] = data[5];                                                                                              \
  B_mma_reg[4] = data[2];                                                                                              \
  B_mma_reg[5] = data[3];                                                                                              \
  B_mma_reg[6] = data[6];                                                                                              \
  B_mma_reg[7] = data[7];                                                                                              \
  }

#define ldg_reg_2_shared()                                                                                             \
  {                                                                                                                    \
    for (int loop = 0; loop < A_LDG_LOOP_COUNT; ++loop) {                                                              \
      int m = (loop * WARP_COUNT + warp_id) * 8 + lane_id % 16 / 2;                                                    \
      int k = lane_id / 16 * 8 + lane_id % 2 * 4;                                                                      \
      STORE_FLOAT2(A_sm[SM_LDG_INDEX][k / 8][m][k % 8], A_ldg_reg[loop * 4]);                                          \
    }                                                                                                                  \
  }

#define sm_2_A_mma_reg()                                                                                               \
  {                                                                                                                    \
    const int mma_m = m_loop_offset + lane_id / 4;                                                                     \
    const int k     = lane_id % 4 * 2;                                                                                 \
    FETCH_FLOAT(A_mma_reg[A_MMA_REG_LDS_INDEX][0], A_sm[SM_COM_INDEX][0][mma_m][k]);                                   \
    FETCH_FLOAT(A_mma_reg[A_MMA_REG_LDS_INDEX][2], A_sm[SM_COM_INDEX][1][mma_m][k]);                                   \
  }

  {
    constexpr int k_loop_offset = 0;
    global_2_ldg_reg();
    global_2_B_mma_reg_staging();
  }
  {
    constexpr int SM_LDG_INDEX = 0;
    ldg_reg_2_shared();
    move_staging_2_B_mma_reg_pre();
    move_staging_2_B_mma_reg_post();
  }
  int k_loop_offset = LOOP_TILE_K;
  int SM_COM_INDEX  = 0;
  int SM_LDG_INDEX  = 1;
  __syncthreads();

  int A_MMA_REG_LDS_INDEX = 0;
  int m_loop_offset       = 0;
  sm_2_A_mma_reg();

  while (k_loop_offset < K) {
    global_2_ldg_reg();

    global_2_B_mma_reg_staging();

    m_loop_offset = LOOP_TILE_M;
    A_MMA_REG_LDS_INDEX ^= 1;
    while (m_loop_offset < BLOCK_TILE_M) {
      sm_2_A_mma_reg();
      A_MMA_REG_LDS_INDEX ^= 1;
      mma_m16n8k16_row_col(C_mma_reg[m_loop_offset / LOOP_TILE_M - 1],
                           B_mma_reg,
                           A_mma_reg[A_MMA_REG_LDS_INDEX],
                           C_mma_reg[m_loop_offset / LOOP_TILE_M - 1]);
      m_loop_offset += LOOP_TILE_M;
    }
    ldg_reg_2_shared();
    __syncthreads();
    k_loop_offset += LOOP_TILE_K;
    SM_COM_INDEX ^= 1;
    SM_LDG_INDEX ^= 1;

    m_loop_offset = 0;
    sm_2_A_mma_reg();

    move_staging_2_B_mma_reg_pre();
    mma_m16n8k16_row_col(C_mma_reg[BLOCK_TILE_M / LOOP_TILE_M - 1],
                         B_mma_reg,
                         A_mma_reg[A_MMA_REG_LDS_INDEX ^ 1],
                         C_mma_reg[BLOCK_TILE_M / LOOP_TILE_M - 1]);
    move_staging_2_B_mma_reg_post();
  }
  {
    m_loop_offset       = LOOP_TILE_M;
    A_MMA_REG_LDS_INDEX = 1;
    while (m_loop_offset < BLOCK_TILE_M) {
      sm_2_A_mma_reg();
      A_MMA_REG_LDS_INDEX ^= 1;
      mma_m16n8k16_row_col(C_mma_reg[m_loop_offset / LOOP_TILE_M - 1],
                           B_mma_reg,
                           A_mma_reg[A_MMA_REG_LDS_INDEX],
                           C_mma_reg[m_loop_offset / LOOP_TILE_M - 1]);
      m_loop_offset += LOOP_TILE_M;
    }
    mma_m16n8k16_row_col(C_mma_reg[m_loop_offset / LOOP_TILE_M - 1],
                         B_mma_reg,
                         A_mma_reg[A_MMA_REG_LDS_INDEX ^ 1],
                         C_mma_reg[m_loop_offset / LOOP_TILE_M - 1]);
  }

  for (int m_loop = 0; m_loop < BLOCK_TILE_M / LOOP_TILE_M; m_loop++) {
    T casted[4] = {
      C_mma_reg[m_loop][0],
      C_mma_reg[m_loop][1],
      C_mma_reg[m_loop][2],
      C_mma_reg[m_loop][3],
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
    int              m                  = m_block_offset + m_loop * LOOP_TILE_M + lane_id % 4 * 2;
    int              n                  = n_block_offset + n_warp_offset + lane_2_n_offset[lane_id / 4];
    /* clang-format off */
    asm volatile("st.global.wt.f32 [%0], %1;" : : "l"(&C[OFFSET(m, n, N)]), "f"(*(const float*)&store[0]) : "memory");
    asm volatile("st.global.wt.f32 [%0], %1;" : : "l"(&C[OFFSET(m + 1, n, N)]), "f"(*(const float*)&store[2]) : "memory");
    /* clang-format on */
  }
#undef global_2_ldg_reg
#undef ldg_reg_2_shared
#undef global_2_B_mma_reg_staging
#undef move_staging_2_B_mma_reg_pre
#undef move_staging_2_B_mma_reg_post
}

#define define_check_function(function)                                                                                                     \
  template<typename T>                                                                                                                      \
  void launch_##function(const T* A, const T* B, T* C, int M, int N, int K)                                                                 \
  {                                                                                                                                         \
    constexpr int BLOCK_TILE_M = 128;                                                                                                       \
    constexpr int BLOCK_TILE_N = 128;                                                                                                       \
    constexpr int LOOP_TILE_K  = 16;                                                                                                        \
    if (!(M % BLOCK_TILE_M == 0 && N % BLOCK_TILE_N == 0 && K % LOOP_TILE_K == 0)) {                                                        \
      throw std::runtime_error("M or N or K are not aligned.");                                                                             \
    }                                                                                                                                       \
    static_assert(8 <= BLOCK_TILE_M && BLOCK_TILE_M <= 256 && (BLOCK_TILE_M & (BLOCK_TILE_M - 1)) == 0);                                    \
    static_assert(16 <= BLOCK_TILE_N && BLOCK_TILE_N <= 256 && (BLOCK_TILE_N & (BLOCK_TILE_N - 1)) == 0);                                   \
    static_assert(LOOP_TILE_K == 16);                                                                                                       \
    constexpr int LOOP_TILE_M = 8;                                                                                                          \
    constexpr int WARP_TILE_N = 16;                                                                                                         \
    static_assert(BLOCK_TILE_M % LOOP_TILE_M == 0 && BLOCK_TILE_N % WARP_TILE_N == 0);                                                      \
    static_assert(LOOP_TILE_M == 8);                                                                                                        \
    static_assert(WARP_TILE_N == 16);                                                                                                       \
    constexpr int WARP_COUNT = BLOCK_TILE_N / WARP_TILE_N;                                                                                  \
    static_assert(1 <= WARP_COUNT && WARP_COUNT <= 16 && (WARP_COUNT & (WARP_COUNT - 1)) == 0);                                             \
    dim3 grid(N / BLOCK_TILE_N, M / BLOCK_TILE_M);                                                                                          \
    dim3 block(WARP_COUNT * 32);                                                                                                            \
    function<T, BLOCK_TILE_M, BLOCK_TILE_N, LOOP_TILE_M, WARP_TILE_N, LOOP_TILE_K><<<grid, block>>>(A, B, C, M, N, K);                      \
    CHECK_CUDA_ERROR();                                                                                                                     \
  }                                                                                                                                         \
  template<typename T>                                                                                                                      \
  void function##_check_relative_error(                                                                                                     \
    const T* A, const T* B, T* C, int M, int N, int K, const std::vector<float>& base)                                                      \
  {                                                                                                                                         \
    std::vector<T> host_C(M* N);                                                                                                            \
    memset(host_C.data(), 0, sizeof(T) * host_C.size());                                                                                    \
    launch_##function(A, B, C, M, N, K);                                                                                                    \
    cudaMemcpy(host_C.data(), C, sizeof(T) * host_C.size(), cudaMemcpyDefault);                                                             \
    float max_error = 0, base_value, current_value;                                                                                         \
    int   position  = 0;                                                                                                                    \
    for (int i = 0; i < host_C.size(); ++i) {                                                                                               \
      if (fabs(float(host_C[i]) - base[i]) > max_error) {                                                                                   \
        max_error     = fabs(float(host_C[i]) - base[i]);                                                                                   \
        base_value    = base[i];                                                                                                            \
        current_value = host_C[i];                                                                                                          \
        position      = i;                                                                                                                  \
      }                                                                                                                                     \
    }                                                                                                                                       \
    const char* type = std::is_same<T, half>::value ? "half" : "__nv_bfloat16";                                                             \
    const char* name = #function;                                                                                                           \
    name += 15;                                                                                                                             \
    printf(                                                                                                                                 \
      "relative_error = %10.6f, max_error = %10.3f, base_value = %10.3f, current_value = %10.3f, position = %8d, type=%16s, function=%s\n", \
      fabs(max_error / base_value),                                                                                                         \
      max_error,                                                                                                                            \
      base_value,                                                                                                                           \
      current_value,                                                                                                                        \
      position,                                                                                                                             \
      type,                                                                                                                                 \
      name);                                                                                                                                \
  }

/* clang-format off */
define_check_function(llmmm_fp16_mma_m16n8k16);
define_check_function(llmmm_fp16_mma_m16n8k16__overlap_global_2_sm);
define_check_function(llmmm_fp16_mma_m16n8k16__overlap_global_2_sm__st_global_wt);
define_check_function(llmmm_fp16_mma_m16n8k16__overlap_global_2_sm__overlap_sm_2_reg__st_global_wt);
define_check_function(llmmm_fp16_mma_m16n8k16__overlap_global_2_sm__overlap_sm_2_reg__remove_B_sm__st_global_wt);
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
  llmmm_fp16_mma_m16n8k16_check_relative_error(fp16_A, fp16_B, fp16_C, M, N, K, host_C);
  llmmm_fp16_mma_m16n8k16__overlap_global_2_sm_check_relative_error(fp16_A, fp16_B, fp16_C, M, N, K, host_C);
  llmmm_fp16_mma_m16n8k16__overlap_global_2_sm__st_global_wt_check_relative_error( fp16_A, fp16_B, fp16_C, M, N, K, host_C);
  llmmm_fp16_mma_m16n8k16__overlap_global_2_sm__overlap_sm_2_reg__st_global_wt_check_relative_error(fp16_A, fp16_B, fp16_C, M, N, K, host_C);
  llmmm_fp16_mma_m16n8k16__overlap_global_2_sm__overlap_sm_2_reg__remove_B_sm__st_global_wt_check_relative_error(fp16_A, fp16_B, fp16_C, M, N, K, host_C);
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
        // if (row < limit && col < limit) {
        //   vec->operator[](i) = row * limit + col;
        // }
        // else {
        //   vec->operator[](i) = 0;
        // }
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
