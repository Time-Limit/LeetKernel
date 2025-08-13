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
__device__ void
device_entry(const T* A, const T* B, T* C, int M, int N, int K, T* A_sm, T* B_sm, int m_offset, int n_offset)
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
  // T A_sm[A_sm_dim0 * A_sm_dim1 * A_sm_dim2 * A_sm_dim3];
  // T B_sm[B_sm_dim0 * B_sm_dim1 * B_sm_dim2 * B_sm_dim3];

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

  const int m_block_offset = blockIdx.y * BLOCK_TILE_M + m_offset;
  const int n_block_offset = blockIdx.x * BLOCK_TILE_N + n_offset;

  const int warp_id                 = threadIdx.x / 32;
  const int lane_id                 = threadIdx.x % 32;
  const int transposed_lane_id_mask = (lane_id / 8 == 0 || lane_id / 8 == 3) ? 0x00 : 0x18;
  const int transposed_lane_id      = lane_id ^ transposed_lane_id_mask;

  constexpr int M_MMA_WARP_COUNT       = BLOCK_TILE_M / WARP_TILE_M;
  constexpr int M_GROUP_COUNT_PER_WARP = WARP_TILE_M / 8;
  constexpr int N_GROUP_COUNT_PER_WARP = WARP_TILE_N / 16;

  static_assert(M_GROUP_COUNT_PER_WARP == 8);
  static_assert(N_GROUP_COUNT_PER_WARP == 4);

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

  const T* A_sm_ptr_for_ldg = &A_sm[A_ldg_reg_2_A_sm_partial_offset];
  const T* B_sm_ptr_for_ldg = &B_sm[B_ldg_reg_2_B_sm_partial_offset];

  const int A_sm_2_A_mma_reg_partial_offset =
    lane_id % 16 / 8 * A_sm_dim2 * A_sm_dim3 + (m_warp_offset + lane_id % 8) * A_sm_dim3;

  const int B_sm_2_B_mma_reg_partial_offset = transposed_lane_id % 16 / 8 * B_sm_dim2 * B_sm_dim3
                                              + (n_warp_offset + transposed_lane_id / 16 * 8) / 8 * B_sm_dim3
                                              + transposed_lane_id % 8 * 8;

  const T* A_sm_ptr_for_mma = &A_sm[A_sm_2_A_mma_reg_partial_offset + lane_id / 16 * 8 * A_sm_dim3];
  const T* B_sm_ptr_for_mma = &B_sm[B_sm_2_B_mma_reg_partial_offset];

  enum {
    LDG_SWITCH_OFF             = 0,
    LDG_SWITCH_ON_EVICT_NORMAL = 1,
    LDG_SWITCH_ON_EVICT_LAST   = 2,
  };

#define A_global_2_ldg_reg(A_global_ptr, ldg_reg_buffer_index, loop, cache_policy)                                     \
  {                                                                                                                    \
    /* const int m = (loop * WARP_COUNT + warp_id) * 16 + lane_id % 16; */                                             \
    /* const int k = lane_id / 16 * 8; */                                                                              \
    if constexpr (false && cache_policy == LDG_SWITCH_ON_EVICT_LAST) {                                                 \
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
    if constexpr (false && cache_policy == LDG_SWITCH_ON_EVICT_LAST) {                                                 \
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

  alternate_ldm_mma_sts_stg_ldg(LDG_SM_BUFFER_INDEX + 1,
                                MMA_REG_BUFFER_INDEX_1,
                                MMA_REG_BUFFER_INDEX_0,
                                (LDG_SM_BUFFER_INDEX ^ 2),
                                k_loop_offset_x2,
                                k_loop_offset_x2N,
                                true,
                                false,
                                false,
                                false,
                                LDG_SWITCH_ON_EVICT_LAST);

  while (k_loop_offset + LOOP_TILE_K * 2 < K) {
    alternate_ldm_mma_sts_stg_ldg(LDG_SM_BUFFER_INDEX + 1,
                                  MMA_REG_BUFFER_INDEX_1,
                                  MMA_REG_BUFFER_INDEX_0,
                                  (LDG_SM_BUFFER_INDEX ^ 2),
                                  0,
                                  0,
                                  true,
                                  false,
                                  false,
                                  false,
                                  LDG_SWITCH_OFF);
    alternate_ldm_mma_sts_stg_ldg(LDG_SM_BUFFER_INDEX + 1,
                                  MMA_REG_BUFFER_INDEX_1,
                                  MMA_REG_BUFFER_INDEX_0,
                                  (LDG_SM_BUFFER_INDEX ^ 2),
                                  0,
                                  0,
                                  false,
                                  true,
                                  false,
                                  false,
                                  LDG_SWITCH_OFF);
    alternate_ldm_mma_sts_stg_ldg(LDG_SM_BUFFER_INDEX + 1,
                                  MMA_REG_BUFFER_INDEX_1,
                                  MMA_REG_BUFFER_INDEX_0,
                                  (LDG_SM_BUFFER_INDEX ^ 2),
                                  0,
                                  0,
                                  false,
                                  false,
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
                                  k_loop_offset_x2,
                                  k_loop_offset_x2N,
                                  false,
                                  true,
                                  false,
                                  false,
                                  LDG_SWITCH_ON_EVICT_NORMAL);
    alternate_ldm_mma_sts_stg_ldg(LDG_SM_BUFFER_INDEX,
                                  MMA_REG_BUFFER_INDEX_0,
                                  MMA_REG_BUFFER_INDEX_1,
                                  0,
                                  k_loop_offset_x2,
                                  k_loop_offset_x2N,
                                  true,
                                  false,
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
  constexpr int LOOP_TILE_K         = 16;
  constexpr int LDG_SM_BUFFER_SIZE  = 4;
  constexpr int LDG_REG_BUFFER_SIZE = 2;

  constexpr int A_sm_dim0 = LDG_SM_BUFFER_SIZE;
  constexpr int A_sm_dim1 = 2;
  constexpr int A_sm_dim2 = BLOCK_TILE_M / 2;
  constexpr int A_sm_dim3 = LOOP_TILE_K / 2;
  constexpr int B_sm_dim0 = LDG_SM_BUFFER_SIZE;
  constexpr int B_sm_dim1 = LOOP_TILE_K / 8;
  constexpr int B_sm_dim2 = BLOCK_TILE_N / 8 / 2;
  constexpr int B_sm_dim3 = 64;

  // The 64 elements of type T in each 8x8 matrix are stored consecutively in a single layer of shared memory.
  __shared__ T A_sm[A_sm_dim0 * A_sm_dim1 * A_sm_dim2 * A_sm_dim3];
  __shared__ T B_sm[B_sm_dim0 * B_sm_dim1 * B_sm_dim2 * B_sm_dim3];
  device_entry<T, BLOCK_TILE_M / 2, BLOCK_TILE_N / 2, WARP_TILE_M, WARP_TILE_N>(A, B, C, M, N, K, A_sm, B_sm, 0, 0);
  device_entry<T, BLOCK_TILE_M / 2, BLOCK_TILE_N / 2, WARP_TILE_M, WARP_TILE_N>(A, B, C, M, N, K, A_sm, B_sm, 0, 2048);
  device_entry<T, BLOCK_TILE_M / 2, BLOCK_TILE_N / 2, WARP_TILE_M, WARP_TILE_N>(A, B, C, M, N, K, A_sm, B_sm, 2048, 0);
  device_entry<T, BLOCK_TILE_M / 2, BLOCK_TILE_N / 2, WARP_TILE_M, WARP_TILE_N>(
    A, B, C, M, N, K, A_sm, B_sm, 2048, 2048);
  // device_entry<T, BLOCK_TILE_M, BLOCK_TILE_N / 2, WARP_TILE_M, WARP_TILE_N, 2048>(
  //   A, B, C, M, N, K, C_mma_reg, A_sm, B_sm);
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
    dim3 block(WARP_COUNT * 32 / 2 / 2);                                                                                               \
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
    launch_##function<T, 256, 256, 64, 64>(A, B, C, M, N, K);                                                                          \
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
define_check_function(fp16_mma_m16n8k16_ldmatrix_trans__overlap_global_2_sm__quadra_buffer__reduce_instructions__reorder_instructions__overlap_reg_2_global__stg_memory_coalecesing__overlap_sts__stg_wt__reduce_IMAD_AND_LEA__reduce_MIO);
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
  fp16_mma_m16n8k16_ldmatrix_trans__overlap_global_2_sm__quadra_buffer__reduce_instructions__reorder_instructions__overlap_reg_2_global__stg_memory_coalecesing__overlap_sts__stg_wt__reduce_IMAD_AND_LEA__reduce_MIO___check_relative_error(fp16_A, fp16_B, fp16_C, M, N, K, host_C);
  fp16_mma_m16n8k16_ldmatrix_trans__overlap_global_2_sm__quadra_buffer__reduce_instructions__reorder_instructions__overlap_reg_2_global__stg_memory_coalecesing__overlap_sts__stg_wt__reduce_IMAD_AND_LEA__reduce_MIO___check_relative_error(fp16_A, fp16_B, fp16_C, M, N, K, host_C);
  fp16_mma_m16n8k16_ldmatrix_trans__overlap_global_2_sm__quadra_buffer__reduce_instructions__reorder_instructions__overlap_reg_2_global__stg_memory_coalecesing__overlap_sts__stg_wt__reduce_IMAD_AND_LEA__reduce_MIO___check_relative_error(fp16_A, fp16_B, fp16_C, M, N, K, host_C);
  fp16_mma_m16n8k16_ldmatrix_trans__overlap_global_2_sm__quadra_buffer__reduce_instructions__reorder_instructions__overlap_reg_2_global__stg_memory_coalecesing__overlap_sts__stg_wt__reduce_IMAD_AND_LEA__reduce_MIO___check_relative_error(fp16_A, fp16_B, fp16_C, M, N, K, host_C);
  fp16_mma_m16n8k16_ldmatrix_trans__overlap_global_2_sm__quadra_buffer__reduce_instructions__reorder_instructions__overlap_reg_2_global__stg_memory_coalecesing__overlap_sts__stg_wt__reduce_IMAD_AND_LEA__reduce_MIO___check_relative_error(fp16_A, fp16_B, fp16_C, M, N, K, host_C);
  fp16_mma_m16n8k16_ldmatrix_trans__overlap_global_2_sm__quadra_buffer__reduce_instructions__reorder_instructions__overlap_reg_2_global__stg_memory_coalecesing__overlap_sts__stg_wt__reduce_IMAD_AND_LEA__reduce_MIO___check_relative_error(fp16_A, fp16_B, fp16_C, M, N, K, host_C);
  fp16_mma_m16n8k16_ldmatrix_trans__overlap_global_2_sm__quadra_buffer__reduce_instructions__reorder_instructions__overlap_reg_2_global__stg_memory_coalecesing__overlap_sts__stg_wt__reduce_IMAD_AND_LEA__reduce_MIO___check_relative_error(fp16_A, fp16_B, fp16_C, M, N, K, host_C);
  fp16_mma_m16n8k16_ldmatrix_trans__overlap_global_2_sm__quadra_buffer__reduce_instructions__reorder_instructions__overlap_reg_2_global__stg_memory_coalecesing__overlap_sts__stg_wt__reduce_IMAD_AND_LEA__reduce_MIO___check_relative_error(fp16_A, fp16_B, fp16_C, M, N, K, host_C);
  fp16_mma_m16n8k16_ldmatrix_trans__overlap_global_2_sm__quadra_buffer__reduce_instructions__reorder_instructions__overlap_reg_2_global__stg_memory_coalecesing__overlap_sts__stg_wt__reduce_IMAD_AND_LEA__reduce_MIO___check_relative_error(fp16_A, fp16_B, fp16_C, M, N, K, host_C);
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
