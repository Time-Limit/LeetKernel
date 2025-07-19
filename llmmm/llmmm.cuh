#pragma once

#include "llmmm/llmmm.h"
#include "util/util.cuh"
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace LLMMM {

template<int BLOCK_TILE_SIZE, int THREAD_TILE_SIZE, bool IS_ALIGNED_REDUCE_BLOCK_TILE>
__global__ void llmmm_vetor_addition(float* result, const float* vectors, int count, int size)
{
  static_assert(BLOCK_TILE_SIZE % THREAD_TILE_SIZE == 0);
  constexpr int THREAD_COUNT = (BLOCK_TILE_SIZE + THREAD_TILE_SIZE - 1) / THREAD_TILE_SIZE;
  static_assert(THREAD_COUNT >= 32 && (THREAD_COUNT & (THREAD_COUNT - 1)) == 0);

  // The following checks are to ensure that data can be loaded via float4.
  constexpr int REG_COUNT = THREAD_TILE_SIZE;
  static_assert(REG_COUNT % 4 == 0);
  constexpr int LDG_LOOP_COUNT = REG_COUNT / 4;

  float ldg_reg[2][REG_COUNT];
  float res_reg[REG_COUNT] = {0};

  const int     block_offset  = blockIdx.x * BLOCK_TILE_SIZE;
  const int     thread_offset = block_offset + threadIdx.x * 4;
  constexpr int thread_stride = THREAD_COUNT * 4;

#define global_2_ldg_reg()                                                                                             \
  {                                                                                                                    \
    const float* vector = vectors + vector_index * size;                                                               \
    for (int loop = 0; loop < LDG_LOOP_COUNT; ++loop) {                                                                \
      if constexpr (IS_ALIGNED_REDUCE_BLOCK_TILE) {                                                                    \
        FETCH_FLOAT4(ldg_reg[LDG_BUF_INDEX][loop * 4], vector[thread_offset + loop * thread_stride]);                  \
      }                                                                                                                \
      else {                                                                                                           \
        FETCH_FLOAT4(ldg_reg[LDG_BUF_INDEX][loop * 4], vector[(thread_offset + loop * thread_stride) % size]);         \
      }                                                                                                                \
    }                                                                                                                  \
  }

  {
    constexpr int LDG_BUF_INDEX = 0;
    constexpr int vector_index  = 0;
    global_2_ldg_reg();
  }

  int LDG_BUF_INDEX = 1;
  int COM_BUF_INDEX = 0;
  for (int vector_index = 1; vector_index < count; ++vector_index) {
    global_2_ldg_reg();
    for (int i = 0; i < REG_COUNT; ++i) {
      res_reg[i] += ldg_reg[COM_BUF_INDEX][i];
    }
    LDG_BUF_INDEX ^= 1;
    COM_BUF_INDEX ^= 1;
  }
  for (int i = 0; i < REG_COUNT; ++i) {
    res_reg[i] += ldg_reg[COM_BUF_INDEX][i];
  }
  for (int loop = 0; loop < REG_COUNT; loop += 4) {
    if constexpr (IS_ALIGNED_REDUCE_BLOCK_TILE) {
      asm volatile("st.global.wt.v4.f32 [%0], {%1, %2, %3, %4};"
                   :
                   : "l"(&result[thread_offset + loop * thread_stride]),
                     "f"(res_reg[loop]),
                     "f"(res_reg[loop + 1]),
                     "f"(res_reg[loop + 2]),
                     "f"(res_reg[loop + 3]));
    }
    else {
      int index = thread_offset + loop * thread_stride;
      asm volatile("{\n"
                   "  .reg .pred p;\n"
                   "  setp.ne.b32 p, %0, 0;\n"
                   "  @p st.global.wt.v4.f32 [%1], {%2, %3, %4, %5};"
                   "}\n"
                   :
                   : "r"(int(index < size)),
                     "l"(&result[thread_offset + loop * thread_stride]),
                     "f"(res_reg[loop]),
                     "f"(res_reg[loop + 1]),
                     "f"(res_reg[loop + 2]),
                     "f"(res_reg[loop + 3]));
    }
  }
#undef global_2_ldg_reg
}

template<int  BLOCK_TILE_M,
         int  BLOCK_TILE_N,
         int  THREAD_TILE_M,
         int  THREAD_TILE_N,
         int  TILE_K,
         bool IS_ALIGNED_M,
         int  SPLIT_K_TILES>
__global__ void llmmm__overlap_global2sm2reg__quardra_buffer__double_ldg_reg__st_global_wt(
  const float* A, const float* B, float* C, int M, int N, int K)
{
  constexpr int THREAD_COUNT =
    device_thread_count_calculator<BLOCK_TILE_M, BLOCK_TILE_N, THREAD_TILE_M, THREAD_TILE_N>();

  static_assert((THREAD_COUNT & (THREAD_COUNT - 1)) == 0);

  // The following checks are to ensure that data can be loaded via float4.
  static_assert(BLOCK_TILE_M * TILE_K % THREAD_COUNT == 0);
  static_assert(BLOCK_TILE_N * TILE_K % THREAD_COUNT == 0);
  static_assert(BLOCK_TILE_M * TILE_K / THREAD_COUNT % 4 == 0);
  static_assert(BLOCK_TILE_N * TILE_K / THREAD_COUNT % 4 == 0);
  constexpr int A_LDG_REG_COUNT  = BLOCK_TILE_M * TILE_K / THREAD_COUNT;
  constexpr int B_LDG_REG_COUNT  = BLOCK_TILE_N * TILE_K / THREAD_COUNT;
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
  const int block_k_offset = blockIdx.z * K;

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
  if constexpr (SPLIT_K_TILES == 1) {                                                                                  \
    /* Load A, global -> register */                                                                                   \
    _Pragma("unroll") for (int loop = 0; loop < A_LDG_LOOP_COUNT; ++loop)                                              \
    {                                                                                                                  \
      const int m = (loop * THREAD_COUNT + threadIdx.x) * 4 / TILE_K;                                                  \
      const int k = (loop * THREAD_COUNT + threadIdx.x) * 4 % TILE_K;                                                  \
      if constexpr (IS_ALIGNED_M) {                                                                                    \
        FETCH_FLOAT4(A_ldg_reg[LDG_REG_INDEX][loop * 4], A[OFFSET(block_m_offset + m, k_iter_offset + k, K)]);         \
      }                                                                                                                \
      else {                                                                                                           \
        FETCH_FLOAT4(A_ldg_reg[LDG_REG_INDEX][loop * 4], A[OFFSET((block_m_offset + m) % M, k_iter_offset + k, K)]);   \
      }                                                                                                                \
    }                                                                                                                  \
    /* Load B, global->register */                                                                                     \
    _Pragma("unroll") for (int loop = 0; loop < B_LDG_LOOP_COUNT; ++loop)                                              \
    {                                                                                                                  \
      const int k = (loop * THREAD_COUNT + threadIdx.x) * 4 / BLOCK_TILE_N;                                            \
      const int n = (loop * THREAD_COUNT + threadIdx.x) * 4 % BLOCK_TILE_N;                                            \
      FETCH_FLOAT4(B_ldg_reg[LDG_REG_INDEX][loop * 4], B[OFFSET(k_iter_offset + k, block_n_offset + n, N)]);           \
    }                                                                                                                  \
  }                                                                                                                    \
  else {                                                                                                               \
    /* Load A, global -> register */                                                                                   \
    _Pragma("unroll") for (int loop = 0; loop < A_LDG_LOOP_COUNT; ++loop)                                              \
    {                                                                                                                  \
      const int m = (loop * THREAD_COUNT + threadIdx.x) * 4 / TILE_K;                                                  \
      const int k = (loop * THREAD_COUNT + threadIdx.x) * 4 % TILE_K;                                                  \
      if constexpr (IS_ALIGNED_M) {                                                                                    \
        FETCH_FLOAT4(A_ldg_reg[LDG_REG_INDEX][loop * 4],                                                               \
                     A[OFFSET(block_m_offset + m, block_k_offset + k_iter_offset + k, K * gridDim.z)]);                \
      }                                                                                                                \
      else {                                                                                                           \
        FETCH_FLOAT4(A_ldg_reg[LDG_REG_INDEX][loop * 4],                                                               \
                     A[OFFSET((block_m_offset + m) % M, block_k_offset + k_iter_offset + k, K * gridDim.z)]);          \
      }                                                                                                                \
    }                                                                                                                  \
    /* Load B, global->register */                                                                                     \
    _Pragma("unroll") for (int loop = 0; loop < B_LDG_LOOP_COUNT; ++loop)                                              \
    {                                                                                                                  \
      const int k = (loop * THREAD_COUNT + threadIdx.x) * 4 / BLOCK_TILE_N;                                            \
      const int n = (loop * THREAD_COUNT + threadIdx.x) * 4 % BLOCK_TILE_N;                                            \
      FETCH_FLOAT4(B_ldg_reg[LDG_REG_INDEX][loop * 4],                                                                 \
                   B[OFFSET(block_k_offset + k_iter_offset + k, block_n_offset + n, N)]);                              \
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

  const int C_offset_for_split_k = blockIdx.z * M * N;

  if constexpr (IS_ALIGNED_M) {
#pragma unroll
    for (int i = 0; i < THREAD_TILE_M; ++i) {
      const int m = block_m_offset + comp_thread_m_offset + i;
#pragma unroll
      for (int j = 0; j < THREAD_TILE_N; j += 4) {
        const int n = block_n_offset + comp_thread_n_offset + j / 4 * CAL_THREAD_N_STRIDE;
        if constexpr (SPLIT_K_TILES == 1) {
          asm volatile("st.global.wt.v4.f32 [%0], {%1, %2, %3, %4};"
                       :
                       : "l"(&C[OFFSET(m, n, N)]),
                         "f"(C_reg[i][j]),
                         "f"(C_reg[i][j + 1]),
                         "f"(C_reg[i][j + 2]),
                         "f"(C_reg[i][j + 3]));
        }
        else {
          asm volatile("st.global.wt.v4.f32 [%0], {%1, %2, %3, %4};"
                       :
                       : "l"(&C[C_offset_for_split_k + OFFSET(m, n, N)]),
                         "f"(C_reg[i][j]),
                         "f"(C_reg[i][j + 1]),
                         "f"(C_reg[i][j + 2]),
                         "f"(C_reg[i][j + 3]));
        }
      }
    }
  }
  else {
#pragma unroll
    for (int i = 0; i < THREAD_TILE_M; ++i) {
      const int m = block_m_offset + comp_thread_m_offset + i;
#pragma unroll
      for (int j = 0; j < THREAD_TILE_N; j += 4) {
        const int n = block_n_offset + comp_thread_n_offset + j / 4 * CAL_THREAD_N_STRIDE;
        if constexpr (SPLIT_K_TILES == 1) {
          asm volatile("{\n"
                       "  .reg .pred p;\n"
                       "  setp.ne.b32 p, %0, 0;\n"
                       "  @p st.global.wt.v4.f32 [%1], {%2, %3, %4, %5};"
                       "}\n"
                       :
                       : "r"(int(m < M)),
                         "l"(&C[OFFSET(m, n, N)]),
                         "f"(C_reg[i][j]),
                         "f"(C_reg[i][j + 1]),
                         "f"(C_reg[i][j + 2]),
                         "f"(C_reg[i][j + 3]));
        }
        else {
          asm volatile("{\n"
                       "  .reg .pred p;\n"
                       "  setp.ne.b32 p, %0, 0;\n"
                       "  @p st.global.wt.v4.f32 [%1], {%2, %3, %4, %5};"
                       "}\n"
                       :
                       : "r"(int(m < M)),
                         "l"(&C[C_offset_for_split_k + OFFSET(m, n, N)]),
                         "f"(C_reg[i][j]),
                         "f"(C_reg[i][j + 1]),
                         "f"(C_reg[i][j + 2]),
                         "f"(C_reg[i][j + 3]));
        }
      }
    }
  }
#undef global_2_ldg_reg
#undef ldg_reg_2_sm
#undef sm_2_comp_reg
#undef compute
}

// C = A * B, (M,N) = (M,K) * (K,N)
template<int  BLOCK_TILE_M,
         int  BLOCK_TILE_N,
         int  THREAD_TILE_M,
         int  THREAD_TILE_N,
         int  TILE_K,
         bool IS_ALIGNED_M,
         int  SPLIT_K_TILES,
         int  REDUCE_BLOCK_TILE,
         int  REDUCE_THREAD_TILE,
         bool IS_ALIGNED_REDUCE_BLOCK_TILE>
void launch_llmmm(const float* A,
                  const float* B,
                  float*       C,
                  int          M,
                  int          N,
                  int          K,
                  void*        workspace,
                  size_t       workspace_bytes,
                  cudaStream_t stream)
{
  static_assert(16 <= BLOCK_TILE_M && BLOCK_TILE_M <= 128 && (BLOCK_TILE_M & (BLOCK_TILE_M - 1)) == 0);
  static_assert(8 <= BLOCK_TILE_N && BLOCK_TILE_N <= 256 && (BLOCK_TILE_N & (BLOCK_TILE_N - 1)) == 0);
  static_assert(8 <= TILE_K && TILE_K <= 128 && ((TILE_K & (TILE_K - 1)) == 0));
  static_assert(1 <= SPLIT_K_TILES && SPLIT_K_TILES <= 128 && (SPLIT_K_TILES & (SPLIT_K_TILES - 1)) == 0);

  dim3 grid(N / BLOCK_TILE_N, (M + BLOCK_TILE_M - 1) / BLOCK_TILE_M, SPLIT_K_TILES);
  dim3 block(host_thread_count_calculator<BLOCK_TILE_M, BLOCK_TILE_N, THREAD_TILE_M, THREAD_TILE_N>());
  auto mm_kernel = &llmmm__overlap_global2sm2reg__quardra_buffer__double_ldg_reg__st_global_wt<BLOCK_TILE_M,
                                                                                                 BLOCK_TILE_N,
                                                                                                 THREAD_TILE_M,
                                                                                                 THREAD_TILE_N,
                                                                                                 TILE_K,
                                                                                                 IS_ALIGNED_M,
                                                                                                 SPLIT_K_TILES>;

  static auto result = [&]() -> bool {
    auto err = cudaFuncSetAttribute(mm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 0);
    if (err) {
      std::stringstream log;
      log << "llmmm__overlap_global2sm2reg__quardra_buffer__double_ldg_reg__st_global_wt, " << BLOCK_TILE_M << "x"
          << BLOCK_TILE_N << "x" << TILE_K << ", " << THREAD_TILE_M << "x" << THREAD_TILE_N
          << ", IS_ALIGNED_M=" << IS_ALIGNED_M << "," << cudaGetErrorString(err) << std::endl;
      throw std::runtime_error(log.str());
    }
    return true;
  }();

  if constexpr (SPLIT_K_TILES == 1) {
    mm_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
  }
  else {
    const int expected_workspace_bytes = M * N * SPLIT_K_TILES * sizeof(float);
    if (workspace == nullptr) {
      throw std::runtime_error("workspace is nullptr");
    }
    if (workspace_bytes < expected_workspace_bytes) {
      throw std::runtime_error("The workspace is too small. It requires " + std::to_string(expected_workspace_bytes)
                               + " bytes, but only " + std::to_string(workspace_bytes) + " bytes are available.");
    }
    mm_kernel<<<grid, block, 0, stream>>>(A, B, (float*)workspace, M, N, K / SPLIT_K_TILES);
    dim3 reduce_grid((M * N + REDUCE_BLOCK_TILE - 1) / REDUCE_BLOCK_TILE);
    dim3 reduce_block(REDUCE_BLOCK_TILE / REDUCE_THREAD_TILE);
    llmmm_vetor_addition<REDUCE_BLOCK_TILE, REDUCE_THREAD_TILE, IS_ALIGNED_REDUCE_BLOCK_TILE>
      <<<grid, block, 0, stream>>>(C, (float*)workspace, SPLIT_K_TILES, M * N);
  }
}

struct MMInstantiatorWrapper {
  template<int BLOCK_TILE_M>
  static void apply()
  {
#define AddMMWithAlignedParam(block_tile_m,                                                                            \
                              block_tile_n,                                                                            \
                              thread_tile_m,                                                                           \
                              thread_tile_n,                                                                           \
                              tile_k,                                                                                  \
                              is_aligne_m,                                                                             \
                              split_k_tiles,                                                                           \
                              reduce_block_tile,                                                                       \
                              reduce_thread_tile,                                                                      \
                              is_aligned_reduce_block_tile)                                                            \
  {                                                                                                                    \
    LLMMM::MMConfig config{                                                                                            \
      .BLOCK_TILE_M                 = block_tile_m,                                                                    \
      .BLOCK_TILE_N                 = block_tile_n,                                                                    \
      .THREAD_TILE_M                = thread_tile_m,                                                                   \
      .THREAD_TILE_N                = thread_tile_n,                                                                   \
      .TILE_K                       = tile_k,                                                                          \
      .IS_ALIGNED_M                 = is_aligne_m,                                                                     \
      .SPLIT_K_TILES                = split_k_tiles,                                                                   \
      .REDUCE_BLOCK_TILE            = reduce_block_tile,                                                               \
      .REDUCE_THREAD_TILE           = reduce_block_tile,                                                               \
      .IS_ALIGNED_REDUCE_BLOCK_TILE = is_aligned_reduce_block_tile,                                                    \
    };                                                                                                                 \
    LLMMM::Instance().mm_list.emplace_back(config,                                                                     \
                                           &launch_llmmm<block_tile_m,                                                 \
                                                         block_tile_n,                                                 \
                                                         thread_tile_m,                                                \
                                                         thread_tile_n,                                                \
                                                         tile_k,                                                       \
                                                         is_aligne_m,                                                  \
                                                         split_k_tiles,                                                \
                                                         reduce_block_tile,                                            \
                                                         reduce_thread_tile,                                           \
                                                         is_aligned_reduce_block_tile>);                               \
    std::cout << "MM, " << config.info() << std::endl;                                                                 \
  }

#define AddMM(block_tile_m,                                                                                            \
              block_tile_n,                                                                                            \
              thread_tile_m,                                                                                           \
              thread_tile_n,                                                                                           \
              tile_k,                                                                                                  \
              split_k_tiles,                                                                                           \
              reduce_block_tile,                                                                                       \
              reduce_thread_tile)                                                                                      \
  {                                                                                                                    \
    AddMMWithAlignedParam(block_tile_m,                                                                                \
                          block_tile_n,                                                                                \
                          thread_tile_m,                                                                               \
                          thread_tile_n,                                                                               \
                          tile_k,                                                                                      \
                          false,                                                                                       \
                          split_k_tiles,                                                                               \
                          reduce_block_tile,                                                                           \
                          reduce_thread_tile,                                                                          \
                          false);                                                                                      \
    AddMMWithAlignedParam(block_tile_m,                                                                                \
                          block_tile_n,                                                                                \
                          thread_tile_m,                                                                               \
                          thread_tile_n,                                                                               \
                          tile_k,                                                                                      \
                          true,                                                                                        \
                          split_k_tiles,                                                                               \
                          reduce_block_tile,                                                                           \
                          reduce_thread_tile,                                                                          \
                          true);                                                                                       \
    AddMMWithAlignedParam(block_tile_m,                                                                                \
                          block_tile_n,                                                                                \
                          thread_tile_m,                                                                               \
                          thread_tile_n,                                                                               \
                          tile_k,                                                                                      \
                          false,                                                                                       \
                          split_k_tiles,                                                                               \
                          reduce_block_tile,                                                                           \
                          reduce_thread_tile,                                                                          \
                          true);                                                                                       \
    AddMMWithAlignedParam(block_tile_m,                                                                                \
                          block_tile_n,                                                                                \
                          thread_tile_m,                                                                               \
                          thread_tile_n,                                                                               \
                          tile_k,                                                                                      \
                          true,                                                                                        \
                          split_k_tiles,                                                                               \
                          reduce_block_tile,                                                                           \
                          reduce_thread_tile,                                                                          \
                          false);                                                                                      \
  }

#include "llmmm/mm_config.h"

#undef AddMM
#undef AddMMWithAlignedParam
  }

  template<int BLOCK_TILE_M>
  struct MMInstantiator {
    static void apply()
    {
      MMInstantiatorWrapper::apply<BLOCK_TILE_M>();
    }
  };
};

}  // namespace LLMMM
