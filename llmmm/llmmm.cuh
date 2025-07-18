#pragma once

#include "llmmm/llmmm.h"
#include "util/util.cuh"
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace LLMMM {

template<int BLOCK_TILE_M, int BLOCK_TILE_N, int THREAD_TILE_M, int THREAD_TILE_N, int TILE_K, bool IS_ALIGNED_M>
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

  if constexpr (IS_ALIGNED_M) {
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
  }
  else {
#pragma unroll
    for (int i = 0; i < THREAD_TILE_M; ++i) {
      const int m = block_m_offset + comp_thread_m_offset + i;
#pragma unroll
      for (int j = 0; j < THREAD_TILE_N; j += 4) {
        const int n = block_n_offset + comp_thread_n_offset + j / 4 * CAL_THREAD_N_STRIDE;
        asm volatile("{\n"
                     "  .reg .pred p;\n"
                     "  setp.ne.b32 p, %0, 0;\n"
                     "  st.global.wt.v4.f32 [%1], {%2, %3, %4, %5};"
                     "}\n"
                     :
                     : "r"(int(m < M)),
                       "l"(&C[OFFSET(m, n, N)]),
                       "f"(C_reg[i][j]),
                       "f"(C_reg[i][j + 1]),
                       "f"(C_reg[i][j + 2]),
                       "f"(C_reg[i][j + 3]));
      }
    }
  }
#undef global_2_ldg_reg
#undef ldg_reg_2_sm
#undef sm_2_comp_reg
#undef compute
}

// C = A * B, (M,N) = (M,K) * (K,N)
template<int BLOCK_TILE_M, int BLOCK_TILE_N, int THREAD_TILE_M, int THREAD_TILE_N, int TILE_K, bool IS_ALIGNED_M>
void launch_llmmm(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t stream)
{
  // if (N % 128 != 0 || N <= 0) {
  //   throw std::runtime_error("Not support N = " + std::to_string(N));
  // }
  // if (K % 128 != 0 || K < TILE_K * 4 || K / TILE_K % 2 != 0) {
  //   throw std::runtime_error("Not support K = " + std::to_string(K));
  // }
  // if (M % BLOCK_TILE_M != 0) {
  //   throw std::runtime_error("Not support M = " + std::to_string(M));
  // }
  static_assert(16 <= BLOCK_TILE_M && BLOCK_TILE_M <= 128 && (BLOCK_TILE_M & (BLOCK_TILE_M - 1)) == 0);
  static_assert(8 <= BLOCK_TILE_N && BLOCK_TILE_N <= 256 && (BLOCK_TILE_N & (BLOCK_TILE_N - 1)) == 0);
  static_assert(8 <= TILE_K && TILE_K <= 128 && ((TILE_K & (TILE_K - 1)) == 0));

  dim3 grid(N / BLOCK_TILE_N, (M + BLOCK_TILE_M - 1) / BLOCK_TILE_M);
  dim3 block(host_thread_count_calculator<BLOCK_TILE_M, BLOCK_TILE_N, THREAD_TILE_M, THREAD_TILE_N>());
  auto kernel_func = &llmmm__overlap_global2sm2reg__quardra_buffer__double_ldg_reg__st_global_wt<BLOCK_TILE_M,
                                                                                                 BLOCK_TILE_N,
                                                                                                 THREAD_TILE_M,
                                                                                                 THREAD_TILE_N,
                                                                                                 TILE_K,
                                                                                                 IS_ALIGNED_M>;

  static auto result = [&]() -> bool {
    auto err = cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, 0);
    if (err) {
      std::stringstream log;
      log << "llmmm__overlap_global2sm2reg__quardra_buffer__double_ldg_reg__st_global_wt, " << BLOCK_TILE_M << "x"
          << BLOCK_TILE_N << "x" << TILE_K << ", " << THREAD_TILE_M << "x" << THREAD_TILE_N
          << ", IS_ALIGNED_M=" << IS_ALIGNED_M << "," << cudaGetErrorString(err) << std::endl;
      throw std::runtime_error(log.str());
    }
    return true;
  }();

  kernel_func<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}

struct MMInstantiatorWrapper {
  template<int BLOCK_TILE_M>
  static void apply()
  {
#define AddMM(block_tile_m, block_tile_n, thread_tile_m, thread_tile_n, tile_k)                                        \
  {                                                                                                                    \
    {                                                                                                                  \
      LLMMM::MMConfig config{.BLOCK_TILE_M  = block_tile_m,                                                            \
                             .BLOCK_TILE_N  = block_tile_n,                                                            \
                             .THREAD_TILE_M = thread_tile_m,                                                           \
                             .THREAD_TILE_N = thread_tile_n,                                                           \
                             .TILE_K        = tile_k,                                                                  \
                             .IS_ALIGNED_M  = true};                                                                    \
      LLMMM::Instance().aligned_M_mm_list.emplace_back(                                                                \
        config, &launch_llmmm<block_tile_m, block_tile_n, thread_tile_m, thread_tile_n, tile_k, true>);                \
      std::cout << "MatrixMultiplication" << ", BLOCK_TILE_M=" << std::setw(3) << block_tile_m                         \
                << ", BLOCK_TILE_N=" << std::setw(3) << block_tile_n << ", THREAD_TILE_M=" << std::setw(3)             \
                << thread_tile_m << ", THREAD_TILE_N=" << std::setw(3) << thread_tile_n << ", TILE_K=" << std::setw(3) \
                << tile_k << ", IS_ALIGNED_M=" << true << std::endl;                                                  \
    }                                                                                                                  \
    {                                                                                                                  \
      LLMMM::MMConfig config{.BLOCK_TILE_M  = block_tile_m,                                                            \
                             .BLOCK_TILE_N  = block_tile_n,                                                            \
                             .THREAD_TILE_M = thread_tile_m,                                                           \
                             .THREAD_TILE_N = thread_tile_n,                                                           \
                             .TILE_K        = tile_k,                                                                  \
                             .IS_ALIGNED_M  = false};                                                                   \
      LLMMM::Instance().unaligned_M_mm_list.emplace_back(                                                              \
        config, &launch_llmmm<block_tile_m, block_tile_n, thread_tile_m, thread_tile_n, tile_k, false>);               \
      std::cout << "MatrixMultiplication" << ", BLOCK_TILE_M=" << std::setw(3) << block_tile_m                         \
                << ", BLOCK_TILE_N=" << std::setw(3) << block_tile_n << ", THREAD_TILE_M=" << std::setw(3)             \
                << thread_tile_m << ", THREAD_TILE_N=" << std::setw(3) << thread_tile_n << ", TILE_K=" << std::setw(3) \
                << tile_k << ", IS_ALIGNED_M=" << false << std::endl;                                                  \
    }                                                                                                                  \
  }
    if constexpr (BLOCK_TILE_M == 16) {
      /* clang-format off */
      AddMM(16,  32,   4,   4,   8); // block_tile_n= 32, tile_k=  8, thread_tile_m=  4, thread_tile_n=  4, thread_count= 32, register_count_per_thread= 56, total_register_count= 2040, shared_memory_bytes=  6144
      AddMM(16,  32,   4,   4,  16); // block_tile_n= 32, tile_k= 16, thread_tile_m=  4, thread_tile_n=  4, thread_count= 32, register_count_per_thread= 80, total_register_count= 2805, shared_memory_bytes= 12288
      AddMM(16,  32,   4,   4,  32); // block_tile_n= 32, tile_k= 32, thread_tile_m=  4, thread_tile_n=  4, thread_count= 32, register_count_per_thread=128, total_register_count= 4335, shared_memory_bytes= 24576
      AddMM(16,  32,   4,   4,  64); // block_tile_n= 32, tile_k= 64, thread_tile_m=  4, thread_tile_n=  4, thread_count= 32, register_count_per_thread=224, total_register_count= 7395, shared_memory_bytes= 49152
      AddMM(16,  64,   4,   4,  16); // block_tile_n= 64, tile_k= 16, thread_tile_m=  4, thread_tile_n=  4, thread_count= 64, register_count_per_thread= 72, total_register_count= 4845, shared_memory_bytes= 20480
      AddMM(16,  64,   4,   4,  32); // block_tile_n= 64, tile_k= 32, thread_tile_m=  4, thread_tile_n=  4, thread_count= 64, register_count_per_thread=112, total_register_count= 7395, shared_memory_bytes= 40960
      AddMM(16,  64,   4,   4,  64); // block_tile_n= 64, tile_k= 64, thread_tile_m=  4, thread_tile_n=  4, thread_count= 64, register_count_per_thread=192, total_register_count=12495, shared_memory_bytes= 81920
      AddMM(16,  64,   4,   8,   8); // block_tile_n= 64, tile_k=  8, thread_tile_m=  4, thread_tile_n=  8, thread_count= 32, register_count_per_thread= 96, total_register_count= 3315, shared_memory_bytes= 10240
      AddMM(16,  64,   4,   8,  16); // block_tile_n= 64, tile_k= 16, thread_tile_m=  4, thread_tile_n=  8, thread_count= 32, register_count_per_thread=136, total_register_count= 4590, shared_memory_bytes= 20480
      AddMM(16,  64,   4,   8,  32); // block_tile_n= 64, tile_k= 32, thread_tile_m=  4, thread_tile_n=  8, thread_count= 32, register_count_per_thread=216, total_register_count= 7140, shared_memory_bytes= 40960
      AddMM(16,  64,   8,   4,   8); // block_tile_n= 64, tile_k=  8, thread_tile_m=  8, thread_tile_n=  4, thread_count= 32, register_count_per_thread= 96, total_register_count= 3315, shared_memory_bytes= 10240
      AddMM(16,  64,   8,   4,  16); // block_tile_n= 64, tile_k= 16, thread_tile_m=  8, thread_tile_n=  4, thread_count= 32, register_count_per_thread=136, total_register_count= 4590, shared_memory_bytes= 20480
      AddMM(16,  64,   8,   4,  32); // block_tile_n= 64, tile_k= 32, thread_tile_m=  8, thread_tile_n=  4, thread_count= 32, register_count_per_thread=216, total_register_count= 7140, shared_memory_bytes= 40960
      AddMM(16, 128,   4,   4,  32); // block_tile_n=128, tile_k= 32, thread_tile_m=  4, thread_tile_n=  4, thread_count=128, register_count_per_thread=104, total_register_count=13515, shared_memory_bytes= 73728
      AddMM(16, 128,   4,   8,  16); // block_tile_n=128, tile_k= 16, thread_tile_m=  4, thread_tile_n=  8, thread_count= 64, register_count_per_thread=128, total_register_count= 8415, shared_memory_bytes= 36864
      AddMM(16, 128,   4,   8,  32); // block_tile_n=128, tile_k= 32, thread_tile_m=  4, thread_tile_n=  8, thread_count= 64, register_count_per_thread=200, total_register_count=13005, shared_memory_bytes= 73728
      AddMM(16, 128,   4,  16,   8); // block_tile_n=128, tile_k=  8, thread_tile_m=  4, thread_tile_n= 16, thread_count= 32, register_count_per_thread=176, total_register_count= 5865, shared_memory_bytes= 18432
      AddMM(16, 128,   8,   4,  16); // block_tile_n=128, tile_k= 16, thread_tile_m=  8, thread_tile_n=  4, thread_count= 64, register_count_per_thread=128, total_register_count= 8415, shared_memory_bytes= 36864
      AddMM(16, 128,   8,   4,  32); // block_tile_n=128, tile_k= 32, thread_tile_m=  8, thread_tile_n=  4, thread_count= 64, register_count_per_thread=200, total_register_count=13005, shared_memory_bytes= 73728
      AddMM(16, 128,   8,   8,   8); // block_tile_n=128, tile_k=  8, thread_tile_m=  8, thread_tile_n=  8, thread_count= 32, register_count_per_thread=168, total_register_count= 5610, shared_memory_bytes= 18432
      AddMM(16, 128,  16,   4,   8); // block_tile_n=128, tile_k=  8, thread_tile_m= 16, thread_tile_n=  4, thread_count= 32, register_count_per_thread=176, total_register_count= 5865, shared_memory_bytes= 18432
      /* clang-format on */
    }
    else if constexpr (BLOCK_TILE_M == 32) {
      /* clang-format off */
      AddMM(32,  16,   4,   4,   8); // block_tile_n= 16, tile_k=  8, thread_tile_m=  4, thread_tile_n=  4, thread_count= 32, register_count_per_thread= 56, total_register_count= 2040, shared_memory_bytes=  6144
      AddMM(32,  16,   4,   4,  16); // block_tile_n= 16, tile_k= 16, thread_tile_m=  4, thread_tile_n=  4, thread_count= 32, register_count_per_thread= 80, total_register_count= 2805, shared_memory_bytes= 12288
      AddMM(32,  16,   4,   4,  32); // block_tile_n= 16, tile_k= 32, thread_tile_m=  4, thread_tile_n=  4, thread_count= 32, register_count_per_thread=128, total_register_count= 4335, shared_memory_bytes= 24576
      AddMM(32,  16,   4,   4,  64); // block_tile_n= 16, tile_k= 64, thread_tile_m=  4, thread_tile_n=  4, thread_count= 32, register_count_per_thread=224, total_register_count= 7395, shared_memory_bytes= 49152
      AddMM(32,  32,   4,   4,   8); // block_tile_n= 32, tile_k=  8, thread_tile_m=  4, thread_tile_n=  4, thread_count= 64, register_count_per_thread= 48, total_register_count= 3315, shared_memory_bytes=  8192
      AddMM(32,  32,   4,   4,  16); // block_tile_n= 32, tile_k= 16, thread_tile_m=  4, thread_tile_n=  4, thread_count= 64, register_count_per_thread= 64, total_register_count= 4335, shared_memory_bytes= 16384
      AddMM(32,  32,   4,   4,  32); // block_tile_n= 32, tile_k= 32, thread_tile_m=  4, thread_tile_n=  4, thread_count= 64, register_count_per_thread= 96, total_register_count= 6375, shared_memory_bytes= 32768
      AddMM(32,  32,   4,   4,  64); // block_tile_n= 32, tile_k= 64, thread_tile_m=  4, thread_tile_n=  4, thread_count= 64, register_count_per_thread=160, total_register_count=10455, shared_memory_bytes= 65536
      AddMM(32,  32,   4,   8,   8); // block_tile_n= 32, tile_k=  8, thread_tile_m=  4, thread_tile_n=  8, thread_count= 32, register_count_per_thread= 88, total_register_count= 3060, shared_memory_bytes=  8192
      AddMM(32,  32,   4,   8,  16); // block_tile_n= 32, tile_k= 16, thread_tile_m=  4, thread_tile_n=  8, thread_count= 32, register_count_per_thread=120, total_register_count= 4080, shared_memory_bytes= 16384
      AddMM(32,  32,   4,   8,  32); // block_tile_n= 32, tile_k= 32, thread_tile_m=  4, thread_tile_n=  8, thread_count= 32, register_count_per_thread=184, total_register_count= 6120, shared_memory_bytes= 32768
      AddMM(32,  32,   8,   4,   8); // block_tile_n= 32, tile_k=  8, thread_tile_m=  8, thread_tile_n=  4, thread_count= 32, register_count_per_thread= 88, total_register_count= 3060, shared_memory_bytes=  8192
      AddMM(32,  32,   8,   4,  16); // block_tile_n= 32, tile_k= 16, thread_tile_m=  8, thread_tile_n=  4, thread_count= 32, register_count_per_thread=120, total_register_count= 4080, shared_memory_bytes= 16384
      AddMM(32,  32,   8,   4,  32); // block_tile_n= 32, tile_k= 32, thread_tile_m=  8, thread_tile_n=  4, thread_count= 32, register_count_per_thread=184, total_register_count= 6120, shared_memory_bytes= 32768
      AddMM(32,  64,   4,   4,  16); // block_tile_n= 64, tile_k= 16, thread_tile_m=  4, thread_tile_n=  4, thread_count=128, register_count_per_thread= 56, total_register_count= 7395, shared_memory_bytes= 24576
      AddMM(32,  64,   4,   4,  32); // block_tile_n= 64, tile_k= 32, thread_tile_m=  4, thread_tile_n=  4, thread_count=128, register_count_per_thread= 80, total_register_count=10455, shared_memory_bytes= 49152
      AddMM(32,  64,   4,   4,  64); // block_tile_n= 64, tile_k= 64, thread_tile_m=  4, thread_tile_n=  4, thread_count=128, register_count_per_thread=128, total_register_count=16575, shared_memory_bytes= 98304
      AddMM(32,  64,   4,   8,   8); // block_tile_n= 64, tile_k=  8, thread_tile_m=  4, thread_tile_n=  8, thread_count= 64, register_count_per_thread= 80, total_register_count= 5355, shared_memory_bytes= 12288
      AddMM(32,  64,   4,   8,  16); // block_tile_n= 64, tile_k= 16, thread_tile_m=  4, thread_tile_n=  8, thread_count= 64, register_count_per_thread=104, total_register_count= 6885, shared_memory_bytes= 24576
      AddMM(32,  64,   4,   8,  32); // block_tile_n= 64, tile_k= 32, thread_tile_m=  4, thread_tile_n=  8, thread_count= 64, register_count_per_thread=152, total_register_count= 9945, shared_memory_bytes= 49152
      AddMM(32,  64,   4,  16,   8); // block_tile_n= 64, tile_k=  8, thread_tile_m=  4, thread_tile_n= 16, thread_count= 32, register_count_per_thread=152, total_register_count= 5100, shared_memory_bytes= 12288
      AddMM(32,  64,   4,  16,  16); // block_tile_n= 64, tile_k= 16, thread_tile_m=  4, thread_tile_n= 16, thread_count= 32, register_count_per_thread=200, total_register_count= 6630, shared_memory_bytes= 24576
      AddMM(32,  64,   8,   4,   8); // block_tile_n= 64, tile_k=  8, thread_tile_m=  8, thread_tile_n=  4, thread_count= 64, register_count_per_thread= 80, total_register_count= 5355, shared_memory_bytes= 12288
      AddMM(32,  64,   8,   4,  16); // block_tile_n= 64, tile_k= 16, thread_tile_m=  8, thread_tile_n=  4, thread_count= 64, register_count_per_thread=104, total_register_count= 6885, shared_memory_bytes= 24576
      AddMM(32,  64,   8,   4,  32); // block_tile_n= 64, tile_k= 32, thread_tile_m=  8, thread_tile_n=  4, thread_count= 64, register_count_per_thread=152, total_register_count= 9945, shared_memory_bytes= 49152
      AddMM(32,  64,   8,   8,   8); // block_tile_n= 64, tile_k=  8, thread_tile_m=  8, thread_tile_n=  8, thread_count= 32, register_count_per_thread=144, total_register_count= 4845, shared_memory_bytes= 12288
      AddMM(32,  64,   8,   8,  16); // block_tile_n= 64, tile_k= 16, thread_tile_m=  8, thread_tile_n=  8, thread_count= 32, register_count_per_thread=192, total_register_count= 6375, shared_memory_bytes= 24576
      AddMM(32,  64,  16,   4,   8); // block_tile_n= 64, tile_k=  8, thread_tile_m= 16, thread_tile_n=  4, thread_count= 32, register_count_per_thread=152, total_register_count= 5100, shared_memory_bytes= 12288
      AddMM(32,  64,  16,   4,  16); // block_tile_n= 64, tile_k= 16, thread_tile_m= 16, thread_tile_n=  4, thread_count= 32, register_count_per_thread=200, total_register_count= 6630, shared_memory_bytes= 24576
      AddMM(32, 128,   4,   4,  32); // block_tile_n=128, tile_k= 32, thread_tile_m=  4, thread_tile_n=  4, thread_count=256, register_count_per_thread= 72, total_register_count=18615, shared_memory_bytes= 81920
      AddMM(32, 128,   4,   8,  16); // block_tile_n=128, tile_k= 16, thread_tile_m=  4, thread_tile_n=  8, thread_count=128, register_count_per_thread= 96, total_register_count=12495, shared_memory_bytes= 40960
      AddMM(32, 128,   4,   8,  32); // block_tile_n=128, tile_k= 32, thread_tile_m=  4, thread_tile_n=  8, thread_count=128, register_count_per_thread=136, total_register_count=17595, shared_memory_bytes= 81920
      AddMM(32, 128,   4,  16,   8); // block_tile_n=128, tile_k=  8, thread_tile_m=  4, thread_tile_n= 16, thread_count= 64, register_count_per_thread=144, total_register_count= 9435, shared_memory_bytes= 20480
      AddMM(32, 128,   4,  16,  16); // block_tile_n=128, tile_k= 16, thread_tile_m=  4, thread_tile_n= 16, thread_count= 64, register_count_per_thread=184, total_register_count=11985, shared_memory_bytes= 40960
      AddMM(32, 128,   8,   4,  16); // block_tile_n=128, tile_k= 16, thread_tile_m=  8, thread_tile_n=  4, thread_count=128, register_count_per_thread= 96, total_register_count=12495, shared_memory_bytes= 40960
      AddMM(32, 128,   8,   4,  32); // block_tile_n=128, tile_k= 32, thread_tile_m=  8, thread_tile_n=  4, thread_count=128, register_count_per_thread=136, total_register_count=17595, shared_memory_bytes= 81920
      AddMM(32, 128,   8,   8,   8); // block_tile_n=128, tile_k=  8, thread_tile_m=  8, thread_tile_n=  8, thread_count= 64, register_count_per_thread=136, total_register_count= 8925, shared_memory_bytes= 20480
      AddMM(32, 128,   8,   8,  16); // block_tile_n=128, tile_k= 16, thread_tile_m=  8, thread_tile_n=  8, thread_count= 64, register_count_per_thread=176, total_register_count=11475, shared_memory_bytes= 40960
      AddMM(32, 128,  16,   4,   8); // block_tile_n=128, tile_k=  8, thread_tile_m= 16, thread_tile_n=  4, thread_count= 64, register_count_per_thread=144, total_register_count= 9435, shared_memory_bytes= 20480
      AddMM(32, 128,  16,   4,  16); // block_tile_n=128, tile_k= 16, thread_tile_m= 16, thread_tile_n=  4, thread_count= 64, register_count_per_thread=184, total_register_count=11985, shared_memory_bytes= 40960
      AddMM(32, 256,   4,  16,  16); // block_tile_n=256, tile_k= 16, thread_tile_m=  4, thread_tile_n= 16, thread_count=128, register_count_per_thread=176, total_register_count=22695, shared_memory_bytes= 73728
      AddMM(32, 256,   8,   8,  16); // block_tile_n=256, tile_k= 16, thread_tile_m=  8, thread_tile_n=  8, thread_count=128, register_count_per_thread=168, total_register_count=21675, shared_memory_bytes= 73728
      AddMM(32, 256,  16,   4,  16); // block_tile_n=256, tile_k= 16, thread_tile_m= 16, thread_tile_n=  4, thread_count=128, register_count_per_thread=176, total_register_count=22695, shared_memory_bytes= 73728
      /* clang-format on */
    }
    else if constexpr (BLOCK_TILE_M == 64) {
      /* clang-format off */
      AddMM(64,   8,   4,   4,  16); // block_tile_n=  8, tile_k= 16, thread_tile_m=  4, thread_tile_n=  4, thread_count= 32, register_count_per_thread=104, total_register_count= 3570, shared_memory_bytes= 18432
      AddMM(64,   8,   4,   4,  32); // block_tile_n=  8, tile_k= 32, thread_tile_m=  4, thread_tile_n=  4, thread_count= 32, register_count_per_thread=176, total_register_count= 5865, shared_memory_bytes= 36864
      AddMM(64,  16,   4,   4,  16); // block_tile_n= 16, tile_k= 16, thread_tile_m=  4, thread_tile_n=  4, thread_count= 64, register_count_per_thread= 72, total_register_count= 4845, shared_memory_bytes= 20480
      AddMM(64,  16,   4,   4,  32); // block_tile_n= 16, tile_k= 32, thread_tile_m=  4, thread_tile_n=  4, thread_count= 64, register_count_per_thread=112, total_register_count= 7395, shared_memory_bytes= 40960
      AddMM(64,  16,   4,   4,  64); // block_tile_n= 16, tile_k= 64, thread_tile_m=  4, thread_tile_n=  4, thread_count= 64, register_count_per_thread=192, total_register_count=12495, shared_memory_bytes= 81920
      AddMM(64,  16,   4,   8,   8); // block_tile_n= 16, tile_k=  8, thread_tile_m=  4, thread_tile_n=  8, thread_count= 32, register_count_per_thread= 96, total_register_count= 3315, shared_memory_bytes= 10240
      AddMM(64,  16,   4,   8,  16); // block_tile_n= 16, tile_k= 16, thread_tile_m=  4, thread_tile_n=  8, thread_count= 32, register_count_per_thread=136, total_register_count= 4590, shared_memory_bytes= 20480
      AddMM(64,  16,   4,   8,  32); // block_tile_n= 16, tile_k= 32, thread_tile_m=  4, thread_tile_n=  8, thread_count= 32, register_count_per_thread=216, total_register_count= 7140, shared_memory_bytes= 40960
      AddMM(64,  16,   8,   4,   8); // block_tile_n= 16, tile_k=  8, thread_tile_m=  8, thread_tile_n=  4, thread_count= 32, register_count_per_thread= 96, total_register_count= 3315, shared_memory_bytes= 10240
      AddMM(64,  16,   8,   4,  16); // block_tile_n= 16, tile_k= 16, thread_tile_m=  8, thread_tile_n=  4, thread_count= 32, register_count_per_thread=136, total_register_count= 4590, shared_memory_bytes= 20480
      AddMM(64,  16,   8,   4,  32); // block_tile_n= 16, tile_k= 32, thread_tile_m=  8, thread_tile_n=  4, thread_count= 32, register_count_per_thread=216, total_register_count= 7140, shared_memory_bytes= 40960
      AddMM(64,  32,   4,   4,  16); // block_tile_n= 32, tile_k= 16, thread_tile_m=  4, thread_tile_n=  4, thread_count=128, register_count_per_thread= 56, total_register_count= 7395, shared_memory_bytes= 24576
      AddMM(64,  32,   4,   4,  32); // block_tile_n= 32, tile_k= 32, thread_tile_m=  4, thread_tile_n=  4, thread_count=128, register_count_per_thread= 80, total_register_count=10455, shared_memory_bytes= 49152
      AddMM(64,  32,   4,   4,  64); // block_tile_n= 32, tile_k= 64, thread_tile_m=  4, thread_tile_n=  4, thread_count=128, register_count_per_thread=128, total_register_count=16575, shared_memory_bytes= 98304
      AddMM(64,  32,   4,   8,   8); // block_tile_n= 32, tile_k=  8, thread_tile_m=  4, thread_tile_n=  8, thread_count= 64, register_count_per_thread= 80, total_register_count= 5355, shared_memory_bytes= 12288
      AddMM(64,  32,   4,   8,  16); // block_tile_n= 32, tile_k= 16, thread_tile_m=  4, thread_tile_n=  8, thread_count= 64, register_count_per_thread=104, total_register_count= 6885, shared_memory_bytes= 24576
      AddMM(64,  32,   4,   8,  32); // block_tile_n= 32, tile_k= 32, thread_tile_m=  4, thread_tile_n=  8, thread_count= 64, register_count_per_thread=152, total_register_count= 9945, shared_memory_bytes= 49152
      AddMM(64,  32,   4,  16,   8); // block_tile_n= 32, tile_k=  8, thread_tile_m=  4, thread_tile_n= 16, thread_count= 32, register_count_per_thread=152, total_register_count= 5100, shared_memory_bytes= 12288
      AddMM(64,  32,   4,  16,  16); // block_tile_n= 32, tile_k= 16, thread_tile_m=  4, thread_tile_n= 16, thread_count= 32, register_count_per_thread=200, total_register_count= 6630, shared_memory_bytes= 24576
      AddMM(64,  32,   8,   4,   8); // block_tile_n= 32, tile_k=  8, thread_tile_m=  8, thread_tile_n=  4, thread_count= 64, register_count_per_thread= 80, total_register_count= 5355, shared_memory_bytes= 12288
      AddMM(64,  32,   8,   4,  16); // block_tile_n= 32, tile_k= 16, thread_tile_m=  8, thread_tile_n=  4, thread_count= 64, register_count_per_thread=104, total_register_count= 6885, shared_memory_bytes= 24576
      AddMM(64,  32,   8,   4,  32); // block_tile_n= 32, tile_k= 32, thread_tile_m=  8, thread_tile_n=  4, thread_count= 64, register_count_per_thread=152, total_register_count= 9945, shared_memory_bytes= 49152
      AddMM(64,  32,   8,   8,   8); // block_tile_n= 32, tile_k=  8, thread_tile_m=  8, thread_tile_n=  8, thread_count= 32, register_count_per_thread=144, total_register_count= 4845, shared_memory_bytes= 12288
      AddMM(64,  32,   8,   8,  16); // block_tile_n= 32, tile_k= 16, thread_tile_m=  8, thread_tile_n=  8, thread_count= 32, register_count_per_thread=192, total_register_count= 6375, shared_memory_bytes= 24576
      AddMM(64,  32,  16,   4,   8); // block_tile_n= 32, tile_k=  8, thread_tile_m= 16, thread_tile_n=  4, thread_count= 32, register_count_per_thread=152, total_register_count= 5100, shared_memory_bytes= 12288
      AddMM(64,  32,  16,   4,  16); // block_tile_n= 32, tile_k= 16, thread_tile_m= 16, thread_tile_n=  4, thread_count= 32, register_count_per_thread=200, total_register_count= 6630, shared_memory_bytes= 24576
      AddMM(64,  64,   4,   4,  16); // block_tile_n= 64, tile_k= 16, thread_tile_m=  4, thread_tile_n=  4, thread_count=256, register_count_per_thread= 48, total_register_count=12495, shared_memory_bytes= 32768
      AddMM(64,  64,   4,   4,  32); // block_tile_n= 64, tile_k= 32, thread_tile_m=  4, thread_tile_n=  4, thread_count=256, register_count_per_thread= 64, total_register_count=16575, shared_memory_bytes= 65536
      AddMM(64,  64,   4,   8,   8); // block_tile_n= 64, tile_k=  8, thread_tile_m=  4, thread_tile_n=  8, thread_count=128, register_count_per_thread= 72, total_register_count= 9435, shared_memory_bytes= 16384
      AddMM(64,  64,   4,   8,  16); // block_tile_n= 64, tile_k= 16, thread_tile_m=  4, thread_tile_n=  8, thread_count=128, register_count_per_thread= 88, total_register_count=11475, shared_memory_bytes= 32768
      AddMM(64,  64,   4,   8,  32); // block_tile_n= 64, tile_k= 32, thread_tile_m=  4, thread_tile_n=  8, thread_count=128, register_count_per_thread=120, total_register_count=15555, shared_memory_bytes= 65536
      AddMM(64,  64,   4,  16,   8); // block_tile_n= 64, tile_k=  8, thread_tile_m=  4, thread_tile_n= 16, thread_count= 64, register_count_per_thread=136, total_register_count= 8925, shared_memory_bytes= 16384
      AddMM(64,  64,   4,  16,  16); // block_tile_n= 64, tile_k= 16, thread_tile_m=  4, thread_tile_n= 16, thread_count= 64, register_count_per_thread=168, total_register_count=10965, shared_memory_bytes= 32768
      AddMM(64,  64,   8,   4,   8); // block_tile_n= 64, tile_k=  8, thread_tile_m=  8, thread_tile_n=  4, thread_count=128, register_count_per_thread= 72, total_register_count= 9435, shared_memory_bytes= 16384
      AddMM(64,  64,   8,   4,  16); // block_tile_n= 64, tile_k= 16, thread_tile_m=  8, thread_tile_n=  4, thread_count=128, register_count_per_thread= 88, total_register_count=11475, shared_memory_bytes= 32768
      AddMM(64,  64,   8,   4,  32); // block_tile_n= 64, tile_k= 32, thread_tile_m=  8, thread_tile_n=  4, thread_count=128, register_count_per_thread=120, total_register_count=15555, shared_memory_bytes= 65536
      AddMM(64,  64,   8,   8,   8); // block_tile_n= 64, tile_k=  8, thread_tile_m=  8, thread_tile_n=  8, thread_count= 64, register_count_per_thread=128, total_register_count= 8415, shared_memory_bytes= 16384
      AddMM(64,  64,   8,   8,  16); // block_tile_n= 64, tile_k= 16, thread_tile_m=  8, thread_tile_n=  8, thread_count= 64, register_count_per_thread=160, total_register_count=10455, shared_memory_bytes= 32768
      AddMM(64,  64,   8,   8,  32); // block_tile_n= 64, tile_k= 32, thread_tile_m=  8, thread_tile_n=  8, thread_count= 64, register_count_per_thread=224, total_register_count=14535, shared_memory_bytes= 65536
      AddMM(64,  64,  16,   4,   8); // block_tile_n= 64, tile_k=  8, thread_tile_m= 16, thread_tile_n=  4, thread_count= 64, register_count_per_thread=136, total_register_count= 8925, shared_memory_bytes= 16384
      AddMM(64,  64,  16,   4,  16); // block_tile_n= 64, tile_k= 16, thread_tile_m= 16, thread_tile_n=  4, thread_count= 64, register_count_per_thread=168, total_register_count=10965, shared_memory_bytes= 32768
      AddMM(64, 128,   4,   8,  16); // block_tile_n=128, tile_k= 16, thread_tile_m=  4, thread_tile_n=  8, thread_count=256, register_count_per_thread= 80, total_register_count=20655, shared_memory_bytes= 49152
      AddMM(64, 128,   4,   8,  32); // block_tile_n=128, tile_k= 32, thread_tile_m=  4, thread_tile_n=  8, thread_count=256, register_count_per_thread=104, total_register_count=26775, shared_memory_bytes= 98304
      AddMM(64, 128,   4,  16,   8); // block_tile_n=128, tile_k=  8, thread_tile_m=  4, thread_tile_n= 16, thread_count=128, register_count_per_thread=128, total_register_count=16575, shared_memory_bytes= 24576
      AddMM(64, 128,   4,  16,  16); // block_tile_n=128, tile_k= 16, thread_tile_m=  4, thread_tile_n= 16, thread_count=128, register_count_per_thread=152, total_register_count=19635, shared_memory_bytes= 49152
      AddMM(64, 128,   4,  16,  32); // block_tile_n=128, tile_k= 32, thread_tile_m=  4, thread_tile_n= 16, thread_count=128, register_count_per_thread=200, total_register_count=25755, shared_memory_bytes= 98304
      AddMM(64, 128,   8,   4,  16); // block_tile_n=128, tile_k= 16, thread_tile_m=  8, thread_tile_n=  4, thread_count=256, register_count_per_thread= 80, total_register_count=20655, shared_memory_bytes= 49152
      AddMM(64, 128,   8,   4,  32); // block_tile_n=128, tile_k= 32, thread_tile_m=  8, thread_tile_n=  4, thread_count=256, register_count_per_thread=104, total_register_count=26775, shared_memory_bytes= 98304
      AddMM(64, 128,   8,   8,   8); // block_tile_n=128, tile_k=  8, thread_tile_m=  8, thread_tile_n=  8, thread_count=128, register_count_per_thread=120, total_register_count=15555, shared_memory_bytes= 24576
      AddMM(64, 128,   8,   8,  16); // block_tile_n=128, tile_k= 16, thread_tile_m=  8, thread_tile_n=  8, thread_count=128, register_count_per_thread=144, total_register_count=18615, shared_memory_bytes= 49152
      AddMM(64, 128,   8,   8,  32); // block_tile_n=128, tile_k= 32, thread_tile_m=  8, thread_tile_n=  8, thread_count=128, register_count_per_thread=192, total_register_count=24735, shared_memory_bytes= 98304
      AddMM(64, 128,   8,  16,   8); // block_tile_n=128, tile_k=  8, thread_tile_m=  8, thread_tile_n= 16, thread_count= 64, register_count_per_thread=224, total_register_count=14535, shared_memory_bytes= 24576
      AddMM(64, 128,  16,   4,   8); // block_tile_n=128, tile_k=  8, thread_tile_m= 16, thread_tile_n=  4, thread_count=128, register_count_per_thread=128, total_register_count=16575, shared_memory_bytes= 24576
      AddMM(64, 128,  16,   4,  16); // block_tile_n=128, tile_k= 16, thread_tile_m= 16, thread_tile_n=  4, thread_count=128, register_count_per_thread=152, total_register_count=19635, shared_memory_bytes= 49152
      AddMM(64, 128,  16,   4,  32); // block_tile_n=128, tile_k= 32, thread_tile_m= 16, thread_tile_n=  4, thread_count=128, register_count_per_thread=200, total_register_count=25755, shared_memory_bytes= 98304
      AddMM(64, 128,  16,   8,   8); // block_tile_n=128, tile_k=  8, thread_tile_m= 16, thread_tile_n=  8, thread_count= 64, register_count_per_thread=224, total_register_count=14535, shared_memory_bytes= 24576
      AddMM(64, 256,   4,  16,  16); // block_tile_n=256, tile_k= 16, thread_tile_m=  4, thread_tile_n= 16, thread_count=256, register_count_per_thread=144, total_register_count=36975, shared_memory_bytes= 81920
      AddMM(64, 256,   8,   8,  16); // block_tile_n=256, tile_k= 16, thread_tile_m=  8, thread_tile_n=  8, thread_count=256, register_count_per_thread=136, total_register_count=34935, shared_memory_bytes= 81920
      AddMM(64, 256,   8,  16,   8); // block_tile_n=256, tile_k=  8, thread_tile_m=  8, thread_tile_n= 16, thread_count=128, register_count_per_thread=216, total_register_count=27795, shared_memory_bytes= 40960
      AddMM(64, 256,  16,   4,  16); // block_tile_n=256, tile_k= 16, thread_tile_m= 16, thread_tile_n=  4, thread_count=256, register_count_per_thread=144, total_register_count=36975, shared_memory_bytes= 81920
      AddMM(64, 256,  16,   8,   8); // block_tile_n=256, tile_k=  8, thread_tile_m= 16, thread_tile_n=  8, thread_count=128, register_count_per_thread=216, total_register_count=27795, shared_memory_bytes= 40960
      /* clang-format on */
    }
    else if constexpr (BLOCK_TILE_M == 128) {
      /* clang-format off */
      AddMM(128, 128,   4,  16,   8); // block_tile_n=128, tile_k=  8, thread_tile_m=  4, thread_tile_n= 16, thread_count=256, register_count_per_thread=120, total_register_count=30855, shared_memory_bytes= 32768
      AddMM(128, 128,   4,  16,  16); // block_tile_n=128, tile_k= 16, thread_tile_m=  4, thread_tile_n= 16, thread_count=256, register_count_per_thread=136, total_register_count=34935, shared_memory_bytes= 65536
      AddMM(128, 128,   8,   8,   8); // block_tile_n=128, tile_k=  8, thread_tile_m=  8, thread_tile_n=  8, thread_count=256, register_count_per_thread=112, total_register_count=28815, shared_memory_bytes= 32768
      AddMM(128, 128,   8,   8,  16); // block_tile_n=128, tile_k= 16, thread_tile_m=  8, thread_tile_n=  8, thread_count=256, register_count_per_thread=128, total_register_count=32895, shared_memory_bytes= 65536
      AddMM(128, 128,   8,  16,   8); // block_tile_n=128, tile_k=  8, thread_tile_m=  8, thread_tile_n= 16, thread_count=128, register_count_per_thread=208, total_register_count=26775, shared_memory_bytes= 32768
      AddMM(128, 128,  16,   4,   8); // block_tile_n=128, tile_k=  8, thread_tile_m= 16, thread_tile_n=  4, thread_count=256, register_count_per_thread=120, total_register_count=30855, shared_memory_bytes= 32768
      AddMM(128, 128,  16,   4,  16); // block_tile_n=128, tile_k= 16, thread_tile_m= 16, thread_tile_n=  4, thread_count=256, register_count_per_thread=136, total_register_count=34935, shared_memory_bytes= 65536
      AddMM(128, 128,  16,   8,   8); // block_tile_n=128, tile_k=  8, thread_tile_m= 16, thread_tile_n=  8, thread_count=128, register_count_per_thread=208, total_register_count=26775, shared_memory_bytes= 32768
      AddMM(128, 256,   4,  32,   8); // block_tile_n=256, tile_k=  8, thread_tile_m=  4, thread_tile_n= 32, thread_count=256, register_count_per_thread=224, total_register_count=57375, shared_memory_bytes= 49152
      AddMM(128, 256,   8,  16,   8); // block_tile_n=256, tile_k=  8, thread_tile_m=  8, thread_tile_n= 16, thread_count=256, register_count_per_thread=200, total_register_count=51255, shared_memory_bytes= 49152
      AddMM(128, 256,   8,  16,  16); // block_tile_n=256, tile_k= 16, thread_tile_m=  8, thread_tile_n= 16, thread_count=256, register_count_per_thread=224, total_register_count=57375, shared_memory_bytes= 98304
      AddMM(128, 256,  16,   8,   8); // block_tile_n=256, tile_k=  8, thread_tile_m= 16, thread_tile_n=  8, thread_count=256, register_count_per_thread=200, total_register_count=51255, shared_memory_bytes= 49152
      AddMM(128, 256,  16,   8,  16); // block_tile_n=256, tile_k= 16, thread_tile_m= 16, thread_tile_n=  8, thread_count=256, register_count_per_thread=224, total_register_count=57375, shared_memory_bytes= 98304
      AddMM(128, 256,  32,   4,   8); // block_tile_n=256, tile_k=  8, thread_tile_m= 32, thread_tile_n=  4, thread_count=256, register_count_per_thread=224, total_register_count=57375, shared_memory_bytes= 49152
      /* clang-format on */
    }
    else {
      static_assert(16 <= BLOCK_TILE_M && BLOCK_TILE_M <= 128 && (BLOCK_TILE_M & (BLOCK_TILE_M - 1)) == 0);
    }
#undef AddMM
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
