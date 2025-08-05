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
__global__ void fp16_mma_m8n8k16_ldmatrix(const T* A, const T* B, const T* C, int M, int N, int K)
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
define_check_function(fp16_mma_m8n8k16_ldmatrix);
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
  fp16_mma_m8n8k16_ldmatrix___check_relative_error(fp16_A, fp16_B, fp16_C, M, N, K, host_C);
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
  // test<__nv_bfloat16>(host_A, host_B, host_C, M, N, K);

  CHECK_CUDA_RETURN(cudaFree(A));
  CHECK_CUDA_RETURN(cudaFree(B));
  CHECK_CUDA_RETURN(cudaFree(C));
  return 0;
}
