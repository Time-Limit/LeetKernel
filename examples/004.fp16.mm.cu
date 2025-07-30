#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_pipeline_primitives.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <random>
#include <stdexcept>
#include <vector>

#include "util/error.h"
#include "util/util.cuh"

const int limit = 32;

__global__ void fp32_naive_mm(const float* A, const float* B, float* C, int M, int N, int K)
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
  static_assert(std::is_same<T, half>::value || std::is_same<T, __nv_bfloat16>::value);
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
__global__ void llmmm_fp16_mma_m16n8k8(const T* A, const T* B, T* C, int M, int N, int K)
{
}

template<typename T>
__global__ void llmmm_fp16_mma_m16n8k16(const T* A, const T* B, T* C, int M, int N, int K)
{
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
  llmmm_fp16_mma_m8n8k4<T, BLOCK_TILE_M, BLOCK_TILE_N, m_per_warp, n_per_warp, TILE_K>
    <<<grid, block>>>(A, B, C, M, N, K);
  CHECK_CUDA_ERROR();
}

template<typename T, typename = std::enable_if_t<std::is_same<T, half>::value || std::is_same<T, __nv_bfloat16>::value>>
int test()
{
  static const int M = (1 << 12), N = (1 << 12), K = (1 << 12);
  // static const int M = 128, N = 128, K = 128;

  std::vector<float>                    host_A(M * K), host_B(K * N), host_C(M * N);
  std::vector<T>                        host_fp16_A(M * K), host_fp16_B(K * N), host_fp16_C(M * N);
  std::random_device                    rd;
  std::mt19937                          gen(rd());
  std::uniform_real_distribution<float> dis(-1, 1);
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
  for (auto [fp32, fp16] : {std::make_pair(&host_A, &host_fp16_A),
                            std::make_pair(&host_B, &host_fp16_B),
                            std::make_pair(&host_C, &host_fp16_C)}) {
    for (int i = 0; i < fp16->size(); ++i) {
      fp16->at(i) = T(fp32->at(i));
    }
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

  launch_llmmm_fp16_mma_m8n8k4(fp16_A, fp16_B, fp16_C, M, N, K);
  cudaMemcpy(host_fp16_C.data(), fp16_C, sizeof(T) * host_fp16_C.size(), cudaMemcpyDefault);
  for (int i = 0; i < 16; ++i) {
    for (int j = 0; j < 16; ++j) {
      printf("%8.3f ", float(host_fp16_C[i * N + j]));
    }
    printf("\n");
  }
  for (int i = 0; i < host_fp16_C.size(); ++i) {
    if (fabs(float(host_fp16_C[i]) - host_C[i]) > 1e-1) {
      printf("i = %d, fp16 = %8.3f, fp32 = %8.3f\n", i, float(host_fp16_C[i]), host_C[i]);
      break;
    }
  }

  return 0;
}

int main()
{
  test<half>();
  // test<__nv_bfloat16>();
  return 0;
}
