#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "util/error.h"
#include "util/util.cuh"

// The version that uses a float type accumulator and applies Kahan's algorithm
// to reduce precision loss.
__global__ void kahan(const float *A, const float *B, float *C, int M, int N,
                      int K);
void launch_kahan(const float *A, const float *B, float *C, int M, int N,
                  int K);

template <int BLOCK_TILE_M, int BLOCK_TILE_N, int THREAD_TILE_M,
          int THREAD_TILE_N>
__global__ void sgemm(const float *A, const float *B, float *C, const int M, const int N,
                      const int K) {
  static_assert(BLOCK_TILE_M == 128 || BLOCK_TILE_M == 256);
  static_assert(BLOCK_TILE_N == 128 || BLOCK_TILE_N == 256);
  static_assert(THREAD_TILE_M == 8 || THREAD_TILE_M == 16);
  static_assert(THREAD_TILE_N == 8 || THREAD_TILE_N == 16);
  static_assert(THREAD_TILE_M * THREAD_TILE_N == 128 ||
                THREAD_TILE_M * THREAD_TILE_N == 64);

  constexpr int thread_count =
      BLOCK_TILE_M / THREAD_TILE_M * BLOCK_TILE_N / THREAD_TILE_N;

  static_assert(thread_count == 256 || thread_count == 512);

  const int C_m_top = blockIdx.y * BLOCK_TILE_M;
  const int C_n_left = blockIdx.x * BLOCK_TILE_N;
  const int C_m_bottom = min(C_m_top + BLOCK_TILE_M, M);
  const int C_n_right = min(C_n_left + BLOCK_TILE_N, N);

  // if (threadIdx.x == 0) {
  //   printf("BLOCK_TILE_M = %03d, BLOCK_TILE_N = %03d, THREAD_TILE_M = %03d, "
  //          "THREAD_TILE_N = %03d, lock = %03d %03d %03d, top = %03d, left = "
  //          "%03d\n",
  //          BLOCK_TILE_M, BLOCK_TILE_N, THREAD_TILE_M, THREAD_TILE_N, blockIdx.x,
  //          blockIdx.y, blockIdx.z, C_m_top, C_n_left);
  // }

  const int thread_index = threadIdx.x;
  const int warp_id = thread_index / 32;
  const int lane_id = thread_index % 32;

  constexpr int m_thread_per_warp = 4;
  constexpr int m_float_per_warp = m_thread_per_warp * THREAD_TILE_M;
  static_assert(32 % m_thread_per_warp == 0, "invalid m_thread_per_warp");
  constexpr int n_thread_per_warp = 32 / m_thread_per_warp;
  constexpr int n_float_per_warp = n_thread_per_warp * THREAD_TILE_N;

  static_assert(BLOCK_TILE_M % m_float_per_warp == 0,
                "invalid m_float_per_warp");
  constexpr int m_warp_per_block = BLOCK_TILE_M / m_float_per_warp;
  static_assert(BLOCK_TILE_N % n_float_per_warp == 0,
                "invalid n_float_per_warp");
  constexpr int n_warp_per_block = BLOCK_TILE_N / n_float_per_warp;

  static_assert(m_warp_per_block * n_warp_per_block * 32 == thread_count);

  static_assert(THREAD_TILE_M % 8 == 0, "invalid THREAD_TILE_M");
  static_assert(THREAD_TILE_N % 8 == 0, "invalid THREAD_TILE_N");
  constexpr int ld_float_count_per_iter_per_thread = 8;
  constexpr int ld_m_iter_count =
      THREAD_TILE_M / ld_float_count_per_iter_per_thread;
  constexpr int ld_n_iter_count =
      THREAD_TILE_N / ld_float_count_per_iter_per_thread;

  // (warp_m_top,warp_n_left) is the coordinate relative to the top-left
  // point(0,0) of the current block.
  const int warp_m_top = warp_id / n_warp_per_block * m_float_per_warp;
  const int warp_n_left = warp_id % n_warp_per_block * n_float_per_warp;

  constexpr int k_per_iter = 8;
  constexpr int A_sm_floats = BLOCK_TILE_M * k_per_iter;
  constexpr int B_sm_floats = BLOCK_TILE_N * k_per_iter;
  constexpr int total_sm_floats = (A_sm_floats + B_sm_floats);
  __shared__ float shared_memory_pool[total_sm_floats];

  float *A_sm = shared_memory_pool;
  float *B_sm = shared_memory_pool + A_sm_floats;

  // Both A_ldg_iters and B_ldg_iters can only be 2 or 4.
  constexpr int A_ldg_iters = BLOCK_TILE_M * k_per_iter / thread_count;
  constexpr int B_ldg_iters = BLOCK_TILE_N * k_per_iter / thread_count;
  // registers for LDG. 
  float A_ldg_reg[A_ldg_iters];
  float B_ldg_reg[B_ldg_iters];

  // registers for computing.
  float A_reg[THREAD_TILE_M];
  float B_reg[THREAD_TILE_N];
  float C_reg[THREAD_TILE_M][THREAD_TILE_N] = {0};

  // main loop
  for (int k_offset = 0; k_offset < K; k_offset += k_per_iter) {
#pragma unroll
    for (int i = 0; i < A_ldg_iters; ++i) {
      const int m = i * thread_count / 8 + thread_index / 8;
      const int k = thread_index % 8;
      if (C_m_top + m < M && k_offset + k < K) {
        A_ldg_reg[i] = A[OFFSET(C_m_top + m, k_offset + k, K)];
        const int sm_layer = m / 32 * 8 + k;
        const int sm_bank = (k + m % 32 / 4) % 8 + m % 4 * 8;
        A_sm[sm_layer * 32 + sm_bank] = A_ldg_reg[i];
      }
    }
#pragma unroll
    for (int i = 0; i < B_ldg_iters; ++i) {
      int k = (i * thread_count + thread_index) / BLOCK_TILE_N;
      int n = (i * thread_count + thread_index) % BLOCK_TILE_N;
      if (k_offset + k < K && C_n_left + n < N) {
        B_ldg_reg[i] = B[OFFSET(k_offset + k, C_n_left + n, N)];
        const int sm_layer = n / 32 * 8 + k;
        const int sm_bank = (k + n) % 4 + (n & 0x1c);
        B_sm[sm_layer * 32 + sm_bank] = B_ldg_reg[i];
      }
    }
    __syncthreads();

#pragma unroll
    for (int k = 0; k < k_per_iter; ++k) {
      for (int i = 0; i < ld_m_iter_count; ++i) {
        const int m_in_block_0 =
            warp_m_top + i * m_thread_per_warp * 8 + lane_id % 8 / 2 * 8;
        FETCH_FLOAT4(
            A_reg[i * 8],
            A_sm[m_in_block_0 / 32 * 256 + k * 32 + m_in_block_0 % 32]);
        const int m_in_block_4 = m_in_block_0 + 4;
        FETCH_FLOAT4(
            A_reg[i * 8 + 4],
            A_sm[m_in_block_4 / 32 * 256 + k * 32 + m_in_block_4 % 32]);
      }
#pragma unroll
      for (int i = 0; i < ld_n_iter_count; ++i) {
        const int n_in_block_0 = warp_n_left + i * n_thread_per_warp * 8 +
                                 (lane_id % 8 + lane_id / 8 * 2) % 8 * 4;
        FETCH_FLOAT4(
            B_reg[i * 8],
            B_sm[n_in_block_0 / 32 * 256 + k * 32 + n_in_block_0 % 32]);
        const int n_in_block_32 = warp_n_left + i * n_thread_per_warp * 8 +
                                  (lane_id % 8 + lane_id / 8 * 2) % 8 * 4 + 32;
        FETCH_FLOAT4(
            B_reg[i * 8 + 4],
            B_sm[n_in_block_32 / 32 * 256 + k * 32 + n_in_block_32 % 32]);
      }
#pragma unroll
      for (int i = 0; i < THREAD_TILE_M; ++i) {
#pragma unroll
        for (int j = 0; j < THREAD_TILE_N; ++j) {
          C_reg[i][j] +=
              A_reg[(i + k) % 8 + (i & 0xf8)] * B_reg[(j + k) % 4 + (j & 0xfc)];
        }
      }
    }
    __syncthreads();
  }

  for (int i = 0; i < THREAD_TILE_M; ++i) {
    const int m =
        C_m_top + warp_m_top + i / 8 * 32 + i % 8 * 4 + lane_id % 8 / 2;
    if (m < M) {
      for (int j = 0; j < THREAD_TILE_N; j += 4) {
        const int n = C_n_left + warp_n_left + j / 4 * 32 +
                      (lane_id % 8 + lane_id / 8 * 2) % 8 * 4;
        STORE_FLOAT4(C[OFFSET(m, n, N)], C_reg[i][j]);
      }
    }
  }
}

template <int BLOCK_TILE_M, int BLOCK_TILE_N, int THREAD_TILE_M,
          int THREAD_TILE_N>
void launch_sgemm(const float *A, const float *B, float *C, int M, int N,
                  int K) {
  if (N % 128 != 0 || K % 128 != 0) {
    throw std::runtime_error("invalid arguments");
  }

  dim3 grid(N / BLOCK_TILE_N, M / BLOCK_TILE_M);
  int block(BLOCK_TILE_M / THREAD_TILE_M * BLOCK_TILE_N / THREAD_TILE_N);

  sgemm<BLOCK_TILE_M, BLOCK_TILE_N, THREAD_TILE_M, THREAD_TILE_N>
      <<<grid, block>>>(A, B, C, M, N, K);
  CHECK_CUDA_ERROR();
}

template <int BLOCK_TILE_M, int BLOCK_TILE_N, int THREAD_TILE_M,
          int THREAD_TILE_N>
void test_sgemm(const float *A, const float *B, float *C, int M, int N, int K,
                const float *result, std::vector<float> &host_C) {
  cudaMemset(C, 0, M * N * sizeof(float));
  launch_sgemm<BLOCK_TILE_M, BLOCK_TILE_N, THREAD_TILE_M, THREAD_TILE_N>(A, B, C, M, N, K);
  cudaMemcpy(host_C.data(), C, sizeof(float) * host_C.size(),
             cudaMemcpyDefault);
  const float(*host_result_ptr)[N] =
      reinterpret_cast<const float(*)[N]>(result);
  const float(*device_result_ptr)[N] =
      reinterpret_cast<const float(*)[N]>(host_C.data());

  constexpr float EPS = 1e-1;

  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      if (fabs(host_result_ptr[i][j] - device_result_ptr[i][j]) > EPS) {
        printf("%.7f, %.7f\n", host_result_ptr[i][j], device_result_ptr[i][j]);
        std::stringstream ss;
        ss << "sgemm, " << BLOCK_TILE_M << "x" << BLOCK_TILE_N << ", "
           << THREAD_TILE_M << "x" << THREAD_TILE_N
           << ", invalid result, m=" << i << ", n=" << j << ", expected "
           << host_result_ptr[i][j] << ", got " << device_result_ptr[i][j];
        throw std::runtime_error(ss.str());
      }
    }
  }
  std::cout << "success, BLOCK_TILE_M=" << BLOCK_TILE_M
            << ", BLOCK_TILE_N=" << BLOCK_TILE_N
            << ", THREAD_TILE_M=" << THREAD_TILE_M
            << ", THREAD_TILE_N=" << THREAD_TILE_N << std::endl;
}

int main() {
  static const int M = (1 << 12), N = (1 << 12), K = (1 << 12);
  // static const int M = 128, N = 128, K = 128;

  std::vector<float> host_A(M * K), host_B(K * N), host_C(M * N),
      host_result(M * N);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(-5, 5);
  for (auto &vec : {&host_A, &host_B}) {
#if 1
#if 1
    for (auto &data : *vec) {
      data = dis(gen);
    }
#else
    if (vec == &host_A) {
      for (size_t i = 0; i < vec->size(); ++i) {
        int row = i / K;
        int col = i % K;
        if (row < 4096 && col < 4096) {
          vec->operator[](i) = dis(gen);
        } else {
          vec->operator[](i) = 0;
        }
      }
      for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; ++j) {
          printf("%8.3f ", vec->at(i * K + j));
        }
        printf("\n");
      }
      printf("\n");
      printf("\n");
    }
    if (vec == &host_B) {
      for (size_t i = 0; i < vec->size(); ++i) {
        int row = i / N;
        int col = i % N;
        if (row < 8 && col < 9) {
          vec->operator[](i) = dis(gen);
        } else {
          vec->operator[](i) = 0;
        }
      }
      for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 16; ++j) {
          printf("%8.3f ", vec->at(i * N + j));
        }
        printf("\n");
      }
      printf("\n");
      printf("\n");
    }
#endif
#else
    if (vec == &host_A) {
      for (size_t i = 0; i < vec->size(); ++i) {
        int row = i / K;
        int col = i % K;
        if (row < 128 && col < 128) {
          vec->operator[](i) = row * 128 + col;
        } else {
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
        } else {
          vec->operator[](i) = 0;
        }
      }
    }
#endif
  }

  float *A, *B, *C;
  for (auto &pair : {std::make_pair(host_A, &A), std::make_pair(host_B, &B),
                     std::make_pair(host_C, &C)}) {
    const std::vector<float> &host = pair.first;
    float *&device = *pair.second;
    cudaMalloc(&device, sizeof(float) * host.size());
    cudaMemcpy(device, host.data(), sizeof(float) * host.size(),
               cudaMemcpyDefault);
    CHECK_CUDA_ERROR();
  }

  {
    cudaMemset(C, 0, M * N * sizeof(float));
    launch_kahan(A, B, C, M, N, K);
    cudaMemcpy(host_result.data(), C, sizeof(float) * host_C.size(),
               cudaMemcpyDefault);
  }

  { test_sgemm<128, 128, 8, 8>(A, B, C, M, N, K, host_result.data(), host_C); }
  { test_sgemm<128, 256, 8, 16>(A, B, C, M, N, K, host_result.data(), host_C); }
  { test_sgemm<128, 256, 16, 8>(A, B, C, M, N, K, host_result.data(), host_C); }
  { test_sgemm<256, 128, 16, 8>(A, B, C, M, N, K, host_result.data(), host_C); }
  { test_sgemm<256, 128, 8, 16>(A, B, C, M, N, K, host_result.data(), host_C); }

  return 0;
}

__global__ void kahan(const float *A, const float *B, float *C, int M, int N,
                      int K) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  int m = blockIdx.y * blockDim.y + threadIdx.y;

  if (n >= N || m >= M) {
    return;
  }

  A += m * K;
  B += n;
  float sum = 0.0, diff = 0.0;
  for (int k = 0; k < K; ++k) {
    float y = A[k] * B[k * N] - diff;
    float t = sum + y;
    diff = (t - sum) - y;
    sum = t;
  }
  C[m * N + n] = sum;
}

void launch_kahan(const float *A, const float *B, float *C, int M, int N,
                  int K) {
  dim3 threads_per_block(16, 16);
  dim3 blocks_per_grid((K + threads_per_block.x - 1) / threads_per_block.x,
                       (M + threads_per_block.y - 1) / threads_per_block.y);

  kahan<<<blocks_per_grid, threads_per_block>>>(A, B, C, M, N, K);
  CHECK_CUDA_ERROR();
}
