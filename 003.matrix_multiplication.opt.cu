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

// C = A*B
// N and K must divided by 128.
template <int BLOCK_TILE_M, int BLOCK_TILE_N, int THREAD_TILE_M,
          int THREAD_TILE_N>
__global__ void sgemm(const float *A, const float *B, float *C, int M, int N,
                      int K) {
  static_assert(BLOCK_TILE_M == 128);
  static_assert(BLOCK_TILE_N == 128 || BLOCK_TILE_N == 256);
  static_assert(THREAD_TILE_M == 8);
  static_assert(THREAD_TILE_N == 8 || THREAD_TILE_N == 16);

  int C_m_top = blockIdx.y * BLOCK_TILE_M;
  int C_n_left = blockIdx.x * BLOCK_TILE_N;
  int C_m_bottom = min(C_m_top + BLOCK_TILE_M, M);
  int C_n_right = min(C_n_left + BLOCK_TILE_N, N);
  

  constexpr int k_per_iter = 8;
  constexpr int A_sm_floats = BLOCK_TILE_M * k_per_iter;
  constexpr int B_sm_floats = BLOCK_TILE_N * k_per_iter;
  constexpr int total_sm_bytes = (A_sm_floats + B_sm_floats);
  __shared__ float shared_memory_pool[total_sm_bytes];

  float *A_sm = shared_memory_pool;
  float *B_sm = shared_memory_pool + A_sm_floats;

  constexpr int thread_count =
      BLOCK_TILE_M / THREAD_TILE_M * BLOCK_TILE_N / THREAD_TILE_N;
  const int thread_index = threadIdx.x;

  static_assert(thread_count == 128 || thread_count == 256);

  // Both A_ldg_iters and B_ldg_iters can only be 2 or 4.
  constexpr int A_ldg_iters = BLOCK_TILE_M * k_per_iter / thread_count;
  constexpr int B_ldg_iters = BLOCK_TILE_N * k_per_iter / thread_count;
  // registers for LDG. 
  float A_ldg_reg[A_ldg_iters];
  float B_ldg_reg[B_ldg_iters];

  // registers for computing.
  float A_reg[THREAD_TILE_M];
  float B_reg[THREAD_TILE_N];
  float C_reg[THREAD_TILE_M * THREAD_TILE_N];

  // main loop
  for (int k = 0; k < K; k += k_per_iter) {
    for (int i = 0; i < A_ldg_iters; ++i) {
    }
    for (int i = 0; i < B_ldg_iters; ++i) {
    }
    __syncthreads();

    float reg = 0;
    for (int i = 0; i < A_sm_floats; ++i) {
      reg += A_sm[i];
    }
    for (int i = 0; i < B_sm_floats; ++i) {
      reg += B_sm[i];
    }
    C[0] += reg;

    // if (k < 128 && thread_index == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
    //   printf("\nA_sm begin\n");
    //   for (int i = 0; i < 128 * 8; ++i) {
    //     if (i % 32 == 0) {
    //       printf("\n");
    //     }
    //     printf("%03d_%03d ", int(A_sm[i]) / 128, int(A_sm[i]) % 128);
    //   }
    //   printf("\nA_sm end\n");
    // }

    // if (k < 128 && thread_index == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
    //   printf("\nB_sm begin\n");
    //   for (int i = 0; i < 128 * 8; ++i) {
    //     if (i % 32 == 0) {
    //       printf("\n");
    //     }
    //     printf("%03d_%03d ", int(B_sm[i]) % 128, int(B_sm[i]) / 128);
    //   }
    //   printf("\nB_sm end\n");
    // }
    __syncthreads();
  }
}

template <int BLOCK_TILE_M, int BLOCK_TILE_N, int THREAD_TILE_M,
          int THREAD_TILE_N>
void launch_sgemm(const float *A, const float *B, float *C, int M, int N,
                  int K) {
  if (N % 128 != 0 || K % 128 != 0) {
    throw std::runtime_error("invalid arguments");
  }

  dim3 grid(M / BLOCK_TILE_M, N / BLOCK_TILE_N);
  int block(BLOCK_TILE_M / THREAD_TILE_M * BLOCK_TILE_N / THREAD_TILE_N);

  sgemm<BLOCK_TILE_M, BLOCK_TILE_N, THREAD_TILE_M, THREAD_TILE_N>
      <<<grid, block>>>(A, B, C, M, N, K);
  CHECK_CUDA_ERROR();
}

int main() {
  static const int M = (1 << 12), N = (1 << 12), K = (1 << 12);
  // static const int M = 128, N = 128, K = 128;
  const float EPS = 1e-1;

  std::vector<float> host_A(M * K), host_B(K * N), host_C(M * N),
      host_result(M * N);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(-5, 5);
  for (auto &vec : {&host_A, &host_B}) {
#if 0
#if 0
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
  { launch_sgemm<128, 128, 8, 8>(A, B, C, M, N, K); }

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
