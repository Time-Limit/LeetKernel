#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "util/error.h"

__global__ void vector_add_base(const float *A, const float *B, float *C,
                                int N) {
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index < N) {
    C[index] = A[index] + B[index];
  }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
void solve_base(const float *A, const float *B, float *C, int N, int thread_per_block) {
  int block_per_grid = (N + thread_per_block - 1) / thread_per_block;

  vector_add_base<<<block_per_grid, thread_per_block>>>(A, B, C, N);
  CHECK_CUDA_ERROR();
  cudaDeviceSynchronize();
}

template <int DATA_PER_THREAD>
__global__ void vector_add_merge_request(const float *A, const float *B,
                                                float *C, int N) {
  static_assert(DATA_PER_THREAD > 0 && DATA_PER_THREAD % 4 == 0);
  const int data_per_block = blockDim.x * DATA_PER_THREAD;
  const int offset = blockIdx.x * data_per_block;

  for (int i = 0; i < DATA_PER_THREAD; i += 4) {
    int index = offset + i * blockDim.x + threadIdx.x * 4;
    if (index + 3 < N) {
      float4 a = reinterpret_cast<const float4 *>(A + index)[0];
      float4 b = reinterpret_cast<const float4 *>(B + index)[0];
      float4 c{a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};
      reinterpret_cast<float4 *>(C + index)[0] = c;
    } else {
      for (int j = 0; j < 3; ++j) {
        if (index + j < N) {
          C[index + j] = A[index + j] + B[index + j];
        }
      }
    }
  }
}

template<int DATA_PER_THREAD>
void solve_merge_request(const float *A, const float *B, float *C,
                                int N) {
  int thread_per_block = 128;
  int data_per_block = DATA_PER_THREAD * thread_per_block;
  int block_per_grid = (N + data_per_block - 1) / (data_per_block);

  vector_add_merge_request<DATA_PER_THREAD>
      <<<block_per_grid, thread_per_block>>>(A, B, C, N);
  CHECK_CUDA_ERROR();
  cudaDeviceSynchronize();
}

int main() {
  static const int N = (1 << 28);

  std::vector<float> host_A(N), host_B(N), host_C(N);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(-1000.0f, 1000.0f);
  for (auto &h : host_A) {
    h = dis(gen);
  }
  for (auto &h : host_B) {
    h = dis(gen);
  }

  float *A, *B, *C;
  cudaMalloc(&A, sizeof(float) * N);
  cudaMalloc(&B, sizeof(float) * N);
  cudaMalloc(&C, sizeof(float) * N);
  cudaMemcpy(A, host_A.data(), sizeof(float) * N, cudaMemcpyDefault);
  cudaMemcpy(B, host_B.data(), sizeof(float) * N, cudaMemcpyDefault);

#define launch_vector_add_base(thread_per_block)                              \
  {                                                                            \
    cudaMemset(C, 0, N * sizeof(float));                                       \
    solve_base(A, B, C, N, thread_per_block);                                 \
    memset(host_C.data(), 0, sizeof(float) * N);                               \
    cudaMemcpy(host_C.data(), C, sizeof(float) * N, cudaMemcpyDefault);        \
    for (int i = 0; i < N; ++i) {                                              \
      if (host_C[i] != host_A[i] + host_B[i]) {                                \
        std::stringstream ss;                                                  \
        ss << "base, invalid result, i=" << i << ", A=" << host_A[i]           \
           << ", B=" << host_B[i] << ", C=" << host_C[i];                      \
        throw std::runtime_error(ss.str());                                    \
      }                                                                        \
    }                                                                          \
  }

  launch_vector_add_base(1);
  launch_vector_add_base(2);
  launch_vector_add_base(4);
  launch_vector_add_base(8);
  launch_vector_add_base(16);
  launch_vector_add_base(32);
  launch_vector_add_base(64);
  launch_vector_add_base(128);
  launch_vector_add_base(256);
  launch_vector_add_base(512);
  launch_vector_add_base(768);
  launch_vector_add_base(1024);

#define launch_vector_add_merge_request(data_per_thread)                      \
  {                                                                            \
    cudaMemset(C, 0, N * sizeof(float));                                       \
    solve_merge_request<data_per_thread>(A, B, C, N);                          \
    memset(host_C.data(), 0, sizeof(float) * N);                               \
    cudaMemcpy(host_C.data(), C, sizeof(float) * N, cudaMemcpyDefault);        \
    for (int i = 0; i < N; ++i) {                                              \
      if (host_C[i] != host_A[i] + host_B[i]) {                                \
        std::stringstream ss;                                                  \
        ss << "merge request, invalid result, i=" << i << ", A=" << host_A[i]  \
           << ", B=" << host_B[i] << ", C=" << host_C[i];                      \
        throw std::runtime_error(ss.str());                                    \
      }                                                                        \
    }                                                                          \
  }

  launch_vector_add_merge_request(4);
  launch_vector_add_merge_request(8);
  launch_vector_add_merge_request(16);
  launch_vector_add_merge_request(32);
  launch_vector_add_merge_request(64);
  launch_vector_add_merge_request(128);
  launch_vector_add_merge_request(256);
  launch_vector_add_merge_request(512);
  launch_vector_add_merge_request(768);
  launch_vector_add_merge_request(1024 * 1);
  launch_vector_add_merge_request(1024 * 2);
  launch_vector_add_merge_request(1024 * 4);
  launch_vector_add_merge_request(1024 * 8);
  launch_vector_add_merge_request(1024 * 16);
  launch_vector_add_merge_request(1024 * 32);
  launch_vector_add_merge_request(1024 * 64);
  launch_vector_add_merge_request(1024 * 128);
  launch_vector_add_merge_request(1024 * 256);
  launch_vector_add_merge_request(1024 * 512);

  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
  return 0;
}
