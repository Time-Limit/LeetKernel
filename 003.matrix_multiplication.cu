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

// C has M * K floats.
// The kernel has to load N(in matrix A) + N(in B) floats for calculating a
// float in C. So the kernel need to load  2 * M * K * N and store M * K floats.
// And, the kernel will execute N additions and multiplications for calculating
// a float in C.
// So the computational intensity is I = 2MNK / (2MNK + MK), this is very low
// for 4090.
__global__ void matrix_multiplication_kernel_base(const float *A, const float *B,
                                             float *C, int M, int N, int K) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  int m = blockIdx.y * blockDim.y + threadIdx.y;

  float accumulator = 0.0;
  A += m * N;
  B += k;
  for (int i = 0; i < N; ++i) {
    accumulator += A[i] * B[i * K];
  }
  C[m * K + k] = accumulator;
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
void solve_base(const float *A, const float *B, float *C, int M, int N, int K) {
  dim3 threads_per_block(16, 16);
  dim3 blocks_per_grid((K + threads_per_block.x - 1) / threads_per_block.x,
                       (M + threads_per_block.y - 1) / threads_per_block.y);

  matrix_multiplication_kernel_base<<<blocks_per_grid, threads_per_block>>>(
      A, B, C, M, N, K);
  cudaDeviceSynchronize();
}

int main() {
  static const int M = (1 << 6), N = (1 << 5), K = (1 << 4);

  std::vector<float> host_A(M * N), host_B(N * K), host_C(M * K),
      host_result(M * K);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(-5.0f, 5.0f);
  for (auto &vec : {&host_A, &host_B, &host_C}) {
    for (auto &data : *vec) {
      data = dis(gen);
    }
  }

  {
    const float(*A_ptr)[N] = reinterpret_cast<const float(*)[N]>(host_A.data());
    const float(*B_ptr)[K] = reinterpret_cast<const float(*)[K]>(host_B.data());
    float(*C_ptr)[K] = reinterpret_cast<float(*)[K]>(host_result.data());

    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < K; ++j) {
        C_ptr[i][j] = 0;
        for (int k = 0; k < N; ++k) {
          C_ptr[i][j] += A_ptr[i][k] * B_ptr[k][j];
        }
      }
    }
  }

  float *A, *B, *C;
  for (auto &pair : {std::make_pair(host_A, &A), std::make_pair(host_B, &B),
                     std::make_pair(host_C, &C)}) {
    const std::vector<float> &host = pair.first;
    float *device = *pair.second;
    cudaMalloc(&device, sizeof(float) * host.size());
    cudaMemcpy(device, host.data(), sizeof(float) * host.size(),
               cudaMemcpyDefault);
  }

  {
    cudaMemset(C, 0, M * K * sizeof(float));
    solve_base(A, B, C, M, N, K);
    memset(host_C.data(), 0, sizeof(float) * host_C.size());
    cudaMemcpy(host_C.data(), C, sizeof(float) * host_C.size(),
               cudaMemcpyDefault);
    const float(*host_result_ptr)[K] =
        reinterpret_cast<const float(*)[K]>(host_result.data());
    const float(*device_result_ptr)[K] =
        reinterpret_cast<const float(*)[K]>(host_C.data());

    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        if (host_result_ptr[i][j] != device_result_ptr[i][j]) {
          std::stringstream ss;
          ss << "base, invalid result, m=" << i << ", n=" << j << ", expected "
             << host_result_ptr[i][j] << ", got " << device_result_ptr[i][j];
          throw std::runtime_error(ss.str());
        }
      }
    }
  }

  return 0;
}
