#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "util/error.h"

__global__ void dim1_load_float(const float *A, float *B, int N) {
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index < N) {
    B[index] = A[index];
  }
}

void test_dim1_load_float() {
  const int ALLOC_N = (1 << 20);
  const int REAL_N = (1 << 20);
  float *A, *B;
  cudaMalloc(&A, sizeof(float) * ALLOC_N);
  cudaMalloc(&B, sizeof(float) * ALLOC_N);

  dim3 threads_per_block(64);
  dim3 blocks_per_grid(REAL_N / 64);

  printf("A = %p, B = %p\n", A, B);

  dim1_load_float<<<blocks_per_grid, threads_per_block>>>(A, B, REAL_N);
  CHECK_CUDA_ERROR();

  cudaFree(A);
  cudaFree(B);
}

__global__ void dim2_load_float_x_major(const float *A, float *B, int N) {
  int index = blockIdx.x * blockDim.x * blockDim.y + threadIdx.x * blockDim.y +
              threadIdx.y;
  if (index < N) {
    B[index] = A[index];
  }
}

__global__ void dim2_load_float_y_major(const float *A, float *B, int N) {
  int index = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
              threadIdx.x;
  if (index < N) {
    B[index] = A[index];
  }
}

void test_dim2_load_float() {
  const int ALLOC_N = (1 << 20);
  const int REAL_N = (1 << 20);
  float *A, *B;
  cudaMalloc(&A, sizeof(float) * ALLOC_N);
  cudaMalloc(&B, sizeof(float) * ALLOC_N);

  dim3 threads_per_block(16, 16);
  dim3 blocks_per_grid(REAL_N / (16 * 16));

  printf("A = %p, B = %p\n", A, B);

  dim2_load_float_x_major<<<blocks_per_grid, threads_per_block>>>(A, B, REAL_N);
  dim2_load_float_y_major<<<blocks_per_grid, threads_per_block>>>(A, B, REAL_N);
  CHECK_CUDA_ERROR();

  cudaFree(A);
  cudaFree(B);
}

int main() {
  test_dim1_load_float();
  test_dim2_load_float();
  return 0;
}
