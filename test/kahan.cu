#include "util/error.h"

// The version that uses a float type accumulator and applies Kahan's algorithm
// to reduce precision loss.
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
#pragma unroll
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
  dim3 blocks_per_grid((N + threads_per_block.x - 1) / threads_per_block.x,
                       (M + threads_per_block.y - 1) / threads_per_block.y);

  kahan<<<blocks_per_grid, threads_per_block>>>(A, B, C, M, N, K);
  CHECK_CUDA_ERROR();
}
