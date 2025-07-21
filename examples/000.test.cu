#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_pipeline_primitives.h>
#include <cuda_runtime.h>

#include "util/error.h"

__global__ void test(const float* dram, size_t size)
{
  __shared__ float sm[1024];
  for (int i = 0; i < size && i < 1024; i += blockDim.x) {
    if (i + threadIdx.x < 1024) {
      sm[i + threadIdx.x] = dram[i + threadIdx.x];
    }
  }
  if (threadIdx.x == 0) {
    float sum = 0;
    for (int i = 0; i < 1024 && i < size; ++i) {
      sum += sm[i];
    }
    printf("sum = %8.3f\n", sum);
  }
}

int main() {
  float *dram;
  CHECK_CUDA_RETURN(cudaMalloc(&dram, 1024 * sizeof(float)));
  test<<<1, 32>>>(dram, 1024);
  cudaDeviceSynchronize();
  CHECK_CUDA_ERROR();
  return 0;
}
