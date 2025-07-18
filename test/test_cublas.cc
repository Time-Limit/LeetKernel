#include "util/error.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <vector>

int main()
{
  constexpr auto loop_count = 8192 * 4;
  constexpr auto M = 4096, N = 4096, K = 4096;
  // constexpr auto M = 16, N = 16, K = 16;
  float *        A, *B, *C;
  CHECK_CUDA_RETURN(cudaMalloc(&A, sizeof(float) * M * K));
  CHECK_CUDA_RETURN(cudaMalloc(&B, sizeof(float) * K * N));
  CHECK_CUDA_RETURN(cudaMalloc(&C, sizeof(float) * M * N));
  float        milliseconds;
  cudaEvent_t  start, stop;

  CHECK_CUDA_RETURN(cudaEventCreate(&start));
  CHECK_CUDA_RETURN(cudaEventCreate(&stop));

  cublasHandle_t blas_handle;
  cublasCreate(&blas_handle);
  float alpha = 1.0;
  float beta  = 0;
  CHECK_CUDA_RETURN(cudaEventRecord(start));
  for (int loop = 0; loop < loop_count; ++loop) {
    auto result = cublasSgemm(blas_handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &alpha, A, K, B, N, &beta, C, N);
    if (result != 0) {
      throw std::runtime_error("failed to launch cublasSgemm");
    }
  }
  CHECK_CUDA_RETURN(cudaEventRecord(stop));
  CHECK_CUDA_ERROR();
  cudaDeviceSynchronize();
  CHECK_CUDA_RETURN(cudaEventElapsedTime(&milliseconds, start, stop));
  CHECK_CUDA_RETURN(cudaEventDestroy(start));
  CHECK_CUDA_RETURN(cudaEventDestroy(stop));
  printf("milliseconds = %8.3f avg = %8.3f\n", milliseconds, milliseconds / loop_count);
  return 0;
}
