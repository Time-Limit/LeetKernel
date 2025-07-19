#include "util/error.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>

int main()
{
  constexpr auto loop_count = 1;
  constexpr auto M = 4, N = 4096, K = 4096;
  float *        A, *B, *C, *workspace;
  CHECK_CUDA_RETURN(cudaMalloc(&A, sizeof(float) * M * K));
  CHECK_CUDA_RETURN(cudaMalloc(&B, sizeof(float) * K * N));
  CHECK_CUDA_RETURN(cudaMalloc(&C, sizeof(float) * M * N));
  CHECK_CUDA_RETURN(cudaMalloc(&workspace, sizeof(float) * (M + N) * 16));
  float       milliseconds;
  cudaEvent_t start, stop;

  cublasHandle_t cublas_handle;
  cublasSetWorkspace_v2(cublas_handle, workspace, sizeof(float) * M * N * 16);
  cublasCreate(&cublas_handle);
  for (int m = 1; m <= M; m += (m < 128 ? 1 : 32)) {
    CHECK_CUDA_RETURN(cudaEventCreate(&start));
    CHECK_CUDA_RETURN(cudaEventCreate(&stop));

    float alpha = 1.0;
    float beta  = 0;
    CHECK_CUDA_RETURN(cudaEventRecord(start));
    for (int loop = 0; loop < 1; ++loop) {
      auto result = cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T, m, N, K, &alpha, A, K, B, N, &beta, C, M);
      if (result != 0) {
        // throw std::runtime_error("failed to launch cublasSgemm, result=" + std::to_string(result));
        std::cout << "failed to launch cublasSgemm, result=" + std::to_string(result) << std::endl;
      }
    }
    CHECK_CUDA_RETURN(cudaEventRecord(stop));
    CHECK_CUDA_ERROR();
    cudaDeviceSynchronize();
    CHECK_CUDA_RETURN(cudaEventElapsedTime(&milliseconds, start, stop));
    CHECK_CUDA_RETURN(cudaEventDestroy(start));
    CHECK_CUDA_RETURN(cudaEventDestroy(stop));
    printf("warmup m = %4d, cost = %8.3f\n", m, milliseconds);
  }

  for (int m = 1; m <= M; m += (m < 128 ? 1 : 32)) {
    CHECK_CUDA_RETURN(cudaEventCreate(&start));
    CHECK_CUDA_RETURN(cudaEventCreate(&stop));

    float alpha = 1.0;
    float beta  = 0;
    CHECK_CUDA_RETURN(cudaEventRecord(start));
    for (int loop = 0; loop < loop_count; ++loop) {
      auto result = cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T, m, N, K, &alpha, A, K, B, N, &beta, C, M);
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
    printf("m = %4d, cost = %8.3f\n", m, milliseconds / loop_count);
  }
  cublasDestroy_v2(cublas_handle);
  return 0;
}
