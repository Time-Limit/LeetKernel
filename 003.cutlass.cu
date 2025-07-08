#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

using namespace std;

// 检查CUDA调用的错误
void checkCudaError(cudaError_t error, const char* message) {
    if (error != cudaSuccess) {
        cerr << "CUDA Error: " << cudaGetErrorString(error) << " - " << message << endl;
        exit(EXIT_FAILURE);
    }
}

template <typename SUM_T>
__global__ void base(const float *A, const float *B, float *C, int M, int N,
                     int K) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  int m = blockIdx.y * blockDim.y + threadIdx.y;

  if (n >= N || m >= M) {
    return;
  }

  A += m * K;
  B += n;
  SUM_T sum = 0.0;
  for (int k = 0; k < K; ++k) {
    sum += A[k] * B[k * N];
  }
  C[m * N + n] = sum;
}

void launch_base(const float *A, const float *B, float *C, int M, int N,
                 int K) {
  dim3 threads_per_block(16, 16);
  dim3 blocks_per_grid((N + threads_per_block.x - 1) / threads_per_block.x,
                       (M + threads_per_block.y - 1) / threads_per_block.y);
  // base<float><<<blocks_per_grid, threads_per_block>>>(A, B, C, M, N, K);
  base<double><<<blocks_per_grid, threads_per_block>>>(A, B, C, M, N, K);
}

int main() {
    // 矩阵尺寸
    const int M = 4096;
    const int N = 4096;
    const int K = 4096;

    // 分配主机内存
    float* h_A = (float*)malloc(M * K * sizeof(float));
    float* h_B = (float*)malloc(K * N * sizeof(float));
    float* h_C = (float*)malloc(M * N * sizeof(float));

    // 初始化矩阵
    for (int i = 0; i < M * K; i++) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < K * N; i++) {
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // 分配设备内存
    float* d_A, *d_B, *d_C;
    checkCudaError(cudaMalloc(&d_A, M * K * sizeof(float)), "cudaMalloc d_A failed");
    checkCudaError(cudaMalloc(&d_B, K * N * sizeof(float)), "cudaMalloc d_B failed");
    checkCudaError(cudaMalloc(&d_C, M * N * sizeof(float)), "cudaMalloc d_C failed");

    // 将数据从主机复制到设备
    checkCudaError(cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy h_A to d_A failed");
    checkCudaError(cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy h_B to d_B failed");

    // 定义CUTLASS矩阵乘法类型
    using Gemm = cutlass::gemm::device::Gemm<float, cutlass::layout::RowMajor,
                                            float, cutlass::layout::RowMajor,
                                            float, cutlass::layout::RowMajor>;

    // 配置矩阵乘法参数
    Gemm::Arguments args({M, N, K},  // 矩阵尺寸
                        {d_A, K},     // A矩阵和其leading dimension
                        {d_B, N},     // B矩阵和其leading dimension
                        {d_C, N},     // C矩阵和其leading dimension
                        {d_C, N},     // D矩阵和其leading dimension
                        {1.0f, 0.0f}); // 阿尔法和贝塔标量

    // 创建并运行GEMM实例
    Gemm gemm_op;
    cutlass::Status status = gemm_op(args);
    if (status != cutlass::Status::kSuccess) {
        cerr << "CUTLASS GEMM operation failed!" << endl;
        return EXIT_FAILURE;
    }

    // 同步设备
    checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");

    // 将结果从设备复制到主机
    checkCudaError(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy d_C to h_C failed");

    launch_base(d_A, d_B, d_C, M, N, K);

    std::vector<float> base_C(M * N);
    checkCudaError(cudaMemcpy(base_C.data(), d_C, M * N * sizeof(float),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy d_C to h_C failed");

    for (int  i = 0;i < base_C.size(); ++i) {
      if (fabs(base_C[i] - h_C[i]) > 1e-1) {
        printf("i = %d, base = %f, cublas = %f\n", i, base_C[i], h_C[i]);
      }
    }

    // 释放设备内存
    checkCudaError(cudaFree(d_A), "cudaFree d_A failed");
    checkCudaError(cudaFree(d_B), "cudaFree d_B failed");
    checkCudaError(cudaFree(d_C), "cudaFree d_C failed");

    // 释放主机内存
    free(h_A);
    free(h_B);
    free(h_C);

    cout << "CUTLASS GEMM completed successfully!" << endl;

    return 0;
}    
