#include <cublasLt.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define CUDA_CHECK(call)                                                                                               \
  do {                                                                                                                 \
    cudaError_t err = call;                                                                                            \
    if (err != cudaSuccess) {                                                                                          \
      printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));                                \
      exit(EXIT_FAILURE);                                                                                              \
    }                                                                                                                  \
  } while (0)

#define CUBLAS_CHECK(call)                                                                                             \
  do {                                                                                                                 \
    cublasStatus_t status = call;                                                                                      \
    if (status != CUBLAS_STATUS_SUCCESS) {                                                                             \
      printf("cuBLAS error at %s %d: %d\n", __FILE__, __LINE__, status);                                               \
      exit(EXIT_FAILURE);                                                                                              \
    }                                                                                                                  \
  } while (0)

void printMatrix(const char* name, half* matrix, int rows, int cols)
{
  printf("%s:\n", name);
  float* h_matrix = new float[rows * cols];
  CUDA_CHECK(cudaMemcpy(h_matrix, matrix, rows * cols * sizeof(half), cudaMemcpyDeviceToHost));
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      printf("%.2f ", __half2float(h_matrix[i * cols + j]));
    }
    printf("\n");
  }
  printf("\n");
  delete[] h_matrix;
}

int main()
{
  // 矩阵维度 (M, N, K 为 8 的倍数)
  const int M = 4096;
  const int N = 4096;
  const int K = 4096;

  // 初始化矩阵
  half* h_A = new half[M * K];
  half* h_B = new half[K * N];
  half* h_C = new half[M * N];
  for (int i = 0; i < M * K; i++)
    h_A[i] = __float2half(static_cast<float>(i % 10 / 100.0));
  for (int i = 0; i < K * N; i++)
    h_B[i] = __float2half(static_cast<float>(i % 10 / 100.0));
  for (int i = 0; i < M * N; i++)
    h_C[i] = __float2half(0.0f);

  // 分配 GPU 内存
  half *d_A, *d_B, *d_C;
  CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(half)));
  CUDA_CHECK(cudaMemcpy(d_A, h_A, M * K * sizeof(half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B, K * N * sizeof(half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_C, h_C, M * N * sizeof(half), cudaMemcpyHostToDevice));

  // 初始化 cuBLASLt
  cublasLtHandle_t ltHandle;
  CUBLAS_CHECK(cublasLtCreate(&ltHandle));

  // 创建矩阵描述符
  cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc;
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_16F, M, K, M));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_16F, K, N, K));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16F, M, N, M));

  // 创建操作描述符
  cublasLtMatmulDesc_t operationDesc;
  CUBLAS_CHECK(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

  // 设置矩阵乘法参数
  cublasOperation_t transa = CUBLAS_OP_N;
  cublasOperation_t transb = CUBLAS_OP_N;
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));

  // 设置 alpha 和 beta
  float alpha = 1.0f;
  float beta  = 0.0f;

  // 配置启发式算法
  cublasLtMatmulPreference_t preference;
  CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
  size_t workspaceSize = 32 * 1024 * 1024;  // 32 MB 工作空间
  CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
    preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));
  cublasLtMatmulHeuristicResult_t heuristicResult;
  int                             returnedResults = 0;
  CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
    ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Cdesc, preference, 1, &heuristicResult, &returnedResults));

  // 分配工作空间
  void* workspace;
  CUDA_CHECK(cudaMalloc(&workspace, workspaceSize));

  // 执行矩阵乘法
  CUBLAS_CHECK(cublasLtMatmul(ltHandle,
                              operationDesc,
                              &alpha,
                              d_A,
                              Adesc,
                              d_B,
                              Bdesc,
                              &beta,
                              d_C,
                              Cdesc,
                              d_C,
                              Cdesc,
                              &heuristicResult.algo,
                              workspace,
                              workspaceSize,
                              0));
  CUBLAS_CHECK(cublasLtMatmul(ltHandle,
                              operationDesc,
                              &alpha,
                              d_A,
                              Adesc,
                              d_B,
                              Bdesc,
                              &beta,
                              d_C,
                              Cdesc,
                              d_C,
                              Cdesc,
                              &heuristicResult.algo,
                              workspace,
                              workspaceSize,
                              0));
  CUBLAS_CHECK(cublasLtMatmul(ltHandle,
                              operationDesc,
                              &alpha,
                              d_A,
                              Adesc,
                              d_B,
                              Bdesc,
                              &beta,
                              d_C,
                              Cdesc,
                              d_C,
                              Cdesc,
                              &heuristicResult.algo,
                              workspace,
                              workspaceSize,
                              0));
  CUBLAS_CHECK(cublasLtMatmul(ltHandle,
                              operationDesc,
                              &alpha,
                              d_A,
                              Adesc,
                              d_B,
                              Bdesc,
                              &beta,
                              d_C,
                              Cdesc,
                              d_C,
                              Cdesc,
                              &heuristicResult.algo,
                              workspace,
                              workspaceSize,
                              0));
  CUBLAS_CHECK(cublasLtMatmul(ltHandle,
                              operationDesc,
                              &alpha,
                              d_A,
                              Adesc,
                              d_B,
                              Bdesc,
                              &beta,
                              d_C,
                              Cdesc,
                              d_C,
                              Cdesc,
                              &heuristicResult.algo,
                              workspace,
                              workspaceSize,
                              0));
  CUBLAS_CHECK(cublasLtMatmul(ltHandle,
                              operationDesc,
                              &alpha,
                              d_A,
                              Adesc,
                              d_B,
                              Bdesc,
                              &beta,
                              d_C,
                              Cdesc,
                              d_C,
                              Cdesc,
                              &heuristicResult.algo,
                              workspace,
                              workspaceSize,
                              0));
  CUBLAS_CHECK(cublasLtMatmul(ltHandle,
                              operationDesc,
                              &alpha,
                              d_A,
                              Adesc,
                              d_B,
                              Bdesc,
                              &beta,
                              d_C,
                              Cdesc,
                              d_C,
                              Cdesc,
                              &heuristicResult.algo,
                              workspace,
                              workspaceSize,
                              0));
  CUBLAS_CHECK(cublasLtMatmul(ltHandle,
                              operationDesc,
                              &alpha,
                              d_A,
                              Adesc,
                              d_B,
                              Bdesc,
                              &beta,
                              d_C,
                              Cdesc,
                              d_C,
                              Cdesc,
                              &heuristicResult.algo,
                              workspace,
                              workspaceSize,
                              0));
  CUBLAS_CHECK(cublasLtMatmul(ltHandle,
                              operationDesc,
                              &alpha,
                              d_A,
                              Adesc,
                              d_B,
                              Bdesc,
                              &beta,
                              d_C,
                              Cdesc,
                              d_C,
                              Cdesc,
                              &heuristicResult.algo,
                              workspace,
                              workspaceSize,
                              0));
  CUBLAS_CHECK(cublasLtMatmul(ltHandle,
                              operationDesc,
                              &alpha,
                              d_A,
                              Adesc,
                              d_B,
                              Bdesc,
                              &beta,
                              d_C,
                              Cdesc,
                              d_C,
                              Cdesc,
                              &heuristicResult.algo,
                              workspace,
                              workspaceSize,
                              0));
  CUBLAS_CHECK(cublasLtMatmul(ltHandle,
                              operationDesc,
                              &alpha,
                              d_A,
                              Adesc,
                              d_B,
                              Bdesc,
                              &beta,
                              d_C,
                              Cdesc,
                              d_C,
                              Cdesc,
                              &heuristicResult.algo,
                              workspace,
                              workspaceSize,
                              0));

  // 打印结果（可选）
  // printMatrix("Result Matrix C", d_C, M, N);

  // 清理资源
  CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(Adesc));
  CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(Bdesc));
  CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(Cdesc));
  CUBLAS_CHECK(cublasLtMatmulDescDestroy(operationDesc));
  CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(preference));
  CUBLAS_CHECK(cublasLtDestroy(ltHandle));
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));
  CUDA_CHECK(cudaFree(workspace));
  delete[] h_A;
  delete[] h_B;
  delete[] h_C;

  return 0;
}
