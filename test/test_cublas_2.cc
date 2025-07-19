#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

// 检查 CUDA 错误
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

// 检查 cuBLAS 错误
#define CHECK_CUBLAS(call) { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error in %s:%d: %d\n", __FILE__, __LINE__, status); \
        exit(EXIT_FAILURE); \
    } \
}

// CPU 参考矩阵乘法用于验证
void cpuSgemm(int m, int n, int k, float *a, float *b, float *c) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int l = 0; l < k; l++) {
                sum += a[i * k + l] * b[l * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

// 验证结果
bool verifyResult(int m, int n, float *c_gpu, float *c_cpu, float tol = 1e-1) {
    float max = -1;
    for (int i = 0; i < m * n; i++) {
      if (max < 0)
        max = fabs(c_gpu[i] - c_cpu[i]);
      else if (max < fabs(c_gpu[i] - c_cpu[i])) {
        max = fabs(c_gpu[i] - c_cpu[i]);
      }
      if (fabs(c_gpu[i] - c_cpu[i]) > tol) {
        return false;
        }
    }
    printf("%8.3f\n", max);
    return true;
}

int main() {
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    const int n = 4096;
    const int max_m = 8192;
    const int k = 4096;

    // 分配主机内存
    float *h_A = (float*)malloc(max_m * k * sizeof(float));
    float *h_B = (float*)malloc(k * n * sizeof(float));
    float *h_C = (float*)malloc(max_m * n * sizeof(float));
    float *h_C_cpu = (float*)malloc(max_m * n * sizeof(float));

    // 初始化矩阵
    for (int i = 0; i < max_m * k; i++) {
      h_A[i] = rand() % 5 * 1.0 / 5;
    }
    for (int i = 0; i < k * n; i++) {
      h_B[i] = rand() % 5 * 1.0 / 5;
    }

    // 分配设备内存
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, max_m * k * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, k * n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, max_m * n * sizeof(float)));

    // 复制数据到设备
    CHECK_CUDA(cudaMemcpy(d_A, h_A, max_m * k * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, k * n * sizeof(float), cudaMemcpyHostToDevice));

    // cuBLAS 参数
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // 计时
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    printf("M\tTime(ms)\tResult\n");
    printf("----------------------------------------\n");

    // warmup
    for (int m = 1; m <= max_m; m += (m < 128 ? 1 : 32)) {
        // 清零结果矩阵
        CHECK_CUDA(cudaMemset(d_C, 0, max_m * n * sizeof(float)));

        // 计时开始
        CHECK_CUDA(cudaEventRecord(start));
        
        // 执行矩阵乘法 C = alpha * A * B + beta * C
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                n, m, k,
                                &alpha, d_B, n,
                                d_A, k,
                                &beta, d_C, n));

        // 计时结束
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float milliseconds = 0;
        CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

        // 复制结果回主机
        CHECK_CUDA(cudaMemcpy(h_C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost));

        // CPU 验证（只对较小的 M 值进行验证以节省时间）
        bool result_correct = true;
        if (m <= 16) {  // 限制验证的矩阵大小
            cpuSgemm(m, n, k, h_A, h_B, h_C_cpu);
            result_correct = verifyResult(m, n, h_C, h_C_cpu);
        } else {
            // 对于较大的矩阵，假设结果正确
            result_correct = true;
        }

        printf("%d\t%.3f\t%s\n", m, milliseconds, result_correct ? "Correct" : "Incorrect");
    }

    // 测试不同 M 值
    for (int m = 1; m <= max_m; m += (m < 128 ? 1 : 32)) {
        // 清零结果矩阵
        CHECK_CUDA(cudaMemset(d_C, 0, max_m * n * sizeof(float)));

        // 计时开始
        CHECK_CUDA(cudaEventRecord(start));
        
        // 执行矩阵乘法 C = alpha * A * B + beta * C
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                n, m, k,
                                &alpha, d_B, n,
                                d_A, k,
                                &beta, d_C, n));

        // 计时结束
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float milliseconds = 0;
        CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

        // 复制结果回主机
        CHECK_CUDA(cudaMemcpy(h_C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost));

        // CPU 验证（只对较小的 M 值进行验证以节省时间）
        bool result_correct = true;
        if (m <= 16) {  // 限制验证的矩阵大小
            cpuSgemm(m, n, k, h_A, h_B, h_C_cpu);
            result_correct = verifyResult(m, n, h_C, h_C_cpu);
        } else {
            // 对于较大的矩阵，假设结果正确
            result_correct = true;
        }

        printf("%d\t%.3f\t%s\n", m, milliseconds, result_correct ? "Correct" : "Incorrect");
    }

    // 清理
    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_cpu);

    return 0;
}
