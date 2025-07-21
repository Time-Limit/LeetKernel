#include "llmmm/llmmm.h"
#include "util/error.h"
#include <cmath>
#include <limits>
#include <stdexcept>

namespace LLMMM {

void LLMMM::tune(const int N, const int K)
{
  realloc_split_k_workspace(N);
  float *A, *B, *C;
  constexpr int MAX_M = 8192;
  CHECK_CUDA_RETURN(cudaMalloc(&A, sizeof(float) * MAX_M * K));
  CHECK_CUDA_RETURN(cudaMalloc(&B, sizeof(float) * K * N));
  CHECK_CUDA_RETURN(cudaMalloc(&C, sizeof(float) * MAX_M * N));
  for (uint32_t M = 1; M <= MAX_M; M += (M < 128 ? 1 : 32)) {
    MMConfig opt_config;
    double   opt_microseconds = std::numeric_limits<float>::max();
    for (const auto& [config, mm] : mm_list) {
      if (config.is_suitable(M, N, K)) {
        constexpr auto loop_count = 32;
        float          milliseconds;
        cudaStream_t   stream;
        cudaEvent_t    start, stop;

        CHECK_CUDA_RETURN(cudaStreamCreate(&stream));
        CHECK_CUDA_RETURN(cudaEventCreate(&start));
        CHECK_CUDA_RETURN(cudaEventCreate(&stop));

        CHECK_CUDA_RETURN(cudaEventRecord(start, stream));
        for (int loop = 0; loop < loop_count; ++loop) {
          mm(A, B, C, M, N, K, split_k_workspace.get(), split_k_workspace_bytes, stream);
        }
        CHECK_CUDA_RETURN(cudaEventRecord(stop, stream));
        CHECK_CUDA_RETURN(cudaStreamSynchronize(stream));
        CHECK_CUDA_RETURN(cudaEventElapsedTime(&milliseconds, start, stop));
        CHECK_CUDA_RETURN(cudaEventDestroy(start));
        CHECK_CUDA_RETURN(cudaEventDestroy(stop));
        CHECK_CUDA_RETURN(cudaStreamDestroy(stream));
        if (milliseconds / loop_count < opt_microseconds) {
          opt_microseconds = milliseconds / loop_count;
          opt_config       = config;
        }
      }
    }
    printf("M=%3d, N=%3d, K=%3d, cost=%8.3lf, config=%s\n", M, N, K, opt_microseconds, opt_config.info().c_str());
    fflush(stdout);
  }
  CHECK_CUDA_RETURN(cudaFree(A));
  CHECK_CUDA_RETURN(cudaFree(B));
  CHECK_CUDA_RETURN(cudaFree(C));
}

void LLMMM::mm(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t stream)
{
  throw std::runtime_error("no implemention");
}

void LLMMM::verify(const float* A, const float* B, const float* benchmark, int M, int N, int K, const float EPS)
{
  realloc_split_k_workspace(N);
  float* C;
  CHECK_CUDA_RETURN(cudaMalloc(&C, sizeof(float) * M * N));
  std::vector<float> host_C(M * N);
  int                total = 0, correct = 0, error = 0;
  for (const auto& [config, mm] : mm_list) {
    total++;
    if (!config.is_suitable(M, N, K)) {
      continue;
    }
    CHECK_CUDA_RETURN(cudaMemset(C, 0, sizeof(float) * M * N));
    mm(A, B, C, M, N, K, split_k_workspace.get(), split_k_workspace_bytes, nullptr);
    CHECK_CUDA_ERROR_WITH_INFO(config.info().c_str());
    CHECK_CUDA_RETURN(cudaMemcpy(host_C.data(), C, sizeof(float) * M * N, cudaMemcpyDeviceToHost));
    double max_diff = -1, A = 0, B = 0;
    int    index = 0;
    for (int i = 0; i < M * N; ++i) {
      double tmp = fabs(host_C[i] - benchmark[i]);
      if (tmp > max_diff) {
        max_diff = tmp;
        A        = host_C[i];
        B        = benchmark[i];
        index    = i;
      }
    }
    if (max_diff > EPS) {
      error++;
      // std::cout << config.info() << ", M=" << M << ", N=" << N << ", K=" << K
      //           << ", the error in precision is excessive, max=" << max_diff << ", base=" << B << ", exp=" << A
      //           << ", index=(" << (index / N) << "," << index % N << ")" << ", EPS=" << EPS << std::endl;
    }
    else {
      correct++;
      // std::cout << config.info() << ", M=" << M << ", N=" << N << ", K=" << K << ", correct, max=" << max_diff
      //           << ", base=" << B << ", exp=" << A << ", index=(" << (index / N) << "," << index % N << ")"
      //           << std::endl;
    }
  }
  std::cout << "M=" << std::setw(4) << M << ", N=" << std::setw(4) << N << ", K=" << std::setw(4) << K
            << ", total=" << std::setw(3) << total << ", skip=" << std::setw(3) << (total - error - correct)
            << ", correct=" << correct << ", error=" << std::setw(3) << error << std::endl;
  cudaFree(C);
}

void LLMMM::realloc_split_k_workspace(const int N)
{
  size_t new_split_k_workspace_bytes = MAX_M_SUPPOR_SPLIT_K * N * MAX_SPLIT_K_TILES * sizeof(float4);
  if (new_split_k_workspace_bytes > split_k_workspace_bytes) {
    float* tmp = nullptr;
    CHECK_CUDA_RETURN(cudaMalloc(&tmp, new_split_k_workspace_bytes));
    split_k_workspace.reset(tmp);
    split_k_workspace_bytes = new_split_k_workspace_bytes;
  }
}

}  // namespace LLMMM
