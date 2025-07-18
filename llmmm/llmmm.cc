#include "llmmm/llmmm.h"
#include "util/error.h"
#include <cmath>
#include <limits>
#include <stdexcept>

namespace LLMMM {

void LLMMM::tune(const uint32_t N, const uint32_t K)
{
  float *A, *B, *C;
  CHECK_CUDA_RETURN(cudaMalloc(&A, sizeof(float) * 1024 * K));
  CHECK_CUDA_RETURN(cudaMalloc(&B, sizeof(float) * K * N));
  CHECK_CUDA_RETURN(cudaMalloc(&C, sizeof(float) * 1024 * N));
  for (uint32_t M = 1; M <= 1024; ++M) {
    MMConfig opt_config;
    double   opt_microseconds = std::numeric_limits<float>::max();
    for (const auto& list : {unaligned_M_mm_list, aligned_M_mm_list}) {
      for (const auto& [config, mm] : list) {
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
            mm(A, B, C, M, N, K, stream);
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
    }
    printf("M=%3d, N=%3d, K=%3d, cost=%8.3lf, config=%s\n", M, N, K, opt_microseconds, opt_config.info().c_str());
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
  float* C;
  CHECK_CUDA_RETURN(cudaMalloc(&C, sizeof(float) * M * N));
  std::vector<float> host_C(M * N);
  int                total = 0, correct = 0, error = 0;
  for (auto list : {aligned_M_mm_list, unaligned_M_mm_list}) {
    for (const auto& [config, mm] : list) {
      total++;
      if (!config.is_suitable(M, N, K)) {
        continue;
      }
      CHECK_CUDA_RETURN(cudaMemset(C, 0, sizeof(float) * M * N));
      mm(A, B, C, M, N, K, nullptr);
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
  }
  std::cout << "M=" << std::setw(4) << M << ", N=" << std::setw(4) << N << ", K=" << std::setw(4) << K
            << ", total=" << std::setw(3) << total << ", skip=" << std::setw(3) << (total - error - correct)
            << ", correct=" << correct << ", error=" << std::setw(3) << error << std::endl;
  cudaFree(C);
}

}  // namespace LLMMM
