#include "llmmm/llmmm.h"
#include "util/error.h"
#include <cmath>
#include <limits>
#include <stdexcept>

namespace LLMMM {

void LLMMM::tune(const uint32_t N, const uint32_t K)
{
  float *A, *B, *C;
  CHECK_RETURN(cudaMalloc(&A, sizeof(float) * 128 * K));
  CHECK_RETURN(cudaMalloc(&B, sizeof(float) * K * N));
  CHECK_RETURN(cudaMalloc(&C, sizeof(float) * 128 * N));
  for (uint32_t M = 1; M <= 128; ++M) {
    MMConfig opt_config;
    double   opt_microseconds = std::numeric_limits<float>::max();
    for (const auto& list : {unaligned_M_mm_list, aligned_M_mm_list}) {
      for (const auto& [config, mm] : list) {
        if (config.is_suitable(M, N, K)) {
          constexpr auto loop_count = 16;
          double         sum        = 0;
          for (int loop = 0; loop < loop_count; ++loop) {
            cudaStream_t stream;
            cudaEvent_t  start, stop;

            CHECK_RETURN(cudaStreamCreate(&stream));
            CHECK_RETURN(cudaEventCreate(&start));
            CHECK_RETURN(cudaEventCreate(&stop));

            CHECK_RETURN(cudaEventRecord(start, stream));
            mm(A, B, C, M, N, K, stream);
            CHECK_RETURN(cudaEventRecord(stop, stream));
            CHECK_RETURN(cudaStreamSynchronize(stream));
            float milliseconds;
            CHECK_RETURN(cudaEventElapsedTime(&milliseconds, start, stop));
            sum += milliseconds;
            CHECK_RETURN(cudaEventDestroy(start));
            CHECK_RETURN(cudaEventDestroy(stop));
            CHECK_RETURN(cudaStreamDestroy(stream));
          }
          if (sum / loop_count < opt_microseconds) {
            opt_microseconds = sum / loop_count;
            opt_config       = config;
          }
        }
      }
    }
    printf("M=%3d, N=%3d, K=%3d, cost=%8.3lf, config=%s", M, N, K, opt_microseconds, opt_config.info().c_str());
  }
  CHECK_RETURN(cudaFree(A));
  CHECK_RETURN(cudaFree(B));
  CHECK_RETURN(cudaFree(C));
}

void LLMMM::mm(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t stream)
{
  throw std::runtime_error("no implemention");
}

void LLMMM::verify(const float* A, const float* B, const float* benchmark, int M, int N, int K, const float EPS)
{
  float* C;
  CHECK_RETURN(cudaMalloc(&C, sizeof(float) * M * N));
  std::vector<float> host_C(M * N);
  for (auto list : {aligned_M_mm_list, unaligned_M_mm_list}) {
    for (const auto& [config, mm] : list) {
      if (!config.is_suitable(M, N, K)) {
        continue;
      }
      CHECK_RETURN(cudaMemset(C, 0, sizeof(float) * M * N));
      int real_M = (config.BLOCK_TILE_M == 128 ? M : config.BLOCK_TILE_M);
      mm(A, B, C, real_M, N, K, nullptr);
      CHECK_CUDA_ERROR_WITH_INFO(config.info().c_str());
      CHECK_RETURN(cudaMemcpy(host_C.data(), C, sizeof(float) * M * N, cudaMemcpyDeviceToHost));
      double max_diff = -1, A = 0, B = 0;
      int    index = 0;
      for (int i = 0; i < real_M * N; ++i) {
        double tmp = fabs(host_C[i] - benchmark[i]);
        if (tmp > max_diff) {
          max_diff = tmp;
          A        = host_C[i];
          B        = benchmark[i];
          index    = i;
        }
      }
      if (max_diff > EPS) {
        std::cout << config.info() << ", M=" << M << ", N=" << N << ", K=" << K
                  << ", the error in precision is excessive, max=" << max_diff << ", base=" << B << ", exp=" << A
                  << ", index=(" << (index / N) << "," << index % N << ")" << ", EPS=" << EPS << std::endl;
      }
      else {
        std::cout << config.info() << ", M=" << M << ", N=" << N << ", K=" << K << ", correct, max=" << max_diff
                  << ", base=" << B << ", exp=" << A << ", index=(" << (index / N) << "," << index % N << ")"
                  << std::endl;
      }
    }
  }
  cudaFree(C);
}

}  // namespace LLMMM
