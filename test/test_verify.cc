#include "llmmm/llmmm.h"
#include "test/kahan.h"
#include "util/error.h"
#include "util/util.h"
#include <cuda_runtime.h>
#include <random>
#include <vector>

int main()
{
  static const int M = 4096, N = 4096, K = 4096;

  std::vector<float>                    host_A(M * K), host_B(K * N), host_C(M * N), host_result(M * N);
  std::random_device                    rd;
  std::mt19937                          gen(rd());
  std::uniform_real_distribution<float> dis(-5, 5);
  for (auto& vec : {&host_A, &host_B}) {
#if 1
    for (auto& data : *vec) {
      data = dis(gen);
    }
#else
    if (vec == &host_A) {
      for (size_t i = 0; i < vec->size(); ++i) {
        int row = i / K;
        int col = i % K;
        if (row < 128 && col < 128) {
          vec->operator[](i) = row * 128 + col;
          vec->operator[](i) = (row == col);
        }
        else {
          vec->operator[](i) = 0;
        }
      }
    }
    if (vec == &host_B) {
      for (size_t i = 0; i < vec->size(); ++i) {
        int row = i / N;
        int col = i % N;
        if (row < 128 && col < 128) {
          vec->operator[](i) = row * 128 + col;
        }
        else {
          vec->operator[](i) = 0;
        }
      }
    }
#endif
  }

  float *A, *B, *C;
  for (auto& pair : {std::make_pair(host_A, &A), std::make_pair(host_B, &B), std::make_pair(host_C, &C)}) {
    const std::vector<float>& host   = pair.first;
    float*&                   device = *pair.second;
    cudaMalloc(&device, sizeof(float) * host.size());
    cudaMemcpy(device, host.data(), sizeof(float) * host.size(), cudaMemcpyDefault);
    CHECK_CUDA_ERROR();
  }

  for (int m = 1; m <= 256; ++m) {
    for (int n = 128; n <= 1024; n += 128) {
      for (int k = 128; k <= 1024; k += 128) {
        {
          cudaMemset(C, 0, m * n * sizeof(float));
          launch_kahan(A, B, C, m, n, k);
          cudaMemcpy(host_result.data(), C, sizeof(float) * host_C.size(), cudaMemcpyDefault);
          CHECK_CUDA_ERROR();
        }

        LLMMM::LLMMM::Instance().verify(A, B, host_result.data(), m, n, k, 1e-1);
      }
    }
  }

  return 0;
}
