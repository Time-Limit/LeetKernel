#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <unordered_map>
#include <vector>

namespace  LLMMM {

class LLMMM {
  struct NK {
    int n;
    int k;
    NK(int n, int k) : n(n), k(k) {}
  };

  struct NKHash {
    std::size_t operator()(const NK &nk) {
      return std::size_t(nk.n) << 32 | nk.k;
    }
  };

  using MM = void (*)(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t stream);

  std::unordered_map<NK, MM, NKHash> mm_map;

  struct MMConfig {
    int32_t BLOCK_TILE_M;
    int32_t BLOCK_TILE_N;
    int32_t THREAD_TILE_M;
    int32_t THREAD_TILE_N;
    int32_t TILE_K;
    bool IS_ALIGNED_M;

    std::string info() const {
      std::stringstream ss;
      ss << "BLOCK_TILE_M=" << std::setw(3) << BLOCK_TILE_M;
      ss << ", BLOCK_TILE_N=" << std::setw(3) << BLOCK_TILE_N;
      ss << ", TILE_K=" << std::setw(3) << TILE_K;
      ss << ", THREAD_TILE_M=" << std::setw(3) << THREAD_TILE_M;
      ss << ", THREAD_TILE_N=" << std::setw(3) << THREAD_TILE_N;
      ss << ", IS_ALIGNED_M=" << std::setw(3) << IS_ALIGNED_M;
      return ss.str();
    }

    bool is_suitable(int M, int N, int K) const
    {
      if (IS_ALIGNED_M) {
        if (M % BLOCK_TILE_M != 0) {
          return false;
        }
      }
      if (N % BLOCK_TILE_N) {
        return false;
      }
      if (K % TILE_K || K < TILE_K * 4 || K / TILE_K % 2) {
        return false;
      }
      return true;
    }
  };

  std::vector<std::pair<MMConfig, MM>> aligned_M_mm_list;
  std::vector<std::pair<MMConfig, MM>> unaligned_M_mm_list;

  friend class MMInstantiatorWrapper;

public:
  static LLMMM &Instance() {
    static LLMMM instance;
    return instance;
  }

  void tune(const uint32_t N, const uint32_t K);

  // C = A * B, (M,N) = (M,K) * (K,N)
  void mm(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t stream);

  void verify(const float* A, const float* B, const float* benchmark, int M, int N, int K, const float EPS);
};

} // namespace LLMMM
