#pragma once

#include "util/error.h"
#include <cstdint>
#include <cuda_runtime.h>
#include <functional>
#include <iomanip>
#include <iostream>
#include <unordered_map>
#include <vector>

namespace LLMMM {

// FIXME None of the member functions of this class are thread-safe.
class LLMMM {
  struct NK {
    int n;
    int k;
    NK(int n, int k): n(n), k(k) {}
  };

  struct NKHash {
    std::size_t operator()(const NK& nk)
    {
      return std::size_t(nk.n) << 32 | nk.k;
    }
  };

  using MM = void (*)(const float* A,
                      const float* B,
                      float*       C,
                      int          M,
                      int          N,
                      int          K,
                      void*        workspace,
                      size_t       workspace_bytes,
                      cudaStream_t stream);

  std::unordered_map<NK, MM, NKHash> mm_map;

  struct MMConfig {
    int32_t BLOCK_TILE_M;
    int32_t BLOCK_TILE_N;
    int32_t THREAD_TILE_M;
    int32_t THREAD_TILE_N;
    int32_t TILE_K;
    bool    IS_ALIGNED_M;
    int32_t SPLIT_K_TILES;  // SPLIT_K_TILES indicates how many tiles are divided in the K direction.
    int32_t REDUCE_BLOCK_TILE;
    int32_t REDUCE_THREAD_TILE;
    int32_t IS_ALIGNED_REDUCE_BLOCK_TILE;

    std::string info() const
    {
      std::stringstream ss;
      ss << "BLOCK_TILE_M=" << std::setw(3) << BLOCK_TILE_M;
      ss << ", BLOCK_TILE_N=" << std::setw(3) << BLOCK_TILE_N;
      ss << ", TILE_K=" << std::setw(3) << TILE_K;
      ss << ", THREAD_TILE_M=" << std::setw(3) << THREAD_TILE_M;
      ss << ", THREAD_TILE_N=" << std::setw(3) << THREAD_TILE_N;
      ss << ", IS_ALIGNED_M=" << std::setw(3) << IS_ALIGNED_M;
      ss << ", SPLIT_K_TILES=" << std::setw(3) << SPLIT_K_TILES;
      ss << ", REDUCE_BLOCK_TILE=" << std::setw(3) << REDUCE_BLOCK_TILE;
      ss << ", REDUCE_THREAD_TILE=" << std::setw(3) << REDUCE_THREAD_TILE;
      ss << ", IS_ALIGNED_REDUCE_BLOCK_TILE=" << std::setw(3) << IS_ALIGNED_REDUCE_BLOCK_TILE;
      return ss.str();
    }

    bool is_suitable(int M, int N, int K) const
    {
      if (SPLIT_K_TILES && K % SPLIT_K_TILES) {
        return false;
      }
      if (M > 256 && BLOCK_TILE_M <= 16) {
        return false;
      }
      if (M > 640 && BLOCK_TILE_M <= 32) {
        return false;
      }
      if (IS_ALIGNED_M) {
        if (M % BLOCK_TILE_M != 0) {
          return false;
        }
      }
      if (N % BLOCK_TILE_N) {
        return false;
      }
      K /= SPLIT_K_TILES;
      if (K % TILE_K || K < TILE_K * 4 || K / TILE_K % 2) {
        return false;
      }
      if (IS_ALIGNED_REDUCE_BLOCK_TILE && M * N % REDUCE_BLOCK_TILE != 0) {
        return false;
      }
      if (SPLIT_K_TILES != 1 && M > MAX_M_SUPPOR_SPLIT_K) {
        return false;
      }
      if (SPLIT_K_TILES > MAX_SPLIT_K_TILES) {
        return false;
      }
      return true;
    }
  };

  std::vector<std::pair<MMConfig, MM>> mm_list;

  struct SplitKWorkspaceDeleter {
    void operator()(float* split_k_workspace)
    {
      CHECK_CUDA_RETURN(cudaFree(split_k_workspace));
    }
  };

  std::unique_ptr<float, SplitKWorkspaceDeleter> split_k_workspace;
  size_t                 split_k_workspace_bytes = 0;
  static constexpr int   MAX_M_SUPPOR_SPLIT_K    = 64;
  static constexpr int   MIN_SPLIT_K_TILES       = 2;
  static constexpr int   MAX_SPLIT_K_TILES       = 32;

  void realloc_split_k_workspace(const int N);

  friend class MMInstantiatorWrapper;

public:
  static LLMMM& Instance()
  {
    static LLMMM instance;
    return instance;
  }

  void tune(const int N, const int K);

  // C = A * B, (M,N) = (M,K) * (K,N)
  void mm(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t stream);

  void verify(const float* A, const float* B, const float* benchmark, int M, int N, int K, const float EPS);
};

}  // namespace LLMMM
