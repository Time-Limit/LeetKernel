#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "util/error.h"
#include "util/util.cuh"

// The base version using a double type accumulator.
template <typename SUM_T>
__global__ void base(const float *A, const float *B, float *C, int M, int N,
                     int K);
void launch_base(const float *A, const float *B, float *C, int M, int N, int K);

// The version that uses a float type accumulator and applies Kahan's algorithm
// to reduce precision loss.
__global__ void kahan(const float *A, const float *B, float *C, int M, int N,
                      int K);
void launch_kahan(const float *A, const float *B, float *C, int M, int N,
                  int K);

// The memory bandwidth of RTX 4090 is approximately 1TB/s, while its
// single-precision floating-point performance is around 83 TFLOP/s.
template <int block_tile, int thread_tile, int k_per_iter>
__global__ void shared_memory(const float *A, const float *B, float *C, int M,
                              int N, int K);

template <int BLOCK_TILE>
void launch_shared_memory(const float *A, const float *B, float *C, int M,
                          int N, int K);

template <int BLOCK_TILE>
void launch_shared_memory__eliminate_bank_conflict(const float *A,
                                                   const float *B, float *C,
                                                   int M, int N, int K);

int main() {
  static const int M = (1 << 12), N = (1 << 12), K = (1 << 12);
  // static const int M = 128, N = 128, K = 128;
  const float EPS = 1e-2;

  std::vector<float> host_A(M * K), host_B(K * N), host_C(M * N),
      host_result(M * N);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(-5.0f, 5.0f);
  for (auto &vec : {&host_A, &host_B, &host_C}) {
    for (auto &data : *vec) {
      data = dis(gen);
    }
  }

  float *A, *B, *C;
  for (auto &pair : {std::make_pair(host_A, &A), std::make_pair(host_B, &B),
                     std::make_pair(host_C, &C)}) {
    const std::vector<float> &host = pair.first;
    float *&device = *pair.second;
    cudaMalloc(&device, sizeof(float) * host.size());
    cudaMemcpy(device, host.data(), sizeof(float) * host.size(),
               cudaMemcpyDefault);
    CHECK_CUDA_ERROR();
  }

  {
    cudaMemset(C, 0, M * N * sizeof(float));
    launch_base(A, B, C, M, N, K);
    cudaMemcpy(host_result.data(), C, sizeof(float) * host_C.size(),
               cudaMemcpyDefault);
  }

  {
    cudaMemset(C, 0, M * N * sizeof(float));
    launch_kahan(A, B, C, M, N, K);
    memset(host_C.data(), 0, sizeof(float) * host_C.size());
    cudaMemcpy(host_C.data(), C, sizeof(float) * host_C.size(),
               cudaMemcpyDefault);
    const float(*host_result_ptr)[N] =
        reinterpret_cast<const float(*)[N]>(host_result.data());
    const float(*device_result_ptr)[N] =
        reinterpret_cast<const float(*)[N]>(host_C.data());

    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        if (fabs(host_result_ptr[i][j] - device_result_ptr[i][j]) > EPS) {
          printf("%.7f, %.7f\n", host_result_ptr[i][j],
                 device_result_ptr[i][j]);
          std::stringstream ss;
          ss << "kahan, invalid result, m=" << i << ", n=" << j << ", expected "
             << host_result_ptr[i][j] << ", got " << device_result_ptr[i][j];
          throw std::runtime_error(ss.str());
        }
      }
    }
  }

  {
    cudaMemset(C, 0, M * N * sizeof(float));
    launch_shared_memory<128>(A, B, C, M, N, K);
    memset(host_C.data(), 0, sizeof(float) * host_C.size());
    cudaMemcpy(host_C.data(), C, sizeof(float) * host_C.size(),
               cudaMemcpyDefault);
    const float(*host_result_ptr)[N] =
        reinterpret_cast<const float(*)[N]>(host_result.data());
    const float(*device_result_ptr)[N] =
        reinterpret_cast<const float(*)[N]>(host_C.data());

    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        if (fabs(host_result_ptr[i][j] - device_result_ptr[i][j]) > EPS) {
          printf("%.7f, %.7f\n", host_result_ptr[i][j],
                 device_result_ptr[i][j]);
          std::stringstream ss;
          ss << "shared_memory, 128x128, invalid result, m=" << i << ", n=" << j
             << ", expected " << host_result_ptr[i][j] << ", got "
             << device_result_ptr[i][j];
          throw std::runtime_error(ss.str());
        }
      }
    }
  }
  
  {
    cudaMemset(C, 0, M * N * sizeof(float));
    launch_shared_memory__eliminate_bank_conflict<128>(A, B, C, M, N, K);
    memset(host_C.data(), 0, sizeof(float) * host_C.size());
    cudaMemcpy(host_C.data(), C, sizeof(float) * host_C.size(),
               cudaMemcpyDefault);
    const float(*host_result_ptr)[N] =
        reinterpret_cast<const float(*)[N]>(host_result.data());
    const float(*device_result_ptr)[N] =
        reinterpret_cast<const float(*)[N]>(host_C.data());

    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        if (fabs(host_result_ptr[i][j] - device_result_ptr[i][j]) > EPS) {
          printf("%.7f, %.7f\n", host_result_ptr[i][j],
                 device_result_ptr[i][j]);
          std::stringstream ss;
          ss << "shared_memory, eliminate_bank_confict, 128x128, invalid "
                "result, m="
             << i << ", n=" << j << ", expected " << host_result_ptr[i][j]
             << ", got " << device_result_ptr[i][j];
          throw std::runtime_error(ss.str());
        }
      }
    }
  }

  return 0;
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
  base<float><<<blocks_per_grid, threads_per_block>>>(A, B, C, M, N, K);
  base<double><<<blocks_per_grid, threads_per_block>>>(A, B, C, M, N, K);
  CHECK_CUDA_ERROR();
  cudaDeviceSynchronize();
}

__global__ void kahan(const float *A, const float *B, float *C, int M, int N,
                      int K) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  int m = blockIdx.y * blockDim.y + threadIdx.y;

  if (n >= N || m >= M) {
    return;
  }

  A += m * K;
  B += n;
  float sum = 0.0, diff = 0.0;
  for (int k = 0; k < K; ++k) {
    float y = A[k] * B[k * N] - diff;
    float t = sum + y;
    diff = (t - sum) - y;
    sum = t;
  }
  C[m * N + n] = sum;
}

void launch_kahan(const float *A, const float *B, float *C, int M, int N,
                  int K) {
  dim3 threads_per_block(16, 16);
  dim3 blocks_per_grid((K + threads_per_block.x - 1) / threads_per_block.x,
                       (M + threads_per_block.y - 1) / threads_per_block.y);

  kahan<<<blocks_per_grid, threads_per_block>>>(A, B, C, M, N, K);
  CHECK_CUDA_ERROR();
  cudaDeviceSynchronize();
}

// A, B and C are all row-marjor matrices;
template <int block_tile, int thread_tile, int k_per_iter>
__global__ void shared_memory(const float *A, const float *B, float *C, int M,
                              int N, int K) {
  extern __shared__ float shared_memory_buffer[];

  // `A_sm` and `B_sm` are both k-marjor.
  float *A_sm = shared_memory_buffer;
  float *B_sm = A_sm + block_tile * k_per_iter;

  const int k_iter_count = (K + k_per_iter - 1) / k_per_iter;
  const int thread_count = blockDim.x * blockDim.y;

  // The current block needs to compute `block_tile * block_tile` floats in
  // `C`, spanning from `C_m_top` to `C_m_bottom`(exclusive) and from
  // `C_n_left` to `C_n_right`(exclusive).
  int C_m_top = blockIdx.y * block_tile;
  int C_n_left = blockIdx.x * block_tile;
  int C_m_bottom = min(C_m_top + block_tile, M);
  int C_n_right = min(C_n_left + block_tile, N);

  // Set A and B to the address of the first float that needs to be read in
  // this block.
  // A += C_m_top * K;
  // B += C_n_left;

  // `C_reg` is being used to store the partial sum.
  float C_reg[thread_tile][thread_tile] = {0};
  float C_diff_reg[thread_tile][thread_tile] = {0};

  // The following code computes the relative position of the top-right float
  // in the `C` submatrix.
  const int C_sub_m = threadIdx.y * thread_tile;
  const int C_sub_n = threadIdx.x * thread_tile;

  for (int iter = 0; iter < k_iter_count; ++iter) {
    // Read `block_tile * k_per_iter` floats from each of A and B in global
    // memory into shared memory per iteration.
    for (int sm_index = threadIdx.y * blockDim.x + threadIdx.x;
         sm_index < block_tile * k_per_iter; sm_index += thread_count) {
      // Use `sm_index` to get the relative position in the current iteration's
      // submatrix and then add the global offset.
      const int A_m = sm_index % block_tile + C_m_top;
      const int A_k = sm_index / block_tile + iter * k_per_iter;
      if (A_m < M and A_k < K) {
        // (m0,k0), (m1,k0), (m2,k0), ...., (m0, k1), (m1, k1), (m2, k1), ....
        A_sm[sm_index] = A[A_m * K + A_k];
      }

      // The code below implements the similiar logic as described previously.
      const int B_k = sm_index / block_tile + iter * k_per_iter;
      const int B_n = sm_index % block_tile + C_n_left;
      if (B_k < K && B_n < N) {
        // (k0,n1), (k0,n2), (k0,n3), ...., (k1,n1), (k1,n2), (k1,n3), ....
        B_sm[sm_index] = B[B_k * N + B_n];
      }
    }
    __syncthreads();

    // Every thread needs to compute `thread_tile * thread_tile` floats in the
    // `C` matrix.
    for (int k = 0; k < k_per_iter; ++k) {
      // Read `thread_tile` floats from each of `A_sm` and `B_sm`.
      float A_reg[thread_tile], B_reg[thread_tile];
      for (int i = 0; i < thread_tile; ++i) {
        A_reg[i] = A_sm[k * block_tile + C_sub_m + i];
        B_reg[i] = B_sm[k * block_tile + C_sub_n + i];
      }
      for (int i = 0; i < thread_tile; ++i) {
        for (int j = 0; j < thread_tile; ++j) {
          float y = A_reg[i] * B_reg[j];
          float t = C_reg[i][j] + y;
          C_diff_reg[i][j] = (t - C_reg[i][j]) - y;
          C_reg[i][j] = t;
        }
      }
    }
  }

  for (int i = 0; i < thread_tile; ++i) {
    const int m = C_m_top + C_sub_m + i;
    if (m < M) {
      for (int j = 0; j < thread_tile; ++j) {
        const int n = C_n_left + C_sub_n + j;
        if (n < N) {
          C[m * N + n] = C_reg[i][j];
        }
      }
    }
  }
}

// A, B and C are all row-marjor matrices;
template <int block_tile, int thread_tile, int k_per_iter>
__global__ void shared_memory__eliminate_bank_conflict(const float *A,
                                                       const float *B, float *C,
                                                       int M, int N, int K) {
  extern __shared__ float shared_memory_buffer[];

  // `A_sm` and `B_sm` are both k-marjor.
  float *A_sm = shared_memory_buffer;
  float *B_sm = A_sm + block_tile * k_per_iter;

  const int k_iter_count = (K + k_per_iter - 1) / k_per_iter;
  const int thread_count = blockDim.x * blockDim.y;

  // The current block needs to compute `block_tile * block_tile` floats in
  // `C`, spanning from `C_m_top` to `C_m_bottom`(exclusive) and from
  // `C_n_left` to `C_n_right`(exclusive).
  int C_m_top = blockIdx.y * block_tile;
  int C_n_left = blockIdx.x * block_tile;
  int C_m_bottom = min(C_m_top + block_tile, M);
  int C_n_right = min(C_n_left + block_tile, N);

  // Set A and B to the address of the first float that needs to be read in
  // this block.
  // A += C_m_top * K;
  // B += C_n_left;

  // `C_reg` is being used to store the partial sum.
  float C_reg[thread_tile][thread_tile] = {0};
  float C_diff_reg[thread_tile][thread_tile] = {0};

  // The following code computes the relative position of the top-right float
  // in the `C` submatrix.
  const int C_sub_m = threadIdx.y * thread_tile;
  const int C_sub_n = threadIdx.x * thread_tile;

  for (int iter = 0; iter < k_iter_count; ++iter) {
    // Read `block_tile * k_per_iter` floats from each of A and B in global
    // memory into shared memory per iteration.
    for (int sm_index = threadIdx.y * blockDim.x + threadIdx.x;
         sm_index < block_tile * k_per_iter; sm_index += thread_count) {
      // Use `sm_index` to get the relative position in the current iteration's
      // submatrix and then add the global offset.
      const int A_m = sm_index % block_tile + C_m_top;
      const int A_k = sm_index / block_tile + iter * k_per_iter;
      if (A_m < M and A_k < K) {
        // (m0,k0), (m1,k0), (m2,k0), ...., (m0, k1), (m1, k1), (m2, k1), ....
        A_sm[sm_index] = A[A_m * K + A_k];
      }

      // The code below implements the similiar logic as described previously.
      const int B_k = sm_index / block_tile + iter * k_per_iter;
      const int B_n = sm_index % block_tile + C_n_left;
      if (B_k < K && B_n < N) {
        // (k0,n1), (k0,n2), (k0,n3), ...., (k1,n1), (k1,n2), (k1,n3), ....
        B_sm[sm_index] = B[B_k * N + B_n];
      }
    }
    __syncthreads();

    // Every thread needs to compute `thread_tile * thread_tile` floats in the
    // `C` matrix.
    int swizzle = (threadIdx.y * blockDim.x + threadIdx.x) & 15 >> 2;
    for (int k = 0; k < k_per_iter; ++k) {
      // Read `thread_tile` floats from each of `A_sm` and `B_sm`.
      float A_reg[thread_tile], B_reg[thread_tile];
      for (int i = 0; i < thread_tile; ++i) {
        // No bank conflict
        A_reg[i] = A_sm[k * block_tile + C_sub_m + i];
        // Threads 0,4,8,12 read floats from the same bank.
        // B_reg[i] = B_sm[k * block_tile + C_sub_n + i];
        int position = (i + swizzle) & (thread_tile - 1);
        B_reg[position] = B_sm[k * block_tile + C_sub_n + position];
      }
      for (int i = 0; i < thread_tile; ++i) {
        for (int j = 0; j < thread_tile; ++j) {
          float y = A_reg[i] * B_reg[j];
          float t = C_reg[i][j] + y;
          C_diff_reg[i][j] = (t - C_reg[i][j]) - y;
          C_reg[i][j] = t;
        }
      }
    }
  }

  for (int i = 0; i < thread_tile; ++i) {
    const int m = C_m_top + C_sub_m + i;
    if (m < M) {
      for (int j = 0; j < thread_tile; ++j) {
        const int n = C_n_left + C_sub_n + j;
        if (n < N) {
          C[m * N + n] = C_reg[i][j];
        }
      }
    }
  }
}

template <int BLOCK_TILE>
void launch_shared_memory(const float *A, const float *B, float *C, int M,
                          int N, int K) {
  constexpr int THREAD_TILE = 8;
  static_assert(BLOCK_TILE % THREAD_TILE == 0);
  constexpr dim3 block(BLOCK_TILE / THREAD_TILE, BLOCK_TILE / THREAD_TILE);
  constexpr int k_per_iter = 8;
  const dim3 grid_dim((N + BLOCK_TILE - 1) / BLOCK_TILE,
                      (M + BLOCK_TILE - 1) / BLOCK_TILE);
  constexpr dim3 block_dim(BLOCK_TILE / THREAD_TILE, BLOCK_TILE / THREAD_TILE);

  const int32_t shared_memory_bytes =
      k_per_iter * BLOCK_TILE * sizeof(float) * 2;

  static_assert(BLOCK_TILE * k_per_iter % (block_dim.x * block_dim.y) == 0);
  shared_memory<BLOCK_TILE, THREAD_TILE, k_per_iter>
      <<<grid_dim, block_dim, shared_memory_bytes>>>(A, B, C, M, N, K);

  cudaDeviceSynchronize();
  CHECK_CUDA_ERROR();
}

template <int BLOCK_TILE>
void launch_shared_memory__eliminate_bank_conflict(const float *A, const float *B, float *C, int M,
                          int N, int K) {
  constexpr int THREAD_TILE = 8;
  static_assert(BLOCK_TILE % THREAD_TILE == 0);
  constexpr dim3 block(BLOCK_TILE / THREAD_TILE, BLOCK_TILE / THREAD_TILE);
  constexpr int k_per_iter = 8;
  const dim3 grid_dim((N + BLOCK_TILE - 1) / BLOCK_TILE,
                      (M + BLOCK_TILE - 1) / BLOCK_TILE);
  constexpr dim3 block_dim(BLOCK_TILE / THREAD_TILE, BLOCK_TILE / THREAD_TILE);

  const int32_t shared_memory_bytes =
      k_per_iter * BLOCK_TILE * sizeof(float) * 2;

  static_assert(BLOCK_TILE * k_per_iter % (block_dim.x * block_dim.y) == 0);
  shared_memory__eliminate_bank_conflict<BLOCK_TILE, THREAD_TILE, k_per_iter>
      <<<grid_dim, block_dim, shared_memory_bytes>>>(A, B, C, M, N, K);

  cudaDeviceSynchronize();
  CHECK_CUDA_ERROR();
}
