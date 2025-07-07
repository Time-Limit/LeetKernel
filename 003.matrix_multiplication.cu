#include <cassert>
#include <cooperative_groups/memcpy_async.h>
#include <cstdio>
#include <cstdlib>
#include <cuda/pipeline>
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
  const float EPS = 1e-1;

  std::vector<float> host_A(M * K), host_B(K * N), host_C(M * N),
      host_result(M * N);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(-5, 5);
  for (auto &vec : {&host_A, &host_B}) {
#if 1
#if 1
    for (auto &data : *vec) {
      data = dis(gen);
    }
#else
    if (vec == &host_A) {
      for (size_t i = 0; i < vec->size(); ++i) {
        int row = i / K;
        int col = i % K;
        if (row < 4096 && col < 4096) {
          vec->operator[](i) = dis(gen);
        } else {
          vec->operator[](i) = 0;
        }
      }
      for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; ++j) {
          printf("%8.3f ", vec->at(i * K + j));
        }
        printf("\n");
      }
      printf("\n");
      printf("\n");
    }
    if (vec == &host_B) {
      for (size_t i = 0; i < vec->size(); ++i) {
        int row = i / N;
        int col = i % N;
        if (row < 8 && col < 9) {
          vec->operator[](i) = dis(gen);
        } else {
          vec->operator[](i) = 0;
        }
      }
      for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 16; ++j) {
          printf("%8.3f ", vec->at(i * N + j));
        }
        printf("\n");
      }
      printf("\n");
      printf("\n");
    }
#endif
#else
    if (vec == &host_A) {
      for (size_t i = 0; i < vec->size(); ++i) {
        int row = i / K;
        int col = i % K;
        if (row < 128 && col < 128) {
          vec->operator[](i) = row * 128 + col;
        } else {
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
        } else {
          vec->operator[](i) = 0;
        }
      }
    }
#endif
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

  // {
  //   cudaMemset(C, 0, M * N * sizeof(float));
  //   launch_kahan(A, B, C, M, N, K);
  //   memset(host_C.data(), 0, sizeof(float) * host_C.size());
  //   cudaMemcpy(host_C.data(), C, sizeof(float) * host_C.size(),
  //              cudaMemcpyDefault);
  //   const float(*host_result_ptr)[N] =
  //       reinterpret_cast<const float(*)[N]>(host_result.data());
  //   const float(*device_result_ptr)[N] =
  //       reinterpret_cast<const float(*)[N]>(host_C.data());

  //   for (int i = 0; i < M; ++i) {
  //     for (int j = 0; j < N; ++j) {
  //       if (fabs(host_result_ptr[i][j] - device_result_ptr[i][j]) > EPS) {
  //         printf("%.7f, %.7f\n", host_result_ptr[i][j],
  //                device_result_ptr[i][j]);
  //         std::stringstream ss;
  //         ss << "kahan, invalid result, m=" << i << ", n=" << j << ",
  //         expected "
  //            << host_result_ptr[i][j] << ", got " << device_result_ptr[i][j];
  //         throw std::runtime_error(ss.str());
  //       }
  //     }
  //   }
  // }

  // {
  //   cudaMemset(C, 0, M * N * sizeof(float));
  //   launch_shared_memory<128>(A, B, C, M, N, K);
  //   memset(host_C.data(), 0, sizeof(float) * host_C.size());
  //   cudaMemcpy(host_C.data(), C, sizeof(float) * host_C.size(),
  //              cudaMemcpyDefault);
  //   const float(*host_result_ptr)[N] =
  //       reinterpret_cast<const float(*)[N]>(host_result.data());
  //   const float(*device_result_ptr)[N] =
  //       reinterpret_cast<const float(*)[N]>(host_C.data());

  //   for (int i = 0; i < M; ++i) {
  //     for (int j = 0; j < N; ++j) {
  //       if (fabs(host_result_ptr[i][j] - device_result_ptr[i][j]) > EPS) {
  //         printf("%.7f, %.7f\n", host_result_ptr[i][j],
  //                device_result_ptr[i][j]);
  //         std::stringstream ss;
  //         ss << "shared_memory, 128x128, invalid result, m=" << i << ", n="
  //         << j
  //            << ", expected " << host_result_ptr[i][j] << ", got "
  //            << device_result_ptr[i][j];
  //         throw std::runtime_error(ss.str());
  //       }
  //     }
  //   }
  // }

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

    // printf("\nnaive\n");
    // for (int i = 0; i < 128; ++i) {
    //   for (int j = 0; j < 128; ++j) {
    //     printf("%8.2f ", host_result_ptr[i][j]);
    //   }
    //   printf("\n");
    // }
    // printf("shared_memory\n");
    // for (int i = 0; i < 128; ++i) {
    //   for (int j = 0; j < 128; ++j) {
    //     printf("%8.2f ", device_result_ptr[i][j]);
    //   }
    //   printf("\n");
    // }

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
  // base<float><<<blocks_per_grid, threads_per_block>>>(A, B, C, M, N, K);
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
          C_reg[i][j] += A_reg[i] * B_reg[j];
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
__global__ void
shared_memory__eliminate_bank_conflict_v0(const float *A, const float *B,
                                          float *C, int M, int N, int K) {
  extern __shared__ float shared_memory_buffer[];

  float *A_sm = shared_memory_buffer;
  float *B_sm = A_sm + block_tile * k_per_iter;

  const int k_iter_count = (K + k_per_iter - 1) / k_per_iter;

  // The current block needs to compute `block_tile * block_tile` floats in
  // `C`, spanning from `C_m_top` to `C_m_bottom`(exclusive) and from
  // `C_n_left` to `C_n_right`(exclusive).
  int C_m_top = blockIdx.y * block_tile;
  int C_n_left = blockIdx.x * block_tile;
  int C_m_bottom = min(C_m_top + block_tile, M);
  int C_n_right = min(C_n_left + block_tile, N);

  // `C_reg` is being used to store the partial sum.
  float C_reg[thread_tile][thread_tile] = {0};

  // The following code computes the relative position of the top-right float
  // in the `C` submatrix.
  const int lane_id = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  /*
   * Warp-Layout
   * W0 W1
   * W2 W3
   * W4 W5
   * W6 W7
   * Thread-Layout. Each thread computes 8x8 floats in C.
   * T00 T01 T08 T09 T16 T17 T24 T25
   * T02 T03 T10 T11 T18 T19 T26 T27
   * T04 T05 T12 T13 T20 T21 T28 T29
   * T06 T07 T14 T15 T22 T23 T30 T31
   */
  const int C_sub_m = ((lane_id & 0x6) << 2) + ((warp_id & 0xfffffffe) << 4);
  const int C_sub_n = ((lane_id & 0xfffffff8) << 1) + ((lane_id & 1) << 3) +
                      ((warp_id & 1) << 6);

  // if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
  //   printf("threadIdx.x = %3d, C_sub_m = %3d, C_sub_n = %3d\n", threadIdx.x,
  //          C_sub_m, C_sub_n);
  // }

  for (int iter = 0; iter < k_iter_count; ++iter) {
    // Read `block_tile * k_per_iter` floats from each of A and B in global
    // memory into shared memory per iteration.
    for (int sm_index = threadIdx.x; sm_index < block_tile * k_per_iter;
         sm_index += blockDim.x) {
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
      const int A_offset = k * block_tile + C_sub_m;
      const int B_offset = k * block_tile + C_sub_n;
      *(float4 *)(&A_reg[0]) = *(const float4 *)(A_sm + A_offset);
      *(float4 *)(&B_reg[0]) = *(const float4 *)(B_sm + B_offset);
      *(float4 *)(&A_reg[4]) = *(const float4 *)(A_sm + A_offset + 4);
      *(float4 *)(&B_reg[4]) = *(const float4 *)(B_sm + B_offset + 4);
      for (int i = 0; i < thread_tile; ++i) {
        for (int j = 0; j < thread_tile; ++j) {
          C_reg[i][j] += A_reg[i] * B_reg[j];
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
__global__ void
shared_memory__eliminate_bank_conflict_v1(const float *A, const float *B,
                                          float *C, int M, int N, int K) {
  extern __shared__ float shared_memory_buffer[];

  float *A_sm = shared_memory_buffer;
  float *B_sm = A_sm + block_tile * k_per_iter;

  const int k_iter_count = (K + k_per_iter - 1) / k_per_iter;

  // The current block needs to compute `block_tile * block_tile` floats in
  // `C`, spanning from `C_m_top` to `C_m_bottom`(exclusive) and from
  // `C_n_left` to `C_n_right`(exclusive).
  int C_m_top = blockIdx.y * block_tile;
  int C_n_left = blockIdx.x * block_tile;
  int C_m_bottom = min(C_m_top + block_tile, M);
  int C_n_right = min(C_n_left + block_tile, N);

  // `C_reg` is being used to store the partial sum.
  float C_reg[thread_tile][thread_tile] = {0};

  // The following code computes the relative position of the top-right float
  // in the `C` submatrix.
  const int lane_id = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  /*
   * Warp-Layout
   * W0 W1
   * W2 W3
   * W4 W5
   * W6 W7
   * Thread-Layout. Each thread computes 8x8 floats in C.
   * T00 T01 T08 T09 T16 T17 T24 T25
   * T02 T03 T10 T11 T18 T19 T26 T27
   * T04 T05 T12 T13 T20 T21 T28 T29
   * T06 T07 T14 T15 T22 T23 T30 T31
   */
  const int C_sub_m = ((lane_id & 0x6) << 2) + ((warp_id & 0xfffffffe) << 4);
  const int C_sub_n = ((lane_id & 0xfffffff8) << 1) + ((lane_id & 1) << 3) +
                      ((warp_id & 1) << 6);

  // if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
  //   printf("threadIdx.x = %3d, C_sub_m = %3d, C_sub_n = %3d\n", threadIdx.x,
  //          C_sub_m, C_sub_n);
  // }

  for (int iter = 0; iter < k_iter_count; ++iter) {
    // Read `block_tile * k_per_iter` floats from each of A and B in global
    // memory into shared memory per iteration.
    for (int sm_index = threadIdx.x; sm_index < block_tile * k_per_iter;
         sm_index += blockDim.x) {
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
      const int A_offset = k * block_tile + C_sub_m;
      const int B_offset = k * block_tile + C_sub_n;
      *(float4 *)(&A_reg[0]) = *(const float4 *)(A_sm + A_offset);
      *(float4 *)(&B_reg[0]) = *(const float4 *)(B_sm + B_offset);
      *(float4 *)(&A_reg[4]) = *(const float4 *)(A_sm + A_offset + 4);
      *(float4 *)(&B_reg[4]) = *(const float4 *)(B_sm + B_offset + 4);
      for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
          C_reg[i][j] += A_reg[i] * B_reg[j];
        }
      }
      for (int i = 0; i < 4; ++i) {
        for (int j = 4; j < thread_tile; ++j) {
          C_reg[i][j] += A_reg[i] * B_reg[j];
        }
      }
      for (int i = 4; i < thread_tile; ++i) {
        for (int j = 0; j < thread_tile; ++j) {
          C_reg[i][j] += A_reg[i] * B_reg[j];
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
__global__ void
shared_memory__eliminate_bank_conflict_v2(const float *A, const float *B,
                                          float *C, int M, int N, int K) {
  extern __shared__ float shared_memory_buffer[];

  float *A_sm = shared_memory_buffer;
  float *B_sm = A_sm + block_tile * k_per_iter;

  const int k_iter_count = (K + k_per_iter - 1) / k_per_iter;

  // The current block needs to compute `block_tile * block_tile` floats in
  // `C`, spanning from `C_m_top` to `C_m_bottom`(exclusive) and from
  // `C_n_left` to `C_n_right`(exclusive).
  int C_m_top = blockIdx.y * block_tile;
  int C_n_left = blockIdx.x * block_tile;
  int C_m_bottom = min(C_m_top + block_tile, M);
  int C_n_right = min(C_n_left + block_tile, N);

  // `C_reg` is being used to store the partial sum.
  float C_reg[thread_tile][thread_tile] = {0};

  // The following code computes the relative position of the top-right float
  // in the `C` submatrix.
  const int lane_id = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  /*
   * Warp-Layout
   * W0 W1
   * W2 W3
   * W4 W5
   * W6 W7
   * Thread-Layout. Each thread computes 8x8 floats in C.
   * T00 T01 T08 T09 T16 T17 T24 T25
   * T02 T03 T10 T11 T18 T19 T26 T27
   * T04 T05 T12 T13 T20 T21 T28 T29
   * T06 T07 T14 T15 T22 T23 T30 T31
   */
  const int C_sub_m = ((lane_id & 0x6) << 2) + ((warp_id & 0xfffffffe) << 4);
  const int C_sub_n = ((lane_id & 0xfffffff8) << 1) + ((lane_id & 1) << 3) +
                      ((warp_id & 1) << 6);

  // if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
  //   printf("threadIdx.x = %3d, C_sub_m = %3d, C_sub_n = %3d\n", threadIdx.x,
  //          C_sub_m, C_sub_n);
  // }

  for (int iter = 0; iter < k_iter_count; ++iter) {
    // Read `block_tile * k_per_iter` floats from each of A and B in global
    // memory into shared memory per iteration.
    for (int sm_index = threadIdx.x; sm_index < block_tile * k_per_iter;
         sm_index += blockDim.x) {
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
      const int A_offset = k * block_tile + C_sub_m;
      const int B_offset = k * block_tile + C_sub_n;
      *(float4 *)(&A_reg[0]) = *(const float4 *)(A_sm + A_offset);
      *(float4 *)(&B_reg[0]) = *(const float4 *)(B_sm + B_offset);
      for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
          C_reg[i][j] += A_reg[i] * B_reg[j];
        }
      }
      *(float4 *)(&A_reg[4]) = *(const float4 *)(A_sm + A_offset + 4);
      *(float4 *)(&B_reg[4]) = *(const float4 *)(B_sm + B_offset + 4);
      for (int i = 0; i < 4; ++i) {
        for (int j = 4; j < thread_tile; ++j) {
          C_reg[i][j] += A_reg[i] * B_reg[j];
        }
      }
      for (int i = 4; i < thread_tile; ++i) {
        for (int j = 0; j < thread_tile; ++j) {
          C_reg[i][j] += A_reg[i] * B_reg[j];
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
__global__ void
shared_memory__eliminate_bank_conflict_v3(const float *A, const float *B,
                                          float *C, int M, int N, int K) {
  extern __shared__ float shared_memory_buffer[];

  float *A_sm = shared_memory_buffer;
  float *B_sm = A_sm + block_tile * k_per_iter;

  const int k_iter_count = (K + k_per_iter - 1) / k_per_iter;

  // The current block needs to compute `block_tile * block_tile` floats in
  // `C`, spanning from `C_m_top` to `C_m_bottom`(exclusive) and from
  // `C_n_left` to `C_n_right`(exclusive).
  int C_m_top = blockIdx.y * block_tile;
  int C_n_left = blockIdx.x * block_tile;
  int C_m_bottom = min(C_m_top + block_tile, M);
  int C_n_right = min(C_n_left + block_tile, N);

  // `C_reg` is being used to store the partial sum.
  float C_reg[thread_tile][thread_tile] = {0};

  // The following code computes the relative position of the top-right float
  // in the `C` submatrix.
  const int lane_id = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  /*
   * Warp-Layout
   * W0 W1
   * W2 W3
   * W4 W5
   * W6 W7
   * Thread-Layout. Each thread computes 8x8 floats in C.
   * T00 T01 T08 T09 T16 T17 T24 T25
   * T02 T03 T10 T11 T18 T19 T26 T27
   * T04 T05 T12 T13 T20 T21 T28 T29
   * T06 T07 T14 T15 T22 T23 T30 T31
   */
  const int C_sub_m = ((lane_id & 0x6) << 2) + ((warp_id & 0xfffffffe) << 4);
  const int C_sub_n = ((lane_id & 0xfffffff8) << 1) + ((lane_id & 1) << 3) +
                      ((warp_id & 1) << 6);

  // if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
  //   printf("threadIdx.x = %3d, C_sub_m = %3d, C_sub_n = %3d\n", threadIdx.x,
  //          C_sub_m, C_sub_n);
  // }

  for (int iter = 0; iter < k_iter_count; ++iter) {
    // Read `block_tile * k_per_iter` floats from each of A and B in global
    // memory into shared memory per iteration.
    for (int sm_index = threadIdx.x; sm_index < block_tile * k_per_iter;
         sm_index += blockDim.x) {
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
      const int A_offset = k * block_tile + C_sub_m;
      const int B_offset = k * block_tile + C_sub_n;
      *(float4 *)(&A_reg[0]) = *(const float4 *)(A_sm + A_offset);
      *(float4 *)(&B_reg[0]) = *(const float4 *)(B_sm + B_offset);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          C_reg[i][j] += A_reg[i] * B_reg[j];
        }
      }
      *(float4 *)(&A_reg[4]) = *(const float4 *)(A_sm + A_offset + 4);
      *(float4 *)(&B_reg[4]) = *(const float4 *)(B_sm + B_offset + 4);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int j = 4; j < thread_tile; ++j) {
          C_reg[i][j] += A_reg[i] * B_reg[j];
        }
      }
#pragma unroll
      for (int i = 4; i < thread_tile; ++i) {
#pragma unroll
        for (int j = 0; j < thread_tile; ++j) {
          C_reg[i][j] += A_reg[i] * B_reg[j];
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
__global__ void
shared_memory__eliminate_bank_conflict_v4(const float *A, const float *B,
                                          float *C, int M, int N, int K) {
  extern __shared__ float shared_memory_buffer[];

  float *A_sm = shared_memory_buffer;
  float *B_sm = A_sm + block_tile * k_per_iter;

  const int k_iter_count = (K + k_per_iter - 1) / k_per_iter;

  // The current block needs to compute `block_tile * block_tile` floats in
  // `C`, spanning from `C_m_top` to `C_m_bottom`(exclusive) and from
  // `C_n_left` to `C_n_right`(exclusive).
  int C_m_top = blockIdx.y * block_tile;
  int C_n_left = blockIdx.x * block_tile;
  int C_m_bottom = min(C_m_top + block_tile, M);
  int C_n_right = min(C_n_left + block_tile, N);

  // `C_reg` is being used to store the partial sum.
  float C_reg[thread_tile][thread_tile] = {0};

  // The following code computes the relative position of the top-right float
  // in the `C` submatrix.
  const int lane_id = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  /*
   * Warp-Layout
   * W0 W1
   * W2 W3
   * W4 W5
   * W6 W7
   * Thread-Layout. Each thread computes 8x8 floats in C.
   * T00 T01 T08 T09 T16 T17 T24 T25
   * T02 T03 T10 T11 T18 T19 T26 T27
   * T04 T05 T12 T13 T20 T21 T28 T29
   * T06 T07 T14 T15 T22 T23 T30 T31
   */
  const int C_sub_m = ((lane_id & 0x6) << 2) + ((warp_id & 0xfffffffe) << 4);
  const int C_sub_n = ((lane_id & 0xfffffff8) << 1) + ((lane_id & 1) << 3) +
                      ((warp_id & 1) << 6);

  // if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
  //   printf("threadIdx.x = %3d, C_sub_m = %3d, C_sub_n = %3d\n", threadIdx.x,
  //          C_sub_m, C_sub_n);
  // }

  for (int iter = 0; iter < k_iter_count; ++iter) {
    // Read `block_tile * k_per_iter` floats from each of A and B in global
    // memory into shared memory per iteration.
    for (int sm_index = threadIdx.x; sm_index < block_tile * k_per_iter;
         sm_index += blockDim.x) {
      // Use `sm_index` to get the relative position in the current iteration's
      // submatrix and then add the global offset.
      const int A_m = sm_index % block_tile + C_m_top;
      const int A_k = sm_index / block_tile + iter * k_per_iter;
      // (m0,k0), (m1,k0), (m2,k0), ...., (m0, k1), (m1, k1), (m2, k1), ....
      A_sm[sm_index] = A[A_m * K + A_k];

      // The code below implements the similiar logic as described previously.
      const int B_k = sm_index / block_tile + iter * k_per_iter;
      const int B_n = sm_index % block_tile + C_n_left;
      // (k0,n1), (k0,n2), (k0,n3), ...., (k1,n1), (k1,n2), (k1,n3), ....
      B_sm[sm_index] = B[B_k * N + B_n];
    }
    __syncthreads();

    // Every thread needs to compute `thread_tile * thread_tile` floats in the
    // `C` matrix.
    for (int k = 0; k < k_per_iter; ++k) {
      // Read `thread_tile` floats from each of `A_sm` and `B_sm`.
      float A_reg[thread_tile], B_reg[thread_tile];
      const int A_offset = k * block_tile + C_sub_m;
      const int B_offset = k * block_tile + C_sub_n;
      *(float4 *)(&A_reg[0]) = *(const float4 *)(A_sm + A_offset);
      *(float4 *)(&B_reg[0]) = *(const float4 *)(B_sm + B_offset);
      *(float4 *)(&A_reg[4]) = *(const float4 *)(A_sm + A_offset + 4);
      *(float4 *)(&B_reg[4]) = *(const float4 *)(B_sm + B_offset + 4);
      for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
          C_reg[i][j] += A_reg[i] * B_reg[j];
        }
      }
      for (int i = 0; i < 4; ++i) {
        for (int j = 4; j < thread_tile; ++j) {
          C_reg[i][j] += A_reg[i] * B_reg[j];
        }
      }
      for (int i = 4; i < thread_tile; ++i) {
        for (int j = 0; j < thread_tile; ++j) {
          C_reg[i][j] += A_reg[i] * B_reg[j];
        }
      }
    }
  }

  for (int i = 0; i < thread_tile; ++i) {
    const int m = C_m_top + C_sub_m + i;
    for (int j = 0; j < thread_tile; ++j) {
      const int n = C_n_left + C_sub_n + j;
      C[m * N + n] = C_reg[i][j];
    }
  }
}

// A, B and C are all row-marjor matrices;
template <int block_tile, int thread_tile, int k_per_iter>
__global__ void
shared_memory__eliminate_bank_conflict__global_store_memory_colacesing(
    const float *A, const float *B, float *C, int M, int N, int K) {
  extern __shared__ float shared_memory_buffer[];

  float *A_sm = shared_memory_buffer;
  float *B_sm = A_sm + block_tile * k_per_iter;

  const int k_iter_count = (K + k_per_iter - 1) / k_per_iter;

  // The current block needs to compute `block_tile * block_tile` floats in
  // `C`, spanning from `C_m_top` to `C_m_bottom`(exclusive) and from
  // `C_n_left` to `C_n_right`(exclusive).
  int C_m_top = blockIdx.y * block_tile;
  int C_n_left = blockIdx.x * block_tile;
  int C_m_bottom = min(C_m_top + block_tile, M);
  int C_n_right = min(C_n_left + block_tile, N);

  // `C_reg` is being used to store the partial sum.
  float C_reg[thread_tile][thread_tile] = {0};

  // The following code computes the relative position of the top-right float
  // in the `C` submatrix.
  const int lane_id = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;

  // Every thread needs to compute `thread_tile * thread_tile` floats in the
  // `C` matrix.
  // A_reg is K-marjor; B_reg is N-marjor;
  const int k_per_iter_half = k_per_iter / 2;
  float A_reg[thread_tile * k_per_iter_half];
  float B_reg[thread_tile * k_per_iter_half];
  const int A_offset = lane_id / 16 * 64 + lane_id % 16 / 2 * 4;

  // Each thread shifts `warp_id` position to the right within its own
  // half-warp.
  const int B_shifted_lane_id =
      ((lane_id + warp_id * 4) % 16 + (lane_id & 0x10)) ^ (warp_id / 4 * 16);
  const int B_offset = B_shifted_lane_id / 16 * 64 +
                       ((B_shifted_lane_id % 16) & 0xfd) / 4 * 8 +
                       (B_shifted_lane_id & 1) * 4;

  for (int iter = 0; iter < k_iter_count; ++iter) {
    if (iter != 0) {
      __syncthreads();
    }

    // Read `block_tile * k_per_iter` floats from each of A and B in global
    // memory into shared memory per iteration.
    {
      *(float4 *)(&A_sm[warp_id * block_tile + lane_id * 4]) = __ldg(
          (const float4
               *)(&A[(C_m_top + warp_id * 16 + lane_id % 8 + lane_id / 16 * 8) *
                         K +
                     iter * k_per_iter + lane_id % 16 / 8 * 4]));
    }
    {
      *(float4 *)(&B_sm[warp_id * block_tile + lane_id * 4]) = __ldg(
          (const float4 *)(&B[(iter * k_per_iter + warp_id) * N + C_n_left +
                              lane_id % 8 / 2 * 16 + (lane_id & 1) * 4 +
                              (lane_id & 0x8) + lane_id / 16 * 64]));
    }
    __syncthreads();

    // if (iter == 0 && warp_id == 0 && lane_id == 0 && blockIdx.x == 0 &&
    // blockIdx.y == 0) {
    //   printf("\nA_sm begin\n");
    //   for (int i = 0; i < 128 * 8; ++i) {
    //     if (i % 32 == 0) {
    //       printf("\n");
    //     }
    //     printf("%03d_%03d ", int(A_sm[i]) / 128, int(A_sm[i]) % 128);
    //   }
    //   printf("\nA_sm end\n");
    // }

    // if (iter == 0 && warp_id == 0 && lane_id == 0 && blockIdx.x == 0 &&
    // blockIdx.y == 0) {
    //   printf("\nB_sm begin\n");
    //   for (int i = 0; i < 128 * 8; ++i) {
    //     if (i % 32 == 0) {
    //       printf("\n");
    //     }
    //     printf("%03d_%03d ", int(B_sm[i]) / 128, int(B_sm[i]) % 128);
    //   }
    //   printf("\nB_sm end\n");
    // }

    for (int i = 0; i < thread_tile; ++i) {
      *(float4 *)&A_reg[i * k_per_iter_half] =
          *(float4 *)&A_sm[A_offset + i * 128];
    }
    for (int i = 0; i < k_per_iter_half; ++i) {
      *(float4 *)&B_reg[i * thread_tile] = *(float4 *)&B_sm[B_offset + i * 128];
      *(float4 *)&B_reg[i * thread_tile + 4] =
          *(float4 *)&B_sm[B_offset + i * 128 + 32];
    }

    for (int i = 0; i < thread_tile; ++i) {
      for (int j = 0; j < thread_tile; ++j) {
        for (int k = 0; k < k_per_iter_half; ++k) {
          C_reg[i][j] +=
              A_reg[i * k_per_iter_half + k] * B_reg[k * thread_tile + j];
        }
      }
    }

    for (int i = 0; i < thread_tile; ++i) {
      *(float4 *)&A_reg[i * k_per_iter_half] =
          *(float4 *)&A_sm[A_offset + i * 128 + 32];
    }
    for (int i = 4; i < k_per_iter; ++i) {
      *(float4 *)&B_reg[(i - 4) * thread_tile] =
          *(float4 *)&B_sm[B_offset + i * 128];
      *(float4 *)&B_reg[(i - 4) * thread_tile + 4] =
          *(float4 *)&B_sm[B_offset + i * 128 + 32];
    }

    for (int i = 0; i < thread_tile; ++i) {
      for (int j = 0; j < thread_tile; ++j) {
        for (int k = 0; k < k_per_iter_half; ++k) {
          C_reg[i][j] +=
              A_reg[i * k_per_iter_half + k] * B_reg[k * thread_tile + j];
        }
      }
    }
  }

  // C_reg is N-marjor
  if (C_m_top + block_tile <= M && C_n_left + block_tile <= N) {
    for (int i = 0; i < thread_tile; ++i) {
      int m = C_m_top + i * 16 + lane_id / 16 * 8 + lane_id % 16 / 2;
      int n = C_n_left + B_shifted_lane_id % 16 / 4 * 16 +
              B_shifted_lane_id / 16 * 64 + (B_shifted_lane_id & 1) * 4;
      *(float4 *)&C[m * N + n] = *(const float4 *)(&C_reg[i][0]);
      *(float4 *)&C[m * N + n + 8] = *(const float4 *)(&C_reg[i][4]);
      // if (blockIdx.x == 0 && blockIdx.y == 0) {
      //   for (int t = 0; t < 256; ++t) {
      //     if (t == threadIdx.x) {
      //       printf("i = %03d, warp_id = %03d, lane_id = %03d, shifted = %03d,
      //       "
      //              "m = %03d, n = "
      //              "%03d\n",
      //              i, warp_id, lane_id, B_shifted_lane_id, m, n);
      //     }
      //     __syncthreads();
      //   }
      // }
    }
  } else {
    // for (int i = 0; i < thread_tile; ++i) {
    //   const int m = C_m_top + C_sub_m + i;
    //   for (int j = 0; j < thread_tile; ++j) {
    //     const int n = C_n_left + C_sub_n + (j < 4 ? j : (32 + j - 4));
    //     if (m < M && n < N) {
    //       C[m * N + n] = C_reg[i][j];
    //     }
    //   }
    // }
  }
}

// A, B and C are all row-marjor matrices;
template <int block_tile, int thread_tile, int k_per_iter>
__global__ void
shared_memory__eliminate_bank_conflict__global_store_memory_colacesing__pipeline_1_stage(
    const float * __restrict__ A, const float * __restrict__ B, float * __restrict__ C, int M, int N, int K) {
  extern __shared__ float shared_memory_buffer[];

  float *A_sm = shared_memory_buffer;
  float *B_sm = A_sm + block_tile * k_per_iter;

  const int k_iter_count = (K + k_per_iter - 1) / k_per_iter;

  // The current block needs to compute `block_tile * block_tile` floats in
  // `C`, spanning from `C_m_top` to `C_m_bottom`(exclusive) and from
  // `C_n_left` to `C_n_right`(exclusive).
  int C_m_top = blockIdx.y * block_tile;
  int C_n_left = blockIdx.x * block_tile;
  int C_m_bottom = min(C_m_top + block_tile, M);
  int C_n_right = min(C_n_left + block_tile, N);

  // `C_reg` is being used to store the partial sum.
  float C_reg[thread_tile][thread_tile] = {0};

  // The following code computes the relative position of the top-right float
  // in the `C` submatrix.
  const int lane_id = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;

  // Every thread needs to compute `thread_tile * thread_tile` floats in the
  // `C` matrix.
  // A_reg is K-marjor; B_reg is N-marjor;
  const int k_per_iter_half = k_per_iter / 2;
  float A_reg[thread_tile * k_per_iter_half];
  float B_reg[thread_tile * k_per_iter_half];
  const int A_offset = lane_id / 16 * 64 + lane_id % 16 / 2 * 4;

  // Each thread shifts `warp_id` position to the right within its own
  // half-warp.
  const int B_shifted_lane_id =
      ((lane_id + warp_id * 4) % 16 + (lane_id & 0x10)) ^ (warp_id / 4 * 16);
  const int B_offset = B_shifted_lane_id / 16 * 64 +
                       ((B_shifted_lane_id % 16) & 0xfd) / 4 * 8 +
                       (B_shifted_lane_id & 1) * 4;

  auto grid = cooperative_groups::this_grid();
  auto block = cooperative_groups::this_thread_block();

  constexpr size_t stages_count = 1;
  __shared__ cuda::pipeline_shared_state<cuda::thread_scope::thread_scope_block, stages_count> shared_state;

  auto pipeline = cuda::make_pipeline(block, &shared_state);

  for (int iter = 0; iter < k_iter_count; ++iter) {
    if (iter != 0) {
      block.sync();
    }

    // Read `block_tile * k_per_iter` floats from each of A and B in global
    // memory into shared memory per iteration.
    void *A_sm_dst = A_sm + warp_id * block_tile + lane_id * 4;
    const void *A_global_src =
        A + ((C_m_top + warp_id * 16 + lane_id % 8 + lane_id / 16 * 8) * K +
             iter * k_per_iter + lane_id % 16 / 8 * 4);
    void *B_sm_dst = B_sm + warp_id * block_tile + lane_id * 4;
    const void *B_global_src = B + (iter * k_per_iter + warp_id) * N +
                               C_n_left + lane_id % 8 / 2 * 16 +
                               (lane_id & 1) * 4 + (lane_id & 0x8) +
                               lane_id / 16 * 64;
    pipeline.producer_acquire();
    cuda::memcpy_async(A_sm_dst, A_global_src, cuda::aligned_size_t<16>(1),
                       pipeline);
    cuda::memcpy_async(B_sm_dst, B_global_src, cuda::aligned_size_t<16>(1), pipeline);
    pipeline.producer_commit();
    pipeline.consumer_wait();
    block.sync();
    // if (iter < 16 && blockIdx.x == 1 && blockIdx.y == 1) {
    //   for (int i = 0; i < block.size(); ++i) {
    //     if (i == block.thread_rank()) {
    //       printf("thread = %03d, A_sm = %p, A_global = %p, B_sm = %p, B_global "
    //              "= %p\n",
    //              block.thread_rank(),
    //              (float4 *)(A_sm + warp_id * block_tile + lane_id * 4),
    //              (float4 *)(A + ((C_m_top + warp_id * 16 + lane_id % 8 +
    //                               lane_id / 16 * 8) *
    //                                  K +
    //                              iter * k_per_iter + lane_id % 16 / 8 * 4)),
    //              (float4 *)(B_sm + warp_id * block_tile + lane_id * 4),
    //              (float4 *)(B + ((iter * k_per_iter + warp_id) * N + C_n_left +
    //                              lane_id % 8 / 2 * 16 + (lane_id & 1) * 4 +
    //                              (lane_id & 0x8) + lane_id / 16 * 64)));
    //     }
    //     block.sync();
    //   }
    // }

    // if (iter == 0 && warp_id == 0 && lane_id == 0 && blockIdx.x == 0 &&
    //     blockIdx.y == 0) {
    //   printf("\nA_sm begin\n");
    //   for (int i = 0; i < 128 * 8; ++i) {
    //     if (i % 32 == 0) {
    //       printf("\n");
    //     }
    //     printf("%03d_%03d ", int(A_sm[i]) / 128, int(A_sm[i]) % 128);
    //   }
    //   printf("\nA_sm end\n");
    // }

    // if (iter == 0 && warp_id == 0 && lane_id == 0 && blockIdx.x == 0 &&
    // blockIdx.y == 0) {
    //   printf("\nB_sm begin\n");
    //   for (int i = 0; i < 128 * 8; ++i) {
    //     if (i % 32 == 0) {
    //       printf("\n");
    //     }
    //     printf("%03d_%03d ", int(B_sm[i]) / 128, int(B_sm[i]) % 128);
    //   }
    //   printf("\nB_sm end\n");
    // }

    for (int i = 0; i < thread_tile; ++i) {
      *(float4 *)&A_reg[i * k_per_iter_half] =
          *(float4 *)&A_sm[A_offset + i * 128];
    }
    for (int i = 0; i < k_per_iter_half; ++i) {
      *(float4 *)&B_reg[i * thread_tile] = *(float4 *)&B_sm[B_offset + i * 128];
      *(float4 *)&B_reg[i * thread_tile + 4] =
          *(float4 *)&B_sm[B_offset + i * 128 + 32];
    }

    for (int i = 0; i < thread_tile; ++i) {
      for (int j = 0; j < thread_tile; ++j) {
        for (int k = 0; k < k_per_iter_half; ++k) {
          C_reg[i][j] +=
              A_reg[i * k_per_iter_half + k] * B_reg[k * thread_tile + j];
        }
      }
    }

    for (int i = 0; i < thread_tile; ++i) {
      *(float4 *)&A_reg[i * k_per_iter_half] =
          *(float4 *)&A_sm[A_offset + i * 128 + 32];
    }
    for (int i = 4; i < k_per_iter; ++i) {
      *(float4 *)&B_reg[(i - 4) * thread_tile] =
          *(float4 *)&B_sm[B_offset + i * 128];
      *(float4 *)&B_reg[(i - 4) * thread_tile + 4] =
          *(float4 *)&B_sm[B_offset + i * 128 + 32];
    }

    for (int i = 0; i < thread_tile; ++i) {
      for (int j = 0; j < thread_tile; ++j) {
        for (int k = 0; k < k_per_iter_half; ++k) {
          C_reg[i][j] +=
              A_reg[i * k_per_iter_half + k] * B_reg[k * thread_tile + j];
        }
      }
    }
    pipeline.consumer_release();
  }

  // C_reg is N-marjor
  if (C_m_top + block_tile <= M && C_n_left + block_tile <= N) {
    for (int i = 0; i < thread_tile; ++i) {
      int m = C_m_top + i * 16 + lane_id / 16 * 8 + lane_id % 16 / 2;
      int n = C_n_left + B_shifted_lane_id % 16 / 4 * 16 +
              B_shifted_lane_id / 16 * 64 + (B_shifted_lane_id & 1) * 4;
      *(float4 *)&C[m * N + n] = *(const float4 *)(&C_reg[i][0]);
      *(float4 *)&C[m * N + n + 8] = *(const float4 *)(&C_reg[i][4]);
      // if (blockIdx.x == 0 && blockIdx.y == 0) {
      //   for (int t = 0; t < 256; ++t) {
      //     if (t == threadIdx.x) {
      //       printf("i = %03d, warp_id = %03d, lane_id = %03d, shifted = %03d,
      //       "
      //              "m = %03d, n = "
      //              "%03d\n",
      //              i, warp_id, lane_id, B_shifted_lane_id, m, n);
      //     }
      //     __syncthreads();
      //   }
      // }
    }
  } else {
    // for (int i = 0; i < thread_tile; ++i) {
    //   const int m = C_m_top + C_sub_m + i;
    //   for (int j = 0; j < thread_tile; ++j) {
    //     const int n = C_n_left + C_sub_n + (j < 4 ? j : (32 + j - 4));
    //     if (m < M && n < N) {
    //       C[m * N + n] = C_reg[i][j];
    //     }
    //   }
    // }
  }
}

// A, B and C are all row-marjor matrices;
template <int block_tile, int thread_tile, int k_per_iter>
__global__ void
shared_memory__eliminate_bank_conflict__global_store_memory_colacesing__pipeline_primitive_double_stage(
    const float *__restrict__ A, const float *__restrict__ B,
    float *__restrict__ C, int M, int N, int K) {
  extern __shared__ float shared_memory_buffer[];

  float *A_sm = shared_memory_buffer;
  float *B_sm = shared_memory_buffer + block_tile * k_per_iter * 2;

  const int k_iter_count = (K + k_per_iter - 1) / k_per_iter;

  // The current block needs to compute `block_tile * block_tile` floats in
  // `C`, spanning from `C_m_top` to `C_m_bottom`(exclusive) and from
  // `C_n_left` to `C_n_right`(exclusive).
  int C_m_top = blockIdx.y * block_tile;
  int C_n_left = blockIdx.x * block_tile;
  int C_m_bottom = min(C_m_top + block_tile, M);
  int C_n_right = min(C_n_left + block_tile, N);

  // `C_reg` is being used to store the partial sum.
  float C_reg[thread_tile][thread_tile] = {0};

  // The following code computes the relative position of the top-right float
  // in the `C` submatrix.
  const int lane_id = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;

  // Every thread needs to compute `thread_tile * thread_tile` floats in the
  // `C` matrix.
  // A_reg is K-marjor; B_reg is N-marjor;
  const int k_per_iter_half = k_per_iter / 2;
  float A_reg[thread_tile * k_per_iter_half * 2];
  float B_reg[thread_tile * k_per_iter_half * 2];
  const int A_offset =
      lane_id / 16 * 64 + lane_id % 16 / 4 * 8 + (lane_id & 0x2) * 18;

  // Each thread shifts `warp_id` position to the right within its own
  // half-warp.
  const int B_shifted_lane_id =
      ((lane_id + warp_id * 4) % 16 + (lane_id & 0x10)) ^ (warp_id / 4 * 16);
  const int B_offset = B_shifted_lane_id / 16 * 64 +
                       ((B_shifted_lane_id % 16) & 0xfd) / 4 * 8 +
                       (B_shifted_lane_id & 1) * 4;

  {
    // Read `block_tile * k_per_iter` floats from each of A and B in global
    // memory into shared memory per iteration.
    void *A_sm_dst =
        A_sm + warp_id * block_tile + (lane_id ^ ((lane_id >> 3) & 0x1)) * 4;
    const void *A_global_src =
        A +
        ((C_m_top + warp_id * 16 + lane_id / 16 * 8 + lane_id % 16 / 2) * K +
         +lane_id % 2 * 4);
    void *B_sm_dst = B_sm + warp_id * block_tile + lane_id * 4;
    const void *B_global_src = B + (+warp_id) * N + C_n_left +
                               lane_id % 8 / 2 * 16 + (lane_id & 1) * 4 +
                               (lane_id & 0x8) + lane_id / 16 * 64;
    __pipeline_memcpy_async(A_sm_dst, A_global_src, sizeof(float4));
    __pipeline_memcpy_async(B_sm_dst, B_global_src, sizeof(float4));
    __pipeline_commit();
  }

  for (int iter = 0; iter < k_iter_count; ++iter) {
    const int shared_memory_computing_offset =
        iter % 2 * block_tile * k_per_iter;
    if (iter != 0) {
      __syncthreads();
    }

    // Read `block_tile * k_per_iter` floats from each of A and B in global
    // memory into shared memory per iteration.
    if (iter + 1 < k_iter_count) {
      const int shared_memory_loading_offset =
          (iter + 1) % 2 * block_tile * k_per_iter;
      void *A_sm_dst = A_sm + shared_memory_loading_offset +
                       warp_id * block_tile +
                       (lane_id ^ ((lane_id >> 3) & 0x1)) * 4;
      const void *A_global_src =
          A +
          ((C_m_top + warp_id * 16 + lane_id / 16 * 8 + lane_id % 16 / 2) * K +
           (iter + 1) * k_per_iter + lane_id % 2 * 4);
      void *B_sm_dst = B_sm + shared_memory_loading_offset +
                       warp_id * block_tile + lane_id * 4;
      const void *B_global_src = B + ((iter + 1) * k_per_iter + warp_id) * N +
                                 C_n_left + lane_id % 8 / 2 * 16 +
                                 (lane_id & 1) * 4 + (lane_id & 0x8) +
                                 lane_id / 16 * 64;
      __pipeline_memcpy_async(A_sm_dst, A_global_src, sizeof(float4));
      __pipeline_memcpy_async(B_sm_dst, B_global_src, sizeof(float4));
      __pipeline_commit();
    }
    __pipeline_wait_prior(0);
    __syncthreads();
    // if (iter == 0 && warp_id == 0 && lane_id == 0 && blockIdx.x == 0 &&
    //     blockIdx.y == 0) {
    //   printf("A_SM");
    //   for (int i = 0; i < 8 * 128; ++i) {
    //     if (i % 32 == 0) {
    //       printf("\n");
    //     }
    //     printf("m%03d_k%03d ", int(A_sm[i]) / 128, int(A_sm[i]) % 128);
    //   }
    //   printf("\n");
    //   printf("B_SM");
    //   for (int i = 0; i < 8 * 128; ++i) {
    //     if (i % 32 == 0) {
    //       printf("\n");
    //     }
    //     printf("n%03d_k%03d ", int(B_sm[i]) % 128, int(B_sm[i]) / 128);
    //   }
    //   printf("\n");
    // }

    for (int i = 0; i < thread_tile; ++i) {
      *(float4 *)&A_reg[i * k_per_iter_half] =
          *(float4 *)&A_sm[shared_memory_computing_offset + A_offset + i * 128];
    }
    // if (iter == 0 && warp_id == 0 && lane_id == 0 && blockIdx.x == 0 &&
    //     blockIdx.y == 0) {
    //   printf("A_reg\n");
    //   for (int i = 0; i < thread_tile; ++i) {
    //     for (int j = 0; j < k_per_iter_half; ++j) {
    //       printf("m%03d_k%03d ", int(A_reg[i * k_per_iter_half + j]) / 128,
    //              int(A_reg[i * k_per_iter_half + j]) % 128);
    //     }
    //     printf("\n");
    //   }
    // }
    for (int i = 0; i < k_per_iter_half; ++i) {
      *(float4 *)&B_reg[i * thread_tile] =
          *(float4 *)&B_sm[shared_memory_computing_offset + B_offset + i * 128];
      *(float4 *)&B_reg[i * thread_tile + 4] =
          *(float4 *)&B_sm[shared_memory_computing_offset + B_offset + i * 128 +
                           32];
    }

    // if (iter == 0 && warp_id == 0 && lane_id == 0 && blockIdx.x == 0 &&
    //     blockIdx.y == 0) {
    //   printf("B_reg\n");
    //   for (int j = 0; j < k_per_iter_half; ++j) {
    //     for (int i = 0; i < thread_tile; ++i) {
    //       printf("k%03d_n%03d ", int(B_reg[j * thread_tile + i]) / 128,
    //              int(B_reg[j * thread_tile + i]) % 128);
    //     }
    //     printf("\n");
    //   }
    // }

    for (int i = 0; i < thread_tile; ++i) {
      *(float4 *)&A_reg[i * k_per_iter_half + thread_tile * k_per_iter_half] =
          *(float4 *)&A_sm[shared_memory_computing_offset + (A_offset ^ 4) + i * 128];
    }
    for (int i = 4; i < k_per_iter; ++i) {
      *(float4 *)&B_reg[(i - 4) * thread_tile + thread_tile * k_per_iter_half] =
          *(float4 *)&B_sm[shared_memory_computing_offset + B_offset + i * 128];
      *(float4 *)&B_reg[(i - 4) * thread_tile + 4 +
                        thread_tile * k_per_iter_half] =
          *(float4 *)&B_sm[shared_memory_computing_offset + B_offset + i * 128 +
                           32];
    }

    for (int i = 0; i < thread_tile; ++i) {
      for (int j = 0; j < thread_tile; ++j) {
        for (int k = 0; k < k_per_iter_half; ++k) {
          C_reg[i][j] +=
              A_reg[i * k_per_iter_half + k] * B_reg[k * thread_tile + j];
        }
      }
    }

    for (int i = 0; i < thread_tile; ++i) {
      for (int j = 0; j < thread_tile; ++j) {
        for (int k = 0; k < k_per_iter_half; ++k) {
          C_reg[i][j] +=
              A_reg[i * k_per_iter_half + k + thread_tile * k_per_iter_half] *
              B_reg[k * thread_tile + j + thread_tile * k_per_iter_half];
        }
      }
    }
  }

  // C_reg is N-marjor
  if (C_m_top + block_tile <= M && C_n_left + block_tile <= N) {
    for (int i = 0; i < thread_tile; ++i) {
      int m = C_m_top + i * 16 + lane_id / 16 * 8 + lane_id % 16 / 4 +
              (lane_id & 0x2) * 2;
      int n = C_n_left + B_shifted_lane_id % 16 / 4 * 16 +
              B_shifted_lane_id / 16 * 64 + (B_shifted_lane_id & 1) * 4;
      *(float4 *)&C[m * N + n] = *(const float4 *)(&C_reg[i][0]);
      *(float4 *)&C[m * N + n + 8] = *(const float4 *)(&C_reg[i][4]);
      // if (blockIdx.x == 0 && blockIdx.y == 0) {
      //   for (int t = 0; t < 256; ++t) {
      //     if (t == threadIdx.x) {
      //       printf("i = %03d, warp_id = %03d, lane_id = %03d, shifted = %03d,
      //       "
      //              "m = %03d, n = "
      //              "%03d\n",
      //              i, warp_id, lane_id, B_shifted_lane_id, m, n);
      //     }
      //     __syncthreads();
      //   }
      // }
    }
  } else {
    // for (int i = 0; i < thread_tile; ++i) {
    //   const int m = C_m_top + C_sub_m + i;
    //   for (int j = 0; j < thread_tile; ++j) {
    //     const int n = C_n_left + C_sub_n + (j < 4 ? j : (32 + j - 4));
    //     if (m < M && n < N) {
    //       C[m * N + n] = C_reg[i][j];
    //     }
    //   }
    // }
  }
}

// A, B and C are all row-marjor matrices;
template <int block_tile, int thread_tile, int k_per_iter>
__global__ void
shared_memory__eliminate_bank_conflict__global_store_memory_colacesing__pipeline_primitive_single_stage(
    const float *__restrict__ A, const float *__restrict__ B,
    float *__restrict__ C, int M, int N, int K) {
  extern __shared__ float shared_memory_buffer[];

  float *A_sm = shared_memory_buffer;
  float *B_sm = A_sm + block_tile * k_per_iter;

  const int k_iter_count = (K + k_per_iter - 1) / k_per_iter;

  // The current block needs to compute `block_tile * block_tile` floats in
  // `C`, spanning from `C_m_top` to `C_m_bottom`(exclusive) and from
  // `C_n_left` to `C_n_right`(exclusive).
  int C_m_top = blockIdx.y * block_tile;
  int C_n_left = blockIdx.x * block_tile;
  int C_m_bottom = min(C_m_top + block_tile, M);
  int C_n_right = min(C_n_left + block_tile, N);

  // `C_reg` is being used to store the partial sum.
  float C_reg[thread_tile][thread_tile] = {0};

  // The following code computes the relative position of the top-right float
  // in the `C` submatrix.
  const int lane_id = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;

  // Every thread needs to compute `thread_tile * thread_tile` floats in the
  // `C` matrix.
  // A_reg is K-marjor; B_reg is N-marjor;
  const int k_per_iter_half = k_per_iter / 2;
  float A_reg[thread_tile * k_per_iter_half];
  float B_reg[thread_tile * k_per_iter_half];
  const int A_offset =
      lane_id / 16 * 64 + lane_id % 16 / 4 * 8 + (lane_id & 0x2) * 18;

  // Each thread shifts `warp_id` position to the right within its own
  // half-warp.
  const int B_shifted_lane_id =
      ((lane_id + warp_id * 4) % 16 + (lane_id & 0x10)) ^ (warp_id / 4 * 16);
  const int B_offset = B_shifted_lane_id / 16 * 64 +
                       ((B_shifted_lane_id % 16) & 0xfd) / 4 * 8 +
                       (B_shifted_lane_id & 1) * 4;

  for (int iter = 0; iter < k_iter_count; ++iter) {
    if (iter != 0) {
      __syncthreads();
    }

    // Read `block_tile * k_per_iter` floats from each of A and B in global
    // memory into shared memory per iteration.
    void *A_sm_dst =
        A_sm + warp_id * block_tile + (lane_id ^ ((lane_id >> 3) & 0x1)) * 4;
    const void *A_global_src =
        A +
        ((C_m_top + warp_id * 16 + lane_id / 16 * 8 + lane_id % 16 / 2) * K +
         iter * k_per_iter + lane_id % 2 * 4);
    void *B_sm_dst = B_sm + warp_id * block_tile + lane_id * 4;
    const void *B_global_src = B + (iter * k_per_iter + warp_id) * N +
                               C_n_left + lane_id % 8 / 2 * 16 +
                               (lane_id & 1) * 4 + (lane_id & 0x8) +
                               lane_id / 16 * 64;
    __pipeline_memcpy_async(A_sm_dst, A_global_src, sizeof(float4));
    __pipeline_memcpy_async(B_sm_dst, B_global_src, sizeof(float4));
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();
    // if (iter == 0 && warp_id == 0 && lane_id == 0 && blockIdx.x == 0 &&
    //     blockIdx.y == 0) {
    //   printf("A_SM");
    //   for (int i = 0; i < 8 * 128; ++i) {
    //     if (i % 32 == 0) {
    //       printf("\n");
    //     }
    //     printf("m%03d_k%03d ", int(A_sm[i]) / 128, int(A_sm[i]) % 128);
    //   }
    //   printf("\n");
    //   printf("B_SM");
    //   for (int i = 0; i < 8 * 128; ++i) {
    //     if (i % 32 == 0) {
    //       printf("\n");
    //     }
    //     printf("n%03d_k%03d ", int(B_sm[i]) % 128, int(B_sm[i]) / 128);
    //   }
    //   printf("\n");
    // }

    for (int i = 0; i < thread_tile; ++i) {
      *(float4 *)&A_reg[i * k_per_iter_half] =
          *(float4 *)&A_sm[A_offset + i * 128];
    }
    // if (iter == 0 && warp_id == 0 && lane_id == 0 && blockIdx.x == 0 &&
    //     blockIdx.y == 0) {
    //   printf("A_reg\n");
    //   for (int i = 0; i < thread_tile; ++i) {
    //     for (int j = 0; j < k_per_iter_half; ++j) {
    //       printf("m%03d_k%03d ", int(A_reg[i * k_per_iter_half + j]) / 128,
    //              int(A_reg[i * k_per_iter_half + j]) % 128);
    //     }
    //     printf("\n");
    //   }
    // }
    for (int i = 0; i < k_per_iter_half; ++i) {
      *(float4 *)&B_reg[i * thread_tile] = *(float4 *)&B_sm[B_offset + i * 128];
      *(float4 *)&B_reg[i * thread_tile + 4] =
          *(float4 *)&B_sm[B_offset + i * 128 + 32];
    }

    // if (iter == 0 && warp_id == 0 && lane_id == 0 && blockIdx.x == 0 &&
    //     blockIdx.y == 0) {
    //   printf("B_reg\n");
    //   for (int j = 0; j < k_per_iter_half; ++j) {
    //     for (int i = 0; i < thread_tile; ++i) {
    //       printf("k%03d_n%03d ", int(B_reg[j * thread_tile + i]) / 128,
    //              int(B_reg[j * thread_tile + i]) % 128);
    //     }
    //     printf("\n");
    //   }
    // }

    for (int i = 0; i < thread_tile; ++i) {
      for (int j = 0; j < thread_tile; ++j) {
        for (int k = 0; k < k_per_iter_half; ++k) {
          C_reg[i][j] +=
              A_reg[i * k_per_iter_half + k] * B_reg[k * thread_tile + j];
        }
      }
    }

    for (int i = 0; i < thread_tile; ++i) {
      *(float4 *)&A_reg[i * k_per_iter_half] =
          *(float4 *)&A_sm[(A_offset ^ 4) + i * 128];
    }
    for (int i = 4; i < k_per_iter; ++i) {
      *(float4 *)&B_reg[(i - 4) * thread_tile] =
          *(float4 *)&B_sm[B_offset + i * 128];
      *(float4 *)&B_reg[(i - 4) * thread_tile + 4] =
          *(float4 *)&B_sm[B_offset + i * 128 + 32];
    }

    for (int i = 0; i < thread_tile; ++i) {
      for (int j = 0; j < thread_tile; ++j) {
        for (int k = 0; k < k_per_iter_half; ++k) {
          C_reg[i][j] +=
              A_reg[i * k_per_iter_half + k] * B_reg[k * thread_tile + j];
        }
      }
    }
  }

  // C_reg is N-marjor
  if (C_m_top + block_tile <= M && C_n_left + block_tile <= N) {
    for (int i = 0; i < thread_tile; ++i) {
      int m = C_m_top + i * 16 + lane_id / 16 * 8 + lane_id % 16 / 4 +
              (lane_id & 0x2) * 2;
      int n = C_n_left + B_shifted_lane_id % 16 / 4 * 16 +
              B_shifted_lane_id / 16 * 64 + (B_shifted_lane_id & 1) * 4;
      *(float4 *)&C[m * N + n] = *(const float4 *)(&C_reg[i][0]);
      *(float4 *)&C[m * N + n + 8] = *(const float4 *)(&C_reg[i][4]);
      // if (blockIdx.x == 0 && blockIdx.y == 0) {
      //   for (int t = 0; t < 256; ++t) {
      //     if (t == threadIdx.x) {
      //       printf("i = %03d, warp_id = %03d, lane_id = %03d, shifted = %03d,
      //       "
      //              "m = %03d, n = "
      //              "%03d\n",
      //              i, warp_id, lane_id, B_shifted_lane_id, m, n);
      //     }
      //     __syncthreads();
      //   }
      // }
    }
  } else {
    // for (int i = 0; i < thread_tile; ++i) {
    //   const int m = C_m_top + C_sub_m + i;
    //   for (int j = 0; j < thread_tile; ++j) {
    //     const int n = C_n_left + C_sub_n + (j < 4 ? j : (32 + j - 4));
    //     if (m < M && n < N) {
    //       C[m * N + n] = C_reg[i][j];
    //     }
    //   }
    // }
  }
}

// A, B and C are all row-marjor matrices;
template <int block_tile, int thread_tile, int k_per_iter>
__global__ void
shared_memory__eliminate_bank_conflict__global_store_memory_colacesing__pipeline_primitive_single_stage__overlap_sm_latency(
    const float *__restrict__ A, const float *__restrict__ B,
    float *__restrict__ C, int M, int N, int K) {
  extern __shared__ float shared_memory_buffer[];

  float *A_sm = shared_memory_buffer;
  float *B_sm = A_sm + block_tile * k_per_iter;

  const int k_iter_count = (K + k_per_iter - 1) / k_per_iter;

  // The current block needs to compute `block_tile * block_tile` floats in
  // `C`, spanning from `C_m_top` to `C_m_bottom`(exclusive) and from
  // `C_n_left` to `C_n_right`(exclusive).
  int C_m_top = blockIdx.y * block_tile;
  int C_n_left = blockIdx.x * block_tile;
  int C_m_bottom = min(C_m_top + block_tile, M);
  int C_n_right = min(C_n_left + block_tile, N);

  // `C_reg` is being used to store the partial sum.
  float C_reg[thread_tile][thread_tile] = {0};

  // The following code computes the relative position of the top-right float
  // in the `C` submatrix.
  const int lane_id = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;

  // Every thread needs to compute `thread_tile * thread_tile` floats in the
  // `C` matrix.
  // A_reg is K-marjor; B_reg is N-marjor;
  float A_reg[thread_tile * 2];
  float B_reg[thread_tile * 2];
  const int A_offset =
      lane_id / 16 * 64 + lane_id % 16 / 4 * 8 + (lane_id & 0x2) * 18;

  // Each thread shifts `warp_id` position to the right within its own
  // warp.
  const int B_shifted_lane_id =
      ((lane_id + warp_id * 4) % 16 + (lane_id & 0x10)) ^ (warp_id / 4 * 16);
  const int B_offset = B_shifted_lane_id / 16 * 64 +
                       ((B_shifted_lane_id % 16) & 0xfd) / 4 * 8 +
                       (B_shifted_lane_id & 1) * 4;

  for (int iter = 0; iter < k_iter_count; ++iter) {
    if (iter != 0) {
      __syncthreads();
    }

    // Read `block_tile * k_per_iter` floats from each of A and B in global
    // memory into shared memory per iteration.
    for (int i = 0; i < 4; ++i) {
      void *A_sm_dst = A_sm + i * 256 + (lane_id % 8) * 32 + (lane_id & 0xf8) +
                       (lane_id + warp_id) % 8;
      const void *A_global_src =
          A + ((C_m_top + i * 32 + warp_id * 4 + lane_id / 8) * K +
               iter * k_per_iter + lane_id % 8);
      __pipeline_memcpy_async(A_sm_dst, A_global_src, sizeof(float));
    }

    void *B_sm_dst = B_sm + warp_id * block_tile + lane_id * 4;
    const void *B_global_src =
        B + (iter * k_per_iter + warp_id) * N + C_n_left + lane_id * 4;
    __pipeline_memcpy_async(B_sm_dst, B_global_src, sizeof(float4));
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();
    // if (iter == 0 && warp_id == 0 && lane_id == 0 && blockIdx.x == 0 &&
    //     blockIdx.y == 0) {
    //   printf("A_SM");
    //   for (int i = 0; i < 8 * 128; ++i) {
    //     if (i % 32 == 0) {
    //       printf("\n");
    //     }
    //     printf("m%03d_k%03d ", int(A_sm[i]) / 128, int(A_sm[i]) % 128);
    //   }
    //   printf("\n");
    //   printf("B_SM");
    //   for (int i = 0; i < 8 * 128; ++i) {
    //     if (i % 32 == 0) {
    //       printf("\n");
    //     }
    //     printf("n%03d_k%03d ", int(B_sm[i]) % 128, int(B_sm[i]) / 128);
    //   }
    //   printf("\n");
    // }
    {
      constexpr int k = 0;
      const int A_offset = lane_id / 8 * 256 + k * 32 + lane_id % 8 * 4;
      *(float4 *)&A_reg[0] = *(const float4 *)&A_sm[A_offset];
      *(float4 *)&A_reg[4] = *(const float4 *)&A_sm[A_offset ^ 4];
      const int B_offset = k * 128 + B_shifted_lane_id / 4 * 16 +
                           (B_shifted_lane_id & 0x1) * 8 +
                           (B_shifted_lane_id & 0x2) * 2;
      *(float4 *)&B_reg[0] = *(const float4 *)&B_sm[B_offset];
      *(float4 *)&B_reg[4] = *(const float4 *)&B_sm[B_offset ^ 4];
    }
#pragma unroll
    for (int k = 1; k < k_per_iter; ++k) {
      const int loading_reg_offset = k % 2 * thread_tile;
      const int A_offset = lane_id / 8 * 256 + k * 32 + lane_id % 8 * 4;
      *(float4 *)&A_reg[loading_reg_offset] = *(const float4 *)&A_sm[A_offset];
      *(float4 *)&A_reg[4 + loading_reg_offset] =
          *(const float4 *)&A_sm[A_offset ^ 4];
      const int B_offset = k * 128 + B_shifted_lane_id / 4 * 16 +
                           (B_shifted_lane_id & 0x1) * 8 +
                           (B_shifted_lane_id & 0x2) * 2;
      *(float4 *)&B_reg[loading_reg_offset] = *(const float4 *)&B_sm[B_offset];
      *(float4 *)&B_reg[4 + loading_reg_offset] =
          *(const float4 *)&B_sm[B_offset ^ 4];

      const int computing_reg_offset = (k - 1) % 2 * thread_tile;

      for (int i = 0; i < thread_tile; ++i) {
        for (int j = 0; j < thread_tile; ++j) {
          C_reg[i][j] +=
              A_reg[i + computing_reg_offset] * B_reg[j + computing_reg_offset];
        }
      }
    }
    for (int i = 0; i < thread_tile; ++i) {
      for (int j = 0; j < thread_tile; ++j) {
        C_reg[i][j] += A_reg[i + thread_tile] * B_reg[j + thread_tile];
      }
    }
  }

  // C_reg is N-marjor
  if (C_m_top + block_tile <= M && C_n_left + block_tile <= N) {
    for (int i = 0; i < thread_tile; ++i) {
      int m = C_m_top + i * 16 + lane_id / 16 * 8 + lane_id % 16 / 4 +
              (lane_id & 0x2) * 2;
      int n = C_n_left + B_shifted_lane_id % 16 / 4 * 16 +
              B_shifted_lane_id / 16 * 64 + (B_shifted_lane_id & 1) * 4;
      *(float4 *)&C[m * N + n] = *(const float4 *)(&C_reg[i][0]);
      *(float4 *)&C[m * N + n + 8] = *(const float4 *)(&C_reg[i][4]);
      // if (blockIdx.x == 0 && blockIdx.y == 0) {
      //   for (int t = 0; t < 256; ++t) {
      //     if (t == threadIdx.x) {
      //       printf("i = %03d, warp_id = %03d, lane_id = %03d, shifted = %03d,
      //       "
      //              "m = %03d, n = "
      //              "%03d\n",
      //              i, warp_id, lane_id, B_shifted_lane_id, m, n);
      //     }
      //     __syncthreads();
      //   }
      // }
    }
  } else {
    // for (int i = 0; i < thread_tile; ++i) {
    //   const int m = C_m_top + C_sub_m + i;
    //   for (int j = 0; j < thread_tile; ++j) {
    //     const int n = C_n_left + C_sub_n + (j < 4 ? j : (32 + j - 4));
    //     if (m < M && n < N) {
    //       C[m * N + n] = C_reg[i][j];
    //     }
    //   }
    // }
  }
}

// A, B and C are all row-marjor matrices;
template <int block_tile, int thread_tile, int k_per_iter>
__global__ void
shared_memory__eliminate_bank_conflict__global_store_memory_colacesing__pipeline_2_stage__only_A(
    const float *A, const float *B, float *C, int M, int N, int K) {
  extern __shared__ float shared_memory_buffer[];

  float *A_sm = shared_memory_buffer;
  float *B_sm = shared_memory_buffer + block_tile * k_per_iter * 2;

  const int k_iter_count = (K + k_per_iter - 1) / k_per_iter;

  // The current block needs to compute `block_tile * block_tile` floats in
  // `C`, spanning from `C_m_top` to `C_m_bottom`(exclusive) and from
  // `C_n_left` to `C_n_right`(exclusive).
  int C_m_top = blockIdx.y * block_tile;
  int C_n_left = blockIdx.x * block_tile;
  int C_m_bottom = min(C_m_top + block_tile, M);
  int C_n_right = min(C_n_left + block_tile, N);

  // `C_reg` is being used to store the partial sum.
  float C_reg[thread_tile][thread_tile] = {0};

  // The following code computes the relative position of the top-right float
  // in the `C` submatrix.
  const int lane_id = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;

  // Every thread needs to compute `thread_tile * thread_tile` floats in the
  // `C` matrix.
  // A_reg is K-marjor; B_reg is N-marjor;
  const int k_per_iter_half = k_per_iter / 2;
  float A_reg[thread_tile * k_per_iter_half];
  float B_reg[thread_tile * k_per_iter_half];
  const int A_offset = lane_id / 16 * 64 + lane_id % 16 / 2 * 4;

  // Each thread shifts `warp_id` position to the right within its own
  // half-warp.
  const int B_shifted_lane_id =
      ((lane_id + warp_id * 4) % 16 + (lane_id & 0x10)) ^ (warp_id / 4 * 16);
  const int B_offset = B_shifted_lane_id / 16 * 64 +
                       ((B_shifted_lane_id % 16) & 0xfd) / 4 * 8 +
                       (B_shifted_lane_id & 1) * 4;

  auto grid = cooperative_groups::this_grid();
  auto block = cooperative_groups::this_thread_block();

  constexpr size_t stages_count = 2;
  __shared__ cuda::pipeline_shared_state<cuda::thread_scope::thread_scope_block, stages_count> shared_state;

  auto pipeline = cuda::make_pipeline(block, &shared_state);

  {
    void *A_sm_dst = A_sm + warp_id * block_tile + lane_id * 4;
    const void *A_global_src =
        A + (C_m_top + warp_id * 16 + lane_id % 8 + lane_id / 16 * 8) * K +
        lane_id % 16 / 8 * 4;
    void *B_sm_dst = B_sm + warp_id * block_tile + lane_id * 4;
    const void *B_global_src = B + warp_id * N + C_n_left +
                               lane_id % 8 / 2 * 16 + (lane_id & 1) * 4 +
                               (lane_id & 0x8) + lane_id / 16 * 64;
    pipeline.producer_acquire();
    cuda::memcpy_async(A_sm_dst, A_global_src, cuda::aligned_size_t<16>(1),
                       pipeline);
    // cuda::memcpy_async(B_sm_dst, B_global_src, cuda::aligned_size_t<16>(1),
    //                    pipeline);
    pipeline.producer_commit();
  }

  for (int iter = 0; iter < k_iter_count; ++iter) {
    // Read `block_tile * k_per_iter` floats from each of A and B in global
    // memory into shared memory per iteration.
    const int shared_memory_computing_offset =
        iter % 2 * block_tile * k_per_iter;

    if (iter + 1 < k_iter_count) {
      const int shared_memory_loading_offset =
          (iter + 1) % 2 * block_tile * k_per_iter;
      void *A_sm_dst = A_sm + warp_id * block_tile + lane_id * 4 +
                       shared_memory_loading_offset;
      const void *A_global_src =
          A + (C_m_top + warp_id * 16 + lane_id % 8 + lane_id / 16 * 8) * K +
          (iter + 1) * k_per_iter + lane_id % 16 / 8 * 4;
      void *B_sm_dst = B_sm + warp_id * block_tile + lane_id * 4 +
                       shared_memory_loading_offset;
      const void *B_global_src = B + ((iter + 1) * k_per_iter + warp_id) * N +
                                 C_n_left + lane_id % 8 / 2 * 16 +
                                 (lane_id & 1) * 4 + (lane_id & 0x8) +
                                 lane_id / 16 * 64;
      pipeline.producer_acquire();
      cuda::memcpy_async(A_sm_dst, A_global_src, cuda::aligned_size_t<16>(1),
                         pipeline);
      // cuda::memcpy_async(B_sm_dst, B_global_src, cuda::aligned_size_t<16>(1),
      //                    pipeline);
      pipeline.producer_commit();
    }

    pipeline.consumer_wait();

    for (int i = 0; i < thread_tile; ++i) {
      *(float4 *)&A_reg[i * k_per_iter_half] =
          *(float4 *)&A_sm[shared_memory_computing_offset + A_offset + i * 128];
    }
    for (int i = 0; i < k_per_iter_half; ++i) {
      *(float4 *)&B_reg[i * thread_tile] =
          *(float4 *)&B_sm[shared_memory_computing_offset + B_offset + i * 128];
      *(float4 *)&B_reg[i * thread_tile + 4] =
          *(float4 *)&B_sm[shared_memory_computing_offset + B_offset + i * 128 +
                           32];
    }

    for (int i = 0; i < thread_tile; ++i) {
      for (int j = 0; j < thread_tile; ++j) {
        for (int k = 0; k < k_per_iter_half; ++k) {
          C_reg[i][j] +=
              A_reg[i * k_per_iter_half + k] * B_reg[k * thread_tile + j];
        }
      }
    }

    for (int i = 0; i < thread_tile; ++i) {
      *(float4 *)&A_reg[i * k_per_iter_half] =
          *(float4 *)&A_sm[shared_memory_computing_offset + A_offset + i * 128 +
                           32];
    }
    for (int i = 4; i < k_per_iter; ++i) {
      *(float4 *)&B_reg[(i - 4) * thread_tile] =
          *(float4 *)&B_sm[shared_memory_computing_offset + B_offset + i * 128];
      *(float4 *)&B_reg[(i - 4) * thread_tile + 4] =
          *(float4 *)&B_sm[shared_memory_computing_offset + B_offset + i * 128 +
                           32];
    }

    for (int i = 0; i < thread_tile; ++i) {
      for (int j = 0; j < thread_tile; ++j) {
        for (int k = 0; k < k_per_iter_half; ++k) {
          C_reg[i][j] +=
              A_reg[i * k_per_iter_half + k] * B_reg[k * thread_tile + j];
        }
      }
    }
    pipeline.consumer_release();
  }

  // C_reg is N-marjor
  if (C_m_top + block_tile <= M && C_n_left + block_tile <= N) {
    for (int i = 0; i < thread_tile; ++i) {
      int m = C_m_top + i * 16 + lane_id / 16 * 8 + lane_id % 16 / 2;
      int n = C_n_left + B_shifted_lane_id % 16 / 4 * 16 +
              B_shifted_lane_id / 16 * 64 + (B_shifted_lane_id & 1) * 4;
      *(float4 *)&C[m * N + n] = *(const float4 *)(&C_reg[i][0]);
      *(float4 *)&C[m * N + n + 8] = *(const float4 *)(&C_reg[i][4]);
    }
  } else {
    // for (int i = 0; i < thread_tile; ++i) {
    //   const int m = C_m_top + C_sub_m + i;
    //   for (int j = 0; j < thread_tile; ++j) {
    //     const int n = C_n_left + C_sub_n + (j < 4 ? j : (32 + j - 4));
    //     if (m < M && n < N) {
    //       C[m * N + n] = C_reg[i][j];
    //     }
    //   }
    // }
  }
}

// A, B and C are all row-marjor matrices;
template <int block_tile, int thread_tile, int k_per_iter>
__global__ void
shared_memory__eliminate_bank_conflict__global_store_memory_colacesing__pipeline_2_stage__only_B(
    const float *A, const float *B, float *C, int M, int N, int K) {
  extern __shared__ float shared_memory_buffer[];

  float *A_sm = shared_memory_buffer;
  float *B_sm = shared_memory_buffer + block_tile * k_per_iter * 2;

  const int k_iter_count = (K + k_per_iter - 1) / k_per_iter;

  // The current block needs to compute `block_tile * block_tile` floats in
  // `C`, spanning from `C_m_top` to `C_m_bottom`(exclusive) and from
  // `C_n_left` to `C_n_right`(exclusive).
  int C_m_top = blockIdx.y * block_tile;
  int C_n_left = blockIdx.x * block_tile;
  int C_m_bottom = min(C_m_top + block_tile, M);
  int C_n_right = min(C_n_left + block_tile, N);

  // `C_reg` is being used to store the partial sum.
  float C_reg[thread_tile][thread_tile] = {0};

  // The following code computes the relative position of the top-right float
  // in the `C` submatrix.
  const int lane_id = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;

  // Every thread needs to compute `thread_tile * thread_tile` floats in the
  // `C` matrix.
  // A_reg is K-marjor; B_reg is N-marjor;
  const int k_per_iter_half = k_per_iter / 2;
  float A_reg[thread_tile * k_per_iter_half];
  float B_reg[thread_tile * k_per_iter_half];
  const int A_offset = lane_id / 16 * 64 + lane_id % 16 / 2 * 4;

  // Each thread shifts `warp_id` position to the right within its own
  // half-warp.
  // const int B_shifted_lane_id =
  //     ((lane_id + warp_id * 4) % 16 + (lane_id & 0x10)) ^ (warp_id / 4 * 16);
  const int B_shifted_lane_id = lane_id;
  const int B_offset = B_shifted_lane_id / 16 * 64 +
                       ((B_shifted_lane_id % 16) & 0xfd) / 4 * 8 +
                       (B_shifted_lane_id & 1) * 4;

  auto grid = cooperative_groups::this_grid();
  auto block = cooperative_groups::this_thread_block();

  constexpr size_t stages_count = 2;
  __shared__ cuda::pipeline_shared_state<cuda::thread_scope::thread_scope_block, stages_count> shared_state;

  auto pipeline = cuda::make_pipeline(block, &shared_state);

  {
    void *A_sm_dst = A_sm + warp_id * block_tile + lane_id * 4;
    const void *A_global_src =
        A + (C_m_top + warp_id * 16 + lane_id % 8 + lane_id / 16 * 8) * K +
        lane_id % 16 / 8 * 4;
    void *B_sm_dst = B_sm + warp_id * block_tile + lane_id * 4;
    const void *B_global_src = B + warp_id * N + C_n_left +
                               lane_id % 8 / 2 * 16 + (lane_id & 1) * 4 +
                               (lane_id & 0x8) + lane_id / 16 * 64;
    pipeline.producer_acquire();
    // cuda::memcpy_async(A_sm_dst, A_global_src, cuda::aligned_size_t<16>(1),
    //                    pipeline);
    cuda::memcpy_async(B_sm_dst, B_global_src, cuda::aligned_size_t<16>(1),
                       pipeline);
    pipeline.producer_commit();
  }

  for (int iter = 0; iter < k_iter_count; ++iter) {
    // Read `block_tile * k_per_iter` floats from each of A and B in global
    // memory into shared memory per iteration.
    const int shared_memory_computing_offset =
        iter % 2 * block_tile * k_per_iter;

    if (iter + 1 < k_iter_count) {
      const int shared_memory_loading_offset =
          (iter + 1) % 2 * block_tile * k_per_iter;
      void *A_sm_dst = A_sm + warp_id * block_tile + lane_id * 4 +
                       shared_memory_loading_offset;
      const void *A_global_src =
          A + (C_m_top + warp_id * 16 + lane_id % 8 + lane_id / 16 * 8) * K +
          (iter + 1) * k_per_iter + lane_id % 16 / 8 * 4;
      void *B_sm_dst = B_sm + warp_id * block_tile + lane_id * 4 +
                       shared_memory_loading_offset;
      const void *B_global_src = B + ((iter + 1) * k_per_iter + warp_id) * N +
                                 C_n_left + lane_id % 8 / 2 * 16 +
                                 (lane_id & 1) * 4 + (lane_id & 0x8) +
                                 lane_id / 16 * 64;
      pipeline.producer_acquire();
      // cuda::memcpy_async(A_sm_dst, A_global_src, cuda::aligned_size_t<16>(1),
      //                    pipeline);
      cuda::memcpy_async(B_sm_dst, B_global_src, cuda::aligned_size_t<16>(1),
                         pipeline);
      pipeline.producer_commit();
    }

    pipeline.consumer_wait();

    for (int i = 0; i < thread_tile; ++i) {
      *(float4 *)&A_reg[i * k_per_iter_half] =
          *(float4 *)&A_sm[shared_memory_computing_offset + A_offset + i * 128];
    }
    for (int i = 0; i < k_per_iter_half; ++i) {
      *(float4 *)&B_reg[i * thread_tile] =
          *(float4 *)&B_sm[shared_memory_computing_offset + B_offset + i * 128];
      *(float4 *)&B_reg[i * thread_tile + 4] =
          *(float4 *)&B_sm[shared_memory_computing_offset + B_offset + i * 128 +
                           32];
    }

    for (int i = 0; i < thread_tile; ++i) {
      for (int j = 0; j < thread_tile; ++j) {
        for (int k = 0; k < k_per_iter_half; ++k) {
          C_reg[i][j] +=
              A_reg[i * k_per_iter_half + k] * B_reg[k * thread_tile + j];
        }
      }
    }

    for (int i = 0; i < thread_tile; ++i) {
      *(float4 *)&A_reg[i * k_per_iter_half] =
          *(float4 *)&A_sm[shared_memory_computing_offset + A_offset + i * 128 +
                           32];
    }
    for (int i = 4; i < k_per_iter; ++i) {
      *(float4 *)&B_reg[(i - 4) * thread_tile] =
          *(float4 *)&B_sm[shared_memory_computing_offset + B_offset + i * 128];
      *(float4 *)&B_reg[(i - 4) * thread_tile + 4] =
          *(float4 *)&B_sm[shared_memory_computing_offset + B_offset + i * 128 +
                           32];
    }

    for (int i = 0; i < thread_tile; ++i) {
      for (int j = 0; j < thread_tile; ++j) {
        for (int k = 0; k < k_per_iter_half; ++k) {
          C_reg[i][j] +=
              A_reg[i * k_per_iter_half + k] * B_reg[k * thread_tile + j];
        }
      }
    }
    pipeline.consumer_release();
  }

  // C_reg is N-marjor
  if (C_m_top + block_tile <= M && C_n_left + block_tile <= N) {
    for (int i = 0; i < thread_tile; ++i) {
      int m = C_m_top + i * 16 + lane_id / 16 * 8 + lane_id % 16 / 2;
      int n = C_n_left + B_shifted_lane_id % 16 / 4 * 16 +
              B_shifted_lane_id / 16 * 64 + (B_shifted_lane_id & 1) * 4;
      *(float4 *)&C[m * N + n] = *(const float4 *)(&C_reg[i][0]);
      *(float4 *)&C[m * N + n + 8] = *(const float4 *)(&C_reg[i][4]);
    }
  } else {
    // for (int i = 0; i < thread_tile; ++i) {
    //   const int m = C_m_top + C_sub_m + i;
    //   for (int j = 0; j < thread_tile; ++j) {
    //     const int n = C_n_left + C_sub_n + (j < 4 ? j : (32 + j - 4));
    //     if (m < M && n < N) {
    //       C[m * N + n] = C_reg[i][j];
    //     }
    //   }
    // }
  }
}

// A, B and C are all row-marjor matrices;
template <int block_tile, int thread_tile, int k_per_iter>
__global__ void
shared_memory__eliminate_bank_conflict__global_store_memory_colacesing__pipeline_2_stage(
    const float *A, const float *B, float *C, int M, int N, int K) {
  extern __shared__ float shared_memory_buffer[];

  float *A_sm = shared_memory_buffer;
  float *B_sm = shared_memory_buffer + block_tile * k_per_iter * 2;

  const int k_iter_count = (K + k_per_iter - 1) / k_per_iter;

  // The current block needs to compute `block_tile * block_tile` floats in
  // `C`, spanning from `C_m_top` to `C_m_bottom`(exclusive) and from
  // `C_n_left` to `C_n_right`(exclusive).
  int C_m_top = blockIdx.y * block_tile;
  int C_n_left = blockIdx.x * block_tile;
  int C_m_bottom = min(C_m_top + block_tile, M);
  int C_n_right = min(C_n_left + block_tile, N);

  // `C_reg` is being used to store the partial sum.
  float C_reg[thread_tile][thread_tile] = {0};

  // The following code computes the relative position of the top-right float
  // in the `C` submatrix.
  const int lane_id = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;

  // Every thread needs to compute `thread_tile * thread_tile` floats in the
  // `C` matrix.
  // A_reg is K-marjor; B_reg is N-marjor;
  const int k_per_iter_half = k_per_iter / 2;
  float A_reg[thread_tile * k_per_iter_half];
  float B_reg[thread_tile * k_per_iter_half];
  const int A_offset = lane_id / 16 * 64 + lane_id % 16 / 2 * 4;

  // Each thread shifts `warp_id` position to the right within its own
  // half-warp.
  const int B_shifted_lane_id =
      ((lane_id + warp_id * 4) % 16 + (lane_id & 0x10)) ^ (warp_id / 4 * 16);
  const int B_offset = B_shifted_lane_id / 16 * 64 +
                       ((B_shifted_lane_id % 16) & 0xfd) / 4 * 8 +
                       (B_shifted_lane_id & 1) * 4;

  auto grid = cooperative_groups::this_grid();
  auto block = cooperative_groups::this_thread_block();

  constexpr size_t stages_count = 2;
  __shared__ cuda::pipeline_shared_state<cuda::thread_scope::thread_scope_block, stages_count> shared_state;

  auto pipeline = cuda::make_pipeline(block, &shared_state);

  {
    void *A_sm_dst = A_sm + warp_id * block_tile + lane_id * 4;
    const void *A_global_src =
        A + (C_m_top + warp_id * 16 + lane_id % 8 + lane_id / 16 * 8) * K +
        lane_id % 16 / 8 * 4;
    void *B_sm_dst = B_sm + warp_id * block_tile + lane_id * 4;
    const void *B_global_src = B + warp_id * N + C_n_left +
                               lane_id % 8 / 2 * 16 + (lane_id & 1) * 4 +
                               (lane_id & 0x8) + lane_id / 16 * 64;
    pipeline.producer_acquire();
    cuda::memcpy_async(A_sm_dst, A_global_src, cuda::aligned_size_t<16>(1),
                       pipeline);
    cuda::memcpy_async(B_sm_dst, B_global_src, cuda::aligned_size_t<16>(1),
                       pipeline);
    pipeline.producer_commit();
  }

  for (int iter = 0; iter < k_iter_count; ++iter) {
    // Read `block_tile * k_per_iter` floats from each of A and B in global
    // memory into shared memory per iteration.
    const int shared_memory_computing_offset =
        iter % 2 * block_tile * k_per_iter;

    if (iter + 1 < k_iter_count) {
      const int shared_memory_loading_offset =
          (iter + 1) % 2 * block_tile * k_per_iter;
      void *A_sm_dst = A_sm + warp_id * block_tile + lane_id * 4 +
                       shared_memory_loading_offset;
      const void *A_global_src =
          A + (C_m_top + warp_id * 16 + lane_id % 8 + lane_id / 16 * 8) * K +
          (iter + 1) * k_per_iter + lane_id % 16 / 8 * 4;
      void *B_sm_dst = B_sm + warp_id * block_tile + lane_id * 4 +
                       shared_memory_loading_offset;
      const void *B_global_src = B + ((iter + 1) * k_per_iter + warp_id) * N +
                                 C_n_left + lane_id % 8 / 2 * 16 +
                                 (lane_id & 1) * 4 + (lane_id & 0x8) +
                                 lane_id / 16 * 64;
      pipeline.producer_acquire();
      cuda::memcpy_async(A_sm_dst, A_global_src, cuda::aligned_size_t<16>(1),
                         pipeline);
      cuda::memcpy_async(B_sm_dst, B_global_src, cuda::aligned_size_t<16>(1),
                         pipeline);
      pipeline.producer_commit();
    }

    pipeline.consumer_wait();

    for (int i = 0; i < thread_tile; ++i) {
      *(float4 *)&A_reg[i * k_per_iter_half] =
          *(float4 *)&A_sm[shared_memory_computing_offset + A_offset + i * 128];
    }
    for (int i = 0; i < k_per_iter_half; ++i) {
      *(float4 *)&B_reg[i * thread_tile] =
          *(float4 *)&B_sm[shared_memory_computing_offset + B_offset + i * 128];
      *(float4 *)&B_reg[i * thread_tile + 4] =
          *(float4 *)&B_sm[shared_memory_computing_offset + B_offset + i * 128 +
                           32];
    }

    for (int i = 0; i < thread_tile; ++i) {
      for (int j = 0; j < thread_tile; ++j) {
        for (int k = 0; k < k_per_iter_half; ++k) {
          C_reg[i][j] +=
              A_reg[i * k_per_iter_half + k] * B_reg[k * thread_tile + j];
        }
      }
    }

    for (int i = 0; i < thread_tile; ++i) {
      *(float4 *)&A_reg[i * k_per_iter_half] =
          *(float4 *)&A_sm[shared_memory_computing_offset + A_offset + i * 128 +
                           32];
    }
    for (int i = 4; i < k_per_iter; ++i) {
      *(float4 *)&B_reg[(i - 4) * thread_tile] =
          *(float4 *)&B_sm[shared_memory_computing_offset + B_offset + i * 128];
      *(float4 *)&B_reg[(i - 4) * thread_tile + 4] =
          *(float4 *)&B_sm[shared_memory_computing_offset + B_offset + i * 128 +
                           32];
    }

    for (int i = 0; i < thread_tile; ++i) {
      for (int j = 0; j < thread_tile; ++j) {
        for (int k = 0; k < k_per_iter_half; ++k) {
          C_reg[i][j] +=
              A_reg[i * k_per_iter_half + k] * B_reg[k * thread_tile + j];
        }
      }
    }
    pipeline.consumer_release();
  }

  // C_reg is N-marjor
  if (C_m_top + block_tile <= M && C_n_left + block_tile <= N) {
    for (int i = 0; i < thread_tile; ++i) {
      int m = C_m_top + i * 16 + lane_id / 16 * 8 + lane_id % 16 / 2;
      int n = C_n_left + B_shifted_lane_id % 16 / 4 * 16 +
              B_shifted_lane_id / 16 * 64 + (B_shifted_lane_id & 1) * 4;
      *(float4 *)&C[m * N + n] = *(const float4 *)(&C_reg[i][0]);
      *(float4 *)&C[m * N + n + 8] = *(const float4 *)(&C_reg[i][4]);
    }
  } else {
    // for (int i = 0; i < thread_tile; ++i) {
    //   const int m = C_m_top + C_sub_m + i;
    //   for (int j = 0; j < thread_tile; ++j) {
    //     const int n = C_n_left + C_sub_n + (j < 4 ? j : (32 + j - 4));
    //     if (m < M && n < N) {
    //       C[m * N + n] = C_reg[i][j];
    //     }
    //   }
    // }
  }
}

// A, B and C are all row-marjor matrices;
template <int block_tile, int thread_tile, int k_per_iter>
__global__ void
shared_memory__eliminate_bank_conflict__global_store_memory_colacesing__less_register(
    const float *A, const float *B, float *C, int M, int N, int K) {
  extern __shared__ float shared_memory_buffer[];

  float *A_sm = shared_memory_buffer;
  float *B_sm = A_sm + block_tile * k_per_iter;

  const int k_iter_count = (K + k_per_iter - 1) / k_per_iter;

  // The current block needs to compute `block_tile * block_tile` floats in
  // `C`, spanning from `C_m_top` to `C_m_bottom`(exclusive) and from
  // `C_n_left` to `C_n_right`(exclusive).
  const int C_m_top = blockIdx.y * block_tile;
  const int C_n_left = blockIdx.x * block_tile;
  const int C_m_bottom = min(C_m_top + block_tile, M);
  const int C_n_right = min(C_n_left + block_tile, N);

  // The following code computes the relative position of the top-right float
  // in the `C` submatrix.
  const int lane_id = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;

  constexpr int k_per_iter_half = k_per_iter / 2;
  constexpr int k_per_iter_quarter = k_per_iter / 4;
  // Every thread needs to compute `thread_tile * thread_tile` floats in the
  // `C` matrix.
  // A_reg is K-marjor; B_reg is N-marjor;
  float A_reg[thread_tile * k_per_iter_quarter];
  float B_reg[thread_tile * k_per_iter_half];
  // `C_reg` is used to store the partial sum. N-Marjor.
  float C_reg[thread_tile][thread_tile] = {0};

  const int A_offset = lane_id / 16 * 64 + lane_id % 16 / 2 * 4;

  // Each thread shifts `warp_id` position to the right within its own
  // half-warp.
  const int B_shifted_lane_id =
      ((lane_id + warp_id * 4) % 16 + (lane_id & 0x10)) ^ (warp_id / 4 * 16);
  const int B_offset = B_shifted_lane_id / 16 * 64 +
                       ((B_shifted_lane_id % 16) & 0xfd) / 4 * 8 +
                       (B_shifted_lane_id & 1) * 4;

  for (int iter = 0; iter < k_iter_count; ++iter) {
    if (iter != 0) {
      __syncthreads();
    }

    // Read `block_tile * k_per_iter` floats from each of A and B in global
    // memory into shared memory per iteration.
    {
      *(float4 *)(&A_sm[warp_id * block_tile + lane_id * 4]) = __ldg(
          (const float4
               *)(&A[(C_m_top + warp_id * 16 + lane_id % 8 + lane_id / 16 * 8) *
                         K +
                     iter * k_per_iter + lane_id % 16 / 8 * 4]));
    }
    {
      *(float4 *)(&B_sm[warp_id * block_tile + lane_id * 4]) = __ldg(
          (const float4 *)(&B[(iter * k_per_iter + warp_id) * N + C_n_left +
                              lane_id % 8 / 2 * 16 + (lane_id & 1) * 4 +
                              (lane_id & 0x8) + lane_id / 16 * 64]));
    }
    __syncthreads();

    int B_k_swizzle = (lane_id < 16 ? 0 : 2);
    for (int k_iter_offset = 0; k_iter_offset <= 4; k_iter_offset += 4) {
      for (int k_iter = 0; k_iter < 4; ++k_iter) {
        *(float4 *)&B_reg[k_iter * thread_tile] =
            *(float4 *)&B_sm[B_offset +
                             ((k_iter ^ B_k_swizzle) + k_iter_offset) * 128];
        *(float4 *)&B_reg[k_iter * thread_tile + 4] = *(
            float4 *)&B_sm[B_offset +
                           ((k_iter ^ B_k_swizzle) + k_iter_offset) * 128 + 32];
      }

      for (int A_k_swizzle = 0; A_k_swizzle <= 2; A_k_swizzle += 2) {
        int k_offset = B_k_swizzle ^ A_k_swizzle;
        for (int i = 0; i < thread_tile; ++i) {
          *(float2 *)&A_reg[i * k_per_iter_quarter] = *(
              float2 *)&A_sm[A_offset + i * 128 + k_offset + k_iter_offset * 8];
        }

        for (int i = 0; i < thread_tile; ++i) {
          for (int j = 0; j < thread_tile; ++j) {
            for (int k = 0; k < k_per_iter_quarter; ++k) {
              C_reg[i][j] += A_reg[i * k_per_iter_quarter + k] *
                             B_reg[(k + A_k_swizzle) * thread_tile + j];
            }
          }
        }
      }
    }
  }

  // C_reg is N-marjor
  if (C_m_top + block_tile <= M && C_n_left + block_tile <= N) {
    for (int i = 0; i < thread_tile; ++i) {
      int m = C_m_top + i * 16 + lane_id / 16 * 8 + lane_id % 16 / 2;
      int n = C_n_left + B_shifted_lane_id % 16 / 4 * 16 +
              B_shifted_lane_id / 16 * 64 + (B_shifted_lane_id & 1) * 4;
      *(float4 *)&C[m * N + n] = *(const float4 *)(&C_reg[i][0]);
      *(float4 *)&C[m * N + n + 8] = *(const float4 *)(&C_reg[i][4]);
    }
  } else {
    // for (int i = 0; i < thread_tile; ++i) {
    //   const int m = C_m_top + C_sub_m + i;
    //   for (int j = 0; j < thread_tile; ++j) {
    //     const int n = C_n_left + C_sub_n + (j < 4 ? j : (32 + j - 4));
    //     if (m < M && n < N) {
    //       C[m * N + n] = C_reg[i][j];
    //     }
    //   }
    // }
  }
}

// A, B and C are all row-marjor matrices;
template <int block_tile, int thread_tile, int k_per_iter>
__global__ void
shared_memory__eliminate_bank_conflict__global_store_memory_colacesing__double_buffer__overlap_sm_latency(
    const float *A, const float *B, float *C, int M, int N, int K) {
  extern __shared__ float shared_memory_buffer[];

  float *A_sm = shared_memory_buffer;
  float *B_sm = shared_memory_buffer + block_tile * k_per_iter * 2;

  const int k_iter_count = (K + k_per_iter - 1) / k_per_iter;

  // The current block needs to compute `block_tile * block_tile` floats in
  // `C`, spanning from `C_m_top` to `C_m_bottom`(exclusive) and from
  // `C_n_left` to `C_n_right`(exclusive).
  int C_m_top = blockIdx.y * block_tile;
  int C_n_left = blockIdx.x * block_tile;
  int C_m_bottom = min(C_m_top + block_tile, M);
  int C_n_right = min(C_n_left + block_tile, N);

  // `C_reg` is being used to store the partial sum.
  float C_reg[thread_tile][thread_tile] = {0};

  // The following code computes the relative position of the top-right float
  // in the `C` submatrix.
  const int lane_id = threadIdx.x % 32;
  const int warp_id = threadIdx.x / 32;
  const int half_warp_id = warp_id / 16;
  const int half_warp_lane_id = lane_id % 16;
  const int quarter_warp_id = warp_id / 8;
  const int quarter_warp_lane_id = lane_id % 8;

  // Every thread needs to compute `thread_tile * thread_tile` floats in the
  // `C` matrix.
  // A_reg is K-marjor; B_reg is N-marjor;
  float A_reg[thread_tile * 2];
  float B_reg[thread_tile * 2];

  // Each thread shifts `warp_id` position to the right within its own
  // half-warp.
  const int B_shifted_lane_id =
      ((lane_id + warp_id * 4) % 16 + (lane_id & 0x10)) ^ (warp_id / 4 * 16);

  const int A_sm2reg_offset_mask_0x1 =
      lane_id / 8 * 256 + (lane_id & 0xfe) % 8 * 4 + lane_id / 8 % 2 * 4;
  const int A_sm2reg_offset_or_0x1 =
      lane_id / 8 * 256 + (lane_id | 0x1) % 8 * 4 - lane_id / 8 % 2 * 4;

  const int B_sm2reg_offset =
      B_shifted_lane_id / 8 * 256 + B_shifted_lane_id % 8 / 4 * 16 +
      B_shifted_lane_id % 2 * 8 + B_shifted_lane_id / 8 % 2 * 4;

  const int A_sm_dst_offset =
      (lane_id & 0xf8) + (warp_id + lane_id) % 8 + lane_id % 8 * 32;
  const int B_sm_dst_offset =
      warp_id % 4 * 256 + warp_id / 4 * 32 + (lane_id & 0xf8);
  const int A_global_src_offset =
      (C_m_top + lane_id / 8 * 8 + warp_id) * K + lane_id % 8;
  const int B_global_src_offset =
      warp_id / 4 * N + C_n_left + warp_id % 4 * 32 + lane_id;

  // Read `block_tile * k_per_iter` floats from each of A and B in global
  // memory into shared memory per iteration.
  {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      float *A_sm_dst = A_sm + i * 256 + A_sm_dst_offset;
      const float *A_global_src = A + i * 32 * K + A_global_src_offset;
      float reg;
      asm volatile("ld.global.cg.f32 %0, [%1];"
                   : "=f"(reg)
                   : "l"(A_global_src));
      *A_sm_dst = reg;
    }
  }
  {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float *B_global_src = B + i * 2 * N + B_global_src_offset;
      float reg;
      asm volatile("ld.global.cg.f32 %0, [%1];"
                   : "=f"(reg)
                   : "l"(B_global_src));
      float *B_sm_dst = B_sm + i * 2 * 32 +
                        (lane_id + i * 2 + warp_id / 4) % 8 + B_sm_dst_offset;
      *B_sm_dst = reg;
    }
  }

  for (int iter = 0; iter < k_iter_count; ++iter) {
    // Read `block_tile * k_per_iter` floats from each of A and B in global
    // memory into shared memory per iteration.
    const int shared_memory_computing_offset =
        iter % 2 * block_tile * k_per_iter;
    __syncthreads();
    if (iter + 1 < k_iter_count) {
      const int shared_memory_loading_offset =
          (iter + 1) % 2 * block_tile * k_per_iter;
      {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          float *A_sm_dst =
              A_sm + shared_memory_loading_offset + i * 256 + A_sm_dst_offset;
          const float *A_global_src =
              A + (iter + 1) * k_per_iter + i * 32 * K + A_global_src_offset;
          float reg;
          asm volatile("ld.global.cg.f32 %0, [%1];"
                       : "=f"(reg)
                       : "l"(A_global_src));
          *A_sm_dst = reg;
        }
      }
      {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const float *B_global_src =
              B + ((iter + 1) * k_per_iter + i * 2) * N + B_global_src_offset;
          float reg;
          asm volatile("ld.global.cg.f32 %0, [%1];"
                       : "=f"(reg)
                       : "l"(B_global_src));
          float *B_sm_dst = B_sm + shared_memory_loading_offset + i * 2 * 32 +
                            (lane_id + i * 2 + warp_id / 4) % 8 +
                            B_sm_dst_offset;
          *B_sm_dst = reg;
        }
      }
    }
    {
      *(float4 *)&A_reg[0] =
          *(const float4 *)&A_sm[shared_memory_computing_offset +
                                 A_sm2reg_offset_mask_0x1];
      *(float4 *)&A_reg[4] =
          *(const float4
                *)&A_sm[shared_memory_computing_offset + A_sm2reg_offset_or_0x1];
      *(float4 *)&B_reg[0] =
          *(const float4 *)&B_sm[shared_memory_computing_offset + B_sm2reg_offset];
      *(float4 *)&B_reg[4] =
          *(const float4
                *)&B_sm[shared_memory_computing_offset + (B_sm2reg_offset ^ 0x4)];
    }
#pragma unroll
    for (int k = 0; k < k_per_iter; ++k) {
      if (k + 1 < k_per_iter) {
        const int reg_loading_offset = (k + 1) % 2 * thread_tile;
        {
          *(float4 *)&A_reg[0 + reg_loading_offset] =
              *(const float4 *)&A_sm[shared_memory_computing_offset +
                                     (k + 1) * 32 + A_sm2reg_offset_mask_0x1];
          *(float4 *)&A_reg[4 + reg_loading_offset] =
              *(const float4 *)&A_sm[shared_memory_computing_offset +
                                     (k + 1) * 32 + A_sm2reg_offset_or_0x1];
          *(float4 *)&B_reg[0 + reg_loading_offset] =
              *(const float4 *)&B_sm[shared_memory_computing_offset +
                                     (k + 1) * 32 + B_sm2reg_offset];
          *(float4 *)&B_reg[4 + reg_loading_offset] =
              *(const float4 *)&B_sm[shared_memory_computing_offset +
                                     (k + 1) * 32 + (B_sm2reg_offset ^ 0x4)];
        }
      }
      const int reg_computing_offset = k % 2 * thread_tile;
#pragma unroll
      for (int i = 0; i < thread_tile; ++i) {
#pragma unroll
        for (int j = 0; j < thread_tile; ++j) {
          C_reg[i][j] += A_reg[(i + k) % thread_tile + reg_computing_offset] *
                         B_reg[(j + k) % thread_tile + reg_computing_offset];
        }
      }
    }
  }

  // C_reg is N-marjor
  if (C_m_top + block_tile <= M && C_n_left + block_tile <= N) {
#pragma unroll
    for (int i = 0; i < thread_tile; ++i) {
      int m =
          C_m_top + lane_id / 2 * 8 + (i + lane_id / 8 % 2 * 4) % thread_tile;
      int n = C_n_left + B_shifted_lane_id / 4 * 16 + B_shifted_lane_id % 2 * 8;
      *(float4 *)&C[m * N + n + B_shifted_lane_id / 8 % 2 * 4] =
          *(const float4 *)(&C_reg[i][0]);
      *(float4 *)&C[m * N + n + 4 - B_shifted_lane_id / 8 % 2 * 4] =
          *(const float4 *)(&C_reg[i][4]);
    }
  } else {
    // for (int i = 0; i < thread_tile; ++i) {
    //   const int m = C_m_top + C_sub_m + i;
    //   for (int j = 0; j < thread_tile; ++j) {
    //     const int n = C_n_left + C_sub_n + (j < 4 ? j : (32 + j - 4));
    //     if (m < M && n < N) {
    //       C[m * N + n] = C_reg[i][j];
    //     }
    //   }
    // }
  }
}

// A, B and C are all row-marjor matrices;
template <int block_tile, int thread_tile, int k_per_iter>
__global__ void
shared_memory__eliminate_bank_conflict__global_store_memory_colacesing__double_buffer(
    const float *A, const float *B, float *C, int M, int N, int K) {
  extern __shared__ float shared_memory_buffer[];

  float *A_sm = shared_memory_buffer;
  float *B_sm = shared_memory_buffer + block_tile * k_per_iter * 2;

  const int k_iter_count = (K + k_per_iter - 1) / k_per_iter;

  // The current block needs to compute `block_tile * block_tile` floats in
  // `C`, spanning from `C_m_top` to `C_m_bottom`(exclusive) and from
  // `C_n_left` to `C_n_right`(exclusive).
  int C_m_top = blockIdx.y * block_tile;
  int C_n_left = blockIdx.x * block_tile;
  int C_m_bottom = min(C_m_top + block_tile, M);
  int C_n_right = min(C_n_left + block_tile, N);

  // `C_reg` is being used to store the partial sum.
  float C_reg[thread_tile][thread_tile] = {0};

  // The following code computes the relative position of the top-right float
  // in the `C` submatrix.
  const int lane_id = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;

  // Every thread needs to compute `thread_tile * thread_tile` floats in the
  // `C` matrix.
  // A_reg is K-marjor; B_reg is N-marjor;
  const int k_per_iter_half = k_per_iter / 2;
  float A_reg[thread_tile * k_per_iter_half];
  float B_reg[thread_tile * k_per_iter_half];
  const int A_offset = lane_id / 16 * 64 + lane_id % 16 / 2 * 4;

  // Each thread shifts `warp_id` position to the right within its own
  // half-warp.
  const int B_shifted_lane_id =
      ((lane_id + warp_id * 4) % 16 + (lane_id & 0x10)) ^ (warp_id / 4 * 16);
  const int B_offset = B_shifted_lane_id / 16 * 64 +
                       ((B_shifted_lane_id % 16) & 0xfd) / 4 * 8 +
                       (B_shifted_lane_id & 1) * 4;

  // Read `block_tile * k_per_iter` floats from each of A and B in global
  // memory into shared memory per iteration.
  {
    *(float4 *)(&A_sm[warp_id * block_tile + lane_id * 4]) = __ldg(
        (const float4
             *)(&A[(C_m_top + warp_id * 16 + lane_id % 8 + lane_id / 16 * 8) *
                       K +
                   lane_id % 16 / 8 * 4]));
  }
  {
    *(float4 *)(&B_sm[warp_id * block_tile + lane_id * 4]) =
        __ldg((const float4 *)(&B[warp_id * N + C_n_left +
                                  lane_id % 8 / 2 * 16 + (lane_id & 1) * 4 +
                                  (lane_id & 0x8) + lane_id / 16 * 64]));
  }

  for (int iter = 0; iter < k_iter_count; ++iter) {
    // Read `block_tile * k_per_iter` floats from each of A and B in global
    // memory into shared memory per iteration.
    const int shared_memory_computing_offset =
        iter % 2 * block_tile * k_per_iter;
    __syncthreads();
    if (iter + 1 < k_iter_count) {
      const int shared_memory_loading_offset =
          (iter + 1) % 2 * block_tile * k_per_iter;
      {
        *(float4 *)(&A_sm[warp_id * block_tile + lane_id * 4 +
                          shared_memory_loading_offset]) =
            __ldg((const float4
                       *)(&A[(C_m_top + warp_id * 16 + lane_id % 8 +
                              lane_id / 16 * 8) *
                                 K +
                             (iter + 1) * k_per_iter + lane_id % 16 / 8 * 4]));
      }
      {
        *(float4 *)(&B_sm[warp_id * block_tile + lane_id * 4 +
                          shared_memory_loading_offset]) =
            __ldg((const float4 *)(&B[((iter + 1) * k_per_iter + warp_id) * N +
                                      C_n_left + lane_id % 8 / 2 * 16 +
                                      (lane_id & 1) * 4 + (lane_id & 0x8) +
                                      lane_id / 16 * 64]));
      }
    }

    for (int i = 0; i < thread_tile; ++i) {
      *(float4 *)&A_reg[i * k_per_iter_half] =
          *(float4 *)&A_sm[shared_memory_computing_offset + A_offset + i * 128];
    }
    for (int i = 0; i < k_per_iter_half; ++i) {
      *(float4 *)&B_reg[i * thread_tile] =
          *(float4 *)&B_sm[shared_memory_computing_offset + B_offset + i * 128];
      *(float4 *)&B_reg[i * thread_tile + 4] =
          *(float4 *)&B_sm[shared_memory_computing_offset + B_offset + i * 128 +
                           32];
    }

    for (int i = 0; i < thread_tile; ++i) {
      for (int j = 0; j < thread_tile; ++j) {
        for (int k = 0; k < k_per_iter_half; ++k) {
          C_reg[i][j] +=
              A_reg[i * k_per_iter_half + k] * B_reg[k * thread_tile + j];
        }
      }
    }

    for (int i = 0; i < thread_tile; ++i) {
      *(float4 *)&A_reg[i * k_per_iter_half] =
          *(float4 *)&A_sm[shared_memory_computing_offset + A_offset + i * 128 +
                           32];
    }
    for (int i = 4; i < k_per_iter; ++i) {
      *(float4 *)&B_reg[(i - 4) * thread_tile] =
          *(float4 *)&B_sm[shared_memory_computing_offset + B_offset + i * 128];
      *(float4 *)&B_reg[(i - 4) * thread_tile + 4] =
          *(float4 *)&B_sm[shared_memory_computing_offset + B_offset + i * 128 +
                           32];
    }

    for (int i = 0; i < thread_tile; ++i) {
      for (int j = 0; j < thread_tile; ++j) {
        for (int k = 0; k < k_per_iter_half; ++k) {
          C_reg[i][j] +=
              A_reg[i * k_per_iter_half + k] * B_reg[k * thread_tile + j];
        }
      }
    }
  }

  // C_reg is N-marjor
  if (C_m_top + block_tile <= M && C_n_left + block_tile <= N) {
    for (int i = 0; i < thread_tile; ++i) {
      int m = C_m_top + i * 16 + lane_id / 16 * 8 + lane_id % 16 / 2;
      int n = C_n_left + B_shifted_lane_id % 16 / 4 * 16 +
              B_shifted_lane_id / 16 * 64 + (B_shifted_lane_id & 1) * 4;
      *(float4 *)&C[m * N + n] = *(const float4 *)(&C_reg[i][0]);
      *(float4 *)&C[m * N + n + 8] = *(const float4 *)(&C_reg[i][4]);
    }
  } else {
    // for (int i = 0; i < thread_tile; ++i) {
    //   const int m = C_m_top + C_sub_m + i;
    //   for (int j = 0; j < thread_tile; ++j) {
    //     const int n = C_n_left + C_sub_n + (j < 4 ? j : (32 + j - 4));
    //     if (m < M && n < N) {
    //       C[m * N + n] = C_reg[i][j];
    //     }
    //   }
    // }
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
void launch_shared_memory__eliminate_bank_conflict(const float *A,
                                                   const float *B, float *C,
                                                   int M, int N, int K) {
  constexpr int THREAD_TILE = 8;
  static_assert(BLOCK_TILE % THREAD_TILE == 0);
  constexpr dim3 block(BLOCK_TILE / THREAD_TILE, BLOCK_TILE / THREAD_TILE);
  constexpr int k_per_iter = 8;
  const dim3 grid_dim((N + BLOCK_TILE - 1) / BLOCK_TILE,
                      (M + BLOCK_TILE - 1) / BLOCK_TILE);
  constexpr dim3 block_dim(BLOCK_TILE / THREAD_TILE *
                           (BLOCK_TILE / THREAD_TILE));

  const int32_t shared_memory_bytes =
      k_per_iter * BLOCK_TILE * sizeof(float) * 2;

  static_assert(BLOCK_TILE * k_per_iter % (block_dim.x * block_dim.y) == 0);
  static_assert(THREAD_TILE == 8);

  // shared_memory__eliminate_bank_conflict_v0<BLOCK_TILE, THREAD_TILE,
  // k_per_iter>
  //     <<<grid_dim, block_dim, shared_memory_bytes>>>(A, B, C, M, N, K);
  // CHECK_CUDA_ERROR();

  // shared_memory__eliminate_bank_conflict_v1<BLOCK_TILE, THREAD_TILE,
  // k_per_iter>
  //     <<<grid_dim, block_dim, shared_memory_bytes>>>(A, B, C, M, N, K);
  // CHECK_CUDA_ERROR();

  // shared_memory__eliminate_bank_conflict_v2<BLOCK_TILE, THREAD_TILE,
  // k_per_iter>
  //     <<<grid_dim, block_dim, shared_memory_bytes>>>(A, B, C, M, N, K);
  // CHECK_CUDA_ERROR();

  // shared_memory__eliminate_bank_conflict_v3<BLOCK_TILE, THREAD_TILE,
  // k_per_iter>
  //     <<<grid_dim, block_dim, shared_memory_bytes>>>(A, B, C, M, N, K);
  // CHECK_CUDA_ERROR();

  if (M % 128 == 0 && N % 128 == 0 && K % 8 == 0) {
    shared_memory__eliminate_bank_conflict__global_store_memory_colacesing__double_buffer<
        BLOCK_TILE, THREAD_TILE, k_per_iter>
        <<<grid_dim, block_dim, shared_memory_bytes * 2>>>(A, B, C, M, N, K);
    CHECK_CUDA_ERROR();
    shared_memory__eliminate_bank_conflict__global_store_memory_colacesing__pipeline_primitive_single_stage__overlap_sm_latency<
        BLOCK_TILE, THREAD_TILE, k_per_iter>
        <<<grid_dim, block_dim, shared_memory_bytes>>>(A, B, C, M, N, K);
    CHECK_CUDA_ERROR();
    shared_memory__eliminate_bank_conflict__global_store_memory_colacesing__pipeline_primitive_single_stage<
        BLOCK_TILE, THREAD_TILE, k_per_iter>
        <<<grid_dim, block_dim, shared_memory_bytes>>>(A, B, C, M, N, K);
    CHECK_CUDA_ERROR();
    for (int i = 0; i < 4; ++i) {
      shared_memory__eliminate_bank_conflict__global_store_memory_colacesing__double_buffer__overlap_sm_latency<
          BLOCK_TILE, THREAD_TILE, k_per_iter>
          <<<grid_dim, block_dim, shared_memory_bytes * 2>>>(A, B, C, M, N, K);
      CHECK_CUDA_ERROR();
    }
    // shared_memory__eliminate_bank_conflict__global_store_memory_colacesing__pipeline_primitive_double_stage<
    //     BLOCK_TILE, THREAD_TILE, k_per_iter>
    //     <<<grid_dim, block_dim, shared_memory_bytes * 2>>>(A, B, C, M, N, K);
    // CHECK_CUDA_ERROR();
    // shared_memory__eliminate_bank_conflict__global_store_memory_colacesing__pipeline_primitive_single_stage__32_k_per_iter<<<
    //     grid_dim, block_dim, 128 * (32 + 8) * 4>>>(A, B, C, M, N, K);
    // CHECK_CUDA_ERROR();
    // shared_memory__eliminate_bank_conflict__global_store_memory_colacesing__pipeline_primitive_single_stage__32_k_per_iter_only_A<<<
    //     grid_dim, block_dim, 128 * (32 + 8) * 4>>>(A, B, C, M, N, K);
    // CHECK_CUDA_ERROR();
    // shared_memory__eliminate_bank_conflict__global_store_memory_colacesing__pipeline_primitive_single_stage__32_k_per_iter_only_B<<<
    //     grid_dim, block_dim, 128 * (32 + 8) * 4>>>(A, B, C, M, N, K);
    // CHECK_CUDA_ERROR();
    // shared_memory__eliminate_bank_conflict__global_store_memory_colacesing__pipeline_primitive_double_stage<
    //     BLOCK_TILE, THREAD_TILE, k_per_iter>
    //     <<<grid_dim, block_dim, shared_memory_bytes * 2>>>(A, B, C, M, N, K);
    // CHECK_CUDA_ERROR();
    // shared_memory__eliminate_bank_conflict__global_store_memory_colacesing__pipeline_2_stage__only_A<
    //     BLOCK_TILE, THREAD_TILE, k_per_iter>
    //     <<<grid_dim, block_dim, shared_memory_bytes * 2>>>(A, B, C, M, N, K);
    // CHECK_CUDA_ERROR();
    // shared_memory__eliminate_bank_conflict__global_store_memory_colacesing__pipeline_2_stage__only_B<
    //     BLOCK_TILE, THREAD_TILE, k_per_iter>
    //     <<<grid_dim, block_dim, shared_memory_bytes * 2>>>(A, B, C, M, N, K);
    // CHECK_CUDA_ERROR();
    // shared_memory__eliminate_bank_conflict__global_store_memory_colacesing__pipeline_2_stage<
    //     BLOCK_TILE, THREAD_TILE, k_per_iter>
    //     <<<grid_dim, block_dim, shared_memory_bytes * 2>>>(A, B, C, M, N, K);
    // CHECK_CUDA_ERROR();
    // shared_memory__eliminate_bank_conflict_v4<BLOCK_TILE, THREAD_TILE,
    //                                           k_per_iter>
    //     <<<grid_dim, block_dim, shared_memory_bytes>>>(A, B, C, M, N, K);
    // CHECK_CUDA_ERROR();
    // shared_memory__eliminate_bank_conflict__global_store_memory_colacesing<
    //     BLOCK_TILE, THREAD_TILE, k_per_iter>
    //     <<<grid_dim, block_dim, shared_memory_bytes>>>(A, B, C, M, N, K);
    // CHECK_CUDA_ERROR();
    // shared_memory__eliminate_bank_conflict__global_store_memory_colacesing__less_register<
    //     BLOCK_TILE, THREAD_TILE, k_per_iter>
    //     <<<grid_dim, block_dim, shared_memory_bytes>>>(A, B, C, M, N, K);
    // CHECK_CUDA_ERROR();
  } else {
    printf("Skipping shared_memory__eliminate_bank_conflict_v4 because M or N or K are not suitable");
  }
}
