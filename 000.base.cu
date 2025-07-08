#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <random>

#include "util/error.h"

__global__ void dim1_load_float(const float *A, float *B, int N) {
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index < N) {
    B[index] = A[index];
  }
}

void test_dim1_load_float() {
  const int ALLOC_N = (1 << 20);
  const int REAL_N = (1 << 20);
  float *A, *B;
  cudaMalloc(&A, sizeof(float) * ALLOC_N);
  cudaMalloc(&B, sizeof(float) * ALLOC_N);

  dim3 threads_per_block(64);
  dim3 blocks_per_grid(REAL_N / 64);

  printf("A = %p, B = %p\n", A, B);

  dim1_load_float<<<blocks_per_grid, threads_per_block>>>(A, B, REAL_N);
  CHECK_CUDA_ERROR();

  cudaFree(A);
  cudaFree(B);
}

__global__ void dim2_load_float_x_major(const float *A, float *B, int N) {
  int index = blockIdx.x * blockDim.x * blockDim.y + threadIdx.x * blockDim.y +
              threadIdx.y;
  if (index < N) {
    B[index] = A[index];
  }
}

__global__ void dim2_load_float_y_major(const float *A, float *B, int N) {
  int index = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
              threadIdx.x;
  if (index < N) {
    B[index] = A[index];
  }
}

void test_dim2_load_float() {
  const int ALLOC_N = (1 << 20);
  const int REAL_N = (1 << 20);
  float *A, *B;
  cudaMalloc(&A, sizeof(float) * ALLOC_N);
  cudaMalloc(&B, sizeof(float) * ALLOC_N);

  dim3 threads_per_block(16, 16);
  dim3 blocks_per_grid(REAL_N / (16 * 16));

  printf("A = %p, B = %p\n", A, B);

  dim2_load_float_x_major<<<blocks_per_grid, threads_per_block>>>(A, B, REAL_N);
  dim2_load_float_y_major<<<blocks_per_grid, threads_per_block>>>(A, B, REAL_N);
  CHECK_CUDA_ERROR();

  cudaFree(A);
  cudaFree(B);
}

__device__ void init_shared_memory(const int *global, int *sm, int size) {
  if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 &&
      threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    for (int i = 0; i < size; ++i) {
      sm[i] = global[i];
    }
  }
  __syncthreads();
}

__global__ void shared_memory_instructions__lds32(const int *input,
                                                  int *result) {
  extern __shared__ int shared_memory[];
  init_shared_memory(input, shared_memory, 10240 / 4);
  int sum = 0;
  int iter = 10240 / 4 / 32;
  for (int loop = 0; loop < 512; ++loop) {
    for (int i = 0; i < iter; ++i) {
      const int offset = (i + loop) % iter * blockDim.x + threadIdx.x;
      sum += shared_memory[offset];
      sum += shared_memory[offset ^ 1];
      sum += shared_memory[offset ^ 2];
      sum += shared_memory[offset ^ 3];
    }
  }
  for (int i = 0; i < 32; ++i) {
    if (i == threadIdx.x) {
      *result += sum;
    }
    __syncthreads();
  }
}

__global__ void shared_memory_instructions__lds64(const int *input, int *result) {
  extern __shared__ int shared_memory[];
  init_shared_memory(input, shared_memory, 10240 / 4);
  int sum = 0;
  int iter = 10240 / 4 / 32;
  for (int loop = 0; loop < 512; ++loop) {
    for (int i = 0; i < iter; ++i) {
      const int offset =
          (i + loop) % iter * blockDim.x + (threadIdx.x & 0xfffffffe);
      {
        int2 data = *reinterpret_cast<int2 *>(&shared_memory[offset]);
        sum += data.x + data.y;
      }
      {
        int2 data = *reinterpret_cast<int2 *>(&shared_memory[offset ^ 2]);
        sum += data.x + data.y;
      }
    }
  }
  for (int i = 0; i < 32; ++i) {
    if (i == threadIdx.x) {
      *result += sum;
    }
    __syncthreads();
  }
}

__global__ void shared_memory_instructions__lds64__uniform_datapath(const int *input, int *result) {
  extern __shared__ int shared_memory[];
  init_shared_memory(input, shared_memory, 10240 / 4);
  int sum = 0;
  int iter = 10240 / 4 / 32 / 2;
  for (int loop = 0; loop < 2048; ++loop) {
    for (int i = 0; i < iter; ++i) {
      const int offset = ((i + loop) % iter * blockDim.x + threadIdx.x) * 2;
      int2 data = *reinterpret_cast<int2 *>(&shared_memory[offset]);
      sum += data.x + data.y;
    }
  }
  for (int i = 0; i < 32; ++i) {
    if (i == threadIdx.x) {
      *result += sum;
    }
    __syncthreads();
  }
}

__global__ void shared_memory_instructions__lds128(const int *input, int *result) {
  extern __shared__ int shared_memory[];
  init_shared_memory(input, shared_memory, 10240 / 4);
  int sum = 0;
  int iter = 10240 / 4 / 32;
  for (int loop = 0; loop < 512; ++loop) {
    for (int i = 0; i < iter; ++i) {
      const int base =
          ((i + loop) % iter * blockDim.x + threadIdx.x) & 0xfffffffc;
      int4 data = *reinterpret_cast<int4 *>(&shared_memory[base]);
      sum += data.x + data.y + data.z + data.w;
    }
  }
  for (int i = 0; i < 32; ++i) {
    if (i == threadIdx.x) {
      *result += sum;
    }
    __syncthreads();
  }
}

__global__ void shared_memory_instructions__lds128__uniform_datapath(const int *input, int *result) {
  extern __shared__ int shared_memory[];
  init_shared_memory(input, shared_memory, 10240 / 4);
  int sum = 0;
  constexpr int iter = 10240 / 4 / 32 / 4;
  for (int loop = 0; loop < 2048; ++loop) {
    for (int i = 0; i < iter; ++i) {
      const int base = (((i + loop) % iter) * blockDim.x + threadIdx.x) * 4;
      int4 data = *reinterpret_cast<int4 *>(&shared_memory[base]);
      sum += data.x + data.y + data.z + data.w;
    }
  }
  for (int i = 0; i < 32; ++i) {
    if (i == threadIdx.x) {
      *result += sum;
    }
    __syncthreads();
  }
}

__global__ void shared_memory_instructions__lds128__reuse_offset(const int * input, int *result) {
  extern __shared__ int shared_memory[];
  init_shared_memory(input, shared_memory, 10240 / 4);
  int sum = 0;
  int iter = 10240 / 4 / 32;
  for (int loop = 0; loop < 1024; loop += 2) {
    for (int i = 0; i < iter; i += 2) {
      const int offset =
          ((i + loop) % iter * blockDim.x + threadIdx.x) & 0xfffffffc;
      {
        int4 data = *reinterpret_cast<int4 *>(&shared_memory[offset]);
        sum += data.x + data.y + data.z + data.w;
      }
      {
        int4 data =
            *reinterpret_cast<int4 *>(&shared_memory[offset + blockDim.x]);
        sum += data.x + data.y + data.z + data.w;
      }
    }
  }
  for (int i = 0; i < 32; ++i) {
    if (i == threadIdx.x) {
      *result += sum;
    }
    __syncthreads();
  }
}

__global__ void
shared_memory_instructions__lds128__bank_conflict(const int *input,
                                                  int *result) {
  extern __shared__ int shared_memory[];
  init_shared_memory(input, shared_memory, 10240 / 4);
  int sum = 0;
  int iter = 10240 / 4 / 32;
  for (int loop = 0; loop < 512; loop += 1) {
    // In shared memory access, thread i and thread i^1 will cause a bank
    // conflict.
    for (int i = 0; i < iter; i += 2) {
      int4 data = *reinterpret_cast<int4 *>(
          &shared_memory[((i + (threadIdx.x & 1) + loop) % iter * blockDim.x +
                          threadIdx.x) &
                         0xfffffffc]);
      sum += data.x + data.y + data.z + data.w;
    }
    for (int i = 0; i < iter; i += 2) {
      int4 data = *reinterpret_cast<int4 *>(
          &shared_memory[((i + (threadIdx.x & 1 ^ 1) + loop) % iter *
                              blockDim.x +
                          threadIdx.x) &
                         0xfffffffc]);
      sum += data.x + data.y + data.z + data.w;
    }
  }
  for (int i = 0; i < 32; ++i) {
    if (i == threadIdx.x) {
      *result += sum;
    }
    __syncthreads();
  }
}

__global__ void shared_memory_instructions__lds128__bank_conflict__reuse_offset(
    const int *input, int *result) {
  extern __shared__ int shared_memory[];
  init_shared_memory(input, shared_memory, 10240 / 4);
  int sum = 0;
  int iter = 10240 / 4 / 32;
  for (int loop = 0; loop < 1024; loop += 2) {
    // In shared memory access, thread i and thread i^1 will cause a bank
    // conflict.
    for (int i = 0; i < iter; i += 2) {
      int offset = ((i + loop) % iter * blockDim.x + threadIdx.x) & 0xfffffffc;
      {
        int4 data = *reinterpret_cast<int4 *>(
            &shared_memory[offset + (threadIdx.x & 1) * blockDim.x]);
        sum += data.x + data.y + data.z + data.w;
      }
      {
        int4 data = *reinterpret_cast<int4 *>(
            &shared_memory[offset + (threadIdx.x & 1 ^ 1) * blockDim.x]);
        sum += data.x + data.y + data.z + data.w;
      }
    }
  }
  for (int i = 0; i < 32; ++i) {
    if (i == threadIdx.x) {
      *result += sum;
    }
    __syncthreads();
  }
}

void test_shared_memory_instructions() {
  // On 4090, Each thread can have at most 255 registers, each register can
  // store one float. Each block can have at most 15360 bytes of shared memory.
  int *input;
  cudaMalloc(&input, sizeof(int) * 10240);
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::mt19937 generator(seed);
  std::uniform_int_distribution<> distribution(1, 100);
  std::vector<int> host_input(10240);
  for (auto &data : host_input) {
    data = distribution(generator);
  }
  cudaMemcpy(input, host_input.data(), sizeof(int) * host_input.size(),
             cudaMemcpyDefault);
  int *result;
  int host_result;
  cudaMalloc(&result, sizeof(int));

  host_result = 0;
  cudaMemcpy(result, &host_result, sizeof(float), cudaMemcpyDefault);
  shared_memory_instructions__lds32<<<1, 32, 10240>>>(input, result);
  CHECK_CUDA_ERROR();
  cudaMemcpy(&host_result, result, sizeof(float), cudaMemcpyDefault);
  printf("host_result = %d\n", host_result);

  host_result = 0;
  cudaMemcpy(result, &host_result, sizeof(float), cudaMemcpyDefault);
  shared_memory_instructions__lds64<<<1, 32, 10240>>>(input, result);
  CHECK_CUDA_ERROR();
  cudaMemcpy(&host_result, result, sizeof(float), cudaMemcpyDefault);
  printf("host_result = %d\n", host_result);

  host_result = 0;
  cudaMemcpy(result, &host_result, sizeof(float), cudaMemcpyDefault);
  shared_memory_instructions__lds64__uniform_datapath<<<1, 32, 10240>>>(input,
                                                                        result);
  CHECK_CUDA_ERROR();
  cudaMemcpy(&host_result, result, sizeof(float), cudaMemcpyDefault);
  printf("host_result = %d\n", host_result);

  host_result = 0;
  cudaMemcpy(result, &host_result, sizeof(float), cudaMemcpyDefault);
  shared_memory_instructions__lds128<<<1, 32, 10240>>>(input, result);
  CHECK_CUDA_ERROR();
  cudaMemcpy(&host_result, result, sizeof(float), cudaMemcpyDefault);
  printf("host_result = %d\n", host_result);

  host_result = 0;
  cudaMemcpy(result, &host_result, sizeof(float), cudaMemcpyDefault);
  shared_memory_instructions__lds128__uniform_datapath<<<1, 32, 10240>>>(
      input, result);
  CHECK_CUDA_ERROR();
  cudaMemcpy(&host_result, result, sizeof(float), cudaMemcpyDefault);
  printf("host_result = %d\n", host_result);

  host_result = 0;
  cudaMemcpy(result, &host_result, sizeof(float), cudaMemcpyDefault);
  shared_memory_instructions__lds128__reuse_offset<<<1, 32, 10240>>>(input, result);
  CHECK_CUDA_ERROR();
  cudaMemcpy(&host_result, result, sizeof(float), cudaMemcpyDefault);
  printf("host_result = %d\n", host_result);

  host_result = 0;
  cudaMemcpy(result, &host_result, sizeof(float), cudaMemcpyDefault);
  shared_memory_instructions__lds128__bank_conflict<<<1, 32, 10240>>>(input, result);
  CHECK_CUDA_ERROR();
  cudaMemcpy(&host_result, result, sizeof(float), cudaMemcpyDefault);
  printf("host_result = %d\n", host_result);

  host_result = 0;
  cudaMemcpy(result, &host_result, sizeof(float), cudaMemcpyDefault);
  shared_memory_instructions__lds128__bank_conflict__reuse_offset<<<1, 32,
                                                                    10240>>>(
      input, result);
  CHECK_CUDA_ERROR();
  cudaMemcpy(&host_result, result, sizeof(float), cudaMemcpyDefault);
  printf("host_result = %d\n", host_result);

  cudaFree(input);
  cudaFree(result);
}

__global__ void thread_layouts_in_sgemm_v0(const int *input, int *result) {
  /*
   *  Matrix B                                         
   *  NxK=64x8 floats, each thread loads eight floats, four threads load the same floats, repeat K times.                                     
   *  N00N01N02N03N32N33N34N35 -> T00T08T16T24                      
   *  N04N05N06N07N36N37N38N39 -> T01T09T10T11                      
   *  ...                                                            
   *  N28N29N30N31N60N61N62N63 -> T07T15T23T31                      
   *  ┌─────────────────────────────────┐            Matrix A                                                                                                                
   *  │ T00 T01 T02 T03 T04 T05 T06 T07 │            MxK=32x8 floats, each thread loads eight floats, eight threads load the same floats, repeat eight times.
   *  │ T08 T09 T10 T11 T12 T13 T14 T15 │            M00M01M02M03M04M05M06M07 -> T00T01T02T03T04T05T06T07 
   *  │ T16 T17 T18 T19 T20 T21 T22 T23 │            ...
   *  │ T24 T25 T26 T27 T28 T29 T30 T31 │            M24M25M26M27M28M29M30M31 -> T24T25T26T27T28T29T30T31
   *  └─────────────────────────────────┘            
   *                                                 
   */
  extern __shared__ int sm[];
  constexpr int M = 32;
  constexpr int N = 64;
  constexpr int K = 8;
  init_shared_memory(input, sm, (M + N) * K * 4 /*loops*/);
  // A is M-major.
  // M0K0, ..., M31K0, M0K1, ..., M31K1, ..., ..., M0K7, ..., M31K7
  const int *A_matrix = sm;
  // B is N-major.
  // K0N0, ..., K0N63, K1N0, ..., K1N63, ..., ..., K7N0, ..., K7N63
  const int *B_matrix = A_matrix + M * K;

  int sum = 0;
  int A_reg[8] = {0};
  int B_reg[8] = {0};

  for (int loop = 0; loop < 1024; ++loop) {
    const int *A = A_matrix + (loop & 3) * (M + N) * K;
    const int *B = B_matrix + (loop & 3) * (M + N) * K;
    for (int k = 0; k < 8; ++k) {
      {
        int offset = (threadIdx.x & 0xfffffff8) + k * M;
        *(int4 *)(&A_reg[0]) = *(const int4 *)(A + offset);
        *(int4 *)(&A_reg[4]) = *(const int4 *)(A + offset + 4);
      }
      {
        int offset = (threadIdx.x & 0x7 << 4) + k * N;
        *(int4 *)(&B_reg[0]) = *(const int4 *)(B + offset);
        *(int4 *)(&B_reg[4]) = *(const int4 *)(B + offset + 32);
      }
      for (int i = 0; i < 8; ++i) {
        sum += A_reg[i] + B_reg[i];
      }
    }
  }

  for (int i = 0; i < 32; ++i) {
    if (i == threadIdx.x) {
      *result += sum;
    }
    __syncthreads();
  }
}

__global__ void thread_layouts_in_sgemm_v1(const int *input, int *result) {
  /*
   *  Matrix B                                         
   *  NxK=128x8 floats, each thread loads four floats, repeat K times.                                     
   *  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐            
   *  │ T00 T01 T02 T03 T04 T05 T06 T07 T08 T09 T10 T11 T12 T13 T14 T15 T16 T17 T18 T19 T20 T21 T22 T23 T24 T25 T26 T27 T28 T29 T30 T31 │           
   *  └─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘            
   *  Matrix A
   *  M*K=16*8 floats, each thread loads sixteen floats, all threads load same floats, repeat K times.
   *                                                 
   */
  extern __shared__ int sm[];
  constexpr int M = 16;
  constexpr int N = 128;
  constexpr int K = 8;
  init_shared_memory(input, sm, (M + N) * K * 2 /*loops*/);
  // A is M-major.
  const int *A_matrix = sm;
  // B is N-major.
  const int *B_matrix = A_matrix + M * K;

  int sum = 0;
  int A_reg[16] = {0};
  int B_reg[4] = {0};

  for (int loop = 0; loop < 1024; ++loop) {
    const int *A = A_matrix + (loop & 1) * (M + N) * K;
    const int *B = B_matrix + (loop & 1) * (M + N) * K;
    for (int k = 0; k < 8; ++k) {
      {
        int offset = k * M;
        *(int4 *)(&A_reg[0]) = *(const int4 *)(A + offset);
        *(int4 *)(&A_reg[4]) = *(const int4 *)(A + offset + 4);
        *(int4 *)(&A_reg[8]) = *(const int4 *)(A + offset + 8);
        *(int4 *)(&A_reg[12]) = *(const int4 *)(A + offset + 12);
      }
      {
        int offset = (threadIdx.x * 4) + k * N;
        *(int4 *)(&B_reg[0]) = *(const int4 *)(B + offset);
      }
      for (int i = 0; i < 4; ++i) {
        sum += B_reg[i];
      }
      for (int i = 0; i < 16; ++i) {
        sum += A_reg[i];
      }
    }
  }

  for (int i = 0; i < 32; ++i) {
    if (i == threadIdx.x) {
      *result += sum;
    }
    __syncthreads();
  }
}

__global__ void thread_layouts_in_sgemm_v2(const int *input, int *result) {
  /*
   * v0 and v2 just have different thread layouts.
   *  ┌─────────────────────────────────┐            
   *  │ T00 T01 T08 T09 T16 T17 T24 T25 │ 
   *  │ T02 T03 T10 T11 T18 T19 T26 T27 │
   *  │ T04 T05 T12 T13 T20 T21 T28 T29 │
   *  │ T06 T07 T14 T15 T22 T23 T30 T31 │
   *  └─────────────────────────────────┘            
   */
  extern __shared__ int sm[];
  constexpr int M = 32;
  constexpr int N = 64;
  constexpr int K = 8;
  init_shared_memory(input, sm, (M + N) * K * 4 /*loops*/);
  // A is M-major.
  const int *A_matrix = sm;
  // B is N-major.
  const int *B_matrix = A_matrix + M * K;

  int sum = 0;
  int A_reg[8] = {0};
  int B_reg[8] = {0};

  for (int loop = 0; loop < 1024; ++loop) {
    const int *A = A_matrix + (loop & 3) * (M + N) * K;
    const int *B = B_matrix + (loop & 3) * (M + N) * K;
    for (int k = 0; k < 8; ++k) {
      {
        int offset = k * M + (threadIdx.x & 0x6 << 1);
        *(int4 *)(&A_reg[0]) = *(const int4 *)(A + offset);
        *(int4 *)(&A_reg[4]) = *(const int4 *)(A + offset + 16);
      }
      {
        int offset = (((threadIdx.x >> 3) + (threadIdx.x & 1)) << 2) + k * N;
        *(int4 *)(&B_reg[0]) = *(const int4 *)(B + offset);
        *(int4 *)(&B_reg[4]) = *(const int4 *)(B + offset + 32);
      }
      for (int i = 0; i < 8; ++i) {
        sum += A_reg[i] + B_reg[i];
      }
    }
  }

  for (int i = 0; i < 32; ++i) {
    if (i == threadIdx.x) {
      *result += sum;
    }
    __syncthreads();
  }
}

__global__ void thread_layouts_in_sgemm_v3(const int *input, int *result) {
  /*
   * V2 and V3 only differ in the relative position of loading and computing.
   */
  extern __shared__ int sm[];
  constexpr int M = 32;
  constexpr int N = 64;
  constexpr int K = 8;
  init_shared_memory(input, sm, (M + N) * K * 4 /*loops*/);
  // A is M-major.
  const int *A_matrix = sm;
  // B is N-major.
  const int *B_matrix = A_matrix + M * K;

  int sum = 0;
  int A_reg[8] = {0};
  int B_reg[8] = {0};

  for (int loop = 0; loop < 1024; ++loop) {
    const int *A = A_matrix + (loop & 3) * (M + N) * K;
    const int *B = B_matrix + (loop & 3) * (M + N) * K;
    for (int k = 0; k < 8; ++k) {
      int A_offset = k * M + (threadIdx.x & 0x6 << 1);
      *(int4 *)(&A_reg[0]) = *(const int4 *)(A + A_offset);
      int B_offset = (((threadIdx.x >> 3) + (threadIdx.x & 1)) << 2) + k * N;
      *(int4 *)(&B_reg[0]) = *(const int4 *)(B + B_offset);
      for (int i = 0; i < 4; ++i) {
        sum += A_reg[i] + B_reg[i];
      }
      *(int4 *)(&A_reg[4]) = *(const int4 *)(A + A_offset + 16);
      *(int4 *)(&B_reg[4]) = *(const int4 *)(B + B_offset + 32);
      for (int i = 4; i < 8; ++i) {
        sum += A_reg[i] + B_reg[i];
      }
    }
  }

  for (int i = 0; i < 32; ++i) {
    if (i == threadIdx.x) {
      *result += sum;
    }
    __syncthreads();
  }
}

void test_thread_layouts_in_sgemm() {
  // On 4090, Each thread can have at most 255 registers, each register can
  // store one float. Each block can have at most 15360 bytes of shared memory.

  // This function is being used to compare different threads-layouts' executed
  // duration. Each thread loads (M(32) + N(64)) * K (8) floats from shared
  // memory into registers.

  // All kernels called in this function only load floats but don't compute
  // them.
  const int shared_memory_bytes = (32 + 64) * 8 * 4 /*bytes*/ * 4 /*loops*/;
  int *input;
  cudaMalloc(&input, shared_memory_bytes);
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::mt19937 generator(seed);
  std::uniform_int_distribution<> distribution(1, 100);
  std::vector<int> host_input(shared_memory_bytes / 4);
  for (auto &data : host_input) {
    data = distribution(generator);
  }
  cudaMemcpy(input, host_input.data(), sizeof(int) * host_input.size(),
             cudaMemcpyDefault);

  int *result;
  int host_result;
  cudaMalloc(&result, sizeof(float));

  host_result = 0;
  cudaMemcpy(result, &host_result, sizeof(float), cudaMemcpyDefault);
  thread_layouts_in_sgemm_v0<<<1, 32, shared_memory_bytes>>>(input, result);
  CHECK_CUDA_ERROR();
  cudaMemcpy(&host_result, result, sizeof(float), cudaMemcpyDefault);
  printf("host_result = %d\n", host_result);

  host_result = 0;
  cudaMemcpy(result, &host_result, sizeof(float), cudaMemcpyDefault);
  thread_layouts_in_sgemm_v1<<<1, 32, shared_memory_bytes>>>(input, result);
  CHECK_CUDA_ERROR();
  cudaMemcpy(&host_result, result, sizeof(float), cudaMemcpyDefault);
  printf("host_result = %d\n", host_result);

  host_result = 0;
  cudaMemcpy(result, &host_result, sizeof(float), cudaMemcpyDefault);
  thread_layouts_in_sgemm_v2<<<1, 32, shared_memory_bytes>>>(input, result);
  CHECK_CUDA_ERROR();
  cudaMemcpy(&host_result, result, sizeof(float), cudaMemcpyDefault);
  printf("host_result = %d\n", host_result);

  host_result = 0;
  cudaMemcpy(result, &host_result, sizeof(float), cudaMemcpyDefault);
  thread_layouts_in_sgemm_v3<<<1, 32, shared_memory_bytes>>>(input, result);
  CHECK_CUDA_ERROR();
  cudaMemcpy(&host_result, result, sizeof(float), cudaMemcpyDefault);
  printf("host_result = %d\n", host_result);

  host_result = 0;
  cudaMemcpy(result, &host_result, sizeof(float), cudaMemcpyDefault);
  thread_layouts_in_sgemm_v4<<<1, 32, shared_memory_bytes>>>(input, result);
  CHECK_CUDA_ERROR();
  cudaMemcpy(&host_result, result, sizeof(float), cudaMemcpyDefault);
  printf("host_result = %d\n", host_result);

  cudaFree(result);
}

int main() {
  // test_dim1_load_float();
  // test_dim2_load_float();
  test_shared_memory_instructions();
  test_thread_layouts_in_sgemm();
  return 0;
}
