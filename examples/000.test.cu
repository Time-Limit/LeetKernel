#include "util/error.h"
#include "util/util.cuh"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

__global__ void test_ld_global_and_st_shared(const float* dram, size_t size)
{
  __shared__ float sm[1024];
  for (int i = 0; i < size && i < 1024; i += blockDim.x) {
    if (i + threadIdx.x < 1024) {
      sm[i + threadIdx.x] = dram[i + threadIdx.x];
    }
  }
  if (threadIdx.x == 0) {
    float sum = 0;
    for (int i = 0; i < 1024 && i < size; ++i) {
      sum += sm[i];
    }
    printf("sum = %8.3f\n", sum);
  }
}


__global__ void test_mma_m8n8k4()
{
  float c[8] = {0};
  half  a[4] = {0};
  half  b[4] = {threadIdx.x * 100 + 0, threadIdx.x * 100 + 1, threadIdx.x * 100 + 2, threadIdx.x * 100 + 3};
  if (threadIdx.x == 0 || threadIdx.x == 16) {
    a[0] = 1;
  }
  if (threadIdx.x == 1 || threadIdx.x == 17) {
    a[1] = 1;
  }
  if (threadIdx.x == 2 || threadIdx.x == 18) {
    a[2] = 1;
  }
  if (threadIdx.x == 3 || threadIdx.x == 19) {
    a[3] = 1;
  }
  uint32_t const* A    = reinterpret_cast<uint32_t const*>(&a);
  uint32_t const* B    = reinterpret_cast<uint32_t const*>(&b);
  // clang-format off
  asm volatile(
      "mma.sync.aligned.m8n8k4.row.row.f32.f16.f16.f32"
      "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11},"
      "{%12, %13, %14, %15, %16, %17, %18, %19};"
      : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3]), "=f"(c[4]), "=f"(c[5]), "=f"(c[6]), "=f"(c[7])
      : "r"(A[0]), "r"(A[1]),
        "r"(B[0]), "r"(B[1]),
        "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]), "f"(c[4]), "f"(c[5]), "f"(c[6]), "f"(c[7]));
  // clang-format on
  for (auto i : {0, 1, 2, 3, 16, 17, 18, 19}) {
    if (this_thread_can_log(i)) {
      printf("threadIdx.x = %03d, %8.3f, %8.3f, %8.3f, %8.3f, %8.3f, %8.3f, %8.3f, %8.3f\n",
             threadIdx.x,
             float(c[0]),
             float(c[1]),
             float(c[2]),
             float(c[3]),
             float(c[4]),
             float(c[5]),
             float(c[6]),
             float(c[7]));
    }
    __syncthreads();
  }
}

__inline__ __device__ void mma_m16n8k16_row_col(float (&d)[4], const half (&a)[8], const half (&b)[4], float (&c)[4])
{
  uint32_t const* A = reinterpret_cast<uint32_t const*>(&a);
  uint32_t const* B = reinterpret_cast<uint32_t const*>(&b);
  float const*    C = reinterpret_cast<float const*>(&c);
  float*          D = reinterpret_cast<float*>(&d);
  asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32  {%0,%1,%2,%3}, "
    "{%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
    : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
    : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));
}

__global__ void test_mma_m16n8k16_row_col()
{
  int             lane_id          = threadIdx.x % 32;
  int             group_id         = threadIdx.x / 4;
  int             lane_id_in_group = lane_id % 4;
  float           d[4]             = {0, 0, 0, 0};
  half            a[8]             = {0, 0, 0, 0, 0, 0, 0, 0};
  half            b[4]             = {0, 0, 0, 0};
  float           c[4]             = {0, 0, 0, 0};

  a[group_id & 1]       = half(group_id == (lane_id_in_group * 2 + (group_id & 1)) ? 1 : 0);
  a[6 + (group_id & 1)] = half(group_id == (lane_id_in_group * 2 + (group_id & 1)) ? 1 : 0);
  b[0]                  = half(lane_id_in_group * 2 * 8 + group_id);
  b[1]                  = half((lane_id_in_group * 2 + 1) * 8 + group_id);
  b[2]                  = half((lane_id_in_group * 2 + 8) * 8 + group_id);
  b[3]                  = half((lane_id_in_group * 2 + 9) * 8 + group_id);
  mma_m16n8k16_row_col(d, a, b, c);
  for (int i = 0; i < 32; ++i) {
    if (threadIdx.x == i) {
      printf("threadIdx.x = %03d, a = %f, %f, %f, %f, %f, %f, %f, %f\n",
             threadIdx.x,
             float(a[0]),
             float(a[1]),
             float(a[2]),
             float(a[3]),
             float(a[4]),
             float(a[5]),
             float(a[6]),
             float(a[7]));
    }
    __syncthreads();
  }
  for (int i = 0; i < 32; ++i) {
    if (threadIdx.x == i) {
      printf("threadIdx.x = %03d, b = k%03dn%03d, k%03dn%03d, k%03dn%03d, k%03dn%03d\n",
             threadIdx.x,
             int(b[0]) / 8,
             int(b[0]) % 8,
             int(b[1]) / 8,
             int(b[1]) % 8,
             int(b[2]) / 8,
             int(b[2]) % 8,
             int(b[3]) / 8,
             int(b[3]) % 8);
    }
    __syncthreads();
  }
  for (int i = 0; i < 32; ++i) {
    if (threadIdx.x == i) {
      printf("threadIdx.x = %03d, d = k%03dn%03d, k%03dn%03d, k%03dn%03d, k%03dn%03d\n",
             threadIdx.x,
             int(d[0]) / 8,
             int(d[0]) % 8,
             int(d[1]) / 8,
             int(d[1]) % 8,
             int(d[2]) / 8,
             int(d[2]) % 8,
             int(d[3]) / 8,
             int(d[3]) % 8);
    }
    __syncthreads();
  }
}

void launch_test_mma_m16n8k16_row_col() {
  test_mma_m16n8k16_row_col<<<dim3(1), dim3(32)>>>();
}

int main()
{
  float* dram;
  CHECK_CUDA_RETURN(cudaMalloc(&dram, 1024 * sizeof(float)));
  test_ld_global_and_st_shared<<<1, 32>>>(dram, 1024);
  CHECK_CUDA_ERROR();
  test_mma_m8n8k4<<<1, 32>>>();
  CHECK_CUDA_ERROR();
  launch_test_mma_m16n8k16_row_col();
  CHECK_CUDA_ERROR();
  return 0;
}
