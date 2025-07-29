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

// __inline__ __device__ void
// mma_m8n8k4_row_col(Array<float, 8>& c, const Array<half, 4>& a, const Array<half, 4>& b, Array<float, 8>& c)
// {
// #if TURBOMIND_ARCH_SM70
//     uint32_t const* A = reinterpret_cast<uint32_t const*>(&a);
//     uint32_t const* B = reinterpret_cast<uint32_t const*>(&b);
//     // clang-format off
//     asm volatile(
//         "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32"
//         "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7},"
//         "{%8,  %9},"
//         "{%10, %11},"
//         "{%12, %13, %14, %15, %16, %17, %18, %19};"
//         : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3]), "=f"(c[4]), "=f"(d[5]), "=f"(d[6]), "=f"(d[7])
//         : "r"(A[0]), "r"(A[1]),
//           "r"(B[0]), "r"(B[1]),
//           "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]), "f"(c[4]), "f"(c[5]), "f"(c[6]), "f"(c[7]));
// // clang-format on
// #endif
// }

// m8n8k4
// m16n8k8
// m16n8k16
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

int main()
{
  float* dram;
  CHECK_CUDA_RETURN(cudaMalloc(&dram, 1024 * sizeof(float)));
  test_ld_global_and_st_shared<<<1, 32>>>(dram, 1024);
  CHECK_CUDA_ERROR();
  test_mma_m8n8k4<<<1, 32>>>();
  CHECK_CUDA_ERROR();
  return 0;
}
