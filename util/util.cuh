#pragma once

#include "util/util.h"
#include <cstdio>
#include <cuda_fp16.h>

template <typename T> __device__ void swap(T &a, T &b) {
  T tmp = a;
  a = b;
  b = tmp;
}

__device__ __inline__ bool this_block_can_log() {
  return blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0;
}

__device__ __inline__ bool this_thread_can_log(int thread_x = -1) {
  if (thread_x == -1) {
    return blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 &&
           threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0;
  }
  return blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 &&
         threadIdx.x == thread_x && threadIdx.y == 0 && threadIdx.z == 0;
}

__device__ __inline__ void print_thread_info(const char *prefix) {
  printf("%s, block = %03d, %03d, %03d, thread = %03d %03d %03d\n", prefix,
         blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y,
         threadIdx.z);
}

#define OFFSET(row, col, stride) ((row) * (stride) + (col))

#define FETCH_FLOAT(dst, src) *(float *)(&(dst)) = *(const float *)(&(src))

#define FETCH_FLOAT2(dst, src) *(float2 *)(&(dst)) = *(const float2 *)(&(src))

#define FETCH_FLOAT4(dst, src) *(float4 *)(&(dst)) = *(const float4 *)(&(src))

#define FETCH_FLOAT4_WITH_PTR(dst, src) *(float4*)(dst) = *(const float4*)(src)

#define FETCH_FLOAT4_PREFETCH_256B_WITH_SRC_PTR(dst, src)                                                              \
  {                                                                                                                    \
    asm volatile("ld.global.L2::256B.v4.f32 {%0, %1, %2, %3}, [%4];"                                   \
                 : "=f"(dst[0]), "=f"(dst[1]), "=f"(dst[2]), "=f"(dst[3])                                              \
                 : "l"((const float*)(src)));                                                                          \
  }

#define FETCH_FLOAT4_EVICT_LAST_AND_PREFETCH_256B_WITH_SRC_PTR(dst, src)                                               \
  {                                                                                                                    \
    asm volatile("ld.global.L1::evict_last.L2::256B.v4.f32 {%0, %1, %2, %3}, [%4];"                                    \
                 : "=f"(dst[0]), "=f"(dst[1]), "=f"(dst[2]), "=f"(dst[3])                                              \
                 : "l"((const float*)(src)));                                                                          \
  }

#define FETCH_FLOAT4_CONST_PREFETCH_256B_WITH_SRC_PTR(dst, src)                                                        \
  {                                                                                                                    \
    asm volatile("ld.global.nc.L2::128B.v4.f32 {%0, %1, %2, %3}, [%4];"                                                \
                 : "=f"(dst[0]), "=f"(dst[1]), "=f"(dst[2]), "=f"(dst[3])                                              \
                 : "l"((const float*)(src)));                                                                          \
  }

#define FETCH_FLOAT4_CONST_EVICT_LAST_WITH_SRC_PTR(dst, src)                                                           \
  {                                                                                                                    \
    asm volatile("ld.global.nc.L1::evict_last.v4.f32 {%0, %1, %2, %3}, [%4];"                                          \
                 : "=f"(dst[0]), "=f"(dst[1]), "=f"(dst[2]), "=f"(dst[3])                                              \
                 : "l"((const float*)(src)));                                                                          \
  }

#define STORE_FLOAT(dst, src) *(float *)(&(dst)) = *(const float *)(&(src))

#define STORE_FLOAT_WITH_PTR(dst, src) *(float*)((dst)) = *(const float*)((src))

#define STORE_FLOAT2(dst, src) *(float2 *)(&(dst)) = *(const float2 *)(&(src))

#define STORE_FLOAT4(dst, src) *(float4 *)(&(dst)) = *(const float4 *)(&(src))

#define STORE_FLOAT4_WITH_PTR(dst, src) *(float4*)(dst) = *(const float4*)(src)

template <int BLOCK_TILE_M, int BLOCK_TILE_N, int THREAD_TILE_M,
          int THREAD_TILE_N>
__device__ __inline__ constexpr int device_thread_count_calculator() {
  thread_count_calculator();
}

template <int UNALIGNED_BLOCK_TILE_M>
__device__ __inline__ constexpr int
device_corresponding_aligned_M_calculator() {
  corresponding_aligned_M_calculator();
}

template<typename T>
__inline__ __device__ void
mma_sync_aligned_m8n8k4_row_row_f32_f16_f16_f32(float (&D)[8], const T (&A)[4], const T (&B)[4], const float (&C)[8])
{
  asm volatile("mma.sync.aligned.m8n8k4.row.row.f32.f16.f16.f32"
               "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7},"
               "{%8,  %9},"
               "{%10, %11},"
               "{%12, %13, %14, %15, %16, %17, %18, %19};"
               : "=f"(D[0]), "=f"(D[1]), "=f"(D[4]), "=f"(D[5]), "=f"(D[2]), "=f"(D[3]), "=f"(D[6]), "=f"(D[7])
               : "r"(*(uint32_t*)&A[0]),
                 "r"(*(uint32_t*)&A[2]),
                 "r"(*(uint32_t*)&B[0]),
                 "r"(*(uint32_t*)&B[2]),
                 "f"(C[0]),
                 "f"(C[1]),
                 "f"(C[4]),
                 "f"(C[5]),
                 "f"(C[2]),
                 "f"(C[3]),
                 "f"(C[6]),
                 "f"(C[7]));
}

template<typename T>
__inline__ __device__ void mma_m16n8k16_row_col(float (&d)[4], const T (&a)[8], const T (&b)[4], const float (&c)[4])
{
  uint32_t const* A = reinterpret_cast<uint32_t const*>(&a);
  uint32_t const* B = reinterpret_cast<uint32_t const*>(&b);
  float const*    C = reinterpret_cast<float const*>(&c);
  float*          D = reinterpret_cast<float*>(&d);
  if constexpr (std::is_same<T, half>::value) {
    asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32  {%0,%1,%2,%3}, "
      "{%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
      : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));
  }
  else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
    asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32  {%0,%1,%2,%3}, "
      "{%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
      : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));
  }
  else {
    static_assert(std::is_same<T, half>::value == false && std::is_same<T, __nv_bfloat16>::value == false);
  }
}

__forceinline__ __device__ uint32_t get_smid()
{
  uint32_t smid;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
  return smid;
}

__forceinline__ uint32_t get_sm_number()
{
  uint32_t sm_number;
  asm volatile("mov.u32 %0, %%nsmid;" : "=r"(sm_number));
  return sm_number;
}
