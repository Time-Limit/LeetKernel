#pragma once

#include "util/util.h"
#include <cstdio>

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

#define FETCH_FLOAT4(dst, src) *(float4 *)(&(dst)) = *(const float4 *)(&(src))

#define STORE_FLOAT4(dst, src) *(float4 *)(&(dst)) = *(const float4 *)(&(src))

#define STORE_FLOAT2(dst, src) *(float2 *)(&(dst)) = *(const float2 *)(&(src))

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
