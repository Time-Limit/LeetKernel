#pragma once

#include "util/util.cuh"

template<typename T, int mg, int ng, int max_mg, int max_ng>
__inline__ __device__ void mg_ng_mma_m16n8k16_row_col(float (&d)[max_mg][max_ng][4],
                                                      const T (&b)[max_ng][8],
                                                      const T (&a)[max_mg][4],
                                                      const float (&c)[max_mg][max_ng][4])
{
  mma_m16n8k16_row_col(d[mg][ng], b[ng], a[mg], c[mg][ng]);
  if constexpr (mg + 1 == max_mg && ng + 1 == max_ng) {
    return;
  }
  if constexpr (mg + 1 == max_mg && ng + 1 < max_ng) {
    mg_ng_mma_m16n8k16_row_col<T, 0, ng + 1, max_mg, max_ng>(d, b, a, c);
  }
  else if constexpr (mg + 1 < max_mg) {
    mg_ng_mma_m16n8k16_row_col<T, mg + 1, ng, max_mg, max_ng>(d, b, a, c);
  }
}

template<typename T, typename = std::enable_if_t<sizeof(T) == 2>>
__inline__ __device__ void shfl_23_and_01(T (&data)[4], uint32_t mask, int lane_id)
{
  uint32_t& _01  = *(uint32_t*)(&data[0]);
  uint32_t& _23 = *(uint32_t*)(&data[2]);
  uint32_t  swap   = (_01 ^ _23) * (!(lane_id & mask));
  _01 ^= swap;
  _23 ^= swap;
  _01 = __shfl_xor_sync(0xffffffff, _01, mask);
  swap = (_01 ^ _23) * (!(lane_id & mask));
  _01 ^= swap;
  _23 ^= swap;
}

template<typename T, typename = std::enable_if_t<sizeof(T) == 2>>
__inline__ __device__ void shfl_4567_and_0123(T (&data)[8], uint32_t mask, int lane_id)
{
  uint64_t& _0123 = *(uint64_t*)(&data[0]);
  uint64_t& _4567 = *(uint64_t*)(&data[4]);
  uint64_t  swap  = (_0123 ^ _4567) * (!(lane_id & mask));
  _0123 ^= swap;
  _4567 ^= swap;
  _0123 = __shfl_xor_sync(0xffffffff, _0123, mask);
  swap  = (_0123 ^ _4567) * (!(lane_id & mask));
  _0123 ^= swap;
  _4567 ^= swap;
}
