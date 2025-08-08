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

