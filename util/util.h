#pragma once

#define thread_count_calculator()                                              \
  static_assert(BLOCK_TILE_M % THREAD_TILE_M == 0);                            \
  static_assert(BLOCK_TILE_N % THREAD_TILE_N == 0);                            \
  int thread_count =                                                           \
      (BLOCK_TILE_M / THREAD_TILE_M) * (BLOCK_TILE_N / THREAD_TILE_N);         \
  return thread_count >= 32 ? thread_count : 32;

#define corresponding_aligned_M_calculator()                                   \
  static_assert(0 < UNALIGNED_BLOCK_TILE_M && UNALIGNED_BLOCK_TILE_M < 128);   \
  if (UNALIGNED_BLOCK_TILE_M <= 16)                                            \
    return 16;                                                                 \
  if (UNALIGNED_BLOCK_TILE_M <= 32)                                            \
    return 32;                                                                 \
  if (UNALIGNED_BLOCK_TILE_M <= 64)                                            \
    return 64;                                                                 \
  if (UNALIGNED_BLOCK_TILE_M < 128)                                            \
    return 128;

template <int UNALIGNED_BLOCK_TILE_M>
constexpr int host_corresponding_aligned_M_calculator() {
  corresponding_aligned_M_calculator()
}

template <int BLOCK_TILE_M, int BLOCK_TILE_N, int THREAD_TILE_M,
          int THREAD_TILE_N>
constexpr int host_thread_count_calculator() {
  thread_count_calculator();
}
