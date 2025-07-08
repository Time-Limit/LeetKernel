template <typename T> __device__ void swap(T &a, T &b) {
  T tmp = a;
  a = b;
  b = tmp;
}

__device__ __inline__ void FETCH_FLOAT4(void *dst, const void *src) {
  *(float4 *)dst = *(float4 *)src;
}
