template <typename T> __device__ void swap(T &a, T &b) {
  T tmp = a;
  a = b;
  b = tmp;
}

#define OFFSET(row, col, stride) \
  ((row) * (stride) + (col))
#define FETCH_FLOAT4(dst, src) \
  *(float4 *)(&(dst)) = *(const float4 *)(&(src))
