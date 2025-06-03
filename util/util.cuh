template <typename T> __device__ void swap(T &a, T &b) {
  T tmp = a;
  a = b;
  b = tmp;
}
