#pragma once

#include <cuda_runtime.h>
#include <iostream>

#define CHECK_CUDA_ERROR()                                                                                             \
  do {                                                                                                                 \
    cudaDeviceSynchronize();                                                                                           \
    cudaError_t err = cudaGetLastError();                                                                              \
    if (err != cudaSuccess) {                                                                                          \
      fprintf(stderr, "CUDA error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__);                       \
      exit(EXIT_FAILURE);                                                                                              \
    }                                                                                                                  \
  } while (0)

#define CHECK_CUDA_ERROR_WITH_INFO(prefix)                                                                             \
  do {                                                                                                                 \
    cudaDeviceSynchronize();                                                                                           \
    cudaError_t err = cudaGetLastError();                                                                              \
    if (err != cudaSuccess) {                                                                                          \
      fprintf(stderr, "%s, CUDA error: %s at %s:%d\n", prefix, cudaGetErrorString(err), __FILE__, __LINE__);           \
      exit(EXIT_FAILURE);                                                                                              \
    }                                                                                                                  \
  } while (0)

#define CHECK_CUDA_RETURN(command)                                                                                     \
  do {                                                                                                                 \
    auto err = command;                                                                                                \
    if (err != cudaSuccess) {                                                                                          \
      fprintf(stderr, "CUDA error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__);                       \
      exit(EXIT_FAILURE);                                                                                              \
    }                                                                                                                  \
  } while (0)

#define CHECK_CUDA_RETURN_WITH_INFO(command, prefix)                                                                   \
  do {                                                                                                                 \
    auto err = command;                                                                                                \
    if (err != cudaSuccess) {                                                                                          \
      fprintf(stderr, "%s, CUDA error: %s at %s:%d\n", prefix, cudaGetErrorString(err), __FILE__, __LINE__);           \
      exit(EXIT_FAILURE);                                                                                              \
    }                                                                                                                  \
  } while (0)
