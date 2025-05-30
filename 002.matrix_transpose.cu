#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "util/error.h"

__global__ void matrix_transpose_base(const float *input, float *output,
                                      int rows, int cols) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < rows && y < cols) {
    int src_index = x * cols + y;
    int dst_index = y * rows + x;
    printf("data.x=%f row=%d, col=%d\n", input[src_index], src_index, dst_index);
    output[dst_index] = input[src_index];
  }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
void solve_base(const float *input, float *output, int rows, int cols) {
  const int32_t BLOCK_SIZE = 8;
  dim3 threads_per_block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 blocks_per_grid((rows + BLOCK_SIZE - 1) / BLOCK_SIZE,
                       (cols + BLOCK_SIZE - 1) / BLOCK_SIZE);

  matrix_transpose_base<<<blocks_per_grid, threads_per_block>>>(input, output,
                                                                rows, cols);
  CHECK_CUDA_ERROR();
  cudaDeviceSynchronize();
}

__global__ void matrix_transpose_register(const float *input, float *output,
                                          int rows, int cols) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = (blockIdx.y * blockDim.y + threadIdx.y) * 4;

  int src_index = row * cols + col;
  float4 data;
  printf("data.x=%f row=%d, col=%d\n", data.x, row, col);
  if (col + 3 < cols) {
    data = *reinterpret_cast<const float4 *>(input + src_index);
  } else {
    float *ptr = &data.x;
    for (int i = cols - col - 1; i >= 0; --i) {
      ptr[i] = input[src_index + i];
    }
  }

  if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 &&
      threadIdx.y == 0) {
    printf("data.x=%f row=%d, col=%d\n", data.x, row, col);
  }

  auto tmp = rows;
  rows = cols;
  cols = tmp;

  tmp = row;
  row = col;
  col = tmp;

  if (row + 3 < rows) {
    output[row * cols + col] = data.x;
    output[(row + 1) * cols + col] = data.y;
    output[(row + 2) * cols + col] = data.z;
    output[(row + 3) * cols + col] = data.w;
  } else {
    float *ptr = &data.x;
    for (int i = rows - row - 1; i >= 0; --i) {
      output[(row + i) * cols + col] = ptr[i];
    }
  }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
void solve_register(const float *input, float *output, int rows, int cols) {
  const int32_t row_threads_per_block = 16;
  const int32_t col_threads_per_block = 4;
  const int32_t data_per_thread = 4;
  dim3 threads_per_block(row_threads_per_block, col_threads_per_block);

  const int32_t row_blocks_per_grid =
      (rows + row_threads_per_block - 1) / row_threads_per_block;
  const int32_t col_blocks_per_grid =
      (cols + col_threads_per_block * data_per_thread - 1) /
      col_threads_per_block * data_per_thread;

  dim3 blocks_per_grid(row_blocks_per_grid, col_blocks_per_grid);

  matrix_transpose_register<<<blocks_per_grid, threads_per_block>>>(
      input, output, rows, cols);
  cudaDeviceSynchronize();
}

int main() {
  static const int rows = (1 << 13), cols = (1 << 13);

  std::vector<float> host_input(rows * cols), host_output(rows * cols);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(-1000.0f, 1000.0f);
  for (auto &h : host_input) {
    h = dis(gen);
  }

  float *input, *output;
  cudaMalloc(&input, sizeof(float) * rows * cols);
  cudaMalloc(&output, sizeof(float) * rows * cols);
  cudaMemcpy(input, host_input.data(), sizeof(float) * rows * cols,
             cudaMemcpyDefault);

  {
    cudaMemset(output, 0, rows * cols * sizeof(float));
    solve_base(input, output, rows, cols);
    memset(host_output.data(), 0, sizeof(float) * rows * cols);
    cudaMemcpy(host_output.data(), output, sizeof(float) * rows * cols,
               cudaMemcpyDefault);
    const float(*in_ptr)[cols] =
        reinterpret_cast<const float(*)[cols]>(host_input.data());
    const float(*out_ptr)[rows] =
        reinterpret_cast<const float(*)[rows]>(host_output.data());

    for (int x = 0; x < rows; ++x) {
      for (int y = 0; y < cols; ++y) {
        if (in_ptr[x][y] != out_ptr[y][x]) {
          std::stringstream ss;
          ss << "base, invalid value, x=" << x << ", y=" << y
             << ", in=" << in_ptr[x][y] << ", out=" << out_ptr[y][x];
          throw std::runtime_error(ss.str());
        }
      }
    }
  }

  {
    cudaMemset(output, 0, rows * cols * sizeof(float));
    solve_register(input, output, rows, cols);
    memset(host_output.data(), 0, sizeof(float) * rows * cols);
    cudaMemcpy(host_output.data(), output, sizeof(float) * rows * cols,
               cudaMemcpyDefault);
    const float(*in_ptr)[cols] =
        reinterpret_cast<const float(*)[cols]>(host_input.data());
    const float(*out_ptr)[rows] =
        reinterpret_cast<const float(*)[rows]>(host_output.data());

    for (int x = 0; x < rows; ++x) {
      for (int y = 0; y < cols; ++y) {
        if (in_ptr[x][y] != out_ptr[y][x]) {
          std::stringstream ss;
          ss << "register, invalid value, x=" << x << ", y=" << y
             << ", in=" << in_ptr[x][y] << ", out=" << out_ptr[y][x];
          throw std::runtime_error(ss.str());
        }
      }
    }
  }
  return 0;
}
