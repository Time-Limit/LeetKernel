#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "util/error.h"
#include "util/util.cuh"

__global__ void matrix_transpose_base(const float *input, float *output,
                                      int rows, int cols) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < rows && y < cols) {
    int src_index = x * cols + y;
    int dst_index = y * rows + x;
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

__global__ void matrix_transpose_register_x_major(const float *input,
                                                  float *output, int rows,
                                                  int cols) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = (blockIdx.y * blockDim.y + threadIdx.y) * 4;

  int src_index = row * cols + col;
  float4 data;
  if (row < rows) {
    if (col + 3 < cols) {
      data = *reinterpret_cast<const float4 *>(input + src_index);
    } else {
      float *ptr = &data.x;
      for (int i = cols - col - 1; i >= 0; --i) {
        ptr[i] = input[src_index + i];
      }
    }
  }

  swap(rows, cols);
  swap(row, col);

  if (col < cols) {
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
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
void solve_register_x_major(const float *input, float *output, int rows,
                            int cols) {
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

  matrix_transpose_register_x_major<<<blocks_per_grid, threads_per_block>>>(
      input, output, rows, cols);
  CHECK_CUDA_ERROR();
  cudaDeviceSynchronize();
}

__global__ void matrix_transpose_register_y_major(const float *input,
                                                  float *output, int rows,
                                                  int cols) {
  int cols_per_thread = 4;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = (blockIdx.x * blockDim.x + threadIdx.x) * cols_per_thread;

  int src_index = row * cols + col;
  float data[4];
  if (row < rows) {
    if (col + 3 < cols) {
      *reinterpret_cast<float4 *>(data) =
          *reinterpret_cast<const float4 *>(input + src_index);
    } else if (row < rows) {
      for (int i = cols - col - 1; i >= 0; --i) {
        data[i] = input[src_index + i];
      }
    }
  }

  int transposed_rows = cols;
  int transposed_cols = rows;
  int transposed_row = col;
  int transposed_col = row;

  if (transposed_col < transposed_cols) {
    if (transposed_row + 3 < transposed_rows) {
      int dst_index = transposed_row * transposed_cols + transposed_col;
      output[dst_index] = data[0];
      output[dst_index += transposed_cols] = data[1];
      output[dst_index += transposed_cols] = data[2];
      output[dst_index += transposed_cols] = data[3];
    } else {
      for (int i = transposed_rows - transposed_row - 1; i >= 0; --i) {
        output[(transposed_row + i) * transposed_cols + transposed_col] =
            data[i];
      }
    }
  }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
void solve_register_y_major(const float *input, float *output, int rows,
                            int cols) {
  const int32_t row_threads_per_block = 16;
  const int32_t col_threads_per_block = 4;
  const int32_t cols_per_thread = 4;
  dim3 threads_per_block(col_threads_per_block, row_threads_per_block);

  const int32_t row_blocks_per_grid =
      (rows + row_threads_per_block - 1) / row_threads_per_block;
  const int32_t col_blocks_per_grid =
      (cols + col_threads_per_block * cols_per_thread - 1) /
      (col_threads_per_block * cols_per_thread);

  dim3 blocks_per_grid(col_blocks_per_grid, row_blocks_per_grid);

  matrix_transpose_register_y_major<<<blocks_per_grid, threads_per_block>>>(
      input, output, rows, cols);
  CHECK_CUDA_ERROR();
  cudaDeviceSynchronize();
}

__global__ void matrix_transpose_register_y_major_eliminate_local_load(
    const float *input, float *output, int rows, int cols) {
  int rows_per_thread = 2;
  int row = (blockIdx.y * blockDim.y + threadIdx.y) * rows_per_thread;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  float data[2];
  if (row < rows && col < cols) {
    data[0] = input[row * cols + col];
  }
  if (row + 1 < rows && col < cols) {
    data[1] = input[(row + 1) * cols + col];
  }


  int transposed_rows = cols;
  int transposed_cols = rows;
  int transposed_row = col;
  int transposed_col = row;

  if (transposed_row < transposed_rows &&
      transposed_col + 1 < transposed_cols) {
    float2 merge{data[0], data[1]};
    (reinterpret_cast<float2 &>(
        output[transposed_row * transposed_cols + transposed_col])) = merge;
  } else if (transposed_row < transposed_rows &&
             transposed_col < transposed_cols) {
    output[transposed_row * transposed_cols + transposed_col] = data[0];
  }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
void solve_register_y_major_eliminate_local_load(const float *input,
                                                 float *output, int rows,
                                                 int cols) {
  const int32_t row_threads_per_block = 8;
  const int32_t col_threads_per_block = 8;
  const int32_t rows_per_thread = 2;
  dim3 threads_per_block(col_threads_per_block, row_threads_per_block);

  const int32_t row_blocks_per_grid =
      (rows + row_threads_per_block * rows_per_thread - 1) /
      (row_threads_per_block * rows_per_thread);
  const int32_t col_blocks_per_grid =
      (cols + col_threads_per_block - 1) /
      (col_threads_per_block);

  dim3 blocks_per_grid(col_blocks_per_grid, row_blocks_per_grid);

  matrix_transpose_register_y_major_eliminate_local_load<<<blocks_per_grid,
                                                           threads_per_block>>>(
      input, output, rows, cols);
  CHECK_CUDA_ERROR();
  cudaDeviceSynchronize();
}

int main() {
  static const int rows = (1 << 13), cols = (1 << 13);
  // static const int rows = 3, cols = 3;

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
    solve_register_y_major_eliminate_local_load(input, output, rows, cols);
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
          ss << "eliminate local load, invalid value, x=" << x << ", y=" << y
             << ", in=" << in_ptr[x][y] << ", out=" << out_ptr[y][x];
          throw std::runtime_error(ss.str());
        }
      }
    }
  }

  {
    cudaMemset(output, 0, rows * cols * sizeof(float));
    solve_register_y_major(input, output, rows, cols);
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
          ss << "register y_major, invalid value, x=" << x << ", y=" << y
             << ", in=" << in_ptr[x][y] << ", out=" << out_ptr[y][x];
          throw std::runtime_error(ss.str());
        }
      }
    }
  }

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
    solve_register_x_major(input, output, rows, cols);
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
          ss << "register x_major, invalid value, x=" << x << ", y=" << y
             << ", in=" << in_ptr[x][y] << ", out=" << out_ptr[y][x];
          throw std::runtime_error(ss.str());
        }
      }
    }
  }

  return 0;
}
