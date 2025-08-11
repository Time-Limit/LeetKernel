#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        printf("Failed to get device count: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("Number of CUDA devices: %d\n", deviceCount);

    for (int device = 0; device < deviceCount; ++device) {
        cudaDeviceProp deviceProp;
        err = cudaGetDeviceProperties(&deviceProp, device);
        if (err != cudaSuccess) {
            printf("Failed to get properties for device %d: %s\n", device, cudaGetErrorString(err));
            continue;
        }

        printf("\n=== Device %d: %s ===\n", device, deviceProp.name);
        printf("CUDA Capability: %d.%d\n", deviceProp.major, deviceProp.minor);
        printf("Streaming Multiprocessors (SMs): %d\n", deviceProp.multiProcessorCount);
        printf("CUDA Cores (approx): %d\n", deviceProp.multiProcessorCount * 128); // 假设每 SM 128 CUDA 核心（Ampere/Ada Lovelace 架构）
        printf("Global Memory: %.2f GB\n", (float)deviceProp.totalGlobalMem / (1024 * 1024 * 1024));
        printf("Memory Clock Rate: %.2f MHz\n", (float)deviceProp.memoryClockRate / 1000);
        printf("Memory Bus Width: %d bits\n", deviceProp.memoryBusWidth);
        printf("L2 Cache Size: %.2f MB\n", (float)deviceProp.l2CacheSize / (1024 * 1024));
        printf("Max Threads per SM: %d\n", deviceProp.maxThreadsPerMultiProcessor);
        printf("Max Threads per Block: %d\n", deviceProp.maxThreadsPerBlock);
        printf("Max Block Dimensions: [%d, %d, %d]\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
        printf("Max Grid Dimensions: [%d, %d, %d]\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
        printf("Shared Memory per Block: %.2f KB\n", (float)deviceProp.sharedMemPerBlock / 1024);
        printf("Shared Memory per SM: %.2f KB\n", (float)deviceProp.sharedMemPerMultiprocessor / 1024);
        printf("Registers per Block: %d\n", deviceProp.regsPerBlock);
        printf("Warp Size: %d\n", deviceProp.warpSize);
        printf("Clock Rate: %.2f GHz\n", (float)deviceProp.clockRate / (1000 * 1000));
        printf("Concurrent Kernels: %s\n", deviceProp.concurrentKernels ? "Yes" : "No");
        printf("ECC Enabled: %s\n", deviceProp.ECCEnabled ? "Yes" : "No");
        printf("Compute Mode: %d\n", deviceProp.computeMode);
        printf("Total Constant Memory: %zu bytes\n", deviceProp.totalConstMem);
        printf("Max Texture Memory (1D): %d bytes\n", deviceProp.maxTexture1D);
        printf("Max Texture Memory (2D): %d x %d\n", deviceProp.maxTexture2D[0], deviceProp.maxTexture2D[1]);
        printf("Max Texture Memory (3D): %d x %d x %d\n",
               deviceProp.maxTexture3D[0],
               deviceProp.maxTexture3D[1],
               deviceProp.maxTexture3D[2]);
    }

    return 0;
}
