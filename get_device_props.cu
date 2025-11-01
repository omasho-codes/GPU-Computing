#include <stdio.h>
#include <cuda_runtime.h>

float bytesToMB(size_t bytes) {
        return static_cast<float>(bytes) / (1024.0f * 1024.0f);
}

int main() {
        int deviceCount = 0;
        cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

        if (error_id != cudaSuccess) {
                fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(error_id));
                return 1;
        }

        if (deviceCount == 0) {
                printf("There are no available CUDA-capable devices.\n");
                return 0;
        }

        printf("Found %d CUDA-capable device(s).\n", deviceCount);
        printf("------------------------------------------\n");

        for (int dev = 0; dev < deviceCount; ++dev) {
                // Select the device
                cudaSetDevice(dev);

                cudaDeviceProp deviceProp;
                cudaGetDeviceProperties(&deviceProp, dev);

                printf("Device %d: \"%s\"\n", dev, deviceProp.name);

                printf("\n  --- General Information ---\n");
                printf("  Compute capability:          %d.%d\n", deviceProp.major, deviceProp.minor);
                printf("  Clock rate:                  %.2f GHz\n", deviceProp.clockRate / 1000000.0f);
                printf("  Device is integrated:        %s\n", (deviceProp.integrated ? "Yes" : "No"));
                printf("  Can map host memory:         %s\n", (deviceProp.canMapHostMemory ? "Yes" : "No"));

                printf("\n  --- Memory Information ---\n");
                printf("  Total global memory:         %.2f MB\n", bytesToMB(deviceProp.totalGlobalMem));
                printf("  Total constant memory:       %.2f KB\n", deviceProp.totalConstMem / 1024.0f);
                printf("  Shared memory per block:     %.2f KB\n", deviceProp.sharedMemPerBlock / 1024.0f);
                printf("  Registers per block:         %d\n", deviceProp.regsPerBlock);
                printf("  L2 Cache Size:               %.2f KB\n", deviceProp.l2CacheSize / 1024.0f);
                printf("  Memory Clock Rate:           %.2f MHz\n", deviceProp.memoryClockRate / 1000.0f);
                printf("  Memory Bus Width:            %d-bit\n", deviceProp.memoryBusWidth);

                printf("\n  --- Multiprocessor & Thread Information ---\n");
                printf("  Multiprocessor count:        %d SMs\n", deviceProp.multiProcessorCount);
                printf("  Warp size:                   %d threads\n", deviceProp.warpSize);
                printf("  Max threads per multiprocessor: %d\n", deviceProp.maxThreadsPerMultiProcessor);
                printf("  Max blocks per multiprocessor: %d\n", deviceProp.maxBlocksPerMultiProcessor);
                printf("  Max threads per block:       %d\n", deviceProp.maxThreadsPerBlock);
                printf("  Max grid dimensions:         (%d, %d, %d)\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
                printf("  Max block dimensions:        (%d, %d, %d)\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);

                printf("------------------------------------------\n");
        }

        return 0;
}
