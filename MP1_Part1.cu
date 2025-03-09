#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

int main() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    printf("Number of CUDA Devices: %d\n\n", deviceCount);

    for (int dev = 0; dev < deviceCount; dev++) { // Iterate through devices
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&deviceProps, dev);
        printf("Device %d: %s\n", dev, devProp.name);
        printf("Clock Rate: %d kHz\n", deviceProps.clockRate);
        printf("Number of Streaming Multiprocessors (SMs): %d\n", deviceProps.multiProcessorCount);

        int major = deviceProps.major;
        int minor = deviceProps.minor;
        int coresPerSM;
        if (major == 8 && minor == 6) {
            coresPerSM = 128;  // Ampere architecture (RTX 3060 Ti, RTX A2000)
        } else if (major == 7 && minor == 5) {
            coresPerSM = 64;   // Turing architecture (T600)
        } else {
            coresPerSM = -1;   // Unknown 
        }
        printf("Compute Capability: %d.%d\n", major, minor);
        int totalCores = coresPerSM * deviceProps.multiProcessorCount;
        printf("Number of CUDA Cores: %d\n", totalCores);
        printf("Warp Size: %d\n", deviceProps.warpSize);
        printf("Global Memory: %.2f GB\n", (float)deviceProps.totalGlobalMem / (1024 * 1024 * 1024));
        printf("Constant Memory: %zu bytes\n", deviceProps.totalConstMem);
        printf("Shared Memory per Block: %zu bytes\n", deviceProps.sharedMemPerBlock);
        printf("Registers per Block: %d\n", deviceProps.regsPerBlock);
        printf("Max Threads per Block: %d\n", deviceProps.maxThreadsPerBlock);
        printf("Max Block Dimensions: [%d, %d, %d]\n", deviceProps.maxThreadsDim[0], deviceProps.maxThreadsDim[1], deviceProps.maxThreadsDim[2]);
        printf("Max Grid Dimensions: [%d, %d, %d]\n", deviceProps.maxGridSize[0], deviceProps.maxThreadsDim[1], deviceProps.maxGridSize[2]);
    }
    return 0;
}
