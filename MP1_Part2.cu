#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <time.h>    //for random seeds on host

__global__ void matMulKernel(const float* M, const float* N, float* P, int Width)
{
    //calculate the row index of the P element
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    // Calculate the column index of the P element
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < Width && col < Width) {
        float sum = 0.0f;
        for (int k = 0; k < Width; k++) {
            sum += M[row * Width + k] * N[k * Width + col];
        }
        P[row * Width + col] = sum;
    }
} 

void matMulCPU(const float* A, const float* B, float* C, int Width)
{
    for (int row = 0; row < Width; row++) {
        for (int col = 0; col < Width; col++) {
            float sum = 0.0f;
            for (int k = 0; k < Width; k++) {
                sum += A[row * Width + k] * B[k * Width + col];
            }
            C[row * Width + col] = sum;
        }
    }
}

void randomInit(float* data, int size)
{
    //generate random floats from 0 to 1
    for (int i = 0; i < size; i++) {
        data[i] = (float)(rand() / (float)RAND_MAX);
    }
}

bool compareArrays(const float* ref, const float* gpu, int size, float tolerance = 0.001f)
{
    for (int i = 0; i < size; i++) {
        float diff = fabs(ref[i] - gpu[i]);
        if (diff > tolerance) {
            return false;
        }
    }
    return true;
}

int main()
{
    int testSizes[] = {256, 512, 1024, 2048, 4096};
    int numTests = sizeof(testSizes)/sizeof(testSizes[0]);

    //set seed for host random initialization
    srand((unsigned int)time(NULL));

    for (int t = 0; t < numTests; t++) {
        int Width = testSizes[t];
        int size  = Width * Width;
        size_t bytes = size * sizeof(float);
        printf("Matrix Multiplication Test: %d x %d\n", Width, Width);
      
        //allocate Host Memory
        float* h_M    = (float*)malloc(bytes); //host input matrix M
        float* h_N    = (float*)malloc(bytes); //host input matrix N
        float* h_P    = (float*)malloc(bytes); //host result from GPU
        float* h_Pcpu = (float*)malloc(bytes); //host result from CPU

        //Initialize Host Memory
        randomInit(h_M, size);
        randomInit(h_N, size);

        //Allocate Device Memory
        float *d_M, *d_N, *d_P;
        cudaMalloc((void**)&d_M, bytes);
        cudaMalloc((void**)&d_N, bytes);
        cudaMalloc((void**)&d_P, bytes);

        //CUDA events for measuring times
        cudaEvent_t start, stop;
        float elapsedTime = 0.0f;

        //time Host-to-Device transfer
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        cudaMemcpy(d_M, h_M, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_N, h_N, bytes, cudaMemcpyHostToDevice);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        printf("Host to Device transfer time: %.3f ms\n", elapsedTime);

         //configure kernel launch and time kernel
        dim3 block(32, 32);  // can also vary block.x and block.y
        dim3 grid((Width + block.x - 1) / block.x, (Width + block.y - 1) / block.y);

        cudaEventRecord(start, 0);

        matMulKernel<<<grid, block>>>(d_M, d_N, d_P, Width);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        printf("Kernel (GPU) execution time: %.3f ms\n", elapsedTime);

        //ignoring memory allocation/free times

        //time device-to-host data transfer
        cudaEventRecord(start, 0);

        cudaMemcpy(h_P, d_P, bytes, cudaMemcpyDeviceToHost);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float d2hTime = 0.0f;
        cudaEventElapsedTime(&d2hTime, start, stop);
        printf("Device to Host transfer time: %.3f ms\n\n", d2hTime);

        //clean up events
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        //CPU reference computation and timing
       // clock_t cpuStart = clock();
       // matMulCPU(h_M, h_N, h_Pcpu, Width);
       // clock_t cpuEnd = clock();
       // float cpuTimeMs = 1000.0f * (float)(cpuEnd - cpuStart) / CLOCKS_PER_SEC;
       // printf("CPU matrix multiplication time: %.3f ms\n", cpuTimeMs);

         //compare results
        //bool correct = compareArrays(h_Pcpu, h_P, size, 1e-3f);
        //if (correct) {
        //    printf("Test PASSED for %dx%d!\n\n", Width, Width);
        //} else {
        //    printf("Test FAILED for %dx%d!\n\n", Width, Width);
        //}

        //clean up and free mem
        cudaFree(d_M);
        cudaFree(d_N);
        cudaFree(d_P);
        free(h_M);
        free(h_N);
        free(h_P);
        free(h_Pcpu);
    }
    return 0;
}
