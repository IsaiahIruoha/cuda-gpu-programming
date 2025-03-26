// Isaiah Iruoha 20346489
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <time.h>

#define TOLERANCE 0.001f
#define TILE_WIDTH 16  // change this value manually for different tile sizes

// cpu reference for square matrix multiplication
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

// init matrix with random floats [0,1]
void randomInit(float* data, int size)
{
    for (int i = 0; i < size; i++) {
        data[i] = (float)rand() / (float)RAND_MAX;
    }
}

// compare two arrays elementwise with tolerance
bool compareArrays(const float* ref, const float* gpu, int size, float tolerance = TOLERANCE)
{
    for (int i = 0; i < size; i++) {
        float diff = fabs(ref[i] - gpu[i]);
        if (diff > tolerance) {
            return false;
        }
    }
    return true;
}

// tiled matrix multiplication kernel using shared memory, templated by tile width
template <int TILE_WIDTH>
__global__ void tiledMatMulKernel(const float* M, const float* N, float* P, int Width)
{
    __shared__ float tileM[TILE_WIDTH][TILE_WIDTH];  // shared tile for m
    __shared__ float tileN[TILE_WIDTH][TILE_WIDTH];  // shared tile for n

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;  // row index in p
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;  // col index in p

    float Pvalue = 0.0f;
    int numTiles = (Width + TILE_WIDTH - 1) / TILE_WIDTH;  // number of tiles in k dim

    for (int ph = 0; ph < numTiles; ph++) {
        int tiledCol = ph * TILE_WIDTH + threadIdx.x;  // index for m tile
        if (row < Width && tiledCol < Width)
            tileM[threadIdx.y][threadIdx.x] = M[row * Width + tiledCol];
        else
            tileM[threadIdx.y][threadIdx.x] = 0.0f;

        int tiledRow = ph * TILE_WIDTH + threadIdx.y;  // index for n tile
        if (tiledRow < Width && col < Width)
            tileN[threadIdx.y][threadIdx.x] = N[tiledRow * Width + col];
        else
            tileN[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; k++) {
            Pvalue += tileM[threadIdx.y][k] * tileN[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < Width && col < Width)
        P[row * Width + col] = Pvalue;
}

int main()
{
    int testSizes[] = {256, 512, 1024, 2048, 4096};  // matrix sizes to test
    int numTests = sizeof(testSizes) / sizeof(testSizes[0]);

    srand((unsigned int)time(NULL));

    for (int t = 0; t < numTests; t++) {
        int Width = testSizes[t];
        int size = Width * Width;
        size_t bytes = size * sizeof(float);
        printf("tiled matrix multiplication test: %d x %d\n", Width, Width);

        float* h_M = (float*)malloc(bytes);  // host m
        float* h_N = (float*)malloc(bytes);  // host n
        float* h_P = (float*)malloc(bytes);  // host p (gpu result)
        float* h_Pcpu = (float*)malloc(bytes);  // host p (cpu ref)

        randomInit(h_M, size);
        randomInit(h_N, size);

        float *d_M, *d_N, *d_P;
        cudaMalloc((void**)&d_M, bytes);
        cudaMalloc((void**)&d_N, bytes);
        cudaMalloc((void**)&d_P, bytes);

        cudaEvent_t start, stop;
        float elapsedTime = 0.0f;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // host to device transfer
        cudaEventRecord(start, 0);
        cudaMemcpy(d_M, h_M, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_N, h_N, bytes, cudaMemcpyHostToDevice);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        printf("host to device transfer time: %.3f ms\n", elapsedTime);

        // set block and grid dimensions based on global TILE_WIDTH
        dim3 block(TILE_WIDTH, TILE_WIDTH);
        dim3 grid((Width + TILE_WIDTH - 1) / TILE_WIDTH, (Width + TILE_WIDTH - 1) / TILE_WIDTH);

        cudaEventRecord(start, 0);
        tiledMatMulKernel<TILE_WIDTH><<<grid, block>>>(d_M, d_N, d_P, Width);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        printf("tile width: %2d, kernel execution time: %.3f ms\n", TILE_WIDTH, elapsedTime);

        // copy device result to host
        cudaMemcpy(h_P, d_P, bytes, cudaMemcpyDeviceToHost);

        // cpu reference computation and timing
        clock_t cpuStart = clock();
        matMulCPU(h_M, h_N, h_Pcpu, Width);
        clock_t cpuEnd = clock();
        float cpuTimeMs = 1000.0f * (float)(cpuEnd - cpuStart) / CLOCKS_PER_SEC;
        printf("cpu matrix multiplication time: %.3f ms\n", cpuTimeMs);

        // compare results
        bool correct = compareArrays(h_Pcpu, h_P, size, 1e-3f);
        if (correct) {
            printf("test PASSED for %dx%d!\n\n", Width, Width);
        } else {
            printf("test FAILED for %dx%d!\n\n", Width, Width);
        }

        // device to host transfer timing (for completeness)
        cudaEventRecord(start, 0);
        cudaMemcpy(h_P, d_P, bytes, cudaMemcpyDeviceToHost);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float d2hTime = 0.0f;
        cudaEventElapsedTime(&d2hTime, start, stop);
        printf("device to host transfer time: %.3f ms\n\n", d2hTime);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
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
