// Isaiah Iruoha 20346489
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <time.h>

#define TOLERANCE 0.001f         // tolerance for result comparison
#define TILE_WIDTH 16            // change this value manually for different tile sizes

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

// init matrix with random floats between 0 and 1
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

// tiled matrix multiplication kernel using dynamic shared memory
// tileWidth is passed as a kernel parameter so that the shared memory size is computed at runtime
__global__ void tiledMatMulKernel(const float* M, const float* N, float* P, int Width, int tileWidth)
{
    extern __shared__ float shared[];  // shared memory for both tiles
    float* tileM = shared;              // first tileWidth*tileWidth floats for M
    float* tileN = shared + tileWidth * tileWidth;  // next tileWidth*tileWidth floats for N

    int row = blockIdx.y * tileWidth + threadIdx.y;  // row index in P
    int col = blockIdx.x * tileWidth + threadIdx.x;  // col index in P
    float Pvalue = 0.0f;
    int numTiles = (Width + tileWidth - 1) / tileWidth;  // number of phases in k-dim

    for (int ph = 0; ph < numTiles; ph++) {
        int tiledCol = ph * tileWidth + threadIdx.x;  // index for current tile in M
        if (row < Width && tiledCol < Width)
            tileM[threadIdx.y * tileWidth + threadIdx.x] = M[row * Width + tiledCol];
        else
            tileM[threadIdx.y * tileWidth + threadIdx.x] = 0.0f;

        int tiledRow = ph * tileWidth + threadIdx.y;  // index for current tile in N
        if (tiledRow < Width && col < Width)
            tileN[threadIdx.y * tileWidth + threadIdx.x] = N[tiledRow * Width + col];
        else
            tileN[threadIdx.y * tileWidth + threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < tileWidth; k++) {
            Pvalue += tileM[threadIdx.y * tileWidth + k] * tileN[k * tileWidth + threadIdx.x];
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

        float* h_M = (float*)malloc(bytes);  // host input matrix M
        float* h_N = (float*)malloc(bytes);  // host input matrix N
        float* h_P = (float*)malloc(bytes);  // host result from gpu
        float* h_Pcpu = (float*)malloc(bytes);  // host result from cpu

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

        // time host to device transfer
        cudaEventRecord(start, 0);
        cudaMemcpy(d_M, h_M, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_N, h_N, bytes, cudaMemcpyHostToDevice);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        printf("host to device transfer time: %.3f ms\n", elapsedTime);

        dim3 block(TILE_WIDTH, TILE_WIDTH);
        dim3 grid((Width + TILE_WIDTH - 1) / TILE_WIDTH, (Width + TILE_WIDTH - 1) / TILE_WIDTH);
        int sharedSize = 2 * TILE_WIDTH * TILE_WIDTH * sizeof(float); // shared memory for two tiles

        cudaEventRecord(start, 0);
        // launch kernel with dynamic shared memory and pass tile width as parameter
        tiledMatMulKernel<<<grid, block, sharedSize>>>(d_M, d_N, d_P, Width, TILE_WIDTH);
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
        if (correct)
            printf("test PASSED for %dx%d!\n\n", Width, Width);
        else
            printf("test FAILED for %dx%d!\n\n", Width, Width);

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
