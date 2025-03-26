// Isaiah Iruoha 20346489
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <time.h>

#define TOLERANCE 0.001f    // tolerance for result comparison
#define TILE_ROWS 12        // tile height for output (and M)
#define TILE_COLS 18        // tile width for output (and N)
#define TILE_K TILE_ROWS    // tile size in the K dimension

// CPU reference for matrix multiplication
void matMulCPU(const float* A, const float* B, float* C, int heightM, int widthM, int widthN)
{
    for (int row = 0; row < heightM; row++) {
        for (int col = 0; col < widthN; col++) {
            float sum = 0.0f;
            for (int k = 0; k < widthM; k++) {
                sum += A[row * widthM + k] * B[k * widthN + col];
            }
            C[row * widthN + col] = sum;
        }
    }
}

// initialize matrix with random floats in between 0 and 1
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
        if (fabs(ref[i] - gpu[i]) > tolerance)
            return false;
    }
    return true;
}

// revised tiled matrix multiplication kernel with boundary checks for non-square matrices
__global__ void revisedTiledMatMulKernel(const float* M, const float* N, float* P,
                                         int heightM, int widthM, int widthN)
{
    // static shared memory for tiles
    __shared__ float tileM[TILE_ROWS][TILE_K];   // tile from M of size (TILE_ROWS x TILE_K)
    __shared__ float tileN[TILE_K][TILE_COLS];     // tile from N of size (TILE_K x TILE_COLS)

    int row = blockIdx.y * TILE_ROWS + threadIdx.y;  // output row index
    int col = blockIdx.x * TILE_COLS + threadIdx.x;    // output col index

    float Pvalue = 0.0f;
    
    // loop over tiles in K dimension; widthM is the K dimension size
    int numTiles = (widthM + TILE_K - 1) / TILE_K;
    for (int t = 0; t < numTiles; t++) {
        // load tile from M into shared memory:
        // each thread with threadIdx.x < TILE_K loads one element of tileM if available.
        if (threadIdx.x < TILE_K) {
            int colM = t * TILE_K + threadIdx.x;
            if (row < heightM && colM < widthM)
                tileM[threadIdx.y][threadIdx.x] = M[row * widthM + colM];
            else
                tileM[threadIdx.y][threadIdx.x] = 0.0f;
        }
        // load tile from N into shared memory:
        // each thread with threadIdx.y < TILE_K loads one element of tileN if available.
        if (threadIdx.y < TILE_K) {
            int rowN = t * TILE_K + threadIdx.y;
            if (rowN < widthM && col < widthN)
                tileN[threadIdx.y][threadIdx.x] = N[rowN * widthN + col];
            else
                tileN[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();
        
        // compute partial dot product for the output element
        for (int k = 0; k < TILE_K; k++) {
            Pvalue += tileM[threadIdx.y][k] * tileN[k][threadIdx.x];
        }
        __syncthreads();
    }
    
    // write the result if within bounds
    if (row < heightM && col < widthN)
        P[row * widthN + col] = Pvalue;
}

int main()
{
    // define two test cases:
    // test 1: M is 750x800, N is 800x850, so P is 750x850
    // test 2: M is 2000x1750, N is 1750x1900, so P is 2000x1900
    int testCases = 2;
    
    // For each test case, define the dimensions: {heightM, widthM, widthN}
    int dims[2][3] = {
        {750, 800, 850},    // Test 1
        {2000, 1750, 1900}   // Test 2
    };
    
    srand((unsigned int)time(NULL));
    
    for (int t = 0; t < testCases; t++) {
        int heightM = dims[t][0];
        int widthM  = dims[t][1];  // also the height of N
        int widthN  = dims[t][2];
        int sizeM = heightM * widthM;
        int sizeN = widthM * widthN;
        int sizeP = heightM * widthN;
        
        printf("Revised Tiled Matrix Multiplication Test: M(%dx%d) * N(%dx%d) = P(%dx%d)\n",
               heightM, widthM, widthM, widthN, heightM, widthN);
        
        size_t bytesM = sizeM * sizeof(float);
        size_t bytesN = sizeN * sizeof(float);
        size_t bytesP = sizeP * sizeof(float);
        
        float *h_M = (float*)malloc(bytesM);
        float *h_N = (float*)malloc(bytesN);
        float *h_P = (float*)malloc(bytesP);
        float *h_Pcpu = (float*)malloc(bytesP);
        
        randomInit(h_M, sizeM);
        randomInit(h_N, sizeN);
        
        float *d_M, *d_N, *d_P;
        cudaMalloc((void**)&d_M, bytesM);
        cudaMalloc((void**)&d_N, bytesN);
        cudaMalloc((void**)&d_P, bytesP);
        
        // copy inputs to device (transfer time not measured)
        cudaMemcpy(d_M, h_M, bytesM, cudaMemcpyHostToDevice);
        cudaMemcpy(d_N, h_N, bytesN, cudaMemcpyHostToDevice);
        
        // set up grid and block dimensions using TILE_ROWS and TILE_COLS
        dim3 block(TILE_COLS, TILE_ROWS);
        dim3 grid((widthN + TILE_COLS - 1) / TILE_COLS, (heightM + TILE_ROWS - 1) / TILE_ROWS);
        
        cudaEvent_t start, stop;
        float kernelTime = 0.0f;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        // measure kernel execution time
        cudaEventRecord(start, 0);
        revisedTiledMatMulKernel<<<grid, block>>>(d_M, d_N, d_P, heightM, widthM, widthN);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&kernelTime, start, stop);
        printf("Kernel (GPU) execution time: %.3f ms\n", kernelTime);
        
        // copy result back to host
        cudaMemcpy(h_P, d_P, bytesP, cudaMemcpyDeviceToHost);
        
        // CPU reference computation and timing
        clock_t cpuStart = clock();
        matMulCPU(h_M, h_N, h_Pcpu, heightM, widthM, widthN);
        clock_t cpuEnd = clock();
        float cpuTimeMs = 1000.0f * (float)(cpuEnd - cpuStart) / CLOCKS_PER_SEC;
        printf("CPU matrix multiplication time: %.3f ms\n", cpuTimeMs);
        
        // compare results
        bool correct = compareArrays(h_Pcpu, h_P, sizeP, 1e-3f);
        if (correct)
            printf("Test PASSED for P(%dx%d)!\n\n", heightM, widthN);
        else
            printf("Test FAILED for P(%dx%d)!\n\n", heightM, widthN);
        
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
