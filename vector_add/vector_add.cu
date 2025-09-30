#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <ctime>

// OBSERVATIONS:
// Device memory allocation and data transfer dominate execution time,
// wiping out benefits of data parallel execution.

#define CUDA_CHECK(errorObj) \
    if (errorObj != cudaSuccess) { \
        printf("%s in %s at line number %d", \
        cudaGetErrorString(errorObj), __FILE__, __LINE__); \
    } \

void initRand(float* h_A, int n) {
    for (int i = 0; i < n; i++)
        h_A[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}

void vectorAdd(float* h_A, float* h_B, float* h_C, int n) {
    for (int i = 0; i < n; i++)
        h_C[i] = h_A[i] + h_B[i];
}

__global__ void vectorAddKernel(float *A, float *B, float *C, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < n)
        C[i] = A[i] + B[i];
}

void vectorAddGpu(float* h_A, float* h_B, float* h_C, int n) {
    int size = n * sizeof(float);

    float* d_A, *d_B, *d_C;

    CUDA_CHECK(cudaMalloc((void**)&d_A, size));
    CUDA_CHECK(cudaMalloc((void**)&d_B, size));
    CUDA_CHECK(cudaMalloc((void**)&d_C, size));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    vectorAddKernel<<<ceil(n / 256.0), 256.0>>>(d_A, d_B, d_C, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

float getTimeElapsed(struct timespec *start, struct timespec *end) {
    return (end->tv_sec - start->tv_sec) + (end->tv_nsec - start->tv_nsec) / 1e9;
}

int main() {
    int n = 10000000;

    // MEM ALLOCATION + INITIALIZATION

    float* h_A = (float*) malloc(n * sizeof(float));
    float* h_B = (float*) malloc(n * sizeof(float));

    // Keep 2 copies of result vector to compare results for correctness
    float* h_C = (float*) calloc(n, sizeof(float));
    float* h_C_gpu = (float*) calloc(n, sizeof(float));

    initRand(h_A, n);
    initRand(h_B, n);

    // EXECUTION + BENCHMARKING

    struct timespec start, end;

    clock_gettime(CLOCK_MONOTONIC, &start);
    vectorAdd(h_A, h_B, h_C, n);
    clock_gettime(CLOCK_MONOTONIC, &end);

    float cpuExecutionTime = getTimeElapsed(&start, &end);

    clock_gettime(CLOCK_MONOTONIC, &start);
    vectorAddGpu(h_A, h_B, h_C_gpu, n);
    clock_gettime(CLOCK_MONOTONIC, &end);

    float gpuExecutionTime = getTimeElapsed(&start, &end);

    bool correct = true;
    for (int i = 0; i < n; i++) {
        if (fabs(h_C[i] - h_C_gpu[i]) > 1e-5) {
            correct = false;
            break;
        }
    }

    if (correct) {
        printf("Results are correct\n");
        printf("CPU Execution Time: %.6f seconds\n", cpuExecutionTime);
        printf("GPU Execution Time: %.6f seconds\n", gpuExecutionTime);
    } else {
        printf("Results are incorrect\n");
    }

    // MEM FREE

    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_gpu);
    
    return 0;
}
