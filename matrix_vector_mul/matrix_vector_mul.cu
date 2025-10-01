#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <ctime>

// Output: (run on leetgpu.com)
// Results are correct
// Total CPU Execution Time: 0.052665 seconds
// Total GPU Execution Time: 0.018001 seconds

#define CUDA_CHECK(errorObj) \
    if (errorObj != cudaSuccess) { \
        printf("%s in %s at line number %d", \
        cudaGetErrorString(errorObj), __FILE__, __LINE__); \
    } \

float getTimeElapsed(struct timespec *start, struct timespec *end) {
    return (end->tv_sec - start->tv_sec) + (end->tv_nsec - start->tv_nsec) / 1e9;
}

void initRand(float* h_A, int n) {
    for (int i = 0; i < n; i++)
        h_A[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}

void matrixVectorMul(/*input matrix*/ float* h_A, /*input vector*/ float* h_B, /*output vector*/ float* h_C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            h_C[i] += h_A[i * n + j] * h_B[j];
        }
    }
}

__global__ void matrixVectorMulKernel(/*input matrix*/ float *A, /*input vector*/ float *B, /*output vector*/ float *C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        for (int j = 0; j < n; j++) {
            C[i] += A[i * n + j] * B[j];
        }
    }
}

void matrixVectorMulGpu(/*input*/ float* h_A, /*input*/ float* h_B, /*output*/ float* h_C, int n) {
    float *d_A, *d_B, *d_C;

    int size = n * n * sizeof(float);

    CUDA_CHECK(cudaMalloc((void**) &d_A, size));
    CUDA_CHECK(cudaMalloc((void**) &d_B, size / n));
    CUDA_CHECK(cudaMalloc((void**) &d_C, size / n));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size / n, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(d_C, size / n, 0));

    matrixVectorMulKernel<<<(int) ceil(n / 1024.0), 1024>>>(d_A, d_B, d_C, n);

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_C, d_C, size / n, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

int main() {
    int n = 3500;

    // MEM ALLOCATION + INITIALIZATION

    float* h_A = (float*) malloc(n * n * sizeof(float));
    float* h_B = (float*) malloc(n * sizeof(float));

    initRand(h_A, n * n);
    initRand(h_B, n);

    // Keep 2 copies of result matrix to compare results for correctness
    float* h_C = (float*) calloc(n, sizeof(float));
    float* h_C_gpu = (float*) calloc(n, sizeof(float));

    // EXECUTION + BENCHMARKING

    struct timespec start, end;

    clock_gettime(CLOCK_MONOTONIC, &start);
    matrixVectorMul(h_A, h_B, h_C, n);
    clock_gettime(CLOCK_MONOTONIC, &end);

    float cpuExecutionTime = getTimeElapsed(&start, &end);

    clock_gettime(CLOCK_MONOTONIC, &start);
    matrixVectorMulGpu(h_A, h_B, h_C_gpu, n);
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
        printf("Total CPU Execution Time: %.6f seconds\n", cpuExecutionTime);
        printf("Total GPU Execution Time: %.6f seconds\n", gpuExecutionTime);

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
