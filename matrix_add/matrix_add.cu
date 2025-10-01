#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <ctime>

// Output: (run on leetgpu.com)
// Results are correct
// Total CPU Execution Time: 0.055757 seconds
// Total GPU Element-wise Execution Time: 5.725004 seconds
// Total GPU Column-wise Execution Time: 1.267537 seconds
// Total GPU Row-wise Execution Time: 0.332614 seconds

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

void matrixAdd(/*input*/ float* h_A, /*input*/ float* h_B, /*output*/ float* h_C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            h_C[i * n + j] = h_A[i * n + j] + h_B[i * n + j];
        }
    }
}

// type 0 = each thread computes an output element
// type 1 = each thread computes an output col
// type 2 = each thread computes an output row
__global__ void matrixAddKernel(/*input*/ float *A, /*input*/ float *B, /*output*/ float *C, int n, int type) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (type == 0) {
        if (i < n && j < n) 
            C[i * n + j] = A[i * n + j] + B[i * n + j];
    } else if (type == 1) {
        if (j < n) {
            for (int i = 0; i < n; i++)
                C[i * n + j] = A[i * n + j] + B[i * n + j];
        }
    } else if (type == 2) {
        if (i < n) {
            for (int j = 0; j < n; j++)
                C[i * n + j] = A[i * n + j] + B[i * n + j];
        }
    }

    
}

// type 0 = each thread computes an output element
// type 1 = each thread computes an output col
// type 2 = each thread computes an output row
void matrixAddGpu(/*input*/ float* h_A, /*input*/ float* h_B, /*output*/ float* h_C, int n, int type) {
    float *d_A, *d_B, *d_C;

    int size = n * n * sizeof(float);

    CUDA_CHECK(cudaMalloc((void**) &d_A, size));
    CUDA_CHECK(cudaMalloc((void**) &d_B, size));
    CUDA_CHECK(cudaMalloc((void**) &d_C, size));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C, size, cudaMemcpyHostToDevice));

    dim3 _gridDim;
    dim3 _blockDim;

    if (type == 0) {
        _gridDim = dim3(ceil(n / 32.0), ceil(n / 32.0), 1);
        _blockDim = dim3(32, 32, 1);
    } else if (type == 1) {
        _gridDim = dim3(1, ceil(n / 32.0), 1);
        _blockDim = dim3(1, 32, 1);
    } else if (type == 2) {
        _gridDim = dim3(ceil(n / 32.0), 1, 1);
        _blockDim = dim3(32, 1, 1);
    }

    matrixAddKernel<<<_gridDim, _blockDim>>>(d_A, d_B, d_C, n, type);

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

int main() {
    int n = 3500;

    // MEM ALLOCATION + INITIALIZATION

    float* h_A = (float*) malloc(n * n * sizeof(float));
    float* h_B = (float*) malloc(n * n * sizeof(float));

    initRand(h_A, n * n);
    initRand(h_B, n * n);

    // Keep 2 copies of result matrix to compare results for correctness
    float* h_C = (float*) calloc(n * n, sizeof(float));
    float* h_C_gpu_0 = (float*) calloc(n * n, sizeof(float));
    float* h_C_gpu_1 = (float*) calloc(n * n, sizeof(float));
    float* h_C_gpu_2 = (float*) calloc(n * n, sizeof(float));

    // EXECUTION + BENCHMARKING

    struct timespec start, end;

    clock_gettime(CLOCK_MONOTONIC, &start);
    matrixAdd(h_A, h_B, h_C, n);
    clock_gettime(CLOCK_MONOTONIC, &end);

    float cpuExecutionTime = getTimeElapsed(&start, &end);

    clock_gettime(CLOCK_MONOTONIC, &start);
    matrixAddGpu(h_A, h_B, h_C_gpu_0, n, 0);
    clock_gettime(CLOCK_MONOTONIC, &end);

    float gpuExecutionTime0 = getTimeElapsed(&start, &end);

    clock_gettime(CLOCK_MONOTONIC, &start);
    matrixAddGpu(h_A, h_B, h_C_gpu_1, n, 1);
    clock_gettime(CLOCK_MONOTONIC, &end);

    float gpuExecutionTime1 = getTimeElapsed(&start, &end);

    clock_gettime(CLOCK_MONOTONIC, &start);
    matrixAddGpu(h_A, h_B, h_C_gpu_2, n, 2);
    clock_gettime(CLOCK_MONOTONIC, &end);

    float gpuExecutionTime2 = getTimeElapsed(&start, &end);

    bool correct = true;
    for (int i = 0; i < n * n; i++) {
        float diff_0 = fabs(h_C[i] - h_C_gpu_0[i]);
        float diff_1 = fabs(h_C[i] - h_C_gpu_1[i]);
        float diff_2 = fabs(h_C[i] - h_C_gpu_2[i]);

        if (diff_0 > 1e-5 || diff_1 > 1e-5 || diff_2 > 1e-5) {
            correct = false;
            break;
        }
    }

    if (correct) {
        printf("Results are correct\n");
        printf("Total CPU Execution Time: %.6f seconds\n", cpuExecutionTime);
        printf("Total GPU Element-wise Execution Time: %.6f seconds\n", gpuExecutionTime0);
        printf("Total GPU Column-wise Execution Time: %.6f seconds\n", gpuExecutionTime1);
        printf("Total GPU Row-wise Execution Time: %.6f seconds\n", gpuExecutionTime2);

    } else {
        printf("Results are incorrect\n");
    }

    // MEM FREE

    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_gpu_0);
    free(h_C_gpu_1);
    free(h_C_gpu_2);
    
    return 0;
}
