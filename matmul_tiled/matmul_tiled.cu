#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <ctime>

// Output for Tesla T4 GPU
// Max Blocks / SM: 16
// Max Threads / SM: 1024
// Max Regs / SM: 65536
// Max Shared Memory / SM: 65536
// Max Threads / Block: 1024
// Max Regs / Block: 65536
// === Executing kernel ===
// Kernel execution time: 0.000224
// === Completed kernel exec ===
// === Executing kernel ===
// Kernel execution time: 0.000093
// === Completed kernel exec ===
// Results are correct
// Total CPU Execution Time: 0.024360 seconds
// Total GPU (Naive) Execution Time: 0.080274 seconds
// Total GPU (Tiled) Execution Time: 0.000532 seconds

// Observation:
// If TILE_SIZE 16 is picked:
// Num blocks per SM limited by threads per SM = 1024 / (16 * 16) = 4 blocks
// Num blocks per SM limited by shared memory = 65536 / (16 * 16 * 2) = 32 blocks
// Therefore, the limiting factor on this GPU is threads per block, not shared memory.

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error: %s in %s at line %d\n", \
               cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

#define TILE_WIDTH 16

void printDeviceProperties() {
    cudaDeviceProp deviceProp;

    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0));

    printf("Max Blocks / SM: %d\n", deviceProp.maxBlocksPerMultiProcessor);
    printf("Max Threads / SM: %d\n", deviceProp.maxThreadsPerMultiProcessor);
    printf("Max Regs / SM: %d\n", deviceProp.regsPerMultiprocessor);
    printf("Max Shared Memory / SM: %zu\n", deviceProp.sharedMemPerMultiprocessor);

    printf("Max Threads / Block: %d\n", deviceProp.maxThreadsPerBlock);
    printf("Max Regs / Block: %d\n", deviceProp.regsPerBlock);

}

float getTimeElapsed(struct timespec *start, struct timespec *end) {
    return (end->tv_sec - start->tv_sec) + (end->tv_nsec - start->tv_nsec) / 1e9;
}

void initRand(float* h_A, int n) {
    for (int i = 0; i < n; i++)
        h_A[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}

void matmul(/*input matrix*/ float* h_A, /*input matrix*/ float* h_B, /*output matrix*/ float* h_C, 
    int m, int k, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0;
            for (int k_ = 0; k_ < k; k_++) {
                sum += h_A[i * k + k_] * h_B[k_ * n + j];
            }
            h_C[i * n + j] = sum;
        }
    }
}

__global__ void matmulNaiveGpu(/*input matrix*/ float* A, /*input matrix*/ float* B, /*output matrix*/ float* C, 
    int m, int k, int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < m && col < n) {
        float sum = 0;

        for (int k_ = 0; k_ < k; k_++) {
            sum += A[row * k + k_] * B[k_ * n + col];
        }

        C[row * n + col] = sum;
    }
}

__global__ void matmulTiledGpu(/*input matrix*/ float* A, /*input matrix*/ float* B, /*output matrix*/ float* C, 
    int m, int k, int n) {
    extern __shared__ float shmem[];
    
    float* tiledA = shmem;
    float* tiledB = shmem + TILE_WIDTH * TILE_WIDTH;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int col = bx * TILE_WIDTH + threadIdx.x;
    int row = by * TILE_WIDTH + threadIdx.y;

    float sum = 0.0;

    for (int ph = 0; ph < ceil(k / (float) TILE_WIDTH); ph++) {
        if (row < m && (ph * TILE_WIDTH + tx) < k)
            tiledA[ty * TILE_WIDTH + tx] = A[row * k + ph * TILE_WIDTH + tx];
        else 
            tiledA[ty * TILE_WIDTH + tx] = 0.0f;
        
        if ((ph * TILE_WIDTH + ty) < k && col < n)
            tiledB[ty * TILE_WIDTH + tx] = B[(ph * TILE_WIDTH + ty) * n + col];
        else
            tiledB[ty * TILE_WIDTH + tx] = 0.0f;

        __syncthreads();

        for (int tileIdx = 0; tileIdx < TILE_WIDTH; tileIdx++) {
            sum += tiledA[ty * TILE_WIDTH + tileIdx] * tiledB[tileIdx * TILE_WIDTH + tx];
        }

        __syncthreads();
    }

    if (row < m && col < n)
        C[row * n + col] = sum;
}

// type 0 = naive matmul
// type 1 = tiled matmul
void matmulGpu(/*input*/ float* h_A, /*input*/ float* h_B, /*output*/ float* h_C, int m, int k, int n, int type) {
    float *d_A, *d_B, *d_C;

    int sizeA = m * k * sizeof(float);
    int sizeB = k * n * sizeof(float);
    int sizeC = m * n * sizeof(float);

    CUDA_CHECK(cudaMalloc((void**) &d_A, sizeA));
    CUDA_CHECK(cudaMalloc((void**) &d_B, sizeB));
    CUDA_CHECK(cudaMalloc((void**) &d_C, sizeC));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(d_C, 0, sizeC));

    dim3 gridDim_((n - 1 + TILE_WIDTH) / TILE_WIDTH, (m - 1 + TILE_WIDTH) / TILE_WIDTH);
    dim3 blockDim_(TILE_WIDTH, TILE_WIDTH);

    struct timespec start, end;

    printf("=== Executing kernel ===\n");
 
    clock_gettime(CLOCK_MONOTONIC, &start);

    if (type == 0) {
        matmulNaiveGpu<<<gridDim_, blockDim_>>>(d_A, d_B, d_C, m, k, n);
    } else if (type == 1) {
        size_t shmemSize = sizeof(float) * TILE_WIDTH * TILE_WIDTH * 2;
        matmulTiledGpu<<<gridDim_, blockDim_, shmemSize>>>(d_A, d_B, d_C, m, k, n);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    clock_gettime(CLOCK_MONOTONIC, &end);

    float kernelExecutionTime = getTimeElapsed(&start, &end);

    printf("Kernel execution time: %.6f\n", kernelExecutionTime);
    printf("=== Completed kernel exec ===\n");

    CUDA_CHECK(cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

int main() {
    printDeviceProperties();

    int m = 200;
    int k = 200;
    int n = 200;

    srand(time(NULL));

    // MEM ALLOCATION + INITIALIZATION

    float* h_A = (float*) malloc(m * k * sizeof(float));
    float* h_B = (float*) malloc(k * n * sizeof(float));

    initRand(h_A, m * k);
    initRand(h_B, k * n);

    // Keep 2 copies of result matrix to compare results for correctness
    float* h_C = (float*) calloc(m * n, sizeof(float));
    float* h_C_gpu_0 = (float*) calloc(m * n, sizeof(float));
    float* h_C_gpu_1 = (float*) calloc(m * n, sizeof(float));

    // EXECUTION + BENCHMARKING

    struct timespec start, end;

    clock_gettime(CLOCK_MONOTONIC, &start);
    matmul(h_A, h_B, h_C, m, k, n);
    clock_gettime(CLOCK_MONOTONIC, &end);

    float cpuExecutionTime = getTimeElapsed(&start, &end);

    clock_gettime(CLOCK_MONOTONIC, &start);
    matmulGpu(h_A, h_B, h_C_gpu_0, m, k, n, 0);
    clock_gettime(CLOCK_MONOTONIC, &end);

    float gpuExecutionTime_0 = getTimeElapsed(&start, &end);

    clock_gettime(CLOCK_MONOTONIC, &start);
    matmulGpu(h_A, h_B, h_C_gpu_1, m, k, n, 1);
    clock_gettime(CLOCK_MONOTONIC, &end);

    float gpuExecutionTime_1 = getTimeElapsed(&start, &end);

    bool correct = true;
    for (int i = 0; i < m * n; i++) {
        float ferror0 = fabs(h_C[i] - h_C_gpu_0[i]);
        float ferror1 = fabs(h_C[i] - h_C_gpu_1[i]);

        if (ferror0 > 1e-4 || ferror1 > 1e-4) {
            printf("%.6f %.6f %.6f\n", h_C_gpu_0[i], h_C_gpu_1[i], h_C[i]);
            correct = false;
            break;
        }
    }

    if (correct) {
        printf("Results are correct\n");
        printf("Total CPU Execution Time: %.6f seconds\n", cpuExecutionTime);
        printf("Total GPU (Naive) Execution Time: %.6f seconds\n", gpuExecutionTime_0);
        printf("Total GPU (Tiled) Execution Time: %.6f seconds\n", gpuExecutionTime_1);
    } else {
        printf("Results are incorrect\n");
    }

    // MEM FREE

    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_gpu_0);
    free(h_C_gpu_1);
    
    return 0;
}
