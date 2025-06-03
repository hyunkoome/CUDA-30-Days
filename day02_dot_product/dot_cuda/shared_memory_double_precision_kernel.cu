// shared_memory_double_precision_kernel.cu

#include <cuda_runtime.h>

#define THREADS 256

extern "C"
__global__ void dotProductDouble(const double* a, const double* b, double* result, int size) {
    __shared__ double cache[THREADS];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIdx = threadIdx.x;

    double temp = 0;
    while (tid < size) {
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }
    cache[cacheIdx] = temp;
    __syncthreads();

    for (int i = blockDim.x / 2; i > 0; i >>= 1) {
        if (cacheIdx < i)
            cache[cacheIdx] += cache[cacheIdx + i];
        __syncthreads();
    }

    if (cacheIdx == 0)
        atomicAdd(result, cache[0]);
}
