// CUDA-30-Days/day03_matrix_add/src/kernel_matrix_add.cu

#include <cuda_runtime.h>
#include <stdio.h>  // printf 사용 시 필요

extern "C"
__global__ void matrix_add_kernel(const float* a, const float* b, float* result, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row * cols + col;
    if (row < rows && col < cols) {
        result[idx] = a[idx] + b[idx];
    }
}

// 커널 런처 함수 (.cpp에서 호출할 수 있도록)
// 주의: 이 함수는 __host__ CPU 코드이므로 cpp에서 사용 가능
// ✅ C++ 스타일 함수 정의 (extern "C" 제거)
// extern "C"
void cuda_launch_matrix_add(const float* a, const float* b, float* result, int rows, int cols) {
    const int THREADS = 16;  // CUDA block당 쓰레드 수
    dim3 threads(THREADS, THREADS);
    dim3 blocks((cols + THREADS - 1) / THREADS, (rows + THREADS - 1) / THREADS);

    matrix_add_kernel<<<blocks, threads>>>(
        a,        // ✅ 이미 float* 이므로 그대로 전달
        b,
        result,
        rows,
        cols
    );

    // PyTorch 확장에서 CUDA 오류가 발생하면 바로 catch되지 않아서 cudaGetLastError()로 확인
    // low-level CUDA 디버깅 메시지
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("[CUDA ERROR] matrix_add_kernel failed: %s\n", cudaGetErrorString(err));
    }
}