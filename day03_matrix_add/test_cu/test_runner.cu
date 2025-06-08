// CUDA-30-Days/day03_matrix_add/test/test_runner.cu

#include <iostream>
#include <cuda_runtime.h>

// 커널 선언 (kernel_matrix_add.cu 에 정의되어 있어야 함)
void cuda_launch_matrix_add(const float* a, const float* b, float* out, int M, int N);

void print_matrix(const float* mat, int M, int N, const std::string& name) {
    std::cout << name << " = \n";
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << mat[i * N + j] << "\t";
        }
        std::cout << "\n";
    }
}

int main() {
    const int M = 2;
    const int N = 3;
    const int size = M * N;
    const int bytes = size * sizeof(float);

    // 호스트 메모리
    float h_a[size] = {1, 2, 3, 4, 5, 6};
    float h_b[size] = {6, 5, 4, 3, 2, 1};
    float h_out[size] = {0};

    // 디바이스 메모리
    float *d_a, *d_b, *d_out;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_out, bytes);

    // 복사: host → device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // 커널 실행
    cuda_launch_matrix_add(d_a, d_b, d_out, M, N);

    // 결과 복사: device → host
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    // 결과 출력
    print_matrix(h_a, M, N, "A");
    print_matrix(h_b, M, N, "B");
    print_matrix(h_out, M, N, "A + B");

    // 메모리 해제
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    return 0;
}
