// vec_add_kernel.cu

#include <cuda_runtime.h>  // CUDA 런타임 API를 사용하기 위한 헤더
#include <stdio.h>         // 표준 입출력 사용 (선택사항)

/*
1) __global__ 함수
    : GPU에서 실행되는 커널

2) extern "C"
    : C++ 코드에서 CUDA 함수를 링크하기 위해 필요

3) cudaMalloc / cudaMemcpy / cudaFree
    :   CUDA의 기본 메모리 관리 함수

4) <<<>>> 구문:
    GPU에서 병렬 실행할 커널 함수 호출 방식
*/

// ======================================================
// GPU에서 병렬 실행될 벡터 덧셈 커널 함수 정의
// - 각 스레드가 하나의 인덱스를 담당하여 덧셈 수행
// - 실행 구성: <<<grid, block>>> 으로 호출됨
// ======================================================
__global__ void vec_add(float *a, float *b, float *c, int N) {
    // 현재 스레드의 글로벌 인덱스 계산
    int i = threadIdx.x + blockDim.x * blockIdx.x;

    // 배열 경계 초과 방지를 위한 조건문
    if (i < N) {
        c[i] = a[i] + b[i];  // 각 원소에 대해 a + b 값을 c에 저장
    }
}

// ======================================================
// 외부 C++ 코드에서 호출 가능한 CUDA 런처 함수 정의
// - "__global__" 커널은 직접 호출할 수 없으므로,
//   이 런처 함수가 중간 다리 역할을 수행함
// - extern "C"는 C++의 이름 변경(name mangling)을 방지하여
//   C++ 코드에서 호출 가능하도록 함
// ======================================================
extern "C" void vec_add_launcher(float* a, float* b, float* c, int N) {
    float *d_a, *d_b, *d_c;  // GPU 메모리 포인터

    // GPU 메모리 할당
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));

    // 호스트 메모리 → 디바이스 메모리로 데이터 복사
    cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * sizeof(float), cudaMemcpyHostToDevice);

    // 커널 함수 실행 (그리드 크기 계산 포함)
    vec_add<<<(N + 255) / 256, 256>>>(d_a, d_b, d_c, N);
    // - 256개의 스레드로 구성된 블록을 사용
    // - 총 N개의 데이터를 처리하기 위해 필요한 블록 수는 (N + 255) / 256

    // 디바이스 메모리 → 호스트 메모리로 결과 복사
    cudaMemcpy(c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);

    // GPU 메모리 해제
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

