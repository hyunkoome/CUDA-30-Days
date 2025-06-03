// test_runner.cu
// CUDA에서 세 가지 방식의 내적(dot product) 구현을 비교 평가하는 테스트 프로그램
// 1. Shared Memory 기반 병렬 reduction
// 2. Double Precision 정확도 비교
// 3. Warp Shuffle 기반 고속 reduction
// 결과 출력과 실행 시간(ms) 측정 포함

#include <iostream>
#include <cuda_runtime.h>
#include <cuda_fp16.h>  // ✅ half 자료형 사용을 위한 헤더
#include <chrono>

#define N (1 << 20)         // 총 벡터 길이: 2^20 = 1,048,576 (약 100만 개 원소)
#define THREADS 256         // CUDA 블록 당 스레드 수

// 커널 함수 선언 (각 구현은 다른 .cu 파일에서 정의됨)
extern "C" __global__ void dotProductShared(const float* a, const float* b, float* result, int size);
extern "C" __global__ void dotProductDouble(const double* a, const double* b, double* result, int size);
extern "C" __global__ void dotProductWarp(const float* a, const float* b, float* result, int size);
extern "C" __global__ void dotProductHalf(const __half* a, const __half* b, float* result, int size);  // ✅ 추가

int main() {
    // -------------------------------
    // ✅ 1. 호스트 메모리 초기화
    // -------------------------------
    // float (32비트 실수형) 배열 생성 및 초기화
    float *h_a = new float[N];
    float *h_b = new float[N];
    for (int i = 0; i < N; ++i) {
        h_a[i] = 1.0f;      // a = [1.0, 1.0, ..., 1.0]
        h_b[i] = 2.0f;      // b = [2.0, 2.0, ..., 2.0]
        // → 결과는 1*2가 N번 더해져 2 * N = 2097152.0 이 되어야 정상
    }

    // -------------------------------
    // ✅ 2. 디바이스 메모리 할당 및 복사 (float)
    // -------------------------------
    float *d_a, *d_b, *d_result_f32;
    cudaMalloc(&d_a, sizeof(float) * N);     // 입력 벡터 a (device)
    cudaMalloc(&d_b, sizeof(float) * N);     // 입력 벡터 b (device)
    cudaMalloc(&d_result_f32, sizeof(float)); // 결과 저장용 (device)
    cudaMemcpy(d_a, h_a, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(float) * N, cudaMemcpyHostToDevice);

    // -------------------------------
    // ✅ 3. double precision 입력 준비
    // -------------------------------
    double *h_a64 = new double[N];
    double *h_b64 = new double[N];
    for (int i = 0; i < N; ++i) {
        h_a64[i] = 1.0;     // double로 동일한 초기화
        h_b64[i] = 2.0;
    }

    double *d_a64, *d_b64, *d_result_f64;
    cudaMalloc(&d_a64, sizeof(double) * N);
    cudaMalloc(&d_b64, sizeof(double) * N);
    cudaMalloc(&d_result_f64, sizeof(double));
    cudaMemcpy(d_a64, h_a64, sizeof(double) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b64, h_b64, sizeof(double) * N, cudaMemcpyHostToDevice);

    // -------------------------------
    // ✅ 4. half precision 준비
    // -------------------------------
    __half *h_a16 = new __half[N];
    __half *h_b16 = new __half[N];
    for (int i = 0; i < N; ++i) {
        h_a16[i] = __float2half(1.0f);
        h_b16[i] = __float2half(2.0f);
    }

    __half *d_a16, *d_b16;
    cudaMalloc(&d_a16, sizeof(__half) * N);
    cudaMalloc(&d_b16, sizeof(__half) * N);
    cudaMemcpy(d_a16, h_a16, sizeof(__half) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b16, h_b16, sizeof(__half) * N, cudaMemcpyHostToDevice);

    // -------------------------------
    // ✅ 5. 커널 실행 설정
    // -------------------------------
    int blocks = (N + THREADS - 1) / THREADS;  // 총 블록 수 계산
    float result_f32 = 0;      // float 결과
    double result_f64 = 0;     // double 결과
    std::chrono::high_resolution_clock::time_point t1, t2;  // 시간 측정용

    // ===============================
    // [1] Shared Memory Kernel 실행
    // ===============================
    cudaMemset(d_result_f32, 0, sizeof(float));  // GPU 메모리 초기화
    t1 = std::chrono::high_resolution_clock::now();  // 시작 시간 기록
    dotProductShared<<<blocks, THREADS>>>(d_a, d_b, d_result_f32, N);
    cudaDeviceSynchronize();                      // 커널 실행 완료까지 대기
    t2 = std::chrono::high_resolution_clock::now();  // 종료 시간 기록
    cudaMemcpy(&result_f32, d_result_f32, sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "[1] Shared Memory:\t" << result_f32 << "\t("
              << std::chrono::duration<float, std::milli>(t2 - t1).count() << " ms)" << std::endl;

    // ===============================
    // [2] Double Precision Kernel 실행
    // ===============================
    cudaMemset(d_result_f64, 0, sizeof(double));
    t1 = std::chrono::high_resolution_clock::now();
    dotProductDouble<<<blocks, THREADS>>>(d_a64, d_b64, d_result_f64, N);
    cudaDeviceSynchronize();
    t2 = std::chrono::high_resolution_clock::now();
    cudaMemcpy(&result_f64, d_result_f64, sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "[2] Double Precision:\t" << result_f64 << "\t("
              << std::chrono::duration<float, std::milli>(t2 - t1).count() << " ms)" << std::endl;

    // ===============================
    // [3] Warp Shuffle Kernel 실행
    // ===============================
    cudaMemset(d_result_f32, 0, sizeof(float));
    t1 = std::chrono::high_resolution_clock::now();
    dotProductWarp<<<blocks, THREADS>>>(d_a, d_b, d_result_f32, N);
    cudaDeviceSynchronize();
    t2 = std::chrono::high_resolution_clock::now();
    cudaMemcpy(&result_f32, d_result_f32, sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "[3] Warp Shuffle:\t" << result_f32 << "\t("
              << std::chrono::duration<float, std::milli>(t2 - t1).count() << " ms)" << std::endl;

    // -----------------------
    // [4] Half Precision
    // -----------------------
    cudaMemset(d_result_f32, 0, sizeof(float));
    t1 = std::chrono::high_resolution_clock::now();
    dotProductHalf<<<blocks, THREADS>>>(d_a16, d_b16, d_result_f32, N);
    cudaDeviceSynchronize();
    t2 = std::chrono::high_resolution_clock::now();
    cudaMemcpy(&result_f32, d_result_f32, sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "[4] Half Precision:\t" << result_f32 << "\t("
              << std::chrono::duration<float, std::milli>(t2 - t1).count() << " ms)" << std::endl;

    // -------------------------------
    // ✅ 6. 메모리 해제 (clean-up)
    // -------------------------------
    delete[] h_a; delete[] h_b;
    delete[] h_a64; delete[] h_b64;
    delete[] h_a16; delete[] h_b16;
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_result_f32);
    cudaFree(d_a64); cudaFree(d_b64); cudaFree(d_result_f64);
    cudaFree(d_a16); cudaFree(d_b16);

    return 0;
}
