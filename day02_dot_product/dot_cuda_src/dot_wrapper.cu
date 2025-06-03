#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>  // ⬅️ half 자료형 지원
#include <vector>

#define THREADS 256  // CUDA block당 쓰레드 수

/*
함수 이름       	방식              	입력 자료형 → 출력 자료형
dot_shared()	Shared Memory	    float32 → float32
dot_double()	Double Precision	float64 → float64
dot_warp()	    Warp Shuffle	    float32 → float32
dot_half()	    Mixed Precision	    float16 → float32

🔧 주의 사항
    PyTorch에서 float16은 at::Half이지만, 커널에서는 __half이므로 reinterpret_cast 필요
    dot_half() 결과는 float32 텐서로 반환됩니다 (연산 정확도 때문).
    커널 dotProductHalf가 float 결과를 atomicAdd로 반환하는 구조이기 때문에, wrapper에서도 float32로 받는 것이 맞습니다.
*/

// ----------------------------------------------------
// 외부 CUDA 커널 선언 (각 .cu 파일에서 정의됨)
// ----------------------------------------------------
extern "C" __global__ void dotProductShared(const float* a, const float* b, float* result, int size);
extern "C" __global__ void dotProductDouble(const double* a, const double* b, double* result, int size);
extern "C" __global__ void dotProductWarp(const float* a, const float* b, float* result, int size);
extern "C" __global__ void dotProductHalf(const __half* a, const __half* b, float* result, int size);  // ✅ 추가됨

// -----------------------------------------------------
// [1] Shared Memory 버전 내적 연산 함수 (float32 전용)
// -----------------------------------------------------
torch::Tensor dot_shared(torch::Tensor a, torch::Tensor b) {
    const int size = a.numel();
    const int blocks = (size + THREADS - 1) / THREADS;

    auto result = torch::zeros({1}, a.options());  // float32 결과

    dotProductShared<<<blocks, THREADS>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        result.data_ptr<float>(),
        size
    );

    return result;
}

// --------------------------------------------------------
// [2] Double Precision 버전 내적 연산 함수 (float64 전용)
// --------------------------------------------------------
torch::Tensor dot_double(torch::Tensor a, torch::Tensor b) {
    const int size = a.numel();
    const int blocks = (size + THREADS - 1) / THREADS;

    auto result = torch::zeros({1}, a.options().dtype(torch::kFloat64));

    dotProductDouble<<<blocks, THREADS>>>(
        a.data_ptr<double>(),
        b.data_ptr<double>(),
        result.data_ptr<double>(),
        size
    );

    return result;
}

// ------------------------------------------------------
// [3] Warp Shuffle 버전 내적 연산 함수 (float32 전용)
// ------------------------------------------------------
torch::Tensor dot_warp(torch::Tensor a, torch::Tensor b) {
    const int size = a.numel();
    const int blocks = (size + THREADS - 1) / THREADS;

    auto result = torch::zeros({1}, a.options());

    dotProductWarp<<<blocks, THREADS>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        result.data_ptr<float>(),
        size
    );

    return result;
}

// -------------------------------------------------------------
// [4] Mixed Precision 버전 내적 연산 함수 (float16 → float32 출력)
// -------------------------------------------------------------
torch::Tensor dot_half(torch::Tensor a, torch::Tensor b) {
    const int size = a.numel();
    const int blocks = (size + THREADS - 1) / THREADS;

    // float16 입력 → float32 출력
    auto result = torch::zeros({1}, a.options().dtype(torch::kFloat32));

    dotProductHalf<<<blocks, THREADS>>>(
        reinterpret_cast<const __half*>(a.data_ptr<at::Half>()),  // ⬅️ PyTorch half → CUDA half
        reinterpret_cast<const __half*>(b.data_ptr<at::Half>()),
        result.data_ptr<float>(),
        size
    );

    return result;
}

// ----------------------------
// PyTorch 확장 바인딩 등록
// ----------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dot_shared", &dot_shared, "Dot Product (Shared Memory, float32)");
    m.def("dot_double", &dot_double, "Dot Product (Double Precision, float64)");
    m.def("dot_warp",   &dot_warp,   "Dot Product (Warp Shuffle, float32)");
    m.def("dot_half",   &dot_half,   "Dot Product (Mixed Precision, float16→float32)");
}
