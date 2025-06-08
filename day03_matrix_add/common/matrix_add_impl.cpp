// CUDA-30-Days/day03_matrix_add/src/matrix_add_impl.cpp

#include "matrix_add_impl.h"
#include <cuda_runtime.h>

// 외부 CUDA 커널 선언: 공통 커널 런처 함수
// 핵심 연산 함수 (Python/C++ 둘 다 여기만 호출)
//extern "C" __global__ void matrix_add_kernel(const float* a, const float* b, float* result, int rows, int cols);
//extern "C" void cuda_launch_matrix_add(const float* a, const float* b, float* result, int rows, int cols);
// ✅ C++ 스타일 함수 정의 (extern "C" 제거)
void cuda_launch_matrix_add(const float* a, const float* b, float* result, int rows, int cols);

/*
🔹 __global__은 CUDA 커널 함수에만 붙입니다:
    __global__ 함수는 GPU에서 병렬 실행되는 커널 함수이고,
    반드시 <<<...>>> 구문으로 GPU에서 호출해야 합니다.
    __global__ void matrix_add_kernel(...) {
        // CUDA 커널 함수 (GPU에서 실행)
    }
🔹 반면 cuda_launch_matrix_add()는:
    일반적인 호스트(CPU) 함수입니다.
    역할: GPU 커널(matrix_add_kernel<<<...>>>)을 호스트에서 호출해주는 래퍼입니다.
    따라서 __global__, __device__, __host__ 아무것도 붙일 필요 없습니다. (기본값은 __host__)
*/

// -----------------------------------------------------
// PyTorch wrapper 함수
// -----------------------------------------------------
torch::Tensor pytorch_wrapper_matrix_add(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.sizes() == b.sizes(), "Input sizes must match");

    // PyTorch 텐서가 float64일 수 있으므로 float32를 명시적으로 확인
    TORCH_CHECK(a.device().is_cuda(), "Tensor a must be a CUDA tensor");
    TORCH_CHECK(b.device().is_cuda(), "Tensor b must be a CUDA tensor");

    TORCH_CHECK(a.dtype() == torch::kFloat32, "Tensor a must be float32");
    TORCH_CHECK(b.dtype() == torch::kFloat32, "Tensor b must be float32");

    // PyTorch 연산에서 텐서가 연속된 메모리(contiguous())가 아닐 경우 CUDA 커널이 오동작할 수 있음
    a = a.contiguous();
    b = b.contiguous();

    // ✅ torch.zeros_like가 아니라 at::zeros_like
    //    auto result = torch.zeros_like(a);
    auto result = at::zeros_like(a);

    int rows = a.size(0);
    int cols = a.size(1);

    // ✅ data_ptr<float>()로 접근
    cuda_launch_matrix_add(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        result.data_ptr<float>(),
        rows, cols
    );

    // PyTorch exception으로 명확하게 사용자에게 전달
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return result;
}


/*
현재 코드에서는 cudaMemcpy() 나 **공유 메모리(__shared__)**를 사용하지 않았음

각각의 의미와 현재 코드에서 왜 생략되었는지를 아래에 정리

✅ 1. cudaMemcpy()가 없는 이유
🔹 cudaMemcpy()는 CPU ↔ GPU 사이 메모리 복사에 사용됩니다.
    하지만 지금 이 wrapper는 PyTorch 텐서(GPU 텐서)를 직접 받아서 GPU 메모리 주소(data_ptr<float>())를 커널에 넘기고 있으므로,
    별도의 cudaMemcpy()가 필요하지 않습니다.

    즉, 이렇게 됩니다:
        a, b, result: 모두 torch::Tensor이고, .cuda() 상태일 것 → 이미 GPU 상의 메모리임

    따라서 PyTorch가 이미 GPU에 메모리를 할당해주므로 cudaMemcpy() 없이 곧장 커널 실행 가능
    a.data_ptr<float>()  // => GPU 메모리 주소

✅ 2. 공유 메모리(__shared__)가 없는 이유
🔹 공유 메모리는 스레드 블록 내에서 빠르게 데이터를 공유할 때 사용됩니다.
    하지만 현재 커널은 매우 단순한 "행렬 요소별 덧셈" 연산입니다:

    result[idx] = a[idx] + b[idx];
    이런 경우: 스레드 간 데이터 공유 필요 없음
    따라서 __shared__ 메모리 사용 이유 없음
    모든 스레드가 자기 인덱스만 처리하므로 병목도 없음

    예시 (공유 메모리 사용이 필요한 경우)
        행렬 곱(MM), Convolution
        Block 내 Reduction (합산, max 등)
        Data Tiling

✨ 요약
항목	            현재 코드에서의 상태	이유
cudaMemcpy()	❌ 필요 없음	        PyTorch 텐서가 이미 GPU 메모리 상에 존재
__shared__	    ❌ 필요 없음	        스레드 간 데이터 공유가 없는 단순 연산
*/
