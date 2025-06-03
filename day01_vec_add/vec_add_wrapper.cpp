// ===============================================
// 🔧 PyTorch 확장 모듈을 만들기 위한 헤더 파일
// - 텐서 자료형 (torch::Tensor) 사용
// - pybind11 기반의 모듈 등록 기능 포함
// ===============================================


/*
extern "C"
    CUDA에서 정의된 함수가 이름 맹글링 없이 C 스타일로 export되도록 함
data_ptr<float>()
    PyTorch 텐서 내부 데이터를 CUDA에서 접근 가능한 raw 포인터로 변환
PYBIND11_MODULE
    PyTorch에서 사용할 수 있는 Python 모듈로 등록하는 pybind11 매크로
*/

#include <torch/extension.h>
#include <vector>  // (여기선 필수는 아니지만, 텐서 리스트 쓸 경우 필요)

// ===============================================
// 🔗 CUDA 런처 함수 선언부
// - 실제 CUDA 코드(vec_add_kernel.cu)에 구현된 함수
// - extern "C" 를 붙여야 C++ 이름 맹글링을 방지할 수 있음
// - 그렇지 않으면 PyTorch 확장 시 symbol을 찾지 못함
// ===============================================
extern "C" void vec_add_launcher(float* a, float* b, float* c, int N); // 🔒 반드시 extern "C" 필요

// ===============================================
// 🚀 Python → C++ → CUDA 연결 함수 정의
// - Python에서 전달받은 torch.Tensor를 CUDA가 처리할 수 있는 raw pointer로 변환
// - a.data_ptr<float>() → CUDA에서 사용할 수 있는 float* 포인터
// ===============================================
void vec_add(torch::Tensor a, torch::Tensor b, torch::Tensor c, int N) {
    vec_add_launcher(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        N
    );
}

// ===============================================
// 📦 PyTorch 확장 모듈 등록
// - TORCH_EXTENSION_NAME: PyTorch에서 자동으로 모듈 이름으로 바꿔줌
// - m.def("함수이름", &C++함수포인터, "문서 설명")
// - Python에서 `vec_add_launcher()` 라는 이름으로 사용할 수 있음
// ===============================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vec_add_launcher", &vec_add, "Vector Add CUDA");
}

