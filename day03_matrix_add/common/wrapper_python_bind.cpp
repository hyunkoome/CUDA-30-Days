// CUDA-30-Days/day03_matrix_add/src/wrapper_python_bind.cpp
// (Python 바인딩)

#include <torch/extension.h>
#include "matrix_add_impl.h"  // ✅ CUDA 구현 포함된 헤더

// 🔹 여기서는 별도 함수 정의 없이 바로 바인딩만
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("my_matrix_add", &pytorch_wrapper_matrix_add, "Matrix Add (CUDA, float32)");
}