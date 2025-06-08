// CUDA-30-Days/day03_matrix_add/src/wrapper_libtorch.cpp
#include "matrix_add_impl.h"
#include "wrapper_libtorch.h"

// 실제 호출은 공통 구현부를 통해 수행
torch::Tensor cpp_wrapper_matrix_add(torch::Tensor a, torch::Tensor b) {
    return pytorch_wrapper_matrix_add(a, b);
}
