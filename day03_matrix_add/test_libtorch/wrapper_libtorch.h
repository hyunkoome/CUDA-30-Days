// CUDA-30-Days/day03_matrix_add/src/wrapper_libtorch.h
#pragma once

// 🔥 변경 전 (Python 바인딩용)
//#include <torch/extension.h>

// ✅ 변경 후 (C++ 전용)
#include <torch/torch.h>


// C++ 코드에서 호출할 수 있도록 외부 선언
torch::Tensor cpp_wrapper_matrix_add(torch::Tensor a, torch::Tensor b);
