// CUDA-30-Days/day03_matrix_add/src/matrix_add_impl.h

#pragma once
//#include <torch/extension.h>  // ❌ PyTorch + Python 바인딩용
#include <torch/torch.h>  // ✅ PyTorch C++ API만 필요할 경우

// 선언만
torch::Tensor pytorch_wrapper_matrix_add(torch::Tensor a, torch::Tensor b);