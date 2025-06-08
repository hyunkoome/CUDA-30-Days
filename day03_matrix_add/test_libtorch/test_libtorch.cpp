// test/test_libtorch.cpp

#include <iostream>
#include <torch/torch.h>
#include "wrapper_libtorch.h"  // cpp_wrapper_matrix_add 선언 포함

int main() {
    try {
        // 입력 텐서 생성 (GPU 텐서, float32, 4x4)
        int rows = 4, cols = 4;
        torch::Tensor a = torch::rand({rows, cols}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
        torch::Tensor b = torch::rand({rows, cols}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));

        std::cout << "[Input A]\n" << a.cpu() << "\n";
        std::cout << "[Input B]\n" << b.cpu() << "\n";

        // CUDA 행렬 덧셈 함수 호출
        torch::Tensor result = cpp_wrapper_matrix_add(a, b);

        std::cout << "[Result A + B]\n" << result.cpu() << "\n";
    } catch (const c10::Error& e) {
        std::cerr << "C10 Error: " << e.what() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Std Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
