import os
import torch
import time
from torch.utils.cpp_extension import load

# ✅ 환경 설정 (4090 아키텍처에 맞춤)
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"

# ✅ CUDA 커널 컴파일 및 PyTorch에 모듈로 로드
cuda_study_matrix_add = load(
    name="cuda_lib",  # 이름은 임의로 지정 가능 (PyTorch가 내부적으로 .so 캐시 저장할 때 쓰는 모듈 이름, import용 아님)
    sources=[
        "common/kernel_matrix_add.cu",        # CUDA 커널
        "common/matrix_add_impl.cpp",         # C++ 구현부 (커널 호출 포함)
        # "common/matrix_add_impl.h",         # 해더 넣으면 안됨!!!
        "common/wrapper_python_bind.cpp",     # ✅ 바인딩 정의 포함
    ],
    verbose=True,
)

def test_matrix_add():
    a = torch.rand(64, 64, device="cuda", dtype=torch.float32)
    b = torch.rand(64, 64, device="cuda", dtype=torch.float32)

    # CUDA 확장 모듈 호출
    result = cuda_study_matrix_add.my_matrix_add(a, b)

    # 정답 계산
    expected = a + b

    max_error = (result - expected).abs().max().item()
    print(f"result: {result}")
    print(f"expected: {expected}")
    print(f"[Python Test] Max error: {max_error:.6f}")
    assert max_error < 1e-5, "Matrix addition failed!"

# ✅ 진입점
if __name__ == "__main__":
    test_matrix_add()
