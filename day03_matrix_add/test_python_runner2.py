import torch
import time

"""
아래와 같이 패키지 설치
cd day03_matrix_add
python setup.py build_ext --inplace
# cp build/lib.linux-x86_64-cpython-310/cuda_study*.so dot_cuda/
   
"""
import torch
import cuda_study_matrix_add  # setup.py로 빌드한 모듈 이름

def test_matrix_add():
    a = torch.rand(64, 64, device="cuda", dtype=torch.float32)
    b = torch.rand(64, 64, device="cuda", dtype=torch.float32)

    result = cuda_study_matrix_add.my_matrix_add(a, b)
    expected = a + b

    max_error = (result - expected).abs().max().item()
    print(f"result: {result}")
    print(f"expected: {expected}")
    print(f"[Python Test] Max error: {max_error:.6f}")
    assert max_error < 1e-5, "Matrix addition failed!"

if __name__ == "__main__":
    test_matrix_add()
