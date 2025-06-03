import os
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"  # RTX 4090

# PyTorch 모듈 불러오기
import torch

# PyTorch에서 C++/CUDA 확장을 동적으로 컴파일하고 로드하기 위한 유틸
from torch.utils.cpp_extension import load

# 메인 실행 영역 (스크립트가 직접 실행될 때만 실행됨)
if __name__ == '__main__':
    # ----------------------------------------
    # ✅ CUDA 커널과 C++ 래퍼 모듈을 PyTorch에 로드
    # - name: 이 모듈을 PyTorch 내부에서 어떤 이름으로 쓸지
    # - sources: 컴파일할 .cu (CUDA), .cpp (pybind11) 파일 목록
    # ----------------------------------------
    vec_add_mod = load(
        name="vec_add",
        sources=["vec_add_wrapper.cpp", "vec_add_kernel.cu"]
    )

    # ----------------------------------------
    # ✅ 입력 벡터 a, b 생성 (GPU에 저장)
    # - a = [0, 1, 2, ..., 99]
    # - b = [0, 2, 4, ..., 198]
    # ----------------------------------------
    N = 100  # 벡터 길이
    a = torch.arange(N, dtype=torch.float32, device="cuda")
    b = 2 * a
    c = torch.empty_like(a)  # 결과를 저장할 빈 벡터 (GPU상)
    print('a', a.shape, a[:10])
    print('b', b.shape, b[:10])
    print('c', c.shape, c[:10])

    # ----------------------------------------
    # ✅ CUDA 커널 실행
    # - C++ 래퍼 함수(vec_add_launcher)를 호출
    # - 내부에서 vec_add_kernel.cu의 CUDA 커널 실행됨
    # ----------------------------------------
    vec_add_mod.vec_add_launcher(a, b, c, N)

    # ----------------------------------------
    # ✅ PyTorch 기준 정답값 계산
    # - 동일한 연산을 PyTorch에서 직접 수행하여 비교 기준 생성
    # ----------------------------------------
    ref = a + b

    # ----------------------------------------
    # ✅ 정확성 비교
    # - torch.allclose: 두 텐서가 거의 같은 값인지 비교 (부동소수 오차 허용)
    # - atol=1e-5: 절대 오차 허용 한계
    # ----------------------------------------
    print("✅ 정확성 확인:", torch.allclose(c, ref, atol=1e-5))
    print('c', c.shape, c[:10])
    print('ref', ref.shape, ref[:10])

    # ----------------------------------------
    # ✅ 앞부분 결과를 눈으로 확인해보기
    # - 예: 0 + 0 = 0, 1 + 2 = 3, ...
    # ----------------------------------------
    print("▶ 결과 비교 (앞부분):")
    for i in range(5):
        print(f"{a[i].item()} + {b[i].item()} = {c[i].item()} (ref: {ref[i].item()})")
