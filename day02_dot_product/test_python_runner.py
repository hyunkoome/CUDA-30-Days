import os
import torch
import time
from torch.utils.cpp_extension import load

# ✅ 환경 설정 (4090 아키텍처에 맞춤)
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"

# ✅ CUDA 커널 컴파일 및 PyTorch에 모듈로 로드
dot_mod = load(
    name="dot_cuda",
    sources=[
        "dot_wrapper.cu",           # 래퍼
        "shared_memory_float_kernel.cu",    # shared memory 방식
        "shared_memory_double_precision_kernel.cu",    # double precision 방식
        "warp_shuffle_kernel.cu",      # warp shuffle 방식
        "shared_memory_mixed_precision_kernel.cu"       # mixed precision 방식
    ]
)

def benchmark_dot(name, func, a, b):
    torch.cuda.synchronize()
    start = time.time()
    result = func(a, b)
    torch.cuda.synchronize()
    end = time.time()
    return result.item(), (end - start) * 1000  # ms 단위 시간 반환

if __name__ == "__main__":
    # ----------------------------------------
    # ✅ 공통 입력 설정
    # ----------------------------------------
    N = 1 << 20  # 2^20 = 1048576
    expected = 2.0 * N  # 각 원소 1*2 → 2*N

    print("내적 테스트 시작 (N = 2^20 = 1,048,576)\n")

    # -----------------------
    # [1] Shared Memory (float32)
    # -----------------------
    a = torch.ones(N, dtype=torch.float32, device="cuda")
    b = 2 * a
    res, t = benchmark_dot("shared", dot_mod.dot_shared, a, b)
    print(f"[1] Shared Memory   : {res:.2f} (time: {t:.2f} ms) ✅ {abs(res - expected) < 1e-3}")

    # -----------------------
    # [2] Double Precision (float64)
    # -----------------------
    a64 = torch.ones(N, dtype=torch.float64, device="cuda")
    b64 = 2 * a64
    res, t = benchmark_dot("double", dot_mod.dot_double, a64, b64)
    print(f"[2] Double Precision: {res:.2f} (time: {t:.2f} ms) ✅ {abs(res - expected) < 1e-3}")

    # -----------------------
    # [3] Warp Shuffle (float32)
    # -----------------------
    a = torch.ones(N, dtype=torch.float32, device="cuda")
    b = 2 * a
    res, t = benchmark_dot("warp", dot_mod.dot_warp, a, b)
    print(f"[3] Warp Shuffle    : {res:.2f} (time: {t:.2f} ms) ✅ {abs(res - expected) < 1e-3}")

    # -----------------------
    # [4] Mixed Precision (float16 → float32)
    # -----------------------
    a16 = torch.ones(N, dtype=torch.float16, device="cuda")
    b16 = 2 * a16
    res, t = benchmark_dot("half", dot_mod.dot_half, a16, b16)
    print(f"[4] Mixed Precision : {res:.2f} (time: {t:.2f} ms) ✅ {abs(res - expected) < 1e-1}")
