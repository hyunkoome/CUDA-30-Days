import torch
import time

"""
아래와 같이 패키지 설치
cd day02_dot_product
python setup.py build_ext --inplace
cp build/lib.linux-x86_64-cpython-310/cuda_study_dot*.so dot_cuda/
   
"""
# import cuda_study_dot
# print("✅ dot_cuda module loaded from:", cuda_study_dot.__file__)
# print("✅ dot_cuda contents:", dir(cuda_study_dot))

# ✅ dot_cuda 내부에서 .so 모듈 불러오도록 구성되어 있어야 함
import dot_cuda as cuda_study_dot

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
    res, t = benchmark_dot("shared", cuda_study_dot.dot_shared, a, b)
    print(f"[1] Shared Memory   : {res:.2f} (time: {t:.2f} ms) ✅ {abs(res - expected) < 1e-3}")

    # -----------------------
    # [2] Double Precision (float64)
    # -----------------------
    a64 = torch.ones(N, dtype=torch.float64, device="cuda")
    b64 = 2 * a64
    res, t = benchmark_dot("double", cuda_study_dot.dot_double, a64, b64)
    print(f"[2] Double Precision: {res:.2f} (time: {t:.2f} ms) ✅ {abs(res - expected) < 1e-3}")

    # -----------------------
    # [3] Warp Shuffle (float32)
    # -----------------------
    a = torch.ones(N, dtype=torch.float32, device="cuda")
    b = 2 * a
    res, t = benchmark_dot("warp", cuda_study_dot.dot_warp, a, b)
    print(f"[3] Warp Shuffle    : {res:.2f} (time: {t:.2f} ms) ✅ {abs(res - expected) < 1e-3}")

    # -----------------------
    # [4] Mixed Precision (float16 → float32)
    # -----------------------
    a16 = torch.ones(N, dtype=torch.float16, device="cuda")
    b16 = 2 * a16
    res, t = benchmark_dot("half", cuda_study_dot.dot_half, a16, b16)
    print(f"[4] Mixed Precision : {res:.2f} (time: {t:.2f} ms) ✅ {abs(res - expected) < 1e-1}")
