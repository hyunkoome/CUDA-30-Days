# 🚀 Day XX - [연산이름] CUDA 커널 구현

## 📌 목표

- CUDA로 `[연산이름]` 커널을 직접 구현합니다.
- PyTorch와 C++에서 연동하여 정확성 검증을 수행합니다.
- 기존 PyTorch 연산과 성능을 비교합니다.

---

## 📁 파일 구성

| 파일명 | 설명 |
|--------|------|
| `op_kernel.cu` | CUDA 커널 메인 구현 파일 |
| `op_bindings.py` | PyTorch에서 CUDA 커널을 불러오는 확장 모듈 |
| `test_python_runner.py` | PyTorch와 결과 비교 및 정확도 검증 |
| `test_runner.cpp` | C++ 환경에서 직접 CUDA 실행 결과 확인 |
| `op_cpu.cpp` (optional) | CPU 비교용 함수 (정확도 참고) |
| `op_onnx.py` (optional) | ONNX export 용 코드 |
| `op_trt.cpp` (optional) | TensorRT Plugin 연동 실험용 코드 |
| `op_webgpu.wgsl` (optional) | WebGPU 대응 WGSL 구현 |
| `README.md` | 실습 요약 문서 (현재 파일) |

---

## 🧪 실행 방법

### ▶ Python (PyTorch 연동)

```bash
python test_python_runner.py
```

> 결과: PyTorch 연산과 비교하여 `torch.allclose()` 검증

---

### ▶ C++ (단독 실행)

```bash
nvcc -o test_runner test_runner.cpp op_kernel.cu
./test_runner
```

> 결과: C++ 내에서 직접 실행 후 수치 출력 확인

---

### ▶ ONNX (선택)

```bash
python op_onnx.py
```

> ONNXRuntime 또는 Netron으로 구조 확인 가능

---

## 🔍 결과 비교 예시

| 항목 | PyTorch | CUDA | 차이 여부 |
|------|---------|------|------------|
| 결과 값 | `[0.1, 0.2]` | `[0.1001, 0.2001]` | PASS |
| 실행 시간(ms) | `12.4` | `3.1` | +4x |
| 정확도 비교 | `MAE < 1e-4` |  | OK |

---

## 📚 주요 개념 정리

- **스레드 구조**: `blockIdx`, `threadIdx`, `gridDim` 등 CUDA 병렬처리 구조 이해
- **메모리 활용**: global vs shared memory, coalesced access
- **최적화 포인트**: `__syncthreads()`, tiling, unrolling 등

---

## 🧠 생각해보기

- 이 커널에서 shared memory를 사용하면 성능이 더 향상될까?
- 이 연산은 WebGPU에서도 유사한 방식으로 구현 가능한가?
- 이 커널은 ONNX → TensorRT 변환 시 연산 재활용이 가능할까?

---