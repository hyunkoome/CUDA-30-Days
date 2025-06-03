# 🔧 CUDA 벡터 내적 실습 정리

## 📚 내적(Dot Product) 연산 실습 요약

### ✅ 실습 목표
- CUDA를 이용해 벡터 내적을 다양한 방식(Shared Memory, Double Precision, Warp Shuffle, Mixed Precision)으로 구현
- PyTorch 연동을 통해 Python에서 직접 내적 성능을 비교하고 정확도를 검증

## 📁 파일 구성
| 파일명                                        | 설명                                          |
|--------------------------------------------|---------------------------------------------|
| `shared_memory_float_kernel.cu`            | CUDA 커널 메인 구현 파일 (공유 메모리 O, float)          |
| `shared_memory_double_precision_kernel.cu` | CUDA 커널 메인 구현 파일 (공유 메모리 O, double)         |
| `shared_memory_mixed_precision_kernel.cu`  | CUDA 커널 메인 구현 파일 (공유 메모리 O, mixed precision) |
| `warp_shuffle_kernel.cu`                   | CUDA 커널 메인 구현 파일 (공유 메모리 X, 와프 사용)          |
| `dot_wrapper.cu`                           | PyTorch 확장 모듈을 만들기 위한 헤더 파일                 |
| `test_python_runner.py`                    | PyTorch와 결과 비교 및 정확도 검증 (그냥 load)           |
| `test_python_runner2.py`                   | PyTorch와 결과 비교 및 정확도 검증 (모듈 설치후 load)       |
| `test_runner.cu`                           | C++ 환경에서 직접 CUDA 실행 결과 확인                   |
| `README.md`                                | 실습 요약 문서 (현재 파일)                            |

---

## 🧪 실행 방법

### ▶ Python (PyTorch 연동)

```bash
python test_python_runner.py
```
> 결과: PyTorch 연산과 비교하여 `torch.allclose()` 검증
```shell
내적 테스트 시작 (N = 2^20 = 1,048,576)

[1] Shared Memory   : 2097152.00 (time: 0.17 ms) ✅ True
[2] Double Precision: 2097152.00 (time: 0.07 ms) ✅ True
[3] Warp Shuffle    : 2097152.00 (time: 0.08 ms) ✅ True
[4] Mixed Precision : 2097152.00 (time: 0.04 ms) ✅ True
```
---

### ▶ Python (setup.py & PyTorch 연동)

```bash
cd day02_dot_product
# sudo ln -s /usr/include/x86_64-linux-gnu/python3.10 /usr/include/python3.10/x86_64-linux-gnu
python setup.py build_ext --inplace
cp build/lib.linux-x86_64-cpython-310/cuda_study_dot*.so dot_cuda/
#pip install .

python test_python_runner2.py
```
> 결과: PyTorch 연산과 비교하여 `torch.allclose()` 검증
```shell
내적 테스트 시작 (N = 2^20 = 1,048,576)

[1] Shared Memory   : 2097152.00 (time: 0.16 ms) ✅ True
[2] Double Precision: 2097152.00 (time: 0.07 ms) ✅ True
[3] Warp Shuffle    : 2097152.00 (time: 0.08 ms) ✅ True
[4] Mixed Precision : 2097152.00 (time: 0.04 ms) ✅ True
```
---

### ▶ C++ (단독 실행)

```bash
mkdir build & cd build 
cmake .. & make
./bin/dot_product_all
```

> 결과: C++ 내에서 직접 실행 후 수치 출력 확인
```shell
[1] Shared Memory:      2.09715e+06     (0.099056 ms)
[2] Double Precision:   2.09715e+06     (0.030687 ms)
[3] Warp Shuffle:       2.09715e+06     (0.05348 ms)
[4] Half Precision:     2.09715e+06     (0.016438 ms)
```


## 🚀 주요 개념 정리

| 항목 | 설명 |
|------|------|
| **Dot Product** | 두 벡터의 각 원소를 곱한 뒤 모두 더하는 연산: `a ⋅ b = Σ aᵢbᵢ` |
| **Shared Memory** | 블록 내 스레드가 공유하는 고속 메모리, 반복적 합산(reduction)에서 활용 |
| **Warp Shuffle** | 같은 warp 내에서 스레드 간 데이터를 공유하는 연산. 성능 최적화에 유리 |
| **Mixed Precision** | 연산은 float16으로 하고 합산 결과는 float32로 저장하여 성능+정확도 절충 |
| **Double Precision** | float64 연산. 높은 정확도는 유지되지만 성능은 낮음 |
| **CUDA 커널 구조** | `__global__`로 정의하며, 블록/스레드 인덱스로 데이터 접근 |
| **PyTorch 연동** | `torch::Tensor`를 `.data_ptr<T>()`로 접근, `PYBIND11_MODULE`로 바인딩 |
| **동기화(synchronize)** | 커널 실행 후 Python에서 측정할 때 정확한 타이밍을 위해 동기화 필수 |
| **에러 방지** | `if (idx < N)` 조건문으로 경계 체크는 필수 |
| **정확도 검증** | `abs(res - expected) < ε` 형태로 float 오차를 허용하며 비교 |

---

## 🧠 생각해보기

### Q1. 왜 Shared Memory가 내적 연산에 유리한가?

| 항목 | 설명 |
|------|------|
| 반복적 합산 | `reduce` 연산에서는 shared memory를 통해 중간 결과 공유 가능 |
| register pressure 완화 | 각 스레드가 결과를 공유하므로 불필요한 global memory 접근 감소 |
| 병렬화 효율 증가 | thread 간 협업을 통해 빠른 합산 수행 가능 (예: 1024 → 512 → 256 ... 1) |

✅ **정리**: 내적 연산처럼 reduction이 포함된 연산에는 shared memory가 매우 효과적

---

### Q2. Mixed Precision은 정확도가 떨어지지 않을까?

**A:** 연산은 float16이지만, 합산은 float32로 수행하므로 실제 정확도 손실은 거의 없음.

| 항목 | 설명 |
|------|------|
| 곱셈 정밀도 | float16 → 오차 있음 |
| 합산 정밀도 | float32 → 오차 축적 억제 |
| 실제 오차 | `1e-1` 수준 내외. 대부분의 실시간 시스템에는 허용 가능 수준 |

✅ **정리**: 내적 연산에서는 float32 누적만 보장되면 float16 곱셈도 충분히 안정적임

---

### Q3. Warp Shuffle 방식은 언제 유리한가?

| 항목 | 설명 |
|------|------|
| Thread 간 공유 | shared memory 없이도 동일 warp 내에서 빠른 데이터 교환 가능 |
| overhead 없음 | `__shfl_xor_sync()` 등은 별도 메모리 할당 없이 동작 |
| 제한 조건 | warp 크기(32) 이상 스레드 간 공유 불가. 블록 간 공유도 불가 |

✅ **정리**: **32 스레드 이하의 reduction에는 최고 성능**, 다만 확장성은 한계 있음

---

### Q4. ONNX나 TensorRT에서 이 내적 커널을 활용하려면?

| 항목 | 설명 |
|------|------|
| PyTorch 연산 | `torch.dot()` 또는 `torch.sum(a * b)`는 ONNX export 가능 |
| 직접 커널 | ONNX로는 export 불가. TensorRT 플러그인으로 따로 구현 필요 |
| 플러그인 작성 | `IPluginV2DynamicExt` 인터페이스를 사용해 TensorRT용 커스텀 연산 정의 가능 |

✅ **정리**: 벡터 내적 자체는 ONNX 변환 시 표준 연산으로 대체되지만, 커스텀 커널을 쓰고 싶다면 TensorRT 플러그인으로 따로 구현 필요

---

## 💾 추가 팁

- `TORCH_CUDA_ARCH_LIST="8.9"` 환경 설정으로 4090 아키텍처 전용 커널 생성 가능
- `.so` 모듈 이름과 Python import 이름은 `setup(... name=, CUDAExtension(name=))`에 따라 다름
- `__init__.py`를 통해 `from .cuda_study_dot import *` 방식으로 외부에서 간편하게 import 가능