# 🚀 Day 01 - `vector add` CUDA 커널 구현

## 📌 목표

- CUDA로 `vector add` 커널을 직접 구현합니다.
- PyTorch와 C++에서 연동하여 정확성 검증을 수행합니다.
- 기존 PyTorch 연산과 성능을 비교합니다.

---

## 📁 파일 구성

| 파일명 | 설명                             |
|--------|--------------------------------|
| `shared_memory_kernel.cu` | CUDA 커널 메인 구현 파일               |
| `vec_add_wrapper.cpp` | PyTorch 확장 모듈을 만들기 위한 헤더 파일 |
| `test_python_runner.py` | PyTorch와 결과 비교 및 정확도 검증        |
| `test_runner.cpp` | C++ 환경에서 직접 CUDA 실행 결과 확인      |
| `README.md` | 실습 요약 문서 (현재 파일)               |

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
[4] Mixed Precision : 2097152.00 (time: 0.05 ms) ✅ True
```
---

### ▶ Python (setup.py & PyTorch 연동)

```bash
python test_python_runner.py
```
> 결과: PyTorch 연산과 비교하여 `torch.allclose()` 검증
```shell
내적 테스트 시작 (N = 2^20 = 1,048,576)

[1] Shared Memory   : 2097152.00 (time: 0.17 ms) ✅ True
[2] Double Precision: 2097152.00 (time: 0.07 ms) ✅ True
[3] Warp Shuffle    : 2097152.00 (time: 0.08 ms) ✅ True
[4] Mixed Precision : 2097152.00 (time: 0.05 ms) ✅ True
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
[1] Shared Memory:      2.09715e+06     (0.102252 ms)
[2] Double Precision:   2.09715e+06     (0.029117 ms)
[3] Warp Shuffle:       2.09715e+06     (0.054394 ms)
[4] Half Precision:     2.09715e+06     (0.01524 ms)
```


## 📚 주요 개념 정리

| 항목 | 설명 |
|------|------|
| **CUDA 커널 (`__global__`)** | GPU에서 병렬로 실행되는 함수로, 각 스레드는 입력 벡터의 하나의 인덱스를 처리함 |
| **그리드/블록 인덱싱** | `int i = threadIdx.x + blockIdx.x * blockDim.x`를 통해 각 스레드의 전역 인덱스를 계산함 |
| **경계 조건 체크** | `if (i < N)` 구문으로 배열 범위를 초과하지 않도록 안전하게 처리 |
| **cudaMalloc / cudaMemcpy / cudaFree** | GPU 메모리 할당, 데이터 복사, 메모리 해제를 수행하는 CUDA API 함수들 |
| **torch::Tensor → data_ptr<T>()** | PyTorch 텐서를 CUDA에서 사용할 수 있도록 포인터(raw pointer)로 변환 |
| **pybind11 모듈 등록 (`PYBIND11_MODULE`)** | C++과 Python을 연결해주는 매크로로, C++ 함수를 Python에서 사용할 수 있게 함 |
| **extern "C"** | C++의 함수 이름 변경(name mangling)을 방지하여 CUDA 런처를 C 스타일로 노출함 |
| **torch.utils.cpp_extension.load()** | PyTorch에서 `.cu`, `.cpp` 소스코드를 런타임에 컴파일하고 모듈로 로드해주는 유틸 함수 |
| **torch.allclose()** | PyTorch 연산 결과와 CUDA 연산 결과를 비교하여 정확성을 검증하는 함수 |
| **TORCH_CUDA_ARCH_LIST** | 컴파일 대상 GPU 아키텍처를 제한하여 불필요한 빌드 시간과 용량을 줄임 (예: `"8.9"` for RTX 4090) |

---

## 🧠 생각해보기

### Q1. 이 커널에서 shared memory를 사용하면 성능이 더 향상될까?

**A: 일반적인 벡터 덧셈(`c[i] = a[i] + b[i]`)에서는 shared memory를 써도 성능 향상은 거의 없음.**

| 항목 | 설명 |
|------|------|
| 메모리 접근 패턴 | `a[i]`, `b[i]`는 연속적인 global memory → coalesced access already |
| shared memory overhead | 데이터를 shared memory로 복사 → 오히려 느려질 수 있음 |
| 연산 밀도 | 연산량이 너무 작음 (`1 FLOP` per element) → 메모리 대역폭이 병목 아님 |

✅ **정리**: shared memory는 matmul, conv, reduce, histogram 등에 더 효과적이고, 단순 vec_add에는 비효율적임.

---

### Q2. 이 연산은 WebGPU에서도 유사한 방식으로 구현 가능한가?

**A: 예, 매우 쉽게 구현 가능함. WebGPU의 WGSL에서도 유사한 병렬 벡터 덧셈을 쉽게 처리할 수 있음.**

| 항목 | 설명 |
|------|------|
| 병렬 처리 | `@compute @workgroup_size(256)`으로 스레드 그룹 선언 가능 |
| 인덱싱 방식 | `global_id = global_invocation_id.x` 구조는 CUDA와 유사 |
| 메모리 모델 | WGSL의 `storage` buffer는 global memory처럼 사용 가능 |

✅ **정리**: vec_add는 WebGPU용 WGSL로 그대로 이식 가능하며, 구조적 차이가 거의 없음.

---

### Q3. 이 커널은 ONNX → TensorRT 변환 시 연산 재활용이 가능할까?

**A: 직접적으로는 재활용 불가. vec_add 커널은 ONNX의 표준 노드로 매핑되지 않기 때문.**

| 항목 | 설명 |
|------|------|
| ONNX 표준 연산 | `Add`는 있음 → `aten::add` or `onnx::Add` |
| 문제 | 커스텀 CUDA 커널(vec_add_launcher)는 ONNX로 export 불가 (Symbol 없음) |
| 해결 방법 | PyTorch의 `torch.add()`로 구성된 모델을 ONNX로 export하면 내부적으로 `Add` 노드가 됨 |
| TensorRT에서 재활용 | 가능하지만, 커스텀 커널은 별도로 플러그인으로 등록해야 함 |

✅ **정리**: vec_add는 ONNX 변환 시 직접 사용되진 않지만, PyTorch 연산으로 구성된 모델에서 `Add` 노드로 대체 가능함.

---