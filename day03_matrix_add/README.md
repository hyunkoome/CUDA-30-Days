# 🔧 CUDA 벡터 내적 실습 정리

## 📚 내적(Dot Product) 연산 실습 요약

### ✅ 실습 목표
- 4090 GPU 아키텍처에 최적화된 CUDA 벡터 내적 커널을 구현하고, PyTorch와 연동하여 정확도 검증 및 성능 최적화를 진행한 실습을 정리


## 📁 파일 구성
```shell
CUDA-30-Days/
└── day03_matrix_add/
    ├── CMakeLists.txt                # (libtorch용 빌드 시 사용)
    ├── setup.py                      # (PyTorch Python extension용)
    ├── common/
    │   ├── kernel_matrix_add.cu        # CUDA 커널 정의
    │   ├── matrix_add_impl.h            # 공통 wrapper 선언
    │   ├── matrix_add_impl.cu           # 공통 wrapper 정의
    │   ├── wrapper_python_bind.cpp      # Python 바인딩용 pybind11        
    ├── test_libtorch/
    │   ├── test_libtorch.cpp
    │   ├── wrapper_libtorch.h           # C++ wrapper 선언
    │   └── wrapper_libtorch.cpp         # C++ wrapper 정의
    ├── test_python_runner.py       # ✅ Python 테스트 (loading)
    └── test_python_runner2.py     # ✅ Python 테스트 (패키지 설치 버전)
```


---

## 🧪 실행 방법

### ▶ Python (PyTorch 연동, no install 버전)

```bash
cd day03_matrix_add
python test_python_runner.py 
```
> 결과: PyTorch 연산과 비교하여 `torch.allclose()` 검증
```shell
result: tensor([[1.4881, 0.9589, 0.2772,  ..., 1.4806, 0.9984, 1.0113],
        [1.4939, 0.2483, 0.8625,  ..., 1.1338, 0.8665, 1.2062],
        [0.4593, 1.3059, 1.8273,  ..., 1.0087, 1.5997, 0.6258],
        ...,
        [0.9225, 0.1950, 0.3334,  ..., 0.8414, 0.3744, 1.4235],
        [0.8761, 1.5879, 1.3888,  ..., 0.9018, 1.0699, 1.2689],
        [0.2746, 0.8931, 1.2160,  ..., 1.2432, 0.9642, 1.1217]],
       device='cuda:0')
expected: tensor([[1.4881, 0.9589, 0.2772,  ..., 1.4806, 0.9984, 1.0113],
        [1.4939, 0.2483, 0.8625,  ..., 1.1338, 0.8665, 1.2062],
        [0.4593, 1.3059, 1.8273,  ..., 1.0087, 1.5997, 0.6258],
        ...,
        [0.9225, 0.1950, 0.3334,  ..., 0.8414, 0.3744, 1.4235],
        [0.8761, 1.5879, 1.3888,  ..., 0.9018, 1.0699, 1.2689],
        [0.2746, 0.8931, 1.2160,  ..., 1.2432, 0.9642, 1.1217]],
       device='cuda:0')
[Python Test] Max error: 0.000000
```

### ▶ Python (setup.py & PyTorch 연동)

```bash
cd day03_matrix_add
# sudo ln -s /usr/include/x86_64-linux-gnu/python3.10 /usr/include/python3.10/x86_64-linux-gnu
python setup.py build_ext --inplace
#pip install .

python test_python_runner2.py
```
> 결과: PyTorch 연산과 비교하여 `torch.allclose()` 검증
```shell
result: tensor([[0.8437, 0.4761, 0.2981,  ..., 1.1601, 0.9497, 0.4230],
        [1.3469, 0.6381, 1.6465,  ..., 1.2882, 0.6879, 0.6243],
        [0.6532, 1.0501, 0.9184,  ..., 1.0903, 1.0370, 0.8266],
        ...,
        [0.9170, 0.9104, 0.2181,  ..., 1.3447, 1.3101, 0.8663],
        [0.7380, 0.4955, 0.6292,  ..., 1.2493, 1.1754, 1.1914],
        [1.1636, 1.4477, 1.0678,  ..., 0.5290, 1.2105, 0.6150]],
       device='cuda:0')
expected: tensor([[0.8437, 0.4761, 0.2981,  ..., 1.1601, 0.9497, 0.4230],
        [1.3469, 0.6381, 1.6465,  ..., 1.2882, 0.6879, 0.6243],
        [0.6532, 1.0501, 0.9184,  ..., 1.0903, 1.0370, 0.8266],
        ...,
        [0.9170, 0.9104, 0.2181,  ..., 1.3447, 1.3101, 0.8663],
        [0.7380, 0.4955, 0.6292,  ..., 1.2493, 1.1754, 1.1914],
        [1.1636, 1.4477, 1.0678,  ..., 0.5290, 1.2105, 0.6150]],
       device='cuda:0')
[Python Test] Max error: 0.000000
```
---

### ▶ C++ (단독 실행)

```bash
cd day03_matrix_add/
mkdir build & cd build 

cmake -DCMAKE_PREFIX_PATH="$HOME/libtorch" ..
make -j
```
> 결과: C++ 내에서 직접 실행 후 수치 출력 확인
```shell
./bin/test_runner

A = 
1       2       3       
4       5       6       
B = 
6       5       4       
3       2       1       
A + B = 
7       7       7       
7       7       7       
```
> 결과: C++ libtorch 내에서 직접 실행 후 수치 출력 확인
```shell
./bin/test_libtorch

[Input A]
 0.0808  0.4388  0.7249  0.9828
 0.9849  0.7434  0.2034  0.2962
 0.4865  0.0152  0.4204  0.5473
 0.0410  0.4053  0.3369  0.3410
[ CPUFloatType{4,4} ]
[Input B]
 0.7652  0.9674  0.7438  0.5216
 0.7792  0.1286  0.2778  0.7159
 0.4839  0.8414  0.0291  0.1472
 0.6466  0.2508  0.4426  0.0535
[ CPUFloatType{4,4} ]
[Result A + B]
 0.8459  1.4062  1.4687  1.5044
 1.7641  0.8721  0.4812  1.0121
 0.9704  0.8565  0.4495  0.6945
 0.6876  0.6561  0.7795  0.3945
[ CPUFloatType{4,4} ]

```
## 📚 주요 개념 정리

| 항목 | 설명 |
|------|------|
| **Dot Product** | 두 벡터의 각 원소를 곱한 후 모두 더하는 연산: `a ⋅ b = Σ aᵢbᵢ` |
| **Shared Memory** | 블록 내 스레드가 공유하는 고속 메모리. 반복적 합산(Reduction)에 적합 |
| **Warp Shuffle** | 같은 warp 내 스레드 간 직접 데이터를 교환하여 성능 최적화 |
| **Mixed Precision** | 연산은 float16으로, 합산은 float32로 수행하여 성능+정확도 균형 |
| **Double Precision** | float64 연산. 정확도는 높지만 속도는 낮음 |
| **CUDA 커널 구조** | `__global__` 함수 정의 후 스레드/블록 인덱스로 데이터 접근 |
| **PyTorch 연동** | `torch::Tensor.data_ptr<T>()`로 CUDA 메모리 접근, `PYBIND11_MODULE`로 Python 바인딩 |
| **동기화** | Python에서 커널 실행 후 정확한 타이밍 측정을 위해 `cudaDeviceSynchronize()` 필요 |
| **에러 방지** | `if (idx < N)` 형태로 경계 검사 필수 |
| **정확도 검증** | `abs(result - expected) < ε` 방식으로 float 오차 허용 범위 내 비교 |

---

## 🧠 질문과 정리

### Q1. 왜 Shared Memory가 내적 연산에 유리한가?

| 항목 | 설명 |
|------|------|
| 반복 합산 | thread 간 중간 결과 공유 가능 |
| Register pressure 완화 | global memory 접근 최소화 |
| 병렬화 | reduction 단계별로 병렬 수행 가능 |

✅ **결론**: 내적 연산처럼 `reduction`이 필요한 연산에선 shared memory가 필수에 가깝다.

---

### Q2. Mixed Precision은 정확도가 떨어지지 않을까?

| 항목 | 설명 |
|------|------|
| 곱셈: float16 | 오차 존재하지만 빠름 |
| 합산: float32 | 오차 누적 방지 |
| 최종 오차 | `1e-1` 수준, 실시간 AI에는 허용 가능 범위 |

✅ **결론**: float16 연산이라도 float32 누적이면 충분히 안정적이다.

---

### Q3. Warp Shuffle 방식은 언제 유리한가?

| 항목 | 설명 |
|------|------|
| 공유 메모리 없이도 빠른 데이터 교환 |
| `__shfl_xor_sync()` 등으로 구현 가능 |
| warp(32 threads) 내부에서만 유효 |

✅ **결론**: 32 스레드 이하의 reduction에는 최고의 성능, 블록 간 확장은 제한됨.

---

### Q4. ONNX나 TensorRT에서 이 커널을 활용하려면?

| 항목 | 설명 |
|------|------|
| PyTorch 내장 연산 | `torch.dot()` 또는 `torch.sum(a * b)`는 ONNX로 변환 가능 |
| 커스텀 CUDA 커널 | ONNX 미지원. TensorRT 플러그인 필요 |
| TensorRT 구현 | `IPluginV2DynamicExt` 기반으로 플러그인 작성해야 사용 가능 |

✅ **결론**: ONNX 변환 시엔 기본 연산 사용, 커스텀 커널은 TensorRT 플러그인으로 구현해야 함.

---

## 💾 실습 팁 및 환경 설정

- `TORCH_CUDA_ARCH_LIST="8.9"` 환경 변수 설정 시, RTX 4090 아키텍처에 최적화된 PTX 코드 생성 가능
- `torch.utils.cpp_extension.load()` 또는 `setup.py` + `CUDAExtension` 으로 C++/CUDA 코드 PyTorch에 연동
- `.so` 모듈은 자동으로 `~/.cache/torch_extensions/` 경로에 저장되며 이름은 `name=` 파라미터로 제어
- `__init__.py`를 활용하면 외부 모듈처럼 `from mycuda import my_dot` 형식으로 import 가능

---
