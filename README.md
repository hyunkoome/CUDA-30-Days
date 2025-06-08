# CUDA-30-Days

# 🔥 CUDA Kernel Study: Conv + Transformer + Attention + SPConv

> 30일간의 CUDA 커널 직접 구현과 PyTorch/C++/TensorRT/WebGPU 연동까지  
> Conv2D, Transformer, Attention, SparseConv 핵심 연산을 한 번에 마스터합니다.

---

## 📌 목표

- CUDA 커널 기본 개념 및 실습
- Conv/Transformer/Attention 커널 직접 구현
- Python(PyTorch) 및 C++ 연동 테스트
- PyTorch/ONNX/TensorRT/WebGPU 비교 분석

---

## 🛠️ 실습 환경

| 구성 | 버전                               |
|------|----------------------------------|
| CUDA | 12.6                             |
| PyTorch | 2.x+, `여기에서는 2.3.0 (CUDA 12.6 전용)` |
| Python | 3.10+                            |
| C++ | CMake 또는 g++                     |
| 선택사항 | ONNXRuntime, TensorRT, WebGPU 환경 |

# Python 3.10이 설치되어 있다는 전제 하에
```
python3.10 -m venv cudastudy
source cudastudy/bin/activate   # Windows: cudastudy\\Scripts\\activate
pip install --upgrade pip
pip install -r requirements.txt
```

# libtorch 설치 
PyTorch는 CUDA minor 버전이 달라도 보통 잘 동작합니다.
CUDA 12.6에서 CUDA 12.1 빌드된 libtorch 사용 → ✔️ 작동 확인됨
```shell
wget https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu121.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.1.0+cu121.zip -d ~/
```
~/libtorch 경로 아래에는 보통 이런 구조가 생김:
```shell
~/libtorch/
├── include/
├── lib/
├── share/
└── ...
```

---

## 🗓️ 30일 커리큘럼

| Day | 커널 이름                                        | 핵심 내용 | 연동 및 비교 |
|-----|----------------------------------------------|-----------|---------------|
| 01 | [vec_add](./day01_vec_add/README.md)         | 벡터 덧셈 | PyTorch + C++ |
| 02 | [dot_product](./day02_dot_product/README.md) | 내적, reduction | PyTorch + C++ |
| 03 | [matrix_add](./day03_matrix_add/README.md)   | 2D 행렬 합 | PyTorch + C++ |
| 04 | relu                                         | 활성화 함수 | PyTorch |
| 05 | matrix_transpose                             | 전치 | memory coalescing |
| 06 | matmul_basic                                 | naive GEMM | PyTorch |
| 07 | matmul_sharedmem                             | tiling | PyTorch + 성능 |
| 08 | global_avg_pool                              | GAP | PyTorch 비교 |
| 09 | batch_norm_forward                           | 정규화 | ONNX export |
| 10 | instance_norm                                | 채널별 정규화 | PyTorch |
| 11 | group_norm                                   | 그룹 정규화 | TensorRT plugin |
| 12 | layernorm_forward                            | LayerNorm FWD | PyTorch |
| 13 | layernorm_backward                           | LayerNorm BWD | 성능 튜닝 |
| 14 | softmax_kernel                               | softmax | PyTorch |
| 15 | softmax_2d_kernel                            | Attention용 | PyTorch |
| 16 | einsum_qk_attn                               | QK^T | Attention 연산 |
| 17 | attention_softmax_fused                      | fused attention | 효율 비교 |
| 18 | gelu_kernel                                  | GELU | activation 성능 |
| 19 | topk_kernel                                  | Top-K | PyTorch |
| 20 | prefix_sum_kernel                            | scan | warp-level sum |
| 21 | warp_shuffle_max                             | warp max | warp shuffle |
| 22 | conv2d_basic                                 | naive Conv | PyTorch |
| 23 | conv2d_shared_tile                           | tile + shared | PyTorch |
| 24 | conv2d_unroll                                | unroll 최적화 | 성능 비교 |
| 25 | spconv_sparse_gemm                           | sparse matmul | PyTorch |
| 26 | spconv_forward                               | sparse forward | TensorRT |
| 27 | spconv_backward                              | grad 구현 | 실습 |
| 28 | transformer_block                            | full block | residual 포함 |
| 29 | flash_attention                              | FlashAttention | memory opt |
| 30 | final_review_day                             | 전체 정리 | 성능 분석 |

---

## 📂 디렉토리 구조 예시
```bash
cuda-kernel-study/
├── README.md
├── day01_vec_add/
│ ├── vec_add_kernel.cu
│ ├── vec_add_bindings.py
│ ├── test_python_runner.py
│ ├── test_runner.cpp
│ └── README.md
├── day02_dot_product/
│ ├── ...
```

## ✅ 실습 방식

1. CUDA 커널 직접 구현 (`*.cu`)
2. Python(PyTorch) 연동 (`*.py`)
3. C++ 실행 비교 (`test_runner.cpp`)
4. PyTorch / TensorRT / ONNX / WebGPU 결과와 비교
5. 성능 측정 및 결과 정리 (`README.md`)

---

## 📈 성능 비교 예시

| 연산 | PyTorch(ms) | CUDA(ms) | 성능 향상 |
|------|-------------|----------|------------|
| matmul | 12.5 | 3.2 | ↑ 3.9x |
| softmax | 4.3 | 1.8 | ↑ 2.4x |

---

## 🧠 Special Thanks

- 다양한 연산들을 최적화하고 직접 비교하며, 
- 실무에 바로 활용할 수 있는 실력을 기르기 위해, 
- 실전 AI 커널 엔지니어링을 목표로 구성되었음
---

## 📄 [라이선스 / License](LICENSE)
[![CC BY-NC 4.0](https://licensebuttons.net/l/by-nc/4.0/88x31.png)](https://creativecommons.org/licenses/by-nc/4.0/)

- 본 프로젝트는 **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** 라이선스를 따릅니다.  
- 누구나 **비영리 목적**에 한해 자유롭게 사용할 수 있으며, **상업적 이용 시 저작자의 허가**가 필요합니다.  
- 단, **저작자(김현구)**는 본 자료를 **자유롭게 상업적 목적으로 사용할 수 있습니다.**

This project is licensed under a **Creative Commons Attribution-NonCommercial 4.0 International License**.  
Non-commercial use and distribution are permitted. **Commercial use requires permission**, except for the original author (**Hyunkoo Kim**), who retains full rights for educational, business, and commercial use.




