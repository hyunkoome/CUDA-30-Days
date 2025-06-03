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

| 구성 | 버전 |
|------|------|
| CUDA | 12.6 |
| PyTorch | 2.x |
| Python | 3.10+ |
| C++ | CMake 또는 g++ |
| 선택사항 | ONNXRuntime, TensorRT, WebGPU 환경 |

---

## 🗓️ 30일 커리큘럼

| Day | 커널 이름 | 핵심 내용 | 연동 및 비교 |
|-----|-----------|-----------|---------------|
| 01 | vec_add | 벡터 덧셈 | PyTorch + C++ |
| 02 | dot_product | 내적, reduction | PyTorch + C++ |
| 03 | matrix_add | 2D 행렬 합 | PyTorch + C++ |
| 04 | relu | 활성화 함수 | PyTorch |
| 05 | matrix_transpose | 전치 | memory coalescing |
| 06 | matmul_basic | naive GEMM | PyTorch |
| 07 | matmul_sharedmem | tiling | PyTorch + 성능 |
| 08 | global_avg_pool | GAP | PyTorch 비교 |
| 09 | batch_norm_forward | 정규화 | ONNX export |
| 10 | instance_norm | 채널별 정규화 | PyTorch |
| 11 | group_norm | 그룹 정규화 | TensorRT plugin |
| 12 | layernorm_forward | LayerNorm FWD | PyTorch |
| 13 | layernorm_backward | LayerNorm BWD | 성능 튜닝 |
| 14 | softmax_kernel | softmax | PyTorch |
| 15 | softmax_2d_kernel | Attention용 | PyTorch |
| 16 | einsum_qk_attn | QK^T | Attention 연산 |
| 17 | attention_softmax_fused | fused attention | 효율 비교 |
| 18 | gelu_kernel | GELU | activation 성능 |
| 19 | topk_kernel | Top-K | PyTorch |
| 20 | prefix_sum_kernel | scan | warp-level sum |
| 21 | warp_shuffle_max | warp max | warp shuffle |
| 22 | conv2d_basic | naive Conv | PyTorch |
| 23 | conv2d_shared_tile | tile + shared | PyTorch |
| 24 | conv2d_unroll | unroll 최적화 | 성능 비교 |
| 25 | spconv_sparse_gemm | sparse matmul | PyTorch |
| 26 | spconv_forward | sparse forward | TensorRT |
| 27 | spconv_backward | grad 구현 | 실습 |
| 28 | transformer_block | full block | residual 포함 |
| 29 | flash_attention | FlashAttention | memory opt |
| 30 | final_review_day | 전체 정리 | 성능 분석 |

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

이 실습은 실전 AI 커널 엔지니어링을 목표로 구성되었습니다.  
다양한 연산들을 최적화하고 직접 비교하며, 실무에 바로 활용할 수 있는 실력을 기릅니다.

---

