from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="cuda-study-day02-dot",  # pip 패키지 이름 (중요 X)
    ext_modules=[
        CUDAExtension(
            name="cuda_study_dot",  # import 시 사용하는 이름 (.so 파일 이름도 이걸로 생성됨)
            sources=[
                "dot_cuda_src/dot_wrapper.cu",
                "dot_cuda_src/shared_memory_float_kernel.cu",
                "dot_cuda_src/shared_memory_double_precision_kernel.cu",
                "dot_cuda_src/shared_memory_mixed_precision_kernel.cu",
                "dot_cuda_src/warp_shuffle_kernel.cu",
            ],
            include_dirs=[
                "/usr/include/python3.10",
                "/usr/include/x86_64-linux-gnu/python3.10",
            ],
            extra_compile_args={
                "cxx": ["-std=c++17"],
                "nvcc": ["-std=c++17"]
            },
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)
