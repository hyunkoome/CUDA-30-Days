from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="cuda-study-day03-matrix-add",
    ext_modules=[
        CUDAExtension(
            name="cuda_study_matrix_add",
            sources=[
                'common/kernel_matrix_add.cu',
                'common/matrix_add_impl.cpp',
                'common/wrapper_python_bind.cpp',
            ],
            include_dirs=[
                "common",  # ✅ 추가
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
