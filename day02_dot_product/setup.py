from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="dot_cuda",
    ext_modules=[
        CUDAExtension(
            name="dot_cuda",
            sources=[
                "dot_cuda/dot_wrapper.cu",
                "dot_cuda/shared_memory_float_kernel.cu",
                "dot_cuda/shared_memory_double_precision_kernel.cu",
                "dot_cuda/shared_memory_mixed_precision_kernel.cu",
                "dot_cuda/warp_shuffle_kernel.cu",
            ],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)
