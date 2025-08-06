from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

import torch

# For building custom cuda kernel faster
compute_capability = [str(elem) for elem in torch.cuda.get_device_capability()]

CXX_FLAGS = ["-g", "-O2", "-std=c++17"]
NVCC_FLAGS = [f"-arch=sm_{''.join(compute_capability)}", "-std=c++17"]

ABI = 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0
CXX_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]
NVCC_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]

csrcs = [
    "csrc/wait_kernels.cu",
    "csrc/pybind.cpp",
]

ext_modules = []

cuda_ext = CUDAExtension(
    name="async_wait_kernel",
    sources=csrcs,
    extra_compile_args={
        "cxx": CXX_FLAGS,
        "nvcc": NVCC_FLAGS,
    },
)
ext_modules.append(cuda_ext)

setup(
    name="async_wait_kernel",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)