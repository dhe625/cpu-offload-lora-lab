import os
from torch.utils.cpp_extension import load

_THIS = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_THIS, "csrc", "busy_wait.cu")

_ext = load(
    name="busy_wait_ext",
    sources=[_SRC],            # ← 모듈 정의 포함 파일 ‘하나’만
    extra_cuda_cflags=["-O3", "-std=c++17"],
    verbose=True,
)

def wait(flag):
    return _ext.wait(flag)

def set_flag(flag):
    return _ext.set_flag(flag)

def clear_flag(flag):
    return _ext.clear_flag(flag)
