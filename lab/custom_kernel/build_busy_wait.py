#!/usr/bin/env python3
import os
from torch.utils.cpp_extension import load

_THIS = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.join(_THIS, "csrc", "busy_wait.cu")

print(f"Compiling {_SRC} -> busy_wait_ext ...")
_ext = load(
    name="busy_wait_ext",
    sources=[_SRC],
    extra_cuda_cflags=["-O3", "-std=c++17"],
    verbose=True,
)
print("busy_wait_ext build complete.")