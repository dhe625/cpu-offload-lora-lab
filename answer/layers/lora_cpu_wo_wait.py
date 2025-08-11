#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F

from contextlib import contextmanager
from torch.cuda import nvtx as torch_nvtx

@contextmanager
def nvtx_range(msg: str):
    torch_nvtx.range_push(msg)
    try:
        yield
    finally:
        torch_nvtx.range_pop()

class BaseLayerWithLoRACPU(nn.Module):
    def __init__(self, base: nn.Linear, lora_A: torch.Tensor, lora_B: torch.Tensor):
        super().__init__()
        self.base = base
        self.lora_A = lora_A.contiguous()
        self.lora_B = lora_B.contiguous()

        # Streams & events
        self.lora_stream  = torch.cuda.Stream()
        self.evt_copy_done = torch.cuda.Event()
        self.evt_lora_done = torch.cuda.Event()

        cap = 256
        self.x_cpu  = torch.empty((cap, self.lora_A.size(1)), device="cpu",  dtype=self.lora_A.dtype, pin_memory=True)
        self.cpu_out= torch.empty((cap, self.lora_B.size(0)), device="cpu",  dtype=self.lora_B.dtype, pin_memory=True)
        self.gpu_out= torch.empty((cap, self.lora_B.size(0)), device="cuda:0", dtype=self.lora_B.dtype)

    def forward(self, x: torch.Tensor):
        with nvtx_range("shape&views"):
            B, S, D = x.shape
            N = B * S
            x_cpu_v   = self.x_cpu[:N, :]
            cpu_out_v = self.cpu_out[:N, :]
            gpu_out_v = self.gpu_out[:N, :]

        with nvtx_range("streams::lora_wait_current"):
            self.lora_stream.wait_stream(torch.cuda.current_stream())

        with nvtx_range("lora_stream::D2H_x_copy+record"):
            with torch.cuda.stream(self.lora_stream):
                with nvtx_range("D2H x copy (non_blocking)"):
                    x_cpu_v.copy_(x.view(N, D), non_blocking=True)
                with nvtx_range("record evt_copy_done"):
                    self.evt_copy_done.record()

        with nvtx_range("GPU::base_linear+record"):
            base_out = F.linear(x, self.base.weight, self.base.bias)

        with nvtx_range("CPU::wait evt_copy_done"):
            self.evt_copy_done.synchronize()
        with nvtx_range("CPU::LoRA multi_dot"):
            torch.linalg.multi_dot([x_cpu_v, self.lora_A.t(), self.lora_B.t()], out=cpu_out_v)

        with nvtx_range("lora_stream(from thread)::H2D lora_out + record"):
            with torch.cuda.stream(self.lora_stream):
                with nvtx_range("H2D lora_out (non_blocking)"):
                    gpu_out_v.copy_(cpu_out_v, non_blocking=True)
                with nvtx_range("record evt_lora_done"):
                    self.evt_lora_done.record()

        with nvtx_range("main_stream::wait evt_lora_done"):
            torch.cuda.current_stream().wait_event(self.evt_lora_done)

        with nvtx_range("main_stream::add_ lora_out"):
            base_out.add_(gpu_out_v.view(B, S, -1))

        return base_out.contiguous()