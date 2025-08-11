#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F

import custom_kernel.cuda_utils as cu  # wait, set_flag, (clear_flag)

p = torch.cuda.nvtx.range_push
q = torch.cuda.nvtx.range_pop

class BaseLayerWithLoRACPU(nn.Module):
    def __init__(self, base: nn.Linear, lora_A: torch.Tensor, lora_B: torch.Tensor):
        super().__init__()
        self.base = base
        self.lora_A = lora_A.contiguous()
        self.lora_B = lora_B.contiguous()

        self.lora_stream = torch.cuda.Stream()
        self.flag_stream = torch.cuda.Stream()

        self.evt_prev_layer_done = torch.cuda.Event()
        self.evt_copy_done = torch.cuda.Event()
        self.evt_base_done = torch.cuda.Event()
        self.evt_add_done = torch.cuda.Event()

        cap = 128
        Din  = self.lora_A.size(1)
        Dout = self.lora_B.size(0)
        cpu_dtype = self.lora_A.dtype
        self.x_cpu   = torch.empty((cap, Din),  device="cpu",  dtype=cpu_dtype, pin_memory=True)
        self.cpu_out = torch.empty((cap, Dout), device="cpu",  dtype=cpu_dtype, pin_memory=True)
        self.gpu_out = torch.empty((cap, Dout), device="cuda", dtype=torch.bfloat16)

        self.flag = torch.zeros(1, device="cuda", dtype=torch.int32)
        self.flag_one = torch.ones(1, device="cuda", dtype=torch.int32)



    def forward(self, x: torch.Tensor):
        p("BaseLayerWithLoRACPU::forward")
        p("Shape & Views")
        cur_stream = torch.cuda.current_stream()
        self.flag.zero_()

        B, S, _ = x.shape
        N = B * S
        is_decode = True if N == B * 1 else False

        x_cpu_v   = self.x_cpu[:N, :]
        cpu_out_v = self.cpu_out[:N, :]
        gpu_out_v = self.gpu_out[:N, :]
        q()

        p("lora_stream wait defualt stream")
        with torch.cuda.stream(self.lora_stream):
            self.evt_prev_layer_done.record(cur_stream)
            self.evt_prev_layer_done.wait(self.lora_stream)
            x_cpu_v.copy_(x.view(N, -1), non_blocking=True)
            self.evt_copy_done.record()
        q()

        p("main::base_linear + record")
        base_out = F.linear(x, self.base.weight, self.base.bias)
        self.evt_base_done.record(cur_stream)
        q()

        if not is_decode:
            p("CPU Matrix Multiplication")
            self.evt_copy_done.synchronize()
            torch.linalg.multi_dot([x_cpu_v, self.lora_A.t(), self.lora_B.t()], out=cpu_out_v)
            q()

            p("Prefill::issue gpu kernels")
            with torch.cuda.stream(self.lora_stream):
                gpu_out_v.copy_(cpu_out_v, non_blocking=True)
                self.evt_base_done.wait(self.lora_stream)
                base_out.add_(gpu_out_v.view(B, S, -1))
                self.evt_add_done.record(self.lora_stream)
            q()

        else:
            p("Decode::Preissue gpu kernels")
            with torch.cuda.stream(self.lora_stream):
                cu.wait(self.flag)
                gpu_out_v.copy_(cpu_out_v, non_blocking=True)
                self.evt_copy_done.wait(self.lora_stream)
                base_out.add_(gpu_out_v.view(B, S, -1))
                self.evt_add_done.record(self.lora_stream)
            q()
            
            p("CPU Matrix Multiplication + Set flag 0 to 1")
            with torch.cuda.stream(self.flag_stream):
                self.evt_copy_done.synchronize()
                torch.linalg.multi_dot([x_cpu_v, self.lora_A.t(), self.lora_B.t()], out=cpu_out_v)
                self.flag.copy_(self.flag_one, non_blocking=True)    
            q()

        p("default stream waits lora_stream")
        self.evt_add_done.wait(cur_stream)
        q()
        
        q()

        return base_out.contiguous()