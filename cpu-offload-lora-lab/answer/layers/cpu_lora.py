#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F

import async_wait_kernel.cuda_utils as cuda_utils


class BaseLayerWithLoRA(nn.Module):
    def __init__(self, base: nn.Linear, lora_A: torch.Tensor, lora_B: torch.Tensor):
        super().__init__()
        self.base = base
        self.lora_A = lora_A.contiguous()
        self.lora_B = lora_B.contiguous()

        self.evt_prev_layer_done = torch.cuda.Event()
        self.evt_curr_layer_done = torch.cuda.Event()

        self.lora_stream = torch.cuda.Stream()
        self.evt_copy_done = torch.cuda.Event()
        self.evt_base_done = torch.cuda.Event()

        self.x_cpu = None
        self.cpu_lora_out_buffer = None
        self.gpu_lora_out_buffer = None

        self.sync_flag = torch.zeros(1, device="cuda:0", dtype=torch.int32)


    def forward(self, x: torch.Tensor):
        self.evt_prev_layer_done.record()
        self.lora_stream.wait_event(self.evt_prev_layer_done)

        B, S, _ = x.shape
        O = self.lora_B.size(0)
        flat_size = B * S
        buffer_shape = (flat_size, O)

        if self.x_cpu is None or self.x_cpu.shape != x.view(flat_size, -1).shape:
            self.x_cpu = torch.empty(x.view(flat_size, -1).shape, 
                                     device="cpu", dtype=x.dtype, pin_memory=True)

        if self.cpu_lora_out_buffer is None or self.cpu_lora_out_buffer.shape != buffer_shape:
            self.cpu_lora_out_buffer = torch.empty(buffer_shape, device="cpu", dtype=x.dtype, pin_memory=True)
            self.gpu_lora_out_buffer = torch.empty(buffer_shape, device=x.device, dtype=x.dtype)

        with torch.cuda.stream(self.lora_stream):
            self.x_cpu.copy_(x.view(flat_size, -1), non_blocking=True)
            self.evt_copy_done.record()

        base_out = F.linear(x, self.base.weight, self.base.bias)
        self.evt_base_done.record()

        with torch.cuda.stream(self.lora_stream):
            cuda_utils.wait(self.sync_flag)
            self.gpu_lora_out_buffer.copy_(self.cpu_lora_out_buffer, non_blocking=True)
            torch.cuda.current_stream().wait_event(self.evt_base_done)
            base_out.add_(self.gpu_lora_out_buffer.view(B, S, -1))
            self.evt_curr_layer_done.record()

        self.evt_copy_done.synchronize()
        torch.linalg.multi_dot([self.x_cpu, self.lora_A.t(), self.lora_B.t()], out=self.cpu_lora_out_buffer)

        self.sync_flag.fill_(1)
        torch.cuda.current_stream().wait_event(self.evt_curr_layer_done)

        return base_out.contiguous()


class VocabEmbeddingWithLoRA(nn.Module):
    def __init__(self, base: nn.Embedding, lora_A: torch.Tensor, lora_B: torch.Tensor):
        super().__init__()
        self.base = base
        self.lora_A = lora_A.contiguous().to("cuda:0")
        self.lora_B = lora_B.contiguous().to("cuda:0")

        self.evt_prev_layer_done = torch.cuda.Event()
        self.evt_curr_layer_done = torch.cuda.Event()

        self.lora_stream = torch.cuda.Stream()
        self.evt_copy_done = torch.cuda.Event()
        self.evt_base_done = torch.cuda.Event()
        self.evt_lora_done = torch.cuda.Event()

        self.x_cpu = None
        self.cpu_lora_out_buffer = None
        self.gpu_lora_out_buffer = None

        self.sync_flag = torch.zeros(1, device="cuda:0", dtype=torch.int32)

    def forward(self, x: torch.Tensor): # TODO: implement lora computation from GPU to CPU
        self.evt_prev_layer_done.record()
        self.lora_stream.wait_event(self.evt_prev_layer_done)

        B, S = x.shape

        base_out = self.base(x)

        with torch.cuda.stream(self.lora_stream):
            ar = F.embedding(x.view(-1), self.lora_A.t())
            lora_out = torch.mm(ar, self.lora_B.t())
            self.evt_lora_done.record(self.lora_stream)

        torch.cuda.current_stream().wait_event(self.evt_lora_done)
        base_out.add_(lora_out.view(B, S, -1))

        return base_out.contiguous()