#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F

import custom_kernel.cuda_utils as cu  # wait(), set_flag(), (clear_flag)


class BaseLayerWithLoRACPU(nn.Module):
    """
    CPU-offloaded LoRA layer that overlaps CPU LoRA matmul with GPU base matmul
    using CUDA streams and events.

    Notes for students (exercise template):
      - The base path (F.linear) is implemented and runs on the default CUDA stream.
      - You must implement the LoRA path with proper stream/event synchronization.
      - Prefer event waits on streams over host-side synchronize() to keep pipelines overlapped.
    """

    def __init__(self, base: nn.Linear, lora_A: torch.Tensor, lora_B: torch.Tensor):
        super().__init__()
        self.base = base
        self.lora_A = lora_A.contiguous()
        self.lora_B = lora_B.contiguous()

        # Streams
        self.lora_stream = torch.cuda.Stream()
        self.flag_stream = torch.cuda.Stream()

        # Events
        self.evt_prev_layer_done = torch.cuda.Event()
        self.evt_copy_done = torch.cuda.Event()
        self.evt_base_done = torch.cuda.Event()
        self.evt_add_done = torch.cuda.Event()

        # Pre-allocated pinned CPU / GPU work buffers (capacity can be grown if needed)
        cap = 128
        Din = self.lora_A.size(1)
        Dout = self.lora_B.size(0)
        cpu_dtype = self.lora_A.dtype
        gpu_dtype = base.weight.dtype

        self.x_cpu   = torch.empty((cap, Din),  device="cpu",  dtype=cpu_dtype, pin_memory=True)
        self.cpu_out = torch.empty((cap, Dout), device="cpu",  dtype=cpu_dtype, pin_memory=True)
        self.gpu_out = torch.empty((cap, Dout), device="cuda", dtype=gpu_dtype)

        # GPU-side flag used by a custom wait-kernel (cu.wait) to gate decode pre-issue path
        self.flag = torch.zeros(1, device="cuda", dtype=torch.int32)
        self.flag_one = torch.ones(1, device="cuda", dtype=torch.int32)

    def forward(self, x: torch.Tensor):
        """
        Forward pass.

        The base linear computation runs on the default stream.
        LoRA contribution is computed on CPU, transferred to GPU, and added to base_out
        with proper stream/event synchronization.

        You must fill in the TODO sections to:
          1) Record an event on the current stream and make lora_stream wait for it.
          2) Asynchronously copy inputs to pinned CPU buffers and record evt_copy_done.
          3) Record evt_base_done after the base path completes.
          4) Implement Prefill path (B*S > B): CPU LoRA matmul → copy to GPU → wait base → add → record evt_add_done.
          5) Implement Decode path (B*S == B): pre-issue add on lora_stream gated by flag; run CPU matmul on flag_stream; set flag=1.
          6) Make current stream wait for evt_add_done before returning.
        """

        # Shapes and views
        B, S, _ = x.shape
        N = B * S
        is_decode = True if N == B * 1 else False

        x_cpu_v   = self.x_cpu[:N, :]
        cpu_out_v = self.cpu_out[:N, :]
        gpu_out_v = self.gpu_out[:N, :]

        cur_stream = torch.cuda.current_stream()
        self.flag.zero_()

        # TODO: Make lora_stream wait for current_stream (previous layer done), then
        #       async-copy x -> x_cpu_v (pinned) and record evt_copy_done.
        ...

        base_out = F.linear(x, self.base.weight, self.base.bias)

        # TODO: Record base completion on current stream
        ...

        if not is_decode:
            self.evt_copy_done.synchronize()
            torch.linalg.multi_dot([x_cpu_v, self.lora_A.t(), self.lora_B.t()], out=cpu_out_v)

            with torch.cuda.stream(self.lora_stream):
                gpu_out_v.copy_(cpu_out_v, non_blocking=True)
                self.evt_base_done.wait(self.lora_stream)
                base_out.add_(gpu_out_v.view(B, S, -1))
                self.evt_add_done.record(self.lora_stream)
        else:
            # TODO (DECODE): issue GPU kernels first, then run CPU matmul
            ...


        # TODO: Make current stream wait for LoRA add completion before returning
        ...

        return base_out.contiguous()
