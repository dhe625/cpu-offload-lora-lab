#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseLayerWithLoRAMultiStream(nn.Module):
    def __init__(self, base: nn.Linear, lora_A: torch.Tensor, lora_B: torch.Tensor):
        super().__init__()
        self.base = base
        self.lora_A = lora_A.contiguous()
        self.lora_B = lora_B.contiguous()

        self.lora_stream = torch.cuda.Stream()

        self.evt_prev_layer_done = torch.cuda.Event()
        self.evt_lora_done = torch.cuda.Event()

    def forward(self, x: torch.Tensor):
        self.evt_prev_layer_done.record()
        self.lora_stream.wait_event(self.evt_prev_layer_done)

        base_out = F.linear(x, self.base.weight, self.base.bias)

        with torch.cuda.stream(self.lora_stream):
            ar = F.linear(x, self.lora_A)
            lora_out = F.linear(ar, self.lora_B)
            self.evt_lora_done.record(self.lora_stream)

        torch.cuda.current_stream().wait_event(self.evt_lora_done)
        base_out.add_(lora_out)

        return base_out.contiguous()


class VocabEmbeddingWithLoRAMultiStream(nn.Module):
    def __init__(self, base: nn.Embedding, lora_A: torch.Tensor, lora_B: torch.Tensor):
        super().__init__()
        self.base = base
        self.lora_A = lora_A.contiguous()
        self.lora_B = lora_B.contiguous()

        self.lora_stream = torch.cuda.Stream()

        self.evt_prev_layer_done = torch.cuda.Event()
        self.evt_lora_done = torch.cuda.Event()

    def forward(self, x: torch.Tensor):
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