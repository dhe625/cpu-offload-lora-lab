#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseLayerWithLoRASingleStream(nn.Module):
    def __init__(self, base: nn.Linear, lora_A: torch.Tensor, lora_B: torch.Tensor):
        super().__init__()
        self.base = base
        self.lora_A = lora_A.contiguous()
        self.lora_B = lora_B.contiguous()

    def forward(self, x: torch.Tensor):
        B, S, I = x.shape

        base_out = F.linear(x, self.base.weight, self.base.bias)

        ar = torch.mm(x.view(-1, I), self.lora_A.t())
        base_out.view(B * S, -1).addmm_(ar, self.lora_B.t())

        return base_out.view(B, S, -1).contiguous()


class VocabEmbeddingWithLoRASingleStream(nn.Module):
    def __init__(self, base: nn.Embedding, lora_A: torch.Tensor, lora_B: torch.Tensor):
        super().__init__()
        self.base = base
        self.lora_A = lora_A.contiguous()
        self.lora_B = lora_B.contiguous()

    def forward(self, x: torch.Tensor):
        B, S = x.shape

        base_out = self.base(x)

        ar = F.embedding(x.view(-1), self.lora_A.t())
        base_out.view(B * S, -1).addmm_(ar,  self.lora_B.t())

        return base_out.view(B, S, -1).contiguous()