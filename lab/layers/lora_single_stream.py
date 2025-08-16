#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F

p = torch.cuda.nvtx.range_push
q = torch.cuda.nvtx.range_pop


class BaseLayerWithLoRASingleStream(nn.Module):
    """
    Linear layer with LoRA (Low-Rank Adaptation) augmentation
    using a single CUDA stream.

    This module wraps a standard `nn.Linear` and adds a LoRA contribution:
      y = base(x) + (x @ A^T) @ B^T

    Attributes:
        base (nn.Linear): The base linear layer.
        lora_A (torch.Tensor): LoRA A matrix (low-rank projection).
        lora_B (torch.Tensor): LoRA B matrix (re-projection).
    """

    def __init__(self, base: nn.Linear, lora_A: torch.Tensor, lora_B: torch.Tensor):
        """
        Initialize the linear layer with LoRA weights.

        Args:
            base (nn.Linear): Base linear layer to wrap.
            lora_A (torch.Tensor): LoRA A matrix.
            lora_B (torch.Tensor): LoRA B matrix.
        """
        super().__init__()
        self.base = base
        self.lora_A = lora_A.contiguous()
        self.lora_B = lora_B.contiguous()

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the LoRA-augmented linear layer.

        Args:
            x (torch.Tensor): Input tensor of shape [B, S, I].

        Returns:
            torch.Tensor: Output tensor of shape [B, S, O].

        TODO:
            Implement the LoRA contribution using lora_A and lora_B.
        """
        B, S, I = x.shape

        base_out = F.linear(x, self.base.weight, self.base.bias)

        # TODO: implement LoRA contribution using lora_A and lora_B
        ...

        return base_out.view(B, S, -1).contiguous()


class VocabEmbeddingWithLoRASingleStream(nn.Module):
    """
    Embedding layer with LoRA (Low-Rank Adaptation) augmentation
    using a single CUDA stream.

    This module wraps a standard `nn.Embedding` and adds a LoRA contribution
    to its output embeddings.
    """

    def __init__(self, base: nn.Embedding, lora_A: torch.Tensor, lora_B: torch.Tensor):
        """
        Initialize the embedding layer with LoRA weights.

        Args:
            base (nn.Embedding): Base embedding layer to wrap.
            lora_A (torch.Tensor): LoRA A matrix.
            lora_B (torch.Tensor): LoRA B matrix.
        """
        super().__init__()
        self.base = base
        self.lora_A = lora_A.contiguous()
        self.lora_B = lora_B.contiguous()

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the LoRA-augmented embedding layer.

        Args:
            x (torch.Tensor): Input tensor of shape [B, S].

        Returns:
            torch.Tensor: Output tensor of shape [B, S, D].

        TODO:
            Implement the LoRA contribution for embeddings using lora_A and lora_B.
        """
        B, S = x.shape

        base_out = self.base(x)

        # TODO: implement LoRA contribution for embeddings
        ...

        return base_out.view(B, S, -1).contiguous()
