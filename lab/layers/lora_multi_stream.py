#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F

p = torch.cuda.nvtx.range_push
q = torch.cuda.nvtx.range_pop


class BaseLayerWithLoRAMultiStream(nn.Module):
    """
    Linear layer with LoRA (Low-Rank Adaptation) augmentation
    using a separate CUDA stream for the LoRA path.

    Unlike the single-stream version, this design overlaps
    base computation and LoRA computation by running them
    on different CUDA streams, synchronized via events.

    Attributes:
        base (nn.Linear): The base linear layer.
        lora_A (torch.Tensor): LoRA A matrix (low-rank projection).
        lora_B (torch.Tensor): LoRA B matrix (re-projection).
        lora_stream (torch.cuda.Stream): CUDA stream dedicated to LoRA operations.
        evt_prev_layer_done (torch.cuda.Event): Marks completion of prior work.
        evt_lora_done (torch.cuda.Event): Marks completion of LoRA computation.
    """

    def __init__(self, base: nn.Linear, lora_A: torch.Tensor, lora_B: torch.Tensor):
        """
        Initialize the linear layer with LoRA weights and its own CUDA stream.

        Args:
            base (nn.Linear): Base linear layer to wrap.
            lora_A (torch.Tensor): LoRA A matrix.
            lora_B (torch.Tensor): LoRA B matrix.
        """
        super().__init__()
        self.base = base
        self.lora_A = lora_A.contiguous()
        self.lora_B = lora_B.contiguous()

        self.lora_stream = torch.cuda.Stream()
        self.evt_prev_layer_done = torch.cuda.Event()
        self.evt_lora_done = torch.cuda.Event()

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the LoRA-augmented linear layer with multi-stream.

        Args:
            x (torch.Tensor): Input tensor of shape [B, S, I].

        Returns:
            torch.Tensor: Output tensor of shape [B, S, O].

        TODO:
            1. Record an event on the current stream and make lora_stream wait for it,
               so LoRA starts only after the base input is ready.
            2. Launch LoRA matmul on lora_stream using lora_A and lora_B.
            3. Ensure synchronization (evt_lora_done) before returning.
        """
        # TODO: record an event on current_stream and make lora_stream wait for it,
        #       ensuring LoRA computation starts only after current_stream is done
        ...

        B, S, I = x.shape

        base_out = F.linear(x, self.base.weight, self.base.bias)

        # TODO: implement LoRA contribution using lora_A and lora_B
        ...

        return base_out.view(B, S, -1).contiguous()


class VocabEmbeddingWithLoRAMultiStream(nn.Module):
    """
    Embedding layer with LoRA (Low-Rank Adaptation) augmentation
    using a separate CUDA stream for LoRA path.

    This class wraps a standard `nn.Embedding` and overlaps base
    embedding lookup with LoRA computation.
    """

    def __init__(self, base: nn.Embedding, lora_A: torch.Tensor, lora_B: torch.Tensor):
        """
        Initialize the embedding layer with LoRA weights and its own CUDA stream.

        Args:
            base (nn.Embedding): Base embedding layer to wrap.
            lora_A (torch.Tensor): LoRA A matrix.
            lora_B (torch.Tensor): LoRA B matrix.
        """
        super().__init__()
        self.base = base
        self.lora_A = lora_A.contiguous()
        self.lora_B = lora_B.contiguous()

        self.lora_stream = torch.cuda.Stream()
        self.evt_prev_layer_done = torch.cuda.Event()
        self.evt_lora_done = torch.cuda.Event()

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the LoRA-augmented embedding layer with multi-stream.

        Args:
            x (torch.Tensor): Input tensor of shape [B, S].

        Returns:
            torch.Tensor: Output tensor of shape [B, S, D].

        TODO:
            1. Record an event on the current stream and make lora_stream wait for it.
            2. Launch LoRA contribution on lora_stream using lora_A and lora_B.
            3. Synchronize with evt_lora_done before returning.
        """
        # TODO: record an event on current_stream and make lora_stream wait for it,
        #       ensuring LoRA computation starts only after current_stream is done
        ...

        B, S = x.shape

        base_out = self.base(x)

        # TODO: implement LoRA contribution using lora_A and lora_B
        ...

        return base_out.view(B, S, -1).contiguous()
