#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GPU(base) + CPU(LoRA) 데모 (이벤트 동기화만 사용)
- base 연산은 GPU, LoRA 보정(A·B)은 CPU
- torch.cuda.Event/Stream으로 D2H/H2D 동기화
"""

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from load_lora_adapter import LoRAModelManager


# -------------------- Wrappers (events only) --------------------
class BaseLayerWithLoRA(nn.Module):

    def __init__(self, base: nn.Linear, A: torch.Tensor, B: torch.Tensor):
        super().__init__()
        self.base = base
        self.A_cpu = A.to("cpu").contiguous()
        self.B_cpu = B.to("cpu").contiguous()
        self.lora_out_cached = None
        self.x_cpu_pinned = None
        self.lora_out_cpu_pinned = None

        self.copy_stream = torch.cuda.Stream()
        self.evt_prev_layer_done = torch.cuda.Event()
        self.evt_copy_done = torch.cuda.Event()
        self.evt_lora_ready = torch.cuda.Event()


    def forward(self, x: torch.Tensor):
        # TODO: implement the code below

        # Record event for previous layer's completion and wait on copy stream
        self.evt_prev_layer_done.record()
        self.evt_prev_layer_done.wait(self.copy_stream)

        # Asynchronously copy input to CPU for LoRA computation
        with torch.cuda.stream(self.copy_stream):
            # Initialize or resize out_cached tensor if needed
            if self.lora_out_cached is None or self.lora_out_cached.shape[1] != x.shape[1]:
                y_shape = x.shape[:-1] + (self.B_cpu.size(0),)
                self.x_cpu_pinned = torch.empty(x.shape, dtype=x.dtype, device='cpu', pin_memory=True)
                self.lora_out_cpu_pinned = torch.empty(y_shape, dtype=x.dtype, device='cpu', pin_memory=True)
                self.lora_out_cached = torch.empty(y_shape, dtype=x.dtype, device=x.device)

            self.x_cpu_pinned.copy_(x, non_blocking=True)
            self.evt_copy_done.record()

        # Perform base layer computation on GPU
        base_out = F.linear(x, self.base.weight, self.base.bias)

        # Synchronize with copy stream and perform LoRA computation on CPU
        self.evt_copy_done.synchronize()
        self.lora_out_cpu_pinned.copy_(torch.einsum('bsi,ri,or->bso', self.x_cpu_pinned, self.A_cpu, self.B_cpu))

        # Asynchronously copy LoRA output back to GPU
        with torch.cuda.stream(self.copy_stream):
            self.lora_out_cached.copy_(self.lora_out_cpu_pinned, non_blocking=True)
            self.evt_lora_ready.record()
        
        # Wait for LoRA output to be ready on GPU and add to base output
        torch.cuda.current_stream().wait_event(self.evt_lora_ready) # Ensure lora_out is on GPU before addition

        final_output = base_out + self.lora_out_cached[:, :, :base_out.size(-1)]
        return final_output.contiguous()


class VocabEmbeddingWithLoRA(nn.Module):
    def __init__(self, base: nn.Embedding, A: torch.Tensor, B: torch.Tensor):
        super().__init__()
        self.base = base
        self.A_cpu = A.to("cpu").contiguous()
        self.B_cpu = B.to("cpu").contiguous()
        self.lora_out_cached = None
        self.x_cpu_pinned = None
        self.lora_out_cpu_pinned = None

        self.copy_stream = torch.cuda.Stream()
        self.evt_prev_layer_done = torch.cuda.Event()
        self.evt_copy_done = torch.cuda.Event()
        self.evt_lora_ready = torch.cuda.Event()

    def forward(self, x: torch.Tensor):
        # TODO: implement the code below
        self.evt_prev_layer_done.record()
        self.evt_prev_layer_done.wait(self.copy_stream)

        with torch.cuda.stream(self.copy_stream):
            if self.lora_out_cached is None or self.lora_out_cached.shape[1] != x.shape[1]:
                y_shape = x.shape + (self.B_cpu.size(0),)
                self.x_cpu_pinned = torch.empty(x.shape, dtype=x.dtype, device='cpu', pin_memory=True)
                self.lora_out_cpu_pinned = torch.empty(y_shape, dtype=x.dtype, device='cpu', pin_memory=True)
                self.lora_out_cached = torch.empty(y_shape, dtype=x.dtype, device=x.device)

            self.x_cpu_pinned.copy_(x, non_blocking=True)
            self.evt_copy_done.record()

        base_out = self.base(x)

        self.evt_copy_done.synchronize()
        lora_A_out = F.embedding(self.x_cpu_pinned, self.A_cpu.T)
        self.lora_out_cpu_pinned.copy_(F.linear(lora_A_out, self.B_cpu))

        with torch.cuda.stream(self.copy_stream):
            self.lora_out_cached.copy_(self.lora_out_cpu_pinned, non_blocking=True)
            self.evt_lora_ready.record()

        torch.cuda.current_stream().wait_event(self.evt_lora_ready)

        final_output = base_out + self.lora_out_cached
        return final_output.contiguous()


# -------------------- Apply wrappers --------------------
def wrap_model(model: AutoModelForCausalLM, lora_dir: str, skip_embed: bool):
    linear_ops = [
        ("self_attn", "q_proj"), ("self_attn", "k_proj"),
        ("self_attn", "v_proj"), ("self_attn", "o_proj"),
        ("mlp", "gate_proj"), ("mlp", "up_proj"), ("mlp", "down_proj"),
    ]
    lora_manager = LoRAModelManager(lora_dir)

    # 1. Wrap linear layers
    for layer_idx, layer in enumerate(model.model.layers):
        for block_name, op_name in linear_ops:
            target_module = getattr(layer, block_name)
            base_module = getattr(target_module, op_name)
            A, B = lora_manager.get_linear_AB(layer_idx, block_name, op_name)
            wrapped_module = BaseLayerWithLoRA(base_module, A, B)
            setattr(target_module, op_name, wrapped_module)

    # 2. Wrap embedding layer
    if not skip_embed:
        base_embedding = model.model.embed_tokens
        A_embed, B_embed = lora_manager.get_embedding_AB()
        if A_embed is not None and B_embed is not None:
            wrapped_embedding = VocabEmbeddingWithLoRA(base_embedding, A_embed, B_embed)
            model.model.embed_tokens = wrapped_embedding

    # 3. Wrap lm_head
    base_lm_head = model.lm_head
    A_lm_head, B_lm_head = lora_manager.get_lm_head_AB()
    if A_lm_head is not None and B_lm_head is not None:
        wrapped_lm_head = BaseLayerWithLoRA(base_lm_head, A_lm_head, B_lm_head)
        model.lm_head = wrapped_lm_head



# -------------------- Main --------------------
def main():
    parser = argparse.ArgumentParser(
        description="Inference with GPU base layers and CPU LoRA adapters"
    )
    parser.add_argument("--base-model-id", default="meta-llama/Llama-2-7b-hf",
                        help="Base model ID to load from HF")
    parser.add_argument("--lora-dir", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--skip-embed", action="store_true")
    parser.add_argument("--no-cuda", action="store_true",
                        help="Disable CUDA and run entirely on CPU")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    device_is_cuda = torch.cuda.is_available() and not args.no_cuda
    device_map = "auto" if device_is_cuda else None
    dtype      = torch.bfloat16 if device_is_cuda else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_id,
        device_map=device_map,
        torch_dtype=dtype,
        trust_remote_code=False
    )
    model.eval()

    wrap_model(model, lora_dir=args.lora_dir, skip_embed=args.skip_embed)

    inputs = tokenizer(
        args.prompt,
        return_tensors="pt",
        padding=True,
        return_attention_mask=True
    ).to(model.device)

    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id":   tokenizer.pad_token_id,
        "eos_token_id":   tokenizer.eos_token_id,
        "do_sample":      args.do_sample,
    }
    if args.do_sample:
        gen_kwargs.update({
            "temperature": args.temperature,
            "top_p":       args.top_p,
        })

    output_ids = model.generate(**inputs, **gen_kwargs);
    text = tokenizer.decode(
        output_ids[0, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )
    print("Generated text:")
    print(text)

if __name__ == "__main__":
    main()