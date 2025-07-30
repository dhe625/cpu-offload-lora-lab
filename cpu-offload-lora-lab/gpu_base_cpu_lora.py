#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GPU(base) + CPU(LoRA) 데모 (이벤트 동기화만 사용)
- base 연산은 GPU, LoRA 보정(A·B)은 CPU
- torch.cuda.Event/Stream으로 D2H/H2D 동기화
"""

import argparse
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from load_lora_adapter import LoRAModelManager


# -------------------- Utilities --------------------
def read_scaling(lora_dir: str) -> float:
    for fname in ("adapter_config.json", "peft_config.json"):
        p = os.path.join(lora_dir, fname)
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            alpha = cfg.get("lora_alpha", cfg.get("alpha"))
            r = cfg.get("r", cfg.get("rank"))
            if alpha and r:
                return float(alpha) / float(r)
    return 1.0


LINEAR_OPS = [
    ("self_attn", "q_proj"), ("self_attn", "k_proj"),
    ("self_attn", "v_proj"), ("self_attn", "o_proj"),
    ("mlp", "gate_proj"), ("mlp", "up_proj"), ("mlp", "down_proj"),
]


# -------------------- Wrappers (events only) --------------------
class BaseLayerWithLoRA(nn.Module):
    """nn.Linear: GPU(base) + CPU(LoRA), 단일 스레드 + 이벤트 동기화."""

    def __init__(self, base: nn.Linear, A: torch.Tensor, B: torch.Tensor, scale: float):
        super().__init__()
        self.base = base
        self.A_cpu = A.to("cpu").contiguous()
        self.B_cpu = B.to("cpu").contiguous()
        self.scale = float(scale)

        self.copy_stream = torch.cuda.Stream()
        self.evt_copy_done = torch.cuda.Event()
        self.evt_lora_ready = torch.cuda.Event()

        self.delta_gpu = None
        self._buf_shape = None

    def _ensure_delta_buf(self, shape, device, dtype):
        if self.delta_gpu is None or self._buf_shape != tuple(shape):
            self.delta_gpu = torch.empty(shape, device=device, dtype=dtype)
            self._buf_shape = tuple(shape)

    def forward(self, x: torch.Tensor):
        assert x.is_cuda
        x2 = x.reshape(-1, self.base.in_features)

        base_out = F.linear(x2, self.base.weight, self.base.bias)
        self._ensure_delta_buf((x2.shape[0], self.base.out_features), base_out.device, base_out.dtype)

        with torch.cuda.stream(self.copy_stream):
            x_cpu_pinned = torch.empty_like(x2, device="cpu", pin_memory=True)
            x_cpu_pinned.copy_(x2, non_blocking=True)
            self.evt_copy_done.record(self.copy_stream)

        self.evt_copy_done.synchronize()
        x_cpu = x_cpu_pinned.to(dtype=torch.float32, copy=False)
        delta_cpu = (x_cpu @ self.A_cpu.to(dtype=torch.float32)) @ self.B_cpu.to(dtype=torch.float32)
        if self.scale != 1.0:
            delta_cpu.mul_(self.scale)

        with torch.cuda.stream(self.copy_stream):
            self.delta_gpu.copy_(delta_cpu.to(self.delta_gpu.dtype), non_blocking=True)
            self.evt_lora_ready.record(self.copy_stream)

        stream_gpu = torch.cuda.current_stream(device=x.device)
        stream_gpu.wait_event(self.evt_lora_ready)
        out = base_out.add(self.delta_gpu)
        return out.reshape(x.shape[:-1] + (self.base.out_features,))


class GpuCpuEmbedding(nn.Module):
    """nn.Embedding: GPU(base) + CPU(LoRA), 단일 스레드 + 이벤트 동기화."""

    def __init__(self, base: nn.Embedding, A: torch.Tensor, B: torch.Tensor, scale: float):
        super().__init__()
        self.base = base
        self.A_cpu = A.to("cpu").contiguous()
        self.B_cpu = B.to("cpu").contiguous()
        self.scale = float(scale)

        self.copy_stream = torch.cuda.Stream()
        self.evt_copy_done = torch.cuda.Event()
        self.evt_lora_ready = torch.cuda.Event()

        self.delta_gpu = None
        self._buf_shape = None

    def _ensure_delta_buf(self, shape, device, dtype):
        if self.delta_gpu is None or self._buf_shape != tuple(shape):
            self.delta_gpu = torch.empty(shape, device=device, dtype=dtype)
            self._buf_shape = tuple(shape)

    def forward(self, ids: torch.LongTensor):
        assert ids.is_cuda and ids.dtype == torch.long

        base_out = self.base(ids)  # (B,S,H)
        Bsz, S, H = base_out.shape
        self._ensure_delta_buf((Bsz, S, H), base_out.device, base_out.dtype)

        with torch.cuda.stream(self.copy_stream):
            ids_cpu_pinned = torch.empty((Bsz * S,), dtype=torch.long, device="cpu", pin_memory=True)
            ids_cpu_pinned.copy_(ids.reshape(-1), non_blocking=True)
            self.evt_copy_done.record(self.copy_stream)

        self.evt_copy_done.synchronize()
        rows = self.A_cpu.index_select(0, ids_cpu_pinned)  # (N, r)
        delta_cpu = rows.to(dtype=torch.float32) @ self.B_cpu.to(dtype=torch.float32)  # (N, H)
        if self.scale != 1.0:
            delta_cpu.mul_(self.scale)

        with torch.cuda.stream(self.copy_stream):
            self.delta_gpu.copy_(delta_cpu.to(self.delta_gpu.dtype).view(Bsz, S, H), non_blocking=True)
            self.evt_lora_ready.record(self.copy_stream)

        stream_gpu = torch.cuda.current_stream(device=ids.device)
        stream_gpu.wait_event(self.evt_lora_ready)
        return base_out.add(self.delta_gpu)


# -------------------- Apply wrappers --------------------
def wrap_model(model: AutoModelForCausalLM, lora_dir: str, skip_embed: bool):
    mgr = LoRAModelManager(lora_dir)
    scale = read_scaling(lora_dir)

    A_e, B_e = mgr.get_embedding_AB()
    if not skip_embed and A_e is not None and B_e is not None:
        model.model.embed_tokens = GpuCpuEmbedding(model.model.embed_tokens, A_e, B_e, scale)

    for i, layer in enumerate(model.model.layers):
        for block, op in LINEAR_OPS:
            A, B = mgr.get_linear_AB(i, block, op)
            if A is None or B is None:
                continue
            sub = getattr(getattr(layer, block), op)
            setattr(getattr(layer, block), op, BaseLayerWithLoRA(sub, A, B, scale))

    A_l, B_l = mgr.get_lm_head_AB()
    if A_l is not None and B_l is not None:
        model.lm_head = BaseLayerWithLoRA(model.lm_head, A_l, B_l, scale)


# -------------------- Main --------------------
def main():
    parser = argparse.ArgumentParser(
        description="Inference with GPU base layers and CPU LoRA adapters (events-only sync)"
    )
    parser.add_argument("--base-model-id", default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--lora-dir", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--skip-embed", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_id, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=False
    ).eval()

    wrap_model(model, lora_dir=args.lora_dir, skip_embed=args.skip_embed)

    inputs = tokenizer(
        args.prompt, return_tensors="pt", padding=True, return_attention_mask=True
    ).to(model.device)

    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": args.do_sample,
    }
    if args.do_sample:
        gen_kwargs.update({"temperature": args.temperature, "top_p": args.top_p})

    output_ids = model.generate(**inputs, **gen_kwargs)
    text = tokenizer.decode(output_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(text)


if __name__ == "__main__":
    main()