#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GPU(base) + CPU(LoRA)
"""

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from load_lora_adapter import LoRAModelManager
import safetensors.torch


from torch.cuda import nvtx

# -------------------- Wrappers (events only) --------------------
class BaseLayerWithLoRA(nn.Module):

    def __init__(self, base: nn.Linear, lora_A: torch.Tensor, lora_B: torch.Tensor):
        super().__init__()
        self.base = base
        self.lora_A = lora_A.contiguous()
        self.lora_B = lora_B.contiguous()

        self.copy_stream = torch.cuda.Stream()
        self.evt_prev_layer_done = torch.cuda.Event(enable_timing=False)
        self.evt_copy_done = torch.cuda.Event(enable_timing=False)
        self.evt_base_ready = torch.cuda.Event(enable_timing=False)
        self.evt_lora_ready = torch.cuda.Event(enable_timing=False)

        self.x_cpu_pinned = None
        self.lora_out_cpu_pinned = None
        self.lora_out_cached = None


    def forward(self, x: torch.Tensor):
        nvtx.range_push("BaseLayerWithLoRA")
        B, S, I = x.shape
        O = self.lora_B.size(0)

        # ---------- 0) 1회만(또는 shape 변경시에만) 버퍼 할당 ----------
        if (self.lora_out_cached is None) or (self.lora_out_cached.shape[:2] != (B, S)):
            y_shape = (B, S, O)
            self.x_cpu_pinned = torch.empty((B, S, I), dtype=x.dtype, device="cpu", pin_memory=True)
            self.lora_out_cpu_pinned = torch.empty(y_shape, dtype=x.dtype, device="cpu", pin_memory=True)
            self.lora_out_cached = torch.empty(y_shape, dtype=x.dtype, device=x.device)

        # ---------- 1) 입력 준비(이전 레이어 완료) 이벤트 기록: compute_stream ----------
        nvtx.range_push("record_prev_layer_done")
        self.evt_prev_layer_done.record()
        nvtx.range_pop()

        # ---------- 2) D2H: x -> x_cpu_pinned (copy_stream, 비동기) ----------
        # copy_stream이 evt_prev_layer_done을 '미리' 대기하게 하고, 복사를 큐잉
        nvtx.range_push("async_copy_to_cpu")
        self.copy_stream.wait_event(self.evt_prev_layer_done)
        with torch.cuda.stream(self.copy_stream):
            # GPU -> CPU(pinned) 비동기 복사
            self.x_cpu_pinned.copy_(x, non_blocking=True)
            # D2H 완료 이벤트 기록
            self.evt_copy_done.record(self.copy_stream)
        nvtx.range_pop()

        # ---------- 3) GPU 베이스 연산: compute_stream ----------
        nvtx.range_push("base_linear")
        # 현재 compute_stream 상에서 실행됨 (별도 with 필요 없음)
        base_out = F.linear(x, self.base.weight, self.base.bias)
        nvtx.range_pop()

        # ---------- 4) CPU LoRA 연산 ----------
        # CPU는 CUDA 이벤트를 직접 기다릴 수 없으므로, 여기서 1회 동기화 필요
        # 이 시점에는 GPU가 base_out 계산 중이므로 파이프라인은 겹침
        nvtx.range_push("wait_copy_done_cpu_block")
        self.evt_copy_done.synchronize()  # D2H 완료 확인
        nvtx.range_pop()

        nvtx.range_push("lora_cpu_gemm")
        # einsum("bsi,ri,or->bso") 대신 BLAS-friendly 2단 matmul:
        #   (B,S,I) x (I,R)^T -> (B,S,R) -> (B,S,O)
        x_flat = self.x_cpu_pinned.view(-1, I) # (B*S, I)
        ar = torch.matmul(x_flat, self.lora_A.t()) # (B*S, R)
        out = torch.matmul(ar, self.lora_B.t()).view(B, S, O) # (B,S,O)
        self.lora_out_cpu_pinned.copy_(out)
        nvtx.range_pop()

        # ---------- 5) H2D: lora_out_cpu_pinned -> lora_out_cached (copy_stream, 비동기) ----------
        nvtx.range_push("async_copy_to_gpu")
        with torch.cuda.stream(self.copy_stream):
            self.lora_out_cached.copy_(self.lora_out_cpu_pinned, non_blocking=True)
            self.evt_lora_ready.record(self.copy_stream)   # H2D 완료 시그널
        nvtx.range_pop()

        # ---------- 6) (미리 큐잉) compute_stream에서 lora_ready 대기 후 in-place add ----------
        nvtx.range_push("preissue_wait_then_add")
        torch.cuda.current_stream().wait_event(self.evt_lora_ready)  # 의존성만 걸고 블록하지 않음
        # in-place add (AXPY 스타일) → 메모리 대역폭 절약
        base_out.add_(self.lora_out_cached[:, :, :base_out.size(-1)])
        nvtx.range_pop()
        nvtx.range_pop()

        return base_out.contiguous()


class VocabEmbeddingWithLoRA(nn.Module):
    def __init__(self, base: nn.Embedding, lora_A: torch.Tensor, lora_B: torch.Tensor):
        super().__init__()
        self.base = base
        self.lora_A = lora_A.contiguous()
        self.lora_B = lora_B.contiguous()

        self.copy_stream = torch.cuda.Stream()
        self.evt_prev_layer_done = torch.cuda.Event(enable_timing=False)
        self.evt_copy_done = torch.cuda.Event(enable_timing=False)
        self.evt_base_ready = torch.cuda.Event(enable_timing=False)
        self.evt_lora_ready = torch.cuda.Event(enable_timing=False)

        self.x_cpu_pinned = None
        self.lora_out_cpu_pinned = None
        self.lora_out_cached = None

    def forward(self, x: torch.Tensor):
        nvtx.range_push("VocabEmbeddingWithLoRA")
        B, S = x.shape
        O = self.lora_B.size(0)

        # ---------- 0) 1회만(또는 shape 변경시에만) 버퍼 할당 ----------
        if (self.lora_out_cached is None) or (self.lora_out_cached.shape[:2] != (B, S)):
            y_shape = (B, S, O)
            self.x_cpu_pinned = torch.empty((B, S), dtype=x.dtype, device="cpu", pin_memory=True)
            self.lora_out_cpu_pinned = torch.empty(y_shape, dtype=x.dtype, device="cpu", pin_memory=True)
            self.lora_out_cached = torch.empty(y_shape, dtype=x.dtype, device=x.device)

        # ---------- 1) 입력 준비(이전 레이어 완료) 이벤트 기록: compute_stream ----------
        nvtx.range_push("record_prev_layer_done")
        self.evt_prev_layer_done.record()
        nvtx.range_pop()

        # ---------- 2) D2H: x -> x_cpu_pinned (copy_stream, 비동기) ----------
        # copy_stream이 evt_prev_layer_done을 '미리' 대기하게 하고, 복사를 큐잉
        nvtx.range_push("async_copy_to_cpu")
        self.copy_stream.wait_event(self.evt_prev_layer_done)
        with torch.cuda.stream(self.copy_stream):
            # GPU -> CPU(pinned) 비동기 복사
            self.x_cpu_pinned.copy_(x, non_blocking=True)
            # D2H 완료 이벤트 기록
            self.evt_copy_done.record(self.copy_stream)
        nvtx.range_pop()

        # ---------- 3) GPU 베이스 연산: compute_stream ----------
        nvtx.range_push("base_linear")
        # 현재 compute_stream 상에서 실행됨 (별도 with 필요 없음)
        base_out = self.base(x)
        nvtx.range_pop()

        # ---------- 4) CPU LoRA 연산 ----------
        # CPU는 CUDA 이벤트를 직접 기다릴 수 없으므로, 여기서 1회 동기화 필요
        # 이 시점에는 GPU가 base_out 계산 중이므로 파이프라인은 겹침
        nvtx.range_push("wait_copy_done_cpu_block")
        self.evt_copy_done.synchronize()  # D2H 완료 확인
        nvtx.range_pop()

        nvtx.range_push("lora_cpu_gemm")
        # einsum("bsi,ri,or->bso") 대신 BLAS-friendly 2단 matmul:
        #   (B,S) ~ (V,R)^T -> (B,S,R) -> (B,S,O)
        x_flat = self.x_cpu_pinned.view(-1) # (B*S, I)
        ar = F.embedding(x_flat, self.lora_A.t())
        out = torch.matmul(ar, self.lora_B.t()).view(B, S, O) # (B,S,O)
        self.lora_out_cpu_pinned.copy_(out)
        nvtx.range_pop()

        # ---------- 5) H2D: lora_out_cpu_pinned -> lora_out_cached (copy_stream, 비동기) ----------
        nvtx.range_push("async_copy_to_gpu")
        with torch.cuda.stream(self.copy_stream):
            self.lora_out_cached.copy_(self.lora_out_cpu_pinned, non_blocking=True)
            self.evt_lora_ready.record(self.copy_stream)   # H2D 완료 시그널
        nvtx.range_pop()

        # ---------- 6) (미리 큐잉) compute_stream에서 lora_ready 대기 후 in-place add ----------
        nvtx.range_push("preissue_wait_then_add")
        torch.cuda.current_stream().wait_event(self.evt_lora_ready)  # 의존성만 걸고 블록하지 않음
        # in-place add (AXPY 스타일) → 메모리 대역폭 절약
        base_out.add_(self.lora_out_cached[:, :, :base_out.size(-1)])
        nvtx.range_pop()
        nvtx.range_pop()

        return base_out.contiguous()


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
            lora_A, lora_B = lora_manager.get_linear_AB(layer_idx, block_name, op_name)
            wrapped_module = BaseLayerWithLoRA(base_module, lora_A, lora_B)
            setattr(target_module, op_name, wrapped_module)

    # 2. Wrap embedding layer
    if not skip_embed:
        base_embedding = model.model.embed_tokens
        lora_A, lora_B = lora_manager.get_embedding_AB()
        if lora_A is not None and lora_B is not None:
            wrapped_embedding = VocabEmbeddingWithLoRA(base_embedding, lora_A, lora_B)
            model.model.embed_tokens = wrapped_embedding

    # 3. Wrap lm_head
    base_lm_head = model.lm_head
    lora_A, lora_B = lora_manager.get_lm_head_AB()
    if lora_A is not None and lora_B is not None:
        wrapped_lm_head = BaseLayerWithLoRA(base_lm_head, lora_A, lora_B)
        model.lm_head = wrapped_lm_head



# -------------------- Main --------------------
def main():
    parser = argparse.ArgumentParser(
        description="Inference with GPU base layers and CPU LoRA adapters"
    )
    parser.add_argument("--base-model-id",
                        default="meta-llama/Llama-2-7b-hf",
                        help="Base model ID to load from HF")
    parser.add_argument("--lora-dir", required=True,
                        help="Path to LoRA snapshot directory")
    parser.add_argument("--prompt", required=True,
                        help="Text prompt to generate from")
    args = parser.parse_args()

    # 1) 토크나이저 + 모델 로드
    tokenizer = AutoTokenizer.from_pretrained(args.lora_dir, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16 if device.startswith("cuda") else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_id,
        device_map={"": 0} if device.startswith("cuda") else None,
        torch_dtype=dtype,
        trust_remote_code=False
    )
    # 임베딩 크기 맞추기
    model.resize_token_embeddings(len(tokenizer))

    # 2) extra vocab 임베딩 덮어쓰기
    loaded = safetensors.torch.load_file(f"{args.lora_dir}/new_embeddings.safetensors")
    new_inp = loaded["input_embeddings"].to(device=device, dtype=dtype)
    new_out = loaded["output_embeddings"].to(device=device, dtype=dtype)
    with torch.no_grad():
        inp_emb = model.get_input_embeddings().weight
        inp_emb[-new_inp.size(0):] = new_inp

        # lm_head 가 weight tying 되어 있지 않으면 get_output_embeddings() 사용
        lm_w = (model.lm_head.weight
                if hasattr(model, "lm_head") and model.lm_head.weight.shape == inp_emb.shape
                else model.get_output_embeddings().weight)
        lm_w[-new_out.size(0):] = new_out

    # 3) LoRA 래핑
    model.eval()
    wrap_model(model, lora_dir=args.lora_dir, skip_embed=False)

    # 4) 토크나이징
    inputs = tokenizer(
        args.prompt,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)

    # 5) vLLM 예제와 동일한 샘플링 파라미터로 generate()
    stop_id = tokenizer.convert_tokens_to_ids("[/assistant]")  # 32003
    gen_out = model.generate(
        **inputs,
        max_new_tokens=128,
        temperature=0.0,                # greedy
        do_sample=False,
        eos_token_id=[stop_id],         # stop on [/assistant]
        output_scores=True,             # logits/scores 리턴
        return_dict_in_generate=True
    )

    # 7) 디코딩
    text = tokenizer.decode(
        gen_out.sequences[0,
                         inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )
    print("\n=== Generated Text ===")
    print(text)
    print("========================")

if __name__ == "__main__":
    main()