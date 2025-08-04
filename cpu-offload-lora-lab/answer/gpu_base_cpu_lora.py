#!/usr/bin/env python
"""
LoRA adapter GPU/CPU hybrid offloading implementation.

- BaseLayerWithLoRA: CPU offload wrapper for nn.Linear layers
- VocabEmbeddingWithLoRA: CPU offload wrapper for nn.Embedding layers
- wrap_model: Replaces model modules with LoRA-enabled wrappers
- LoRAModelManager: Loads and manages LoRA A and B matrices from safetensors files
"""
import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import safetensors.torch
from huggingface_hub import snapshot_download

from load_lora_adapter import LoRAModelManager
from torch.cuda import nvtx

 # --- LoRA Wrapper Modules ---
class BaseLayerWithLoRA(nn.Module):
    """
    Wrapper for nn.Linear layers that offloads LoRA computations to CPU.

    - Performs base linear operation on GPU.
    - Offloads LoRA adapter computation to CPU.
    - Merges adapter output with base output asynchronously.
    """

    def __init__(self, base: nn.Linear, lora_A: torch.Tensor, lora_B: torch.Tensor):
        super().__init__()
        self.base = base # The base (pre-trained) linear layer
        self.lora_A = lora_A.contiguous() # LoRA A matrix (contiguous for performance)
        self.lora_B = lora_B.contiguous() # LoRA B matrix (contiguous for performance)

        # CUDA streams and events for asynchronous operations
        self.copy_stream = torch.cuda.Stream() # Dedicated stream for data transfers
        self.evt_prev_layer_done = torch.cuda.Event(enable_timing=False) # Marks when input `x` is ready on the compute stream
        self.evt_copy_done = torch.cuda.Event(enable_timing=False) # Marks when `x` has been copied from GPU to CPU
        self.evt_lora_ready = torch.cuda.Event(enable_timing=False) # Marks when LoRA output has been copied from CPU to GPU

        # Pinned memory buffers for efficient CPU-GPU data transfer
        self.x_cpu_pinned = None # Pinned memory for input tensor on CPU
        self.lora_out_cpu_pinned = None # Pinned memory for LoRA output tensor on CPU
        self.lora_out_cached = None # Cached GPU tensor for LoRA output


    def forward(self, x: torch.Tensor):
        nvtx.range_push("BaseLayerWithLoRA")
        B, S, I = x.shape # Batch size, Sequence length, Input dimension
        O = self.lora_B.size(0) # Output dimension

        # ---------- 1) Record event for input readiness (previous layer done) on compute_stream ----------
        nvtx.range_push("record_prev_layer_done")
        self.evt_prev_layer_done.record() # Record event on the default (compute) CUDA stream
        nvtx.range_pop()

        # ---------- 2) D2H: x (GPU) -> x_cpu_pinned (CPU, async on copy_stream) ----------
        # Make the copy_stream wait for the input `x` to be ready on the compute stream
        nvtx.range_push("async_copy_to_cpu")
        self.copy_stream.wait_event(self.evt_prev_layer_done)
        with torch.cuda.stream(self.copy_stream):
            # ---------- 2-1) Allocate buffers once (or if shape changes) ----------
            # Reallocate pinned memory buffers if batch size (B) or sequence length (S) changes
            if (self.lora_out_cached is None) or (self.lora_out_cached.shape[:2] != (B, S)):
                self._init_lora_tensor_cached(x)

            self.x_cpu_pinned.copy_(x, non_blocking=True) # Asynchronously copy `x` from GPU to CPU pinned memory
            self.evt_copy_done.record(self.copy_stream) # Record event on copy_stream when D2H copy is finished
        nvtx.range_pop() 

        # ---------- 3) GPU Base Computation: compute_stream ----------
        nvtx.range_push("base_linear")
        base_out = F.linear(x, self.base.weight, self.base.bias) # Perform the base linear operation on GPU (on the default stream)
        nvtx.range_pop()

        # ---------- 4) CPU LoRA Computation ----------
        # The CPU needs to wait for the D2H copy to complete before starting LoRA computation.
        # This allows for overlap: GPU computes `base_out` while CPU waits for `x_cpu_pinned`.
        nvtx.range_push("wait_copy_done_cpu_block")
        self.evt_copy_done.synchronize() # Synchronize CPU with the copy_stream's D2H completion event (blocks CPU)
        nvtx.range_pop()

        nvtx.range_push("lora_cpu_gemm")
        out = self._lora_cpu_matmul(self.x_cpu_pinned, self.lora_A, self.lora_B)
        self.lora_out_cpu_pinned.copy_(out) # Copy the computed LoRA output to CPU pinned memory (host-side copy)
        nvtx.range_pop()

        # ---------- 5) H2D: lora_out_cpu_pinned (CPU) -> lora_out_cached (GPU, async on copy_stream) ----------
        nvtx.range_push("async_copy_to_gpu")
        with torch.cuda.stream(self.copy_stream):
            self.lora_out_cached.copy_(self.lora_out_cpu_pinned, non_blocking=True) # Asynchronously copy LoRA output from CPU pinned to GPU
            self.evt_lora_ready.record(self.copy_stream) # Record event on copy_stream when H2D copy is finished
        nvtx.range_pop()

        # ---------- 6) (Pre-queued) compute_stream waits for lora_ready, then performs in-place add ----------
        nvtx.range_push("preissue_wait_then_add")
        torch.cuda.current_stream().wait_event(self.evt_lora_ready) # Establish dependency for the default (compute) stream, but does not block CPU
        # In-place add (AXPY style) to save memory bandwidth
        base_out.add_(self.lora_out_cached) # Add LoRA output to base output
        nvtx.range_pop()
        nvtx.range_pop()

        return base_out.contiguous()
    
    def _init_lora_tensor_cached(self, x: torch.Tensor):
        B, S, I= x.shape
        O = self.lora_B.size(0) # Output dimension
        y_shape = (B, S, O) # Output shape (Batch, Sequence, Output_dim)

        # Use BLAS-friendly two-stage matmul for embedding:
        # (B*S) @ (V, R)^T -> (B*S, R) @ (R, O)^T -> (B*S, O)
        self.x_cpu_pinned = torch.empty((B, S, I), dtype=x.dtype, device="cpu", pin_memory=True)
        self.lora_out_cpu_pinned = torch.empty(y_shape, dtype=x.dtype, device="cpu", pin_memory=True)
        self.lora_out_cached = torch.empty(y_shape, dtype=x.dtype, device=x.device)

    @torch.no_grad()
    def _lora_cpu_matmul(self, x_cpu_pinned: torch.Tensor, lora_A: torch.Tensor, lora_B: torch.Tensor) -> torch.Tensor:
        """
        Performs the LoRA matrix multiplication on CPU.

        Args:
            x_cpu_pinned (torch.Tensor): The input tensor, already copied to CPU pinned memory.
            lora_A (torch.Tensor): The LoRA A matrix.
            lora_B (torch.Tensor): The LoRA B matrix.

        Returns:
            torch.Tensor: The computed LoRA output tensor.
        """
        B, S, I = x_cpu_pinned.shape
        O = lora_B.size(0)

        x_flat = x_cpu_pinned.view(-1, I) # Reshape input for matrix multiplication
        ar = torch.matmul(x_flat, lora_A.t()) # First matmul: (B*S, I) @ (I, R) -> (B*S, R)
        out = torch.matmul(ar, lora_B.t()).view(B, S, O) # Second matmul: (B*S, R) @ (R, O) -> (B*S, O), then reshape to original dims
        return out



class VocabEmbeddingWithLoRA(nn.Module):
    """
    Wrapper for nn.Embedding layers that offloads LoRA computations to CPU.

    - Computes base embedding on GPU.
    - Offloads adapter computation to CPU.
    - Adds adapter output to base embedding asynchronously.
    """
    def __init__(self, base: nn.Embedding, lora_A: torch.Tensor, lora_B: torch.Tensor):
        super().__init__()
        self.base = base # The base (pre-trained) linear layer
        self.lora_A = lora_A.contiguous() # LoRA A matrix for embedding (contiguous for performance)
        self.lora_B = lora_B.contiguous() # LoRA B matrix for embedding (contiguous for performance)

        # CUDA streams and events for asynchronous operations
        self.copy_stream = torch.cuda.Stream() # Dedicated stream for data transfers
        self.evt_prev_layer_done = torch.cuda.Event(enable_timing=False) # Marks when input `x` is ready on the compute stream
        self.evt_copy_done = torch.cuda.Event(enable_timing=False) # Marks when `x` has been copied from GPU to CPU
        self.evt_lora_ready = torch.cuda.Event(enable_timing=False) # Marks when LoRA output has been copied from CPU to GPU

        # Pinned memory buffers for efficient CPU-GPU data transfer
        self.x_cpu_pinned = None # Pinned memory for input tensor on CPU
        self.lora_out_cpu_pinned = None # Pinned memory for LoRA output tensor on CPU
        self.lora_out_cached = None # Cached GPU tensor for LoRA output

    def forward(self, x: torch.Tensor):
        nvtx.range_push("VocabEmbeddingWithLoRA")
        B, S = x.shape # Batch size, Sequence length
        O = self.lora_B.size(0) # Output dimension

        # ---------- 1) Record event for input readiness (previous layer done) on compute_stream ----------
        nvtx.range_push("record_prev_layer_done")
        self.evt_prev_layer_done.record() # Record event on the default (compute) CUDA stream
        nvtx.range_pop()

        # ---------- 2) D2H: x (GPU) -> x_cpu_pinned (CPU, async on copy_stream) ----------
        # Make the copy_stream wait for the input `x` to be ready on the compute stream
        nvtx.range_push("async_copy_to_cpu")
        self.copy_stream.wait_event(self.evt_prev_layer_done)
        with torch.cuda.stream(self.copy_stream):
            # ---------- 2-1) Allocate buffers once (or if shape changes) ----------
            # Reallocate pinned memory buffers if batch size (B) or sequence length (S) changes
            if (self.lora_out_cached is None) or (self.lora_out_cached.shape[:2] != (B, S)):
                self._init_lora_tensor_cached(x)

            self.x_cpu_pinned.copy_(x, non_blocking=True) # Asynchronously copy `x` from GPU to CPU pinned memory
            self.evt_copy_done.record(self.copy_stream) # Record event on copy_stream when D2H copy is finished
        nvtx.range_pop()

        # ---------- 3) GPU Base Computation: compute_stream ----------
        nvtx.range_push("base_linear")
        base_out = self.base(x) # Perform the base embedding operation on GPU (on the default stream)
        nvtx.range_pop()

        # ---------- 4) CPU LoRA Computation ----------
        # The CPU needs to wait for the D2H copy to complete before starting LoRA computation.
        # This allows for overlap: GPU computes `base_out` while CPU waits for `x_cpu_pinned`.
        nvtx.range_push("wait_copy_done_cpu_block")
        self.evt_copy_done.synchronize() # Synchronize CPU with the copy_stream's D2H completion event (blocks CPU)
        nvtx.range_pop()

        nvtx.range_push("lora_cpu_gemm")
        out = self._lora_cpu_matmul(self.x_cpu_pinned, self.lora_A, self.lora_B)
        self.lora_out_cpu_pinned.copy_(out) # Copy the computed LoRA output to CPU pinned memory (host-side copy)
        nvtx.range_pop()

        # ---------- 5) H2D: lora_out_cpu_pinned (CPU) -> lora_out_cached (GPU, async on copy_stream) ----------
        nvtx.range_push("async_copy_to_gpu")
        with torch.cuda.stream(self.copy_stream):
            self.lora_out_cached.copy_(self.lora_out_cpu_pinned, non_blocking=True) # Asynchronously copy LoRA output from CPU pinned to GPU
            self.evt_lora_ready.record(self.copy_stream) # Record event on copy_stream when H2D copy is finished
        nvtx.range_pop()

        # ---------- 6) (Pre-queued) compute_stream waits for lora_ready, then performs in-place add ----------
        nvtx.range_push("preissue_wait_then_add")
        torch.cuda.current_stream().wait_event(self.evt_lora_ready) # Establish dependency for the default (compute) stream, but does not block CPU
        # In-place add (AXPY style) to save memory bandwidth
        base_out.add_(self.lora_out_cached) # Add LoRA output to base output
        nvtx.range_pop()
        nvtx.range_pop()

        return base_out.contiguous()
    
    def _init_lora_tensor_cached(self, x: torch.Tensor):
        B, S = x.shape
        O = self.lora_B.size(0) # Output dimension
        y_shape = (B, S, O)

        self.x_cpu_pinned = torch.empty((B, S), dtype=x.dtype, device="cpu", pin_memory=True)
        self.lora_out_cpu_pinned = torch.empty(y_shape, dtype=x.dtype, device="cpu", pin_memory=True)
        self.lora_out_cached = torch.empty(y_shape, dtype=x.dtype, device=x.device)

    @torch.no_grad()
    def _lora_cpu_matmul(self, x_cpu_pinned: torch.Tensor, lora_A: torch.Tensor, lora_B: torch.Tensor) -> torch.Tensor:
        """
        Performs the LoRA matrix multiplication for embedding layers on CPU.

        Args:
            x_cpu_pinned (torch.Tensor): The input tensor (token IDs), already copied to CPU pinned memory.
            lora_A (torch.Tensor): The LoRA A matrix (embedding matrix for LoRA).
            lora_B (torch.Tensor): The LoRA B matrix.

        Returns:
            torch.Tensor: The computed LoRA output tensor.
        """
        B, S = x_cpu_pinned.shape
        O = lora_B.size(0)

        # Use BLAS-friendly two-stage matmul for embedding:
        # (B*S) @ (V, R)^T -> (B*S, R) @ (R, O)^T -> (B*S, O)
        x_flat = x_cpu_pinned.view(-1) # Flatten input for embedding lookup
        ar = F.embedding(x_flat, lora_A.t()) # First step: embedding lookup with LoRA A
        out = torch.matmul(ar, lora_B.t()).view(B, S, O) # Second step: matmul with LoRA B, then reshape to original dims
        return out


# -------------------- Apply wrappers --------------------
def wrap_model(model: AutoModelForCausalLM, lora_dir: str, skip_embed: bool):
    linear_ops = [
        ("self_attn", "q_proj"), ("self_attn", "k_proj"),
        ("self_attn", "v_proj"), ("self_attn", "o_proj"),
        ("mlp", "gate_proj"), ("mlp", "up_proj"), ("mlp", "down_proj"),
    ]
    lora_manager = LoRAModelManager(lora_dir)

    # 1. Wrap linear layers in all transformer blocks
    # Iterate through each transformer layer and replace specified linear modules with LoRA-wrapped versions.
    for layer_idx, layer in enumerate(model.model.layers):
        for block_name, op_name in linear_ops: # Iterate through each type of linear operation (e.g., q_proj, gate_proj)
            target_module = getattr(layer, block_name) # Get the parent module (e.g., `self_attn` or `mlp`)
            base_module = getattr(target_module, op_name) # Get the original linear layer (e.g., `q_proj`)
            lora_A, lora_B = lora_manager.get_linear_AB(layer_idx, block_name, op_name) # Retrieve LoRA A and B matrices
            wrapped_module = BaseLayerWithLoRA(base_module, lora_A, lora_B) # Instantiate the custom wrapper
            setattr(target_module, op_name, wrapped_module) # Replace the original linear layer with the wrapped one

    # 2. Wrap embedding layer (model.embed_tokens)
    # If not skipping embedding, replace the base embedding layer with its LoRA-wrapped version.
    if not skip_embed:
        base_embedding = model.model.embed_tokens # Get the original embedding layer
        lora_A, lora_B = lora_manager.get_embedding_AB() # Retrieve LoRA A and B matrices for embedding
        if lora_A is not None and lora_B is not None:
            model.model.embed_tokens = VocabEmbeddingWithLoRA(base_embedding, lora_A, lora_B) # Replace with wrapped version

    # 3. Wrap lm_head (language model head)
    base_lm_head = model.lm_head # Get the original language model head
    lora_A, lora_B = lora_manager.get_lm_head_AB() # Retrieve LoRA A and B matrices for lm_head
    if lora_A is not None and lora_B is not None:
        wrapped_lm_head = BaseLayerWithLoRA(base_lm_head, lora_A, lora_B) # Create a wrapped lm_head module
        model.lm_head = wrapped_lm_head # Replace the original lm_head



# -------------------- Main --------------------
def main():
    parser = argparse.ArgumentParser(
        description="Inference with GPU base layers and CPU LoRA adapters"
    )
    parser.add_argument("--base-model-id",
                        default="meta-llama/Llama-2-7b-hf",
                        help="Base model ID to load from HF")
    parser.add_argument("--prompt", required=True,
                        help="Text prompt to generate from")
    args = parser.parse_args()

    repo_id = "yard1/llama-2-7b-sql-lora-test"
    lora_dir = snapshot_download(repo_id=repo_id)
    if not os.path.isdir(lora_dir):
        raise RuntimeError(f"Failed to download LoRA adapter to {lora_dir}")
    print(f"LoRA adapter downloaded to {lora_dir}")

    # 1) Load tokenizer and base model
    tokenizer = AutoTokenizer.from_pretrained(lora_dir, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda:0" if torch.cuda.is_available() else "cpu" # Determine the device for model execution
    dtype  = torch.bfloat16 if device.startswith("cuda") else torch.float32 # Use bfloat16 for GPU, float32 for CPU

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_id,
        device_map={"": 0} if device.startswith("cuda") else None, # Automatically map model to GPU if CUDA is used
        torch_dtype=dtype, # Set model's data type
        trust_remote_code=False
    )
    # Resize token embeddings to match tokenizer's vocabulary size
    model.resize_token_embeddings(len(tokenizer)) # Important for models that might have added tokens during fine-tuning

    # 2) Overwrite extra vocabulary embeddings (if any)
    loaded = safetensors.torch.load_file(f"{lora_dir}/new_embeddings.safetensors") # Load new embeddings from safetensors file
    new_inp = loaded["input_embeddings"].to(device=device, dtype=dtype) # Input embeddings for new tokens
    new_out = loaded["output_embeddings"].to(device=device, dtype=dtype) # Output embeddings for new tokens
    with torch.no_grad():
        inp_emb = model.get_input_embeddings().weight # Get the input embedding weights
        inp_emb[-new_inp.size(0):] = new_inp # Overwrite the last `new_inp.size(0)` rows with new input embeddings

        # Get the language model head's weights, handling cases where it might be weight-tied or not
        lm_w = (model.lm_head.weight
                if hasattr(model, "lm_head") and model.lm_head.weight.shape == inp_emb.shape
                else model.get_output_embeddings().weight)
        lm_w[-new_out.size(0):] = new_out

    # 3) Wrap model with LoRA adapters
    model.eval()
    wrap_model(model, lora_dir=lora_dir, skip_embed=False)
    
    # 4) Tokenize the input prompt
    inputs = tokenizer(
        args.prompt, # The text prompt from command line arguments
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)

    # 5) Generate text using sampling parameters similar to vLLM example
    stop_id = tokenizer.convert_tokens_to_ids("[/assistant]") # Convert the stop token string to its ID
    gen_out = model.generate( # Call the model's generate method
        **inputs,
        max_new_tokens=128, # Maximum number of new tokens to generate
        temperature=0.0, # Set temperature to 0.0 for greedy decoding (deterministic output)
        do_sample=False, # Disable sampling for greedy decoding
        eos_token_id=[stop_id], # Stop generation when this token is encountered
        output_scores=True, # Return logits/scores
        return_dict_in_generate=True # Return output as a dictionary
    )

    # 7) Decode
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