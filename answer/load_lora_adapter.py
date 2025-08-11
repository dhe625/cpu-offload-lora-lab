"""
LoRA Adapter Loader

Provides a singleton manager for loading and accessing LoRA A/B weights 
from safetensors files, a utility to inspect adapter_model.safetensors, 
and a CLI entry point to download and test a LoRA adapter.
"""

import os
import json
from typing import Optional, Tuple

from safetensors import safe_open
from huggingface_hub import snapshot_download
import torch


class LoRAModelManager:
    """
    Singleton manager for LoRA weights.

    Loads LoRA A and B tensors from a safetensors file into a nested dictionary:
      - layers: per-layer [self_attn, mlp] blocks with [q,k,v,o] or [gate,up,down] weights.
      - embed_tokens: LoRA embedding weights.
      - lm_head: LoRA head weights.

    Methods:
      get_linear_AB(layer_idx, block, op) -> (A, B)
      get_embedding_AB() -> (A, B)
      get_lm_head_AB() -> (A, B)
    """
    _instance = None

    def __init__(self):
        raise RuntimeError('Call instance() instead')

    @classmethod
    def getInstance(cls, lora_dir: str = None):
        # Instantiate on first call using bypass __init__
        if cls._instance is None:
            if lora_dir is None:
                raise ValueError("lora_dir must be provided on the first call.")
            cls._instance = cls.__new__(cls)
            cls._instance._init_weights()
            cls._instance._load_weights(lora_dir)
        return cls._instance

    def _init_weights(self):
        """Initialize storage structure for LoRA weights."""
        self.lora_weights = {
            'layers': [
                [
                    [[None, None] for _ in range(4)],   # self_attn: q,k,v,o
                    [[None, None] for _ in range(3)]    # mlp: gate,up,down
                ] for _ in range(32)                    # num_layer: 32
            ],
            'embed_tokens': [None, None],
            'lm_head': [None, None]
        }

    def _load_weights(self, lora_dir: str):
        """Load and scale LoRA tensors from adapter_model.safetensors."""
        path = os.path.join(lora_dir, 'adapter_model.safetensors')
        scale = self._read_scaling(lora_dir) # Apply scaling to LoRA B weights to control adapter magnitude

        with safe_open(path, framework='pt') as f:
            for key in f.keys():
                parts = key.split('.')

                # Layer weights
                if parts[3] == 'layers':
                    layer_idx = int(parts[4])
                    block_type = parts[5]
                    op_name = parts[6]
                    adapter_type = parts[7]
                    
                    ops = ['q_proj','k_proj','v_proj','o_proj'] if block_type == 'self_attn' else ['gate_proj','up_proj','down_proj']
                    block_idx = 0 if block_type == 'self_attn' else 1
                    op_idx = ops.index(op_name)
                    adapter_idx = 0 if adapter_type == 'lora_A' else 1

                    tensor = f.get_tensor(key)

                    if adapter_type == 'lora_B':
                        tensor = tensor * scale

                    self.lora_weights['layers'][layer_idx][block_idx][op_idx][adapter_idx] = tensor

                # Embedding weights
                elif parts[3] == 'embed_tokens':
                    emb = parts[4]
                    tensor = f.get_tensor(key)

                    if emb.endswith('_B'):
                        tensor = tensor * scale
                    
                    adapter_idx = 0 if emb.endswith('_A') else 1
                    self.lora_weights['embed_tokens'][adapter_idx] = tensor

                # LM head weights
                elif parts[2] == 'lm_head':
                    adapter_type = parts[3]
                    tensor = f.get_tensor(key)

                    if adapter_type == 'lora_B':
                        tensor = tensor * scale

                    adapter_idx = 0 if adapter_type == 'lora_A' else 1
                    self.lora_weights['lm_head'][adapter_idx] = tensor
                else:
                    raise ValueError(f"Unexpected key: {key}")
                
    def _read_scaling(self,lora_dir: str) -> float:
        """Read adapter_config.json and compute scale (alpha/r); default to 1.0."""
        fname = "adapter_config.json"
        p = os.path.join(lora_dir, fname)
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            alpha = cfg.get("lora_alpha", cfg.get("alpha"))
            r = cfg.get("r", cfg.get("rank"))
            if alpha and r:
                return float(alpha) / float(r)
        return 1.0

    def get_linear_AB(self, layer_idx: int, block: str, op: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (A, B) for a given layer, block, and operation."""
        bt = 0 if block == 'self_attn' else 1
        ops = ['q_proj','k_proj','v_proj','o_proj'] if block == 'self_attn' else ['gate_proj','up_proj','down_proj']
        op_idx = ops.index(op)
        return tuple(self.lora_weights['layers'][layer_idx][bt][op_idx])

    def get_embedding_AB(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Return (A, B) tensors for embedding layer."""
        return tuple(self.lora_weights['embed_tokens'])

    def get_lm_head_AB(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Return (A, B) tensors for language model head."""
        return tuple(self.lora_weights['lm_head'])


def inspect_safetensors(lora_dir: str):
    """
    Print metadata and tensor info from adapter_model.safetensors.

    Args:
        lora_dir (str): Directory containing adapter_model.safetensors.
    """
    safetensors_path = os.path.join(lora_dir, 'adapter_model.safetensors')
    if not os.path.exists(safetensors_path):
        print(f"Error: File not found at {safetensors_path}")
        return

    print(f"Inspecting safetensors file: {safetensors_path}")
    with safe_open(safetensors_path, framework="pt") as f:
        print("\n--- Metadata ---")
        for k, v in f.metadata().items():
            print(f"  {k}: {v}")

        print("\n--- Tensor Keys ---")
        for key in f.keys():
            t = f.get_tensor(key)
            print(f"  {key}: shape={tuple(t.shape)}, dtype={t.dtype}")
    print("-------------------\n")

def test(mgr=None):
    """
    Rigorous test for LoRAModelManager:
    - Verifies top-level keys.
    - Checks lm_head and embedding shapes.
    - Validates all transformer layers (self_attn and mlp) for correct presence, tensor types, and dimensions.
    """
    import torch

    if mgr is None:
        raise ValueError("LoRAModelManager instance is required for testing.")

    # 1) Top-level keys
    keys = list(mgr.lora_weights.keys())
    print("Top-level keys:", keys)
    assert set(keys) == {'layers', 'embed_tokens', 'lm_head'}, f"Unexpected top-level keys: {keys}"

    # 2) lm_head shapes
    A_lm, B_lm = mgr.get_lm_head_AB()
    print(f"lm_head A: {A_lm.shape}, B: {B_lm.shape}")
    for tensor, name in [(A_lm, 'lm_head A'), (B_lm, 'lm_head B')]:
        assert isinstance(tensor, torch.Tensor), f"{name} is not a Tensor"
        assert tensor.ndim == 2, f"{name} should be 2D"

    # 3) embed_tokens shapes
    A_e, B_e = mgr.get_embedding_AB()
    print(f"embed_tokens A: {A_e.shape}, B: {B_e.shape}")
    for tensor, name in [(A_e, 'embed_tokens A'), (B_e, 'embed_tokens B')]:
        assert isinstance(tensor, torch.Tensor), f"{name} is not a Tensor"
        assert tensor.ndim == 2, f"{name} should be 2D"

    # 4) Transformer layers
    layers = mgr.lora_weights['layers']
    assert isinstance(layers, list) and len(layers) == 32, "There should be 32 transformer layers."
    blocks = {
        'self_attn': ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        'mlp': ['gate_proj', 'up_proj', 'down_proj'],
    }
    for idx, layer in enumerate(layers):
        for block_name, ops in blocks.items():
            for op in ops:
                A, B = mgr.get_linear_AB(idx, block_name, op)
                nameA = f"layer {idx} {block_name}.{op} A"
                nameB = f"layer {idx} {block_name}.{op} B"
                # Presence and type
                assert A is not None and B is not None, f"{nameA} or {nameB} is None"
                assert isinstance(A, torch.Tensor), f"{nameA} is not a Tensor"
                assert isinstance(B, torch.Tensor), f"{nameB} is not a Tensor"
                # Dimensionality
                assert A.ndim == 2 and B.ndim == 2, f"{nameA} or {nameB} is not 2D"

    print("All rigorous tests passed for LoRAModelManager.")


def main():
    # Download LoRA adapter
    repo_id = "yard1/llama-2-7b-sql-lora-test"
    print(f"Downloading LoRA adapter from '{repo_id}'...")
    lora_dir = snapshot_download(repo_id=repo_id)
    # lora_dir = "/root/.cache/huggingface/hub/models--yard1--llama-2-7b-sql-lora-test/snapshots/0dfa347e8877a4d4ed19ee56c140fa518470028c"
    print("====================================================================")
    print(f"Adapter downloaded to:\n{lora_dir}")
    print("====================================================================")

    # Inspect adapter (optional)
    inspect_safetensors(lora_dir=lora_dir)

    # Instantiate manager and run tests
    mgr = LoRAModelManager.getInstance(lora_dir=lora_dir)
    test(mgr=mgr)

if __name__ == '__main__':
    main()