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

from utils import test


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
    def getInstance(cls, lora_dir: str = None, device: Optional[torch.device] = None):
        # Instantiate on first call using bypass __init__
        if cls._instance is None:
            if lora_dir is None:
                raise ValueError("lora_dir must be provided on the first call.")
            cls._instance = cls.__new__(cls)
            cls._instance._device = device or torch.device('cpu')
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
                # print(key) # (optional)
                parts = key.split('.')

                # Layer weights (e.g., base_model.model.model.layers.0.self_attn.q_proj.lora_A)
                if parts[3] == 'layers':
                    # TODO: Extract layer index, block type, op name, and kind from parts
                    layer_idx = ...
                    block_type = ...
                    op_name = ...
                    adapter_type = ...
                    
                    # Determine indices based on extracted information
                    ops = ['q_proj','k_proj','v_proj','o_proj'] if block_type == 'self_attn' else ['gate_proj','up_proj','down_proj']
                    block_idx = 0 if block_type == 'self_attn' else 1
                    op_idx = ops.index(op_name)
                    adapter_idx = 0 if adapter_type == 'lora_A' else 1

                    # Load tensor from safetensors file
                    tensor = f.get_tensor(key)

                    # TODO: Apply scaling factor to t if this is a lora_B
                    ...

                    # Move tensor to target device
                    tensor = tensor.to(self._device)

                    # TODO: Store tensor in self.lora_weights
                    ...

                # Embedding weights
                elif parts[3] == 'embed_tokens':
                    emb = parts[4]  # 'lora_embedding_A' or 'lora_embedding_B'
                    tensor = f.get_tensor(key)

                    if emb.endswith('_B'):
                        tensor = tensor * scale
                    
                    tensor = tensor.to(self._device)
                    
                    adapter_idx = 0 if emb.endswith('_A') else 1
                    self.lora_weights['embed_tokens'][adapter_idx] = tensor

                # LM head weights
                elif parts[2] == 'lm_head':
                    adapter_type = parts[3]
                    tensor = f.get_tensor(key)

                    # TODO: Apply scaling factor to t if this is a lora_B
                    ...

                    tensor = tensor.to(self._device)

                    # TODO: Determine an index based on adapter_type
                    ...

                    # TODO: Store tensor in self.lora_weights
                    ...
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



def main():
    # Download LoRA adapter
    repo_id = "yard1/llama-2-7b-sql-lora-test"
    print(f"Downloading LoRA adapter from '{repo_id}'...")
    lora_dir = snapshot_download(repo_id=repo_id)
    lora_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("====================================================================")
    print(f"Adapter downloaded to:\n{lora_dir}")
    print("====================================================================")

    # Inspect adapter (optional)
    # inspect_safetensors(lora_dir=lora_dir)

    # Instantiate manager and run tests
    mgr = LoRAModelManager.getInstance(lora_dir=lora_dir, device=lora_device)
    test(mgr=mgr)

if __name__ == '__main__':
    main()