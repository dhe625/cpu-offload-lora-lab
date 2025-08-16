"""
LoRA Adapter Loader

This module provides utilities for working with LoRA adapters:
  - A singleton manager (`LoRAModelManager`) to load and access LoRA A/B weights 
    from `adapter_model.safetensors`.
  - A utility (`inspect_safetensors`) to inspect the safetensors file.
  - A test function (`test`) to validate correctness of loaded weights.
  - A CLI entry point (`main`) to download and verify a sample LoRA adapter.

LoRA weights are stored in a nested dictionary structure with:
  - Per-layer attention and MLP blocks
  - Embedding weights
  - LM head weights
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

    Responsibilities:
      - Load LoRA A and B tensors from `adapter_model.safetensors`.
      - Organize them into a structured dictionary for easy access.
      - Apply scaling to B weights using values from `adapter_config.json`.

    Data structure:
      lora_weights = {
        'layers': [
          [ [ [A, B], ... ],   # self_attn: q_proj, k_proj, v_proj, o_proj
            [ [A, B], ... ]    # mlp: gate_proj, up_proj, down_proj
          ] for _ in range(num_layers)
        ],
        'embed_tokens': [A, B],
        'lm_head': [A, B]
      }

    Public methods:
      - get_linear_AB(layer_idx, block, op) -> (A, B)
      - get_embedding_AB() -> (A, B)
      - get_lm_head_AB() -> (A, B)
    """
    _instance = None

    def __init__(self):
        raise RuntimeError('Use LoRAModelManager.getInstance() instead of direct construction.')

    @classmethod
    def getInstance(cls, lora_dir: str = None):
        """
        Return the singleton instance of LoRAModelManager.

        Args:
            lora_dir (str, optional): Directory containing `adapter_model.safetensors`.
                                      Must be provided the first time.

        Returns:
            LoRAModelManager: singleton instance
        """
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
        """
        Load LoRA weights from adapter_model.safetensors.

        - Applies scaling (alpha/r) from adapter_config.json to B weights.
        - Populates `self.lora_weights`.
        """
        path = os.path.join(lora_dir, 'adapter_model.safetensors')
        scale = self._read_scaling(lora_dir) # scaling factor for B matrices

        with safe_open(path, framework='pt') as f:
            for key in f.keys():
                parts = key.split('.')

                # -------------------------
                # Transformer layer weights
                # -------------------------
                if parts[3] == 'layers':
                    layer_idx = int(parts[4])   # which transformer layer
                    block_type = parts[5]       # self_attn or mlp
                    op_name = parts[6]          # q_proj, k_proj, etc.
                    adapter_type = parts[7]     # lora_A or lora_B
                    
                    ops = ['q_proj','k_proj','v_proj','o_proj'] if block_type == 'self_attn' else ['gate_proj','up_proj','down_proj']
                    block_idx = 0 if block_type == 'self_attn' else 1
                    op_idx = ops.index(op_name)
                    adapter_idx = 0 if adapter_type == 'lora_A' else 1

                    tensor = f.get_tensor(key)

                    if adapter_type == 'lora_B':
                        tensor = tensor * scale

                    self.lora_weights['layers'][layer_idx][block_idx][op_idx][adapter_idx] = tensor

                # -------------------------
                # Embedding weights
                # -------------------------
                elif parts[3] == 'embed_tokens':
                    emb = parts[4]
                    tensor = f.get_tensor(key)

                    if emb.endswith('_B'):
                        tensor = tensor * scale
                    
                    adapter_idx = 0 if emb.endswith('_A') else 1
                    self.lora_weights['embed_tokens'][adapter_idx] = tensor

                # -------------------------
                # LM head weights
                # -------------------------
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
        """
        Read scaling factor from adapter_config.json.

        Scale = alpha / r
        If missing, defaults to 1.0.
        """
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
        """
        Retrieve (A, B) tensors for a given linear projection.

        Args:
            layer_idx (int): Transformer layer index (0–31).
            block (str): 'self_attn' or 'mlp'.
            op (str): Operation name (e.g., 'q_proj', 'up_proj').

        Returns:
            (torch.Tensor, torch.Tensor): LoRA A and B matrices.
        """
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
    Inspect the contents of `adapter_model.safetensors`.

    Prints:
      - Metadata
      - Keys, shapes, and dtypes of all tensors
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
    Rigorous validation test for LoRAModelManager.

    Verifies:
      1. Presence of top-level keys
      2. lm_head A/B tensors: type and shape
      3. embed_tokens A/B tensors: type and shape
      4. Each transformer layer's attention and MLP ops:
         - A and B exist
         - Are torch.Tensors
         - Are 2D matrices
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
                # Validate presence and types
                assert A is not None and B is not None
                assert isinstance(A, torch.Tensor)
                assert isinstance(B, torch.Tensor)
                # Validate dimensions
                assert A.ndim == 2 and B.ndim == 2

    print("All rigorous tests passed for LoRAModelManager.")


def main():
    """
    CLI entry point.

    - Downloads a sample LoRA adapter from HuggingFace Hub
    - Inspects the safetensors file
    - Instantiates LoRAModelManager
    - Runs validation tests
    """
    repo_id = "yard1/llama-2-7b-sql-lora-test"
    print(f"Downloading LoRA adapter from '{repo_id}'...")
    lora_dir = snapshot_download(repo_id=repo_id)
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