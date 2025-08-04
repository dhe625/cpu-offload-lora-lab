"""
LoRA adapter loader script.

This script provides:
- LoRAModelManager: loads and manages LoRA weights from safetensors files.
- inspect_safetensors: introspects a safetensors file, printing metadata and tensor shapes.
- CLI entry point: downloads a LoRA adapter, inspects it, and runs basic tests.
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
    Manages the loading and access of LoRA (Low-Rank Adaptation) weights from safetensors files.

    This class is implemented as a singleton to ensure that LoRA weights are loaded only once
    and shared across the application. It provides methods to retrieve LoRA A and B matrices
    for different parts of a transformer model, including linear layers, embedding layers,
    and the language model head.

    The LoRA weights are stored in a nested dictionary structure, organized by model
    components (e.g., layers, self_attn, mlp, embed_tokens, lm_head). Scaling factors
    (lora_alpha / r) are applied to the LoRA B weights during loading.

    Attributes:
        lora_weights (dict): A nested dictionary holding the loaded LoRA A and B tensors.
                             The structure mirrors the model's architecture for easy access.
    """

    _instance = None

    def __new__(cls, lora_dir: str = None):
        if cls._instance is None:
            if not lora_dir:
                raise ValueError("First init requires lora_dir.")
            cls._instance = super().__new__(cls)
            cls._instance._lora_dir = lora_dir
            cls._instance._init_weights()
            cls._instance._load_weights(lora_dir)
        else:
            if lora_dir is not None and lora_dir != cls._instance._lora_dir:
                raise ValueError(f"LoRAModelManager already initialized with lora_dir={cls._instance._lora_dir}, cannot reinitialize with different lora_dir={lora_dir}")
        return cls._instance

    # Initializes the nested dictionary structure to store LoRA weights.
    def _init_weights(self):
        """
        Initialize nested dictionary for LoRA weights:
        model.layers (32 layers × [self_attn(4), mlp(3)] × [A,B]),
        model.embed_tokens [A,B], and lm_head [A,B].
        """
        self.lora_weights = {
            'model': {
                'layers': [
                    [
                        [[None, None] for _ in range(4)],  # self_attn: q,k,v,o
                        [[None, None] for _ in range(3)]   # mlp: gate,up,down
                    ] for _ in range(32)
                ],
                'embed_tokens': [None, None]
            },
            'lm_head': [None, None]
        }

    def _load_weights(self, lora_dir: str):
        path = os.path.join(lora_dir, 'adapter_model.safetensors')
        scale = self._read_scaling(lora_dir) # Apply scaling to LoRA B weights to control adapter magnitude

        with safe_open(path, framework='pt') as f:
            for key in f.keys():
                parts = key.split('.')

                # 1) Load LoRA weights for transformer layers (e.g., model.layers.0.self_attn.q_proj.lora_A)
                if parts[3] == 'layers':
                    # TODO: Extract layer index, block type, op name, and kind from parts
                    idx = ...
                    block = ...   # 'self_attn' or 'mlp'
                    op = ...      # 'q_proj'
                    kind = ...    # 'lora_A' or 'lora_B'

                    # Determine operations based on block type ('self_attn' or 'mlp')
                    ops = [...] if block == 'self_attn' else [...]
                    bt = 0 if block == 'self_attn' else 1
                    op_idx = ops.index(op)

                    # Load tensor from safetensors file
                    t = f.get_tensor(key)

                    # TODO: If this is a B-weight, apply scaling factor
                    ...

                    # TODO: Store tensor in self.lora_weights
                    w_idx = 0 if kind == 'lora_A' else 1
                    self.lora_weights[...][...][...][...][...][...] = t

                # 2) Load LoRA weights for embedding layer entries
                elif parts[3] == 'embed_tokens':
                    # TODO: Determine A or B embedding from parts[4] (emb)
                    emb = ...

                    # Load tensor
                    t = f.get_tensor(key)

                    # TODO: Apply scaling for B embeddings
                    ...

                    # TODO: Compute index i (0 for A, 1 for B)
                    i = ...

                    # TODO: Store in self.lora_weights
                    self.lora_weights[...][...][i] = t

                # 3) Load LoRA weights for language model head
                elif parts[2] == 'lm_head':
                    # TODO: Determine kind ('lora_A' or 'lora_B') from parts[3]
                    kind = ...

                    # Load tensor
                    t = f.get_tensor(key)

                    # TODO: Apply scaling if B-weight
                    ...

                    # TODO: Compute i (0 for A, 1 for B)
                    i = ...

                    # TODO: Store in self.lora_weights
                    self.lora_weights[...][...] = t
                else:
                    raise ValueError(f"Unexpected key: {key}")
                
    # Compute scaling factor from adapter_config.json (alpha / rank)
    def _read_scaling(self,lora_dir: str) -> float:
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
        bt = 0 if block == 'self_attn' else 1
        ops = ['q_proj','k_proj','v_proj','o_proj'] if block == 'self_attn' else ['gate_proj','up_proj','down_proj']
        op_idx = ops.index(op)
        return tuple(self.lora_weights['model']['layers'][layer_idx][bt][op_idx])

    def get_embedding_AB(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        return tuple(self.lora_weights['model']['embed_tokens'])

    def get_lm_head_AB(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        return tuple(self.lora_weights['lm_head'])


def inspect_safetensors(lora_dir: str):
    """
    Inspects a safetensors file and prints its metadata and tensor keys.

    Args:
        lora_dir (str): Directory containing 'adapter_model.safetensors'.
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
    # 1) Download LoRA adapter
    repo_id = "yard1/llama-2-7b-sql-lora-test"
    print(f"Downloading LoRA adapter from '{repo_id}'...")
    lora_dir = snapshot_download(repo_id=repo_id)
    print("====================================================================")
    print(f"Adapter downloaded to:\n{lora_dir}")
    print("====================================================================")

    # 2) Inspect LoRA adapter
    inspect_safetensors(lora_dir=lora_dir)

    # 3) Instantiate manager and check top-level keys
    mgr = LoRAModelManager(lora_dir=lora_dir)
    test(mgr=mgr)

if __name__ == '__main__':
    main()