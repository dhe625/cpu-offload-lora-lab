from huggingface_hub import snapshot_download

import os
from safetensors import safe_open
from typing import Optional, Tuple
import json

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
            cls._instance._init_weights()
            cls._instance._load_weights(lora_dir)
        return cls._instance

    # Initializes the nested dictionary structure to store LoRA weights.
    def _init_weights(self): # Llama-2-7B: 32 layers × [self_attn(4), mlp(3)] × [A,B]
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
        scale = self._get_scaling(lora_dir)

        with safe_open(path, framework='pt') as f:
            for key in f.keys():
                parts = key.split('.')

                # 1) Load LoRA weights for transformer layers (e.g., model.layers.0.self_attn.q_proj.lora_A)
                if parts[3] == 'layers':
                    idx = int(parts[4])
                    block = parts[5]          # 'self_attn' or 'mlp'
                    op = parts[6]             # e.g. 'q_proj'
                    kind = parts[7]           # 'lora_A' or 'lora_B'
                    
                    # Determine operations based on block type ('self_attn' or 'mlp')
                    ops = ['q_proj','k_proj','v_proj','o_proj'] if block == 'self_attn' else ['gate_proj','up_proj','down_proj']
                    bt = 0 if block == 'self_attn' else 1
                    op_idx = ops.index(op)

                    t = f.get_tensor(key)

                    # Apply scaling to LoRA B weights to control adapter magnitude
                    if kind == 'lora_B':
                        t = t * scale

                    # Store tensor in self.lora_weights: index 0 for A, 1 for B
                    w_idx = 0 if kind == 'lora_A' else 1
                    self.lora_weights['model']['layers'][idx][bt][op_idx][w_idx] = t

                # 2) Load LoRA weights for embedding layer entries
                elif parts[3] == 'embed_tokens':
                    emb = parts[4]  # 'lora_embedding_A' or 'lora_embedding_B'
                    t = f.get_tensor(key)

                    # Apply scaling to LoRA B weights to control adapter magnitude
                    if emb.endswith('_B'):
                        t = t * scale
                    
                    # Store tensor in self.lora_weights: index 0 for A, 1 for B
                    i = 0 if emb.endswith('_A') else 1
                    self.lora_weights['model']['embed_tokens'][i] = t

                # 3) Load LoRA weights for language model head
                elif parts[2] == 'lm_head':
                    kind = parts[3]
                    t = f.get_tensor(key)

                    # Apply scaling to LoRA B weights to control adapter magnitude
                    if kind == 'lora_B':
                        t = t * scale

                    # Store tensor in self.lora_weights: index 0 for A, 1 for B
                    i = 0 if kind == 'lora_A' else 1
                    self.lora_weights['lm_head'][i] = t
                else:
                    raise ValueError(f"Unexpected key: {key}")
                
    # Compute scaling factor from adapter_config.json (alpha / rank)
    def _get_scaling(self,lora_dir: str) -> float:
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
    
                     
def main():
    # 1) Download LoRA adapter
    repo_id = "yard1/llama-2-7b-sql-lora-test"
    print(f"Downloading LoRA adapter from '{repo_id}'...")
    lora_dir = snapshot_download(repo_id=repo_id)
    print("====================================================================")
    print(f"Adapter downloaded to:\n{lora_dir}")
    print("====================================================================")

    # 2) Instantiate manager and check top-level keys
    mgr = LoRAModelManager(lora_dir=lora_dir)
    test(mgr=mgr)

if __name__ == '__main__':
    main()