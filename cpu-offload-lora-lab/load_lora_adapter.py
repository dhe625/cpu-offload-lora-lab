from huggingface_hub import snapshot_download

import os
from safetensors import safe_open
from typing import Optional, Tuple
import json

import torch


class LoRAModelManager:
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
    def _init_weights(self):
        # 32 layers × [self_attn(4), mlp(3)] × [A,B]
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
        scale = read_scaling(lora_dir)

        with safe_open(path, framework='pt') as f:
            for key in f.keys():
                parts = key.split('.')
                # Transformer layers
                if parts[3] == 'layers':
                    idx = int(parts[4])
                    block = parts[5]          # 'self_attn' or 'mlp'
                    op = parts[6]             # e.g. 'q_proj'
                    kind = parts[7]           # 'lora_A' or 'lora_B'
                    # choose block-specific ops
                    ops = ['q_proj','k_proj','v_proj','o_proj'] if block == 'self_attn' else ['gate_proj','up_proj','down_proj']
                    bt = 0 if block == 'self_attn' else 1
                    op_idx = ops.index(op)

                    t = f.get_tensor(key)
                    # Apply scaling factor only to LoRA B weights.
                    # This is a common practice in LoRA to control the magnitude of the adapter's contribution.
                    if kind == 'lora_B':
                        t = t * scale

                    # Assign the tensor to the correct A (0) or B (1) slot.
                    w_idx = 0 if kind == 'lora_A' else 1
                    self.lora_weights['model']['layers'][idx][bt][op_idx][w_idx] = t
                # Handle Embedding layer weights (e.g., model.embed_tokens.lora_embedding_A)
                elif parts[3] == 'embed_tokens':
                    emb = parts[4]  # 'lora_embedding_A' or 'lora_embedding_B'
                    t = f.get_tensor(key)
                    # Apply scaling factor to LoRA B embedding weights.
                    if emb.endswith('_B'):
                        t = t * scale
                    # Assign the tensor to the correct A (0) or B (1) slot.
                    i = 0 if emb.endswith('_A') else 1
                    self.lora_weights['model']['embed_tokens'][i] = t
                # Handle lm_head weights (e.g., lm_head.lora_A)
                elif parts[2] == 'lm_head':
                    kind = parts[3]
                    t = f.get_tensor(key)
                    # Apply scaling factor to LoRA B lm_head weights.
                    if kind == 'lora_B':
                        t = t * scale
                    # Assign the tensor to the correct A (0) or B (1) slot.
                    i = 0 if kind == 'lora_A' else 1
                    self.lora_weights['lm_head'][i] = t
                else:
                    raise ValueError(f"Unexpected key: {key}")

    def get_linear_AB(self, layer_idx: int, block: str, op: str) -> Tuple[torch.Tensor, torch.Tensor]:
        bt = 0 if block == 'self_attn' else 1
        ops = ['q_proj','k_proj','v_proj','o_proj'] if block == 'self_attn' else ['gate_proj','up_proj','down_proj']
        op_idx = ops.index(op)
        return tuple(self.lora_weights['model']['layers'][layer_idx][bt][op_idx])

    def get_embedding_AB(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        return tuple(self.lora_weights['model']['embed_tokens'])

    def get_lm_head_AB(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        return tuple(self.lora_weights['lm_head'])
    
    
def read_scaling(lora_dir: str) -> float:
    fname = "adapter_config.json"
    p = os.path.join(lora_dir, fname)
    if os.path.exists(p): # Check if the file exists
        with open(p, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        alpha = cfg.get("lora_alpha", cfg.get("alpha"))
        r = cfg.get("r", cfg.get("rank"))
        if alpha and r:
            return float(alpha) / float(r)
    return 1.0


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
    keys = list(mgr.lora_weights.keys())
    print("Top-level keys:", keys)
    assert keys == ['model', 'lm_head'], f"Unexpected keys: {keys}"
    print("====================================================================")

    # 3) Verify lm_head shapes
    A_lm, B_lm = mgr.get_lm_head_AB()
    print(f"lm_head A: {A_lm.shape}\nB: {B_lm.shape}")
    assert A_lm.shape == (8, 4096), f"lm_head A shape mismatch: {A_lm.shape}"
    assert B_lm.shape == (32004, 8), f"lm_head B shape mismatch: {B_lm.shape}"
    print("====================================================================")

    # 4) Verify embed_tokens shapes
    A_e, B_e = mgr.get_embedding_AB()
    print(f"embed_tokens A: {A_e.shape}\nB: {B_e.shape}")
    assert A_e.shape == (8, 32004), f"embed A shape mismatch: {A_e.shape}"
    assert B_e.shape == (4096, 8), f"embed B shape mismatch: {B_e.shape}"
    print("====================================================================")

    # 5) Verify layers 0 and 1 linear shapes
    expected_self = {'q_proj': (8, 4096), 'k_proj': (8, 4096), 'v_proj': (8, 4096), 'o_proj': (8, 4096)}
    expected_self_B = {'q_proj': (4096, 8), 'k_proj': (4096, 8), 'v_proj': (4096, 8), 'o_proj': (4096, 8)}
    expected_mlp = {'gate_proj': (8, 4096), 'up_proj': (8, 4096), 'down_proj': (8, 11008)}
    expected_mlp_B = {'gate_proj': (11008, 8), 'up_proj': (11008, 8), 'down_proj': (4096, 8)}

    for layer in [0, 1]:
        print(f"Checking layer: {layer}\n")
        for op, shapeA in expected_self.items():
            A, B = mgr.get_linear_AB(layer, 'self_attn', op)
            print(f" self_attn.{op}: A {A.shape}, B {B.shape}")
            assert A.shape == shapeA, f"Layer {layer} self_attn {op} A: got {A.shape}"
            assert B.shape == expected_self_B[op], f"Layer {layer} self_attn {op} B: got {B.shape}"
        for op, shapeA in expected_mlp.items():
            A, B = mgr.get_linear_AB(layer, 'mlp', op)
            print(f" mlp.{op}: A {A.shape}, B {B.shape}")
            assert A.shape == shapeA, f"Layer {layer} mlp {op} A: got {A.shape}"
            assert B.shape == expected_mlp_B[op], f"Layer {layer} mlp {op} B: got {B.shape}"
        print("====================================================================")

    print("All assertions passed. LoRAModelManager is working correctly.")
    print("====================================================================")


if __name__ == '__main__':
    main()