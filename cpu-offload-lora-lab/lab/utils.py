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