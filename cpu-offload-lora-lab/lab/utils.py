def test(mgr=None):
    if not mgr:
        return ValueError("There is no LoRAModelManager instance.")

    # 2) Instantiate manager and check top-level keys
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