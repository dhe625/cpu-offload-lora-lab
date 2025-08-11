def wrap_model(model, lora_dir: str, lora_mode: str, device: str, skip_embed: bool):
    # Lazy import heavy dependencies
    from load_lora_adapter import LoRAModelManager

    linear_ops = [
        ("self_attn", "q_proj"), ("self_attn", "k_proj"),
        ("self_attn", "v_proj"), ("self_attn", "o_proj"),
        ("mlp", "gate_proj"), ("mlp", "up_proj"), ("mlp", "down_proj"),
    ]
    lora_manager = LoRAModelManager.getInstance(lora_dir=lora_dir)

    # Select LoRA wrapper implementations based on mode
    if lora_mode == "single_stream":
        from layers.lora_single_stream import BaseLayerWithLoRASingleStream, VocabEmbeddingWithLoRASingleStream
        LayerWrapper = BaseLayerWithLoRASingleStream
        EmbeddingWrapper = VocabEmbeddingWithLoRASingleStream
    elif lora_mode == "multi_stream":
        from layers.lora_multi_stream import BaseLayerWithLoRAMultiStream, VocabEmbeddingWithLoRAMultiStream
        LayerWrapper = BaseLayerWithLoRAMultiStream
        EmbeddingWrapper = VocabEmbeddingWithLoRAMultiStream
    elif lora_mode == "cpu":
        from layers.lora_cpu import BaseLayerWithLoRACPU
        from layers.lora_single_stream import VocabEmbeddingWithLoRASingleStream
        LayerWrapper = BaseLayerWithLoRACPU
        EmbeddingWrapper = VocabEmbeddingWithLoRASingleStream
    else:
        raise ValueError(f"Unsupported lora_mode: {lora_mode}")

    for layer_idx, layer in enumerate(model.model.layers):
        for block_name, op_name in linear_ops:
            target_module = getattr(layer, block_name)
            base_module = getattr(target_module, op_name)
            lora_A, lora_B = lora_manager.get_linear_AB(layer_idx, block_name, op_name)
            if lora_mode != "cpu":
                lora_A = lora_A.to(device)
                lora_B = lora_B.to(device)

            wrapped_module = LayerWrapper(base_module, lora_A, lora_B)
            setattr(target_module, op_name, wrapped_module)

    if not skip_embed:
        base_embedding = model.model.embed_tokens
        lora_A, lora_B = lora_manager.get_embedding_AB()
        if lora_A is not None and lora_B is not None:
            lora_A = lora_A.to(device)
            lora_B = lora_B.to(device)

            model.model.embed_tokens = EmbeddingWrapper(base_embedding, lora_A, lora_B)

    base_lm_head = model.lm_head
    lora_A, lora_B = lora_manager.get_lm_head_AB()
    if lora_A is not None and lora_B is not None:
        if lora_mode != "cpu":
            lora_A = lora_A.to(device)
            lora_B = lora_B.to(device)

        wrapped_lm_head = LayerWrapper(base_lm_head, lora_A, lora_B)
        model.lm_head = wrapped_lm_head