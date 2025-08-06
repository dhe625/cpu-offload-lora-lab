from transformers import AutoModelForCausalLM
from load_lora_adapter import LoRAModelManager

# from layers.gpu_lora_single_stream import BaseLayerWithLoRA, VocabEmbeddingWithLoRA
# from layers.gpu_lora_multi_stream import BaseLayerWithLoRA, VocabEmbeddingWithLoRA
from layers.cpu_lora import BaseLayerWithLoRA, VocabEmbeddingWithLoRA


def wrap_model(model: AutoModelForCausalLM, lora_dir: str, lora_mode: int, lora_device: str,skip_embed: bool):
    linear_ops = [
        ("self_attn", "q_proj"), ("self_attn", "k_proj"),
        ("self_attn", "v_proj"), ("self_attn", "o_proj"),
        ("mlp", "gate_proj"), ("mlp", "up_proj"), ("mlp", "down_proj"),
    ]
    lora_manager = LoRAModelManager.getInstance(lora_dir=lora_dir)

    # 1. Wrap linear layers in all transformer blocks
    # Iterate through each transformer layer and replace specified linear modules with LoRA-wrapped versions.
    for layer_idx, layer in enumerate(model.model.layers):
        for block_name, op_name in linear_ops: # Iterate through each type of linear operation (e.g., q_proj, gate_proj)
            target_module = getattr(layer, block_name) # Get the parent module (e.g., `self_attn` or `mlp`)
            base_module = getattr(target_module, op_name) # Get the original linear layer (e.g., `q_proj`)
            lora_A, lora_B = lora_manager.get_linear_AB(layer_idx, block_name, op_name) # Retrieve LoRA A and B matrices
            
            if lora_device != "cpu":
                lora_A = lora_A.to(lora_device)
                lora_B = lora_B.to(lora_device)

            # TODO: change module class
            wrapped_module = BaseLayerWithLoRA(base_module, lora_A, lora_B) # Instantiate the custom wrapper
            setattr(target_module, op_name, wrapped_module) # Replace the original linear layer with the wrapped one

    # 2. Wrap embedding layer (model.embed_tokens)
    # If not skipping embedding, replace the base embedding layer with its LoRA-wrapped version.
    if not skip_embed:
        base_embedding = model.model.embed_tokens # Get the original embedding layer
        lora_A, lora_B = lora_manager.get_embedding_AB() # Retrieve LoRA A and B matrices for embedding
        if lora_A is not None and lora_B is not None:
            if lora_device != "cpu":
                lora_A = lora_A.to(lora_device)
                lora_B = lora_B.to(lora_device)

            # TODO: change module class
            model.model.embed_tokens = VocabEmbeddingWithLoRA(base_embedding, lora_A, lora_B) # Replace with wrapped version

    # 3. Wrap lm_head (language model head)
    base_lm_head = model.lm_head # Get the original language model head
    lora_A, lora_B = lora_manager.get_lm_head_AB() # Retrieve LoRA A and B matrices for lm_head
    if lora_A is not None and lora_B is not None:
        if lora_device != "cpu":
            lora_A = lora_A.to(lora_device)
            lora_B = lora_B.to(lora_device)

        # TODO: change module class
        wrapped_lm_head = BaseLayerWithLoRA(base_lm_head, lora_A, lora_B) # Create a wrapped lm_head module
        model.lm_head = wrapped_lm_head # Replace the original lm_head