#!/usr/bin/env python
import argparse
import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import safetensors.torch
from huggingface_hub import snapshot_download

from utils import wrap_model

HF_TOKEN = "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"


def main():
    parser = argparse.ArgumentParser(
        description="Inference with LoRA adapters"
    )
    parser.add_argument("--base-model-id",
                        default="meta-llama/Llama-2-7b-hf",
                        help="Base model ID to load from HF")
    parser.add_argument("--prompt", required=True,
                        help="Text prompt to generate from")
    parser.add_argument("--batch-size", required=True,
                        help="Batch size")
    parser.add_argument("--lora-mode", required=True,
                        help="single-stream: LoRA using single stream, multi-stream: LoRA using multi stream, cpu: LoRA using CPU offload")
    args = parser.parse_args()

    repo_id = "yard1/llama-2-7b-sql-lora-test"
    lora_dir = snapshot_download(repo_id=repo_id)
    # lora_dir = "/root/.cache/huggingface/hub/models--yard1--llama-2-7b-sql-lora-test/snapshots/0dfa347e8877a4d4ed19ee56c140fa518470028c"

    if not os.path.isdir(lora_dir):
        raise RuntimeError(f"Failed to download LoRA adapter to {lora_dir}")
    print(f"LoRA adapter downloaded to {lora_dir}")

    # 1) Load base model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(lora_dir, use_fast=False, token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16 if device.startswith("cuda") else torch.float32 # Use bfloat16 for GPU, float32 for CPU

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_id,
        device_map={"": 0} if device.startswith("cuda") else None, # Automatically map model to GPU if CUDA is used
        torch_dtype=dtype, # Set model's data type
        trust_remote_code=False,
        token=HF_TOKEN,
    )
    # Resize token embeddings to match tokenizer's vocabulary size
    model.resize_token_embeddings(len(tokenizer)) # Important for models that might have added tokens during fine-tuning

    # 2) Overwrite extra vocabulary embeddings (if any)
    loaded = safetensors.torch.load_file(f"{lora_dir}/new_embeddings.safetensors") # Load new embeddings from safetensors file
    new_inp = loaded["input_embeddings"].to(device=device, dtype=dtype) # Input embeddings for new tokens
    new_out = loaded["output_embeddings"].to(device=device, dtype=dtype) # Output embeddings for new tokens
    with torch.no_grad():
        inp_emb = model.get_input_embeddings().weight # Get the input embedding weights
        inp_emb[-new_inp.size(0):] = new_inp # Overwrite the last `new_inp.size(0)` rows with new input embeddings

        # Get the language model head's weights, handling cases where it might be weight-tied or not
        lm_w = (model.lm_head.weight
                if hasattr(model, "lm_head") and model.lm_head.weight.shape == inp_emb.shape
                else model.get_output_embeddings().weight)
        lm_w[-new_out.size(0):] = new_out

    # 3) Wrap model with LoRA adapters
    model.eval()

    wrap_model(model, lora_dir=lora_dir, lora_mode=args.lora_mode, device=device, skip_embed=False)
    
    # 4) Prepare a batch of identical prompts for batch inference
    batch_size = int(args.batch_size)
    prompts = [args.prompt] * batch_size
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)

    # 5) Generate text using sampling parameters similar to vLLM example
    stop_id = tokenizer.convert_tokens_to_ids("[/assistant]") # Convert the stop token string to its ID
    gen_out = model.generate( # Call the model's generate method
        **inputs,
        max_new_tokens=128, # Maximum number of new tokens to generate
        temperature=0.0, # Set temperature to 0.0 for greedy decoding (deterministic output)
        do_sample=False, # Disable sampling for greedy decoding
        eos_token_id=[stop_id], # Stop generation when this token is encountered
        output_scores=True, # Return logits/scores
        return_dict_in_generate=True # Return output as a dictionary
    )

    # 7) Decode and print each of the batch outputs
    for idx in range(batch_size):
        text = tokenizer.decode(
            gen_out.sequences[idx, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        print(f"\n=== Generated Text {idx} ===")
        print(text)
        print("========================")

if __name__ == "__main__":
    main()