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
                        help="single-stream | multi-stream | cpu")
    args = parser.parse_args()

    repo_id = "yard1/llama-2-7b-sql-lora-test"
    lora_dir = snapshot_download(repo_id=repo_id)

    if not os.path.isdir(lora_dir):
        raise RuntimeError(f"Failed to download LoRA adapter to {lora_dir}")
    print(f"LoRA adapter downloaded to {lora_dir}")

    # Load base model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(lora_dir, use_fast=False, token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16 if device.startswith("cuda") else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_id,
        device_map={"": 0} if device.startswith("cuda") else None,
        torch_dtype=dtype,
        trust_remote_code=False,
        token=HF_TOKEN,
    )
    model.resize_token_embeddings(len(tokenizer))

    # Overwrite new embeddings (if added during fine-tuning)
    loaded = safetensors.torch.load_file(f"{lora_dir}/new_embeddings.safetensors")
    new_inp = loaded["input_embeddings"].to(device=device, dtype=dtype)
    new_out = loaded["output_embeddings"].to(device=device, dtype=dtype)
    with torch.no_grad():
        inp_emb = model.get_input_embeddings().weight
        inp_emb[-new_inp.size(0):] = new_inp

        lm_w = (model.lm_head.weight
                if hasattr(model, "lm_head") and model.lm_head.weight.shape == inp_emb.shape
                else model.get_output_embeddings().weight)
        lm_w[-new_out.size(0):] = new_out

    # Wrap model with LoRA adapters
    model.eval()
    wrap_model(model, lora_dir=lora_dir, lora_mode=args.lora_mode, device=device, skip_embed=False)
    
    # Prepare batch prompts
    batch_size = int(args.batch_size)
    prompts = [args.prompt] * batch_size
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)

    # Generate text
    stop_id = tokenizer.convert_tokens_to_ids("[/assistant]")
    gen_out = model.generate(
        **inputs,
        max_new_tokens=128,
        temperature=0.0,
        do_sample=False,
        eos_token_id=[stop_id],
        output_scores=True,
        return_dict_in_generate=True
    )

    # Decode outputs
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
