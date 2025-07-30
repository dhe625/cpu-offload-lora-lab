#!/usr/bin/env bash
# run_lora.sh

clear

python gpu_base_cpu_lora.py \
  --lora-dir "/root/.cache/huggingface/hub/models--yard1--llama-2-7b-sql-lora-test/snapshots/0dfa347e8877a4d4ed19ee56c140fa518470028c" \
  --prompt "Hello, my name is"