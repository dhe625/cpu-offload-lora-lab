clear
CUDA_VISIBLE_DEVICES=0 \
nsys profile \
  --output=/workspace/cpu-offload-lora-lab/profiles/gpu_base_cpu_lora_profile \
  --trace=cuda,nvtx,osrt \
  --sample=cpu \
  python gpu_base_cpu_lora.py \
    --lora-dir "/root/.cache/huggingface/hub/models--yard1--llama-2-7b-sql-lora-test/snapshots/0dfa347e8877a4d4ed19ee56c140fa518470028c" \
    --prompt "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_74 (icao VARCHAR, airport VARCHAR)\n\n question: Name the ICAO for lilongwe international airport [/user] [assistant]"
