clear
CUDA_VISIBLE_DEVICES=0 \
nsys profile \
  --output=profiles/gpu_base_cpu_lora_profile \
  --trace=cuda,nvtx,osrt \
  --sample=cpu \
  python gpu_base_cpu_lora.py \
    --prompt "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_74 (icao VARCHAR, airport VARCHAR)\n\n question: Name the ICAO for lilongwe international airport [/user] [assistant]"
