clear

CUDA_VISIBLE_DEVICES=0 \
nsys profile \
  --output=cpu_lora_profile \
  --trace=cuda,nvtx,osrt \
  --force-overwrite true \
  --sample=cpu \
  python ../run_inference.py \
    --prompt "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_74 (icao VARCHAR, airport VARCHAR)\n\n question: Name the ICAO for lilongwe international airport [/user] [assistant]" \
    --batch-size 2  