#!/usr/bin/env bash
# run_lora.sh

clear

nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  --force-overwrite=true \
  --output=profiling/nsys_profile \
  python run_inference.py \
    --prompt "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_74 (icao VARCHAR, airport VARCHAR)\n\n question: Name the ICAO for lilongwe international airport [/user] [assistant]" \
    --batch-size 1 \
    --lora-mode "cpu" # single_stream, multi_stream, cpu