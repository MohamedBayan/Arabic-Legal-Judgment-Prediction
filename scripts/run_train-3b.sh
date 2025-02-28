#!/bin/bash

python3 train.py \
  --model_name "Model directory" \
  --max_seq_length 2048 \
  --quant_bits -1 \
  --use_nested_quant False \
  --batch_size 8 \
  --grad_size 4 \
  --epochs 2 \
  --out_dir "Output directory" \
  --save_steps 500 \
  --train_set_dir "Training data directory" \
  --start_from_last_checkpoint False \
  --lora_adapter_dir "Lora adapter directory" \
  --merged_model_dir "Merged model directory"
