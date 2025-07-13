#!/bin/bash

# Go to project root (assumes you're in scripts/)
cd "$(dirname "$0")/.."

# Export project root to PYTHONPATH
export PYTHONPATH="$(pwd)"
export TOKENIZERS_PARALLELISM=true

# Optional: restrict GPUs
# export CUDA_VISIBLE_DEVICES=0,1

# Run training
accelerate launch \
  --mixed_precision fp16 \
  --main_process_port 29601 \
  src/train.py
