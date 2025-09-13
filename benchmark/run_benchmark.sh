#!/bin/bash

docker run --gpus=all \
   -v .:/workspace/cgen \
   -v ${HOME}/.cache/huggingface:/root/.cache/huggingface \
   --shm-size 64g \
   cgen:latest \
   python3 benchmark/benchmark_cgen.py \
   benchmark/configs/llama3-15b-4xl4.json \
   --dataset arxiv \
   --num-requests 500 \
   --max-input-len 4096 \
   --sort-prompts-desc \
   --hf-token ${HF_TOKEN} \
   --print-output