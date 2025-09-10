#!/bin/bash

docker run --gpus=all \
   -v .:/workspace/cgen \
   -v ${HOME}/.cache/huggingface:/root/.cache/huggingface \
   --shm-size 64g \
   cgen:latest \
   python3 benchmark/benchmark_cgen.py \
   benchmark/configs/llama2-7b-2xl4.json \
   --dataset arxiv \
   --num-requests 50 \
   --hf-token ${HF_TOKEN} \
   --print-output