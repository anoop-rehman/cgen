#!/bin/bash

docker run --gpus=all \
   -v .:/workspace/cgen \
   -v ${HOME}/.cache/huggingface:/root/.cache/huggingface \
   --shm-size 64g \
   cgen:latest \
   python3 benchmark/benchmark_cgen.py \
   benchmark/configs/llama3-15b-4xl4.json \
   --dataset sharegpt \
   --dataset-path /datasets/sharegpt.json \
   --num-requests 10 \
   --hf-token ${HF_TOKEN} \
   --print-output