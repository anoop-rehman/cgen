#!/bin/bash

docker run --gpus=all \
   -v .:/workspace/cgen \
   -v ${HOME}/.cache/huggingface:/root/.cache/huggingface \
   --shm-size 64g \
   cgen:latest \
   python3 benchmark/benchmark_vllm.py \
   --model meta-llama/Llama-2-7b-hf \
   --tokenizer meta-llama/Llama-2-7b-hf \
   --dataset arxiv \
   --num-requests 10 \
   --tensor-parallel-size 2 \
   --pipeline-parallel-size 2 \
   --enable-chunked-prefill \
   --max-model-len 4096 \
   --max-num-batched-tokens 2048 \
   --hf-token ${HF_TOKEN} \
   --print-output
