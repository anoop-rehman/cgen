#!/usr/bin/env bash
set -euo pipefail

# ---- Config you can tweak via env or CLI defaults below ----
IMAGE="${IMAGE:-cgen:latest}"              # your existing image
HF_TOKEN="${HF_TOKEN:-}"                   # export HF_TOKEN in your shell if needed
SHM_SIZE="${SHM_SIZE:-64g}"                # large shm for tokenizer/cuda graphs
NGPUS_FLAG="${NGPUS_FLAG:---gpus=all}"     # or set e.g. "-e CUDA_VISIBLE_DEVICES=0,1,2,3 --gpus=all"

# Optional: restrict GPUs with CUDA_VISIBLE_DEVICES
# export CUDA_VISIBLE_DEVICES=0,1,2,3

# ---- Args to pass through to the Python script ----
MODEL="${MODEL:-meta-llama/Llama-2-7b-hf}"
TOKENIZER="${TOKENIZER:-$MODEL}"
TP="${TP:-2}"
PP="${PP:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
CHUNKED_PREFILL="${CHUNKED_PREFILL:-1}"        # 1 to enable, 0 to disable
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-4096}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"            # 1 to disable CUDA graphs

DATASET="${DATASET:-arxiv}"
DATASET_PATH="${DATASET_PATH:-}"
NUM_REQUESTS="${NUM_REQUESTS:-200}"
INPUT_LEN="${INPUT_LEN:-1000}"
OUTPUT_LEN="${OUTPUT_LEN:-100}"

SORT_PROMPTS="${SORT_PROMPTS:-1}"
PRINT_OUTPUT="${PRINT_OUTPUT:-0}"
CSV="${CSV:-runs.csv}"
CSV_APPEND="${CSV_APPEND:-1}"

# ---- Compose CLI flags ----
ENABLE_CHUNKED=""; [[ "$CHUNKED_PREFILL" == "1" ]] && ENABLE_CHUNKED="--enable-chunked-prefill"
DO_ENFORCE_EAGER=""; [[ "$ENFORCE_EAGER" == "1" ]] && DO_ENFORCE_EAGER="--enforce-eager"
DO_SORT=""; [[ "$SORT_PROMPTS" == "1" ]] && DO_SORT="--sort-prompts"
DO_PRINT=""; [[ "$PRINT_OUTPUT" == "1" ]] && DO_PRINT="--print-output"
DO_APPEND=""; [[ "$CSV_APPEND" == "1" ]] && DO_APPEND="--csv-append"

# ---- Run container ----
docker run ${NGPUS_FLAG} \
  -v "$(pwd)":/workspace/cgen \
  -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
  -e HF_TOKEN="${HF_TOKEN}" \
  -e TOKENIZERS_PARALLELISM=false \
  --shm-size "${SHM_SIZE}" \
  "${IMAGE}" \
  bash -lc "
    cd /workspace/cgen && \
    python3 benchmark/benchmark_vllm_async.py \
      --model '${MODEL}' \
      --tokenizer '${TOKENIZER}' \
      --tensor-parallel-size ${TP} \
      --pipeline-parallel-size ${PP} \
      --max-model-len ${MAX_MODEL_LEN} \
      ${ENABLE_CHUNKED} \
      --max-num-batched-tokens ${MAX_BATCHED_TOKENS} \
      --gpu-memory-utilization ${GPU_MEM_UTIL} \
      ${DO_ENFORCE_EAGER} \
      --dataset ${DATASET} \
      --dataset-path '${DATASET_PATH}' \
      --num-requests ${NUM_REQUESTS} \
      --input-len ${INPUT_LEN} \
      --output-len ${OUTPUT_LEN} \
      ${DO_SORT} \
      ${DO_PRINT} \
      --csv '${CSV}' \
      ${DO_APPEND} \
      ${HF_TOKEN:+--hf-token \$HF_TOKEN}
  "
