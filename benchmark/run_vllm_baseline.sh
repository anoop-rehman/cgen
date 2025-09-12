export CUDA_VISIBLE_DEVICES=0,1

# PP-heavy (prefill-friendly)
python3 benchmark/benchmark_vllm_async.py \
  --model meta-llama/Llama-2-7b-hf \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 1 \
  --enable-chunked-prefill \
  --max-num-batched-tokens 4096 \
  --max-model-len 4096 \
  --dataset arxiv --num-requests 200 \
  --input-len 1000 --output-len 100 \
  --csv runs.csv --csv-append
