#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List, Tuple
import argparse
import asyncio
import os
import time
import csv
from pathlib import Path

import transformers

# your loader (must return List[Tuple[prompt:str, input_tokens:int, out_tokens:int]])
from dataload import load_dataset

# vLLM (0.6.x)
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine


async def run_vllm_async(
    requests: List[Tuple[str, int, int]],
    model_name: str,
    tokenizer_name: str,
    tensor_parallel_size: int = 2,
    pipeline_parallel_size: int = 1,
    print_output: bool = False,
    max_model_len: int = 8192,
    enable_chunked_prefill: bool = True,
    max_num_batched_tokens: int = 4096,
    gpu_memory_utilization: float = 0.9,
    enforce_eager: bool = False,
) -> tuple[float, int, int]:
    """
    Run vLLM with TP+PP (via AsyncLLMEngine) and optional chunked prefill.
    Returns: (duration_sec, total_input_tokens, total_output_tokens)
    """

    # ---- Engine setup (Async) ----
    eargs = AsyncEngineArgs(
        model=model_name,
        tokenizer=tokenizer_name,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        max_model_len=max_model_len,
        enable_chunked_prefill=enable_chunked_prefill,
        max_num_batched_tokens=max_num_batched_tokens,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=False,
        enforce_eager=enforce_eager,  # helpful if CUDA graphs OOM
        distributed_executor_backend="mp",   # <— force MP, avoid Ray
    )
    engine = AsyncLLMEngine.from_engine_args(eargs)

    # ---- Prepare inputs & sampling ----
    inputs = [x[0] for x in requests]
    max_output_tokens = max((x[2] for x in requests), default=100)

    # HuggingFace tokenizer for stop tokens + token counting
    hf_tok = transformers.AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    stop_tokens = []
    if getattr(hf_tok, "eos_token", None):
        stop_tokens.append(hf_tok.eos_token)
    if getattr(hf_tok, "pad_token", None):
        stop_tokens.append(hf_tok.pad_token)

    sampling_params = SamplingParams(
        temperature=0.0,                 # greedy (as in Seesaw evals)
        max_tokens=max_output_tokens,    # cap on decoded tokens
        stop=stop_tokens or None,
    )

    print(f"Running vLLM (Async) with {len(inputs)} requests...")
    print(f"Model: {model_name}")
    print(f"TP={tensor_parallel_size}, PP={pipeline_parallel_size}")
    print(f"Chunked Prefill: {enable_chunked_prefill}, Max Batched Tokens: {max_num_batched_tokens}")
    if enforce_eager:
        print("NOTE: enforce_eager=True (CUDA graphs disabled)")

    # ---- Submit/collect using generate(...) (0.6.x async stream) ----
    async def collect_one(prompt: str, idx: int):
        # In 0.6.3, AsyncLLMEngine.generate(...) returns an async generator of RequestOutput
        try:
            agen = engine.generate(prompt, sampling_params, request_id=f"req-{idx}")
        except TypeError:
            # if request_id kw not supported in your exact subversion
            agen = engine.generate(prompt, sampling_params)
        final = None
        async for out in agen:
            if out.finished:
                final = out
                break
        return idx, final

    start = time.time()
    tasks = [asyncio.create_task(collect_one(p, i)) for i, p in enumerate(inputs)]
    done_results = await asyncio.gather(*tasks)
    duration = time.time() - start

    # Reassemble in input order
    results = [None] * len(done_results)
    for idx, out in done_results:
        results[idx] = out

    # ---- Optional: show some outputs ----
    if print_output:
        print("=== Sample vLLM Outputs (First 20) ===")
        for i, r in enumerate(results[:20]):
            if r is None or not r.outputs:
                continue
            text = (r.outputs[0].text or "")
            if text.startswith(" "):
                text = text[1:]
            print(f"\nSample {i+1}:")
            print(f"Input: {inputs[i]}")
            print(f"Generated: {text}")
            print("-" * 80)

    # ---- Stats: input/output/total tokens ----
    total_input_tokens = 0
    for inp in inputs:
        total_input_tokens += len(hf_tok.encode(inp))

    total_output_tokens = 0
    for r in results:
        if r and r.outputs:
            total_output_tokens += len(r.outputs[0].token_ids or [])

    input_tp = total_input_tokens / duration if duration > 0 else 0.0
    output_tp = total_output_tokens / duration if duration > 0 else 0.0
    total_tp = (total_input_tokens + total_output_tokens) / duration if duration > 0 else 0.0

    print(f"Input tokens: {total_input_tokens}, Output tokens: {total_output_tokens}")
    print(f"Input throughput: {input_tp:.2f} tokens/s")
    print(f"Output throughput: {output_tp:.2f} tokens/s")
    print(f"Total throughput: {total_tp:.2f} tokens/s")
    print(f"Wall time: {duration:.2f}s")

    return duration, total_input_tokens, total_output_tokens


def main():
    parser = argparse.ArgumentParser(description="vLLM async TP+PP benchmark (chunked prefill capable) [vLLM 0.6.x]")

    # Model / tokenizer
    parser.add_argument("--model", type=str, required=True, help="HF hub id or local path")
    parser.add_argument("--tokenizer", type=str, default=None, help="Tokenizer id/path (defaults to --model)")

    # Parallelism
    parser.add_argument("--tensor-parallel-size", type=int, default=2, help="Tensor parallel size (TP)")
    parser.add_argument("--pipeline-parallel-size", type=int, default=1, help="Pipeline parallel size (PP)")

    # vLLM runtime
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--enable-chunked-prefill", action="store_true")
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--enforce-eager", action="store_true", help="Disable CUDA graphs (helps if PP OOM)")

    parser.add_argument("--hf-token", type=str, required=False, help="HF token for gated models")

    # Dataset
    parser.add_argument("--dataset", type=str, default="arxiv",
                        help="sharegpt, cnn, arxiv, constant (as in your dataload.py)")
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--num-requests", type=int, default=50)
    parser.add_argument("--input-len", type=int, default=1000)
    parser.add_argument("--output-len", type=int, default=100)

    # QoL
    parser.add_argument("--sort-prompts", action="store_true", help="Sort by length (helps PP)")
    parser.add_argument("--print-output", action="store_true")

    # CSV logging (optional)
    parser.add_argument("--csv", type=str, default=None, help="Write results to CSV path")
    parser.add_argument("--csv-append", action="store_true", help="Append to CSV if exists")

    args = parser.parse_args()

    if args.tokenizer is None:
        args.tokenizer = args.model
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    # Prepare dataset
    hf_tok = transformers.AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    dataset = load_dataset(
        args.dataset,
        args.num_requests,
        hf_tok,
        max_length=args.max_model_len,
        dataset_path=args.dataset_path,
        input_len=args.input_len,
        output_len=args.output_len,
    )
    if args.sort_prompts:
        dataset.sort(key=lambda x: x[1])  # sort by input length

    # Run
    dur, in_tok, out_tok = asyncio.run(
        run_vllm_async(
            dataset,
            args.model,
            args.tokenizer,
            tensor_parallel_size=args.tensor_parallel_size,
            pipeline_parallel_size=args.pipeline_parallel_size,
            print_output=args.print_output,
            max_model_len=args.max_model_len,
            enable_chunked_prefill=args.enable_chunked_prefill,
            max_num_batched_tokens=args.max_num_batched_tokens,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enforce_eager=args.enforce_eager,
        )
    )
    print(f"Total time: {dur:.2f}s")

    # CSV logging
    if args.csv:
        path = Path(args.csv)
        write_header = (not path.exists()) or (not args.csv_append)
        mode = "a" if args.csv_append and path.exists() else "w"
        with open(path, mode, newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow([
                    "model", "tp", "pp", "max_model_len",
                    "chunked_prefill", "max_batched_tokens",
                    "gpu_mem_util", "enforce_eager",
                    "num_requests", "input_len", "output_len",
                    "duration_s", "input_tokens", "output_tokens",
                    "input_toks_per_s", "output_toks_per_s", "total_toks_per_s"
                ])
            total_tps = (in_tok + out_tok) / dur if dur > 0 else 0.0
            w.writerow([
                args.model, args.tensor_parallel_size, args.pipeline_parallel_size, args.max_model_len,
                int(args.enable_chunked_prefill), args.max_num_batched_tokens,
                args.gpu_memory_utilization, int(args.enforce_eager),
                args.num_requests, args.input_len, args.output_len,
                f"{dur:.4f}", in_tok, out_tok,
                f"{(in_tok/dur) if dur>0 else 0.0:.2f}",
                f"{(out_tok/dur) if dur>0 else 0.0:.2f}",
                f"{total_tps:.2f}"
            ])
        print(f"Wrote CSV row to: {path.resolve()}")


if __name__ == "__main__":
    main()
