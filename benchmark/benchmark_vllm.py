from typing import List, Optional, Tuple

import argparse
import transformers
import torch
import os
import time
import json
from dataload import load_dataset

# vLLM imports
from vllm import LLM, SamplingParams


def run_vllm(
    requests: List[Tuple[str, int, int]],
    model_name: str,
    tokenizer_name: str,
    tensor_parallel_size: int = 2,
    pipeline_parallel_size: int = 1,
    print_output: bool = False,
    max_model_len: int = 8192,
    enable_chunked_prefill: bool = True,
    max_num_batched_tokens: int = 4096,
) -> float:
    """Run vLLM baseline benchmark."""
    
    # Initialize vLLM engine
    llm = LLM(
        model=model_name,
        tokenizer=tokenizer_name,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        max_model_len=max_model_len,
        enable_chunked_prefill=enable_chunked_prefill,
        max_num_batched_tokens=max_num_batched_tokens,
        gpu_memory_utilization=0.9,  # Use most of GPU memory
        trust_remote_code=False,
    )
    
    # Prepare inputs and sampling parameters
    inputs = [x[0] for x in requests]
    max_tokens_list = [x[2] for x in requests]  # output length only
    
    # Create sampling params (we'll use the max output length from requests)
    max_output_tokens = max(max_tokens_list) if max_tokens_list else 100
    
    # Get tokenizer for stop tokens
    tokenizer = llm.get_tokenizer()
    stop_tokens = []
    if hasattr(tokenizer, 'eos_token') and tokenizer.eos_token:
        stop_tokens.append(tokenizer.eos_token)
    if hasattr(tokenizer, 'pad_token') and tokenizer.pad_token:
        stop_tokens.append(tokenizer.pad_token)
    
    sampling_params = SamplingParams(
        temperature=0.1,  # Small temperature for more diverse but still consistent output
        top_p=0.9,  # Nucleus sampling to prevent repetitive loops
        max_tokens=max_output_tokens,
        stop=stop_tokens,
        repetition_penalty=1.1,  # Penalize repetition
    )
    
    print(f"Running vLLM benchmark with {len(inputs)} requests...")
    print(f"Model: {model_name}")
    print(f"Tensor Parallel: {tensor_parallel_size}, Pipeline Parallel: {pipeline_parallel_size}")
    print(f"Chunked Prefill: {enable_chunked_prefill}, Max Batched Tokens: {max_num_batched_tokens}")
    
    # Run benchmark
    start_time = time.time()
    outputs = llm.generate(inputs, sampling_params)
    end_time = time.time()
    
    duration = end_time - start_time
    
    # Extract generated text
    generated_texts = [output.outputs[0].text for output in outputs]
    
    if print_output:
        print("=== Sample vLLM Outputs (First 20) ===")
        for i in range(min(20, len(generated_texts))):
            print(f"\nSample {i+1}:")
            print(f"Input: {inputs[i]}")
            print(f"Generated: {generated_texts[i]}")
            print("-" * 80)
    
    # Calculate statistics
    tokenizer = llm.get_tokenizer()
    total_input_tokens = sum(len(tokenizer.encode(inp)) for inp in inputs)
    total_output_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    
    input_throughput = total_input_tokens / duration
    output_throughput = total_output_tokens / duration
    
    print(f"Input tokens: {total_input_tokens}, Output tokens: {total_output_tokens}")
    print(f"Input throughput: {input_throughput:.2f} tokens/s")
    print(f"Output throughput: {output_throughput:.2f} tokens/s")
    print(f"Total throughput: {(total_input_tokens + total_output_tokens) / duration:.2f} tokens/s")
    
    return duration


def launch(args):
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
    
    # Load tokenizer to prepare dataset
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer)
    
    # Load dataset
    datasets = load_dataset(
        args.dataset,
        args.num_requests,
        tokenizer,
        max_length=args.max_model_len,  # Use the same max length as the model
        dataset_path=args.dataset_path,
        input_len=args.input_len,
        output_len=args.output_len
    )
    
    if args.sort_prompts:
        datasets.sort(key=lambda x: x[1])
    
    # Run vLLM benchmark
    dur = run_vllm(
        datasets,
        args.model,
        args.tokenizer,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        print_output=args.print_output,
        max_model_len=args.max_model_len,
        enable_chunked_prefill=args.enable_chunked_prefill,
        max_num_batched_tokens=args.max_num_batched_tokens,
    )
    
    print(f"Total time: {dur:.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="vLLM baseline benchmark")
    
    # Model configuration
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--tokenizer", type=str, default=None, help="Tokenizer name or path (defaults to model)")
    
    # Parallelization settings
    parser.add_argument("--tensor-parallel-size", type=int, default=2, help="Tensor parallel size")
    parser.add_argument("--pipeline-parallel-size", type=int, default=1, help="Pipeline parallel size")
    
    # vLLM specific settings
    parser.add_argument("--max-model-len", type=int, default=8192, help="Maximum model sequence length")
    parser.add_argument("--enable-chunked-prefill", action="store_true", help="Enable chunked prefill")
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096, help="Maximum number of batched tokens for chunked prefill")
    
    # Dataset configuration
    parser.add_argument("--dataset", type=str, default="arxiv", help="Dataset used for benchmarking. can be sharegpt, cnn, arxiv, constant.")
    parser.add_argument("--dataset-path", type=str, default=None, help="Path to the dataset if needed.")
    parser.add_argument("--num-requests", type=int, default=50, help="Number of requests sent to the engine.")
    
    # Other options
    parser.add_argument("--sort-prompts", action="store_true", help="Sort the prompt by length. Beneficial for pipeline parallelism.")
    parser.add_argument("--print-output", action='store_true', help="Print generation results after benchmarking.")
    parser.add_argument("--input-len", type=int, default=1000, help="Number of prompt tokens in a request(when dataset=constant)")
    parser.add_argument("--output-len", type=int, default=100, help="Number of generated tokens (when dataset=constant)")
    parser.add_argument("--hf-token", type=str, required=False, help="HuggingFace token")
    
    args = parser.parse_args()
    
    # Set tokenizer to model if not specified
    if args.tokenizer is None:
        args.tokenizer = args.model
    
    launch(args)
