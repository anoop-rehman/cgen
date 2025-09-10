#!/bin/bash

echo "=========================================="
echo "Seesaw vs vLLM Baseline Comparison"
echo "=========================================="

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo "Warning: HF_TOKEN not set. You may need to set it for model access."
fi

echo ""
echo "Running Seesaw (cgen) benchmark..."
echo "----------------------------------"
time sh benchmark/run_benchmark.sh 2>&1 | tee seesaw_results.txt

echo ""
echo "Running vLLM baseline benchmark..."
echo "--------------------------------"
time sh benchmark/run_vllm_baseline.sh 2>&1 | tee vllm_results.txt

echo ""
echo "=========================================="
echo "COMPARISON SUMMARY"
echo "=========================================="

echo ""
echo "Seesaw Results:"
echo "---------------"
grep -E "Total time:|Input throughput:|Output throughput:|Total throughput:" seesaw_results.txt || \
grep -E "Total time:" seesaw_results.txt

echo ""
echo "vLLM Results:"
echo "-------------"
grep -E "Total time:|Input throughput:|Output throughput:|Total throughput:" vllm_results.txt

echo ""
echo "Results saved to:"
echo "- seesaw_results.txt"
echo "- vllm_results.txt"
