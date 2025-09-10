#!/bin/bash

# Quality comparison script between Seesaw and vLLM with greedy decoding
echo "Starting Seesaw vs vLLM Quality Comparison with Greedy Decoding..."
echo "================================================================"

# Create outputs directory
mkdir -p benchmark/outputs

# Define output files  
SEESAW_OUTPUT="benchmark/outputs/seesaw_quality_output.log"
VLLM_OUTPUT="benchmark/outputs/vllm_quality_output.log"
COMPARISON_OUTPUT="benchmark/outputs/output_quality_comparison.txt"

echo "Step 1: Running Seesaw benchmark (greedy decoding)..."
echo "Running: sh benchmark/run_benchmark.sh"
sh benchmark/run_benchmark.sh 2>&1 | tee ${SEESAW_OUTPUT}

echo ""
echo "Step 2: Running vLLM baseline (greedy decoding - temperature=0.0)..."
echo "Running: sh benchmark/run_vllm_baseline.sh"  
sh benchmark/run_vllm_baseline.sh 2>&1 | tee ${VLLM_OUTPUT}

echo ""
echo "Step 3: Creating side-by-side comparison..."

# Create the comparison report
cat > ${COMPARISON_OUTPUT} << EOF
================================================================================
SEESAW vs vLLM QUALITY COMPARISON - GREEDY DECODING
================================================================================
Generated on: $(date)
Model: meta-llama/Llama-2-7b-hf
Seesaw decoding: Pure greedy (torch.argmax)
vLLM decoding: Greedy (temperature=0.0) 
Dataset: arXiv summarization
Number of requests: 10
================================================================================

PERFORMANCE METRICS
===================

Seesaw Results:
$(grep -E "(Total time|Input tokens|Output tokens|Input throughput|Output throughput)" ${SEESAW_OUTPUT} || echo "Performance metrics not found")

vLLM Results:
$(grep -E "(Total time|Input tokens|Output tokens|Input throughput|Output throughput)" ${VLLM_OUTPUT} || echo "Performance metrics not found")

================================================================================
SAMPLE OUTPUTS COMPARISON (First 10 samples)
================================================================================

EOF

# Extract and compare sample outputs using a simple approach
echo "Extracting sample outputs..."

# Extract Seesaw samples
echo "SEESAW SAMPLES:" >> ${COMPARISON_OUTPUT}
echo "===============" >> ${COMPARISON_OUTPUT}
grep -A 2 "Sample [0-9]*:" ${SEESAW_OUTPUT} | head -30 >> ${COMPARISON_OUTPUT}
echo "" >> ${COMPARISON_OUTPUT}

# Extract vLLM samples  
echo "vLLM SAMPLES:" >> ${COMPARISON_OUTPUT}
echo "=============" >> ${COMPARISON_OUTPUT}
grep -A 2 "Sample [0-9]*:" ${VLLM_OUTPUT} | head -30 >> ${COMPARISON_OUTPUT}
echo "" >> ${COMPARISON_OUTPUT}

echo "DETAILED SIDE-BY-SIDE COMPARISON:" >> ${COMPARISON_OUTPUT}
echo "==================================" >> ${COMPARISON_OUTPUT}

# Use a simple Python script to do detailed comparison
python3 << 'PYTHON_SCRIPT'
import re

def extract_samples(filename):
    """Extract sample inputs and outputs from benchmark log"""
    samples = []
    try:
        with open(filename, 'r') as f:
            content = f.read()
        
        # More flexible pattern to catch sample outputs
        lines = content.split('\n')
        current_sample = None
        current_input = None
        current_generated = None
        
        for line in lines:
            line = line.strip()
            if line.startswith('Sample ') and ':' in line:
                # Save previous sample if complete
                if current_sample and current_input and current_generated:
                    samples.append({
                        'sample': current_sample,
                        'input': current_input,
                        'generated': current_generated
                    })
                
                current_sample = line
                current_input = None
                current_generated = None
            elif line.startswith('Input: '):
                current_input = line[7:]
            elif line.startswith('Generated: '):
                current_generated = line[11:]
        
        # Don't forget the last sample
        if current_sample and current_input and current_generated:
            samples.append({
                'sample': current_sample,
                'input': current_input,
                'generated': current_generated
            })
                
    except Exception as e:
        print(f"Error reading {filename}: {e}")
    
    return samples

# Extract samples
seesaw_samples = extract_samples('benchmark/outputs/seesaw_quality_output.log')
vllm_samples = extract_samples('benchmark/outputs/vllm_quality_output.log')

print(f"Found {len(seesaw_samples)} Seesaw samples and {len(vllm_samples)} vLLM samples")

# Write detailed comparison
with open('benchmark/outputs/output_quality_comparison.txt', 'a') as f:
    max_samples = min(len(seesaw_samples), len(vllm_samples), 10)
    
    for i in range(max_samples):
        f.write(f"\nSAMPLE {i+1}\n")
        f.write("-" * 40 + "\n")
        
        # Input (should be same for both)
        if i < len(seesaw_samples):
            input_text = seesaw_samples[i]['input']
            if len(input_text) > 300:
                input_text = input_text[:300] + "..."
            f.write(f"INPUT:\n{input_text}\n\n")
        
        # Seesaw output
        f.write("SEESAW OUTPUT:\n")
        if i < len(seesaw_samples):
            f.write(f"{seesaw_samples[i]['generated']}\n")
        else:
            f.write("(No output found)\n")
        
        f.write("\n" + "-" * 20 + "\n\n")
        
        # vLLM output
        f.write("vLLM OUTPUT:\n")
        if i < len(vllm_samples):
            f.write(f"{vllm_samples[i]['generated']}\n")
        else:
            f.write("(No output found)\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    # Summary
    f.write(f"\nSUMMARY\n")
    f.write("-" * 20 + "\n")
    f.write(f"Seesaw samples: {len(seesaw_samples)}\n")
    f.write(f"vLLM samples: {len(vllm_samples)}\n")
    f.write(f"Samples compared: {max_samples}\n")
    
    # Check for identical outputs
    identical = 0
    for i in range(max_samples):
        if (i < len(seesaw_samples) and i < len(vllm_samples) and 
            seesaw_samples[i]['generated'] == vllm_samples[i]['generated']):
            identical += 1
    
    if max_samples > 0:
        f.write(f"Identical outputs: {identical}/{max_samples} ({identical/max_samples*100:.1f}%)\n")

PYTHON_SCRIPT

echo ""
echo "Quality comparison complete!"
echo "Results saved to: benchmark/outputs/output_quality_comparison.txt"
echo ""
echo "Quick summary:"
tail -5 benchmark/outputs/output_quality_comparison.txt
