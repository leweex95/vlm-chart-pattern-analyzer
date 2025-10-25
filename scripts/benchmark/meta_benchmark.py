#!/usr/bin/env python3
"""
Meta orchestrator to run VLM benchmarks for SmolVLM2-2.2B across fp32, bfloat16, int8, int4.
Runs each config sequentially and ensures all results are aggregated in the same CSV file.
"""
import subprocess
import sys
import os


# Test SmolVLM2-2.2B-Instruct across all four precision values
CONFIGS = [
    "fp32",
    "fp16",
    "quantized-int8",
    "quantized-int4"
]
MODEL = "smolvlm2-2.2b"
NOTEBOOK = "src/vlm_chart_pattern_analyzer/kaggle/config/vlm-inference-benchmark.ipynb"

def run_benchmark(config):
    print(f"\n=== Running benchmark for config: {config} ===")
    cmd = [
        sys.executable, "scripts/benchmark.py",
        "--mode", "kaggle",
        "--models", MODEL,
        "--config", config,
        "--notebook", NOTEBOOK
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"ERROR: Benchmark failed for config {config}")
    else:
        print(f"SUCCESS: Benchmark completed for config {config}")

def main():
    for config in CONFIGS:
        run_benchmark(config)
    print("\nAll benchmarks complete. Check the results CSV for aggregated outputs.")

if __name__ == "__main__":
    main()