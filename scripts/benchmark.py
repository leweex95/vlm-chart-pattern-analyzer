#!/usr/bin/env python3
"""Benchmark VLM across multiple images and export results to CSV."""
import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from vlm_chart_pattern_analyzer.models import load_model, get_available_models
from vlm_chart_pattern_analyzer.inference import run_inference


def main():
    parser = argparse.ArgumentParser(description="Benchmark VLM across multiple images")
    parser.add_argument("--model", type=str, default="qwen2-vl-2b", choices=get_available_models(), help="Model to benchmark")
    parser.add_argument("--precision", type=str, default="fp32", choices=["fp32", "fp16", "int8"])
    parser.add_argument("--images-dir", type=str, default="data/images", help="Directory with chart images")
    parser.add_argument("--output", type=str, default="data/results/benchmark.csv", help="Output CSV file")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images to process")
    
    args = parser.parse_args()
    
    # Load model
    model, processor = load_model(args.model, args.precision)
    
    # Get images
    images_dir = Path(args.images_dir)
    image_files = sorted(images_dir.glob("*.png"))[:args.limit]
    
    if not image_files:
        print(f"No images found in {images_dir}")
        return
    
    print(f"\nBenchmarking {len(image_files)} images with {args.model} ({args.precision})...")
    
    # Prepare output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Run benchmark
    results = []
    timestamp = datetime.now().isoformat()
    
    for i, image_path in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] Processing {image_path.name}...")
        
        try:
            metrics = run_inference(image_path, model, processor, args.model)
            
            results.append({
                "timestamp": timestamp,
                "model": args.model,
                "precision": args.precision,
                "image": image_path.name,
                "latency_ms": f"{metrics['latency_ms']:.2f}",
                "memory_mb": f"{metrics['memory_mb']:.2f}",
                "tokens": metrics["tokens"],
                "result": metrics["result"]
            })
            
            print(f"  ✓ Latency: {metrics['latency_ms']:.2f}ms, Memory: {metrics['memory_mb']:.2f}MB")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    # Export to CSV
    if results:
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        
        print(f"\n✓ Results saved to {output_path}")
        print(f"  Total images: {len(results)}")
        print(f"  Avg latency: {sum(float(r['latency_ms']) for r in results) / len(results):.2f}ms")
    else:
        print("\n✗ No results to export")


if __name__ == "__main__":
    main()
