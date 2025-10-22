"""Benchmark VLM across multiple images and export results to CSV."""
import argparse
import csv
from datetime import datetime
from pathlib import Path
import sys
import time
import psutil

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig

# Model registry
MODEL_REGISTRY = {
    "qwen2-vl-2b": {
        "model_id": "Qwen/Qwen2-VL-2B-Instruct",
        "processor_kwargs": {},
        "model_kwargs": {"trust_remote_code": True},
    },
    "llava-1.6-8b": {
        "model_id": "llava-hf/llava-1.6-8b-hf",
        "processor_kwargs": {},
        "model_kwargs": {},
    },
    "phi-3-vision": {
        "model_id": "microsoft/Phi-3-vision-128k-instruct",
        "processor_kwargs": {},
        "model_kwargs": {},
    },
}


def load_model(model_name="qwen2-vl-2b", precision="fp32"):
    """Load VLM model with specified precision."""
    print(f"\nLoading {model_name} ({precision})...")
    
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")
    
    config = MODEL_REGISTRY[model_name]
    model_id = config["model_id"]
    
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, **config["processor_kwargs"])
    
    # Configure precision
    kwargs = config["model_kwargs"].copy()
    kwargs.update({
        "device_map": "cpu",
    })
    
    if precision == "fp16":
        kwargs["torch_dtype"] = torch.float16
    elif precision == "int8":
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        kwargs["quantization_config"] = quantization_config
        kwargs["device_map"] = "auto"
    else:
        kwargs["torch_dtype"] = torch.float32
    
    model = AutoModelForVision2Seq.from_pretrained(model_id, **kwargs)
    print(f"✓ Model loaded")
    
    return model, processor


def run_inference(image_path, model, processor, model_name="qwen2-vl-2b"):
    """Run inference with metrics collection."""
    image = Image.open(image_path)
    
    prompt = "Analyze this trading chart and identify any chart patterns present (e.g., head and shoulders, double top, triangle, flag, wedge). Describe the pattern and trend direction."
    
    # Prepare inputs based on model
    if model_name == "qwen2-vl-2b":
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text_prompt], images=[image], return_tensors="pt")
    elif model_name == "llava-1.6-8b":
        inputs = processor(prompt, image, return_tensors="pt")
    elif model_name == "phi-3-vision":
        messages = [
            {"role": "user", "content": f"<|image_1|>\n{prompt}"}
        ]
        inputs = processor(messages, images=[image], return_tensors="pt")
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Measure metrics
    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024
    
    start_time = time.perf_counter()
    outputs = model.generate(**inputs, max_new_tokens=200)
    end_time = time.perf_counter()
    
    mem_after = process.memory_info().rss / 1024 / 1024
    
    latency_ms = (end_time - start_time) * 1000
    memory_mb = mem_after - mem_before
    
    result = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    tokens = len(result.split())
    
    return {
        "latency_ms": latency_ms,
        "memory_mb": memory_mb,
        "tokens": tokens,
        "result": result
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark VLM across multiple images")
    parser.add_argument("--model", type=str, default="qwen2-vl-2b", choices=list(MODEL_REGISTRY.keys()), help="Model to benchmark")
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
