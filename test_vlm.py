"""Simple VLM inference test script - supports multiple models."""
from PIL import Image
from pathlib import Path
import sys
import time
import psutil
import argparse

# Test imports first
try:
    import torch
    print(f"✓ PyTorch {torch.__version__} loaded")
except Exception as e:
    print(f"✗ PyTorch failed to load: {e}")
    sys.exit(1)

try:
    from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
    print(f"✓ Transformers loaded")
except Exception as e:
    print(f"✗ Transformers failed to load: {e}")
    sys.exit(1)

# Model registry - easily extensible
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
    """Load VLM model with specified precision.
    
    Args:
        model_name: Model identifier (qwen2-vl-2b, llava-1.6-8b, phi-3-vision)
        precision: One of 'fp32', 'fp16', 'int8'
    """
    print(f"\nLoading {model_name} ({precision})...")
    
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")
    
    config = MODEL_REGISTRY[model_name]
    model_id = config["model_id"]
    
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, **config["processor_kwargs"])
    
    # Configure precision
    kwargs = config["model_kwargs"].copy()
    kwargs.update({
        "device_map": "cpu",  # Use CPU for now
    })
    
    if precision == "fp16":
        kwargs["torch_dtype"] = torch.float16
    elif precision == "int8":
        # INT8 quantization
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        kwargs["quantization_config"] = quantization_config
        kwargs["device_map"] = "auto"  # INT8 needs auto device mapping
    else:  # fp32
        kwargs["torch_dtype"] = torch.float32
    
    model = AutoModelForVision2Seq.from_pretrained(model_id, **kwargs)
    
    print(f"✓ Model loaded successfully ({precision})")
    return model, processor


def analyze_chart(image_path, model, processor, model_name="qwen2-vl-2b"):
    """Analyze a chart image with metrics."""
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
    print(f"\nAnalyzing {image_path.name} with {model_name}...")
    
    # Get memory before
    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024  # MB
    
    # Measure inference time
    start_time = time.perf_counter()
    outputs = model.generate(**inputs, max_new_tokens=200)
    end_time = time.perf_counter()
    
    # Get memory after
    mem_after = process.memory_info().rss / 1024 / 1024  # MB
    
    # Calculate metrics
    latency_ms = (end_time - start_time) * 1000
    memory_used = mem_after - mem_before
    
    result = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    
    # Count tokens (approximate)
    tokens_generated = len(result.split())
    
    return {
        'result': result,
        'latency_ms': latency_ms,
        'memory_mb': memory_used,
        'tokens': tokens_generated
    }


def main():
    """Test VLM on one chart image."""
    parser = argparse.ArgumentParser(description="Test VLM inference with metrics")
    parser.add_argument(
        "--model",
        type=str,
        default="qwen2-vl-2b",
        choices=list(MODEL_REGISTRY.keys()),
        help="Model to test (default: qwen2-vl-2b)"
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        choices=["fp32", "fp16", "int8"],
        help="Model precision (default: fp32)"
    )
    args = parser.parse_args()
    
    # Load model
    model, processor = load_model(model_name=args.model, precision=args.precision)
    
    # Get first chart
    image_path = Path("data/images/chart_001.png")
    
    if not image_path.exists():
        print(f"Chart not found: {image_path}")
        print("Run generate_charts.py first")
        return
    
    # Analyze with metrics
    metrics = analyze_chart(image_path, model, processor, model_name=args.model)
    
    print("\n" + "="*80)
    print(f"METRICS ({args.model} - {args.precision}):")
    print("="*80)
    print(f"Latency:       {metrics['latency_ms']:.2f} ms")
    print(f"Memory used:   {metrics['memory_mb']:.2f} MB")
    print(f"Tokens:        {metrics['tokens']}")
    print("="*80)
    print("\nRESULT:")
    print("="*80)
    print(metrics['result'])
    print("="*80)


if __name__ == "__main__":
    main()
