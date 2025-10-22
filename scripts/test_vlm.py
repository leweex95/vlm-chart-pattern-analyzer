#!/usr/bin/env python3
"""Test VLM inference on a single chart image."""
import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

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

from vlm_chart_pattern_analyzer.models import load_model, get_available_models
from vlm_chart_pattern_analyzer.inference import run_inference


def main():
    """Test VLM on one chart image."""
    parser = argparse.ArgumentParser(description="Test VLM inference with metrics")
    parser.add_argument(
        "--model",
        type=str,
        default="qwen2-vl-2b",
        choices=get_available_models(),
        help="Model to test (default: qwen2-vl-2b)"
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        choices=["fp32", "fp16", "int8"],
        help="Model precision (default: fp32)"
    )
    parser.add_argument(
        "--image",
        type=str,
        default="data/images/chart_001.png",
        help="Path to chart image"
    )
    args = parser.parse_args()
    
    # Load model
    model, processor = load_model(model_name=args.model, precision=args.precision)
    
    # Get chart
    image_path = Path(args.image)
    
    if not image_path.exists():
        print(f"Chart not found: {image_path}")
        print("Run generate_charts.py first")
        return
    
    # Analyze with metrics
    metrics = run_inference(image_path, model, processor, model_name=args.model)
    
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
