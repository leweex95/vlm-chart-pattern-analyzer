"""Simple VLM inference test script - starts with one model."""
from PIL import Image
from pathlib import Path
import sys

# Test imports first
try:
    import torch
    print(f"✓ PyTorch {torch.__version__} loaded")
except Exception as e:
    print(f"✗ PyTorch failed to load: {e}")
    sys.exit(1)

try:
    from transformers import AutoProcessor, AutoModelForVision2Seq
    print(f"✓ Transformers loaded")
except Exception as e:
    print(f"✗ Transformers failed to load: {e}")
    sys.exit(1)


def load_model():
    """Load Qwen2-VL 2B model (smaller, easier to test)."""
    print("\nLoading Qwen2-VL-2B model...")
    model_id = "Qwen/Qwen2-VL-2B-Instruct"
    
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map="cpu",  # Use CPU for now
        trust_remote_code=True
    )
    
    print("✓ Model loaded successfully")
    return model, processor


def analyze_chart(image_path, model, processor):
    """Analyze a chart image."""
    image = Image.open(image_path)
    
    prompt = "Analyze this trading chart and identify any chart patterns present (e.g., head and shoulders, double top, triangle, flag, wedge). Describe the pattern and trend direction."
    
    # Prepare inputs for Qwen2-VL
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
    
    # Generate
    print(f"\nAnalyzing {image_path.name}...")
    outputs = model.generate(**inputs, max_new_tokens=200)
    
    result = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return result


def main():
    """Test VLM on one chart image."""
    # Load model
    model, processor = load_model()
    
    # Get first chart
    image_path = Path("data/images/chart_001.png")
    
    if not image_path.exists():
        print(f"Chart not found: {image_path}")
        print("Run generate_charts.py first")
        return
    
    # Analyze
    result = analyze_chart(image_path, model, processor)
    
    print("\n" + "="*80)
    print("RESULT:")
    print("="*80)
    print(result)
    print("="*80)


if __name__ == "__main__":
    main()
