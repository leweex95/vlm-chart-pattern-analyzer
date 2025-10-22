"""Simple VLM inference test script - starts with one model."""
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
from pathlib import Path


def load_model():
    """Load LLaVA model."""
    print("Loading LLaVA model...")
    model_id = "llava-hf/llava-1.5-7b-hf"
    
    processor = AutoProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto"
    )
    
    print("Model loaded")
    return model, processor


def analyze_chart(image_path, model, processor):
    """Analyze a chart image."""
    image = Image.open(image_path)
    
    prompt = "Analyze this trading chart and identify any chart patterns present (e.g., head and shoulders, double top, triangle, flag, wedge). Describe the pattern and trend direction."
    
    # Prepare inputs
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    
    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(model.device)
    
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
