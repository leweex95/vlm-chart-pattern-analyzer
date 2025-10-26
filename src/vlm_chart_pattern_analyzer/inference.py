"""VLM inference utilities with metrics collection."""
import time
import psutil
from pathlib import Path
from typing import Dict, Any
from PIL import Image


def run_inference(image_path: Path, model, processor, model_name: str = "qwen2-vl-2b") -> Dict[str, Any]:
    """Run inference with metrics collection.
    
    Args:
        image_path: Path to chart image
        model: VLM model
        processor: Model processor
        model_name: Name of the model being used
        
    Returns:
        Dictionary with metrics and results
    """
    print(f"Opening image: {image_path}")
    image = Image.open(image_path)
    
    prompt = "Analyze this trading chart and identify any chart patterns present (e.g., head and shoulders, double top, triangle, flag, wedge). Describe the pattern and trend direction."
    print("Prompt prepared, processing inputs...")
    
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
    print("Starting model generation...")
    start_time = time.perf_counter()
    outputs = model.generate(**inputs, max_new_tokens=200)
    end_time = time.perf_counter()
    print("Model generation completed")
    
    # Get memory after
    mem_after = process.memory_info().rss / 1024 / 1024  # MB
    
    # Calculate metrics
    latency_ms = (end_time - start_time) * 1000
    memory_used = mem_after - mem_before
    
    print("Decoding outputs...")
    result = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    
    # Count tokens (approximate)
    tokens_generated = len(result.split())
    
    print(f"Inference done. Latency: {latency_ms:.2f}ms, Memory: {memory_used:.2f}MB, Tokens: {tokens_generated}")
    return {
        'result': result,
        'latency_ms': latency_ms,
        'memory_mb': memory_used,
        'tokens': tokens_generated
    }
