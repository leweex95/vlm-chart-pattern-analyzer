#!/usr/bin/env python3
"""Test LLaVA model loading locally."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image

def test_llava_loading():
    print("Testing LLaVA model loading...")

    model_id = "llava-hf/llava-v1.6-mistral-7b-hf"

    try:
        print("Loading processor...")
        processor = LlavaNextProcessor.from_pretrained(model_id, trust_remote_code=True)
        print("✓ Processor loaded successfully")

        print("Loading model...")
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        print("✓ Model loaded successfully")

        # Test inference with chat template
        print("Testing inference...")
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello, can you see this test message?"},
                ],
            }
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(text=prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)

        response = processor.decode(outputs[0], skip_special_tokens=True)
        print(f"✓ Inference successful: {response[:100]}...")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_llava_loading()
    sys.exit(0 if success else 1)