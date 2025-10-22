"""VLM model registry and loading utilities."""
from typing import Dict, Any, Tuple
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig

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


def load_model(model_name: str = "qwen2-vl-2b", precision: str = "fp32") -> Tuple:
    """Load VLM model with specified precision.
    
    Args:
        model_name: Model identifier (qwen2-vl-2b, llava-1.6-8b, phi-3-vision)
        precision: One of 'fp32', 'fp16', 'int8'
        
    Returns:
        Tuple of (model, processor)
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


def get_available_models() -> list:
    """Get list of available model names."""
    return list(MODEL_REGISTRY.keys())
