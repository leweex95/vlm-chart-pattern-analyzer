# Use official PyTorch image with CUDA
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y python3-pip \
    && pip install torch==2.8.0 torchvision --index-url https://download.pytorch.org/whl/cu121 \
    && pip install transformers>=4.57.1 datasets pillow accelerate einops flash-attn

# Copy fine-tuning scripts and data
COPY finetune.py ./finetune.py
COPY dataset/ ./dataset/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_LAUNCH_BLOCKING=1

# Launch with torchrun (PyTorchJob for DDP)
CMD ["torchrun", "--nproc_per_node=4", "finetune.py"]
