# Use official PyTorch image with CUDA
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

RUN pip install --upgrade pip \
    && pip install "transformers" "datasets" "pillow" "accelerate" "einops" "flash-attn<=2.8.3"

# Copy fine-tuning scripts and data
COPY finetune.py ./finetune.py
COPY dataset/ ./dataset/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_LAUNCH_BLOCKING=1

# Launch with torchrun (PyTorchJob for DDP)
CMD ["torchrun", "--nproc_per_node=4", "finetune.py"]
