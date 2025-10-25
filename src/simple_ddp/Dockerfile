FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime
WORKDIR /app
COPY train.py .
CMD ["torchrun", "--nproc_per_node=1", "train.py"]
