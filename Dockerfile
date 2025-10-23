# Multi-stage build for optimized image size
FROM python:3.11-slim AS builder

# Install poetry and system dependencies in one layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && curl -sSL https://install.python-poetry.org | python3 - \
    && ln -s /root/.local/bin/poetry /usr/local/bin/poetry

# Configure poetry for Docker environment
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_CACHE_DIR=/tmp/poetry_cache \
    POETRY_HOME=/root/.local

WORKDIR /app

# Copy dependency files first for better layer caching
COPY pyproject.toml ./

# Install dependencies (excluding dev dependencies)
# Install PyTorch CPU version explicitly first to avoid pulling CUDA
# Note: Removed cache mount for Windows compatibility
RUN pip install --no-cache-dir torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cpu && \
    poetry install --only main --no-root --no-directory --no-cache

# Runtime stage - minimal image
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application files
COPY pyproject.toml ./
COPY src ./src
COPY scripts ./scripts

# Create necessary directories
RUN mkdir -p \
    /app/data/images \
    /app/data/results \
    /app/.cache/huggingface

# Environment variables for runtime
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    HF_HOME=/app/.cache/huggingface

# Use non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Default command - run benchmark with limited images
CMD ["python", "scripts/benchmark.py", "--limit", "5"]
