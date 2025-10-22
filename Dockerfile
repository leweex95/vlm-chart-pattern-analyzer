# Multi-stage build for optimized image size
FROM python:3.11-slim AS builder

# Install poetry and system dependencies in one layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
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
RUN --mount=type=cache,target=/tmp/poetry_cache \
    pip install --no-cache-dir torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cpu && \
    poetry install --only main --with inference --with charting --no-root --no-directory

# Runtime stage - minimal image
FROM python:3.11-slim AS runtime

# Install only git (required for transformers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application files
COPY *.py ./
COPY pyproject.toml ./

# Create necessary directories
RUN mkdir -p \
    /app/data/images \
    /app/data/results \
    /app/.cache/huggingface

# Environment variables for runtime
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    HF_HOME=/app/.cache/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=1

# Health check to verify container is responsive
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Use non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Default command - run benchmark with limited images
CMD ["python", "benchmark.py", "--limit", "5"]
