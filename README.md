[![Build and push Docker image](https://github.com/leweex95/vlm-chart-pattern-analyzer/actions/workflows/docker-build.yml/badge.svg)](https://github.com/leweex95/vlm-chart-pattern-analyzer/actions/workflows/docker-build.yml)

# VLM Chart Pattern Analyzer

A high-performance benchmarking suite for Vision Language Models (VLMs) detecting chart patterns in financial trading data (stock/forex).

## Overview

This project provides a comprehensive framework for:

1. **Chart Generation** - Generate candlestick charts from real market data (MetaTrader5)
2. **VLM Inference** - Test multiple state-of-the-art VLMs on chart pattern recognition
3. **Performance Benchmarking** - Measure latency, memory usage, and token output across models and precision levels
4. **Quantization Support** - Test models in FP32, FP16, and INT8 precision
5. **Interactive Visualizations** - Analyze results with Plotly-based interactive charts
6. **Docker Deployment** - Containerized workflow with GitHub Actions CI/CD

## Supported Vision Language Models

| Model | Size | Notes |
|-------|------|-------|
| **Qwen2-VL-2B** | 2B params | Fastest, most efficient |
| **LLaVA-1.6-8B** | 8B params | Balanced performance |
| **Phi-3-Vision** | 3.8B params | Efficient alternative |

## Installation

### Prerequisites

- Python 3.11+
- PyTorch 2.3.0 (CPU by default)
- Poetry (dependency management)
- MetaTrader5 terminal (optional, for real market data)

### Setup

```bash
# Clone repository
git clone https://github.com/leweex95/vlm-chart-pattern-analyzer.git
cd vlm-chart-pattern-analyzer

# Install dependencies via Poetry
poetry install

# Activate Poetry shell (optional)
poetry shell
```

## Quick Start

### 1. Generate Chart Images

Generate candlestick charts from market data:

```bash
poetry run python scripts/generate_charts.py --num-charts 25
```

This creates 25 random EURUSD H1 charts in `data/images/`.

Options:
- `--num-charts N` - Number of charts to generate (default: 25)
- `--symbol SYMBOL` - Currency pair (default: EURUSD)
- `--timeframe {M1,M5,M15,M30,H1,H4,D1,W1,MN1}` - Candle timeframe (default: H1)
- `--days DAYS` - Historical data range (default: 30)
- `--output-dir DIR` - Output directory (default: data/images)

### 2. Test Single Image

Test a single chart image with a VLM:

```bash
poetry run python scripts/test_vlm.py data/images/chart_001.png --model qwen2-vl-2b --precision fp32
```

Options:
- `--model {qwen2-vl-2b,llava-1.6-8b,phi-3-vision}` - Model to use
- `--precision {fp32,fp16,int8}` - Precision level
- `--output FILE` - Save results to JSON file

### 3. Run Benchmarks

Benchmark models on a dataset of images:

```bash
poetry run python scripts/benchmark.py --model qwen2-vl-2b --precision fp32 --limit 10
```

Options:
- `--model` - Model name
- `--precision` - Precision level (fp32, fp16, int8)
- `--images-dir` - Input directory with chart images
- `--limit N` - Limit number of images to benchmark
- `--output FILE` - Output CSV path

Results are saved to `data/results/benchmark.csv`.

### 4. Generate Visualizations

Create interactive performance analysis charts:

```bash
poetry run python scripts/visualize.py
```

This generates:
- 6 interactive HTML files with Plotly visualizations
- Summary statistics text report

Options:
- `--input FILE` - Benchmark CSV to visualize
- `--output DIR` - Output directory for HTML files
- `--summary FILE` - Summary statistics output path

Open the HTML files in your browser to explore interactive charts!

## Project Structure

```
vlm-chart-pattern-analyzer/
├── src/vlm_chart_pattern_analyzer/
│   ├── __init__.py              # Package exports
│   ├── models.py                # Model registry and loading
│   ├── inference.py             # VLM inference pipeline
│   └── visualization.py         # Plotly chart generation
├── scripts/
│   ├── generate_charts.py       # MT5 chart generation
│   ├── test_vlm.py              # Single image inference test
│   ├── benchmark.py             # Batch benchmarking suite
│   ├── visualize.py             # Visualization CLI
│   ├── ci_auto_fix.py           # GitHub Actions trigger
│   └── test_visualizations.py   # Visualization testing
├── data/
│   ├── images/                  # Generated chart images
│   └── results/
│       ├── benchmark.csv        # Benchmark results
│       ├── visualizations/      # HTML chart files
│       └── summary.txt          # Summary statistics
├── tests/                       # Unit tests
├── Dockerfile                   # Docker build configuration
├── pyproject.toml               # Poetry dependencies
└── README.md                    # This file
```

## Core Modules

### Models (`src/vlm_chart_pattern_analyzer/models.py`)

Model loading with quantization support:

```python
from vlm_chart_pattern_analyzer import load_model, MODEL_REGISTRY

# See available models
print(MODEL_REGISTRY.keys())

# Load model with specific precision
model, processor = load_model('qwen2-vl-2b', precision='fp16')
```

### Inference (`src/vlm_chart_pattern_analyzer/inference.py`)

Run inference on chart images:

```python
from vlm_chart_pattern_analyzer import run_inference

results = run_inference(
    image_path='data/images/chart_001.png',
    model=model,
    processor=processor,
    model_name='qwen2-vl-2b'
)
# Returns: {
#   'response': str,
#   'latency_ms': float,
#   'memory_mb': float,
#   'tokens': int
# }
```

### Visualization (`src/vlm_chart_pattern_analyzer/visualization.py`)

Generate interactive charts:

```python
from vlm_chart_pattern_analyzer import (
    load_benchmark_results,
    plot_comprehensive_dashboard
)

df = load_benchmark_results('data/results/benchmark.csv')
plot_comprehensive_dashboard(df, 'data/results/visualizations')
```

## Benchmark Metrics

Each benchmark records:

- **Latency (ms)** - Inference time in milliseconds
- **Memory (MB)** - Peak memory usage during inference
- **Tokens** - Number of tokens generated by model
- **Model** - Which VLM was used
- **Precision** - FP32, FP16, or INT8
- **Image** - Source image filename
- **Timestamp** - When benchmark was run

## Quantization Levels

Supported precision levels with tradeoffs:

| Precision | Memory | Speed | Quality | Use Case |
|-----------|--------|-------|---------|----------|
| FP32 | High | Slower | Best | Accuracy-critical |
| FP16 | ~50% | Faster | Good | Balanced |
| INT8 | ~25% | Fastest | Fair | Speed-critical |

## Docker Deployment

Build and run in Docker:

```bash
# Build image
docker build -t vlm-analyzer .

# Run benchmark
docker run --rm -v $(pwd)/data:/app/data vlm-analyzer \
    poetry run python scripts/benchmark.py --limit 5

# View results
ls data/results/
```

## GitHub Actions CI/CD

Automated Docker builds and tests on every push (via self-hosted Windows runner):

```yaml
# .github/workflows/docker-build.yml
- Builds Docker image
- Runs benchmarks in container
- Pushes to GitHub Container Registry (ghcr.io)
```

View workflow status and artifacts on GitHub Actions tab.

## Performance Tips

### Model Selection

- **Qwen2-VL-2B** - Recommended for speed-critical applications (~250ms latency)
- **LLaVA-1.6-8B** - Recommended for quality-critical applications (~900ms latency)
- **Phi-3-Vision** - Balanced option (~400ms latency)

### Precision Selection

- Use **INT8** for real-time applications (>3x speedup, 25% memory)
- Use **FP16** for balanced performance (2x speedup, 50% memory)
- Use **FP32** for highest accuracy (baseline reference)

### Batch Processing

```bash
# Benchmark multiple configurations
for model in qwen2-vl-2b llava-1.6-8b phi-3-vision; do
  for precision in fp32 fp16 int8; do
    poetry run python scripts/benchmark.py \
      --model $model --precision $precision --limit 100
  done
done
```

## Development

### Running Tests

```bash
# Test visualization module with sample data
poetry run python scripts/test_visualizations.py

# Run pytest suite
poetry run pytest tests/
```

### Code Quality

```bash
# Format code
poetry run black src/ scripts/ tests/

# Lint
poetry run ruff check src/ scripts/ tests/
```

### Adding New Models

1. Update `MODEL_REGISTRY` in `src/vlm_chart_pattern_analyzer/models.py`
2. Add model-specific prompt in `src/vlm_chart_pattern_analyzer/inference.py`
3. Test with `poetry run python scripts/test_vlm.py --model <new-model>`

## Troubleshooting

### PyTorch Installation Issues

If you encounter PyTorch DLL errors on Windows:

```bash
# Reinstall PyTorch via Poetry
poetry lock --no-cache
poetry install
```

### MetaTrader5 Connection

If chart generation fails:
- Ensure MetaTrader5 terminal is running
- Check EURUSD (or specified symbol) is available
- Verify 30+ days of historical data is available

### GPU Support

This project uses CPU by default. For GPU support:

1. Update `torch` in `pyproject.toml` to CUDA-enabled version
2. Run `poetry lock && poetry install`
3. GPU inference will be automatic

## License

[Your License Here]

## Citation

If you use this project in research, please cite:

```bibtex
@software{vlm_chart_pattern_analyzer,
  title={VLM Chart Pattern Analyzer},
  author={[Your Name]},
  year={2025},
  url={https://github.com/leweex95/vlm-chart-pattern-analyzer}
}
```

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit pull request

## Support

For issues, questions, or suggestions:
- Open a GitHub Issue
- Check existing documentation in `STEP*.md` files
- Review benchmark results in `data/results/`

## Roadmap

- [ ] Step 9: Pattern similarity metrics (cosine similarity between model outputs)
- [ ] Step 10: Helm deployment for Kubernetes
- [ ] Step 11: GitHub Pages dashboard with result history
- [ ] Step 12: Real-time monitoring with Prometheus metrics

## Changelog

### Version 0.1.0 (2025-10-22)

**Steps Completed:**
- Step 1: Basic project setup ✓
- Step 2: Chart generation from MT5 ✓
- Step 3: VLM inference testing ✓
- Step 4: Performance metrics collection ✓
- Step 5: Quantization support (FP32/FP16/INT8) ✓
- Step 6: Batch benchmarking with CSV export ✓
- Step 7: Multi-model support (Qwen2/LLaVA/Phi-3) ✓
- Step 8: Plotly interactive visualizations ✓

**Key Features:**
- 3 state-of-the-art VLMs supported
- 3 precision levels (FP32, FP16, INT8)
- Real market data (MetaTrader5)
- Interactive performance visualizations
- Docker containerization
- GitHub Actions CI/CD
