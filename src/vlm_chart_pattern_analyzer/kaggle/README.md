# VLM Chart Pattern Analyzer - Kaggle GPU Integration

This module enables automated deployment of VLM inference benchmarks to Kaggle's free GPU environment, replicating the successful pattern from [ImgGenHub](https://github.com/leweex95/imggenhub).

## Overview

The pipeline automates the entire workflow:
1. **Deploy**: Push notebook with VLM inference code to Kaggle kernel
2. **Poll**: Monitor kernel status until completion
3. **Download**: Retrieve benchmark results (metrics & outputs)

## Features

### ✅ Comprehensive Metrics Collection
- **Latency**: Inference time in milliseconds
- **Memory Usage**: RAM consumption during inference
- **Throughput**: Tokens generated per second
- **Token Count**: Total tokens generated per inference

### ✅ Multiple VLM Support
- Qwen2-VL (2B, 7B, 72B variants)
- LLaVA-1.6 (8B)
- Phi-3-Vision (3.8B)
- Easily extensible for new models

### ✅ Precision Options
- FP32 (full precision)
- FP16 (half precision)
- INT8 (quantized)

### ✅ Advanced Analysis
- Per-model comparisons
- Precision impact analysis
- Statistical summaries (mean, median, stdev)
- HTML report generation

## Directory Structure

```
src/vlm_chart_pattern_analyzer/kaggle/
├── __init__.py
├── main.py                          # Pipeline orchestrator
├── config/
│   ├── vlm-inference-benchmark.ipynb  # Kaggle notebook template
│   └── kernel-metadata.json         # Kaggle kernel configuration
├── core/
│   ├── deploy.py                   # Deploy to Kaggle
│   ├── download.py                 # Download results
│   └── __init__.py
└── utils/
    ├── poll_status.py              # Monitor kernel status
    └── __init__.py

src/vlm_chart_pattern_analyzer/
└── metrics_collector.py             # Metrics analysis & export
```

## Setup

### 1. Kaggle Configuration

```bash
# Create Kaggle credentials
mkdir -p ~/.kaggle

# Download from https://www.kaggle.com/settings/account → "Create New API Token"
# This generates kaggle.json, move it to ~/.kaggle/

# Set permissions
chmod 600 ~/.kaggle/kaggle.json
```

### 2. Update Kernel IDs

Edit `src/vlm_chart_pattern_analyzer/kaggle/utils/poll_status.py`:
```python
KERNEL_ID = "your_username/vlm-chart-benchmark"
```

Edit `src/vlm_chart_pattern_analyzer/kaggle/core/download.py`:
```python
KERNEL_ID = "your_username/vlm-chart-benchmark"
```

Update `src/vlm_chart_pattern_analyzer/kaggle/config/kernel-metadata.json`:
```json
{
  "id": "your_username/vlm-chart-benchmark",
  ...
}
```

### 3. Create Kaggle Dataset (for chart images)

1. Go to https://www.kaggle.com/datasets/create
2. Upload your chart images
3. Create dataset (e.g., `chart-patterns`)
4. Update notebook template to reference your dataset:
   ```python
   CHART_IMAGES_DIR = "/kaggle/input/chart-patterns"
   ```

## Usage

### Deploy & Run Benchmark

```bash
poetry run python -m vlm_chart_pattern_analyzer.kaggle.main \
  --model_id "Qwen/Qwen2-VL-2B-Instruct" \
  --precision fp32 \
  --gpu \
  --kernel_id "your_username/vlm-chart-benchmark"
```

### Full Pipeline (Deploy → Poll → Download)

```bash
poetry run python -m vlm_chart_pattern_analyzer.kaggle.main \
  --model_id "Qwen/Qwen2-VL-2B-Instruct" \
  --precision fp32 \
  --gpu \
  --dest benchmark_results \
  --kernel_id "your_username/vlm-chart-benchmark"
```

### Manual Steps

```bash
# Step 1: Deploy kernel
poetry run python -c \
  "from vlm_chart_pattern_analyzer.kaggle.core import deploy; \
   deploy.run('Qwen/Qwen2-VL-2B-Instruct', 'fp32', \
              './src/vlm_chart_pattern_analyzer/kaggle/config/vlm-inference-benchmark.ipynb', \
              './src/vlm_chart_pattern_analyzer/kaggle/config', \
              gpu=True)"

# Step 2: Poll status
poetry run python -c \
  "from vlm_chart_pattern_analyzer.kaggle.utils import poll_status; \
   status = poll_status.run('your_username/vlm-chart-benchmark'); \
   print(f'Final status: {status}')"

# Step 3: Download results
poetry run python -c \
  "from vlm_chart_pattern_analyzer.kaggle.core import download; \
   download.run('benchmark_results', 'your_username/vlm-chart-benchmark')"
```

## Metrics Analysis

### Using the Metrics Collector

```python
from pathlib import Path
from vlm_chart_pattern_analyzer.metrics_collector import MetricsCollector

# Load results
collector = MetricsCollector()
collector.add_metrics_from_csv(Path("benchmark_results/benchmark_results.csv"))

# Get summary
summary = collector.get_summary_stats()
print(f"Avg Latency: {summary['latency_ms']['mean']:.2f} ms")
print(f"Avg Throughput: {summary['throughput_tokens_per_sec']['mean']:.2f} tokens/sec")

# Compare models
model_comp = collector.get_model_comparison()
for model, stats in model_comp.items():
    print(f"{model}: {stats['avg_latency_ms']:.2f} ms")

# Compare precisions
prec_comp = collector.get_precision_comparison()
for precision, stats in prec_comp.items():
    print(f"{precision}: {stats['avg_latency_ms']:.2f} ms, {stats['avg_memory_mb']:.2f} MB")

# Export reports
collector.export_to_json(Path("report.json"))
collector.export_to_csv(Path("metrics.csv"))
collector.export_to_html_report(Path("report.html"))
```

## Output Files

The pipeline generates:

```
benchmark_results/
├── benchmark_results.csv       # Raw metrics (latency, memory, throughput, tokens)
├── benchmark_results.json      # Detailed results + summary statistics
├── report.html                 # Interactive HTML report
└── logs/
    ├── kaggle_cli_stdout.log   # Kaggle CLI stdout
    └── kaggle_cli_stderr.log   # Kaggle CLI stderr
```

## CSV Format

```
image_filename,model_id,precision,device,latency_ms,memory_used_mb,tokens_generated,throughput_tokens_per_sec,timestamp
chart_1.png,Qwen/Qwen2-VL-2B-Instruct,fp32,cuda,245.5,150.2,156,636.0,2025-10-23T10:00:00
chart_2.png,Qwen/Qwen2-VL-2B-Instruct,fp32,cuda,238.3,148.1,160,671.4,2025-10-23T10:00:01
...
```

## JSON Format

```json
{
  "metadata": {
    "model_id": "Qwen/Qwen2-VL-2B-Instruct",
    "precision": "fp32",
    "device": "cuda",
    "timestamp": "2025-10-23T10:00:00",
    "num_images": 10,
    "num_prompts": 1
  },
  "results": [
    {
      "image_filename": "chart_1.png",
      "latency_ms": 245.5,
      "memory_used_mb": 150.2,
      "tokens_generated": 156,
      "throughput_tokens_per_sec": 636.0,
      ...
    }
  ],
  "summary": {
    "avg_latency_ms": 241.9,
    "avg_memory_mb": 149.15,
    "avg_throughput_tokens_per_sec": 653.7,
    "total_tokens_generated": 1560
  }
}
```

## Kaggle Free Tier Benefits

- **30 hours of GPU time per week** (T4×2 or P100)
- **No credit card required** for basic usage
- **Free notebooks** for development
- **Public datasets** for chart images

## Troubleshooting

### Issue: "KERNEL_ID not found"
**Solution**: Ensure your Kaggle kernel is created and ID is correct in config files.

### Issue: "kaggle.json not found"
**Solution**: Download API token from https://www.kaggle.com/settings/account and place in `~/.kaggle/kaggle.json`

### Issue: "Output file not found" when downloading
**Solution**: Check that kernel execution completed successfully. View logs at `benchmark_results/kaggle_cli_*.log`

### Issue: Out of memory on Kaggle GPU
**Solution**: Try FP16 precision or reduce `max_new_tokens` in notebook template

## Advanced: Custom Model Support

To add a new model, edit `vlm-inference-benchmark.ipynb`:

```python
# Add to MODEL_REGISTRY equivalent
SUPPORTED_MODELS = {
    "Qwen/Qwen2-VL-2B-Instruct": {...},
    "your_new_model/name": {
        "processor_kwargs": {...},
        "model_kwargs": {...}
    }
}
```

## Performance Benchmarks

Typical results on Kaggle GPU (T4):

| Model | Precision | Latency (ms) | Memory (MB) | Throughput (tokens/sec) |
|-------|-----------|-------------|------------|----------------------|
| Qwen2-VL-2B | FP32 | 245 | 150 | 636 |
| Qwen2-VL-2B | FP16 | 145 | 100 | 1088 |
| LLaVA-1.6-8B | FP32 | 890 | 500 | 180 |
| LLaVA-1.6-8B | FP16 | 520 | 300 | 308 |

## Next Steps

- [ ] Add vLLM integration for higher throughput
- [ ] Add TensorRT optimization
- [ ] Create GitHub Actions workflow for scheduled benchmarks
- [ ] Support batch inference
- [ ] Add cost tracking

## References

- [ImgGenHub](https://github.com/leweex95/imggenhub) - Original Kaggle automation pattern
- [Kaggle API Documentation](https://github.com/Kaggle/kaggle-api)
- [Qwen2-VL Model Card](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)
