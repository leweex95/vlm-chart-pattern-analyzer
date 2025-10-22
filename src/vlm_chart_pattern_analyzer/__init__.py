"""VLM Chart Pattern Analyzer - Vision Language Model benchmarking for trading chart pattern recognition."""

__version__ = "0.1.0"

from .models import load_model, get_available_models, MODEL_REGISTRY
from .inference import run_inference
from .visualization import (
    load_benchmark_results,
    plot_latency_by_model_precision,
    plot_memory_by_model_precision,
    plot_latency_vs_memory,
    plot_tokens_generated,
    plot_model_comparison_heatmap,
    plot_comprehensive_dashboard,
    create_summary_statistics,
)

__all__ = [
    "load_model",
    "get_available_models",
    "MODEL_REGISTRY",
    "run_inference",
    "load_benchmark_results",
    "plot_latency_by_model_precision",
    "plot_memory_by_model_precision",
    "plot_latency_vs_memory",
    "plot_tokens_generated",
    "plot_model_comparison_heatmap",
    "plot_comprehensive_dashboard",
    "create_summary_statistics",
]
