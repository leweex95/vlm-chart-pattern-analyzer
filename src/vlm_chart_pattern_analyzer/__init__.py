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
    plot_similarity_heatmap,
    plot_model_agreement_bars,
    plot_similarity_distribution,
)
from .similarity import (
    compute_similarity_matrix,
    compute_pairwise_similarity,
    analyze_benchmark_similarities,
    load_benchmark_with_responses,
    create_agreement_summary,
    save_similarity_analysis,
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
    "plot_similarity_heatmap",
    "plot_model_agreement_bars",
    "plot_similarity_distribution",
    "compute_similarity_matrix",
    "compute_pairwise_similarity",
    "analyze_benchmark_similarities",
    "load_benchmark_with_responses",
    "create_agreement_summary",
    "save_similarity_analysis",
]
