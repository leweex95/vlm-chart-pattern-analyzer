#!/usr/bin/env python3
"""
Parse benchmark results and update BENCHMARK_EXPERIMENTS.md table.

Reads CSV results from completed Kaggle runs and appends rows to the experiments table.
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vlm_chart_pattern_analyzer.metrics_collector import MetricsCollector


def get_git_commit_hash() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return "unknown"


def calculate_percentile(values: list, percentile: float) -> float:
    """Calculate percentile from list of values."""
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = int(len(sorted_values) * percentile / 100)
    return sorted_values[min(index, len(sorted_values) - 1)]


def analyze_model_results(collector: MetricsCollector, model_name: str, precision: str) -> Dict[str, Any]:
    """Analyze results for a specific model and precision."""
    # Filter metrics for this model/precision
    filtered = [
        m for m in collector.metrics
        if model_name in m.model_id and m.precision == precision
    ]
    
    if not filtered:
        return {}
    
    latencies = [m.latency_ms for m in filtered]
    memory_usage = [m.memory_used_mb for m in filtered]
    throughputs = [m.throughput_tokens_per_sec for m in filtered]
    
    total_time_minutes = sum(latencies) / 1000 / 60
    device = filtered[0].device if filtered else "unknown"
    
    return {
        "num_images": len(filtered),
        "avg_latency_ms": sum(latencies) / len(latencies),
        "p50_latency_ms": calculate_percentile(latencies, 50),
        "p95_latency_ms": calculate_percentile(latencies, 95),
        "avg_memory_mb": sum(memory_usage) / len(memory_usage),
        "peak_memory_mb": max(memory_usage),
        "avg_throughput": sum(throughputs) / len(throughputs),
        "total_gpu_time_min": total_time_minutes,
        "device": device,
    }


def format_table_row(
    datetime_str: str,
    model: str,
    precision: str,
    stats: Dict[str, Any],
    notes: str,
    commit_hash: str,
) -> str:
    """Format a row for the experiments table."""
    if not stats:
        return f"| {datetime_str} | {model} | {precision} | - | - | - | - | - | - | - | - | - | {notes} | {commit_hash} |"
    
    return (
        f"| {datetime_str} "
        f"| {model} "
        f"| {precision} "
        f"| {stats['num_images']} "
        f"| {stats['avg_latency_ms']:.1f} "
        f"| {stats['p50_latency_ms']:.1f} "
        f"| {stats['p95_latency_ms']:.1f} "
        f"| {stats['avg_memory_mb']:.1f} "
        f"| {stats['peak_memory_mb']:.1f} "
        f"| {stats['avg_throughput']:.1f} "
        f"| {stats['total_gpu_time_min']:.2f} "
        f"| {stats['device']} "
        f"| {notes} "
        f"| {commit_hash} |"
    )


def update_experiments_log(
    results_dir: Path,
    experiment_name: str,
    notes: str = "Baseline unoptimized",
):
    """Update BENCHMARK_EXPERIMENTS.md with new results."""
    
    # Load all results
    collector = MetricsCollector()
    
    model_keys = ["qwen2-vl-2b", "llava-1.6-8b", "phi-3-vision"]
    display_names = ["Qwen2-VL-2B", "LLaVA-1.6-8B", "Phi-3-Vision"]
    
    for model_key in model_keys:
        csv_file = results_dir / model_key / "benchmark_results.csv"
        if csv_file.exists():
            collector.add_metrics_from_csv(csv_file)
            print(f"✓ Loaded results for {model_key}")
        else:
            print(f"✗ No results found for {model_key}")
    
    if not collector.metrics:
        print("No metrics collected. Cannot update experiments log.")
        return
    
    # Get commit hash and timestamp
    commit_hash = get_git_commit_hash()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    # Analyze results for each model (FP32 baseline)
    precision = "FP32"
    new_rows = []
    
    for model_key, display_name in zip(model_keys, display_names):
        stats = analyze_model_results(collector, model_key, "fp32")
        if stats:
            row = format_table_row(
                timestamp,
                display_name,
                precision,
                stats,
                notes,
                commit_hash,
            )
            new_rows.append(row)
            print(f"\n{display_name} Results:")
            print(f"  Images: {stats['num_images']}")
            print(f"  Avg Latency: {stats['avg_latency_ms']:.1f} ms")
            print(f"  P95 Latency: {stats['p95_latency_ms']:.1f} ms")
            print(f"  Avg Memory: {stats['avg_memory_mb']:.1f} MB")
            print(f"  Peak Memory: {stats['peak_memory_mb']:.1f} MB")
            print(f"  Throughput: {stats['avg_throughput']:.1f} tok/s")
            print(f"  GPU Time: {stats['total_gpu_time_min']:.2f} min")
            print(f"  Device: {stats['device']}")
    
    # Read existing experiments log
    experiments_file = Path("BENCHMARK_EXPERIMENTS.md")
    with open(experiments_file, "r") as f:
        content = f.read()
    
    # Find the table and insert new rows
    lines = content.split("\n")
    table_found = False
    insert_index = -1
    
    for i, line in enumerate(lines):
        if line.startswith("| Datetime |"):
            table_found = True
        elif table_found and line.startswith("| _Awaiting"):
            insert_index = i
            break
    
    if insert_index > 0:
        # Replace the "Awaiting" line with actual results
        lines[insert_index:insert_index+1] = new_rows
        
        # Write back
        with open(experiments_file, "w") as f:
            f.write("\n".join(lines))
        
        print(f"\n✓ Updated {experiments_file} with {len(new_rows)} result rows")
    else:
        print(f"✗ Could not find insertion point in {experiments_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Run baseline benchmark and update experiments log"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("data/results/kaggle_benchmark"),
        help="Directory containing benchmark results",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="baseline-unoptimized",
        help="Name of the experiment",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default="Baseline unoptimized",
        help="Notes for this experiment",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't actually update the file, just show what would be added",
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("BASELINE BENCHMARK RESULTS ANALYZER")
    print("=" * 70)
    print(f"Results directory: {args.results_dir}")
    print(f"Experiment: {args.experiment_name}")
    print(f"Notes: {args.notes}")
    print(f"Dry run: {args.dry_run}")
    print("=" * 70)
    
    if not args.results_dir.exists():
        print(f"\n✗ Results directory not found: {args.results_dir}")
        print("Run the benchmark first using: python scripts/benchmark.py")
        return 1
    
    update_experiments_log(
        args.results_dir,
        args.experiment_name,
        args.notes,
    )
    
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
