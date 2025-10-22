#!/usr/bin/env python
"""CLI script for generating visualizations from benchmark results."""
import argparse
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from vlm_chart_pattern_analyzer.visualization import (
    load_benchmark_results,
    plot_comprehensive_dashboard,
    create_summary_statistics,
    plot_latency_by_model_precision,
    plot_memory_by_model_precision,
    plot_latency_vs_memory,
    plot_tokens_generated,
    plot_model_comparison_heatmap
)


def main():
    parser = argparse.ArgumentParser(
        description='Generate Plotly visualizations from VLM benchmark results'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='data/results/benchmark.csv',
        help='Path to benchmark CSV file (default: data/results/benchmark.csv)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/results/visualizations',
        help='Output directory for visualizations (default: data/results/visualizations)'
    )
    parser.add_argument(
        '--summary',
        type=str,
        default='data/results/summary.txt',
        help='Output path for summary statistics (default: data/results/summary.txt)'
    )
    parser.add_argument(
        '--no-dashboard',
        action='store_true',
        help='Skip comprehensive dashboard generation'
    )
    parser.add_argument(
        '--no-summary',
        action='store_true',
        help='Skip summary statistics generation'
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Error: Benchmark results file not found: {args.input}")
        print(f"   Run 'poetry run python scripts/benchmark.py' first to generate results")
        sys.exit(1)
    
    print(f"\n📊 Loading benchmark results from {args.input}...")
    try:
        df = load_benchmark_results(str(input_path))
        print(f"✓ Loaded {len(df)} benchmark results")
        print(f"  Models: {', '.join(df['model'].unique())}")
        print(f"  Precisions: {', '.join(df['precision'].unique())}")
    except Exception as e:
        print(f"❌ Error loading results: {e}")
        sys.exit(1)
    
    # Generate visualizations
    if not args.no_dashboard:
        try:
            plot_comprehensive_dashboard(df, args.output)
        except Exception as e:
            print(f"❌ Error generating visualizations: {e}")
            sys.exit(1)
    
    # Generate summary statistics
    if not args.no_summary:
        try:
            create_summary_statistics(df, args.summary)
        except Exception as e:
            print(f"❌ Error generating summary: {e}")
            sys.exit(1)
    
    print("\n✅ Visualization complete!")
    print(f"   📁 Open {args.output}/ in your browser to view interactive charts")


if __name__ == '__main__':
    main()
