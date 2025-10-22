#!/usr/bin/env python
"""CLI script for analyzing pattern similarity between VLM model outputs."""
import argparse
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from vlm_chart_pattern_analyzer.similarity import (
    analyze_benchmark_similarities,
    load_benchmark_with_responses,
    create_agreement_summary,
    save_similarity_analysis
)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze pattern similarity across VLM model outputs'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='data/results/benchmark.csv',
        help='Path to benchmark CSV file with model responses (default: data/results/benchmark.csv)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/results/similarity_analysis.json',
        help='Output path for similarity analysis JSON (default: data/results/similarity_analysis.json)'
    )
    parser.add_argument(
        '--summary',
        type=str,
        default='data/results/similarity_summary.txt',
        help='Output path for summary report (default: data/results/similarity_summary.txt)'
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Error: Benchmark results file not found: {args.input}")
        print(f"   Run 'poetry run python scripts/benchmark.py' first to generate results")
        print(f"   Note: Benchmark CSV must include 'response' column for similarity analysis")
        sys.exit(1)
    
    print(f"\n📊 Loading benchmark results from {args.input}...")
    df = load_benchmark_with_responses(str(input_path))
    
    if df is None:
        sys.exit(1)
    
    print(f"✓ Loaded {len(df)} benchmark results")
    print(f"  Models: {', '.join(df['model'].unique())}")
    print(f"  Images: {df['image'].nunique()}")
    
    # Analyze similarities
    print(f"\n🔍 Analyzing pattern similarities...")
    try:
        analysis = analyze_benchmark_similarities(df)
        
        if not analysis['image_similarities']:
            print("❌ No similarity data generated. Check benchmark results.")
            sys.exit(1)
        
        print(f"✓ Analyzed {analysis['image_statistics']['total_images_analyzed']} images")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Save JSON analysis
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_similarity_analysis(analysis, str(output_path))
    
    # Generate and save summary
    summary = create_agreement_summary(analysis)
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(summary_path, 'w') as f:
        f.write(summary)
    print(f"✓ Summary saved to {args.summary}")
    
    # Print summary to console
    print("\n" + summary)
    
    print("\n✅ Similarity analysis complete!")
    print(f"   📁 Results saved to {args.output}")


if __name__ == '__main__':
    main()
