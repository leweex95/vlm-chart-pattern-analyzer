#!/usr/bin/env python
"""Test similarity analysis with sample data."""
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from vlm_chart_pattern_analyzer.similarity import (
    analyze_benchmark_similarities,
    create_agreement_summary,
    save_similarity_analysis
)
from vlm_chart_pattern_analyzer.visualization import (
    plot_similarity_heatmap,
    plot_model_agreement_bars,
    plot_similarity_distribution
)


def test_similarity_analysis():
    """Test similarity analysis with sample data."""
    
    print("\n🧪 Testing similarity analysis module...\n")
    
    # Create sample benchmark data with responses
    sample_data = {
        'timestamp': [datetime.now().isoformat()] * 12,
        'image': ['chart_001.png'] * 3 + ['chart_002.png'] * 3 + ['chart_003.png'] * 3 + ['chart_004.png'] * 3,
        'model': ['qwen2-vl-2b', 'llava-1.6-8b', 'phi-3-vision'] * 4,
        'precision': ['fp32'] * 12,
        'latency_ms': [250, 900, 400] * 4,
        'memory_mb': [2100, 7800, 3500] * 4,
        'tokens': [128, 256, 200] * 4,
        'response': [
            # Chart 1
            "The chart shows a bullish trend with support at lower levels and resistance at upper levels.",
            "Strong uptrend detected. Price above MA200. Bullish momentum continues.",
            "Consolidation pattern forming. Support holding at key level.",
            # Chart 2
            "Downtrend is weakening. Potential reversal setup forming.",
            "Strong bearish pressure. Volume confirms weakness.",
            "Support broken. Downtrend continuation expected.",
            # Chart 3
            "Triangle breakout imminent. Bullish bias above resistance.",
            "Consolidation area. Waiting for breakout confirmation.",
            "Resistance at key level. Watch for breakout signal.",
            # Chart 4
            "Pullback in uptrend provides entry opportunity.",
            "Strong uptrend continues. Buy on dips.",
            "Bullish structure maintained. Support holding."
        ]
    }
    
    df = pd.DataFrame(sample_data)
    
    print(f"✓ Created sample dataset with {len(df)} records")
    print(f"  Models: {', '.join(df['model'].unique())}")
    print(f"  Images: {df['image'].nunique()}")
    
    # Analyze similarities
    print(f"\n🔍 Analyzing pattern similarities...")
    try:
        analysis = analyze_benchmark_similarities(df)
        print(f"✓ Analyzed {analysis['image_statistics']['total_images_analyzed']} images")
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Generate summary
    print(f"\n📋 Generating summary...")
    summary = create_agreement_summary(analysis)
    print(summary)
    
    # Save analysis
    output_dir = Path('data/results/test_similarity')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    analysis_file = output_dir / 'similarity_analysis.json'
    save_similarity_analysis(analysis, str(analysis_file))
    
    # Generate visualizations
    print(f"\n📊 Generating similarity visualizations...")
    
    try:
        plot_similarity_heatmap(
            analysis['image_similarities'],
            str(output_dir / 'similarity_heatmap.html')
        )
        
        plot_model_agreement_bars(
            analysis['model_agreement'],
            str(output_dir / 'model_agreement.html')
        )
        
        plot_similarity_distribution(
            analysis['image_similarities'],
            str(output_dir / 'similarity_distribution.html')
        )
        
        print(f"\n✅ Test complete!")
        print(f"   Generated files in {output_dir}/:")
        for f in sorted(output_dir.glob('*')):
            print(f"     - {f.name}")
        
    except Exception as e:
        print(f"❌ Visualization failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    try:
        test_similarity_analysis()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
