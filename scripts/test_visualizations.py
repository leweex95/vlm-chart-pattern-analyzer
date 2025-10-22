#!/usr/bin/env python
"""Quick test to verify visualization module works correctly."""
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from vlm_chart_pattern_analyzer.visualization import (
    plot_comprehensive_dashboard,
    create_summary_statistics
)


def test_visualizations():
    """Test visualization module with sample data."""
    
    print("\n🧪 Testing visualization module...\n")
    
    # Create sample benchmark data
    sample_data = {
        'timestamp': [datetime.now().isoformat()] * 12,
        'image': ['test_' + str(i % 3) + '.jpg' for i in range(12)],
        'model': ['Qwen2-VL-2B'] * 4 + ['LLaVA-1.6-8B'] * 4 + ['Phi-3-Vision'] * 4,
        'precision': ['fp32', 'fp32', 'fp16', 'int8'] * 3,
        'latency_ms': [
            245.5, 248.3, 165.2, 198.7,  # Qwen2-VL-2B
            892.1, 905.3, 458.2, 602.5,  # LLaVA-1.6-8B
            412.3, 418.5, 285.1, 355.2,  # Phi-3-Vision
        ],
        'memory_mb': [
            2145, 2148, 1872, 1950,  # Qwen2-VL-2B
            7823, 7845, 5423, 6120,  # LLaVA-1.6-8B
            3542, 3565, 2845, 3120,  # Phi-3-Vision
        ],
        'tokens': [127, 124, 128, 126, 256, 258, 255, 257, 198, 195, 199, 197]
    }
    
    df = pd.DataFrame(sample_data)
    
    print(f"✓ Created sample dataset with {len(df)} records")
    print(f"  Models: {', '.join(df['model'].unique())}")
    print(f"  Precisions: {', '.join(df['precision'].unique())}")
    print(f"  Avg Latency: {df['latency_ms'].mean():.2f}ms")
    print(f"  Avg Memory: {df['memory_mb'].mean():.2f}MB")
    
    # Test visualization generation
    output_dir = Path('data/results/test_visualizations')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📊 Generating test visualizations...\n")
    plot_comprehensive_dashboard(df, str(output_dir))
    
    # Test summary generation
    print(f"\n📋 Generating summary statistics...\n")
    create_summary_statistics(df, str(output_dir / 'summary.txt'))
    
    # Verify outputs
    html_files = list(output_dir.glob('*.html'))
    print(f"\n✅ Test complete!")
    print(f"   Generated {len(html_files)} HTML visualization files:")
    for f in sorted(html_files):
        print(f"     - {f.name}")
    
    summary_file = output_dir / 'summary.txt'
    if summary_file.exists():
        print(f"   Generated summary: {summary_file.name}")


if __name__ == '__main__':
    try:
        test_visualizations()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
