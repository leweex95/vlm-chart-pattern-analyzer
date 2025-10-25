"""Pattern similarity analysis for comparing VLM model outputs."""
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import json


def extract_text_features(text: str) -> np.ndarray:
    """Extract simple text features from model output."""
    features = []
    
    # Character-level features
    features.append(len(text))  # Length
    features.append(text.count(' '))  # Word count (approx)
    features.append(text.count('.'))  # Sentence count (approx)
    
    # Token/word patterns
    words = text.lower().split()
    features.append(len(set(words)))  # Vocabulary size
    
    # Common pattern indicators
    pattern_keywords = [
        'uptrend', 'downtrend', 'consolidation', 'support', 'resistance',
        'breakout', 'reversal', 'bullish', 'bearish', 'momentum',
        'volume', 'pressure', 'strength', 'weakness', 'pattern'
    ]
    keyword_count = sum(1 for word in words if word in pattern_keywords)
    features.append(keyword_count)
    
    # Sentiment/direction indicators
    bullish_words = ['up', 'bullish', 'long', 'buy', 'strength', 'support', 'above']
    bearish_words = ['down', 'bearish', 'short', 'sell', 'weakness', 'resistance', 'below']
    bullish_count = sum(1 for word in words if word in bullish_words)
    bearish_count = sum(1 for word in words if word in bearish_words)
    features.append(bullish_count - bearish_count)  # Sentiment bias
    
    return np.array(features, dtype=np.float32)


def normalize_features(features: List[np.ndarray]) -> np.ndarray:
    """Normalize feature vectors for similarity computation."""
    stack = np.vstack(features)
    scaler = StandardScaler()
    normalized = scaler.fit_transform(stack)
    return normalized


def compute_similarity_matrix(model_outputs: Dict[str, str]) -> Tuple[np.ndarray, List[str]]:
    """
    Compute cosine similarity matrix between model outputs.
    
    Args:
        model_outputs: Dict mapping model names to their outputs
        
    Returns:
        Tuple of (similarity_matrix, model_names)
    """
    if not model_outputs:
        raise ValueError("model_outputs cannot be empty")
    
    model_names = list(model_outputs.keys())
    
    # Extract and normalize features
    features = [extract_text_features(model_outputs[name]) for name in model_names]
    normalized_features = normalize_features(features)
    
    # Compute cosine similarity
    similarity_matrix = cosine_similarity(normalized_features)
    
    return similarity_matrix, model_names


def compute_pairwise_similarity(responses1: str, responses2: str) -> float:
    """Compute similarity between two model responses."""
    feat1 = extract_text_features(responses1)
    feat2 = extract_text_features(responses2)
    
    # Normalize
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(np.vstack([feat1, feat2]))
    
    # Compute similarity
    similarity = cosine_similarity([features_normalized[0]], [features_normalized[1]])[0][0]
    return float(similarity)


def analyze_benchmark_similarities(results_df: pd.DataFrame) -> Dict:
    """
    Analyze similarities across benchmark results grouped by image.
    
    Args:
        results_df: DataFrame with columns: image, model, response, latency_ms, memory_mb
        
    Returns:
        Dict with similarity analysis results
    """
    analysis = {
        'image_similarities': {},
        'model_agreement': {},
        'image_statistics': {}
    }
    
    # Group by image
    for image_name in results_df['image'].unique():
        image_df = results_df[results_df['image'] == image_name]
        
        if len(image_df) < 2:
            continue
        
        # Get model outputs for this image
        model_outputs = dict(zip(image_df['model'], image_df['response']))
        
        # Compute similarity matrix
        try:
            sim_matrix, models = compute_similarity_matrix(model_outputs)
            
            # Store results
            analysis['image_similarities'][image_name] = {
                'models': models,
                'similarity_matrix': sim_matrix.tolist(),
                'avg_similarity': float(np.mean(sim_matrix[np.triu_indices_from(sim_matrix, k=1)])),
                'min_similarity': float(np.min(sim_matrix[np.triu_indices_from(sim_matrix, k=1)])),
                'max_similarity': float(np.max(sim_matrix[np.triu_indices_from(sim_matrix, k=1)]))
            }
        except Exception as e:
            print(f"  Error analyzing {image_name}: {e}")
            continue
    
    # Compute model-pair agreement across all images
    if analysis['image_similarities']:
        all_similarities = []
        model_pairs = {}
        
        for image_data in analysis['image_similarities'].values():
            models = image_data['models']
            sim_matrix = np.array(image_data['similarity_matrix'])
            
            # Extract pairwise similarities
            for i, m1 in enumerate(models):
                for j, m2 in enumerate(models):
                    if i < j:
                        pair_key = f"{m1} vs {m2}"
                        sim_value = sim_matrix[i][j]
                        all_similarities.append(sim_value)
                        
                        if pair_key not in model_pairs:
                            model_pairs[pair_key] = []
                        model_pairs[pair_key].append(sim_value)
        
        # Aggregate model agreement
        for pair_key, similarities in model_pairs.items():
            analysis['model_agreement'][pair_key] = {
                'mean_similarity': float(np.mean(similarities)),
                'std_similarity': float(np.std(similarities)),
                'min_similarity': float(np.min(similarities)),
                'max_similarity': float(np.max(similarities)),
                'num_comparisons': len(similarities)
            }
        
        # Overall statistics
        analysis['image_statistics'] = {
            'total_images_analyzed': len(analysis['image_similarities']),
            'avg_across_all': float(np.mean(all_similarities)),
            'std_across_all': float(np.std(all_similarities)),
            'min_across_all': float(np.min(all_similarities)),
            'max_across_all': float(np.max(all_similarities))
        }
    
    return analysis


def load_benchmark_with_responses(csv_path: str) -> Optional[pd.DataFrame]:
    """Load benchmark results that include model responses."""
    try:
        df = pd.read_csv(csv_path)
        
        # Check for required columns
        required = ['image', 'model', 'response']
        missing = [col for col in required if col not in df.columns]
        
        if missing:
            print(f"❌ Benchmark CSV missing columns: {', '.join(missing)}")
            print(f"   Available columns: {', '.join(df.columns)}")
            return None
        
        return df
    except Exception as e:
        print(f"❌ Error loading benchmark file: {e}")
        return None


def create_agreement_summary(analysis: Dict) -> str:
    """Create human-readable agreement summary."""
    summary = []
    summary.append("=" * 80)
    summary.append("MODEL AGREEMENT ANALYSIS")
    summary.append("=" * 80)
    summary.append("")
    
    if not analysis['image_statistics']:
        summary.append("No similarity data available.")
        return '\n'.join(summary)
    
    stats = analysis['image_statistics']
    summary.append("OVERALL STATISTICS:")
    summary.append(f"  Images analyzed: {stats['total_images_analyzed']}")
    summary.append(f"  Average similarity: {stats['avg_across_all']:.3f}")
    summary.append(f"  Std deviation: {stats['std_across_all']:.3f}")
    summary.append(f"  Min similarity: {stats['min_across_all']:.3f}")
    summary.append(f"  Max similarity: {stats['max_across_all']:.3f}")
    summary.append("")
    
    summary.append("MODEL PAIR AGREEMENT:")
    summary.append("")
    
    for pair_key in sorted(analysis['model_agreement'].keys()):
        pair_data = analysis['model_agreement'][pair_key]
        summary.append(f"  {pair_key}:")
        summary.append(f"    Mean: {pair_data['mean_similarity']:.3f}")
        summary.append(f"    Std:  {pair_data['std_similarity']:.3f}")
        summary.append(f"    Range: [{pair_data['min_similarity']:.3f}, {pair_data['max_similarity']:.3f}]")
        summary.append(f"    Comparisons: {pair_data['num_comparisons']}")
        summary.append("")
    
    summary.append("=" * 80)
    
    return '\n'.join(summary)


def save_similarity_analysis(analysis: Dict, output_path: str) -> None:
    """Save similarity analysis to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"Similarity analysis saved to {output_path}")
