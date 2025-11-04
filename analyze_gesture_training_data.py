#!/usr/bin/env python3
"""
Comprehensive Gesture Token Analysis for Nineteen Training Data
==============================================================

This script recreates the gesture token generation process from the Nineteen training data
to determine if the issue with reduced rhythmic variety stems from:
1. Training data limitations (limited token diversity in source material)
2. Performance-time processing (smoothing, coordination issues)

Analysis approach:
1. Load 768D features from training frames
2. Train quantizer exactly as done during live performance
3. Generate tokens and analyze diversity, patterns, and rhythmic context
4. Compare with quantizer statistics to identify discrepancies
"""

import json
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from typing import Dict, List

# Import the same quantizer used in live performance
import sys
sys.path.append('.')
from listener.symbolic_quantizer import SymbolicQuantizer

def load_training_features(model_file: str) -> tuple:
    """Load 768D features from training model file"""
    print(f"Loading features from {model_file}...")
    
    with open(model_file, 'r') as f:
        model_data = json.load(f)
    
    audio_frames = model_data['audio_frames']
    features = []
    timestamps = []
    
    print(f"   Found {len(audio_frames)} frames")
    
    for frame_id in sorted(audio_frames.keys(), key=int):
        frame = audio_frames[frame_id]
        if 'features' in frame:
            features.append(frame['features'])
            timestamps.append(frame.get('timestamp', 0.0))
    
    return np.array(features), timestamps

def analyze_gesture_tokens(features: np.ndarray, timestamps: List[float]) -> Dict:
    """Recreate gesture token generation and analyze diversity"""
    print(f"\n📊 Analyzing gesture token generation...")
    print(f"   Feature shape: {features.shape}")
    
    # Train quantizer exactly as in live performance
    quantizer = SymbolicQuantizer(vocabulary_size=64, use_l2_norm=True)
    quantizer.fit(features)
    
    # Generate tokens
    tokens = quantizer.transform(features)
    
    # Basic statistics
    unique_tokens = len(set(tokens))
    token_counts = Counter(tokens)
    
    print(f"\n🎵 Token Diversity Analysis:")
    print(f"   Total frames: {len(tokens)}")
    print(f"   Unique tokens: {unique_tokens}/64 ({unique_tokens/64*100:.1f}%)")
    print(f"   Most common tokens: {token_counts.most_common(10)}")
    print(f"   Least common tokens: {token_counts.most_common()[-10:]}")
    
    # Entropy calculation
    total = len(tokens)
    entropy = -sum((count/total) * np.log2(count/total) for count in token_counts.values())
    max_entropy = np.log2(unique_tokens)
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
    
    print(f"   Entropy: {entropy:.2f} bits")
    print(f"   Max entropy: {max_entropy:.2f} bits")
    print(f"   Normalized entropy: {normalized_entropy:.3f} (1.0 = perfectly diverse)")
    
    # Temporal analysis - look for repetitive patterns
    consecutive_repeats = 0
    for i in range(1, len(tokens)):
        if tokens[i] == tokens[i-1]:
            consecutive_repeats += 1
    
    repeat_ratio = consecutive_repeats / len(tokens)
    print(f"   Consecutive repeats: {consecutive_repeats}/{len(tokens)} ({repeat_ratio*100:.1f}%)")
    
    # Check for rhythmic context patterns
    if len(timestamps) > 1:
        intervals = np.diff(timestamps)
        avg_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        print(f"   Avg time interval: {avg_interval:.3f}s ± {std_interval:.3f}s")
    
    return {
        'tokens': tokens,
        'unique_count': unique_tokens,
        'total_count': len(tokens),
        'diversity_ratio': unique_tokens / 64,
        'entropy': entropy,
        'normalized_entropy': normalized_entropy,
        'repeat_ratio': repeat_ratio,
        'token_counts': token_counts,
        'quantizer_stats': quantizer.get_codebook_statistics()
    }

def analyze_temporal_patterns(tokens: np.ndarray, timestamps: List[float]) -> Dict:
    """Analyze temporal patterns in token generation"""
    print(f"\n⏱️  Temporal Pattern Analysis:")
    
    # Look for rhythmic periodicity
    if len(timestamps) < 2:
        print("   Insufficient timestamp data")
        return {}
    
    intervals = np.diff(timestamps)
    
    # Sliding window analysis - check for repetitive token patterns
    window_sizes = [4, 8, 16]  # Different pattern lengths
    pattern_analysis = {}
    
    for window_size in window_sizes:
        if len(tokens) < window_size * 2:
            continue
            
        patterns = {}
        for i in range(len(tokens) - window_size + 1):
            pattern = tuple(tokens[i:i+window_size])
            patterns[pattern] = patterns.get(pattern, 0) + 1
        
        # Find most common patterns
        common_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Calculate pattern diversity
        unique_patterns = len(patterns)
        max_possible = min(64**window_size, len(tokens) - window_size + 1)
        pattern_diversity = unique_patterns / max_possible
        
        pattern_analysis[window_size] = {
            'unique_patterns': unique_patterns,
            'max_possible': max_possible,
            'diversity': pattern_diversity,
            'most_common': common_patterns
        }
        
        print(f"   Window size {window_size}:")
        print(f"     Unique patterns: {unique_patterns}/{max_possible} ({pattern_diversity*100:.1f}%)")
        if common_patterns:
            print(f"     Most common pattern: {common_patterns[0][0]} (appears {common_patterns[0][1]} times)")
    
    return pattern_analysis

def compare_with_performance_expectations() -> None:
    """Compare training data analysis with expected performance behavior"""
    print(f"\n🎯 Performance Expectations vs Training Reality:")
    
    print(f"   Expected during performance:")
    print(f"   • Gesture smoothing window: 1.5s")
    print(f"   • Minimum unique tokens: 2 per window")
    print(f"   • Rapid rhythmic changes should generate different tokens")
    print(f"   • Complex music (Nineteen/Daybreak) should have high diversity")
    
    print(f"\n   If training data shows:")
    print(f"   • Low diversity → Issue is in source material, need retraining")
    print(f"   • High diversity → Issue is in performance-time processing")
    print(f"   • Repetitive patterns → May explain reduced rhythmic variety")

def find_latest_training_file() -> str:
    """Find the most recent training model file automatically"""
    import os
    from pathlib import Path
    
    json_dir = Path('JSON')
    if not json_dir.exists():
        raise FileNotFoundError("JSON directory not found")
    
    # Look for training model files
    model_files = list(json_dir.glob('*training_model.json'))
    
    if not model_files:
        raise FileNotFoundError("No training model files found in JSON directory")
    
    # Sort by modification time and get the newest
    latest_file = max(model_files, key=lambda f: f.stat().st_mtime)
    
    # Extract the base name for display
    base_name = latest_file.stem.replace('_training_model', '')
    
    return str(latest_file), base_name

def main():
    """Main analysis function"""
    print("🎼 Comprehensive Gesture Token Analysis for Latest Training Data")
    print("=" * 70)
    
    # Find the most recent training data automatically
    try:
        model_file, dataset_name = find_latest_training_file()
        print(f"📁 Auto-detected latest training file: {model_file}")
        print(f"📊 Dataset: {dataset_name}")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return
    
    try:
        features, timestamps = load_training_features(model_file)
        
        # Analyze gesture token generation
        token_analysis = analyze_gesture_tokens(features, timestamps)
        
        # Analyze temporal patterns
        temporal_analysis = analyze_temporal_patterns(token_analysis['tokens'], timestamps)
        
        # Compare with expectations
        compare_with_performance_expectations()
        
        # Summary and diagnosis
        print(f"\n🎯 DIAGNOSIS:")
        print(f"   Dataset: {dataset_name}")
        print(f"   Diversity Ratio: {token_analysis['diversity_ratio']:.3f}")
        print(f"   Normalized Entropy: {token_analysis['normalized_entropy']:.3f}")
        print(f"   Consecutive Repeats: {token_analysis['repeat_ratio']*100:.1f}%")
        
        if token_analysis['diversity_ratio'] < 0.5:
            print(f"   ❌ LOW DIVERSITY: Training data uses <50% of available tokens")
            print(f"      → Issue likely in source material or feature extraction")
            print(f"      → Consider retraining with different parameters")
        elif token_analysis['diversity_ratio'] > 0.8:
            print(f"   ✅ HIGH DIVERSITY: Training data uses >80% of available tokens")
            print(f"      → Issue likely in performance-time processing")
            print(f"      → Check gesture smoothing, behavioral modes, voice coordination")
        else:
            print(f"   ⚠️  MODERATE DIVERSITY: Training data uses ~50-80% of tokens")
            print(f"      → May be normal, check temporal patterns for repetitiveness")
        
        if token_analysis['repeat_ratio'] > 0.3:
            print(f"   ❌ HIGH REPETITION: >30% consecutive identical tokens")
            print(f"      → May explain reduced rhythmic variety")
            print(f"      → Check temporal smoothing during training")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        print("   Available files:")
        import os
        json_files = [f for f in os.listdir('JSON') if f.endswith('_training_model.json')]
        for f in sorted(json_files)[-5:]:  # Show last 5 files
            print(f"   - {f}")

if __name__ == "__main__":
    main()