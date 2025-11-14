#!/usr/bin/env python3
"""
Analyze WHY Wav2Vec features collapsed during training.

Investigates:
1. Feature extraction configuration
2. Feature normalization issues
3. Quantization clustering
4. Input audio characteristics
"""

import json
import numpy as np
import matplotlib.pyplot as plt

def analyze_feature_collapse(model_path):
    print("FEATURE COLLAPSE INVESTIGATION")
    print("=" * 80)
    
    with open(model_path, 'r') as f:
        data = json.load(f)
    
    frames = data['audio_frames']
    
    # Extract all features
    print("\n1. EXTRACTING FEATURES...")
    all_features = []
    for i in range(min(1000, len(frames))):  # Sample 1000 frames
        frame_key = str(i)
        if frame_key in frames and 'features' in frames[frame_key]:
            all_features.append(frames[frame_key]['features'])
    
    feature_array = np.array(all_features)
    print(f"   Sampled {len(all_features)} frames")
    print(f"   Feature shape: {feature_array.shape}")
    
    # Analyze per-dimension statistics
    print("\n2. PER-DIMENSION ANALYSIS")
    means = np.mean(feature_array, axis=0)
    stds = np.std(feature_array, axis=0)
    variances = np.var(feature_array, axis=0)
    
    print(f"   Mean range: [{np.min(means):.3f}, {np.max(means):.3f}]")
    print(f"   Std range: [{np.min(stds):.3f}, {np.max(stds):.3f}]")
    print(f"   Variance range: [{np.min(variances):.4f}, {np.max(variances):.4f}]")
    
    # Find active vs dead dimensions
    active_dims = np.where(variances > 0.01)[0]
    dead_dims = np.where(variances <= 0.01)[0]
    
    print(f"\n   Active dimensions (var > 0.01): {len(active_dims)}/768")
    print(f"   Dead dimensions (var ≤ 0.01): {len(dead_dims)}/768")
    
    if len(active_dims) > 0:
        print(f"\n   Active dimension indices: {active_dims[:20]}...")
        print(f"   Their variances: {variances[active_dims[:20]]}")
    
    # Check if features are all near zero
    print("\n3. ZERO-CENTERING CHECK")
    near_zero = np.sum(np.abs(means) < 0.1)
    print(f"   Dimensions with mean ≈ 0 (|mean| < 0.1): {near_zero}/768")
    
    # Check for constant features
    print("\n4. CONSTANT FEATURE CHECK")
    constant_dims = np.where(stds < 0.001)[0]
    print(f"   Nearly constant dimensions (std < 0.001): {len(constant_dims)}/768")
    
    # Analyze feature value distribution
    print("\n5. FEATURE VALUE DISTRIBUTION")
    all_values = feature_array.flatten()
    print(f"   All values range: [{np.min(all_values):.3f}, {np.max(all_values):.3f}]")
    print(f"   All values mean: {np.mean(all_values):.3f}")
    print(f"   All values std: {np.std(all_values):.3f}")
    
    # Check for quantization artifacts
    unique_values = len(np.unique(all_values))
    print(f"   Unique values: {unique_values} (out of {len(all_values)} total)")
    
    if unique_values < 100:
        print("   ⚠️  Very few unique values - might be over-quantized!")
    
    # Histogram of variances
    print("\n6. VARIANCE DISTRIBUTION")
    var_bins = [0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    var_hist, _ = np.histogram(variances, bins=var_bins)
    for i in range(len(var_bins)-1):
        print(f"   Variance [{var_bins[i]}, {var_bins[i+1]}): {var_hist[i]} dims")
    
    # Save statistics
    print("\n7. SAVING DETAILED STATISTICS...")
    stats = {
        'sample_size': len(all_features),
        'active_dimensions': int(len(active_dims)),
        'dead_dimensions': int(len(dead_dims)),
        'constant_dimensions': int(len(constant_dims)),
        'near_zero_means': int(near_zero),
        'unique_values': int(unique_values),
        'mean_range': [float(np.min(means)), float(np.max(means))],
        'std_range': [float(np.min(stds)), float(np.max(stds))],
        'variance_range': [float(np.min(variances)), float(np.max(variances))],
        'active_dim_indices': active_dims.tolist() if len(active_dims) < 100 else active_dims[:100].tolist(),
    }
    
    with open('feature_collapse_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("   Saved to: feature_collapse_stats.json")
    
    print("\n" + "=" * 80)
    print("DIAGNOSIS:")
    print("=" * 80)
    
    if len(dead_dims) > 700:
        print("\n❌ SEVERE FEATURE COLLAPSE:")
        print(f"   {len(dead_dims)}/768 dimensions are effectively dead (no variation)")
        print("\n   Possible causes:")
        print("   1. Wav2Vec model not properly loaded/initialized")
        print("   2. Input audio preprocessing removed all variation")
        print("   3. Features normalized incorrectly (all squashed to same range)")
        print("   4. Quantization step destroyed information")
        print("   5. Training on silent/uniform audio sections")
    
    if unique_values < 1000:
        print(f"\n⚠️  LOW FEATURE DIVERSITY:")
        print(f"   Only {unique_values} unique feature values")
        print("   Features may have been over-quantized or rounded")
    
    if near_zero > 600:
        print(f"\n⚠️  ZERO-CENTERED COLLAPSE:")
        print(f"   {near_zero}/768 dimensions centered near zero")
        print("   Suggests features were normalized then variance was lost")
    
    print("\n")

if __name__ == "__main__":
    analyze_feature_collapse("JSON/Itzama_061125_2255_training_model.json")
