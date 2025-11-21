#!/usr/bin/env python3
"""
Deep inspection of vocabulary codebook to see if clusters actually exist.
"""

import joblib
import numpy as np
from pathlib import Path

def inspect_codebook(vocab_path: Path):
    """Check if vocabulary clusters are actually distinct."""
    print(f"\n{'='*80}")
    print(f"CODEBOOK INSPECTION: {vocab_path.name}")
    print(f"{'='*80}")
    
    vocab = joblib.load(vocab_path)
    
    if 'kmeans' in vocab:
        kmeans = vocab['kmeans']
        codebook = kmeans.cluster_centers_
        
        print(f"\nCodebook shape: {codebook.shape}")
        print(f"Number of clusters: {codebook.shape[0]}")
        print(f"Feature dimension: {codebook.shape[1]}")
        
        # Check if clusters are distinct
        print(f"\nCluster distinctiveness:")
        
        # Compute pairwise distances between cluster centers
        from scipy.spatial.distance import pdist, squareform
        distances = squareform(pdist(codebook, metric='euclidean'))
        
        # Get min, max, mean distance between clusters
        # Exclude diagonal (distance to self = 0)
        np.fill_diagonal(distances, np.inf)
        min_distance = distances.min()
        max_distance = distances.max()
        mean_distance = distances[distances != np.inf].mean()
        
        print(f"  Min distance between any 2 clusters: {min_distance:.6f}")
        print(f"  Max distance between any 2 clusters: {max_distance:.6f}")
        print(f"  Mean distance between clusters: {mean_distance:.6f}")
        
        # Find clusters that are TOO close (potential duplicates)
        close_threshold = 0.01
        close_pairs = np.argwhere((distances < close_threshold) & (distances > 0))
        
        if len(close_pairs) > 0:
            print(f"\n⚠️  WARNING: {len(close_pairs)} cluster pairs closer than {close_threshold}!")
            print(f"  These clusters may be duplicates:")
            for i, j in close_pairs[:10]:  # Show first 10
                print(f"    Cluster {i} and {j}: distance = {distances[i, j]:.6f}")
        else:
            print(f"\n✅ All clusters are well-separated (>{close_threshold} apart)")
        
        # Check cluster sizes (if labels were stored during training)
        if 'token_frequencies' in vocab:
            freqs = vocab['token_frequencies']
            print(f"\nToken frequencies (from training):")
            print(f"  Non-zero tokens: {np.count_nonzero(freqs)}/{len(freqs)}")
            print(f"  Top 10 tokens:")
            top_indices = np.argsort(freqs)[::-1][:10]
            for idx in top_indices:
                print(f"    Token {idx}: {freqs[idx]} samples")
        
        # Check PCA if present
        if 'pca' in vocab:
            pca = vocab['pca']
            print(f"\nPCA transformation:")
            print(f"  Input dim: {vocab.get('feature_dim', 'unknown')}")
            print(f"  Output dim: {vocab.get('reduced_dim', pca.n_components_)}")
            print(f"  Variance explained (first 5): {pca.explained_variance_ratio_[:5]}")
            print(f"  Total variance explained: {pca.explained_variance_ratio_.sum():.4f}")

if __name__ == '__main__':
    vocab_files = [
        Path("input_audio/General_idea_harmonic_vocab.joblib"),
        Path("input_audio/General_idea_percussive_vocab.joblib"),
    ]
    
    for vocab_file in vocab_files:
        if vocab_file.exists():
            inspect_codebook(vocab_file)
        else:
            print(f"\n❌ File not found: {vocab_file}")
