#!/usr/bin/env python3
"""
Symbolic Vector Quantization for Musical Features
==================================================

Based on Bujard et al. (2025) - IRCAM research on learning musical relationships

Creates a discrete "musical alphabet" from continuous audio features using K-means VQ.
This symbolic representation:
1. Is more efficient for memory (AudioOracle)
2. Makes patterns more explicit
3. Enables Transformer-based relational learning

Key insight from paper:
- Vocabulary size 16-64: Best for learning relationships
- Vocabulary size 256: More diversity but harder to learn
- Our choice: 64 classes (good balance)

Usage:
    >>> quantizer = SymbolicQuantizer(vocabulary_size=64)
    >>> quantizer.fit(training_features)  # Learn codebook from data
    >>> tokens = quantizer.transform(new_features)  # Convert to symbols
    >>> # tokens = [3, 15, 3, 42, 15, ...]  # Discrete symbolic sequence
"""

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
import joblib
from typing import Dict, Optional


class SymbolicQuantizer:
    """
    Vector Quantizer for creating symbolic musical representations
    
    Converts continuous audio features into discrete tokens from a learned vocabulary.
    Based on K-means clustering (like Bujard et al. 2025).
    """
    
    def __init__(self, vocabulary_size: int = 64,
                 n_init: int = 10,
                 random_state: int = 42,
                 use_l2_norm: bool = True,
                 use_pca: bool = True,
                 pca_components: int = 128,
                 min_cluster_separation: float = 0.1):
        """
        Initialize symbolic quantizer

        Args:
            vocabulary_size: Size of musical alphabet (16, 64, or 256 recommended)
                           Paper found 64 works best for musical relationships
            n_init: Number of K-means initializations
            random_state: Random seed for reproducibility
            use_l2_norm: Use L2 normalization (IRCAM approach, recommended for Wav2Vec)
                       If False, uses StandardScaler (for other feature types)
            use_pca: Apply PCA dimensionality reduction before clustering
                    Highly recommended for 768D Wav2Vec features to improve cluster quality
            pca_components: Number of PCA components (128 recommended for 768D features)
            min_cluster_separation: Minimum distance between cluster centers (0.1 = 10% of feature range)
                                   Higher values enforce more diverse tokens
        """
        self.vocabulary_size = vocabulary_size
        self.n_init = n_init
        self.random_state = random_state
        self.use_l2_norm = use_l2_norm
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.min_cluster_separation = min_cluster_separation

        # PCA for dimensionality reduction (768D → 128D)
        self.pca = PCA(n_components=pca_components, random_state=random_state) if use_pca else None

        # K-means for vector quantization
        self.kmeans = MiniBatchKMeans(
            n_clusters=vocabulary_size,
            n_init=n_init,
            random_state=random_state,
            batch_size=1024,
            max_iter=300
        )

        # Scaler for feature normalization (only if not using L2)
        self.scaler = None if use_l2_norm else StandardScaler()

        # Codebook (learned cluster centers)
        self.codebook = None
        self.is_fitted = False

        # Statistics
        self.token_frequencies = None  # How often each token appears
        self.feature_dim = None
        self.reduced_dim = None  # Dimension after PCA
    
    def fit(self, features: np.ndarray) -> 'SymbolicQuantizer':
        """
        Learn the musical vocabulary (codebook) from features

        Args:
            features: Array of shape (n_samples, n_features)
                     From audio analysis (chroma, spectral, ratio features, etc.)

        Returns:
            self (for chaining)
        """
        if len(features) < self.vocabulary_size:
            raise ValueError(f"Need at least {self.vocabulary_size} samples to learn "
                           f"{self.vocabulary_size} classes")

        print(f"📚 Learning musical vocabulary ({self.vocabulary_size} classes)...")
        print(f"   Training samples: {len(features)}")
        print(f"   Feature dimension: {features.shape[1]}")
        print(f"   Normalization: {'L2 (IRCAM)' if self.use_l2_norm else 'StandardScaler'}")

        self.feature_dim = features.shape[1]

        # Normalize features first
        if self.use_l2_norm:
            # L2 normalization: Each vector has unit length (IRCAM approach)
            features_scaled = normalize(features, norm='l2', axis=1)
        else:
            # StandardScaler: Mean=0, Std=1 (traditional approach)
            features_scaled = self.scaler.fit_transform(features)

        # Apply PCA dimensionality reduction (768D → 128D)
        if self.use_pca and self.pca is not None:
            # Adjust PCA components if feature dim is smaller
            actual_components = min(self.pca_components, features.shape[1], len(features) - 1)
            if actual_components != self.pca_components:
                self.pca = PCA(n_components=actual_components, random_state=self.random_state)

            features_reduced = self.pca.fit_transform(features_scaled)
            self.reduced_dim = features_reduced.shape[1]
            variance_retained = sum(self.pca.explained_variance_ratio_) * 100
            print(f"   PCA: {features.shape[1]}D → {self.reduced_dim}D ({variance_retained:.1f}% variance retained)")
        else:
            features_reduced = features_scaled
            self.reduced_dim = features_scaled.shape[1]

        # Learn codebook via K-means
        self.kmeans.fit(features_reduced)
        self.codebook = self.kmeans.cluster_centers_

        # Enforce minimum cluster separation (merge similar clusters and re-cluster)
        if self.min_cluster_separation > 0:
            self._enforce_cluster_separation(features_reduced)

        self.is_fitted = True

        # Calculate token frequencies (for analysis)
        tokens = self.kmeans.predict(features_reduced)
        unique, counts = np.unique(tokens, return_counts=True)
        self.token_frequencies = dict(zip(unique, counts))

        # Calculate cluster separation statistics
        min_dist, avg_dist = self._calculate_cluster_distances()

        print(f"   ✅ Vocabulary learned!")
        print(f"   Codebook shape: {self.codebook.shape}")
        print(f"   Active tokens: {len(unique)}/{self.vocabulary_size}")
        print(f"   Cluster separation: min={min_dist:.3f}, avg={avg_dist:.3f}")

        return self

    def _enforce_cluster_separation(self, features: np.ndarray):
        """
        Enforce minimum distance between cluster centers to improve diversity.
        Merges similar clusters and re-runs k-means with better initialization.
        """
        max_iterations = 3
        for iteration in range(max_iterations):
            # Calculate pairwise distances between cluster centers
            centers = self.kmeans.cluster_centers_
            n_clusters = len(centers)

            # Find pairs of clusters that are too close
            too_close = []
            for i in range(n_clusters):
                for j in range(i + 1, n_clusters):
                    dist = np.linalg.norm(centers[i] - centers[j])
                    if dist < self.min_cluster_separation:
                        too_close.append((i, j, dist))

            if not too_close:
                break  # All clusters are well-separated

            # Perturb close cluster centers to encourage separation
            for i, j, dist in too_close:
                # Add noise to push clusters apart
                noise_scale = self.min_cluster_separation - dist
                centers[i] += np.random.randn(centers[i].shape[0]) * noise_scale * 0.5
                centers[j] -= np.random.randn(centers[j].shape[0]) * noise_scale * 0.5

            # Re-fit with perturbed centers as initialization
            self.kmeans = MiniBatchKMeans(
                n_clusters=self.vocabulary_size,
                n_init=1,  # Single init from our perturbed centers
                init=centers,
                random_state=self.random_state + iteration,
                batch_size=1024,
                max_iter=100
            )
            self.kmeans.fit(features)

        self.codebook = self.kmeans.cluster_centers_

    def _calculate_cluster_distances(self) -> tuple:
        """Calculate min and average pairwise distances between cluster centers."""
        if self.codebook is None:
            return 0.0, 0.0

        centers = self.codebook
        n_clusters = len(centers)
        distances = []

        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                dist = np.linalg.norm(centers[i] - centers[j])
                distances.append(dist)

        if not distances:
            return 0.0, 0.0

        return float(np.min(distances)), float(np.mean(distances))
    
    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        Convert features to symbolic tokens

        Args:
            features: Array of shape (n_samples, n_features) or (n_features,)

        Returns:
            Array of token IDs (integers 0 to vocabulary_size-1)
        """
        if not self.is_fitted:
            raise ValueError("Must call fit() before transform()")

        # Handle single sample
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Normalize first
        if self.use_l2_norm:
            features_scaled = normalize(features, norm='l2', axis=1)
        else:
            features_scaled = self.scaler.transform(features)

        # Apply PCA if enabled
        if self.use_pca and self.pca is not None:
            features_reduced = self.pca.transform(features_scaled)
        else:
            features_reduced = features_scaled

        # Quantize to tokens
        tokens = self.kmeans.predict(features_reduced)

        return tokens
    
    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """Fit and transform in one step"""
        self.fit(features)
        return self.transform(features)
    
    def inverse_transform(self, tokens: np.ndarray) -> np.ndarray:
        """
        Convert tokens back to feature vectors
        
        Args:
            tokens: Array of token IDs
            
        Returns:
            Reconstructed features (from codebook centroids)
        """
        if not self.is_fitted:
            raise ValueError("Must call fit() before inverse_transform()")
        
        # Get codebook vectors for these tokens
        quantized_scaled = self.codebook[tokens]
        
        # Inverse scaling (if not using L2)
        if self.use_l2_norm:
            # L2-normalized vectors stay as is (on unit hypersphere)
            quantized = quantized_scaled
        else:
            quantized = self.scaler.inverse_transform(quantized_scaled)
        
        return quantized
    
    def get_token_label(self, token_id: int) -> str:
        """
        Get human-readable label for a token
        
        Args:
            token_id: Token ID (0 to vocabulary_size-1)
            
        Returns:
            String label like "Token_42"
        """
        return f"Token_{token_id:03d}"
    
    def get_codebook_statistics(self) -> Dict:
        """
        Get statistics about the learned codebook
        
        Returns:
            Dictionary with usage statistics
        """
        if not self.is_fitted:
            return {}
        
        total_usage = sum(self.token_frequencies.values())
        
        stats = {
            'vocabulary_size': self.vocabulary_size,
            'active_tokens': len(self.token_frequencies),
            'unused_tokens': self.vocabulary_size - len(self.token_frequencies),
            'total_assignments': total_usage,
            'token_frequencies': self.token_frequencies,
            'entropy': self._calculate_entropy(),
        }
        
        return stats
    
    def _calculate_entropy(self) -> float:
        """Calculate entropy of token distribution (measure of diversity)"""
        if not self.token_frequencies:
            return 0.0
        
        total = sum(self.token_frequencies.values())
        probs = np.array([count / total for count in self.token_frequencies.values()])
        
        # Shannon entropy
        entropy = -np.sum(probs * np.log2(probs + 1e-9))
        
        return float(entropy)
    
    def save(self, filepath: str):
        """Save quantizer to disk"""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted quantizer")

        data = {
            'kmeans': self.kmeans,
            'scaler': self.scaler,
            'codebook': self.codebook,
            'vocabulary_size': self.vocabulary_size,
            'feature_dim': self.feature_dim,
            'token_frequencies': self.token_frequencies,
            'use_l2_norm': self.use_l2_norm,
            # New PCA-related fields
            'pca': self.pca,
            'use_pca': self.use_pca,
            'pca_components': self.pca_components,
            'reduced_dim': self.reduced_dim,
            'min_cluster_separation': self.min_cluster_separation
        }

        joblib.dump(data, filepath)

        # Determine type from filename to provide appropriate message
        if 'gesture' in filepath.lower():
            print(f"💾 Gesture vocabulary saved: {filepath}")
        elif 'symbolic' in filepath.lower():
            print(f"💾 Symbolic quantizer saved: {filepath}")
        else:
            # Generic message for backward compatibility
            print(f"💾 Quantizer saved: {filepath}")
    
    def load(self, filepath: str):
        """Load quantizer from disk"""
        data = joblib.load(filepath)

        self.kmeans = data['kmeans']
        self.scaler = data.get('scaler')  # May be None if L2 norm
        self.codebook = data['codebook']
        self.vocabulary_size = data['vocabulary_size']
        self.feature_dim = data['feature_dim']
        self.token_frequencies = data.get('token_frequencies', {})
        self.use_l2_norm = data.get('use_l2_norm', False)  # Default to old behavior

        # Load PCA-related fields (with backward compatibility)
        self.pca = data.get('pca', None)
        self.use_pca = data.get('use_pca', False)  # Default to no PCA for old models
        self.pca_components = data.get('pca_components', 128)
        self.reduced_dim = data.get('reduced_dim', self.feature_dim)
        self.min_cluster_separation = data.get('min_cluster_separation', 0.0)

        self.is_fitted = True

        print(f"📚 Symbolic quantizer loaded: {filepath}")
        print(f"   Vocabulary: {self.vocabulary_size} classes")
        print(f"   Feature dim: {self.feature_dim}")
        if self.use_pca:
            print(f"   PCA: {self.feature_dim}D → {self.reduced_dim}D")
        print(f"   Normalization: {'L2 (IRCAM)' if self.use_l2_norm else 'StandardScaler'}")


def demo():
    """Demonstrate symbolic quantization"""
    print("=" * 70)
    print("Symbolic Vector Quantization - Demo")
    print("=" * 70)
    
    # Simulate features from different chords
    # In reality, these come from audio analysis
    np.random.seed(42)
    
    # Create synthetic features for 3 chord types
    n_samples_per_chord = 50
    n_features = 25  # e.g., 12 chroma + ratio features + spectral
    
    # C major cluster
    c_major = np.random.randn(n_samples_per_chord, n_features) + np.array([1, 0, 0, 0, 1, 0, 0, 1] + [0]*17)
    
    # C minor cluster  
    c_minor = np.random.randn(n_samples_per_chord, n_features) + np.array([1, 0, 0, 1, 0, 0, 0, 1] + [0]*17)
    
    # G7 cluster
    g7 = np.random.randn(n_samples_per_chord, n_features) + np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1] + [0]*14)
    
    all_features = np.vstack([c_major, c_minor, g7])
    
    print(f"\n📊 Training Data:")
    print(f"   Total samples: {len(all_features)}")
    print(f"   Feature dimension: {n_features}")
    print(f"   Representing 3 chord types\n")
    
    # Create quantizer with small vocabulary
    quantizer = SymbolicQuantizer(vocabulary_size=8)
    
    # Learn codebook
    tokens = quantizer.fit_transform(all_features)
    
    print(f"\n🎵 Symbolic Representation:")
    print(f"   C major samples → tokens: {tokens[0:5]}")
    print(f"   C minor samples → tokens: {tokens[50:55]}")
    print(f"   G7 samples      → tokens: {tokens[100:105]}")
    
    # Statistics
    stats = quantizer.get_codebook_statistics()
    print(f"\n📈 Codebook Statistics:")
    print(f"   Active tokens: {stats['active_tokens']}/{stats['vocabulary_size']}")
    print(f"   Entropy: {stats['entropy']:.2f} bits")
    print(f"   Token frequencies: {stats['token_frequencies']}")
    
    print("\n" + "=" * 70)
    print("Notice: Similar chords get mapped to similar tokens!")
    print("This makes pattern learning more efficient for AudioOracle/Transformers")
    print("=" * 70)


if __name__ == "__main__":
    demo()


