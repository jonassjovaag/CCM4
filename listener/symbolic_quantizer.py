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
import joblib
from typing import Dict


class SymbolicQuantizer:
    """
    Vector Quantizer for creating symbolic musical representations
    
    Converts continuous audio features into discrete tokens from a learned vocabulary.
    Based on K-means clustering (like Bujard et al. 2025).
    """
    
    def __init__(self, vocabulary_size: int = 64, 
                 n_init: int = 10,
                 random_state: int = 42,
                 use_l2_norm: bool = True):
        """
        Initialize symbolic quantizer
        
        Args:
            vocabulary_size: Size of musical alphabet (16, 64, or 256 recommended)
                           Paper found 64 works best for musical relationships
            n_init: Number of K-means initializations
            random_state: Random seed for reproducibility
            use_l2_norm: Use L2 normalization (IRCAM approach, recommended for Wav2Vec)
                       If False, uses StandardScaler (for other feature types)
        """
        self.vocabulary_size = vocabulary_size
        self.n_init = n_init
        self.random_state = random_state
        self.use_l2_norm = use_l2_norm
        
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
        print(f"   Normalization: {'L2 (IRCAM)' if self.use_l2_norm else 'StandardScaler'}")
        
        self.feature_dim = features.shape[1]
        
        # Normalize features
        if self.use_l2_norm:
            # L2 normalization: Each vector has unit length (IRCAM approach)
            # This makes all vectors lie on a hypersphere
            features_scaled = normalize(features, norm='l2', axis=1)
        else:
            # StandardScaler: Mean=0, Std=1 (traditional approach)
            features_scaled = self.scaler.fit_transform(features)
        
        # Learn codebook via K-means
        self.kmeans.fit(features_scaled)
        self.codebook = self.kmeans.cluster_centers_
        
        self.is_fitted = True
        
        # Calculate token frequencies (for analysis)
        tokens = self.kmeans.predict(features_scaled)
        unique, counts = np.unique(tokens, return_counts=True)
        self.token_frequencies = dict(zip(unique, counts))
        
        print(f"   ✅ Vocabulary learned!")
        print(f"   Codebook shape: {self.codebook.shape}")
        print(f"   Active tokens: {len(unique)}/{self.vocabulary_size}")
        
        return self
    
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
        
        # Normalize and quantize
        if self.use_l2_norm:
            features_scaled = normalize(features, norm='l2', axis=1)
        else:
            features_scaled = self.scaler.transform(features)
        tokens = self.kmeans.predict(features_scaled)
        
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
            'use_l2_norm': self.use_l2_norm
        }
        
        joblib.dump(data, filepath)
        print(f"💾 Symbolic quantizer saved: {filepath}")
    
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
        self.is_fitted = True
        
        print(f"📚 Symbolic quantizer loaded: {filepath}")
        print(f"   Vocabulary: {self.vocabulary_size} classes")
        print(f"   Feature dim: {self.feature_dim}")
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


