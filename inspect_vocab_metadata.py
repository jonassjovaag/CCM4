#!/usr/bin/env python3
"""
Inspect vocabulary file metadata to determine which model was used during training.
"""

import joblib
import sys
from pathlib import Path

def inspect_vocabulary(vocab_path: Path):
    """Load vocabulary and check metadata."""
    print(f"\n{'='*80}")
    print(f"Inspecting: {vocab_path.name}")
    print(f"{'='*80}")
    
    vocab_data = joblib.load(vocab_path)
    
    # Check structure
    print(f"\nVocabulary Type: {type(vocab_data)}")
    
    if isinstance(vocab_data, dict):
        print(f"\nKeys: {list(vocab_data.keys())}")
        
        # Check for metadata
        if 'metadata' in vocab_data:
            print(f"\nMetadata:")
            for key, value in vocab_data['metadata'].items():
                print(f"  {key}: {value}")
        
        # Check for model info
        if 'model_name' in vocab_data:
            print(f"\nModel Name: {vocab_data['model_name']}")
        
        if 'config' in vocab_data:
            print(f"\nConfig:")
            for key, value in vocab_data['config'].items():
                print(f"  {key}: {value}")
        
        # Check quantizer
        if 'quantizer' in vocab_data:
            quantizer = vocab_data['quantizer']
            print(f"\nQuantizer Type: {type(quantizer)}")
            if hasattr(quantizer, 'n_clusters'):
                print(f"N Clusters: {quantizer.n_clusters}")
        
        # Check PCA
        if 'pca' in vocab_data:
            pca = vocab_data['pca']
            print(f"\nPCA Type: {type(pca)}")
            if hasattr(pca, 'n_components'):
                print(f"PCA Components: {pca.n_components}")
            if hasattr(pca, 'explained_variance_ratio_'):
                print(f"Explained Variance (first 3): {pca.explained_variance_ratio_[:3]}")
    else:
        # Legacy format - just the quantizer
        print("\nLegacy format (no metadata)")
        if hasattr(vocab_data, 'n_clusters'):
            print(f"N Clusters: {vocab_data.n_clusters}")

if __name__ == '__main__':
    vocab_files = [
        Path("input_audio/General_idea_harmonic_vocab.joblib"),
        Path("input_audio/General_idea_percussive_vocab.joblib"),
    ]
    
    for vocab_file in vocab_files:
        if vocab_file.exists():
            inspect_vocabulary(vocab_file)
        else:
            print(f"\n❌ File not found: {vocab_file}")
