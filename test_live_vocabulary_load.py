#!/usr/bin/env python3
"""
Test what happens when we load vocabularies and extract features in REAL TIME.
This simulates exactly what MusicHal_9000 does during performance.
"""

import numpy as np
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from listener.dual_perception import DualPerceptionModule

def test_live_extraction():
    """Test the full live pipeline."""
    
    print("\n" + "="*80)
    print("LIVE VOCABULARY LOAD TEST")
    print("="*80)
    
    # Create perception module (exactly as MusicHal_9000 does)
    print("\n1. Creating DualPerceptionModule...")
    perception = DualPerceptionModule(
        vocabulary_size=64,
        wav2vec_model="m-a-p/MERT-v1-95M",  # This is the default
        use_gpu=True,
        enable_symbolic=True,
        enable_dual_vocabulary=True,
        enable_wav2vec=True
    )
    
    print(f"   ✓ Created with model: m-a-p/MERT-v1-95M")
    print(f"   ✓ Dual vocabulary enabled: True")
    
    # Load vocabularies (exactly as MusicHal_9000 does)
    print("\n2. Loading vocabularies...")
    harmonic_vocab = "input_audio/General_idea_harmonic_vocab.joblib"
    percussive_vocab = "input_audio/General_idea_percussive_vocab.joblib"
    
    perception.load_vocabulary(harmonic_vocab, vocabulary_type="harmonic")
    perception.load_vocabulary(percussive_vocab, vocabulary_type="percussive")
    
    print(f"   ✓ Loaded harmonic vocab: {harmonic_vocab}")
    print(f"   ✓ Loaded percussive vocab: {percussive_vocab}")
    
    # Check if fitted
    if perception.harmonic_quantizer and perception.harmonic_quantizer.is_fitted:
        print(f"   ✓ Harmonic quantizer is fitted")
    else:
        print(f"   ❌ Harmonic quantizer NOT fitted!")
    
    # Generate some test audio (440 Hz sine wave)
    print("\n3. Generating test audio (440 Hz sine)...")
    sr = 44100
    duration = 0.1
    t = np.linspace(0, duration, int(sr * duration))
    audio = 0.3 * np.sin(2 * np.pi * 440 * t)
    
    # Extract features
    print("\n4. Extracting features (as live performance would)...")
    result = perception.extract_features(
        audio=audio,
        sr=sr,
        timestamp=0.0,
        detected_f0=440.0
    )
    
    print(f"\n5. Results:")
    print(f"   Wav2Vec Features Shape: {result.wav2vec_features.shape if result.wav2vec_features is not None else 'None'}")
    if result.wav2vec_features is not None:
        print(f"   Wav2Vec Features Mean: {result.wav2vec_features.mean():.4f}")
        print(f"   Wav2Vec Features Std: {result.wav2vec_features.std():.4f}")
        print(f"   Wav2Vec Features Min: {result.wav2vec_features.min():.4f}")
        print(f"   Wav2Vec Features Max: {result.wav2vec_features.max():.4f}")
        print(f"   First 10 values: {result.wav2vec_features[:10]}")
    
    print(f"\n   Gesture Token: {result.gesture_token}")
    print(f"   Chord: {result.chord_label}")
    print(f"   Consonance: {result.consonance:.3f}")
    
    # Extract 100 more events and check diversity
    print("\n6. Testing diversity (100 random events)...")
    tokens = []
    for i in range(100):
        # Random frequency between 200-800 Hz
        freq = np.random.uniform(200, 800)
        t = np.linspace(0, duration, int(sr * duration))
        audio = 0.3 * np.sin(2 * np.pi * freq * t)
        
        result = perception.extract_features(
            audio=audio,
            sr=sr,
            timestamp=i * 0.1,
            detected_f0=freq
        )
        
        if result.gesture_token is not None:
            tokens.append(result.gesture_token)
    
    if tokens:
        unique_tokens = len(set(tokens))
        diversity = unique_tokens / 64
        
        print(f"\n   Unique tokens: {unique_tokens}/64")
        print(f"   Diversity: {diversity:.2%}")
        
        # Token distribution
        from collections import Counter
        token_counts = Counter(tokens)
        print(f"\n   Top 10 tokens:")
        for token, count in token_counts.most_common(10):
            print(f"      Token {token}: {count} times ({count/len(tokens)*100:.1f}%)")
        
        if unique_tokens < 10:
            print(f"\n   ⚠️  WARNING: Very low diversity ({unique_tokens} tokens)!")
            print(f"      This suggests a problem with vocabulary or feature extraction.")
    else:
        print(f"\n   ❌ No tokens extracted!")
    
    print("\n" + "="*80)

if __name__ == '__main__':
    test_live_extraction()
