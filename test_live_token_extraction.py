#!/usr/bin/env python3
"""
Test live token extraction pipeline to diagnose token collapse during performance.

This simulates the exact code path used during live performance:
1. Load vocabulary
2. Generate random wav2vec features (simulating live audio)
3. Apply transform (L2 norm → PCA → KMeans predict)
4. Check if tokens vary or collapse to one value
"""

import numpy as np
import joblib
from listener.symbolic_quantizer import SymbolicQuantizer

print("=" * 80)
print("Live Token Extraction Test")
print("=" * 80)

# Step 1: Load vocabulary (same as live performance)
print("\n[1] Loading harmonic vocabulary...")
harmonic_quantizer = SymbolicQuantizer(vocabulary_size=64)
harmonic_quantizer.load('input_audio/General_idea_harmonic_vocab.joblib')

print(f"  ✓ Vocabulary loaded successfully")
print(f"  ✓ is_fitted: {harmonic_quantizer.is_fitted}")
print(f"  ✓ Feature dim: {harmonic_quantizer.feature_dim}")
print(f"  ✓ Reduced dim: {harmonic_quantizer.reduced_dim}")
print(f"  ✓ Use PCA: {harmonic_quantizer.use_pca}")
print(f"  ✓ Use L2 norm: {harmonic_quantizer.use_l2_norm}")

# Step 2: Simulate live wav2vec features
print("\n[2] Simulating 50 live audio events (random wav2vec features)...")
n_events = 50
np.random.seed(42)  # Reproducible

tokens = []
for i in range(n_events):
    # Simulate wav2vec output: 768D normalized features
    # Real wav2vec typically outputs values in [-5, 5] range
    wav2vec_features = np.random.randn(768).astype(np.float64) * 2.0
    
    # Transform (same as dual_perception.py line 244)
    features_64 = wav2vec_features.astype(np.float64)
    token = int(harmonic_quantizer.transform(features_64.reshape(1, -1))[0])
    tokens.append(token)
    
    if i % 10 == 0:
        print(f"  Event {i}: token={token}")

# Step 3: Analyze token diversity
print("\n[3] Analyzing token diversity...")
unique_tokens = len(set(tokens))
most_common_token = max(set(tokens), key=tokens.count)
most_common_count = tokens.count(most_common_token)
diversity = unique_tokens / harmonic_quantizer.vocabulary_size

print(f"  ✓ Total events: {n_events}")
print(f"  ✓ Unique tokens: {unique_tokens}/{harmonic_quantizer.vocabulary_size}")
print(f"  ✓ Gesture diversity: {diversity:.2f}")
print(f"  ✓ Most common token: {most_common_token} ({most_common_count} times, {most_common_count/n_events*100:.1f}%)")

if unique_tokens == 1:
    print(f"\n  ❌ CRITICAL: Token collapse detected!")
    print(f"     All {n_events} events map to token {tokens[0]}")
    print(f"     This is EXACTLY what we saw during the singing test!")
elif diversity < 0.3:
    print(f"\n  ⚠️  WARNING: Low token diversity ({diversity:.2f})")
    print(f"     Expected ~0.5-0.8 for random features")
else:
    print(f"\n  ✓ HEALTHY: Good token diversity ({diversity:.2f})")

print(f"\n  Token distribution (first 50):")
from collections import Counter
token_counts = Counter(tokens)
for token, count in sorted(token_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"    Token {token}: {count} times ({count/n_events*100:.1f}%)")

# Step 4: Test with REAL wav2vec features (if we can load them)
print("\n[4] Testing with REAL features (if available)...")
try:
    # Try to load actual wav2vec features from training
    import json
    with open('JSON/General_idea.json', 'r') as f:
        model_data = json.load(f)
    
    if 'data' in model_data and 'audio_oracle' in model_data['data']:
        audio_frames = model_data['data']['audio_oracle'].get('audio_frames', {})
        
        if len(audio_frames) > 0:
            print(f"  ✓ Found {len(audio_frames)} training frames")
            
            # Extract first 50 wav2vec features from training
            real_tokens = []
            frame_ids = sorted([int(fid) for fid in audio_frames.keys()])[:50]
            
            for fid in frame_ids:
                frame = audio_frames[str(fid)]
                if 'features' in frame and len(frame['features']) == 768:
                    features = np.array(frame['features'], dtype=np.float64)
                    token = int(harmonic_quantizer.transform(features.reshape(1, -1))[0])
                    real_tokens.append(token)
            
            if len(real_tokens) > 0:
                unique_real = len(set(real_tokens))
                diversity_real = unique_real / harmonic_quantizer.vocabulary_size
                
                print(f"  ✓ Processed {len(real_tokens)} real training features")
                print(f"  ✓ Unique tokens: {unique_real}/{harmonic_quantizer.vocabulary_size}")
                print(f"  ✓ Gesture diversity: {diversity_real:.2f}")
                
                if unique_real == 1:
                    print(f"\n  ❌ CRITICAL: Training features ALSO collapse to token {real_tokens[0]}!")
                    print(f"     This suggests the vocabulary was trained on collapsed/degenerate features!")
                elif diversity_real < 0.3:
                    print(f"\n  ⚠️  WARNING: Training features have low diversity")
                else:
                    print(f"\n  ✓ HEALTHY: Training features have good diversity")
                
                print(f"\n  Real token distribution:")
                real_counts = Counter(real_tokens)
                for token, count in sorted(real_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                    print(f"    Token {token}: {count} times ({count/len(real_tokens)*100:.1f}%)")
except Exception as e:
    print(f"  ⚠️  Could not test real features: {e}")

print("\n" + "=" * 80)
print("CONCLUSION:")
print("=" * 80)
print("""
If random features show good diversity BUT real training features collapse:
→ The vocabulary itself is corrupt (trained on bad data)
→ Solution: Retrain vocabulary from scratch

If BOTH random and real features show good diversity:
→ The issue is elsewhere (MERT encoder, live feature extraction path)
→ Check if wav2vec_encoder is actually running during live performance

If random features collapse:
→ The transform() method has a bug
→ Check PCA/L2 normalization pipeline
""")
