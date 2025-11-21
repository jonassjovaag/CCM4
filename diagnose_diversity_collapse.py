#!/usr/bin/env python3
"""
Diagnose why live feature extraction has low diversity (8/64 tokens)
when training had high diversity (64/64 tokens, 49K samples).

Tests:
1. Gesture smoothing effect
2. Audio segment size effect  
3. Training vs live preprocessing differences
4. Raw token extraction (bypass all processing)
"""

import librosa
import numpy as np
from listener.dual_perception import DualPerceptionModule
from collections import Counter

print("="*80)
print("DIVERSITY COLLAPSE DIAGNOSTIC")
print("="*80)

# Load audio
print("\nLoading Itzama.wav...")
audio, sr = librosa.load('input_audio/Itzama.wav', sr=44100, duration=60)
print(f"Loaded {len(audio)/sr:.1f}s of audio")

# Test 1: Gesture smoothing effect
print("\n" + "="*80)
print("TEST 1: Gesture Smoothing Effect")
print("="*80)

print("\nCreating perception module WITH gesture smoothing (default)...")
perception_smooth = DualPerceptionModule(
    vocabulary_size=64,
    wav2vec_model='m-a-p/MERT-v1-95M',
    use_gpu=True,
    enable_symbolic=True,
    enable_dual_vocabulary=True,
    enable_wav2vec=True
)
perception_smooth.load_vocabulary('input_audio/General_idea_harmonic_vocab.joblib', vocabulary_type='harmonic')

print("\nCreating perception module WITHOUT gesture smoothing...")
perception_raw = DualPerceptionModule(
    vocabulary_size=64,
    wav2vec_model='m-a-p/MERT-v1-95M',
    use_gpu=True,
    enable_symbolic=True,
    enable_dual_vocabulary=True,
    enable_wav2vec=True
)
perception_raw.load_vocabulary('input_audio/General_idea_harmonic_vocab.joblib', vocabulary_type='harmonic')

# Disable smoothing by setting window to 0
if hasattr(perception_raw, 'gesture_smoother'):
    perception_raw.gesture_smoother.window_size = 0.0
    print("✓ Disabled gesture smoothing (window_size = 0)")

# Extract from 100 random positions
tokens_smooth = []
tokens_raw = []
np.random.seed(42)

for i in range(100):
    start = np.random.randint(0, len(audio) - int(sr * 0.1))
    segment = audio[start:start + int(sr * 0.1)]
    
    # With smoothing
    result_smooth = perception_smooth.extract_features(audio=segment, sr=sr, timestamp=start/sr, detected_f0=None)
    if result_smooth.gesture_token is not None:
        tokens_smooth.append(result_smooth.gesture_token)
    
    # Without smoothing
    result_raw = perception_raw.extract_features(audio=segment, sr=sr, timestamp=start/sr, detected_f0=None)
    if result_raw.gesture_token is not None:
        tokens_raw.append(result_raw.gesture_token)

print(f"\nRESULTS (100 random 0.1s segments):")
print(f"  WITH smoothing:    {len(set(tokens_smooth))}/64 tokens ({len(set(tokens_smooth))/64:.1%} diversity)")
print(f"  WITHOUT smoothing: {len(set(tokens_raw))}/64 tokens ({len(set(tokens_raw))/64:.1%} diversity)")

if len(set(tokens_raw)) > len(set(tokens_smooth)) * 1.5:
    print(f"\n✅ Gesture smoothing is REDUCING diversity significantly!")
else:
    print(f"\n⚠️  Smoothing not the main issue - diversity similar with/without")

# Test 2: Audio segment size effect
print("\n" + "="*80)
print("TEST 2: Audio Segment Size Effect")
print("="*80)

perception_test = DualPerceptionModule(
    vocabulary_size=64,
    wav2vec_model='m-a-p/MERT-v1-95M',
    use_gpu=True,
    enable_symbolic=True,
    enable_dual_vocabulary=True,
    enable_wav2vec=True
)
perception_test.load_vocabulary('input_audio/General_idea_harmonic_vocab.joblib', vocabulary_type='harmonic')

if hasattr(perception_test, 'gesture_smoother'):
    perception_test.gesture_smoother.window_size = 0.0

segment_sizes = [0.05, 0.1, 0.2, 0.5, 1.0]  # seconds
results = {}

for size in segment_sizes:
    tokens = []
    np.random.seed(42)
    for i in range(50):  # 50 samples per size
        start = np.random.randint(0, len(audio) - int(sr * size))
        segment = audio[start:start + int(sr * size)]
        result = perception_test.extract_features(audio=segment, sr=sr, timestamp=start/sr, detected_f0=None)
        if result.gesture_token is not None:
            tokens.append(result.gesture_token)
    
    results[size] = {
        'unique': len(set(tokens)),
        'diversity': len(set(tokens))/64,
        'tokens': tokens
    }
    print(f"  {size:4.2f}s segments: {results[size]['unique']:2d}/64 tokens ({results[size]['diversity']:.1%})")

best_size = max(results.items(), key=lambda x: x[1]['unique'])
print(f"\n✅ Best diversity at {best_size[0]:.2f}s segments: {best_size[1]['unique']}/64 tokens")

# Test 3: Check vocabulary cluster overlap
print("\n" + "="*80)
print("TEST 3: Vocabulary Cluster Analysis")
print("="*80)

print("\nExamining KMeans cluster centers...")
import joblib
vocab = joblib.load('input_audio/General_idea_harmonic_vocab.joblib')

codebook = vocab['kmeans'].cluster_centers_  # (64, 128)
from scipy.spatial.distance import pdist, squareform
distances = squareform(pdist(codebook, 'euclidean'))

# Find closest clusters
np.fill_diagonal(distances, np.inf)
min_distances = distances.min(axis=1)

print(f"  Codebook shape: {codebook.shape}")
print(f"  Min inter-cluster distance: {min_distances.min():.6f}")
print(f"  Mean inter-cluster distance: {distances[distances != np.inf].mean():.6f}")
print(f"  Max inter-cluster distance: {distances[distances != np.inf].max():.6f}")

# Find tokens that are very close together
close_pairs = []
for i in range(64):
    for j in range(i+1, 64):
        if distances[i, j] < 0.3:  # Very close clusters
            close_pairs.append((i, j, distances[i, j]))

if close_pairs:
    print(f"\n  ⚠️  Found {len(close_pairs)} cluster pairs with distance < 0.3:")
    for i, j, dist in sorted(close_pairs, key=lambda x: x[2])[:5]:
        print(f"     Tokens {i} and {j}: distance {dist:.6f}")
else:
    print(f"\n  ✅ All clusters well-separated (min distance > 0.3)")

# Test 4: Extract features from continuous audio (not random segments)
print("\n" + "="*80)
print("TEST 3: Direct Quantizer Access (Raw Features)")
print("="*80)

print("\nExtracting features directly from MERT + quantizing...")
from listener.wav2vec_perception import Wav2VecMusicEncoder

wav2vec = Wav2VecMusicEncoder(model_name='m-a-p/MERT-v1-95M', use_gpu=True)

# Extract features from 100 segments
raw_tokens = []
np.random.seed(42)

for i in range(100):
    start = np.random.randint(0, len(audio) - int(sr * 0.1))
    segment = audio[start:start + int(sr * 0.1)]
    
    # Direct MERT extraction
    features = wav2vec.extract_features(segment, sr)
    
    # Direct quantization
    if perception_test.harmonic_quantizer and perception_test.harmonic_quantizer.is_fitted:
        features_64 = features.astype(np.float64)
        token = int(perception_test.harmonic_quantizer.transform(features_64.reshape(1, -1))[0])
        raw_tokens.append(token)

print(f"\nDirect MERT → Quantizer (bypassing DualPerception):")
print(f"  Unique tokens: {len(set(raw_tokens))}/64")
print(f"  Diversity: {len(set(raw_tokens))/64:.1%}")

token_counts = Counter(raw_tokens)
print(f"\n  Top 5 tokens:")
for token, count in token_counts.most_common(5):
    print(f"    Token {token}: {count} times ({count/len(raw_tokens)*100:.1f}%)")

# Test 4: Check if training used different preprocessing
print("\n" + "="*80)
print("TEST 4: Training vs Live Preprocessing")
print("="*80)

print("\nChecking vocabulary metadata for training config...")
import joblib
vocab = joblib.load('input_audio/General_idea_harmonic_vocab.joblib')

if 'use_l2_norm' in vocab:
    print(f"  L2 normalization: {vocab['use_l2_norm']}")
if 'use_pca' in vocab:
    print(f"  PCA enabled: {vocab['use_pca']}")
if 'pca_components' in vocab:
    print(f"  PCA components: {vocab['pca_components']}")
if 'feature_dim' in vocab:
    print(f"  Feature dimension: {vocab['feature_dim']}")

print("\nChecking if DualPerception applies same preprocessing...")
print(f"  Quantizer has L2 norm: {hasattr(perception_test.harmonic_quantizer, 'use_l2_norm')}")
print(f"  Quantizer has PCA: {hasattr(perception_test.harmonic_quantizer, 'pca')}")

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"\nTraining diversity: 64/64 tokens (100%), 49,865 samples")
print(f"Live diversity:     {len(set(tokens_raw))}/64 tokens ({len(set(tokens_raw))/64:.1%})")

print(f"\nLikely causes ranked:")
causes = []

if len(set(tokens_raw)) < len(set(tokens_smooth)) * 0.7:
    causes.append("1. Gesture smoothing (reduces diversity significantly)")
elif len(set(tokens_smooth)) < len(set(tokens_raw)) * 0.7:
    causes.append("1. Gesture smoothing (INCREASES diversity - strange!)")

diversity_by_size = [results[size]['diversity'] for size in segment_sizes]
if max(diversity_by_size) > min(diversity_by_size) * 1.5:
    best_idx = diversity_by_size.index(max(diversity_by_size))
    causes.append(f"2. Segment size (0.1s too short, use {segment_sizes[best_idx]:.2f}s)")

if len(set(raw_tokens)) < 20:
    causes.append("3. MERT feature extraction itself has low variance on this audio")

if not causes:
    causes.append("Unknown - all tests show similar low diversity")

for cause in causes:
    print(f"   {cause}")

print("\n" + "="*80)
