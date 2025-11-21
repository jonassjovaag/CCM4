#!/usr/bin/env python3
"""
Root Cause Analysis: Token 16 Dominance
========================================

The singing test revealed all 97 events mapped to token 16.
This script analyzes WHY this happens during live performance.

EVIDENCE SO FAR:
1. Vocabulary is healthy (64/64 tokens active, good cluster separation)
2. Real training features work (0.17 diversity, token 16 at 44%)
3. Random Gaussian features collapse to token 7 (not 16!)
4. Live singing test → all events to token 16

HYPOTHESIS:
Live audio → Wav2Vec → features land in a specific region of PCA space
that corresponds to token 16's cluster. This could be due to:
A) Audio preprocessing (buffer length, resampling) differs from training
B) Wav2Vec model outputs different distribution for live vs recorded audio  
C) Temporal smoothing amplifies token 16 bias
D) MERT vs Wav2Vec model mismatch (training used one, live uses other)
"""

import numpy as np
import json
import joblib
from sklearn.preprocessing import normalize

print("=" * 80)
print("ROOT CAUSE ANALYSIS: Token 16 Dominance")
print("=" * 80)

# Load vocabulary and model
vocab = joblib.load('input_audio/General_idea_harmonic_vocab.joblib')
pca = vocab['pca']
kmeans = vocab['kmeans']

with open('JSON/General_idea.json', 'r') as f:
    model_data = json.load(f)

audio_frames = model_data['data']['audio_oracle']['audio_frames']

# Extract token distribution from training data
print("\n[1] TRAINING DATA TOKEN DISTRIBUTION")
print("-" * 80)

training_tokens = []
training_features_list = []

for fid in range(len(audio_frames)):
    frame = audio_frames.get(str(fid))
    if frame and 'features' in frame and len(frame['features']) == 768:
        features = np.array(frame['features'], dtype=np.float64)
        features_l2 = normalize(features.reshape(1, -1), norm='l2', axis=1)
        features_pca = pca.transform(features_l2)
        token = int(kmeans.predict(features_pca)[0])
        training_tokens.append(token)
        training_features_list.append(features)

from collections import Counter
token_counts = Counter(training_tokens)

print(f"  ✓ Total training frames: {len(training_tokens)}")
print(f"  ✓ Unique tokens: {len(token_counts)}/64")
print(f"  ✓ Diversity: {len(token_counts)/64:.2f}")

print(f"\n  Top 10 most frequent tokens during training:")
for token, count in sorted(token_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
    pct = (count / len(training_tokens)) * 100
    print(f"    Token {token}: {count} times ({pct:.1f}%)")

token_16_count = token_counts.get(16, 0)
token_16_pct = (token_16_count / len(training_tokens)) * 100
print(f"\n  ⚠️  Token 16 appears {token_16_count} times ({token_16_pct:.1f}%) in training")
print(f"     This is {token_16_count} out of {len(training_tokens)} frames")

# Analyze token 16 cluster center
print("\n[2] TOKEN 16 CLUSTER CHARACTERISTICS")
print("-" * 80)

token_16_center = kmeans.cluster_centers_[16]
all_centers = kmeans.cluster_centers_

# Find distances from token 16 to all other cluster centers
distances_from_16 = np.linalg.norm(all_centers - token_16_center, axis=1)
sorted_neighbors = np.argsort(distances_from_16)

print(f"  Cluster center 16 statistics:")
print(f"    Mean: {token_16_center.mean():.4f}")
print(f"    Std: {token_16_center.std():.4f}")
print(f"    Min: {token_16_center.min():.4f}")
print(f"    Max: {token_16_center.max():.4f}")

print(f"\n  Nearest neighbor clusters to token 16:")
for i in range(1, 6):  # Skip 0 (itself)
    neighbor_id = sorted_neighbors[i]
    dist = distances_from_16[neighbor_id]
    print(f"    Token {neighbor_id}: distance {dist:.4f}")

print(f"\n  Farthest clusters from token 16:")
for i in range(1, 6):
    neighbor_id = sorted_neighbors[-i]
    dist = distances_from_16[neighbor_id]
    print(f"    Token {neighbor_id}: distance {dist:.4f}")

# Check if token 16 is at the "center" of the PCA space (would attract defaults)
print(f"\n  Is token 16 near the PCA space origin?")
origin_distances = np.linalg.norm(all_centers, axis=1)
token_16_dist_from_origin = origin_distances[16]
print(f"    Token 16 distance from origin: {token_16_dist_from_origin:.4f}")
print(f"    Mean cluster distance from origin: {origin_distances.mean():.4f}")
print(f"    Min cluster distance from origin: {origin_distances.min():.4f} (cluster {origin_distances.argmin()})")
print(f"    Max cluster distance from origin: {origin_distances.max():.4f} (cluster {origin_distances.argmax()})")

if token_16_dist_from_origin < origin_distances.mean():
    print(f"    ⚠️  Token 16 is CLOSER to origin than average!")
    print(f"       This means 'generic' or 'default-like' features may gravitate to token 16")

# Analyze features that map to token 16
print("\n[3] FEATURES THAT MAP TO TOKEN 16")
print("-" * 80)

token_16_features = []
for i, token in enumerate(training_tokens):
    if token == 16:
        token_16_features.append(training_features_list[i])

if len(token_16_features) > 0:
    token_16_features = np.array(token_16_features)
    
    print(f"  ✓ Collected {len(token_16_features)} features that mapped to token 16")
    print(f"  Raw feature statistics (before L2 norm):")
    print(f"    Mean: {token_16_features.mean():.4f}")
    print(f"    Std: {token_16_features.std():.4f}")
    print(f"    Min: {token_16_features.min():.4f}")
    print(f"    Max: {token_16_features.max():.4f}")
    
    # Apply same transform as live performance
    token_16_l2 = normalize(token_16_features, norm='l2', axis=1)
    token_16_pca = pca.transform(token_16_l2)
    
    print(f"\n  After L2 normalization:")
    print(f"    Mean: {token_16_l2.mean():.4f}")
    print(f"    Std: {token_16_l2.std():.4f}")
    
    print(f"\n  After PCA projection:")
    print(f"    Mean: {token_16_pca.mean():.4f}")
    print(f"    Std: {token_16_pca.std():.4f}")
    print(f"    Min: {token_16_pca.min():.4f}")
    print(f"    Max: {token_16_pca.max():.4f}")
    
    # Check if these are "typical" or "outlier" features
    all_pca_features = []
    all_l2 = normalize(np.array(training_features_list), norm='l2', axis=1)
    all_pca = pca.transform(all_l2)
    
    print(f"\n  Comparison to ALL training features:")
    print(f"    Token 16 PCA mean: {token_16_pca.mean():.4f}")
    print(f"    All features PCA mean: {all_pca.mean():.4f}")
    print(f"    Token 16 PCA std: {token_16_pca.std():.4f}")
    print(f"    All features PCA std: {all_pca.std():.4f}")

print("\n" + "=" * 80)
print("DIAGNOSIS:")
print("=" * 80)
print("""
LIKELY CAUSES OF TOKEN 16 DOMINANCE:

1. GENERIC FEATURE BIAS:
   If token 16's cluster center is near the PCA space origin, it acts as a
   "default" or "generic" category. Any featureless/repetitive audio will
   gravitate to it.

2. TRAINING DATA BIAS:
   If training audio had long stretches of similar material (e.g., sustained
   notes, electronic pads), token 16 may represent "sustained harmonic content"
   rather than distinct musical gestures.

3. LIVE AUDIO CHARACTERISTICS:
   If singing test used:
   - Similar pitch/timbre throughout (limited vocal variation)
   - Long sustained notes (not rhythmically varied)
   - Electronic processing (reverb, compression)
   Then features would cluster around token 16's "sustained" pattern.

SOLUTIONS:

A) Retrain vocabulary with MORE DIVERSE audio data
   - Include short staccato notes, varied timbres, different instruments
   - Ensure no single musical pattern dominates training set

B) Adjust query during live performance
   - Don't rely solely on gesture tokens
   - Weight consonance, ratio features, and rhythmic patterns more heavily
   - Use request masking to override token bias

C) Add token diversity penalty
   - If last N tokens were all 16, force system to try other tokens
   - Implement anti-repetition filter in gesture smoothing

D) Check MERT vs Wav2Vec model consistency
   - Verify training used same model as live performance
   - If training used Wav2Vec but live uses MERT (or vice versa),
     feature distributions will mismatch
""")

# Final check: Which model was used for training?
print("\n[4] MODEL CONSISTENCY CHECK")
print("-" * 80)
print("  Checking if training and live performance use same encoder...")

# Check model metadata in General_idea.json
if 'training_config' in model_data:
    config = model_data['training_config']
    training_model = config.get('wav2vec_model', 'UNKNOWN')
    print(f"  Training model: {training_model}")
else:
    print(f"  ⚠️  No training_config found in model file")
    print(f"     Cannot verify model consistency!")

print(f"\n  Live performance model (from MusicHal_9000.py default):")
print(f"    Default: m-a-p/MERT-v1-95M")
print(f"    Can override with --wav2vec-model flag")

print(f"\n  If training used facebook/wav2vec2-base but live uses MERT-v1-95M:")
print(f"    → Feature distributions will DIFFER significantly!")
print(f"    → This could explain token collapse!")
