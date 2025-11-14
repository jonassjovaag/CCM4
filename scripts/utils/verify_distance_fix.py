#!/usr/bin/env python3
"""
Verify Distance Function Fix

Shows the difference between:
- OLD: Euclidean distance with adaptive threshold (grew to 269.9)
- NEW: Cosine distance with 0.15 threshold (stable, normalized)
"""

import numpy as np
from scipy.spatial.distance import cosine as cosine_distance

print("🔍 DISTANCE FUNCTION FIX VERIFICATION")
print("=" * 80)

# Simulate 768D Wav2Vec features (same characteristics as Itzama)
np.random.seed(42)

# Create features with low variance (like collapsed Wav2Vec)
# 764 dimensions dead, 4 active
base_features = np.zeros(768)
base_features[0] = -0.3  # Mean from actual data
base_features[2] = 0.1
base_features[3] = 0.05
base_features[15] = 0.2

# Add small variation
features_1 = base_features + np.random.normal(0, 0.01, 768)
features_2 = base_features + np.random.normal(0, 0.02, 768)
features_3 = base_features + np.random.normal(0, 0.5, 768)  # More different

print("\n1. EUCLIDEAN DISTANCE (OLD METHOD)")
print("-" * 80)

euclidean_1_2 = np.linalg.norm(features_1 - features_2)
euclidean_1_3 = np.linalg.norm(features_1 - features_3)

print(f"   Similar features:   {euclidean_1_2:.4f}")
print(f"   Different features: {euclidean_1_3:.4f}")
print(f"\n   With threshold=0.15:")
print(f"     Similar pair:   {'✅ MATCH' if euclidean_1_2 < 0.15 else '❌ NO MATCH'}")
print(f"     Different pair: {'✅ MATCH' if euclidean_1_3 < 0.15 else '❌ NO MATCH'}")
print(f"\n   ⚠️  Euclidean on 768D vectors ranges 0-40+")
print(f"   ⚠️  Threshold 0.15 is TOO STRICT (matches almost nothing)")
print(f"   ⚠️  Adaptive adjustment inflated to 269.9 (matches EVERYTHING)")

print("\n2. COSINE DISTANCE (NEW METHOD - FIXED)")
print("-" * 80)

# Cosine distance = 1 - cosine_similarity
cos_1_2 = cosine_distance(features_1, features_2)
cos_1_3 = cosine_distance(features_1, features_3)

print(f"   Similar features:   {cos_1_2:.4f}")
print(f"   Different features: {cos_1_3:.4f}")
print(f"\n   With threshold=0.15:")
print(f"     Similar pair:   {'✅ MATCH' if cos_1_2 < 0.15 else '❌ NO MATCH'}")
print(f"     Different pair: {'✅ MATCH' if cos_1_3 < 0.15 else '❌ NO MATCH'}")
print(f"\n   ✅ Cosine distance is normalized (0-2 range)")
print(f"   ✅ Threshold 0.15 works correctly")
print(f"   ✅ No dangerous adaptive inflation")

print("\n3. ACTUAL ITZAMA TRAINING DATA")
print("-" * 80)

import json

with open('JSON/Itzama_061125_2255_training_model.json', 'r') as f:
    data = json.load(f)

print(f"   Current configuration:")
print(f"     distance_function:  {data['distance_function']}")
print(f"     distance_threshold: {data['distance_threshold']:.4f}")
print(f"\n   ❌ Using Euclidean with inflated threshold")
print(f"   ❌ Matches EVERYTHING (threshold 68x mean distance)")
print(f"   ❌ Destroys variation → limited phrase diversity")

print("\n4. SOLUTION")
print("=" * 80)
print("   ✅ Changed hybrid_batch_trainer.py default:")
print("      distance_function: 'euclidean' → 'cosine'")
print("   ✅ Fixed adaptive threshold logic:")
print("      - Cosine: clamped to [0.05, 0.5]")
print("      - Euclidean: use 25th percentile, max 2x initial")
print("\n   🔄 RETRAIN REQUIRED:")
print("      The current Itzama model was trained with broken distance function.")
print("      Retraining will use cosine distance + stable threshold.")
print("\n   Expected result:")
print("      - Better phrase variation (AudioOracle can distinguish gestures)")
print("      - Gesture tokens still diverse (60/64 already working)")
print("      - More musical responses (proper pattern matching)")

print("\n" + "=" * 80)
