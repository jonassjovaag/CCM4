#!/usr/bin/env python3
"""
Final validation: Test token diversity with all-frames mode vs averaged mode
"""

import numpy as np
import librosa
from listener.dual_perception import DualPerceptionModule
from collections import Counter

print("🎯 FINAL VALIDATION: All-Frames vs Averaged Mode")
print("=" * 80)

# Load audio (10 seconds for meaningful diversity test - faster than 60s)
audio, sr = librosa.load('input_audio/General_idea.wav', sr=44100, duration=10, offset=10)
print(f"🎵 Audio: {len(audio)/sr:.1f}s at {sr}Hz")

# Test 1: All-frames mode (NEW - matches training)
print("\n" + "=" * 80)
print("TEST 1: All-Frames Mode (matches training)")
print("=" * 80)

perception_all_frames = DualPerceptionModule(
    vocabulary_size=64,
    wav2vec_model="m-a-p/MERT-v1-95M",
    use_gpu=False,
    enable_symbolic=True,
    enable_dual_vocabulary=True,
    extract_all_frames=True  # MATCH TRAINING
)

perception_all_frames.load_vocabulary("input_audio/General_idea_harmonic_vocab.joblib", "harmonic")
perception_all_frames.load_vocabulary("input_audio/General_idea_percussive_vocab.joblib", "percussive")

# Process 350ms segments
segment_dur = 0.35
hop = 0.35
tokens_all_frames = []

print(f"\nProcessing {len(audio)/sr:.1f}s audio in 350ms segments...")
num_segments = 0
num_frames_per_segment = []

for start_sec in np.arange(0, len(audio)/sr - segment_dur, hop):
    start = int(start_sec * sr)
    end = int((start_sec + segment_dur) * sr)
    segment = audio[start:end]
    
    result = perception_all_frames.extract_features(
        audio=segment,
        sr=int(sr),
        timestamp=start_sec,
        detected_f0=None
    )
    
    num_segments += 1
    
    # Extract tokens from all frames
    if isinstance(result, list):
        frame_tokens = [r.harmonic_token for r in result if r.harmonic_token is not None]
        num_frames_per_segment.append(len(frame_tokens))
        tokens_all_frames.extend(frame_tokens)
    else:
        num_frames_per_segment.append(1)
        if result.harmonic_token is not None:
            tokens_all_frames.append(result.harmonic_token)

print(f"✅ Processed {num_segments} segments")
print(f"   Frames per segment: {num_frames_per_segment}")
print(f"   Total tokens extracted: {len(tokens_all_frames)}")
print(f"   Expected: {num_segments} segments × ~26 frames = ~{num_segments * 26} tokens")

# Analyze diversity
token_counts = Counter(tokens_all_frames)
unique_tokens = len(token_counts)

print(f"\n📊 All-Frames Mode Results:")
print(f"   Total tokens: {len(tokens_all_frames)}")
print(f"   Unique tokens: {unique_tokens}/64 ({100*unique_tokens/64:.1f}%)")
print(f"\n   Top 10 tokens:")
for token_id, count in token_counts.most_common(10):
    print(f"      Token {token_id}: {count} times ({100*count/len(tokens_all_frames):.1f}%)")

top_3_pct = sum(c for _, c in token_counts.most_common(3)) / len(tokens_all_frames) * 100
print(f"\n   Top 3 concentration: {top_3_pct:.1f}%")

# Test 2: Averaged mode (LEGACY - causes diversity collapse)
print("\n" + "=" * 80)
print("TEST 2: Averaged Mode (legacy - causes collapse)")
print("=" * 80)

perception_averaged = DualPerceptionModule(
    vocabulary_size=64,
    wav2vec_model="m-a-p/MERT-v1-95M",
    use_gpu=False,
    enable_symbolic=True,
    enable_dual_vocabulary=True,
    extract_all_frames=False  # LEGACY AVERAGING
)

perception_averaged.load_vocabulary("input_audio/General_idea_harmonic_vocab.joblib", "harmonic")
perception_averaged.load_vocabulary("input_audio/General_idea_percussive_vocab.joblib", "percussive")

tokens_averaged = []

print(f"\nProcessing {len(audio)/sr:.1f}s audio in 350ms segments...")
for start_sec in np.arange(0, len(audio)/sr - segment_dur, hop):
    start = int(start_sec * sr)
    end = int((start_sec + segment_dur) * sr)
    segment = audio[start:end]
    
    result = perception_averaged.extract_features(
        audio=segment,
        sr=int(sr),
        timestamp=start_sec,
        detected_f0=None
    )
    
    if isinstance(result, list):
        # Shouldn't happen with averaged mode
        frame_tokens = [r.harmonic_token for r in result if r.harmonic_token is not None]
        tokens_averaged.extend(frame_tokens)
    else:
        if result.harmonic_token is not None:
            tokens_averaged.append(result.harmonic_token)

print(f"✅ Extracted {len(tokens_averaged)} tokens")

# Analyze diversity
token_counts_avg = Counter(tokens_averaged)
unique_tokens_avg = len(token_counts_avg)

print(f"\n📊 Averaged Mode Results:")
print(f"   Total tokens: {len(tokens_averaged)}")
print(f"   Unique tokens: {unique_tokens_avg}/64 ({100*unique_tokens_avg/64:.1f}%)")
print(f"\n   Top 10 tokens:")
for token_id, count in token_counts_avg.most_common(10):
    print(f"      Token {token_id}: {count} times ({100*count/len(tokens_averaged):.1f}%)")

top_3_pct_avg = sum(c for _, c in token_counts_avg.most_common(3)) / len(tokens_averaged) * 100
print(f"\n   Top 3 concentration: {top_3_pct_avg:.1f}%")

# COMPARISON
print("\n" + "=" * 80)
print("📈 COMPARISON")
print("=" * 80)

print(f"\nTraining baseline (from vocabulary):")
print(f"  Unique tokens: 64/64 (100%)")
print(f"  Top token: 3.5%")
print(f"  Distribution: Even, no dominance")

print(f"\nLive extraction (averaged mode - BROKEN):")
print(f"  Unique tokens: {unique_tokens_avg}/64 ({100*unique_tokens_avg/64:.1f}%)")
print(f"  Top token: {100*token_counts_avg.most_common(1)[0][1]/len(tokens_averaged):.1f}%")
print(f"  Top 3: {top_3_pct_avg:.1f}%")

print(f"\nLive extraction (all-frames mode - FIXED):")
print(f"  Unique tokens: {unique_tokens}/64 ({100*unique_tokens/64:.1f}%)")
print(f"  Top token: {100*token_counts.most_common(1)[0][1]/len(tokens_all_frames):.1f}%")
print(f"  Top 3: {top_3_pct:.1f}%")

# Calculate improvement
diversity_improvement = (unique_tokens / unique_tokens_avg - 1) * 100 if unique_tokens_avg > 0 else 0
print(f"\n✨ Improvement: {diversity_improvement:+.1f}% more unique tokens!")

if unique_tokens >= 45:
    print(f"\n✅ SUCCESS! All-frames mode achieves 70%+ diversity (close to training)")
elif unique_tokens >= 35:
    print(f"\n⚠️ GOOD but not perfect. Expected 60-80% diversity, got {100*unique_tokens/64:.1f}%")
else:
    print(f"\n❌ ISSUE: Expected 60-80% diversity, got {100*unique_tokens/64:.1f}%")

print("\n" + "=" * 80)
print("🎯 Validation complete!")
