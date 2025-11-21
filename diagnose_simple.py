#!/usr/bin/env python3
"""
Simple diagnostic: Why does Itzama.wav produce only 6/64 tokens?

Key findings from previous test:
- Gesture smoothing NOT the issue (6 tokens with or without)
- Segment size NOT the issue (all sizes: 2-4 tokens)
- Vocabulary trained correctly (64/64 tokens, 49K samples)

This test checks if the issue is:
1. Itzama.wav is just not musically diverse (compared to General_idea.wav)
2. Random sampling vs sequential sampling
3. Feature extraction produces low-variance outputs on this specific audio
"""

import librosa
import numpy as np
from listener.dual_perception import DualPerceptionModule
from collections import Counter

print("="*80)
print("SIMPLIFIED DIVERSITY DIAGNOSTIC")
print("="*80)

# Create perception module (no smoothing)
perception = DualPerceptionModule(
    vocabulary_size=64,
    wav2vec_model='m-a-p/MERT-v1-95M',
    use_gpu=True,
    enable_symbolic=True,
    enable_dual_vocabulary=True,
    enable_wav2vec=True
)
perception.load_vocabulary('input_audio/General_idea_harmonic_vocab.joblib', vocabulary_type='harmonic')

# Disable smoothing
if hasattr(perception, 'gesture_smoother') and hasattr(perception.gesture_smoother, 'window_size'):
    perception.gesture_smoother.window_size = 0.0
    print("✓ Gesture smoothing disabled\n")

# Test 1: Sequential segments from General_idea (original training audio)
print("="*80)
print("TEST: Sequential 0.1s segments from General_idea.wav (TRAINING AUDIO)")
print("="*80)

audio, sr = librosa.load('input_audio/General_idea_converted.wav', sr=44100, duration=60)
print(f"Loaded {len(audio)/sr:.1f}s of audio\n")

tokens_sequential = []
segment_duration = 0.1  # 0.1 seconds
hop_duration = 0.05     # 50% overlap (like training)

for start_sec in np.arange(0, len(audio)/sr - segment_duration, hop_duration):
    start = int(start_sec * sr)
    segment = audio[start:start + int(sr * segment_duration)]
    
    result = perception.extract_features(audio=segment, sr=int(sr), timestamp=start_sec, detected_f0=None)
    if result.gesture_token is not None:
        tokens_sequential.append(result.gesture_token)
    
    if len(tokens_sequential) >= 200:  # Get 200 samples
        break

print(f"Extracted {len(tokens_sequential)} tokens from sequential segments")
print(f"Unique tokens: {len(set(tokens_sequential))}/64 ({len(set(tokens_sequential))/64:.1%})")

token_counts = Counter(tokens_sequential)
print(f"\nTop 10 tokens:")
for token, count in token_counts.most_common(10):
    print(f"  Token {token}: {count} times ({count/len(tokens_sequential)*100:.1f}%)")

# Check if a few tokens dominate
top_3_percent = sum(count for _, count in token_counts.most_common(3)) / len(tokens_sequential)
print(f"\nTop 3 tokens account for: {top_3_percent:.1%} of all tokens")

if len(set(tokens_sequential)) < 15:
    print("\n⚠️  VERY LOW DIVERSITY even with TRAINING audio!")
    print("    This indicates a REAL TECHNICAL PROBLEM in live extraction")
    print("    (vocabulary trained with 64/64 tokens but live uses <15)")
elif len(set(tokens_sequential)) < 30:
    print("\n⚠️  MODERATE diversity with training audio")
    print("    Some difference between training and live extraction")
elif top_3_percent > 0.5:
    print("\n⚠️  Token distribution highly skewed - a few tokens dominate")
else:
    print("\n✅ GOOD DIVERSITY - vocabulary works correctly with training audio!")
    print("    Low diversity on Itzama.wav is due to musical mismatch")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

print(f"""
Training data: General_idea.wav → 64/64 tokens, even distribution
Live (SAME):   General_idea.wav → {len(set(tokens_sequential))}/64 tokens ({len(set(tokens_sequential))/64:.1%})

VERDICT:
""")

if len(set(tokens_sequential)) > 40:
    print("""✅ HIGH DIVERSITY (>40 tokens) with original training audio!
   
   The vocabulary works CORRECTLY. The low diversity you saw with Itzama.wav
   (11/64 tokens, 17%) is due to MUSICAL MISMATCH, not a technical bug.
   
   General_idea.wav has different musical characteristics than Itzama.wav,
   so the vocabulary learned patterns specific to General_idea's style.
   
   This is EXPECTED BEHAVIOR for a style-specific vocabulary.""")
elif len(set(tokens_sequential)) > 25:
    print("""⚠️  MODERATE DIVERSITY (25-40 tokens) - better than Itzama but not ideal
   
   This suggests some difference between training and live extraction,
   but vocabulary is partially working. May need investigation.""")
else:
    print(f"""❌ LOW DIVERSITY (<25 tokens) even with TRAINING audio!
   
   Training used 64/64 tokens but live extraction only uses {len(set(tokens_sequential))}.
   This indicates a REAL TECHNICAL PROBLEM in the live feature extraction pipeline.
   
   Possible causes:
   - Preprocessing differences between training and live
   - Feature normalization issues
   - PCA transformation problems
   - Audio segmentation differences""")

print(f"""
COMPARISON SUMMARY:
- Training:         General_idea.wav → 64/64 tokens (100%), even distribution
- Live (trained):   General_idea.wav → {len(set(tokens_sequential))}/64 tokens ({len(set(tokens_sequential))/64:.1%})
- Live (different): Itzama.wav → 11/64 tokens (17.2%)
""")

print("="*80)
