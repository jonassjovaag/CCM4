#!/usr/bin/env python3
"""
Test all-frames mode in DualPerceptionModule
"""

import numpy as np
import librosa
from listener.dual_perception import DualPerceptionModule

print("🧪 Testing All-Frames Mode in DualPerceptionModule")
print("=" * 80)

# Load vocabulary
perception_all_frames = DualPerceptionModule(
    vocabulary_size=64,
    wav2vec_model="m-a-p/MERT-v1-95M",
    use_gpu=False,
    enable_symbolic=True,
    enable_dual_vocabulary=True,
    extract_all_frames=True  # NEW: Enable all-frames extraction
)

# Load vocabularies
perception_all_frames.load_vocabulary("input_audio/General_idea_harmonic_vocab.joblib", "harmonic")
perception_all_frames.load_vocabulary("input_audio/General_idea_percussive_vocab.joblib", "percussive")

# Load audio
audio, sr = librosa.load('input_audio/General_idea.wav', sr=44100, duration=10, offset=10)
print(f"\n🎵 Audio: {len(audio)/sr:.1f}s at {sr}Hz")

# Extract features for 350ms segment
segment_dur = 0.35
segment = audio[:int(segment_dur * sr)]

print(f"\n⚡ Extracting features with all-frames mode...")
result = perception_all_frames.extract_features(
    audio=segment,
    sr=sr,
    timestamp=0.0,
    detected_f0=None
)

print(f"\n📊 Results:")
if isinstance(result, list):
    print(f"✅ Returned list of {len(result)} DualPerceptionResult objects (one per MERT frame)")
    print(f"   Expected ~26 frames for 350ms segment")
    
    # Check tokens
    tokens = [r.harmonic_token for r in result if r.harmonic_token is not None]
    unique_tokens = len(set(tokens))
    
    print(f"\n🎯 Tokens:")
    print(f"   Total tokens: {len(tokens)}")
    print(f"   Unique tokens: {unique_tokens}")
    
    if len(tokens) > 0:
        from collections import Counter
        token_counts = Counter(tokens)
        print(f"   Top 5 tokens:")
        for token_id, count in token_counts.most_common(5):
            print(f"      Token {token_id}: {count} times ({100*count/len(tokens):.1f}%)")
    
    # Check timestamps
    timestamps = [r.timestamp for r in result]
    print(f"\n⏱️  Timestamps:")
    print(f"   First: {timestamps[0]:.4f}s")
    print(f"   Last: {timestamps[-1]:.4f}s")
    print(f"   Span: {timestamps[-1] - timestamps[0]:.4f}s")
    
    print(f"\n✅ SUCCESS! All-frames mode working correctly")
else:
    print(f"❌ ERROR: Expected list, got single result")
    print(f"   Type: {type(result)}")

print("\n" + "=" * 80)
print("🔬 Now testing legacy averaged mode...")

perception_averaged = DualPerceptionModule(
    vocabulary_size=64,
    wav2vec_model="m-a-p/MERT-v1-95M",
    use_gpu=False,
    enable_symbolic=True,
    enable_dual_vocabulary=True,
    extract_all_frames=False  # Legacy averaging
)

perception_averaged.load_vocabulary("input_audio/General_idea_harmonic_vocab.joblib", "harmonic")
perception_averaged.load_vocabulary("input_audio/General_idea_percussive_vocab.joblib", "percussive")

result_averaged = perception_averaged.extract_features(
    audio=segment,
    sr=sr,
    timestamp=0.0,
    detected_f0=None
)

print(f"\n📊 Results (averaged mode):")
if not isinstance(result_averaged, list):
    print(f"✅ Returned single DualPerceptionResult object (legacy behavior)")
    print(f"   Token: {result_averaged.harmonic_token}")
else:
    print(f"❌ ERROR: Expected single result, got list of {len(result_averaged)}")

print("\n" + "=" * 80)
print("✅ Test complete!")
