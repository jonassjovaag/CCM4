#!/usr/bin/env python3
"""
Debug: Check if all-frames mode is actually returning multiple tokens per segment
"""

import numpy as np
import librosa
from listener.dual_perception import DualPerceptionModule

print("🔍 DEBUG: Are we getting multiple tokens per segment?")
print("=" * 80)

perception = DualPerceptionModule(
    vocabulary_size=64,
    wav2vec_model="m-a-p/MERT-v1-95M",
    use_gpu=False,
    enable_symbolic=True,
    enable_dual_vocabulary=True,
    extract_all_frames=True,
    gesture_window=0.0,  # DISABLE smoothing to see raw tokens
    gesture_min_tokens=1
)

perception.load_vocabulary("input_audio/General_idea_harmonic_vocab.joblib", "harmonic")
perception.load_vocabulary("input_audio/General_idea_percussive_vocab.joblib", "percussive")

# Load one segment
audio, sr = librosa.load('input_audio/General_idea.wav', sr=44100, duration=0.35, offset=10)
print(f"🎵 Single 350ms segment")

result = perception.extract_features(
    audio=audio,
    sr=int(sr),
    timestamp=0.0,
    detected_f0=None
)

print(f"\n📊 Result type: {type(result)}")
if isinstance(result, list):
    print(f"✅ List of {len(result)} results")
    tokens = [r.harmonic_token for r in result if r.harmonic_token is not None]
    print(f"   Tokens: {tokens}")
    print(f"   Unique: {len(set(tokens))}")
else:
    print(f"❌ Single result (not a list)")
    print(f"   Token: {result.harmonic_token}")
