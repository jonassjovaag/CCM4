#!/usr/bin/env python3
"""
Verify: What does training extract NOW with current code?
"""

import numpy as np
import librosa
from listener.dual_perception import DualPerceptionModule

print("🔍 VERIFICATION: Training extraction with current code")
print("=" * 80)

# Simulate training: Create DualPerceptionModule WITHOUT extract_all_frames parameter
# (training code doesn't pass this parameter, so it uses the default)
perception_training = DualPerceptionModule(
    vocabulary_size=64,
    wav2vec_model="m-a-p/MERT-v1-95M",
    use_gpu=False,
    enable_symbolic=True,
    enable_dual_vocabulary=True
    # NOTE: extract_all_frames parameter NOT specified - uses default!
)

print(f"\n📋 Training perception config:")
print(f"   extract_all_frames: {perception_training.extract_all_frames}")
print(f"   (This is what training will use)")

# Test what it extracts
audio, sr = librosa.load('input_audio/General_idea.wav', sr=44100, duration=0.35, offset=10)

# Simulate what training does: call wav2vec_encoder.encode() directly
result = perception_training.wav2vec_encoder.encode(
    audio=audio,
    sr=sr,
    timestamp=0.0
    # NOTE: No return_all_frames parameter - uses default!
)

print(f"\n📊 What training extracts:")
print(f"   Result type: {type(result)}")
if isinstance(result, list):
    print(f"   ✅ List of {len(result)} frames (all-frames mode)")
else:
    print(f"   ❌ Single result (averaged mode)")
    print(f"   This means training will get AVERAGED features, not all frames!")

print("\n" + "=" * 80)
print("⚠️ CRITICAL CHECK:")
print("   If training gets averaged features, it will NOT match the existing vocabulary!")
print("   Existing vocabulary has 49,865 samples (implies all-frames extraction)")
print("   Current training code would get ~991 samples (averaged extraction)")
