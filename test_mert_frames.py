#!/usr/bin/env python3
"""
Test MERT frame-level output to understand the 49,865 sample mystery
"""

import numpy as np
import librosa
from listener.wav2vec_perception import Wav2VecMusicEncoder
import torch

# Load a 350ms segment from General_idea.wav
audio, sr = librosa.load('input_audio/General_idea.wav', sr=44100, duration=0.35, offset=10.0)

print(f"🎵 Audio loaded:")
print(f"   Duration: {len(audio)/sr:.3f}s ({len(audio)} samples at {sr}Hz)")

# Initialize MERT encoder
encoder = Wav2VecMusicEncoder(model_name="m-a-p/MERT-v1-95M", use_gpu=False)
encoder._initialize_model()

# Resample to 24kHz (MERT's expected rate)
audio_24k = librosa.resample(audio, orig_sr=sr, target_sr=24000)
sr = 24000

print(f"   Resampled: {len(audio_24k)} samples at {sr}Hz")

# Normalize
if np.abs(audio_24k).max() > 0:
    audio_24k = audio_24k / np.abs(audio_24k).max()

# Process with MERT
inputs = encoder.processor(
    audio_24k, 
    sampling_rate=sr, 
    return_tensors="pt",
    padding=True
)

print(f"\n🔍 MERT inputs:")
print(f"   Keys: {inputs.keys()}")
for key, value in inputs.items():
    print(f"   {key}: shape={value.shape}, dtype={value.dtype}")

# Extract features (raw, before averaging)
with torch.no_grad():
    outputs = encoder.model(**inputs)

hidden_states = outputs.last_hidden_state

print(f"\n🎯 MERT outputs:")
print(f"   hidden_states shape: {hidden_states.shape}")
print(f"   [batch, time_steps, hidden_dim]")

# Calculate how many frames per segment
time_steps = hidden_states.shape[1]
segment_duration = 0.35  # 350ms
frames_per_segment = time_steps

print(f"\n📊 Frame analysis:")
print(f"   Time steps (frames) per 350ms segment: {frames_per_segment}")
print(f"   If training extracted ALL frames from ALL segments...")

# Calculate for full audio
audio_duration = 347.2  # General_idea.wav duration in seconds
expected_segments = int(audio_duration / 0.35)
total_frames = expected_segments * frames_per_segment

print(f"\n💡 Hypothesis: Frame-level extraction")
print(f"   Expected segments (350ms/no overlap): {expected_segments}")
print(f"   Frames per segment: {frames_per_segment}")
print(f"   Total frames if ALL extracted: {expected_segments} × {frames_per_segment} = {total_frames}")
print(f"   Actual training samples: 49,865")
print(f"   Match? {total_frames} vs 49,865 = {'✅ YES!' if abs(total_frames - 49865) < 1000 else '❌ NO'}")

# Show what averaging does
averaged_features = hidden_states.mean(dim=1).squeeze().cpu().numpy()
print(f"\n🧮 Current encoding (with averaging):")
print(f"   Shape: {averaged_features.shape}")
print(f"   This is what live extraction uses: single 768D vector per segment")

print(f"\n⚠️ If training used frame-level extraction WITHOUT averaging:")
print(f"   Each 350ms segment → {frames_per_segment} separate 768D vectors")
print(f"   Live extraction with averaging → 1 vector per segment")
print(f"   This would explain the diversity collapse!")
