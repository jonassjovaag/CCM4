#!/usr/bin/env python3
"""
Test how many frames MERT outputs for different audio segment sizes
"""

import numpy as np
import librosa
from listener.wav2vec_perception import Wav2VecMusicEncoder
import torch

# Initialize MERT encoder
encoder = Wav2VecMusicEncoder(model_name="m-a-p/MERT-v1-95M", use_gpu=False)
encoder._initialize_model()

# Load audio
audio, sr = librosa.load('input_audio/General_idea.wav', sr=44100, duration=60)

# Test different segment sizes
segment_sizes_ms = [100, 200, 350, 500, 1000]

print("🔍 Testing MERT frame output for different segment sizes:")
print("=" * 80)

for seg_ms in segment_sizes_ms:
    seg_sec = seg_ms / 1000.0
    seg_samples = int(sr * seg_sec)
    
    # Extract one segment
    segment = audio[:seg_samples]
    
    # Resample to 24kHz
    segment_24k = librosa.resample(segment, orig_sr=sr, target_sr=24000)
    sr_24k = 24000
    
    # Normalize
    if np.abs(segment_24k).max() > 0:
        segment_24k = segment_24k / np.abs(segment_24k).max()
    
    # Process with MERT
    inputs = encoder.processor(
        segment_24k, 
        sampling_rate=sr_24k, 
        return_tensors="pt",
        padding=True
    )
    
    # Extract features
    with torch.no_grad():
        outputs = encoder.model(**inputs)
    
    hidden_states = outputs.last_hidden_state
    time_steps = hidden_states.shape[1]
    
    print(f"{seg_ms:4d}ms → {time_steps:3d} frames ({time_steps / seg_sec:.1f} frames/sec)")

print("\n" + "=" * 80)
print("💡 Calculation for General_idea.wav:")
audio_duration = 347.2  # seconds
print(f"Audio duration: {audio_duration}s")

# Using 350ms segments
seg_ms = 350
seg_sec = seg_ms / 1000.0
# Assume ~26 frames per 350ms (based on earlier test)
frames_per_350ms = 26

# With NO overlap
segments_no_overlap = int(audio_duration / seg_sec)
total_frames_no_overlap = segments_no_overlap * frames_per_350ms

print(f"\n350ms segments, NO overlap:")
print(f"  Segments: {segments_no_overlap}")
print(f"  Frames per segment: {frames_per_350ms}")
print(f"  Total frames: {segments_no_overlap} × {frames_per_350ms} = {total_frames_no_overlap}")
print(f"  Actual training samples: 49,865")
print(f"  Ratio: {49865 / total_frames_no_overlap:.2f}x")

# What overlap would give us 49,865?
required_segments = 49865 / frames_per_350ms
print(f"\n🔍 To get 49,865 frames:")
print(f"  Required segments: {49865} / {frames_per_350ms} = {required_segments:.0f}")
print(f"  Current segments (no overlap): {segments_no_overlap}")
print(f"  Ratio: {required_segments / segments_no_overlap:.2f}x more segments needed")

# Calculate hop size
hop_ms = (audio_duration * 1000) / required_segments
overlap_ratio = 1.0 - (hop_ms / seg_ms)
print(f"  Implied hop size: {hop_ms:.1f}ms")
print(f"  Implied overlap ratio: {overlap_ratio:.1%}")
