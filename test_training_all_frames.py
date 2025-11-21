#!/usr/bin/env python3
"""
Test that training extraction now uses all-frames mode
Simulates the exact code path from Chandra_trainer.py
"""

import numpy as np
import librosa
from listener.wav2vec_perception import Wav2VecMusicEncoder

print("=" * 80)
print("🧪 Testing Training All-Frames Extraction")
print("=" * 80)

# Load test audio
audio_path = "input_audio/General_idea.wav"
print(f"\n📂 Loading audio: {audio_path}")
audio, sr = librosa.load(audio_path, sr=None, duration=1.0)  # 1 second test
print(f"   Duration: {len(audio) / sr:.2f}s @ {sr}Hz")

# Initialize encoder
print("\n🔄 Initializing MERT encoder...")
encoder = Wav2VecMusicEncoder(
    model_name="m-a-p/MERT-v1-95M"
)
print(f"   Model: {encoder.model_name}")

# Extract features with return_all_frames=True (NEW training behavior)
print("\n📊 Extracting features with return_all_frames=True...")
results = encoder.encode(
    audio=audio,
    sr=sr,
    timestamp=0.0,
    return_all_frames=True  # ✅ This is what training now uses
)

# Check result type
print(f"\n🔍 Result type: {type(results)}")
if isinstance(results, list):
    print(f"   ✅ List of {len(results)} frame results")
    print(f"   Each result type: {type(results[0])}")
    print(f"   Each feature shape: {results[0].features.shape}")
    print(f"   Total features collected: {sum(1 for r in results)}")
    
    # Simulate what training does now
    features_list = []
    for frame_result in results:
        features_list.append(frame_result.features)
    
    print(f"\n📈 Simulated training extraction:")
    print(f"   Features collected: {len(features_list)}")
    print(f"   Each feature dimension: {features_list[0].shape}")
    print(f"   ✅ Training will now get {len(features_list)} features per 1s segment")
    
    # Calculate expected for full training
    segment_duration = 0.35  # 350ms
    audio_duration = 347.2  # General_idea.wav duration
    num_segments = int(audio_duration / segment_duration)
    frames_per_segment = len(results) / (1.0 / segment_duration)  # Scale from 1s to 350ms
    expected_frames_per_segment = int(frames_per_segment * segment_duration)
    
    # Calculate expected vocabulary size
    expected_total = num_segments * expected_frames_per_segment * 2  # × 2 for harmonic+percussive
    
    print(f"\n🎯 Expected full training extraction:")
    print(f"   Audio duration: {audio_duration}s")
    print(f"   Segment duration: {segment_duration * 1000}ms")
    print(f"   Number of segments: {num_segments}")
    print(f"   Frames per segment: ~{expected_frames_per_segment}")
    print(f"   Sources: 2 (harmonic + percussive)")
    print(f"   Expected vocabulary size: {expected_total:,} samples")
    print(f"   Existing vocabulary: 49,865 samples")
    print(f"   Match: {'✅ YES' if abs(expected_total - 49865) < 5000 else '❌ NO'}")
    
else:
    print(f"   ❌ Single result (averaged mode)")
    print(f"   This means return_all_frames=True is NOT working!")

print("\n" + "=" * 80)
print("✅ Test complete!")
print("=" * 80)
