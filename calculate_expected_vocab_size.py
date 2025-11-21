#!/usr/bin/env python3
"""
Calculate expected vocabulary size with all-frames extraction
"""

# MERT outputs ~74 frames/second
frames_per_second = 74

# Training uses 350ms segments with no overlap
segment_duration_s = 0.35
frames_per_segment = int(frames_per_second * segment_duration_s)

# General_idea.wav duration
audio_duration_s = 347.2

# Number of segments
num_segments = int(audio_duration_s / segment_duration_s)

# Dual vocabulary: harmonic + percussive
num_sources = 2

# Total vocabulary size
total_features = num_segments * frames_per_segment * num_sources

print("=" * 80)
print("📊 Expected Vocabulary Size Calculation")
print("=" * 80)
print(f"\n🎵 Audio: General_idea.wav")
print(f"   Duration: {audio_duration_s}s")
print(f"\n🔧 Segmentation:")
print(f"   Segment duration: {segment_duration_s * 1000}ms")
print(f"   Overlap: 0% (no overlap)")
print(f"   Number of segments: {num_segments}")
print(f"\n🧠 MERT Frame Extraction:")
print(f"   Frames per second: {frames_per_second}")
print(f"   Frames per {segment_duration_s * 1000}ms segment: {frames_per_segment}")
print(f"\n🎭 Dual Vocabulary:")
print(f"   Sources: {num_sources} (harmonic + percussive)")
print(f"   Features per source: {num_segments} segments × {frames_per_segment} frames = {num_segments * frames_per_segment:,}")
print(f"\n📈 Total Vocabulary Size:")
print(f"   {num_segments} segments × {frames_per_segment} frames × {num_sources} sources = {total_features:,} features")
print(f"\n🔍 Comparison with Existing:")
existing_vocab_size = 49865
print(f"   Existing vocabulary: {existing_vocab_size:,} samples")
print(f"   Expected with all-frames: {total_features:,} samples")
print(f"   Difference: {abs(existing_vocab_size - total_features):,} samples")
ratio = existing_vocab_size / total_features
print(f"   Ratio: {ratio:.2f}x")

if abs(existing_vocab_size - total_features) < 3000:
    print(f"\n   ✅ Match! Existing vocabulary was created with all-frames extraction")
else:
    print(f"\n   ⚠️ Mismatch - investigating...")
    # Maybe different segment count due to processing limit?
    if existing_vocab_size > total_features:
        print(f"   Existing has MORE samples - possibly different audio duration or overlap")
    else:
        print(f"   Existing has FEWER samples - possibly max_events limit during training")

print("=" * 80)
