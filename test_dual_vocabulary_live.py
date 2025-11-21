#!/usr/bin/env python3
"""
Test dual vocabulary extraction in live performance mode
Verifies HPSS separation and distinct harmonic/percussive token extraction
"""

import numpy as np
import librosa
from listener.dual_perception import DualPerceptionModule

print("=" * 80)
print("🧪 Testing Dual Vocabulary Live Extraction")
print("=" * 80)

# Load test audio (1 second)
audio_path = "input_audio/General_idea.wav"
print(f"\n📂 Loading audio: {audio_path}")
audio, sr = librosa.load(audio_path, sr=None, duration=1.0, offset=10.0)
print(f"   Duration: {len(audio) / sr:.2f}s @ {sr}Hz")

# Initialize dual perception module
print("\n🔄 Initializing DualPerceptionModule with dual vocabulary...")
dual_perception = DualPerceptionModule(
    vocabulary_size=64,
    wav2vec_model="m-a-p/MERT-v1-95M",
    use_gpu=False,
    enable_symbolic=True,
    enable_dual_vocabulary=True,  # Enable dual vocabulary mode
    extract_all_frames=True  # Extract all MERT frames
)

# Load vocabularies
harmonic_vocab = "input_audio/General_idea_harmonic_vocab.joblib"
percussive_vocab = "input_audio/General_idea_percussive_vocab.joblib"

print(f"\n📚 Loading vocabularies...")
print(f"   Harmonic: {harmonic_vocab}")
print(f"   Percussive: {percussive_vocab}")

dual_perception.load_vocabulary(harmonic_vocab, vocabulary_type="harmonic")
dual_perception.load_vocabulary(percussive_vocab, vocabulary_type="percussive")

print(f"   ✅ Vocabularies loaded")

# Extract features (should perform HPSS and dual extraction)
print(f"\n🎵 Extracting features with HPSS separation...")
results = dual_perception.extract_features(
    audio=audio,
    sr=sr,
    timestamp=0.0
)

# Check results
print(f"\n📊 Results:")
if isinstance(results, list):
    print(f"   ✅ All-frames mode: {len(results)} frames extracted")
    
    # Analyze first few frames
    print(f"\n   Analyzing first 5 frames:")
    for i, result in enumerate(results[:5]):
        print(f"\n   Frame {i}:")
        print(f"      Timestamp: {result.timestamp:.3f}s")
        print(f"      Harmonic token: {result.harmonic_token}")
        print(f"      Percussive token: {result.percussive_token}")
        print(f"      Gesture token (legacy): {result.gesture_token}")
        print(f"      Content type: {result.content_type}")
        print(f"      Harmonic ratio: {result.harmonic_ratio:.2f}")
        print(f"      Percussive ratio: {result.percussive_ratio:.2f}")
    
    # Check token diversity
    harmonic_tokens = [r.harmonic_token for r in results if r.harmonic_token is not None]
    percussive_tokens = [r.percussive_token for r in results if r.percussive_token is not None]
    
    unique_harmonic = len(set(harmonic_tokens))
    unique_percussive = len(set(percussive_tokens))
    
    print(f"\n   Token Diversity:")
    print(f"      Harmonic tokens: {len(harmonic_tokens)} total, {unique_harmonic} unique")
    print(f"      Percussive tokens: {len(percussive_tokens)} total, {unique_percussive} unique")
    
    # Check if tokens differ
    different_tokens = sum(1 for r in results if r.harmonic_token != r.percussive_token)
    print(f"\n   Token Differentiation:")
    print(f"      Frames with different h/p tokens: {different_tokens}/{len(results)} ({different_tokens/len(results)*100:.1f}%)")
    
    if different_tokens > 0:
        print(f"      ✅ HPSS separation is working - tokens differ!")
    else:
        print(f"      ⚠️  All tokens identical - may indicate issue with HPSS or quantization")
    
    # Content type distribution
    content_types = [r.content_type for r in results]
    from collections import Counter
    type_counts = Counter(content_types)
    print(f"\n   Content Type Distribution:")
    for ctype, count in type_counts.items():
        print(f"      {ctype}: {count} frames ({count/len(results)*100:.1f}%)")
    
else:
    print(f"   ⚠️  Single frame mode (expected list)")
    print(f"      Harmonic token: {results.harmonic_token}")
    print(f"      Percussive token: {results.percussive_token}")
    print(f"      Content type: {results.content_type}")

print("\n" + "=" * 80)
print("✅ Test complete!")
print("=" * 80)

# Summary
print(f"\n📝 Summary:")
print(f"   - HPSS separation: {'✅ Active' if isinstance(results, list) and different_tokens > 0 else '❌ Not working'}")
print(f"   - Dual token extraction: {'✅ Working' if isinstance(results, list) else '❌ Single mode'}")
print(f"   - Content type detection: {'✅ Working' if isinstance(results, list) and len(type_counts) > 1 else '⚠️  Single type only'}")
print(f"   - All-frames mode: {'✅ Active' if isinstance(results, list) else '❌ Disabled'}")
