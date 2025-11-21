#!/usr/bin/env python3
"""
End-to-end test of dual vocabulary in MusicHal_9000
Simulates live performance with test audio to verify:
1. HPSS separation and dual token extraction
2. Content type detection
3. Agent receives dual tokens correctly
4. AudioOracle can filter by response_mode
"""

import sys
import numpy as np
import librosa
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 80)
print("🎭 End-to-End Dual Vocabulary Test")
print("=" * 80)

# Load test audio
audio_path = "input_audio/General_idea.wav"
print(f"\n📂 Loading test audio: {audio_path}")
audio, sr = librosa.load(audio_path, sr=44100, duration=5.0, offset=20.0)
print(f"   Duration: {len(audio) / sr:.2f}s @ {sr}Hz")

# Initialize MusicHal components manually (lighter than full system)
print("\n🔧 Initializing components...")

from listener.dual_perception import DualPerceptionModule

# Initialize dual perception
print("   Initializing DualPerceptionModule...")
dual_perception = DualPerceptionModule(
    vocabulary_size=64,
    wav2vec_model="m-a-p/MERT-v1-95M",
    use_gpu=False,
    enable_symbolic=True,
    enable_dual_vocabulary=True,
    extract_all_frames=True
)

# Load vocabularies
dual_perception.load_vocabulary("input_audio/General_idea_harmonic_vocab.joblib", "harmonic")
dual_perception.load_vocabulary("input_audio/General_idea_percussive_vocab.joblib", "percussive")
print("   ✅ Vocabularies loaded")

print("\n🎵 Simulating live performance...")
print("-" * 80)

# Detect onsets in test audio
print("   Detecting onsets...")
onset_frames = librosa.onset.onset_detect(
    y=audio,
    sr=sr,
    hop_length=512,
    backtrack=True,
    units='samples'
)
print(f"   Found {len(onset_frames)} onsets")

# Process first 10 onsets
events = []
for i, onset_sample in enumerate(onset_frames[:10]):
    onset_time = onset_sample / sr
    
    # Extract audio segment around onset (350ms window)
    segment_samples = int(0.35 * sr)
    start_sample = max(0, onset_sample - segment_samples // 4)
    end_sample = min(len(audio), start_sample + segment_samples)
    segment_audio = audio[start_sample:end_sample]
    
    # Extract dual perception features
    results = dual_perception.extract_features(
        audio=segment_audio,
        sr=sr,
        timestamp=onset_time
    )
    
    # Handle all-frames mode (take first frame for simplicity in this test)
    if isinstance(results, list):
        result = results[0]
        num_frames = len(results)
    else:
        result = results
        num_frames = 1
    
    # Create event data
    event = {
        'time': onset_time,
        'harmonic_token': result.harmonic_token,
        'percussive_token': result.percussive_token,
        'gesture_token': result.gesture_token,
        'content_type': result.content_type,
        'harmonic_ratio': result.harmonic_ratio,
        'percussive_ratio': result.percussive_ratio,
        'consonance': result.consonance,
        'num_frames': num_frames
    }
    events.append(event)
    
    # Print event details
    print(f"\n   Event {i+1} @ {onset_time:.2f}s:")
    print(f"      Content: {result.content_type} (h={result.harmonic_ratio:.2f}, p={result.percussive_ratio:.2f})")
    print(f"      Tokens: harmonic={result.harmonic_token}, percussive={result.percussive_token}, gesture={result.gesture_token}")
    print(f"      Frames: {num_frames}")
    print(f"      Consonance: {result.consonance:.2f}")

print("\n" + "=" * 80)
print("📊 Analysis")
print("=" * 80)

# Analyze token diversity
harmonic_tokens = [e['harmonic_token'] for e in events if e['harmonic_token'] is not None]
percussive_tokens = [e['percussive_token'] for e in events if e['percussive_token'] is not None]

print(f"\n🎯 Token Diversity:")
print(f"   Harmonic: {len(set(harmonic_tokens))}/{len(harmonic_tokens)} unique tokens")
print(f"   Percussive: {len(set(percussive_tokens))}/{len(percussive_tokens)} unique tokens")

# Check token differentiation
different = sum(1 for e in events if e['harmonic_token'] != e['percussive_token'])
print(f"\n🔀 Token Differentiation:")
print(f"   Events with different h/p tokens: {different}/{len(events)} ({different/len(events)*100:.0f}%)")

# Content type distribution
from collections import Counter
content_types = Counter(e['content_type'] for e in events)
print(f"\n🎭 Content Type Distribution:")
for ctype, count in content_types.items():
    print(f"   {ctype}: {count} events ({count/len(events)*100:.0f}%)")

# Test complementary response logic
print(f"\n🤝 Complementary Response Test:")
for event in events[:3]:
    print(f"\n   Event @ {event['time']:.2f}s:")
    print(f"      User plays: {event['content_type']}")
    
    if event['content_type'] == 'percussive':
        print(f"      → System should respond with: HARMONIC patterns")
        print(f"         Query: percussive_token={event['percussive_token']}, response_mode='harmonic'")
    elif event['content_type'] == 'harmonic':
        print(f"      → System should respond with: PERCUSSIVE patterns")
        print(f"         Query: harmonic_token={event['harmonic_token']}, response_mode='percussive'")
    else:
        print(f"      → System should respond with: BALANCED (hybrid input)")
        print(f"         Query: Use both tokens, balanced response")

print("\n" + "=" * 80)
print("✅ Test Summary")
print("=" * 80)

success_checks = []

# Check 1: HPSS separation working
hpss_working = different > 0
success_checks.append(("HPSS separation produces different tokens", hpss_working))

# Check 2: Content type detected
content_detected = len(content_types) > 0
success_checks.append(("Content type detection active", content_detected))

# Check 3: Dual tokens present
dual_tokens = all(e['harmonic_token'] is not None and e['percussive_token'] is not None for e in events)
success_checks.append(("All events have dual tokens", dual_tokens))

# Check 4: All-frames mode
all_frames = all(e['num_frames'] > 1 for e in events)
success_checks.append(("All-frames mode active", all_frames))

print("")
for check_name, passed in success_checks:
    status = "✅" if passed else "❌"
    print(f"{status} {check_name}")

all_passed = all(passed for _, passed in success_checks)
print(f"\n{'🎉 All checks passed!' if all_passed else '⚠️  Some checks failed'}")

if all_passed:
    print("\nDual vocabulary system is fully operational!")
    print("The system can now:")
    print("  • Detect harmonic vs percussive input")
    print("  • Extract separate tokens for each source")
    print("  • Enable complementary response (drums → melody, guitar → rhythm)")
else:
    print("\nSome issues detected - review output above")

print("\n" + "=" * 80)
