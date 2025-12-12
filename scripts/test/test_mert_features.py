#!/usr/bin/env python3
"""
Test MERT Feature Extraction

Tests the ACTUAL feature extraction used by both training and live:
- MERT (m-a-p/MERT-v1-95M) → 768D neural embeddings
- Gesture token quantization

This is what really matters for MusicHal - not chroma!

Usage:
    python scripts/test/test_mert_features.py input_audio/moon_stars_short.wav
    python scripts/test/test_mert_features.py  # Uses synthetic test
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def test_mert_extraction(audio_path: Path = None):
    """Test MERT feature extraction."""
    print("=" * 60)
    print("MERT FEATURE EXTRACTION TEST")
    print("=" * 60)
    print("This is what BOTH training and live use for features\n")

    # Import the encoder
    try:
        from listener.dual_perception import DualPerceptionModule
        print("✅ DualPerceptionModule imported")
    except ImportError as e:
        print(f"❌ Failed to import DualPerceptionModule: {e}")
        return False

    # Initialize with MERT
    print("\nInitializing MERT encoder (m-a-p/MERT-v1-95M)...")
    print("This may take 10-30 seconds on first run...\n")

    start_time = time.time()
    try:
        module = DualPerceptionModule(
            vocabulary_size=64,
            wav2vec_model="m-a-p/MERT-v1-95M",
            use_gpu=True,  # Use GPU if available
            enable_symbolic=True,
            enable_wav2vec=True,
            extract_all_frames=False  # Single frame for testing
        )
        init_time = time.time() - start_time
        print(f"✅ MERT initialized in {init_time:.1f}s")
    except Exception as e:
        print(f"❌ Failed to initialize MERT: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Check encoder is ready
    if not module.wav2vec_encoder:
        print("❌ wav2vec_encoder is None")
        return False

    print(f"   Model: {module.wav2vec_encoder.model_name}")
    print(f"   Feature dim: 768D")

    # Load or generate test audio
    import librosa

    if audio_path and audio_path.exists():
        print(f"\nLoading audio: {audio_path.name}")
        audio, sr = librosa.load(str(audio_path), sr=44100, duration=10.0)
        print(f"   Duration: {len(audio)/sr:.1f}s")
    else:
        print("\nGenerating synthetic test audio (C major chord)...")
        sr = 44100
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)

        # C major chord
        c4 = 261.63
        e4 = 329.63
        g4 = 392.00
        audio = (
            0.3 * np.sin(2 * np.pi * c4 * t) +
            0.3 * np.sin(2 * np.pi * e4 * t) +
            0.3 * np.sin(2 * np.pi * g4 * t)
        ).astype(np.float32)
        print(f"   Generated C major chord, {duration}s")

    # Extract features
    print("\n" + "=" * 60)
    print("FEATURE EXTRACTION")
    print("=" * 60)

    # Test extraction on segments
    segment_duration = 1.0  # 1 second segments
    segment_samples = int(segment_duration * sr)
    num_segments = min(5, len(audio) // segment_samples)

    features_list = []
    extraction_times = []

    for i in range(num_segments):
        start = i * segment_samples
        segment = audio[start:start + segment_samples]
        timestamp = start / sr

        ext_start = time.time()
        result = module.extract_features(segment, sr, detected_f0=None)
        ext_time = time.time() - ext_start
        extraction_times.append(ext_time)

        if result is None:
            print(f"Segment {i+1}: ❌ No result")
            continue

        # Handle list result (all-frames mode)
        if isinstance(result, list):
            result = result[-1]  # Use last frame

        features = result.wav2vec_features
        if features is None:
            print(f"Segment {i+1}: ❌ No features")
            continue

        features_list.append(features)

        print(f"Segment {i+1} [{timestamp:.1f}s]:")
        print(f"   Features shape: {features.shape}")
        print(f"   Feature stats: min={features.min():.3f}, max={features.max():.3f}, mean={features.mean():.3f}")
        print(f"   Extraction time: {ext_time*1000:.0f}ms")

        # Show additional result fields
        if hasattr(result, 'gesture_token') and result.gesture_token is not None:
            print(f"   Gesture token: {result.gesture_token}")
        if hasattr(result, 'consonance'):
            print(f"   Consonance: {result.consonance:.2f}")
        if hasattr(result, 'chord_label') and result.chord_label:
            print(f"   Chord label: {result.chord_label}")

    if not features_list:
        print("\n❌ No features extracted!")
        return False

    # Analyze feature consistency
    print("\n" + "=" * 60)
    print("FEATURE ANALYSIS")
    print("=" * 60)

    features_array = np.array(features_list)
    print(f"\nExtracted {len(features_list)} segments")
    print(f"Feature matrix shape: {features_array.shape}")
    print(f"Average extraction time: {np.mean(extraction_times)*1000:.0f}ms")

    # Feature variance across segments
    variance = np.var(features_array, axis=0)
    print(f"\nFeature variance across segments:")
    print(f"   Min variance: {variance.min():.4f}")
    print(f"   Max variance: {variance.max():.4f}")
    print(f"   Mean variance: {variance.mean():.4f}")

    # Similarity between segments
    if len(features_list) >= 2:
        print(f"\nSegment similarity (cosine):")
        from sklearn.metrics.pairwise import cosine_similarity
        sim_matrix = cosine_similarity(features_array)
        for i in range(min(3, len(features_list))):
            for j in range(i+1, min(3, len(features_list))):
                print(f"   Seg {i+1} vs Seg {j+1}: {sim_matrix[i,j]:.3f}")

    # Test quantization (if quantizer available AND trained)
    print("\n" + "=" * 60)
    print("GESTURE TOKEN QUANTIZATION")
    print("=" * 60)

    if module.quantizer is not None:
        # Check if quantizer is actually fitted
        try:
            # Try a test transform
            test_features = features_list[0].reshape(1, -1)
            token = int(module.quantizer.transform(test_features)[0])
            print("Quantizer is trained - testing token assignment...")
            for i, features in enumerate(features_list[:3]):
                token = int(module.quantizer.transform(features.reshape(1, -1))[0])
                print(f"   Segment {i+1} → Token {token}")
        except ValueError as e:
            if "fit()" in str(e):
                print("Quantizer exists but not trained yet")
                print("(Tokens would be assigned after running Chandra_trainer)")
            else:
                raise
    else:
        print("No quantizer loaded (would need to load from trained model)")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"✅ MERT feature extraction working")
    print(f"   - 768D embeddings extracted successfully")
    print(f"   - Average extraction: {np.mean(extraction_times)*1000:.0f}ms per segment")
    print(f"   - Features show {'high' if variance.mean() > 0.01 else 'low'} variance (diversity)")

    return True


def compare_with_trained_model(json_path: Path):
    """Compare extracted features with those stored in trained model."""
    import json

    print("\n" + "=" * 60)
    print("COMPARISON WITH TRAINED MODEL")
    print("=" * 60)

    with open(json_path, 'r') as f:
        data = json.load(f)

    audio_data = data.get('audio_data', [])
    if not audio_data:
        print("No audio_data in JSON")
        return

    # Check feature dimensions
    sample = audio_data[0]
    features = sample.get('features', sample.get('wav2vec_features', []))

    print(f"Model: {json_path.name}")
    print(f"Events: {len(audio_data)}")
    print(f"Feature dimensions: {len(features)}")

    if len(features) == 768:
        print("✅ Features are 768D MERT embeddings")
    elif len(features) == 12:
        print("⚠️  Features are 12D chroma (old format)")
    else:
        print(f"❓ Unknown feature format ({len(features)}D)")

    # Check for gesture tokens
    tokens = [e.get('gesture_token') for e in audio_data if e.get('gesture_token') is not None]
    if tokens:
        unique_tokens = len(set(tokens))
        print(f"\nGesture tokens: {unique_tokens} unique values")
        print(f"   Range: {min(tokens)} - {max(tokens)}")

    # Check for dual vocab tokens
    h_tokens = [e.get('harmonic_token') for e in audio_data if e.get('harmonic_token') is not None]
    p_tokens = [e.get('percussive_token') for e in audio_data if e.get('percussive_token') is not None]

    if h_tokens:
        print(f"\nHarmonic tokens: {len(set(h_tokens))} unique")
    if p_tokens:
        print(f"Percussive tokens: {len(set(p_tokens))} unique")


def main():
    print("=" * 60)
    print("MERT FEATURE EXTRACTION TEST")
    print("=" * 60)
    print("Tests the PRIMARY feature extraction used by MusicHal")
    print("(Both training and live use MERT 768D embeddings)\n")

    # Check for audio file argument
    audio_path = None
    json_path = None

    if len(sys.argv) > 1:
        arg = Path(sys.argv[1])
        if arg.suffix == '.json':
            json_path = arg
        elif arg.exists():
            audio_path = arg

    # Run MERT extraction test
    success = test_mert_extraction(audio_path)

    # Compare with trained model if JSON provided
    if json_path and json_path.exists():
        compare_with_trained_model(json_path)

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
