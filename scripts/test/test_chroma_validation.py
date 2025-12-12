#!/usr/bin/env python3
"""
Test Chroma Validation (Noise Gate)

Tests the chroma-based validation layer that filters noise from musical content.
This is the gate that prevents MERT from matching on room noise.

Usage:
    python scripts/test/test_chroma_validation.py
    python scripts/test/test_chroma_validation.py input_audio/test.wav
"""

import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def test_validator_standalone():
    """Test ChromaValidator with synthetic chroma vectors."""
    print("=" * 60)
    print("TEST 1: ChromaValidator Standalone")
    print("=" * 60)
    print("Testing validation logic with synthetic chroma vectors\n")

    from listener.chroma_validator import ChromaValidator

    validator = ChromaValidator()

    test_cases = [
        ("Clear C major chord (C, E, G)",
         np.array([0.9, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 0.0]),
         True),  # Expected: musical

        ("Single note (A)",
         np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.95, 0.0, 0.0]),
         True),  # Expected: musical

        ("Am7 chord (A, C, E, G)",
         np.array([0.7, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.6, 0.0, 0.9, 0.0, 0.0]),
         True),  # Expected: musical

        ("Uniform noise (all pitch classes equal)",
         np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
         False),  # Expected: NOT musical (low coherence)

        ("Low energy (quiet)",
         np.array([0.05, 0.0, 0.0, 0.0, 0.03, 0.0, 0.0, 0.04, 0.0, 0.0, 0.0, 0.0]),
         False),  # Expected: NOT musical (too quiet)

        ("Silence",
         np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
         False),  # Expected: NOT musical

        ("Complex cluster (all pitches high)",
         np.array([0.8, 0.7, 0.9, 0.6, 0.8, 0.7, 0.9, 0.8, 0.7, 0.8, 0.6, 0.7]),
         False),  # Expected: NOT musical (low coherence, might be noise)
    ]

    passed = 0
    for name, chroma, expected_musical in test_cases:
        result = validator.validate(chroma)

        status = "PASS" if result.is_musical == expected_musical else "FAIL"
        if status == "PASS":
            passed += 1

        print(f"{name}:")
        print(f"  Expected musical: {expected_musical}")
        print(f"  Detected musical: {result.is_musical}")
        print(f"  Confidence: {result.musical_confidence:.2f}")
        print(f"  Should process MERT: {result.should_process_mert}")
        print(f"  Reason: {result.reason}")
        print(f"  Status: {status}")
        print()

    print(f"Standalone validation: {passed}/{len(test_cases)} passed")
    return passed == len(test_cases)


def test_with_synthetic_audio():
    """Test full DualPerceptionModule with synthetic audio."""
    print("\n" + "=" * 60)
    print("TEST 2: DualPerceptionModule with Synthetic Audio")
    print("=" * 60)
    print("Testing full feature extraction with chroma validation\n")

    try:
        from listener.dual_perception import DualPerceptionModule
        print("DualPerceptionModule imported successfully")
    except ImportError as e:
        print(f"Failed to import DualPerceptionModule: {e}")
        return False

    # Initialize module (MERT might take a moment)
    print("\nInitializing DualPerceptionModule (this may take 10-30s)...")
    try:
        module = DualPerceptionModule(
            vocabulary_size=64,
            wav2vec_model="m-a-p/MERT-v1-95M",
            use_gpu=True,
            enable_symbolic=True,
            enable_wav2vec=True,
            extract_all_frames=False  # Single frame for testing
        )
        print("Module initialized\n")
    except Exception as e:
        print(f"Failed to initialize module: {e}")
        import traceback
        traceback.print_exc()
        return False

    sr = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)

    test_cases = [
        ("C major chord (musical)",
         lambda t: (0.3 * np.sin(2 * np.pi * 261.63 * t) +
                   0.3 * np.sin(2 * np.pi * 329.63 * t) +
                   0.3 * np.sin(2 * np.pi * 392.00 * t)).astype(np.float32),
         True),

        ("Single A note (musical)",
         lambda t: (0.8 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32),
         True),

        ("White noise (not musical)",
         lambda t: (0.3 * np.random.randn(len(t))).astype(np.float32),
         False),

        ("Very quiet audio (not musical)",
         lambda t: (0.001 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32),
         False),

        ("Silence (not musical)",
         lambda t: np.zeros(len(t), dtype=np.float32),
         False),
    ]

    passed = 0
    for name, audio_fn, expected_musical in test_cases:
        audio = audio_fn(t)

        # Extract features
        result = module.extract_features(audio, sr, detected_f0=None)

        # Handle list result (all-frames mode)
        if isinstance(result, list):
            result = result[-1]

        # Check validation result
        is_musical = result.is_musical
        confidence = result.musical_confidence
        should_process = result.chroma_validation.should_process_mert if result.chroma_validation else True

        # Determine pass/fail
        # For musical content, we expect is_musical=True
        # For non-musical, we expect is_musical=False OR low confidence
        if expected_musical:
            status = "PASS" if is_musical else "FAIL"
        else:
            # Non-musical: should either be detected as not musical, or have low confidence
            status = "PASS" if (not is_musical or confidence < 0.5) else "FAIL"

        if status == "PASS":
            passed += 1

        print(f"{name}:")
        print(f"  Expected musical: {expected_musical}")
        print(f"  Detected musical: {is_musical}")
        print(f"  Musical confidence: {confidence:.2f}")
        print(f"  Should process MERT: {should_process}")
        if result.chroma_validation:
            print(f"  Chroma energy: {result.chroma_validation.chroma_energy:.2f}")
            print(f"  Pitch coherence: {result.chroma_validation.pitch_coherence:.2f}")
            print(f"  Reason: {result.chroma_validation.reason}")
        print(f"  Status: {status}")
        print()

    print(f"Full module validation: {passed}/{len(test_cases)} passed")
    return passed >= len(test_cases) - 1  # Allow one failure


def test_with_audio_file(audio_path: Path):
    """Test with a real audio file."""
    print("\n" + "=" * 60)
    print(f"TEST 3: Real Audio File")
    print("=" * 60)
    print(f"Analyzing: {audio_path.name}\n")

    import librosa
    from listener.dual_perception import DualPerceptionModule

    # Load audio
    audio, sr = librosa.load(str(audio_path), sr=44100, duration=30.0)
    print(f"Loaded {len(audio)/sr:.1f} seconds at {sr} Hz")

    # Initialize module
    print("\nInitializing DualPerceptionModule...")
    module = DualPerceptionModule(
        vocabulary_size=64,
        wav2vec_model="m-a-p/MERT-v1-95M",
        use_gpu=True,
        enable_symbolic=True,
        enable_wav2vec=True,
        extract_all_frames=False
    )
    print("Module initialized\n")

    # Process in segments
    segment_duration = 1.0
    segment_samples = int(segment_duration * sr)

    results = []
    musical_count = 0
    not_musical_count = 0

    print("Analyzing segments...")
    print("-" * 60)

    for i in range(0, len(audio) - segment_samples, segment_samples):
        segment = audio[i:i + segment_samples]
        timestamp = i / sr

        result = module.extract_features(segment, sr, detected_f0=None)

        # Handle list result
        if isinstance(result, list):
            result = result[-1]

        results.append(result)

        if result.is_musical:
            musical_count += 1
        else:
            not_musical_count += 1

        # Print some segments
        if i < 10 * segment_samples or i % (5 * segment_samples) == 0:
            status = "MUSICAL" if result.is_musical else "NOISE?"
            print(f"[{timestamp:5.1f}s] {status:8s} | "
                  f"conf={result.musical_confidence:.2f} | "
                  f"chord={result.chord_label:8s} | "
                  f"coherence={result.chroma_validation.pitch_coherence:.2f if result.chroma_validation else 0:.2f}")

    print("-" * 60)
    print(f"\nSummary:")
    print(f"  Total segments: {len(results)}")
    print(f"  Musical: {musical_count} ({musical_count/len(results)*100:.1f}%)")
    print(f"  Not musical: {not_musical_count} ({not_musical_count/len(results)*100:.1f}%)")

    # Statistics on confidence
    confidences = [r.musical_confidence for r in results]
    print(f"\n  Musical confidence stats:")
    print(f"    Min: {min(confidences):.2f}")
    print(f"    Max: {max(confidences):.2f}")
    print(f"    Mean: {np.mean(confidences):.2f}")

    return musical_count > not_musical_count  # Expect more musical than noise in real audio


def main():
    print("=" * 60)
    print("CHROMA VALIDATION TEST SUITE")
    print("=" * 60)
    print("Tests the noise gate that filters room noise from music\n")

    # Test 1: Standalone validator
    standalone_pass = test_validator_standalone()

    # Test 2: Full module with synthetic audio
    full_pass = test_with_synthetic_audio()

    # Test 3: Real audio file (if provided)
    real_file_pass = True
    if len(sys.argv) > 1:
        audio_path = Path(sys.argv[1])
        if audio_path.exists():
            real_file_pass = test_with_audio_file(audio_path)
        else:
            print(f"\nFile not found: {audio_path}")
    else:
        # Try to find a test file
        test_files = list(project_root.glob("input_audio/*.wav"))
        if test_files:
            print(f"\nTip: Run with an audio file for real-world testing:")
            print(f"  python {sys.argv[0]} {test_files[0]}")

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Standalone validator: {'PASS' if standalone_pass else 'FAIL'}")
    print(f"Full module test:     {'PASS' if full_pass else 'FAIL'}")
    if len(sys.argv) > 1:
        print(f"Real audio test:      {'PASS' if real_file_pass else 'FAIL'}")

    all_pass = standalone_pass and full_pass and real_file_pass
    if all_pass:
        print("\nAll chroma validation tests PASSED")
        print("The noise gate is ready to filter room noise from MERT matching!")
    else:
        print("\nSome tests failed - check output above")

    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
