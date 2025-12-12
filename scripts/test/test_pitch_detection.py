#!/usr/bin/env python3
"""
Test Pitch Detection

Tests the pitch detection pipeline used in both:
1. Training (polyphonic_processor.py - librosa multi-pitch)
2. Live (jhs_listener_core.py - YIN algorithm)

Usage:
    python scripts/test/test_pitch_detection.py input_audio/test.wav
    python scripts/test/test_pitch_detection.py  # Uses test tone
"""

import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def generate_test_tones() -> tuple:
    """Generate test audio with known pitches for validation."""
    sr = 44100
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)

    # Test tones with known frequencies
    test_cases = [
        ("A4 (440 Hz)", 440.0, 69),
        ("C4 (261.63 Hz)", 261.63, 60),
        ("E4 (329.63 Hz)", 329.63, 64),
        ("G4 (392.00 Hz)", 392.00, 67),
    ]

    return sr, t, test_cases


def test_yin_pitch_detection():
    """Test the YIN pitch detection algorithm (live system)."""
    print("\n" + "="*60)
    print("TEST 1: YIN PITCH DETECTION (Live System)")
    print("="*60)
    print("This is what the live listener uses for real-time pitch tracking\n")

    from listener.jhs_listener_core import DriftListener

    sr, t, test_cases = generate_test_tones()

    # Create listener with test settings
    listener = DriftListener(
        sr=sr,
        fmin=80.0,
        fmax=1000.0,
        buffer_seconds=0.1
    )

    results = []
    for name, freq, expected_midi in test_cases:
        # Generate pure tone
        audio = 0.5 * np.sin(2 * np.pi * freq * t[:4096]).astype(np.float32)

        # Detect pitch using YIN
        detected_f0 = listener._yin_pitch(audio)

        # Convert to MIDI
        if detected_f0 > 0:
            detected_midi = 69 + 12 * np.log2(detected_f0 / 440.0)
            midi_error = abs(detected_midi - expected_midi)
            freq_error = abs(detected_f0 - freq)
            status = "PASS" if midi_error < 0.5 else "FAIL"
        else:
            detected_midi = 0
            midi_error = float('inf')
            freq_error = float('inf')
            status = "FAIL"

        results.append({
            'name': name,
            'expected_freq': freq,
            'detected_freq': detected_f0,
            'expected_midi': expected_midi,
            'detected_midi': detected_midi,
            'freq_error': freq_error,
            'midi_error': midi_error,
            'status': status
        })

        print(f"{name}:")
        print(f"  Expected: {freq:.2f} Hz (MIDI {expected_midi})")
        print(f"  Detected: {detected_f0:.2f} Hz (MIDI {detected_midi:.1f})")
        print(f"  Error:    {freq_error:.2f} Hz ({midi_error:.2f} semitones)")
        print(f"  Status:   {status}")
        print()

    passed = sum(1 for r in results if r['status'] == 'PASS')
    print(f"YIN Results: {passed}/{len(results)} passed")
    return results


def test_librosa_pitch_detection():
    """Test librosa-based pitch detection (training system)."""
    print("\n" + "="*60)
    print("TEST 2: LIBROSA PITCH DETECTION (Training System)")
    print("="*60)
    print("This is what the training pipeline uses for offline analysis\n")

    import librosa

    sr, t, test_cases = generate_test_tones()

    results = []
    for name, freq, expected_midi in test_cases:
        # Generate 1 second of audio
        audio = 0.5 * np.sin(2 * np.pi * freq * t[:sr]).astype(np.float32)

        # Detect pitch using pyin (probabilistic YIN)
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr
        )

        # Get median of voiced frames
        voiced_f0 = f0[voiced_flag]
        if len(voiced_f0) > 0:
            detected_f0 = np.median(voiced_f0)
            detected_midi = 69 + 12 * np.log2(detected_f0 / 440.0)
            midi_error = abs(detected_midi - expected_midi)
            freq_error = abs(detected_f0 - freq)
            status = "PASS" if midi_error < 0.5 else "FAIL"
            voicing_ratio = len(voiced_f0) / len(f0)
        else:
            detected_f0 = 0
            detected_midi = 0
            midi_error = float('inf')
            freq_error = float('inf')
            status = "FAIL"
            voicing_ratio = 0

        results.append({
            'name': name,
            'expected_freq': freq,
            'detected_freq': detected_f0,
            'expected_midi': expected_midi,
            'detected_midi': detected_midi,
            'freq_error': freq_error,
            'midi_error': midi_error,
            'voicing_ratio': voicing_ratio,
            'status': status
        })

        print(f"{name}:")
        print(f"  Expected: {freq:.2f} Hz (MIDI {expected_midi})")
        print(f"  Detected: {detected_f0:.2f} Hz (MIDI {detected_midi:.1f})")
        print(f"  Error:    {freq_error:.2f} Hz ({midi_error:.2f} semitones)")
        print(f"  Voicing:  {voicing_ratio*100:.1f}% of frames detected as voiced")
        print(f"  Status:   {status}")
        print()

    passed = sum(1 for r in results if r['status'] == 'PASS')
    print(f"Librosa Results: {passed}/{len(results)} passed")
    return results


def test_polyphonic_detection():
    """Test polyphonic (multi-pitch) detection."""
    print("\n" + "="*60)
    print("TEST 3: POLYPHONIC DETECTION (Chord Detection)")
    print("="*60)
    print("Tests multi-pitch detection for chords\n")

    import librosa

    sr = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)

    # Generate C major chord (C4, E4, G4)
    c4 = 261.63  # MIDI 60
    e4 = 329.63  # MIDI 64
    g4 = 392.00  # MIDI 67

    chord_audio = (
        0.3 * np.sin(2 * np.pi * c4 * t) +
        0.3 * np.sin(2 * np.pi * e4 * t) +
        0.3 * np.sin(2 * np.pi * g4 * t)
    ).astype(np.float32)

    print("Test chord: C Major (C4 + E4 + G4)")
    print(f"  Expected frequencies: {c4:.1f}, {e4:.1f}, {g4:.1f} Hz")
    print(f"  Expected MIDI notes:  60, 64, 67")
    print()

    # Use librosa's multi-pitch estimation via CQT
    C = np.abs(librosa.cqt(chord_audio, sr=sr, hop_length=512))

    # Get mean energy across time
    energy = np.mean(C, axis=1)

    # Find peaks (potential pitches)
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(energy, height=np.max(energy) * 0.1)

    # Convert CQT bins to frequencies
    freqs = librosa.cqt_frequencies(C.shape[0], fmin=librosa.note_to_hz('C1'))
    detected_freqs = freqs[peaks[:6]]  # Top 6 peaks

    print(f"Detected frequencies (top peaks):")
    expected = [c4, e4, g4]
    matches = 0
    for f in detected_freqs[:6]:
        midi = 69 + 12 * np.log2(f / 440.0)
        # Check if it matches any expected note
        match = any(abs(f - exp) < 10 for exp in expected)
        if match:
            matches += 1
            print(f"  {f:.1f} Hz (MIDI {midi:.0f}) - MATCH")
        else:
            print(f"  {f:.1f} Hz (MIDI {midi:.0f})")

    print()
    print(f"Matched {matches}/3 chord tones")
    return matches >= 2  # Pass if at least 2/3 detected


def test_with_audio_file(audio_path: Path):
    """Test pitch detection on a real audio file."""
    print("\n" + "="*60)
    print(f"TEST 4: REAL AUDIO FILE")
    print("="*60)
    print(f"Analyzing: {audio_path.name}\n")

    import librosa

    # Load first 10 seconds
    audio, sr = librosa.load(str(audio_path), sr=44100, duration=10.0)

    print(f"Loaded {len(audio)/sr:.1f} seconds of audio at {sr} Hz")

    # Detect pitch with pyin
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        sr=sr
    )

    # Analyze results
    voiced_f0 = f0[voiced_flag]

    if len(voiced_f0) > 0:
        # Statistics
        mean_f0 = np.mean(voiced_f0)
        min_f0 = np.min(voiced_f0)
        max_f0 = np.max(voiced_f0)

        mean_midi = 69 + 12 * np.log2(mean_f0 / 440.0)
        min_midi = 69 + 12 * np.log2(min_f0 / 440.0)
        max_midi = 69 + 12 * np.log2(max_f0 / 440.0)

        voicing_ratio = len(voiced_f0) / len(f0)

        print(f"Pitch Statistics:")
        print(f"  Voiced frames: {voicing_ratio*100:.1f}% ({len(voiced_f0)}/{len(f0)})")
        print(f"  Mean pitch:    {mean_f0:.1f} Hz (MIDI {mean_midi:.0f})")
        print(f"  Pitch range:   {min_f0:.1f} - {max_f0:.1f} Hz")
        print(f"  MIDI range:    {min_midi:.0f} - {max_midi:.0f}")

        # Show pitch distribution
        midi_notes = 69 + 12 * np.log2(voiced_f0 / 440.0)
        midi_rounded = np.round(midi_notes).astype(int)
        unique, counts = np.unique(midi_rounded, return_counts=True)

        print(f"\n  Most common pitches:")
        sorted_idx = np.argsort(counts)[::-1][:5]
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        for idx in sorted_idx:
            midi = unique[idx]
            count = counts[idx]
            note = note_names[midi % 12]
            octave = (midi // 12) - 1
            print(f"    {note}{octave} (MIDI {midi}): {count} frames ({count/len(midi_rounded)*100:.1f}%)")

        return True
    else:
        print("WARNING: No pitched content detected!")
        print("This could mean:")
        print("  - Audio is unpitched (drums, noise)")
        print("  - Audio is too quiet")
        print("  - Pitch detection parameters need adjustment")
        return False


def main():
    print("="*60)
    print("PITCH DETECTION TEST SUITE")
    print("="*60)
    print("Tests both live (YIN) and training (librosa) pitch detection")

    # Run synthetic tests
    yin_results = test_yin_pitch_detection()
    librosa_results = test_librosa_pitch_detection()
    poly_passed = test_polyphonic_detection()

    # Test with real audio if provided
    if len(sys.argv) > 1:
        audio_path = Path(sys.argv[1])
        if audio_path.exists():
            test_with_audio_file(audio_path)
        else:
            print(f"\nERROR: File not found: {audio_path}")
    else:
        # Try to find a test file
        test_files = list(project_root.glob("input_audio/*.wav"))
        if test_files:
            print(f"\nTip: Run with an audio file for real-world testing:")
            print(f"  python {sys.argv[0]} {test_files[0]}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    yin_passed = sum(1 for r in yin_results if r['status'] == 'PASS')
    lib_passed = sum(1 for r in librosa_results if r['status'] == 'PASS')

    print(f"YIN (live):      {yin_passed}/{len(yin_results)} tests passed")
    print(f"Librosa (train): {lib_passed}/{len(librosa_results)} tests passed")
    print(f"Polyphonic:      {'PASS' if poly_passed else 'FAIL'}")

    all_passed = (
        yin_passed == len(yin_results) and
        lib_passed == len(librosa_results) and
        poly_passed
    )

    if all_passed:
        print("\nAll pitch detection tests PASSED")
    else:
        print("\nSome tests failed - check output above")

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
