#!/usr/bin/env python3
"""
Test Chord Detection

Tests polyphonic chord detection using the ACTUAL HarmonicAwareChromaExtractor
that the training pipeline uses (not plain librosa).

Usage:
    python scripts/test/test_chord_detection.py input_audio/moon_stars_short.wav
"""

import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import librosa
from listener.harmonic_chroma import HarmonicAwareChromaExtractor

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Common chord templates (relative to root)
CHORD_TEMPLATES = {
    'maj': [0, 4, 7],           # Major triad
    'min': [0, 3, 7],           # Minor triad
    'dim': [0, 3, 6],           # Diminished
    'aug': [0, 4, 8],           # Augmented
    'maj7': [0, 4, 7, 11],      # Major 7th
    'min7': [0, 3, 7, 10],      # Minor 7th
    '7': [0, 4, 7, 10],         # Dominant 7th
    'sus4': [0, 5, 7],          # Suspended 4th
    'sus2': [0, 2, 7],          # Suspended 2nd
}


def detect_chord_from_chroma(chroma_vector: np.ndarray, threshold: float = 0.3) -> tuple:
    """
    Detect chord from chroma vector using template matching.

    Returns: (chord_name, confidence, active_notes)
    """
    # Normalize chroma
    chroma_norm = chroma_vector / (np.max(chroma_vector) + 1e-10)

    # Find active pitch classes
    active_pcs = np.where(chroma_norm > threshold)[0]
    active_notes = [NOTE_NAMES[pc] for pc in active_pcs]

    if len(active_pcs) < 2:
        return ("N/C", 0.0, active_notes)

    best_chord = None
    best_score = 0

    # Try each root note
    for root in range(12):
        # Try each chord type
        for chord_type, intervals in CHORD_TEMPLATES.items():
            # Build expected pitch classes for this chord
            expected_pcs = [(root + interval) % 12 for interval in intervals]

            # Calculate match score
            score = 0
            for pc in expected_pcs:
                score += chroma_norm[pc]

            # Penalize extra notes not in chord
            for pc in active_pcs:
                if pc not in expected_pcs:
                    score -= chroma_norm[pc] * 0.3

            # Normalize by chord size
            score /= len(intervals)

            if score > best_score:
                best_score = score
                root_name = NOTE_NAMES[root]
                if chord_type == 'maj':
                    best_chord = root_name
                elif chord_type == 'min':
                    best_chord = f"{root_name}m"
                else:
                    best_chord = f"{root_name}{chord_type}"

    return (best_chord or "N/C", best_score, active_notes)


def analyze_audio_chords(audio_path: Path, segment_duration: float = 2.0):
    """Analyze chord progression in audio file using HarmonicAwareChromaExtractor."""
    print(f"\nAnalyzing: {audio_path.name}")
    print("=" * 60)

    # Load audio
    audio, sr = librosa.load(str(audio_path), sr=44100)
    duration = len(audio) / sr

    print(f"Duration: {duration:.1f} seconds")
    print(f"Analyzing in {segment_duration}s segments...")
    print("Using HarmonicAwareChromaExtractor (same as training pipeline)\n")

    # Create the SAME chroma extractor used in training
    chroma_extractor = HarmonicAwareChromaExtractor()

    segment_samples = int(segment_duration * sr)
    chords_detected = []

    for start in range(0, len(audio) - segment_samples, segment_samples):
        segment = audio[start:start + segment_samples]
        timestamp = start / sr

        # Extract chroma using HarmonicAwareChromaExtractor (NOT plain librosa)
        # This applies harmonic weighting to suppress overtones
        chroma_avg = chroma_extractor.extract(segment, sr, use_temporal=True, live_mode=False)

        # Detect chord
        chord, confidence, active_notes = detect_chord_from_chroma(chroma_avg)

        chords_detected.append({
            'time': timestamp,
            'chord': chord,
            'confidence': confidence,
            'notes': active_notes
        })

        print(f"[{timestamp:5.1f}s - {timestamp + segment_duration:5.1f}s] "
              f"{chord:8s} (conf: {confidence:.0%}) "
              f"Notes: {', '.join(active_notes)}")

    return chords_detected


def analyze_full_chroma(audio_path: Path):
    """Show full chroma analysis with beat-synced display using HarmonicAwareChromaExtractor."""
    print("\n" + "=" * 60)
    print("DETAILED CHROMA ANALYSIS (HarmonicAwareChromaExtractor)")
    print("=" * 60)

    audio, sr = librosa.load(str(audio_path), sr=44100)

    # Create the SAME chroma extractor used in training
    chroma_extractor = HarmonicAwareChromaExtractor()

    # Beat detection
    tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
    tempo_val = float(tempo[0]) if hasattr(tempo, '__len__') else float(tempo)
    print(f"Detected tempo: {tempo_val:.0f} BPM")

    # Get beat times
    beat_times = librosa.frames_to_time(beats, sr=sr)

    print(f"\nChroma per beat (first 16 beats):")
    print("-" * 60)

    # Extract chroma at each beat using HarmonicAwareChromaExtractor
    hop_samples = int(0.5 * sr)  # ~0.5 seconds around each beat

    beat_chromas = []
    for i, beat_time in enumerate(beat_times[:16]):
        beat_sample = int(beat_time * sr)
        start = max(0, beat_sample - hop_samples // 2)
        end = min(len(audio), beat_sample + hop_samples // 2)
        segment = audio[start:end]

        if len(segment) > 2048:
            chroma_vec = chroma_extractor.extract(segment, sr, use_temporal=True, live_mode=False)
            beat_chromas.append(chroma_vec)

            chord, conf, notes = detect_chord_from_chroma(chroma_vec)

            # Show top 3 pitch classes
            top_pcs = np.argsort(chroma_vec)[::-1][:3]
            top_notes = [f"{NOTE_NAMES[pc]}:{chroma_vec[pc]:.2f}" for pc in top_pcs]

            print(f"Beat {i+1:2d} [{beat_time:5.2f}s]: {chord:8s} | {', '.join(top_notes)}")

    # Overall pitch class distribution (using HarmonicAwareChromaExtractor on full audio)
    print("\n" + "=" * 60)
    print("OVERALL PITCH CLASS DISTRIBUTION")
    print("=" * 60)

    # Compute chroma for entire file in chunks and average
    chunk_size = sr * 2  # 2 second chunks
    all_chromas = []
    for start in range(0, len(audio) - chunk_size, chunk_size):
        chunk = audio[start:start + chunk_size]
        chroma = chroma_extractor.extract(chunk, sr, use_temporal=True, live_mode=False)
        all_chromas.append(chroma)

    chroma_mean = np.mean(all_chromas, axis=0) if all_chromas else np.zeros(12)
    chroma_norm = chroma_mean / (np.max(chroma_mean) + 1e-10)

    # Sort by energy
    sorted_pcs = np.argsort(chroma_norm)[::-1]

    print("\nRanked by energy:")
    for pc in sorted_pcs:
        bar = "#" * int(chroma_norm[pc] * 30)
        print(f"  {NOTE_NAMES[pc]:2s}: {bar} ({chroma_norm[pc]:.2f})")

    # Identify likely key
    print("\n" + "=" * 60)
    print("KEY ESTIMATION")
    print("=" * 60)

    # Simple key detection using Krumhansl-Schmuckler
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

    best_corr = -1
    best_key = None

    for shift in range(12):
        # Try major
        shifted = np.roll(major_profile, shift)
        corr = np.corrcoef(chroma_norm, shifted)[0, 1]
        if corr > best_corr:
            best_corr = corr
            best_key = f"{NOTE_NAMES[shift]} major"

        # Try minor
        shifted = np.roll(minor_profile, shift)
        corr = np.corrcoef(chroma_norm, shifted)[0, 1]
        if corr > best_corr:
            best_corr = corr
            best_key = f"{NOTE_NAMES[shift]} minor"

    print(f"Estimated key: {best_key} (correlation: {best_corr:.2f})")


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_chord_detection.py <audio_file>")
        print("\nExample:")
        print("  python test_chord_detection.py input_audio/moon_stars_short.wav")
        return 1

    audio_path = Path(sys.argv[1])
    if not audio_path.exists():
        print(f"ERROR: File not found: {audio_path}")
        return 1

    print("=" * 60)
    print("CHORD DETECTION TEST")
    print("=" * 60)
    print("Uses chroma features for polyphonic chord analysis")

    # Segment-based chord detection
    chords = analyze_audio_chords(audio_path, segment_duration=2.0)

    # Full chroma analysis
    analyze_full_chroma(audio_path)

    # Summary
    print("\n" + "=" * 60)
    print("CHORD PROGRESSION SUMMARY")
    print("=" * 60)

    unique_chords = []
    for c in chords:
        if c['chord'] not in unique_chords and c['chord'] != 'N/C':
            unique_chords.append(c['chord'])

    print(f"Detected progression: {' -> '.join(unique_chords)}")
    print(f"\nYou mentioned: Am -> Bm -> C -> Dm")
    print("Compare with detected chords above.")

    return 0


if __name__ == '__main__':
    sys.exit(main())
