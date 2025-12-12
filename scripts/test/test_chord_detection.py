#!/usr/bin/env python3
"""
Test Chord Detection

Tests polyphonic chord detection using chroma features.
This is what the training pipeline uses for harmonic analysis.

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
    """Analyze chord progression in audio file."""
    print(f"\nAnalyzing: {audio_path.name}")
    print("=" * 60)

    # Load audio
    audio, sr = librosa.load(str(audio_path), sr=44100)
    duration = len(audio) / sr

    print(f"Duration: {duration:.1f} seconds")
    print(f"Analyzing in {segment_duration}s segments...\n")

    # Analyze in segments
    hop_length = 512
    segment_samples = int(segment_duration * sr)

    chords_detected = []

    for start in range(0, len(audio) - segment_samples, segment_samples):
        segment = audio[start:start + segment_samples]
        timestamp = start / sr

        # Extract chroma using CQT (better for harmony)
        chroma = librosa.feature.chroma_cqt(y=segment, sr=sr, hop_length=hop_length)

        # Average chroma across time
        chroma_avg = np.mean(chroma, axis=1)

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
    """Show full chroma analysis with beat-synced display."""
    print("\n" + "=" * 60)
    print("DETAILED CHROMA ANALYSIS")
    print("=" * 60)

    audio, sr = librosa.load(str(audio_path), sr=44100)

    # Beat-synced chroma
    tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
    print(f"Detected tempo: {tempo:.0f} BPM")

    # Get chroma synced to beats
    chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)
    chroma_sync = librosa.util.sync(chroma, beats, aggregate=np.median)

    print(f"\nChroma per beat (first 16 beats):")
    print("-" * 60)

    beat_times = librosa.frames_to_time(beats, sr=sr)

    for i, beat_time in enumerate(beat_times[:16]):
        if i < chroma_sync.shape[1]:
            chroma_vec = chroma_sync[:, i]
            chord, conf, notes = detect_chord_from_chroma(chroma_vec)

            # Show top 3 pitch classes
            top_pcs = np.argsort(chroma_vec)[::-1][:3]
            top_notes = [f"{NOTE_NAMES[pc]}:{chroma_vec[pc]:.2f}" for pc in top_pcs]

            print(f"Beat {i+1:2d} [{beat_time:5.2f}s]: {chord:8s} | {', '.join(top_notes)}")

    # Overall pitch class distribution
    print("\n" + "=" * 60)
    print("OVERALL PITCH CLASS DISTRIBUTION")
    print("=" * 60)

    chroma_mean = np.mean(chroma, axis=1)
    chroma_norm = chroma_mean / np.max(chroma_mean)

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
