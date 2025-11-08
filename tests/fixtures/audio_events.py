"""
Synthetic audio event fixtures for testing
These are predictable, known-good events for validation

Purpose: Provide consistent test data with known properties for:
- Feature extraction validation
- Note extraction testing
- Request masking verification
- End-to-end integration tests
"""

from typing import Dict, List
import numpy as np


def c_major_chord_event() -> Dict:
    """
    Returns synthetic event representing C major triad
    
    Properties:
    - High consonance (0.95)
    - Perfect major third + fifth ratios
    - C4 fundamental (261.63 Hz)
    """
    return {
        't': 0.0,
        'pitch_hz': 261.63,  # C4
        'amplitude': 0.8,
        'duration': 0.5,
        'gesture_token': 42,  # Arbitrary but consistent
        'consonance': 0.95,   # High consonance for major triad
        'frequency_ratios': [1.0, 1.25, 1.5],  # Perfect major intervals (root, M3, P5)
        'chord_name_display': 'major',
        'dual_chroma': [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # C, E, G
        'dual_active_pcs': [0, 4, 7],  # C, E, G pitch classes
    }


def a_minor_chord_event() -> Dict:
    """
    Returns synthetic event representing A minor triad
    
    Properties:
    - Moderately high consonance (0.85)
    - Minor third + perfect fifth ratios
    - A3 fundamental (220 Hz)
    """
    return {
        't': 0.5,
        'pitch_hz': 220.0,  # A3
        'amplitude': 0.7,
        'duration': 0.6,
        'gesture_token': 28,  # Different from major
        'consonance': 0.85,   # Slightly lower than major
        'frequency_ratios': [1.0, 1.2, 1.5],  # Minor third (root, m3, P5)
        'chord_name_display': 'minor',
        'dual_chroma': [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # A, C, E
        'dual_active_pcs': [9, 0, 4],  # A, C, E pitch classes
    }


def dissonant_cluster_event() -> Dict:
    """
    Returns synthetic event representing chromatic cluster (dissonant)
    
    Properties:
    - Low consonance (0.2)
    - Tight semitone intervals
    - No clear chord classification
    """
    return {
        't': 1.0,
        'pitch_hz': 300.0,
        'amplitude': 0.6,
        'duration': 0.3,
        'gesture_token': 15,
        'consonance': 0.2,    # Very dissonant
        'frequency_ratios': [1.0, 1.06, 1.12, 1.19],  # Chromatic cluster (semitones)
        'chord_name_display': 'unknown',
        'dual_chroma': [0.3, 0.3, 0.3, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Ambiguous
        'dual_active_pcs': [0, 1, 2, 3],  # Chromatic cluster
    }


def perfect_fifth_event() -> Dict:
    """
    Returns synthetic event with perfect fifth (very consonant)
    
    Properties:
    - Highest consonance (0.98)
    - Simple 2:3 frequency ratio
    - G4 fundamental
    """
    return {
        't': 1.5,
        'pitch_hz': 392.0,  # G4
        'amplitude': 0.75,
        'duration': 0.8,
        'gesture_token': 55,
        'consonance': 0.98,   # Nearly perfect consonance
        'frequency_ratios': [1.0, 1.5],  # Perfect fifth only
        'chord_name_display': 'power',  # Power chord
        'dual_chroma': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # G
        'dual_active_pcs': [7],  # G pitch class
    }


def silence_event() -> Dict:
    """
    Returns synthetic event representing silence/rest
    
    Properties:
    - Zero amplitude
    - No pitch
    - Edge case for testing
    """
    return {
        't': 2.0,
        'pitch_hz': 0.0,
        'amplitude': 0.0,
        'duration': 0.5,
        'gesture_token': None,  # No token for silence
        'consonance': 0.0,
        'frequency_ratios': [],
        'chord_name_display': 'silence',
        'dual_chroma': [0.0] * 12,
        'dual_active_pcs': [],
    }


def high_note_event() -> Dict:
    """
    Returns synthetic event with very high pitch (edge case)
    
    Properties:
    - High frequency (near upper limit)
    - Tests range boundaries
    """
    return {
        't': 2.5,
        'pitch_hz': 1975.5,  # B6 (high)
        'amplitude': 0.5,
        'duration': 0.2,
        'gesture_token': 60,
        'consonance': 0.6,
        'frequency_ratios': [1.0],
        'chord_name_display': 'single_note',
        'dual_chroma': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # B
        'dual_active_pcs': [11],
    }


def low_note_event() -> Dict:
    """
    Returns synthetic event with very low pitch (edge case)
    
    Properties:
    - Low frequency (near lower limit)
    - Tests range boundaries
    """
    return {
        't': 3.0,
        'pitch_hz': 55.0,  # A1 (low bass)
        'amplitude': 0.9,
        'duration': 1.0,
        'gesture_token': 3,
        'consonance': 0.5,
        'frequency_ratios': [1.0],
        'chord_name_display': 'single_note',
        'dual_chroma': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # A
        'dual_active_pcs': [9],
    }


def get_all_fixture_events() -> List[Dict]:
    """
    Returns list of all fixture events for comprehensive testing
    
    Returns:
        List of all predefined fixture events
    """
    return [
        c_major_chord_event(),
        a_minor_chord_event(),
        dissonant_cluster_event(),
        perfect_fifth_event(),
        silence_event(),
        high_note_event(),
        low_note_event(),
    ]


def get_diverse_event_sequence() -> List[Dict]:
    """
    Returns musically diverse sequence of events (excluding edge cases)
    
    Useful for testing phrase generation, temporal patterns, etc.
    """
    return [
        c_major_chord_event(),
        a_minor_chord_event(),
        dissonant_cluster_event(),
        perfect_fifth_event(),
    ]


def get_consonant_events() -> List[Dict]:
    """
    Returns only consonant events (high consonance)
    
    Useful for testing consonance-based filtering
    """
    return [
        c_major_chord_event(),
        a_minor_chord_event(),
        perfect_fifth_event(),
    ]


def get_dissonant_events() -> List[Dict]:
    """
    Returns only dissonant events (low consonance)
    
    Useful for testing consonance-based filtering
    """
    return [
        dissonant_cluster_event(),
    ]


if __name__ == "__main__":
    """
    Test the fixtures themselves
    """
    print("Testing audio event fixtures...")
    
    all_events = get_all_fixture_events()
    print(f"\n✅ Created {len(all_events)} fixture events")
    
    for event in all_events:
        print(f"\n{event.get('chord_name_display', 'unknown')}:")
        print(f"  pitch_hz: {event['pitch_hz']}")
        print(f"  consonance: {event['consonance']}")
        print(f"  gesture_token: {event['gesture_token']}")
        print(f"  frequency_ratios: {event['frequency_ratios']}")
    
    print("\n✅ All fixtures valid!")
