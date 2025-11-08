#!/usr/bin/env python3
"""
Unit Tests for Interval-Based Harmonic Translation
===================================================

Tests IntervalExtractor and HarmonicTranslator components
to ensure correct behavior before integration.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agent.interval_extractor import IntervalExtractor
from agent.harmonic_translator import HarmonicTranslator


def test_interval_extraction():
    """Test IntervalExtractor with known MIDI sequence"""
    print("="*80)
    print("TEST 1: Interval Extraction")
    print("="*80)
    
    # Create mock audio_frames
    audio_frames = {
        0: {'audio_data': {'midi': 67}},  # G4
        1: {'audio_data': {'midi': 67}},  # G4 (sustain)
        2: {'audio_data': {'midi': 65}},  # F4 (interval -2)
        3: {'audio_data': {'midi': 69}},  # A4 (interval +4)
        4: {'audio_data': {'midi': 69}},  # A4 (sustain)
    }
    
    frame_ids = [0, 1, 2, 3, 4]
    
    extractor = IntervalExtractor()
    intervals = extractor.extract_intervals(frame_ids, audio_frames)
    
    expected_intervals = [0, -2, +4, 0]
    
    print(f"Input MIDI sequence: [67, 67, 65, 69, 69]")
    print(f"Expected intervals:  {expected_intervals}")
    print(f"Extracted intervals: {intervals}")
    
    assert intervals == expected_intervals, f"Mismatch! Got {intervals}"
    print("✅ PASS: Intervals extracted correctly\n")


def test_harmonic_translation_c_major():
    """Test HarmonicTranslator with C major context"""
    print("="*80)
    print("TEST 2: Harmonic Translation - C Major")
    print("="*80)
    
    # Mock scale constraint function (C major scale)
    def mock_scale_constraint(note, scale_degrees):
        """Snap to C major scale"""
        pitch_class = note % 12
        
        # Find nearest scale degree
        min_distance = float('inf')
        nearest_degree = pitch_class
        
        for degree in scale_degrees:
            distance = min(abs(pitch_class - degree), 
                          abs(pitch_class - degree + 12), 
                          abs(pitch_class - degree - 12))
            if distance < min_distance:
                min_distance = distance
                nearest_degree = degree
        
        adjustment = nearest_degree - pitch_class
        if adjustment > 6:
            adjustment -= 12
        elif adjustment < -6:
            adjustment += 12
        
        return note + adjustment
    
    translator = HarmonicTranslator(scale_constraint_func=mock_scale_constraint)
    
    # Test intervals from Token 27 pattern
    intervals = [0, -2, +4, 0]
    
    harmonic_context = {
        'current_chord': 'C',
        'current_key': 'C',
        'scale_degrees': [0, 2, 4, 5, 7, 9, 11]  # C major
    }
    
    midi_notes = translator.translate_intervals_to_midi(
        intervals=intervals,
        harmonic_context=harmonic_context,
        voice_type="melodic",
        apply_constraints=True
    )
    
    print(f"Intervals:       {intervals}")
    print(f"Harmonic context: C major (root = C4 = MIDI 60)")
    print(f"Output MIDI:     {midi_notes}")
    
    # Verify all notes are in C major scale
    c_major_pcs = [0, 2, 4, 5, 7, 9, 11]
    for note in midi_notes:
        pc = note % 12
        assert pc in c_major_pcs, f"Note {note} (PC {pc}) not in C major!"
    
    print(f"✅ PASS: All notes in C major scale\n")


def test_harmonic_translation_f_sharp_minor():
    """Test HarmonicTranslator with F# minor context"""
    print("="*80)
    print("TEST 3: Harmonic Translation - F# Minor")
    print("="*80)
    
    # Mock scale constraint for F# minor
    def mock_scale_constraint(note, scale_degrees):
        """Snap to F# minor scale"""
        pitch_class = note % 12
        
        min_distance = float('inf')
        nearest_degree = pitch_class
        
        for degree in scale_degrees:
            distance = min(abs(pitch_class - degree), 
                          abs(pitch_class - degree + 12), 
                          abs(pitch_class - degree - 12))
            if distance < min_distance:
                min_distance = distance
                nearest_degree = degree
        
        adjustment = nearest_degree - pitch_class
        if adjustment > 6:
            adjustment -= 12
        elif adjustment < -6:
            adjustment += 12
        
        return note + adjustment
    
    translator = HarmonicTranslator(scale_constraint_func=mock_scale_constraint)
    
    # Same intervals as C major test
    intervals = [0, -2, +4, 0]
    
    harmonic_context = {
        'current_chord': 'F#m',
        'current_key': 'F#_minor',
        'scale_degrees': [6, 8, 9, 11, 1, 2, 4]  # F# minor (F#, G#, A, B, C#, D, E)
    }
    
    midi_notes = translator.translate_intervals_to_midi(
        intervals=intervals,
        harmonic_context=harmonic_context,
        voice_type="melodic",
        apply_constraints=True
    )
    
    print(f"Intervals:       {intervals}")
    print(f"Harmonic context: F# minor (root = F#4 = MIDI 66)")
    print(f"Output MIDI:     {midi_notes}")
    
    # Verify all notes are in F# minor scale
    f_sharp_minor_pcs = [6, 8, 9, 11, 1, 2, 4]  # F#, G#, A, B, C#, D, E
    for note in midi_notes:
        pc = note % 12
        assert pc in f_sharp_minor_pcs, f"Note {note} (PC {pc}) not in F# minor!"
    
    print(f"✅ PASS: All notes in F# minor scale\n")


def test_root_note_parsing():
    """Test chord root parsing for various chord names"""
    print("="*80)
    print("TEST 4: Chord Root Parsing")
    print("="*80)
    
    # Mock scale constraint (not used in this test)
    def mock_scale_constraint(note, scale_degrees):
        return note
    
    translator = HarmonicTranslator(scale_constraint_func=mock_scale_constraint)
    
    test_cases = [
        ('C', 'C'),
        ('Dm', 'D'),
        ('F#m', 'F#'),
        ('Bb7', 'Bb'),
        ('C#maj7', 'C#'),
        ('Gmaj', 'G'),
        ('Abm', 'Ab'),
    ]
    
    print("Chord Name → Parsed Root:")
    for chord_name, expected_root in test_cases:
        parsed_root = translator._parse_root_from_chord(chord_name)
        print(f"  {chord_name:10s} → {parsed_root:3s} (expected {expected_root})")
        assert parsed_root == expected_root, f"Mismatch for {chord_name}!"
    
    print("✅ PASS: All chord roots parsed correctly\n")


def test_consonant_intervals_preservation():
    """Test that learned consonant intervals (±5, ±7, ±12) work correctly"""
    print("="*80)
    print("TEST 5: Consonant Interval Preservation")
    print("="*80)
    
    def mock_scale_constraint(note, scale_degrees):
        return note  # No constraint for this test
    
    translator = HarmonicTranslator(scale_constraint_func=mock_scale_constraint)
    
    # Test learned intervals from analysis: ±7 (perfect 5th), ±5 (perfect 4th)
    intervals = [+7, -5, +12, -7]  # Perfect 5th up, 4th down, octave up, 5th down
    
    harmonic_context = {
        'current_chord': 'C',
        'current_key': 'C',
        'scale_degrees': [0, 2, 4, 5, 7, 9, 11]
    }
    
    midi_notes = translator.translate_intervals_to_midi(
        intervals=intervals,
        harmonic_context=harmonic_context,
        voice_type="melodic",
        apply_constraints=False  # Test raw intervals
    )
    
    print(f"Input intervals:  {intervals}")
    print(f"Start note (C4):  60")
    print(f"Output MIDI:      {midi_notes}")
    print(f"Expected:         [60, 67, 62, 74, 67]")
    
    expected = [60, 67, 62, 74, 67]
    assert midi_notes == expected, f"Mismatch! Got {midi_notes}"
    
    print("✅ PASS: Consonant intervals preserved correctly\n")


def main():
    """Run all unit tests"""
    print("\n" + "="*80)
    print("INTERVAL-BASED HARMONIC TRANSLATION - UNIT TESTS")
    print("="*80 + "\n")
    
    try:
        test_interval_extraction()
        test_harmonic_translation_c_major()
        test_harmonic_translation_f_sharp_minor()
        test_root_note_parsing()
        test_consonant_intervals_preservation()
        
        print("="*80)
        print("ALL TESTS PASSED ✅")
        print("="*80)
        print("\nComponents are ready for integration into PhraseGenerator.\n")
        return True
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
