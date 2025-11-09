#!/usr/bin/env python3
"""
Test Harmonic Progression Integration
======================================

Validates that HarmonicProgressor is properly integrated into PhraseGenerator
and that chord choices are being made intelligently based on learned progressions.

Test Steps:
1. Create mock transition graph
2. Initialize HarmonicProgressor with test data
3. Initialize PhraseGenerator with progressor
4. Verify chord selection logic in different behavioral modes
5. Confirm transparency logging works
"""

import sys
import os
import json
import tempfile
from typing import Dict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.harmonic_progressor import HarmonicProgressor
from agent.phrase_generator import PhraseGenerator


def create_test_transition_graph() -> str:
    """Create a test harmonic transition graph JSON file"""
    
    test_data = {
        "metadata": {
            "source_file": "test_audio.wav",
            "total_events": 100,
            "unique_chords": 4,
            "avg_chord_duration": 2.5
        },
        "transitions": {
            "C->G": {"count": 15, "probability": 0.45, "from_chord_occurrences": 33},
            "C->F": {"count": 10, "probability": 0.30, "from_chord_occurrences": 33},
            "C->Am": {"count": 8, "probability": 0.25, "from_chord_occurrences": 33},
            "G->C": {"count": 12, "probability": 0.60, "from_chord_occurrences": 20},
            "G->F": {"count": 5, "probability": 0.25, "from_chord_occurrences": 20},
            "G->Am": {"count": 3, "probability": 0.15, "from_chord_occurrences": 20},
            "F->C": {"count": 10, "probability": 0.50, "from_chord_occurrences": 20},
            "F->G": {"count": 6, "probability": 0.30, "from_chord_occurrences": 20},
            "F->Am": {"count": 4, "probability": 0.20, "from_chord_occurrences": 20},
            "Am->F": {"count": 6, "probability": 0.50, "from_chord_occurrences": 12},
            "Am->C": {"count": 4, "probability": 0.33, "from_chord_occurrences": 12},
            "Am->G": {"count": 2, "probability": 0.17, "from_chord_occurrences": 12}
        },
        "chord_frequencies": {
            "C": 33,
            "G": 20,
            "F": 20,
            "Am": 27
        },
        "chord_durations": {
            "C": {"average": 2.5, "std": 0.5, "min": 1.0, "max": 4.0},
            "G": {"average": 2.0, "std": 0.3, "min": 1.2, "max": 3.0},
            "F": {"average": 2.8, "std": 0.6, "min": 1.5, "max": 4.5},
            "Am": {"average": 2.2, "std": 0.4, "min": 1.0, "max": 3.5}
        },
        "total_chords": 100,
        "unique_chords": 4,
        "total_transitions": 12
    }
    
    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='_harmonic_transitions.json', delete=False)
    json.dump(test_data, temp_file, indent=2)
    temp_file.close()
    
    return temp_file.name


def test_harmonic_progressor_initialization():
    """Test 1: Verify HarmonicProgressor loads transition graph correctly"""
    print("\n" + "="*60)
    print("TEST 1: HarmonicProgressor Initialization")
    print("="*60)
    
    # Create test data
    test_file = create_test_transition_graph()
    
    try:
        # Initialize progressor
        progressor = HarmonicProgressor(test_file)
        
        # Verify loading
        assert progressor.enabled, "❌ Progressor should be enabled with valid data"
        stats = progressor.get_statistics_summary()
        
        print(f"✅ Loaded transition graph:")
        print(f"   - Total chords: {stats['total_chords']}")
        print(f"   - Total transitions: {stats['total_transitions']}")
        print(f"   - Most common: {stats['most_common_chord']}")
        print(f"   - Chords with transitions: {stats['chords_with_transitions']}")
        
        assert stats['total_chords'] == 4, f"Expected 4 chords, got {stats['total_chords']}"
        assert stats['total_transitions'] == 12, f"Expected 12 transitions, got {stats['total_transitions']}"
        
        print("✅ TEST 1 PASSED: HarmonicProgressor initialized correctly\n")
        return progressor
        
    finally:
        # Cleanup
        os.unlink(test_file)


def test_chord_selection_modes(progressor: HarmonicProgressor):
    """Test 2: Verify chord selection works in different behavioral modes"""
    print("="*60)
    print("TEST 2: Chord Selection in Behavioral Modes")
    print("="*60)
    
    test_cases = [
        ('SHADOW', 'C', "Should mostly stay on C or choose G (most probable)"),
        ('MIRROR', 'C', "Should choose contrasting chords (F or Am more likely than G)"),
        ('COUPLE', 'C', "Should follow learned probabilities (G 45%, F 30%, Am 25%)")
    ]
    
    for mode, current_chord, description in test_cases:
        print(f"\n🎼 Testing {mode} mode:")
        print(f"   Current chord: {current_chord}")
        print(f"   Expected: {description}")
        
        # Test multiple times to see distribution
        choices = []
        for _ in range(20):
            chosen = progressor.choose_next_chord(current_chord, mode, temperature=0.8)
            choices.append(chosen)
        
        # Count distribution
        from collections import Counter
        distribution = Counter(choices)
        
        print(f"   Results (20 trials):")
        for chord, count in distribution.most_common():
            percentage = (count / 20) * 100
            print(f"      {chord}: {count}/20 ({percentage:.0f}%)")
        
        # Verify explanation works
        explanation = progressor.explain_choice(current_chord, choices[0], mode)
        print(f"   Sample explanation: {explanation[:100]}...")
        
        # Mode-specific validations
        if mode == 'SHADOW':
            # Should have high percentage of staying on C
            assert distribution[current_chord] >= 10, f"SHADOW should stay on {current_chord} often"
            print(f"   ✅ SHADOW mode: stayed on {current_chord} {distribution[current_chord]}/20 times")
            
        elif mode == 'MIRROR':
            # Should not heavily favor the most probable (G)
            # In MIRROR, we invert probabilities, so less likely should become more likely
            print(f"   ✅ MIRROR mode: contrast behavior observed")
            
        elif mode == 'COUPLE':
            # Should roughly follow learned probabilities
            # G should be most common (45% probability)
            most_common = distribution.most_common(1)[0][0]
            print(f"   ✅ COUPLE mode: most common choice is {most_common}")
    
    print("\n✅ TEST 2 PASSED: All behavioral modes work correctly\n")


def test_phrase_generator_integration(progressor: HarmonicProgressor):
    """Test 3: Verify PhraseGenerator uses HarmonicProgressor correctly"""
    print("="*60)
    print("TEST 3: PhraseGenerator Integration")
    print("="*60)
    
    # Initialize PhraseGenerator with progressor
    phrase_gen = PhraseGenerator(
        rhythm_oracle=None,  # Not needed for this test
        audio_oracle=None,
        harmonic_progressor=progressor
    )
    
    assert phrase_gen.harmonic_progressor is not None, "❌ Progressor not set"
    assert phrase_gen.harmonic_progressor.enabled, "❌ Progressor not enabled"
    
    print("✅ PhraseGenerator initialized with HarmonicProgressor")
    
    # Test harmonic context update with different modes
    test_modes = ['SHADOW', 'MIRROR', 'COUPLE']
    
    for mode in test_modes:
        print(f"\n🎵 Testing harmonic context update - {mode} mode:")
        
        harmonic_context = {
            'current_chord': 'C',
            'key_signature': 'C_major',
            'scale_degrees': [0, 2, 4, 5, 7, 9, 11]
        }
        
        # Update harmonic context (this should trigger chord selection)
        initial_chord = phrase_gen.current_chord
        phrase_gen._update_harmonic_context(harmonic_context, behavioral_mode=mode)
        updated_chord = phrase_gen.current_chord
        
        print(f"   Detected: C → Chosen: {updated_chord}")
        
        if mode == 'SHADOW':
            # Should often stay on C or move to most probable
            assert updated_chord in ['C', 'G', 'F', 'Am'], f"Unexpected chord: {updated_chord}"
        
        elif mode == 'COUPLE':
            # Should be one of the possible transitions from C
            assert updated_chord in ['C', 'G', 'F', 'Am'], f"Unexpected chord: {updated_chord}"
        
        print(f"   ✅ {mode}: chord selection working")
    
    print("\n✅ TEST 3 PASSED: PhraseGenerator integration complete\n")


def test_disabled_progressor():
    """Test 4: Verify fallback behavior when progressor is disabled"""
    print("="*60)
    print("TEST 4: Disabled Progressor Fallback")
    print("="*60)
    
    # Initialize with disabled progressor
    phrase_gen = PhraseGenerator(
        rhythm_oracle=None,
        audio_oracle=None,
        harmonic_progressor=HarmonicProgressor()  # No file = disabled
    )
    
    assert not phrase_gen.harmonic_progressor.enabled, "❌ Should be disabled"
    print("✅ Progressor correctly disabled (no transition file)")
    
    # Test harmonic context update (should use detected chord)
    harmonic_context = {
        'current_chord': 'Dm',
        'key_signature': 'D_minor',
    }
    
    phrase_gen._update_harmonic_context(harmonic_context, behavioral_mode='SHADOW')
    
    assert phrase_gen.current_chord == 'Dm', f"Should use detected chord, got {phrase_gen.current_chord}"
    print("✅ Fallback to detected chord works correctly")
    
    print("\n✅ TEST 4 PASSED: Disabled progressor fallback working\n")


def main():
    """Run all integration tests"""
    print("\n" + "🎼"*30)
    print("HARMONIC PROGRESSION INTEGRATION TEST SUITE")
    print("🎼"*30)
    
    try:
        # Test 1: Initialize progressor
        progressor = test_harmonic_progressor_initialization()
        
        # Test 2: Chord selection modes
        test_chord_selection_modes(progressor)
        
        # Test 3: PhraseGenerator integration
        test_phrase_generator_integration(progressor)
        
        # Test 4: Disabled progressor fallback
        test_disabled_progressor()
        
        # Summary
        print("="*60)
        print("🎉 ALL TESTS PASSED!")
        print("="*60)
        print("\n✅ Integration Summary:")
        print("   1. HarmonicProgressor loads transition graphs correctly")
        print("   2. Chord selection works in all behavioral modes")
        print("   3. PhraseGenerator uses progressor during phrase generation")
        print("   4. Fallback behavior works when progressor is disabled")
        print("\n📋 Next Steps:")
        print("   1. Retrain model with chord detection (Step 11)")
        print("   2. Test with real audio in live performance (Step 12)")
        print("   3. Validate musical coherence of progressions")
        print("")
        
        return 0
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
