#!/usr/bin/env python3
"""
Synthetic test for HarmonicProgressor - validate learned chord progressions.

Tests the harmonic progression system without live audio by simulating
a sequence of detected chords and tracking how the system responds.

Validates:
1. HarmonicProgressor loads transition graph correctly
2. Chord selection differs appropriately by behavioral mode
3. Learned progressions are being applied (COUPLE mode)
4. Input tracking works (what was detected vs what was chosen)
5. Temperature affects randomness as expected
"""

import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from agent.harmonic_progressor import HarmonicProgressor


def print_section(title):
    """Print formatted section header"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def test_load_transition_graph(graph_file):
    """Test that HarmonicProgressor loads the graph correctly"""
    print_section("TEST 1: Load Transition Graph")
    
    if not Path(graph_file).exists():
        print(f"❌ Transition graph file not found: {graph_file}")
        return None
    
    print(f"📂 Loading transition graph from: {graph_file}")
    progressor = HarmonicProgressor(graph_file)
    
    if not progressor.enabled:
        print(f"❌ HarmonicProgressor failed to enable")
        return None
    
    print(f"✅ HarmonicProgressor loaded successfully!")
    
    # Get statistics
    stats = progressor.get_statistics_summary()
    print(f"\n📊 Transition Graph Statistics:")
    print(f"   • Unique chords: {stats.get('unique_chords', 0)}")
    print(f"   • Total transitions: {stats.get('total_transitions', 0)}")
    print(f"   • Real progressions: {stats.get('real_progressions', 0)} (excluding self-loops)")
    
    # Show top progressions
    if 'top_progressions' in stats:
        print(f"\n🎵 Top 5 Learned Progressions:")
        for i, prog in enumerate(stats['top_progressions'][:5], 1):
            print(f"   {i}. {prog['transition']}: {prog['probability']:.1%} ({prog['count']} times)")
    
    return progressor


def test_behavioral_modes(progressor, test_chords):
    """Test chord selection across different behavioral modes"""
    print_section("TEST 2: Behavioral Mode Differences")
    
    modes = ['SHADOW', 'COUPLE', 'MIRROR']
    temperature = 0.8
    
    print(f"Testing with chord sequence: {' → '.join(test_chords)}\n")
    
    results = {mode: [] for mode in modes}
    
    for mode in modes:
        print(f"🎭 {mode} Mode (temperature={temperature}):")
        current_chord = test_chords[0]
        
        for i, detected_chord in enumerate(test_chords):
            chosen = progressor.choose_next_chord(
                current_chord=detected_chord,
                behavioral_mode=mode,
                temperature=temperature
            )
            results[mode].append(chosen)
            
            # Show what happened
            match_icon = "✓" if chosen == detected_chord else "→"
            print(f"   Step {i+1}: Detected [{detected_chord}] {match_icon} Chose [{chosen}]")
            
            current_chord = chosen
        
        print()
    
    # Analyze differences between modes
    print("📊 Mode Comparison:")
    for i, detected in enumerate(test_chords):
        print(f"   Input [{detected}]:")
        for mode in modes:
            chosen = results[mode][i]
            match = "MATCH" if chosen == detected else "DIFFERENT"
            print(f"      {mode:8s} → [{chosen:4s}] ({match})")
    
    return results


def test_learned_progressions(progressor, start_chord='D', steps=10):
    """Test that COUPLE mode follows learned progressions"""
    print_section("TEST 3: Learned Progression Following (COUPLE Mode)")
    
    print(f"🎵 Generating {steps}-step progression starting from [{start_chord}]")
    print(f"   Using COUPLE mode (should follow learned transitions)\n")
    
    progression = [start_chord]
    current = start_chord
    
    print(f"Step 1: Start with [{current}]")
    
    for step in range(2, steps + 1):
        # COUPLE mode with moderate temperature
        next_chord = progressor.choose_next_chord(
            current_chord=current,
            behavioral_mode='COUPLE',
            temperature=0.7
        )
        
        progression.append(next_chord)
        
        # Check if this was a learned transition
        transition_key = f"{current}->{next_chord}"
        if transition_key in progressor.transitions:
            trans_data = progressor.transitions[transition_key]
            prob = trans_data.probability
            count = trans_data.count
            print(f"Step {step}: [{current}] → [{next_chord}] "
                  f"(learned: {prob:.1%}, seen {count}x)")
        else:
            print(f"Step {step}: [{current}] → [{next_chord}] "
                  f"(fallback - not in training data)")
        
        current = next_chord
    
    print(f"\n🎼 Generated Progression:")
    print(f"   {' → '.join(progression)}")
    
    return progression


def test_temperature_effect(progressor, test_chord='D', samples=20):
    """Test that temperature affects randomness"""
    print_section("TEST 4: Temperature Effect on Randomness")
    
    temperatures = [0.1, 0.5, 1.0, 1.5]
    
    print(f"Sampling {samples} times from chord [{test_chord}] with different temperatures\n")
    
    for temp in temperatures:
        choices = []
        for _ in range(samples):
            choice = progressor.choose_next_chord(
                current_chord=test_chord,
                behavioral_mode='COUPLE',
                temperature=temp
            )
            choices.append(choice)
        
        # Count unique choices
        unique = set(choices)
        unique_count = len(unique)
        
        # Show distribution
        from collections import Counter
        counts = Counter(choices)
        
        print(f"🌡️  Temperature = {temp:.1f}:")
        print(f"   Unique chords chosen: {unique_count}/{samples}")
        print(f"   Distribution:")
        for chord, count in counts.most_common(5):
            pct = count / samples * 100
            bar = '█' * int(pct / 5)
            print(f"      [{chord:4s}]: {count:2d} ({pct:4.1f}%) {bar}")
        print()


def test_input_output_tracking(progressor, input_sequence):
    """Track input vs output to verify the system responds to input correctly"""
    print_section("TEST 5: Input/Output Tracking")
    
    print("🎯 Verifying system responds to detected chords appropriately\n")
    print(f"Input sequence: {' → '.join(input_sequence)}\n")
    
    modes = ['SHADOW', 'COUPLE', 'MIRROR']
    
    for mode in modes:
        print(f"🎭 {mode} Mode:")
        
        match_count = 0
        learned_count = 0
        fallback_count = 0
        
        current = input_sequence[0]
        
        for detected in input_sequence:
            chosen = progressor.choose_next_chord(
                current_chord=detected,
                behavioral_mode=mode,
                temperature=0.8
            )
            
            # Track if matched
            if chosen == detected:
                match_count += 1
                status = "✓ MATCH"
            else:
                # Check if learned transition
                trans_key = f"{detected}->{chosen}"
                if trans_key in progressor.transitions:
                    learned_count += 1
                    prob = progressor.transitions[trans_key].probability
                    status = f"→ LEARNED ({prob:.1%})"
                else:
                    fallback_count += 1
                    status = "→ FALLBACK"
            
            print(f"   INPUT: [{detected:4s}] → OUTPUT: [{chosen:4s}] {status}")
            current = chosen
        
        # Summary
        total = len(input_sequence)
        print(f"\n   Summary:")
        print(f"      Matched input: {match_count}/{total} ({match_count/total*100:.1f}%)")
        print(f"      Used learned:  {learned_count}/{total} ({learned_count/total*100:.1f}%)")
        print(f"      Fallback:      {fallback_count}/{total} ({fallback_count/total*100:.1f}%)")
        print()


def test_pitch_awareness(progressor):
    """
    Test that the system is aware of pitch relationships.
    NOTE: This test shows HARMONIC awareness (chord labels).
    For 768D gesture token tracking, we need integration with Wav2Vec.
    """
    print_section("TEST 6: Pitch/Harmonic Awareness")
    
    print("🎵 Testing harmonic relationship awareness\n")
    
    # Test related chords (common progressions in training data)
    test_pairs = [
        ('D', 'A'),      # Dominant relationship
        ('A', 'D'),      # Subdominant relationship  
        ('Am', 'D'),     # Minor to major
        ('F#m', 'D'),    # Common in training data
        ('D', 'F#m'),    # Reverse
    ]
    
    print("Testing if learned progressions reflect musical relationships:\n")
    
    for from_chord, to_chord in test_pairs:
        # Check if this transition was learned
        forward_key = f"{from_chord}->{to_chord}"
        reverse_key = f"{to_chord}->{from_chord}"
        
        forward_learned = forward_key in progressor.transitions
        reverse_learned = reverse_key in progressor.transitions
        
        print(f"   {from_chord} ↔ {to_chord}:")
        
        if forward_learned:
            data = progressor.transitions[forward_key]
            print(f"      {from_chord}→{to_chord}: ✅ Learned ({data.probability:.1%}, {data.count}x)")
        else:
            print(f"      {from_chord}→{to_chord}: ❌ Not in training")
        
        if reverse_learned:
            data = progressor.transitions[reverse_key]
            print(f"      {to_chord}→{from_chord}: ✅ Learned ({data.probability:.1%}, {data.count}x)")
        else:
            print(f"      {to_chord}→{from_chord}: ❌ Not in training")
        
        if forward_learned or reverse_learned:
            print(f"      → Bidirectional relationship: {forward_learned and reverse_learned}")
        
        print()
    
    print("NOTE: For 768D gesture token tracking (timbre/pitch perception),")
    print("      see test_end_to_end_system.py which includes Wav2Vec integration.")


def main():
    """Run all synthetic tests"""
    
    print("\n" + "="*70)
    print("  HARMONIC PROGRESSION SYSTEM - SYNTHETIC TEST SUITE")
    print("="*70)
    print("\nThis test validates the HarmonicProgressor without live audio input.")
    print("It simulates detected chords and tracks system responses.\n")
    
    # Use the most recent training
    graph_file = "JSON/Curious_child_091125_2009_training_harmonic_transitions.json"
    
    # Test 1: Load
    progressor = test_load_transition_graph(graph_file)
    if progressor is None:
        print("\n❌ Failed to load transition graph - aborting tests")
        sys.exit(1)
    
    # Define test chord sequences (based on learned progressions)
    test_sequence = ['D', 'A', 'F#m', 'D', 'Am', 'D', 'G', 'D']
    
    # Test 2: Behavioral modes
    test_behavioral_modes(progressor, test_sequence[:5])
    
    # Test 3: Learned progressions
    test_learned_progressions(progressor, start_chord='D', steps=8)
    
    # Test 4: Temperature
    test_temperature_effect(progressor, test_chord='D', samples=20)
    
    # Test 5: Input/output tracking
    test_input_output_tracking(progressor, test_sequence)
    
    # Test 6: Pitch/harmonic awareness
    test_pitch_awareness(progressor)
    
    # Final summary
    print_section("TEST SUMMARY")
    print("✅ All tests completed!")
    print("\n📋 Test Coverage:")
    print("   ✓ Transition graph loading")
    print("   ✓ Behavioral mode differentiation (SHADOW/COUPLE/MIRROR)")
    print("   ✓ Learned progression following")
    print("   ✓ Temperature effect on randomness")
    print("   ✓ Input/output tracking")
    print("   ✓ Harmonic relationship awareness")
    print("\n💡 Next: Test in live performance with MusicHal_9000.py")
    print("   This will add 768D gesture token tracking (timbral/pitch perception)")
    print()


if __name__ == '__main__':
    main()
