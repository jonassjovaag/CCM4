#!/usr/bin/env python3
"""
Test script for melodic enhancement validation

Generates sample phrases with and without melodic constraints,
then analyzes and compares:
- Interval distribution (steps vs leaps)
- Pitch range usage
- Scale adherence (diatonic vs chromatic)
- Melodic contour smoothness
"""

import sys
import numpy as np
from collections import Counter
from typing import List, Dict

# Add project root to path
sys.path.insert(0, '/Users/jonashsj/Jottacloud/PhD - UiA/CCM4')

from agent.phrase_generator import PhraseGenerator


def analyze_intervals(notes: List[int]) -> Dict[str, float]:
    """Analyze interval distribution in a melody"""
    if len(notes) < 2:
        return {}
    
    intervals = [notes[i+1] - notes[i] for i in range(len(notes)-1)]
    abs_intervals = [abs(i) for i in intervals]
    
    # Count interval types
    steps = sum(1 for i in abs_intervals if i <= 2)  # Half/whole steps
    small_leaps = sum(1 for i in abs_intervals if 3 <= i <= 4)  # Minor/major 3rds
    medium_leaps = sum(1 for i in abs_intervals if 5 <= i <= 7)  # 4ths/5ths
    large_leaps = sum(1 for i in abs_intervals if i > 7)  # 6ths and larger
    tritones = sum(1 for i in abs_intervals if i == 6)  # Tritone count
    
    total = len(intervals)
    
    return {
        'steps_pct': (steps / total * 100) if total > 0 else 0,
        'small_leaps_pct': (small_leaps / total * 100) if total > 0 else 0,
        'medium_leaps_pct': (medium_leaps / total * 100) if total > 0 else 0,
        'large_leaps_pct': (large_leaps / total * 100) if total > 0 else 0,
        'tritone_count': tritones,
        'avg_interval': np.mean(abs_intervals),
        'max_interval': max(abs_intervals),
        'intervals': intervals  # For contour analysis
    }


def analyze_pitch_range(notes: List[int]) -> Dict[str, any]:
    """Analyze pitch range usage"""
    if not notes:
        return {}
    
    return {
        'min_note': min(notes),
        'max_note': max(notes),
        'range_semitones': max(notes) - min(notes),
        'mean_pitch': np.mean(notes),
        'pitch_std': np.std(notes)
    }


def analyze_scale_adherence(notes: List[int], scale_degrees: List[int] = None) -> Dict[str, float]:
    """Analyze how well notes adhere to diatonic scale"""
    if scale_degrees is None:
        scale_degrees = [0, 2, 4, 5, 7, 9, 11]  # C major
    
    pitch_classes = [n % 12 for n in notes]
    diatonic_notes = sum(1 for pc in pitch_classes if pc in scale_degrees)
    chromatic_notes = len(pitch_classes) - diatonic_notes
    
    # Count pitch class distribution
    pc_counts = Counter(pitch_classes)
    
    return {
        'diatonic_pct': (diatonic_notes / len(notes) * 100) if notes else 0,
        'chromatic_pct': (chromatic_notes / len(notes) * 100) if notes else 0,
        'pitch_class_distribution': dict(pc_counts),
        'unique_pitch_classes': len(pc_counts)
    }


def analyze_contour(intervals: List[int]) -> Dict[str, any]:
    """Analyze melodic contour characteristics"""
    if not intervals:
        return {}
    
    # Direction changes
    directions = [1 if i > 0 else (-1 if i < 0 else 0) for i in intervals]
    direction_changes = sum(1 for i in range(len(directions)-1) 
                           if directions[i] != 0 and directions[i+1] != 0 
                           and directions[i] != directions[i+1])
    
    # Calculate run lengths (consecutive steps in same direction)
    run_lengths = []
    current_run = 1
    for i in range(len(directions)-1):
        if directions[i] != 0 and directions[i+1] != 0:
            if directions[i] == directions[i+1]:
                current_run += 1
            else:
                run_lengths.append(current_run)
                current_run = 1
        else:
            if current_run > 1:
                run_lengths.append(current_run)
            current_run = 1
    if current_run > 1:
        run_lengths.append(current_run)
    
    # Arch detection (rise then fall, or fall then rise)
    has_arch = False
    if len(run_lengths) >= 2:
        # Look for alternating direction runs (arch pattern)
        has_arch = any(run_lengths[i] >= 2 and run_lengths[i+1] >= 2 
                      for i in range(len(run_lengths)-1))
    
    return {
        'direction_changes': direction_changes,
        'avg_run_length': np.mean(run_lengths) if run_lengths else 0,
        'max_run_length': max(run_lengths) if run_lengths else 0,
        'has_arch_shape': has_arch,
        'run_lengths': run_lengths
    }


def print_analysis(name: str, notes: List[int], verbose: bool = False):
    """Print comprehensive analysis of a melody"""
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")
    
    if verbose:
        print(f"\nNotes ({len(notes)}): {notes[:20]}{'...' if len(notes) > 20 else ''}")
    
    # Pitch range analysis
    range_data = analyze_pitch_range(notes)
    print(f"\n📊 Pitch Range:")
    print(f"  Range: MIDI {range_data['min_note']}-{range_data['max_note']} "
          f"({range_data['range_semitones']} semitones)")
    print(f"  Mean: {range_data['mean_pitch']:.1f} (±{range_data['pitch_std']:.1f})")
    
    # Interval analysis
    interval_data = analyze_intervals(notes)
    print(f"\n🎵 Interval Distribution:")
    print(f"  Steps (≤2 semitones):      {interval_data['steps_pct']:5.1f}%")
    print(f"  Small leaps (3-4):         {interval_data['small_leaps_pct']:5.1f}%")
    print(f"  Medium leaps (5-7):        {interval_data['medium_leaps_pct']:5.1f}%")
    print(f"  Large leaps (>7):          {interval_data['large_leaps_pct']:5.1f}%")
    print(f"  Tritones (6):              {interval_data['tritone_count']} occurrences")
    print(f"  Average interval:          {interval_data['avg_interval']:.2f} semitones")
    print(f"  Maximum interval:          {interval_data['max_interval']} semitones")
    
    # Scale adherence
    scale_data = analyze_scale_adherence(notes)
    print(f"\n🎼 Scale Adherence (C major):")
    print(f"  Diatonic notes:            {scale_data['diatonic_pct']:5.1f}%")
    print(f"  Chromatic notes:           {scale_data['chromatic_pct']:5.1f}%")
    print(f"  Unique pitch classes:      {scale_data['unique_pitch_classes']}/12")
    
    if verbose:
        print(f"  Pitch class distribution:  {scale_data['pitch_class_distribution']}")
    
    # Contour analysis
    contour_data = analyze_contour(interval_data['intervals'])
    print(f"\n📈 Melodic Contour:")
    print(f"  Direction changes:         {contour_data['direction_changes']}")
    print(f"  Average run length:        {contour_data['avg_run_length']:.2f} steps")
    print(f"  Maximum run length:        {contour_data['max_run_length']} steps")
    print(f"  Has arch shape:            {'✓ Yes' if contour_data['has_arch_shape'] else '✗ No'}")
    
    if verbose and contour_data['run_lengths']:
        print(f"  Run lengths:               {contour_data['run_lengths'][:10]}")


def generate_test_phrases(phrase_gen: PhraseGenerator, 
                          num_phrases: int = 10,
                          phrase_type: str = "buildup") -> List[List[int]]:
    """Generate multiple test phrases"""
    phrases = []
    
    # Mock current_event
    current_event = {
        'instrument': 'keys',
        'rms_db': -40.0,
        'consonance': 0.5
    }
    
    for i in range(num_phrases):
        if phrase_type == "buildup":
            phrase = phrase_gen._generate_buildup_phrase(
                mode="SHADOW",
                voice_type="melodic",
                timestamp=float(i),
                current_event=current_event,
                temperature=0.8,
                activity_multiplier=1.0
            )
        elif phrase_type == "peak":
            phrase = phrase_gen._generate_peak_phrase(
                mode="SHADOW",
                voice_type="melodic",
                timestamp=float(i),
                current_event=current_event,
                temperature=0.8,
                activity_multiplier=1.0
            )
        
        if phrase and phrase.notes:
            phrases.append(phrase.notes)
    
    return phrases


def run_comparison_test():
    """Main test: Compare melodies with and without constraints"""
    
    print("\n" + "="*70)
    print("  MELODIC ENHANCEMENT VALIDATION TEST")
    print("="*70)
    print("\nGenerating test phrases...")
    
    # Test 1: WITH constraints (current settings)
    print("\n" + "="*70)
    print("  TEST 1: WITH MELODIC CONSTRAINTS (Enhanced)")
    print("="*70)
    
    phrase_gen_enhanced = PhraseGenerator(
        rhythm_oracle=None,
        audio_oracle=None,
        enable_silence=True
    )
    
    # Verify settings
    print(f"\nSettings:")
    print(f"  melodic_range: {phrase_gen_enhanced.melodic_range}")
    print(f"  max_leap: {phrase_gen_enhanced.max_leap}")
    print(f"  prefer_steps: {phrase_gen_enhanced.prefer_steps}")
    print(f"  penalize_tritone: {phrase_gen_enhanced.penalize_tritone}")
    print(f"  scale_constraint: {phrase_gen_enhanced.scale_constraint}")
    
    # Generate buildup phrases
    buildup_enhanced = generate_test_phrases(phrase_gen_enhanced, num_phrases=10, phrase_type="buildup")
    all_notes_enhanced_buildup = [note for phrase in buildup_enhanced for note in phrase]
    
    print_analysis("BUILDUP PHRASES (Enhanced)", all_notes_enhanced_buildup)
    
    # Generate peak phrases
    peak_enhanced = generate_test_phrases(phrase_gen_enhanced, num_phrases=10, phrase_type="peak")
    all_notes_enhanced_peak = [note for phrase in peak_enhanced for note in phrase]
    
    print_analysis("PEAK PHRASES (Enhanced)", all_notes_enhanced_peak)
    
    # Test 2: WITHOUT constraints (old behavior)
    print("\n\n" + "="*70)
    print("  TEST 2: WITHOUT MELODIC CONSTRAINTS (Original)")
    print("="*70)
    
    phrase_gen_original = PhraseGenerator(
        rhythm_oracle=None,
        audio_oracle=None,
        enable_silence=True
    )
    
    # Disable all constraints
    phrase_gen_original.melodic_range = (60, 96)  # OLD: Wide range
    phrase_gen_original.max_leap = 12  # Allow octave leaps
    phrase_gen_original.penalize_tritone = False  # Allow tritones
    phrase_gen_original.scale_constraint = False  # No scale snapping
    
    print(f"\nSettings:")
    print(f"  melodic_range: {phrase_gen_original.melodic_range}")
    print(f"  max_leap: {phrase_gen_original.max_leap}")
    print(f"  prefer_steps: {phrase_gen_original.prefer_steps}")
    print(f"  penalize_tritone: {phrase_gen_original.penalize_tritone}")
    print(f"  scale_constraint: {phrase_gen_original.scale_constraint}")
    
    # Generate buildup phrases
    buildup_original = generate_test_phrases(phrase_gen_original, num_phrases=10, phrase_type="buildup")
    all_notes_original_buildup = [note for phrase in buildup_original for note in phrase]
    
    print_analysis("BUILDUP PHRASES (Original)", all_notes_original_buildup)
    
    # Generate peak phrases
    peak_original = generate_test_phrases(phrase_gen_original, num_phrases=10, phrase_type="peak")
    all_notes_original_peak = [note for phrase in peak_original for note in phrase]
    
    print_analysis("PEAK PHRASES (Original)", all_notes_original_peak)
    
    # Comparison summary
    print("\n\n" + "="*70)
    print("  COMPARISON SUMMARY")
    print("="*70)
    
    # Buildup comparison
    enhanced_buildup_intervals = analyze_intervals(all_notes_enhanced_buildup)
    original_buildup_intervals = analyze_intervals(all_notes_original_buildup)
    enhanced_buildup_scale = analyze_scale_adherence(all_notes_enhanced_buildup)
    original_buildup_scale = analyze_scale_adherence(all_notes_original_buildup)
    
    print("\n📊 BUILDUP PHRASES - Key Metrics:")
    print(f"\n  Stepwise Motion (target >70%):")
    print(f"    Enhanced:  {enhanced_buildup_intervals['steps_pct']:5.1f}%")
    print(f"    Original:  {original_buildup_intervals['steps_pct']:5.1f}%")
    print(f"    Change:    {enhanced_buildup_intervals['steps_pct'] - original_buildup_intervals['steps_pct']:+5.1f}%")
    
    print(f"\n  Diatonic Notes (target >80%):")
    print(f"    Enhanced:  {enhanced_buildup_scale['diatonic_pct']:5.1f}%")
    print(f"    Original:  {original_buildup_scale['diatonic_pct']:5.1f}%")
    print(f"    Change:    {enhanced_buildup_scale['diatonic_pct'] - original_buildup_scale['diatonic_pct']:+5.1f}%")
    
    print(f"\n  Tritone Count (target <5):")
    print(f"    Enhanced:  {enhanced_buildup_intervals['tritone_count']}")
    print(f"    Original:  {original_buildup_intervals['tritone_count']}")
    
    print(f"\n  Pitch Range (target <24 semitones):")
    enhanced_buildup_range = analyze_pitch_range(all_notes_enhanced_buildup)
    original_buildup_range = analyze_pitch_range(all_notes_original_buildup)
    print(f"    Enhanced:  {enhanced_buildup_range['range_semitones']} semitones")
    print(f"    Original:  {original_buildup_range['range_semitones']} semitones")
    
    # Peak comparison
    enhanced_peak_intervals = analyze_intervals(all_notes_enhanced_peak)
    original_peak_intervals = analyze_intervals(all_notes_original_peak)
    enhanced_peak_scale = analyze_scale_adherence(all_notes_enhanced_peak)
    original_peak_scale = analyze_scale_adherence(all_notes_original_peak)
    
    print("\n📊 PEAK PHRASES - Key Metrics:")
    print(f"\n  Stepwise Motion:")
    print(f"    Enhanced:  {enhanced_peak_intervals['steps_pct']:5.1f}%")
    print(f"    Original:  {original_peak_intervals['steps_pct']:5.1f}%")
    print(f"    Change:    {enhanced_peak_intervals['steps_pct'] - original_peak_intervals['steps_pct']:+5.1f}%")
    
    print(f"\n  Diatonic Notes:")
    print(f"    Enhanced:  {enhanced_peak_scale['diatonic_pct']:5.1f}%")
    print(f"    Original:  {original_peak_scale['diatonic_pct']:5.1f}%")
    print(f"    Change:    {enhanced_peak_scale['diatonic_pct'] - original_peak_scale['diatonic_pct']:+5.1f}%")
    
    print(f"\n  Tritone Count:")
    print(f"    Enhanced:  {enhanced_peak_intervals['tritone_count']}")
    print(f"    Original:  {original_peak_intervals['tritone_count']}")
    
    # Success criteria
    print("\n\n" + "="*70)
    print("  SUCCESS CRITERIA")
    print("="*70)
    
    success_checks = []
    
    # Check 1: Stepwise motion increase
    buildup_steps_improved = enhanced_buildup_intervals['steps_pct'] > original_buildup_intervals['steps_pct']
    success_checks.append(("Buildup stepwise motion increased", buildup_steps_improved))
    
    # Check 2: Diatonic adherence increase
    buildup_diatonic_improved = enhanced_buildup_scale['diatonic_pct'] > original_buildup_scale['diatonic_pct']
    success_checks.append(("Buildup diatonic adherence increased", buildup_diatonic_improved))
    
    # Check 3: Tritone reduction
    tritones_reduced = enhanced_buildup_intervals['tritone_count'] <= original_buildup_intervals['tritone_count']
    success_checks.append(("Tritones reduced or equal", tritones_reduced))
    
    # Check 4: Range constraint
    range_constrained = enhanced_buildup_range['range_semitones'] <= 24  # 2 octaves
    success_checks.append(("Range constrained to ≤24 semitones", range_constrained))
    
    # Check 5: Peak also improved
    peak_steps_improved = enhanced_peak_intervals['steps_pct'] > original_peak_intervals['steps_pct']
    success_checks.append(("Peak stepwise motion increased", peak_steps_improved))
    
    print()
    for check_name, passed in success_checks:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}  {check_name}")
    
    all_passed = all(passed for _, passed in success_checks)
    
    print("\n" + "="*70)
    if all_passed:
        print("  ✅ ALL TESTS PASSED - Melodic enhancements working!")
    else:
        print("  ⚠️  SOME TESTS FAILED - Review results above")
    print("="*70 + "\n")
    
    return all_passed


if __name__ == "__main__":
    try:
        success = run_comparison_test()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ TEST ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
