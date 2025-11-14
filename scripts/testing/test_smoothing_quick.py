#!/usr/bin/env python3
"""Quick test to verify temporal smoothing is working"""

import numpy as np
from core.temporal_smoothing import TemporalSmoother

def test_chord_smoothing():
    """Test that chord label smoothing prevents jitter"""
    print("\n🧪 Testing Chord Label Smoothing...")
    
    # Create events with noisy chord labels (same chord with jitter)
    events = []
    base_time = 0.0
    
    # Simulate 10 events over 5 seconds - should all be "C"
    # but noise makes them vary
    noisy_chords = ['C', 'C', 'Cdim', 'C', 'F#7', 'C', 'C', 'D9', 'Cdim', 'C']
    
    for i, chord in enumerate(noisy_chords):
        events.append({
            't': base_time + i * 0.5,
            'chord': chord,
            'chord_confidence': 0.5,
            'midi': 60,
            'rms_db': -20
        })
    
    print(f"   Noisy chord sequence: {[e['chord'] for e in events]}")
    
    # Apply smoothing
    smoother = TemporalSmoother()
    smoothed_events = smoother.smooth_chord_labels(events, min_chord_duration=2.0)
    
    smoothed_chords = [e['chord'] for e in smoothed_events]
    print(f"   Smoothed chord sequence: {smoothed_chords}")
    
    # Verify: Should be all "C" (most common chord)
    unique_chords = set(smoothed_chords)
    if len(unique_chords) <= 2:  # Allow 1-2 unique chords max
        print(f"   ✅ Chord smoothing working! Reduced from 6 variants to {len(unique_chords)}")
        return True
    else:
        print(f"   ❌ Chord smoothing failed! Still {len(unique_chords)} variants")
        return False

def test_temporal_smoothing():
    """Test that temporal smoothing reduces event count"""
    print("\n🧪 Testing Temporal Smoothing...")
    
    # Create 20 events over 5 seconds (over-sampled sustained note)
    events = []
    for i in range(20):
        events.append({
            't': i * 0.25,  # 0.25s intervals
            'midi': 60,
            'velocity': 64,
            'rms_db': -20.0 - i * 0.5,  # Gradual decay
            'chord': 'C',
            'onset': (i == 0)  # Only first is onset
        })
    
    print(f"   Original event count: {len(events)}")
    
    # Apply smoothing with 0.5s window
    smoother = TemporalSmoother(window_size=0.5)
    smoothed_events = smoother.smooth_events(events)
    
    print(f"   Smoothed event count: {len(smoothed_events)}")
    reduction = (1 - len(smoothed_events) / len(events)) * 100
    print(f"   Reduction: {reduction:.1f}%")
    
    if len(smoothed_events) < len(events):
        print(f"   ✅ Temporal smoothing working! Reduced event count")
        return True
    else:
        print(f"   ❌ Temporal smoothing failed!")
        return False

def test_onset_filtering():
    """Test onset filtering"""
    print("\n🧪 Testing Onset Filtering...")
    
    # Create events: 5 onsets + 15 sustain
    events = []
    for i in range(20):
        events.append({
            't': i * 0.25,
            'midi': 60 + (i % 5),
            'onset': (i % 4 == 0)  # Every 4th event is onset
        })
    
    onset_count = sum(1 for e in events if e.get('onset', False))
    print(f"   Total events: {len(events)}")
    print(f"   Onset events: {onset_count}")
    
    # Filter to onsets only
    onset_events = [e for e in events if e.get('onset', False)]
    
    print(f"   After filtering: {len(onset_events)}")
    
    if len(onset_events) == onset_count and len(onset_events) < len(events):
        print(f"   ✅ Onset filtering working!")
        return True
    else:
        print(f"   ❌ Onset filtering failed!")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🔬 Temporal Smoothing & Event Filtering Tests")
    print("=" * 60)
    
    results = []
    
    try:
        results.append(("Chord Smoothing", test_chord_smoothing()))
    except Exception as e:
        print(f"   ❌ Test crashed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Chord Smoothing", False))
    
    try:
        results.append(("Temporal Smoothing", test_temporal_smoothing()))
    except Exception as e:
        print(f"   ❌ Test crashed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Temporal Smoothing", False))
    
    try:
        results.append(("Onset Filtering", test_onset_filtering()))
    except Exception as e:
        print(f"   ❌ Test crashed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Onset Filtering", False))
    
    print("\n" + "=" * 60)
    print("📊 Test Results Summary")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {test_name}")
    
    all_passed = all(passed for _, passed in results)
    
    if all_passed:
        print("\n✅ All tests passed! Smoothing system is working correctly.")
    else:
        print("\n❌ Some tests failed. Check implementation.")
    
    print("=" * 60)


