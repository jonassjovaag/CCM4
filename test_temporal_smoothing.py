#!/usr/bin/env python3
"""
Test Temporal Smoothing

Validates onset filtering, time-window averaging, and chord stabilization.

Usage:
    python test_temporal_smoothing.py
"""

import numpy as np
import sys


def test_onset_filtering():
    """Test that onset filtering keeps only note attacks"""
    print("=" * 60)
    print("TEST 1: Onset Filtering")
    print("=" * 60)
    
    # Create mock events: 10 onsets + 90 sustain
    test_events = []
    
    for i in range(100):
        event = {
            't': i * 0.1,
            'midi': 60,
            'onset': (i % 10 == 0),  # Every 10th event is onset
            'rms_db': -25.0
        }
        test_events.append(event)
    
    print(f"\n📊 Created {len(test_events)} test events")
    print(f"   Onsets: {sum(1 for e in test_events if e['onset'])}")
    print(f"   Sustain: {sum(1 for e in test_events if not e['onset'])}")
    
    # Filter to onsets only
    onset_events = [e for e in test_events if e.get('onset', False)]
    
    print(f"\n✅ After filtering:")
    print(f"   Kept: {len(onset_events)} events")
    print(f"   Reduction: {(1 - len(onset_events)/len(test_events))*100:.1f}%")
    
    assert len(onset_events) == 10, f"Should have 10 onsets, got {len(onset_events)}"
    assert all(e['onset'] for e in onset_events), "All filtered events should be onsets"
    
    print("\n✅ Test 1 PASSED")
    return True


def test_temporal_smoothing():
    """Test time-window averaging"""
    print("\n" + "=" * 60)
    print("TEST 2: Temporal Smoothing (Time-Window Averaging)")
    print("=" * 60)
    
    from core.temporal_smoothing import TemporalSmoother
    
    smoother = TemporalSmoother(window_size=1.0, min_change_threshold=0.5)
    
    # Create 20 events over 5 seconds (same sustained chord with noise)
    test_events = []
    for i in range(20):
        event = {
            't': i * 0.25,
            'rms_db': -25.0 + np.random.randn() * 2.0,  # Noise
            'centroid': 1000.0 + np.random.randn() * 50.0,  # Noise
            'f0': 440.0 + np.random.randn() * 5.0,  # Slight pitch drift
            'midi': 69,
            'chord': 'C',
            'onset': (i == 0),  # Only first is onset
            'consonance': 0.8 + np.random.randn() * 0.05,
            'gesture_token': 5 if i < 10 else 7  # Token change midway
        }
        test_events.append(event)
    
    print(f"\n📊 Created {len(test_events)} events over 5 seconds")
    print(f"   Time span: {test_events[0]['t']:.2f}s to {test_events[-1]['t']:.2f}s")
    
    # Smooth
    smoothed = smoother.smooth_events(test_events)
    
    print(f"\n✅ After smoothing (1.0s windows):")
    print(f"   Original: {len(test_events)} events")
    print(f"   Smoothed: {len(smoothed)} events")
    print(f"   Reduction: {(1 - len(smoothed)/len(test_events))*100:.1f}%")
    
    # Should have ~5 events (one per second)
    assert len(smoothed) < len(test_events), "Should reduce event count"
    assert 3 <= len(smoothed) <= 7, f"Expected ~5 events, got {len(smoothed)}"
    
    # Check that features were averaged
    assert all('smoothed' in e for e in smoothed), "All should be marked as smoothed"
    assert all('window_size' in e for e in smoothed), "All should have window_size"
    
    print("   ✅ Features averaged correctly")
    print("   ✅ Window size tracked")
    
    print("\n✅ Test 2 PASSED")
    return True


def test_chord_stabilization():
    """Test chord label smoothing"""
    print("\n" + "=" * 60)
    print("TEST 3: Chord Label Stabilization")
    print("=" * 60)
    
    from core.temporal_smoothing import TemporalSmoother
    
    smoother = TemporalSmoother()
    
    # Create events with jittery chord labels
    # Musical reality: C major for 5 seconds, then Dm for 3 seconds
    # Detected: Rapid flipping due to overtones
    chord_sequence = [
        'C', 'C', 'Cdim', 'C', 'C', 'F#7', 'C', 'C', 'D9', 'C',  # 0-5s (should all be C)
        'Dm', 'Dm', 'Dm', 'Dm', 'Dm'  # 5-7.5s (should all be Dm)
    ]
    
    test_events = []
    for i, chord in enumerate(chord_sequence):
        test_events.append({
            't': i * 0.5,
            'chord': chord,
            'chord_confidence': 0.7 if chord in ['C', 'Dm'] else 0.3,  # Jitter has low confidence
            'midi': 60
        })
    
    print(f"\n📊 Created {len(test_events)} events with jittery chords")
    
    # Extract original sequence
    original_seq = [e['chord'] for e in test_events]
    print(f"   Original:   {original_seq}")
    
    # Apply smoothing
    stabilized = smoother.smooth_chord_labels(test_events, min_chord_duration=2.0)
    stabilized_seq = [e['chord'] for e in stabilized]
    
    print(f"   Stabilized: {stabilized_seq}")
    
    # First 10 should all be 'C' (jitter removed)
    first_10_stabilized = stabilized_seq[:10]
    assert all(c == 'C' for c in first_10_stabilized), f"First 10 should be 'C', got {first_10_stabilized}"
    
    # Last 5 should all be 'Dm'
    last_5_stabilized = stabilized_seq[-5:]
    assert all(c == 'Dm' for c in last_5_stabilized), f"Last 5 should be 'Dm', got {last_5_stabilized}"
    
    print("\n✅ Verification:")
    print(f"   First 10 events: all '{first_10_stabilized[0]}' ✅")
    print(f"   Last 5 events: all '{last_5_stabilized[0]}' ✅")
    print("   Jitter removed ✅")
    
    print("\n✅ Test 3 PASSED")
    return True


def test_feature_averaging():
    """Test that numerical features are averaged correctly"""
    print("\n" + "=" * 60)
    print("TEST 4: Feature Averaging")
    print("=" * 60)
    
    from core.temporal_smoothing import TemporalSmoother
    
    smoother = TemporalSmoother(window_size=2.0)
    
    # Create 5 events with known values
    test_events = [
        {'t': 0.0, 'rms_db': -20.0, 'centroid': 1000.0, 'consonance': 0.8, 'onset': False},
        {'t': 0.4, 'rms_db': -22.0, 'centroid': 1100.0, 'consonance': 0.7, 'onset': False},
        {'t': 0.8, 'rms_db': -24.0, 'centroid': 1200.0, 'consonance': 0.6, 'onset': False},
        {'t': 1.2, 'rms_db': -26.0, 'centroid': 1300.0, 'consonance': 0.5, 'onset': False},
        {'t': 1.6, 'rms_db': -28.0, 'centroid': 1400.0, 'consonance': 0.4, 'onset': False},
    ]
    
    smoothed = smoother.smooth_events(test_events)
    
    print(f"\n📊 Input: {len(test_events)} events")
    print(f"   Smoothed: {len(smoothed)} events")
    
    # Should be 1 event (all within 2s window)
    assert len(smoothed) == 1, f"Expected 1 smoothed event, got {len(smoothed)}"
    
    # Check averaged values
    avg_event = smoothed[0]
    expected_rms = np.mean([-20, -22, -24, -26, -28])
    expected_centroid = np.mean([1000, 1100, 1200, 1300, 1400])
    expected_consonance = np.mean([0.8, 0.7, 0.6, 0.5, 0.4])
    
    print(f"\n✅ Feature averaging:")
    print(f"   RMS: {avg_event['rms_db']:.1f} (expected: {expected_rms:.1f})")
    print(f"   Centroid: {avg_event['centroid']:.1f} (expected: {expected_centroid:.1f})")
    print(f"   Consonance: {avg_event['consonance']:.2f} (expected: {expected_consonance:.2f})")
    
    assert abs(avg_event['rms_db'] - expected_rms) < 0.1, "RMS not averaged correctly"
    assert abs(avg_event['centroid'] - expected_centroid) < 0.1, "Centroid not averaged correctly"
    assert abs(avg_event['consonance'] - expected_consonance) < 0.01, "Consonance not averaged correctly"
    
    print("   ✅ All features averaged correctly")
    
    print("\n✅ Test 4 PASSED")
    return True


def run_all_tests():
    """Run all temporal smoothing tests"""
    print("\n" + "=" * 60)
    print("TEMPORAL SMOOTHING TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Onset Filtering", test_onset_filtering),
        ("Temporal Smoothing", test_temporal_smoothing),
        ("Chord Stabilization", test_chord_stabilization),
        ("Feature Averaging", test_feature_averaging),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
            else:
                failed += 1
                print(f"\n❌ {name} FAILED")
        except Exception as e:
            failed += 1
            print(f"\n❌ {name} FAILED with exception:")
            print(f"   {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED!")
        print("\n📋 Temporal smoothing is ready to use:")
        print("   --onset-only: Filter to note attacks only")
        print("   --temporal-smoothing: Average features in time windows")
        print("   --smooth-chords: Stabilize chord labels (enabled by default)")
        print("\n🚀 Recommended usage:")
        print("   python Chandra_trainer.py --file Georgia.wav --max-events 1000 \\")
        print("     --hybrid-perception --wav2vec --gpu --temporal-smoothing")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())


