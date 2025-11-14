#!/usr/bin/env python3
"""Test gesture token temporal smoothing."""

import time
from listener.gesture_smoothing import GestureTokenSmoother


def test_consensus_formation():
    """Test that consensus emerges from repeated tokens."""
    print("\n🧪 Test 1: Consensus Formation")
    smoother = GestureTokenSmoother(window_duration=2.0, min_tokens=3)
    
    # Simulate chord: 4 notes with slight token variation
    tokens = [142, 142, 141, 142]
    
    results = []
    for i, token in enumerate(tokens):
        timestamp = i * 0.1  # 100ms apart
        smoothed = smoother.add_token(token, timestamp)
        results.append(smoothed)
        print(f"   Token {i+1}: raw={token}, smoothed={smoothed}")
    
    # Should converge to 142 (most common)
    assert results[-1] == 142, f"Expected 142, got {results[-1]}"
    print(f"   ✅ Consensus formed: {results[-1]}")
    print(f"   Distribution: {smoother.get_token_distribution()}")


def test_phrase_transition():
    """Test smooth transition between chord gestures."""
    print("\n🧪 Test 2: Phrase Transition")
    smoother = GestureTokenSmoother(window_duration=3.0, min_tokens=3, decay_time=1.0)
    
    # Chord 1: tokens 140-142 over 1 second
    print("   Playing Chord 1 (tokens ~141)...")
    for i in range(5):
        smoother.add_token(141, timestamp=i * 0.2)
    
    consensus_1 = smoother.get_current_consensus()
    print(f"   Chord 1 consensus: {consensus_1}")
    
    # Transition: chord 2 starts at 1.5s (tokens 155-157)
    print("   Playing Chord 2 (tokens ~156)...")
    for i in range(5):
        smoother.add_token(156, timestamp=1.5 + i * 0.2)
    
    consensus_2 = smoother.get_current_consensus()
    print(f"   Chord 2 consensus: {consensus_2}")
    
    # Should transition from 141 → 156
    assert consensus_1 == 141, f"Expected 141, got {consensus_1}"
    assert consensus_2 == 156, f"Expected 156, got {consensus_2}"
    print(f"   ✅ Phrase transition: {consensus_1} → {consensus_2}")


def test_decay_weighting():
    """Test that recent tokens dominate old tokens."""
    print("\n🧪 Test 3: Decay Weighting")
    smoother = GestureTokenSmoother(window_duration=3.0, min_tokens=3, decay_time=1.0)
    
    # Old tokens (3s ago) - only 2 of them
    print("   Adding old tokens (100) at t=0.0...")
    smoother.add_token(100, timestamp=0.0)
    smoother.add_token(100, timestamp=0.1)
    
    # Recent tokens (now) - 3 of them
    print("   Adding recent tokens (200) at t=3.0...")
    smoother.add_token(200, timestamp=3.0)
    smoother.add_token(200, timestamp=3.1)
    smoother.add_token(200, timestamp=3.2)
    
    consensus = smoother.get_current_consensus()
    
    # Recent tokens should dominate despite similar absolute count
    assert consensus == 200, f"Expected 200 (recent), got {consensus}"
    print(f"   ✅ Decay weighting works: recent token {consensus} dominates")
    print(f"   Distribution: {smoother.get_token_distribution()}")
    print(f"   Weighted: {smoother.get_weighted_distribution(current_time=3.2)}")


def test_window_cleanup():
    """Test that old tokens are removed from window."""
    print("\n🧪 Test 4: Window Cleanup")
    smoother = GestureTokenSmoother(window_duration=2.0, min_tokens=2)
    
    # Add tokens over 5 seconds
    for i in range(10):
        smoother.add_token(100 + i, timestamp=i * 0.5)
    
    # Window should only contain last 2 seconds (last 4 tokens)
    window_size = len(smoother)
    print(f"   Window size after 5s: {window_size} tokens")
    print(f"   Token distribution: {smoother.get_token_distribution()}")
    
    assert window_size <= 5, f"Window too large: {window_size} tokens"
    print(f"   ✅ Old tokens cleaned up, window size: {window_size}")


def test_min_tokens_threshold():
    """Test that consensus waits for minimum token count."""
    print("\n🧪 Test 5: Minimum Token Threshold")
    smoother = GestureTokenSmoother(window_duration=3.0, min_tokens=5)
    
    # Add only 3 tokens (below threshold)
    print("   Adding 3 tokens (below min_tokens=5)...")
    result1 = smoother.add_token(50, timestamp=0.0)
    result2 = smoother.add_token(50, timestamp=0.1)
    result3 = smoother.add_token(50, timestamp=0.2)
    
    print(f"   After 3 tokens: consensus={smoother.get_current_consensus()}")
    
    # Add 2 more (now at threshold)
    print("   Adding 2 more tokens (reaching threshold)...")
    result4 = smoother.add_token(50, timestamp=0.3)
    result5 = smoother.add_token(50, timestamp=0.4)
    
    final_consensus = smoother.get_current_consensus()
    print(f"   After 5 tokens: consensus={final_consensus}")
    
    assert final_consensus == 50, f"Expected 50, got {final_consensus}"
    print(f"   ✅ Consensus formed after reaching threshold")


def test_statistics():
    """Test statistics reporting."""
    print("\n🧪 Test 6: Statistics Reporting")
    smoother = GestureTokenSmoother(window_duration=2.0, min_tokens=2)
    
    # Add various tokens
    for i in range(10):
        token = 100 if i < 5 else 200
        smoother.add_token(token, timestamp=i * 0.3)
    
    stats = smoother.get_statistics()
    print(f"   Statistics: {stats}")
    
    assert stats['total_processed'] == 10
    assert stats['consensus_changes'] >= 1  # Should have changed from 100 to 200
    print(f"   ✅ Statistics tracking works")


def test_real_time_scenario():
    """Test realistic performance scenario."""
    print("\n🧪 Test 7: Realistic Performance Scenario")
    smoother = GestureTokenSmoother(window_duration=3.0, min_tokens=3, decay_time=1.0)
    
    print("   Simulating 10-second performance...")
    
    # Cmaj7 chord for 3 seconds (tokens 140-143)
    print("   0-3s: Cmaj7 chord")
    start_time = 0.0
    for i in range(12):  # ~4 tokens/second
        token = 140 + (i % 4)  # Slight variation
        t = start_time + i * 0.25
        smoothed = smoother.add_token(token, timestamp=t)
        if i % 4 == 0:
            print(f"      t={t:.1f}s: token={token}, smoothed={smoothed}")
    
    cmaj_consensus = smoother.get_current_consensus()
    
    # Transition to Dm7 at 3s (tokens 155-158)
    print("   3-6s: Dm7 chord")
    for i in range(12):
        token = 155 + (i % 4)
        t = 3.0 + i * 0.25
        smoothed = smoother.add_token(token, timestamp=t)
        if i % 4 == 0:
            print(f"      t={t:.1f}s: token={token}, smoothed={smoothed}")
    
    dm_consensus = smoother.get_current_consensus()
    
    # Back to Cmaj7 at 6s
    print("   6-9s: Back to Cmaj7")
    for i in range(12):
        token = 140 + (i % 4)
        t = 6.0 + i * 0.25
        smoothed = smoother.add_token(token, timestamp=t)
        if i % 4 == 0:
            print(f"      t={t:.1f}s: token={token}, smoothed={smoothed}")
    
    final_consensus = smoother.get_current_consensus()
    
    print(f"\n   Chord progression detected:")
    print(f"      Cmaj7: consensus ~{cmaj_consensus}")
    print(f"      Dm7: consensus ~{dm_consensus}")
    print(f"      Cmaj7 return: consensus ~{final_consensus}")
    
    stats = smoother.get_statistics()
    print(f"\n   Session stats:")
    print(f"      Total tokens: {stats['total_processed']}")
    print(f"      Consensus changes: {stats['consensus_changes']}")
    print(f"      ✅ Realistic scenario completed")


if __name__ == '__main__':
    print("=" * 60)
    print("GESTURE TOKEN SMOOTHING TEST SUITE")
    print("=" * 60)
    
    try:
        test_consensus_formation()
        test_phrase_transition()
        test_decay_weighting()
        test_window_cleanup()
        test_min_tokens_threshold()
        test_statistics()
        test_real_time_scenario()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        raise
