#!/usr/bin/env python3
"""
Test the tuned gesture smoothing parameters
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from listener.dual_perception import DualPerceptionModule
import numpy as np
from collections import Counter

def test_gesture_smoothing():
    """Test the new smoothing parameters"""
    
    print("🧪 Testing tuned gesture smoothing parameters...")
    print("=" * 50)
    
    # Initialize with new tuned parameters
    dual_perception = DualPerceptionModule(
        vocabulary_size=64,
        enable_symbolic=True,
        enable_dual_vocabulary=False,
        # These should now use the new defaults: 1.5s window, 2 min tokens, 0.5s decay
    )
    
    print("\n📊 Current smoothing settings:")
    smoother = dual_perception.gesture_smoother
    print(f"   Window duration: {smoother.window_duration}s")
    print(f"   Min tokens: {smoother.min_tokens}")
    print(f"   Decay time: {smoother.decay_time}s")
    
    # Simulate rapid token changes
    print(f"\n🎯 Simulating rapid gesture token changes...")
    
    import time
    start_time = time.time()
    
    # Simulate 10 different tokens arriving quickly (like rapid musical changes)
    test_tokens = [5, 12, 5, 23, 12, 45, 23, 12, 30, 45, 30, 12]
    smoothed_tokens = []
    
    for i, token in enumerate(test_tokens):
        timestamp = start_time + i * 0.2  # 200ms apart (5 tokens/second)
        smoothed = smoother.add_token(token, timestamp)
        smoothed_tokens.append(smoothed)
        
        print(f"   {i+1:2d}. Raw: {token:2d} → Smoothed: {smoothed if smoothed else 'None'}")
    
    # Analyze results
    print(f"\n📈 Analysis:")
    raw_unique = len(set(test_tokens))
    smoothed_unique = len(set([t for t in smoothed_tokens if t is not None]))
    
    print(f"   Raw tokens unique: {raw_unique}/{len(test_tokens)}")
    print(f"   Smoothed tokens unique: {smoothed_unique}/{len([t for t in smoothed_tokens if t is not None])}")
    
    # Token distribution
    raw_counts = Counter(test_tokens)
    smoothed_counts = Counter([t for t in smoothed_tokens if t is not None])
    
    print(f"   Raw distribution: {dict(raw_counts)}")
    print(f"   Smoothed distribution: {dict(smoothed_counts)}")
    
    # Check responsiveness
    response_delay = 0
    for i, smoothed in enumerate(smoothed_tokens):
        if smoothed is not None:
            response_delay = i
            break
    
    print(f"   Responsiveness: First consensus after {response_delay} tokens ({response_delay * 0.2:.1f}s)")
    
    # Final smoother stats
    stats = smoother.get_statistics()
    print(f"\n📊 Final smoother statistics:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print(f"\n✅ Tuning assessment:")
    if response_delay <= 2:
        print("   ✅ Good responsiveness (consensus within 2 tokens)")
    else:
        print("   ⚠️  Slow responsiveness - consider reducing min_tokens further")
    
    if smoothed_unique >= raw_unique * 0.7:
        print("   ✅ Good diversity preservation")
    else:
        print("   ⚠️  Smoothing reducing diversity too much")
    
    return {
        'raw_unique': raw_unique,
        'smoothed_unique': smoothed_unique,
        'response_delay': response_delay,
        'stats': stats
    }

if __name__ == "__main__":
    test_gesture_smoothing()