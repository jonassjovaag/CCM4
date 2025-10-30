#!/usr/bin/env python3
"""Quick test to verify timeline manager works without arc file"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from performance_timeline_manager import PerformanceTimelineManager, PerformanceConfig
import time

# Test with 1 minute duration
config = PerformanceConfig(
    duration_minutes=0.5,  # 30 seconds for quick test
    arc_file_path="nonexistent.json",  # Doesn't exist
    engagement_profile="balanced",
    silence_tolerance=5.0,
    adaptation_rate=0.1
)

print("Creating timeline manager...")
manager = PerformanceTimelineManager(config)
print(f"✅ Timeline initialized")
print(f"Total duration: {manager.performance_state.total_duration}s")
print(f"Start time: {manager.performance_state.start_time}")

print("\nSimulating 30-second performance...")
for i in range(35):  # 35 seconds
    manager.update_performance_state(human_activity=False)
    
    remaining = manager.get_time_remaining()
    is_complete = manager.is_complete()
    elapsed = manager.performance_state.current_time
    
    if i % 5 == 0:  # Every 5 seconds
        print(f"t={elapsed:.1f}s | remaining={remaining:.1f}s | complete={is_complete}")
    
    if is_complete:
        print(f"\n✅ Timeline marked complete at {elapsed:.1f}s!")
        break
    
    time.sleep(1)

print(f"\nFinal state:")
print(f"  Elapsed: {manager.performance_state.current_time:.1f}s")
print(f"  Total: {manager.performance_state.total_duration:.1f}s")
print(f"  Complete: {manager.is_complete()}")
