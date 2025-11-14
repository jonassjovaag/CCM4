#!/usr/bin/env python3
"""
Test 3: Verify Waypoint Extraction from Performance Arc

Tests that PerformanceTimelineManager correctly extracts root waypoints
from performance arc JSON files.
"""

import json
from performance_timeline_manager import PerformanceTimelineManager, PerformanceConfig


def test_waypoint_extraction():
    """Test waypoint extraction from arc"""
    
    print("="*70)
    print("  TEST 3: WAYPOINT EXTRACTION FROM PERFORMANCE ARC")
    print("="*70)
    
    # Create config pointing to example arc
    config = PerformanceConfig(
        duration_minutes=15,
        arc_file_path="performance_arcs/simple_root_progression.json",
        engagement_profile="balanced",
        silence_tolerance=5.0,
        adaptation_rate=0.5
    )
    
    print(f"\n📂 Loading arc: {config.arc_file_path}")
    
    # Initialize timeline manager
    manager = PerformanceTimelineManager(config)
    
    # Check if arc was loaded
    if not manager.scaled_arc:
        print("❌ Arc not loaded!")
        return False
    
    print(f"✅ Arc loaded: {len(manager.scaled_arc.phases)} phases")
    
    # Extract waypoints
    print("\n🎯 Extracting waypoints from phases...")
    waypoints = manager._extract_waypoints_from_phases()
    
    if not waypoints:
        print("❌ No waypoints extracted!")
        print("   Check if phases have 'root_hint_frequency' fields")
        
        # Debug: Show what's in the phases
        print("\n🔍 Phase inspection:")
        for i, phase in enumerate(manager.scaled_arc.phases):
            print(f"\nPhase {i+1}:")
            print(f"  Type: {phase.phase_type}")
            print(f"  Time: {phase.start_time}s - {phase.end_time}s")
            print(f"  Has root_hint_frequency: {hasattr(phase, 'root_hint_frequency')}")
            if hasattr(phase, 'root_hint_frequency'):
                print(f"  Value: {phase.root_hint_frequency}")
        return False
    
    print(f"✅ Extracted {len(waypoints)} waypoints\n")
    
    # Display waypoints
    print("="*70)
    print("  EXTRACTED WAYPOINTS")
    print("="*70)
    
    for i, wp in enumerate(waypoints):
        # Convert Hz to note name
        note_name = _hz_to_note_name(wp.root_hz)
        minutes = wp.time / 60.0
        
        print(f"\nWaypoint {i+1}:")
        print(f"  Time: {wp.time:.1f}s ({minutes:.1f} min)")
        print(f"  Root: {wp.root_hz:.2f} Hz ({note_name})")
        print(f"  Comment: {wp.comment}")
    
    # Verify waypoints match expected structure
    print("\n" + "="*70)
    print("  VERIFICATION")
    print("="*70)
    
    # Load original JSON to compare
    with open(config.arc_file_path, 'r') as f:
        arc_data = json.load(f)
    
    expected_count = sum(
        1 for phase in arc_data.get('phases', [])
        if 'root_hint_frequency' in phase and phase['root_hint_frequency'] is not None
    )
    
    print(f"\nExpected waypoints (from JSON): {expected_count}")
    print(f"Extracted waypoints: {len(waypoints)}")
    
    if len(waypoints) == expected_count:
        print("✅ Count matches!")
    else:
        print("❌ Count mismatch!")
        return False
    
    # Verify first and last waypoints
    if waypoints:
        first_wp = waypoints[0]
        last_wp = waypoints[-1]
        
        print(f"\nFirst waypoint: {first_wp.time}s @ {first_wp.root_hz:.2f} Hz")
        print(f"Last waypoint: {last_wp.time}s @ {last_wp.root_hz:.2f} Hz")
        
        # Check if times are in order
        for i in range(len(waypoints) - 1):
            if waypoints[i].time > waypoints[i+1].time:
                print(f"❌ Waypoints not in chronological order!")
                return False
        
        print("✅ Waypoints in chronological order")
    
    # Test timeline manager state
    print("\n" + "="*70)
    print("  TIMELINE MANAGER STATE")
    print("="*70)
    
    print(f"\nRoot explorer initialized: {manager.root_explorer is not None}")
    print(f"Last exploration time: {manager.last_root_exploration_time}")
    print(f"Exploration interval: {manager.exploration_interval}s")
    
    if manager.performance_state:
        print(f"\nPerformance state:")
        print(f"  Total duration: {manager.performance_state.total_duration}s")
        print(f"  Current time: {manager.performance_state.current_time}s")
        print(f"  Current root hint: {manager.performance_state.current_root_hint}")
        print(f"  Current tension: {manager.performance_state.current_tension_target}")
    
    print("\n" + "="*70)
    print("  ✅ TEST 3 PASSED: WAYPOINT EXTRACTION WORKING")
    print("="*70)
    
    return True


def _hz_to_note_name(hz: float) -> str:
    """Convert frequency to note name (approximate)"""
    import numpy as np
    
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    # A4 = 440 Hz is MIDI 69
    midi = 12 * np.log2(hz / 440.0) + 69
    note_class = int(round(midi)) % 12
    octave = int(round(midi)) // 12 - 1
    
    return f"{note_names[note_class]}{octave}"


if __name__ == "__main__":
    success = test_waypoint_extraction()
    
    if success:
        print("\n🎉 All checks passed!")
    else:
        print("\n⚠️  Some checks failed - see details above")
