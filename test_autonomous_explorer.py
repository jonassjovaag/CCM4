#!/usr/bin/env python3
"""
Test Autonomous Root Explorer Logic

Demonstrates how the hybrid intelligence system makes decisions
(without requiring a trained model yet)
"""

import numpy as np
from agent.autonomous_root_explorer import (
    AutonomousRootExplorer, 
    RootWaypoint, 
    ExplorationConfig
)


class MockAudioOracle:
    """Mock AudioOracle with sample harmonic data"""
    
    def __init__(self):
        # Simulate training data fundamentals (like Itzama.wav might have)
        # In key of A minor: A, C, D, E, F, G
        self.fundamentals = {
            0: 220.0,   # A3 - appears often (tonic)
            1: 264.0,   # C4 - appears very often (relative major)
            2: 294.0,   # D4 - appears often (subdominant)
            3: 330.0,   # E4 - appears often (dominant)
            4: 349.0,   # F4 - appears sometimes
            5: 196.0,   # G3 - appears sometimes
            10: 247.0,  # B3 - rare (leading tone)
            15: 440.0,  # A4 - octave
            20: 233.0,  # Bb3 - chromatic, very rare
        }
        
        # Consonance scores (higher = more consonant)
        self.consonances = {
            0: 0.9,   # A3 - very consonant (tonic)
            1: 0.85,  # C4 - consonant
            2: 0.75,  # D4 - fairly consonant
            3: 0.8,   # E4 - consonant
            4: 0.7,   # F4 - medium
            5: 0.65,  # G3 - medium
            10: 0.5,  # B3 - less consonant (leading tone tension)
            15: 0.95, # A4 - very consonant (octave)
            20: 0.3,  # Bb3 - dissonant (chromatic)
        }
        
        # Simulate state transition frequencies (how often each state appears)
        self.transitions = {
            (0, '0'): 0,  (0, '1'): 1,  (0, '2'): 2,
            (1, '0'): 1,  (1, '1'): 2,  (1, '2'): 3,
            (2, '0'): 2,  (2, '1'): 3,  (2, '2'): 0,
            (3, '0'): 3,  (3, '1'): 0,  (3, '2'): 1,
            # State 1 (C4) gets lots of transitions - it's very common
            (4, '0'): 1,  (5, '0'): 1,  (6, '0'): 1,  (7, '0'): 1,
            (8, '0'): 1,  (9, '0'): 1,  (10, '0'): 1, (11, '0'): 1,
            (12, '0'): 1, (13, '0'): 1, (14, '0'): 1,
        }
        
        self.states = {i: {} for i in range(20)}


def test_exploration_logic():
    """Test the core exploration decision-making"""
    
    print("="*70)
    print("  AUTONOMOUS ROOT EXPLORER - HYBRID INTELLIGENCE TEST")
    print("="*70)
    
    # Create mock AudioOracle
    audio_oracle = MockAudioOracle()
    
    # Define waypoints (A minor progression)
    waypoints = [
        RootWaypoint(time=0.0, root_hz=220.0, comment="Start on A3 (tonic)"),
        RootWaypoint(time=600.0, root_hz=330.0, comment="10 min: shift to E4 (dominant)"),
        RootWaypoint(time=1200.0, root_hz=220.0, comment="20 min: return to A3"),
    ]
    
    # Create explorer with hybrid config
    config = ExplorationConfig(
        training_weight=0.6,
        input_response_weight=0.3,
        theory_bonus_weight=0.1,
        max_drift_semitones=7,
        update_interval=60.0
    )
    
    explorer = AutonomousRootExplorer(audio_oracle, waypoints, config)
    
    print("\n" + "="*70)
    print("  SCENARIO 1: Early exploration (0-5 minutes)")
    print("  Anchor: A3 (220 Hz), No input from user")
    print("="*70)
    
    # Simulate several exploration decisions
    for i in range(5):
        elapsed_time = i * 60.0  # Every minute
        root = explorer.update(elapsed_time=elapsed_time, input_fundamental=None)
        
        interval = 12 * np.log2(root / 220.0)
        root_name = _hz_to_note_name(root)
        
        print(f"\n{elapsed_time:.0f}s: Chose {root:.1f} Hz ({root_name})")
        print(f"       Interval from anchor: {interval:+.1f} semitones")
        
        if explorer.exploration_history:
            last_decision = explorer.exploration_history[-1]
            print(f"       Reason: {last_decision['reason']}")
    
    print("\n" + "="*70)
    print("  SCENARIO 2: Response to live input")
    print("  Anchor: A3 (220 Hz), User playing C4 (264 Hz)")
    print("="*70)
    
    # Reset for new scenario
    explorer.last_update_time = 0
    
    for i in range(3):
        elapsed_time = 300.0 + i * 60.0
        # User is playing C4 - system should bias toward C
        root = explorer.update(elapsed_time=elapsed_time, input_fundamental=264.0)
        
        interval = 12 * np.log2(root / 220.0)
        root_name = _hz_to_note_name(root)
        
        print(f"\n{elapsed_time:.0f}s: Chose {root:.1f} Hz ({root_name})")
        print(f"       Interval from anchor: {interval:+.1f} semitones")
        print(f"       Input was: 264 Hz (C4)")
        
        if explorer.exploration_history:
            last_decision = explorer.exploration_history[-1]
            print(f"       Reason: {last_decision['reason']}")
    
    print("\n" + "="*70)
    print("  SCENARIO 3: Transition to next waypoint")
    print("  Approaching E4 waypoint at 600s (10 min)")
    print("="*70)
    
    # Simulate times approaching waypoint
    times = [480, 540, 570, 590, 600, 610]  # 2 min before → at → after
    for elapsed_time in times:
        root = explorer.update(elapsed_time=elapsed_time, input_fundamental=None)
        
        root_name = _hz_to_note_name(root)
        
        print(f"\n{elapsed_time:.0f}s: Target root = {root:.1f} Hz ({root_name})")
        
        if explorer.exploration_history:
            last_decision = explorer.exploration_history[-1]
            print(f"       Reason: {last_decision['reason']}")
    
    print("\n" + "="*70)
    print("  EXPLORATION SUMMARY")
    print("="*70)
    
    summary = explorer.get_exploration_summary()
    print(f"\nTotal decisions: {summary.get('total_decisions', 0)}")
    print(f"Mean interval from anchor: {summary.get('mean_interval', 0):+.1f} semitones")
    print(f"Max drift: {summary.get('max_interval', 0):.1f} semitones")
    print(f"Input responses: {summary.get('input_responses', 0)}")
    print(f"Explorations: {summary.get('explorations', 0)}")
    print(f"Transitions: {summary.get('transitions', 0)}")
    
    print("\nRecent roots (last 10):")
    for root_hz in summary.get('recent_roots', []):
        print(f"  {root_hz:.1f} Hz ({_hz_to_note_name(root_hz)})")
    
    print("\n" + "="*70)
    print("  ✅ HYBRID INTELLIGENCE DEMONSTRATION COMPLETE")
    print("="*70)
    print("\nKey observations:")
    print("  • System explores around waypoints (not stuck on one note)")
    print("  • Responds to live input (biases toward what you play)")
    print("  • Smoothly transitions between waypoints")
    print("  • Prefers roots from training data (C, D, E appear often)")
    print("  • Avoids extreme drift (stays within ±7 semitones)")
    print("  • Transparent reasoning for each decision")


def _hz_to_note_name(hz: float) -> str:
    """Convert frequency to note name (approximate)"""
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    # A4 = 440 Hz is MIDI 69
    midi = 12 * np.log2(hz / 440.0) + 69
    note_class = int(round(midi)) % 12
    octave = int(round(midi)) // 12 - 1
    
    return f"{note_names[note_class]}{octave}"


if __name__ == "__main__":
    test_exploration_logic()
