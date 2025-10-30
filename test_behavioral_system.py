#!/usr/bin/env python3
"""
Test Behavioral System Functionality
Tests request parameters, mode transitions, and thematic recall end-to-end
"""

import sys
import os
import time
import numpy as np
from typing import Dict, List, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import components to test
from agent.behaviors import BehaviorEngine, BehaviorMode
from agent.phrase_generator import PhraseGenerator
from agent.phrase_memory import PhraseMemory
from memory.polyphonic_audio_oracle import PolyphonicAudioOracle

class MockVisualizationManager:
    """Mock visualization manager to track events"""
    def __init__(self):
        self.events = []
    
    def emit_mode_change(self, mode: str, duration: float, request_params: Dict, temperature: float):
        event = {
            'type': 'mode_change',
            'mode': mode,
            'duration': duration,
            'request_params': request_params,
            'temperature': temperature,
            'timestamp': time.time()
        }
        self.events.append(event)
        print(f"📊 MODE CHANGE: {mode} (duration: {duration:.1f}s, temp: {temperature})")
    
    def emit_timeline_update(self, event_type: str, mode: Optional[str] = None, timestamp: Optional[float] = None):
        event = {
            'type': 'timeline_update',
            'event_type': event_type,
            'mode': mode,
            'timestamp': timestamp or time.time()
        }
        self.events.append(event)
        print(f"📈 TIMELINE: {event_type} (mode: {mode})")
    
    def emit_request_params(self, mode: str, request: Dict, voice_type: str):
        event = {
            'type': 'request_params',
            'mode': mode,
            'request': request,
            'voice_type': voice_type,
            'timestamp': time.time()
        }
        self.events.append(event)
        print(f"🎯 REQUEST: {mode} -> {request} ({voice_type})")

def test_request_parameter_generation():
    """Test that behavioral modes generate proper request parameters"""
    print("\n🧪 TEST 1: Request Parameter Generation")
    
    viz_manager = MockVisualizationManager()
    phrase_generator = PhraseGenerator(rhythm_oracle=None, visualization_manager=viz_manager)
    
    # Test each mode
    modes = ['shadow', 'mirror', 'couple', 'imitate', 'contrast', 'lead']
    
    for mode in modes:
        print(f"\n--- Testing {mode.upper()} mode ---")
        request = phrase_generator._build_request_for_mode(mode)
        
        if request:
            print(f"✅ {mode}: {request}")
            # Verify request format compatibility with AudioOracle
            required_keys = ['parameter', 'type', 'value', 'weight']
            if all(key in request for key in required_keys):
                print(f"   ✅ Compatible format with AudioOracle")
            else:
                print(f"   ❌ Missing keys: {[k for k in required_keys if k not in request]}")
        else:
            print(f"❌ {mode}: No request generated")

def test_mode_transitions():
    """Test behavioral mode transition timing and logging"""
    print("\n🧪 TEST 2: Mode Transition Timing")
    
    viz_manager = MockVisualizationManager()
    behavior_engine = BehaviorEngine(rhythm_oracle=None, visualization_manager=viz_manager)
    
    # Create mock event data
    mock_event = {
        'timestamp': time.time(),
        'consonance': 0.5,
        'gesture_token': 142,
        'instrument': 'piano',
        'harmonic_context': {}
    }
    
    # Create minimal mock memory objects
    class MockMemoryBuffer:
        def get_recent_events(self, n=5):
            return [mock_event] * n
        
        def get_recent_moments(self, duration=10.0):
            return [mock_event] * 5
        
        def get_activity_level(self):
            return 0.5
    
    class MockClustering:
        def __init__(self):
            self.is_trained = True
            self.audio_frames = {}
        def generate_with_request(self, context, request=None, temperature=1.0, max_length=5):
            return [1, 2, 3, 4, 5]  # Mock frame IDs
    
    memory_buffer = MockMemoryBuffer()
    clustering = MockClustering()
    
    # Force mode change by setting old timestamp
    behavior_engine.mode_start_time = time.time() - 100  # 100 seconds ago
    initial_mode = behavior_engine.current_mode
    
    # Trigger decision (should change mode)
    _ = behavior_engine.decide_behavior(mock_event, memory_buffer, clustering)
    
    # Check if mode changed
    if behavior_engine.current_mode != initial_mode:
        print(f"✅ Mode changed: {initial_mode} -> {behavior_engine.current_mode}")
    else:
        print(f"❌ Mode did not change (still {initial_mode})")
    
    # Check visualization events
    mode_change_events = [e for e in viz_manager.events if e['type'] == 'mode_change']
    timeline_events = [e for e in viz_manager.events if e['type'] == 'timeline_update']
    
    print(f"✅ Mode change events: {len(mode_change_events)}")
    print(f"✅ Timeline events: {len(timeline_events)}")

def test_thematic_recall():
    """Test thematic recall system"""
    print("\n🧪 TEST 3: Thematic Recall System")
    
    viz_manager = MockVisualizationManager()
    phrase_memory = PhraseMemory(visualization_manager=viz_manager)
    
    # Add some phrases to memory
    test_phrases = [
        ([60, 62, 64, 67], [0.5, 0.5, 0.5, 1.0]),  # C, D, E, G
        ([60, 62, 64, 67], [0.25, 0.25, 0.5, 1.0]), # Same melody, different rhythm
        ([67, 64, 62, 60], [0.5, 0.5, 0.5, 1.0]),  # Retrograde
    ]
    
    for i, (notes, durations) in enumerate(test_phrases):
        phrase_memory.add_phrase(notes, durations, time.time() - (10 * i), f"mode_{i}")
        print(f"Added phrase {i+1}: {notes}")
    
    # Check motif extraction
    print(f"✅ Motifs stored: {len(phrase_memory.motifs)}")
    for motif_id, motif in phrase_memory.motifs.items():
        print(f"   Motif {motif_id}: {motif.notes} (occurrences: {motif.occurrence_count})")
    
    # Force recall time
    phrase_memory.next_recall_time = time.time() - 1  # Force recall
    
    # Test recall decision
    should_recall = phrase_memory.should_recall_theme()
    print(f"✅ Should recall theme: {should_recall}")
    
    if should_recall:
        theme = phrase_memory.get_current_theme()
        if theme:
            print(f"✅ Current theme: {theme.notes}")
            
            # Test variation
            variation = phrase_memory.get_variation(theme, 'transpose')
            print(f"✅ Transposed variation: {variation['notes']}")
        else:
            print("❌ No theme available")

def test_end_to_end_integration():
    """Test full behavioral system integration"""
    print("\n🧪 TEST 4: End-to-End Integration")
    
    viz_manager = MockVisualizationManager()
    
    # Create minimal AudioOracle
    oracle = PolyphonicAudioOracle(distance_threshold=0.15)
    
    # Add some mock audio frames
    for i in range(10):
        features = np.random.random(15)  # 15D features
        audio_data = {
            'consonance': np.random.random(),
            'gesture_token': np.random.randint(0, 256),
            'midi_relative': np.random.randint(-12, 12),
            'rhythm_ratio': np.random.random()
        }
        oracle.add_audio_frame(features, audio_data)
    
    # Create behavior system
    behavior_engine = BehaviorEngine(rhythm_oracle=None, visualization_manager=viz_manager)
    
    # Test request generation and AudioOracle query
    mock_event = {
        'timestamp': time.time(),
        'consonance': 0.7,
        'gesture_token': 142,
        'instrument': 'piano',
        'harmonic_context': {}
    }
    
    class MockMemoryBuffer:
        def get_recent_events(self, n=5):
            return [mock_event] * n
    
    memory_buffer = MockMemoryBuffer()
    
    # Generate decisions
    decisions = behavior_engine.decide_behavior(mock_event, memory_buffer, oracle)
    
    print(f"✅ Decisions generated: {len(decisions)}")
    for decision in decisions:
        print(f"   Mode: {decision.mode}, Voice: {decision.voice_type}, Confidence: {decision.confidence:.2f}")
    
    # Check request parameter events
    request_events = [e for e in viz_manager.events if e['type'] == 'request_params']
    print(f"✅ Request parameter events: {len(request_events)}")

def main():
    """Run all behavioral system tests"""
    print("🎵 MusicHal 9000 - Behavioral System Test Suite")
    print("=" * 50)
    
    try:
        test_request_parameter_generation()
        test_mode_transitions()
        test_thematic_recall()
        test_end_to_end_integration()
        
        print("\n" + "=" * 50)
        print("✅ ALL TESTS COMPLETED")
        print("\nNext steps:")
        print("1. Run live system and observe viewport for:")
        print("   - Mode changes every 30-90s")
        print("   - Request parameters varying by mode")
        print("   - Thematic recalls every 30-60s")
        print("2. Check logs/ directory for behavioral reasoning")
        print("3. Monitor consonance values changing with mode parameters")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()