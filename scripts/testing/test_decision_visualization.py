#!/usr/bin/env python3
"""
Test that decision methods emit visualization events correctly
"""

import sys
import numpy as np
from agent.behaviors import BehaviorEngine, BehaviorMode
from memory.memory_buffer import MemoryBuffer


class MockVisualizationManager:
    """Mock visualization manager to capture events"""
    def __init__(self):
        self.request_params_calls = []
        self.mode_change_calls = []
    
    def update_request_params(self, mode, request_params):
        """Capture request parameter updates"""
        print(f"📊 MockVisualizationManager received: mode={mode}, params={request_params}")
        self.request_params_calls.append({
            'mode': mode,
            'request_params': request_params
        })
    
    def emit_mode_change(self, mode, duration, request_params, reasoning="", temperature=0.8):
        """Capture mode changes"""
        self.mode_change_calls.append({
            'mode': mode,
            'duration': duration,
            'request_params': request_params,
            'reasoning': reasoning,
            'temperature': temperature
        })


def test_decision_visualization():
    """Test that decisions emit visualization events"""
    print("=" * 80)
    print("Testing Decision Visualization Event Emission")
    print("=" * 80)
    
    # Create mock visualization manager
    viz_manager = MockVisualizationManager()
    
    # Create behavior engine with visualization
    engine = BehaviorEngine(rhythm_oracle=None, visualization_manager=viz_manager)
    
    # Create memory buffer
    memory_buffer = MemoryBuffer(max_duration_seconds=180.0)
    
    # Create mock event with harmonic context
    mock_event = {
        'centroid': 1000.0,
        'rms_db': -20.0,
        'cents': 0.0,
        'ioi': 0.5,
        'instrument': 'piano',
        'harmonic_context': {
            'current_chord': 'Cmaj7',
            'key_signature': 'C major',
            'stability': 0.85,
            'chord_root': 60,
            'scale_degrees': [0, 2, 4, 5, 7, 9, 11]
        }
    }
    
    # Test 1: Imitate decision
    print("\n" + "=" * 80)
    print("Test 1: Imitate Decision")
    print("=" * 80)
    decision = engine._imitate_decision(mock_event, memory_buffer, None, "melodic")
    print(f"✅ Created imitate decision: mode={decision.mode.value}, confidence={decision.confidence}")
    print(f"✅ Visualization calls: {len(viz_manager.request_params_calls)}")
    if viz_manager.request_params_calls:
        print(f"✅ Last call: {viz_manager.request_params_calls[-1]}")
    
    # Test 2: Contrast decision
    print("\n" + "=" * 80)
    print("Test 2: Contrast Decision")
    print("=" * 80)
    decision = engine._contrast_decision(mock_event, memory_buffer, None, "melodic")
    print(f"✅ Created contrast decision: mode={decision.mode.value}, confidence={decision.confidence}")
    print(f"✅ Visualization calls: {len(viz_manager.request_params_calls)}")
    if len(viz_manager.request_params_calls) >= 2:
        print(f"✅ Last call: {viz_manager.request_params_calls[-1]}")
    
    # Test 3: Lead decision
    print("\n" + "=" * 80)
    print("Test 3: Lead Decision")
    print("=" * 80)
    decision = engine._lead_decision(mock_event, memory_buffer, None, "melodic")
    print(f"✅ Created lead decision: mode={decision.mode.value}, confidence={decision.confidence}")
    print(f"✅ Visualization calls: {len(viz_manager.request_params_calls)}")
    if len(viz_manager.request_params_calls) >= 3:
        print(f"✅ Last call: {viz_manager.request_params_calls[-1]}")
    
    # Verify all calls
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total visualization events emitted: {len(viz_manager.request_params_calls)}")
    
    if len(viz_manager.request_params_calls) == 3:
        print("✅ SUCCESS: All three decision methods emitted visualization events")
        
        # Check parameter structure
        all_valid = True
        for i, call in enumerate(viz_manager.request_params_calls):
            params = call['request_params']
            if not params:
                print(f"❌ Call {i+1}: No parameters")
                all_valid = False
                continue
            
            # Check each parameter has required fields
            for param in params:
                if not all(k in param for k in ['parameter', 'type', 'value', 'weight']):
                    print(f"❌ Call {i+1}: Missing required fields in {param}")
                    all_valid = False
                    break
            
            if all_valid:
                print(f"✅ Call {i+1}: {len(params)} parameters, all valid structure")
        
        if all_valid:
            print("\n🎉 ALL TESTS PASSED! Decision methods emit proper visualization events")
            return True
        else:
            print("\n❌ TESTS FAILED: Invalid parameter structure")
            return False
    else:
        print(f"❌ TESTS FAILED: Expected 3 visualization events, got {len(viz_manager.request_params_calls)}")
        return False


if __name__ == "__main__":
    success = test_decision_visualization()
    sys.exit(0 if success else 1)
