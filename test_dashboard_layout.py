#!/usr/bin/env python3
"""
Test script to verify dashboard layout renders correctly without running full system.
Tests the visualization manager with mock data to ensure layout structure is correct.
"""

import sys
from PyQt5.QtWidgets import QApplication
from visualization.visualization_manager import VisualizationManager

def test_dashboard_layout():
    """Test that dashboard layout renders with correct structure"""
    
    print("Creating visualization manager (creates Qt app internally)...")
    viz_manager = VisualizationManager()
    
    # Verify viewports were created
    expected_viewports = [
        'status_bar',
        'pattern_matching', 
        'request_parameters',
        'audio_analysis',
        'rhythm_oracle',
        'performance_timeline',
        'performance_controls'
    ]
    
    print("\nChecking viewport creation:")
    for vp_id in expected_viewports:
        if vp_id in viz_manager.viewports:
            print(f"  ✓ {vp_id} created")
        else:
            print(f"  ✗ {vp_id} MISSING!")
    
    # Check that optional viewports were created but won't be displayed
    optional_viewports = ['phrase_memory', 'gpt_reflection', 'webcam']
    print("\nOptional viewports (created but hidden in dashboard):")
    for vp_id in optional_viewports:
        if vp_id in viz_manager.viewports:
            print(f"  ✓ {vp_id} created (hidden)")
        else:
            print(f"  - {vp_id} not created")
    
    # Send test data using event bus methods
    print("\nSending test data to viewports...")
    
    # Test mode change (goes to status_bar and request_parameters)
    viz_manager.emit_mode_change(
        mode='SHADOW',
        duration=45.0,
        request_params={'consonance': 0.8, 'gesture_token': 142},
        temperature=0.7
    )
    
    # Test audio analysis (goes to audio_analysis viewport)
    import numpy as np
    viz_manager.emit_audio_analysis(
        waveform=np.random.randn(1000),
        onset=True,
        ratio=[1.5, 2.0, 3.0],
        consonance=0.75,
        timestamp=0.0,
        complexity=0.5,
        chord_label='D minor',
        chord_confidence=0.87
    )
    
    # Test timeline update
    viz_manager.emit_timeline_update(
        event_type='PHASE_CHANGE',
        mode='SHADOW',
        timestamp=123.5
    )
    
    # Show the window
    print("\nShowing visualization window...")
    viz_manager.show()
    
    # Process events to render
    viz_manager.app.processEvents()
    
    print("\n" + "="*60)
    print("DASHBOARD LAYOUT TEST")
    print("="*60)
    print("\nLayout structure:")
    print("  Row 0 (10%):  Status Bar (full width)")
    print("  Row 1 (35%):  Audio Analysis | Pattern Matching")
    print("  Row 2 (35%):  Rhythm Oracle | Request Parameters")
    print("  Row 3 (15%):  Performance Timeline (full width)")
    print("  Row 4 (5%):   Performance Controls (full width)")
    print("\nExpected behavior:")
    print("  • Status bar shows behavior mode and phase")
    print("  • Audio analysis shows detected chord: D minor")
    print("  • Request parameters shows SHADOW mode")
    print("  • Layout fills window with proper proportions")
    print("\nClose the window to exit test...")
    print("="*60 + "\n")
    
    # Run event loop
    sys.exit(viz_manager.app.exec_())

if __name__ == '__main__':
    test_dashboard_layout()
