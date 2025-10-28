#!/usr/bin/env python3
"""
Visualization Manager
Main coordinator for multi-viewport system
"""

from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout
from PyQt5.QtCore import Qt, QThread
from typing import Dict, List, Optional, Any
import sys

from .layout_manager import LayoutManager, ViewportPosition
from .event_bus import VisualizationEventBus
from .pattern_match_viewport import PatternMatchViewport
from .request_params_viewport import RequestParamsViewport
from .phrase_memory_viewport import PhraseMemoryViewport
from .audio_analysis_viewport import AudioAnalysisViewport
from .timeline_viewport import TimelineViewport


class VisualizationManager:
    """
    Main coordinator for multi-viewport visualization system
    
    Responsibilities:
    - Initialize Qt application
    - Create and position viewports using LayoutManager
    - Connect viewports to VisualizationEventBus
    - Provide API for MusicHal_9000 to emit events
    """
    
    def __init__(self, 
                 viewports_config: Optional[List[str]] = None,
                 padding: int = 10,
                 margin: int = 20):
        """
        Initialize visualization manager
        
        Args:
            viewports_config: List of viewport IDs to display
                             Default: all 5 essential viewports
            padding: Padding between viewports in pixels
            margin: Margin around screen edges in pixels
        """
        # Create Qt application
        self.app = QApplication.instance()
        if self.app is None:
            self.app = QApplication(sys.argv)
        
        # Initialize components
        self.layout_manager = LayoutManager(padding=padding, margin=margin)
        self.event_bus = VisualizationEventBus()
        
        # Default viewport configuration
        if viewports_config is None:
            viewports_config = [
                'pattern_matching',
                'request_parameters',
                'phrase_memory',
                'audio_analysis',
                'performance_timeline'
            ]
        
        self.viewports_config = viewports_config
        self.viewports: Dict[str, Any] = {}
        self.main_window = None
        
        # Initialize viewports
        self._create_viewports()
        self._connect_event_bus()
        self._arrange_viewports()
    
    def _create_viewports(self):
        """Create viewport instances based on configuration"""
        viewport_classes = {
            'pattern_matching': PatternMatchViewport,
            'request_parameters': RequestParamsViewport,
            'phrase_memory': PhraseMemoryViewport,
            'audio_analysis': AudioAnalysisViewport,
            'performance_timeline': TimelineViewport
        }
        
        for viewport_id in self.viewports_config:
            if viewport_id in viewport_classes:
                viewport_class = viewport_classes[viewport_id]
                self.viewports[viewport_id] = viewport_class()
                print(f"✅ Created viewport: {viewport_id}")
            else:
                print(f"⚠️  Unknown viewport ID: {viewport_id}")
    
    def _connect_event_bus(self):
        """Connect viewports to event bus signals"""
        from PyQt5.QtCore import Qt
        
        # Pattern matching viewport (FORCE queued connection for thread safety)
        if 'pattern_matching' in self.viewports:
            self.event_bus.pattern_match_signal.connect(
                self.viewports['pattern_matching'].update_data,
                Qt.QueuedConnection  # CRITICAL: Forces cross-thread delivery
            )
        
        # Request parameters viewport
        if 'request_parameters' in self.viewports:
            self.event_bus.mode_change_signal.connect(
                self.viewports['request_parameters'].update_data,
                Qt.QueuedConnection
            )
        
        # Phrase memory viewport
        if 'phrase_memory' in self.viewports:
            self.event_bus.phrase_memory_signal.connect(
                self.viewports['phrase_memory'].update_data,
                Qt.QueuedConnection
            )
        
        # Audio analysis viewport
        if 'audio_analysis' in self.viewports:
            self.event_bus.audio_analysis_signal.connect(
                self.viewports['audio_analysis'].update_data,
                Qt.QueuedConnection
            )
        
        # Timeline viewport
        if 'performance_timeline' in self.viewports:
            self.event_bus.timeline_update_signal.connect(
                self.viewports['performance_timeline'].update_data,
                Qt.QueuedConnection
            )
        
        print("✅ Connected viewports to event bus (with queued connections for thread safety)")
        
        # TEST: Immediately emit test signals to verify connections work
        print("🧪 Testing signal connections...")
        self.event_bus.pattern_match_signal.emit({'gesture_token': 999, 'score': 99.9, 'state_id': 1})
        self.event_bus.audio_analysis_signal.emit({'waveform': None, 'onset': True, 'consonance': 0.99})
        self.app.processEvents()  # Force immediate processing
        print("🧪 Test signals emitted and processed")
    
    def _arrange_viewports(self):
        """Arrange viewports using layout manager"""
        # Get screen dimensions
        screen_width, screen_height = self.layout_manager.get_screen_dimensions()
        
        # Calculate layout
        viewport_ids = list(self.viewports.keys())
        positions = self.layout_manager.calculate_layout(
            num_viewports=len(viewport_ids),
            screen_width=screen_width,
            screen_height=screen_height,
            viewport_ids=viewport_ids
        )
        
        # Position and show viewports
        for pos in positions:
            viewport = self.viewports[pos.viewport_id]
            viewport.setGeometry(pos.x, pos.y, pos.width, pos.height)
            viewport.show()
        
        print(f"✅ Arranged {len(positions)} viewports")
        print(self.layout_manager.format_layout_info(positions))
    
    def start(self):
        """
        Start the visualization system (non-blocking)
        
        Returns Qt application for event loop integration
        """
        print("\n🎨 Visualization system started!")
        print("💡 Viewports are now receiving events...")
        return self.app
    
    def close(self):
        """Close all viewports and cleanup"""
        for viewport in self.viewports.values():
            viewport.close()
        print("🎨 Visualization system closed")
    
    # ===== API for MusicHal_9000 =====
    # These methods are called from the main MusicHal thread
    
    def emit_pattern_match(self, score: float, state_id: int, gesture_token: int, context: Optional[Dict] = None):
        """Emit pattern matching event"""
        self.event_bus.emit_pattern_match(score, state_id, gesture_token, context)
    
    def emit_mode_change(self, mode: str, duration: float, request_params: Dict, temperature: float):
        """Emit behavior mode change event"""
        self.event_bus.emit_mode_change(mode, duration, request_params, temperature)
    
    def emit_request_params(self, mode: str, request: Dict, voice_type: str):
        """Emit request parameters event (for real-time phrase generation)"""
        self.event_bus.emit_request_params(mode, request, voice_type)
    
    def emit_phrase_memory(self, action: str, motif: Optional[List] = None, 
                          variation_type: Optional[str] = None, timestamp: Optional[float] = None):
        """Emit phrase memory event"""
        self.event_bus.emit_phrase_memory(action, motif, variation_type, timestamp)
    
    def emit_audio_analysis(self, waveform: Optional[Any] = None, onset: bool = False,
                           ratio: Optional[List] = None, consonance: Optional[float] = None,
                           timestamp: Optional[float] = None, complexity: Optional[float] = None):
        """Emit audio analysis event"""
        self.event_bus.emit_audio_analysis(waveform, onset, ratio, consonance, timestamp, complexity)
    
    def emit_timeline_update(self, event_type: str, mode: Optional[str] = None, 
                            timestamp: Optional[float] = None):
        """Emit timeline event"""
        self.event_bus.emit_timeline_update(event_type, mode, timestamp)
    
    def process_events(self):
        """Process pending Qt events (call from main loop)"""
        self.app.processEvents()


def main():
    """Test the complete visualization system"""
    print("🧪 Testing Complete Visualization System\n")
    
    # Create manager
    manager = VisualizationManager()
    app = manager.start()
    
    print("\n✅ Visualization manager created")
    print("✅ All viewports displayed")
    print("\n🧪 Simulating events...\n")
    
    import time
    
    # Simulate a realistic session
    test_sequence = [
        # Start with SHADOW mode
        lambda: manager.emit_mode_change('SHADOW', 47.0, {
            'primary': {'parameter': 'gesture_token', 'type': '==', 'value': 142, 'weight': 0.95}
        }, 0.7),
        lambda: manager.emit_timeline_update('mode_change', 'SHADOW'),
        
        # Human input
        lambda: manager.emit_audio_analysis(onset=True, ratio=[3, 2], consonance=0.75),
        lambda: manager.emit_pattern_match(87.5, 234, 142),
        lambda: manager.emit_timeline_update('human_input'),
        
        # Machine response
        lambda: manager.emit_timeline_update('response'),
        
        # Store motif
        lambda: manager.emit_phrase_memory('store', [65, 67, 69, 67], timestamp=45.2),
        
        # More interaction
        lambda: manager.emit_pattern_match(72.3, 267, 143),
        lambda: manager.emit_audio_analysis(onset=True, ratio=[4, 3], consonance=0.62),
        
        # Recall theme
        lambda: manager.emit_phrase_memory('update_probability', timestamp=90.0),
        lambda: manager.emit_phrase_memory('recall', [65, 67, 69, 67], timestamp=98.7),
        lambda: manager.emit_timeline_update('thematic_recall'),
        
        # Switch to COUPLE mode
        lambda: manager.emit_mode_change('COUPLE', 52.0, {
            'primary': {'parameter': 'consonance', 'type': 'gradient', 'value': 3.0, 'weight': 0.8}
        }, 1.2),
        lambda: manager.emit_timeline_update('mode_change', 'COUPLE'),
        
        # Continue interaction
        lambda: manager.emit_pattern_match(45.8, 289, 144),
        lambda: manager.emit_audio_analysis(onset=False, ratio=[5, 4], consonance=0.45),
        
        # Apply variation
        lambda: manager.emit_phrase_memory('variation', [72, 74, 76, 74], 'transpose', timestamp=105.4),
    ]
    
    # Execute test sequence
    for i, action in enumerate(test_sequence):
        print(f"Event {i+1}/{len(test_sequence)}")
        action()
        app.processEvents()
        time.sleep(1.5)
    
    print("\n✅ Test sequence complete!")
    print("💡 Viewports are updating in real-time")
    print("🎨 Close any viewport window to exit...\n")
    
    # Run Qt event loop
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

