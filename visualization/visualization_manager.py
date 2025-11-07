#!/usr/bin/env python3
"""
Visualization Manager
Main coordinator for multi-viewport system
"""

from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QGridLayout
from PyQt5.QtCore import Qt, QThread, QTimer, QMetaObject, Q_ARG
from typing import Dict, List, Optional, Any
import sys

from .layout_manager import LayoutManager, ViewportPosition
from .event_bus import VisualizationEventBus
from .pattern_match_viewport import PatternMatchViewport
from .request_params_viewport import RequestParamsViewport
from .phrase_memory_viewport import PhraseMemoryViewport
from .audio_analysis_viewport import AudioAnalysisViewport
from .rhythm_oracle_viewport import RhythmOracleViewport
from .timeline_viewport import TimelineViewport
from .webcam_viewport import WebcamViewport
from .gpt_reflection_viewport import GPTReflectionViewport
from .performance_controls_viewport import PerformanceControlsViewport


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
                'rhythm_oracle',
                'performance_timeline',
                'webcam',
                'gpt_reflection'
            ]
        
        self.viewports_config = viewports_config
        self.viewports: Dict[str, Any] = {}
        self.main_window = None
        
        # Initialize viewports
        self._create_viewports()
        self._connect_event_bus()
        self._create_fullscreen_container()
    
    def _create_viewports(self):
        """Create viewport instances based on configuration"""
        viewport_classes = {
            'pattern_matching': PatternMatchViewport,
            'request_parameters': RequestParamsViewport,
            'phrase_memory': PhraseMemoryViewport,
            'audio_analysis': AudioAnalysisViewport,
            'rhythm_oracle': RhythmOracleViewport,
            'performance_timeline': TimelineViewport,
            'performance_controls': PerformanceControlsViewport,
            'webcam': WebcamViewport,
            'gpt_reflection': GPTReflectionViewport
        }
        
        for viewport_id in self.viewports_config:
            if viewport_id in viewport_classes:
                viewport_class = viewport_classes[viewport_id]
                self.viewports[viewport_id] = viewport_class()
                print(f"✅ Created viewport: {viewport_id}")
            else:
                print(f"⚠️  Unknown viewport ID: {viewport_id}")
    
    def _create_fullscreen_container(self):
        """Create single fullscreen window containing all viewports in grid layout"""
        self.main_window = QMainWindow()
        self.main_window.setWindowTitle("MusicHal 9000 - Performance Visualization")
        self.main_window.setWindowState(Qt.WindowFullScreen)
        
        # Create central widget with grid layout
        central_widget = QWidget()
        grid_layout = QGridLayout()
        grid_layout.setSpacing(10)  # Padding between viewports
        grid_layout.setContentsMargins(20, 20, 20, 20)  # Margin around edges
        
        # Arrange viewports in 3-column layout
        # Column 1 (3 rows @ 33% each): pattern_matching, request_parameters, phrase_memory
        # Column 2 (3 rows @ 33% each): audio_analysis, rhythm_oracle (NEW), performance_timeline
        # Column 3 (3 rows): gpt_reflection (spans rows 1-2 @ 66%), webcam (row 3 @ 33%)
        
        viewport_positions = {
            'pattern_matching': (0, 0, 1, 1),      # Col 1, Row 1
            'request_parameters': (1, 0, 1, 1),    # Col 1, Row 2
            'phrase_memory': (2, 0, 1, 1),         # Col 1, Row 3
            'audio_analysis': (0, 1, 1, 1),        # Col 2, Row 1
            'rhythm_oracle': (1, 1, 1, 1),         # Col 2, Row 2 (NEW - RhythmOracle in middle!)
            'performance_timeline': (2, 1, 1, 1),  # Col 2, Row 3
            'gpt_reflection': (0, 2, 2, 1),        # Col 3, Rows 1-2 (spans 2 rows for 66% height)
            'webcam': (2, 2, 1, 1)                 # Col 3, Row 3
        }
        
        for viewport_id, (row, col, rowspan, colspan) in viewport_positions.items():
            if viewport_id in self.viewports:
                viewport = self.viewports[viewport_id]
                # Remove fixed size - let widgets fill their grid cells
                viewport.setMinimumSize(400, 300)  # Set minimum size instead
                grid_layout.addWidget(viewport, row, col, rowspan, colspan)
        
        # Set column stretches (all equal width - 3 columns)
        grid_layout.setColumnStretch(0, 1)
        grid_layout.setColumnStretch(1, 1)
        grid_layout.setColumnStretch(2, 1)
        
        # Set row stretches for proper proportions:
        # Row 0: 33% height
        # Row 1: 33% height  
        # Row 2: 34% height
        grid_layout.setRowStretch(0, 33)
        grid_layout.setRowStretch(1, 33)
        grid_layout.setRowStretch(2, 34)
        
        central_widget.setLayout(grid_layout)
        self.main_window.setCentralWidget(central_widget)
        
        print("✅ Created fullscreen container with 3-column grid layout (Col1: 3 rows | Col2: 3 rows | Col3: 2+1 rows)")
    
    def show(self):
        """Show the main visualization window"""
        if self.main_window:
            self.main_window.show()
            print("🎨 Visualization window displayed")
    
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
        
        # RhythmOracle viewport (listens to rhythm_oracle_signal for timing engine data)
        if 'rhythm_oracle' in self.viewports:
            self.event_bus.rhythm_oracle_signal.connect(
                self.viewports['rhythm_oracle'].update_data,
                Qt.QueuedConnection
            )
        
        # Timeline viewport
        if 'performance_timeline' in self.viewports:
            self.event_bus.timeline_update_signal.connect(
                self.viewports['performance_timeline'].update_data,
                Qt.QueuedConnection
            )
        
        # GPT Reflection viewport
        if 'gpt_reflection' in self.viewports:
            self.event_bus.gpt_reflection_signal.connect(
                self.viewports['gpt_reflection'].update_data,
                Qt.QueuedConnection
            )
        
        print("✅ Connected viewports to event bus (with queued connections for thread safety)")
        
        # TEST: Immediately emit test signals to verify connections work
        print("🧪 Testing signal connections...")
        self.event_bus.pattern_match_signal.emit({'gesture_token': 999, 'score': 99.9, 'state_id': 1})
        self.event_bus.audio_analysis_signal.emit({'waveform': None, 'onset': True, 'consonance': 0.99})
        self.app.processEvents()  # Force immediate processing
        print("🧪 Test signals emitted and processed")
    
    def start(self):
        """
        Start the visualization system (non-blocking)
        
        Shows the main window and returns Qt application for event loop integration
        """
        # Show the main window
        self.show()
        
        print("\n🎨 Visualization system started!")
        print("💡 Viewports are now receiving events...")
        return self.app
    
    def close(self):
        """Close all viewports and cleanup (THREAD-SAFE)"""
        # If we're not on the Qt main thread, schedule close on main thread
        if QThread.currentThread() != self.app.thread():
            # Use QTimer.singleShot to safely execute close on main thread
            QTimer.singleShot(0, self._close_viewports)
        else:
            # Already on main thread, close directly
            self._close_viewports()
    
    def _close_viewports(self):
        """Internal method to actually close viewports (must be on main thread)"""
        if self.main_window:
            try:
                self.main_window.close()
                print("🎨 Visualization window closed")
            except Exception as e:
                print(f"⚠️  Error closing visualization window: {e}")
        else:
            # Fallback: close individual viewports if no main window
            for viewport in self.viewports.values():
                try:
                    viewport.close()
                except Exception as e:
                    print(f"⚠️  Error closing viewport: {e}")
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
                           timestamp: Optional[float] = None, complexity: Optional[float] = None,
                           gesture_token: Optional[int] = None, raw_gesture_token: Optional[int] = None,
                           chord_label: Optional[str] = None, chord_confidence: Optional[float] = None):
        """Emit audio analysis event"""
        self.event_bus.emit_audio_analysis(waveform, onset, ratio, consonance, timestamp, complexity,
                                          gesture_token, raw_gesture_token, chord_label, chord_confidence)
    
    def emit_timeline_update(self, event_type: str, mode: Optional[str] = None, 
                            timestamp: Optional[float] = None):
        """Emit timeline event"""
        self.event_bus.emit_timeline_update(event_type, mode, timestamp)
    
    def emit_rhythm_oracle(self, pattern_id: str, tempo: float, density: float,
                          similarity: float, duration_pattern: str, pulse: int,
                          syncopation: float, timestamp: Optional[float] = None):
        """Emit RhythmOracle pattern matching event"""
        self.event_bus.emit_rhythm_oracle(pattern_id, tempo, density, similarity,
                                          duration_pattern, pulse, syncopation, timestamp)
    
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

