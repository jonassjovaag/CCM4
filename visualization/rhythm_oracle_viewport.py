"""
RhythmOracle Viewport - Real-time display of RhythmOracle pattern matching
Shows selected patterns, tempo, density, similarity scores, and duration patterns
"""

from PyQt5.QtWidgets import QVBoxLayout, QLabel, QFrame
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui import QFont
from typing import Dict, Any
from .base_viewport import BaseViewport


class RhythmOracleViewport(BaseViewport):
    """
    Viewport for displaying RhythmOracle pattern matching data
    Shows which rhythmic patterns are being selected and their characteristics
    """
    
    def __init__(self):
        super().__init__(
            viewport_id="rhythm_oracle",
            title="🥁 RhythmOracle",
            update_rate_ms=100
        )
        self._setup_content()
        
    def _setup_content(self):
        """Set up the viewport UI"""
        content_layout = QVBoxLayout()
        self.content_widget.setLayout(content_layout)
        
        content_layout.setSpacing(5)
        content_layout.setContentsMargins(10, 5, 10, 5)
        
        # Pattern ID label (smaller font)
        self.pattern_label = QLabel("Pattern: —")
        self.pattern_label.setFont(QFont("Monaco", 11, QFont.Bold))
        self.pattern_label.setStyleSheet("color: #00ff88; padding: 2px;")
        content_layout.addWidget(self.pattern_label)
        
        # Duration pattern display (smaller)
        self.duration_label = QLabel("Duration: —")
        self.duration_label.setFont(QFont("Monaco", 9))
        self.duration_label.setStyleSheet("color: #88ddff; padding: 1px;")
        self.duration_label.setWordWrap(True)
        content_layout.addWidget(self.duration_label)
        
        # Tempo and density (compact)
        metrics_frame = QFrame()
        metrics_frame.setStyleSheet("background-color: #0f0f1e; border-radius: 3px; padding: 5px;")
        metrics_layout = QVBoxLayout()
        metrics_layout.setSpacing(2)
        metrics_layout.setContentsMargins(5, 5, 5, 5)
        
        self.tempo_label = QLabel("Tempo: —")
        self.tempo_label.setFont(QFont("Monaco", 9))
        self.tempo_label.setStyleSheet("color: #ffaa00;")
        metrics_layout.addWidget(self.tempo_label)
        
        self.density_label = QLabel("Density: —")
        self.density_label.setFont(QFont("Monaco", 9))
        self.density_label.setStyleSheet("color: #ff66dd;")
        metrics_layout.addWidget(self.density_label)
        
        self.pulse_label = QLabel("Pulse: —")
        self.pulse_label.setFont(QFont("Monaco", 9))
        self.pulse_label.setStyleSheet("color: #66ddff;")
        metrics_layout.addWidget(self.pulse_label)
        
        self.syncopation_label = QLabel("Syncopation: —")
        self.syncopation_label.setFont(QFont("Monaco", 9))
        self.syncopation_label.setStyleSheet("color: #dd88ff;")
        metrics_layout.addWidget(self.syncopation_label)
        
        metrics_frame.setLayout(metrics_layout)
        content_layout.addWidget(metrics_frame)
        
        # Similarity score (smaller)
        self.similarity_label = QLabel("Match: —")
        self.similarity_label.setFont(QFont("Monaco", 11, QFont.Bold))
        self.similarity_label.setStyleSheet("color: #00ff88; padding: 2px;")
        self.similarity_label.setAlignment(Qt.AlignCenter)
        content_layout.addWidget(self.similarity_label)
        
        # Status message (smaller)
        self.status_label = QLabel("Waiting for RhythmOracle...")
        self.status_label.setFont(QFont("Monaco", 8))
        self.status_label.setStyleSheet("color: #666666; font-style: italic;")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setWordWrap(True)
        content_layout.addWidget(self.status_label)
        
        content_layout.addStretch()
    
    @pyqtSlot(dict)
    def update_data(self, data: Dict[str, Any]):
        """
        Update viewport with new RhythmOracle data
        
        RhythmOracle is a TIMING ENGINE that provides rhythmic phrasing
        within behavioral modes (MIRROR/CONTRAST/LEAD).
        
        Args:
            data: Dictionary with keys:
                - pattern_id: ID of matched pattern
                - tempo: Current estimated tempo in BPM
                - density: Pattern density (events per second)
                - similarity: Pattern match similarity (0.0-1.0)
                - duration_pattern: String representation of pattern
                - pulse: Detected pulse subdivision
                - syncopation: Syncopation score (0.0-1.0)
                - timestamp: Event timestamp
        """
        try:
            print("🎨 RhythmOracleViewport: Updating with RhythmOracle data!")
            
            # Update pattern ID
            pattern_id = data.get('pattern_id', '—')
            self.pattern_label.setText(f"Pattern: {pattern_id}")
            
            # Update duration pattern
            duration_pattern = data.get('duration_pattern', '—')
            self.duration_label.setText(f"Duration: {duration_pattern}")
            
            # Update tempo
            tempo = data.get('tempo', 0)
            self.tempo_label.setText(f"Tempo: {tempo:.1f} BPM")
            
            # Update density
            density = data.get('density', 0)
            self.density_label.setText(f"Density: {density:.2f} events/sec")
            
            # Update pulse
            pulse = data.get('pulse', 0)
            self.pulse_label.setText(f"Pulse: {pulse}")
            
            # Update syncopation
            syncopation = data.get('syncopation', 0)
            self.syncopation_label.setText(f"Syncopation: {syncopation:.2f}")
            
            # Update similarity score
            similarity = data.get('similarity', 0)
            self.similarity_label.setText(f"Match: {similarity:.1%}")
            
            # Color similarity label based on score
            if similarity >= 0.7:
                color = "#00ff88"  # Green - excellent match
            elif similarity >= 0.5:
                color = "#ffaa00"  # Orange - good match
            else:
                color = "#ff6666"  # Red - weak match
            self.similarity_label.setStyleSheet(f"color: {color}; padding: 5px; font-weight: bold;")
            
            # Update status
            timestamp = data.get('timestamp', 0)
            self.status_label.setText(f"✅ Active at {timestamp:.1f}s")
            self.status_label.setStyleSheet("color: #00ff88; font-style: italic;")
            
        except Exception as e:
            print(f"⚠️  RhythmOracleViewport update error: {e}")
            self.status_label.setText(f"⚠️ Update error: {str(e)}")
            self.status_label.setStyleSheet("color: #ff6666; font-style: italic;")
