#!/usr/bin/env python3
"""
Audio Analysis Viewport
Displays real-time audio analysis (waveform, onsets, ratios, consonance)
"""

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QFrame
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPainter, QColor, QPen
from typing import Dict, Any, List, Optional
import numpy as np
from .base_viewport import BaseViewport


class WaveformWidget(QWidget):
    """Custom widget for drawing waveform"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.waveform_data: Optional[np.ndarray] = None
        self.onset_detected = False
        self.setMinimumHeight(80)
        
        # Styling
        self.setStyleSheet("background-color: #1E1E1E;")
    
    def set_waveform(self, waveform: Optional[np.ndarray], onset: bool = False):
        """Update waveform data"""
        self.waveform_data = waveform
        self.onset_detected = onset
        self.update()  # Trigger repaint
    
    def paintEvent(self, event):
        """Paint the waveform"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        width = self.width()
        height = self.height()
        center_y = height // 2
        
        # Draw center line
        painter.setPen(QPen(QColor(80, 80, 80), 1))
        painter.drawLine(0, center_y, width, center_y)
        
        # Draw waveform if available
        if self.waveform_data is not None and len(self.waveform_data) > 0:
            # Choose color based on onset
            if self.onset_detected:
                waveform_color = QColor(255, 152, 0)  # Orange for onset
                pen_width = 2
            else:
                waveform_color = QColor(33, 150, 243)  # Blue for normal
                pen_width = 1
            
            painter.setPen(QPen(waveform_color, pen_width))
            
            # Downsample waveform to fit width
            num_samples = len(self.waveform_data)
            samples_per_pixel = max(1, num_samples // width)
            
            for x in range(width):
                start_idx = x * samples_per_pixel
                end_idx = min(start_idx + samples_per_pixel, num_samples)
                
                if start_idx < num_samples:
                    # Get max amplitude in this pixel's range
                    segment = self.waveform_data[start_idx:end_idx]
                    if len(segment) > 0:
                        max_val = np.max(np.abs(segment))
                        y_offset = int(max_val * (height / 2 - 5))
                        
                        # Draw line from center to amplitude
                        painter.drawLine(x, center_y - y_offset, x, center_y + y_offset)
        
        painter.end()


class AudioAnalysisViewport(BaseViewport):
    """
    Displays audio analysis information:
    - Waveform display (with onset highlighting)
    - Onset detection indicator
    - Rhythm ratio (Brandtsegg)
    - Consonance value
    - Barlow complexity
    """
    
    def __init__(self):
        super().__init__(
            viewport_id="audio_analysis",
            title="Audio Analysis",
            update_rate_ms=30  # Update every 30ms for smooth waveform
        )
        
        # Setup content
        self._setup_content()
    
    def _setup_content(self):
        """Setup viewport-specific content"""
        content_layout = QVBoxLayout()
        self.content_widget.setLayout(content_layout)
        
        # Waveform display
        waveform_label = QLabel("Waveform:")
        waveform_font = QFont()
        waveform_font.setPointSize(9)
        waveform_font.setBold(True)
        waveform_label.setFont(waveform_font)
        content_layout.addWidget(waveform_label)
        
        self.waveform_widget = WaveformWidget()
        content_layout.addWidget(self.waveform_widget)
        
        # Onset indicator
        self.onset_label = QLabel("Onset: ---")
        self.onset_label.setAlignment(Qt.AlignCenter)
        onset_font = QFont()
        onset_font.setPointSize(14)
        onset_font.setBold(True)
        self.onset_label.setFont(onset_font)
        content_layout.addWidget(self.onset_label)
        
        # Analysis data frame
        analysis_frame = QFrame()
        analysis_frame.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        analysis_layout = QVBoxLayout()
        analysis_frame.setLayout(analysis_layout)
        
        # Rhythm ratio
        self.ratio_label = QLabel("Rhythm Ratio: ---")
        ratio_font = QFont()
        ratio_font.setPointSize(11)
        self.ratio_label.setFont(ratio_font)
        analysis_layout.addWidget(self.ratio_label)
        
        # Consonance
        self.consonance_label = QLabel("Consonance: ---")
        consonance_font = QFont()
        consonance_font.setPointSize(11)
        self.consonance_label.setFont(consonance_font)
        analysis_layout.addWidget(self.consonance_label)
        
        # Barlow complexity
        self.complexity_label = QLabel("Barlow Complexity: ---")
        complexity_font = QFont()
        complexity_font.setPointSize(11)
        self.complexity_label.setFont(complexity_font)
        analysis_layout.addWidget(self.complexity_label)
        
        content_layout.addWidget(analysis_frame)
        content_layout.addStretch()
    
    def _update_display(self, data: Dict[str, Any]):
        """Update display with new audio analysis data"""
        # Debug: Print first few updates
        if not hasattr(self, '_update_count'):
            self._update_count = 0
        if self._update_count < 3:
            print(f"🎨 VIEWPORT: Audio analysis received data keys: {list(data.keys())}")
            self._update_count += 1
        
        # Update waveform
        waveform = data.get('waveform', None)
        onset = data.get('onset', False)
        
        if waveform is not None:
            # Convert to numpy array if not already
            if not isinstance(waveform, np.ndarray):
                waveform = np.array(waveform)
            self.waveform_widget.set_waveform(waveform, onset)
        
        # Update onset indicator
        if onset:
            self.onset_label.setText("Onset: DETECTED")
            self.onset_label.setStyleSheet("color: #FF9800;")  # Orange
        else:
            self.onset_label.setText("Onset: ---")
            self.onset_label.setStyleSheet("color: #DCDCDC;")  # Normal
        
        # Update rhythm ratio
        if 'ratio' in data and data['ratio'] is not None:
            ratio = data['ratio']
            if isinstance(ratio, (list, tuple)) and len(ratio) >= 2:
                self.ratio_label.setText(f"Rhythm Ratio: {ratio[0]}:{ratio[1]}")
            else:
                self.ratio_label.setText(f"Rhythm Ratio: {ratio}")
        else:
            # Show placeholder if no data
            self.ratio_label.setText("Rhythm Ratio: ---")
        
        # Update consonance
        if 'consonance' in data and data['consonance'] is not None:
            consonance = data['consonance']
            self.consonance_label.setText(f"Consonance: {consonance:.3f}")
            
            # Color code based on consonance
            if consonance >= 0.7:
                color = "#4CAF50"  # Green (consonant)
            elif consonance >= 0.4:
                color = "#FFC107"  # Yellow (neutral)
            else:
                color = "#F44336"  # Red (dissonant)
            
            self.consonance_label.setStyleSheet(f"color: {color};")
        
        # Update Barlow complexity
        if 'complexity' in data and data['complexity'] is not None:
            complexity = data['complexity']
            self.complexity_label.setText(f"Barlow Complexity: {complexity:.2f}")
        else:
            # Show placeholder if no data
            self.complexity_label.setText("Barlow Complexity: ---")
    
    def clear(self):
        """Clear the viewport"""
        self.waveform_widget.set_waveform(None, False)
        self.onset_label.setText("Onset: ---")
        self.onset_label.setStyleSheet("color: #DCDCDC;")
        self.ratio_label.setText("Rhythm Ratio: ---")
        self.consonance_label.setText("Consonance: ---")
        self.complexity_label.setText("Barlow Complexity: ---")


if __name__ == "__main__":
    # Test the audio analysis viewport
    print("🧪 Testing AudioAnalysisViewport")
    
    from PyQt5.QtWidgets import QApplication
    import sys
    import time
    
    app = QApplication(sys.argv)
    
    viewport = AudioAnalysisViewport()
    viewport.setGeometry(100, 100, 500, 500)
    viewport.show()
    
    print("✅ Viewport created")
    print("✅ Simulating audio analysis updates...")
    
    # Simulate updates
    for i in range(10):
        # Generate fake waveform
        waveform = np.random.randn(1024) * 0.5
        onset = (i % 3 == 0)  # Onset every 3rd update
        
        data = {
            'waveform': waveform,
            'onset': onset,
            'ratio': [3, 2] if i % 2 == 0 else [4, 3],
            'consonance': np.random.uniform(0.3, 0.9),
            'complexity': np.random.uniform(1.0, 4.0)
        }
        
        print(f"   Update {i+1}: Onset={onset}, Consonance={data['consonance']:.2f}")
        viewport.update_data(data)
        app.processEvents()
        time.sleep(0.3)
    
    print("\n✅ AudioAnalysisViewport tests complete!")
    print("Close the window to exit...")
    
    sys.exit(app.exec_())

