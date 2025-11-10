#!/usr/bin/env python3
"""
Status Bar Viewport
Top bar showing critical performance state at a glance
"""

from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLabel, QFrame
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QColor
from typing import Dict, Any
from .base_viewport import BaseViewport


class StatusBarViewport(BaseViewport):
    """
    Displays critical status information in a horizontal bar:
    - Behavior mode (large, color-coded)
    - Override warning (if active, flashing)
    - Performance phase
    - Detected chord (what system hears)
    - Time elapsed
    """
    
    def __init__(self):
        super().__init__(
            viewport_id="status_bar",
            title="",  # No title for status bar
            update_rate_ms=100
        )
        
        # Remove title bar for cleaner look
        self.title_label.hide()
        
        # Setup content
        self._setup_content()
        
        # Mode colors
        self.mode_colors = {
            'SHADOW': '#2196F3',    # Blue
            'MIRROR': '#4CAF50',    # Green
            'COUPLE': '#FF9800',    # Orange
            'IMITATE': '#9C27B0',   # Purple
            'CONTRAST': '#F44336',  # Red
            'LEAD': '#00BCD4'       # Cyan
        }
        
        # Override flash state
        self.override_flash = False
        
    def _setup_content(self):
        """Setup horizontal status bar layout"""
        layout = QHBoxLayout()
        layout.setContentsMargins(10, 5, 10, 5)
        layout.setSpacing(20)
        self.content_widget.setLayout(layout)
        
        # Set minimum height for status bar
        self.setMinimumHeight(80)
        self.setMaximumHeight(100)
        
        # 1. Behavior Mode (large, color-coded)
        mode_frame = QFrame()
        mode_frame.setFrameStyle(QFrame.Panel | QFrame.Raised)
        mode_frame.setMinimumWidth(200)
        mode_layout = QHBoxLayout()
        mode_frame.setLayout(mode_layout)
        
        self.mode_label = QLabel("---")
        self.mode_label.setAlignment(Qt.AlignCenter)
        mode_font = QFont()
        mode_font.setPointSize(24)
        mode_font.setBold(True)
        self.mode_label.setFont(mode_font)
        mode_layout.addWidget(self.mode_label)
        layout.addWidget(mode_frame)
        self.mode_frame = mode_frame
        
        # 2. Duration
        self.duration_label = QLabel("---")
        self.duration_label.setAlignment(Qt.AlignCenter)
        duration_font = QFont()
        duration_font.setPointSize(12)
        self.duration_label.setFont(duration_font)
        layout.addWidget(self.duration_label)
        
        # 3. Override Warning (initially hidden)
        self.override_frame = QFrame()
        self.override_frame.setFrameStyle(QFrame.Panel | QFrame.Raised)
        self.override_frame.setStyleSheet("background-color: #FF5722; border-radius: 5px;")
        override_layout = QHBoxLayout()
        self.override_frame.setLayout(override_layout)
        
        self.override_label = QLabel("⚠️ OVERRIDE ACTIVE")
        self.override_label.setAlignment(Qt.AlignCenter)
        override_font = QFont()
        override_font.setPointSize(14)
        override_font.setBold(True)
        self.override_label.setFont(override_font)
        self.override_label.setStyleSheet("color: white; padding: 5px;")
        override_layout.addWidget(self.override_label)
        
        layout.addWidget(self.override_frame)
        self.override_frame.hide()  # Hidden by default
        
        # 4. Phase
        phase_frame = QFrame()
        phase_layout = QHBoxLayout()
        phase_frame.setLayout(phase_layout)
        
        phase_label = QLabel("Phase:")
        phase_label_font = QFont()
        phase_label_font.setPointSize(10)
        phase_label.setFont(phase_label_font)
        phase_layout.addWidget(phase_label)
        
        self.phase_label = QLabel("---")
        phase_font = QFont()
        phase_font.setPointSize(12)
        phase_font.setBold(True)
        self.phase_label.setFont(phase_font)
        phase_layout.addWidget(self.phase_label)
        
        layout.addWidget(phase_frame)
        
        # 5. Detected Chord (what system hears)
        chord_frame = QFrame()
        chord_frame.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        chord_layout = QHBoxLayout()
        chord_frame.setLayout(chord_layout)
        
        chord_label = QLabel("Detected:")
        chord_label_font = QFont()
        chord_label_font.setPointSize(10)
        chord_label.setFont(chord_label_font)
        chord_layout.addWidget(chord_label)
        
        self.detected_chord_label = QLabel("---")
        detected_font = QFont()
        detected_font.setPointSize(14)
        detected_font.setBold(True)
        self.detected_chord_label.setFont(detected_font)
        chord_layout.addWidget(self.detected_chord_label)
        
        layout.addWidget(chord_frame)
        
        # 6. Time Elapsed
        self.time_label = QLabel("0:00")
        time_font = QFont()
        time_font.setPointSize(12)
        self.time_label.setFont(time_font)
        layout.addWidget(self.time_label)
        
        # Add stretch to push everything to reasonable spacing
        layout.addStretch()
    
    def _update_display(self, data: Dict[str, Any]):
        """Update status bar with new data"""
        
        # Update behavior mode
        if 'mode' in data:
            mode = data['mode']
            self.mode_label.setText(mode)
            
            # Update color
            color = self.mode_colors.get(mode, '#808080')
            self.mode_frame.setStyleSheet(f"""
                QFrame {{
                    background-color: {color};
                    border-radius: 8px;
                    padding: 5px;
                }}
            """)
        
        # Update duration
        if 'mode_duration' in data:
            duration = data['mode_duration']
            self.duration_label.setText(f"{duration:.0f}s")
        
        # Update override status
        if 'harmonic_context' in data:
            harmonic = data['harmonic_context']
            override_active = harmonic.get('override_active', False)
            
            if override_active:
                # Show override warning
                self.override_frame.show()
                
                # Update text with time remaining
                time_left = harmonic.get('override_time_left', 0)
                active_chord = harmonic.get('active_chord', '---')
                detected_chord = harmonic.get('detected_chord', '---')
                
                self.override_label.setText(
                    f"⚠️ OVERRIDE: {active_chord} ({time_left:.0f}s) | "
                    f"Detecting: {detected_chord} (ignored)"
                )
                
                # Flash effect
                self.override_flash = not self.override_flash
                if self.override_flash:
                    self.override_frame.setStyleSheet("background-color: #FF5722; border-radius: 5px;")
                else:
                    self.override_frame.setStyleSheet("background-color: #FF8A65; border-radius: 5px;")
            else:
                # Hide override warning
                self.override_frame.hide()
        
        # Update detected chord
        if 'chord' in data:
            chord = data['chord']
            confidence = data.get('chord_confidence', 0.0)
            self.detected_chord_label.setText(f"{chord} ({confidence:.0%})")
        
        # Update phase
        if 'phase' in data:
            phase = data['phase']
            self.phase_label.setText(phase)
        
        # Update time
        if 'elapsed_time' in data:
            elapsed = data['elapsed_time']
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            self.time_label.setText(f"{minutes}:{seconds:02d}")
