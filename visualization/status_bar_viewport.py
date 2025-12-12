#!/usr/bin/env python3
"""
Status Bar Viewport - Simplified Display
Shows only essential performance info at a glance
"""

from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLabel, QFrame
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QColor
from typing import Dict, Any
from .base_viewport import BaseViewport


class StatusBarViewport(BaseViewport):
    """
    Simplified status bar showing only essential info:
    1. Time left in concert mode
    2. Detected pitch (monophonic) + Chord (from Chroma)
    3. Live input indicator
    4. Response mode
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
            'shadow': '#2196F3',     # Blue
            'mirror': '#4CAF50',     # Green
            'couple': '#FF9800',     # Orange
            'imitate': '#9C27B0',    # Purple
            'contrast': '#F44336',   # Red
            'lead': '#00BCD4',       # Cyan
            'support': '#8BC34A',    # Light Green
            '---': '#808080'         # Gray (unknown)
        }

    def _setup_content(self):
        """Setup simplified horizontal status bar layout"""
        layout = QHBoxLayout()
        layout.setContentsMargins(15, 10, 15, 10)
        layout.setSpacing(30)
        self.content_widget.setLayout(layout)

        # Set height for status bar
        self.setMinimumHeight(90)
        self.setMaximumHeight(110)

        # === 1. TIME LEFT ===
        time_frame = self._create_info_frame("TIME", "---")
        self.time_label = time_frame.findChild(QLabel, "value_label")
        layout.addWidget(time_frame)

        # === 2. PITCH (monophonic) ===
        pitch_frame = self._create_info_frame("PITCH", "---")
        self.pitch_label = pitch_frame.findChild(QLabel, "value_label")
        layout.addWidget(pitch_frame)

        # === 3. CHORD (from Chroma) ===
        chord_frame = self._create_info_frame("CHORD", "---")
        self.chord_label = chord_frame.findChild(QLabel, "value_label")
        layout.addWidget(chord_frame)

        # === 4. INPUT LEVEL ===
        input_frame = self._create_input_indicator()
        layout.addWidget(input_frame)

        # === 5. RESPONSE MODE ===
        mode_frame = QFrame()
        mode_frame.setFrameStyle(QFrame.Panel | QFrame.Raised)
        mode_frame.setMinimumWidth(150)
        mode_layout = QVBoxLayout()
        mode_layout.setContentsMargins(10, 5, 10, 5)
        mode_frame.setLayout(mode_layout)

        mode_title = QLabel("MODE")
        mode_title.setAlignment(Qt.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(9)
        mode_title.setFont(title_font)
        mode_title.setStyleSheet("color: #888;")
        mode_layout.addWidget(mode_title)

        self.mode_label = QLabel("---")
        self.mode_label.setAlignment(Qt.AlignCenter)
        mode_font = QFont()
        mode_font.setPointSize(18)
        mode_font.setBold(True)
        self.mode_label.setFont(mode_font)
        mode_layout.addWidget(self.mode_label)

        layout.addWidget(mode_frame)
        self.mode_frame = mode_frame

        # Add stretch at end
        layout.addStretch()

    def _create_info_frame(self, title: str, initial_value: str) -> QFrame:
        """Create a labeled info frame"""
        frame = QFrame()
        frame.setMinimumWidth(100)
        frame_layout = QVBoxLayout()
        frame_layout.setContentsMargins(10, 5, 10, 5)
        frame_layout.setSpacing(2)
        frame.setLayout(frame_layout)

        # Title
        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(9)
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: #888;")
        frame_layout.addWidget(title_label)

        # Value
        value_label = QLabel(initial_value)
        value_label.setObjectName("value_label")
        value_label.setAlignment(Qt.AlignCenter)
        value_font = QFont()
        value_font.setPointSize(16)
        value_font.setBold(True)
        value_label.setFont(value_font)
        frame_layout.addWidget(value_label)

        return frame

    def _create_input_indicator(self) -> QFrame:
        """Create the live input level indicator"""
        frame = QFrame()
        frame.setMinimumWidth(120)
        frame_layout = QVBoxLayout()
        frame_layout.setContentsMargins(10, 5, 10, 5)
        frame_layout.setSpacing(2)
        frame.setLayout(frame_layout)

        # Title
        title_label = QLabel("INPUT")
        title_label.setAlignment(Qt.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(9)
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: #888;")
        frame_layout.addWidget(title_label)

        # Level bars container
        bars_layout = QHBoxLayout()
        bars_layout.setSpacing(3)

        self.input_bars = []
        for i in range(5):
            bar = QFrame()
            bar.setFixedSize(15, 30)
            bar.setStyleSheet("background-color: #333; border-radius: 2px;")
            bars_layout.addWidget(bar)
            self.input_bars.append(bar)

        frame_layout.addLayout(bars_layout)

        # Status text (MUSIC / NOISE)
        self.input_status_label = QLabel("")
        self.input_status_label.setAlignment(Qt.AlignCenter)
        status_font = QFont()
        status_font.setPointSize(8)
        self.input_status_label.setFont(status_font)
        frame_layout.addWidget(self.input_status_label)

        return frame

    def _update_input_level(self, rms_db: float, is_musical: bool = True):
        """Update input level indicator bars"""
        # Map rms_db to 0-5 bars
        # -60dB = 0 bars, -30dB = 5 bars
        if rms_db < -60:
            level = 0
        elif rms_db > -30:
            level = 5
        else:
            level = int((rms_db + 60) / 6)  # Each bar = 6dB

        # Update bars
        for i, bar in enumerate(self.input_bars):
            if i < level:
                if is_musical:
                    # Green gradient based on level
                    if i < 2:
                        color = "#4CAF50"  # Green
                    elif i < 4:
                        color = "#FFC107"  # Yellow
                    else:
                        color = "#FF5722"  # Orange/red (loud)
                else:
                    color = "#9E9E9E"  # Gray for noise
                bar.setStyleSheet(f"background-color: {color}; border-radius: 2px;")
            else:
                bar.setStyleSheet("background-color: #333; border-radius: 2px;")

        # Update status text
        if not is_musical:
            self.input_status_label.setText("NOISE?")
            self.input_status_label.setStyleSheet("color: #FF5722;")
        elif level == 0:
            self.input_status_label.setText("silent")
            self.input_status_label.setStyleSheet("color: #666;")
        else:
            self.input_status_label.setText("")

    def _update_display(self, data: Dict[str, Any]):
        """Update status bar with new data"""

        # === 1. TIME LEFT / ELAPSED ===
        if 'time_remaining' in data:
            remaining = data['time_remaining']
            if remaining > 0:
                mins = int(remaining // 60)
                secs = int(remaining % 60)
                self.time_label.setText(f"{mins:02d}:{secs:02d}")
            else:
                self.time_label.setText("00:00")
        elif 'elapsed_time' in data:
            elapsed = data['elapsed_time']
            mins = int(elapsed // 60)
            secs = int(elapsed % 60)
            self.time_label.setText(f"+{mins:02d}:{secs:02d}")

        # === 2. PITCH (monophonic) ===
        if 'pitch' in data:
            pitch = data['pitch']
            if pitch and pitch != 0:
                # Convert frequency to note name if needed
                if isinstance(pitch, (int, float)) and pitch > 20:
                    # It's a frequency, convert to note
                    import math
                    midi = round(12 * math.log2(pitch / 440.0) + 69)
                    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                    note = note_names[midi % 12]
                    octave = (midi // 12) - 1
                    self.pitch_label.setText(f"{note}{octave}")
                else:
                    self.pitch_label.setText(str(pitch))
            else:
                self.pitch_label.setText("---")

        # === 3. CHORD (from Chroma) ===
        if 'chord' in data:
            chord = data['chord']
            self.chord_label.setText(chord if chord else "---")
        elif 'harmonic_context' in data:
            harmonic = data['harmonic_context']
            detected = harmonic.get('detected_chord', '---')
            self.chord_label.setText(detected if detected else "---")

        # === 4. INPUT LEVEL ===
        rms_db = data.get('rms_db', -80)
        is_musical = data.get('is_musical', True)
        self._update_input_level(rms_db, is_musical)

        # === 5. RESPONSE MODE ===
        if 'mode' in data:
            mode = data['mode']
            if mode:
                mode_lower = mode.lower()
                self.mode_label.setText(mode_lower)

                # Update color
                color = self.mode_colors.get(mode_lower, '#808080')
                self.mode_frame.setStyleSheet(f"""
                    QFrame {{
                        background-color: {color};
                        border-radius: 8px;
                    }}
                """)
                self.mode_label.setStyleSheet("color: white;")
