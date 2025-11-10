#!/usr/bin/env python3
"""
Performance Timeline Viewport
Displays timeline of session events (mode changes, thematic recalls, responses)
"""

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QScrollArea, QFrame
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPainter, QColor, QPen, QBrush
from typing import Dict, Any, List, Tuple
from .base_viewport import BaseViewport
import time


class TimelineWidget(QWidget):
    """Custom widget for drawing timeline"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.events: List[Dict[str, Any]] = []
        self.session_start_time = time.time()
        self.timeline_duration = 300  # Show 5 minutes of timeline
        self.setMinimumHeight(120)
        
        # Event colors
        self.event_colors = {
            'mode_change': QColor(33, 150, 243),     # Blue
            'thematic_recall': QColor(255, 193, 7),  # Yellow
            'response': QColor(76, 175, 80),         # Green
            'human_input': QColor(156, 39, 176)      # Purple
        }
        
        # Mode colors
        self.mode_colors = {
            'SHADOW': QColor(33, 150, 243),     # Blue
            'MIRROR': QColor(76, 175, 80),      # Green
            'COUPLE': QColor(255, 152, 0),      # Orange
            'IMITATE': QColor(156, 39, 176),    # Purple
            'CONTRAST': QColor(244, 67, 54),    # Red
            'LEAD': QColor(0, 188, 212)         # Cyan
        }
        
        self.setStyleSheet("background-color: #1E1E1E;")
    
    def add_event(self, event_type: str, mode: str = None, timestamp: float = None):
        """Add an event to the timeline"""
        if timestamp is None:
            timestamp = time.time()
        
        event = {
            'type': event_type,
            'mode': mode,
            'timestamp': timestamp
        }
        
        self.events.append(event)
        
        # Trim old events outside timeline duration
        cutoff_time = time.time() - self.timeline_duration
        self.events = [e for e in self.events if e['timestamp'] > cutoff_time]
        
        self.update()  # Trigger repaint
    
    def paintEvent(self, event):
        """Paint the timeline"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        width = self.width()
        height = self.height()
        
        current_time = time.time()
        
        # Draw time axis
        painter.setPen(QPen(QColor(80, 80, 80), 2))
        axis_y = height // 2
        painter.drawLine(0, axis_y, width, axis_y)
        
        # Draw time markers (every 30 seconds)
        marker_interval = 30  # seconds
        for i in range(int(self.timeline_duration / marker_interval) + 1):
            marker_time = current_time - (i * marker_interval)
            x = self._time_to_x(marker_time, width, current_time)
            
            if 0 <= x <= width:
                # Draw marker line
                painter.setPen(QPen(QColor(80, 80, 80), 1))
                painter.drawLine(x, axis_y - 5, x, axis_y + 5)
                
                # Draw time label
                minutes_ago = i * marker_interval // 60
                seconds_ago = (i * marker_interval) % 60
                label = f"-{minutes_ago}:{seconds_ago:02d}"
                
                painter.setPen(QPen(QColor(150, 150, 150), 1))
                painter.drawText(x - 20, axis_y + 20, label)
        
        # Draw events
        for evt in self.events:
            evt_time = evt['timestamp']
            x = self._time_to_x(evt_time, width, current_time)
            
            if 0 <= x <= width:
                evt_type = evt['type']
                mode = evt.get('mode', None)
                
                # Choose color based on event type or mode
                if mode and evt_type == 'mode_change':
                    color = self.mode_colors.get(mode, QColor(128, 128, 128))
                else:
                    color = self.event_colors.get(evt_type, QColor(128, 128, 128))
                
                # Draw event marker
                if evt_type == 'mode_change':
                    # Draw as rectangle
                    painter.setBrush(QBrush(color))
                    painter.setPen(QPen(color, 1))
                    painter.drawRect(x - 3, axis_y - 30, 6, 30)
                    
                    # Add mode label
                    if mode:
                        painter.setPen(QPen(QColor(220, 220, 220), 1))
                        painter.drawText(x - 20, axis_y - 35, mode[:3])
                
                elif evt_type == 'thematic_recall':
                    # Draw as star
                    painter.setBrush(QBrush(color))
                    painter.setPen(QPen(color, 2))
                    self._draw_star(painter, x, axis_y - 20, 8)
                
                elif evt_type == 'response':
                    # Draw as circle
                    painter.setBrush(QBrush(color))
                    painter.setPen(QPen(color, 1))
                    painter.drawEllipse(x - 4, axis_y - 4, 8, 8)
                
                elif evt_type == 'human_input':
                    # Draw as triangle
                    painter.setBrush(QBrush(color))
                    painter.setPen(QPen(color, 1))
                    self._draw_triangle(painter, x, axis_y + 20, 8)
        
        # Draw "NOW" indicator
        now_x = width - 10
        painter.setPen(QPen(QColor(255, 87, 34), 3))
        painter.drawLine(now_x, 10, now_x, height - 10)
        painter.drawText(now_x - 20, 10, "NOW")
        
        painter.end()
    
    def _time_to_x(self, event_time: float, width: int, current_time: float) -> int:
        """Convert timestamp to x coordinate"""
        time_ago = current_time - event_time
        fraction = 1 - (time_ago / self.timeline_duration)
        return int(fraction * (width - 20))  # Leave space for NOW indicator
    
    def _draw_star(self, painter, cx, cy, size):
        """Draw a star shape"""
        from PyQt5.QtCore import QPoint
        points = []
        for i in range(5):
            angle = (i * 4 * 3.14159 / 5) - (3.14159 / 2)
            x = cx + int(size * (1 if i % 2 == 0 else 0.5) * (1 if i % 2 == 0 else -1) * abs(1))
            y = cy + int(size * (1 if i % 2 == 0 else 0.5) * (1 if i % 2 == 0 else -1) * abs(1))
            # Simplified: just draw a circle for now
        painter.drawEllipse(cx - size//2, cy - size//2, size, size)
    
    def _draw_triangle(self, painter, cx, cy, size):
        """Draw a triangle shape"""
        from PyQt5.QtCore import QPoint
        from PyQt5.QtGui import QPolygon
        points = QPolygon([
            QPoint(cx, cy - size),
            QPoint(cx - size, cy + size),
            QPoint(cx + size, cy + size)
        ])
        painter.drawPolygon(points)
    
    def clear(self):
        """Clear all events"""
        self.events.clear()
        self.session_start_time = time.time()
        self.update()


class TimelineViewport(BaseViewport):
    """
    Displays performance timeline:
    - Visual timeline with events
    - Event legend
    - Session duration
    """
    
    def __init__(self):
        super().__init__(
            viewport_id="performance_timeline",
            title="Performance Timeline",
            update_rate_ms=1000  # Update every second (for time axis)
        )
        
        # Setup content
        self._setup_content()
        
        # Start time
        self.session_start_time = time.time()
    
    def _setup_content(self):
        """Setup viewport-specific content"""
        content_layout = QVBoxLayout()
        self.content_widget.setLayout(content_layout)
        
        content_layout.setSpacing(5)
        content_layout.setContentsMargins(10, 5, 10, 5)
        
        # Session info bar (compact, styled like rhythm oracle)
        info_frame = QFrame()
        info_frame.setStyleSheet("background-color: #0f0f1e; border-radius: 3px; padding: 5px;")
        info_layout = QVBoxLayout()
        info_layout.setSpacing(2)
        info_layout.setContentsMargins(5, 5, 5, 5)
        
        # Session duration (prominent but not huge)
        self.duration_label = QLabel("Session: 0:00")
        self.duration_label.setFont(QFont("Monaco", 14, QFont.Bold))
        self.duration_label.setStyleSheet("color: #00ff88;")
        self.duration_label.setAlignment(Qt.AlignCenter)
        info_layout.addWidget(self.duration_label)
        
        # Current phase (if using performance arc)
        self.phase_label = QLabel("Phase: Waiting...")
        self.phase_label.setFont(QFont("Monaco", 9))
        self.phase_label.setStyleSheet("color: #88ddff;")
        self.phase_label.setAlignment(Qt.AlignCenter)
        info_layout.addWidget(self.phase_label)
        
        info_frame.setLayout(info_layout)
        content_layout.addWidget(info_frame)
        
        # Timeline widget
        self.timeline_widget = TimelineWidget()
        content_layout.addWidget(self.timeline_widget)
        
        # Simplified legend (mode colors only, since mode is the main thing tracked)
        legend_frame = QFrame()
        legend_frame.setStyleSheet("background-color: #0f0f1e; border-radius: 3px; padding: 3px;")
        legend_layout = QVBoxLayout()
        legend_layout.setSpacing(1)
        legend_layout.setContentsMargins(5, 3, 5, 3)
        legend_frame.setLayout(legend_layout)
        
        legend_label = QLabel("Mode Colors:")
        legend_label.setFont(QFont("Monaco", 8, QFont.Bold))
        legend_label.setStyleSheet("color: #888888;")
        legend_layout.addWidget(legend_label)
        
        # Mode color legend (horizontal, compact)
        mode_colors_text = "SHADOW=Blue  MIRROR=Green  COUPLE=Orange  IMITATE=Purple  CONTRAST=Red  LEAD=Cyan"
        legend_text = QLabel(mode_colors_text)
        legend_text.setFont(QFont("Monaco", 7))
        legend_text.setStyleSheet("color: #666666;")
        legend_text.setWordWrap(True)
        legend_layout.addWidget(legend_text)
        
        content_layout.addWidget(legend_frame)
        
        # Start duration update timer
        from PyQt5.QtCore import QTimer
        self.duration_timer = QTimer()
        self.duration_timer.timeout.connect(self._update_duration)
        self.duration_timer.start(1000)  # Update every second
    
    def _update_display(self, data: Dict[str, Any]):
        """Update timeline with new event"""
        event_type = data.get('event_type', 'unknown')
        mode = data.get('mode', None)
        timestamp = data.get('timestamp', None)
        
        # Update phase if provided
        phase = data.get('phase', None)
        if phase:
            self.phase_label.setText(f"Phase: {phase}")
        
        self.timeline_widget.add_event(event_type, mode, timestamp)
    
    def _update_duration(self):
        """Update session duration display"""
        elapsed = time.time() - self.session_start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        self.duration_label.setText(f"Session: {minutes}:{seconds:02d}")
    
    def clear(self):
        """Clear the viewport"""
        self.timeline_widget.clear()
        self.session_start_time = time.time()
        self.duration_label.setText("Session: 0:00")
        self.phase_label.setText("Phase: Waiting...")


if __name__ == "__main__":
    # Test the timeline viewport
    print("🧪 Testing TimelineViewport")
    
    from PyQt5.QtWidgets import QApplication
    import sys
    
    app = QApplication(sys.argv)
    
    viewport = TimelineViewport()
    viewport.setGeometry(100, 100, 800, 400)
    viewport.show()
    
    print("✅ Viewport created")
    print("✅ Simulating timeline events...")
    
    # Simulate events over time
    test_events = [
        {'event_type': 'mode_change', 'mode': 'SHADOW'},
        {'event_type': 'human_input'},
        {'event_type': 'response'},
        {'event_type': 'human_input'},
        {'event_type': 'response'},
        {'event_type': 'thematic_recall'},
        {'event_type': 'mode_change', 'mode': 'COUPLE'},
        {'event_type': 'response'},
    ]
    
    for i, event in enumerate(test_events):
        print(f"   Event {i+1}: {event['event_type']}")
        viewport.update_data(event)
        app.processEvents()
        time.sleep(2)  # 2 seconds between events
    
    print("\n✅ TimelineViewport tests complete!")
    print("Watch the timeline... Close the window to exit...")
    
    sys.exit(app.exec_())

