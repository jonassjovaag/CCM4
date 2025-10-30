#!/usr/bin/env python3
"""
Request Parameters Viewport
Displays current behavior mode and request parameters
"""

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QColor
from typing import Dict, Any
from .base_viewport import BaseViewport
import time


class RequestParamsViewport(BaseViewport):
    """
    Displays request parameters:
    - Current behavior mode (large, color-coded)
    - Mode duration remaining (countdown timer)
    - Request structure (primary/secondary/tertiary with weights)
    - Temperature setting
    """
    
    def __init__(self):
        super().__init__(
            viewport_id="request_parameters",
            title="Behavior Mode & Requests",
            update_rate_ms=100
        )
        
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
        
        # Countdown timer
        self.mode_end_time = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_countdown)
        self.timer.start(100)  # Update every 100ms
    
    def _setup_content(self):
        """Setup viewport-specific content"""
        content_layout = QVBoxLayout()
        self.content_widget.setLayout(content_layout)
        
        # Current mode (large, color-coded badge)
        self.mode_label = QLabel("---")
        self.mode_label.setAlignment(Qt.AlignCenter)
        mode_font = QFont()
        mode_font.setPointSize(36)
        mode_font.setBold(True)
        self.mode_label.setFont(mode_font)
        self.mode_label.setMinimumHeight(80)
        self.mode_label.setStyleSheet("""
            QLabel {
                background-color: #404040;
                border-radius: 10px;
                padding: 10px;
            }
        """)
        content_layout.addWidget(self.mode_label)
        
        # Duration remaining
        self.duration_label = QLabel("Duration: ---")
        self.duration_label.setAlignment(Qt.AlignCenter)
        duration_font = QFont()
        duration_font.setPointSize(12)
        self.duration_label.setFont(duration_font)
        content_layout.addWidget(self.duration_label)
        
        # Request parameters frame
        request_frame = QFrame()
        request_frame.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        request_layout = QVBoxLayout()
        request_frame.setLayout(request_layout)
        
        request_title = QLabel("Request Parameters:")
        request_title_font = QFont()
        request_title_font.setPointSize(10)
        request_title_font.setBold(True)
        request_title.setFont(request_title_font)
        request_layout.addWidget(request_title)
        
        self.request_label = QLabel("No request data")
        self.request_label.setWordWrap(True)
        self.request_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        request_label_font = QFont()
        request_label_font.setPointSize(9)
        self.request_label.setFont(request_label_font)
        request_layout.addWidget(self.request_label)
        
        content_layout.addWidget(request_frame)
        
        # Temperature setting
        self.temp_label = QLabel("Temperature: ---")
        self.temp_label.setAlignment(Qt.AlignCenter)
        temp_font = QFont()
        temp_font.setPointSize(11)
        self.temp_label.setFont(temp_font)
        content_layout.addWidget(self.temp_label)
        
        content_layout.addStretch()
    
    def _update_display(self, data: Dict[str, Any]):
        """Update display with new mode/request data"""
        # Update current mode
        if 'mode' in data:
            mode = data['mode']
            self.mode_label.setText(mode)
            
            # Apply mode-specific color
            color = self.mode_colors.get(mode, '#808080')
            self.mode_label.setStyleSheet(f"""
                QLabel {{
                    background-color: {color};
                    border-radius: 10px;
                    padding: 10px;
                    color: white;
                }}
            """)
        
        # Update duration
        if 'duration' in data:
            duration = data['duration']
            self.mode_end_time = time.time() + duration
            self._update_countdown()
        
        # Update request parameters
        if 'request_params' in data:
            request_params = data['request_params']
            request_text = self._format_request_params(request_params)
            self.request_label.setText(request_text)
        
        # Update temperature
        if 'temperature' in data:
            temp = data['temperature']
            self.temp_label.setText(f"Temperature: {temp:.2f}")
    
    def _format_request_params(self, params: Dict[str, Any]) -> str:
        """Format request parameters for display"""
        if not params:
            return "No request data"
        
        lines = []
        
        # Check if it's the new flat format (single request dict)
        if 'parameter' in params and 'type' in params and 'value' in params:
            # Flat format from phrase_generator
            param_name = params.get('parameter', '?')
            param_type = params.get('type', '?')
            param_value = params.get('value', '?')
            weight = params.get('weight', 1.0)
            lines.append(f"REQUEST ({weight:.2f}):")
            lines.append(f"  {param_name} {param_type} {param_value}")
            return "\n".join(lines)
        
        # Legacy nested format (primary/secondary/tertiary)
        # Primary parameter
        if 'primary' in params:
            primary = params['primary']
            param_name = primary.get('parameter', '?')
            param_type = primary.get('type', '?')
            param_value = primary.get('value', '?')
            weight = primary.get('weight', 0)
            lines.append(f"PRIMARY ({weight:.2f}):")
            lines.append(f"  {param_name} {param_type} {param_value}")
        
        # Secondary parameter
        if 'secondary' in params:
            secondary = params['secondary']
            param_name = secondary.get('parameter', '?')
            param_type = secondary.get('type', '?')
            param_value = secondary.get('value', '?')
            weight = secondary.get('weight', 0)
            lines.append(f"SECONDARY ({weight:.2f}):")
            lines.append(f"  {param_name} {param_type} {param_value}")
        
        # Tertiary parameter
        if 'tertiary' in params:
            tertiary = params['tertiary']
            param_name = tertiary.get('parameter', '?')
            param_type = tertiary.get('type', '?')
            param_value = tertiary.get('value', '?')
            weight = tertiary.get('weight', 0)
            lines.append(f"TERTIARY ({weight:.2f}):")
            lines.append(f"  {param_name} {param_type} {param_value}")
        
        return "\n".join(lines) if lines else "No parameters"
    
    def _update_countdown(self):
        """Update countdown timer display"""
        if self.mode_end_time > 0:
            remaining = self.mode_end_time - time.time()
            if remaining > 0:
                self.duration_label.setText(f"Duration: {remaining:.1f}s remaining")
            else:
                self.duration_label.setText("Duration: Expired (switching...)")
        else:
            self.duration_label.setText("Duration: ---")
    
    def clear(self):
        """Clear the viewport"""
        self.mode_label.setText("---")
        self.mode_label.setStyleSheet("""
            QLabel {
                background-color: #404040;
                border-radius: 10px;
                padding: 10px;
            }
        """)
        self.duration_label.setText("Duration: ---")
        self.request_label.setText("No request data")
        self.temp_label.setText("Temperature: ---")
        self.mode_end_time = 0


if __name__ == "__main__":
    # Test the request params viewport
    print("🧪 Testing RequestParamsViewport")
    
    from PyQt5.QtWidgets import QApplication
    import sys
    
    app = QApplication(sys.argv)
    
    viewport = RequestParamsViewport()
    viewport.setGeometry(100, 100, 400, 500)
    viewport.show()
    
    print("✅ Viewport created")
    print("✅ Simulating mode changes...")
    
    # Simulate SHADOW mode
    shadow_data = {
        'mode': 'SHADOW',
        'duration': 47.0,
        'request_params': {
            'primary': {
                'parameter': 'gesture_token',
                'type': '==',
                'value': 142,
                'weight': 0.95
            },
            'secondary': {
                'parameter': 'consonance',
                'type': 'gradient',
                'value': 2.0,
                'weight': 0.5
            }
        },
        'temperature': 0.7
    }
    
    print("   Setting SHADOW mode (47s duration)...")
    viewport.update_data(shadow_data)
    app.processEvents()
    
    print("\n✅ RequestParamsViewport tests complete!")
    print("Watch the countdown timer...")
    print("Close the window to exit...")
    
    sys.exit(app.exec_())

