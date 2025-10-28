#!/usr/bin/env python3
"""
Base Viewport Class
All specific viewport types inherit from this base
"""

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QPalette, QColor
from typing import Dict, Any, Optional


class BaseViewport(QWidget):
    """
    Base class for all viewport types
    
    Provides:
    - Standard styling
    - Title display
    - Update rate limiting
    - Common layout structure
    """
    
    def __init__(self, 
                 viewport_id: str,
                 title: str,
                 update_rate_ms: int = 50):
        """
        Initialize base viewport
        
        Args:
            viewport_id: Unique identifier for this viewport
            title: Display title
            update_rate_ms: Minimum milliseconds between updates (rate limiting)
        """
        super().__init__()
        
        self.viewport_id = viewport_id
        self.title = title
        self.update_rate_ms = update_rate_ms
        
        # Rate limiting
        self._last_update_time = 0
        self._update_pending = False
        self._pending_data = None
        
        # Setup UI
        self._setup_ui()
        self._apply_styling()
    
    def _setup_ui(self):
        """Setup basic UI structure"""
        # Main layout
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        # Title label
        self.title_label = QLabel(self.title)
        self.title_label.setAlignment(Qt.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(12)
        title_font.setBold(True)
        self.title_label.setFont(title_font)
        self.layout.addWidget(self.title_label)
        
        # Content widget (subclasses override this)
        self.content_widget = QWidget()
        self.layout.addWidget(self.content_widget)
    
    def _apply_styling(self):
        """Apply standard viewport styling"""
        # Dark theme colors
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(45, 45, 48))  # Dark gray background
        palette.setColor(QPalette.WindowText, QColor(220, 220, 220))  # Light gray text
        palette.setColor(QPalette.Base, QColor(30, 30, 30))  # Even darker for content
        palette.setColor(QPalette.AlternateBase, QColor(50, 50, 50))
        palette.setColor(QPalette.Text, QColor(220, 220, 220))
        self.setPalette(palette)
        self.setAutoFillBackground(True)
        
        # Border styling
        self.setStyleSheet("""
            BaseViewport {
                border: 2px solid #404040;
                border-radius: 5px;
            }
            QLabel {
                color: #DCDCDC;
            }
        """)
    
    def update_data(self, data: Dict[str, Any]):
        """
        Update viewport with new data (rate-limited)
        
        Args:
            data: Dictionary of data to display
        """
        import time
        
        current_time = time.time() * 1000  # milliseconds
        time_since_last_update = current_time - self._last_update_time
        
        if time_since_last_update >= self.update_rate_ms:
            # Immediate update
            self._last_update_time = current_time
            self._update_display(data)
            self._update_pending = False
            self._pending_data = None
        else:
            # Queue update for later
            self._update_pending = True
            self._pending_data = data
            
            # Schedule delayed update
            delay = self.update_rate_ms - int(time_since_last_update)
            QTimer.singleShot(delay, self._process_pending_update)
    
    def _process_pending_update(self):
        """Process queued update if one is pending"""
        if self._update_pending and self._pending_data is not None:
            import time
            self._last_update_time = time.time() * 1000
            self._update_display(self._pending_data)
            self._update_pending = False
            self._pending_data = None
    
    def _update_display(self, data: Dict[str, Any]):
        """
        Update the display with new data
        
        Subclasses MUST override this method
        
        Args:
            data: Dictionary of data to display
        """
        raise NotImplementedError("Subclasses must implement _update_display()")
    
    def clear(self):
        """
        Clear viewport display
        
        Subclasses should override if they need special clearing logic
        """
        pass
    
    def set_highlighted(self, highlighted: bool):
        """
        Highlight or unhighlight viewport (e.g., during events)
        
        Args:
            highlighted: True to highlight, False to return to normal
        """
        if highlighted:
            self.setStyleSheet("""
                BaseViewport {
                    border: 2px solid #FF8C00;
                    border-radius: 5px;
                }
                QLabel {
                    color: #DCDCDC;
                }
            """)
        else:
            self.setStyleSheet("""
                BaseViewport {
                    border: 2px solid #404040;
                    border-radius: 5px;
                }
                QLabel {
                    color: #DCDCDC;
                }
            """)


class LabelViewport(BaseViewport):
    """
    Simple viewport that displays text labels
    Useful for testing and simple data display
    """
    
    def __init__(self, viewport_id: str, title: str):
        """Initialize label viewport"""
        super().__init__(viewport_id, title, update_rate_ms=100)
        
        # Create content label
        self.content_label = QLabel("No data")
        self.content_label.setAlignment(Qt.AlignCenter)
        content_font = QFont()
        content_font.setPointSize(24)
        self.content_label.setFont(content_font)
        
        # Replace content widget
        self.layout.removeWidget(self.content_widget)
        self.content_widget = self.content_label
        self.layout.addWidget(self.content_widget)
    
    def _update_display(self, data: Dict[str, Any]):
        """Update label with data"""
        if 'text' in data:
            self.content_label.setText(str(data['text']))
        elif 'value' in data:
            self.content_label.setText(str(data['value']))
        else:
            # Display first key-value pair
            if data:
                key = list(data.keys())[0]
                value = data[key]
                self.content_label.setText(f"{key}: {value}")


if __name__ == "__main__":
    # Test the base viewport
    print("🧪 Testing BaseViewport")
    
    from PyQt5.QtWidgets import QApplication
    import sys
    import time
    
    app = QApplication(sys.argv)
    
    # Create a simple label viewport for testing
    viewport = LabelViewport("test_viewport", "Test Viewport")
    viewport.setGeometry(100, 100, 400, 300)
    viewport.show()
    
    # Test updating data
    print("✅ Viewport created and displayed")
    print("✅ Test: Updating data 5 times...")
    
    for i in range(5):
        viewport.update_data({'text': f'Update {i+1}'})
        app.processEvents()
        time.sleep(0.5)
    
    print("✅ Test: Highlighting viewport...")
    viewport.set_highlighted(True)
    app.processEvents()
    time.sleep(1)
    
    viewport.set_highlighted(False)
    app.processEvents()
    
    print("\n✅ BaseViewport tests complete!")
    print("Close the window to exit...")
    
    sys.exit(app.exec_())

