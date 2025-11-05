#!/usr/bin/env python3
"""
GPT Reflection Viewport - Display AI's live musical reflections
"""

from PyQt5.QtWidgets import QLabel, QVBoxLayout, QTextEdit, QSizePolicy
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QTextOption
from .base_viewport import BaseViewport
import time


class GPTReflectionViewport(BaseViewport):
    """
    Viewport for displaying GPT-OSS live reflections on musical interaction.
    
    Features:
    - Display reflection text (2-3 sentence musical analysis)
    - Show last update time and countdown to next reflection
    - Auto-scroll for long reflections
    - Graceful handling of errors/timeouts
    """
    
    def __init__(self, parent=None):
        """Initialize GPT reflection viewport"""
        super().__init__(viewport_id="gpt_reflection", title="GPT-OSS Live Reflection")
        
        # State
        self.current_reflection = "Initializing GPT-OSS live reflection engine...\n\nWaiting for first reflection..."
        self.last_update_time = 0.0
        self.reflection_interval = 60.0  # Default, will be updated
        
        self._init_ui()
        
        # Update timer for countdown
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self._update_status)
        self.update_timer.start(1000)  # Update every second
        
    def _init_ui(self):
        """Initialize the user interface"""
        # Use BaseViewport's content_widget instead of creating new layout
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        self.content_widget.setLayout(layout)
        
        # Reflection text area (read-only, word wrap)
        self.reflection_text = QTextEdit()
        self.reflection_text.setReadOnly(True)
        self.reflection_text.setWordWrapMode(QTextOption.WrapAtWordBoundaryOrAnywhere)
        self.reflection_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #e0e0e0;
                border: 1px solid #555;
                font-size: 13px;
                line-height: 1.6;
                padding: 10px;
            }
        """)
        
        # Set font
        font = QFont("SF Mono", 13)
        font.setStyleHint(QFont.Monospace)
        self.reflection_text.setFont(font)
        
        # Initial text
        self.reflection_text.setPlainText(self.current_reflection)
        
        layout.addWidget(self.reflection_text, stretch=1)
        
        # Status label
        self.status_label = QLabel("Waiting for first reflection...")
        self.status_label.setStyleSheet("""
            QLabel {
                color: #aaa;
                font-size: 10px;
                padding: 3px;
            }
        """)
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
        
    def update_reflection(self, reflection_text: str, timestamp: float):
        """
        Update the displayed reflection (called via signal).
        
        Args:
            reflection_text: New reflection from GPT-OSS
            timestamp: Time of reflection completion
        """
        self.current_reflection = reflection_text
        self.last_update_time = timestamp
        
        # Update display
        self.reflection_text.setPlainText(reflection_text)
        
        # Scroll to top
        cursor = self.reflection_text.textCursor()
        cursor.movePosition(cursor.Start)
        self.reflection_text.setTextCursor(cursor)
        
        # Update status
        self._update_status()
        
    def _update_status(self):
        """Update status label with time info"""
        current_time = time.time()
        
        if self.last_update_time == 0.0:
            # No reflection yet
            self.status_label.setText("⏳ Waiting for first reflection...")
            self.status_label.setStyleSheet("""
                QLabel {
                    color: #888;
                    font-size: 10px;
                    padding: 3px;
                }
            """)
        else:
            # Calculate elapsed and remaining
            elapsed = current_time - self.last_update_time
            remaining = max(0, self.reflection_interval - elapsed)
            
            # Format status message
            if remaining > 0:
                status = f"✅ Updated {elapsed:.0f}s ago  |  Next reflection in {remaining:.0f}s"
                color = "#4ade80"  # Green
            else:
                status = f"⏳ Reflecting now...  (last update: {elapsed:.0f}s ago)"
                color = "#fbbf24"  # Yellow
            
            self.status_label.setText(status)
            self.status_label.setStyleSheet(f"""
                QLabel {{
                    color: {color};
                    font-size: 10px;
                    padding: 3px;
                }}
            """)
    
    def set_reflection_interval(self, interval: float):
        """
        Set the reflection interval (for countdown accuracy).
        
        Args:
            interval: Reflection interval in seconds
        """
        self.reflection_interval = interval
        
    def update_data(self, data: dict):
        """
        Update viewport with new data (called via event bus).
        
        Args:
            data: Dict with 'reflection' and 'timestamp' keys
        """
        reflection = data.get('reflection', '')
        timestamp = data.get('timestamp', time.time())
        
        self.update_reflection(reflection, timestamp)


# Test if run directly
if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication
    import sys
    
    app = QApplication(sys.argv)
    
    viewport = GPTReflectionViewport()
    viewport.setGeometry(100, 100, 500, 400)
    viewport.show()
    
    # Simulate reflection update after 3 seconds
    def test_update():
        test_reflection = """The human is exploring modal territory around D, while the AI provides contrapuntal bass patterns in shadow mode. Consonance is gradually increasing (0.72 avg, rising trend), suggesting convergence toward a shared harmonic space. Gesture tokens 142 and 158 dominate (65% of phrases), indicating gestural lockup around specific timbral qualities."""
        
        viewport.update_reflection(test_reflection, time.time())
    
    QTimer.singleShot(3000, test_update)
    
    sys.exit(app.exec_())
