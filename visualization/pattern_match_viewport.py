#!/usr/bin/env python3
"""
Pattern Matching Viewport
Displays current pattern matching information
"""

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QProgressBar, QScrollArea, QFrame
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QColor
from typing import Dict, Any, List
from .base_viewport import BaseViewport


class PatternMatchViewport(BaseViewport):
    """
    Displays pattern matching information:
    - Current gesture token (large display)
    - Pattern match score (progress bar with color coding)
    - Training state matched
    - Recent gesture token history (scrolling list)
    """
    
    def __init__(self):
        super().__init__(
            viewport_id="pattern_matching",
            title="Pattern Matching",
            update_rate_ms=50  # Update every 50ms
        )
        
        # Setup content
        self._setup_content()
        
        # Initialize state
        self.recent_tokens: List[int] = []
        self.max_recent_tokens = 10
    
    def _setup_content(self):
        """Setup viewport-specific content"""
        content_layout = QVBoxLayout()
        self.content_widget.setLayout(content_layout)
        
        # Current gesture token (large display)
        self.token_label = QLabel("---")
        self.token_label.setAlignment(Qt.AlignCenter)
        token_font = QFont()
        token_font.setPointSize(48)
        token_font.setBold(True)
        self.token_label.setFont(token_font)
        content_layout.addWidget(self.token_label)
        
        # Token description
        self.token_desc_label = QLabel("Gesture Token")
        self.token_desc_label.setAlignment(Qt.AlignCenter)
        desc_font = QFont()
        desc_font.setPointSize(10)
        self.token_desc_label.setFont(desc_font)
        content_layout.addWidget(self.token_desc_label)
        
        # Pattern match score (progress bar)
        self.score_label = QLabel("Match Score: ---")
        self.score_label.setAlignment(Qt.AlignCenter)
        content_layout.addWidget(self.score_label)
        
        self.score_bar = QProgressBar()
        self.score_bar.setRange(0, 100)
        self.score_bar.setValue(0)
        self.score_bar.setTextVisible(True)
        self.score_bar.setFormat("%v%")
        self.score_bar.setMinimumHeight(30)
        content_layout.addWidget(self.score_bar)
        
        # Training state info
        self.state_label = QLabel("State: ---")
        self.state_label.setAlignment(Qt.AlignCenter)
        state_font = QFont()
        state_font.setPointSize(10)
        self.state_label.setFont(state_font)
        content_layout.addWidget(self.state_label)
        
        # Recent token history
        history_label = QLabel("Recent Tokens:")
        history_label.setAlignment(Qt.AlignLeft)
        history_font = QFont()
        history_font.setPointSize(9)
        history_font.setBold(True)
        history_label.setFont(history_font)
        content_layout.addWidget(history_label)
        
        # Scrolling token list
        self.token_history_widget = QLabel("...")
        self.token_history_widget.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.token_history_widget.setWordWrap(True)
        self.token_history_widget.setMinimumHeight(60)
        content_layout.addWidget(self.token_history_widget)
        
        content_layout.addStretch()
    
    def _update_display(self, data: Dict[str, Any]):
        """Update display with new pattern matching data"""
        # Debug: Print first few updates
        if not hasattr(self, '_update_count'):
            self._update_count = 0
        if self._update_count < 3:
            print(f"🎨 VIEWPORT: Pattern match received data: {data}")
            self._update_count += 1
        
        # Update current gesture token
        if 'gesture_token' in data:
            token = data['gesture_token']
            self.token_label.setText(str(token))
            
            # Add to history
            if token not in self.recent_tokens or (self.recent_tokens and self.recent_tokens[-1] != token):
                self.recent_tokens.append(token)
                if len(self.recent_tokens) > self.max_recent_tokens:
                    self.recent_tokens.pop(0)
                
                # Update history display
                history_text = " → ".join(str(t) for t in self.recent_tokens)
                self.token_history_widget.setText(history_text)
        
        # Update pattern match score
        if 'score' in data:
            score = float(data['score'])
            self.score_bar.setValue(int(score))
            self.score_label.setText(f"Match Score: {score:.1f}%")
            
            # Color code based on score
            if score >= 70:
                # Green for high match
                self.score_bar.setStyleSheet("""
                    QProgressBar {
                        border: 2px solid grey;
                        border-radius: 5px;
                        text-align: center;
                    }
                    QProgressBar::chunk {
                        background-color: #4CAF50;
                    }
                """)
            elif score >= 40:
                # Yellow for medium match
                self.score_bar.setStyleSheet("""
                    QProgressBar {
                        border: 2px solid grey;
                        border-radius: 5px;
                        text-align: center;
                    }
                    QProgressBar::chunk {
                        background-color: #FFC107;
                    }
                """)
            else:
                # Red for low match
                self.score_bar.setStyleSheet("""
                    QProgressBar {
                        border: 2px solid grey;
                        border-radius: 5px;
                        text-align: center;
                    }
                    QProgressBar::chunk {
                        background-color: #F44336;
                    }
                """)
        
        # Update training state info
        if 'state_id' in data:
            state_id = data['state_id']
            # Estimate bar number (roughly state_id / 10, but approximate)
            bar_approx = state_id // 10
            self.state_label.setText(f"State: {state_id} (bar ~{bar_approx})")
    
    def clear(self):
        """Clear the viewport"""
        self.token_label.setText("---")
        self.score_bar.setValue(0)
        self.score_label.setText("Match Score: ---")
        self.state_label.setText("State: ---")
        self.recent_tokens.clear()
        self.token_history_widget.setText("...")


if __name__ == "__main__":
    # Test the pattern match viewport
    print("🧪 Testing PatternMatchViewport")
    
    from PyQt5.QtWidgets import QApplication
    import sys
    import time
    
    app = QApplication(sys.argv)
    
    viewport = PatternMatchViewport()
    viewport.setGeometry(100, 100, 400, 500)
    viewport.show()
    
    print("✅ Viewport created")
    print("✅ Simulating pattern matching updates...")
    
    # Simulate updates
    test_data = [
        {'gesture_token': 142, 'score': 87.5, 'state_id': 234},
        {'gesture_token': 143, 'score': 72.3, 'state_id': 267},
        {'gesture_token': 144, 'score': 45.8, 'state_id': 289},
        {'gesture_token': 145, 'score': 23.1, 'state_id': 312},
        {'gesture_token': 142, 'score': 91.2, 'state_id': 234},
    ]
    
    for i, data in enumerate(test_data):
        print(f"   Update {i+1}: Token {data['gesture_token']}, Score {data['score']}%")
        viewport.update_data(data)
        app.processEvents()
        time.sleep(1)
    
    print("\n✅ PatternMatchViewport tests complete!")
    print("Close the window to exit...")
    
    sys.exit(app.exec_())

