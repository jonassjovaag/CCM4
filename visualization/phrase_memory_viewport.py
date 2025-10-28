#!/usr/bin/env python3
"""
Phrase Memory Viewport
Displays phrase memory events (motif storage, recall, variations)
"""

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QListWidget, QListWidgetItem, QFrame
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QColor
from typing import Dict, Any, List, Optional
from .base_viewport import BaseViewport


class PhraseMemoryViewport(BaseViewport):
    """
    Displays phrase memory information:
    - Current stored motifs (list)
    - Recent memory events (store/recall/variation with timestamps)
    - Recall probability indicator
    - Current theme indicator
    """
    
    def __init__(self):
        super().__init__(
            viewport_id="phrase_memory",
            title="Phrase Memory",
            update_rate_ms=100
        )
        
        # Setup content
        self._setup_content()
        
        # State tracking
        self.stored_motifs: List[Dict[str, Any]] = []
        self.max_events = 20  # Show last 20 events
    
    def _setup_content(self):
        """Setup viewport-specific content"""
        content_layout = QVBoxLayout()
        self.content_widget.setLayout(content_layout)
        
        # Current theme indicator
        self.theme_label = QLabel("Current Theme: None")
        self.theme_label.setAlignment(Qt.AlignCenter)
        theme_font = QFont()
        theme_font.setPointSize(12)
        theme_font.setBold(True)
        self.theme_label.setFont(theme_font)
        self.theme_label.setStyleSheet("""
            QLabel {
                background-color: #2C2C2C;
                border-radius: 5px;
                padding: 8px;
                color: #FFC107;
            }
        """)
        content_layout.addWidget(self.theme_label)
        
        # Recall probability
        self.recall_prob_label = QLabel("Recall Probability: 0%")
        self.recall_prob_label.setAlignment(Qt.AlignCenter)
        prob_font = QFont()
        prob_font.setPointSize(10)
        self.recall_prob_label.setFont(prob_font)
        content_layout.addWidget(self.recall_prob_label)
        
        # Stored motifs section
        motifs_title = QLabel("Stored Motifs:")
        motifs_title_font = QFont()
        motifs_title_font.setPointSize(10)
        motifs_title_font.setBold(True)
        motifs_title.setFont(motifs_title_font)
        content_layout.addWidget(motifs_title)
        
        self.motifs_list = QListWidget()
        self.motifs_list.setMaximumHeight(100)
        self.motifs_list.setStyleSheet("""
            QListWidget {
                background-color: #2C2C2C;
                border: 1px solid #404040;
                color: #DCDCDC;
            }
        """)
        content_layout.addWidget(self.motifs_list)
        
        # Recent events section
        events_title = QLabel("Recent Events:")
        events_title_font = QFont()
        events_title_font.setPointSize(10)
        events_title_font.setBold(True)
        events_title.setFont(events_title_font)
        content_layout.addWidget(events_title)
        
        self.events_list = QListWidget()
        self.events_list.setStyleSheet("""
            QListWidget {
                background-color: #2C2C2C;
                border: 1px solid #404040;
                color: #DCDCDC;
                font-size: 9pt;
            }
        """)
        content_layout.addWidget(self.events_list)
    
    def _update_display(self, data: Dict[str, Any]):
        """Update display with new phrase memory data"""
        action = data.get('action', 'unknown')
        
        # Handle different action types
        if action == 'store':
            self._handle_store(data)
        elif action == 'recall':
            self._handle_recall(data)
        elif action == 'variation':
            self._handle_variation(data)
        elif action == 'update_probability':
            self._handle_probability_update(data)
        elif action == 'theme_set':
            self._handle_theme_set(data)
    
    def _handle_store(self, data: Dict[str, Any]):
        """Handle motif storage event"""
        motif = data.get('motif', [])
        timestamp = data.get('timestamp', 0)
        
        # Add to stored motifs
        motif_str = self._format_motif(motif)
        self.stored_motifs.append({'motif': motif, 'timestamp': timestamp})
        
        # Update motifs list
        self.motifs_list.addItem(f"Motif {len(self.stored_motifs)}: {motif_str}")
        
        # Add to events list
        event_text = f"[{self._format_time(timestamp)}] STORED: {motif_str}"
        self._add_event(event_text, QColor(76, 175, 80))  # Green
    
    def _handle_recall(self, data: Dict[str, Any]):
        """Handle motif recall event"""
        motif = data.get('motif', [])
        timestamp = data.get('timestamp', 0)
        
        motif_str = self._format_motif(motif)
        
        # Add to events list
        event_text = f"[{self._format_time(timestamp)}] RECALLED: {motif_str}"
        self._add_event(event_text, QColor(33, 150, 243))  # Blue
        
        # Highlight theme
        self.theme_label.setText(f"Current Theme: {motif_str}")
    
    def _handle_variation(self, data: Dict[str, Any]):
        """Handle variation application event"""
        motif = data.get('motif', [])
        variation_type = data.get('variation_type', 'unknown')
        timestamp = data.get('timestamp', 0)
        
        motif_str = self._format_motif(motif)
        
        # Add to events list
        event_text = f"[{self._format_time(timestamp)}] VARIATION ({variation_type}): {motif_str}"
        self._add_event(event_text, QColor(255, 152, 0))  # Orange
    
    def _handle_probability_update(self, data: Dict[str, Any]):
        """Handle recall probability update"""
        probability = data.get('probability', 0)
        
        # Update probability display
        self.recall_prob_label.setText(f"Recall Probability: {int(probability * 100)}%")
        
        # Color code based on probability
        if probability >= 0.7:
            color = "#4CAF50"  # Green (high probability)
        elif probability >= 0.4:
            color = "#FFC107"  # Yellow (medium)
        else:
            color = "#F44336"  # Red (low)
        
        self.recall_prob_label.setStyleSheet(f"""
            QLabel {{
                color: {color};
            }}
        """)
    
    def _handle_theme_set(self, data: Dict[str, Any]):
        """Handle current theme update"""
        motif = data.get('motif', [])
        
        if motif:
            motif_str = self._format_motif(motif)
            self.theme_label.setText(f"Current Theme: {motif_str}")
        else:
            self.theme_label.setText("Current Theme: None")
    
    def _add_event(self, text: str, color: QColor):
        """Add an event to the events list"""
        item = QListWidgetItem(text)
        item.setForeground(color)
        self.events_list.insertItem(0, item)  # Add to top
        
        # Trim list if too long
        while self.events_list.count() > self.max_events:
            self.events_list.takeItem(self.events_list.count() - 1)
    
    def _format_motif(self, motif: List[int]) -> str:
        """Format motif as note names"""
        if not motif:
            return "[]"
        
        # Simple MIDI to note name conversion (C4 = 60)
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        notes = []
        for midi in motif[:5]:  # Show first 5 notes
            octave = (midi // 12) - 1
            note = note_names[midi % 12]
            notes.append(f"{note}{octave}")
        
        if len(motif) > 5:
            notes.append("...")
        
        return "[" + " ".join(notes) + "]"
    
    def _format_time(self, timestamp: float) -> str:
        """Format timestamp as MM:SS"""
        minutes = int(timestamp // 60)
        seconds = int(timestamp % 60)
        return f"{minutes:02d}:{seconds:02d}"
    
    def clear(self):
        """Clear the viewport"""
        self.theme_label.setText("Current Theme: None")
        self.recall_prob_label.setText("Recall Probability: 0%")
        self.motifs_list.clear()
        self.events_list.clear()
        self.stored_motifs.clear()


if __name__ == "__main__":
    # Test the phrase memory viewport
    print("🧪 Testing PhraseMemoryViewport")
    
    from PyQt5.QtWidgets import QApplication
    import sys
    import time
    
    app = QApplication(sys.argv)
    
    viewport = PhraseMemoryViewport()
    viewport.setGeometry(100, 100, 500, 600)
    viewport.show()
    
    print("✅ Viewport created")
    print("✅ Simulating phrase memory events...")
    
    # Simulate events
    test_events = [
        {'action': 'store', 'motif': [65, 67, 69, 67], 'timestamp': 45.2},
        {'action': 'update_probability', 'probability': 0.3},
        {'action': 'update_probability', 'probability': 0.6},
        {'action': 'recall', 'motif': [65, 67, 69, 67], 'timestamp': 98.7},
        {'action': 'variation', 'motif': [72, 74, 76, 74], 'variation_type': 'transpose', 'timestamp': 105.4},
        {'action': 'variation', 'motif': [67, 69, 67, 65], 'variation_type': 'retrograde', 'timestamp': 112.1},
    ]
    
    for i, event in enumerate(test_events):
        print(f"   Event {i+1}: {event['action']}")
        viewport.update_data(event)
        app.processEvents()
        time.sleep(1.5)
    
    print("\n✅ PhraseMemoryViewport tests complete!")
    print("Close the window to exit...")
    
    sys.exit(app.exec_())

