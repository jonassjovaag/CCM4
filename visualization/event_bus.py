#!/usr/bin/env python3
"""
Visualization Event Bus
Handles thread-safe communication between MusicHal_9000 and viewports
"""

import time
from PyQt5.QtCore import QObject, pyqtSignal
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum


class EventType(Enum):
    """Types of events that can be broadcast"""
    PATTERN_MATCH = "pattern_match"
    MODE_CHANGE = "mode_change"
    PHRASE_MEMORY = "phrase_memory"
    AUDIO_ANALYSIS = "audio_analysis"
    TIMELINE_UPDATE = "timeline_update"
    HUMAN_INPUT = "human_input"
    MACHINE_OUTPUT = "machine_output"


@dataclass
class VisualizationEvent:
    """Container for visualization event data"""
    event_type: EventType
    timestamp: float
    data: Dict[str, Any]


class VisualizationEventBus(QObject):
    """
    Central event bus for viewport updates using Qt signals
    
    Thread-safe communication between:
    - MusicHal_9000 main thread (emits events)
    - Viewport update threads (receive events)
    
    Uses Qt's signal/slot mechanism for thread safety
    """
    
    # Qt signals for each event type (thread-safe)
    pattern_match_signal = pyqtSignal(dict)  # {score, state_id, gesture_token, context}
    mode_change_signal = pyqtSignal(dict)    # {mode, duration, request_params, temperature}
    phrase_memory_signal = pyqtSignal(dict)  # {action, motif, variation_type, timestamp}
    audio_analysis_signal = pyqtSignal(dict) # {waveform, onset, ratio, consonance, timestamp}
    timeline_update_signal = pyqtSignal(dict) # {event_type, mode, timestamp}
    human_input_signal = pyqtSignal(dict)    # {midi, gesture_token, timestamp}
    machine_output_signal = pyqtSignal(dict) # {notes, durations, mode, timestamp}
    gpt_reflection_signal = pyqtSignal(dict)  # {reflection, timestamp}
    
    def __init__(self):
        """Initialize event bus"""
        super().__init__()
        self._event_history: List[VisualizationEvent] = []
        self._max_history = 1000  # Keep last 1000 events
    
    def emit_pattern_match(self, 
                          score: float, 
                          state_id: int, 
                          gesture_token: int,
                          context: Optional[Dict[str, Any]] = None):
        """
        Emit pattern matching update
        
        Args:
            score: Match score 0-100
            state_id: AudioOracle state ID matched
            gesture_token: Current gesture token
            context: Additional context (recent tokens, consonance, etc.)
        """
        data = {
            'score': score,
            'state_id': state_id,
            'gesture_token': gesture_token,
            'context': context or {}
        }
        self.pattern_match_signal.emit(data)
        self._record_event(EventType.PATTERN_MATCH, data)
    
    def emit_mode_change(self,
                        mode: str,
                        duration: float,
                        request_params: Dict[str, Any],
                        temperature: float):
        """
        Emit behavior mode change
        
        Args:
            mode: Mode name (SHADOW, MIRROR, COUPLE, etc.)
            duration: Mode duration in seconds
            request_params: Request structure (primary, secondary, tertiary)
            temperature: Temperature setting
        """
        data = {
            'mode': mode,
            'duration': duration,
            'request_params': request_params,
            'temperature': temperature
        }
        self.mode_change_signal.emit(data)
        self._record_event(EventType.MODE_CHANGE, data)
    
    def emit_request_params(self,
                           mode: str,
                           request: Dict[str, Any],
                           voice_type: str):
        """
        Emit request parameters event (separate from mode change for real-time updates)
        
        Args:
            mode: Current behavior mode
            request: Request parameters dict (primary, secondary, tertiary constraints)
            voice_type: 'melodic' or 'bass'
        """
        data = {
            'mode': mode,
            'request': request,
            'voice_type': voice_type,
            'timestamp': time.time()
        }
        self.mode_change_signal.emit(data)  # Reuse mode_change_signal for request params
        self._record_event(EventType.MODE_CHANGE, data)
    
    def emit_phrase_memory(self,
                          action: str,
                          motif: Optional[List[int]] = None,
                          variation_type: Optional[str] = None,
                          timestamp: Optional[float] = None):
        """
        Emit phrase memory event
        
        Args:
            action: 'store', 'recall', 'variation'
            motif: MIDI notes of motif (if applicable)
            variation_type: Type of variation applied (if applicable)
            timestamp: Event timestamp
        """
        data = {
            'action': action,
            'motif': motif,
            'variation_type': variation_type,
            'timestamp': timestamp
        }
        self.phrase_memory_signal.emit(data)
        self._record_event(EventType.PHRASE_MEMORY, data)
    
    def emit_audio_analysis(self,
                           waveform: Optional[Any] = None,
                           onset: bool = False,
                           ratio: Optional[List[int]] = None,
                           consonance: Optional[float] = None,
                           timestamp: Optional[float] = None,
                           complexity: Optional[float] = None,
                           gesture_token: Optional[int] = None,
                           raw_gesture_token: Optional[int] = None,
                           chord_label: Optional[str] = None,
                           chord_confidence: Optional[float] = None):
        """
        Emit audio analysis update
        
        Args:
            waveform: Audio waveform data (numpy array or None)
            onset: Whether onset was detected
            ratio: Brandtsegg rhythm ratio [numerator, denominator]
            consonance: Consonance value 0-1
            timestamp: Event timestamp
            complexity: Barlow complexity value
            gesture_token: Smoothed gesture token (phrase-level)
            raw_gesture_token: Raw gesture token (onset-level)
            chord_label: Interpreted chord name from smoothed gesture
            chord_confidence: Confidence of chord interpretation
        """
        data = {
            'waveform': waveform,
            'onset': onset,
            'ratio': ratio,
            'consonance': consonance,
            'timestamp': timestamp,
            'complexity': complexity,
            'gesture_token': gesture_token,
            'raw_gesture_token': raw_gesture_token,
            'chord_label': chord_label,
            'chord_confidence': chord_confidence
        }
        self.audio_analysis_signal.emit(data)
        # Don't record audio data to history (too large)
        # self._record_event(EventType.AUDIO_ANALYSIS, data)
    
    def emit_timeline_update(self,
                            event_type: str,
                            mode: Optional[str] = None,
                            timestamp: Optional[float] = None):
        """
        Emit timeline event
        
        Args:
            event_type: 'mode_change', 'thematic_recall', 'response', 'human_input'
            mode: Current mode (if applicable)
            timestamp: Event timestamp
        """
        data = {
            'event_type': event_type,
            'mode': mode,
            'timestamp': timestamp
        }
        self.timeline_update_signal.emit(data)
        self._record_event(EventType.TIMELINE_UPDATE, data)
    
    def emit_human_input(self,
                        midi: int,
                        gesture_token: Optional[int] = None,
                        timestamp: Optional[float] = None):
        """
        Emit human input event
        
        Args:
            midi: MIDI note number
            gesture_token: Gesture token extracted from input
            timestamp: Event timestamp
        """
        data = {
            'midi': midi,
            'gesture_token': gesture_token,
            'timestamp': timestamp
        }
        self.human_input_signal.emit(data)
        self._record_event(EventType.HUMAN_INPUT, data)
    
    def emit_machine_output(self,
                           notes: List[int],
                           durations: List[float],
                           mode: str,
                           timestamp: Optional[float] = None):
        """
        Emit machine output event
        
        Args:
            notes: MIDI notes generated
            durations: Note durations in seconds
            mode: Current behavior mode
            timestamp: Event timestamp
        """
        data = {
            'notes': notes,
            'durations': durations,
            'mode': mode,
            'timestamp': timestamp
        }
        self.machine_output_signal.emit(data)
        self._record_event(EventType.MACHINE_OUTPUT, data)
    
    def emit_gpt_reflection(self,
                           reflection: str,
                           timestamp: float):
        """
        Emit GPT-OSS reflection update
        
        Args:
            reflection: Reflection text from GPT-OSS
            timestamp: Time of reflection completion
        """
        data = {
            'reflection': reflection,
            'timestamp': timestamp
        }
        self.gpt_reflection_signal.emit(data)
        # Note: Not recording in _event_history to avoid bloat
    
    def _record_event(self, event_type: EventType, data: Dict[str, Any]):
        """
        Record event to history (for replay/analysis)
        
        Args:
            event_type: Type of event
            data: Event data
        """
        import time
        
        event = VisualizationEvent(
            event_type=event_type,
            timestamp=time.time(),
            data=data
        )
        
        self._event_history.append(event)
        
        # Trim history if too long
        if len(self._event_history) > self._max_history:
            self._event_history = self._event_history[-self._max_history:]
    
    def get_event_history(self, 
                         event_type: Optional[EventType] = None,
                         last_n: Optional[int] = None) -> List[VisualizationEvent]:
        """
        Get event history
        
        Args:
            event_type: Filter by event type (None = all types)
            last_n: Return only last N events (None = all)
        
        Returns:
            List of VisualizationEvent objects
        """
        events = self._event_history
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        if last_n:
            events = events[-last_n:]
        
        return events
    
    def clear_history(self):
        """Clear event history"""
        self._event_history.clear()


if __name__ == "__main__":
    # Test event bus (requires Qt application)
    print("🧪 Testing VisualizationEventBus")
    print("Note: Full testing requires Qt application context")
    
    from PyQt5.QtWidgets import QApplication
    import sys
    
    app = QApplication(sys.argv)
    
    bus = VisualizationEventBus()
    
    # Test emitting events
    print("\n✅ Emitting test events...")
    bus.emit_pattern_match(87.5, 234, 142, {'recent_tokens': [140, 141, 142]})
    bus.emit_mode_change('SHADOW', 47.0, {'primary': {'param': 'gesture_token'}}, 0.7)
    bus.emit_phrase_memory('store', [65, 67, 69, 67], None, 1.23)
    
    # Check history
    history = bus.get_event_history()
    print(f"\n✅ Event history: {len(history)} events recorded")
    for event in history:
        print(f"   {event.event_type.value}: {list(event.data.keys())}")
    
    print("\n✅ Event bus tests complete!")

