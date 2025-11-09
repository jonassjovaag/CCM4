"""
Harmonic Context Manager - Manual Override System for MusicHal 9000

This module provides a transparent override mechanism for harmonic context,
enabling human-machine dialogue about chord perception. Part of artistic research
PhD exploring trust through transparency.

ARCHITECTURE:
- Dual perception: Detected (auto) vs Active (override or detected)
- Timeout mechanism: Overrides expire after duration (default 30s)
- Logging: All override events recorded with timestamps for research analysis
- Integration: Works with HarmonicTranslator without affecting 768D AudioOracle

RESEARCH CONTEXT:
The system's chord detection may be uncertain (20-29% confidence on real audio).
Manual override provides:
1. Transparency: See what machine thinks vs what you know
2. Agency: Correct perception when machine is wrong
3. Trust: Control builds confidence in collaboration
4. Documentation: Override logs capture perception differences

USAGE:
    manager = HarmonicContextManager()
    
    # Normal operation
    manager.set_detected_chord('D', confidence=0.24)
    chord = manager.get_active_chord()  # Returns 'D'
    
    # Override for 30 seconds
    manager.set_override('C', duration=30.0)
    chord = manager.get_active_chord()  # Returns 'C' (override active)
    
    # After 30s timeout
    chord = manager.get_active_chord()  # Returns 'D' (detected)
    
    # Check status
    status = manager.get_status()
    # {
    #     'detected_chord': 'D',
    #     'detected_confidence': 0.24,
    #     'override_chord': 'C',
    #     'override_expires_in': 23.5,  # seconds
    #     'active_chord': 'C',
    #     'override_active': True
    # }

Jonas Sjøvaag - PhD Artistic Research - University of Agder
"""

import time
from typing import Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime


@dataclass
class OverrideEvent:
    """Record of a manual override event for research documentation."""
    timestamp: float
    detected_chord: str
    detected_confidence: float
    override_chord: str
    duration: float
    reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            'timestamp': self.timestamp,
            'datetime': datetime.fromtimestamp(self.timestamp).isoformat(),
            'detected_chord': self.detected_chord,
            'detected_confidence': self.detected_confidence,
            'override_chord': self.override_chord,
            'duration': self.duration,
            'reason': self.reason
        }


class HarmonicContextManager:
    """
    Manages harmonic context with manual override capability.
    
    This class maintains two parallel states:
    1. DETECTED: What the machine perceives (auto chord detection)
    2. ACTIVE: What is actually used (override if active, otherwise detected)
    
    Overrides have a timeout mechanism - they expire after a set duration,
    returning control to automatic detection. This prevents "stuck" overrides
    from lasting entire performance.
    
    All override events are logged for post-performance analysis and artistic
    research documentation.
    """
    
    def __init__(self, default_override_duration: float = 30.0):
        """
        Initialize harmonic context manager.
        
        Args:
            default_override_duration: Default timeout for overrides in seconds (default: 30s)
        """
        # Detected state (automatic chord detection)
        self.detected_chord: Optional[str] = None
        self.detected_confidence: float = 0.0
        self.detected_timestamp: float = 0.0
        
        # Override state (manual correction)
        self.override_chord: Optional[str] = None
        self.override_timeout: float = 0.0
        self.default_override_duration = default_override_duration
        
        # Override history for research documentation
        self.override_history: list[OverrideEvent] = []
        
        # Statistics
        self.total_detections: int = 0
        self.total_overrides: int = 0
        
    def set_detected_chord(self, chord: str, confidence: float = 0.0) -> None:
        """
        Update the detected chord from automatic chord detection.
        
        This is called by the system's RealtimeHarmonicDetector during performance.
        
        Args:
            chord: Detected chord label (e.g., 'C', 'Dm', 'G7')
            confidence: Detection confidence (0.0-1.0)
        """
        self.detected_chord = chord
        self.detected_confidence = confidence
        self.detected_timestamp = time.time()
        self.total_detections += 1
        
    def set_override(self, chord: str, duration: Optional[float] = None, reason: Optional[str] = None) -> None:
        """
        Manually override the harmonic context.
        
        This is the key method for human-machine dialogue. When you disagree with
        the detected chord, you can override it. The override lasts for `duration`
        seconds, then automatically reverts to detected chord.
        
        Args:
            chord: Override chord label (e.g., 'C', 'Dm', 'G7')
            duration: Override timeout in seconds (None = use default)
            reason: Optional explanation for research logs (e.g., "detected D, heard F#m")
        """
        if duration is None:
            duration = self.default_override_duration
            
        self.override_chord = chord
        self.override_timeout = time.time() + duration
        self.total_overrides += 1
        
        # Log override event for research documentation
        event = OverrideEvent(
            timestamp=time.time(),
            detected_chord=self.detected_chord or "None",
            detected_confidence=self.detected_confidence,
            override_chord=chord,
            duration=duration,
            reason=reason
        )
        self.override_history.append(event)
        
        # Console feedback with visual distinction
        print(f"🎹 MANUAL OVERRIDE: {chord} (for {duration:.0f}s)")
        if self.detected_chord and self.detected_chord != chord:
            print(f"   Machine detected: {self.detected_chord} (conf: {self.detected_confidence:.2f})")
            print(f"   You corrected to: {chord}")
        
    def clear_override(self) -> None:
        """
        Immediately clear any active override.
        
        Useful for "cancel override" command or emergency reset.
        """
        if self.override_chord is not None:
            print(f"🎹 Override cleared: {self.override_chord} → {self.detected_chord}")
        self.override_chord = None
        self.override_timeout = 0.0
        
    def get_active_chord(self) -> Optional[str]:
        """
        Get the currently active chord (override if active, otherwise detected).
        
        This is the main method used by PhraseGenerator to determine harmonic context.
        
        Returns:
            Active chord label, or None if no chord detected/overridden
        """
        # Check if override is still active (hasn't timed out)
        if self.override_chord is not None and time.time() < self.override_timeout:
            return self.override_chord
        
        # Override expired or not set - use detected chord
        if self.override_chord is not None and time.time() >= self.override_timeout:
            # Override just expired - log transition
            print(f"🎹 Override expired: {self.override_chord} → {self.detected_chord}")
            self.override_chord = None
            self.override_timeout = 0.0
            
        return self.detected_chord
    
    def is_override_active(self) -> bool:
        """
        Check if manual override is currently active.
        
        Returns:
            True if override active and hasn't timed out, False otherwise
        """
        return self.override_chord is not None and time.time() < self.override_timeout
    
    def get_override_time_remaining(self) -> float:
        """
        Get remaining time for active override.
        
        Returns:
            Seconds remaining, or 0.0 if no override active
        """
        if self.is_override_active():
            return max(0.0, self.override_timeout - time.time())
        return 0.0
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status for display/logging.
        
        Returns:
            Dictionary with all relevant state information
        """
        return {
            'detected_chord': self.detected_chord,
            'detected_confidence': self.detected_confidence,
            'override_chord': self.override_chord if self.is_override_active() else None,
            'override_expires_in': self.get_override_time_remaining(),
            'active_chord': self.get_active_chord(),
            'override_active': self.is_override_active(),
            'total_detections': self.total_detections,
            'total_overrides': self.total_overrides,
            'override_rate': self.total_overrides / max(1, self.total_detections)
        }
    
    def get_display_string(self) -> str:
        """
        Get formatted string for console display.
        
        Returns:
            Human-readable status string
        """
        active = self.get_active_chord() or "None"
        
        if self.is_override_active():
            remaining = self.get_override_time_remaining()
            return (f"Detected: {self.detected_chord or 'None'} (conf: {self.detected_confidence:.2f}) | "
                   f"Active: {active} (OVERRIDE - {remaining:.0f}s left)")
        else:
            return f"Detected: {self.detected_chord or 'None'} (conf: {self.detected_confidence:.2f}) | Active: {active}"
    
    def get_override_history(self) -> list[Dict[str, Any]]:
        """
        Get all override events for research analysis.
        
        Returns:
            List of override event dictionaries
        """
        return [event.to_dict() for event in self.override_history]
    
    def export_override_log(self, filepath: str) -> None:
        """
        Export override history to JSON file for research documentation.
        
        Args:
            filepath: Path to save JSON log
        """
        import json
        
        log_data = {
            'session_statistics': {
                'total_detections': self.total_detections,
                'total_overrides': self.total_overrides,
                'override_rate': self.total_overrides / max(1, self.total_detections),
                'average_confidence': sum(e.detected_confidence for e in self.override_history) / max(1, len(self.override_history))
            },
            'override_events': self.get_override_history()
        }
        
        with open(filepath, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        print(f"📊 Override log exported: {filepath}")
        print(f"   Total overrides: {self.total_overrides}")
        print(f"   Override rate: {log_data['session_statistics']['override_rate']:.1%}")


if __name__ == "__main__":
    """Test the HarmonicContextManager."""
    
    print("Testing HarmonicContextManager...\n")
    
    # Create manager with 5s timeout for testing
    manager = HarmonicContextManager(default_override_duration=5.0)
    
    # Simulate detected chord
    print("1. Detecting chord: D (confidence 0.24)")
    manager.set_detected_chord('D', confidence=0.24)
    print(f"   Active chord: {manager.get_active_chord()}")
    print(f"   Status: {manager.get_display_string()}\n")
    
    # Override chord
    print("2. Manual override to C (for 5s)")
    manager.set_override('C', reason="Detected D, but I played C")
    print(f"   Active chord: {manager.get_active_chord()}")
    print(f"   Status: {manager.get_display_string()}\n")
    
    # Wait 2 seconds
    print("3. After 2 seconds...")
    time.sleep(2)
    print(f"   Active chord: {manager.get_active_chord()}")
    print(f"   Time remaining: {manager.get_override_time_remaining():.1f}s")
    print(f"   Status: {manager.get_display_string()}\n")
    
    # Wait for timeout
    print("4. After override expires (3 more seconds)...")
    time.sleep(3.5)
    print(f"   Active chord: {manager.get_active_chord()}")
    print(f"   Status: {manager.get_display_string()}\n")
    
    # New detection
    print("5. New detection: Am (confidence 0.31)")
    manager.set_detected_chord('Am', confidence=0.31)
    print(f"   Active chord: {manager.get_active_chord()}")
    print(f"   Status: {manager.get_display_string()}\n")
    
    # Statistics
    status = manager.get_status()
    print("6. Session statistics:")
    print(f"   Total detections: {status['total_detections']}")
    print(f"   Total overrides: {status['total_overrides']}")
    print(f"   Override rate: {status['override_rate']:.1%}\n")
    
    print("✅ HarmonicContextManager test complete!")
