#!/usr/bin/env python3
"""
Simple Chord Detection Test
Focus: Test harmonic detection accuracy for C - F - Gm - Bb - Cm - C progression
"""

import time
import json
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

# Import our core components
from listener.jhs_listener_core import DriftListener, Event
from listener.harmonic_context import HarmonicContext

@dataclass
class ChordDetection:
    """Record of chord detection"""
    timestamp: float
    detected_chord: str
    detected_key: str
    confidence: float
    f0: float
    midi: int
    rms_db: float

class SimpleChordTest:
    """Simple test for chord detection accuracy"""
    
    def __init__(self):
        self.detections = []
        self.current_chord = None
        self.last_chord_time = 0.0
        
        # Expected progression
        self.expected_progression = ["C", "F", "Gm", "Bb", "Cm", "C"]
        self.progression_index = 0
        
        # Test settings
        self.test_duration = 60.0
        self.chord_change_threshold = 1.5  # seconds
        
        # Initialize components
        self.listener = DriftListener(
            ref_fn=self._get_reference_frequency,
            a4_fn=lambda: 440.0,
            device=None,
            sr=44100,
            frame=2048,
            hop=512
        )
        
        self.harmonic_context = HarmonicContext()
        
    def _get_reference_frequency(self, midi_note: int) -> float:
        """Get reference frequency for MIDI note"""
        return 440.0 * (2 ** ((midi_note - 69) / 12))
        
    def _on_audio_event(self, event: Event):
        """Handle audio events"""
        current_time = time.time()
        
        # Update harmonic context
        self.harmonic_context.update_context(event)
        
        # Check for chord changes
        if event.harmonic_context and event.harmonic_context.current_chord:
            detected_chord = event.harmonic_context.current_chord
            detected_key = event.harmonic_context.key_signature
            
            # Detect chord change
            if (self.current_chord != detected_chord and 
                current_time - self.last_chord_time > self.chord_change_threshold):
                
                self._record_chord_change(event, current_time, detected_chord, detected_key)
                self.current_chord = detected_chord
                self.last_chord_time = current_time
        
        # Print live status
        self._print_status(event)
    
    def _record_chord_change(self, event: Event, timestamp: float, chord: str, key: str):
        """Record detected chord change"""
        confidence = getattr(event.harmonic_context, 'confidence', 0.5)
        
        # Check if matches expected
        expected = None
        if self.progression_index < len(self.expected_progression):
            expected = self.expected_progression[self.progression_index]
        
        match = "✅" if chord == expected else "❌"
        
        print(f"\n🎵 CHORD DETECTED: {chord} in {key}")
        print(f"   Expected: {expected} | Status: {match}")
        print(f"   Confidence: {confidence:.2f}")
        print(f"   Pitch: {event.f0:.1f}Hz (MIDI: {event.midi})")
        
        # Record detection
        detection = ChordDetection(
            timestamp=timestamp,
            detected_chord=chord,
            detected_key=key,
            confidence=confidence,
            f0=event.f0,
            midi=event.midi,
            rms_db=event.rms_db
        )
        self.detections.append(detection)
        
        # Move to next expected chord if match
        if chord == expected:
            self.progression_index += 1
    
    def _print_status(self, event: Event):
        """Print live status"""
        if event.f0 > 0:
            note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            note_name = note_names[event.midi % 12]
            octave = (event.midi // 12) - 1
            
            chord_info = ""
            if event.harmonic_context and event.harmonic_context.current_chord:
                chord_info = f"| Chord: {event.harmonic_context.current_chord}"
            
            status = (f"\r🎹 {note_name}{octave} ({event.f0:.1f}Hz) "
                     f"| RMS: {event.rms_db:.1f}dB {chord_info} "
                     f"| Progress: {self.progression_index}/{len(self.expected_progression)}")
            
            print(status.ljust(100), end='', flush=True)
    
    def start(self):
        """Start the test"""
        print("🎵 Simple Chord Detection Test")
        print("=" * 40)
        print("Expected: C → F → Gm → Bb → Cm → C")
        print("Duration: 60 seconds")
        print("=" * 40)
        
        try:
            if not self.listener.start(self._on_audio_event):
                print("❌ Failed to start listener")
                return False
            
            print("✅ Test started! Play the chord progression...")
            print("   Expected: C - F - Gm - Bb - Cm - C")
            print("   Press Ctrl+C to stop early")
            
            start_time = time.time()
            
            while time.time() - start_time < self.test_duration:
                time.sleep(0.1)
            
            self._generate_report()
            return True
            
        except KeyboardInterrupt:
            print(f"\n⏹️ Test stopped by user")
            self._generate_report()
            return True
        except Exception as e:
            print(f"❌ Test failed: {e}")
            return False
    
    def _generate_report(self):
        """Generate test report"""
        print(f"\n🏁 TEST COMPLETED!")
        print(f"Chord detections: {len(self.detections)}")
        print(f"Progression completion: {self.progression_index}/{len(self.expected_progression)}")
        
        if self.detections:
            print(f"\n📊 DETECTION SUMMARY:")
            for i, det in enumerate(self.detections):
                expected = self.expected_progression[i] if i < len(self.expected_progression) else "?"
                match = "✅" if det.detected_chord == expected else "❌"
                print(f"   {i+1}. {det.detected_chord} (expected: {expected}) {match}")
                print(f"      Key: {det.detected_key}, Confidence: {det.confidence:.2f}")
        
        # Save detailed report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"logs/simple_chord_test_{timestamp}.json"
        
        import os
        os.makedirs("logs", exist_ok=True)
        
        report = {
            'test_info': {
                'timestamp': timestamp,
                'expected_progression': self.expected_progression,
                'detections_count': len(self.detections),
                'completion': f"{self.progression_index}/{len(self.expected_progression)}"
            },
            'detections': [
                {
                    'timestamp': d.timestamp,
                    'detected_chord': d.detected_chord,
                    'detected_key': d.detected_key,
                    'confidence': d.confidence,
                    'f0': d.f0,
                    'midi': d.midi,
                    'rms_db': d.rms_db
                } for d in self.detections
            ]
        }
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📊 Report saved to: {report_file}")
    
    def stop(self):
        """Stop the test"""
        if self.listener:
            self.listener.stop()

def main():
    """Main function"""
    test = SimpleChordTest()
    
    try:
        test.start()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        test.stop()

if __name__ == "__main__":
    main()
