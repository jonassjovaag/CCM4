#!/usr/bin/env python3
"""
Simple Bass Response Test - No MIDI Output
Focus: Test chord detection and simulate bass responses
"""

import time
import json
import os
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

# Import our core components
from listener.jhs_listener_core import DriftListener, Event
from listener.harmonic_context import RealtimeHarmonicDetector

@dataclass
class BassResponse:
    """Record of bass response to chord change"""
    timestamp: float
    chord_context: str
    bass_notes: List[int]
    response_type: str
    confidence: float

class SimpleBassTest:
    """Test bass response to chord changes (simulation only)"""
    
    def __init__(self):
        self.responses = []
        self.current_chord = None
        self.last_chord_time = 0.0
        self.last_bass_time = 0.0
        
        # Expected progression with typical bass notes
        self.expected_progression = [
            ("C", [36, 48]),      # C major: C2, C3
            ("F", [41, 53]),      # F major: F2, F3  
            ("Gm", [43, 55]),     # G minor: G2, G3
            ("Bb", [46, 58]),     # Bb major: Bb2, Bb3
            ("Cm", [36, 48]),     # C minor: C2, C3
            ("C", [36, 48])       # C major: C2, C3
        ]
        self.progression_index = 0
        
        # Test settings
        self.test_duration = 60.0
        self.chord_change_threshold = 2.0  # seconds for chord stability
        self.bass_response_delay = 1.0  # seconds after chord change
        
        # Initialize components
        self._initialize_components()
        
    def _initialize_components(self):
        """Initialize test components"""
        print("🔧 Initializing simple bass response test...")
        
        # Audio listener
        self.listener = DriftListener(
            ref_fn=self._get_reference_frequency,
            a4_fn=lambda: 440.0,
            device=None,
            sr=44100,
            frame=2048,
            hop=512
        )
        
        # Initialize REAL harmonic detector (same as MusicHal_9000)
        self.harmonic_detector = RealtimeHarmonicDetector()
        
        print("✅ Components initialized")
    
    def _get_reference_frequency(self, midi_note: int) -> float:
        """Get reference frequency for MIDI note"""
        return 440.0 * (2 ** ((midi_note - 69) / 12))
    
    def _on_audio_event(self, *args):
        """Handle audio events"""
        if not args or len(args) == 0:
            return
            
        # Handle different callback patterns
        if len(args) == 1 and isinstance(args[0], Event):
            event = args[0]
        elif len(args) == 9:
            # Handle the old callback pattern: (None, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False)
            # This indicates silence, so we can skip processing
            return
        else:
            return
        
        current_time = time.time()
        
        # Use REAL harmonic detection (same as MusicHal_9000)
        self._detect_harmonic_context(event, current_time)
        
        # Check if we should generate bass response
        self._check_bass_response(event, current_time)
        
        # Print live status
        self._print_status(event)
    
    def _detect_harmonic_context(self, event: Event, current_time: float):
        """Use REAL harmonic detection with actual audio from microphone"""
        if event.f0 <= 0:
            return
        
        # Get the actual audio buffer from DriftListener
        # The DriftListener already has the real audio frame from your microphone
        audio_buffer = self._get_real_audio_buffer()
        
        if audio_buffer is not None and len(audio_buffer) > 0:
            # Use the real harmonic detector with actual audio
            harmonic_context = self.harmonic_detector.update_from_audio(audio_buffer, sr=44100)
            
            # Check for chord changes
            detected_chord = harmonic_context.current_chord
            if detected_chord and detected_chord != self.current_chord:
                if current_time - self.last_chord_time > self.chord_change_threshold:
                    self._handle_chord_change(event, current_time, detected_chord, harmonic_context)
                    self.current_chord = detected_chord
                    self.last_chord_time = current_time
    
    def _get_real_audio_buffer(self) -> Optional[np.ndarray]:
        """Get the actual audio buffer from DriftListener"""
        # Access the DriftListener's internal audio buffer
        if hasattr(self.listener, '_ring') and self.listener._ring is not None:
            # Get the most recent audio frame
            ring_pos = getattr(self.listener, '_ring_pos', 0)
            frame_size = getattr(self.listener, 'frame', 2048)
            
            # Extract the current frame from the ring buffer
            idx = (np.arange(frame_size) + ring_pos) % frame_size
            return self.listener._ring[idx].copy()
        
        return None
    
    def _midi_to_note_name(self, midi_note: int) -> str:
        """Convert MIDI note to note name"""
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        return note_names[midi_note % 12]
    
    def _handle_chord_change(self, event: Event, timestamp: float, chord: str, harmonic_context=None):
        """Handle detected chord change"""
        # Check if matches expected
        expected_chord = None
        expected_bass = None
        if self.progression_index < len(self.expected_progression):
            expected_chord, expected_bass = self.expected_progression[self.progression_index]
        
        match = "✅" if chord == expected_chord else "❌"
        
        print(f"\n🎵 CHORD CHANGE: {chord}")
        print(f"   Expected: {expected_chord} | Status: {match}")
        print(f"   Expected bass: {expected_bass}")
        print(f"   Current MIDI: {event.midi} ({self._midi_to_note_name(event.midi)})")
        
        if harmonic_context:
            print(f"   Key: {harmonic_context.key_signature}")
            print(f"   Confidence: {harmonic_context.confidence:.2f}")
            print(f"   Stability: {harmonic_context.stability:.2f}")
        
        if not match:
            print(f"   💡 Tip: Try playing {expected_chord} bass note (MIDI: {expected_bass[0] if expected_bass else '?'})")
        
        # Move to next expected chord if match
        if chord == expected_chord:
            self.progression_index += 1
            print(f"   🎉 Progress: {self.progression_index}/{len(self.expected_progression)}")
        
        # Update current chord
        self.current_chord = chord
        self.last_chord_time = timestamp
    
    def _check_bass_response(self, event: Event, current_time: float):
        """Check if we should generate bass response"""
        if not self.current_chord:
            return
        
        # Wait for response delay after chord change
        if current_time - self.last_chord_time < self.bass_response_delay:
            return
        
        # Don't respond too frequently
        if current_time - self.last_bass_time < 3.0:
            return
        
        # Generate bass response
        self._generate_bass_response(event, current_time)
        self.last_bass_time = current_time
    
    def _generate_bass_response(self, event: Event, current_time: float):
        """Generate bass response to current chord (simulation)"""
        try:
            # Simple bass response based on chord
            bass_notes = self._get_chord_bass_notes(self.current_chord)
            
            if bass_notes:
                # Simulate sending bass notes (no actual MIDI output)
                print(f"🎸 SIMULATED Bass response: {bass_notes} (chord: {self.current_chord})")
                
                # Record response
                response = BassResponse(
                    timestamp=current_time,
                    chord_context=self.current_chord,
                    bass_notes=bass_notes,
                    response_type="chord_bass",
                    confidence=0.8
                )
                self.responses.append(response)
                
        except Exception as e:
            print(f"⚠️ Bass response error: {e}")
    
    def _get_chord_bass_notes(self, chord: str) -> List[int]:
        """Get appropriate bass notes for chord"""
        # Simple mapping of chord roots to bass notes
        chord_bass_map = {
            "C": [36, 48],    # C2, C3
            "F": [41, 53],    # F2, F3
            "G": [43, 55],    # G2, G3
            "Bb": [46, 58],   # Bb2, Bb3
            "A": [45, 57],    # A2, A3
            "D": [38, 50],    # D2, D3
            "E": [40, 52],    # E2, E3
            "D#": [39, 51],   # D#2, D#3 (Eb)
            "C#": [37, 49],   # C#2, C#3 (Db)
            "F#": [42, 54],   # F#2, F#3 (Gb)
            "G#": [44, 56],   # G#2, G#3 (Ab)
            "A#": [46, 58],   # A#2, A#3 (Bb)
        }
        
        return chord_bass_map.get(chord, [36])  # Default to C2 if unknown
    
    def _print_status(self, event: Event):
        """Print live status"""
        if event.f0 > 0:
            note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            note_name = note_names[event.midi % 12]
            octave = (event.midi // 12) - 1
            
            chord_info = f"| Chord: {self.current_chord}" if self.current_chord else ""
            
            status = (f"\r🎹 {note_name}{octave} ({event.f0:.1f}Hz) "
                     f"| RMS: {event.rms_db:.1f}dB {chord_info} "
                     f"| Progress: {self.progression_index}/{len(self.expected_progression)} "
                     f"| Bass responses: {len(self.responses)}")
            
            print(status.ljust(120), end='', flush=True)
    
    def start(self):
        """Start the bass response test"""
        print("🎵 Simple Bass Response Test (Simulation)")
        print("=" * 50)
        print("Expected progression: C → F → Gm → Bb → Cm → C")
        print("Focus: Bass response simulation to harmonic changes")
        print("Duration: 60 seconds")
        print("=" * 50)
        
        try:
            # Start listener
            self.listener.start(self._on_audio_event)
            print("✅ Test started! Play the chord progression...")
            print("   Expected: C - F - Gm - Bb - Cm - C")
            print("   Focus on bass notes (C2-C4 range)")
            print("   Play each chord for 3+ seconds before moving to next")
            print("   Bass responses will be simulated (no actual MIDI output)")
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
        print(f"\n🏁 BASS RESPONSE TEST COMPLETED!")
        print(f"Chord progression completion: {self.progression_index}/{len(self.expected_progression)}")
        print(f"Bass responses generated: {len(self.responses)}")
        
        if self.responses:
            print(f"\n🎸 BASS RESPONSES:")
            for i, resp in enumerate(self.responses):
                print(f"   {i+1}. Chord: {resp.chord_context}")
                print(f"      Bass notes: {resp.bass_notes}")
                print(f"      Response type: {resp.response_type}")
                print(f"      Confidence: {resp.confidence:.2f}")
        
        # Save detailed report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"logs/simple_bass_test_{timestamp}.json"
        
        os.makedirs("logs", exist_ok=True)
        
        report = {
            'test_info': {
                'timestamp': timestamp,
                'expected_progression': [p[0] for p in self.expected_progression],
                'expected_bass_notes': [p[1] for p in self.expected_progression],
                'completion': f"{self.progression_index}/{len(self.expected_progression)}",
                'bass_responses_count': len(self.responses)
            },
            'bass_responses': [
                {
                    'timestamp': r.timestamp,
                    'chord_context': r.chord_context,
                    'bass_notes': r.bass_notes,
                    'response_type': r.response_type,
                    'confidence': r.confidence
                } for r in self.responses
            ]
        }
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📊 Report saved to: {report_file}")
    
    def stop(self):
        """Stop the test"""
        print("\n🛑 Stopping test...")
        
        if self.listener:
            self.listener.stop()
        
        print("✅ Test stopped")

def main():
    """Main function"""
    test = SimpleBassTest()
    
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
