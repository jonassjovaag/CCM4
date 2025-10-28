#!/usr/bin/env python3
"""
Bass Response Test - Fixed Version
Focus: Test how well the bass responds to chord changes in C - F - Gm - Bb - Cm - C progression
"""

import time
import json
import os
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

# Import our core components
from listener.jhs_listener_core import DriftListener, Event
from midi_io.mpe_midi_output import MPEMIDIOutput
from mapping.feature_mapper import MIDIParameters

@dataclass
class BassResponse:
    """Record of bass response to chord change"""
    timestamp: float
    chord_context: str
    bass_notes: List[int]
    response_type: str
    confidence: float

class BassResponseTest:
    """Test bass response to chord changes"""
    
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
        self.chord_change_threshold = 2.0
        self.bass_response_delay = 1.0  # Wait 1s after chord change before responding
        
        # Initialize components
        self._initialize_components()
        
    def _initialize_components(self):
        """Initialize test components"""
        print("🔧 Initializing bass response test components...")
        
        # Audio listener
        self.listener = DriftListener(
            ref_fn=self._get_reference_frequency,
            a4_fn=lambda: 440.0,
            device=None,
            sr=44100,
            frame=2048,
            hop=512
        )
        
        # MIDI output (bass only)
        self.midi_output = MPEMIDIOutput(
            output_port_name="IAC Driver Bass",
            enable_mpe=True
        )
        
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
        
        # Simple chord detection
        detected_chord = self._simple_chord_detection(event)
        
        # Check for chord changes
        if detected_chord and detected_chord != self.current_chord:
            if current_time - self.last_chord_time > self.chord_change_threshold:
                self._handle_chord_change(event, current_time, detected_chord)
                self.current_chord = detected_chord
                self.last_chord_time = current_time
        
        # Check if we should generate bass response
        self._check_bass_response(event, current_time)
        
        # Print live status
        self._print_status(event)
    
    def _simple_chord_detection(self, event: Event) -> Optional[str]:
        """Simple chord detection based on pitch analysis"""
        if event.f0 <= 0:
            return None
        
        # Convert frequency to note name
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        note_name = note_names[event.midi % 12]
        
        # Simple heuristic: if we're in the bass range, assume it's a chord root
        if event.midi >= 36 and event.midi <= 60:  # C2 to C4
            return note_name
        
        return None
    
    def _handle_chord_change(self, event: Event, timestamp: float, chord: str):
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
        
        # Move to next expected chord if match
        if chord == expected_chord:
            self.progression_index += 1
    
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
        """Generate bass response to current chord"""
        try:
            # Simple bass response based on chord
            bass_notes = self._get_chord_bass_notes(self.current_chord)
            
            if bass_notes:
                # Send bass notes
                for note in bass_notes:
                    midi_params = MIDIParameters(
                        note=note,
                        velocity=80,
                        duration=1.0,
                        attack_time=0.1,
                        release_time=0.1,
                        filter_cutoff=0.5,
                        modulation_depth=0.0,
                        pan=0.0,
                        reverb_amount=0.0
                    )
                    self.midi_output.send_note(
                        midi_params,
                        voice_type="bass"
                    )
                
                # Record response
                response = BassResponse(
                    timestamp=current_time,
                    chord_context=self.current_chord,
                    bass_notes=bass_notes,
                    response_type="chord_bass",
                    confidence=0.8
                )
                self.responses.append(response)
                
                print(f"🎸 Bass response: {bass_notes} (chord: {self.current_chord})")
                
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
        print("🎵 Bass Response Test")
        print("=" * 50)
        print("Expected progression: C → F → Gm → Bb → Cm → C")
        print("Focus: Bass response to harmonic changes")
        print("Duration: 60 seconds")
        print("=" * 50)
        
        try:
            # Start MIDI output
            if not self.midi_output.start():
                print("❌ Failed to start MIDI output")
                return False
            
            # Start listener
            self.listener.start(self._on_audio_event)
            print("✅ Test started! Play the chord progression...")
            print("   Expected: C - F - Gm - Bb - Cm - C")
            print("   Bass will respond to detected chord changes")
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
        report_file = f"logs/bass_response_test_{timestamp}.json"
        
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
        
        if self.midi_output:
            self.midi_output.stop()
        
        print("✅ Test stopped")

def main():
    """Main function"""
    test = BassResponseTest()
    
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
