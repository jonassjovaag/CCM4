#!/usr/bin/env python3
"""
Bass Response Test
Focus: Test how well the bass responds to chord changes in C - F - Gm - Bb - Cm - C progression
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
from agent.behaviors import BehaviorEngine
from agent.phrase_generator import PhraseGenerator
from midi_io.mpe_midi_output import MPEMIDIOutput

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
        
        # Harmonic context
        self.harmonic_context = HarmonicContext()
        
        # Behavior engine (bass only)
        self.behavior_engine = BehaviorEngine()
        
        # Phrase generator (bass only)
        self.phrase_generator = PhraseGenerator(
            voice_type="bass",
            enable_silence=False
        )
        
        # MIDI output (bass only)
        self.midi_output = MPEMIDIOutput(
            port_name="IAC Driver Bass",
            channel=1,
            voice_type="bass"
        )
        
        print("✅ Components initialized")
    
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
            
            # Detect chord change
            if (self.current_chord != detected_chord and 
                current_time - self.last_chord_time > self.chord_change_threshold):
                
                self._handle_chord_change(event, current_time, detected_chord)
                self.current_chord = detected_chord
                self.last_chord_time = current_time
        
        # Check if we should generate bass response
        self._check_bass_response(event, current_time)
        
        # Print live status
        self._print_status(event)
    
    def _handle_chord_change(self, event: Event, timestamp: float, chord: str):
        """Handle detected chord change"""
        key = event.harmonic_context.key_signature if event.harmonic_context else "unknown"
        
        # Check if matches expected
        expected_chord = None
        expected_bass = None
        if self.progression_index < len(self.expected_progression):
            expected_chord, expected_bass = self.expected_progression[self.progression_index]
        
        match = "✅" if chord == expected_chord else "❌"
        
        print(f"\n🎵 CHORD CHANGE: {chord} in {key}")
        print(f"   Expected: {expected_chord} | Status: {match}")
        print(f"   Expected bass: {expected_bass}")
        
        # Move to next expected chord if match
        if chord == expected_chord:
            self.progression_index += 1
    
    def _check_bass_response(self, event: Event, current_time: float):
        """Check if we should generate bass response"""
        if not event.harmonic_context or not event.harmonic_context.current_chord:
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
            # Convert event to dict for phrase generator
            event_data = {
                'f0': event.f0,
                'midi': event.midi,
                'rms_db': event.rms_db,
                'harmonic_context': event.harmonic_context,
                'rhythmic_context': event.rhythmic_context
            }
            
            # Generate bass phrase
            phrase = self.phrase_generator.generate_phrase(
                mode="contrast",  # Respond to harmonic changes
                voice_type="bass",
                current_time=current_time,
                current_event=event_data
            )
            
            if phrase and phrase.notes:
                # Send bass notes
                bass_notes = []
                for note_data in phrase.notes:
                    self.midi_output.send_note(
                        note=note_data['note'],
                        velocity=note_data['velocity'],
                        duration=note_data['duration']
                    )
                    bass_notes.append(note_data['note'])
                
                # Record response
                response = BassResponse(
                    timestamp=current_time,
                    chord_context=event.harmonic_context.current_chord,
                    bass_notes=bass_notes,
                    response_type=phrase.arc.value if hasattr(phrase, 'arc') else 'unknown',
                    confidence=getattr(event.harmonic_context, 'confidence', 0.5)
                )
                self.responses.append(response)
                
                print(f"🎸 Bass response: {bass_notes} (chord: {event.harmonic_context.current_chord})")
                
        except Exception as e:
            print(f"⚠️ Bass response error: {e}")
    
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
            if not self.listener.start(self._on_audio_event):
                print("❌ Failed to start listener")
                return False
            
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
        
        import os
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
