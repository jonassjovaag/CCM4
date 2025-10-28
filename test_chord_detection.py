#!/usr/bin/env python3
"""
Test script for chord detection and response analysis
Tests: C - F - Gminor - Bb - Cminor - C progression
Focus: Bass response to harmonic changes
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
from listener.harmonic_context import HarmonicContext
from listener.rhythmic_context import RealtimeRhythmicDetector
from agent.behaviors import BehaviorEngine
from agent.phrase_generator import PhraseGenerator
from midi_io.mpe_midi_output import MPEMIDIOutput
from core.audio_oracle import PolyphonicAudioOracle

@dataclass
class ChordAnalysis:
    """Analysis of detected chord changes"""
    timestamp: float
    detected_chord: str
    detected_key: str
    confidence: float
    bass_response: Optional[str] = None
    response_time: Optional[float] = None
    notes_played: List[int] = None

class ChordDetectionTest:
    """Test harness for chord detection and bass response"""
    
    def __init__(self):
        self.test_results = []
        self.current_chord = None
        self.last_chord_change = 0.0
        self.bass_responses = []
        
        # Expected chord progression
        self.expected_progression = [
            "C",      # C major
            "F",      # F major  
            "Gm",     # G minor
            "Bb",     # Bb major
            "Cm",     # C minor
            "C"       # C major (return)
        ]
        self.progression_index = 0
        self.chord_change_threshold = 2.0  # seconds to detect chord change
        
        # Initialize components
        self._initialize_components()
        
        # Test configuration
        self.test_duration = 60.0  # 60 seconds test
        self.start_time = None
        
    def _initialize_components(self):
        """Initialize all required components"""
        print("🔧 Initializing test components...")
        
        # Initialize listener
        self.listener = DriftListener(
            ref_fn=self._get_reference_frequency,
            a4_fn=lambda: 440.0,
            device=None,
            sr=44100,
            frame=2048,
            hop=512
        )
        
        # Initialize harmonic context
        self.harmonic_context = HarmonicContext()
        
        # Initialize rhythmic detector
        self.rhythmic_detector = RealtimeRhythmicDetector()
        
        # Initialize behavior engine (bass only)
        self.behavior_engine = BehaviorEngine()
        
        # Initialize phrase generator (bass only)
        self.phrase_generator = PhraseGenerator(
            voice_type="bass",
            enable_silence=False  # No silence for this test
        )
        
        # Initialize MIDI output (bass only)
        self.midi_output = MPEMIDIOutput(
            port_name="IAC Driver Bass",
            channel=1,
            voice_type="bass"
        )
        
        # Initialize AudioOracle (load most recent model)
        self.audio_oracle = self._load_audio_oracle()
        
        print("✅ Components initialized")
        
    def _get_reference_frequency(self, midi_note: int) -> float:
        """Get reference frequency for MIDI note"""
        return 440.0 * (2 ** ((midi_note - 69) / 12))
        
    def _load_audio_oracle(self) -> Optional[PolyphonicAudioOracle]:
        """Load the most recent AudioOracle model"""
        try:
            import glob
            model_files = glob.glob("JSON/*_model.json")
            if not model_files:
                print("⚠️ No AudioOracle models found")
                return None
                
            # Get most recent model
            latest_model = max(model_files, key=lambda x: os.path.getmtime(x))
            print(f"🎓 Loading AudioOracle: {latest_model}")
            
            oracle = PolyphonicAudioOracle()
            oracle.load_model(latest_model)
            print(f"✅ Loaded {len(oracle.audio_frames)} audio frames")
            return oracle
            
        except Exception as e:
            print(f"⚠️ Could not load AudioOracle: {e}")
            return None
    
    def _on_audio_event(self, event: Event):
        """Handle audio events from listener"""
        if not self.start_time:
            return
            
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        # Update harmonic context
        self.harmonic_context.update_context(event)
        
        # Detect chord changes
        chord_change = self._detect_chord_change(event, current_time)
        
        if chord_change:
            self._analyze_chord_change(chord_change, event, current_time)
        
        # Generate bass response if needed
        self._generate_bass_response(event, current_time)
        
        # Print live status
        self._print_status(event, elapsed)
        
        # Check if test is complete
        if elapsed >= self.test_duration:
            self._complete_test()
    
    def _detect_chord_change(self, event: Event, current_time: float) -> bool:
        """Detect if chord has changed"""
        if not event.harmonic_context:
            return False
            
        current_chord = event.harmonic_context.current_chord
        if not current_chord:
            return False
        
        # Check if chord is different from last detected
        if self.current_chord != current_chord:
            time_since_change = current_time - self.last_chord_change
            
            # Only register change if enough time has passed
            if time_since_change >= self.chord_change_threshold:
                self.current_chord = current_chord
                self.last_chord_change = current_time
                return True
                
        return False
    
    def _analyze_chord_change(self, chord_change: bool, event: Event, current_time: float):
        """Analyze detected chord change"""
        if not event.harmonic_context:
            return
            
        detected_chord = event.harmonic_context.current_chord
        detected_key = event.harmonic_context.key_signature
        confidence = getattr(event.harmonic_context, 'confidence', 0.5)
        
        # Check if this matches expected progression
        expected_chord = None
        if self.progression_index < len(self.expected_progression):
            expected_chord = self.expected_progression[self.progression_index]
        
        match_status = "✅ MATCH" if detected_chord == expected_chord else "❌ MISMATCH"
        
        print(f"\n🎵 CHORD CHANGE DETECTED!")
        print(f"   Detected: {detected_chord} in {detected_key}")
        print(f"   Expected: {expected_chord}")
        print(f"   Status: {match_status}")
        print(f"   Confidence: {confidence:.2f}")
        
        # Record analysis
        analysis = ChordAnalysis(
            timestamp=current_time,
            detected_chord=detected_chord,
            detected_key=detected_key,
            confidence=confidence
        )
        self.test_results.append(analysis)
        
        # Move to next expected chord
        if match_status == "✅ MATCH":
            self.progression_index += 1
    
    def _generate_bass_response(self, event: Event, current_time: float):
        """Generate bass response to harmonic context"""
        if not event.harmonic_context or not self.audio_oracle:
            return
            
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
            for note_data in phrase.notes:
                self.midi_output.send_note(
                    note=note_data['note'],
                    velocity=note_data['velocity'],
                    duration=note_data['duration']
                )
            
            # Record response
            response = {
                'timestamp': current_time,
                'chord': event.harmonic_context.current_chord,
                'notes': [n['note'] for n in phrase.notes],
                'response_type': phrase.arc.value if hasattr(phrase, 'arc') else 'unknown'
            }
            self.bass_responses.append(response)
            
            print(f"🎸 Bass response: {[n['note'] for n in phrase.notes]} (chord: {event.harmonic_context.current_chord})")
    
    def _print_status(self, event: Event, elapsed: float):
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
                     f"| Time: {elapsed:.1f}s "
                     f"| Progress: {self.progression_index}/{len(self.expected_progression)}")
            
            print(status.ljust(120), end='', flush=True)
    
    def _complete_test(self):
        """Complete the test and generate report"""
        print(f"\n\n🏁 TEST COMPLETED!")
        print(f"Duration: {self.test_duration:.1f} seconds")
        print(f"Chord changes detected: {len(self.test_results)}")
        print(f"Bass responses generated: {len(self.bass_responses)}")
        
        # Generate detailed report
        self._generate_report()
        
        # Stop components
        self.stop()
    
    def _generate_report(self):
        """Generate detailed test report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"logs/chord_detection_test_{timestamp}.json"
        
        report = {
            'test_info': {
                'timestamp': timestamp,
                'duration': self.test_duration,
                'expected_progression': self.expected_progression
            },
            'results': {
                'chord_changes_detected': len(self.test_results),
                'bass_responses_generated': len(self.bass_responses),
                'progression_completion': f"{self.progression_index}/{len(self.expected_progression)}"
            },
            'chord_analyses': [
                {
                    'timestamp': r.timestamp,
                    'detected_chord': r.detected_chord,
                    'detected_key': r.detected_key,
                    'confidence': r.confidence
                } for r in self.test_results
            ],
            'bass_responses': self.bass_responses
        }
        
        # Save report
        import os
        os.makedirs("logs", exist_ok=True)
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📊 Detailed report saved to: {report_file}")
        
        # Print summary
        print(f"\n📈 SUMMARY:")
        print(f"   Expected progression: {' → '.join(self.expected_progression)}")
        print(f"   Detected chords: {len(self.test_results)}")
        print(f"   Bass responses: {len(self.bass_responses)}")
        
        if self.test_results:
            print(f"\n🎵 DETECTED CHORD CHANGES:")
            for i, result in enumerate(self.test_results):
                print(f"   {i+1}. {result.detected_chord} in {result.detected_key} (conf: {result.confidence:.2f})")
    
    def start(self):
        """Start the chord detection test"""
        print("🎵 Starting Chord Detection Test")
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
                print("❌ Failed to start audio listener")
                return False
            
            print("✅ Test started! Play the chord progression on your piano...")
            print("   Expected: C - F - Gm - Bb - Cm - C")
            print("   Focus on bass notes, harmonic changes will be detected")
            print("   Press Ctrl+C to stop early")
            
            self.start_time = time.time()
            
            # Keep running until test duration
            while time.time() - self.start_time < self.test_duration:
                time.sleep(0.1)
            
            return True
            
        except KeyboardInterrupt:
            print(f"\n⏹️ Test stopped by user")
            self._complete_test()
            return True
        except Exception as e:
            print(f"❌ Test failed: {e}")
            return False
    
    def stop(self):
        """Stop the test"""
        print("\n🛑 Stopping test...")
        
        if self.listener:
            self.listener.stop()
        
        if self.midi_output:
            self.midi_output.stop()
        
        print("✅ Test stopped")

def main():
    """Main test function"""
    print("🎵 Chord Detection Test")
    print("Testing harmonic detection and bass response")
    print("Expected progression: C - F - Gm - Bb - Cm - C")
    print()
    
    test = ChordDetectionTest()
    
    try:
        success = test.start()
        if success:
            print("✅ Test completed successfully")
        else:
            print("❌ Test failed")
    except Exception as e:
        print(f"❌ Test error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        test.stop()

if __name__ == "__main__":
    main()
