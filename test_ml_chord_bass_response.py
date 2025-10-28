#!/usr/bin/env python3
"""
ML-Trained Chord Detection with Bass Response
============================================

This script demonstrates how the ML-trained chord detection can be integrated
with bass response using MPE MIDI output, similar to MusicHal_9000.

Features:
- Uses ML-trained chord detection (more accurate than rule-based)
- Generates bass responses based on detected chords
- Outputs to IAC Driver Bass using MPE MIDI
- Shows how this can benefit MusicHal_9000 and Chandra_trainer
"""

import os
import sys
import time
import json
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from listener.jhs_listener_core import DriftListener, Event
from listener.harmonic_context import RealtimeHarmonicDetector, HarmonicContext
from midi_io.mpe_midi_output import MPEMIDIOutput
from mapping.feature_mapper import MIDIParameters
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

class MLChordBassResponse:
    """ML-trained chord detection with bass response system"""
    
    def __init__(self):
        self.listener = None
        self.harmonic_detector = RealtimeHarmonicDetector()
        self.midi_output = None
        self.ml_model = None
        self.scaler = None
        
        # Bass response mapping
        self.bass_mappings = {
            # Major chords
            'C': [36, 48], 'C#': [37, 49], 'D': [38, 50], 'D#': [39, 51], 
            'E': [40, 52], 'F': [41, 53], 'F#': [42, 54], 'G': [43, 55],
            'G#': [44, 56], 'A': [45, 57], 'A#': [46, 58], 'B': [47, 59],
            
            # Minor chords
            'Cm': [36, 48], 'C#m': [37, 49], 'Dm': [38, 50], 'D#m': [39, 51],
            'Em': [40, 52], 'Fm': [41, 53], 'F#m': [42, 54], 'Gm': [43, 55],
            'G#m': [44, 56], 'Am': [45, 57], 'A#m': [46, 58], 'Bm': [47, 59],
            
            # 7th chords
            'C7': [36, 48], 'Cm7': [36, 48], 'Cmaj7': [36, 48],
            'D7': [38, 50], 'Dm7': [38, 50], 'Dmaj7': [38, 50],
            'E7': [40, 52], 'Em7': [40, 52], 'Emaj7': [40, 52],
            'F7': [41, 53], 'Fm7': [41, 53], 'Fmaj7': [41, 53],
            'G7': [43, 55], 'Gm7': [43, 55], 'Gmaj7': [43, 55],
            'A7': [45, 57], 'Am7': [45, 57], 'Amaj7': [45, 57],
            'B7': [47, 59], 'Bm7': [47, 59], 'Bmaj7': [47, 59],
            
            # Jazz extensions (use root + 5th)
            'C9': [36, 48], 'Cm9': [36, 48], 'Cmaj9': [36, 48],
            'C11': [36, 48], 'Cm11': [36, 48], 'Cmaj11': [36, 48],
            'C13': [36, 48], 'Cm13': [36, 48], 'Cmaj13': [36, 48],
            
            # Diminished
            'Cdim': [36, 48], 'Cdim7': [36, 48], 'Cm7b5': [36, 48],
            
            # Augmented
            'Caug': [36, 48],
            
            # Suspended
            'Csus2': [36, 48], 'Csus4': [36, 48],
            
            # Altered dominants
            'C7alt': [36, 48], 'C7b9': [36, 48], 'C7#9': [36, 48],
            'C7b5': [36, 48], 'C7#5': [36, 48],
        }
        
        # Extend mappings for all chromatic notes
        self._extend_chromatic_mappings()
        
        self.current_chord = None
        self.last_bass_time = 0
        self.bass_response_delay = 1.0  # seconds
        
    def _extend_chromatic_mappings(self):
        """Extend bass mappings to all chromatic variations"""
        chromatic_notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        chromatic_notes_flat = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
        
        # Add flat equivalents
        for sharp, flat in zip(chromatic_notes, chromatic_notes_flat):
            if sharp != flat:
                for suffix in ['', 'm', '7', 'm7', 'maj7', '9', 'm9', 'maj9', '11', 'm11', 'maj11', '13', 'm13', 'maj13', 'dim', 'dim7', 'm7b5', 'aug', 'sus2', 'sus4', '7alt', '7b9', '7#9', '7b5', '7#5']:
                    if sharp + suffix in self.bass_mappings:
                        self.bass_mappings[flat + suffix] = self.bass_mappings[sharp + suffix]
    
    def _get_reference_frequency(self, midi_note: int) -> float:
        """Get reference frequency for MIDI note"""
        return 440.0 * (2 ** ((midi_note - 69) / 12))
    
    def load_ml_model(self):
        """Load the trained ML model"""
        model_path = "models/chord_model.pkl"
        scaler_path = "models/chord_scaler.pkl"
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            try:
                self.ml_model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                print(f"✅ Loaded ML model: {model_path}")
                print(f"✅ Loaded scaler: {scaler_path}")
                return True
            except Exception as e:
                print(f"❌ Error loading ML model: {e}")
                return False
        else:
            print(f"❌ ML model not found: {model_path}")
            print("💡 Train a model first using test_interactive_chord_trainer.py")
            return False
    
    def _extract_features(self, event: Event, harmonic_context) -> np.ndarray:
        """Extract features for ML prediction"""
        # Chroma vector (12 features)
        chroma = harmonic_context.chroma if hasattr(harmonic_context, 'chroma') else np.zeros(12)
        
        # Additional features
        confidence = harmonic_context.confidence
        stability = harmonic_context.stability
        rms_db = event.rms_db
        f0 = event.f0
        chord_history_length = 0  # Not available in HarmonicContext
        
        # Combine all features
        features = np.concatenate([
            chroma,
            [confidence, stability, rms_db, f0, chord_history_length]
        ])
        
        return features.reshape(1, -1)
    
    def predict_chord_ml(self, event: Event, harmonic_context) -> str:
        """Predict chord using ML model"""
        if self.ml_model is None or self.scaler is None:
            return harmonic_context.current_chord
        
        try:
            features = self._extract_features(event, harmonic_context)
            features_scaled = self.scaler.transform(features)
            prediction = self.ml_model.predict(features_scaled)[0]
            confidence = self.ml_model.predict_proba(features_scaled).max()
            
            print(f"🤖 ML Prediction: {prediction} (confidence: {confidence:.2f})")
            return prediction
        except Exception as e:
            print(f"⚠️ ML prediction error: {e}")
            return harmonic_context.current_chord
    
    def get_bass_notes(self, chord) -> list:
        """Get bass notes for a chord"""
        # Handle numeric predictions from ML model (numpy.int64 or string)
        chord_str = str(chord)
        if chord_str.isdigit():
            # Convert numeric prediction to chord name
            chord_names = ['C', 'Cm', 'D', 'Dm', 'E', 'Em', 'F', 'Fm', 'G', 'Gm', 'A', 'Am', 'B', 'Bm']
            chord_idx = int(chord_str) % len(chord_names)
            chord_str = chord_names[chord_idx]
        
        if chord_str in self.bass_mappings:
            return self.bass_mappings[chord_str]
        else:
            # Default fallback - use root note
            root_midi = self._chord_to_root_midi(chord_str)
            return [root_midi, root_midi + 12]
    
    def _midi_to_note_name(self, midi: int) -> str:
        """Convert MIDI note number to note name"""
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        note = note_names[midi % 12]
        octave = (midi // 12) - 1
        return f"{note}{octave}"
    
    def _chord_to_root_midi(self, chord: str) -> int:
        """Convert chord name to root MIDI note"""
        note_map = {
            'C': 60, 'C#': 61, 'Db': 61, 'D': 62, 'D#': 63, 'Eb': 63,
            'E': 64, 'F': 65, 'F#': 66, 'Gb': 66, 'G': 67, 'G#': 68,
            'Ab': 68, 'A': 69, 'A#': 70, 'Bb': 70, 'B': 71
        }
        
        # Extract root note
        root = chord[0]
        if len(chord) > 1 and chord[1] in ['#', 'b']:
            root = chord[:2]
        
        return note_map.get(root, 60)  # Default to C
    
    def send_bass_response(self, chord: str):
        """Send bass response via MPE MIDI"""
        if self.midi_output is None:
            return
        
        bass_notes = self.get_bass_notes(chord)
        current_time = time.time()
        
        # Check if enough time has passed since last bass response
        if current_time - self.last_bass_time < self.bass_response_delay:
            return
        
        print(f"🎸 ML Bass Response: {chord} -> {bass_notes}")
        
        # Send bass notes
        for i, note in enumerate(bass_notes):
            # Create MIDI parameters
            midi_params = MIDIParameters(
                note=note,
                velocity=80,
                duration=2.0,
                attack_time=0.1,
                release_time=0.5,
                filter_cutoff=0.3,
                modulation_depth=0.2,
                pan=0.0,
                reverb_amount=0.3
            )
            
            # Send note with slight delay between notes
            time.sleep(0.1 * i)
            success = self.midi_output.send_note(midi_params, 'bass', 0.0, 0.5, 0.5, 0.5)
            print(f"🎵 MIDI sent: note={note}, success={success}")
        
        self.last_bass_time = current_time
    
    def _on_audio_event(self, *args):
        """Handle audio events"""
        try:
            event = args[0] if args else None
            if event is None:
                return
            
            current_time = time.time()
            
            # Update harmonic context
            audio_buffer = self._get_real_audio_buffer()
            if audio_buffer is not None:
                self.harmonic_detector.update_from_audio(audio_buffer)
                
                # Get harmonic context from the detector's current context
                harmonic_context = self.harmonic_detector.current_context
                
                if harmonic_context is not None:
                    # Predict chord using ML model
                    ml_chord = self.predict_chord_ml(event, harmonic_context)
                else:
                    ml_chord = "unknown"
                
                # Convert ML prediction to chord name for display
                ml_chord_str = str(ml_chord)
                if ml_chord_str.isdigit():
                    chord_names = ['C', 'Cm', 'D', 'Dm', 'E', 'Em', 'F', 'Fm', 'G', 'Gm', 'A', 'Am', 'B', 'Bm']
                    chord_idx = int(ml_chord_str) % len(chord_names)
                    chord_name = chord_names[chord_idx]
                else:
                    chord_name = ml_chord_str
                
                # Check for chord change
                if ml_chord != self.current_chord:
                    self.current_chord = ml_chord
                    print(f"🎵 ML CHORD CHANGE: {chord_name} (prediction: {ml_chord})")
                    print(f"   Confidence: {harmonic_context.confidence:.2f}")
                    print(f"   Stability: {harmonic_context.stability:.2f}")
                    
                    # Send bass response
                    self.send_bass_response(ml_chord)
                
                # Display current status
                note_name = self._midi_to_note_name(event.midi)
                print(f"🎹 {note_name} ({event.f0:.1f}Hz) | RMS: {event.rms_db:.1f}dB | ML: {chord_name} | Bass: {self.get_bass_notes(ml_chord)}")
                
        except Exception as e:
            print(f"⚠️ Audio event error: {e}")
    
    def _get_real_audio_buffer(self):
        """Get real audio buffer from listener"""
        if self.listener and hasattr(self.listener, '_ring'):
            return self.listener._ring
        return None
    
    def start(self):
        """Start the ML chord bass response system"""
        print("🎓 ML Chord Detection with Bass Response")
        print("=" * 50)
        
        # Load ML model
        if not self.load_ml_model():
            print("❌ Cannot start without ML model")
            return False
        
        # Initialize MIDI output
        try:
            # Check available MIDI ports first
            import mido
            available_ports = mido.get_output_names()
            print(f"📡 Available MIDI ports: {available_ports}")
            
            if "IAC Driver Bass" not in available_ports:
                print("⚠️ IAC Driver Bass not found in available ports!")
                print("💡 Make sure IAC Driver is enabled in Audio MIDI Setup")
                # Try to use any available port for testing
                if available_ports:
                    print(f"🔄 Using first available port: {available_ports[0]}")
                    self.midi_output = MPEMIDIOutput(output_port_name=available_ports[0])
                else:
                    print("❌ No MIDI ports available")
                    return False
            else:
                print("✅ IAC Driver Bass found in available ports")
                self.midi_output = MPEMIDIOutput(output_port_name="IAC Driver Bass")
            
            print("✅ MIDI output initialized")
            
            # Start the MIDI output
            if not self.midi_output.start():
                print("❌ Failed to start MIDI output")
                return False
            print("✅ MIDI output started")
            
        except Exception as e:
            print(f"❌ MIDI output error: {e}")
            return False
        
        # Initialize audio listener
        try:
            def ref_fn(midi_note: int) -> float:
                return self._get_reference_frequency(midi_note)
            
            def a4_fn() -> float:
                return 440.0
            
            self.listener = DriftListener(ref_fn, a4_fn)
            self.listener.start(self._on_audio_event)
            print("✅ Audio system started")
            return True
        except Exception as e:
            print(f"❌ Audio system error: {e}")
            return False
    
    def stop(self):
        """Stop the system"""
        if self.listener:
            self.listener.stop()
        if self.midi_output:
            self.midi_output.stop()
        print("✅ System stopped")

def main():
    """Main function"""
    print("🎵 ML Chord Detection with Bass Response")
    print("=" * 50)
    print("This system demonstrates:")
    print("1. ML-trained chord detection (more accurate)")
    print("2. Bass response via MPE MIDI to IAC Driver Bass")
    print("3. Integration potential with MusicHal_9000")
    print("=" * 50)
    
    system = MLChordBassResponse()
    
    try:
        if system.start():
            print("\n🎹 Play chords on your piano...")
            print("   The system will detect chords using ML and respond with bass notes")
            print("   Press Ctrl+C to stop")
            
            while True:
                time.sleep(0.1)
        else:
            print("❌ Failed to start system")
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping...")
        system.stop()
        print("✅ Stopped")

if __name__ == "__main__":
    main()
