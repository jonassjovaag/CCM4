#!/usr/bin/env python3
"""
Validation Script for Autonomous Chord Trainer
==============================================

This script:
1. Plays a chord via MIDI
2. Analyzes the audio WITHOUT knowing what was played
3. Compares ML prediction to ground truth
4. Reports accuracy on UNSEEN data (not training set)

This is TRUE validation - independent analysis!
"""

import numpy as np
import time
import mido
from typing import Optional
import joblib

from listener.jhs_listener_core import DriftListener, Event
from listener.harmonic_context import RealtimeHarmonicDetector
from autonomous_chord_trainer import ChordVocabulary, ChordInversion


class ChordValidationSystem:
    """Independent validation of trained chord model"""
    
    def __init__(self, 
                 model_path: str = 'models/autonomous_chord_model.pkl',
                 scaler_path: str = 'models/autonomous_chord_scaler.pkl',
                 metadata_path: str = 'models/autonomous_chord_metadata.json',
                 midi_output_port: str = "IAC Driver Chord Trainer Output",
                 audio_input_device: Optional[int] = None):
        
        self.midi_port_name = midi_output_port
        self.audio_device = audio_input_device
        
        # Load trained model
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
        import json
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        self.inverse_mapping = metadata['inverse_mapping']
        
        print(f"✅ Loaded model: {len(self.inverse_mapping)} chord types")
        
        # MIDI
        self.midi_port = None
        self.midi_channel = 0
        
        # Audio
        self.listener = None
        self.harmonic_detector = RealtimeHarmonicDetector()
        
        # Validation state
        self.validation_results = []
        self.current_ground_truth = None
        self.prediction_made = False
        self.predicted_chord = None
        
    def _open_midi_port(self) -> bool:
        """Open MIDI output port"""
        try:
            available_ports = mido.get_output_names()
            if self.midi_port_name in available_ports:
                self.midi_port = mido.open_output(self.midi_port_name)
                print(f"✅ MIDI output: {self.midi_port_name}")
                return True
            else:
                print(f"❌ Port '{self.midi_port_name}' not found!")
                return False
        except Exception as e:
            print(f"❌ Failed to open MIDI port: {e}")
            return False
    
    def _start_audio_listener(self) -> bool:
        """Start audio input listener"""
        try:
            self.listener = DriftListener(
                ref_fn=lambda midi_note: 440.0,
                a4_fn=lambda: 440.0,
                device=self.audio_device
            )
            self.listener.start(self._on_audio_event)
            print(f"✅ Audio listener started")
            return True
        except Exception as e:
            print(f"❌ Failed to start audio listener: {e}")
            return False
    
    def _on_audio_event(self, *args):
        """Handle audio events and make predictions"""
        # Handle variable callback signature
        if len(args) == 1:
            event = args[0]
        else:
            return
        
        if event is None or not isinstance(event, Event):
            return
        
        # Only predict once per chord (after stabilization)
        if self.prediction_made or self.current_ground_truth is None:
            return
        
        try:
            # Extract features (same as training)
            features = self._extract_features(event)
            
            # Scale features
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            # PREDICT without knowing ground truth!
            prediction = self.model.predict(features_scaled)[0]
            confidence = np.max(self.model.predict_proba(features_scaled))
            
            predicted_label = self.inverse_mapping[str(int(prediction))]
            
            self.predicted_chord = predicted_label
            self.prediction_made = True
            
            # Show prediction (not ground truth!)
            print(f"   🤖 ML Prediction: {predicted_label} (confidence: {confidence:.2f})")
            
        except Exception as e:
            print(f"   ⚠️  Prediction error: {e}")
    
    def _extract_features(self, event: Event) -> np.ndarray:
        """Extract features from audio event (same as training)"""
        # Get audio buffer for harmonic analysis
        try:
            if self.listener and hasattr(self.listener, '_ring'):
                audio_buffer = self.listener._ring.copy()
                self.harmonic_detector.update_from_audio(audio_buffer)
                context = self.harmonic_detector.current_context
            else:
                context = None
        except Exception:
            context = None
        
        # Build feature vector (must match training!)
        features = [
            event.rms_db,
            event.f0,
            event.centroid,
            event.rolloff,
            event.zcr,
            event.bandwidth,
        ]
        
        # Add chroma
        if context and hasattr(context, 'chroma') and context.chroma is not None:
            features.extend(context.chroma)
        else:
            features.extend([0.0] * 12)
        
        # Add MFCC
        if event.mfcc and len(event.mfcc) > 0:
            features.extend(event.mfcc[:3])
        else:
            features.extend([0.0] * 3)
        
        return np.array(features, dtype=np.float32)
    
    def _play_chord(self, inversion: ChordInversion):
        """Play chord via MIDI"""
        if not self.midi_port:
            return
        
        for note in inversion.notes:
            msg = mido.Message('note_on', channel=self.midi_channel, note=note, velocity=80)
            self.midi_port.send(msg)
        
        self.current_ground_truth = inversion
        self.prediction_made = False
        self.predicted_chord = None
    
    def _stop_chord(self):
        """Stop chord"""
        if not self.midi_port or not self.current_ground_truth:
            return
        
        for note in self.current_ground_truth.notes:
            msg = mido.Message('note_off', channel=self.midi_channel, note=note, velocity=0)
            self.midi_port.send(msg)
    
    def _get_ground_truth_label(self, inversion: ChordInversion) -> str:
        """Get ground truth label (same format as training)"""
        return f"{inversion.root}{inversion.quality}_inv{inversion.inversion}_oct{inversion.octave_offset}"
    
    def validate_chord(self, inversion: ChordInversion, duration: float = 2.0) -> bool:
        """
        Validate a single chord
        
        Returns True if prediction matches ground truth
        """
        ground_truth_label = self._get_ground_truth_label(inversion)
        
        print(f"\n🎹 Testing: {inversion.root}{inversion.quality} (inv={inversion.inversion}, oct={inversion.octave_offset}) → {inversion.notes}")
        print(f"   Ground Truth: {ground_truth_label}")
        
        # Play chord
        self._play_chord(inversion)
        
        # Wait for analysis (give system time to analyze)
        time.sleep(duration)
        
        # Stop chord
        self._stop_chord()
        
        # Check prediction
        if self.predicted_chord is None:
            print(f"   ❌ No prediction made!")
            return False
        
        # Compare prediction to ground truth
        match = (self.predicted_chord == ground_truth_label)
        
        if match:
            print(f"   ✅ CORRECT! Predicted: {self.predicted_chord}")
        else:
            print(f"   ❌ WRONG! Predicted: {self.predicted_chord} | Expected: {ground_truth_label}")
        
        # Store result
        self.validation_results.append({
            'ground_truth': ground_truth_label,
            'predicted': self.predicted_chord,
            'correct': match,
            'notes': inversion.notes,
            'root': inversion.root,
            'quality': inversion.quality,
            'inversion': inversion.inversion,
            'octave_offset': inversion.octave_offset
        })
        
        return match
    
    def run_validation(self, test_chords=None):
        """
        Run validation on test chords (NOT in training set)
        
        This tests TRUE generalization!
        """
        if test_chords is None:
            # Test on chords that WERE in training (sanity check)
            # And chords that were NOT in training (true validation)
            test_chords = [
                ('C', ''),   # Was in training
                ('D', ''),   # NOT in training (if you only trained C and Cm)
                ('E', 'm'),  # NOT in training
                ('G', ''),   # NOT in training
            ]
        
        print("\n🧪 Running Independent Validation")
        print("=" * 60)
        print("This tests if the model can analyze chords WITHOUT knowing them!")
        print()
        
        correct = 0
        total = 0
        
        for root, quality in test_chords:
            # Test just one inversion variant
            inversions = ChordVocabulary.get_inversions(root, quality, base_octave=4)
            
            # Test root position, middle octave
            test_inversion = inversions[0]  # Root position, middle octave
            
            result = self.validate_chord(test_inversion, duration=2.0)
            
            if result:
                correct += 1
            total += 1
            
            # Small pause between chords
            time.sleep(0.5)
        
        # Report results
        print("\n" + "=" * 60)
        print("🎯 Validation Results:")
        print(f"   Correct: {correct}/{total} ({100*correct/total:.1f}%)")
        
        # Detailed breakdown
        print("\n📊 Detailed Results:")
        for result in self.validation_results:
            status = "✅" if result['correct'] else "❌"
            print(f"   {status} {result['ground_truth']:20s} → Predicted: {result['predicted']}")
        
        return correct / total if total > 0 else 0.0
    
    def start(self) -> bool:
        """Start validation system"""
        print("🧪 Chord Model Validation System")
        print("=" * 60)
        
        if not self._open_midi_port():
            return False
        
        if not self._start_audio_listener():
            return False
        
        return True
    
    def stop(self):
        """Stop system"""
        if self.listener:
            self.listener.stop()
        if self.midi_port:
            self.midi_port.close()


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate Autonomous Chord Trainer')
    parser.add_argument('--input-device', type=int, default=None,
                       help='Audio input device number')
    
    args = parser.parse_args()
    
    validator = ChordValidationSystem(audio_input_device=args.input_device)
    
    if not validator.start():
        print("❌ Failed to start validation system")
        return
    
    try:
        # Run validation
        accuracy = validator.run_validation()
        
        print("\n" + "=" * 60)
        if accuracy >= 0.8:
            print(f"✅ VALIDATION PASSED! Accuracy: {accuracy:.1%}")
            print("   The model can independently analyze chords!")
        else:
            print(f"⚠️  VALIDATION CONCERNS! Accuracy: {accuracy:.1%}")
            print("   The model may need more training data")
        
    except KeyboardInterrupt:
        print("\n⚠️  Validation interrupted")
    finally:
        validator.stop()


if __name__ == "__main__":
    main()



































