#!/usr/bin/env python3
"""
Live Chord Model Test - Independent Validation
==============================================

This script:
1. Loads your trained model
2. You play chords on your piano
3. System analyzes and predicts WITHOUT being told what you played
4. Shows if predictions match what you actually played

This is TRUE validation - completely independent analysis!

Usage:
    python test_chord_model_live.py --input-device 2
    
Then play the chords you trained on and see if it recognizes them!
"""

import numpy as np
import time
import joblib
import json
from typing import Optional

from listener.jhs_listener_core import DriftListener, Event
from listener.harmonic_context import RealtimeHarmonicDetector


class LiveChordValidator:
    """Test chord model on live piano input"""
    
    def __init__(self, 
                 model_path: str = 'models/autonomous_chord_model.pkl',
                 scaler_path: str = 'models/autonomous_chord_scaler.pkl',
                 metadata_path: str = 'models/autonomous_chord_metadata.json',
                 audio_input_device: Optional[int] = None):
        
        # Load model
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.inverse_mapping = metadata['inverse_mapping']
            self.chord_mapping = metadata['chord_mapping']
            
            print(f"✅ Loaded model with {len(self.inverse_mapping)} chord types")
            print(f"📊 Trained on {metadata['num_samples']} samples")
            
            # Show what chords the model knows
            print(f"\n🎹 Model knows these chords:")
            chord_labels = sorted(self.chord_mapping.keys())
            for i, label in enumerate(chord_labels, 1):
                print(f"   {i:2d}. {label}")
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise
        
        self.audio_device = audio_input_device
        self.listener = None
        self.harmonic_detector = RealtimeHarmonicDetector()
        
        # Prediction state
        self.last_prediction = None
        self.last_prediction_time = 0
        self.prediction_interval = 1.0  # Predict every 1 second
        
        # Statistics
        self.total_predictions = 0
    
    def _start_audio_listener(self) -> bool:
        """Start audio listener"""
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
            print(f"❌ Failed to start listener: {e}")
            return False
    
    def _on_audio_event(self, *args):
        """Analyze audio and predict chord"""
        # Handle variable callback signature
        if len(args) == 1:
            event = args[0]
        else:
            return
        
        if event is None or not isinstance(event, Event):
            return
        
        # Rate limit predictions
        current_time = time.time()
        if current_time - self.last_prediction_time < self.prediction_interval:
            return
        
        try:
            # Extract features
            features = self._extract_features(event)
            
            # Scale and predict
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            prediction = self.model.predict(features_scaled)[0]
            confidence = np.max(self.model.predict_proba(features_scaled))
            
            # Get chord name
            predicted_label = self.inverse_mapping[str(int(prediction))]
            
            # Parse label for display
            parts = predicted_label.split('_')
            chord_name = parts[0]
            inversion = parts[1] if len(parts) > 1 else "inv0"
            octave = parts[2] if len(parts) > 2 else "oct0"
            
            # Show prediction
            print(f"\n🎵 DETECTED: {chord_name:15s} | {inversion:5s} | {octave:6s} | Confidence: {confidence:.2%}")
            print(f"   Audio: {event.f0:.1f}Hz | RMS: {event.rms_db:.1f}dB")
            
            self.last_prediction = predicted_label
            self.last_prediction_time = current_time
            self.total_predictions += 1
            
        except Exception as e:
            print(f"⚠️  Prediction error: {e}")
    
    def _extract_features(self, event: Event) -> np.ndarray:
        """Extract features (must match training!)"""
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
        
        # Build feature vector (same as training)
        features = [
            event.rms_db,
            event.f0,
            event.centroid,
            event.rolloff,
            event.zcr,
            event.bandwidth,
        ]
        
        # Chroma
        if context and hasattr(context, 'chroma') and context.chroma is not None:
            features.extend(context.chroma)
        else:
            features.extend([0.0] * 12)
        
        # MFCC
        if event.mfcc and len(event.mfcc) > 0:
            features.extend(event.mfcc[:3])
        else:
            features.extend([0.0] * 3)
        
        return np.array(features, dtype=np.float32)
    
    def run(self):
        """Run live chord detection test"""
        print("\n🎹 Play chords on your piano to test the model!")
        print("   The system will analyze and predict what you're playing")
        print("   Press Ctrl+C to stop\n")
        
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print(f"\n\n📊 Session stats: {self.total_predictions} predictions made")
            print("✅ Stopped")
    
    def start(self) -> bool:
        """Start the system"""
        print("🧪 Live Chord Model Validation")
        print("=" * 60)
        
        if not self._start_audio_listener():
            return False
        
        print("\n✅ System ready!")
        return True
    
    def stop(self):
        """Stop the system"""
        if self.listener:
            self.listener.stop()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Live Chord Model Test')
    parser.add_argument('--input-device', type=int, default=None,
                       help='Audio input device number')
    
    args = parser.parse_args()
    
    validator = LiveChordValidator(audio_input_device=args.input_device)
    
    if not validator.start():
        return
    
    try:
        validator.run()
    finally:
        validator.stop()


if __name__ == "__main__":
    main()































