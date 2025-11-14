#!/usr/bin/env python3
"""
Simple Chord Training System
============================

Interactive system to train chord detection using your piano playing as ground truth.

Usage:
1. Run the script
2. Play a chord on piano
3. Press the corresponding key to label it
4. Repeat for different chords
5. Press 't' to train the model
6. Press 'q' to quit

Key mappings:
- 1-0, -= : Major chords (C, C#, D, D#, E, F, F#, G, G#, A, A#, B)
- q-p, [] : Minor chords (Cm, C#m, Dm, D#m, Em, Fm, F#m, Gm, G#m, Am, A#m, Bm)
- space : Silence
- t : Train model
- q : Quit
"""

import numpy as np
import time
import json
import os
from datetime import datetime
from typing import Dict, List, Optional
import threading
import queue

# Import our existing components
from listener.jhs_listener_core import DriftListener
from listener.harmonic_context import RealtimeHarmonicDetector
from core.event import Event

# Simple ML
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

class SimpleChordTrainer:
    """Simple chord detection trainer"""
    
    def __init__(self):
        self.listener = None
        self.harmonic_detector = RealtimeHarmonicDetector()
        
        # Training data
        self.samples = []  # List of (features, chord_label)
        self.chord_mapping = {}  # chord -> index
        self.next_chord_id = 0
        
        # State
        self.current_label = None
        self.is_collecting = False
        self.last_collection_time = 0
        
        # ML
        self.model = None
        self.scaler = StandardScaler()
        
        # Key mappings
        self.chord_keys = {
            '1': 'C', '2': 'C#', '3': 'D', '4': 'D#', '5': 'E', '6': 'F',
            '7': 'F#', '8': 'G', '9': 'G#', '0': 'A', '-': 'A#', '=': 'B',
            'q': 'Cm', 'w': 'C#m', 'e': 'Dm', 'r': 'D#m', 't': 'Em', 'y': 'Fm',
            'u': 'F#m', 'i': 'Gm', 'o': 'G#m', 'p': 'Am', '[': 'A#m', ']': 'Bm',
            'space': 'silence'
        }
        
        print("🎓 Simple Chord Training System")
        print("=" * 40)
        self._print_key_mappings()

    def _print_key_mappings(self):
        """Print key mappings"""
        print("Key mappings:")
        print("  Numbers (1-0, -=): Major chords")
        print("  Letters (q-p, []): Minor chords")
        print("  SPACE: Silence")
        print("  T: Train model")
        print("  Q: Quit")
        print("=" * 40)

    def start(self):
        """Start the training system"""
        try:
            self.listener = DriftListener()
            self.listener.start(self._on_audio_event)
            print("✅ Audio system started")
            return True
        except Exception as e:
            print(f"❌ Failed to start audio: {e}")
            return False

    def _on_audio_event(self, *args):
        """Handle audio events"""
        try:
            event = args[0] if len(args) > 0 else None
            if not event or not isinstance(event, Event):
                return
                
            # Only collect samples when we have a label
            if not self.current_label:
                return
                
            current_time = time.time()
            if current_time - self.last_collection_time < 0.5:  # Avoid too frequent collection
                return
                
            # Get audio buffer
            audio_buffer = self._get_audio_buffer()
            if audio_buffer is None or len(audio_buffer) == 0:
                return
                
            # Analyze harmonic content
            harmonic_context = self.harmonic_detector.update_from_audio(audio_buffer, sr=44100)
            
            # Extract features
            features = self._extract_features(event, harmonic_context)
            
            # Store sample
            self.samples.append((features, self.current_label))
            self.last_collection_time = current_time
            
            print(f"📝 Collected: {self.current_label} (total: {len(self.samples)})")
            
        except Exception as e:
            print(f"⚠️ Audio error: {e}")

    def _get_audio_buffer(self) -> Optional[np.ndarray]:
        """Get audio buffer from listener"""
        if hasattr(self.listener, '_ring') and self.listener._ring is not None:
            ring_pos = getattr(self.listener, '_ring_pos', 0)
            frame_size = getattr(self.listener, 'frame', 2048)
            idx = (np.arange(frame_size) + ring_pos) % frame_size
            return self.listener._ring[idx].copy()
        return None

    def _extract_features(self, event: Event, harmonic_context) -> List[float]:
        """Extract features from audio event and harmonic context"""
        # Chroma vector (12 values)
        chroma = harmonic_context.chroma_vector.tolist() if hasattr(harmonic_context, 'chroma_vector') else [0.0] * 12
        
        # Additional features
        additional = [
            harmonic_context.confidence,
            harmonic_context.stability,
            event.rms,
            event.f0,
            len(harmonic_context.chord_history) if hasattr(harmonic_context, 'chord_history') else 0
        ]
        
        return chroma + additional

    def set_label(self, chord: str):
        """Set current chord label"""
        self.current_label = chord
        
        # Add to mapping if new
        if chord not in self.chord_mapping:
            self.chord_mapping[chord] = self.next_chord_id
            self.next_chord_id += 1
            
        print(f"🏷️ Label: {chord}")

    def train_model(self):
        """Train the model"""
        if len(self.samples) < 5:
            print("❌ Need at least 5 samples to train")
            return False
            
        try:
            print("🤖 Training model...")
            
            # Prepare data
            X = []
            y = []
            
            for features, chord in self.samples:
                X.append(features)
                y.append(self.chord_mapping[chord])
            
            X = np.array(X)
            y = np.array(y)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X_scaled, y)
            
            print(f"✅ Model trained on {len(self.samples)} samples")
            print(f"📊 Classes: {list(self.chord_mapping.keys())}")
            
            # Save model
            self._save_model()
            
            return True
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return False

    def test_model(self):
        """Test the trained model"""
        if self.model is None:
            print("❌ No trained model")
            return
            
        print("🧪 Testing model... Press 'q' to stop")
        
        def test_callback(*args):
            try:
                event = args[0] if len(args) > 0 else None
                if not event or not isinstance(event, Event):
                    return
                    
                audio_buffer = self._get_audio_buffer()
                if audio_buffer is None or len(audio_buffer) == 0:
                    return
                    
                harmonic_context = self.harmonic_detector.update_from_audio(audio_buffer, sr=44100)
                features = self._extract_features(event, harmonic_context)
                
                # Predict
                features_scaled = self.scaler.transform([features])
                prediction_id = self.model.predict(features_scaled)[0]
                
                # Get chord name
                chord_name = list(self.chord_mapping.keys())[list(self.chord_mapping.values()).index(prediction_id)]
                
                print(f"🎵 Predicted: {chord_name}")
                
            except Exception as e:
                print(f"⚠️ Test error: {e}")
        
        # Replace callback
        original_callback = self.listener._callback
        self.listener._callback = test_callback
        
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            self.listener._callback = original_callback

    def _save_model(self):
        """Save trained model"""
        try:
            os.makedirs('models', exist_ok=True)
            
            model_data = {
                'chord_mapping': self.chord_mapping,
                'samples_count': len(self.samples),
                'timestamp': datetime.now().isoformat()
            }
            
            # Save model files
            joblib.dump(self.model, 'models/chord_model.pkl')
            joblib.dump(self.scaler, 'models/chord_scaler.pkl')
            
            with open('models/chord_metadata.json', 'w') as f:
                json.dump(model_data, f, indent=2)
                
            print("💾 Model saved to models/")
            
        except Exception as e:
            print(f"❌ Save failed: {e}")

    def run(self):
        """Run interactive training"""
        if not self.start():
            return
            
        print("\n🎹 Start playing chords and press keys to label them!")
        
        try:
            while True:
                # Simple input handling
                try:
                    key = input().strip().lower()
                    
                    if key == 'q':
                        break
                    elif key == 't':
                        self.train_model()
                    elif key in self.chord_keys:
                        chord = self.chord_keys[key]
                        self.set_label(chord)
                    else:
                        print(f"❓ Unknown key: {key}")
                        
                except EOFError:
                    break
                    
        except KeyboardInterrupt:
            pass
        finally:
            if self.listener:
                self.listener.stop()
            print("🛑 Training ended")

def main():
    """Main function"""
    trainer = SimpleChordTrainer()
    trainer.run()

if __name__ == "__main__":
    main()


