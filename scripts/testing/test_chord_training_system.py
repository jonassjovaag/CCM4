#!/usr/bin/env python3
"""
Chord Detection Training System
===============================

This system learns to detect chords by collecting training data:
- Audio input from microphone (features)
- Ground truth chord labels from user input
- Trains a model to map audio features → chord predictions

Usage:
1. Run the training session
2. Play chords on piano
3. Press keys to label the current chord
4. System learns the mapping
5. Test improved accuracy
"""

import numpy as np
import time
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import threading
import queue

# Import our existing components
from listener.jhs_listener_core import DriftListener
from listener.harmonic_context import RealtimeHarmonicDetector, HarmonicContext
from listener.rhythmic_context import RealtimeRhythmicDetector, RhythmicContext
from core.event import Event

# ML imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

@dataclass
class TrainingSample:
    """Single training sample: audio features + ground truth chord"""
    timestamp: float
    audio_features: List[float]  # Chroma vector (12 values)
    harmonic_features: List[float]  # Additional harmonic features
    ground_truth_chord: str
    confidence: float
    stability: float
    key: str
    rms: float
    f0: float

@dataclass
class TrainingSession:
    """Complete training session with multiple samples"""
    session_id: str
    start_time: float
    samples: List[TrainingSample]
    chord_mapping: Dict[str, int]  # chord name -> class index
    model: Optional[object] = None
    scaler: Optional[object] = None
    accuracy: Optional[float] = None

class ChordTrainingSystem:
    """Interactive chord detection training system"""
    
    def __init__(self):
        self.listener = None
        self.harmonic_detector = RealtimeHarmonicDetector()
        self.rhythmic_detector = RealtimeRhythmicDetector()
        
        # Training data
        self.current_session: Optional[TrainingSession] = None
        self.samples: List[TrainingSample] = []
        self.chord_mapping: Dict[str, int] = {}
        self.next_chord_id = 0
        
        # State
        self.is_training = False
        self.current_chord_label = None
        self.last_chord_time = 0
        self.chord_change_threshold = 2.0
        
        # ML components
        self.model = None
        self.scaler = StandardScaler()
        
        # Key mappings for chord input
        self.chord_keys = {
            '1': 'C', '2': 'C#', '3': 'D', '4': 'D#', '5': 'E', '6': 'F',
            '7': 'F#', '8': 'G', '9': 'G#', '0': 'A', '-': 'A#', '=': 'B',
            'q': 'Cm', 'w': 'C#m', 'e': 'Dm', 'r': 'D#m', 't': 'Em', 'y': 'Fm',
            'u': 'F#m', 'i': 'Gm', 'o': 'G#m', 'p': 'Am', '[': 'A#m', ']': 'Bm',
            'a': 'C7', 's': 'C#7', 'd': 'D7', 'f': 'D#7', 'g': 'E7', 'h': 'F7',
            'j': 'F#7', 'k': 'G7', 'l': 'G#7', ';': 'A7', "'": 'A#7', '\\': 'B7',
            'z': 'Cmaj7', 'x': 'C#maj7', 'c': 'Dmaj7', 'v': 'D#maj7', 'b': 'Emaj7',
            'n': 'Fmaj7', 'm': 'F#maj7', ',': 'Gmaj7', '.': 'G#maj7', '/': 'Amaj7',
            'space': 'silence', 'enter': 'unknown'
        }
        
        print("🎓 Chord Detection Training System")
        print("=" * 50)
        print("Key mappings:")
        print("  Numbers (1-0, -=): Major chords")
        print("  Letters (q-p, []): Minor chords") 
        print("  a-\\: Dominant 7th chords")
        print("  z-/: Major 7th chords")
        print("  SPACE: Silence, ENTER: Unknown")
        print("=" * 50)

    def start_training_session(self):
        """Start a new training session"""
        session_id = f"chord_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.current_session = TrainingSession(
            session_id=session_id,
            start_time=time.time(),
            samples=[],
            chord_mapping={}
        )
        
        self.samples = []
        self.chord_mapping = {}
        self.next_chord_id = 0
        self.is_training = True
        
        print(f"🎯 Started training session: {session_id}")
        print("🎹 Play chords and press corresponding keys to label them")
        print("📊 Press 't' to train model, 'q' to quit")

    def _initialize_components(self):
        """Initialize audio components"""
        try:
            self.listener = DriftListener()
            self.listener.start(self._on_audio_event)
            print("✅ Audio components initialized")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize audio: {e}")
            return False

    def _on_audio_event(self, *args):
        """Handle audio events during training"""
        if not self.is_training:
            return
            
        try:
            # Extract event from args (DriftListener passes multiple arguments)
            event = args[0] if len(args) > 0 else None
            if not event or not isinstance(event, Event):
                return
                
            current_time = time.time()
            
            # Get audio buffer for harmonic analysis
            audio_buffer = self._get_real_audio_buffer()
            if audio_buffer is None or len(audio_buffer) == 0:
                return
                
            # Analyze harmonic content
            harmonic_context = self.harmonic_detector.update_from_audio(audio_buffer, sr=44100)
            
            # Only collect samples when we have a chord label
            if self.current_chord_label and current_time - self.last_chord_time > 0.5:
                self._collect_training_sample(event, harmonic_context, current_time)
                self.last_chord_time = current_time
                
        except Exception as e:
            print(f"⚠️ Audio event error: {e}")

    def _get_real_audio_buffer(self) -> Optional[np.ndarray]:
        """Get actual audio buffer from DriftListener"""
        if hasattr(self.listener, '_ring') and self.listener._ring is not None:
            ring_pos = getattr(self.listener, '_ring_pos', 0)
            frame_size = getattr(self.listener, 'frame', 2048)
            
            idx = (np.arange(frame_size) + ring_pos) % frame_size
            return self.listener._ring[idx].copy()
        return None

    def _collect_training_sample(self, event: Event, harmonic_context: HarmonicContext, timestamp: float):
        """Collect a training sample with current chord label"""
        try:
            # Extract features
            chroma_vector = harmonic_context.chroma_vector.tolist() if hasattr(harmonic_context, 'chroma_vector') else [0.0] * 12
            
            # Additional harmonic features
            harmonic_features = [
                harmonic_context.confidence,
                harmonic_context.stability,
                event.rms,
                event.f0,
                len(harmonic_context.chord_history) if hasattr(harmonic_context, 'chord_history') else 0
            ]
            
            # Create training sample
            sample = TrainingSample(
                timestamp=timestamp,
                audio_features=chroma_vector,
                harmonic_features=harmonic_features,
                ground_truth_chord=self.current_chord_label,
                confidence=harmonic_context.confidence,
                stability=harmonic_context.stability,
                key=harmonic_context.current_key,
                rms=event.rms,
                f0=event.f0
            )
            
            self.samples.append(sample)
            
            # Update chord mapping
            if self.current_chord_label not in self.chord_mapping:
                self.chord_mapping[self.current_chord_label] = self.next_chord_id
                self.next_chord_id += 1
            
            print(f"📝 Collected sample: {self.current_chord_label} (total: {len(self.samples)})")
            
        except Exception as e:
            print(f"⚠️ Sample collection error: {e}")

    def set_chord_label(self, chord: str):
        """Set current chord label for training"""
        self.current_chord_label = chord
        self.last_chord_time = time.time()
        print(f"🏷️ Label set: {chord}")

    def train_model(self):
        """Train ML model on collected samples"""
        if len(self.samples) < 10:
            print("❌ Need at least 10 samples to train")
            return False
            
        try:
            print("🤖 Training model...")
            
            # Prepare features and labels
            X = []
            y = []
            
            for sample in self.samples:
                # Combine audio and harmonic features
                features = sample.audio_features + sample.harmonic_features
                X.append(features)
                y.append(self.chord_mapping[sample.ground_truth_chord])
            
            X = np.array(X)
            y = np.array(y)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model (try multiple algorithms)
            models = {
                'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
                'NeuralNetwork': MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=1000, random_state=42)
            }
            
            best_model = None
            best_accuracy = 0
            
            for name, model in models.items():
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)
                
                print(f"  {name}: {accuracy:.3f} accuracy")
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = model
            
            self.model = best_model
            self.current_session.model = best_model
            self.current_session.scaler = self.scaler
            self.current_session.accuracy = best_accuracy
            
            print(f"✅ Model trained! Best accuracy: {best_accuracy:.3f}")
            
            # Save model
            self._save_training_session()
            
            return True
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return False

    def test_model(self):
        """Test the trained model in real-time"""
        if self.model is None:
            print("❌ No trained model available")
            return
            
        print("🧪 Testing trained model...")
        print("🎹 Play chords to see predictions (press 'q' to stop)")
        
        def test_callback(*args):
            try:
                event = args[0] if len(args) > 0 else None
                if not event or not isinstance(event, Event):
                    return
                    
                audio_buffer = self._get_real_audio_buffer()
                if audio_buffer is None or len(audio_buffer) == 0:
                    return
                    
                harmonic_context = self.harmonic_detector.update_from_audio(audio_buffer, sr=44100)
                
                # Extract features
                chroma_vector = harmonic_context.chroma_vector.tolist() if hasattr(harmonic_context, 'chroma_vector') else [0.0] * 12
                harmonic_features = [
                    harmonic_context.confidence,
                    harmonic_context.stability,
                    event.rms,
                    event.f0,
                    len(harmonic_context.chord_history) if hasattr(harmonic_context, 'chord_history') else 0
                ]
                
                features = np.array([chroma_vector + harmonic_features])
                features_scaled = self.scaler.transform(features)
                
                # Predict
                prediction_id = self.model.predict(features_scaled)[0]
                prediction_chord = list(self.chord_mapping.keys())[list(self.chord_mapping.values()).index(prediction_id)]
                
                print(f"🎵 Predicted: {prediction_chord} | Confidence: {harmonic_context.confidence:.2f}")
                
            except Exception as e:
                print(f"⚠️ Test error: {e}")
        
        # Replace callback temporarily
        original_callback = self.listener._callback
        self.listener._callback = test_callback
        
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            self.listener._callback = original_callback

    def _save_training_session(self):
        """Save training session to file"""
        try:
            os.makedirs('logs', exist_ok=True)
            
            session_data = {
                'session_id': self.current_session.session_id,
                'start_time': self.current_session.start_time,
                'samples': [asdict(sample) for sample in self.samples],
                'chord_mapping': self.chord_mapping,
                'accuracy': self.current_session.accuracy,
                'num_samples': len(self.samples)
            }
            
            filename = f"logs/{self.current_session.session_id}.json"
            with open(filename, 'w') as f:
                json.dump(session_data, f, indent=2)
                
            print(f"💾 Training session saved: {filename}")
            
        except Exception as e:
            print(f"❌ Save failed: {e}")

    def run_interactive_training(self):
        """Run interactive training session"""
        if not self._initialize_components():
            return
            
        self.start_training_session()
        
        print("\n🎹 Start playing chords and press corresponding keys to label them!")
        print("Press 't' to train, 'q' to quit")
        
        try:
            while True:
                # Check for keyboard input
                import sys
                import select
                
                if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                    key = sys.stdin.readline().strip().lower()
                    
                    if key == 'q':
                        break
                    elif key == 't':
                        self.train_model()
                    elif key in self.chord_keys:
                        chord = self.chord_keys[key]
                        self.set_chord_label(chord)
                    else:
                        print(f"❓ Unknown key: {key}")
                        
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            pass
        finally:
            self.is_training = False
            if self.listener:
                self.listener.stop()
            print("🛑 Training session ended")

def main():
    """Main function"""
    trainer = ChordTrainingSystem()
    trainer.run_interactive_training()

if __name__ == "__main__":
    main()


