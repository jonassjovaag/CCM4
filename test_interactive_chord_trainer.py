#!/usr/bin/env python3
"""
Interactive Chord Training for Bass Response Test
================================================

This extends the existing bass response test with ML training capabilities.
You play chords, label them, and the system learns to detect them better.

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
from typing import Dict, List, Optional, Tuple
import threading
import queue

# Import our existing components
from listener.jhs_listener_core import DriftListener
from listener.harmonic_context import RealtimeHarmonicDetector
from listener.jhs_listener_core import Event

# Simple ML
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

class InteractiveChordTrainer:
    """Interactive chord detection trainer"""
    
    def __init__(self):
        self.listener = None
        self.harmonic_detector = RealtimeHarmonicDetector()
        
        # Training data
        self.samples = []  # List of (features, chord_label)
        self.chord_mapping = {}  # chord -> index
        self.next_chord_id = 0
        
        # State
        self.current_label = None
        self.last_collection_time = 0
        self.collection_interval = 0.5  # seconds
        
        # ML
        self.model = None
        self.scaler = StandardScaler()
        
        # Model persistence paths
        self.model_path = 'models/chord_model.pkl'
        self.scaler_path = 'models/chord_scaler.pkl'
        self.metadata_path = 'models/chord_metadata.json'
        
        # Key mappings (case insensitive) - Full Jazz Chord Vocabulary
        self.chord_keys = {
            # Basic Major/Minor
            'c': 'C', 'd': 'D', 'e': 'E', 'f': 'F', 'g': 'G', 'a': 'A', 'b': 'B',
            'cm': 'Cm', 'dm': 'Dm', 'em': 'Em', 'fm': 'Fm', 'gm': 'Gm', 'am': 'Am', 'bm': 'Bm',
            
            # Chromatic Major/Minor
            'c#': 'C#', 'd#': 'D#', 'f#': 'F#', 'g#': 'G#', 'a#': 'A#',
            'c#m': 'C#m', 'd#m': 'D#m', 'f#m': 'F#m', 'g#m': 'G#m', 'a#m': 'A#m',
            'db': 'Db', 'eb': 'Eb', 'gb': 'Gb', 'ab': 'Ab', 'bb': 'Bb',
            'dbm': 'Dbm', 'ebm': 'Ebm', 'gbm': 'Gbm', 'abm': 'Abm', 'bbm': 'Bbm',
            
            # Dominant 7th
            'c7': 'C7', 'd7': 'D7', 'e7': 'E7', 'f7': 'F7', 'g7': 'G7', 'a7': 'A7', 'b7': 'B7',
            'c#7': 'C#7', 'd#7': 'D#7', 'f#7': 'F#7', 'g#7': 'G#7', 'a#7': 'A#7',
            'db7': 'Db7', 'eb7': 'Eb7', 'gb7': 'Gb7', 'ab7': 'Ab7', 'bb7': 'Bb7',
            
            # Minor 7th
            'cm7': 'Cm7', 'dm7': 'Dm7', 'em7': 'Em7', 'fm7': 'Fm7', 'gm7': 'Gm7', 'am7': 'Am7', 'bm7': 'Bm7',
            'c#m7': 'C#m7', 'd#m7': 'D#m7', 'f#m7': 'F#m7', 'g#m7': 'G#m7', 'a#m7': 'A#m7',
            'dbm7': 'Dbm7', 'ebm7': 'Ebm7', 'gbm7': 'Gbm7', 'abm7': 'Abm7', 'bbm7': 'Bbm7',
            
            # Major 7th
            'cmaj7': 'Cmaj7', 'dmaj7': 'Dmaj7', 'emaj7': 'Emaj7', 'fmaj7': 'Fmaj7', 'gmaj7': 'Gmaj7', 'amaj7': 'Amaj7', 'bmaj7': 'Bmaj7',
            'c#maj7': 'C#maj7', 'd#maj7': 'D#maj7', 'f#maj7': 'F#maj7', 'g#maj7': 'G#maj7', 'a#maj7': 'A#maj7',
            'dbmaj7': 'Dbmaj7', 'ebmaj7': 'Ebmaj7', 'gbmaj7': 'Gbmaj7', 'abmaj7': 'Abmaj7', 'bbmaj7': 'Bbmaj7',
            
            # Diminished
            'cdim': 'Cdim', 'ddim': 'Ddim', 'edim': 'Edim', 'fdim': 'Fdim', 'gdim': 'Gdim', 'adim': 'Adim', 'bdim': 'Bdim',
            'c#dim': 'C#dim', 'd#dim': 'D#dim', 'f#dim': 'F#dim', 'g#dim': 'G#dim', 'a#dim': 'A#dim',
            'dbdim': 'Dbdim', 'ebdim': 'Ebdim', 'gbdim': 'Gbdim', 'abdim': 'Abdim', 'bbdim': 'Bbdim',
            
            # Diminished 7th
            'cdim7': 'Cdim7', 'ddim7': 'Ddim7', 'edim7': 'Edim7', 'fdim7': 'Fdim7', 'gdim7': 'Gdim7', 'adim7': 'Adim7', 'bdim7': 'Bdim7',
            'c#dim7': 'C#dim7', 'd#dim7': 'D#dim7', 'f#dim7': 'F#dim7', 'g#dim7': 'G#dim7', 'a#dim7': 'A#dim7',
            'dbdim7': 'Dbdim7', 'ebdim7': 'Ebdim7', 'gbdim7': 'Gbdim7', 'abdim7': 'Abdim7', 'bbdim7': 'Bbdim7',
            
            # Half-Diminished (m7b5)
            'cm7b5': 'Cm7b5', 'dm7b5': 'Dm7b5', 'em7b5': 'Em7b5', 'fm7b5': 'Fm7b5', 'gm7b5': 'Gm7b5', 'am7b5': 'Am7b5', 'bm7b5': 'Bm7b5',
            'c#m7b5': 'C#m7b5', 'd#m7b5': 'D#m7b5', 'f#m7b5': 'F#m7b5', 'g#m7b5': 'G#m7b5', 'a#m7b5': 'A#m7b5',
            'dbm7b5': 'Dbm7b5', 'ebm7b5': 'Ebm7b5', 'gbm7b5': 'Gbm7b5', 'abm7b5': 'Abm7b5', 'bbm7b5': 'Bbm7b5',
            
            # Augmented
            'caug': 'Caug', 'daug': 'Daug', 'eaug': 'Eaug', 'faug': 'Faug', 'gaug': 'Gaug', 'aaug': 'Aaug', 'baug': 'Baug',
            'c#aug': 'C#aug', 'd#aug': 'D#aug', 'f#aug': 'F#aug', 'g#aug': 'G#aug', 'a#aug': 'A#aug',
            'dbaug': 'Dbaug', 'ebaug': 'Ebaug', 'gbaug': 'Gbaug', 'abaug': 'Abaug', 'bbaug': 'Bbaug',
            
            # Suspended
            'csus2': 'Csus2', 'dsus2': 'Dsus2', 'esus2': 'Esus2', 'fsus2': 'Fsus2', 'gsus2': 'Gsus2', 'asus2': 'Asus2', 'bsus2': 'Bsus2',
            'csus4': 'Csus4', 'dsus4': 'Dsus4', 'esus4': 'Esus4', 'fsus4': 'Fsus4', 'gsus4': 'Gsus4', 'asus4': 'Asus4', 'bsus4': 'Bsus4',
            'c#sus2': 'C#sus2', 'd#sus2': 'D#sus2', 'f#sus2': 'F#sus2', 'g#sus2': 'G#sus2', 'a#sus2': 'A#sus2',
            'c#sus4': 'C#sus4', 'd#sus4': 'D#sus4', 'f#sus4': 'F#sus4', 'g#sus4': 'G#sus4', 'a#sus4': 'A#sus4',
            'dbsus2': 'Dbsus2', 'ebsus2': 'Ebsus2', 'gbsus2': 'Gbsus2', 'absus2': 'Absus2', 'bbsus2': 'Bbsus2',
            'dbsus4': 'Dbsus4', 'ebsus4': 'Ebsus4', 'gbsus4': 'Gbsus4', 'absus4': 'Absus4', 'bbsus4': 'Bbsus4',
            
            # 9th Chords
            'c9': 'C9', 'd9': 'D9', 'e9': 'E9', 'f9': 'F9', 'g9': 'G9', 'a9': 'A9', 'b9': 'B9',
            'cm9': 'Cm9', 'dm9': 'Dm9', 'em9': 'Em9', 'fm9': 'Fm9', 'gm9': 'Gm9', 'am9': 'Am9', 'bm9': 'Bm9',
            'cmaj9': 'Cmaj9', 'dmaj9': 'Dmaj9', 'emaj9': 'Emaj9', 'fmaj9': 'Fmaj9', 'gmaj9': 'Gmaj9', 'amaj9': 'Amaj9', 'bmaj9': 'Bmaj9',
            'c#9': 'C#9', 'd#9': 'D#9', 'f#9': 'F#9', 'g#9': 'G#9', 'a#9': 'A#9',
            'c#m9': 'C#m9', 'd#m9': 'D#m9', 'f#m9': 'F#m9', 'g#m9': 'G#m9', 'a#m9': 'A#m9',
            'c#maj9': 'C#maj9', 'd#maj9': 'D#maj9', 'f#maj9': 'F#maj9', 'g#maj9': 'G#maj9', 'a#maj9': 'A#maj9',
            'db9': 'Db9', 'eb9': 'Eb9', 'gb9': 'Gb9', 'ab9': 'Ab9', 'bb9': 'Bb9',
            'dbm9': 'Dbm9', 'ebm9': 'Ebm9', 'gbm9': 'Gbm9', 'abm9': 'Abm9', 'bbm9': 'Bbm9',
            'dbmaj9': 'Dbmaj9', 'ebmaj9': 'Ebmaj9', 'gbmaj9': 'Gbmaj9', 'abmaj9': 'Abmaj9', 'bbmaj9': 'Bbmaj9',
            
            # 11th Chords
            'c11': 'C11', 'd11': 'D11', 'e11': 'E11', 'f11': 'F11', 'g11': 'G11', 'a11': 'A11', 'b11': 'B11',
            'cm11': 'Cm11', 'dm11': 'Dm11', 'em11': 'Em11', 'fm11': 'Fm11', 'gm11': 'Gm11', 'am11': 'Am11', 'bm11': 'Bm11',
            'cmaj11': 'Cmaj11', 'dmaj11': 'Dmaj11', 'emaj11': 'Emaj11', 'fmaj11': 'Fmaj11', 'gmaj11': 'Gmaj11', 'amaj11': 'Amaj11', 'bmaj11': 'Bmaj11',
            
            # 13th Chords
            'c13': 'C13', 'd13': 'D13', 'e13': 'E13', 'f13': 'F13', 'g13': 'G13', 'a13': 'A13', 'b13': 'B13',
            'cm13': 'Cm13', 'dm13': 'Dm13', 'em13': 'Em13', 'fm13': 'Fm13', 'gm13': 'Gm13', 'am13': 'Am13', 'bm13': 'Bm13',
            'cmaj13': 'Cmaj13', 'dmaj13': 'Dmaj13', 'emaj13': 'Emaj13', 'fmaj13': 'Fmaj13', 'gmaj13': 'Gmaj13', 'amaj13': 'Amaj13', 'bmaj13': 'Bmaj13',
            
            # Altered Dominants
            'c7alt': 'C7alt', 'd7alt': 'D7alt', 'e7alt': 'E7alt', 'f7alt': 'F7alt', 'g7alt': 'G7alt', 'a7alt': 'A7alt', 'b7alt': 'B7alt',
            'c7b9': 'C7b9', 'd7b9': 'D7b9', 'e7b9': 'E7b9', 'f7b9': 'F7b9', 'g7b9': 'G7b9', 'a7b9': 'A7b9', 'b7b9': 'B7b9',
            'c7#9': 'C7#9', 'd7#9': 'D7#9', 'e7#9': 'E7#9', 'f7#9': 'F7#9', 'g7#9': 'G7#9', 'a7#9': 'A7#9', 'b7#9': 'B7#9',
            'c7b5': 'C7b5', 'd7b5': 'D7b5', 'e7b5': 'E7b5', 'f7b5': 'F7b5', 'g7b5': 'G7b5', 'a7b5': 'A7b5', 'b7b5': 'B7b5',
            'c7#5': 'C7#5', 'd7#5': 'D7#5', 'e7#5': 'E7#5', 'f7#5': 'F7#5', 'g7#5': 'G7#5', 'a7#5': 'A7#5', 'b7#5': 'B7#5',
            
            # Special
            'space': 'silence'
        }
        
        print("🎓 Interactive Chord Training System")
        print("=" * 50)
        self._print_instructions()

    def _print_instructions(self):
        """Print usage instructions"""
        print("Instructions:")
        print("1. Play a chord on your piano")
        print("2. Press the corresponding key to label it")
        print("3. Repeat for different chords")
        print("4. Press 't' to train the model")
        print("5. Press 'q' to quit")
        print()
        print("🎵 FULL JAZZ CHORD VOCABULARY:")
        print("=" * 50)
        print("📋 BASIC CHORDS:")
        print("  Major: c, d, e, f, g, a, b")
        print("  Minor: cm, dm, em, fm, gm, am, bm")
        print("  Chromatic: c#, d#, f#, g#, a# (or db, eb, gb, ab, bb)")
        print()
        print("🎼 7TH CHORDS:")
        print("  Dominant: c7, d7, e7, f7, g7, a7, b7")
        print("  Minor 7th: cm7, dm7, em7, fm7, gm7, am7, bm7")
        print("  Major 7th: cmaj7, dmaj7, emaj7, fmaj7, gmaj7, amaj7, bmaj7")
        print()
        print("🎯 JAZZ EXTENSIONS:")
        print("  9th: c9, cm9, cmaj9, c#9, db9, etc.")
        print("  11th: c11, cm11, cmaj11, etc.")
        print("  13th: c13, cm13, cmaj13, etc.")
        print()
        print("🎪 ALTERED & SPECIAL:")
        print("  Diminished: cdim, cdim7, cm7b5")
        print("  Augmented: caug")
        print("  Suspended: csus2, csus4")
        print("  Altered: c7alt, c7b9, c7#9, c7b5, c7#5")
        print()
        print("💡 EXAMPLES:")
        print("  dm7, bdim, bbmaj9, f#m7b5, g7alt, ebsus4")
        print("  SPACE: Silence")
        print("=" * 50)

    def start(self):
        """Start the training system"""
        try:
            # Create reference frequency functions
            def ref_fn(midi_note: int) -> float:
                return 440.0 * (2 ** ((midi_note - 69) / 12))
            
            def a4_fn() -> float:
                return 440.0
            
            self.listener = DriftListener(ref_fn, a4_fn)
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
            if current_time - self.last_collection_time < self.collection_interval:
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
            event.rms_db,
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

    def load_existing_model(self):
        """Load existing model and training data for continuous learning"""
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
                # Load model and scaler
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                
                # Load metadata
                if os.path.exists(self.metadata_path):
                    with open(self.metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    # Restore chord mapping
                    self.chord_mapping = metadata.get('chord_mapping', {})
                    self.next_chord_id = max(self.chord_mapping.values()) + 1 if self.chord_mapping else 0
                    
                    print(f"✅ Loaded existing model with {metadata.get('samples_count', 0)} samples")
                    print(f"📊 Existing chord vocabulary: {list(self.chord_mapping.keys())}")
                    return True
                else:
                    print("⚠️ Model files found but no metadata - starting fresh")
                    return False
            else:
                print("📝 No existing model found - starting fresh")
                return False
                
        except Exception as e:
            print(f"❌ Error loading existing model: {e}")
            return False

    def train_model(self):
        """Train the model with continuous learning"""
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
            
            # Train model (continuous learning - retrain on all data)
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
        # Load existing model first
        self.load_existing_model()
        
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
                        print(f"❓ Unknown key: '{key}'")
                        print("💡 Try: dm7, bdim, bbmaj9, f#m7b5, g7alt, ebsus4, etc.")
                        print("   Or basic: c, cm, c7, cm7, cmaj7")
                        
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
    trainer = InteractiveChordTrainer()
    trainer.run()

if __name__ == "__main__":
    main()
