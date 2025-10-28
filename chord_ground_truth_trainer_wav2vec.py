#!/usr/bin/env python3
"""
Chord Ground Truth Trainer - Wav2Vec Features Version
======================================================

This version extracts 768D Wav2Vec features for training the ML chord detection model.

KEY DIFFERENCE from hybrid version:
- Uses Wav2Vec (768D neural features) instead of 22D hybrid features
- Same feature space as Chandra trainer with --wav2vec flag
- Enables transfer learning: Ground truth → Chandra → MusicHal

Features extracted:
- 768D Wav2Vec features (self-supervised learned representations)
- Optional: Ratio analysis for validation (not used in training)

Can train on:
1. Pre-validated dataset (validation_results_*.json)
2. Live MIDI → audio validation (future)

Usage:
    # Train from existing validation results
    python chord_ground_truth_trainer_wav2vec.py \\
        --from-validation validation_results_20251007_170413.json \\
        --gpu
    
    # CPU only
    python chord_ground_truth_trainer_wav2vec.py \\
        --from-validation validation_results_20251007_170413.json
"""

import numpy as np
import time
import json
import os
import mido
from datetime import datetime
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, asdict
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report

# Import Wav2Vec perception
from listener.dual_perception import DualPerceptionModule


@dataclass
class ChordGroundTruth:
    """Validated ground truth for a chord"""
    sent_midi_notes: List[int]
    detected_midi_notes: List[int]
    detected_frequencies: List[float]
    chord_name: str  # e.g., "C", "Cm", "C7"
    inversion: int
    octave_offset: int
    match_quality: float  # 0.0-1.0, how well detected matches sent
    audio_features: dict  # Store Wav2Vec features
    timestamp: str


class Wav2VecFeatureExtractor:
    """Extract Wav2Vec features from audio"""
    
    def __init__(self, use_gpu: bool = False):
        """
        Initialize Wav2Vec feature extractor
        
        Args:
            use_gpu: Use GPU acceleration if available
        """
        print("🔬 Initializing Wav2Vec Feature Extractor...")
        self.dual_perception = DualPerceptionModule(
            vocabulary_size=64,
            wav2vec_model='facebook/wav2vec2-base',
            use_gpu=use_gpu,
            enable_symbolic=True  # Enable gesture token quantization
        )
        print("✅ Wav2Vec extractor initialized")
    
    def extract_features_from_audio(self, 
                                   audio: np.ndarray,
                                   sr: int = 44100,
                                   detected_f0: Optional[float] = None) -> Dict:
        """
        Extract Wav2Vec features from audio
        
        Args:
            audio: Audio signal
            sr: Sample rate
            detected_f0: Optional detected fundamental frequency
        
        Returns:
            Dict with:
                - wav2vec_features: 768D feature vector
                - gesture_token: Quantized token (0-63)
                - consonance: Consonance score (from ratio analysis)
                - chord_detected: Chord type from ratio analyzer (for validation)
                - confidence: Confidence score
        """
        # Ensure audio is long enough (Wav2Vec needs minimum length)
        min_samples = int(sr * 0.02)  # 20ms minimum
        if len(audio) < min_samples:
            # Pad with zeros
            audio = np.pad(audio, (0, min_samples - len(audio)))
        
        # Extract features using dual perception
        result = self.dual_perception.extract_features(
            audio, 
            sr=sr,
            timestamp=0.0,
            detected_f0=detected_f0
        )
        
        # Prepare feature dict
        features = {
            'wav2vec_features': result.wav2vec_features.tolist() if result.wav2vec_features is not None else None,
            'gesture_token': result.gesture_token,
            'consonance': result.consonance if result.ratio_analysis else 0.0,
            'chord_detected': None,
            'confidence': 0.0
        }
        
        # Add ratio analysis if available (for validation, not training)
        if result.ratio_analysis:
            features['chord_detected'] = result.chord_label
            features['confidence'] = result.chord_confidence
        
        return features


class ChordGroundTruthTrainer:
    """
    Trainer with Wav2Vec feature extraction
    Can work in two modes:
    1. Live validation (MIDI → audio → detect)
    2. Load from pre-validated dataset
    """
    
    def __init__(self,
                 use_gpu: bool = False,
                 midi_output_port: str = "IAC Driver Chord Trainer Output",
                 audio_input_device: Optional[int] = None,
                 chord_duration: float = 2.5,
                 analysis_window: float = 2.0):
        """
        Initialize trainer
        
        Args:
            use_gpu: Use GPU acceleration for Wav2Vec
            midi_output_port: MIDI output port name (for live mode)
            audio_input_device: Audio input device index (for live mode)
            chord_duration: Duration of each chord (for live mode)
            analysis_window: Analysis window duration (for live mode)
        """
        self.use_gpu = use_gpu
        self.midi_port_name = midi_output_port
        self.audio_device = audio_input_device
        self.chord_duration = chord_duration
        self.analysis_window = analysis_window
        
        # MIDI
        self.midi_port = None
        
        # Audio recording
        self.sample_rate = 44100
        self.recorded_audio = []
        self.recording = False
        
        # Feature extractor
        self.feature_extractor = Wav2VecFeatureExtractor(use_gpu=use_gpu)
        
        # Validated ground truths
        self.validated_chords = []
        
        # Statistics
        self.stats = {
            'total_attempts': 0,
            'successful_validations': 0,
            'failed_validations': 0,
            'retries': 0
        }
    
    def _extract_chord_type(self, chord_name: str) -> str:
        """
        Extract chord type from chord name
        
        Examples:
            "C" → "C"
            "C7" → "C7"
            "Cmaj7" → "Cmaj7"
        """
        return chord_name
    
    def _open_midi_port(self) -> bool:
        """Open MIDI output"""
        try:
            available = mido.get_output_names()
            if self.midi_port_name in available:
                self.midi_port = mido.open_output(self.midi_port_name)
                print("✅ MIDI output: {}".format(self.midi_port_name))
                return True
            else:
                print("❌ Port not found: {}".format(self.midi_port_name))
                print("   Available ports: {}".format(available))
                return False
        except Exception as e:
            print("❌ MIDI error: {}".format(e))
            return False
    
    def _record_audio(self, duration: float) -> np.ndarray:
        """Record audio for specified duration"""
        import sounddevice as sd
        
        samples = int(duration * self.sample_rate)
        recording = sd.rec(samples, samplerate=self.sample_rate, 
                          channels=1, device=self.audio_device, dtype='float32')
        sd.wait()
        
        return recording.flatten()
    
    def _play_chord_midi(self, midi_notes: List[int]):
        """Play MIDI notes"""
        if not self.midi_port:
            return
        
        for note in midi_notes:
            msg = mido.Message('note_on', channel=0, note=note, velocity=80)
            self.midi_port.send(msg)
    
    def _stop_chord_midi(self, midi_notes: List[int]):
        """Stop MIDI notes"""
        if not self.midi_port:
            return
        
        for note in midi_notes:
            msg = mido.Message('note_off', channel=0, note=note, velocity=0)
            self.midi_port.send(msg)
    
    def validate_chord(self, 
                      midi_notes: List[int],
                      chord_name: str,
                      inversion: int,
                      octave_offset: int,
                      max_retries: int = 3) -> Optional[ChordGroundTruth]:
        """
        Validate a single chord with retries
        
        Play MIDI chord → Record audio → Extract Wav2Vec features → Validate
        
        Returns validated ChordGroundTruth or None if validation fails
        """
        for attempt in range(max_retries):
            self.stats['total_attempts'] += 1
            
            if attempt > 0:
                print("   🔄 Retry {}/{}...".format(attempt, max_retries-1))
                self.stats['retries'] += 1
            
            # Play chord and record simultaneously
            print("   🎹 Sending MIDI: {} (notes: {})".format(midi_notes, chord_name))
            
            self._play_chord_midi(midi_notes)
            
            # Record audio while chord plays
            time.sleep(0.1)  # Let sound stabilize
            audio = self._record_audio(self.analysis_window)
            
            # Stop chord
            self._stop_chord_midi(midi_notes)
            
            # Extract Wav2Vec features from recorded audio
            try:
                features = self.feature_extractor.extract_features_from_audio(
                    audio,
                    sr=self.sample_rate,
                    detected_f0=None
                )
                
                # Check if we got valid Wav2Vec features
                if features['wav2vec_features'] is None:
                    print("   ⚠️  Failed to extract Wav2Vec features, retrying...")
                    self.stats['failed_validations'] += 1
                    time.sleep(0.5)
                    continue
                
                # SUCCESS! We have valid features
                print("   ✅ VALIDATED! Wav2Vec features extracted")
                print("   📊 Gesture token: {}".format(features.get('gesture_token', 'N/A')))
                print("   📊 Consonance: {:.2f}".format(features.get('consonance', 0.0)))
                self.stats['successful_validations'] += 1
                
                # Create ground truth record with audio samples
                ground_truth = ChordGroundTruth(
                    sent_midi_notes=midi_notes,
                    detected_midi_notes=[],  # Not used for Wav2Vec training
                    detected_frequencies=[],
                    chord_name=chord_name,
                    inversion=inversion,
                    octave_offset=octave_offset,
                    match_quality=1.0,  # Always 1.0 since we're not doing frequency matching
                    audio_features=features,
                    timestamp=datetime.now().isoformat()
                )
                
                # CRITICAL: Store audio samples for later re-extraction
                ground_truth.audio_features['audio_samples'] = audio.tolist()
                ground_truth.audio_features['sample_rate'] = self.sample_rate
                
                return ground_truth
                
            except Exception as e:
                print("   ❌ Error extracting features: {}".format(e))
                self.stats['failed_validations'] += 1
                time.sleep(0.5)
                continue
        
        print("   ⚠️  Failed after {} attempts".format(max_retries))
        return None
    
    def run_training_session(self, chord_list: List[Tuple] = None,
                            save_to_json: bool = True):
        """
        Run live ground truth validation session
        
        Args:
            chord_list: List of (root, quality, midi_notes, inversion, octave) tuples
            save_to_json: Save validated chords to JSON file
        """
        if chord_list is None:
            # Default test chords
            chord_list = [
                ("C", "", [60, 64, 67], 0, 0),     # Root position
                ("C", "", [48, 52, 55], 0, -1),    # Root, lower octave
                ("C", "", [64, 67, 72], 1, 0),     # 1st inversion
                ("Cm", "m", [60, 63, 67], 0, 0),   # C minor root
                ("D", "", [62, 66, 69], 0, 0),     # D major root
                ("Dm", "m", [62, 65, 69], 0, 0),   # D minor root
                ("E", "", [64, 68, 71], 0, 0),     # E major root
                ("Em", "m", [64, 67, 71], 0, 0),   # E minor root
            ]
        
        print("\n🎯 Ground Truth Validation Session (Wav2Vec)")
        print("📊 Validating {} chord variations".format(len(chord_list)))
        print("⏱️  Chord duration: {}s".format(self.chord_duration))
        print("🔍 Analysis window: {}s".format(self.analysis_window))
        print("=" * 60)
        
        validated_count = 0
        
        for i, (root, quality, midi_notes, inversion, octave) in enumerate(chord_list, 1):
            chord_name = "{}{}".format(root, quality)
            
            print("\n[{}/{}] Validating: {} (inv={}, oct={})".format(
                i, len(chord_list), chord_name, inversion, octave))
            print("─" * 60)
            
            ground_truth = self.validate_chord(
                midi_notes=midi_notes,
                chord_name=chord_name,
                inversion=inversion,
                octave_offset=octave
            )
            
            if ground_truth:
                self.validated_chords.append(ground_truth)
                validated_count += 1
            
            # Small pause between chords
            time.sleep(0.5)
        
        # Print summary
        print("\n" + "=" * 60)
        print("🎯 Session Summary:")
        print("   Total attempts: {}".format(self.stats['total_attempts']))
        print("   Successful validations: {}".format(self.stats['successful_validations']))
        print("   Failed validations: {}".format(self.stats['failed_validations']))
        print("   Retries needed: {}".format(self.stats['retries']))
        print("   Success rate: {:.1%}".format(
            self.stats['successful_validations'] / max(1, self.stats['total_attempts'])
        ))
        
        # Save to JSON
        if save_to_json and self.validated_chords:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = "validation_results_{}.json".format(timestamp)
            
            # Convert to JSON-serializable format
            data = []
            for gt in self.validated_chords:
                gt_dict = asdict(gt)
                data.append(gt_dict)
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            
            print("\n💾 Validation results saved: {}".format(filename))
            print("   You can now train with: --from-validation {}".format(filename))
        
        return validated_count > 0
    
    def load_from_validation_results(self, filepath: str) -> bool:
        """
        Load validated chords from JSON file
        
        Args:
            filepath: Path to validation_results_*.json file
            
        Returns:
            True if successful
        """
        print(f"\n📁 Loading validation results from: {filepath}")
        
        if not os.path.exists(filepath):
            print(f"❌ File not found: {filepath}")
            return False
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            print("✅ Loaded validation data")
            
            # Handle both formats: dict with 'validated_chords' key, or direct list
            if isinstance(data, dict):
                print("   Validation session: {}".format(data.get('timestamp', 'unknown')))
                validated_chords_data = data.get('validated_chords', [])
            elif isinstance(data, list):
                print("   Direct list format (from live validation)")
                validated_chords_data = data
            else:
                print("   ❌ Unknown format")
                return False
            print(f"   Total validated chords: {len(validated_chords_data)}")
            
            # Convert to ChordGroundTruth objects
            print("\n🔄 Re-extracting Wav2Vec features from audio...")
            successful = 0
            failed = 0
            
            for i, chord_data in enumerate(validated_chords_data):
                try:
                    # Get audio data (check both locations for compatibility)
                    audio_samples = chord_data.get('audio_samples', [])
                    if not audio_samples:
                        # Try inside audio_features
                        audio_features_dict = chord_data.get('audio_features', {})
                        audio_samples = audio_features_dict.get('audio_samples', [])
                    
                    if not audio_samples:
                        print(f"   ⚠️  Chord {i+1}: No audio samples")
                        failed += 1
                        continue
                    
                    # Convert to numpy array
                    audio = np.array(audio_samples, dtype=np.float32)
                    
                    # Extract Wav2Vec features
                    detected_f0 = chord_data.get('detected_f0')
                    features = self.feature_extractor.extract_features_from_audio(
                        audio, 
                        sr=self.sample_rate,
                        detected_f0=detected_f0
                    )
                    
                    # Check if we got valid Wav2Vec features
                    if features['wav2vec_features'] is None:
                        print(f"   ⚠️  Chord {i+1}: Failed to extract Wav2Vec features")
                        failed += 1
                        continue
                    
                    # Create ground truth object
                    ground_truth = ChordGroundTruth(
                        sent_midi_notes=chord_data.get('sent_midi_notes', []),
                        detected_midi_notes=chord_data.get('detected_midi_notes', []),
                        detected_frequencies=chord_data.get('detected_frequencies', []),
                        chord_name=chord_data.get('chord_name', ''),
                        inversion=chord_data.get('inversion', 0),
                        octave_offset=chord_data.get('octave_offset', 0),
                        match_quality=chord_data.get('match_quality', 0.0),
                        audio_features=features,
                        timestamp=chord_data.get('timestamp', '')
                    )
                    
                    self.validated_chords.append(ground_truth)
                    successful += 1
                    
                    if (i + 1) % 50 == 0:
                        print("   Processed {}/{} chords...".format(i+1, len(validated_chords_data)))
                
                except Exception as e:
                    print(f"   ⚠️  Chord {i+1}: Error extracting features: {e}")
                    failed += 1
                    continue
            
            print("\n✅ Feature extraction complete:")
            print("   Successful: {}".format(successful))
            print("   Failed: {}".format(failed))
            print("   Ready for training: {} chords".format(len(self.validated_chords)))
            
            return len(self.validated_chords) > 0
            
        except Exception as e:
            print(f"❌ Error loading validation results: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def train_ml_model(self) -> bool:
        """
        Train Random Forest classifier on validated chords
        
        Returns:
            True if successful
        """
        if not self.validated_chords:
            print("❌ No validated chords to train on")
            return False
        
        print("\n🤖 Training ML Chord Classifier (Wav2Vec Features)")
        print("=" * 60)
        
        # Extract features and labels
        X = []
        y = []
        
        for chord in self.validated_chords:
            wav2vec_features = chord.audio_features.get('wav2vec_features')
            if wav2vec_features is None:
                continue
            
            X.append(wav2vec_features)
            y.append(self._extract_chord_type(chord.chord_name))
        
        X = np.array(X)
        y = np.array(y)
        
        print("\n📊 Dataset:")
        print("   Samples: {}".format(len(X)))
        print("   Feature dimensions: {}D (Wav2Vec)".format(X.shape[1]))
        print("   Unique chord types: {}".format(len(np.unique(y))))
        print("   Chord types: {}...".format(sorted(np.unique(y))[:20]))  # Show first 20
        
        # Check minimum samples
        if len(X) < 10:
            print("❌ Not enough samples for training")
            return False
        
        # Calculate appropriate test size based on number of classes
        # For stratified split, need at least 2 samples per class in total
        min_samples_per_class = np.bincount([np.where(np.unique(y) == label)[0][0] for label in y]).min()
        
        # Adjust test_size and stratification based on data
        if min_samples_per_class < 2:
            print(f"   ⚠️  Some classes have only 1 sample, cannot use stratification")
            use_stratify = None
            test_size = 0.2
        elif len(np.unique(y)) > len(y) * 0.3:
            # Too many classes relative to samples, use smaller test set or no stratification
            print(f"   ⚠️  Many classes ({len(np.unique(y))}) relative to samples ({len(y)})")
            print(f"   Using 10% test split without stratification")
            use_stratify = None
            test_size = 0.1
        else:
            use_stratify = y
            test_size = 0.2
        
        # Split train/test
        if use_stratify is not None:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=use_stratify
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
        
        print("\n📊 Train/test split:")
        print("   Training: {} samples".format(len(X_train)))
        print("   Testing: {} samples".format(len(X_test)))
        
        # Scale features
        print("\n🔧 Scaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest classifier
        print("\n🌲 Training Random Forest classifier...")
        print("   (This may take a few minutes with 768D features...)")
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,  # Increased for higher-dimensional features
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = model.score(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)
        
        print("\n📊 Model Performance:")
        print(f"   Training accuracy: {train_score:.1%}")
        print(f"   Testing accuracy: {test_score:.1%}")
        
        # Cross-validation
        print("\n🔄 Running 5-fold cross-validation...")
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, n_jobs=-1)
        print(f"   CV scores: {cv_scores}")
        print(f"   CV mean: {cv_scores.mean():.1%} (+/- {cv_scores.std() * 2:.1%})")
        
        # Detailed classification report
        y_pred = model.predict(X_test_scaled)
        print("\n📊 Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Save model
        print("\n💾 Saving model...")
        os.makedirs('models', exist_ok=True)
        
        model_path = 'models/chord_model_wav2vec.pkl'
        scaler_path = 'models/chord_scaler_wav2vec.pkl'
        
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        
        print("✅ Model saved:")
        print("   {}".format(model_path))
        print("   {}".format(scaler_path))
        
        # Save metadata
        metadata = {
            'training_date': datetime.now().isoformat(),
            'num_samples': len(y),
            'num_classes': len(np.unique(y)),
            'feature_dimensions': X.shape[1],
            'feature_type': 'wav2vec2-base',
            'train_accuracy': float(train_score),
            'test_accuracy': float(test_score),
            'cv_mean': float(cv_scores.mean()),
            'cv_std': float(cv_scores.std()),
            'classes': list(np.unique(y)),
            'model_type': 'RandomForest',
            'n_estimators': 100,
            'max_depth': 20
        }
        
        metadata_path = 'models/chord_model_wav2vec_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"   {metadata_path}")
        
        return True
    
    def start(self) -> bool:
        """Start the system (setup MIDI if needed)"""
        print("🎯 Chord Ground Truth Trainer (Wav2Vec Features)")
        print("=" * 60)
        print("   Feature type: Wav2Vec (768D neural features)")
        print("   GPU acceleration: {}".format('Yes' if self.use_gpu else 'No'))
        
        # If audio device is specified, try to open MIDI for live mode
        if self.audio_device is not None:
            print("\n🎹 Live mode: Setting up MIDI...")
            if not self._open_midi_port():
                print("⚠️  MIDI setup failed. Live mode unavailable.")
                print("   You can still use --from-validation mode")
                return False
            print("🎤 Audio input device: {}".format(self.audio_device))
            print("✅ Live validation ready!")
        
        return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Chord Ground Truth Trainer - Wav2Vec Features',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Mode 1: Train from existing validation results (GPU)
  python chord_ground_truth_trainer_wav2vec.py --from-validation validation_results_20251007_170413.json --gpu
  
  # Mode 1: Train from existing validation results (CPU)
  python chord_ground_truth_trainer_wav2vec.py --from-validation validation_results_20251007_170413.json
  
  # Mode 2: Live validation (create new validation dataset)
  python chord_ground_truth_trainer_wav2vec.py --input-device 0 --gpu
  
  # Mode 2: Live validation then train
  python chord_ground_truth_trainer_wav2vec.py --input-device 0 --gpu --train-after
        """
    )
    
    parser.add_argument('--from-validation', type=str,
                       help='Load validated chords from JSON file (Mode 1)')
    parser.add_argument('--input-device', type=int, default=None,
                       help='Audio input device number for live validation (Mode 2)')
    parser.add_argument('--gpu', action='store_true',
                       help='Use GPU acceleration for Wav2Vec')
    parser.add_argument('--train-after', action='store_true',
                       help='Train ML model after live validation')
    
    args = parser.parse_args()
    
    trainer = ChordGroundTruthTrainer(
        use_gpu=args.gpu,
        audio_input_device=args.input_device
    )
    
    if not trainer.start():
        return
    
    try:
        # Mode 1: Load from validation results and train
        if args.from_validation:
            if trainer.load_from_validation_results(args.from_validation):
                print("\n✅ Ready to train with {} validated chords".format(len(trainer.validated_chords)))
                
                # Train ML model
                if trainer.train_ml_model():
                    print("\n✅ Training complete!")
                    print("\n🎯 Next steps:")
                    print("   1. Update Chandra_trainer.py to load models/chord_model_wav2vec.pkl")
                    print("   2. Retrain Georgia with ground truth chord predictions")
                    print("   3. Test with MusicHal_9000")
                else:
                    print("\n❌ Training failed")
            else:
                print("\n❌ Failed to load validation results")
        
        # Mode 2: Live validation
        elif args.input_device is not None:
            print("\n🎹 Starting live validation session...")
            print("⚠️  Make sure:")
            print("   1. MIDI: '{}' is connected to your synth/Ableton".format(trainer.midi_port_name))
            print("   2. Audio: Device {} is receiving sound".format(args.input_device))
            print("\nPress Enter to start, Ctrl+C to abort...")
            input()
            
            # Run session (will save to JSON automatically)
            if trainer.run_training_session():
                print("\n✅ Live validation complete!")
                
                # Train if requested
                if args.train_after:
                    print("\n🤖 Training ML model...")
                    if trainer.train_ml_model():
                        print("\n✅ Training complete!")
                    else:
                        print("\n❌ Training failed")
                else:
                    print("\n💡 To train on this data, run:")
                    print("   python chord_ground_truth_trainer_wav2vec.py --from-validation validation_results_XXXXXX_XXXXXX.json --gpu")
            else:
                print("\n❌ Live validation failed")
        
        # No mode specified
        else:
            print("\n⚠️  Please specify a mode:")
            print("\n   Mode 1 (from JSON):")
            print("      python chord_ground_truth_trainer_wav2vec.py --from-validation FILE.json --gpu")
            print("\n   Mode 2 (live):")
            print("      python chord_ground_truth_trainer_wav2vec.py --input-device 0 --gpu")
            print("\nRun with --help for more details.")
        
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except Exception as e:
        print("\n❌ Error: {}".format(e))
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

