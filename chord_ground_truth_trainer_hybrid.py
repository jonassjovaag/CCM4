#!/usr/bin/env python3
"""
Chord Ground Truth Trainer - Hybrid Features Version
====================================================

This version extracts 21D hybrid features (chroma + ratio) for training
the ML chord detection model.

Features extracted:
- 12D chroma (harmonic-aware)
- 10D ratio features (consonance, intervals, chord type)
- Total: 22D features

Can train on:
1. Live MIDI → audio validation (original mode)
2. Pre-validated dataset (validation_results_*.json)

Usage:
    # Train from existing validation results
    python chord_ground_truth_trainer_hybrid.py --from-validation validation_results_20251007_170413.json
    
    # Live validation (original mode)
    python chord_ground_truth_trainer_hybrid.py --input-device 7
"""

import numpy as np
import time
import json
import os
from datetime import datetime
from typing import List, Optional, Dict
from dataclasses import dataclass
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report

# Import hybrid perception
from listener.hybrid_perception import HybridPerceptionModule


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
    audio_features: dict  # Store hybrid features
    timestamp: str


class HybridFeatureExtractor:
    """Extract hybrid perception features from audio"""
    
    def __init__(self):
        self.hybrid_perception = HybridPerceptionModule(
            vocabulary_size=64,
            enable_ratio_analysis=True,
            enable_symbolic=False  # Don't need symbolic tokens for training
        )
    
    def extract_features_from_audio(self, 
                                   audio: np.ndarray,
                                   sr: int = 44100,
                                   detected_f0: Optional[float] = None) -> Dict:
        """
        Extract hybrid features from audio
        
        Returns:
            Dict with:
                - hybrid_features: 22D feature vector
                - chroma: 12D chroma
                - ratio_features: 10D ratio features
                - consonance: consonance score
                - chord_detected: chord type from ratio analyzer
                - confidence: confidence score
        """
        # Extract hybrid features
        result = self.hybrid_perception.extract_features(
            audio, 
            sr=sr,
            timestamp=time.time(),
            detected_f0=detected_f0
        )
        
        # Prepare feature dict
        features = {
            'hybrid_features': result.features.tolist(),
            'chroma': result.chroma.tolist(),
            'active_pitch_classes': result.active_pitch_classes.tolist(),
            'consonance': result.consonance,
            'chord_detected': None,
            'confidence': 0.0
        }
        
        # Add ratio analysis if available
        if result.ratio_analysis:
            features['chord_detected'] = result.ratio_analysis.chord_match['type']
            features['confidence'] = result.ratio_analysis.chord_match['confidence']
        
        return features


class ChordGroundTruthTrainer:
    """
    Trainer with hybrid feature extraction
    Can work in two modes:
    1. Live validation (MIDI → audio → detect)
    2. Load from pre-validated dataset
    """
    
    def __init__(self,
                 midi_output_port: str = "IAC Driver Chord Trainer Output",
                 audio_input_device: Optional[int] = None,
                 chord_duration: float = 2.5,
                 analysis_window: float = 2.0):
        """Initialize trainer"""
        self.midi_port_name = midi_output_port
        self.audio_device = audio_input_device
        self.chord_duration = chord_duration
        self.analysis_window = analysis_window
        
        # MIDI
        self.midi_port = None
        
        # Audio recording
        self.sample_rate = 44100
        
        # Feature extractor
        self.feature_extractor = HybridFeatureExtractor()
        
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
        E.g., "C#m7" → "m7", "F#" → "major", "Gdim7" → "dim7"
        
        This reduces the number of classes for better ML training
        """
        # Remove root note (first 1-2 chars)
        if len(chord_name) < 2:
            return "major"
        
        # Handle sharp/flat roots
        if chord_name[1] in ['#', 'b']:
            chord_type = chord_name[2:] if len(chord_name) > 2 else ""
        else:
            chord_type = chord_name[1:]
        
        # Map to standard types
        if chord_type == "" or chord_type == "maj":
            return "major"
        elif chord_type == "m":
            return "minor"
        elif chord_type == "7":
            return "dom7"
        elif chord_type == "m7":
            return "min7"
        elif chord_type == "maj7":
            return "maj7"
        elif chord_type == "m7b5":
            return "m7b5"
        elif chord_type == "dim":
            return "dim"
        elif chord_type == "dim7":
            return "dim7"
        elif chord_type == "aug":
            return "aug"
        elif chord_type in ["sus2", "sus4"]:
            return chord_type
        elif chord_type == "9":
            return "9"
        elif chord_type == "m9":
            return "m9"
        elif chord_type == "maj9":
            return "maj9"
        else:
            return chord_type  # Keep as-is
    
    def _frequencies_to_chroma(self, frequencies: List[float]) -> np.ndarray:
        """
        Convert list of frequencies to 12-dimensional chroma vector
        
        Args:
            frequencies: List of frequencies in Hz
            
        Returns:
            12D chroma vector (normalized)
        """
        chroma = np.zeros(12)
        
        for freq in frequencies:
            if freq <= 0:
                continue
            
            # Convert frequency to MIDI note
            midi = 12 * np.log2(freq / 440.0) + 69
            
            # Get pitch class (0-11)
            pitch_class = int(round(midi)) % 12
            
            # Add to chroma (with Gaussian smoothing)
            chroma[pitch_class] += 1.0
            
            # Also add to adjacent pitch classes (smoothing)
            left = (pitch_class - 1) % 12
            right = (pitch_class + 1) % 12
            chroma[left] += 0.3
            chroma[right] += 0.3
        
        # Normalize
        if chroma.sum() > 0:
            chroma = chroma / chroma.sum()
        
        return chroma
    
    def load_from_validation_results(self, json_path: str) -> bool:
        """
        Load validated chords from validation_results_*.json
        
        This allows us to train on the 600-chord dataset without re-validating
        """
        print(f"📂 Loading validation results from {json_path}")
        
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                print(f"❌ Expected list, got {type(data)}")
                return False
            
            loaded_count = 0
            
            for item in data:
                # Only use validated entries
                if not item.get('validation_passed', False):
                    continue
                
                # Extract chord name (e.g., "C root" → "C")
                chord_name_full = item.get('chord_name', '')
                chord_name = chord_name_full.split()[0] if chord_name_full else 'unknown'
                
                # Create ground truth entry
                gt = ChordGroundTruth(
                    sent_midi_notes=item.get('sent_midi', []),
                    detected_midi_notes=[],  # Not stored in validation results
                    detected_frequencies=item.get('detected_frequencies', []),
                    chord_name=chord_name,
                    inversion=0,  # Could parse from name
                    octave_offset=0,
                    match_quality=1.0 if item.get('validation_passed') else 0.0,
                    audio_features={},  # Will extract hybrid features if audio available
                    timestamp=item.get('timestamp', '')
                )
                
                # Store ratio analysis for feature extraction
                if 'ratio_analysis' in item:
                    gt.audio_features['ratio_analysis'] = item['ratio_analysis']
                
                self.validated_chords.append(gt)
                loaded_count += 1
            
            print(f"✅ Loaded {loaded_count} validated chords")
            return loaded_count > 0
            
        except Exception as e:
            print(f"❌ Error loading validation results: {e}")
            return False
    
    def train_ml_model(self) -> bool:
        """
        Train ML chord detection model using hybrid features
        
        Returns:
            True if training successful
        """
        print("\n🎓 Training ML Chord Detection Model")
        print("=" * 60)
        
        if len(self.validated_chords) == 0:
            print("❌ No validated chords to train on")
            return False
        
        # Prepare training data
        print(f"📊 Preparing training data from {len(self.validated_chords)} chords...")
        
        features_list = []
        labels_list = []
        
        for gt in self.validated_chords:
            # Extract hybrid features from ratio analysis and detected frequencies
            if 'ratio_analysis' in gt.audio_features and len(gt.detected_frequencies) > 0:
                ratio = gt.audio_features['ratio_analysis']
                
                # Build chroma vector from detected frequencies
                chroma = self._frequencies_to_chroma(gt.detected_frequencies)
                
                # Extract ratio features
                fundamental = ratio.get('fundamental', 440.0)
                ratios = ratio.get('ratios', [])
                consonance = ratio.get('consonance_score', 0.5)
                confidence = ratio.get('confidence', 0.5)
                intervals = ratio.get('intervals', [])
                
                # Average interval consonance
                avg_interval_consonance = 0.5
                if intervals:
                    consonances = [i.get('consonance', 0.5) for i in intervals]
                    avg_interval_consonance = np.mean(consonances)
                
                ratio_features = np.array([
                    fundamental / 1000.0,  # Normalized fundamental (0-8)
                    *ratios[:4],  # Up to 4 ratios (typically 1.0, 1.25, 1.5, etc.)
                    *([1.0] * (4 - len(ratios[:4]))),  # Pad with 1.0 (unison)
                    consonance,  # Overall consonance (0-1)
                    len(intervals) / 10.0,  # Number of intervals, normalized
                    confidence,  # Confidence (0-1)
                    avg_interval_consonance  # Average interval quality
                ])[:10]  # Ensure 10D
                
                # Combine: 12D chroma + 10D ratio = 22D
                full_features = np.concatenate([chroma, ratio_features])
                
                # Extract chord TYPE (not full name with root)
                chord_type = self._extract_chord_type(gt.chord_name)
                
                features_list.append(full_features)
                labels_list.append(chord_type)
        
        if len(features_list) == 0:
            print("❌ No features extracted")
            return False
        
        # Convert to arrays
        X = np.array(features_list)
        y = np.array(labels_list)
        
        print(f"✅ Extracted features: {X.shape}")
        print(f"   Feature dimensions: {X.shape[1]}")
        print(f"   Unique chord labels: {len(np.unique(y))}")
        
        # Check for sufficient samples per class
        unique, counts = np.unique(y, return_counts=True)
        print(f"\n📊 Class distribution:")
        for label, count in zip(unique, counts):
            print(f"   {label}: {count} samples")
        
        # Filter out classes with too few samples
        min_samples = 3
        valid_classes = unique[counts >= min_samples]
        mask = np.isin(y, valid_classes)
        X = X[mask]
        y = y[mask]
        
        print(f"\n✅ Using {len(valid_classes)} classes with ≥{min_samples} samples")
        print(f"   Total samples: {len(y)}")
        
        if len(y) < 10:
            print("❌ Not enough samples for training")
            return False
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print("\n📊 Train/test split:")
        print(f"   Training: {len(X_train)} samples")
        print(f"   Testing: {len(X_test)} samples")
        
        # Scale features
        print("\n🔧 Scaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest classifier
        print("\n🌲 Training Random Forest classifier...")
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
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
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        print(f"   CV scores: {cv_scores}")
        print(f"   CV mean: {cv_scores.mean():.1%} (+/- {cv_scores.std() * 2:.1%})")
        
        # Detailed classification report
        y_pred = model.predict(X_test_scaled)
        print("\n📊 Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Save model
        print("\n💾 Saving model...")
        os.makedirs('models', exist_ok=True)
        
        model_path = 'models/chord_model_hybrid.pkl'
        scaler_path = 'models/chord_scaler_hybrid.pkl'
        
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        
        print(f"✅ Model saved:")
        print(f"   {model_path}")
        print(f"   {scaler_path}")
        
        # Save metadata
        metadata = {
            'training_date': datetime.now().isoformat(),
            'num_samples': len(y),
            'num_classes': len(np.unique(y)),
            'feature_dimensions': X.shape[1],
            'train_accuracy': float(train_score),
            'test_accuracy': float(test_score),
            'cv_mean': float(cv_scores.mean()),
            'cv_std': float(cv_scores.std()),
            'classes': list(np.unique(y))
        }
        
        metadata_path = 'models/chord_model_hybrid_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"   {metadata_path}")
        
        return True
    
    def start(self) -> bool:
        """Start the system (if using live mode)"""
        print("🎯 Chord Ground Truth Trainer (Hybrid Features)")
        print("=" * 60)
        return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Chord Ground Truth Trainer - Hybrid Features',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train from existing validation results
  python chord_ground_truth_trainer_hybrid.py --from-validation validation_results_20251007_170413.json
  
  # Live validation mode (not implemented yet)
  python chord_ground_truth_trainer_hybrid.py --input-device 7
        """
    )
    
    parser.add_argument('--from-validation', type=str,
                       help='Load validated chords from JSON file')
    parser.add_argument('--input-device', type=int, default=None,
                       help='Audio input device number (for live mode)')
    
    args = parser.parse_args()
    
    trainer = ChordGroundTruthTrainer(
        audio_input_device=args.input_device
    )
    
    if not trainer.start():
        return
    
    try:
        # Mode 1: Load from validation results
        if args.from_validation:
            if trainer.load_from_validation_results(args.from_validation):
                print(f"\n✅ Ready to train with {len(trainer.validated_chords)} validated chords")
                
                # Train ML model
                if trainer.train_ml_model():
                    print("\n✅ Training complete!")
                else:
                    print("\n❌ Training failed")
            else:
                print("\n❌ Failed to load validation results")
        
        # Mode 2: Live validation (not implemented yet)
        else:
            print("\n⚠️  Live validation mode not implemented yet")
            print("    Use --from-validation to train from existing data")
        
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

