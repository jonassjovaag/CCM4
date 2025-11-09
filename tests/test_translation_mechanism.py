#!/usr/bin/env python3
"""
Test 1: Core Translation Mechanism Validation
==============================================

Validates the complete gesture token → MIDI note translation chain:

1. Audio → Wav2Vec (768D) → Gesture Token (0-63)
2. Gesture Token → AudioOracle State → audio_data
3. audio_data → MIDI note extraction

Critical Questions:
Q1. Token Stability: Do variations of same chord produce similar gesture tokens?
Q2. Harmonic Separation: Do different chords produce distinct tokens?
Q3. MIDI Mapping Accuracy: Do tokens map to correct MIDI notes?
Q4. Consonance Consistency: Does extracted consonance match expected values?

This is THE core test - if this fails, the entire system fails.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.utils.test_helpers import (
    load_trained_model,
    save_test_results,
    frequency_to_midi,
)
from tests.config import EXPECTED_RANGES


class TranslationMechanismTester:
    """Test gesture token → MIDI translation mechanism"""
    
    def __init__(self, test_audio_dir: Path, ground_truth_path: Path):
        self.test_audio_dir = test_audio_dir
        self.ground_truth_path = ground_truth_path
        
        # Load ground truth
        with open(ground_truth_path) as f:
            self.ground_truth = json.load(f)
        
        print("=" * 80)
        print("TEST 1: CORE TRANSLATION MECHANISM VALIDATION")
        print("=" * 80)
        print(f"\n📂 Test audio: {test_audio_dir}")
        print(f"📄 Ground truth: {ground_truth_path}")
        print(f"📊 Samples: {len(self.ground_truth['samples'])}")
    
    def extract_features_from_audio(self, audio_path: Path) -> Dict:
        """
        Extract features from audio file using the actual production pipeline
        
        Returns:
            Dict with: gesture_token, consonance, f0, midi_notes, etc.
        """
        import librosa
        from listener.dual_perception import DualPerceptionModule
        
        # Load audio
        audio, sr = librosa.load(audio_path, sr=44100, mono=True)
        
        # Initialize perception module
        perception = DualPerceptionModule(use_gpu=False)  # CPU for testing consistency
        
        # Extract features using the actual production method
        perception_result = perception.extract_features(audio, int(sr))
        
        # Extract values from DualPerceptionResult dataclass
        gesture_token = perception_result.gesture_token
        consonance = perception_result.consonance
        f0 = perception_result.ratio_analysis.fundamental if perception_result.ratio_analysis else None
        
        return {
            'gesture_token': int(gesture_token) if gesture_token is not None else 0,
            'consonance': float(consonance),
            'dissonance': float(1.0 - consonance),
            'f0': float(f0) if f0 and f0 > 0 else None,
            'chord_label': perception_result.chord_label,
            'chord_confidence': float(perception_result.chord_confidence),
        }
    
    def test_feature_extraction(self) -> Dict:
        """
        Step 1: Extract features from all test samples
        
        Measures:
        - Token stability within same chord
        - Token separation between different chords
        - Consonance accuracy
        """
        print("\n" + "=" * 80)
        print("STEP 1: FEATURE EXTRACTION FROM TEST AUDIO")
        print("=" * 80)
        
        results = {
            'samples': [],
            'by_chord': defaultdict(list),
            'by_type': defaultdict(list),
        }
        
        for i, sample in enumerate(self.ground_truth['samples']):
            print(f"\n[{i+1}/{len(self.ground_truth['samples'])}] {sample['filename']}")
            print(f"   Ground truth: {sample['chord_name']} ({sample['chord_type']})")
            print(f"   MIDI notes: {sample['midi_notes']}")
            
            # Extract features
            audio_path = Path(sample['filepath'])
            features = self.extract_features_from_audio(audio_path)
            
            # Compare to ground truth
            consonance_error = abs(features['consonance'] - sample['expected_consonance'])
            
            print(f"   Extracted:")
            print(f"      Gesture token: {features['gesture_token']}")
            print(f"      Consonance: {features['consonance']:.3f} (expected {sample['expected_consonance']:.3f}, Δ={consonance_error:.3f})")
            print(f"      F0: {features['f0']:.2f} Hz" if features['f0'] else "      F0: None detected")
            
            # Store results
            result = {
                'sample_id': sample['id'],
                'filename': sample['filename'],
                'chord_name': sample['chord_name'],
                'chord_type': sample['chord_type'],
                'variation': sample['variation'],
                'ground_truth_midi': sample['midi_notes'],
                'expected_consonance': sample['expected_consonance'],
                'extracted_features': features,
                'consonance_error': float(consonance_error),
            }
            
            results['samples'].append(result)
            results['by_chord'][sample['chord_name']].append(result)
            results['by_type'][sample['chord_type']].append(result)
        
        # Analyze token stability
        print("\n" + "=" * 80)
        print("TOKEN STABILITY ANALYSIS")
        print("=" * 80)
        
        token_variance_by_chord = {}
        
        for chord_name, samples in results['by_chord'].items():
            tokens = [s['extracted_features']['gesture_token'] for s in samples]
            token_variance = np.var(tokens)
            unique_tokens = len(set(tokens))
            
            token_variance_by_chord[chord_name] = {
                'tokens': tokens,
                'variance': float(token_variance),
                'unique_count': unique_tokens,
                'total_variations': len(tokens),
            }
            
            print(f"\n{chord_name}:")
            print(f"   Tokens: {tokens}")
            print(f"   Unique: {unique_tokens}/{len(tokens)}")
            print(f"   Variance: {token_variance:.2f}")
            
            if unique_tokens == 1:
                print(f"   ✅ Perfect stability - all variations produce same token")
            elif unique_tokens == len(tokens):
                print(f"   ❌ No stability - every variation produces different token")
            else:
                print(f"   ⚠️  Partial stability - some token overlap")
        
        results['token_stability'] = token_variance_by_chord
        
        # Analyze consonance accuracy
        print("\n" + "=" * 80)
        print("CONSONANCE ACCURACY ANALYSIS")
        print("=" * 80)
        
        consonance_errors = [s['consonance_error'] for s in results['samples']]
        mean_error = np.mean(consonance_errors)
        max_error = np.max(consonance_errors)
        
        print(f"\nConsonance prediction:")
        print(f"   Mean error: {mean_error:.3f}")
        print(f"   Max error: {max_error:.3f}")
        print(f"   Samples with error <0.1: {sum(1 for e in consonance_errors if e < 0.1)}/{len(consonance_errors)}")
        
        results['consonance_accuracy'] = {
            'mean_error': float(mean_error),
            'max_error': float(max_error),
            'acceptable_count': int(sum(1 for e in consonance_errors if e < 0.1)),
            'total_samples': len(consonance_errors),
        }
        
        return results
    
    def test_training_and_mapping(self, extraction_results: Dict) -> Dict:
        """
        Step 2: Train AudioOracle on test samples, validate mapping
        
        This tests whether the model learns correct gesture→MIDI associations
        """
        print("\n" + "=" * 80)
        print("STEP 2: TRAINING AUDIOORACLE ON TEST SAMPLES")
        print("=" * 80)
        
        # Create concatenated test audio file
        import soundfile as sf
        import librosa
        import subprocess
        
        print("\n📝 Creating concatenated test audio...")
        
        all_audio = []
        for sample in self.ground_truth['samples']:
            audio, sr = librosa.load(sample['filepath'], sr=44100, mono=True)
            all_audio.append(audio)
        
        concatenated_audio = np.concatenate(all_audio)
        test_audio_path = self.test_audio_dir / 'concatenated_test_audio.wav'
        sf.write(test_audio_path, concatenated_audio, 44100)
        
        print(f"   ✅ Concatenated {len(all_audio)} samples")
        print(f"   Duration: {len(concatenated_audio)/44100:.1f}s")
        print(f"   Saved: {test_audio_path}")
        
        # Train model using Chandra_trainer.py as script
        print("\n🎓 Training AudioOracle (this may take a moment)...")
        
        model_output_path = self.test_audio_dir / 'test_model.json'
        
        # Build command
        cmd = [
            'python',
            'Chandra_trainer.py',
            '--file', str(test_audio_path),
            '--output', str(model_output_path),
            '--max-events', '100',  # Small for quick testing
            '--training-events', '50',  # Even smaller for speed
            '--no-transformer',  # Disable for speed
            '--no-rhythmic',  # Disable for speed
            '--no-gpt-oss',  # Disable for speed
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,  # 2 minute timeout
            )
            
            if result.returncode != 0:
                print(f"❌ Training failed with return code {result.returncode}")
                print(f"Stderr: {result.stderr}")
                return {'success': False, 'error': result.stderr}
            
            print("   ✅ Training complete")
            
        except subprocess.TimeoutExpired:
            print("❌ Training timed out after 2 minutes")
            return {'success': False, 'error': 'Timeout'}
        except Exception as e:
            print(f"❌ Training error: {e}")
            return {'success': False, 'error': str(e)}
        
        # Load trained model
        print("\n📂 Loading trained model...")
        
        if not model_output_path.exists():
            print(f"❌ Model file not created: {model_output_path}")
            return {'success': False, 'error': 'Model file not found'}
        
        model_data = load_trained_model(model_output_path)
        
        audio_frames = model_data.get('audio_frames', {})
        print(f"   Model states: {len(audio_frames)}")
        
        # Analyze gesture token → MIDI mapping
        print("\n" + "=" * 80)
        print("GESTURE TOKEN → MIDI MAPPING ANALYSIS")
        print("=" * 80)
        
        token_to_midi_map = defaultdict(list)
        token_to_consonance_map = defaultdict(list)
        
        for frame_id, frame in audio_frames.items():
            audio_data = frame.get('audio_data', frame)
            
            if 'gesture_token' in audio_data and 'midi' in audio_data:
                token = audio_data['gesture_token']
                midi = audio_data['midi']
                consonance = audio_data.get('consonance', 0.0)
                
                token_to_midi_map[token].append(midi)
                token_to_consonance_map[token].append(consonance)
        
        # Analyze mapping consistency
        mapping_analysis = {}
        
        print(f"\nFound {len(token_to_midi_map)} unique gesture tokens in model")
        
        for token in sorted(token_to_midi_map.keys()):
            midi_notes = token_to_midi_map[token]
            consonances = token_to_consonance_map[token]
            
            midi_variance = np.var(midi_notes)
            consonance_mean = np.mean(consonances)
            
            mapping_analysis[int(token)] = {
                'midi_notes': [int(m) for m in midi_notes],
                'midi_mean': float(np.mean(midi_notes)),
                'midi_variance': float(midi_variance),
                'consonance_mean': float(consonance_mean),
                'frame_count': len(midi_notes),
            }
            
            if len(midi_notes) >= 3:  # Only print tokens with multiple frames
                print(f"\nToken {token}:")
                print(f"   Frames: {len(midi_notes)}")
                print(f"   MIDI notes: {set(midi_notes)} (variance: {midi_variance:.2f})")
                print(f"   Mean consonance: {consonance_mean:.3f}")
                
                if midi_variance < 5:
                    print(f"   ✅ Consistent MIDI mapping")
                else:
                    print(f"   ⚠️  High variance - token maps to multiple notes")
        
        return {
            'success': True,
            'model_path': str(model_output_path),
            'total_states': len(audio_frames),
            'unique_tokens': len(token_to_midi_map),
            'token_midi_mapping': mapping_analysis,
        }
    
    def run_all_tests(self) -> Dict:
        """Run complete translation mechanism validation"""
        
        # Step 1: Feature extraction
        extraction_results = self.test_feature_extraction()
        
        # Step 2: Training and mapping
        training_results = self.test_training_and_mapping(extraction_results)
        
        # Combine results
        complete_results = {
            'test_name': 'translation_mechanism_validation',
            'timestamp': self.ground_truth['metadata']['created'],
            'extraction': extraction_results,
            'training': training_results,
        }
        
        # Save results
        save_test_results('translation_mechanism', complete_results)
        
        # Summary
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        
        # Token stability
        perfect_stability = sum(
            1 for stats in extraction_results['token_stability'].values()
            if stats['unique_count'] == 1
        )
        total_chords = len(extraction_results['token_stability'])
        
        print(f"\n📊 Token Stability:")
        print(f"   Chords with perfect stability: {perfect_stability}/{total_chords}")
        
        # Consonance accuracy  
        cons_acc = extraction_results['consonance_accuracy']
        print(f"\n📊 Consonance Accuracy:")
        print(f"   Mean error: {cons_acc['mean_error']:.3f}")
        print(f"   Acceptable (<0.1 error): {cons_acc['acceptable_count']}/{cons_acc['total_samples']}")
        
        # Mapping quality
        if training_results['success']:
            print(f"\n📊 Gesture→MIDI Mapping:")
            print(f"   Total states: {training_results['total_states']}")
            print(f"   Unique gesture tokens: {training_results['unique_tokens']}")
        
        print("\n" + "=" * 80)
        
        return complete_results


def main():
    """Run translation mechanism validation test"""
    
    test_audio_dir = Path('tests/test_audio')
    ground_truth_path = test_audio_dir / 'ground_truth.json'
    
    if not ground_truth_path.exists():
        print(f"❌ Ground truth not found: {ground_truth_path}")
        print(f"   Run: python tests/generate_test_audio.py")
        return False
    
    tester = TranslationMechanismTester(test_audio_dir, ground_truth_path)
    results = tester.run_all_tests()
    
    print("\n✅ Translation mechanism test complete!")
    print(f"📄 Results saved: tests/test_output/translation_mechanism_results.json")
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
