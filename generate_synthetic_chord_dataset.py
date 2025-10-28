#!/usr/bin/env python3
"""
Generate Synthetic Chord Dataset for Wav2Vec Classifier Training
================================================================

Generates clean synthetic chord audio samples and extracts Wav2Vec features.
This creates training data for the Wav2Vec → Chord classifier.

Approach:
1. Generate synthetic audio for each chord type (additive synthesis)
2. Extract Wav2Vec features from audio
3. Save as ground truth dataset for classifier training

Usage:
    python generate_synthetic_chord_dataset.py --output ground_truth_wav2vec_synthetic.json --gpu
"""

import numpy as np
import argparse
import json
import os
from typing import List, Dict, Tuple
from datetime import datetime

# Import Wav2Vec perception
from listener.dual_perception import DualPerceptionModule


class SyntheticChordGenerator:
    """Generate synthetic chord audio using additive synthesis"""
    
    def __init__(self, sr: int = 44100, duration: float = 2.0):
        """
        Initialize chord generator
        
        Args:
            sr: Sample rate
            duration: Duration of each chord sample in seconds
        """
        self.sr = sr
        self.duration = duration
        self.num_samples = int(sr * duration)
        
        # Define chord templates (intervals from root)
        self.chord_templates = {
            # Triads
            'major': [0, 4, 7],
            'minor': [0, 3, 7],
            'dim': [0, 3, 6],
            'aug': [0, 4, 8],
            'sus2': [0, 2, 7],
            'sus4': [0, 5, 7],
            
            # Seventh chords
            '7': [0, 4, 7, 10],        # Dominant 7th
            'maj7': [0, 4, 7, 11],     # Major 7th
            'm7': [0, 3, 7, 10],       # Minor 7th
            'dim7': [0, 3, 6, 9],      # Diminished 7th
            'ø7': [0, 3, 6, 10],       # Half-diminished (m7b5)
            
            # Extended chords
            '9': [0, 4, 7, 10, 14],    # Dominant 9th (14 = 2 + 12)
            'maj9': [0, 4, 7, 11, 14], # Major 9th
            'm9': [0, 3, 7, 10, 14],   # Minor 9th
            '7#9': [0, 4, 7, 10, 15],  # 7#9 (Hendrix chord)
            'add9': [0, 4, 7, 14],     # Add9 (no 7th)
            
            # Additional
            '6': [0, 4, 7, 9],         # Major 6th
            'm6': [0, 3, 7, 9],        # Minor 6th
        }
        
        # Root notes (all 12 chromatic notes)
        self.note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        # Base octaves to generate
        self.base_octaves = [2, 3, 4]  # Generate in different octaves
        
        # Base MIDI note (C4)
        self.base_midi = 60
    
    def midi_to_freq(self, midi_note: int) -> float:
        """Convert MIDI note to frequency"""
        return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))
    
    def generate_sine_tone(self, frequency: float, amplitude: float = 0.3) -> np.ndarray:
        """Generate sine wave at given frequency"""
        t = np.linspace(0, self.duration, self.num_samples)
        
        # Add envelope (simple ADSR)
        attack = int(0.01 * self.sr)  # 10ms attack
        release = int(0.2 * self.sr)  # 200ms release
        
        envelope = np.ones(self.num_samples)
        envelope[:attack] = np.linspace(0, 1, attack)
        envelope[-release:] = np.linspace(1, 0, release)
        
        wave = amplitude * np.sin(2 * np.pi * frequency * t) * envelope
        
        return wave
    
    def generate_chord(self, root_midi: int, chord_type: str) -> np.ndarray:
        """
        Generate synthetic chord audio
        
        Args:
            root_midi: MIDI note for root
            chord_type: Chord type (e.g., 'major', 'minor', '7')
            
        Returns:
            Audio samples
        """
        if chord_type not in self.chord_templates:
            print(f"⚠️  Unknown chord type: {chord_type}, using major")
            chord_type = 'major'
        
        intervals = self.chord_templates[chord_type]
        
        # Generate each note in the chord
        audio = np.zeros(self.num_samples)
        
        for interval in intervals:
            midi_note = root_midi + interval
            frequency = self.midi_to_freq(midi_note)
            
            # Add some overtones for realism (2nd and 3rd harmonics)
            note_audio = self.generate_sine_tone(frequency, amplitude=0.3)
            note_audio += self.generate_sine_tone(frequency * 2, amplitude=0.1)  # Octave
            note_audio += self.generate_sine_tone(frequency * 3, amplitude=0.05)  # 5th
            
            audio += note_audio
        
        # Normalize
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.8
        
        return audio
    
    def generate_all_chords(self, 
                           inversions: List[int] = [0, 1, 2],
                           variations_per_chord: int = 1) -> List[Tuple[np.ndarray, str, Dict]]:
        """
        Generate all chord combinations
        
        Args:
            inversions: List of inversion numbers to generate
            variations_per_chord: Number of octave variations per chord
            
        Returns:
            List of (audio, label, metadata) tuples
        """
        print(f"\n🎵 Generating synthetic chord dataset...")
        print(f"   Chord types: {len(self.chord_templates)}")
        print(f"   Roots: {len(self.note_names)}")
        print(f"   Octaves: {len(self.base_octaves)}")
        print(f"   Inversions: {len(inversions)}")
        
        dataset = []
        total_chords = len(self.chord_templates) * len(self.note_names) * len(self.base_octaves) * len(inversions)
        
        print(f"   Total chords to generate: {total_chords}\n")
        
        count = 0
        for chord_type in self.chord_templates:
            for root_idx, root_name in enumerate(self.note_names):
                for octave in self.base_octaves:
                    for inversion in inversions:
                        # Calculate root MIDI note
                        root_midi = self.base_midi + root_idx + (octave - 4) * 12
                        
                        # Apply inversion
                        intervals = self.chord_templates[chord_type].copy()
                        if inversion > 0 and inversion < len(intervals):
                            # Move bottom note(s) up an octave
                            for i in range(inversion):
                                intervals[i] += 12
                            intervals.sort()
                        
                        # Generate audio
                        audio = self.generate_chord(root_midi, chord_type)
                        
                        # Create label
                        if chord_type == 'major':
                            label = root_name
                        else:
                            label = f"{root_name}{chord_type}"
                        
                        # Metadata
                        metadata = {
                            'root': root_name,
                            'chord_type': chord_type,
                            'root_midi': root_midi,
                            'octave': octave,
                            'inversion': inversion,
                            'intervals': intervals
                        }
                        
                        dataset.append((audio, label, metadata))
                        count += 1
                        
                        if count % 100 == 0:
                            print(f"   Generated {count}/{total_chords} chords...")
        
        print(f"\n✅ Generated {len(dataset)} chord samples")
        return dataset


def main():
    parser = argparse.ArgumentParser(description="Generate Synthetic Chord Dataset for Wav2Vec Classifier")
    
    parser.add_argument('--output', type=str, 
                       default='ground_truth_wav2vec_synthetic.json',
                       help='Output JSON file')
    parser.add_argument('--gpu', action='store_true',
                       help='Use GPU for Wav2Vec feature extraction')
    parser.add_argument('--duration', type=float, default=2.0,
                       help='Duration of each chord sample (seconds)')
    parser.add_argument('--inversions', type=int, nargs='+', default=[0, 1, 2],
                       help='Chord inversions to generate')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🎵 Synthetic Chord Dataset Generator")
    print("=" * 80)
    
    # Generate synthetic chords
    generator = SyntheticChordGenerator(sr=44100, duration=args.duration)
    dataset = generator.generate_all_chords(inversions=args.inversions)
    
    # Initialize Wav2Vec feature extractor
    print(f"\n🔬 Initializing Wav2Vec Feature Extractor...")
    dual_perception = DualPerceptionModule(
        vocabulary_size=64,
        wav2vec_model='facebook/wav2vec2-base',
        use_gpu=args.gpu,
        enable_symbolic=True
    )
    print("✅ Wav2Vec extractor ready")
    
    # Extract Wav2Vec features for each chord
    print(f"\n🔄 Extracting Wav2Vec features...")
    
    ground_truths = []
    
    for i, (audio, label, metadata) in enumerate(dataset):
        try:
            # Extract Wav2Vec features
            result = dual_perception.extract_features(audio, sr=44100)
            
            # Create ground truth entry
            gt_entry = {
                'chord_label': label,
                'root': metadata['root'],
                'chord_type': metadata['chord_type'],
                'root_midi': metadata['root_midi'],
                'octave': metadata['octave'],
                'inversion': metadata['inversion'],
                'intervals': metadata['intervals'],
                'wav2vec_features': result.wav2vec_features.tolist(),
                'gesture_token': int(result.gesture_token) if result.gesture_token is not None else None,
                'consonance': float(result.consonance) if result.consonance is not None else 0.0,
                'timestamp': datetime.now().isoformat()
            }
            
            ground_truths.append(gt_entry)
            
            if (i + 1) % 100 == 0:
                print(f"   Processed {i + 1}/{len(dataset)} chords...")
        
        except Exception as e:
            print(f"   ⚠️  Failed to extract features for {label}: {e}")
            continue
    
    print(f"\n✅ Feature extraction complete")
    print(f"   Total samples: {len(ground_truths)}")
    
    # Save to JSON
    output_data = {
        'metadata': {
            'generation_date': datetime.now().isoformat(),
            'total_samples': len(ground_truths),
            'unique_chords': len(set(gt['chord_label'] for gt in ground_truths)),
            'unique_chord_types': len(set(gt['chord_type'] for gt in ground_truths)),
            'feature_type': 'wav2vec2-base',
            'feature_dim': 768,
            'duration_per_sample': args.duration,
            'sample_rate': 44100,
            'inversions': args.inversions
        },
        'ground_truths': ground_truths
    }
    
    print(f"\n💾 Saving dataset...")
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✅ Dataset saved to: {args.output}")
    
    # Show statistics
    from collections import Counter
    chord_type_counts = Counter(gt['chord_type'] for gt in ground_truths)
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total samples: {len(ground_truths)}")
    print(f"   Unique chord labels: {len(set(gt['chord_label'] for gt in ground_truths))}")
    print(f"\n   Chord type distribution:")
    for chord_type, count in chord_type_counts.most_common():
        print(f"      {chord_type:10s}: {count:4d} samples")
    
    print("\n" + "=" * 80)
    print("✅ Dataset Generation Complete!")
    print("=" * 80)
    print(f"\nNext step: Train the classifier")
    print(f"   python train_wav2vec_chord_classifier.py \\")
    print(f"       --ground-truth-file {args.output} \\")
    print(f"       --output-model models/wav2vec_chord_classifier.pt \\")
    print(f"       --epochs 50 \\")
    print(f"       --gpu")


if __name__ == "__main__":
    main()

