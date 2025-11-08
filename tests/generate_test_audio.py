#!/usr/bin/env python3
"""
Generate Synthetic Test Audio for Translation Mechanism Validation
===================================================================

Creates controlled test audio to validate: Gesture Token → MIDI mapping

Test Design:
- 5 chord types (C major, D minor, G major, F major, A minor)
- 3 variations per chord (different octaves/timbres)
- Total: 15 audio samples with known ground truth

Ground Truth Properties:
- Exact MIDI notes for each chord
- Expected consonance values
- Expected harmonic category (major/minor)
- Timestamp and duration

This tests whether:
1. Same harmony produces similar gesture tokens (perceptual consistency)
2. Gesture tokens map to correct MIDI notes (translation accuracy)
3. Different harmonies produce distinct tokens (harmonic separation)
"""

import numpy as np
import soundfile as sf
import json
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime


class SyntheticChordGenerator:
    """Generate synthetic chord audio with known properties"""
    
    def __init__(self, sr: int = 44100):
        self.sr = sr
        
        # Chord definitions (intervals from root in semitones)
        self.chord_templates = {
            'major': [0, 4, 7],      # Root, major 3rd, perfect 5th
            'minor': [0, 3, 7],      # Root, minor 3rd, perfect 5th
        }
        
        # Expected consonance values (approximate)
        self.expected_consonance = {
            'major': 0.92,
            'minor': 0.88,
        }
    
    def midi_to_freq(self, midi_note: int) -> float:
        """Convert MIDI note to frequency (Hz)"""
        return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))
    
    def generate_sine_chord(
        self, 
        midi_notes: List[int], 
        duration: float = 2.0,
        amplitude: float = 0.25,
        overtones: int = 0  # Number of harmonic overtones to add
    ) -> np.ndarray:
        """
        Generate chord using additive synthesis
        
        Args:
            midi_notes: List of MIDI note numbers
            duration: Duration in seconds
            amplitude: Base amplitude (0-1)
            overtones: Number of harmonic overtones (0=pure sine, >0 adds timbre)
        
        Returns:
            Audio samples
        """
        num_samples = int(self.sr * duration)
        t = np.linspace(0, duration, num_samples)
        
        # ADSR envelope
        attack_samples = int(0.02 * self.sr)  # 20ms attack
        decay_samples = int(0.1 * self.sr)    # 100ms decay
        sustain_level = 0.7
        release_samples = int(0.3 * self.sr)  # 300ms release
        
        envelope = np.ones(num_samples) * sustain_level
        
        # Attack
        if attack_samples > 0:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        
        # Decay
        if decay_samples > 0:
            decay_end = attack_samples + decay_samples
            envelope[attack_samples:decay_end] = np.linspace(1, sustain_level, decay_samples)
        
        # Release
        if release_samples > 0:
            envelope[-release_samples:] = np.linspace(sustain_level, 0, release_samples)
        
        # Generate audio
        audio = np.zeros(num_samples)
        
        for midi_note in midi_notes:
            fundamental_freq = self.midi_to_freq(midi_note)
            
            # Fundamental frequency (pure sine)
            note_audio = np.sin(2 * np.pi * fundamental_freq * t)
            
            # Add harmonic overtones for timbre variation
            for harmonic in range(1, overtones + 1):
                overtone_freq = fundamental_freq * (harmonic + 1)
                overtone_amplitude = 1.0 / (harmonic + 1)  # Decreasing amplitude
                note_audio += overtone_amplitude * np.sin(2 * np.pi * overtone_freq * t)
            
            # Normalize if overtones added
            if overtones > 0:
                note_audio = note_audio / (1 + sum(1/(h+1) for h in range(overtones)))
            
            audio += note_audio
        
        # Normalize by number of notes
        audio = audio / len(midi_notes)
        
        # Apply envelope and amplitude
        audio = audio * envelope * amplitude
        
        return audio.astype(np.float32)
    
    def generate_test_dataset(
        self, 
        output_dir: Path,
        duration_per_chord: float = 2.0
    ) -> Dict:
        """
        Generate complete test dataset
        
        Returns:
            Dictionary with ground truth information
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Test chord definitions
        test_chords = [
            # (name, root_midi, chord_type, variations)
            ('C_major', 60, 'major', [
                {'octave_shift': 0, 'overtones': 0, 'desc': 'pure_sine'},
                {'octave_shift': 12, 'overtones': 2, 'desc': 'octave_up_overtones'},
                {'octave_shift': -12, 'overtones': 4, 'desc': 'octave_down_rich'},
            ]),
            ('D_minor', 62, 'minor', [
                {'octave_shift': 0, 'overtones': 0, 'desc': 'pure_sine'},
                {'octave_shift': 12, 'overtones': 2, 'desc': 'octave_up_overtones'},
                {'octave_shift': -12, 'overtones': 4, 'desc': 'octave_down_rich'},
            ]),
            ('G_major', 67, 'major', [
                {'octave_shift': 0, 'overtones': 0, 'desc': 'pure_sine'},
                {'octave_shift': 12, 'overtones': 2, 'desc': 'octave_up_overtones'},
                {'octave_shift': -12, 'overtones': 4, 'desc': 'octave_down_rich'},
            ]),
            ('F_major', 65, 'major', [
                {'octave_shift': 0, 'overtones': 0, 'desc': 'pure_sine'},
                {'octave_shift': 12, 'overtones': 2, 'desc': 'octave_up_overtones'},
                {'octave_shift': -12, 'overtones': 4, 'desc': 'octave_down_rich'},
            ]),
            ('A_minor', 57, 'minor', [
                {'octave_shift': 0, 'overtones': 0, 'desc': 'pure_sine'},
                {'octave_shift': 12, 'overtones': 2, 'desc': 'octave_up_overtones'},
                {'octave_shift': -12, 'overtones': 4, 'desc': 'octave_down_rich'},
            ]),
        ]
        
        ground_truth = {
            'metadata': {
                'created': datetime.now().isoformat(),
                'sample_rate': self.sr,
                'duration_per_sample': duration_per_chord,
                'total_samples': sum(len(chord[3]) for chord in test_chords),
            },
            'samples': []
        }
        
        print("=" * 80)
        print("GENERATING SYNTHETIC TEST AUDIO")
        print("=" * 80)
        
        sample_id = 0
        
        for chord_name, root_midi, chord_type, variations in test_chords:
            template = self.chord_templates[chord_type]
            
            print(f"\n🎵 {chord_name} ({chord_type})")
            
            for variation in variations:
                # Calculate MIDI notes with octave shift
                midi_notes = [root_midi + interval + variation['octave_shift'] 
                             for interval in template]
                
                # Generate audio
                audio = self.generate_sine_chord(
                    midi_notes=midi_notes,
                    duration=duration_per_chord,
                    overtones=variation['overtones']
                )
                
                # Create filename
                filename = f"{chord_name}_{variation['desc']}.wav"
                filepath = output_dir / filename
                
                # Save audio
                sf.write(filepath, audio, self.sr)
                
                # Store ground truth
                sample_info = {
                    'id': sample_id,
                    'filename': filename,
                    'filepath': str(filepath),
                    'chord_name': chord_name,
                    'chord_type': chord_type,
                    'root_midi': root_midi,
                    'midi_notes': midi_notes,
                    'variation': variation['desc'],
                    'octave_shift': variation['octave_shift'],
                    'overtones': variation['overtones'],
                    'expected_consonance': self.expected_consonance[chord_type],
                    'duration': duration_per_chord,
                }
                
                ground_truth['samples'].append(sample_info)
                
                print(f"   ✅ {filename}")
                print(f"      MIDI notes: {midi_notes}")
                print(f"      Octave shift: {variation['octave_shift']:+d}")
                print(f"      Overtones: {variation['overtones']}")
                
                sample_id += 1
        
        # Save ground truth
        ground_truth_path = output_dir / 'ground_truth.json'
        with open(ground_truth_path, 'w') as f:
            json.dump(ground_truth, f, indent=2)
        
        print("\n" + "=" * 80)
        print(f"✅ Generated {sample_id} test samples")
        print(f"📄 Ground truth saved: {ground_truth_path}")
        print("=" * 80)
        
        return ground_truth


def main():
    """Generate test audio dataset"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate synthetic test audio for translation mechanism validation"
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('tests/test_audio'),
        help='Output directory for audio files'
    )
    parser.add_argument(
        '--duration',
        type=float,
        default=2.0,
        help='Duration per chord sample (seconds)'
    )
    
    args = parser.parse_args()
    
    generator = SyntheticChordGenerator()
    ground_truth = generator.generate_test_dataset(
        output_dir=args.output_dir,
        duration_per_chord=args.duration
    )
    
    # Print summary
    print("\n📊 Dataset Summary:")
    print(f"   Total samples: {ground_truth['metadata']['total_samples']}")
    print(f"   Sample rate: {ground_truth['metadata']['sample_rate']} Hz")
    print(f"   Duration per sample: {ground_truth['metadata']['duration_per_sample']}s")
    
    # Chord type breakdown
    chord_types = {}
    for sample in ground_truth['samples']:
        ct = sample['chord_type']
        chord_types[ct] = chord_types.get(ct, 0) + 1
    
    print(f"\n   Chord types:")
    for chord_type, count in chord_types.items():
        print(f"      {chord_type}: {count} samples")
    
    print(f"\n✅ Test audio generation complete!")
    print(f"   Next step: Run tests/test_translation_mechanism.py")


if __name__ == '__main__':
    main()
