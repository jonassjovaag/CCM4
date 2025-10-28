#!/usr/bin/env python3
"""
Chord Ground Truth Trainer - Proper Implementation
==================================================

This system:
1. Sends MIDI chord (e.g., C-E-G) → Ableton plays it
2. Records audio for full chord duration
3. Extracts frequency peaks from audio
4. Detects which notes are present (e.g., "I hear C, E, G")
5. Compares detected notes to sent MIDI notes
6. If match → Success! Store as validated ground truth
7. If no match → Replay chord until correct
8. After all chords → Use GPT-OSS to find patterns

This is TRUE ground truth validation!

Usage:
    python chord_ground_truth_trainer.py --input-device 7
"""

import numpy as np
import time
import json
import os
import mido
from datetime import datetime
from typing import List, Optional, Tuple, Set
from dataclasses import dataclass, asdict
import scipy.signal
from scipy.fft import rfft, rfftfreq

from listener.jhs_listener_core import DriftListener, Event


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
    audio_features: dict  # Store for later pattern analysis
    timestamp: str
    

class FrequencyPeakDetector:
    """Detects frequency peaks from audio to identify chord notes"""
    
    @staticmethod
    def detect_notes_from_audio(audio_buffer: np.ndarray, 
                               sr: int = 44100,
                               expected_num_notes: int = 3) -> List[int]:
        """
        Detect MIDI notes from audio using chroma + octave detection
        
        Returns:
            List of MIDI note numbers detected
        """
        import librosa
        
        # Ensure minimum buffer length
        min_length = 2048
        if len(audio_buffer) < min_length:
            audio_buffer = np.pad(audio_buffer, (0, min_length - len(audio_buffer)))
        
        # Extract chroma from audio
        chroma = librosa.feature.chroma_cqt(
            y=audio_buffer,
            sr=sr,
            hop_length=512,
            n_chroma=12,
            n_fft=2048
        )
        
        # Average chroma over time
        chroma_mean = np.mean(chroma, axis=1)
        
        # Find dominant pitch classes - use stronger threshold
        # Take only the N strongest pitch classes
        sorted_indices = np.argsort(chroma_mean)[::-1]
        dominant_pitch_classes = sorted_indices[:expected_num_notes]  # Take only expected number
        
        # Detect octave from strongest frequency component
        # Use FFT to find approximate fundamental
        fft_result = np.abs(rfft(audio_buffer * np.hanning(len(audio_buffer))))
        freqs = rfftfreq(len(audio_buffer), 1/sr)
        
        # Find strongest peak in musical range
        mask = (freqs >= 60) & (freqs <= 1000)
        masked_fft = fft_result.copy()
        masked_fft[~mask] = 0
        
        peak_idx = np.argmax(masked_fft)
        peak_freq = freqs[peak_idx]
        
        # Convert to approximate MIDI to get octave
        if peak_freq > 0:
            approx_midi = 12 * np.log2(peak_freq / 440.0) + 69
            base_octave = int(approx_midi / 12) * 12  # Round to nearest octave (C)
        else:
            base_octave = 48  # Default to C3
        
        # Convert pitch classes to MIDI notes in detected octave
        detected_midi = []
        for pc in dominant_pitch_classes:
            detected_midi.append(int(base_octave + pc))
        
        return sorted(detected_midi)
    
    @staticmethod
    def freq_to_midi(freq: float) -> int:
        """Convert frequency to MIDI note number"""
        if freq <= 0:
            return 0
        midi = 12 * np.log2(freq / 440.0) + 69
        return int(round(midi))
    
    @staticmethod
    def midi_to_note_name(midi: int) -> str:
        """Convert MIDI to note name"""
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        return notes[midi % 12]


class ChordGroundTruthTrainer:
    """Proper ground truth trainer with frequency peak detection"""
    
    def __init__(self,
                 midi_output_port: str = "IAC Driver Chord Trainer Output",
                 audio_input_device: Optional[int] = None,
                 chord_duration: float = 2.5,
                 analysis_window: float = 2.0):
        """
        Initialize trainer
        
        Args:
            chord_duration: How long to play chord
            analysis_window: How much audio to analyze (should be ~= chord_duration)
        """
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
        
        # Validated ground truths
        self.validated_chords = []
        
        # Peak detector
        self.peak_detector = FrequencyPeakDetector()
        
        # Statistics
        self.stats = {
            'total_attempts': 0,
            'successful_validations': 0,
            'failed_validations': 0,
            'retries': 0
        }
    
    def _open_midi_port(self) -> bool:
        """Open MIDI output"""
        try:
            available = mido.get_output_names()
            if self.midi_port_name in available:
                self.midi_port = mido.open_output(self.midi_port_name)
                print(f"✅ MIDI output: {self.midi_port_name}")
                return True
            else:
                print(f"❌ Port not found: {self.midi_port_name}")
                return False
        except Exception as e:
            print(f"❌ MIDI error: {e}")
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
        
        Returns validated ChordGroundTruth or None if validation fails
        """
        for attempt in range(max_retries):
            self.stats['total_attempts'] += 1
            
            if attempt > 0:
                print(f"   🔄 Retry {attempt}/{max_retries-1}...")
                self.stats['retries'] += 1
            
            # Play chord and record simultaneously
            print(f"   🎹 Sending MIDI: {midi_notes} ({[self.peak_detector.midi_to_note_name(n) for n in midi_notes]})")
            
            self._play_chord_midi(midi_notes)
            
            # Record audio while chord plays
            time.sleep(0.1)  # Let sound stabilize
            audio = self._record_audio(self.analysis_window)
            
            # Stop chord
            self._stop_chord_midi(midi_notes)
            
            # Analyze audio to detect notes using chroma
            detected_midi_unique = self.peak_detector.detect_notes_from_audio(
                audio, 
                sr=self.sample_rate,
                expected_num_notes=len(midi_notes)  # Tell it how many notes to find
            )
            sent_midi_unique = sorted(list(set(midi_notes)))
            
            # Calculate match quality
            match_quality = self._calculate_match_quality(sent_midi_unique, detected_midi_unique)
            
            print(f"   🎧 Detected notes: {detected_midi_unique} ({[self.peak_detector.midi_to_note_name(n) for n in detected_midi_unique]})")
            print(f"   🎯 Expected notes: {sent_midi_unique} ({[self.peak_detector.midi_to_note_name(n) for n in sent_midi_unique]})")
            print(f"   📊 Match quality: {match_quality:.1%}")
            
            # Check if validation passed
            if match_quality >= 0.8:  # 80% match threshold
                print(f"   ✅ VALIDATED!")
                self.stats['successful_validations'] += 1
                
                # Create ground truth record
                ground_truth = ChordGroundTruth(
                    sent_midi_notes=midi_notes,
                    detected_midi_notes=detected_midi_unique,
                    detected_frequencies=[],  # Not using frequency data directly anymore
                    chord_name=chord_name,
                    inversion=inversion,
                    octave_offset=octave_offset,
                    match_quality=match_quality,
                    audio_features={
                        'detection_method': 'chroma_cqt',
                        'num_detected': len(detected_midi_unique)
                    },
                    timestamp=datetime.now().isoformat()
                )
                
                return ground_truth
            else:
                print(f"   ❌ Validation failed (match: {match_quality:.1%})")
                self.stats['failed_validations'] += 1
                time.sleep(0.5)  # Wait before retry
        
        print(f"   ⚠️  Failed after {max_retries} attempts")
        return None
    
    def _calculate_match_quality(self, sent: List[int], detected: List[int]) -> float:
        """
        Calculate how well detected notes match sent notes
        
        Returns value 0.0-1.0
        """
        if not sent:
            return 0.0
        
        sent_set = set(sent)
        detected_set = set(detected)
        
        # Count matches (allowing ±1 semitone tolerance for tuning)
        matches = 0
        for s in sent_set:
            if s in detected_set or (s-1) in detected_set or (s+1) in detected_set:
                matches += 1
        
        # Match quality = matches / total sent notes
        return matches / len(sent_set)
    
    def run_training_session(self, chord_list: List[Tuple[str, str, List[int]]] = None):
        """
        Run ground truth validation session
        
        Args:
            chord_list: List of (chord_name, quality, midi_notes) tuples
        """
        if chord_list is None:
            # Basic test: C major in different inversions
            chord_list = [
                ("C", "", [60, 64, 67], 0, 0),     # Root position
                ("C", "", [48, 52, 55], 0, -1),    # Root, lower octave
                ("C", "", [64, 67, 72], 1, 0),     # 1st inversion
                ("Cm", "m", [60, 63, 67], 0, 0),   # C minor root
                ("Cm", "m", [48, 51, 55], 0, -1),  # C minor lower
                ("D", "", [62, 66, 69], 0, 0),     # D major root
                ("Dm", "m", [62, 65, 69], 0, 0),   # D minor root
            ]
        
        print(f"\n🎯 Ground Truth Validation Session")
        print(f"📊 Validating {len(chord_list)} chord variations")
        print(f"⏱️  Chord duration: {self.chord_duration}s")
        print(f"🔍 Analysis window: {self.analysis_window}s")
        print("=" * 60)
        
        validated_count = 0
        
        for i, (root, quality, midi_notes, inversion, octave) in enumerate(chord_list, 1):
            chord_name = f"{root}{quality}"
            
            print(f"\n[{i}/{len(chord_list)}] Validating: {chord_name} (inv={inversion}, oct={octave})")
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
            
            time.sleep(0.5)  # Pause between chords
        
        # Report results
        print(f"\n{'=' * 60}")
        print(f"✅ Validation Complete!")
        print(f"   Successful: {validated_count}/{len(chord_list)}")
        print(f"   Success rate: {100 * validated_count / len(chord_list):.1f}%")
        print(f"   Total attempts: {self.stats['total_attempts']}")
        print(f"   Retries needed: {self.stats['retries']}")
        
        # Save validated ground truths
        self.save_ground_truths()
    
    def save_ground_truths(self):
        """Save validated ground truths to file"""
        os.makedirs('models', exist_ok=True)
        output_path = 'models/chord_ground_truths.json'
        
        data = {
            'validated_chords': [asdict(gt) for gt in self.validated_chords],
            'statistics': self.stats,
            'metadata': {
                'total_chords': len(self.validated_chords),
                'session_date': datetime.now().isoformat(),
                'chord_duration': self.chord_duration,
                'analysis_window': self.analysis_window
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n💾 Ground truths saved: {output_path}")
        print(f"   Ready for GPT-OSS pattern analysis!")
    
    def start(self) -> bool:
        """Start the system"""
        print("🎯 Chord Ground Truth Trainer")
        print("=" * 60)
        
        if not self._open_midi_port():
            return False
        
        print("✅ System ready!")
        return True
    
    def stop(self):
        """Stop the system"""
        if self.midi_port:
            self.midi_port.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Chord Ground Truth Trainer')
    parser.add_argument('--input-device', type=int, default=None,
                       help='Audio input device number')
    parser.add_argument('--chord-duration', type=float, default=2.5,
                       help='How long to play each chord')
    
    args = parser.parse_args()
    
    trainer = ChordGroundTruthTrainer(
        audio_input_device=args.input_device,
        chord_duration=args.chord_duration,
        analysis_window=args.chord_duration - 0.2  # Analyze most of the chord
    )
    
    if not trainer.start():
        return
    
    try:
        # Build comprehensive chord list
        from autonomous_chord_trainer import ChordVocabulary
        
        chord_list = []
        
        # Test with a few chords first
        test_chords = [
            ('C', ''),   # Root, quality (not chord name!)
            ('C', 'm'),  # C, m = C minor
            ('D', ''),
            ('D', 'm'),
        ]
        
        for root, quality in test_chords:
            inversions = ChordVocabulary.get_inversions(root, quality, base_octave=4)
            for inv in inversions:
                chord_list.append((
                    root, 
                    quality,
                    inv.notes,
                    inv.inversion,
                    inv.octave_offset
                ))
        
        trainer.run_training_session(chord_list)
        
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    finally:
        trainer.stop()


if __name__ == "__main__":
    main()

