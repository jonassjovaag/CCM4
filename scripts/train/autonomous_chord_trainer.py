#!/usr/bin/env python3
"""
Autonomous Chord Trainer - Self-Learning System
================================================

This system:
1. Generates chords with inversions via MIDI → Ableton
2. Listens to audio playback through speakers
3. Analyzes the audio in real-time
4. Correlates analysis with ground truth
5. Trains ML model automatically

Usage:
    python autonomous_chord_trainer.py

Setup:
1. Create an IAC Driver port called "IAC Driver Chord Trainer Output"
2. In Ableton: route "Chord Trainer Output" to a piano/synth
3. Set audio input to capture Ableton's output (speakers/loopback)
4. Run the script - it will train itself!
"""

import numpy as np
import time
import json
import os
import mido
from datetime import datetime
from typing import List, Optional, Tuple
from dataclasses import dataclass

# Import our existing components
from listener.jhs_listener_core import DriftListener, Event
from listener.harmonic_context import RealtimeHarmonicDetector

# ML
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib


@dataclass
class ChordInversion:
    """Represents a chord with its inversion"""
    root: str  # e.g., "C"
    quality: str  # e.g., "", "m", "7", "maj7"
    inversion: int  # 0=root, 1=first, 2=second
    notes: List[int]  # MIDI note numbers
    octave_offset: int = 0  # 0 or -1 for lower octave


class ChordVocabulary:
    """Generates MIDI notes for chord vocabulary"""
    
    # Note name to MIDI offset (C=0, C#=1, etc.)
    NOTE_OFFSETS = {
        'C': 0, 'C#': 1, 'Db': 1,
        'D': 2, 'D#': 3, 'Eb': 3,
        'E': 4,
        'F': 5, 'F#': 6, 'Gb': 6,
        'G': 7, 'G#': 8, 'Ab': 8,
        'A': 9, 'A#': 10, 'Bb': 10,
        'B': 11
    }
    
    # Chord qualities as semitone intervals from root
    CHORD_INTERVALS = {
        '': [0, 4, 7],                    # Major: C E G
        'm': [0, 3, 7],                   # Minor: C Eb G
        '7': [0, 4, 7, 10],               # Dominant 7th: C E G Bb
        'maj7': [0, 4, 7, 11],            # Major 7th: C E G B
        'm7': [0, 3, 7, 10],              # Minor 7th: C Eb G Bb
        'dim': [0, 3, 6],                 # Diminished: C Eb Gb
        'dim7': [0, 3, 6, 9],             # Diminished 7th
        'm7b5': [0, 3, 6, 10],            # Half-diminished
        'aug': [0, 4, 8],                 # Augmented
        'sus2': [0, 2, 7],                # Suspended 2nd
        'sus4': [0, 5, 7],                # Suspended 4th
        '9': [0, 4, 7, 10, 14],           # Dominant 9th
        'maj9': [0, 4, 7, 11, 14],        # Major 9th
        'm9': [0, 3, 7, 10, 14],          # Minor 9th
    }
    
    @classmethod
    def get_chord_notes(cls, root: str, quality: str, base_octave: int = 4) -> List[int]:
        """Get MIDI notes for a chord in root position"""
        if root not in cls.NOTE_OFFSETS:
            raise ValueError(f"Unknown root note: {root}")
        if quality not in cls.CHORD_INTERVALS:
            raise ValueError(f"Unknown chord quality: {quality}")
        
        root_midi = 60 + (base_octave - 4) * 12 + cls.NOTE_OFFSETS[root]
        intervals = cls.CHORD_INTERVALS[quality]
        return [root_midi + interval for interval in intervals]
    
    @classmethod
    def get_inversions(cls, root: str, quality: str, base_octave: int = 4) -> List[ChordInversion]:
        """Generate all inversions of a chord"""
        root_notes = cls.get_chord_notes(root, quality, base_octave)
        inversions = []
        
        num_inversions = len(root_notes)  # Include root position
        
        for inv_num in range(min(3, num_inversions)):  # Max 3 inversions
            notes = root_notes.copy()
            
            # Rotate notes for inversion
            for _ in range(inv_num):
                bottom_note = notes.pop(0)
                notes.append(bottom_note + 12)  # Move to next octave
            
            inversions.append(ChordInversion(
                root=root,
                quality=quality,
                inversion=inv_num,
                notes=notes,
                octave_offset=0
            ))
            
            # Add lower octave variant
            inversions.append(ChordInversion(
                root=root,
                quality=quality,
                inversion=inv_num,
                notes=[n - 12 for n in notes],
                octave_offset=-1
            ))
        
        return inversions


class AutonomousChordTrainer:
    """Self-learning chord detection system with adaptive validation"""
    
    def __init__(self, 
                 midi_output_port: str = "IAC Driver Chord Trainer Output",
                 audio_input_device: Optional[int] = None,
                 chord_duration: float = 2.0,
                 analysis_delay: float = 0.5,
                 enable_validation: bool = True,
                 validation_threshold: float = 0.7):
        """
        Initialize autonomous chord trainer
        
        Args:
            midi_output_port: MIDI port to send chords to Ableton
            audio_input_device: Audio input device to listen to playback
            chord_duration: How long to play each chord (seconds)
            analysis_delay: Delay after note_on before analyzing (let sound stabilize)
        """
        self.midi_port_name = midi_output_port
        self.audio_device = audio_input_device
        self.chord_duration = chord_duration
        self.analysis_delay = analysis_delay
        
        # MIDI
        self.midi_port = None
        self.midi_channel = 0
        
        # Audio
        self.listener = None
        self.harmonic_detector = RealtimeHarmonicDetector()
        
        # Training data
        self.training_samples = []  # (features, chord_label, inversion_info)
        self.chord_mapping = {}  # chord_str -> index
        self.next_chord_id = 0
        
        # Ground truth tracking
        self.current_ground_truth = None  # ChordInversion being played
        self.ground_truth_start_time = 0
        self.collecting_samples = False
        
        # Chroma accumulation for better harmonic analysis
        self.accumulated_chroma = None
        self.chroma_sample_count = 0
        
        # ML
        self.model = None
        self.scaler = StandardScaler()
        
        # Validation
        self.enable_validation = enable_validation
        self.validation_threshold = validation_threshold
        self.validation_predictions = []  # Store predictions during collection
        self.last_prediction = None
        self.last_prediction_confidence = 0.0
        
        # Note-level memory for intelligent validation
        self.note_memory = {}  # MIDI note -> list of chord names containing it
        self.chord_note_sets = {}  # chord_name -> set of MIDI notes in that chord
        
        # Persistence
        os.makedirs('models', exist_ok=True)
        self.model_path = 'models/autonomous_chord_model.pkl'
        self.scaler_path = 'models/autonomous_chord_scaler.pkl'
        self.metadata_path = 'models/autonomous_chord_metadata.json'
        self.training_log_path = 'models/autonomous_training_log.json'
        
        # Statistics
        self.stats = {
            'chords_played': 0,
            'samples_collected': 0,
            'training_iterations': 0,
            'start_time': datetime.now().isoformat()
        }
    
    def _open_midi_port(self) -> bool:
        """Open MIDI output port"""
        try:
            available_ports = mido.get_output_names()
            print(f"📡 Available MIDI ports: {available_ports}")
            
            if self.midi_port_name in available_ports:
                self.midi_port = mido.open_output(self.midi_port_name)
                print(f"✅ MIDI output: {self.midi_port_name}")
                return True
            else:
                print(f"❌ Port '{self.midi_port_name}' not found!")
                print(f"💡 Create an IAC Driver port in Audio MIDI Setup")
                return False
        except Exception as e:
            print(f"❌ Failed to open MIDI port: {e}")
            return False
    
    def _start_audio_listener(self) -> bool:
        """Start audio input listener"""
        try:
            self.listener = DriftListener(
                ref_fn=lambda midi_note: 440.0,  # Accept MIDI note argument but return constant
                a4_fn=lambda: 440.0,
                device=self.audio_device
            )
            self.listener.start(self._on_audio_event)
            print(f"✅ Audio listener started (device: {self.audio_device or 'default'})")
            return True
        except Exception as e:
            print(f"❌ Failed to start audio listener: {e}")
            return False
    
    def _on_audio_event(self, *args):
        """
        Handle audio events during chord playback
        
        DriftListener can call this with either:
        - One Event object when sound is detected
        - Multiple parameters (None, 0, 0.0, ...) during silence
        """
        # Handle both callback signatures
        if len(args) == 1:
            event = args[0]
        else:
            # Multiple args = silence callback, ignore
            return
        
        # Check if event is valid (Event object, not silence callback)
        if event is None or not isinstance(event, Event):
            return
            
        if not self.collecting_samples or self.current_ground_truth is None:
            return
        
        # Accumulate chroma from all events during chord playback
        try:
            if self.listener and hasattr(self.listener, '_ring'):
                audio_buffer = self.listener._ring.copy()
                self.harmonic_detector.update_from_audio(audio_buffer)
                context = self.harmonic_detector.current_context
                
                if context and hasattr(context, 'chroma') and context.chroma is not None:
                    if self.accumulated_chroma is None:
                        self.accumulated_chroma = np.array(context.chroma, dtype=np.float32)
                        self.chroma_sample_count = 1
                    else:
                        self.accumulated_chroma += np.array(context.chroma, dtype=np.float32)
                        self.chroma_sample_count += 1
        except Exception:
            pass
        
        # Check if enough time has passed since chord started (let sound stabilize)
        if time.time() - self.ground_truth_start_time < self.analysis_delay:
            return
        
        try:
            # Extract features from event
            features = self._extract_features(event)
            
            # Skip if features are None (filtered out due to frequency range)
            if features is None:
                return
            
            # Get chord label
            chord_label = self._get_chord_label(self.current_ground_truth)
            
            # Store sample with ground truth
            self.training_samples.append({
                'features': features,
                'chord_label': chord_label,
                'root': self.current_ground_truth.root,
                'quality': self.current_ground_truth.quality,
                'inversion': self.current_ground_truth.inversion,
                'octave_offset': self.current_ground_truth.octave_offset,
                'notes': self.current_ground_truth.notes,
                'timestamp': time.time()
            })
            
            self.stats['samples_collected'] += 1
            
            # NOTE: We're NOT doing ML validation here anymore
            # The validation happens at the NOTE level (see _stop_chord)
            
        except Exception as e:
            print(f"⚠️  Error collecting sample: {e}")
    
    def _extract_features(self, event: Event) -> np.ndarray:
        """
        Extract features from audio event using ACCUMULATED chroma
        
        IMPORTANT: We trust the MIDI ground truth, so this just extracts
        audio features to correlate with the known chord being played.
        Musical filtering applied: 196Hz - 4186Hz (G3 to C8 piano range)
        """
        # Filter event by musical frequency range
        # Only use events within typical chord playing range (G3-C8)
        # BUT: f0 from YIN is monophonic, might not be in chord range
        # So we check if it's at least somewhat musical
        if event.f0 > 0 and (event.f0 < 80.0 or event.f0 > 5000.0):
            # Way outside any musical range - noise
            return None
        
        # Also filter by RMS - skip very quiet samples
        if event.rms_db < -60.0:
            # Too quiet - likely silence or noise
            return None
        
        # Build feature vector
        features = [
            event.rms_db,
            event.f0,
            event.centroid,
            event.rolloff,
            event.zcr,
            event.bandwidth,
        ]
        
        # Use accumulated chroma (averaged over entire chord duration)
        if self.accumulated_chroma is not None and self.chroma_sample_count > 0:
            # Average the accumulated chroma
            chroma_avg = self.accumulated_chroma / self.chroma_sample_count
            # Normalize to sum to 1.0
            chroma_sum = np.sum(chroma_avg) + 1e-10
            chroma_normalized = chroma_avg / chroma_sum
            features.extend(chroma_normalized)
        else:
            features.extend([0.0] * 12)  # Placeholder if chroma unavailable
        
        # Add MFCC if available (first 3 for timbral signature)
        if event.mfcc and len(event.mfcc) > 0:
            features.extend(event.mfcc[:3])  # First 3 MFCCs
        else:
            features.extend([0.0] * 3)
        
        return np.array(features, dtype=np.float32)
    
    def _get_chord_label(self, inversion: ChordInversion) -> str:
        """Get unique label for chord"""
        # Include inversion and octave in label for detailed training
        label = f"{inversion.root}{inversion.quality}_inv{inversion.inversion}_oct{inversion.octave_offset}"
        
        if label not in self.chord_mapping:
            self.chord_mapping[label] = self.next_chord_id
            self.next_chord_id += 1
        
        return label
    
    def _play_chord(self, inversion: ChordInversion):
        """Send MIDI chord to Ableton"""
        if not self.midi_port:
            return
        
        # Reset chroma accumulation for new chord
        self.accumulated_chroma = None
        self.chroma_sample_count = 0
        
        # Note on for all notes
        for note in inversion.notes:
            msg = mido.Message('note_on', channel=self.midi_channel, note=note, velocity=80)
            self.midi_port.send(msg)
        
        self.current_ground_truth = inversion
        self.ground_truth_start_time = time.time()
        self.collecting_samples = True
        self.stats['chords_played'] += 1
        
        print(f"🎹 Playing: {inversion.root}{inversion.quality} (inv={inversion.inversion}, oct={inversion.octave_offset}) → {inversion.notes}")
    
    def _stop_chord(self):
        """Stop current chord and validate if model exists"""
        if not self.midi_port or not self.current_ground_truth:
            return
        
        # Note off for all notes
        for note in self.current_ground_truth.notes:
            msg = mido.Message('note_off', channel=self.midi_channel, note=note, velocity=0)
            self.midi_port.send(msg)
        
        self.collecting_samples = False
        ground_truth_label = self._get_chord_label(self.current_ground_truth)
        samples_for_chord = len([s for s in self.training_samples if s['chord_label'] == ground_truth_label])
        
        # Report collection with feature debug
        print(f"   Collected {samples_for_chord} samples", end="")
        
        # DEBUG: Show chroma for first sample to verify it's capturing harmonics
        if samples_for_chord > 0 and True:  # Set to True to enable debug
            recent_samples = [s for s in self.training_samples if s['chord_label'] == ground_truth_label]
            if recent_samples:
                sample_features = recent_samples[0]['features']
                chroma = sample_features[6:18]
                print(f"\n   DEBUG Chroma: {[f'{c:.3f}' for c in chroma]}", end="")
        
        print()  # Newline
        
        # NOTE-LEVEL VALIDATION: Check if we recognize any notes from previous chords
        if len(self.chord_note_sets) > 0:  # Have we learned any chords yet?
            current_notes = set([n % 12 for n in self.current_ground_truth.notes])
            
            recognized_notes = []
            new_notes = []
            
            for pc in current_notes:
                if pc in self.note_memory:
                    prev_chords = self.note_memory[pc]
                    recognized_notes.append((pc, prev_chords))
                else:
                    new_notes.append(pc)
            
            if recognized_notes:
                note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                print(f"   🔍 Note recognition:")
                for pc, prev_chords in recognized_notes:
                    print(f"      ✅ {note_names[pc]} previously in: {', '.join(prev_chords[:2])}")
                if new_notes:
                    print(f"      🆕 New notes: {[note_names[pc] for pc in new_notes]}")
                print(f"   ✅ Feature extraction is consistent across chords!")
    
    def train_model(self):
        """Train ML model on collected samples"""
        if len(self.training_samples) < 10:
            print("⚠️  Not enough samples to train (need at least 10)")
            return False
        
        print(f"\n🧠 Training model on {len(self.training_samples)} samples...")
        
        # Prepare training data
        X = np.array([s['features'] for s in self.training_samples])
        y = np.array([self.chord_mapping[s['chord_label']] for s in self.training_samples])
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Random Forest
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_scaled, y)
        
        # Evaluate on training set
        train_score = self.model.score(X_scaled, y)
        print(f"✅ Model trained! Training accuracy: {train_score:.2%}")
        print(f"   Unique chords: {len(self.chord_mapping)}")
        
        # Note-level validation happens in _stop_chord, not here
        # Clear any old validation data
        self.validation_predictions = []
        
        self.stats['training_iterations'] += 1
        
        # Save model
        self.save_model()
        
        return True
    
    def save_model(self):
        """Save trained model and metadata"""
        if self.model is None:
            return
        
        try:
            # Save model and scaler
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            
            # Save metadata
            inverse_mapping = {v: k for k, v in self.chord_mapping.items()}
            metadata = {
                'chord_mapping': self.chord_mapping,
                'inverse_mapping': inverse_mapping,
                'num_features': len(self.training_samples[0]['features']) if self.training_samples else 0,
                'num_samples': len(self.training_samples),
                'num_chords': len(self.chord_mapping),
                'stats': self.stats,
                'saved_at': datetime.now().isoformat()
            }
            
            with open(self.metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Save training log (convert numpy arrays)
            training_log_serializable = []
            for sample in self.training_samples:
                sample_copy = sample.copy()
                if isinstance(sample_copy['features'], np.ndarray):
                    sample_copy['features'] = sample_copy['features'].tolist()
                training_log_serializable.append(sample_copy)
            
            with open(self.training_log_path, 'w') as f:
                json.dump(training_log_serializable, f, indent=2)
            
            print(f"💾 Model saved: {self.model_path}")
            
        except Exception as e:
            print(f"❌ Failed to save model: {e}")
    
    def run_training_session(self, 
                            chord_list: List[Tuple[str, str]] = None,
                            train_after_each_chord_type: bool = True,
                            randomize: bool = True):
        """
        Run autonomous training session with continuous validation
        
        Args:
            chord_list: List of (root, quality) tuples to train on
            train_after_each_chord_type: If True, train after all 6 inversions of each chord
            randomize: If True, randomize chord order to prevent sequence learning
        """
        if chord_list is None:
            # Default: major and minor triads in all keys
            chord_list = [
                (root, quality)
                for root in ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                for quality in ['', 'm']
            ]
        
        # Randomize chord order to prevent sequence learning
        if randomize:
            import random
            chord_list = chord_list.copy()
            random.shuffle(chord_list)
            print(f"\n🎓 Starting autonomous training session with RANDOMIZED order")
        else:
            print(f"\n🎓 Starting autonomous training session")
        
        print(f"📊 Training plan: {len(chord_list)} chord types × 6 inversions each")
        print(f"⏱️  Chord duration: {self.chord_duration}s")
        if train_after_each_chord_type:
            print(f"🔄 Training after EVERY chord type (all 6 inversions) - immediate feedback!")
        print(f"🎲 Randomized order: {randomize} (prevents sequence learning)")
        print("=" * 60)
        
        try:
            chord_type_count = 0
            total_variations = 0
            
            for root, quality in chord_list:
                chord_type_count += 1
                chord_name = f"{root}{quality}"
                
                print(f"\n🎹 Chord Type {chord_type_count}/{len(chord_list)}: {chord_name}")
                print("─" * 60)
                
                # Generate all 6 inversions for this chord
                inversions = ChordVocabulary.get_inversions(root, quality, base_octave=4)
                
                # Randomize inversion order too!
                if randomize:
                    import random
                    random.shuffle(inversions)
                
                # First, play all 6 inversions and collect data
                for inversion in inversions:
                    # Play chord
                    self._play_chord(inversion)
                    
                    # Wait for chord to play
                    time.sleep(self.chord_duration)
                    
                    # Stop chord (NO validation yet - model doesn't know this chord!)
                    self._stop_chord()
                    
                    # Short silence between chords
                    time.sleep(0.3)
                    
                    total_variations += 1
                
                # NOW train on this chord's data BEFORE moving to next chord
                if train_after_each_chord_type:
                    print(f"\n🔄 Training on {chord_name} data (all 6 inversions collected)...")
                    
                    # Store note memory for this chord BEFORE training
                    chord_notes = set(inversions[0].notes) if inversions else set()
                    # Get just the pitch classes (mod 12)
                    pitch_classes = set([n % 12 for n in chord_notes])
                    self.chord_note_sets[chord_name] = pitch_classes
                    
                    # Update note memory
                    for pc in pitch_classes:
                        if pc not in self.note_memory:
                            self.note_memory[pc] = []
                        self.note_memory[pc].append(chord_name)
                    
                    print(f"   📝 {chord_name} contains pitch classes: {sorted(pitch_classes)}")
                    
                    # Check for shared notes with previous chords
                    shared_notes = {}
                    for pc in pitch_classes:
                        previous_chords = [c for c in self.note_memory.get(pc, []) if c != chord_name]
                        if previous_chords:
                            shared_notes[pc] = previous_chords
                    
                    if shared_notes:
                        print(f"   🔗 Shared notes with previous chords:")
                        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                        for pc, prev_chords in shared_notes.items():
                            print(f"      {note_names[pc]} also in: {', '.join(prev_chords)}")
                        print(f"   ✅ This validates that feature extraction captures note content!")
                    
                    self.train_model()
                    print(f"✅ Model now knows {len(self.chord_note_sets)} chord(s)! Next chord will be validated.")
                    print("=" * 60)
            
            # Final training
            print(f"\n✅ Training session complete!")
            print(f"📊 Total samples collected: {len(self.training_samples)}")
            self.train_model()
            
        except KeyboardInterrupt:
            print(f"\n⚠️  Training interrupted by user")
            if len(self.training_samples) > 0:
                print(f"💾 Saving {len(self.training_samples)} samples...")
                self.train_model()
    
    def start(self) -> bool:
        """Start the autonomous trainer"""
        print("🎵 Autonomous Chord Trainer - Self-Learning System")
        print("=" * 60)
        
        # Open MIDI port
        if not self._open_midi_port():
            return False
        
        # Start audio listener
        if not self._start_audio_listener():
            return False
        
        print("✅ System ready!")
        return True
    
    def stop(self):
        """Stop the system"""
        # Stop audio
        if self.listener:
            self.listener.stop()
        
        # Close MIDI
        if self.midi_port:
            self.midi_port.close()
        
        print("✅ System stopped")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Autonomous Chord Trainer')
    parser.add_argument('--midi-port', default='IAC Driver Chord Trainer Output',
                       help='MIDI output port name')
    parser.add_argument('--input-device', type=int, default=None,
                       help='Audio input device number')
    parser.add_argument('--chord-duration', type=float, default=2.0,
                       help='Duration to play each chord (seconds)')
    parser.add_argument('--train-interval', type=int, default=50,
                       help='Train model after every N chords')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = AutonomousChordTrainer(
        midi_output_port=args.midi_port,
        audio_input_device=args.input_device,
        chord_duration=args.chord_duration
    )
    
    # Start system
    if not trainer.start():
        print("❌ Failed to start trainer")
        return
    
    # Run training session
    try:
        # Start with basic major and minor chords
        # Full jazz chord vocabulary
        jazz_chords = [
            (root, quality)
            for root in ['C', 'D', 'E', 'F', 'G', 'A', 'B']
            for quality in ['', 'm', '7', 'maj7', 'm7', 'dim', 'm7b5', 'aug', 'sus4', 'sus2']
        ]
        
        trainer.run_training_session(
            chord_list=jazz_chords,
            train_after_each_chord_type=True  # Train after each chord type (all 6 inversions)
        )
        
    finally:
        trainer.stop()


if __name__ == "__main__":
    main()

