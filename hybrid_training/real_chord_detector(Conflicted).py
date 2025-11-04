"""
Real Chord Detection System
Analyzes actual audio content to detect real chord progressions
"""

import numpy as np
import librosa
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import torch


@dataclass
class ChordAnalysis:
    """Container for real chord analysis results"""
    chord_progression: List[str]
    key_signature: str
    chord_changes: List[float]  # Timestamps of chord changes
    chord_durations: List[float]  # Duration of each chord
    confidence_scores: List[float]  # Confidence for each chord
    harmonic_rhythm: float  # Average time between chord changes


class RealChordDetector:
    """
    Real chord detection system that analyzes actual audio content
    """
    
    def __init__(self):
        # Note names
        self.note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        # Chord templates for major and minor chords
        self.chord_templates = self._create_chord_templates()
        
        # Common chord progressions
        self.common_progressions = {
            'major': ['I', 'V', 'vi', 'IV'],  # Common major progression
            'minor': ['i', 'V', 'VI', 'iv'],  # Common minor progression
        }
    
    def _create_chord_templates(self) -> Dict[str, List[int]]:
        """Create chord templates for major and minor chords"""
        templates = {}
        
        # Major chords (root, major third, perfect fifth)
        for root in range(12):
            root_name = self.note_names[root]
            major_third = (root + 4) % 12
            perfect_fifth = (root + 7) % 12
            templates[f"{root_name}_major"] = [root, major_third, perfect_fifth]
        
        # Minor chords (root, minor third, perfect fifth)
        for root in range(12):
            root_name = self.note_names[root]
            minor_third = (root + 3) % 12
            perfect_fifth = (root + 7) % 12
            templates[f"{root_name}_minor"] = [root, minor_third, perfect_fifth]
        
        # Add some common extended chords
        for root in range(12):
            root_name = self.note_names[root]
            # Dominant 7th (root, major third, perfect fifth, minor seventh)
            templates[f"{root_name}_7"] = [root, (root + 4) % 12, (root + 7) % 12, (root + 10) % 12]
            # Minor 7th (root, minor third, perfect fifth, minor seventh)
            templates[f"{root_name}_m7"] = [root, (root + 3) % 12, (root + 7) % 12, (root + 10) % 12]
        
        return templates
    
    def analyze_audio_file(self, audio_file_path: str, hop_length: int = 512) -> ChordAnalysis:
        """
        Analyze audio file for real chord progressions
        
        Args:
            audio_file_path: Path to audio file
            hop_length: Hop length for analysis
            
        Returns:
            ChordAnalysis with real chord progression
        """
        print(f"🎵 Analyzing real chord progression in {audio_file_path}...")
        
        # Load audio
        y, sr = librosa.load(audio_file_path, sr=44100)
        
        # Extract chroma features (pitch class profiles)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
        
        # Extract harmonic and percussive components
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        
        # Get chroma for harmonic component (more accurate for chords)
        chroma_harmonic = librosa.feature.chroma_stft(y=y_harmonic, sr=sr, hop_length=hop_length)
        
        # Detect chord changes using onset detection
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop_length, units='frames')
        
        # Analyze chords at onset points and regular intervals
        chord_analysis_points = self._get_chord_analysis_points(chroma_harmonic, onset_frames)
        
        # Detect chords at each analysis point
        chord_progression = []
        chord_changes = []
        chord_durations = []
        confidence_scores = []
        
        for i, frame_idx in enumerate(chord_analysis_points):
            # Get chroma vector for this frame
            chroma_vector = chroma_harmonic[:, frame_idx]
            
            # Detect chord
            chord, confidence = self._detect_chord_from_chroma(chroma_vector)
            chord_progression.append(chord)
            confidence_scores.append(confidence)
            
            # Calculate timing
            time_seconds = frame_idx * hop_length / sr
            chord_changes.append(time_seconds)
            
            # Calculate duration
            if i > 0:
                duration = time_seconds - chord_changes[i-1]
                chord_durations.append(duration)
            else:
                chord_durations.append(0.0)
        
        # Estimate key signature
        key_signature = self._estimate_key_signature(chord_progression)
        
        # Calculate harmonic rhythm
        harmonic_rhythm = np.mean(chord_durations[1:]) if len(chord_durations) > 1 else 0.0
        
        print(f"✅ Real chord analysis complete: {len(chord_progression)} chords detected")
        print(f"🎵 Key: {key_signature}")
        print(f"🎼 Progression: {chord_progression[:10]}...")
        
        return ChordAnalysis(
            chord_progression=chord_progression,
            key_signature=key_signature,
            chord_changes=chord_changes,
            chord_durations=chord_durations,
            confidence_scores=confidence_scores,
            harmonic_rhythm=harmonic_rhythm
        )
    
    def _get_chord_analysis_points(self, chroma: np.ndarray, onset_frames: np.ndarray) -> List[int]:
        """Get frame indices for chord analysis"""
        analysis_points = []
        
        # Add onset points
        analysis_points.extend(onset_frames.tolist())
        
        # Add regular intervals (every 2 seconds)
        hop_length = 512
        sr = 44100
        frames_per_2sec = int(2.0 * sr / hop_length)
        
        for i in range(frames_per_2sec, chroma.shape[1], frames_per_2sec):
            if i not in analysis_points:
                analysis_points.append(i)
        
        # Sort and remove duplicates
        analysis_points = sorted(list(set(analysis_points)))
        
        # Limit to reasonable number of analysis points
        max_points = min(100, len(analysis_points))
        step = len(analysis_points) // max_points if len(analysis_points) > max_points else 1
        
        return analysis_points[::step]
    
    def _detect_chord_from_chroma(self, chroma_vector: np.ndarray) -> Tuple[str, float]:
        """
        Detect chord from chroma vector
        
        Args:
            chroma_vector: 12-dimensional chroma vector
            
        Returns:
            Tuple of (chord_name, confidence)
        """
        # Normalize chroma vector
        chroma_norm = chroma_vector / (np.sum(chroma_vector) + 1e-8)
        
        # Compare with chord templates
        best_chord = "C_major"
        best_score = 0.0
        
        for chord_name, template in self.chord_templates.items():
            # Create template vector
            template_vector = np.zeros(12)
            for pitch_class in template:
                template_vector[pitch_class] = 1.0
            
            # Calculate similarity (cosine similarity)
            similarity = np.dot(chroma_norm, template_vector) / (
                np.linalg.norm(chroma_norm) * np.linalg.norm(template_vector) + 1e-8
            )
            
            if similarity > best_score:
                best_score = similarity
                best_chord = chord_name
        
        # Convert to simple chord notation
        simple_chord = self._simplify_chord_name(best_chord)
        
        return simple_chord, best_score
    
    def _simplify_chord_name(self, chord_name: str) -> str:
        """Simplify chord name to basic notation"""
        if '_major' in chord_name:
            return chord_name.replace('_major', '')
        elif '_minor' in chord_name:
            return chord_name.replace('_minor', 'm')
        elif '_7' in chord_name:
            return chord_name.replace('_7', '7')
        elif '_m7' in chord_name:
            return chord_name.replace('_m7', 'm7')
        else:
            return chord_name
    
    def _estimate_key_signature(self, chord_progression: List[str]) -> str:
        """Estimate key signature from chord progression"""
        # Count chord occurrences
        chord_counts = {}
        for chord in chord_progression:
            # Extract root note
            root = chord[0] if chord[0] != 'm' else chord[1] if len(chord) > 1 else 'C'
            chord_counts[root] = chord_counts.get(root, 0) + 1
        
        # Find most common root
        if chord_counts:
            most_common_root = max(chord_counts.items(), key=lambda x: x[1])[0]
            
            # Determine if major or minor based on chord types
            major_chords = sum(1 for chord in chord_progression if 'm' not in chord and '7' not in chord)
            minor_chords = sum(1 for chord in chord_progression if 'm' in chord)
            
            if minor_chords > major_chords:
                return f"{most_common_root}_minor"
            else:
                return f"{most_common_root}_major"
        
        return "C_major"  # Default
    
    def analyze_events_for_chords(self, events: List[Dict]) -> ChordAnalysis:
        """
        Analyze events for chord progression
        
        Args:
            events: List of event dictionaries with musical features
            
        Returns:
            ChordAnalysis with chord progression
        """
        if not events:
            return ChordAnalysis(
                chord_progression=[],
                key_signature="C_major",
                chord_changes=[],
                chord_durations=[],
                confidence_scores=[],
                harmonic_rhythm=0.0
            )
        
        # Extract MIDI notes and timing from events
        midi_notes = []
        timestamps = []
        
        for event in events:
            if 'midi' in event and 't' in event:
                midi_notes.append(event['midi'])
                timestamps.append(event['t'])
        
        if not midi_notes:
            return ChordAnalysis(
                chord_progression=[],
                key_signature="C_major",
                chord_changes=[],
                chord_durations=[],
                confidence_scores=[],
                harmonic_rhythm=0.0
            )
        
        # Group notes into chords based on timing
        chord_progression = []
        chord_changes = []
        chord_durations = []
        confidence_scores = []
        
        # Simple chord detection: group notes that occur within 0.5 seconds
        current_chord_notes = []
        current_chord_start = timestamps[0]
        
        for i, (midi, timestamp) in enumerate(zip(midi_notes, timestamps)):
            if timestamp - current_chord_start > 0.5:  # New chord
                # Analyze current chord
                if current_chord_notes:
                    chord, confidence = self._detect_chord_from_midi_notes(current_chord_notes)
                    chord_progression.append(chord)
                    confidence_scores.append(confidence)
                    chord_changes.append(current_chord_start)
                    
                    if len(chord_changes) > 1:
                        duration = current_chord_start - chord_changes[-2]
                        chord_durations.append(duration)
                    else:
                        chord_durations.append(0.0)
                
                # Start new chord
                current_chord_notes = [midi]
                current_chord_start = timestamp
            else:
                current_chord_notes.append(midi)
        
        # Process final chord
        if current_chord_notes:
            chord, confidence = self._detect_chord_from_midi_notes(current_chord_notes)
            chord_progression.append(chord)
            confidence_scores.append(confidence)
            chord_changes.append(current_chord_start)
            
            if len(chord_changes) > 1:
                duration = current_chord_start - chord_changes[-2]
                chord_durations.append(duration)
            else:
                chord_durations.append(0.0)
        
        # Estimate key signature
        key_signature = self._estimate_key_signature(chord_progression)
        
        # Calculate harmonic rhythm
        harmonic_rhythm = np.mean(chord_durations[1:]) if len(chord_durations) > 1 else 0.0
        
        return ChordAnalysis(
            chord_progression=chord_progression,
            key_signature=key_signature,
            chord_changes=chord_changes,
            chord_durations=chord_durations,
            confidence_scores=confidence_scores,
            harmonic_rhythm=harmonic_rhythm
        )
    
    def _detect_chord_from_midi_notes(self, midi_notes: List[int]) -> Tuple[str, float]:
        """
        Detect chord from MIDI note numbers
        
        Args:
            midi_notes: List of MIDI note numbers
            
        Returns:
            Tuple of (chord_name, confidence)
        """
        if not midi_notes:
            return "C", 0.0
        
        # Convert MIDI notes to pitch classes
        pitch_classes = [note % 12 for note in midi_notes]
        
        # Count pitch class occurrences
        pitch_class_counts = {}
        for pc in pitch_classes:
            pitch_class_counts[pc] = pitch_class_counts.get(pc, 0) + 1
        
        # Find the most prominent pitch classes
        sorted_pcs = sorted(pitch_class_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Try to match with chord templates
        best_chord = "C"
        best_score = 0.0
        
        for chord_name, template in self.chord_templates.items():
            # Calculate how many template notes are present
            matches = 0
            for pc in template:
                if pc in pitch_class_counts:
                    matches += pitch_class_counts[pc]
            
            # Calculate score (normalized by template length)
            score = matches / len(template)
            
            if score > best_score:
                best_score = score
                best_chord = self._simplify_chord_name(chord_name)
        
        return best_chord, best_score
