"""
Real Chord Detection System
Analyzes actual audio content to detect real chord progressions
Enhanced with voice leading analysis
"""

import numpy as np
import librosa
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import torch
from .voice_leading_analyzer import VoiceLeadingAnalyzer, BassLineAnalyzer, VoiceLeadingAnalysis

# Import custom chroma extraction (works around NumPy 2.2 / numba issues)
try:
    from .chroma_utils import chroma_stft_fallback
    _USE_CHROMA_FALLBACK = True
except ImportError:
    _USE_CHROMA_FALLBACK = False


@dataclass
class ChordAnalysis:
    """Container for real chord analysis results with voice leading"""
    chord_progression: List[str]
    key_signature: str
    chord_changes: List[float]  # Timestamps of chord changes
    chord_durations: List[float]  # Duration of each chord
    confidence_scores: List[float]  # Confidence for each chord
    harmonic_rhythm: float  # Average time between chord changes
    voice_leading: Optional[VoiceLeadingAnalysis] = None  # Voice leading analysis
    bass_line: Optional[Dict] = None  # Bass line analysis


class RealChordDetector:
    """
    Real chord detection system that analyzes actual audio content
    """
    
    def __init__(self, enable_voice_leading: bool = True):
        # Note names
        self.note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        # Chord templates for major and minor chords
        self.chord_templates = self._create_chord_templates()
        
        # Common chord progressions
        self.common_progressions = {
            'major': ['I', 'V', 'vi', 'IV'],  # Common major progression
            'minor': ['i', 'V', 'VI', 'iv'],  # Common minor progression
        }
        
        # Voice leading analysis
        self.enable_voice_leading = enable_voice_leading
        if enable_voice_leading:
            self.voice_analyzer = VoiceLeadingAnalyzer(num_voices=4)
            self.bass_analyzer = BassLineAnalyzer()
        else:
            self.voice_analyzer = None
            self.bass_analyzer = None
    
    def _create_chord_templates(self) -> Dict[str, List[int]]:
        """
        Create chord templates for jazz and contemporary music
        Enhanced with 17 chord types (204 total chords: 17 types × 12 roots)
        """
        templates = {}
        
        for root in range(12):
            root_name = self.note_names[root]
            
            # === TRIADS (3 notes) ===
            
            # Major (root, major third, perfect fifth)
            templates[f"{root_name}_major"] = [root, (root + 4) % 12, (root + 7) % 12]
            
            # Minor (root, minor third, perfect fifth)
            templates[f"{root_name}_minor"] = [root, (root + 3) % 12, (root + 7) % 12]
            
            # Suspended 2nd (root, major 2nd, perfect fifth)
            templates[f"{root_name}_sus2"] = [root, (root + 2) % 12, (root + 7) % 12]
            
            # Suspended 4th (root, perfect 4th, perfect fifth)
            templates[f"{root_name}_sus4"] = [root, (root + 5) % 12, (root + 7) % 12]
            
            # Augmented (root, major third, augmented fifth)
            templates[f"{root_name}_aug"] = [root, (root + 4) % 12, (root + 8) % 12]
            
            # Diminished (root, minor third, diminished fifth)
            templates[f"{root_name}_dim"] = [root, (root + 3) % 12, (root + 6) % 12]
            
            # === SEVENTH CHORDS (4 notes) ===
            
            # Dominant 7th (root, major third, perfect fifth, minor seventh)
            templates[f"{root_name}_7"] = [root, (root + 4) % 12, (root + 7) % 12, (root + 10) % 12]
            
            # Major 7th (root, major third, perfect fifth, major seventh)
            templates[f"{root_name}_maj7"] = [root, (root + 4) % 12, (root + 7) % 12, (root + 11) % 12]
            
            # Minor 7th (root, minor third, perfect fifth, minor seventh)
            templates[f"{root_name}_m7"] = [root, (root + 3) % 12, (root + 7) % 12, (root + 10) % 12]
            
            # Minor 7th flat 5 / Half-diminished (root, minor third, diminished fifth, minor seventh)
            templates[f"{root_name}_m7b5"] = [root, (root + 3) % 12, (root + 6) % 12, (root + 10) % 12]
            
            # Diminished 7th (root, minor third, diminished fifth, diminished seventh)
            templates[f"{root_name}_dim7"] = [root, (root + 3) % 12, (root + 6) % 12, (root + 9) % 12]
            
            # === SIXTH CHORDS (4 notes) ===
            
            # Major 6th (root, major third, perfect fifth, major sixth)
            templates[f"{root_name}_6"] = [root, (root + 4) % 12, (root + 7) % 12, (root + 9) % 12]
            
            # Minor 6th (root, minor third, perfect fifth, major sixth)
            templates[f"{root_name}_m6"] = [root, (root + 3) % 12, (root + 7) % 12, (root + 9) % 12]
            
            # === EXTENDED CHORDS (5+ notes) ===
            
            # Dominant 9th (root, major third, perfect fifth, minor seventh, major ninth)
            templates[f"{root_name}_9"] = [root, (root + 4) % 12, (root + 7) % 12, (root + 10) % 12, (root + 2) % 12]
            
            # Major 9th (root, major third, perfect fifth, major seventh, major ninth)
            templates[f"{root_name}_maj9"] = [root, (root + 4) % 12, (root + 7) % 12, (root + 11) % 12, (root + 2) % 12]
            
            # Minor 9th (root, minor third, perfect fifth, minor seventh, major ninth)
            templates[f"{root_name}_m9"] = [root, (root + 3) % 12, (root + 7) % 12, (root + 10) % 12, (root + 2) % 12]
            
            # Add9 (root, major third, perfect fifth, major ninth) - no 7th!
            templates[f"{root_name}_add9"] = [root, (root + 4) % 12, (root + 7) % 12, (root + 2) % 12]
            
            # === ALTERED DOMINANTS (jazz essentials) ===
            
            # Dominant 7th sharp 9 / Hendrix chord (root, major third, perfect fifth, minor seventh, sharp ninth)
            templates[f"{root_name}_7#9"] = [root, (root + 4) % 12, (root + 7) % 12, (root + 10) % 12, (root + 3) % 12]
        
        print(f"✅ Created {len(templates)} chord templates across 18 chord types")
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
        # Use fallback if librosa chroma is broken (NumPy 2.2 / numba issue)
        if _USE_CHROMA_FALLBACK:
            print("   Using custom chroma extraction (librosa compatibility mode)")
            chroma = chroma_stft_fallback(y=y, sr=sr, hop_length=hop_length)
        else:
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
        
        # Extract harmonic and percussive components
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        
        # Get chroma for harmonic component (more accurate for chords)
        if _USE_CHROMA_FALLBACK:
            chroma_harmonic = chroma_stft_fallback(y=y_harmonic, sr=sr, hop_length=hop_length)
        else:
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
        """
        Simplify chord name to standard jazz notation
        Handles all 18 chord types
        """
        # Handle extended and altered chords first (more specific patterns)
        if '_m7b5' in chord_name:
            return chord_name.replace('_m7b5', 'ø7')  # Half-diminished symbol
        elif '_dim7' in chord_name:
            return chord_name.replace('_dim7', 'dim7')
        elif '_maj9' in chord_name:
            return chord_name.replace('_maj9', 'maj9')
        elif '_maj7' in chord_name:
            return chord_name.replace('_maj7', 'maj7')
        elif '_7#9' in chord_name:
            return chord_name.replace('_7#9', '7#9')
        elif '_m9' in chord_name:
            return chord_name.replace('_m9', 'm9')
        elif '_m7' in chord_name:
            return chord_name.replace('_m7', 'm7')
        elif '_m6' in chord_name:
            return chord_name.replace('_m6', 'm6')
        elif '_9' in chord_name:
            return chord_name.replace('_9', '9')
        elif '_7' in chord_name:
            return chord_name.replace('_7', '7')
        elif '_6' in chord_name:
            return chord_name.replace('_6', '6')
        elif '_add9' in chord_name:
            return chord_name.replace('_add9', 'add9')
        elif '_sus2' in chord_name:
            return chord_name.replace('_sus2', 'sus2')
        elif '_sus4' in chord_name:
            return chord_name.replace('_sus4', 'sus4')
        elif '_aug' in chord_name:
            return chord_name.replace('_aug', 'aug')
        elif '_dim' in chord_name:
            return chord_name.replace('_dim', 'dim')
        elif '_minor' in chord_name:
            return chord_name.replace('_minor', 'm')
        elif '_major' in chord_name:
            return chord_name.replace('_major', '')
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
                harmonic_rhythm=0.0,
                voice_leading=None,
                bass_line=None
            )
        
        print(f"🎵 Analyzing chords from {len(events)} events...")
        
        # Detect chords from chroma features in events
        chord_progression = []
        chord_changes = []
        chord_durations = []
        confidence_scores = []
        
        # Group events into chord analysis windows (every 4-8 events ≈ 1 chord)
        window_size = 8
        
        for i in range(0, len(events), window_size):
            window_events = events[i:i+window_size]
            if not window_events:
                continue
            
            # Extract chroma from features array
            # Features array structure: [chroma(12), spectral features, temporal features]
            chroma_sum = np.zeros(12)
            window_start_time = window_events[0].get('t', 0.0)
            
            for event in window_events:
                if 'features' in event:
                    features = event['features']
                    # First 12 dimensions are chroma
                    if len(features) >= 12:
                        chroma = np.array(features[:12])
                        chroma_sum += chroma
                elif 'midi' in event:
                    # Fallback: create chroma from MIDI note
                    midi = event['midi']
                    pitch_class = midi % 12
                    chroma_sum[pitch_class] += 1.0
                elif 'f0' in event and event['f0'] > 0:
                    # Fallback: create chroma from f0
                    import math
                    try:
                        midi = int(round(69 + 12 * math.log2(event['f0'] / 440.0)))
                        pitch_class = midi % 12
                        chroma_sum[pitch_class] += 1.0
                    except (ValueError, ZeroDivisionError):
                        pass
            
            # Detect chord from accumulated chroma
            if np.sum(chroma_sum) > 0:
                chord, confidence = self._detect_chord_from_chroma(chroma_sum)
                chord_progression.append(chord)
                confidence_scores.append(confidence)
                chord_changes.append(window_start_time)
                
                if len(chord_changes) > 1:
                    duration = window_start_time - chord_changes[-2]
                    chord_durations.append(duration)
        
        if not chord_progression:
            print("⚠️  No chords detected from chroma features")
            return ChordAnalysis(
                chord_progression=[],
                key_signature="C_major",
                chord_changes=[],
                chord_durations=[],
                confidence_scores=[],
                harmonic_rhythm=0.0,
                voice_leading=None,
                bass_line=None
            )
        
        # Estimate key signature
        key_signature = self._estimate_key_signature(chord_progression)
        
        # Calculate harmonic rhythm
        harmonic_rhythm = np.mean(chord_durations) if chord_durations else 0.0
        
        print(f"✅ Detected {len(chord_progression)} chords")
        print(f"   Key: {key_signature}")
        print(f"   First 10 chords: {chord_progression[:10]}")
        print(f"   Average confidence: {np.mean(confidence_scores):.2f}")
        
        # Perform voice leading analysis if enabled
        voice_leading_result = None
        bass_line_result = None
        
        if self.enable_voice_leading and self.voice_analyzer and len(chord_progression) > 1:
            # Get chroma vectors for voice leading analysis
            chroma_vectors = []
            for i in range(0, len(events), 8):  # Same window size as chord detection
                window_events = events[i:i+8]
                chroma_sum = np.zeros(12)
                
                for event in window_events:
                    if 'features' in event and len(event['features']) >= 12:
                        chroma = np.array(event['features'][:12])
                        chroma_sum += chroma
                
                if np.sum(chroma_sum) > 0:
                    chroma_vectors.append(chroma_sum)
            
            # Analyze voice leading
            voice_leading_result = self.voice_analyzer.analyze_chord_progression(
                chord_progression, chroma_vectors[:len(chord_progression)], chord_changes
            )
            
            # Analyze bass line
            voicings = [self.voice_analyzer._assign_voices_from_chroma(
                chord, chroma, timestamp
            ) for chord, chroma, timestamp in zip(chord_progression, chroma_vectors[:len(chord_progression)], chord_changes)]
            
            bass_line_result = self.bass_analyzer.analyze_bass_line(voicings, chord_progression)
        
        return ChordAnalysis(
            chord_progression=chord_progression,
            key_signature=key_signature,
            chord_changes=chord_changes,
            chord_durations=chord_durations,
            confidence_scores=confidence_scores,
            harmonic_rhythm=harmonic_rhythm,
            voice_leading=voice_leading_result,
            bass_line=bass_line_result
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
