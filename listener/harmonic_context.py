"""
Real-time Harmonic Context for Live Performance
Provides chord, scale, and key awareness to the AI agent

Updated: Now uses CQT-based extraction with harmonic weighting from harmonic_chroma.py
This addresses the key problems:
1. FFT can't distinguish fundamentals from harmonics (C4 vs C5)
2. Harmonic weighting: fundamentals >> overtones
3. Proper confidence calculation (no artificial inflation)
4. Zero-vector guard for silence
"""

import numpy as np
import librosa
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import time

# Import superior CQT-based extraction
from .harmonic_chroma import HarmonicAwareChromaExtractor


@dataclass
class HarmonicContext:
    """Real-time harmonic context for AI decision-making"""
    current_chord: str  # e.g., "Cmaj7", "Dm9"
    key_signature: str  # e.g., "C_major", "A_minor"
    scale_degrees: List[int]  # Scale degrees present (e.g., [0, 2, 4, 5, 7, 9, 11] for major)
    chord_root: int  # MIDI note of chord root (0-11)
    chord_type: str  # "major", "minor", "7", "maj7", etc.
    chroma: np.ndarray  # 12-dimensional chroma vector
    confidence: float  # Confidence in detection (0-1)
    timestamp: float  # When this context was detected
    stability: float  # How stable is this chord (0-1, higher = more stable)


class RealtimeHarmonicDetector:
    """
    Lightweight real-time harmonic detection for live performance
    Optimized for low latency (<10ms processing time)
    """
    
    def __init__(self, window_size: int = 8192, hop_length: int = 2048):
        """
        Args:
            window_size: FFT window size (kept for API compatibility)
            hop_length: Hop length (kept for API compatibility)
        """
        # Validate parameters
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")
        if hop_length <= 0:
            raise ValueError(f"hop_length must be positive, got {hop_length}")
        if hop_length >= window_size:
            raise ValueError(f"hop_length ({hop_length}) must be less than window_size ({window_size})")

        self.window_size = window_size
        self.hop_length = hop_length

        # Note names
        self.note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

        # Initialize CQT-based harmonic extractor (superior to FFT)
        # Uses harmonic weighting to distinguish fundamentals from overtones
        self.harmonic_extractor = HarmonicAwareChromaExtractor(
            n_octaves=6,           # C3 to C8 range
            bins_per_octave=12,    # Semitone resolution
            fmin=130.81,           # C3
            fmax=4186.0            # C8
        )

        # Chord templates (expanded vocabulary)
        self.chord_templates = self._create_chord_templates()

        # Scale templates
        self.scale_templates = self._create_scale_templates()

        # Rolling buffer for temporal smoothing
        self.chroma_history = deque(maxlen=5)  # Keep last 5 chroma vectors
        self.chord_history = deque(maxlen=10)  # Keep last 10 detected chords

        # Current state
        self.current_context: Optional[HarmonicContext] = None
        self.last_update_time = 0.0

        # Temporal hysteresis (prevent rapid chord switching)
        self.min_chord_duration = 0.2  # seconds
        self.last_chord_change = 0.0
        
    def _create_chord_templates(self) -> Dict[str, Dict]:
        """Create chord templates for detection"""
        templates = {}
        
        for root in range(12):
            root_name = self.note_names[root]
            
            # Triads
            templates[f"{root_name}"] = {
                'notes': [root, (root + 4) % 12, (root + 7) % 12],
                'type': 'major',
                'weight': 1.0
            }
            templates[f"{root_name}m"] = {
                'notes': [root, (root + 3) % 12, (root + 7) % 12],
                'type': 'minor',
                'weight': 1.0
            }
            
            # Seventh chords
            templates[f"{root_name}7"] = {
                'notes': [root, (root + 4) % 12, (root + 7) % 12, (root + 10) % 12],
                'type': 'dom7',
                'weight': 0.9
            }
            templates[f"{root_name}maj7"] = {
                'notes': [root, (root + 4) % 12, (root + 7) % 12, (root + 11) % 12],
                'type': 'maj7',
                'weight': 0.9
            }
            templates[f"{root_name}m7"] = {
                'notes': [root, (root + 3) % 12, (root + 7) % 12, (root + 10) % 12],
                'type': 'min7',
                'weight': 0.9
            }
            
            # Extended chords (9ths)
            templates[f"{root_name}9"] = {
                'notes': [root, (root + 4) % 12, (root + 7) % 12, (root + 10) % 12, (root + 2) % 12],
                'type': 'dom9',
                'weight': 0.8
            }
            templates[f"{root_name}maj9"] = {
                'notes': [root, (root + 4) % 12, (root + 7) % 12, (root + 11) % 12, (root + 2) % 12],
                'type': 'maj9',
                'weight': 0.8
            }
            templates[f"{root_name}m9"] = {
                'notes': [root, (root + 3) % 12, (root + 7) % 12, (root + 10) % 12, (root + 2) % 12],
                'type': 'min9',
                'weight': 0.8
            }
            
            # Sus chords
            templates[f"{root_name}sus4"] = {
                'notes': [root, (root + 5) % 12, (root + 7) % 12],
                'type': 'sus4',
                'weight': 0.7
            }
            templates[f"{root_name}sus2"] = {
                'notes': [root, (root + 2) % 12, (root + 7) % 12],
                'type': 'sus2',
                'weight': 0.7
            }
            
            # Diminished/Augmented
            templates[f"{root_name}dim"] = {
                'notes': [root, (root + 3) % 12, (root + 6) % 12],
                'type': 'dim',
                'weight': 0.6
            }
            templates[f"{root_name}aug"] = {
                'notes': [root, (root + 4) % 12, (root + 8) % 12],
                'type': 'aug',
                'weight': 0.6
            }
        
        return templates
    
    def _create_scale_templates(self) -> Dict[str, List[int]]:
        """Create scale templates for key detection"""
        return {
            'major': [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1],  # W-W-H-W-W-W-H
            'minor': [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0],  # Natural minor
            'dorian': [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0],
            'mixolydian': [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0],
            'harmonic_minor': [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1],
        }
    
    def extract_chroma_from_audio(self, audio_buffer: np.ndarray, sr: int = 44100) -> np.ndarray:
        """
        Extract chroma features from audio buffer using CQT with harmonic weighting.

        Key improvements over simple FFT:
        1. CQT provides logarithmic frequency spacing (like musical notes)
        2. Harmonic weighting distinguishes fundamentals from overtones
        3. Temporal correlation emphasizes sustained chord tones

        Args:
            audio_buffer: Audio samples (mono)
            sr: Sample rate

        Returns:
            12-dimensional chroma vector (normalized), or zeros for silence
        """
        # Use CQT-based extraction with harmonic weighting
        # live_mode=True applies noise gate and preprocessing for microphone input
        chroma = self.harmonic_extractor.extract(
            audio_buffer,
            sr=sr,
            use_temporal=True,   # Use temporal correlation for stability
            live_mode=True       # Apply noise gate and preprocessing
        )

        return chroma
    
    def detect_chord_from_chroma(self, chroma: np.ndarray, use_history: bool = True) -> Tuple[str, float, str]:
        """
        Detect chord from chroma vector

        Args:
            chroma: 12-dimensional chroma vector
            use_history: Whether to use temporal smoothing

        Returns:
            (chord_name, confidence, chord_type)
            Returns ("N/A", 0.0, "unknown") for silence/invalid input
        """
        # ZERO-VECTOR GUARD: Check for silence or invalid input
        chroma_energy = np.sum(chroma)
        if chroma_energy < 0.05:
            # Too quiet to detect - don't guess
            return ("N/A", 0.0, "unknown")

        # Add to history for temporal smoothing
        if use_history:
            self.chroma_history.append(chroma)
            # Average recent chroma for stability
            chroma_avg = np.mean(list(self.chroma_history), axis=0)
        else:
            chroma_avg = chroma

        best_chord = "N/A"
        best_score = 0.0
        best_type = "unknown"

        # Match against templates
        for chord_name, template_info in self.chord_templates.items():
            # Create binary template
            template = np.zeros(12)
            for note in template_info['notes']:
                template[note] = 1.0

            # Normalize template
            template = template / np.sum(template)

            # Calculate similarity (dot product of normalized vectors)
            similarity = np.dot(chroma_avg, template)

            # Weight by template importance (triads > 7ths > 9ths > sus > dim/aug)
            weighted_score = similarity * template_info['weight']

            if weighted_score > best_score:
                best_score = weighted_score
                best_chord = chord_name
                best_type = template_info['type']

        # HONEST CONFIDENCE: No artificial inflation
        # Raw similarity is already 0-1 (dot product of normalized vectors)
        # Only report high confidence if we have a strong match
        confidence = best_score

        # Minimum threshold: reject weak matches
        if confidence < 0.15:
            return ("N/A", confidence, "unknown")

        return best_chord, confidence, best_type
    
    def detect_key_from_chroma(self, chroma: np.ndarray) -> Tuple[str, float]:
        """
        Detect key signature from chroma
        
        Returns:
            (key_name, confidence)
        """
        best_key = "C_major"
        best_score = 0.0
        
        for root in range(12):
            for scale_name, scale_template in self.scale_templates.items():
                # Rotate template to this root
                rotated_template = np.roll(scale_template, root)
                
                # Calculate correlation
                correlation = np.corrcoef(chroma, rotated_template)[0, 1]
                
                if correlation > best_score:
                    best_score = correlation
                    best_key = f"{self.note_names[root]}_{scale_name}"
        
        confidence = max(0.0, min(1.0, (best_score + 1.0) / 2.0))  # Map -1,1 to 0,1
        return best_key, confidence
    
    def get_scale_degrees(self, key_signature: str) -> List[int]:
        """Get scale degrees for a key signature"""
        parts = key_signature.split('_')
        if len(parts) != 2:
            return [0, 2, 4, 5, 7, 9, 11]  # Default to C major
        
        root_name, scale_type = parts
        
        # Get root offset
        try:
            root = self.note_names.index(root_name)
        except ValueError:
            root = 0
        
        # Get scale pattern
        scale_pattern = self.scale_templates.get(scale_type, self.scale_templates['major'])
        
        # Extract scale degrees
        degrees = []
        for i, present in enumerate(scale_pattern):
            if present:
                degrees.append((root + i) % 12)
        
        return degrees
    
    def update_from_audio(self, audio_buffer: np.ndarray, sr: int = 44100) -> HarmonicContext:
        """
        Main update method: analyze audio and return harmonic context
        
        Args:
            audio_buffer: Audio samples
            sr: Sample rate
            
        Returns:
            HarmonicContext with current harmonic information
        """
        current_time = time.time()
        
        # Extract chroma
        chroma = self.extract_chroma_from_audio(audio_buffer, sr)
        
        # Detect chord
        chord_name, chord_conf, chord_type = self.detect_chord_from_chroma(chroma)
        
        # Detect key (less frequently - only if chord changed significantly)
        if self.current_context is None or current_time - self.last_update_time > 2.0:
            key_name, key_conf = self.detect_key_from_chroma(chroma)
            scale_degrees = self.get_scale_degrees(key_name)
        else:
            key_name = self.current_context.key_signature
            scale_degrees = self.current_context.scale_degrees
        
        # Calculate stability (how consistent is the chord over time?)
        stability = self._calculate_stability(chord_name)
        
        # Extract chord root
        chord_root = self._extract_chord_root(chord_name)
        
        # Create context
        context = HarmonicContext(
            current_chord=chord_name,
            key_signature=key_name,
            scale_degrees=scale_degrees,
            chord_root=chord_root,
            chord_type=chord_type,
            chroma=chroma,
            confidence=chord_conf,
            timestamp=current_time,
            stability=stability
        )
        
        # Update state
        self.chord_history.append(chord_name)
        self.current_context = context
        self.last_update_time = current_time
        
        return context
    
    def _calculate_stability(self, current_chord: str) -> float:
        """Calculate chord stability based on history"""
        if len(self.chord_history) < 3:
            return 0.5
        
        # Count how many recent chords match current
        recent = list(self.chord_history)[-5:]
        matches = sum(1 for c in recent if c == current_chord)
        
        stability = matches / len(recent)
        return stability
    
    def _extract_chord_root(self, chord_name: str) -> int:
        """Extract root note from chord name"""
        # Handle chord names like "C#m7", "Gmaj7", etc.
        for i, note in enumerate(self.note_names):
            if chord_name.startswith(note):
                return i
        return 0  # Default to C
    
    def get_chord_notes(self, chord_name: str) -> List[int]:
        """Get MIDI note numbers for a chord"""
        template = self.chord_templates.get(chord_name)
        if template:
            return template['notes']
        return [0, 4, 7]  # Default to C major triad
    
    def get_related_chords(self, chord_name: str) -> List[str]:
        """Get musically related chords for contrast/variation"""
        root = self._extract_chord_root(chord_name)
        root_name = self.note_names[root]
        
        # Common chord substitutions and related chords
        related = []
        
        # Parallel major/minor
        if 'm' in chord_name and chord_name != f"{root_name}maj7":
            related.append(f"{root_name}")  # Major version
        else:
            related.append(f"{root_name}m")  # Minor version
        
        # Fifth above and below
        fifth_above = self.note_names[(root + 7) % 12]
        fifth_below = self.note_names[(root - 7) % 12]
        related.append(f"{fifth_above}")
        related.append(f"{fifth_below}m")
        
        # Relative major/minor (third)
        if 'm' in chord_name:
            # Minor -> relative major (up minor third)
            rel_major = self.note_names[(root + 3) % 12]
            related.append(f"{rel_major}")
        else:
            # Major -> relative minor (down minor third)
            rel_minor = self.note_names[(root - 3) % 12]
            related.append(f"{rel_minor}m")
        
        return related[:3]  # Return top 3

