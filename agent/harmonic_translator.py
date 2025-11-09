"""
Harmonic Translator for Interval-Based MIDI Generation
=======================================================

Translates interval patterns from AudioOracle to absolute MIDI notes
using harmonic context (current chord, key, scale degrees).

Key insight: Gesture tokens have consistent interval patterns (±5, ±7, ±12)
that are musically consonant. Apply these intervals to the current harmonic
context to generate coherent melodies in the correct key.
"""

from typing import List, Dict, Callable


class HarmonicTranslator:
    """
    Translates interval patterns to absolute MIDI notes using harmonic context
    """
    
    def __init__(self, scale_constraint_func: Callable[[int, List[int]], int]):
        """
        Args:
            scale_constraint_func: Reference to PhraseGenerator._apply_scale_constraint
                Function signature: (note: int, scale_degrees: List[int]) -> int
        """
        self.apply_scale_constraint = scale_constraint_func
    
    def translate_intervals_to_midi(self,
                                   intervals: List[int],
                                   harmonic_context: Dict,
                                   voice_type: str,
                                   apply_constraints: bool = True) -> List[int]:
        """
        Convert interval sequence to absolute MIDI notes
        
        Args:
            intervals: List of intervals from IntervalExtractor
            harmonic_context: {current_chord, current_key, scale_degrees}
            voice_type: "melodic" or "bass"
            apply_constraints: Whether to apply scale constraints
            
        Returns:
            List of absolute MIDI notes in correct key/scale
            
        Example:
            intervals = [0, -2, +4, 0]
            harmonic_context = {'current_chord': 'F#m', 'scale_degrees': [0,2,3,5,7,9,10]}
            voice_type = "melodic"
            
            Returns: [66, 66, 64, 68, 68] (F#-F#-E-G#-G# in F# minor)
        """
        if not intervals:
            return []
        
        # Step 1: Get starting note from harmonic context
        start_note = self._get_root_note_for_voice(harmonic_context, voice_type)
        
        # Step 2: Apply intervals to generate MIDI sequence
        midi_sequence = [start_note]
        current_note = start_note
        
        for interval in intervals:
            current_note = current_note + interval
            
            # Constrain to voice range (prevent extreme outliers)
            current_note = self._constrain_to_voice_range(current_note, voice_type)
            
            midi_sequence.append(current_note)
        
        # Step 3: Apply scale constraints if enabled
        if apply_constraints:
            scale_degrees = harmonic_context.get('scale_degrees', [0, 2, 4, 5, 7, 9, 11])  # Default C major
            constrained_sequence = []
            
            for note in midi_sequence:
                constrained_note = self.apply_scale_constraint(note, scale_degrees)
                constrained_sequence.append(constrained_note)
            
            return constrained_sequence
        
        return midi_sequence
    
    def _get_root_note_for_voice(self, harmonic_context: Dict, voice_type: str) -> int:
        """
        Get appropriate starting note based on current chord and voice type
        
        Args:
            harmonic_context: Contains current_chord (e.g., "Dm", "C", "G7", "F#m")
            voice_type: "melodic" or "bass"
            
        Returns:
            MIDI note number for starting pitch
        """
        current_chord = harmonic_context.get('current_chord', 'C')
        
        # Extract root note from chord name (e.g., "Dm" → "D", "F#m" → "F#")
        root_name = self._parse_root_from_chord(current_chord)
        
        # Convert root name to MIDI pitch class (0-11)
        root_pc = self._root_name_to_pitch_class(root_name)
        
        # Place in appropriate octave for voice
        if voice_type == "melodic":
            # Melody: C4-C6 range (MIDI 60-84), prefer middle octave
            root_note = 60 + root_pc  # C4 octave
            if root_note < 60:
                root_note += 12
            if root_note > 72:  # If > C5, use lower octave
                root_note -= 12
        else:
            # Bass: C2-C4 range (MIDI 36-60), prefer C3 octave
            root_note = 48 + root_pc  # C3 octave
            if root_note < 36:
                root_note += 12
            if root_note > 60:
                root_note -= 12
        
        return root_note
    
    @staticmethod
    def _parse_root_from_chord(chord_name: str) -> str:
        """
        Parse root note from chord name
        
        Examples:
            "C" → "C"
            "Dm" → "D"
            "F#m" → "F#"
            "Bb7" → "Bb"
            "C#maj7" → "C#"
        
        Args:
            chord_name: Chord name string
            
        Returns:
            Root note name (e.g., "C", "F#", "Bb")
        """
        if not chord_name or len(chord_name) == 0:
            return "C"
        
        # First character is always root
        root = chord_name[0].upper()
        
        # Check for accidental (# or b)
        if len(chord_name) > 1 and chord_name[1] in ['#', 'b']:
            root += chord_name[1]
        
        return root
    
    @staticmethod
    def _root_name_to_pitch_class(root_name: str) -> int:
        """
        Convert root note name to MIDI pitch class (0-11)
        
        Args:
            root_name: Note name (e.g., "C", "F#", "Bb")
            
        Returns:
            Pitch class (0-11)
        """
        root_pc_map = {
            'C': 0, 'C#': 1, 'Db': 1,
            'D': 2, 'D#': 3, 'Eb': 3,
            'E': 4,
            'F': 5, 'F#': 6, 'Gb': 6,
            'G': 7, 'G#': 8, 'Ab': 8,
            'A': 9, 'A#': 10, 'Bb': 10,
            'B': 11
        }
        
        return root_pc_map.get(root_name, 0)  # Default to C if not found
    
    @staticmethod
    def _constrain_to_voice_range(note: int, voice_type: str) -> int:
        """
        Constrain MIDI note to appropriate voice range
        
        Prevents extreme octave jumps from interval application
        
        Args:
            note: MIDI note number
            voice_type: "melodic" or "bass"
            
        Returns:
            MIDI note constrained to voice range
        """
        if voice_type == "melodic":
            # Melody: C4-C6 (MIDI 60-84)
            min_note, max_note = 60, 84
        else:
            # Bass: C2-C4 (MIDI 36-60)
            min_note, max_note = 36, 60
        
        # Wrap to range using octave shifts
        while note < min_note:
            note += 12
        while note > max_note:
            note -= 12
        
        # Final clamping (safety)
        return max(min_note, min(note, max_note))
