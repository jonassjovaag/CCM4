"""
Interval Extractor for Harmonic Translation
============================================

Extracts melodic intervals from AudioOracle retrieved frames
instead of absolute MIDI notes.

Key insight: Gesture tokens have consistent INTERVAL patterns (2.23x improvement)
even when absolute MIDI notes are scattered. Extract intervals to preserve
learned melodic gestures while allowing harmonic translation.
"""

import math
from typing import List, Dict, Any, Optional


class IntervalExtractor:
    """
    Extracts melodic intervals from AudioOracle retrieved frames
    """
    
    def extract_intervals(self, frame_ids: List[int], 
                         audio_frames: Dict) -> List[int]:
        """
        Extract intervals between consecutive frames
        
        Args:
            frame_ids: List of frame IDs from AudioOracle
            audio_frames: AudioOracle audio_frames dict
            
        Returns:
            List of intervals (semitones) between consecutive MIDI notes
            
        Example:
            Frames have MIDI [67, 67, 65, 69, 69]
            Returns intervals [0, -2, +4, 0]
        """
        midi_sequence = []
        
        for frame_id in frame_ids:
            frame = audio_frames.get(frame_id, {})
            
            # Handle both dict and AudioFrame object
            if hasattr(frame, 'audio_data'):
                audio_data = frame.audio_data
            else:
                audio_data = frame.get('audio_data', frame)
            
            # Extract MIDI using existing priority order
            midi = self._extract_midi_from_frame(audio_data)
            
            if midi is not None and midi > 0:
                midi_sequence.append(int(midi))
        
        if len(midi_sequence) < 2:
            return []
        
        # Calculate intervals between consecutive notes
        intervals = []
        for i in range(1, len(midi_sequence)):
            interval = midi_sequence[i] - midi_sequence[i-1]
            intervals.append(interval)
        
        return intervals
    
    def _extract_midi_from_frame(self, audio_data: Dict) -> Optional[int]:
        """
        Extract MIDI note from audio_data using priority order
        
        Priority:
        1. 'midi' field (direct MIDI note)
        2. 'midi_note' field (alternative field name)
        3. 'pitch_hz' field (convert Hz to MIDI)
        4. 'f0' field (fundamental frequency to MIDI)
        
        Args:
            audio_data: Frame's audio_data dict
            
        Returns:
            MIDI note number or None if not found
        """
        # Priority 1: Direct MIDI field
        if 'midi' in audio_data and audio_data['midi']:
            return int(audio_data['midi'])
        
        # Priority 2: Alternative MIDI field
        if 'midi_note' in audio_data and audio_data['midi_note']:
            return int(audio_data['midi_note'])
        
        # Priority 3: Convert pitch_hz
        if 'pitch_hz' in audio_data and audio_data['pitch_hz']:
            hz = float(audio_data['pitch_hz'])
            if hz > 0:
                return self._hz_to_midi(hz)
        
        # Priority 4: Convert f0
        if 'f0' in audio_data and audio_data['f0']:
            hz = float(audio_data['f0'])
            if hz > 0:
                return self._hz_to_midi(hz)
        
        return None
    
    @staticmethod
    def _hz_to_midi(hz: float) -> int:
        """
        Convert frequency (Hz) to MIDI note number
        
        Formula: MIDI = 69 + 12 * log2(Hz / 440)
        Where A4 = 440 Hz = MIDI 69
        
        Args:
            hz: Frequency in Hertz
            
        Returns:
            MIDI note number (0-127)
        """
        if hz <= 0:
            return 0
        
        midi = 69 + 12 * math.log2(hz / 440.0)
        return int(round(midi))
