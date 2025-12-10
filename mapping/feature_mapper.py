# feature_mapper.py
# Maps audio features to musical parameters for MIDI output

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class MIDIParameters:
    """Musical parameters for MIDI output"""
    note: int
    velocity: int
    duration: float
    attack_time: float
    release_time: float
    filter_cutoff: float
    modulation_depth: float
    pan: float
    reverb_amount: float
    timing_deviation: float = 0.0  # Timing offset in seconds (-0.05 to +0.05)

class FeatureMapper:
    """
    Maps audio features to musical parameters
    Converts AI agent decisions into MIDI output
    """
    
    def __init__(self):
        # Mapping parameters
        self.brightness_range = (0.0, 1.0)
        self.loudness_range = (-80.0, 0.0)  # dB
        self.cents_range = (-50.0, 50.0)
        self.ioi_range = (0.0, 5.0)  # seconds
        
        # MIDI parameter ranges - More musical values to reduce click noise
        self.velocity_range = (20, 127)
        self.duration_range = (0.5, 4.0)  # Longer minimum duration
        self.attack_range = (0.05, 0.8)   # Slower minimum attack
        self.release_range = (0.2, 3.0)   # Longer minimum release
        self.filter_range = (0.1, 1.0)
        self.modulation_range = (0.0, 1.0)
        
        # Musical scales and modes
        self.scales = {
            'major': [0, 2, 4, 5, 7, 9, 11],
            'minor': [0, 2, 3, 5, 7, 8, 10],
            'dorian': [0, 2, 3, 5, 7, 9, 10],
            'mixolydian': [0, 2, 4, 5, 7, 9, 10],
            'pentatonic': [0, 2, 4, 7, 9],
            'chromatic': list(range(12))
        }
        
        self.current_scale = 'major'
        self.root_note = 60  # Middle C
        
        # Voice leading state (track last notes for smooth motion)
        self.last_melodic_note = 60  # Middle C
        self.last_bass_note = 55  # G3
        
        # Voice separation parameters
        self.melody_range = (48, 96)  # C3 to C7 (expanded range for more variation)
        self.bass_range = (43, 67)    # G2 to G3 (musical bass range)
        self.min_voice_separation = 8   # Minimum 8 semitones (minor 6th) between melody and bass
        
        # Timing separation parameters
        self.last_melody_time = 0.0
        self.last_bass_time = 0.0
        self.min_timing_separation = 0.1  # Minimum 100ms between melody and bass notes

        # Pattern-based timing state (for RatioAnalyzer integration)
        self._pattern_beat_index = 0  # Tracks position in learned rhythmic pattern

        # Chord tone definitions (relative to root)
        self.chord_tones = {
            'major': [0, 4, 7],
            'minor': [0, 3, 7],
            'maj7': [0, 4, 7, 11],
            'min7': [0, 3, 7, 10],
            'dom7': [0, 4, 7, 10],
            'maj9': [0, 4, 7, 11, 2],
            'min9': [0, 3, 7, 10, 2],
            'dom9': [0, 4, 7, 10, 2],
            'sus2': [0, 2, 7],
            'sus4': [0, 5, 7],
            'dim': [0, 3, 6],
            'aug': [0, 4, 8],
        }
        
    def map_features_to_midi(self, event_data: Dict, 
                           decision_data: Dict, voice_type: str = "melodic") -> MIDIParameters:
        """Map audio features and AI decision to MIDI parameters with harmonic awareness"""
        
        # Extract features
        centroid = event_data.get('centroid', 1000.0)
        rms_db = event_data.get('rms_db', -60.0)
        cents = event_data.get('cents', 0.0)
        ioi = event_data.get('ioi', 1.0)
        f0 = event_data.get('f0', 440.0)
        instrument = event_data.get('instrument', 'unknown')
        
        # Extract harmonic context from decision data
        musical_params = decision_data.get('musical_params', {})
        harmonic_context = self._extract_harmonic_context(musical_params)
        
        # USE PHRASE NOTE DIRECTLY if available - don't override adventurous generation!
        if 'target_note' in musical_params:
            note = musical_params['target_note']
            # Using phrase target (silent)
        else:
            # Fallback to old behavior for non-phrase decisions
            base_note = self._map_to_note(f0, cents, voice_type, instrument)
            
            # Apply harmonic intelligence to note selection
            if harmonic_context and instrument not in ['drums']:
                mode = decision_data.get('mode', 'imitate')
                note = self._apply_harmonic_awareness(base_note, harmonic_context, mode, voice_type)
            else:
                note = base_note
            print(f"🎹 Using map_to_note fallback: {note}")
        
        # USE PHRASE PARAMETERS if available - preserve adventurous timing/dynamics!
        if 'velocity' in musical_params:
            velocity = musical_params['velocity']
        else:
            velocity = self._map_loudness_to_velocity(rms_db, voice_type, instrument)
            
        if 'duration' in musical_params:
            duration = musical_params['duration']
        else:
            duration = self._map_ioi_to_duration(ioi, voice_type, instrument)
        attack_time = self._map_brightness_to_attack(centroid, voice_type, instrument)
        release_time = self._map_brightness_to_release(centroid, voice_type, instrument)
        filter_cutoff = self._map_brightness_to_filter(centroid, voice_type, instrument)
        modulation_depth = self._map_activity_to_modulation(event_data, voice_type, instrument)
        pan = self._map_to_pan(event_data, voice_type, instrument)
        reverb_amount = self._map_to_reverb(event_data, voice_type, instrument)
        
        # Apply timing separation to avoid simultaneous melody/bass notes
        duration = self._apply_timing_separation(duration, voice_type)

        # Extract timing deviation from RatioAnalyzer output
        # Use full pattern data if available, otherwise fall back to polarity
        deviations = event_data.get('deviations', None)
        rhythm_tempo = event_data.get('rhythm_tempo', 120.0)
        duration_pattern = event_data.get('duration_pattern', None)
        deviation_polarity = event_data.get('deviation_polarity', 0)

        timing_deviation = self._calculate_timing_deviation_from_pattern(
            deviations, rhythm_tempo, duration_pattern, deviation_polarity, ioi
        )

        # Apply AI decision modifications
        params = MIDIParameters(
            note=note,
            velocity=velocity,
            duration=duration,
            attack_time=attack_time,
            release_time=release_time,
            filter_cutoff=filter_cutoff,
            modulation_depth=modulation_depth,
            pan=pan,
            reverb_amount=reverb_amount,
            timing_deviation=timing_deviation
        )
        
        # Modify based on AI decision
        params = self._apply_decision_modifications(params, decision_data)
        
        return params
    
    def _map_to_note(self, f0: float, cents: float, voice_type: str = "melodic", instrument: str = "unknown") -> int:
        """Map frequency and cents to MIDI note with harmonic awareness"""
        if f0 <= 0:
            base_note = self.root_note
        else:
            # Convert frequency to MIDI note
            midi_note = 69 + 12 * np.log2(f0 / 440.0)
            
            # Apply cents adjustment
            cents_adjustment = cents / 100.0
            midi_note += cents_adjustment
            
            # Clamp to valid MIDI range
            midi_note = int(round(midi_note))
            base_note = max(21, min(108, midi_note))  # A0 to C8
        
        # Adjust for voice type and instrument
        if voice_type == "bass":
            # Bass voice handled separately (will be overridden by harmonic-aware method)
            import random
            bass_offset = random.choice([-12, -24])  # 1 or 2 octaves down
            bass_note = base_note + bass_offset
            bass_note = max(24, min(60, bass_note))
            return bass_note
        else:
            # Melodic voice - adjust based on instrument
            if instrument == "drums":
                # Drums: use percussive notes (no clear pitch)
                import random
                drum_notes = [36, 38, 42, 46, 49, 51, 56, 60, 64, 68, 72, 76]
                return random.choice(drum_notes)
            else:
                # For pitched instruments, return base note
                # (will be quantized to scale/chord by caller)
                return base_note
    
    def _map_loudness_to_velocity(self, rms_db: float, voice_type: str = "melodic", instrument: str = "unknown") -> int:
        """Map RMS level to MIDI velocity"""
        # Normalize RMS to 0-1 range
        normalized = max(0.0, min(1.0, (rms_db + 80.0) / 80.0))
        
        # Map to velocity range
        velocity = int(self.velocity_range[0] + 
                      normalized * (self.velocity_range[1] - self.velocity_range[0]))
        
        # Adjust for voice type and instrument
        if voice_type == "bass":
            velocity = int(velocity * 0.8)  # Slightly softer for bass
        
        # Adjust for instrument
        if instrument == "drums":
            velocity = int(velocity * 1.2)  # Drums are typically louder
            velocity = min(127, velocity)
        elif instrument == "piano":
            velocity = int(velocity * 0.9)  # Piano is typically softer
        elif instrument == "guitar":
            velocity = int(velocity * 1.0)  # Guitar is typically moderate
        elif instrument == "bass":
            velocity = int(velocity * 0.7)  # Bass is typically softer
        elif instrument == "speech":
            velocity = int(velocity * 0.8)  # Speech accompaniment is moderate
        
        return max(self.velocity_range[0], min(self.velocity_range[1], velocity))
    
    def _map_ioi_to_duration(self, ioi: float, voice_type: str = "melodic", instrument: str = "unknown") -> float:
        """Map inter-onset interval to note duration"""
        # Normalize IOI
        normalized = max(0.0, min(1.0, ioi / 5.0))
        
        # Map to duration range
        duration = self.duration_range[0] + normalized * (self.duration_range[1] - self.duration_range[0])
        
        # Adjust for voice type and instrument
        if voice_type == "bass":
            duration *= 1.5  # Longer notes for bass
        
        # Adjust for instrument
        if instrument == "drums":
            duration *= 0.3  # Drums are typically short
        elif instrument == "piano":
            duration *= 1.2  # Piano notes can be longer
        elif instrument == "guitar":
            duration *= 0.8  # Guitar notes are moderate
        elif instrument == "bass":
            duration *= 1.5  # Bass notes are longer
        elif instrument == "speech":
            duration *= 0.8  # Speech accompaniment is typically shorter
        
        return max(self.duration_range[0], min(self.duration_range[1], duration))
    
    def _map_brightness_to_attack(self, centroid: float, voice_type: str = "melodic", instrument: str = "unknown") -> float:
        """Map spectral centroid to attack time"""
        # Higher centroid = brighter = faster attack
        normalized = max(0.0, min(1.0, centroid / 4000.0))
        
        # Inverse mapping: brighter = faster attack
        attack = self.attack_range[1] - normalized * (self.attack_range[1] - self.attack_range[0])
        
        # Adjust for voice type and instrument
        if voice_type == "bass":
            attack *= 1.5  # Slower attack for bass
        
        # Adjust for instrument
        if instrument == "drums":
            attack *= 0.1  # Drums have very fast attack
        elif instrument == "piano":
            attack *= 0.8  # Piano has moderate attack
        elif instrument == "guitar":
            attack *= 0.6  # Guitar has fast attack
        elif instrument == "bass":
            attack *= 1.2  # Bass has slower attack
        elif instrument == "speech":
            attack *= 0.7  # Speech accompaniment has moderate attack
        
        return max(self.attack_range[0], min(self.attack_range[1], attack))
    
    def _map_brightness_to_release(self, centroid: float, voice_type: str = "melodic", instrument: str = "unknown") -> float:
        """Map spectral centroid to release time"""
        # Higher centroid = brighter = shorter release
        normalized = max(0.0, min(1.0, centroid / 4000.0))
        
        # Inverse mapping: brighter = shorter release
        release = self.release_range[1] - normalized * (self.release_range[1] - self.release_range[0])
        
        # Adjust for voice type and instrument
        if voice_type == "bass":
            release *= 1.2  # Slightly longer release for bass
        
        # Adjust for instrument
        if instrument == "drums":
            release *= 0.2  # Drums have very short release
        elif instrument == "piano":
            release *= 1.0  # Piano has moderate release
        elif instrument == "guitar":
            release *= 0.8  # Guitar has shorter release
        elif instrument == "bass":
            release *= 1.3  # Bass has longer release
        elif instrument == "speech":
            release *= 0.9  # Speech accompaniment has moderate release
        
        return max(self.release_range[0], min(self.release_range[1], release))
    
    def _map_brightness_to_filter(self, centroid: float, voice_type: str = "melodic", instrument: str = "unknown") -> float:
        """Map spectral centroid to filter cutoff"""
        # Higher centroid = brighter = higher filter cutoff
        normalized = max(0.0, min(1.0, centroid / 4000.0))
        
        filter_cutoff = self.filter_range[0] + normalized * (self.filter_range[1] - self.filter_range[0])
        
        # Adjust for voice type and instrument
        if voice_type == "bass":
            filter_cutoff *= 0.6  # Lower filter for bass
        
        # Adjust for instrument
        if instrument == "drums":
            filter_cutoff *= 1.2  # Drums are bright
        elif instrument == "piano":
            filter_cutoff *= 0.8  # Piano is moderate
        elif instrument == "guitar":
            filter_cutoff *= 0.9  # Guitar is slightly bright
        elif instrument == "bass":
            filter_cutoff *= 0.5  # Bass is dark
        elif instrument == "speech":
            filter_cutoff *= 0.7  # Speech accompaniment is moderate
        
        return max(self.filter_range[0], min(self.filter_range[1], filter_cutoff))
    
    def _map_activity_to_modulation(self, event_data: Dict, voice_type: str = "melodic", instrument: str = "unknown") -> float:
        """Map activity level to modulation depth"""
        # Calculate activity from multiple features
        rms_db = event_data.get('rms_db', -80.0)
        onset = event_data.get('onset', False)
        f0 = event_data.get('f0', 0.0)
        
        # RMS contribution
        rms_activity = max(0.0, min(1.0, (rms_db + 80.0) / 80.0))
        
        # Onset contribution
        onset_activity = 1.0 if onset else 0.0
        
        # Frequency contribution
        freq_activity = 0.5 if f0 > 0 else 0.0
        
        # Combined activity
        activity = (rms_activity * 0.4 + onset_activity * 0.4 + freq_activity * 0.2)
        
        # Map to modulation depth
        modulation = self.modulation_range[0] + activity * (self.modulation_range[1] - self.modulation_range[0])
        
        # Adjust for voice type and instrument
        if voice_type == "bass":
            modulation *= 0.5  # Less modulation for bass
        
        # Adjust for instrument
        if instrument == "drums":
            modulation *= 0.3  # Drums have less modulation
        elif instrument == "piano":
            modulation *= 0.6  # Piano has moderate modulation
        elif instrument == "guitar":
            modulation *= 0.8  # Guitar has more modulation
        elif instrument == "bass":
            modulation *= 0.4  # Bass has less modulation
        elif instrument == "speech":
            modulation *= 0.5  # Speech accompaniment has moderate modulation
        
        return max(self.modulation_range[0], min(self.modulation_range[1], modulation))
    
    def _map_to_pan(self, event_data: Dict, voice_type: str = "melodic", instrument: str = "unknown") -> float:
        """Map features to pan position"""
        # Use spectral centroid for pan (simplified)
        centroid = event_data.get('centroid', 1000.0)
        
        # Normalize centroid to pan range
        normalized = max(0.0, min(1.0, centroid / 4000.0))
        
        # Map to pan range (-1.0 to 1.0)
        pan = -1.0 + normalized * 2.0
        
        # Adjust for voice type
        if voice_type == "bass":
            pan *= 0.3  # Less pan variation for bass
        
        return max(-1.0, min(1.0, pan))
    
    def _map_to_reverb(self, event_data: Dict, voice_type: str = "melodic", instrument: str = "unknown") -> float:
        """Map features to reverb amount"""
        # Use IOI for reverb (longer intervals = more reverb)
        ioi = event_data.get('ioi', 1.0)
        
        # Normalize IOI
        normalized = max(0.0, min(1.0, ioi / 5.0))
        
        # Map to reverb range
        reverb = normalized * 0.8  # Max 80% reverb
        
        # Adjust for voice type
        if voice_type == "bass":
            reverb *= 0.7  # Slightly less reverb for bass
        
        return max(0.0, min(1.0, reverb))

    def _calculate_timing_deviation_from_pattern(
        self,
        deviations: list,
        rhythm_tempo: float,
        duration_pattern: list,
        deviation_polarity: int,
        ioi: float
    ) -> float:
        """
        Calculate timing deviation from RatioAnalyzer pattern data.

        Uses the full deviations array and tempo to apply pattern-based
        timing feel, not just per-note micro-timing.

        The deviations array contains fractional timing offsets for each
        beat position in the learned pattern. We cycle through these
        to match the human's rhythmic feel.

        Args:
            deviations: Full deviations array from RatioAnalyzer (fractional offsets)
            rhythm_tempo: Tempo in BPM from RatioAnalyzer
            duration_pattern: Duration ratios (e.g., [3, 2] for swing)
            deviation_polarity: Fallback polarity if no pattern data
            ioi: Inter-onset interval for fallback calculation

        Returns:
            Timing offset in seconds
        """
        # If we have full pattern data, use pattern-based timing
        if deviations and len(deviations) > 0 and rhythm_tempo > 0:
            # Get deviation for current beat position in pattern
            pattern_len = len(deviations)
            beat_idx = self._pattern_beat_index % pattern_len

            # Get fractional deviation for this beat position
            # deviations are fractions of the beat duration (e.g., 0.05 = 5% late)
            fractional_deviation = deviations[beat_idx]

            # Convert to seconds using tempo
            # beat_duration = 60 / tempo (seconds per beat)
            beat_duration = 60.0 / rhythm_tempo
            timing_deviation_seconds = fractional_deviation * beat_duration

            # Advance beat position for next note
            self._pattern_beat_index = (self._pattern_beat_index + 1) % pattern_len

            # Clamp to reasonable range (±100ms max for pattern-based)
            timing_deviation_seconds = max(-0.1, min(0.1, timing_deviation_seconds))

            return timing_deviation_seconds

        # Fallback: use polarity-based micro-timing (original behavior)
        if deviation_polarity == 0:
            return 0.0

        # Base deviation: 25ms feels natural
        base_deviation_ms = 25.0
        ioi_scale = min(1.0, max(0.5, ioi / 0.5))
        deviation_seconds = (base_deviation_ms / 1000.0) * ioi_scale * deviation_polarity

        return max(-0.05, min(0.05, deviation_seconds))

    def _apply_decision_modifications(self, params: MIDIParameters, 
                                   decision_data: Dict) -> MIDIParameters:
        """Apply AI decision modifications to parameters"""
        mode = decision_data.get('mode', 'imitate')
        confidence = decision_data.get('confidence', 0.5)
        musical_params = decision_data.get('musical_params', {})
        
        # Apply mode-specific modifications
        if mode == 'imitate':
            # Subtle modifications for imitation
            params.velocity = int(params.velocity * 0.9)
            params.duration *= 1.1
            
        elif mode == 'contrast':
            # Strong modifications for contrast
            params.velocity = int(params.velocity * 1.2)
            params.duration *= 0.8
            params.filter_cutoff = 1.0 - params.filter_cutoff  # Invert filter
            
        elif mode == 'lead':
            # Experimental modifications for leading
            params.velocity = int(params.velocity * 1.1)
            params.modulation_depth = min(1.0, params.modulation_depth * 1.5)
        
        # Apply confidence-based modifications
        confidence_factor = 0.5 + confidence * 0.5
        params.velocity = int(params.velocity * confidence_factor)
        
        # Apply any specific musical parameters from decision
        if 'velocity' in musical_params:
            params.velocity = musical_params['velocity']
        if 'duration' in musical_params:
            params.duration = musical_params['duration']
        if 'attack' in musical_params:
            params.attack_time = musical_params['attack']
        if 'release' in musical_params:
            params.release_time = musical_params['release']
        if 'filter_cutoff' in musical_params:
            params.filter_cutoff = musical_params['filter_cutoff']
        if 'modulation_depth' in musical_params:
            params.modulation_depth = musical_params['modulation_depth']
        
        return params
    
    def set_scale(self, scale_name: str, root_note: int = 60):
        """Set musical scale and root note"""
        if scale_name in self.scales:
            self.current_scale = scale_name
            self.root_note = root_note
    
    def constrain_to_scale(self, note: int) -> int:
        """Constrain MIDI note to current scale"""
        if self.current_scale == 'chromatic':
            return note
        
        scale_intervals = self.scales[self.current_scale]
        
        # Find closest note in scale
        octave = note // 12
        note_class = note % 12
        
        # Find closest scale degree
        closest_interval = min(scale_intervals, key=lambda x: abs(x - note_class))
        
        return octave * 12 + closest_interval
    
    # ========== HARMONIC AWARENESS METHODS ==========
    
    def _extract_harmonic_context(self, musical_params: Dict) -> Optional[Dict]:
        """Extract harmonic context from musical parameters"""
        if not musical_params:
            return None
        
        # Check if harmonic info is present
        if 'scale_degrees' not in musical_params:
            return None
        
        return {
            'current_chord': musical_params.get('current_chord', 'N/A'),
            'key_signature': musical_params.get('key_signature', 'N/A'),
            'scale_degrees': musical_params.get('scale_degrees', []),
            'chord_root': musical_params.get('chord_root', 0),
            'chord_type': self._parse_chord_type(musical_params.get('current_chord', '')),
            'use_contrasting_harmony': musical_params.get('use_contrasting_harmony', False),
            'explore_harmony': musical_params.get('explore_harmony', False),
            'chord_stability': musical_params.get('chord_stability', 0.5)
        }
    
    def _parse_chord_type(self, chord_name: str) -> str:
        """Parse chord type from chord name (e.g., 'Cmaj7' -> 'maj7')"""
        chord_lower = chord_name.lower()
        
        # Check for extended chords first (more specific)
        if 'maj9' in chord_lower:
            return 'maj9'
        elif 'min9' in chord_lower or 'm9' in chord_lower:
            return 'min9'
        elif 'dom9' in chord_lower or (('9' in chord_lower) and ('maj' not in chord_lower) and ('m' not in chord_lower)):
            return 'dom9'
        elif 'maj7' in chord_lower:
            return 'maj7'
        elif 'min7' in chord_lower or 'm7' in chord_lower:
            return 'min7'
        elif '7' in chord_lower:
            return 'dom7'
        elif 'sus4' in chord_lower:
            return 'sus4'
        elif 'sus2' in chord_lower:
            return 'sus2'
        elif 'dim' in chord_lower:
            return 'dim'
        elif 'aug' in chord_lower:
            return 'aug'
        elif 'min' in chord_lower or 'm' in chord_lower:
            return 'minor'
        else:
            return 'major'
    
    def _apply_harmonic_awareness(self, base_note: int, harmonic_context: Dict, 
                                   mode: str, voice_type: str) -> int:
        """Apply harmonic awareness to note selection based on mode and voice type"""
        
        if voice_type == "bass":
            # Bass voice: use chord-based bass notes with strict range enforcement
            return self._select_bass_note(base_note, harmonic_context, mode)
        else:
            # Melodic voice: quantize to scale/chord based on mode with range enforcement
            if mode == 'imitate':
                # Imitate: prefer chord tones, stay in key
                note = self._select_chord_tone(base_note, harmonic_context)
            elif mode == 'contrast':
                # Contrast: use related harmony
                if harmonic_context.get('use_contrasting_harmony'):
                    note = self._select_contrasting_note(base_note, harmonic_context)
                else:
                    note = self._quantize_to_scale(base_note, harmonic_context['scale_degrees'])
            elif mode == 'lead':
                # Lead: explore scale, allow non-chord tones
                if harmonic_context.get('explore_harmony'):
                    note = self._quantize_to_scale(base_note, harmonic_context['scale_degrees'])
                else:
                    note = self._select_chord_tone(base_note, harmonic_context)
            else:
                # Fallback: quantize to scale
                note = self._quantize_to_scale(base_note, harmonic_context['scale_degrees'])
            
            # Enforce melody range and separation from bass
            return self._enforce_voice_separation(note, voice_type)
    
    def _quantize_to_scale(self, note: int, scale_degrees: List[int]) -> int:
        """Quantize MIDI note to nearest scale degree"""
        if not scale_degrees:
            return note
        
        octave = note // 12
        note_class = note % 12
        
        # Find closest scale degree
        closest_degree = min(scale_degrees, key=lambda x: abs(x - note_class))
        quantized_note = octave * 12 + closest_degree
        
        # Apply voice leading (move smoothly from last note)
        quantized_note = self._apply_voice_leading(quantized_note, self.last_melodic_note)
        
        # Update last note
        self.last_melodic_note = quantized_note
        
        return quantized_note
    
    def _select_chord_tone(self, base_note: int, harmonic_context: Dict) -> int:
        """Select nearest chord tone for the current chord"""
        chord_root = harmonic_context.get('chord_root', 0)
        chord_type = harmonic_context.get('chord_type', 'major')
        
        # Get chord tones for this chord type
        chord_intervals = self.chord_tones.get(chord_type, self.chord_tones['major'])
        
        # Build chord tones in MIDI space (multiple octaves)
        octave = base_note // 12
        chord_notes = []
        
        # Include chord tones in nearby octaves
        for oct_offset in [-1, 0, 1]:
            for interval in chord_intervals:
                chord_note = (octave + oct_offset) * 12 + ((chord_root + interval) % 12)
                if 21 <= chord_note <= 108:  # Valid MIDI range
                    chord_notes.append(chord_note)
        
        # Find closest chord tone
        if chord_notes:
            closest_chord_tone = min(chord_notes, key=lambda x: abs(x - base_note))
            
            # Apply voice leading
            closest_chord_tone = self._apply_voice_leading(closest_chord_tone, self.last_melodic_note)
            
            # Update last note
            self.last_melodic_note = closest_chord_tone
            
            return closest_chord_tone
        else:
            # Fallback: quantize to scale
            return self._quantize_to_scale(base_note, harmonic_context.get('scale_degrees', []))
    
    def _select_contrasting_note(self, base_note: int, harmonic_context: Dict) -> int:
        """Select a note that provides harmonic contrast (related chord)"""
        chord_root = harmonic_context.get('chord_root', 0)
        scale_degrees = harmonic_context.get('scale_degrees', [])
        
        # Use related chord roots (tritone sub, relative minor/major, etc.)
        # For simplicity: use scale degrees but prefer non-chord tones
        octave = base_note // 12
        note_class = base_note % 12
        
        # Find scale degrees that are NOT in the current chord
        chord_type = harmonic_context.get('chord_type', 'major')
        chord_intervals = self.chord_tones.get(chord_type, [0, 4, 7])
        chord_pcs = [(chord_root + i) % 12 for i in chord_intervals]
        
        # Prefer scale degrees that aren't chord tones
        non_chord_scale_degrees = [sd for sd in scale_degrees if sd not in chord_pcs]
        
        # For contrast mode: be more adventurous with melodic exploration
        import random
        
        if non_chord_scale_degrees:
            # Pick a random non-chord tone for more variety
            closest_degree = random.choice(non_chord_scale_degrees)
        elif scale_degrees:
            # Pick a random scale degree
            closest_degree = random.choice(scale_degrees)
        else:
            # Chromatic exploration: use semitone neighbors
            closest_degree = (note_class + random.choice([1, 2, -1, -2])) % 12
        
        contrasting_note = octave * 12 + closest_degree
        
        # Apply more relaxed voice leading for contrast mode (allow larger intervals)
        contrasting_note = self._apply_voice_leading(contrasting_note, self.last_melodic_note, max_interval=12)
        
        # Update last note
        self.last_melodic_note = contrasting_note
        
        return contrasting_note
    
    def _select_bass_note(self, base_note: int, harmonic_context: Dict, mode: str) -> int:
        """Select intelligent bass note based on chord and mode with strict range enforcement"""
        chord_root = harmonic_context.get('chord_root', 0)
        chord_type = harmonic_context.get('chord_type', 'major')
        
        import random
        
        # Bass note options (in bass register)
        # Choose octave based on chord root to stay in musical range
        
        # For low chord roots (0-2 = C-C#-D), use lower octave
        # For medium chord roots (3-5 = D#-F-Gb), use middle octave  
        # For high chord roots (6-11 = F#-F#-G-A-A#-B), use higher octave
        if chord_root in [0, 1, 2]:  # C, C#, D
            bass_octave = 3  # MIDI 36-47
        elif chord_root in [3, 4, 5]:  # D#, E, F, F#
            bass_octave = 3  # MIDI 36-47  
        else:  # Other roots
            bass_octave = 3  # MIDI 36-47
        
        if mode == 'imitate':
            # Root position or 5th
            if random.random() < 0.7:  # 70% root
                bass_pc = chord_root
            else:  # 30% fifth
                bass_pc = (chord_root + 7) % 12
        
        elif mode == 'contrast':
            # Use related chord roots or walking bass
            # For now: use 3rd or 6th interval
            if random.random() < 0.5:
                bass_pc = (chord_root + 3) % 12  # Minor 3rd
            else:
                bass_pc = (chord_root + 9) % 12  # Major 6th
        
        elif mode == 'lead':
            # Walking bass: move by scale degrees
            scale_degrees = harmonic_context.get('scale_degrees', [0])
            last_pc = self.last_bass_note % 12
            
            # Find nearby scale degrees
            nearby_degrees = [sd for sd in scale_degrees if abs(sd - last_pc) <= 4]
            if nearby_degrees:
                bass_pc = random.choice(nearby_degrees)
            else:
                bass_pc = chord_root
        
        else:
            # Default: root position
            bass_pc = chord_root
        
        # Build bass note in appropriate octave
        bass_note = bass_octave * 12 + bass_pc
        
        # Apply voice leading (smoother bass motion)
        bass_note = self._apply_voice_leading(bass_note, self.last_bass_note, max_interval=12)
        
        # Enforce strict bass range (MIDI 24-48)
        bass_note = self._enforce_voice_separation(bass_note, "bass")
        
        # Update last bass note
        self.last_bass_note = bass_note
        
        return bass_note
    
    def _enforce_voice_separation(self, note: int, voice_type: str) -> int:
        """Enforce strict voice separation between melody and bass"""
        
        if voice_type == "bass":
            # Bass: enforce lower register (MIDI 24-48)
            min_bass, max_bass = self.bass_range
            
            if note < min_bass:
                # Move up octaves until in range
                while note < min_bass:
                    note += 12
            elif note > max_bass:
                # Move down octaves until in range
                while note > max_bass:
                    note -= 12
            
            # Ensure minimum separation from melody
            if hasattr(self, 'last_melodic_note') and self.last_melodic_note > 0:
                if note > self.last_melodic_note - self.min_voice_separation:
                    note = self.last_melodic_note - self.min_voice_separation
                    # Ensure still in bass range
                    if note > max_bass:
                        note = max_bass
        
        else:
            # Melody: enforce upper register (MIDI 60-84)
            min_melody, max_melody = self.melody_range
            
            if note < min_melody:
                # Move up octaves until in range
                while note < min_melody:
                    note += 12
            elif note > max_melody:
                # Move down octaves until in range
                while note > max_melody:
                    note -= 12
            
            # Ensure minimum separation from bass
            if hasattr(self, 'last_bass_note') and self.last_bass_note > 0:
                if note < self.last_bass_note + self.min_voice_separation:
                    note = self.last_bass_note + self.min_voice_separation
                    # Ensure still in melody range
                    if note < min_melody:
                        note = min_melody
        
        return note
    
    def _apply_timing_separation(self, duration: float, voice_type: str) -> float:
        """Apply timing separation to avoid simultaneous melody/bass notes"""
        import time
        current_time = time.time()
        
        if voice_type == "bass":
            # Check if melody was recently played
            if current_time - self.last_melody_time < self.min_timing_separation:
                # Extend bass duration to avoid overlap
                duration = max(duration, self.min_timing_separation * 2)
            self.last_bass_time = current_time
        else:
            # Check if bass was recently played
            if current_time - self.last_bass_time < self.min_timing_separation:
                # Extend melody duration to avoid overlap
                duration = max(duration, self.min_timing_separation * 2)
            self.last_melody_time = current_time
        
        return duration
    
    def _apply_voice_leading(self, target_note: int, previous_note: int, max_interval: int = 10) -> int:
        """Apply voice leading to prefer smooth motion between notes"""
        interval = abs(target_note - previous_note)
        
        # If interval is small enough, use target note
        if interval <= max_interval:
            return target_note
        
        # Otherwise, find closest note in same pitch class
        target_pc = target_note % 12
        candidates = []
        
        # Check multiple octaves
        for octave_offset in [-2, -1, 0, 1, 2]:
            candidate = previous_note + octave_offset * 12
            if candidate % 12 == target_pc and 21 <= candidate <= 108:
                candidates.append(candidate)
        
        # Choose closest to previous note
        if candidates:
            return min(candidates, key=lambda x: abs(x - previous_note))
        else:
            return target_note
