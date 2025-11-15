#!/usr/bin/env python3
"""
PhraseGenerator
Generates rhythmic musical phrases based on learned patterns from RhythmOracle

This module bridges the gap between rhythmic analysis and musical expression:
- Uses stored rhythmic patterns from Chandra training
- Generates connected musical phrases (not single notes)
- Implements silence/buildup/release musical arcs
- Creates human-like musical timing and feel
"""

import time
import random
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

@dataclass
class MusicalPhrase:
    """Represents a complete musical phrase"""
    phrase_id: str
    notes: List[int]  # MIDI notes in sequence
    timings: List[float]  # Timing for each note (in beats)
    velocities: List[int]  # Velocity for each note
    durations: List[float]  # Duration for each note (in beats)
    mode: str  # imitate, contrast, lead
    phrase_type: str  # melodic_run, arpeggio, silence, buildup, etc.
    rhythm_pattern: str  # Pattern ID from RhythmOracle
    confidence: float
    timestamp: float

class PhraseArc(Enum):
    """Musical phrase development arcs"""
    SILENCE = "silence"
    BUILDUP = "buildup" 
    PEAK = "peak"
    RELEASE = "release"
    CONTEMPLATION = "contemplation"

class PhraseGenerator:
    """
    Generates musical phrases based on learned rhythmic patterns
    
    Creates connected musical expressions with:
    - Rhythmic timing from RhythmOracle
    - Silence and buildup periods
    - Human-like musical phrasing
    - Voice-specific character (melody vs bass)
    """
    
    def __init__(self, rhythm_oracle, enable_silence: bool = True):
        self.rhythm_oracle = rhythm_oracle
        self.enable_silence = enable_silence
        
        # Phrase generation parameters - LONGER PHRASES!
        self.min_phrase_length = 4  # notes minimum
        self.max_phrase_length = 12  # notes maximum
        self.silence_probability = 0.3  # 30% chance of silence
        self.buildup_probability = 0.4  # 40% chance of buildup
        
        # Musical arcs
        self.current_arc = PhraseArc.BUILDUP
        self.arc_start_time = time.time()
        self.arc_duration = 8.0  # seconds
        
        # Phrase state - MORE RESPONSIVE!
        self.last_phrase_time = 0.0
        self.min_phrase_separation = 0.3  # seconds (faster response)
        self.phrases_since_silence = 0
        
        # Harmonic context for phrase generation
        self.current_key = "C"
        self.current_chord = "C"
        self.scale_degrees = [0, 2, 4, 5, 7, 9, 11]  # C major
    
    def generate_phrase(self, current_event: Dict, voice_type: str, 
                       mode: str, harmonic_context: Dict) -> Optional[MusicalPhrase]:
        """Generate a musical phrase based on context and rhythm patterns"""
        
        current_time = time.time()
        
        # Check if enough time passed since last phrase
        if current_time - self.last_phrase_time < self.min_phrase_separation:
            return None
        
        # Update harmonic context
        self._update_harmonic_context(harmonic_context)
        
        # Decide phrase arc
        phrase_arc = self._decide_phrase_arc(current_event)
        
        # Generate phrase based on arc and voice type
        if phrase_arc == PhraseArc.SILENCE and self.enable_silence:
            phrase = self._generate_silence_phrase(current_time)
            self.phrases_since_silence = 0
        elif phrase_arc == PhraseArc.BUILDUP:
            phrase = self._generate_buildup_phrase(mode, voice_type, current_time)
            self.phrases_since_silence += 1
        elif phrase_arc == PhraseArc.PEAK:
            phrase = self._generate_peak_phrase(mode, voice_type, current_time)
            self.phrases_since_silence += 1
        elif phrase_arc == PhraseArc.RELEASE:
            phrase = self._generate_release_phrase(mode, voice_type, current_time)
            self.phrases_since_silence += 1
        else:
            phrase = self._generate_contemplation_phrase(mode, voice_type, current_time)
            self.phrases_since_silence += 1
        
        if phrase:
            self.last_phrase_time = current_time
            
        return phrase
    
    def _decide_phrase_arc(self, current_event: Dict) -> PhraseArc:
        """Decide musical phrase arc based on context"""
        
        current_time = time.time()
        arc_duration = current_time - self.arc_start_time
        
        # Check if arc is completed
        if arc_duration >= self.arc_duration:
            # Transition to new arc
            instrument = current_event.get('instrument', 'unknown')
            activity = current_event.get('rms_db', -80.0)
            
            if instrument == 'drums' and activity > -50:
                # High drum activity -> buildup or peak
                self.current_arc = random.choice([PhraseArc.BUILDUP, PhraseArc.PEAK])
            elif self.phrases_since_silence > 6:
                # Long activity -> release into silence
                self.current_arc = random.choice([PhraseArc.RELEASE, PhraseArc.CONTEMPLATION])
            else:
                # Normal flow
                self.current_arc = random.choice(list(PhraseArc))
            
            self.arc_start_time = current_time
            self.arc_duration = random.uniform(6.0, 12.0)  # Variable arc length
            
        return self.current_arc
    
    def _update_harmonic_context(self, harmonic_context: Dict):
        """Update harmonic context for phrase generation"""
        if not harmonic_context:
            return
            
        self.current_chord = harmonic_context.get('current_chord', 'C')
        key_name = harmonic_context.get('key_signature', 'C_major')
        if '_' in key_name:
            self.current_key = key_name.split('_')[0]
        else:
            self.current_key = key_name
        
        # Update scale degrees based on key
        root_pc = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
        if self.current_key in root_pc:
            root = root_pc[self.current_key]
            major_scale = [2, 2, 1, 2, 2, 2, 1]  # Major scale intervals
            self.scale_degrees = []
            current_pc = root
            for interval in major_scale:
                self.scale_degrees.append(current_pc)
                current_pc = (current_pc + interval) % 12
    
    def _generate_silence_phrase(self, timestamp: float) -> MusicalPhrase:
        """Generate a silence phrase (musical pause)"""
        
        # Silence phrase has no notes but timing
        silence_duration = random.uniform(2.0, 8.0)
        
        return MusicalPhrase(
            phrase_id=f"silence_{int(timestamp)}",
            notes=[],  # No notes in silence
            timings=[silence_duration],  # Duration of silence
            velocities=[],
            durations=[],
            mode="silence",
            phrase_type="silence",
            rhythm_pattern="silence",
            confidence=1.0,
            timestamp=timestamp
        )
    
    def _generate_buildup_phrase(self, mode: str, voice_type: str, timestamp: float) -> MusicalPhrase:
        """Generate a phrase that builds energy"""
        
        phrase_length = random.randint(3, 6)
        # Add octave variety
        if voice_type == "melodic":
            melodic_options = [4, 5, 6, 7]  # Middle to upper register
            base_octave = random.choice(melodic_options)
        else:
            bass_options = [2, 3, 4]  # Low to middle-low register
            base_octave = random.choice(bass_options)
        midi_base = base_octave * 12
        
        # Generate ascending melodic line
        notes = []
        timings = []
        velocities = []
        durations = []
        
        for i in range(phrase_length):
            # More adventurous melodic motion with variety
            if i == 0:
                # First note: start with a strong chord tone
                note_pc = random.choice(self.scale_degrees[:3])  # Root, 3rd, 5th
            elif i < phrase_length - 1:
                # Middle notes: mix of step motion and skips
                interval_options = random.choice([
                    [1, 2, 3, 4, 5],      # Step motion
                    [2, 3, 6, 7],        # Moderate skips  
                    [4, 5, 8, 9],        # Larger leaps
                    [-3, -2, -1, 1, 2, 3] # Allow downward motion
                ])
                interval = random.choice(interval_options)
                # Choose from broader scale + chromaticism
                note_options = self.scale_degrees + random.sample(list(range(12)), 3)
                note_pc = random.choice(note_options)
            else:
                # Last note: resolution tone (typically 1st, 3rd, or 5th)
                note_pc = random.choice(self.scale_degrees[:4])
            
            # Add octave variety within phrase
            octave_adjust = random.choice([-12, 0, 12, 24]) if random.random() < 0.3 else 0
            notes.append(midi_base + note_pc + octave_adjust)
            
            # Gradual timing acceleration
            timing = 1.0 + (phrase_length - i) * 0.2  # Faster toward end
            timings.append(timing)
            
            # Increasing velocity with variation
            base_velocity = 60 + i * 10
            velocity = base_velocity + random.randint(-15, 25)  # Add expressiveness
            velocities.append(max(30, min(127, velocity)))
            
            # Shorter durations toward end with variation
            base_duration = 1.0 - i * 0.1
            duration = base_duration + random.uniform(-0.2, 0.2)  # Add rhythmic interest
            durations.append(max(0.1, duration))
        
        return MusicalPhrase(
            phrase_id=f"buildup_{voice_type}_{int(timestamp)}",
            notes=notes,
            timings=timings,
            velocities=velocities,
            durations=durations,
            mode=mode,
            phrase_type="buildup",
            rhythm_pattern="buildup_pattern",
            confidence=0.8,
            timestamp=timestamp
        )
    
    def _generate_peak_phrase(self, mode: str, voice_type: str, timestamp: float) -> MusicalPhrase:
        """Generate a peak phrase (high energy)"""
        
        phrase_length = random.randint(4, 8)
        # Add octave variety
        if voice_type == "melodic":
            melodic_options = [5, 6, 7, 8]  # Higher register for peak energy
            base_octave = random.choice(melodic_options)
        else:
            bass_options = [2, 3, 4, 5]  # Bass range for peak
            base_octave = random.choice(bass_options)
        midi_base = base_octave * 12
        
        notes = []
        timings = []
        velocities = []
        durations = []
        
        # Generate varied peak content
        possible_notes = list(range(12))  # All 12 chromatic notes
        previous_note = None
        
        for i in range(phrase_length):
            if i == 0:
                # Start with strong chord tone
                note_pc = random.choice(self.scale_degrees[:3])
                previous_note = note_pc
            elif i == phrase_length - 1:
                # End with resolution
                note_pc = random.choice(self.scale_degrees[:2])
            else:
                # Middle notes: mix chords tones with chromatic passing tones
                if random.random() < 0.7:
                    # Chord tones: root, 3rd, 5th, 7th
                    note_pc = random.choice(self.scale_degrees[:min(4, len(self.scale_degrees))])
                else:
                    # Chromatic passing tones for tension
                    note_options = [n for n in possible_notes if n not in self.scale_degrees[:3]]
                    note_pc = random.choice(note_options)
                
                # Add melodic interest: sometimes leap, sometimes step
                if random.random() < 0.6:
                    interval_direction = 1 if random.random() < 0.5 else -1
                    interval_size = random.choice([1, 2, 4, 7, 12])  # Steps, thirds, fifths, octaves
                    note_pc = (previous_note + interval_direction * interval_size) % 12
            
            # Add octave variety within phrase
            octave_adjust = random.choice([-12, 0, 12, 24]) if random.random() < 0.4 else 0
            notes.append(midi_base + note_pc + octave_adjust)
            previous_note = note_pc
            
            # Varied rhythm for staccato feel
            timing = random.uniform(0.15, 0.45)  # Less predictable rhythm
            timings.append(timing)
            
            # Dynamic velocity range
            if i < phrase_length // 2:
                # Building energy: lower to higher
                velocity = 70 + (i * 15) + random.randint(-15, 20)
            else:
                # Peak energy: sustained high
                velocity = 85 + random.randint(-15, 25)
            velocities.append(max(40, min(127, velocity)))
            
            # Varied short durations 
            duration = random.uniform(0.2, 0.5)
            durations.append(duration)
        
        return MusicalPhrase(
            phrase_id=f"peak_{voice_type}_{int(timestamp)}",
            notes=notes,
            timings=timings,
            velocities=velocities,
            durations=durations,
            mode=mode,
            phrase_type="peak",
            rhythm_pattern="peak_pattern",
            confidence=0.9,
            timestamp=timestamp
        )
    
    def _generate_release_phrase(self, mode: str, voice_type: str, timestamp: float) -> MusicalPhrase:
        """Generate a release phrase (settling down)"""
        
        phrase_length = random.randint(3, 6)
        # Add octave variety
        if voice_type == "melodic":
            melodic_options = [4, 5, 6]  # Middle register for release
            base_octave = random.choice(melodic_options)
        else:
            bass_options = [2, 3, 4]  # Bass range for release
            base_octave = random.choice(bass_options)
        midi_base = base_octave * 12
        
        notes = []
        timings = []
        velocities = []
        durations = []
        
        for i in range(phrase_length):
            # Descending motion
            interval = -random.choice([1, 2, 3])
            note_pc = random.choice(self.scale_degrees)
            notes.append(midi_base + note_pc + i * interval)
            
            # Slower timing
            timing = 2.0 - i * 0.2
            timings.append(timing)
            
            # Decreasing velocity
            velocity = 80 - i * 8
            velocities.append(velocity)
            
            # Longer durations
            duration = 1.0 + i * 0.2
            durations.append(duration)
        
        return MusicalPhrase(
            phrase_id=f"release_{voice_type}_{int(timestamp)}",
            notes=notes,
            timings=timings,
            velocities=velocities,
            durations=durations,
            mode=mode,
            phrase_type="release",
            rhythm_pattern="release_pattern",
            confidence=0.7,
            timestamp=timestamp
        )
    
    def _generate_contemplation_phrase(self, mode: str, voice_type: str, timestamp: float) -> MusicalPhrase:
        """Generate a contemplation phrase (meditative)"""
        
        phrase_length = random.randint(2, 4)
        # Add octave variety
        if voice_type == "melodic":
            melodic_options = [3, 4, 5]  # Gentle register for contemplation
            base_octave = random.choice(melodic_options)
        else:
            bass_options = [1, 2, 3]  # Low range for contemplation
            base_octave = random.choice(bass_options)
        midi_base = base_octave * 12
        
        notes = []
        timings = []
        velocities = []
        durations = []
        
        for i in range(phrase_length):
            # Simple stepwise motion
            interval = random.choice([1, 2])
            note_pc = random.choice(self.scale_degrees)
            notes.append(midi_base + note_pc + i * interval)
            
            # Slow, meditative timing
            timing = 2.0 + random.uniform(-0.5, 0.5)
            timings.append(timing)
            
            # Soft velocity
            velocity = 40 + random.randint(-5, 15)
            velocities.append(velocity)
            
            # Long, sustained durations
            duration = 2.0 + random.uniform(-0.5, 1.0)
            durations.append(duration)
        
        return MusicalPhrase(
            phrase_id=f"contemplation_{voice_type}_{int(timestamp)}",
            notes=notes,
            timings=timings,
            velocities=velocities,
            durations=durations,
            mode=mode,
            phrase_type="contemplation",
            rhythm_pattern="contemplation_pattern",
            confidence=0.6,
            timestamp=timestamp
        )
