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
import numpy as np
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
    
    def __init__(self, rhythm_oracle, audio_oracle=None, enable_silence: bool = True, visualization_manager=None):
        self.rhythm_oracle = rhythm_oracle
        self.audio_oracle = audio_oracle  # AudioOracle for learned patterns
        self.visualization_manager = visualization_manager
        
        # Phrase-based context buffer for AudioOracle matching
        self.recent_human_notes = []  # List of recent human MIDI notes
        self.max_context_length = 5  # Remember last 5 notes for context
        self.enable_silence = enable_silence
        
        # Phrase memory for thematic development
        from .phrase_memory import PhraseMemory
        self.phrase_memory = PhraseMemory(max_history=20, visualization_manager=visualization_manager)
        
        # IRCAM Phase 3+: Event tracking for request-based generation
        self.recent_events = []  # Track all recent events with metadata
        self.max_event_history = 50  # Keep last 50 events
        
        # Phrase generation parameters - MODE-SPECIFIC LENGTHS for consistency
        self.phrase_length_by_mode = {
            'SHADOW': (3, 6),    # Shorter, responsive
            'MIRROR': (4, 8),    # Medium, balanced
            'COUPLE': (6, 12),   # Longer, independent
            'IMITATE': (3, 6),   # Similar to SHADOW
            'CONTRAST': (5, 10), # Medium-long
            'LEAD': (4, 9)       # Medium
        }
        # Fallback for unknown modes
        self.min_phrase_length = 4  # notes minimum
        self.max_phrase_length = 12  # notes maximum
        self.silence_probability = 0.05  # 5% chance of silence (reduced from 30%)
        self.buildup_probability = 0.4  # 40% chance of buildup
        
        # Musical arcs - start with active arc
        self.current_arc = PhraseArc.BUILDUP
        self.arc_start_time = time.time()
        self.arc_duration = 15.0  # seconds (shorter cycles)
        
        # Phrase state - TUNED for better musical flow
        self.last_phrase_time = 0.0
        self.min_phrase_separation = 0.5  # seconds (increased for more space)
        self.phrases_since_silence = 0
        
        # Harmonic context for phrase generation
        self.current_key = "C"
        self.current_chord = "C"
        self.scale_degrees = [0, 2, 4, 5, 7, 9, 11]  # C major
        
        # DRAMATICALLY EXPANDED RANGES for maximum variety - ADJUSTMENT REQUESTED
        self.melodic_range = (60, 96)   # C4 to C7 (3 octaves - expressive lead)
        self.bass_range = (36, 60)      # C2 to C4 (2 octaves - foundation)
    
    def track_event(self, event_data: Dict, source: str = 'human'):
        """
        Track events for request-based generation context analysis
        
        Args:
            event_data: Event data dict
            source: 'human' or 'ai' to distinguish input sources
        """
        tagged_event = event_data.copy()
        tagged_event['source'] = source
        tagged_event['timestamp'] = time.time()
        
        # DEBUG: Check if gesture_token is present
        if source == 'human' and not hasattr(self, '_track_debug_count'):
            self._track_debug_count = 0
        if source == 'human' and self._track_debug_count < 3:
            has_token = 'gesture_token' in event_data
            token_value = event_data.get('gesture_token', 'MISSING')
            print(f"🔍 TRACK: source={source}, has_gesture_token={has_token}, token={token_value}, buffer_size={len(self.recent_events)}")
            self._track_debug_count += 1
        
        self.recent_events.append(tagged_event)
        
        # Keep history limited
        if len(self.recent_events) > self.max_event_history:
            self.recent_events.pop(0)
    
    def _get_recent_human_events(self, n: int = 10) -> List[Dict]:
        """Get recent human-only events (exclude AI output)"""
        human_events = [e for e in self.recent_events if e.get('source') == 'human']
        return human_events[-n:] if len(human_events) > n else human_events
    
    def _get_recent_human_tokens(self, n: int = 5) -> List[int]:
        """Extract gesture tokens from recent human input"""
        human_events = self._get_recent_human_events(n * 2)  # Get more events to ensure we have N tokens
        tokens = []
        
        # DEBUG: Show what we're working with
        if len(human_events) == 0:
            print(f"🔍 DEBUG: No human events found in recent_events buffer (size: {len(self.recent_events)})")
        
        for event in human_events:
            token = event.get('gesture_token')
            if token is not None:
                tokens.append(token)
        
        # DEBUG: Show extraction results
        if len(human_events) > 0 and len(tokens) == 0:
            print(f"🔍 DEBUG: Found {len(human_events)} human events but no gesture tokens")
        
        return tokens[-n:] if len(tokens) > n else tokens
    
    def _calculate_avg_consonance(self, n: int = 10) -> float:
        """Calculate average consonance from recent human events"""
        human_events = self._get_recent_human_events(n)
        
        if not human_events:
            return 0.5  # Default
        
        consonances = [e.get('consonance', 0.5) for e in human_events]
        return float(np.mean(consonances))
    
    def _get_melodic_tendency(self, n: int = 5) -> float:
        """Determine recent melodic direction from human input"""
        human_events = self._get_recent_human_events(n)
        
        if len(human_events) < 2:
            return 0.0  # Neutral
        
        # Extract midi_relative values
        intervals = [e.get('midi_relative', 0) for e in human_events if 'midi_relative' in e]
        
        if not intervals:
            return 0.0
        
        # Average interval (positive = ascending, negative = descending)
        return float(np.mean(intervals))
    
    def _get_rhythmic_tendency(self, n: int = 5) -> float:
        """Determine recent rhythmic density from human input"""
        human_events = self._get_recent_human_events(n)
        
        if not human_events:
            return 1.0  # Default
        
        # Extract rhythm_ratio values (higher = longer durations)
        ratios = [e.get('rhythm_ratio', 1) for e in human_events if 'rhythm_ratio' in e]
        
        if not ratios:
            return 1.0
        
        return float(np.mean(ratios))
    
    def _get_barlow_complexity(self, n: int = 5) -> float:
        """Get average Barlow complexity from recent events"""
        human_events = self._get_recent_human_events(n)
        complexities = [e.get('barlow_complexity', 5.0) for e in human_events 
                       if 'barlow_complexity' in e]
        return float(np.mean(complexities)) if complexities else 5.0
    
    def _get_deviation_polarity(self, n: int = 5) -> float:
        """Get average deviation polarity from recent events"""
        human_events = self._get_recent_human_events(n)
        polarities = [e.get('deviation_polarity', 0.0) for e in human_events 
                     if 'deviation_polarity' in e]
        return float(np.mean(polarities)) if polarities else 0.0
    
    def _build_shadowing_request(self, mode_params: Dict = None) -> Dict:
        """
        Enhanced SHADOW: Close imitation using multiple parameters
        Combines gesture tokens, consonance, and rhythmic complexity
        """
        if mode_params is None:
            mode_params = {}
        
        recent_tokens = self._get_recent_human_tokens(n=3)
        avg_consonance = self._calculate_avg_consonance(n=5)
        barlow_complexity = self._get_barlow_complexity(n=3)
        
        print(f"🎯 Shadow: recent_tokens={recent_tokens}, consonance={avg_consonance:.2f}, barlow={barlow_complexity:.1f}")  # DEBUG
        
        # Use primary parameter in AudioOracle-compatible format
        # Focus on gesture token matching for shadow mode
        request = {
            'parameter': 'gesture_token',
            'type': '==',
            'value': recent_tokens[-1] if recent_tokens else None,
            'weight': mode_params.get('request_weight', 0.95)
        }
        
        return request
    
    def _build_mirroring_request(self, mode_params: Dict = None) -> Dict:
        """
        Enhanced MIRROR: Phrase-aware complementary variation
        Contrasts melodic direction while matching harmonic character
        """
        if mode_params is None:
            mode_params = {}
        
        avg_consonance = self._calculate_avg_consonance(n=10)
        
        # Focus on consonance matching for mirror mode
        request = {
            'parameter': 'consonance',
            'type': '==',
            'value': avg_consonance,
            'weight': mode_params.get('request_weight', 0.7)
        }
        
        return request
    
    def _build_coupling_request(self, mode_params: Dict = None) -> Dict:
        """
        Enhanced COUPLE: Independent but rhythmically/harmonically aware
        High consonance with complementary rhythm
        """
        if mode_params is None:
            mode_params = {}
        
        # High consonance constraint for couple mode
        request = {
            'parameter': 'consonance',
            'type': '>',
            'value': 0.7,
            'weight': mode_params.get('request_weight', 0.3)  # Loose constraint
        }
        
        return request
    
    def _build_request_for_mode(self, mode: str, mode_params: Dict = None) -> Optional[Dict]:
        """Build request parameters based on current behavior mode"""
        mode_upper = mode.upper()
        
        if mode_upper == 'SHADOW':
            return self._build_shadowing_request(mode_params)
        elif mode_upper == 'MIRROR':
            return self._build_mirroring_request(mode_params)
        elif mode_upper == 'COUPLE':
            return self._build_coupling_request(mode_params)
        elif mode_upper in ['IMITATE', 'CONTRAST', 'LEAD']:
            # Use shadowing as fallback for compatibility
            return self._build_shadowing_request(mode_params)
        else:
            return None
    
    def generate_phrase(self, current_event: Dict, voice_type: str, 
                       mode: str, harmonic_context: Dict, temperature: float = 0.8, 
                       activity_multiplier: float = 1.0) -> Optional[MusicalPhrase]:
        """Generate a musical phrase based on context and rhythm patterns
        
        Args:
            temperature: Controls randomness in Oracle generation (from mode_params)
            activity_multiplier: Multiplier from performance arc (0.0-1.0)
                - During buildup: 0.3 → 1.0 (sparse → full)
                - During main: 1.0 (full activity)
                - During ending: 1.0 → 0.0 (gradual fade)
        """
        
        current_time = time.time()
        
        # Check if enough time passed since last phrase
        if current_time - self.last_phrase_time < self.min_phrase_separation:
            return None
        
        # THEMATIC DEVELOPMENT: Check if we should recall/develop existing theme
        if self.phrase_memory.should_recall_theme():
            motif = self.phrase_memory.get_current_theme()
            if motif:
                variation_type = random.choice(['transpose', 'invert', 'augment', 'diminish'])
                variation = self.phrase_memory.get_variation(motif, variation_type)
                
                # Create phrase from variation
                phrase = MusicalPhrase(
                    phrase_id=f"thematic_{int(current_time)}",
                    notes=variation['notes'],
                    timings=[0.5] * len(variation['notes']),  # Simple timing
                    velocities=[80] * len(variation['notes']),
                    durations=variation['durations'],
                    mode=mode,
                    phrase_type='thematic_development',
                    rhythm_pattern=f"variation_{variation_type}",
                    confidence=0.85,
                    timestamp=current_time
                )
                
                self.last_phrase_time = current_time
                
                # DON'T store recalled phrases back into memory (prevents feedback loop)
                # Recalled phrases are already in memory - storing them again increases their
                # probability, creating a positive feedback loop where the same motif dominates
                
                # Emit phrase memory event
                if self.visualization_manager:
                    self.visualization_manager.emit_phrase_memory(
                        action='recall',
                        motif=motif,
                        variation_type=variation_type,
                        timestamp=current_time
                    )
                
                # Emit thematic recall visualization event
                if self.visualization_manager:
                    self.visualization_manager.emit_timeline_update(
                        'thematic_recall', 
                        mode=mode,
                        timestamp=current_time
                    )
                    
                    # 🎨 ALSO emit request parameters for viewport (thematic recall path)
                    thematic_request = {
                        'parameter': 'motif_recall',
                        'type': 'thematic',
                        'value': f"variation_{variation_type}",
                        'weight': 0.85
                    }
                    self.visualization_manager.emit_request_params(
                        mode=mode,
                        request=thematic_request,
                        voice_type=voice_type
                    )
                
                return phrase
        
        # Update harmonic context
        self._update_harmonic_context(harmonic_context)
        
        # Build request based on mode (for AudioOracle pattern matching)
        request = self._build_request_for_mode(mode)
        
        # Emit request parameters to visualization (rate-limited to avoid spam)
        if self.visualization_manager and request:
            # Rate limit: only emit if enough time passed since last emission (reduce Qt event queue spam)
            current_emit_time = time.time()
            if not hasattr(self, '_last_viz_emit_time'):
                self._last_viz_emit_time = 0
            
            if current_emit_time - self._last_viz_emit_time > 0.5:  # Max 2 updates per second
                self.visualization_manager.emit_request_params(
                    mode=mode,
                    request=request,
                    voice_type=voice_type
                )
                self._last_viz_emit_time = current_emit_time
        
        # Decide phrase arc
        phrase_arc = self._decide_phrase_arc(current_event)
        
        # Generate phrase based on arc and voice type
        if phrase_arc == PhraseArc.SILENCE and self.enable_silence:
            phrase = self._generate_silence_phrase(current_time)
            self.phrases_since_silence = 0
        elif phrase_arc == PhraseArc.BUILDUP:
            phrase = self._generate_buildup_phrase(mode, voice_type, current_time, current_event, temperature, activity_multiplier)
            self.phrases_since_silence += 1
        elif phrase_arc == PhraseArc.PEAK:
            phrase = self._generate_peak_phrase(mode, voice_type, current_time, current_event, temperature, activity_multiplier)
            self.phrases_since_silence += 1
        elif phrase_arc == PhraseArc.RELEASE:
            phrase = self._generate_release_phrase(mode, voice_type, current_time, current_event, temperature, activity_multiplier)
            self.phrases_since_silence += 1
        else:
            phrase = self._generate_contemplation_phrase(mode, voice_type, current_time, current_event, temperature, activity_multiplier)
            self.phrases_since_silence += 1
        
        if phrase:
            self.last_phrase_time = current_time
            
            # Store phrase in memory for future thematic development
            self.phrase_memory.add_phrase(phrase.notes, phrase.durations, current_time, mode)
            
            # Emit phrase memory event for new storage
            if self.visualization_manager:
                self.visualization_manager.emit_phrase_memory(
                    action='store',
                    motif=phrase.notes,
                    variation_type=None,
                    timestamp=current_time
                )
            
        return phrase
    
    def _decide_phrase_arc(self, current_event: Dict) -> PhraseArc:
        """Decide musical phrase arc based on context"""
        
        current_time = time.time()
        arc_duration = current_time - self.arc_start_time
        
        # Check if arc is completed
        if arc_duration >= self.arc_duration:
            # Transition to new arc with strong bias towards RELEASE
            instrument = current_event.get('instrument', 'unknown')
            activity = current_event.get('rms_db', -80.0)
            
            if instrument == 'drums' and activity > -50:
                # High drum activity -> buildup or peak
                self.current_arc = random.choice([PhraseArc.BUILDUP, PhraseArc.PEAK])
            elif self.phrases_since_silence > 6:
                # Long activity -> release into contemplation (not silence)
                self.current_arc = PhraseArc.CONTEMPLATION
            else:
                # Normal flow - varied arcs, NO SILENCE
                weights = [0.3, 0.3, 0.4, 0.0, 0.0]  # buildup, peak, RELEASE, contemplation, NO SILENCE
                self.current_arc = random.choices(list(PhraseArc), weights=weights)[0]
            
            self.arc_start_time = current_time
            self.arc_duration = random.uniform(10.0, 20.0)  # Shorter arc cycles (10-20 seconds)
            
        return self.current_arc
    
    def _update_harmonic_context(self, harmonic_context: Dict):
        """Update harmonic context for phrase generation"""
        if not harmonic_context:
            return
            
        self.current_chord = harmonic_context.get('current_chord', 'C')
        key_name = harmonic_context.get('key_signature', 'C_major')
        
        # Parse key and mode
        if '_' in key_name:
            self.current_key = key_name.split('_')[0]
            mode = key_name.split('_')[1]
        else:
            self.current_key = key_name
            mode = 'major'
        
        # Scale interval patterns
        scale_intervals = {
            'major': [2, 2, 1, 2, 2, 2, 1],
            'minor': [2, 1, 2, 2, 1, 2, 2],
            'dorian': [2, 1, 2, 2, 2, 1, 2],
            'phrygian': [1, 2, 2, 2, 1, 2, 2],
            'lydian': [2, 2, 2, 1, 2, 2, 1],
            'mixolydian': [2, 2, 1, 2, 2, 1, 2],
            'locrian': [1, 2, 2, 1, 2, 2, 2],
            'harmonic_minor': [2, 1, 2, 2, 1, 3, 1],
            'melodic_minor': [2, 1, 2, 2, 2, 2, 1],
        }
        
        # Update scale degrees based on key and mode
        root_pc = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
        if self.current_key in root_pc and mode in scale_intervals:
            root = root_pc[self.current_key]
            intervals = scale_intervals[mode]
            self.scale_degrees = []
            current_pc = root
            for interval in intervals:
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
    
    def _generate_buildup_phrase(self, mode: str, voice_type: str, timestamp: float, current_event: Dict = None, temperature: float = 0.8, activity_multiplier: float = 1.0) -> MusicalPhrase:
        """Generate a phrase that builds energy
        
        Args:
            activity_multiplier: Scale phrase length (0.3-1.0 during buildup phase)
        """
        
        # Melody: 2-4 notes (short musical phrases), scaled by activity
        # Bass: 1-2 notes (sparse accompaniment), scaled by activity
        if voice_type == "melodic":
            base_length = random.randint(2, 4)
        else:
            base_length = random.randint(1, 2)
        
        # Scale phrase length by activity multiplier
        # activity_multiplier: 0.3 → 1.0 during buildup
        phrase_length = max(1, int(base_length * activity_multiplier))
        
        # Query AudioOracle with request parameters (proper suffix link traversal)
        oracle_notes = None
        print(f"🔍 AudioOracle check: audio_oracle={self.audio_oracle is not None}, has_method={hasattr(self.audio_oracle, 'generate_with_request') if self.audio_oracle else False}")  # DEBUG
        if self.audio_oracle and hasattr(self.audio_oracle, 'generate_with_request'):
            try:
                # Build request from current mode
                request = self._build_request_for_mode(mode)
                print(f"🔍 Oracle query: request={request is not None}, mode={mode}")  # DEBUG
                if request:
                    # Get recent tokens for context
                    recent_tokens = self._get_recent_human_tokens(n=3)
                    if not recent_tokens:
                        recent_tokens = []  # Empty list if no context
                    
                    # Generate from AudioOracle using request masking
                    generated_frames = self.audio_oracle.generate_with_request(
                        current_context=recent_tokens,
                        request=request,
                        max_length=phrase_length,
                        temperature=temperature
                    )
                    print(f"🔍 Oracle returned: {len(generated_frames) if generated_frames else 0} frames")  # DEBUG
                    
                    if generated_frames and len(generated_frames) > 0:
                        # Extract MIDI notes from generated frame IDs
                        oracle_notes = []
                        for frame_id in generated_frames:
                            # frame_id is an integer index into audio_frames
                            if isinstance(frame_id, int) and frame_id < len(self.audio_oracle.audio_frames):
                                frame_obj = self.audio_oracle.audio_frames[frame_id]
                                audio_data = frame_obj.audio_data
                                
                                # Try to extract MIDI note
                                if 'midi_note' in audio_data:
                                    oracle_notes.append(int(audio_data['midi_note']))
                                elif 'pitch_hz' in audio_data and audio_data['pitch_hz'] > 0:
                                    # Convert Hz to MIDI
                                    import math
                                    midi_note = int(round(69 + 12 * math.log2(audio_data['pitch_hz'] / 440.0)))
                                    oracle_notes.append(midi_note)
                        print(f"🔍 Extracted {len(oracle_notes)} notes from {len(generated_frames)} frames: {oracle_notes[:5]}...")  # DEBUG
            except Exception as e:
                # Silently fall back to random generation
                print(f"🔍 Oracle exception: {e}")  # DEBUG
                oracle_notes = None
        
        # Use MASSIVE ranges instead of narrow octave ranges
        if voice_type == "melodic":
            min_note, max_note = self.melodic_range
            # Melody: musical phrasing with breath (1.5-2.5s)
            base_timing = random.uniform(1.5, 2.5)  # More space, more musical
            timing_variation = 0.5
        else:
            min_note, max_note = self.bass_range
            # Bass: sparse, supportive (3.0-5.0s)
            base_timing = random.uniform(3.0, 5.0)  # Much sparser
            timing_variation = 1.0
        
        # DRAMATICALLY EXPANDED generation algorithm
        notes = []
        timings = []
        velocities = []
        durations = []
        
        # Use AudioOracle notes if available, otherwise generate from scratch
        if oracle_notes and len(oracle_notes) > 0:
            # Use learned patterns!
            print(f"✅ Using {len(oracle_notes)} oracle_notes for {voice_type}")
            phrase_length = min(len(oracle_notes), phrase_length)
            notes = oracle_notes[:phrase_length]
            print(f"   Phrase notes: {notes}")
            
            # Generate timings/velocities/durations for oracle notes
            for i in range(len(notes)):
                timing = base_timing + random.uniform(-timing_variation, timing_variation)
                timings.append(max(0.3, timing))
                velocity = random.randint(60, 100) if voice_type == "melodic" else random.randint(70, 110)
                velocities.append(velocity)
                # Melody: longer sustained notes; Bass: shorter punchy notes
                duration = random.uniform(0.4, 1.2) if voice_type == "melodic" else random.uniform(0.2, 0.6)
                durations.append(duration)
            
            return MusicalPhrase(
                phrase_id=f"buildup_oracle_{voice_type}_{int(timestamp)}",
                notes=notes,
                timings=timings,
                velocities=velocities,
                durations=durations,
                mode=mode,
                phrase_type="buildup",
                rhythm_pattern="oracle_pattern",
                confidence=0.9,  # High confidence for learned patterns
                timestamp=timestamp
            )
        
        # Fallback: Generate notes from scratch if AudioOracle didn't provide any
        print(f"⚠️  FALLBACK: Generating {phrase_length} random notes for {voice_type} (oracle_notes was None or empty)")
        print(f"   Range: {min_note}-{max_note}")  # DEBUG
        # Musical note selection with melodic contour
        previous_note = random.randint(min_note, max_note)
        previous_direction = 0  # Track melodic direction: -1=down, 0=static, 1=up
        print(f"   Starting note: {previous_note}")  # DEBUG
        
        for i in range(phrase_length):
            if i == 0:
                # Starting note: totally random in range
                note = random.randint(min_note, max_note)
            else:
                # More musical intervals for melody
                if voice_type == "melodic":
                    # Melodic motion: HEAVILY favor stepwise motion and small leaps
                    # This creates singable, memorable melodies
                    interval_choices = [
                        (-2, -2),   # Descending whole step (major 2nd)
                        (-1, -1),   # Descending half step (minor 2nd)
                        (1, 1),     # Ascending half step (minor 2nd)
                        (2, 2),     # Ascending whole step (major 2nd)
                        (-4, -3),   # Descending minor/major 3rd
                        (3, 4),     # Ascending minor/major 3rd
                        (-5, -5),   # Descending perfect 4th
                        (5, 5),     # Ascending perfect 4th
                        (-7, -7),   # Descending perfect 5th (rare)
                        (7, 7),     # Ascending perfect 5th (rare)
                    ]
                    # STRONG bias toward steps (75% stepwise motion)
                    probs = [0.20, 0.20, 0.20, 0.15, 0.10, 0.05, 0.04, 0.03, 0.02, 0.01]
                else:
                    # Bass: allow bigger leaps but still favor 4ths/5ths (consonant)
                    interval_choices = [
                        (-7, -7),   # Descending 5th (very common in bass)
                        (-5, -5),   # Descending 4th
                        (-4, -3),   # Descending 3rd
                        (0, 0),     # Repeat (pedal point)
                        (3, 4),     # Ascending 3rd
                        (5, 5),     # Ascending 4th
                        (7, 7),     # Ascending 5th (very common in bass)
                        (-12, -12), # Descending octave (rare)
                        (12, 12),   # Ascending octave (rare)
                    ]
                    probs = [0.25, 0.15, 0.15, 0.10, 0.10, 0.10, 0.10, 0.025, 0.025]
                
                chosen_magnitude = random.choices(interval_choices, weights=probs)[0]
                interval = random.randint(chosen_magnitude[0], chosen_magnitude[1])
                
                # Melodic contour: Favor direction changes after 2-3 steps in same direction
                # This creates arch-like melodies (up then down, or down then up)
                if i >= 2:  # Need at least 2 previous notes to detect direction
                    current_direction = 1 if interval > 0 else (-1 if interval < 0 else 0)
                    if current_direction == previous_direction and previous_direction != 0:
                        # Same direction for 2+ steps - bias toward reversing
                        if random.random() < 0.4:  # 40% chance to force direction change
                            interval = -interval  # Reverse direction
                    previous_direction = current_direction
                
                note = previous_note + interval
                
                # Wrap around range boundaries with octave jumping
                attempts = 0
                while (note < min_note or note > max_note) and attempts < 10:
                    if note < min_note:
                        note += 12
                    if note > max_note:
                        note -= 12
                    attempts += 1
                
                # Final clamping to ensure note is in range
                note = max(min_note, min(note, max_note))
                
                # Avoid repetition: if note equals previous, nudge it
                if note == previous_note and (max_note - min_note) > 2:
                    # Try to move by a small interval
                    nudge = random.choice([-2, -1, 1, 2])
                    note = previous_note + nudge
                    # Clamp again
                    note = max(min_note, min(note, max_note))
            
            notes.append(note)
            previous_note = note
            
            # Musical timing - use base_timing already calculated per voice type
            timing = base_timing + random.uniform(-timing_variation, timing_variation)
            timings.append(max(0.3, timing))
            
            # Musical velocity range (not too extreme)
            if voice_type == "melodic":
                velocity = random.randint(60, 100)  # Moderate dynamics for melody
            else:
                velocity = random.randint(70, 110)  # Slightly stronger for bass
            velocities.append(velocity)
            
            # Musical durations
            if voice_type == "melodic":
                duration = random.uniform(0.4, 1.2)  # Sustained melodic notes
            else:
                duration = random.uniform(0.2, 0.6)  # Shorter bass notes
            durations.append(duration)
        
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
    
    def _generate_peak_phrase(self, mode: str, voice_type: str, timestamp: float, current_event: Dict = None, temperature: float = 0.8, activity_multiplier: float = 1.0) -> MusicalPhrase:
        """Generate a peak phrase (high energy)"""
        
        phrase_length = random.randint(4, 30)  # Wide range: 4-30 notes
        
        # Query AudioOracle with request parameters (proper suffix link traversal)
        oracle_notes = None
        if self.audio_oracle and hasattr(self.audio_oracle, 'generate_with_request'):
            try:
                request = self._build_request_for_mode(mode)
                if request:
                    # Get recent tokens for context
                    recent_tokens = self._get_recent_human_tokens(n=3)
                    if not recent_tokens:
                        recent_tokens = []  # Empty list if no context
                    
                    generated_frames = self.audio_oracle.generate_with_request(
                        current_context=recent_tokens,
                        request=request,
                        max_length=phrase_length,
                        temperature=0.8
                    )
                    if generated_frames and len(generated_frames) > 0:
                        oracle_notes = []
                        for frame in generated_frames:
                            if 'midi_note' in frame:
                                oracle_notes.append(int(frame['midi_note']))
                            elif 'pitch_hz' in frame and frame['pitch_hz'] > 0:
                                import math
                                midi_note = int(round(69 + 12 * math.log2(frame['pitch_hz'] / 440.0)))
                                oracle_notes.append(midi_note)
            except:
                oracle_notes = None
        
        # Use MASSIVE ranges for peaks too
        if voice_type == "melodic":
            min_note, max_note = self.melodic_range
        else:
            min_note, max_note = self.bass_range
        
        notes = []
        timings = []
        velocities = []
        durations = []
        
        # DRAMATICALLY EXPANDED peak generation - use SAME algorithm as buildup
        previous_note = random.randint(min_note, max_note)
        
        for i in range(phrase_length):
            if i == 0:
                # Starting note: totally random in range
                note = random.randint(min_note, max_note)
            else:
                # Peak phrases: More energetic but still melodic
                # Allow slightly wider intervals than buildup, but keep it singable
                if voice_type == "melodic":
                    interval_magnitudes = [
                        (-3, -3),   # Descending minor 3rd
                        (-2, -2),   # Descending whole step
                        (-1, -1),   # Descending half step
                        (1, 1),     # Ascending half step
                        (2, 2),     # Ascending whole step
                        (3, 3),     # Ascending minor 3rd
                        (4, 4),     # Ascending major 3rd
                        (5, 5),     # Ascending perfect 4th
                        (7, 7),     # Ascending perfect 5th (peak energy!)
                        (-5, -5),   # Descending perfect 4th
                    ]
                    # Balanced toward steps with some expressive leaps
                    magnitude_probs = [0.12, 0.18, 0.15, 0.15, 0.15, 0.10, 0.08, 0.05, 0.03, 0.04]
                else:
                    # Bass peaks: Strong 4ths/5ths with some octaves
                    interval_magnitudes = [
                        (-7, -7),   # Descending 5th
                        (-5, -5),   # Descending 4th
                        (-12, -12), # Descending octave
                        (0, 0),     # Repeat (pedal)
                        (5, 5),     # Ascending 4th
                        (7, 7),     # Ascending 5th
                        (12, 12),   # Ascending octave (rare)
                    ]
                    magnitude_probs = [0.25, 0.20, 0.10, 0.10, 0.15, 0.15, 0.05]
                
                chosen_magnitude = random.choices(interval_magnitudes, weights=magnitude_probs)[0]
                
                interval = random.randint(chosen_magnitude[0], chosen_magnitude[1])
                note = previous_note + interval
                
                # Wrap around range boundaries
                attempts = 0
                while (note < min_note or note > max_note) and attempts < 10:
                    if note < min_note:
                        note += 12
                    if note > max_note:
                        note -= 12
                    attempts += 1
                
                # Final clamping to ensure note is in range
                note = max(min_note, min(note, max_note))
                
                # Avoid repetition: if note equals previous, nudge it
                if note == previous_note and (max_note - min_note) > 2:
                    nudge = random.choice([-2, -1, 1, 2])
                    note = previous_note + nudge
                    note = max(min_note, min(note, max_note))
            
            notes.append(note)
            previous_note = note
            
            # EXTREME variation for peaks too! (TRIPLED: 0.1-0.2 → 0.3-0.6)
            timing = random.uniform(0.3, 0.6)  # Tripled timing range
            timings.append(timing)
            
            # HIGH velocity with max variation
            velocity = random.randint(60, 127)  # High but variable
            velocities.append(velocity)
            
            # VARIABLE durations - unpredictable
            duration = random.uniform(0.15, 1.2)
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
    
    def _generate_release_phrase(self, mode: str, voice_type: str, timestamp: float, current_event: Dict = None, temperature: float = 0.8, activity_multiplier: float = 1.0) -> MusicalPhrase:
        """Generate a release phrase (settling down)"""
        
        phrase_length = random.randint(4, 30)  # Wide range: 4-30 notes
        
        # Query AudioOracle with request parameters (proper suffix link traversal)
        oracle_notes = None
        if self.audio_oracle and hasattr(self.audio_oracle, 'generate_with_request'):
            try:
                request = self._build_request_for_mode(mode)
                if request:
                    # Get recent tokens for context
                    recent_tokens = self._get_recent_human_tokens(n=3)
                    if not recent_tokens:
                        recent_tokens = []  # Empty list if no context
                    
                    generated_frames = self.audio_oracle.generate_with_request(
                        current_context=recent_tokens,
                        request=request,
                        max_length=phrase_length,
                        temperature=0.8
                    )
                    if generated_frames and len(generated_frames) > 0:
                        oracle_notes = []
                        for frame in generated_frames:
                            if 'midi_note' in frame:
                                oracle_notes.append(int(frame['midi_note']))
                            elif 'pitch_hz' in frame and frame['pitch_hz'] > 0:
                                import math
                                midi_note = int(round(69 + 12 * math.log2(frame['pitch_hz'] / 440.0)))
                                oracle_notes.append(midi_note)
            except:
                oracle_notes = None
        
        # Get proper range constraints
        if voice_type == "melodic":
            min_note, max_note = self.melodic_range
        else:
            min_note, max_note = self.bass_range
        
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
            note = midi_base + note_pc + i * interval
            
            # Clamp to range
            note = max(min_note, min(note, max_note))
            notes.append(note)
            
            # Slower timing (TRIPLED: 0.25-0.5 → 0.75-1.5)
            timing = random.uniform(0.75, 1.5)
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
    
    def _generate_contemplation_phrase(self, mode: str, voice_type: str, timestamp: float, current_event: Dict = None, temperature: float = 0.8, activity_multiplier: float = 1.0) -> MusicalPhrase:
        """Generate a contemplation phrase (meditative)"""
        
        phrase_length = random.randint(4, 30)  # Wide range: 4-30 notes
        
        # Query AudioOracle with request parameters (proper suffix link traversal)
        oracle_notes = None
        if self.audio_oracle and hasattr(self.audio_oracle, 'generate_with_request'):
            try:
                request = self._build_request_for_mode(mode)
                if request:
                    # Get recent tokens for context
                    recent_tokens = self._get_recent_human_tokens(n=3)
                    if not recent_tokens:
                        recent_tokens = []  # Empty list if no context
                    
                    generated_frames = self.audio_oracle.generate_with_request(
                        current_context=recent_tokens,
                        request=request,
                        max_length=phrase_length,
                        temperature=0.8
                    )
                    if generated_frames and len(generated_frames) > 0:
                        oracle_notes = []
                        for frame in generated_frames:
                            if 'midi_note' in frame:
                                oracle_notes.append(int(frame['midi_note']))
                            elif 'pitch_hz' in frame and frame['pitch_hz'] > 0:
                                import math
                                midi_note = int(round(69 + 12 * math.log2(frame['pitch_hz'] / 440.0)))
                                oracle_notes.append(midi_note)
            except:
                oracle_notes = None
        
        # Get proper range constraints
        if voice_type == "melodic":
            min_note, max_note = self.melodic_range
        else:
            min_note, max_note = self.bass_range
        
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
            note = midi_base + note_pc + i * interval
            
            # Clamp to range
            note = max(min_note, min(note, max_note))
            notes.append(note)
            
            # Slow, meditative timing (TRIPLED: 0.4-0.8 → 1.2-2.4)
            timing = random.uniform(1.2, 2.4)
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
