#!/usr/bin/env python3
"""
PhraseGenerator
Generates rhythmic musical phrases based on learned patterns from RhythmOracle

This module bridges the gap between rhythmic analysis and musical expression:
- Uses stored rhythmic patterns from Chandra training
- Generates connected musical phrases (not single notes)
- Implements silence/buildup/release musical arcs
- Creates human-like musical timing and feel
- INTERVAL-BASED HARMONIC TRANSLATION: Preserves learned melodic gestures
  while adapting to current harmonic context
"""

import time
import random
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Interval-based harmonic translation components
from .interval_extractor import IntervalExtractor
from .harmonic_translator import HarmonicTranslator

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
    
    def __init__(self, rhythm_oracle, audio_oracle=None, enable_silence: bool = True, 
                 visualization_manager=None, harmonic_progressor=None, 
                 harmonic_context_manager=None):
        self.rhythm_oracle = rhythm_oracle
        self.audio_oracle = audio_oracle  # AudioOracle for learned patterns
        self.visualization_manager = visualization_manager
        self.harmonic_progressor = harmonic_progressor  # Learned harmonic progressions
        self.harmonic_context_manager = harmonic_context_manager  # Manual override system
        
        # INTERVAL-BASED HARMONIC TRANSLATION
        self.interval_extractor = IntervalExtractor()
        self.harmonic_translator = HarmonicTranslator(
            scale_constraint_func=self._apply_scale_constraint
        )
        
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
        self.arc_duration = 75.0  # seconds (longer cycles)
        
        # Phrase state - TUNED for better musical flow
        self.last_phrase_time = 0.0
        self.min_phrase_separation = 0.5  # seconds (increased for more space)
        self.phrases_since_silence = 0
        
        # Harmonic context for phrase generation
        self.current_key = "C"
        self.current_chord = "C"
        self.scale_degrees = [0, 2, 4, 5, 7, 9, 11]  # C major
        
        # MELODIC ENHANCEMENT: Narrower ranges for more singable, coherent melodies
        # Changed from (60, 96) to (48, 72) for better melodic focus
        self.melodic_range = (48, 72)   # C3 to C5 (2 octaves - singable lead range)
        self.bass_range = (36, 60)      # C2 to C4 (2 octaves - foundation)
        
        # Melodic constraints for more coherent lines
        self.max_leap = 7              # Max interval jump (perfect 5th)
        self.prefer_steps = 0.7        # 70% probability for stepwise motion
        self.penalize_tritone = True   # Avoid augmented 4th/diminished 5th
        self.scale_constraint = True   # Prefer diatonic notes in current key
    
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
        
        # DEBUG: Check for gesture token homogeneity
        if len(tokens) > 0:
            unique_tokens = len(set(tokens))
            total_tokens = len(tokens)
            diversity_ratio = unique_tokens / total_tokens
            print(f"🎯 GESTURE DIVERSITY: {unique_tokens}/{total_tokens} unique tokens (diversity: {diversity_ratio:.2f})")
            if diversity_ratio < 0.3:
                print(f"⚠️ LOW GESTURE DIVERSITY: Most tokens are similar ({tokens})")
            elif diversity_ratio < 0.6:
                print(f"🔍 MODERATE GESTURE DIVERSITY: Some variety ({tokens})")
            else:
                print(f"✅ GOOD GESTURE DIVERSITY: Varied tokens ({tokens})")
        
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
    
    def _get_rhythmic_phrasing_from_oracle(self, current_context: Dict = None) -> Optional[Dict]:
        """
        Query RhythmOracle for rhythmic phrasing pattern based on current context
        
        NOW TEMPO-INDEPENDENT: Queries with duration patterns and scales to current tempo.
        This provides WHEN/HOW to phrase notes (complements AudioOracle's WHAT notes)
        
        Args:
            current_context: Dict with duration_pattern, density, syncopation, pulse
            
        Returns:
            Dict with rhythmic phrasing parameters (including onsets at current tempo) or None
        """
        print(f"🥁 DEBUG: _get_rhythmic_phrasing_from_oracle called, rhythm_oracle={self.rhythm_oracle is not None}")
        
        if self.rhythm_oracle is None:
            print(f"🥁 DEBUG: RhythmOracle is None, returning None")
            return None
        
        # Build query context from recent events if not provided
        if current_context is None:
            human_events = self._get_recent_human_events(n=5)
            if not human_events:
                return None
            
            # Extract TEMPO-INDEPENDENT rhythmic features from recent input
            # Calculate duration pattern from recent onsets
            recent_onsets = [e.get('t', 0.0) for e in human_events if 't' in e]
            
            # Analyze current rhythm ratios
            duration_pattern = [2, 2, 2, 2]  # Default
            if len(recent_onsets) >= 3:
                intervals = np.diff(recent_onsets)
                if len(intervals) > 0:
                    min_interval = np.min(intervals)
                    if min_interval > 0:
                        duration_pattern = [int(round(interval / min_interval)) for interval in intervals]
            
            # Extract other tempo-free features
            densities = [e.get('density', 0.5) for e in human_events if 'density' in e]
            syncopations = [e.get('syncopation', 0.0) for e in human_events if 'syncopation' in e]
            pulses = [e.get('pulse', 4) for e in human_events if 'pulse' in e]
            
            current_context = {
                'duration_pattern': duration_pattern,
                'density': float(np.mean(densities)) if densities else 0.5,
                'syncopation': float(np.mean(syncopations)) if syncopations else 0.0,
                'pulse': int(np.median(pulses)) if pulses else 4
            }
        
        # Query RhythmOracle for similar patterns (TEMPO-FREE MATCHING)
        try:
            print(f"🥁 DEBUG: Querying RhythmOracle with context: {current_context}")
            similar_patterns = self.rhythm_oracle.find_similar_patterns(
                current_context,
                threshold=0.5  # 50% similarity threshold (density scale mismatch fixed in rhythm_oracle.py)
            )
            print(f"🥁 DEBUG: RhythmOracle returned {len(similar_patterns) if similar_patterns else 0} patterns")
            
            if similar_patterns:
                # Use most similar pattern (first in sorted list)
                best_pattern, similarity = similar_patterns[0]
                
                # Estimate current tempo from recent events (for scaling pattern to playback)
                # IMPROVED: Detect subdivision level to avoid tempo inflation
                human_events = self._get_recent_human_events(n=10)
                current_tempo = 120.0  # Default
                if len(human_events) >= 2:
                    recent_onsets = [e.get('t', 0.0) for e in human_events if 't' in e]
                    if len(recent_onsets) >= 2:
                        avg_interval = np.mean(np.diff(recent_onsets))
                        if avg_interval > 0:
                            # Calculate raw tempo (assuming each interval = 1 beat)
                            raw_tempo = 60.0 / avg_interval
                            
                            # SMART SUBDIVISION DETECTION:
                            # If raw tempo > 180, likely playing 8th/16th notes, not quarter notes
                            # Common subdivisions: 1/4, 1/8, 1/16
                            # Divide by subdivision factor to get actual quarter-note tempo
                            if raw_tempo > 350:
                                # Likely 16th notes (1/4 of quarter note)
                                current_tempo = raw_tempo / 4.0
                                subdivision = "16th notes"
                            elif raw_tempo > 180:
                                # Likely 8th notes (1/2 of quarter note)
                                current_tempo = raw_tempo / 2.0
                                subdivision = "8th notes"
                            else:
                                # Likely quarter notes or slower
                                current_tempo = raw_tempo
                                subdivision = "quarter notes"
                            
                            # Clamp to reasonable range (after subdivision correction)
                            current_tempo = max(60.0, min(200.0, current_tempo))
                            
                            print(f"🥁 DEBUG: Tempo estimation - avg_interval={avg_interval:.3f}s, "
                                  f"raw_tempo={raw_tempo:.1f} BPM, subdivision={subdivision}, "
                                  f"corrected_tempo={current_tempo:.1f} BPM")
                
                # Scale pattern to current tempo using to_absolute_timing()
                absolute_onsets = best_pattern.to_absolute_timing(current_tempo, start_time=0.0)
                
                rhythmic_phrasing = {
                    'duration_pattern': best_pattern.duration_pattern,
                    'absolute_onsets': absolute_onsets,  # Scaled to current tempo
                    'current_tempo': current_tempo,  # For reference
                    'density': best_pattern.density,
                    'syncopation': best_pattern.syncopation,
                    'pulse': best_pattern.pulse,
                    'complexity': best_pattern.complexity,
                    'pattern_type': best_pattern.pattern_type,
                    'pattern_id': best_pattern.pattern_id,
                    'confidence': best_pattern.confidence * similarity,  # Scale by similarity
                    'meter': best_pattern.meter
                }
                
                print(f"🥁 RhythmOracle phrasing: pattern={best_pattern.pattern_id}, "
                      f"duration={best_pattern.duration_pattern}, tempo={current_tempo:.1f} BPM, "
                      f"density={best_pattern.density:.2f}, similarity={similarity:.2f}")
                
                # 🎨 Emit RhythmOracle data to viewport (using dedicated rhythm_oracle_signal)
                if self.visualization_manager:
                    duration_str = str(best_pattern.duration_pattern[:8]) + ('...' if len(best_pattern.duration_pattern) > 8 else '')
                    self.visualization_manager.emit_rhythm_oracle(
                        pattern_id=best_pattern.pattern_id,
                        tempo=current_tempo,
                        density=best_pattern.density,
                        similarity=similarity,
                        duration_pattern=duration_str,
                        pulse=best_pattern.pulse,
                        syncopation=best_pattern.syncopation
                    )
                
                return rhythmic_phrasing
            else:
                print(f"🥁 RhythmOracle: No similar patterns found for context {current_context}")
                return None
                
        except Exception as e:
            print(f"⚠️ RhythmOracle query failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _apply_rhythmic_phrasing_to_timing(self, rhythmic_phrasing: Dict, num_notes: int, 
                                          voice_type: str = "melodic") -> List[float]:
        """
        Convert rhythmic phrasing parameters into actual note timing array
        
        NOW TEMPO-INDEPENDENT: Uses absolute_onsets from RhythmOracle pattern scaled to current tempo.
        This is where RhythmOracle patterns become actual MIDI timing!
        
        Args:
            rhythmic_phrasing: Dict with absolute_onsets (already scaled to current tempo)
            num_notes: Number of notes to generate timings for
            voice_type: "melodic" or "bass" for context
            
        Returns:
            List of timing values (inter-onset intervals in seconds)
        """
        if not rhythmic_phrasing:
            # Fallback to default timing
            base = 0.5 if voice_type == "melodic" else 1.0
            return [base] * num_notes
        
        # Use pre-scaled absolute onsets from RhythmOracle pattern
        if 'absolute_onsets' in rhythmic_phrasing:
            absolute_onsets = rhythmic_phrasing['absolute_onsets']
            
            # Convert to inter-onset intervals
            if len(absolute_onsets) > 0:
                intervals = []
                for i in range(len(absolute_onsets) - 1):
                    intervals.append(absolute_onsets[i+1] - absolute_onsets[i])
                
                # If we need more notes than pattern length, repeat pattern
                while len(intervals) < num_notes:
                    intervals.extend(intervals[:min(len(intervals), num_notes - len(intervals))])
                
                # Trim to exact number of notes needed
                timings = intervals[:num_notes]
                
                # BASS FIX: Ensure bass has room to breathe even with RhythmOracle patterns
                if voice_type == "bass":
                    # Scale up timings to allow for longer notes (0.2s - 2.0s)
                    # If intervals are tight, this will slow down the bass line
                    timings = [t * random.uniform(1.5, 2.5) for t in timings]
                
                return timings
        
        # Fallback: use density/syncopation if no absolute_onsets
        # (This should rarely happen now that we have duration patterns)
        tempo = rhythmic_phrasing.get('current_tempo', 120.0)
        density = rhythmic_phrasing.get('density', 0.5)
        syncopation = rhythmic_phrasing.get('syncopation', 0.0)
        
        # Calculate base inter-onset interval from tempo
        beat_duration = 60.0 / tempo  # e.g., 120 BPM = 0.5 seconds per beat
        
        # Density affects note spacing:
        # High density (0.8-1.0) → notes closer together (0.5x-0.75x beat)
        # Medium density (0.4-0.6) → notes on beat (1.0x beat)
        # Low density (0.0-0.3) → notes spread out (1.5x-2.0x beat)
        if voice_type == "bass":
             # BASS: Force sparse spacing to allow long notes
             spacing_multiplier = random.uniform(2.0, 4.0)
        elif density > 0.7:
            spacing_multiplier = random.uniform(0.5, 0.75)  # Dense
        elif density > 0.4:
            spacing_multiplier = random.uniform(0.75, 1.25)  # Medium
        else:
            spacing_multiplier = random.uniform(1.5, 2.0)  # Sparse
        
        base_ioi = beat_duration * spacing_multiplier
        
        # Apply syncopation as timing variation
        syncopation_amount = syncopation * 0.3  # Max 30% deviation
        
        timings = []
        for i in range(num_notes):
            # Apply syncopation as random offset
            if syncopation > 0.3:
                # Syncopated: add random offsets
                offset = random.uniform(-syncopation_amount, syncopation_amount) * beat_duration
            else:
                # Straight: minimal variation
                offset = random.uniform(-0.05, 0.05) * beat_duration
            
            timing = max(0.1, base_ioi + offset)  # Ensure minimum 0.1s
            timing = self._apply_timing_profile(timing, timing_variation=0.1, voice_profile=voice_profile, beat_duration=beat_duration)
            timings.append(timing)
        
        print(f"🥁 Applied rhythmic phrasing: tempo={tempo:.0f}, density={density:.2f}, "
              f"syncopation={syncopation:.2f} → avg_IOI={np.mean(timings):.3f}s")
        
        return timings
    
    def _apply_density_filter(self, notes: List[int], rhythmic_density: float, 
                              voice_type: str = "melodic") -> Tuple[List[int], List[int]]:
        """
        Probabilistically skip notes based on rhythmic_density parameter.
        
        This creates natural sparse/dense phrasing by removing notes from the phrase.
        Lower density = more notes skipped (sparser). Higher density = fewer notes skipped (denser).
        
        Args:
            notes: List of MIDI note numbers
            rhythmic_density: 0.0-1.0 (0.2=very sparse, 0.6=dense, 1.0=all notes)
            voice_type: "melodic" or "bass" (bass is more forgiving with low density)
            
        Returns:
            Tuple of (filtered_notes, kept_indices) - indices track which notes survived
        """
        if rhythmic_density >= 0.95:
            # Very high density - keep all notes
            return notes, list(range(len(notes)))
        
        # Bass should always play at least every other note for foundation
        min_keep_probability = 0.3 if voice_type == "bass" else 0.15
        
        # Convert density to keep probability
        # density=0.2 → 35% keep (sparse)
        # density=0.4 → 55% keep (medium)  
        # density=0.6 → 75% keep (dense)
        keep_probability = min_keep_probability + (rhythmic_density * 0.7)
        
        filtered_notes = []
        kept_indices = []
        
        for i, note in enumerate(notes):
            # Always keep first and last notes for phrase coherence
            if i == 0 or i == len(notes) - 1:
                filtered_notes.append(note)
                kept_indices.append(i)
            elif random.random() < keep_probability:
                filtered_notes.append(note)
                kept_indices.append(i)
        
        # Ensure minimum phrase length (at least 2 notes)
        if len(filtered_notes) < 2 and len(notes) >= 2:
            # Keep first and one random note
            filtered_notes = [notes[0], notes[random.randint(1, len(notes)-1)]]
            kept_indices = [0, random.randint(1, len(notes)-1)]
        
        skipped_count = len(notes) - len(filtered_notes)
        if skipped_count > 0:
            print(f"🎵 Density filter ({rhythmic_density:.2f}): {len(filtered_notes)}/{len(notes)} notes kept ({skipped_count} skipped)")
        
        return filtered_notes, kept_indices
    
    # ========================================
    # Melodic Constraint Helpers (Phase 2-4)
    # ========================================
    
    def _apply_scale_constraint(self, note: int, scale_degrees: List[int] = None) -> int:
        """
        Phase 2: Snap note to nearest scale degree for more diatonic melodies
        
        Args:
            note: MIDI note number to constrain
            scale_degrees: List of scale degrees [0,2,4,5,7,9,11] for major, etc.
            
        Returns:
            MIDI note snapped to nearest scale degree
        """
        if not self.scale_constraint:
            return note
        
        if scale_degrees is None:
            scale_degrees = self.scale_degrees  # Use default (C major)
        
        # Get the note's pitch class (0-11)
        pitch_class = note % 12
        
        # Find nearest scale degree
        min_distance = float('inf')
        nearest_degree = pitch_class
        
        for degree in scale_degrees:
            # Calculate distance (wrapping around octave)
            distance = min(abs(pitch_class - degree), abs(pitch_class - degree + 12), abs(pitch_class - degree - 12))
            if distance < min_distance:
                min_distance = distance
                nearest_degree = degree
        
        # Calculate adjustment
        adjustment = nearest_degree - pitch_class
        
        # Handle wrapping (e.g., B -> C is +1, not +11)
        if adjustment > 6:
            adjustment -= 12
        elif adjustment < -6:
            adjustment += 12
        
        return note + adjustment
    
    def _calculate_interval_penalty(self, interval: int) -> float:
        """
        Phase 3: Calculate penalty for large melodic leaps
        
        Args:
            interval: Interval in semitones (can be negative)
            
        Returns:
            Penalty multiplier [0.0, 1.0] where 1.0 = no penalty, 0.0 = maximum penalty
        """
        if not self.penalize_tritone and abs(interval) <= self.max_leap:
            return 1.0  # No penalty if constraints disabled
        
        abs_interval = abs(interval)
        
        # Tritone penalty (augmented 4th / diminished 5th)
        if self.penalize_tritone and abs_interval == 6:
            return 0.1  # Strong penalty for tritone
        
        # Leap penalty (beyond perfect 5th)
        if abs_interval > self.max_leap:
            # Gradually increase penalty for larger leaps
            # Perfect 5th (7) = baseline
            # Minor 6th (8) = 0.8 penalty
            # Major 6th (9) = 0.6 penalty
            # Minor 7th (10) = 0.4 penalty
            # Octave (12) = 0.2 penalty
            excess = abs_interval - self.max_leap
            penalty = max(0.2, 1.0 - (excess * 0.2))
            return penalty
        
        # Small intervals: no penalty
        return 1.0
    
    def _apply_contour_smoothing(self, notes: List[int], i: int, interval: int, 
                                 previous_direction: int) -> int:
        """
        Phase 4: Smooth melodic contour to create arch-like melodies
        
        Args:
            notes: List of notes generated so far
            i: Current index in note generation
            interval: Proposed interval for next note
            previous_direction: Previous melodic direction (-1, 0, 1)
            
        Returns:
            Adjusted interval (may be reversed for contour smoothing)
        """
        if i < 2 or len(notes) < 2:
            return interval  # Need at least 2 notes for direction detection
        
        current_direction = 1 if interval > 0 else (-1 if interval < 0 else 0)
        
        # Detect if we've been moving in same direction for 2+ steps
        if current_direction == previous_direction and previous_direction != 0:
            # Calculate "run length" - how many steps in same direction?
            run_length = 1
            for j in range(len(notes) - 1, 0, -1):
                if (notes[j] - notes[j-1]) * previous_direction > 0:
                    run_length += 1
                else:
                    break
            
            # Bias toward direction reversal based on run length
            # Longer runs = stronger reversal bias
            reversal_probability = min(0.8, 0.3 + (run_length * 0.15))
            
            if random.random() < reversal_probability:
                # Reverse direction to create arch-like contour
                return -interval
        
        return interval
    
    def _build_shadowing_request(self, mode_params: Dict = None) -> Dict:
        """
        Enhanced SHADOW: Close imitation using multiple parameters
        Combines gesture tokens, consonance, and rhythmic complexity
        
        DUAL VOCABULARY: If input has dual tokens (harmonic/percussive),
        build adaptive request based on content type
        """
        if mode_params is None:
            mode_params = {}
        
        recent_tokens = self._get_recent_human_tokens(n=3)
        avg_consonance = self._calculate_avg_consonance(n=5)
        barlow_complexity = self._get_barlow_complexity(n=3)
        
        print(f"🎯 Shadow: recent_tokens={recent_tokens}, consonance={avg_consonance:.2f}, barlow={barlow_complexity:.1f}")  # DEBUG
        
        # DUAL VOCABULARY: Check if recent events have dual tokens
        recent_human_events = self._get_recent_human_events(n=1)
        if recent_human_events:
            latest_event = recent_human_events[-1]
            content_type = latest_event.get('content_type')
            harmonic_token = latest_event.get('harmonic_token')
            percussive_token = latest_event.get('percussive_token')
            
            # If dual vocabulary is active and content type detected
            if content_type and (harmonic_token is not None or percussive_token is not None):
                # Build adaptive request based on content
                if content_type == "percussive":
                    # Drums input → request harmonic response
                    request = {
                        'percussive_token': percussive_token,
                        'response_mode': 'harmonic',
                        'parameter': 'consonance',  # Enable fallback filtering if dual vocab fails
                        'type': '==',
                        'value': 0.65,  # Match training data median (0.54-0.80 range)
                        'tolerance': 0.35,  # Wide tolerance for more matches
                        'consonance': avg_consonance,
                        'weight': mode_params.get('request_weight', 0.85)
                    }
                    print(f"🥁→🎸 Dual vocab: Drums detected → requesting harmony (perc_tok={percussive_token})")
                elif content_type == "harmonic":
                    # Guitar input → request rhythmic response
                    request = {
                        'harmonic_token': harmonic_token,
                        'response_mode': 'percussive',
                        'parameter': 'consonance',  # Enable fallback filtering if dual vocab fails
                        'type': '==',
                        'value': 0.65,  # Match training data median
                        'tolerance': 0.35,  # Wide tolerance for more matches
                        'weight': mode_params.get('request_weight', 0.85)
                    }
                    print(f"🎸→🥁 Dual vocab: Guitar detected → requesting rhythm (harm_tok={harmonic_token})")
                else:  # hybrid
                    # Both present → request contextual filling
                    request = {
                        'harmonic_token': harmonic_token,
                        'percussive_token': percussive_token,
                        'response_mode': 'hybrid',
                        'parameter': 'consonance',  # Enable fallback filtering if dual vocab fails
                        'type': '==',
                        'value': 0.65,  # Match training data median
                        'tolerance': 0.35,  # Wide tolerance for more matches
                        'consonance': avg_consonance,
                        'weight': mode_params.get('request_weight', 0.75)
                    }
                    print(f"🎵 Dual vocab: Hybrid input → contextual filling (harm={harmonic_token}, perc={percussive_token})")
                
                # Add rhythmic phrasing if RhythmOracle available
                rhythmic_phrasing = self._get_rhythmic_phrasing_from_oracle()
                if rhythmic_phrasing:
                    request['rhythmic_phrasing'] = rhythmic_phrasing
                    print(f"🥁 Added rhythmic phrasing to shadow request: {rhythmic_phrasing}")
                else:
                    print(f"🥁 No rhythmic phrasing available for shadow request")
                
                return request
        
        # Fallback: Use standard gesture token matching (single vocabulary mode)
        # Add tolerance: allow similar tokens instead of exact match
        if recent_tokens:
            request = {
                'parameter': 'gesture_token',
                'type': '==',
                'value': recent_tokens[-1],
                'tolerance': 5,  # Allow tokens within ±5 (cluster neighbors)
                'weight': mode_params.get('request_weight', 0.85),  # Slightly lower weight for range match
                'consonance_range': (max(0.0, avg_consonance - 0.2), min(1.0, avg_consonance + 0.2))  # ±0.2 consonance tolerance
            }
        else:
            # No context - use consonance only
            request = {
                'parameter': 'consonance',
                'type': '==',
                'value': 0.65,  # Match training data median
                'tolerance': 0.4,  # Very wide consonance range when no token context
                'weight': mode_params.get('request_weight', 0.5)
            }
        
        # Add rhythmic phrasing if RhythmOracle available
        rhythmic_phrasing = self._get_rhythmic_phrasing_from_oracle()
        if rhythmic_phrasing:
            request['rhythmic_phrasing'] = rhythmic_phrasing
        
        return request
    
    def _build_mirroring_request(self, mode_params: Dict = None) -> Dict:
        """
        Enhanced MIRROR: Phrase-aware complementary variation
        Contrasts melodic direction while matching harmonic character
        
        DUAL VOCABULARY: Mirrors content type - drums get harmony, guitar gets rhythm
        """
        if mode_params is None:
            mode_params = {}
        
        avg_consonance = self._calculate_avg_consonance(n=10)
        
        # DUAL VOCABULARY: Check for dual tokens
        recent_human_events = self._get_recent_human_events(n=1)
        if recent_human_events:
            latest_event = recent_human_events[-1]
            content_type = latest_event.get('content_type')
            harmonic_token = latest_event.get('harmonic_token')
            percussive_token = latest_event.get('percussive_token')
            
            if content_type and (harmonic_token is not None or percussive_token is not None):
                # Mirror mode: complement the input
                if content_type == "percussive":
                    request = {
                        'percussive_token': percussive_token,
                        'response_mode': 'harmonic',
                        'parameter': 'consonance',  # Enable fallback filtering
                        'type': '==',
                        'value': 0.65,  # Match training data
                        'tolerance': 0.35,  # Wide tolerance for more matches
                        'consonance': avg_consonance,
                        'weight': mode_params.get('request_weight', 0.7)
                    }
                elif content_type == "harmonic":
                    request = {
                        'harmonic_token': harmonic_token,
                        'response_mode': 'percussive',
                        'parameter': 'consonance',  # Enable fallback filtering
                        'type': '==',
                        'value': 0.65,  # Match training data
                        'tolerance': 0.35,  # Wide tolerance for more matches
                        'weight': mode_params.get('request_weight', 0.7)
                    }
                else:  # hybrid
                    request = {
                        'harmonic_token': harmonic_token,
                        'percussive_token': percussive_token,
                        'response_mode': 'hybrid',
                        'parameter': 'consonance',  # Enable fallback filtering
                        'type': '==',
                        'value': 0.65,  # Match training data
                        'tolerance': 0.35,  # Wide tolerance for more matches
                        'consonance': avg_consonance,
                        'weight': mode_params.get('request_weight', 0.6)
                    }
                
                # Add rhythmic phrasing if RhythmOracle available
                rhythmic_phrasing = self._get_rhythmic_phrasing_from_oracle()
                if rhythmic_phrasing:
                    request['rhythmic_phrasing'] = rhythmic_phrasing
                
                return request
        
        # Fallback: Focus on consonance matching for mirror mode
        # Add tolerance for more flexible matching
        request = {
            'parameter': 'consonance',
            'type': '==',
            'value': 0.65,  # Match training data
            'tolerance': 0.4,  # Wide tolerance for more matches
            'weight': mode_params.get('request_weight', 0.7)
        }
        
        # Add rhythmic phrasing if RhythmOracle available
        rhythmic_phrasing = self._get_rhythmic_phrasing_from_oracle()
        if rhythmic_phrasing:
            request['rhythmic_phrasing'] = rhythmic_phrasing
        
        return request
    
    def _build_coupling_request(self, mode_params: Dict = None) -> Dict:
        """
        Enhanced COUPLE: Independent but rhythmically/harmonically aware
        High consonance with complementary rhythm
        
        DUAL VOCABULARY: Loosely coupled - hybrid mode with high consonance
        """
        if mode_params is None:
            mode_params = {}
        
        # DUAL VOCABULARY: Check for dual tokens
        recent_human_events = self._get_recent_human_events(n=1)
        if recent_human_events:
            latest_event = recent_human_events[-1]
            content_type = latest_event.get('content_type')
            harmonic_token = latest_event.get('harmonic_token')
            percussive_token = latest_event.get('percussive_token')
            
            if content_type and (harmonic_token is not None or percussive_token is not None):
                # Couple mode: hybrid with high consonance constraint
                request = {
                    'harmonic_token': harmonic_token,
                    'percussive_token': percussive_token,
                    'response_mode': 'hybrid',
                    'parameter': 'consonance',  # Enable fallback filtering
                    'type': '==',
                    'value': 0.7,
                    'tolerance': 0.3,  # Accept 0.4-1.0 range (±0.3)
                    'consonance': 0.7,  # High consonance target
                    'consonance_range': (0.5, 0.9),  # Kept for compatibility
                    'weight': mode_params.get('request_weight', 0.3)  # Loose constraint
                }
                
                # Add rhythmic phrasing if RhythmOracle available
                rhythmic_phrasing = self._get_rhythmic_phrasing_from_oracle()
                if rhythmic_phrasing:
                    request['rhythmic_phrasing'] = rhythmic_phrasing
                
                return request
        
        # Fallback: High consonance constraint for couple mode
        request = {
            'parameter': 'consonance',
            'type': '>',
            'value': 0.7,
            'weight': mode_params.get('request_weight', 0.3)  # Loose constraint
        }
        
        # Add rhythmic phrasing if RhythmOracle available
        rhythmic_phrasing = self._get_rhythmic_phrasing_from_oracle()
        if rhythmic_phrasing:
            request['rhythmic_phrasing'] = rhythmic_phrasing
        
        return request
    
    def _build_request_for_mode(self, mode: str, mode_params: Dict = None) -> Optional[Dict]:
        """Build request parameters based on current behavior mode"""
        mode_upper = mode.upper()
        
        if mode_upper == 'SHADOW':
            request = self._build_shadowing_request(mode_params)
        elif mode_upper == 'MIRROR':
            request = self._build_mirroring_request(mode_params)
        elif mode_upper == 'COUPLE':
            request = self._build_coupling_request(mode_params)
        elif mode_upper in ['IMITATE', 'CONTRAST', 'LEAD']:
            # Use shadowing as fallback for compatibility
            request = self._build_shadowing_request(mode_params)
        else:
            request = None
        
        # PHASE 8: Add root hints from mode_params (from PerformanceState)
        if request and mode_params:
            request = self._add_root_hints_to_request(request, mode_params)
        
        return request
    
    def _add_root_hints_to_request(self, request: Dict, mode_params: Dict) -> Dict:
        """
        Add autonomous root progression hints to request
        
        PHASE 8: If mode_params contains root_hint_hz and tension_target
        (from PerformanceTimelineManager → PerformanceState), add them
        to the AudioOracle request for soft biasing.
        
        Args:
            request: Existing request dict
            mode_params: Parameters from calling context (may contain root hints)
            
        Returns:
            Request dict with root hints added (if available)
        """
        if 'root_hint_hz' in mode_params and mode_params['root_hint_hz']:
            request['root_hint_hz'] = mode_params['root_hint_hz']
            
        if 'tension_target' in mode_params and mode_params['tension_target'] is not None:
            request['tension_target'] = mode_params['tension_target']
        
        # Optionally override bias strength
        if 'root_bias_strength' in mode_params:
            request['root_bias_strength'] = mode_params['root_bias_strength']
        
        return request
    
    def generate_phrase(self, current_event: Dict, voice_type: str, 
                       mode: str, harmonic_context: Dict, temperature: float = 0.8, 
                       activity_multiplier: float = 1.0,
                       voice_profile: Optional[Dict[str, float]] = None) -> Optional[MusicalPhrase]:
        """Generate a musical phrase based on context and rhythm patterns
        
        Args:
            temperature: Controls randomness in Oracle generation (from mode_params)
            activity_multiplier: Multiplier from performance arc (0.0-1.0)
            voice_profile: Optional timing profile with timing_precision, syncopation_tendency, etc.
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
        
        # Update harmonic context (with learned progression if available)
        self._update_harmonic_context(harmonic_context, behavioral_mode=mode)
        
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
            phrase = self._generate_buildup_phrase(mode, voice_type, current_time, current_event, temperature, activity_multiplier, voice_profile)
            self.phrases_since_silence += 1
        elif phrase_arc == PhraseArc.PEAK:
            phrase = self._generate_peak_phrase(mode, voice_type, current_time, current_event, temperature, activity_multiplier, voice_profile)
            self.phrases_since_silence += 1
        elif phrase_arc == PhraseArc.RELEASE:
            phrase = self._generate_release_phrase(mode, voice_type, current_time, current_event, temperature, activity_multiplier, voice_profile)
            self.phrases_since_silence += 1
        else:
            phrase = self._generate_contemplation_phrase(mode, voice_type, current_time, current_event, temperature, activity_multiplier, voice_profile)
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
                weights = [0.0, 0.3, 0.3, 0.4, 0.0]  # NO SILENCE, buildup, peak, RELEASE, contemplation
                self.current_arc = random.choices(list(PhraseArc), weights=weights)[0]
            
            self.arc_start_time = current_time
            self.arc_duration = random.uniform(75.0, 300.0)  # Longer arc cycles (75-300 seconds)
        
        return self.current_arc
    
    def _update_harmonic_context(self, harmonic_context: Dict, behavioral_mode: str = None):
        """Update harmonic context for phrase generation
        
        If harmonic_progressor is available and enabled, will intelligently
        choose next chord based on learned progressions + behavioral mode.
        Otherwise, uses detected chord from harmonic_context.
        """
        if not harmonic_context:
            return
        
        # Get detected chord from context
        detected_chord = harmonic_context.get('current_chord', 'C')
        detected_confidence = harmonic_context.get('chord_confidence', 0.0)
        
        # UPDATE HARMONIC CONTEXT MANAGER with detection
        if self.harmonic_context_manager:
            self.harmonic_context_manager.set_detected_chord(detected_chord, detected_confidence)
            # Get active chord (override if active, otherwise detected)
            active_chord = self.harmonic_context_manager.get_active_chord()
        else:
            # Fallback if no context manager
            active_chord = detected_chord
        
        # LEARNED HARMONIC PROGRESSION: Choose next chord intelligently
        # Use ACTIVE chord (which respects manual overrides) as starting point
        if self.harmonic_progressor and self.harmonic_progressor.enabled and behavioral_mode:
            # Map behavioral modes (SHADOW/MIRROR/COUPLE match HarmonicProgressor modes)
            chosen_chord = self.harmonic_progressor.choose_next_chord(
                current_chord=active_chord,  # ← Use active chord (respects override)
                behavioral_mode=behavioral_mode,
                temperature=0.8  # Could be parameterized
            )
            
            # Log decision for transparency
            if chosen_chord != active_chord:
                explanation = self.harmonic_progressor.explain_choice(
                    current_chord=active_chord,
                    chosen_chord=chosen_chord,
                    behavioral_mode=behavioral_mode
                )
                print(f"🎼 Harmonic progression: {active_chord} → {chosen_chord}")
                print(f"   {explanation}")
            
            # Use chosen chord for phrase generation
            self.current_chord = chosen_chord
        else:
            # Fallback: use active chord (respects manual override)
            self.current_chord = active_chord
            
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
    
    def _generate_buildup_phrase(self, mode: str, voice_type: str, timestamp: float, current_event: Dict = None, temperature: float = 0.8, activity_multiplier: float = 1.0, voice_profile: Optional[Dict[str, float]] = None) -> MusicalPhrase:
        """Generate a phrase that builds energy
        
        Args:
            activity_multiplier: Scale phrase length (0.3-1.0 during buildup phase)
            voice_profile: Optional timing profile for precision and syncopation
        """
        
        # Melody: 2-4 notes (short musical phrases), scaled by activity
        # Bass: 3-6 notes (more substantial lines), scaled by activity
        if voice_type == "melodic":
            base_length = random.randint(2, 4)
        else:
            base_length = random.randint(3, 6)
        
        # Scale phrase length by activity multiplier
        # activity_multiplier: 0.3 → 1.0 during buildup
        phrase_length = max(1, int(base_length * activity_multiplier))
        
        # Query AudioOracle with request parameters (proper suffix link traversal)
        oracle_notes = None
        request = None  # Initialize request variable for later rhythmic phrasing check
        print(f"🔍 AudioOracle check: audio_oracle={self.audio_oracle is not None}, has_method={hasattr(self.audio_oracle, 'generate_with_request') if self.audio_oracle else False}")  # DEBUG
        if self.audio_oracle and hasattr(self.audio_oracle, 'generate_with_request'):
            try:
                # Build request from current mode
                request = self._build_request_for_mode(mode)
                print(f"🔍 Oracle query: request={request is not None}, mode={mode}")  # DEBUG
                if request:
                    # DIAGNOSTIC: Log request details
                    print(f"🔍 REQUEST DETAILS: {request}")
                    
                    # Get recent tokens for context
                    recent_tokens = self._get_recent_human_tokens(n=3)
                    if not recent_tokens:
                        recent_tokens = []  # Empty list if no context
                    
                    # DIAGNOSTIC: Log context and Oracle state
                    print(f"🔍 CONTEXT: tokens={recent_tokens}, Oracle has {len(self.audio_oracle.audio_frames) if hasattr(self.audio_oracle, 'audio_frames') else 'unknown'} frames")
                    
                    # DIAGNOSTIC: Sample frame structure check
                    if hasattr(self.audio_oracle, 'audio_frames') and len(self.audio_oracle.audio_frames) > 0:
                        sample_frame_id = list(self.audio_oracle.audio_frames.keys())[0]
                        sample_frame = self.audio_oracle.audio_frames[sample_frame_id]
                        if hasattr(sample_frame, 'audio_data'):
                            sample_keys = list(sample_frame.audio_data.keys())[:10]
                            print(f"🔍 FRAME STRUCTURE: Sample frame {sample_frame_id} audio_data keys: {sample_keys}")
                        else:
                            print(f"⚠️  FRAME WARNING: Sample frame has no audio_data attribute")
                    
                    # Generate from AudioOracle using request masking
                    generated_frames = self.audio_oracle.generate_with_request(
                        current_context=recent_tokens,
                        request=request,
                        max_length=phrase_length,
                        temperature=temperature
                    )
                    print(f"🔍 Oracle returned: {len(generated_frames) if generated_frames else 0} frames")  # DEBUG
                    
                    # DIAGNOSTIC: If empty, explain why
                    if not generated_frames or len(generated_frames) == 0:
                        print(f"⚠️  ORACLE RETURNED EMPTY - possible reasons:")
                        print(f"   - Request constraints too strict (check tolerance)")
                        print(f"   - Empty context (recent_tokens={recent_tokens})")
                        print(f"   - Missing audio_data keys in frames")
                        print(f"   - No matching patterns for mode={mode}")
                    
                    if generated_frames and len(generated_frames) > 0:
                        # INTERVAL-BASED HARMONIC TRANSLATION
                        # Extract intervals instead of absolute MIDI notes
                        print(f"🔍 Extracting intervals from {len(generated_frames)} frames: {generated_frames}")  # DEBUG
                        intervals = self.interval_extractor.extract_intervals(
                            frame_ids=generated_frames,
                            audio_frames=self.audio_oracle.audio_frames
                        )
                        print(f"🔍 Interval extractor returned: {intervals}")  # DEBUG
                        
                        if intervals:
                            # Build harmonic context for translation
                            harmonic_context = {
                                'current_chord': self.current_chord,
                                'current_key': self.current_key,
                                'scale_degrees': self.scale_degrees
                            }
                            
                            # Translate intervals to MIDI using harmonic context
                            oracle_notes = self.harmonic_translator.translate_intervals_to_midi(
                                intervals=intervals,
                                harmonic_context=harmonic_context,
                                voice_type=voice_type,
                                apply_constraints=self.scale_constraint  # Use existing flag
                            )
                            
                            print(f"✅ Interval Translation: {len(intervals)} intervals → {len(oracle_notes)} MIDI notes")
                            print(f"   Intervals: {intervals[:5]}...")
                            print(f"   MIDI output: {oracle_notes[:5]}... (in {self.current_key})")
                        else:
                            oracle_notes = None
                            print(f"⚠️  No intervals extracted from {len(generated_frames)} frames")
                        
                        # DEBUG: Safe logging even if oracle_notes is None
                        if oracle_notes:
                            print(f"🔍 Extracted {len(oracle_notes)} notes from {len(generated_frames)} frames: {oracle_notes[:5]}...")
                        else:
                            print(f"🔍 Interval extraction returned None - falling back to chord-aware generation")
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
            # Bass: varied rhythmic feel (1.0-4.0s)
            # 30% chance of faster, more active bass line
            if random.random() < 0.3:
                base_timing = random.uniform(0.8, 1.5)  # Active/walking feel
            else:
                base_timing = random.uniform(2.0, 4.0)  # Supportive feel
            timing_variation = 0.8
        
        # DRAMATICALLY EXPANDED generation algorithm
        notes = []
        timings = []
        velocities = []
        durations = []
        
        # Use AudioOracle notes if available, otherwise generate from scratch
        if oracle_notes and len(oracle_notes) >= 2:  # Use Oracle if we get 2+ notes (it generates 2-8 typically)
            # Use learned patterns!
            print(f"✅ Using {len(oracle_notes)} oracle_notes for {voice_type}")
            phrase_length = min(len(oracle_notes), phrase_length)
            notes = oracle_notes[:phrase_length]
            print(f"   Phrase notes: {notes}")
            
            # Apply density filter ONLY if we have enough notes (>3)
            # For very short phrases, keep all notes for coherence
            if len(notes) > 3:
                rhythmic_density = voice_profile.get('rhythmic_density', 0.5) if voice_profile else 0.5
                notes, kept_indices = self._apply_density_filter(notes, rhythmic_density, voice_type)
                print(f"   After density filter: {notes}")
                
                # Safety check: if filter removed too many notes, use original
                if len(notes) < 2:
                    print(f"   ⚠️ Density filter too aggressive, reverting to original notes")
                    notes = oracle_notes[:phrase_length]
            
            # RHYTHMIC PHRASING: Check if request has rhythmic_phrasing from RhythmOracle
            rhythmic_phrasing = None
            if request and 'rhythmic_phrasing' in request:
                rhythmic_phrasing = request['rhythmic_phrasing']
                print(f"🥁 Using RhythmOracle phrasing for timing generation")
            
            # Generate timings using rhythmic phrasing if available
            if rhythmic_phrasing:
                timings = self._apply_rhythmic_phrasing_to_timing(
                    rhythmic_phrasing, 
                    len(notes), 
                    voice_type
                )
            else:
                # Fallback to default timing
                beat_duration = self._get_beat_duration(current_event)
                timings = []
                for i in range(len(notes)):
                    # BASS: Force more variety in timing (mix of short and long gaps)
                    if voice_type == "bass":
                        # 40% chance of a long gap (2.0-4.0s) to let notes breathe
                        if random.random() < 0.4:
                            timing = random.uniform(2.0, 4.0)
                        else:
                            # 60% chance of active movement (0.5-1.5s)
                            timing = random.uniform(0.5, 1.5)
                    else:
                        timing = base_timing + random.uniform(-timing_variation, timing_variation)
                    
                    timing = self._apply_timing_profile(timing, timing_variation, voice_profile, beat_duration)
                    timings.append(max(0.3, timing))
            
            # Generate velocities and durations
            velocities = []
            durations = []
            for i in range(len(notes)):
                velocity = random.randint(60, 100) if voice_type == "melodic" else random.randint(70, 110)
                velocities.append(velocity)
                # Melody: longer sustained notes; Bass: shorter punchy notes
                if voice_type == "melodic":
                    duration = random.uniform(0.4, 1.2)
                else:
                    # Bass: varied length (0.2s to 2.0s)
                    duration = random.uniform(0.2, 2.0)
                    # Ensure monophonic behavior
                    if i < len(timings):
                        current_ioi = timings[i]
                        if duration > current_ioi:
                            duration = current_ioi * 0.95
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
        
        # Fallback: Generate CHORD-AWARE notes if AudioOracle didn't provide any
        print(f"⚠️  FALLBACK: Generating {phrase_length} chord-aware notes for {voice_type} (oracle_notes was None or empty)")
        print(f"   Range: {min_note}-{max_note}")  # DEBUG
        
        # Get current chord for harmonic awareness
        current_chord_notes = []
        current_chord_name = None
        if self.harmonic_context_manager:
            current_chord_name = self.harmonic_context_manager.get_active_chord()
            if current_chord_name:
                # Parse chord name to get notes (simple heuristic)
                try:
                    from music21 import chord as m21_chord
                    parsed_chord = m21_chord.Chord(current_chord_name)
                    current_chord_notes = [p.midi for p in parsed_chord.pitches]
                    print(f"   Using chord: {current_chord_name} with notes {current_chord_notes}")
                except:
                    # Fallback to current_chord stored in phrase_generator
                    print(f"   Could not parse chord {current_chord_name}, using fallback")
        
        # Chord-aware note selection
        if current_chord_notes:
            # Start with a chord tone
            chord_tones_in_range = [n for n in current_chord_notes if min_note <= n <= max_note]
            # Expand chord tones across all octaves in range
            for note in current_chord_notes:
                octave = min_note // 12
                while octave * 12 + (note % 12) <= max_note:
                    candidate = octave * 12 + (note % 12)
                    if min_note <= candidate <= max_note and candidate not in chord_tones_in_range:
                        chord_tones_in_range.append(candidate)
                    octave += 1
            
            if chord_tones_in_range:
                previous_note = random.choice(chord_tones_in_range)
                print(f"   Starting note: {previous_note} (chord tone)")
            else:
                previous_note = random.randint(min_note, max_note)
                print(f"   Starting note: {previous_note} (random, no chord tones in range)")
        else:
            previous_note = random.randint(min_note, max_note)
            print(f"   Starting note: {previous_note} (random, no chord detected)")
        
        previous_direction = 0  # Track melodic direction: -1=down, 0=static, 1=up
        
        for i in range(phrase_length):
            if i == 0:
                # Starting note: chord tone if available
                if current_chord_notes and len(current_chord_notes) > 0:
                    chord_tones_in_range = [n for n in current_chord_notes if min_note <= n <= max_note]
                    # Expand across octaves
                    for note_val in current_chord_notes:
                        octave = min_note // 12
                        while octave * 12 + (note_val % 12) <= max_note:
                            candidate = octave * 12 + (note_val % 12)
                            if min_note <= candidate <= max_note and candidate not in chord_tones_in_range:
                                chord_tones_in_range.append(candidate)
                            octave += 1
                    note = random.choice(chord_tones_in_range) if chord_tones_in_range else random.randint(min_note, max_note)
                else:
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
                
                # PHASE 4: Apply contour smoothing (arch-like melodies)
                if voice_type == "melodic":
                    interval = self._apply_contour_smoothing(notes, i, interval, previous_direction)
                
                # PHASE 3: Apply interval leap penalty (prefer small intervals)
                # Re-roll interval if it has high penalty
                if voice_type == "melodic":
                    penalty = self._calculate_interval_penalty(interval)
                    if penalty < 1.0 and random.random() > penalty:
                        # Penalty failed - choose smaller interval
                        # Fall back to stepwise motion (most melodic)
                        interval = random.choice([-2, -1, 1, 2])  # Step only
                
                # Track melodic direction for contour analysis
                if i >= 2:
                    current_direction = 1 if interval > 0 else (-1 if interval < 0 else 0)
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
                
                # PHASE 2: Apply scale constraint (snap to diatonic notes)
                if voice_type == "melodic":
                    note = self._apply_scale_constraint(note)
                
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
            beat_duration = self._get_beat_duration(current_event)
            
            # BASS: Force more variety in timing (mix of short and long gaps)
            if voice_type == "bass":
                # 40% chance of a long gap (2.0-4.0s) to let notes breathe
                if random.random() < 0.4:
                    timing = random.uniform(2.0, 4.0)
                else:
                    # 60% chance of active movement (0.5-1.5s)
                    timing = random.uniform(0.5, 1.5)
            else:
                timing = base_timing + random.uniform(-timing_variation, timing_variation)
            
            timing = self._apply_timing_profile(timing, timing_variation, voice_profile, beat_duration)
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
                # Bass: varied length (0.2s to 2.0s)
                duration = random.uniform(0.2, 2.0)
                # Ensure monophonic behavior (no overlap)
                current_ioi = timings[-1]
                if duration > current_ioi:
                    duration = current_ioi * 0.95
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
    
    def _generate_peak_phrase(self, mode: str, voice_type: str, timestamp: float, current_event: Dict = None, temperature: float = 0.8, activity_multiplier: float = 1.0, voice_profile: Optional[Dict[str, float]] = None) -> MusicalPhrase:
        """Generate a peak phrase (high energy)"""
        
        # Scale phrase length by rhythmic_density and activity
        # Dense voices (0.8): 6-16 notes, Sparse voices (0.2): 2-4 notes
        rhythmic_density = voice_profile.get('rhythmic_density', 0.5) if voice_profile else 0.5
        base_length = int(4 + rhythmic_density * 12)  # 4-16 notes based on density
        phrase_length = int(base_length * activity_multiplier)  # Scale by arc activity
        phrase_length = max(2, min(phrase_length, 20))  # Clamp to 2-20 notes
        
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
                        # INTERVAL-BASED HARMONIC TRANSLATION (PEAK)
                        intervals = self.interval_extractor.extract_intervals(
                            frame_ids=generated_frames,
                            audio_frames=self.audio_oracle.audio_frames
                        )
                        
                        if intervals:
                            harmonic_context = {
                                'current_chord': self.current_chord,
                                'current_key': self.current_key,
                                'scale_degrees': self.scale_degrees
                            }
                            
                            oracle_notes = self.harmonic_translator.translate_intervals_to_midi(
                                intervals=intervals,
                                harmonic_context=harmonic_context,
                                voice_type=voice_type,
                                apply_constraints=self.scale_constraint
                            )
                        else:
                            oracle_notes = None
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
        previous_direction = 0  # Track melodic direction: -1=down, 0=static, 1=up
        
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
                
                # PHASE 4: Apply contour smoothing (peak phrases still get arch-like shape)
                if voice_type == "melodic" and i >= 2:
                    interval = self._apply_contour_smoothing(notes, i, interval, 
                                                            previous_direction if 'previous_direction' in locals() else 0)
                
                # PHASE 3: Apply interval leap penalty (even peaks should be singable)
                if voice_type == "melodic":
                    penalty = self._calculate_interval_penalty(interval)
                    if penalty < 1.0 and random.random() > penalty:
                        # Fall back to stepwise motion
                        interval = random.choice([-2, -1, 1, 2])
                
                # Track melodic direction
                if i >= 2:
                    current_direction = 1 if interval > 0 else (-1 if interval < 0 else 0)
                    previous_direction = current_direction
                
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
                
                # PHASE 2: Apply scale constraint (snap to diatonic notes)
                if voice_type == "melodic":
                    note = self._apply_scale_constraint(note)
                
                # Avoid repetition: if note equals previous, nudge it
                if note == previous_note and (max_note - min_note) > 2:
                    nudge = random.choice([-2, -1, 1, 2])
                    note = previous_note + nudge
                    note = max(min_note, min(note, max_note))
                    # Re-apply scale constraint after nudge
                    if voice_type == "melodic":
                        note = self._apply_scale_constraint(note)
            
            notes.append(note)
            previous_note = note
            
            # EXTREME variation for peaks too! (TRIPLED: 0.1-0.2 → 0.3-0.6)
            beat_duration = self._get_beat_duration(current_event)
            
            if voice_type == "bass":
                # Bass peaks: mix of driving fast notes and powerful long notes
                if random.random() < 0.3:
                    timing = random.uniform(1.5, 3.0)  # Long powerful notes
                else:
                    timing = random.uniform(0.3, 0.8)  # Driving fast notes
            else:
                timing = random.uniform(0.3, 0.6)  # Tripled timing range
                
            timing = self._apply_timing_profile(timing, timing_variation=0.15, voice_profile=voice_profile, beat_duration=beat_duration)
            timings.append(timing)
            
            # HIGH velocity with max variation
            velocity = random.randint(60, 127)  # High but variable
            velocities.append(velocity)
            
            # VARIABLE durations - unpredictable
            if voice_type == "melodic":
                duration = random.uniform(0.15, 1.2)
            else:
                # Bass: varied length (0.2s to 2.0s)
                duration = random.uniform(0.2, 2.0)
                # Ensure monophonic behavior
                current_ioi = timings[-1]
                if duration > current_ioi:
                    duration = current_ioi * 0.95
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
    
    def _generate_release_phrase(self, mode: str, voice_type: str, timestamp: float, current_event: Dict = None, temperature: float = 0.8, activity_multiplier: float = 1.0, voice_profile: Optional[Dict[str, float]] = None) -> MusicalPhrase:
        """Generate a release phrase (settling down)"""
        
        # Scale phrase length by rhythmic_density and activity
        # Dense voices: 4-12 notes, Sparse voices: 2-3 notes
        rhythmic_density = voice_profile.get('rhythmic_density', 0.5) if voice_profile else 0.5
        base_length = int(3 + rhythmic_density * 9)  # 3-12 notes based on density
        phrase_length = int(base_length * activity_multiplier)  # Scale by arc activity
        phrase_length = max(2, min(phrase_length, 15))  # Clamp to 2-15 notes
        
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
                        # INTERVAL-BASED HARMONIC TRANSLATION (RELEASE)
                        intervals = self.interval_extractor.extract_intervals(
                            frame_ids=generated_frames,
                            audio_frames=self.audio_oracle.audio_frames
                        )
                        
                        if intervals:
                            harmonic_context = {
                                'current_chord': self.current_chord,
                                'current_key': self.current_key,
                                'scale_degrees': self.scale_degrees
                            }
                            
                            oracle_notes = self.harmonic_translator.translate_intervals_to_midi(
                                intervals=intervals,
                                harmonic_context=harmonic_context,
                                voice_type=voice_type,
                                apply_constraints=self.scale_constraint
                            )
                        else:
                            oracle_notes = None
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
            beat_duration = self._get_beat_duration(current_event)
            
            if voice_type == "bass":
                # Bass release: generally slower, allowing for long decay
                timing = random.uniform(1.0, 3.0)
            else:
                timing = random.uniform(0.75, 1.5)
                
            timing = self._apply_timing_profile(timing, timing_variation=0.375, voice_profile=voice_profile, beat_duration=beat_duration)
            timings.append(timing)
            
            # Decreasing velocity
            velocity = 80 - i * 8
            velocities.append(velocity)
            
            # Longer durations
            if voice_type == "melodic":
                duration = 1.0 + i * 0.2
            else:
                # Bass: varied length (0.2s to 2.0s)
                duration = random.uniform(0.2, 2.0)
                # Ensure monophonic behavior
                current_ioi = timings[-1]
                if duration > current_ioi:
                    duration = current_ioi * 0.95
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
    
    def _generate_contemplation_phrase(self, mode: str, voice_type: str, timestamp: float, current_event: Dict = None, temperature: float = 0.8, activity_multiplier: float = 1.0, voice_profile: Optional[Dict[str, float]] = None) -> MusicalPhrase:
        """Generate a contemplation phrase (meditative)"""
        
        # Scale phrase length by rhythmic_density - contemplation should be sparse
        # Dense voices: 3-8 notes, Sparse voices: 2-3 notes
        rhythmic_density = voice_profile.get('rhythmic_density', 0.5) if voice_profile else 0.5
        base_length = int(2 + rhythmic_density * 6)  # 2-8 notes based on density
        phrase_length = int(base_length * activity_multiplier)  # Scale by arc activity
        phrase_length = max(2, min(phrase_length, 10))  # Clamp to 2-10 notes
        
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
                        # INTERVAL-BASED HARMONIC TRANSLATION (CONTEMPLATION)
                        intervals = self.interval_extractor.extract_intervals(
                            frame_ids=generated_frames,
                            audio_frames=self.audio_oracle.audio_frames
                        )
                        
                        if intervals:
                            harmonic_context = {
                                'current_chord': self.current_chord,
                                'current_key': self.current_key,
                                'scale_degrees': self.scale_degrees
                            }
                            
                            oracle_notes = self.harmonic_translator.translate_intervals_to_midi(
                                intervals=intervals,
                                harmonic_context=harmonic_context,
                                voice_type=voice_type,
                                apply_constraints=self.scale_constraint
                            )
                        else:
                            oracle_notes = None
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
            beat_duration = self._get_beat_duration(current_event)
            
            if voice_type == "bass":
                timing = random.uniform(1.5, 4.0) # Very spacious for bass
            else:
                timing = random.uniform(1.2, 2.4)
                
            timing = self._apply_timing_profile(timing, timing_variation=0.6, voice_profile=voice_profile, beat_duration=beat_duration)
            timings.append(timing)
            
            # Soft velocity
            velocity = 40 + random.randint(-5, 15)
            velocities.append(velocity)
            
            # Long, sustained durations
            if voice_type == "bass":
                # Use user-requested range even in contemplation, but biased toward long
                duration = random.uniform(0.5, 3.0)
                # Ensure monophonic behavior
                current_ioi = timings[-1]
                if duration > current_ioi:
                    duration = current_ioi * 0.95
            else:
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
    
    def _get_beat_duration(self, current_event: Optional[Dict] = None) -> float:
        """
        Extract beat duration from current tempo
        
        Args:
            current_event: Event dict with rhythmic_context containing current_tempo
            
        Returns:
            Beat duration in seconds (default 0.5s = 120 BPM if no tempo available)
        """
        if current_event and 'rhythmic_context' in current_event and current_event['rhythmic_context'] is not None:
            tempo = current_event['rhythmic_context'].get('current_tempo', 120.0)
            if tempo > 0:
                # Convert BPM to beat duration in seconds
                beat_duration = 60.0 / tempo
                return beat_duration
        
        # Default to 120 BPM (0.5s per beat)
        return 0.5
    
    def _apply_timing_profile(self, base_timing: float, timing_variation: float, 
                             voice_profile: Optional[Dict[str, float]], 
                             beat_duration: float = 0.5) -> float:
        """Apply timing precision and syncopation tendency to a timing value.
        
        Args:
            base_timing: Base timing value (IOI in seconds)
            timing_variation: Base variation already applied
            voice_profile: Voice timing profile with timing_precision and syncopation_tendency
            beat_duration: Duration of one beat in seconds (default 0.5s = 120 BPM)
            
        Returns:
            Modified timing value with precision and syncopation applied
        """
        if not voice_profile:
            return base_timing
        
        # Get timing parameters (with defaults)
        timing_precision = voice_profile.get('timing_precision', 0.5)
        syncopation = voice_profile.get('syncopation_tendency', 0.3)
        
        # Apply IOI variance for tempo consistency
        # High precision (0.9) = ±3% variance (steady tempo)
        # Low precision (0.3) = ±21% variance (rubato allowed)
        ioi_variance_factor = (1.0 - timing_precision) * 0.3  # 0-30% variance range
        timing_with_variance = base_timing * (1.0 + random.uniform(-ioi_variance_factor, ioi_variance_factor))
        
        # Apply humanization jitter based on timing_precision
        # precision=1.0 (tight) → minimal jitter (~5ms)
        # precision=0.5 (medium) → moderate jitter (~25ms)
        # precision=0.0 (loose) → high jitter (~50ms)
        if timing_precision < 1.0:
            max_jitter_ms = (1.0 - timing_precision) * 50.0  # Up to 50ms for loose timing
            jitter_seconds = random.uniform(-max_jitter_ms, max_jitter_ms) / 1000.0
            timing_with_variance += jitter_seconds
        
        # NO BEAT QUANTIZATION - let RhythmOracle timing pass through naturally
        # This preserves the learned rhythmic patterns from Brandtsegg ratio analysis
        # and creates off-grid, humanized timing instead of grid-locked notes
        
        return max(0.1, timing_with_variance)  # Ensure minimum 0.1s
