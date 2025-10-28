# behaviors.py
# AI Agent behaviors: Imitate, Contrast, Lead

import time
import random
import numpy as np
from typing import Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
from .phrase_generator import PhraseGenerator, MusicalPhrase

class BehaviorMode(Enum):
    # Original modes
    IMITATE = "imitate"
    CONTRAST = "contrast"
    LEAD = "lead"
    
    # IRCAM Phase 3: Research-based interaction modes
    SHADOW = "shadow"    # Close imitation, quick response
    MIRROR = "mirror"    # Similar with variation, phrase-aware
    COUPLE = "couple"    # Independent but complementary

@dataclass
class MusicalDecision:
    """Represents a musical decision made by the AI agent"""
    mode: BehaviorMode
    confidence: float
    target_features: np.ndarray
    musical_params: Dict
    timestamp: float
    reasoning: str
    voice_type: str = "melodic"  # "melodic" or "bass"
    instrument: str = "unknown"  # Instrument classification


# IRCAM Phase 3: Behavior Controller
class BehaviorController:
    """
    Manages IRCAM-style interaction modes
    Based on IRCAM 2023 research best practices for mixed-initiative systems
    """
    
    def __init__(self):
        self.mode = BehaviorMode.MIRROR  # Default to phrase-aware mode
        
        # Mode-specific parameters (EXAGGERATED for clarity)
        self.mode_params = {
            BehaviorMode.SHADOW: {
                'similarity_threshold': 0.95,  # EXTREME imitation
                'request_weight': 0.98,        # Almost slavish following
                'temperature': 0.3,            # VERY LOW = highly predictable
                'phrase_variation': 0.05,      # Almost no variation
                'response_delay': 0.1,         # IMMEDIATE response
                'volume_factor': 0.6,          # Much quieter (shadow in background)
                'note_density': 1.5,           # More notes (chatter)
                'pitch_offset': 0              # Same register
            },
            BehaviorMode.MIRROR: {
                'similarity_threshold': 0.7,   # Recognizable but varied
                'request_weight': 0.7,         # Balanced following
                'temperature': 1.0,            # MEDIUM = balanced
                'phrase_variation': 0.4,       # Moderate variation
                'response_delay': 'phrase_aware',  # Wait for pauses
                'volume_factor': 0.85,         # Slightly quieter
                'note_density': 1.0,           # Same density
                'pitch_offset': 0              # Same register
            },
            BehaviorMode.COUPLE: {
                'similarity_threshold': 0.05,  # EXTREMELY independent
                'request_weight': 0.2,         # Barely following
                'temperature': 1.8,            # VERY HIGH = wild exploration
                'phrase_variation': 0.9,       # Maximum variation
                'response_delay': 3.0,         # LONG delay (own time)
                'volume_factor': 1.2,          # LOUDER (equal partner)
                'note_density': 0.6,           # Sparser (giving space)
                'pitch_offset': 12             # Octave higher (distinct voice)
            },
            # Keep original modes for compatibility
            BehaviorMode.IMITATE: {
                'similarity_threshold': 0.85,
                'request_weight': 0.85,
                'temperature': 0.6,
                'phrase_variation': 0.15,
                'response_delay': 0.2,
                'volume_factor': 0.7,
                'note_density': 1.0,
                'pitch_offset': 0
            },
            BehaviorMode.CONTRAST: {
                'similarity_threshold': 0.15,
                'request_weight': 0.3,
                'temperature': 1.5,
                'phrase_variation': 0.85,
                'response_delay': 1.5,
                'volume_factor': 1.0,
                'note_density': 0.7,
                'pitch_offset': -12            # Octave lower (bass counterpoint)
            },
            BehaviorMode.LEAD: {
                'similarity_threshold': 0.5,
                'request_weight': 0.6,
                'temperature': 1.1,
                'phrase_variation': 0.5,
                'response_delay': 0.0,         # NO delay (anticipatory)
                'volume_factor': 1.1,          # Slightly louder (leading)
                'note_density': 0.8,
                'pitch_offset': 7              # Perfect 5th above (harmonic lead)
            }
        }
    
    def set_mode(self, mode: BehaviorMode):
        """Change interaction mode"""
        self.mode = mode
        print(f"🎭 Behavior mode: {mode.value}")
    
    def get_similarity_threshold(self) -> float:
        """Get pattern matching threshold for current mode"""
        return self.mode_params[self.mode]['similarity_threshold']
    
    def get_note_density(self) -> float:
        """Get note density multiplier for current mode"""
        return self.mode_params[self.mode].get('note_density', 1.0)
    
    def get_pitch_offset(self) -> int:
        """Get pitch offset in semitones for current mode"""
        return self.mode_params[self.mode].get('pitch_offset', 0)
    
    def get_response_delay(self, phrase_detector) -> Optional[float]:
        """
        Calculate response delay based on mode and phrase state
        
        Args:
            phrase_detector: PhraseBoundaryDetector instance
            
        Returns:
            Delay in seconds, or None if should not respond yet
        """
        delay = self.mode_params[self.mode]['response_delay']
        
        if delay == 'phrase_aware':
            # Wait for phrase boundary in MIRROR mode
            if phrase_detector.is_in_phrase(time.time()):
                return None  # Don't respond yet
            else:
                return 0.1  # Respond quickly at boundary
        else:
            return delay
    
    def get_volume_factor(self) -> float:
        """Get volume scaling for current mode"""
        return self.mode_params[self.mode]['volume_factor']
    
    def filter_pattern_matches(self, matches: List, similarity_scores: List[float]) -> List:
        """
        Filter AudioOracle matches based on mode similarity threshold
        
        Args:
            matches: List of pattern matches
            similarity_scores: Corresponding similarity scores
            
        Returns:
            Filtered list of matches
        """
        threshold = self.get_similarity_threshold()
        filtered = []
        for match, score in zip(matches, similarity_scores):
            if score >= threshold:
                filtered.append(match)
        return filtered


class BehaviorEngine:
    """
    Core behavior engine for AI agent
    Implements imitate, contrast, and lead behaviors
    """
    
    def __init__(self, rhythm_oracle=None, visualization_manager=None):
        self.current_mode = BehaviorMode.IMITATE
        self.mode_history = []
        self.decision_history = []
        self.visualization_manager = visualization_manager
        
        # Phrase generation
        self.phrase_generator = PhraseGenerator(rhythm_oracle, visualization_manager=visualization_manager) if rhythm_oracle else None
        self.active_phrases = {}  # Store active phrases by voice_type
        
        # Behavior parameters
        self.imitate_strength = 0.8
        self.contrast_strength = 0.6
        self.lead_strength = 0.7
        
        # Mode-specific parameters (EXAGGERATED for clarity)
        self.mode_params = {
            BehaviorMode.SHADOW: {
                'similarity_threshold': 0.95,
                'request_weight': 0.98,
                'temperature': 0.3,
                'phrase_variation': 0.05,
                'response_delay': 0.1,
                'volume_factor': 0.6,
                'note_density': 1.5,
                'pitch_offset': 0
            },
            BehaviorMode.MIRROR: {
                'similarity_threshold': 0.7,
                'request_weight': 0.7,
                'temperature': 1.0,
                'phrase_variation': 0.4,
                'response_delay': 'phrase_aware',
                'volume_factor': 0.85,
                'note_density': 1.0,
                'pitch_offset': 0
            },
            BehaviorMode.COUPLE: {
                'similarity_threshold': 0.05,
                'request_weight': 0.2,
                'temperature': 1.8,
                'phrase_variation': 0.9,
                'response_delay': 3.0,
                'volume_factor': 1.2,
                'note_density': 0.6,
                'pitch_offset': 12
            },
            BehaviorMode.IMITATE: {
                'similarity_threshold': 0.85,
                'request_weight': 0.85,
                'temperature': 0.6,
                'phrase_variation': 0.15,
                'response_delay': 0.2,
                'volume_factor': 0.7,
                'note_density': 1.0,
                'pitch_offset': 0
            },
            BehaviorMode.CONTRAST: {
                'similarity_threshold': 0.15,
                'request_weight': 0.3,
                'temperature': 1.5,
                'phrase_variation': 0.85,
                'response_delay': 1.5,
                'volume_factor': 1.0,
                'note_density': 0.7,
                'pitch_offset': -12
            },
            BehaviorMode.LEAD: {
                'similarity_threshold': 0.5,
                'request_weight': 0.6,
                'temperature': 1.1,
                'phrase_variation': 0.5,
                'response_delay': 0.0,
                'volume_factor': 1.1,
                'note_density': 0.8,
                'pitch_offset': 7
            }
        }
        
        # Timing controls - STICKY MODES for sustained character
        self.min_mode_duration = 30.0  # seconds (stay in mode longer)
        self.max_mode_duration = 90.0  # seconds (sustained identity)
        self.mode_start_time = time.time()
        self.current_mode_duration = random.uniform(30.0, 90.0)  # Target duration for current mode
        
        # Decision thresholds
        self.confidence_threshold = 0.5
        self.adaptation_rate = 0.1
        
        # Voice timing separation
        self.last_melody_time = time.time()  # Initialize to current time
        self.last_bass_time = time.time()    # Initialize to current time
        self.min_voice_separation = 0.2  # Minimum 200ms between melody and bass
        
        # Variable pause durations (balanced for responsiveness without click noise)
        self.melody_pause_min = 6.0   # seconds (increased from 10.0 for more responsiveness)
        self.melody_pause_max = 20.0  # seconds (reduced from 40.0 for more activity)
        self.bass_pause_min = 2.5     # seconds (increased from 4.0 for more responsiveness)
        self.bass_pause_max = 8.0     # seconds (reduced from 15.0 for more activity)
        self.current_melody_pause = 6.0   # Start with minimum
        self.current_bass_pause = 2.5     # Start with minimum
        
        # Emit initial mode to visualization (so viewport shows starting mode immediately)
        if self.visualization_manager:
            self.visualization_manager.emit_mode_change(
                mode=self.current_mode.value,
                duration=self.current_mode_duration,
                request_params={},
                temperature=0.8
            )
            print(f"🎭 Initial mode: {self.current_mode.value.upper()} (will persist for {self.current_mode_duration:.0f}s)")
        
    def set_pause_ranges_from_gpt_oss(self, gpt_oss_data: Dict):
        """Set pause ranges from GPT-OSS training analysis"""
        if not gpt_oss_data:
            return
        
        # Extract pause information if available
        if 'silence_strategy' in gpt_oss_data:
            silence = gpt_oss_data['silence_strategy']
            
            # Use silence metrics to inform pause ranges
            if 'avg_silence_duration' in silence:
                avg_silence = silence['avg_silence_duration']
                # Set melody pause based on average silence in training
                self.melody_pause_min = max(10.0, avg_silence * 0.5)
                self.melody_pause_max = max(20.0, avg_silence * 2.0)
                print(f"🎵 GPT-OSS: Melody pause range set to {self.melody_pause_min:.1f}-{self.melody_pause_max:.1f}s")
            
            if 'silence_percentage' in silence:
                silence_pct = silence['silence_percentage']
                # More silence in training = longer pauses
                if silence_pct > 50:
                    self.melody_pause_min = 15.0
                    self.melody_pause_max = 40.0
                    self.bass_pause_min = 6.0
                    self.bass_pause_max = 20.0
                    print(f"🎵 GPT-OSS: High silence detected, using longer pauses")
    
    def decide_behavior(self, current_event: Dict, 
                       memory_buffer, clustering) -> List[MusicalDecision]:
        """Make musical decisions based on current context"""
        current_time = time.time()
        
        # Store last event for drum-triggered evolution
        self._last_event = current_event
        
        # Check if we should change mode
        if self._should_change_mode(current_time):
            self._select_new_mode(current_event, memory_buffer, clustering)
        
        # Generate decisions based on current mode (melodic and bass)
        decisions = self._generate_decision(
            self.current_mode, current_event, memory_buffer, clustering
        )
        
        # Store decisions
        for decision in decisions:
            self.decision_history.append(decision)
        if len(self.decision_history) > 100:
            self.decision_history.pop(0)
        
        return decisions
    
    def continue_phrase_iteration(self) -> List[MusicalDecision]:
        """Continue active phrase iterations - called periodically to play next notes in phrases"""
        decisions = []
        current_time = time.time()
        
        for phrase_id, phrase_data in list(self.active_phrases.items()):
            phrase = phrase_data['phrase']
            note_index = phrase_data['note_index']
            start_time = phrase_data['start_time']
            mode = phrase_data['mode']
            voice_type = phrase_data['voice_type']
            
            # Check if it's time for the next note (use phrase timings)
            elapsed_time = current_time - start_time
            # Sum up all timings up to and including the next note
            expected_time = sum(phrase.timings[:note_index + 1])
            
            if elapsed_time >= expected_time and note_index < len(phrase.notes) - 1:
                # Move to next note
                note_index += 1
                phrase_data['note_index'] = note_index
                
                # Schedule next note with mode-specific modifications
                note = phrase.notes[note_index]
                
                # Apply note density: randomly skip notes in sparse modes (COUPLE, CONTRAST)
                note_density = 1.0  # Default
                pitch_offset = 0    # Default
                if hasattr(self, 'behavior_controller'):
                    note_density = self.behavior_controller.get_note_density()
                    pitch_offset = self.behavior_controller.get_pitch_offset()
                
                # Skip note if density check fails (makes sparse modes feel sparse)
                if random.random() > note_density:
                    continue  # Skip this note
                
                # Apply pitch offset (transpose register: COUPLE +12, CONTRAST -12, LEAD +7)
                modified_note = int(note) + pitch_offset
                modified_note = max(21, min(108, modified_note))  # Keep within MIDI range
                
                musical_params = {
                    'target_note': modified_note,
                    'velocity': phrase.velocities[note_index],
                    'duration': phrase.durations[note_index],
                    'timing': phrase.timings[note_index],
                    'phrase_id': phrase.phrase_id,
                    'phrase_type': phrase.phrase_type,
                    'phrase_scheduler': phrase_id
                }
                
                decision = MusicalDecision(
                    mode=mode,
                    confidence=phrase.confidence,
                    target_features=np.array([note]),
                    musical_params=musical_params,
                    timestamp=current_time,
                    reasoning=f"Phrase continuation: {phrase.phrase_type} note {note_index + 1}/{len(phrase.notes)}",
                    voice_type=voice_type,
                    instrument=voice_type
                )
                decisions.append(decision)
                
                # Phrase continuation (silent)
            
            # Remove phrase if completed
            if note_index >= len(phrase.notes) - 1:
                del self.active_phrases[phrase_id]
                # Phrase completed (silent)
        
        return decisions
    
    def _should_change_mode(self, current_time: float) -> bool:
        """Determine if we should change behavior mode"""
        mode_duration = current_time - self.mode_start_time
        
        # Always change after max duration
        if mode_duration >= self.max_mode_duration:
            return True
        
        # Check for drum-triggered evolution
        if hasattr(self, '_last_event'):
            instrument = self._last_event.get('instrument', 'unknown')
            if instrument == 'drums':
                # Much faster evolution for drums (0.5s minimum)
                if mode_duration >= 0.5:
                    return random.random() < 0.6  # 60% chance with drums
                return random.random() < 0.3  # Still some chance even early
        
        # Sometimes change after min duration
        if mode_duration >= self.min_mode_duration:
            return random.random() < 0.3  # 30% chance
        
        return False
    
    def _select_new_mode(self, current_event: Dict, 
                        memory_buffer, clustering):
        """Select a new behavior mode"""
        # STICKY MODES: Check if we should persist with current mode
        elapsed = time.time() - self.mode_start_time
        
        # Only switch modes if duration exceeded
        if elapsed < self.current_mode_duration:
            return  # Stay in current mode (sticky behavior)
        
        # Time to switch - weighted selection based on context
        modes = list(BehaviorMode)
        weights = self._calculate_mode_weights(current_event, memory_buffer, clustering)
        
        # Select new mode based on weights
        new_mode = random.choices(modes, weights=weights)[0]
        self.current_mode = new_mode
        self.mode_start_time = time.time()
        self.current_mode_duration = random.uniform(self.min_mode_duration, self.max_mode_duration)
        
        # Emit visualization event for mode change
        if self.visualization_manager:
            # Get actual temperature from mode parameters
            current_params = self.mode_params[self.current_mode]
            self.visualization_manager.emit_mode_change(
                mode=self.current_mode.value,
                duration=self.current_mode_duration,
                request_params=current_params,  # Send full mode params (includes temperature, weights, etc.)
                temperature=current_params['temperature']
            )
            # Also emit timeline event
            self.visualization_manager.emit_timeline_update('mode_change', mode=self.current_mode.value)
        
        # Store mode change with instrument context
        instrument = current_event.get('instrument', 'unknown')
        self.mode_history.append({
            'mode': self.current_mode,
            'timestamp': time.time(),
            'reason': 'scheduled_change',
            'instrument': instrument
        })
        
        # Print mode change for visibility (with duration)
        print(f"🎭 Mode shift: {new_mode.value.upper()} (will persist for {self.current_mode_duration:.0f}s)")
    
    def _calculate_mode_weights(self, current_event: Dict, 
                              memory_buffer, clustering) -> List[float]:
        """Calculate weights for mode selection based on context"""
        # FIXED: Base weights for ALL 6 modes
        # Order: [IMITATE, CONTRAST, LEAD, SHADOW, MIRROR, COUPLE]
        weights = [0.15, 0.15, 0.15, 0.25, 0.20, 0.10]
        
        # Adjust based on recent activity
        recent_moments = memory_buffer.get_recent_moments(10.0)
        
        if len(recent_moments) > 5:
            # High activity -> more MIRROR/COUPLE (phrase-aware, independent)
            weights = [0.10, 0.15, 0.15, 0.15, 0.25, 0.20]
        elif len(recent_moments) < 2:
            # Low activity -> more SHADOW/IMITATE (follow closely)
            weights = [0.20, 0.10, 0.10, 0.35, 0.15, 0.10]
        
        # Adjust based on onset activity
        onset_count = sum(1 for moment in recent_moments 
                         if moment.event_data.get('onset', False))
        
        if onset_count > 3:
            # High onset activity -> more LEAD/COUPLE (take initiative)
            weights = [0.10, 0.15, 0.25, 0.10, 0.15, 0.25]
        
        # DRUM-SPECIFIC: Drums trigger more contrast and lead evolution
        instrument = current_event.get('instrument', 'unknown')
        if instrument == 'drums':
            # Drums: prefer LEAD, COUPLE, CONTRAST
            weights = [0.05, 0.25, 0.30, 0.10, 0.15, 0.15]
            print(f"🥁 Drum evolution: weights={weights}")
        
        return weights
    
    def _generate_decision(self, mode: BehaviorMode, current_event: Dict,
                          memory_buffer, clustering) -> List[MusicalDecision]:
        """Generate musical phrase decisions for melody and bass partnership with rhythmic awareness"""
        decisions = []
        instrument = current_event.get('instrument', 'unknown')
        current_time = time.time()
        
        # Update phrase generator with AudioOracle if available
        if self.phrase_generator and clustering:
            self.phrase_generator.audio_oracle = clustering
        
        # If we have phrase generation, use it for musical phrases
        if self.phrase_generator:
            # Generating phrase (silent)
            # Extract harmonic context
            harmonic_context = current_event.get('harmonic_context', {})
            
            # Decide which voice to play based on timing and context
            time_since_melody = current_time - self.last_melody_time
            time_since_bass = current_time - self.last_bass_time
            
            voice_type = None
            # Use variable pause durations (randomized within range each time)
            import random
            required_melody_gap = random.uniform(self.melody_pause_min, self.melody_pause_max)
            required_bass_gap = random.uniform(self.bass_pause_min, self.bass_pause_max)
            
            if time_since_melody > required_melody_gap and time_since_bass > required_bass_gap * 0.75:
                # Both voices can play - alternate (give human lots of space)
                if not hasattr(self, '_voice_alternation_counter'):
                    self._voice_alternation_counter = 0
                
                voice_type = "melodic" if self._voice_alternation_counter % 2 == 0 else "bass"
                print(f"🎵 Voice alternation: chose {voice_type} (counter={self._voice_alternation_counter}, melody={time_since_melody:.1f}s, bass={time_since_bass:.1f}s, required: {required_melody_gap:.1f}s/{required_bass_gap:.1f}s)")
                self._voice_alternation_counter += 1
                
            elif time_since_melody > required_melody_gap:
                # Variable space between melody phrases - let human lead!
                voice_type = "melodic"
                print(f"🎵 Voice selection: melody (since_melody={time_since_melody:.1f}s, required={required_melody_gap:.1f}s)")
            elif time_since_bass > required_bass_gap:
                # Bass can fill in while waiting for melody
                voice_type = "bass"
                print(f"🎵 Voice selection: bass (since_bass={time_since_bass:.1f}s, required={required_bass_gap:.1f}s)")
            else:
                print(f"🎵 Voice selection: BLOCKED (melody={time_since_melody:.1f}s, bass={time_since_bass:.1f}s)")
            
            if voice_type:
                # Generate musical phrase with mode-specific temperature
                current_params = self.mode_params[mode]
                phrase = self.phrase_generator.generate_phrase(
                    current_event, voice_type, mode.value, harmonic_context,
                    temperature=current_params['temperature']
                )
                
                if phrase:
                    if phrase.phrase_type == "silence":
                        # Handle silence phrase - DON'T update timing (no notes sent)
                        # Silence phrases shouldn't block future generation
                        print(f"🤫 Silence phrase: {phrase.timings[0]:.1f}s (not blocking)")
                        # Return empty decisions for silence
                        return decisions
                    
                    else:
                        # Store phrase for scheduling instead of creating multiple immediate decisions
                        phrase_id = f"{voice_type}_{phrase.phrase_id}"
                        self.active_phrases[phrase_id] = {
                            'phrase': phrase,
                            'note_index': 0,
                            'start_time': current_time,
                            'mode': mode,
                            'voice_type':voice_type
                        }
                        
                        # Schedule first note immediately
                        first_note = phrase.notes[0]
                        musical_params = {
                            'target_note': int(first_note),
                            'velocity': phrase.velocities[0],
                            'duration': phrase.durations[0],
                            'timing': phrase.timings[0],
                            'phrase_id': phrase.phrase_id,
                            'phrase_type': phrase.phrase_type,
                            'phrase_scheduler': phrase_id  # Flag for phrase scheduling
                        }
                        
                        decision = MusicalDecision(
                            mode=mode,
                            confidence=phrase.confidence,
                            target_features=np.array([first_note]),
                            musical_params=musical_params,
                            timestamp=current_time,
                            reasoning=f"Rhythmic phrase start: {phrase.phrase_type}",
                            voice_type=voice_type,
                            instrument=voice_type
                        )
                        decisions.append(decision)
                        
                        # Update timing for voice immediately to prevent overlap
                        phrase_duration = (len(phrase.notes) - 1) * 0.25 + phrase.durations[-1]
                        if voice_type == "melodic":
                            self.last_melody_time = current_time + phrase_duration
                        else:
                            self.last_bass_time = current_time + phrase_duration
                        
                        # Generated phrase (silent)
        
        else:
            print(f"🎹 No phrase generator - using fallback single notes")
            # Fallback to single note generation if no phrase generator
            time_since_melody = current_time - self.last_melody_time
            time_since_bass = current_time - self.last_bass_time
            
            if time_since_melody > 0.5 and time_since_bass > 0.5:
                # Both voices can play - alternate between melody and bass
                if not hasattr(self, '_voice_alternation_counter'):
                    self._voice_alternation_counter = 0
                
                if self._voice_alternation_counter % 2 == 0:
                    # Generate melody
                    melodic_decision = self._generate_single_decision(mode, current_event, memory_buffer, clustering, "melodic")
                    decisions.append(melodic_decision)
                    self.last_melody_time = current_time
                    print(f"🎵 Voice alternation: chose melody (counter={self._voice_alternation_counter})")
                else:
                    # Generate bass
                    bass_decision = self._generate_single_decision(mode, current_event, memory_buffer, clustering, "bass")
                    decisions.append(bass_decision)
                    self.last_bass_time = current_time
                    print(f"🎵 Voice alternation: chose bass (counter={self._voice_alternation_counter})")
                
                self._voice_alternation_counter += 1
            elif time_since_melody > 0.3:
                # Melody can play
                melodic_decision = self._generate_single_decision(mode, current_event, memory_buffer, clustering, "melodic")
                decisions.append(melodic_decision)
                self.last_melody_time = current_time
            elif time_since_bass > 0.3:
                # Bass can play
                bass_decision = self._generate_single_decision(mode, current_event, memory_buffer, clustering, "bass")
                decisions.append(bass_decision)
                self.last_bass_time = current_time
        
        return decisions
    
    def _generate_single_decision(self, mode: BehaviorMode, current_event: Dict,
                                 memory_buffer, clustering, voice_type: str) -> MusicalDecision:
        """Generate a single musical decision for a specific voice type"""
        current_time = time.time()
        
        if mode == BehaviorMode.IMITATE:
            return self._imitate_decision(current_event, memory_buffer, clustering, voice_type)
        elif mode == BehaviorMode.CONTRAST:
            return self._contrast_decision(current_event, memory_buffer, clustering, voice_type)
        elif mode == BehaviorMode.LEAD:
            return self._lead_decision(current_event, memory_buffer, clustering, voice_type)
        else:
            # Fallback
            return self._imitate_decision(current_event, memory_buffer, clustering, voice_type)
    
    def _should_generate_bass(self, current_event: Dict) -> bool:
        """Determine if bass should be generated based on musical context, instrument, and timing separation"""
        current_time = time.time()
        
        # Check timing separation - only block if melody was generated very recently (within 100ms)
        time_since_melody = current_time - self.last_melody_time
        if time_since_melody < 0.1:  # Only block if melody was within last 100ms
            print(f"🔇 Bass blocked: melody too recent ({time_since_melody:.3f}s)")
            return False
        
        instrument = current_event.get('instrument', 'unknown')
        rms_db = current_event.get('rms_db', -80.0)
        
        print(f"🎸 Bass check: instrument={instrument}, rms={rms_db:.1f}dB, time_since_melody={time_since_melody:.3f}s")
        
        # Instrument-specific bass generation
        if instrument == "bass":
            # Always generate bass when bass is detected
            print(f"🎸 Bass generated: instrument=bass (always)")
            return True
        elif instrument == "drums":
            # Generate bass more often with drums
            if rms_db > -30.0:
                result = random.random() < 0.7  # 70% chance
                print(f"🎸 Bass check: drums, loud ({rms_db:.1f}dB) -> {result}")
                return result
            elif rms_db > -50.0:
                result = random.random() < 0.4  # 40% chance
                print(f"🎸 Bass check: drums, moderate ({rms_db:.1f}dB) -> {result}")
                return result
            else:
                result = random.random() < 0.1  # 10% chance
                print(f"🎸 Bass check: drums, quiet ({rms_db:.1f}dB) -> {result}")
                return result
        elif instrument == "piano":
            # Generate bass moderately with piano
            if rms_db > -30.0:
                result = random.random() < 0.5  # 50% chance
                print(f"🎸 Bass check: piano, loud ({rms_db:.1f}dB) -> {result}")
                return result
            elif rms_db > -50.0:
                result = random.random() < 0.3  # 30% chance
                print(f"🎸 Bass check: piano, moderate ({rms_db:.1f}dB) -> {result}")
                return result
        elif instrument == "speech":
            # Generate bass less frequently with speech (accompaniment)
            if rms_db > -30.0:
                result = random.random() < 0.3  # 30% chance
                print(f"🎸 Bass check: speech, loud ({rms_db:.1f}dB) -> {result}")
                return result
            elif rms_db > -50.0:
                result = random.random() < 0.2  # 20% chance
                print(f"🎸 Bass check: speech, moderate ({rms_db:.1f}dB) -> {result}")
                return result
            else:
                result = random.random() < 0.1  # 10% chance
                print(f"🎸 Bass check: speech, quiet ({rms_db:.1f}dB) -> {result}")
                return result
        else:
            # Default behavior for unknown instruments
            if rms_db > -30.0:  # Active musical input
                result = random.random() < 0.4  # 40% chance
                print(f"🎸 Bass check: {instrument}, loud ({rms_db:.1f}dB) -> {result}")
                return result
            elif rms_db > -50.0:  # Moderate activity
                result = random.random() < 0.2  # 20% chance
                print(f"🎸 Bass check: {instrument}, moderate ({rms_db:.1f}dB) -> {result}")
                return result
            else:
                result = random.random() < 0.05  # 5% chance for quiet moments
                print(f"🎸 Bass check: {instrument}, quiet ({rms_db:.1f}dB) -> {result}")
                return result

    def _should_generate_percussion(self, current_event: Dict) -> bool:
        """Determine if percussion should be generated for drums"""
        instrument = current_event.get('instrument', 'unknown')
        rms_db = current_event.get('rms_db', -80.0)
        
        if instrument == "drums":
            if rms_db > -30.0:  # Active drum input
                return random.random() < 0.8  # 80% chance
            elif rms_db > -50.0:  # Moderate activity
                return random.random() < 0.5  # 50% chance
            else:
                return random.random() < 0.2  # 20% chance for quiet moments
        elif instrument == "piano":
            # Sometimes add percussion with piano for rhythm
            if rms_db > -30.0:
                return random.random() < 0.2  # 20% chance
            else:
                return random.random() < 0.05  # 5% chance
        elif instrument == "speech":
            # Minimal percussion with speech (subtle accompaniment)
            if rms_db > -30.0:
                return random.random() < 0.1  # 10% chance
            else:
                return random.random() < 0.02  # 2% chance
        return False

    def _should_generate_harmony(self, current_event: Dict) -> bool:
        """Determine if harmony should be generated for piano"""
        instrument = current_event.get('instrument', 'unknown')
        rms_db = current_event.get('rms_db', -80.0)
        
        if instrument == "piano":
            if rms_db > -30.0:  # Active piano input
                return random.random() < 0.7  # 70% chance
            elif rms_db > -50.0:  # Moderate activity
                return random.random() < 0.4  # 40% chance
            else:
                return random.random() < 0.1  # 10% chance for quiet moments
        elif instrument == "guitar":
            # Sometimes add harmony with guitar
            if rms_db > -30.0:
                return random.random() < 0.3  # 30% chance
            else:
                return random.random() < 0.1  # 10% chance
        elif instrument == "speech":
            # Moderate harmony with speech (musical accompaniment)
            if rms_db > -30.0:
                return random.random() < 0.4  # 40% chance
            elif rms_db > -50.0:
                return random.random() < 0.2  # 20% chance
            else:
                return random.random() < 0.05  # 5% chance
        return False
    
    def _imitate_decision(self, current_event: Dict, 
                         memory_buffer, clustering, voice_type: str = "melodic") -> MusicalDecision:
        """Generate imitate behavior decision with harmonic awareness"""
        # Extract harmonic context if available
        harmonic_context = current_event.get('harmonic_context')
        
        # Find similar moments in memory
        current_features = np.array([
            current_event.get('centroid', 0.0),
            current_event.get('rms_db', -80.0),
            current_event.get('cents', 0.0),
            current_event.get('ioi', 0.0)
        ])
        
        # Normalize features (simplified)
        current_features_norm = self._normalize_features(current_features, memory_buffer)
        
        # Find similar moments
        similar_moments = memory_buffer.find_neighbors(
            current_features_norm, radius=1.0, max_results=5
        )
        
        if similar_moments:
            # Imitate the most recent similar moment
            target_moment = similar_moments[-1]
            confidence = 0.8
            reasoning = f"Imitating similar moment from {target_moment.timestamp:.1f}s ago"
        else:
            # No similar moments, use current event
            target_moment = None
            confidence = 0.4
            reasoning = "No similar moments found, using current event"
        
        # Add harmonic information to reasoning
        if harmonic_context:
            chord = harmonic_context.get('current_chord', 'N/A')
            key = harmonic_context.get('key_signature', 'N/A')
            reasoning += f" | Chord: {chord}, Key: {key}"
        
        # Generate musical parameters (with harmonic awareness)
        musical_params = self._generate_imitate_params(current_event, target_moment, voice_type, harmonic_context)
        
        return MusicalDecision(
            mode=BehaviorMode.IMITATE,
            confidence=confidence,
            target_features=current_features_norm,
            musical_params=musical_params,
            timestamp=time.time(),
            reasoning=reasoning,
            voice_type=voice_type,
            instrument=current_event.get('instrument', 'unknown')
        )
    
    def _contrast_decision(self, current_event: Dict, 
                          memory_buffer, clustering, voice_type: str = "melodic") -> MusicalDecision:
        """Generate contrast behavior decision with harmonic awareness"""
        harmonic_context = current_event.get('harmonic_context')
        
        current_features = np.array([
            current_event.get('centroid', 0.0),
            current_event.get('rms_db', -80.0),
            current_event.get('cents', 0.0),
            current_event.get('ioi', 0.0)
        ])
        
        current_features_norm = self._normalize_features(current_features, memory_buffer)
        
        # Find contrasting moments
        contrasting_moments = memory_buffer.find_distant_moments(
            current_features_norm, min_distance=1.5, max_results=3
        )
        
        if contrasting_moments:
            # Use contrasting moment as inspiration
            target_moment = random.choice(contrasting_moments)
            confidence = 0.7
            reasoning = f"Contrasting with moment from {target_moment.timestamp:.1f}s ago"
        else:
            # Generate artificial contrast
            target_moment = None
            confidence = 0.5
            reasoning = "Generating artificial contrast"
        
        # Add harmonic contrast information
        if harmonic_context:
            chord = harmonic_context.get('current_chord', 'N/A')
            reasoning += f" | Using related chords to {chord}"
        
        musical_params = self._generate_contrast_params(current_event, target_moment, voice_type, harmonic_context)
        
        return MusicalDecision(
            mode=BehaviorMode.CONTRAST,
            confidence=confidence,
            target_features=current_features_norm,
            musical_params=musical_params,
            timestamp=time.time(),
            reasoning=reasoning,
            voice_type=voice_type,
            instrument=current_event.get('instrument', 'unknown')
        )
    
    def _lead_decision(self, current_event: Dict, 
                      memory_buffer, clustering, voice_type: str = "melodic") -> MusicalDecision:
        """Generate lead behavior decision with harmonic awareness"""
        harmonic_context = current_event.get('harmonic_context')
        
        # Lead behavior creates new musical directions
        current_features = np.array([
            current_event.get('centroid', 0.0),
            current_event.get('rms_db', -80.0),
            current_event.get('cents', 0.0),
            current_event.get('ioi', 0.0)
        ])
        
        current_features_norm = self._normalize_features(current_features, memory_buffer)
        
        # Generate innovative musical parameters
        musical_params = self._generate_lead_params(current_event, voice_type, harmonic_context)
        
        confidence = 0.6
        reasoning = "Leading with innovative musical direction"
        
        # Add harmonic innovation info
        if harmonic_context:
            key = harmonic_context.get('key_signature', 'N/A')
            reasoning += f" | Exploring new territory in {key}"
        
        return MusicalDecision(
            mode=BehaviorMode.LEAD,
            confidence=confidence,
            target_features=current_features_norm,
            musical_params=musical_params,
            timestamp=time.time(),
            reasoning=reasoning,
            voice_type=voice_type,
            instrument=current_event.get('instrument', 'unknown')
        )
    
    def _normalize_features(self, features: np.ndarray, memory_buffer) -> np.ndarray:
        """Normalize features using memory buffer statistics"""
        stats = memory_buffer.get_buffer_stats()
        feature_stats = stats.get('feature_stats', {})
        
        if not feature_stats:
            return features
        
        normalized = np.zeros_like(features)
        feature_names = ['centroid', 'rms', 'cents', 'ioi']
        
        for i, name in enumerate(feature_names):
            if name in feature_stats:
                mean = feature_stats[name]['mean']
                std = max(feature_stats[name]['std'], 1e-6)
                normalized[i] = (features[i] - mean) / std
        
        return normalized
    
    def _generate_imitate_params(self, current_event: Dict, 
                               target_moment, voice_type: str = "melodic", 
                               harmonic_context: Optional[Dict] = None) -> Dict:
        """Generate musical parameters for imitate behavior with harmonic awareness"""
        params = {
            'velocity': 80,
            'duration': 1.0,
            'attack': 0.1,
            'release': 0.3,
            'filter_cutoff': 0.7,
            'modulation_depth': 0.3
        }
        
        # Add harmonic information if available
        if harmonic_context:
            params['current_chord'] = harmonic_context.get('current_chord', 'N/A')
            params['key_signature'] = harmonic_context.get('key_signature', 'N/A')
            params['scale_degrees'] = harmonic_context.get('scale_degrees', [])
            params['chord_root'] = harmonic_context.get('chord_root', 0)
            params['chord_stability'] = harmonic_context.get('stability', 0.5)
        
        # Adjust parameters based on voice type
        if voice_type == "bass":
            params['velocity'] = 70  # Slightly softer for bass
            params['duration'] = 1.5  # Longer notes for bass
            params['attack'] = 0.15  # Slightly slower attack
            params['filter_cutoff'] = 0.3  # Lower filter for bass
            params['modulation_depth'] = 0.1  # Less modulation
        
        if target_moment:
            # Use target moment's characteristics
            event_data = target_moment.event_data
            params['velocity'] = min(127, max(40, int(80 + event_data.get('rms_db', -60) * 0.5)))
            params['filter_cutoff'] = min(1.0, max(0.1, event_data.get('centroid', 1000) / 2000))
        
        return params
    
    def _generate_contrast_params(self, current_event: Dict, 
                                target_moment, voice_type: str = "melodic",
                                harmonic_context: Optional[Dict] = None) -> Dict:
        """Generate musical parameters for contrast behavior with harmonic awareness"""
        params = {
            'velocity': 60,
            'duration': 0.5,
            'attack': 0.05,
            'release': 0.8,
            'filter_cutoff': 0.3,
            'modulation_depth': 0.8
        }
        
        # Add harmonic information for contrast (use related chords)
        if harmonic_context:
            params['current_chord'] = harmonic_context.get('current_chord', 'N/A')
            params['key_signature'] = harmonic_context.get('key_signature', 'N/A')
            params['scale_degrees'] = harmonic_context.get('scale_degrees', [])
            params['chord_root'] = harmonic_context.get('chord_root', 0)
            params['use_contrasting_harmony'] = True  # Flag for FeatureMapper
        
        # Adjust parameters based on voice type
        if voice_type == "bass":
            params['velocity'] = 50  # Softer for bass contrast
            params['duration'] = 0.8  # Longer bass notes
            params['attack'] = 0.1  # Slower attack
            params['filter_cutoff'] = 0.2  # Lower filter
            params['modulation_depth'] = 0.4  # Less modulation
        
        if target_moment:
            # Contrast with target moment
            event_data = target_moment.event_data
            params['velocity'] = min(127, max(20, int(100 - event_data.get('rms_db', -60) * 0.3)))
            params['filter_cutoff'] = min(1.0, max(0.1, 1.0 - (event_data.get('centroid', 1000) / 2000)))
        
        return params
    
    def _generate_lead_params(self, current_event: Dict, voice_type: str = "melodic",
                           harmonic_context: Optional[Dict] = None) -> Dict:
        """Generate musical parameters for lead behavior with harmonic awareness"""
        # Lead behavior is more experimental
        params = {
            'velocity': random.randint(60, 100),
            'duration': random.uniform(0.3, 2.0),
            'attack': random.uniform(0.02, 0.2),
            'release': random.uniform(0.2, 1.0),
            'filter_cutoff': random.uniform(0.2, 0.9),
            'modulation_depth': random.uniform(0.4, 1.0)
        }
        
        # Add harmonic information for exploration
        if harmonic_context:
            params['current_chord'] = harmonic_context.get('current_chord', 'N/A')
            params['key_signature'] = harmonic_context.get('key_signature', 'N/A')
            params['scale_degrees'] = harmonic_context.get('scale_degrees', [])
            params['chord_root'] = harmonic_context.get('chord_root', 0)
            params['explore_harmony'] = True  # Flag for FeatureMapper
        
        # Adjust parameters based on voice type
        if voice_type == "bass":
            params['velocity'] = random.randint(50, 80)  # Lower velocity range
            params['duration'] = random.uniform(0.8, 2.5)  # Longer notes
            params['attack'] = random.uniform(0.1, 0.4)  # Slower attack
            params['filter_cutoff'] = random.uniform(0.1, 0.5)  # Lower filter range
            params['modulation_depth'] = random.uniform(0.05, 0.4)  # Less modulation
        
        return params
    
    def get_current_mode(self) -> BehaviorMode:
        """Get current behavior mode"""
        return self.current_mode
    
    def get_mode_history(self) -> List[Dict]:
        """Get history of mode changes"""
        return self.mode_history
    
    def get_decision_history(self) -> List[MusicalDecision]:
        """Get history of musical decisions"""
        return self.decision_history

    def _generate_percussion_decision(self, mode: BehaviorMode, current_event: Dict,
                                    memory_buffer, clustering) -> MusicalDecision:
        """Generate percussion-specific decision for drums"""
        # Percussion decisions focus on rhythmic patterns
        current_features = np.array([
            current_event.get('centroid', 0.0),
            current_event.get('rms_db', -80.0),
            current_event.get('zcr', 0.0),
            current_event.get('ioi', 0.0)
        ])
        
        current_features_norm = self._normalize_features(current_features, memory_buffer)
        
        # Generate percussion parameters
        musical_params = {
            'velocity': random.randint(80, 127),  # High velocity for percussion
            'duration': random.uniform(0.1, 0.5),  # Short duration
            'attack': 0.01,  # Very fast attack
            'release': 0.1,  # Short release
            'filter_cutoff': 0.9,  # Bright filter
            'modulation_depth': 0.2  # Low modulation
        }
        
        confidence = 0.7
        reasoning = "Generating percussion response to drums"
        
        return MusicalDecision(
            mode=mode,
            confidence=confidence,
            target_features=current_features_norm,
            musical_params=musical_params,
            timestamp=time.time(),
            reasoning=reasoning,
            voice_type="percussion",
            instrument=current_event.get('instrument', 'unknown')
        )

    def _generate_harmony_decision(self, mode: BehaviorMode, current_event: Dict,
                                 memory_buffer, clustering) -> MusicalDecision:
        """Generate harmony-specific decision for piano"""
        # Harmony decisions focus on chord progressions
        current_features = np.array([
            current_event.get('centroid', 0.0),
            current_event.get('rms_db', -80.0),
            current_event.get('hnr', 0.0),
            current_event.get('f0', 0.0)
        ])
        
        current_features_norm = self._normalize_features(current_features, memory_buffer)
        
        # Generate harmony parameters
        musical_params = {
            'velocity': random.randint(60, 90),  # Moderate velocity
            'duration': random.uniform(1.0, 3.0),  # Longer duration
            'attack': 0.1,  # Moderate attack
            'release': 0.5,  # Longer release
            'filter_cutoff': 0.6,  # Moderate filter
            'modulation_depth': 0.4  # Moderate modulation
        }
        
        confidence = 0.6
        reasoning = "Generating harmonic response to piano"
        
        return MusicalDecision(
            mode=mode,
            confidence=confidence,
            target_features=current_features_norm,
            musical_params=musical_params,
            timestamp=time.time(),
            reasoning=reasoning,
            voice_type="harmony",
            instrument=current_event.get('instrument', 'unknown')
        )
