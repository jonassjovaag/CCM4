import time
import random
from typing import Dict

class RhythmEngine:
    def __init__(self, base_bpm: float = 60.0, logger=None):
        self.base_bpm = base_bpm
        self.logger = logger
        self.time_start = time.time()

        # Musical patterns for voice2 (in beat divisions)
        self.voice2_patterns = [
            [2.25, 2.25, 10.5],           # Short-short-long
            [4.5, 3.25, 2.25],           # Long-short-short
            [4.375, 10.975, 10.625],        # Medium-medium-short
            [1.125, 2.125, 5.25, 20.5]    # Quick-quick-short-long
        ]
        
        # Add pattern variations with Fibonacci-based durations
        self.voice2_patterns.extend([
            [1.618, 2.618, 4.236],        # Golden ratio based
            [3.0, 5.0, 8.0],              # Fibonacci steps
            [1.0, 1.0, 2.0, 3.0, 5.0]     # Accumulating sequence
        ])
        
        # Add intensity levels for pattern selection
        self.intensity_levels = {
            'low': [0, 1],        # Index of calmer patterns
            'medium': [2, 3, 4],  # More active patterns
            'high': [5, 6, 7]     # Most complex patterns
        }
        
        self.current_intensity = 'low'
        self.response_sensitivity = 0.5  # How responsive to phase changes (0-1)
        
        # Pattern control variables
        self.current_pattern_index = 0
        self.pattern_position = 0
        self.pattern_repetitions = 0
        self.max_repetitions = 4

        # Base intervals for other voices (with variation for voice1)
        self.voice_beat_divisions = {
            'voice1': 8.0,    # Base 8-beat intervals with variation
            'voice2': 3.0,    # Reduced from 4.5 to 3.0 for more frequent activation  
            'voice3': 16.0    # Consistent 16-beat intervals
        }
        
        # Voice1 rhythmic variations (Fibonacci-based)
        self.voice1_variations = [
            6.0,    # Shorter burst
            8.0,    # Standard
            10.0,   # Longer contemplation
            5.0,    # Quick response
            13.0,   # Extended silence
            7.0,    # Slight variation
            11.0    # Medium extension
        ]
        self.voice1_variation_weights = [0.2, 0.3, 0.15, 0.15, 0.1, 0.15, 0.1]  # Favor standard and shorter
        self.voice1_last_variation_time = 0
        
        # Phase-dependent timing multipliers for increased activity during complex phases
        self.phase_timing_multipliers = {
            'awakening': 1.0,      # Normal timing
            'conversation': 0.7,   # Faster for more interaction
            'climax': 0.3,         # Much faster - peak activity
            'reflection': 0.6,     # Still active but calming
            'resolution': 1.0      # Return to normal, not slower
        }
        
        # Track last phase for transition detection
        self.last_phase = 'awakening'
        self.phase_transition_time = 0

        # Initialize next trigger times
        self.next_trigger_times = {
            'voice1': self.time_start,
            'voice2': self.time_start,
            'voice3': self.time_start
        }

        self.evolution_start_time = time.time()
        self.evolution_duration = 180  # seconds before evolving
    def get_next_voice2_division(self) -> float:
        pattern = self.voice2_patterns[self.current_pattern_index]
        division = pattern[self.pattern_position]
        
        # Update pattern position
        self.pattern_position += 1
        if self.pattern_position >= len(pattern):
            self.pattern_position = 0
            self.pattern_repetitions += 1
            
        # Change pattern after max repetitions
        if self.pattern_repetitions >= self.max_repetitions:
            self.pattern_repetitions = 0
            self.current_pattern_index = (self.current_pattern_index + 1) % len(self.voice2_patterns)
        
        return division
    
    def get_voice1_variation(self, current_phase: str) -> float:
        """Get varied beat interval for voice1 based on musical context"""
        current_time = time.time()
        
        # Choose variation based on phase characteristics
        if current_phase == 'awakening':
            # Favor longer, more contemplative intervals
            weights = [0.1, 0.2, 0.4, 0.1, 0.15, 0.1, 0.15]  # Favor 10.0 and 13.0
        elif current_phase == 'conversation':
            # Favor shorter, more responsive intervals
            weights = [0.3, 0.3, 0.1, 0.25, 0.05, 0.2, 0.05]  # Favor 6.0, 8.0, 5.0, 7.0
        elif current_phase == 'climax':
            # Favor very short intervals for maximum activity
            weights = [0.35, 0.25, 0.05, 0.3, 0.02, 0.25, 0.03]  # Favor 6.0, 5.0, 7.0
        elif current_phase == 'reflection':
            # Mix of short and medium intervals
            weights = [0.2, 0.25, 0.2, 0.15, 0.1, 0.2, 0.15]  # Balanced but favor standard
        else:  # resolution
            # Return to longer, peaceful intervals
            weights = [0.15, 0.3, 0.25, 0.1, 0.2, 0.15, 0.1]  # Favor 8.0, 10.0, 13.0
        
        # Add some randomness but preserve musical sense
        import random
        variation = random.choices(self.voice1_variations, weights=weights)[0]
        
        # Occasionally apply micro-timing adjustments (±10%)
        if random.random() < 0.3:  # 30% chance
            micro_adjustment = random.uniform(0.9, 1.1)
            variation *= micro_adjustment
        
        self.voice1_last_variation_time = current_time
        return variation

    def beats_to_seconds(self, beats: float) -> float:
        """Convert musical beats to seconds based on BPM"""
        return (60.0 / self.base_bpm) * beats

    def get_voice_timing_info(self, voice: str, current_phase: str = 'conversation') -> Dict:
        """Get timing information for a specific voice, adjusted for current performance phase"""
        interval = (self.get_next_voice2_division() if voice == 'voice2' 
                   else self.voice_beat_divisions[voice])
        
        # Apply phase-dependent timing multiplier
        phase_multiplier = self.phase_timing_multipliers.get(current_phase, 1.0)
        adjusted_interval = interval * phase_multiplier
        
        return {
            'bpm': self.base_bpm,
            'interval': self.beats_to_seconds(adjusted_interval),
            'next_trigger': self.next_trigger_times[voice],
            'phase': current_phase,
            'phase_multiplier': phase_multiplier
        }

    def should_trigger_voice(self, voice: str, current_phase: str = 'conversation') -> bool:
        current_time = time.time()
        
        # Detect phase transitions and apply immediate adjustments
        if current_phase != self.last_phase:
            self.handle_phase_transition(current_phase)
            self.last_phase = current_phase
            
        if current_time >= self.next_trigger_times[voice]:
            # Use beat intervals with variation for voice1
            if voice == 'voice1':
                beat_interval = self.get_voice1_variation(current_phase)
            else:
                beat_interval = self.voice_beat_divisions[voice]
            
            # Apply phase-dependent timing multiplier
            phase_multiplier = self.phase_timing_multipliers.get(current_phase, 1.0)
            adjusted_beat_interval = beat_interval * phase_multiplier
            
            # Apply rhythmic interlocking - voices respond to each other's timing
            adjusted_beat_interval = self.apply_rhythmic_interlocking(voice, adjusted_beat_interval)
            
            time_interval = self.beats_to_seconds(adjusted_beat_interval)
            self.next_trigger_times[voice] = current_time + time_interval

            if self.logger:
                self.logger.log_rhythm_event(
                    'voice_trigger',
                    voice,
                    self.base_bpm,
                    adjusted_beat_interval,
                    time_interval,
                    [],  # Simplified pattern logging
                    current_time
                )
            return True
        return False
    
    def handle_phase_transition(self, new_phase: str):
        """Handle immediate adjustments when phase transitions occur"""
        current_time = time.time()
        self.phase_transition_time = current_time
        
        # Log phase transition
        if self.logger:
            self.logger.log_system_event(
                event_type="phase_transition",
                voice="system",
                trigger_reason=f"{self.last_phase} -> {new_phase}",
                activity_level=0.0,  # Not applicable for phase transitions
                phase=new_phase,
                additional_data=f"time={current_time:.1f}"
            )
        
        # Immediate timing adjustments for dramatic transitions
        transition_urgency = self.get_transition_urgency(self.last_phase, new_phase)
        
        if transition_urgency > 0.5:  # Major transition
            # Force earlier triggers for all voices to create immediate response
            urgency_factor = 0.5 + (transition_urgency * 0.3)  # 0.5 to 0.8
            for voice in self.next_trigger_times:
                remaining_time = self.next_trigger_times[voice] - current_time
                if remaining_time > 0:
                    # Reduce remaining time based on urgency
                    new_remaining = remaining_time * urgency_factor
                    self.next_trigger_times[voice] = current_time + new_remaining
    
    def get_transition_urgency(self, from_phase: str, to_phase: str) -> float:
        """Calculate how dramatic a phase transition is (0.0 to 1.0)"""
        # Define phase intensity levels
        phase_intensities = {
            'awakening': 0.2,
            'conversation': 0.4,
            'climax': 1.0,
            'reflection': 0.6,
            'resolution': 0.3
        }
        
        from_intensity = phase_intensities.get(from_phase, 0.5)
        to_intensity = phase_intensities.get(to_phase, 0.5)
        
        # Calculate urgency based on intensity change
        intensity_change = abs(to_intensity - from_intensity)
        
        # Special cases for dramatic transitions
        if (from_phase == 'conversation' and to_phase == 'climax') or \
           (from_phase == 'climax' and to_phase == 'reflection'):
            return 1.0  # Maximum urgency
        elif to_phase == 'climax':
            return 0.8  # High urgency when entering climax
        elif from_phase == 'climax':
            return 0.7  # High urgency when leaving climax
        else:
            return intensity_change  # Based on intensity difference

    def apply_rhythmic_interlocking(self, voice: str, beat_interval: float) -> float:
        """Apply rhythmic interlocking patterns for collaborative feel"""
        # Track recent triggers for rhythmic analysis
        if not hasattr(self, 'recent_triggers'):
            self.recent_triggers = {}
        
        current_time = time.time()
        
        # 40% chance to apply interlocking, 60% chance for independent timing
        if random.random() > 0.4:
            return beat_interval
            
        # Look for recent triggers from other voices
        other_voices = [v for v in self.voice_beat_divisions.keys() if v != voice]
        recent_other_triggers = []
        
        for other_voice in other_voices:
            if other_voice in self.recent_triggers:
                trigger_time = self.recent_triggers[other_voice]
                # Consider triggers from last 8 seconds
                if current_time - trigger_time < 8.0:
                    time_since = current_time - trigger_time
                    recent_other_triggers.append((other_voice, time_since))
        
        if recent_other_triggers:
            # Sort by most recent first
            recent_other_triggers.sort(key=lambda x: x[1])
            
            # Interlocking patterns based on most recent trigger
            most_recent_voice, time_since_trigger = recent_other_triggers[0]
            
            # Create interlocking rhythms
            if time_since_trigger < 1.0:
                # Very recent trigger - create syncopation (offset timing)
                beat_interval *= 0.7  # Trigger sooner for tight interlocking
            elif time_since_trigger < 3.0:
                # Recent trigger - create complementary rhythm
                beat_interval *= 1.3  # Trigger later for call-and-response
            
            # Polyrhythmic relationships
            if len(recent_other_triggers) >= 2:
                # Multiple voices active - create polyrhythmic feel
                polyrhythm_ratios = [0.66, 0.75, 1.33, 1.5]  # 2:3, 3:4, 3:2, 2:3 ratios
                ratio = random.choice(polyrhythm_ratios)
                beat_interval *= ratio
        
        # Store this trigger time for other voices to reference
        self.recent_triggers[voice] = current_time
        
        return beat_interval

    def update_bpm(self, new_bpm: float):
        """Update base BPM while maintaining fixed intervals"""
        self.base_bpm = new_bpm
    
    def set_response_sensitivity(self, sensitivity):
        """Set how responsive rhythm is to performance changes (0-1)"""
        self.response_sensitivity = max(0.0, min(1.0, sensitivity))
        
        # Adjust timing based on sensitivity
        if sensitivity < 0.3:
            self.current_intensity = 'low'
        elif sensitivity < 0.7:
            self.current_intensity = 'medium'
        else:
            self.current_intensity = 'high'
