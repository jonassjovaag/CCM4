import numpy as np
import time
import random
from typing import List, Dict
from collections import deque

class ResponsiveBassController:
    """
    Voice3 - Responsive bass that reacts directly to audio input activity
    Similar to Voice5 (responsive percussion) but with bass characteristics
    """
    def __init__(self, state_manager=None, thread_ctrl=None, logger=None, pitch_tracker=None):
        self.state_manager = state_manager
        self.thread_ctrl = thread_ctrl
        self.logger = logger
        self.pitch_tracker = pitch_tracker
        
        # Bass sound types with sub-bass frequencies
        self.sound_types = {
            'sub_bass': {'freq': 40, 'duration': 1.2, 'noise_factor': 0.05, 'bass_type': 0},    # Very low sub bass
            'bass': {'freq': 60, 'duration': 0.8, 'noise_factor': 0.1, 'bass_type': 1},        # Standard bass
            'low_mid': {'freq': 80, 'duration': 0.6, 'noise_factor': 0.15, 'bass_type': 2},    # Low mid bass
            'punchy': {'freq': 100, 'duration': 0.4, 'noise_factor': 0.2, 'bass_type': 3}      # Punchy bass
        }
        
        # Activity tracking for responsiveness (similar to Voice5)
        self.activity_buffer = deque(maxlen=100)  # Last 10 seconds at 10Hz
        self.recent_triggers = deque(maxlen=20)   # Last 20 triggers
        self.activity_threshold = 0.2   # Further reduced for more responsiveness
        self.max_response_rate = 1.8    # Increased response rate
        self.cooldown_time = 1.2        # Further reduced cooldown time
        
        # Volume and sensitivity - REDUCED to prevent distortion
        self.base_volume = 0.08         # Reduced from 0.15 to prevent distortion
        self.volume_boost = 0.04        # Reduced from 0.10 to prevent distortion
        self.sensitivity = 0.45         # Increased sensitivity for more frequent triggers
        
        # Response characteristics
        self.echo_probability = 0.2     # Lower chance of creating echo
        self.echo_delay = 0.25         # Longer time between echoes for bass
        self.pitch_influence = 0.8     # Higher pitch influence for musical coherence
        
        self.last_activity_check = time.time()
        self.last_trigger_time = 0.0
        
    def update_activity_tracking(self):
        """Track audio input activity using onset detection (same as Voice5)"""
        current_time = time.time()
        
        # Primary activity source: onset detection from audio input
        onset_activity = 0.0
        if self.pitch_tracker:
            # Get recent onsets (last 2 seconds)
            recent_onsets = self.pitch_tracker.get_recent_onsets(2.0)
            onset_rate = self.pitch_tracker.get_onset_rate()
            
            # Scale onset activity (0-1) based on onset rate
            # 4+ onsets per second = full activity (1.0)
            onset_activity = min(1.0, onset_rate / 4.0)
            
            # Boost if there have been very recent onsets
            if recent_onsets > 0:
                onset_activity = max(onset_activity, recent_onsets / 8.0)
        
        # Secondary: system-generated activity (lower weight)
        system_activity = 0.0
        if self.state_manager:
            system_activity = self.state_manager.current_state.get('activity_level', 0.0) * 0.3
        
        # Combine onset-based and system activity (onset-based is primary)
        total_activity = min(1.0, onset_activity * 0.8 + system_activity * 0.2)
        
        # Debug: print onset activity when significant
        if onset_activity > 0.1 and current_time - getattr(self, 'last_debug_time', 0) > 1.0:
            print(f"Voice3: Onsets={recent_onsets}, Rate={onset_rate:.2f}, Activity={total_activity:.2f}")
            self.last_debug_time = current_time
        
        self.activity_buffer.append({
            'time': current_time,
            'activity': total_activity,
            'onset_activity': onset_activity,
            'onset_rate': self.pitch_tracker.get_onset_rate() if self.pitch_tracker else 0.0
        })
        
        self.last_activity_check = current_time
    
    def get_current_activity_level(self) -> float:
        """Calculate current activity level from recent buffer (same as Voice5)"""
        if not self.activity_buffer:
            return 0.0
            
        current_time = time.time()
        # Only consider last 2 seconds for immediate responsiveness
        recent_activity = [
            entry['activity'] for entry in self.activity_buffer
            if current_time - entry['time'] < 2.0
        ]
        
        if not recent_activity:
            return 0.0
            
        # Weight recent activity more heavily
        weights = np.linspace(0.5, 1.0, len(recent_activity))
        weighted_activity = np.average(recent_activity, weights=weights)
        
        return weighted_activity
    
    def should_trigger_response(self) -> bool:
        """Determine if voice3 should respond based on activity (adapted from Voice5)"""
        current_time = time.time()
        activity_level = self.get_current_activity_level()
        
        # Don't trigger too frequently - use cooldown time
        time_since_last = current_time - self.last_trigger_time
        
        if time_since_last < self.cooldown_time:
            return False
        
        # Activity must exceed threshold
        if activity_level < self.activity_threshold:
            return False
            
        # Get current performance phase for phase-dependent sensitivity
        phase_sensitivity = self.get_phase_sensitivity()
        
        # Base probability is much lower and depends on phase
        base_probability = activity_level * self.sensitivity * phase_sensitivity
        
        # Add randomness factor - more triggering than before
        randomness_gate = random.random() < 0.4   # Increased from 25% to 40% chance to consider triggering
        if not randomness_gate:
            return False
        
        # Strongly reduce probability if we've been triggering recently
        if len(self.recent_triggers) > 1:
            recent_trigger_rate = len(self.recent_triggers) / 10.0  
            if recent_trigger_rate > 0.2:  # If more than 0.2 per second recently
                base_probability *= 0.1  # Drastically reduce probability
        
        # Final random trigger based on probability
        if random.random() < base_probability:
            self.last_trigger_time = current_time
            self.recent_triggers.append(current_time)
            print(f"🎸 Voice3 BASS TRIGGER! Activity={activity_level:.2f}, Phase={phase_sensitivity:.2f}, Prob={base_probability:.2f}")
            
            # Log system trigger event
            if self.logger:
                self.logger.log_system_event(
                    event_type="voice3_trigger",
                    voice="voice3", 
                    trigger_reason="onset_response",
                    activity_level=activity_level,
                    phase=f"sensitivity_{phase_sensitivity:.2f}",
                    additional_data=f"prob={base_probability:.3f}"
                )
            
            return True
            
        return False
    
    def get_phase_sensitivity(self) -> float:
        """Get sensitivity multiplier based on current performance phase (same as Voice5)"""
        current_time = time.time()
        if not hasattr(self, 'start_time'):
            self.start_time = current_time
            
        elapsed_minutes = (current_time - self.start_time) / 60.0
        
        # Approximate phases based on elapsed time (assuming 5-minute performance)
        if elapsed_minutes < 0.5:  # Phase 1: Awakening - bass more active than before
            return 0.2   # Increased for more bass presence
        elif elapsed_minutes < 2.0:  # Phase 2: Conversation - bass MORE active
            return 0.5   # Significantly increased for more bass activity
        elif elapsed_minutes < 3.0:  # Phase 3: Climax - bass VERY active
            return 0.7   # Much more active during climax
        elif elapsed_minutes < 4.0:  # Phase 4: Reflection - bass MUCH MORE active
            return 0.6   # Very active during reflection
        else:  # Phase 5: Resolution - bass MORE active, ending with just bass
            # Bass should be most prominent in final phase
            phase5_elapsed = elapsed_minutes - 4.0
            phase5_progress = min(phase5_elapsed / 1.0, 1.0)  # 0-1 over 1 minute
            # Start high and stay relatively active to end with just bass
            return max(0.3, 0.4 - 0.1 * phase5_progress)  # Fade from 0.4 to 0.3
    
    def generate_responsive_event(self) -> Dict:
        """Generate a bass event that responds to current audio context"""
        activity_level = self.get_current_activity_level()
        
        # Choose bass type based on activity level
        if activity_level > 0.8:
            # High activity - more punchy and bass
            sound_weights = {'punchy': 0.4, 'bass': 0.3, 'low_mid': 0.2, 'sub_bass': 0.1}
        elif activity_level > 0.5:
            # Medium activity - balanced mix
            sound_weights = {'bass': 0.4, 'low_mid': 0.3, 'punchy': 0.2, 'sub_bass': 0.1}
        else:
            # Lower activity - mostly sub bass and bass
            sound_weights = {'sub_bass': 0.4, 'bass': 0.4, 'low_mid': 0.15, 'punchy': 0.05}
        
        sound_type = random.choices(
            list(sound_weights.keys()),
            weights=list(sound_weights.values())
        )[0]
        
        sound_params = self.sound_types[sound_type]
        
        # Start with base frequency
        base_freq = sound_params['freq']
        
        # Add pitch influence based on detected pitch for musical coherence
        if self.pitch_tracker and self.pitch_influence > 0:
            detected_pitch = self.pitch_tracker.get_current_pitch()
            if detected_pitch > 0:
                # Strong pitch influence for musical coherence
                # Map detected pitch to bass frequencies (typically octaves below)
                fundamental = detected_pitch
                while fundamental > 120:  # Bring down to bass range
                    fundamental /= 2.0
                
                # Blend detected fundamental with base frequency
                influence_factor = self.pitch_influence
                base_freq = base_freq * (1.0 - influence_factor) + fundamental * influence_factor
                
                # Ensure we stay in bass range
                base_freq = max(30, min(base_freq, 150))
        
        # Volume responds to activity
        volume = self.base_volume + (activity_level * self.volume_boost)
        volume = max(0.01, min(volume, 0.15))  # Quieter range than drums
        
        # Duration scales with activity (more activity = shorter bass notes)
        base_duration = sound_params['duration']
        duration_factor = 1.5 - (activity_level * 0.5)  # 1.5 to 1.0 range (less variation)
        duration = base_duration * duration_factor
        
        # Other parameters
        noise_factor = sound_params['noise_factor']
        bass_type = sound_params['bass_type']
        
        return {
            'type': sound_type,
            'frequency': base_freq,
            'amplitude': volume,
            'duration': duration,
            'noise_factor': noise_factor,
            'bass_type': bass_type,
            'activity_level': activity_level,
            'timestamp': time.time()
        }
    
    def should_create_echo(self) -> bool:
        """Determine if we should create an echo/repeat effect (lower probability than drums)"""
        activity_level = self.get_current_activity_level()
        
        # Lower echo chance than drums
        echo_chance = self.echo_probability * (0.5 + activity_level * 0.5)
        return random.random() < echo_chance
    
    def run(self):
        """Main responsive bass loop"""
        while self.thread_ctrl.running.is_set():
            # Update activity tracking frequently
            self.update_activity_tracking()
            
            # Check for response triggers
            if self.should_trigger_response():
                event = self.generate_responsive_event()
                
                if self.logger:
                    self.logger.log_performance_event({
                        'event': 'voice3_trigger',
                        'sound_type': event['type'],
                        'activity_level': event['activity_level'],
                        'frequency': event['frequency'],
                        'amplitude': event['amplitude']
                    })
                
                # Store the event for the OSC handler to pick up
                if not hasattr(self, 'pending_events'):
                    self.pending_events = []
                self.pending_events.append(event)
                print(f"🎸 Voice3 bass event created: {event['type']} at {event['frequency']:.1f}Hz")
                
                # Schedule echo if appropriate
                if self.should_create_echo():
                    echo_event = event.copy()
                    echo_event['amplitude'] *= 0.5  # Quieter echo
                    echo_event['timestamp'] += self.echo_delay
                    self.pending_events.append(echo_event)
            
            time.sleep(0.1)  # 10Hz update rate for responsiveness
    
    def get_pending_events(self) -> List[Dict]:
        """Get events ready to be sent"""
        if not hasattr(self, 'pending_events'):
            return []
            
        current_time = time.time()
        ready_events = []
        remaining_events = []
        
        for event in self.pending_events:
            if current_time >= event['timestamp']:
                ready_events.append(event)
            else:
                remaining_events.append(event)
        
        self.pending_events = remaining_events
        return ready_events
    
    def set_sensitivity(self, sensitivity: float):
        """Adjust how responsive the system is (0-1)"""
        self.sensitivity = max(0.0, min(1.0, sensitivity))
        
    def set_activity_threshold(self, threshold: float):
        """Set minimum activity level needed to trigger (0-1)"""
        self.activity_threshold = max(0.0, min(1.0, threshold))
