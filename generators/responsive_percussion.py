import numpy as np
import time
import random
from typing import List, Dict
from collections import deque

class ResponsivePercussionController:
    """
    Voice5 - Responsive percussion that reacts directly to audio input activity
    Different sound characteristics from voice4 (which is time-based)
    """
    def __init__(self, state_manager=None, thread_ctrl=None, logger=None, pitch_tracker=None):
        self.state_manager = state_manager
        self.thread_ctrl = thread_ctrl
        self.logger = logger
        self.pitch_tracker = pitch_tracker
        
        # Drum sound types 
        self.sound_types = {
            'kick': {'freq': 60, 'duration': 0.6, 'noise_factor': 0.1, 'drum_type': 0},  # Bass drum - sine based
            'snare': {'freq': 200, 'duration': 0.15, 'noise_factor': 0.8, 'drum_type': 1},  # Snare - noise based
            'hihat_closed': {'freq': 8000, 'duration': 0.08, 'noise_factor': 0.6, 'drum_type': 2},  # Closed hihat
            'hihat_open': {'freq': 8000, 'duration': 0.4, 'noise_factor': 0.6, 'drum_type': 3}   # Open hihat
        }
        
        # Activity tracking for responsiveness
        self.activity_buffer = deque(maxlen=100)  # Last 10 seconds at 10Hz
        self.recent_triggers = deque(maxlen=20)   # Last 20 triggers
        self.activity_threshold = 0.25  # Reduced threshold for more responsiveness 
        self.max_response_rate = 1.5   # Max 1.5 responses per second
        self.cooldown_time = 1.0       # Reduced minimum time between triggers
        
        # Volume and sensitivity - REDUCED to prevent distortion
        self.base_volume = 0.08        # Reduced from 0.18 to prevent distortion
        self.volume_boost = 0.05       # Reduced from 0.12 to prevent distortion
        self.sensitivity = 0.5         # Increased sensitivity for more frequent triggers
        
        # Response characteristics
        self.echo_probability = 0.4    # Chance of creating echo/repeat
        self.echo_delay = 0.15        # Time between echoes
        self.pitch_influence = 0.6    # How much detected pitch affects frequency
        
        self.last_activity_check = time.time()
        self.last_trigger_time = 0.0
        
    def update_activity_tracking(self):
        """Track audio input activity using onset detection"""
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
                onset_activity = max(onset_activity, recent_onsets / 8.0)  # Up to 8 onsets in 2s = max activity
        
        # Secondary: system-generated activity (lower weight)
        system_activity = 0.0
        if self.state_manager:
            system_activity = self.state_manager.current_state.get('activity_level', 0.0) * 0.3
        
        # Combine onset-based and system activity (onset-based is primary)
        total_activity = min(1.0, onset_activity * 0.8 + system_activity * 0.2)
        
        # Debug: print onset activity when significant
        if onset_activity > 0.1 and current_time - getattr(self, 'last_debug_time', 0) > 1.0:
            print(f"Voice5: Onsets={recent_onsets}, Rate={onset_rate:.2f}, Activity={total_activity:.2f}")
            self.last_debug_time = current_time
        
        self.activity_buffer.append({
            'time': current_time,
            'activity': total_activity,
            'onset_activity': onset_activity,
            'onset_rate': self.pitch_tracker.get_onset_rate() if self.pitch_tracker else 0.0
        })
        
        self.last_activity_check = current_time
    
    def get_current_activity_level(self) -> float:
        """Calculate current activity level from recent buffer"""
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
        """Determine if voice5 should respond based on activity"""
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
        
        # Add randomness factor but more permissive
        randomness_gate = random.random() < 0.6  # 60% chance to consider triggering
        if not randomness_gate:
            return False
        
        # Strongly reduce probability if we've been triggering recently
        if len(self.recent_triggers) > 1:  # If more than 1 trigger in last 10 seconds
            recent_trigger_rate = len(self.recent_triggers) / 10.0  
            if recent_trigger_rate > 0.3:  # If more than 0.3 per second recently
                base_probability *= 0.1  # Drastically reduce probability
        
        # Final random trigger based on probability
        if random.random() < base_probability:
            self.last_trigger_time = current_time
            self.recent_triggers.append(current_time)
            print(f"🔥 Voice5 TRIGGER! Activity={activity_level:.2f}, Phase={phase_sensitivity:.2f}, Prob={base_probability:.2f}")
            
            # Log system trigger event
            if self.logger:
                self.logger.log_system_event(
                    event_type="voice5_trigger",
                    voice="voice5", 
                    trigger_reason="onset_response",
                    activity_level=activity_level,
                    phase=f"sensitivity_{phase_sensitivity:.2f}",
                    additional_data=f"prob={base_probability:.3f}"
                )
            
            return True
            
        return False
    
    def get_phase_sensitivity(self) -> float:
        """Get sensitivity multiplier based on current performance phase"""
        # For now, use a simple time-based approximation since we don't have direct access to performance director
        # This is much simpler and more predictable than trying to guess from system activity
        
        # Very conservative phase sensitivities
        current_time = time.time()
        if not hasattr(self, 'start_time'):
            self.start_time = current_time
            
        elapsed_minutes = (current_time - self.start_time) / 60.0
        
        # Approximate phases based on elapsed time (assuming 5-minute performance)
        if elapsed_minutes < 0.5:  # Phase 1: Awakening - drums MORE active
            return 0.6  # Increased for more drum activity
        elif elapsed_minutes < 2.0:  # Phase 2: Conversation - drums normal
            return 0.4  # Increased normal level
        elif elapsed_minutes < 3.0:  # Phase 3: Climax - drums VERY active
            return 0.8  # Much more active during climax
        elif elapsed_minutes < 4.0:  # Phase 4: Reflection - drums MUCH MORE active
            return 0.7  # Very active during reflection
        else:  # Phase 5: Resolution - drums less active, easing out
            # Gradually reduce activity as we approach the end
            phase5_elapsed = elapsed_minutes - 4.0
            phase5_progress = min(phase5_elapsed / 1.0, 1.0)  # 0-1 over 1 minute
            return max(0.05, 0.2 * (1.0 - phase5_progress))  # Fade from 0.2 to 0.05
    
    def generate_responsive_event(self) -> Dict:
        """Generate a percussion event that responds to current audio context"""
        activity_level = self.get_current_activity_level()
        
        # Choose drum type based on activity level
        if activity_level > 0.8:
            # High activity - more kicks and snares
            sound_weights = {'kick': 0.4, 'snare': 0.3, 'hihat_closed': 0.2, 'hihat_open': 0.1}
        elif activity_level > 0.5:
            # Medium activity - balanced mix
            sound_weights = {'kick': 0.3, 'snare': 0.25, 'hihat_closed': 0.35, 'hihat_open': 0.1}
        else:
            # Lower activity - mostly hihats
            sound_weights = {'kick': 0.2, 'snare': 0.1, 'hihat_closed': 0.6, 'hihat_open': 0.1}
        
        sound_type = random.choices(
            list(sound_weights.keys()),
            weights=list(sound_weights.values())
        )[0]
        
        sound_params = self.sound_types[sound_type]
        
        # Use fixed frequency for each drum type (more realistic)
        base_freq = sound_params['freq']
        
        # Add slight pitch variation for kick and snare based on detected pitch
        if sound_type in ['kick', 'snare'] and self.pitch_tracker and self.pitch_influence > 0:
            detected_pitch = self.pitch_tracker.get_current_pitch()
            if detected_pitch > 0:
                # Slight pitch influence for musical coherence
                pitch_factor = min(2.0, detected_pitch / 220.0)  # Relative to A3
                if sound_type == 'kick':
                    base_freq = base_freq * (0.8 + pitch_factor * 0.4)  # 48-96Hz range
                elif sound_type == 'snare':
                    base_freq = base_freq * (0.7 + pitch_factor * 0.6)  # 140-320Hz range
        
        # Volume responds to activity
        volume = self.base_volume + (activity_level * self.volume_boost)
        volume = max(0.05, min(volume, 0.3))  # Clamp to reasonable range
        
        # Duration scales with activity (more activity = shorter, punchier sounds)
        base_duration = sound_params['duration']
        duration_factor = 1.5 - (activity_level * 0.8)  # 1.5 to 0.7 range
        duration = base_duration * duration_factor
        
        # Other parameters
        noise_factor = sound_params['noise_factor']
        drum_type = sound_params['drum_type']
        
        return {
            'type': sound_type,
            'frequency': base_freq,
            'amplitude': volume,
            'duration': duration,
            'noise_factor': noise_factor,
            'drum_type': drum_type,
            'activity_level': activity_level,
            'timestamp': time.time()
        }
    
    def should_create_echo(self) -> bool:
        """Determine if we should create an echo/repeat effect"""
        activity_level = self.get_current_activity_level()
        
        # Higher activity increases echo probability
        echo_chance = self.echo_probability * (1.0 + activity_level)
        return random.random() < echo_chance
    
    def run(self):
        """Main responsive percussion loop"""
        while self.thread_ctrl.running.is_set():
            # Update activity tracking frequently
            self.update_activity_tracking()
            
            # Check for response triggers
            if self.should_trigger_response():
                event = self.generate_responsive_event()
                
                if self.logger:
                    self.logger.log_performance_event({
                        'event': 'voice5_trigger',
                        'sound_type': event['type'],
                        'activity_level': event['activity_level'],
                        'frequency': event['frequency'],
                        'amplitude': event['amplitude']
                    })
                
                # Store the event for the OSC handler to pick up
                if not hasattr(self, 'pending_events'):
                    self.pending_events = []
                self.pending_events.append(event)
                print(f"🎵 Voice5 event created: {event['type']} at {event['frequency']:.1f}Hz")
                
                # Schedule echo if appropriate
                if self.should_create_echo():
                    echo_event = event.copy()
                    echo_event['amplitude'] *= 0.6  # Quieter echo
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
