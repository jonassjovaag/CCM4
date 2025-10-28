import time
import random
import numpy as np
from typing import Dict, List
from collections import deque

class Voice3OnsetResponsiveController:
    """
    Voice 3: Onset-responsive voice that reacts specifically to audio input onsets
    - Responds directly to onset detection
    - Musical bass-like responses
    - Varied response patterns based on onset characteristics
    """
    def __init__(self, pitch_tracker=None, thread_ctrl=None, logger=None):
        self.pitch_tracker = pitch_tracker
        self.thread_ctrl = thread_ctrl
        self.logger = logger
        
        # Onset response characteristics
        self.onset_threshold = 0.3  # Minimum onset strength to respond
        self.response_probability = 0.6  # 60% chance to respond to valid onset
        self.cooldown_time = 0.8  # Minimum time between responses
        self.last_response_time = time.time()  # Initialize to current time
        
        # Musical response patterns
        self.bass_frequencies = [65.4, 87.3, 98.0, 110.0, 130.8, 146.8]  # C2, F2, G2, A2, C3, D3
        self.response_types = {
            'bass_hit': {'duration': 1.2, 'amplitude': 0.20, 'attack': 0.05, 'decay': 0.3},     # Reduced from 0.6
            'bass_sustain': {'duration': 2.0, 'amplitude': 0.15, 'attack': 0.2, 'decay': 0.8},  # Reduced from 0.4
            'bass_stab': {'duration': 0.6, 'amplitude': 0.25, 'attack': 0.02, 'decay': 0.1},    # Reduced from 0.8!
            'bass_roll': {'duration': 0.4, 'amplitude': 0.18, 'attack': 0.1, 'decay': 0.2}     # Reduced from 0.5
        }
        
        # Onset tracking
        self.recent_onsets = deque(maxlen=20)
        self.onset_pattern_buffer = deque(maxlen=10)
        
        # Harmonic following
        self.pitch_influence = 0.7  # How much detected pitch affects frequency choice
        self.last_chosen_frequency = 110.0
        
    def update_onset_tracking(self):
        """Track recent onsets for pattern analysis"""
        current_time = time.time()
        
        if self.pitch_tracker:
            # Get recent onset information
            onset_rate = self.pitch_tracker.get_onset_rate()
            recent_onsets_count = self.pitch_tracker.get_recent_onsets(2.0)  # Last 2 seconds
            
            # Store onset data
            onset_data = {
                'time': current_time,
                'rate': onset_rate,
                'count': recent_onsets_count,
                'strength': min(1.0, onset_rate / 4.0)  # Normalize to 0-1
            }
            
            self.recent_onsets.append(onset_data)
            
            # Analyze onset patterns
            if len(self.recent_onsets) >= 3:
                recent_rates = [o['rate'] for o in list(self.recent_onsets)[-3:]]
                avg_rate = sum(recent_rates) / len(recent_rates)
                rate_trend = recent_rates[-1] - recent_rates[0]  # Increasing or decreasing
                
                pattern = {
                    'average_rate': avg_rate,
                    'trend': rate_trend,
                    'intensity': onset_data['strength']
                }
                
                self.onset_pattern_buffer.append(pattern)
    
    def get_onset_strength(self) -> float:
        """Get current onset strength (0.0 to 1.0)"""
        if not self.recent_onsets:
            return 0.0
        
        latest_onset = self.recent_onsets[-1]
        return latest_onset['strength']
    
    def choose_bass_frequency(self) -> float:
        """Choose bass frequency based on pitch input and pattern"""
        base_freq = random.choice(self.bass_frequencies)
        
        # Influence frequency choice with detected pitch
        if self.pitch_tracker and self.pitch_influence > 0:
            detected_pitch = self.pitch_tracker.get_current_pitch()
            if detected_pitch > 50 and detected_pitch < 500:
                # Find closest bass frequency to detected pitch
                closest_bass = min(self.bass_frequencies, 
                                 key=lambda f: abs(f - (detected_pitch / 2)))  # Octave down
                
                # Blend with random choice
                if random.random() < self.pitch_influence:
                    base_freq = closest_bass
        
        # Smooth frequency changes
        smoothing = 0.3
        result_freq = self.last_chosen_frequency + smoothing * (base_freq - self.last_chosen_frequency)
        self.last_chosen_frequency = result_freq
        
        return result_freq
    
    def choose_response_type(self, onset_strength: float) -> str:
        """Choose response type based on onset characteristics"""
        if not self.onset_pattern_buffer:
            return 'bass_hit'
        
        latest_pattern = self.onset_pattern_buffer[-1]
        
        # Choose response based on onset pattern
        if onset_strength > 0.8:  # Strong onset
            if latest_pattern['average_rate'] > 6.0:  # Fast activity
                return 'bass_stab'  # Quick, punchy response
            else:
                return 'bass_hit'   # Strong but not rushed
        elif onset_strength > 0.5:  # Medium onset
            if latest_pattern['trend'] > 0:  # Increasing activity
                return 'bass_roll'  # Rhythmic response
            else:
                return 'bass_sustain'  # Sustained response
        else:  # Weaker onset
            return 'bass_sustain'  # Gentle, sustained response
    
    def should_respond_to_onset(self) -> bool:
        """Determine if Voice3 should respond to current onset"""
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_response_time < self.cooldown_time:
            return False
        
        # Check onset strength
        onset_strength = self.get_onset_strength()
        if onset_strength < self.onset_threshold:
            return False
        
        # Probability-based response
        # Higher onset strength = higher probability
        response_chance = self.response_probability * (0.5 + onset_strength * 0.5)
        
        return random.random() < response_chance
    
    def generate_onset_response(self) -> Dict:
        """Generate a Voice3 onset response event"""
        current_time = time.time()
        onset_strength = self.get_onset_strength()
        
        # Choose response characteristics
        frequency = self.choose_bass_frequency()
        response_type = self.choose_response_type(onset_strength)
        response_params = self.response_types[response_type]
        
        # Scale amplitude based on onset strength
        base_amplitude = response_params['amplitude']
        amplitude = base_amplitude * (0.7 + onset_strength * 0.3)
        
        # Update response timing
        self.last_response_time = current_time
        
        event = {
            'type': 'onset_response',
            'frequency': frequency,
            'amplitude': amplitude,
            'duration': response_params['duration'],
            'attack': response_params['attack'],
            'decay': response_params['decay'],
            'response_type': response_type,
            'onset_strength': onset_strength,
            'timestamp': current_time,
            'voice': 'voice3'
        }
        
        return event
    
    def get_pending_events(self) -> List[Dict]:
        """Get events ready to be sent to SuperCollider"""
        if not hasattr(self, 'pending_events'):
            self.pending_events = []
            
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
    
    def run(self):
        """Main Voice3 onset-responsive loop"""
        while self.thread_ctrl.running.is_set():
            # Update onset tracking
            self.update_onset_tracking()
            
            # Check for onset response
            if self.should_respond_to_onset():
                event = self.generate_onset_response()
                
                if self.logger:
                    self.logger.log_performance_event({
                        'event': 'voice3_onset_response',
                        'response_type': event['response_type'],
                        'onset_strength': event['onset_strength'],
                        'frequency': event['frequency']
                    })
                
                # Store event for OSC handler
                if not hasattr(self, 'pending_events'):
                    self.pending_events = []
                self.pending_events.append(event)
                
                print(f"🎯 Voice3 onset response: {event['response_type']} at {event['frequency']:.1f}Hz (strength: {event['onset_strength']:.2f})")
            
            time.sleep(0.05)  # Check every 50ms for responsiveness
