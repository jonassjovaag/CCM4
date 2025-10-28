import time
import random
import numpy as np
from typing import Dict, List

class Voice1MelodicController:
    """
    Voice 1: Longer, melodic notes that follow averaged pitch input
    - Random note lengths between 0.5-1.5 seconds
    - ADSR envelope with fade in/out
    - Stays close to averaged pitch input
    """
    def __init__(self, pitch_tracker=None, thread_ctrl=None, logger=None):
        self.pitch_tracker = pitch_tracker
        self.thread_ctrl = thread_ctrl
        self.logger = logger
        
        # Voice 1 characteristics
        self.min_note_duration = 0.5
        self.max_note_duration = 1.5
        self.pitch_follow_factor = 0.7  # How closely to follow input pitch
        self.base_frequency = 220.0  # Fallback frequency
        
        # Timing control
        self.last_trigger_time = time.time()  # Initialize to current time
        self.current_note_end_time = 0.0
        self.next_trigger_delay = 8.0  # Initial delay between notes - much longer
        
        # ADSR parameters (will be sent to SC)
        self.attack_time = 0.2   # 200ms attack
        self.decay_time = 0.3    # 300ms decay
        self.sustain_level = 0.7 # 70% sustain level
        self.release_time = 0.4  # 400ms release
        
        # Pitch smoothing
        self.last_frequency = 220.0
        self.frequency_smoothing = 0.1  # How quickly to adapt to new pitches
        
    def get_target_frequency(self) -> float:
        """Get target frequency based on averaged pitch input"""
        if self.pitch_tracker:
            current_pitch = self.pitch_tracker.get_current_pitch()
            if current_pitch > 50 and current_pitch < 800:  # Valid range
                # Smooth transition to new pitch
                target = self.last_frequency + self.frequency_smoothing * (current_pitch - self.last_frequency)
                self.last_frequency = target
                return target
        
        # Fallback: slight drift around base frequency
        drift = random.uniform(-20, 20)
        return self.base_frequency + drift
    
    def get_note_duration(self) -> float:
        """Generate random note duration between min and max"""
        return random.uniform(self.min_note_duration, self.max_note_duration)
    
    def get_next_trigger_delay(self) -> float:
        """Calculate delay until next note trigger"""
        # Much longer delays for melodic voice
        base_delay = random.uniform(6.0, 15.0)  # 6-15 seconds between notes
        
        # Occasionally create shorter delays for melodic phrases
        if random.random() < 0.15:  # 15% chance
            base_delay *= 0.4  # Shorter delay for melodic phrases
        
        return base_delay
    
    def should_trigger_note(self) -> bool:
        """Determine if Voice1 should trigger a new note"""
        current_time = time.time()
        
        # Check if enough time has passed since last trigger
        if current_time >= self.last_trigger_time + self.next_trigger_delay:
            return True
        
        return False
    
    def generate_voice1_event(self) -> Dict:
        """Generate a Voice1 melodic event"""
        current_time = time.time()
        
        # Get parameters for this note
        frequency = self.get_target_frequency()
        duration = self.get_note_duration()
        
        # Amplitude varies slightly for expression - MUCH lower for no distortion
        base_amplitude = 0.15  # Reduced from 0.4 to prevent distortion
        amplitude_variation = random.uniform(0.8, 1.2)
        amplitude = base_amplitude * amplitude_variation
        
        # Update timing
        self.last_trigger_time = current_time
        self.current_note_end_time = current_time + duration
        self.next_trigger_delay = self.get_next_trigger_delay()
        
        return {
            'type': 'melodic_note',
            'frequency': frequency,
            'amplitude': amplitude,
            'duration': duration,
            'attack': self.attack_time,
            'decay': self.decay_time,
            'sustain': self.sustain_level,
            'release': self.release_time,
            'timestamp': current_time,
            'voice': 'voice1'
        }
    
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
        """Main Voice1 melodic loop"""
        while self.thread_ctrl.running.is_set():
            if self.should_trigger_note():
                event = self.generate_voice1_event()
                
                if self.logger:
                    self.logger.log_performance_event({
                        'event': 'voice1_melodic',
                        'frequency': event['frequency'],
                        'duration': event['duration'],
                        'amplitude': event['amplitude']
                    })
                
                # Store the event for the OSC handler to pick up
                if not hasattr(self, 'pending_events'):
                    self.pending_events = []
                self.pending_events.append(event)
                
                print(f"🎵 Voice1 melodic note: {event['frequency']:.1f}Hz for {event['duration']:.2f}s")
            
            time.sleep(0.1)  # Check every 100ms
