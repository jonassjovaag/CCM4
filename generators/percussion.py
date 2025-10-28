import numpy as np
import time
import random
from typing import List, Dict

class PercussionController:
    def __init__(self, rhythm_engine=None, thread_ctrl=None, logger=None, base_bpm=30.0):
        self.rhythm_engine = rhythm_engine
        self.thread_ctrl = thread_ctrl
        self.logger = logger
        self.base_bpm = base_bpm
        
        # Percussion sound types with their parameters
        self.sound_types = {
            'click': {'freq_range': (800, 1200), 'duration': 0.05, 'noise_factor': 0.3},
            'pop': {'freq_range': (200, 400), 'duration': 0.08, 'noise_factor': 0.7},
            'tick': {'freq_range': (1500, 2500), 'duration': 0.03, 'noise_factor': 0.1},
            'blip': {'freq_range': (600, 1000), 'duration': 0.12, 'noise_factor': 0.5},
            'noise': {'freq_range': (100, 300), 'duration': 0.15, 'noise_factor': 0.9},
            'scratch': {'freq_range': (400, 1800), 'duration': 0.2, 'noise_factor': 0.8}
        }
        
        # Rhythmic vs random behavior
        self.time_mode = True  # True = rhythmic, False = random
        self.mode_transition_time = time.time()
        self.mode_duration = random.uniform(10, 30)  # How long to stay in each mode
        
        # Volume control
        self.base_volume = 0.16  # Doubled from 0.08
        self.volume_fluctuation = 0.06  # Amount of volume variation
        self.density = 0.3  # Density of percussion events (0-1)
        
        # Timing control
        self.next_event_time = time.time()
        self.rhythmic_subdivision = 4  # Quarter note subdivisions
        
    def update_mode(self):
        """Switch between time-based and random modes"""
        current_time = time.time()
        if current_time - self.mode_transition_time > self.mode_duration:
            self.time_mode = not self.time_mode
            self.mode_transition_time = current_time
            self.mode_duration = random.uniform(8, 25)  # Vary the mode duration
            
            mode_name = "rhythmic" if self.time_mode else "organic"
            if self.logger:
                self.logger.log_performance_event({
                    'event': 'percussion_mode_change',
                    'mode': mode_name,
                    'duration': self.mode_duration
                })
    
    def get_next_event_time(self) -> float:
        """Calculate when the next percussion event should occur"""
        current_time = time.time()
        
        if self.time_mode:
            # Rhythmic mode - sync to BPM with some variation
            beat_duration = 60.0 / self.base_bpm
            subdivision_duration = beat_duration / self.rhythmic_subdivision
            
            # Add some rhythmic variation (swing, syncopation), scaled by density
            variation_factor = random.uniform(0.7, 1.3) / max(0.1, self.density)
            interval = subdivision_duration * variation_factor
            
            # Occasionally skip beats for syncopation (less skipping with higher density)
            skip_probability = 0.3 * (1.0 - self.density)
            if random.random() < skip_probability:
                interval *= random.choice([2, 3, 4])  # Skip 1-3 beats
                
        else:
            # Organic mode - random intervals like raindrops/waves
            # Use Poisson-like distribution for natural randomness, scaled by density
            lambda_param = 1.0 + (self.density * 3.0)  # 1-4 events per second based on density
            interval = np.random.exponential(1.0 / lambda_param)
            
            # Add some clustering (like waves)
            if random.random() < 0.2:  # 20% chance of cluster
                interval *= random.uniform(0.1, 0.3)  # Very short interval
        
        return current_time + interval
    
    def generate_percussion_event(self) -> Dict:
        """Generate a percussion sound event"""
        # Choose sound type with weighted probabilities
        sound_weights = {
            'click': 0.3,
            'tick': 0.25, 
            'pop': 0.15,
            'blip': 0.15,
            'noise': 0.1,
            'scratch': 0.05
        }
        
        sound_type = random.choices(
            list(sound_weights.keys()), 
            weights=list(sound_weights.values())
        )[0]
        
        sound_params = self.sound_types[sound_type]
        
        # Generate frequency (even though it's percussive, we need a base freq)
        freq_min, freq_max = sound_params['freq_range']
        frequency = random.uniform(freq_min, freq_max)
        
        # Generate volume with fluctuation
        volume = self.base_volume + random.uniform(-self.volume_fluctuation, self.volume_fluctuation)
        volume = max(0.01, min(volume, 0.2))  # Clamp between 1% and 20%
        
        # Duration and noise factor
        duration = sound_params['duration'] * random.uniform(0.7, 1.5)
        noise_factor = sound_params['noise_factor']
        
        return {
            'type': sound_type,
            'frequency': frequency,
            'amplitude': volume,
            'duration': duration,
            'noise_factor': noise_factor,
            'timestamp': time.time()
        }
    
    def should_trigger_event(self) -> bool:
        """Check if it's time for a percussion event"""
        current_time = time.time()
        if current_time >= self.next_event_time:
            self.next_event_time = self.get_next_event_time()
            return True
        return False
    
    def run(self):
        """Main percussion controller loop"""
        while self.thread_ctrl.running.is_set():
            self.update_mode()
            
            if self.should_trigger_event():
                event = self.generate_percussion_event()
                
                if self.logger:
                    self.logger.log_percussion_event(event)
            
            time.sleep(0.01)  # 100Hz update rate
    
    def set_density(self, density):
        """Set percussion density (0-1)"""
        self.density = max(0.0, min(1.0, density))
        # Adjust base timing based on density
        self.base_volume = 0.10 + (density * 0.2)  # Doubled from 0.05 + 0.1
