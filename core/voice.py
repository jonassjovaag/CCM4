import numpy as np
import time
from typing import List

class Voice:
    def __init__(self, base_freq: float, sample_rate: int = 48000):
        self.base_frequency = base_freq
        self.sample_rate = sample_rate
        self.is_playing = False
        self.amplitude = 0.0
        self.phase = 0.0
        self.start_time = time.time()
        
        self.lfo_params = {
            'primary': {'depth': 0.3, 'rate': 1/3.3},
            'secondary': {'depth': 0.2, 'rate': 1/7.1},
            'tertiary': {'depth': 0.15, 'rate': 1/11.3}
        }
        
        self.envelope = {
            'state': 'idle',
            'start_time': 0,
            'current_amplitude': 0,
            'target_amplitude': 1.0
        }

    def evolve_frequency(self, harmonic_targets: List[float], tension: float) -> float:
        weighted_pull = 0
        for target in harmonic_targets:
            distance = target - self.base_frequency
            weight = 1.0 / (1.0 + abs(distance/100))
            weighted_pull += distance * weight * 0.15
            
        resistance = np.random.choice([1.01, 1.02, 1.03]) if self.base_frequency < min(harmonic_targets) else np.random.choice([0.97, 0.98, 0.99])
        self.base_frequency *= resistance
        self.base_frequency += weighted_pull
        return self.base_frequency

    def get_amplitude(self, current_time: float) -> float:
        elapsed = current_time - self.envelope['start_time']
        
        if self.envelope['state'] == 'attack':
            if elapsed < self.adsr_params['attack_time']:
                return (elapsed / self.adsr_params['attack_time']) * self.amplitude
            self.envelope['state'] = 'decay'
            self.envelope['start_time'] = current_time
            return self.amplitude
            
        elif self.envelope['state'] == 'decay':
            if elapsed < self.adsr_params['decay_time']:
                decay_factor = elapsed / self.adsr_params['decay_time']
                return self.amplitude * (1.0 - (decay_factor * (1.0 - self.adsr_params['sustain_level'])))
            self.envelope['state'] = 'sustain'
            return self.amplitude * self.adsr_params['sustain_level']
            
        elif self.envelope['state'] == 'release':
            if elapsed < self.adsr_params['release_time']:
                return self.envelope['current_amplitude'] * (1.0 - (elapsed / self.adsr_params['release_time']))
            self.envelope['state'] = 'idle'
            return 0.0
            
        elif self.envelope['state'] == 'sustain':
            return self.amplitude * self.adsr_params['sustain_level']
            
        return 0.0

class ShortTone(Voice):
    def __init__(self, base_freq: float, sample_rate: int = 48000):
        super().__init__(base_freq, sample_rate)
        
        self.adsr_params = {
            'attack_time': 2,
            'decay_time': 0.1,
            'sustain_level': 0.7,
            'release_time': 0.1
        }
        
        self.duration = (self.adsr_params['attack_time'] + 
                        self.adsr_params['decay_time'] + 
                        self.adsr_params['release_time'] + 0.2)
        self.total_duration = self.duration
        
        self.current_overtones = [self.base_frequency]
        self.natural_ratios = [1.0, 2.3, 3.1, 3.9, 5.4, 6.4]

class DroneTone(Voice):
    def __init__(self, frequency: float, fade_duration: float, sample_rate: int = 48000):
        super().__init__(frequency, sample_rate)
        
        self.adsr_params = {
            'min_interval': 0.5,
            'max_interval': 10.0,
            'attack_time': 2.0,
            'release_time': 15.0,
            'sustain_level': 0.8
        }
        
        self.fade_duration = fade_duration
        self.fade_in_duration = fade_duration
        self.fade_out_duration = fade_duration
        self.total_duration = fade_duration * 3
        
        self.rhythmic_mode = False
        self.current_pattern = []
        self.pattern_index = 0
        self.last_note_time = time.time()
