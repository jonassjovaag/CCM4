import numpy as np
from typing import List, Tuple
import time

class DroneController:
    def __init__(self, freq_engine=None, thread_ctrl=None, logger=None):
        # Rhythm patterns
        self.patterns = [
            [0.25, 0.25, 0.5, 0.5],    # eighth-eighth-quarter-quarter
            [0.5, 0.25, 0.25, 0.5],    # quarter-eighth-eighth-quarter
            [0.25, 0.5, 0.25, 0.5]     # eighth-quarter-eighth-quarter
        ]
        
        # Core components
        self.logger = logger
        self.freq_engine = freq_engine
        self.thread_ctrl = thread_ctrl
        
        # Pattern tracking
        self.current_pattern = []
        self.pattern_index = 0
        self.active_drones = []
        
        # Frequency ratios for drone generation
        self.freq_ratios = [1/8, 1/6, 1/4, 4/12]
        self.ratio_weights = [0.4, 0.3, 0.2, 0.1]

    def is_active(self):
        return bool(self.current_pattern)

    def generate_drone_frequency(self, voice1_freq: float) -> float:
        possible_freqs = [voice1_freq * ratio for ratio in self.freq_ratios]
        freq = np.random.choice(possible_freqs, p=self.ratio_weights)
        return self._clamp_bass_frequency(freq)

    def run(self):
        while self.thread_ctrl.running.is_set():
            if self.logger:
                self.logger.log_performance_event({'event': 'drone_generation'})
            self._add_drone_tone()
            time.sleep(np.random.uniform(10.0, 50.0))

    def _add_drone_tone(self):
        voice1_freq = self.freq_engine.base_freq
        drone_freq = self.generate_drone_frequency(voice1_freq)
        
        # Limit to maximum 2 active drones to prevent layering chaos
        if len(self.active_drones) >= 2:
            self.active_drones.pop(0)  # Remove oldest drone
        
        drone_data = {
            'frequency': drone_freq,
            'amplitude': self.calculate_drone_amplitude(time.time()),
            'pattern': self.generate_rhythm_pattern()
        }
        
        self.active_drones.append(drone_data)
        
        if self.logger:
            self.logger.log_drone_event({
                'frequency': drone_freq,
                'amplitude': drone_data['amplitude'],
                'timestamp': time.time()
            })

    def get_next_duration(self) -> float:
        if not self.current_pattern or self.pattern_index >= len(self.current_pattern):
            self.current_pattern = np.random.choice(self.patterns)
            self.pattern_index = 0
        duration = self.current_pattern[self.pattern_index]
        self.pattern_index += 1
        return duration

    def _clamp_bass_frequency(self, freq: float) -> float:
        return np.clip(freq, 32.70, 65.41)  # C1 to C2 range

    def generate_rhythm_pattern(self) -> List[float]:
        patterns = np.array([
            [0.25, 0.25, 0.5, 0.5],
            [0.5, 0.25, 0.25, 0.5],
            [0.25, 0.5, 0.25, 0.5]
        ])
        return patterns[np.random.randint(0, len(patterns))]

    def calculate_drone_amplitude(self, elapsed_time: float) -> float:
        lfo1 = 0.15 * np.sin(2 * np.pi * 1/3.3 * elapsed_time)
        lfo2 = 0.12 * np.sin(2 * np.pi * 1/7.1 * elapsed_time)
        lfo3 = 0.08 * np.sin(2 * np.pi * 1/11.3 * elapsed_time)
        return np.clip(0.15 + lfo1 + lfo2 + lfo3, 0.08, 0.25)  # Reduced significantly to prevent distortion
