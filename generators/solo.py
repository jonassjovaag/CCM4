import numpy as np
from typing import List, Tuple
import time

class SoloGenerator:
    def __init__(self, freq_engine=None, midi_handler=None, state_manager=None, thread_ctrl=None, logger=None):
        # Core intervals for Nordic-inspired melodies
        self.intervals = np.array([0, 2, 3, 5, 7, 9, 10, 12, 14])
        self.base_unit = 5.0
        self.rhythm_scale_range = (0.25, 2.0)
        self.freq_engine = freq_engine
        self.midi_handler = midi_handler
        self.state_manager = state_manager
        self.thread_ctrl = thread_ctrl
        self.logger = logger
        
        # Melodic motifs and rhythm patterns remain the same
        self.motifs = [
            np.array([0, 3, 7, 5, 3]),
            np.array([7, 5, 3, 0, 2]),
            np.array([0, 5, 7, 12, 7, 5]),
            np.array([0, 2, 3, 5, 7, 5, 3, 2])
        ]
        
        self.rhythm_patterns = {
            'halling': np.array([0.25, 0.5, 0.25, 0.5, 0.25, 0.25]),
            'polska': np.array([0.35, 0.35, 0.3, 0.35, 0.35, 0.3]),
            'springar': np.array([0.4, 0.3, 0.3, 0.4, 0.3, 0.3]),
            'gangar': np.array([0.5, 0.25, 0.25, 1.0, 1.0, 0.25, 0.25]),
            'walking': np.array([0.25, 0.25, 0.5, 0.25, 0.25, 0.5]),
            'lyrical': np.array([0.75, 0.25, 0.5, 1.5, 0.25, 0.25])
        }
        
        self.pending_phrase = None  # Store phrases to be played by OSC handler

    def run(self):
        while self.thread_ctrl.running.is_set():
            if self.state_manager.should_trigger_solo():
                print(f"Solo: Triggering solo phrase (intensity: {self.state_manager.current_state['intensity']:.2f})")
                self._trigger_solo()
            
            # Check more frequently during high intensity periods
            intensity = self.state_manager.current_state.get('intensity', 0.5)
            if intensity > 0.6:  # High/climax phases
                time.sleep(8.0)   # Check every 8 seconds
            else:
                time.sleep(15.0)  # Check every 15 seconds normally

    def generate_phrase(self, base_freq: float) -> List[Tuple[float, float, int]]:
        phrase = []
        selected_motifs = [self.motifs[np.random.randint(len(self.motifs))] for _ in range(4)]
        melodic_shape = np.concatenate(selected_motifs)
        
        pattern_type = np.random.choice(list(self.rhythm_patterns.keys()))
        base_rhythms = self.rhythm_patterns[pattern_type]
        num_repeats = len(melodic_shape) // len(base_rhythms) + 1
        rhythms = np.tile(base_rhythms, num_repeats)[:len(melodic_shape)]
        
        volume_curve = self._create_dynamics_curve(len(melodic_shape))
        
        for i, interval in enumerate(melodic_shape):
            freq = base_freq * (2 ** (interval/12))
            freq *= np.random.uniform(0.995, 1.005)  # Slight random tuning
            velocity = int(volume_curve[i] * 127)
            duration = rhythms[i] * self.base_unit
            phrase.append((freq, duration, velocity))
        
        return phrase

    def _create_dynamics_curve(self, length: int) -> np.ndarray:
        positions = np.linspace(0, np.pi, length)
        main_arc = np.sin(positions) * 0.5 + 0.5
        micro_dynamics = np.sin(positions * 4) * 0.15
        emphasis = np.random.choice([0, 0.2], size=length, p=[0.7, 0.3])
        dynamics = main_arc + micro_dynamics + emphasis
        return np.clip(dynamics, 0.2, 0.95)

    def _trigger_solo(self):
        """Generate and play a solo phrase"""
        base_freq = self.freq_engine.base_freq
        phrase = self.generate_phrase(base_freq)
        
        # Store the phrase to be picked up by the OSC handler
        self.pending_phrase = phrase
        
        if self.logger:
            self.logger.log_performance_event({
                'event': 'solo_triggered',
                'base_freq': base_freq,
                'phrase_length': len(phrase),
                'intensity': self.state_manager.current_state['intensity']
            })
