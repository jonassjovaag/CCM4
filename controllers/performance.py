import time
from typing import Dict
from core.logger import PerformanceLogger
from core.manager import HarmonicManager


class PerformanceController:
    def __init__(self, duration_minutes: int = 25, state_manager=None, 
                 thread_ctrl=None, freq_engine=None, logger: PerformanceLogger = None):
        self.harmonic_manager = HarmonicManager() 
        self.performance_duration = duration_minutes * 60
        self.start_time = None
        self.fade_threshold = 0.85
        self.intensity = 1.0
        self.state_manager = state_manager
        self.thread_ctrl = thread_ctrl
        self.freq_engine = freq_engine
        self.logger = logger
        
        self.voice_states = {
            'voice1': {'active': True, 'muted': False},
            'voice2': {'active': True, 'muted': False},
            'voice3': {'active': True, 'muted': False},
            'voice4': {'active': True, 'muted': False}
        }


    def run(self):
        self.start_time = time.time()
        last_overtone_time = time.time()
        last_evolution_time = time.time()
        last_frequency_time = time.time()
        overtone_interval = 10.0
        evolution_interval = 50.0
        frequency_interval = 5

        while self.thread_ctrl.running.is_set():
            current_time = time.time()
            progress = (current_time - self.start_time) / self.performance_duration
            intensity = self.calculate_intensity()

            # Build overtones periodically
            if self.freq_engine and current_time - last_overtone_time >= overtone_interval:
                frequencies = self.freq_engine.generate_voice_frequencies()
                last_overtone_time = current_time
                if self.logger:
                    self.logger.log_performance_event({
                        'event': 'overtone_build',
                        'overtones': len(frequencies['voice1_frequencies'])
                    })

            # Evolve frequencies periodically
            if self.freq_engine and current_time - last_evolution_time >= evolution_interval:
                self.freq_engine.evolve_base_frequency()
                last_evolution_time = current_time
                if self.logger:
                    self.logger.log_performance_event({
                        'event': 'frequency_evolution',
                        'time': current_time - self.start_time
                    })

            # Generate new frequencies more frequently
            if self.freq_engine and current_time - last_frequency_time >= frequency_interval:
                frequencies = self.freq_engine.generate_voice_frequencies()
                state_update = {
                    'intensity': intensity,
                    'tension': progress * 2 if progress < 0.5 else (1 - progress) * 2,
                    'voice1_frequencies': frequencies['voice1_frequencies'],
                    'voice2_frequencies': frequencies['voice2_frequencies'],
                    'voice3_frequencies': frequencies['voice3_frequencies'],
                    'voice_states': self.voice_states,
                    'base_frequency': frequencies['base_frequency']
                }
                self.state_manager.update_state(state_update)
                last_frequency_time = current_time
                
                if self.logger:
                    self.logger.log_performance_event({
                        'event': 'state_update',
                        'state': state_update
                    })
            else:
                self.state_manager.update_state({
                    'intensity': intensity,
                    'voice_states': self.voice_states
                })

            phase = "Opening" if progress < 0.33 else "Middle" if progress < 0.66 else "Closing"
            if self.logger:
                self.logger.log_performance_event({
                    'phase': phase,
                    'intensity': intensity,
                    'progress': progress
                })
            
            time.sleep(0.05)

    def calculate_intensity(self) -> float:
        if not self.start_time:
            self.start_time = time.time()
            return 1.0
            
        elapsed = time.time() - self.start_time
        progress = elapsed / self.performance_duration
        
        if progress < 0.3:
            return 0.5 + (progress/0.3) * 0.5
        elif progress < 0.7:
            return 1.0
        elif progress < 0.85:
            reduction_progress = (progress - 0.7) / 0.15
            return 1.0 - (reduction_progress * 0.3)
        elif progress < 1.0:
            fade_duration = self.performance_duration * (1 - 0.85)
            fade_elapsed = elapsed - (self.performance_duration * 0.85)
            return (1.0 - (fade_elapsed / fade_duration)) * 0.7
        return 0.0

    def get_voice_state(self, voice_name: str) -> Dict:
        return self.voice_states.get(voice_name, {'active': False, 'muted': True})

    def toggle_voice(self, voice_name: str) -> None:
        if voice_name in self.voice_states:
            self.voice_states[voice_name]['muted'] = not self.voice_states[voice_name]['muted']
            if self.logger:
                self.logger.log_performance_event({
                    'event': 'voice_toggle',
                    'voice': voice_name,
                    'state': self.voice_states[voice_name]
                })
