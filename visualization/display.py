import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import time
from typing import Dict, Tuple, List

class PerformanceDisplay:
    def __init__(self, fig, performance_ctrl=None, state_manager=None):
        self.fig = fig
        self.performance_ctrl = performance_ctrl
        self.state_manager = state_manager
        self.performance_duration = self.performance_ctrl.performance_duration if performance_ctrl else 1500
        self.performance_start_time = time.time()
        self.voice_frequencies = {i: [] for i in range(1, 5)}
        
        # Initialize figure and axes
        self.fig.patch.set_facecolor('white')
        self.setup_main_plot()
        self.setup_info_display()
        
    def setup_main_plot(self):
        """Initialize the main plot area and timer display"""
        # Main plotting area
        self.ax_main = self.fig.add_subplot(111)
        self.ax_main.set_facecolor('white')
        self.ax_main.grid(True, alpha=0.3)
        self.ax_main.set_xlim(0, 1000)
        self.ax_main.set_ylim(0, 1)
        self.line1, = self.ax_main.plot([0], [0], 'b-', lw=2)
        
        # Timer display
        self.timer_text = self.fig.text(0.5, 0.95, 'Time Remaining: 00:00',
                                      transform=self.fig.transFigure,
                                      ha='center', va='center',
                                      color='black')
    
    def setup_info_display(self):
        """Initialize frequency information display"""
        self.ax_info = self.fig.add_axes([0.1, 0.02, 0.8, 0.05])
        self.ax_info.set_facecolor('white')
        self.ax_info.axis('off')
    
    def update_frequency_display(self, frequencies: Dict[str, List[float]]):
        freq_text = "Current Frequencies:\n"
        for voice_name, freqs in frequencies.items():
            freq_text += f"{voice_name}: {[round(f, 1) for f in freqs]}\n"
        self.ax_info.clear()
        self.ax_info.text(0, 0.5, freq_text, fontsize=10, color='black')
    
    def plot_spectrum(self, frequencies: np.ndarray, amplitudes: np.ndarray):
        self.ax_main.clear()
        self.ax_main.plot(frequencies, amplitudes, 'b-', lw=2)
        self.ax_main.set_facecolor('white')
        self.ax_main.grid(True, alpha=0.3)
        self.ax_main.set_xlim(0, 1000)
        self.ax_main.set_ylim(0, 1)
    
    def start_animation(self):
        self.animation = FuncAnimation(
            self.fig,
            self._update_display,
            interval=50,
            cache_frame_data=False,
            save_count=1000,
            blit=False  # Changed to False to maintain visibility
        )
    
    def _update_display(self, frame):
        """Updates the display for each animation frame."""
        # Update timer
        elapsed = time.time() - self.performance_start_time
        remaining = max(0, self.performance_duration - elapsed)
        minutes, seconds = divmod(int(remaining), 60)
        self.timer_text.set_text(f"Time Remaining: {minutes:02d}:{seconds:02d}")
        
        # Update frequency display without clearing
        frequencies = self._collect_current_frequencies()
        self.update_frequency_display(frequencies)
        
        # Only update spectrum if new data exists
        if hasattr(self, 'current_spectrum'):
            self.line1.set_data(self.current_spectrum['freq'],
                            self.current_spectrum['amp'])
        
        return (self.line1, self.timer_text)

    
    def cleanup(self):
        if hasattr(self, 'animation'):
            self.animation.event_source.stop()
        plt.close(self.fig)
    
    def _collect_current_frequencies(self) -> Dict[str, List[float]]:
        return {
            'Voice 1': self.voice_frequencies.get(1, []),
            'Voice 2': self.voice_frequencies.get(2, []),
            'Voice 3': self.voice_frequencies.get(3, []),
            'Voice 4': self.voice_frequencies.get(4, [])
        }
