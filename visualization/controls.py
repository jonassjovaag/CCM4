import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons
from typing import Dict, Callable

class PerformanceControls:
    def __init__(self, fig, state_manager=None):
        self.fig = fig
        self.state_manager = state_manager
        self.sliders = {}
        self.buttons = {}
        self.toggles = None
        
    def setup_sliders(self):
        slider_color = 'lightgoldenrodyellow'
        
        # Parameter sliders
        self.sliders['tension'] = Slider(
            plt.axes([0.1, 0.25, 0.3, 0.02]),
            'Tension', 0.0, 1.0, valinit=0.5, 
            color=slider_color
        )
        
        self.sliders['evolution'] = Slider(
            plt.axes([0.1, 0.20, 0.3, 0.02]),
            'Evolution Rate', 0.1, 2.0, valinit=1.0, 
            color=slider_color
        )
        
        self.sliders['spread'] = Slider(
            plt.axes([0.1, 0.15, 0.3, 0.02]),
            'Harmonic Spread', 0.5, 2.0, valinit=1.0, 
            color=slider_color
        )
        
        self.sliders['duration'] = Slider(
            plt.axes([0.1, 0.10, 0.3, 0.02]),
            'Duration (min)', 5, 60, valinit=25, 
            color=slider_color
        )
        
        # Voice balance sliders
        voice_positions = [(0.5, y, 0.3, 0.02) for y in [0.25, 0.20, 0.15, 0.10]]
        for i, pos in enumerate(voice_positions, 1):
            self.sliders[f'voice{i}'] = Slider(
                plt.axes(pos),
                f'Voice {i}', 0.0, 1.0, valinit=0.8, 
                color=slider_color
            )

    def setup_buttons(self):
        # Solo button
        self.buttons['solo'] = Button(
            plt.axes([0.85, 0.35, 0.1, 0.05]),
            'SOLO',
            color='lightgoldenrodyellow'
        )
        
        # Voice toggles
        self.toggles = CheckButtons(
            plt.axes([0.85, 0.1, 0.1, 0.2]),
            ['V1', 'V2', 'V3', 'V4'],
            [True, True, True, True]
        )

    def connect_callbacks(self, callbacks: Dict[str, Callable]):
        for name, slider in self.sliders.items():
            if name in callbacks:
                slider.on_changed(callbacks[name])
                
        for name, button in self.buttons.items():
            if name in callbacks:
                button.on_clicked(callbacks[name])
                
        if self.toggles and 'voice_toggle' in callbacks:
            self.toggles.on_clicked(callbacks['voice_toggle'])

    def get_slider_value(self, name: str) -> float:
        return self.sliders[name].val if name in self.sliders else 0.0
