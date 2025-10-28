from typing import List, Dict


class HarmonicManager:
    def __init__(self):
        # Existing ratios
        self.primary_ratios = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
        self.modal_centers = [1.0, 1.125, 1.25, 1.333, 1.5, 1.667, 1.875]
        self.tension_level = 0.0
        
        # New chord system
        self.chord_system = {
            'major': [1.0, 1.25, 1.5],
            'minor': [1.0, 1.2, 1.5],
            'dim': [1.0, 1.2, 1.4],
            'aug': [1.0, 1.25, 1.6]
        }
        
        self.progression = [
            ('C4', 'major', 8.0),
            ('G3', 'major', 8.0),
            ('Am3', 'minor', 8.0),
            ('F3', 'major', 8.0)
        ]
        self.progression_start_time = None

    def get_weighted_ratios(self, tension: float) -> List[float]:
        weights = [1.0 / (1.0 + tension * i) for i in range(len(self.primary_ratios))]
        total = sum(weights)
        return [w/total for w in weights]

    def evolve_harmonic_field(self, base_freq: float, tension: float) -> List[float]:
        weights = self.get_weighted_ratios(tension)
        ratios = random.choices(self.primary_ratios, weights=weights, k=3)
        return [base_freq * ratio for ratio in ratios]

    def get_current_chord(self, current_time: float):
        if not self.progression_start_time:
            self.progression_start_time = current_time
            
        elapsed = current_time - self.progression_start_time
        cycle_duration = sum(duration for _, _, duration in self.progression)
        position = elapsed % cycle_duration
        
        current_time = 0
        for root, quality, duration in self.progression:
            if current_time + duration > position:
                return root, quality
            current_time += duration
        
        return self.progression[0][0], self.progression[0][1]

    def generate_chord_frequencies(self, base_freq: float, current_time: float) -> Dict[str, List[float]]:
        root, quality = self.get_current_chord(current_time)
        chord_ratios = self.chord_system[quality]
        
        return {
            'voice1_frequencies': [base_freq * chord_ratios[0]],
            'voice2_frequencies': [base_freq * chord_ratios[1]],
            'voice3_frequencies': [base_freq * chord_ratios[2]],
            'base_frequency': base_freq
        }
