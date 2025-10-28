import random
import time

class FrequencyEngine:
    def __init__(self, initial_freq=220.0, state_manager=None, base_bpm=30.0, pitch_tracker=None):
        self.base_freq = initial_freq
        self.pitch_tracker = pitch_tracker
        self.start_time = time.time()
        self.adaptation_delay = 5  # Seconds before adapting
        
        # Norwegian traditional ratios
        self.norwegian_ratios = [
            1.0,        # Root
            1.125,      # Major second (9:8)
            1.2,        # Minor third (6:5)
            1.33,       # Perfect fourth (4:3)
            1.5,        # Perfect fifth (3:2)
            1.6,        # Minor sixth (8:5)
            1.8,         # Minor seventh (9:5)
            2.0,        # Octave (2:1)
            2.25,       # Major ninth (9:4)
            2.4         # Minor tenth (12:5)
        ]
        
        # Specific ratios for voice3 (root, fourth, fifth)
        self.voice3_ratios = [1.0, 1.333, 1.5]  # C, F, G
        
        self.evolution_stage = 0
        self.evolution_time = time.time()
        self.stage_duration = 120  # seconds per evolution stage
        
    def generate_voice_frequencies(self):
        current_time = time.time()
    
        if current_time - self.start_time > self.adaptation_delay and self.pitch_tracker:
            instant = self.pitch_tracker.get_instant_pitch()
            average = self.pitch_tracker.get_current_pitch()
            print(f"\rInstant: {instant:.1f} Hz | Average: {average:.1f} Hz", end='', flush=True)
            
            # Only update base frequency if we have a valid pitch reading
            if average > 50 and average < 1000:  # Reasonable frequency range
                # Smooth adaptation to prevent jarring changes
                adaptation_rate = 0.15  # More responsive adaptation
                self.base_freq = self.base_freq + adaptation_rate * (average - self.base_freq)
    
        # Clamp the base frequency to prevent it from going too high
        self.base_freq = min(self.base_freq, 220.0)  # A3 as maximum base frequency
    
        # Before using weights, create a weights list that matches exactly
        # the length of norwegian_ratios
        num_ratios = len(self.norwegian_ratios)
    
        # Create dynamic weights based on time and pitch input activity
        cycle_position = (current_time % 45) / 45  # Faster cycles for more variation
        evolution_cycle = (current_time % 180) / 180  # Longer evolution cycles
        
        # Add some randomness and pitch-responsive weighting
        import random
        base_weights = [0.1] * num_ratios
        
        if cycle_position < 0.25:
            # Consonant intervals with some randomness
            emphasis = [0.3, 0.1, 0.2, 0.15, 0.25, 0.1, 0.05, 0.1, 0.05, 0.05]
        elif cycle_position < 0.5:
            # Middle intervals
            emphasis = [0.05, 0.15, 0.2, 0.2, 0.15, 0.2, 0.15, 0.1, 0.1, 0.1]
        elif cycle_position < 0.75:
            # Wider intervals 
            emphasis = [0.05, 0.05, 0.1, 0.15, 0.1, 0.15, 0.2, 0.2, 0.15, 0.15]
        else:
            # Experimental/dissonant phase
            emphasis = [0.1, 0.2, 0.05, 0.1, 0.05, 0.1, 0.1, 0.1, 0.2, 0.2]
        
        # Mix base weights with emphasis and add evolution factor
        weights = []
        for i in range(min(len(base_weights), len(emphasis))):
            evolution_factor = 0.7 + 0.3 * evolution_cycle  # Gradually more experimental
            weight = base_weights[i] + (emphasis[i] * evolution_factor)
            weights.append(weight + random.uniform(-0.02, 0.02))  # Small random variation
    
        # Ensure the weights list has exactly the same length as the ratios
        if len(weights) > num_ratios:
            weights = weights[:num_ratios]
        elif len(weights) < num_ratios:
            weights.extend([0.05] * (num_ratios - len(weights)))
    
        # Generate frequencies with smooth transitions using base_freq with weights
        # Store previous frequencies for smoothing
        if not hasattr(self, 'prev_voice1_freq'):
            self.prev_voice1_freq = self.base_freq
            self.prev_voice2_freq = self.base_freq * 2
            self.prev_voice3_freq = self.base_freq / 2
        
        # Generate new target frequencies
        target_voice1_freq = self.base_freq * random.choices(self.norwegian_ratios, weights=weights, k=1)[0]
        target_voice2_freq = (self.base_freq * 2) * random.choices(self.norwegian_ratios, weights=weights, k=1)[0]
        target_voice3_freq = (self.base_freq / 2) * random.choice(self.voice3_ratios)
        
        # Dynamic smoothing based on pitch tracker activity
        if self.pitch_tracker and hasattr(self.pitch_tracker, 'current_instant'):
            current_average = self.pitch_tracker.get_current_pitch()
            current_instant = self.pitch_tracker.get_instant_pitch()
            if current_average > 0:
                pitch_stability = abs(current_instant - current_average) / current_average
                smoothing_factor = 0.05 + min(pitch_stability * 0.3, 0.25)  # More variation when pitch is changing
            else:
                smoothing_factor = 0.1
        else:
            smoothing_factor = 0.1
        voice1_freq = self.prev_voice1_freq + smoothing_factor * (target_voice1_freq - self.prev_voice1_freq)
        voice2_freq = self.prev_voice2_freq + smoothing_factor * (target_voice2_freq - self.prev_voice2_freq)
        voice3_freq = self.prev_voice3_freq + smoothing_factor * (target_voice3_freq - self.prev_voice3_freq)
        
        # Update previous frequencies
        self.prev_voice1_freq = voice1_freq
        self.prev_voice2_freq = voice2_freq
        self.prev_voice3_freq = voice3_freq
    
        # Clamp individual voice frequencies
        voice1_freq = min(voice1_freq, 880.0)  # A5 as maximum
        voice2_freq = min(voice2_freq, 1760.0)  # A6 as maximum
        voice3_freq = min(voice3_freq, 440.0)  # A4 as maximum
    
        return {
            'voice1_frequencies': [voice1_freq],
            'voice2_frequencies': [voice2_freq],
            'voice3_frequencies': [voice3_freq]
        }