#!/usr/bin/env python3
"""
Synthetic audio test for MusicHal_9000
Generates test audio patterns to debug AI behavior
"""

import numpy as np
import sounddevice as sd
import time
import threading

class SyntheticAudioTest:
    def __init__(self, duration=10, sample_rate=44100):
        self.duration = duration
        self.sample_rate = sample_rate
        self.running = False
        self.thread = None
        
    def generate_melodic_pattern(self):
        """Generate a simple melodic pattern for testing"""
        # Simple melody: C4, E4, G4, C5, rest, repeat
        notes = [
            261.63,  # C4
            329.63,  # E4
            392.00,  # G4
            523.25,  # C5
            0,       # rest
        ]
        
        audio = []
        note_duration = 0.8  # seconds per note
        
        for note in notes:
            if note > 0:
                # Generate sine wave
                t = np.linspace(0, note_duration, int(self.sample_rate * note_duration))
                wave = 0.3 * np.sin(2 * np.pi * note * t)
                # Add envelope (attack, decay, sustain, release)
                envelope = np.ones_like(t)
                attack = int(0.1 * len(t))
                release = int(0.2 * len(t))
                envelope[:attack] = np.linspace(0, 1, attack)
                envelope[-release:] = np.linspace(1, 0, release)
                wave *= envelope
                audio.extend(wave)
            else:
                # Rest
                audio.extend(np.zeros(int(self.sample_rate * note_duration)))
        
        return np.array(audio, dtype=np.float32)
    
    def generate_bass_pattern(self):
        """Generate a sparse bass pattern for testing"""
        # Sparse bass: C2, rest, rest, rest, G2, rest, rest, rest
        notes = [
            65.41,   # C2
            0, 0, 0,
            98.00,   # G2
            0, 0, 0,
        ]
        
        audio = []
        note_duration = 0.5
        
        for note in notes:
            if note > 0:
                t = np.linspace(0, note_duration, int(self.sample_rate * note_duration))
                wave = 0.4 * np.sin(2 * np.pi * note * t)
                # Bass envelope - longer sustain
                envelope = np.ones_like(t)
                attack = int(0.05 * len(t))
                release = int(0.3 * len(t))
                envelope[:attack] = np.linspace(0, 1, attack)
                envelope[-release:] = np.linspace(1, 0, release)
                wave *= envelope
                audio.extend(wave)
            else:
                audio.extend(np.zeros(int(self.sample_rate * note_duration)))
        
        return np.array(audio, dtype=np.float32)
    
    def generate_test_audio(self):
        """Generate full test audio"""
        print("🎵 Generating synthetic test audio...")
        
        # Generate patterns
        melody = self.generate_melodic_pattern()
        bass = self.generate_bass_pattern()
        
        # Mix patterns (alternate for variety)
        audio = []
        for i in range(int(self.duration / 4)):  # 4 seconds per cycle
            if i % 2 == 0:
                audio.extend(melody)
            else:
                audio.extend(bass)
        
        # Trim to exact duration
        audio = np.array(audio)
        target_samples = int(self.duration * self.sample_rate)
        if len(audio) > target_samples:
            audio = audio[:target_samples]
        elif len(audio) < target_samples:
            audio = np.pad(audio, (0, target_samples - len(audio)))
        
        return audio
    
    def play_audio(self, audio):
        """Play audio through default output"""
        print(f"🔊 Playing {self.duration}s test audio...")
        sd.play(audio, self.sample_rate)
        sd.wait()
        print("✅ Test audio finished")
    
    def run_test(self):
        """Run the test"""
        audio = self.generate_test_audio()
        self.play_audio(audio)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Synthetic audio test")
    parser.add_argument('--duration', type=int, default=10, help='Test duration in seconds')
    args = parser.parse_args()
    
    test = SyntheticAudioTest(duration=args.duration)
    test.run_test()

