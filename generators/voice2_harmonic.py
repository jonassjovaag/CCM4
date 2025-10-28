import time
import random
import numpy as np
from typing import Dict, List

class Voice2HarmonicController:
    """
    Voice 2: Harmonic voice with grouped note sequences
    - Plays 2-5 notes in groups
    - Random delays between groups
    - Uses musical modes (Nordic, 12-tone, harmonic minor)
    - Musical phrasing and expression
    """
    def __init__(self, pitch_tracker=None, thread_ctrl=None, logger=None, base_freq=220.0):
        self.pitch_tracker = pitch_tracker
        self.thread_ctrl = thread_ctrl
        self.logger = logger
        self.base_freq = base_freq
        
        # Group characteristics
        self.min_notes_per_group = 2
        self.max_notes_per_group = 4  # Reduced max notes per group
        self.min_group_delay = 8.0   # Much longer delays between groups
        self.max_group_delay = 20.0  # Much longer delays between groups
        
        # Note timing within groups
        self.min_note_interval = 0.3
        self.max_note_interval = 1.2
        self.note_duration = 0.8  # Individual note duration
        
        # Musical scales/modes
        self.modes = {
            'nordic': [0, 2, 3, 5, 7, 9, 10, 12],  # Nordic-inspired scale
            'harmonic_minor': [0, 2, 3, 5, 7, 8, 11, 12],  # Harmonic minor
            'dorian': [0, 2, 3, 5, 7, 9, 10, 12],  # Dorian mode
            'mixolydian': [0, 2, 4, 5, 7, 9, 10, 12],  # Mixolydian mode
            'pentatonic': [0, 2, 5, 7, 9, 12],  # Pentatonic scale
        }
        
        # Current state
        self.current_mode = 'nordic'
        self.last_group_time = time.time()  # Initialize to current time
        self.current_group = []
        self.group_index = 0
        self.next_group_delay = 12.0  # Start with longer delay
        
        # Harmonic progression
        self.harmonic_centers = [0, 3, 5, 7]  # Root, minor third, fifth, seventh
        self.current_center = 0
        
    def select_mode(self) -> str:
        """Select musical mode based on context"""
        # Change mode occasionally for variety
        if random.random() < 0.1:  # 10% chance to change mode
            return random.choice(list(self.modes.keys()))
        return self.current_mode
    
    def get_harmonic_base_frequency(self) -> float:
        """Get base frequency with harmonic awareness"""
        if self.pitch_tracker:
            current_pitch = self.pitch_tracker.get_current_pitch()
            if current_pitch > 50 and current_pitch < 600:
                # Use pitch input as harmonic center
                return current_pitch
        
        # Fallback to base frequency with harmonic progression
        harmonic_shift = self.harmonic_centers[self.current_center]
        semitone_ratio = 2 ** (harmonic_shift / 12.0)
        return self.base_freq * semitone_ratio
    
    def generate_note_group(self) -> List[Dict]:
        """Generate a group of harmonic notes"""
        current_time = time.time()
        
        # Select mode and number of notes
        mode = self.select_mode()
        self.current_mode = mode
        scale = self.modes[mode]
        
        num_notes = random.randint(self.min_notes_per_group, self.max_notes_per_group)
        base_freq = self.get_harmonic_base_frequency()
        
        # Generate note sequence
        notes = []
        accumulated_time = 0.0
        
        # Choose a melodic pattern for this group
        if random.random() < 0.3:  # 30% chance for ascending
            scale_indices = sorted(random.choices(range(len(scale)), k=num_notes))
        elif random.random() < 0.3:  # 30% chance for descending
            scale_indices = sorted(random.choices(range(len(scale)), k=num_notes), reverse=True)
        else:  # 40% chance for mixed/random
            scale_indices = random.choices(range(len(scale)), k=num_notes)
        
        for i, scale_index in enumerate(scale_indices):
            semitones = scale[scale_index]
            semitone_ratio = 2 ** (semitones / 12.0)
            frequency = base_freq * semitone_ratio
            
            # Amplitude varies for musical expression - MUCH lower for no distortion
            if i == 0:  # First note slightly louder
                amplitude = random.uniform(0.15, 0.25)  # Reduced significantly
            elif i == len(scale_indices) - 1:  # Last note softer
                amplitude = random.uniform(0.10, 0.20)  # Reduced significantly
            else:  # Middle notes
                amplitude = random.uniform(0.12, 0.22)  # Reduced significantly
            
            # Note interval timing
            if i < len(scale_indices) - 1:  # Not the last note
                interval = random.uniform(self.min_note_interval, self.max_note_interval)
            else:
                interval = 0  # Last note doesn't need interval
            
            note = {
                'type': 'harmonic_note',
                'frequency': frequency,
                'amplitude': amplitude,
                'duration': self.note_duration,
                'timestamp': current_time + accumulated_time,
                'voice': 'voice2',
                'mode': mode,
                'scale_degree': semitones,
                'group_position': i + 1,
                'group_size': num_notes
            }
            
            notes.append(note)
            accumulated_time += interval
        
        # Update harmonic center for next group
        if random.random() < 0.4:  # 40% chance to move harmonic center
            self.current_center = (self.current_center + 1) % len(self.harmonic_centers)
        
        return notes
    
    def should_start_new_group(self) -> bool:
        """Determine if it's time to start a new note group"""
        current_time = time.time()
        
        # Check if enough time has passed since last group
        if current_time >= self.last_group_time + self.next_group_delay:
            return True
        
        return False
    
    def get_pending_events(self) -> List[Dict]:
        """Get events ready to be sent to SuperCollider"""
        if not hasattr(self, 'pending_events'):
            self.pending_events = []
            
        current_time = time.time()
        ready_events = []
        remaining_events = []
        
        for event in self.pending_events:
            if current_time >= event['timestamp']:
                ready_events.append(event)
            else:
                remaining_events.append(event)
        
        self.pending_events = remaining_events
        return ready_events
    
    def run(self):
        """Main Voice2 harmonic loop"""
        while self.thread_ctrl.running.is_set():
            if self.should_start_new_group():
                # Generate new note group
                note_group = self.generate_note_group()
                
                if self.logger:
                    self.logger.log_performance_event({
                        'event': 'voice2_harmonic_group',
                        'num_notes': len(note_group),
                        'mode': self.current_mode,
                        'base_frequency': self.get_harmonic_base_frequency()
                    })
                
                # Store events for OSC handler
                if not hasattr(self, 'pending_events'):
                    self.pending_events = []
                
                for note in note_group:
                    self.pending_events.append(note)
                
                print(f"🎼 Voice2 harmonic group: {len(note_group)} notes in {self.current_mode} mode")
                
                # Update timing for next group
                self.last_group_time = time.time()
                self.next_group_delay = random.uniform(self.min_group_delay, self.max_group_delay)
            
            time.sleep(0.2)  # Check every 200ms
