from typing import Dict, Set, List
import json
from datetime import datetime
import time

class StateManager:
    MODE_GENERATIVE = 1
    MODE_HARMONIC = 2
    
    def __init__(self, logger=None):
        self.logger = logger
        self.current_mode = self.MODE_GENERATIVE
        self.current_state = {
            'intensity': 0.0,  # Start with zero intensity, not 1.0!
            'tension': 0.0,
            'volume_level': 0.0,  # Current volume level (0-1)
            'activity_level': 0.0,  # How much activity is happening (0-1)
            'modulation_pending': False,
            'active_voices': {0, 1, 2, 3},
            'voice1_frequencies': [],
            'voice2_frequencies': [],
            'voice3_frequencies': [],
            'base_frequency': 220.0,
            'recent_events': [],  # Track recent musical events
            'last_solo_time': 0.0  # When last solo was triggered
        }
               
    def get_frequency_generation_mode(self):
        return self.current_mode

    def update_state(self, new_state: Dict) -> None:
        """Updates the current state with new values and logs the change."""
        if not new_state:
            return
            
        # Log state before update
        if self.logger:
            self.logger.log_state({
                'event': 'state_update',
                'timestamp': datetime.now().isoformat(),
                'previous_state': self.current_state.copy(),
                'new_values': new_state
            })
            
        # Update state
        for key, value in new_state.items():
            self.current_state[key] = value
            
        # Log complete updated state
        if self.logger:
            self.logger.log_state({
                'event': 'state_complete',
                'timestamp': datetime.now().isoformat(),
                'current_state': self.current_state,
                'performance_data': {
                    'intensity': self.current_state['intensity'],
                    'tension': self.current_state['tension'],
                    'active_voices': list(self.current_state['active_voices']),
                    'frequencies': {
                        'voice1': self.current_state['voice1_frequencies'],
                        'voice2': self.current_state['voice2_frequencies'],
                        'voice3': self.current_state['voice3_frequencies'],
                        'base': self.current_state['base_frequency']
                    }
                }
            })

    def get_current_notes(self):
        return self.current_state.get('current_notes', {})

    def update_activity(self, voice_events=0, percussion_events=0, drone_active=False):
        """Update the current activity and intensity levels"""
        current_time = time.time()
        
        # Track recent events (last 30 seconds)
        self.current_state['recent_events'] = [
            event for event in self.current_state['recent_events'] 
            if current_time - event['time'] < 30.0
        ]
        
        # Add new events
        if voice_events > 0:
            self.current_state['recent_events'].append({
                'type': 'voice',
                'count': voice_events,
                'time': current_time
            })
        if percussion_events > 0:
            self.current_state['recent_events'].append({
                'type': 'percussion', 
                'count': percussion_events,
                'time': current_time
            })
        if drone_active:
            self.current_state['recent_events'].append({
                'type': 'drone',
                'count': 1,
                'time': current_time
            })
        
        # Calculate activity level (events per second over last 30 seconds)
        total_events = sum(event['count'] for event in self.current_state['recent_events'])
        self.current_state['activity_level'] = min(1.0, total_events / 30.0)
        
        # Calculate volume level based on number of active voices and recent activity
        active_voice_count = len([v for v in ['voice1', 'voice2', 'voice3'] if v in self.current_state.get('active_voices', [])])
        self.current_state['volume_level'] = min(1.0, (active_voice_count / 3.0) * 0.7 + (self.current_state['activity_level'] * 0.3))
        
        # Update overall intensity (combination of volume and activity)
        self.current_state['intensity'] = (self.current_state['volume_level'] * 0.6 + 
                                          self.current_state['activity_level'] * 0.4)

    def should_trigger_solo(self) -> bool:
        """Determine if a solo should be triggered based on intensity and timing"""
        current_time = time.time()
        intensity = self.current_state['intensity']
        
        # Don't trigger if too soon since last solo
        time_since_last_solo = current_time - self.current_state['last_solo_time']
        min_interval = 45.0  # Minimum 45 seconds between solos
        
        if time_since_last_solo < min_interval:
            return False
        
        # Calculate trigger probability based on intensity
        if intensity < 0.2:
            # Very low intensity - very rare solos (2% chance every 15 seconds)
            trigger_probability = 0.02
        elif intensity < 0.4:
            # Low intensity - rare solos (8% chance every 15 seconds)
            trigger_probability = 0.08
        elif intensity < 0.6:
            # Medium intensity - occasional solos (25% chance every 15 seconds)
            trigger_probability = 0.25
        elif intensity < 0.8:
            # High intensity - more frequent solos (50% chance every 15 seconds)
            trigger_probability = 0.50
        else:
            # Very high intensity (climax phase) - very frequent solos (80% chance every 15 seconds)
            trigger_probability = 0.80
        
        # Random trigger based on probability
        import random
        if random.random() < trigger_probability:
            self.current_state['last_solo_time'] = current_time
            return True
            
        return False

    def toggle_voice(self, voice_idx: int) -> None:
        voice_key = f'voice{voice_idx + 1}'
        if voice_key in self.current_state['active_voices']:
            self.current_state['active_voices'].remove(voice_key)
        else:
            self.current_state['active_voices'].add(voice_key)
            
        if self.logger:
            self.logger.log_state({
                'event': 'voice_toggle',
                'voice': voice_key,
                'active': voice_key in self.current_state['active_voices'],
                'timestamp': datetime.now().isoformat()
            })

    def get_voice_states(self) -> Dict[str, bool]:
        return {f'voice{i+1}': i in self.current_state['active_voices']
                for i in range(4)}

    def get_tension_level(self) -> float:
        return self.current_state['tension']

    def is_voice_active(self, voice_idx: int) -> bool:
        return f'voice{voice_idx + 1}' in self.current_state['active_voices']

    def get_state_for_osc(self) -> Dict:
        """Prepares state data formatted for OSC transmission"""
        return {
            'frequencies': {
                f'voice{i+1}': self.current_state[f'voice{i+1}_frequencies']
                for i in range(3)
            },
            'controls': {
                'intensity': self.current_state['intensity'],
                'tension': self.current_state['tension']
            },
            'active_voices': list(self.current_state['active_voices'])
        }
