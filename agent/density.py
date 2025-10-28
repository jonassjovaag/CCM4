# density.py
# Density control and musical flow management

import time
import numpy as np
from typing import Dict, List, Optional
from collections import deque

class DensityController:
    """
    Controls musical density and flow
    Manages when to be active vs giving space
    """
    
    def __init__(self):
        self.target_density = 0.4  # Increased from 0.2 for better responsiveness
        self.current_density = 0.4  # Start with moderate density
        self.density_history = deque(maxlen=100)
        
        # Flow parameters
        self.flow_state = 'building'  # Start in building state for more engagement
        self.flow_intensity = 0.5  # Increased from 0.3 to 0.5
        self.last_flow_change = time.time()
        
        # Activity tracking
        self.activity_window = 30.0  # seconds
        self.activity_history = deque(maxlen=1000)
        
        # Density adaptation
        self.adaptation_rate = 0.1  # Increased from 0.05 to 0.1 for faster adaptation
        self.min_density = 0.05  # Reduced from 0.1 to 0.05
        self.max_density = 0.6  # Reduced from 0.9 to 0.6
        
    def update_activity(self, event_data: Dict):
        """Update activity tracking with new event"""
        current_time = time.time()
        
        # Calculate activity level
        activity_level = self._calculate_activity_level(event_data)
        
        # Store activity
        self.activity_history.append({
            'timestamp': current_time,
            'activity_level': activity_level,
            'rms_db': event_data.get('rms_db', -80.0),
            'onset': event_data.get('onset', False),
            'f0': event_data.get('f0', 0.0)
        })
        
        # Update density
        self._update_density()
        
        # Update flow state
        self._update_flow_state()
    
    def _calculate_activity_level(self, event_data: Dict) -> float:
        """Calculate activity level from event data with instrument awareness"""
        instrument = event_data.get('instrument', 'unknown')
        
        # Base activity from RMS
        rms_db = event_data.get('rms_db', -80.0)
        rms_activity = max(0.0, min(1.0, (rms_db + 80.0) / 60.0))
        
        # Onset activity
        onset_activity = 1.0 if event_data.get('onset', False) else 0.0
        
        # Frequency activity (pitch changes)
        f0 = event_data.get('f0', 0.0)
        freq_activity = 0.5 if f0 > 0 else 0.0
        
        # Instrument-specific activity calculation
        if instrument == "drums":
            # Drums: focus on onset density and spectral energy
            # High onset activity, moderate RMS, low frequency activity
            activity = (onset_activity * 0.6 + rms_activity * 0.3 + freq_activity * 0.1)
        elif instrument == "piano":
            # Piano: focus on pitch changes and harmonic content
            # High frequency activity, moderate RMS, low onset activity
            activity = (freq_activity * 0.6 + rms_activity * 0.3 + onset_activity * 0.1)
        elif instrument == "guitar":
            # Guitar: balanced approach
            # Moderate all activities
            activity = (freq_activity * 0.4 + rms_activity * 0.4 + onset_activity * 0.2)
        elif instrument == "bass":
            # Bass: focus on low-frequency content
            # High frequency activity, moderate RMS, low onset activity
            activity = (freq_activity * 0.5 + rms_activity * 0.4 + onset_activity * 0.1)
        elif instrument == "speech":
            # Speech: focus on spectral changes and moderate activity
            # Moderate frequency activity, moderate RMS, moderate onset activity
            activity = (freq_activity * 0.4 + rms_activity * 0.3 + onset_activity * 0.3)
        else:
            # Unknown instrument: use default calculation
            activity = (rms_activity * 0.4 + onset_activity * 0.4 + freq_activity * 0.2)
        
        return min(1.0, max(0.0, activity))
    
    def _update_density(self):
        """Update current density based on recent activity"""
        if not self.activity_history:
            return
        
        current_time = time.time()
        cutoff_time = current_time - self.activity_window
        
        # Get recent activity
        recent_activity = [
            entry['activity_level'] 
            for entry in self.activity_history 
            if entry['timestamp'] > cutoff_time
        ]
        
        if recent_activity:
            # Calculate average activity
            avg_activity = np.mean(recent_activity)
            
            # Adapt density towards target
            density_delta = (self.target_density - self.current_density) * self.adaptation_rate
            
            # Also adapt based on activity
            activity_delta = (avg_activity - self.current_density) * self.adaptation_rate * 0.5
            
            self.current_density += density_delta + activity_delta
            self.current_density = max(self.min_density, min(self.max_density, self.current_density))
            
            # Store density history
            self.density_history.append({
                'timestamp': current_time,
                'density': self.current_density,
                'avg_activity': avg_activity
            })
    
    def _update_flow_state(self):
        """Update musical flow state"""
        if len(self.density_history) < 10:
            return
        
        current_time = time.time()
        
        # Analyze density trend
        recent_densities = [entry['density'] for entry in list(self.density_history)[-10:]]
        density_trend = np.polyfit(range(len(recent_densities)), recent_densities, 1)[0]
        
        # Determine flow state
        if density_trend > 0.01:
            self.flow_state = 'building'
        elif density_trend < -0.01:
            self.flow_state = 'releasing'
        else:
            self.flow_state = 'balanced'
        
        # Update flow intensity
        self.flow_intensity = np.mean(recent_densities)
        
        self.last_flow_change = current_time
    
    def get_density_recommendation(self) -> Dict:
        """Get density recommendation for AI agent"""
        recommendation = {
            'target_density': self.target_density,
            'current_density': self.current_density,
            'flow_state': self.flow_state,
            'flow_intensity': self.flow_intensity,
            'should_act': self._should_act(),
            'should_give_space': self._should_give_space()
        }
        
        return recommendation
    
    def _should_act(self) -> bool:
        """Determine if agent should be active"""
        # Act more when density is low or building
        if self.flow_state == 'building' and self.current_density < 0.6:
            return True
        
        # Act when density is very low
        if self.current_density < 0.3:
            return True
        
        # Random chance based on flow intensity
        return np.random.random() < self.flow_intensity * 0.3
    
    def _should_give_space(self) -> bool:
        """Determine if agent should give space"""
        # Give space when density is high or releasing
        if self.flow_state == 'releasing' and self.current_density > 0.7:
            return True
        
        # Give space when density is very high
        if self.current_density > 0.8:
            return True
        
        # Random chance based on inverse flow intensity
        return np.random.random() < (1.0 - self.flow_intensity) * 0.4
    
    def set_target_density(self, density: float):
        """Set target density level"""
        self.target_density = max(0.0, min(1.0, density))
    
    def get_activity_summary(self, duration_seconds: float = 30.0) -> Dict:
        """Get activity summary for the last duration_seconds"""
        current_time = time.time()
        cutoff_time = current_time - duration_seconds
        
        recent_activity = [
            entry for entry in self.activity_history 
            if entry['timestamp'] > cutoff_time
        ]
        
        if not recent_activity:
            return {
                'count': 0,
                'avg_activity': 0.0,
                'max_activity': 0.0,
                'onset_count': 0,
                'avg_rms': -80.0
            }
        
        activities = [entry['activity_level'] for entry in recent_activity]
        onsets = [entry['onset'] for entry in recent_activity]
        rms_values = [entry['rms_db'] for entry in recent_activity]
        
        return {
            'count': len(recent_activity),
            'avg_activity': np.mean(activities),
            'max_activity': np.max(activities),
            'onset_count': sum(onsets),
            'avg_rms': np.mean(rms_values)
        }
