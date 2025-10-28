import time
from typing import Dict, List

class PerformanceDirector:
    def __init__(self, duration_minutes=10, base_bpm=30.0):
        self.total_duration = duration_minutes * 60  # Convert to seconds
        self.start_time = None
        self.current_phase_index = 0
        self.base_bpm = base_bpm
        
        # Redesigned for proper musical arc: gentle start → build → climax → wind down → resolution
        phase_ratios = [0.18, 0.22, 0.25, 0.20, 0.15]  # Five phases with climax in middle
        
        self.phases = [
            {
                "name": "awakening",
                "duration": duration_minutes * phase_ratios[0],
                "description": "Gentle introduction, finding each other",
                "characteristics": {
                    "percussion_density": 0.2,
                    "voice_complexity": "simple",
                    "harmonic_palette": "consonant",
                    "response_sensitivity": 0.8,  # High sensitivity
                    "active_voices": ["voice1", "voice2", "voice5", "drone"],  # Add voice2 for more activity
                    "norwegian_ratio_weights": "conservative",
                    "rhythm_variation": 0.1
                }
            },
            {
                "name": "conversation", 
                "duration": duration_minutes * phase_ratios[1],
                "description": "Growing conversation, call-response",
                "characteristics": {
                    "percussion_density": 0.4,
                    "voice_complexity": "moderate", 
                    "harmonic_palette": "expanded",
                    "response_sensitivity": 0.7,  # Building responsiveness
                    "active_voices": ["voice1", "voice2", "voice3", "voice5", "drone"],  # Add voice2 to conversation phase
                    "norwegian_ratio_weights": "balanced",
                    "rhythm_variation": 0.3
                }
            },
            {
                "name": "climax",
                "duration": duration_minutes * phase_ratios[2], 
                "description": "Peak intensity, full engagement",
                "characteristics": {
                    "percussion_density": 1.0,
                    "voice_complexity": "complex",
                    "harmonic_palette": "full_spectrum", 
                    "response_sensitivity": 0.9,  # Peak responsiveness
                    "active_voices": ["voice1", "voice2", "voice3", "voice4", "voice5", "drone", "solo"],  # All voices
                    "norwegian_ratio_weights": "dynamic",
                    "rhythm_variation": 0.9
                }
            },
            {
                "name": "reflection",
                "duration": duration_minutes * phase_ratios[3],
                "description": "Winding down, contemplative",
                "characteristics": {
                    "percussion_density": 0.3,
                    "voice_complexity": "moderate",
                    "harmonic_palette": "consonant_return",
                    "response_sensitivity": 0.5,  # Calming down
                    "active_voices": ["voice1", "voice2", "voice5", "drone", "solo"],  # Keep voice5 for interaction
                    "norwegian_ratio_weights": "conservative",
                    "rhythm_variation": 0.4
                }
            },
            {
                "name": "resolution",
                "duration": duration_minutes * phase_ratios[4],
                "description": "Gentle conclusion, return to simplicity",
                "characteristics": {
                    "percussion_density": 0.1,
                    "voice_complexity": "simple",
                    "harmonic_palette": "resolution",
                    "response_sensitivity": 0.6,  # Gentle interaction
                    "active_voices": ["voice1", "voice5", "drone", "solo"],  # Keep voice5 responsive
                    "norwegian_ratio_weights": "conservative",
                    "rhythm_variation": 0.2
                }
            }
        ]
        
        # Calculate cumulative phase times for easy lookup
        self.phase_start_times = [0]
        cumulative = 0
        for phase in self.phases:
            cumulative += phase["duration"] * 60  # Convert minutes to seconds
            self.phase_start_times.append(cumulative)
    
    def start_performance(self):
        """Start the performance timer"""
        self.start_time = time.time()
        
    def get_elapsed_time(self) -> float:
        """Get elapsed time in seconds since performance start"""
        if self.start_time is None:
            return 0
        return time.time() - self.start_time
    
    def get_current_phase(self) -> Dict:
        """Get the current phase based on elapsed time"""
        if self.start_time is None:
            return self.phases[0]
            
        elapsed = self.get_elapsed_time()
        
        # Find which phase we're in
        for i, phase_start in enumerate(self.phase_start_times[:-1]):
            if elapsed >= phase_start and elapsed < self.phase_start_times[i + 1]:
                self.current_phase_index = i
                return self.phases[i]
        
        # If we're past all phases, return the last one
        self.current_phase_index = len(self.phases) - 1
        return self.phases[-1]
    
    def get_phase_progress(self) -> float:
        """Get progress through current phase (0.0 to 1.0)"""
        current_phase = self.get_current_phase()
        elapsed = self.get_elapsed_time()
        phase_start = self.phase_start_times[self.current_phase_index]
        phase_duration = current_phase["duration"] * 60
        
        if phase_duration == 0:
            return 1.0
            
        progress = (elapsed - phase_start) / phase_duration
        return min(1.0, max(0.0, progress))
    
    def get_performance_parameters(self) -> Dict:
        """Get current performance parameters for all system components"""
        current_phase = self.get_current_phase()
        progress = self.get_phase_progress()
        
        # Calculate dynamic parameters based on phase and progress
        characteristics = current_phase["characteristics"]
        
        return {
            "phase_name": current_phase["name"],
            "phase_progress": progress,
            "elapsed_time": self.get_elapsed_time(),
            "remaining_time": max(0, self.total_duration - self.get_elapsed_time()),
            
            # System parameters
            "percussion_density_multiplier": characteristics["percussion_density"],
            "voice_complexity": characteristics["voice_complexity"],
            "harmonic_palette": characteristics["harmonic_palette"],
            "response_sensitivity": characteristics["response_sensitivity"],
            "active_voices": characteristics["active_voices"],
            "norwegian_ratio_weights": characteristics["norwegian_ratio_weights"],
            "rhythm_variation": characteristics["rhythm_variation"],
            
            # Dynamic modulation based on progress within phase
            "phase_intensity": self._calculate_phase_intensity(progress),
            "transition_approaching": progress > 0.8  # Last 20% of phase
        }
    
    def _calculate_phase_intensity(self, progress: float) -> float:
        """Calculate intensity curve within a phase"""
        # Create an intensity curve that builds up and optionally releases
        if progress < 0.3:
            # Build up
            return progress / 0.3 * 0.5
        elif progress < 0.8:
            # Sustain/develop
            return 0.5 + ((progress - 0.3) / 0.5) * 0.4
        else:
            # Transition preparation or climax
            return 0.9 + ((progress - 0.8) / 0.2) * 0.1
    
    def should_advance_phase(self) -> bool:
        """Check if it's time to move to next phase"""
        elapsed = self.get_elapsed_time()
        current_phase_end = self.phase_start_times[self.current_phase_index + 1]
        return elapsed >= current_phase_end
    
    def is_performance_complete(self) -> bool:
        """Check if the performance duration has been reached"""
        return self.get_elapsed_time() >= self.total_duration
    
    def get_time_display(self) -> str:
        """Get formatted time display for UI"""
        elapsed = self.get_elapsed_time()
        remaining = max(0, self.total_duration - elapsed)
        
        elapsed_min = int(elapsed // 60)
        elapsed_sec = int(elapsed % 60)
        remaining_min = int(remaining // 60)
        remaining_sec = int(remaining % 60)
        
        return f"[{elapsed_min:02d}:{elapsed_sec:02d}] - {remaining_min:02d}:{remaining_sec:02d} remaining"
    
    def get_performance_summary(self) -> List[str]:
        """Get a summary of the performance structure for display"""
        summary = []
        cumulative = 0
        
        for i, phase in enumerate(self.phases):
            start_min = int(cumulative)
            end_min = int(cumulative + phase["duration"])
            summary.append(f"Phase {i+1} ({start_min}-{end_min}min): {phase['name'].title()} - {phase['description']}")
            cumulative += phase["duration"]
            
        return summary
