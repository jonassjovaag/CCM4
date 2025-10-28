#!/usr/bin/env python3
"""
Performance Timeline Manager
Manages performance duration and applies learned musical arcs to live performance
"""

import os
import json
import time
import random
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from performance_arc_analyzer import PerformanceArc, MusicalPhase

@dataclass
class PerformanceState:
    """Current state of the performance"""
    start_time: float
    current_time: float
    total_duration: float
    current_phase: Optional[MusicalPhase]
    engagement_level: float
    instrument_roles: Dict[str, str]
    silence_mode: bool
    last_activity_time: float
    musical_momentum: float
    detected_instrument: str = "unknown"

@dataclass
class PerformanceConfig:
    """Configuration for a performance session"""
    duration_minutes: int
    arc_file_path: str
    engagement_profile: str  # 'conservative', 'balanced', 'experimental'
    silence_tolerance: float  # How long to wait before re-engaging
    adaptation_rate: float  # How quickly to adapt to live input

class PerformanceTimelineManager:
    """Manages performance timeline and applies learned arcs"""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.performance_arc: Optional[PerformanceArc] = None
        self.performance_state: Optional[PerformanceState] = None
        self.scaled_arc: Optional[PerformanceArc] = None
        
        # Load and scale the performance arc
        self._load_and_scale_arc()
        
        # Initialize performance state
        self._initialize_performance_state()
    
    def _load_and_scale_arc(self):
        """Load the performance arc and scale it to the desired duration"""
        print(f"📂 Loading performance arc from: {self.config.arc_file_path}")
        
        if not os.path.exists(self.config.arc_file_path):
            raise FileNotFoundError(f"Performance arc file not found: {self.config.arc_file_path}")
        
        with open(self.config.arc_file_path, 'r') as f:
            arc_data = json.load(f)
        
        # Reconstruct the original arc
        self.performance_arc = self._reconstruct_arc(arc_data)
        
        # Scale the arc to the desired duration
        self.scaled_arc = self._scale_arc_to_duration(
            self.performance_arc, 
            self.config.duration_minutes * 60
        )
        
        print(f"✅ Performance arc loaded and scaled to {self.config.duration_minutes} minutes")
        print(f"📊 Original duration: {self.performance_arc.total_duration:.2f}s")
        print(f"📊 Scaled duration: {self.scaled_arc.total_duration:.2f}s")
        print(f"🎼 Number of phases: {len(self.scaled_arc.phases)}")
    
    def _reconstruct_arc(self, arc_data: Dict) -> PerformanceArc:
        """Reconstruct PerformanceArc from JSON data"""
        phases = []
        for phase_data in arc_data['phases']:
            phases.append(MusicalPhase(**phase_data))
        
        return PerformanceArc(
            total_duration=arc_data['total_duration'],
            phases=phases,
            overall_engagement_curve=arc_data['overall_engagement_curve'],
            instrument_evolution=arc_data['instrument_evolution'],
            silence_patterns=arc_data['silence_patterns'],
            theme_development=arc_data['theme_development'],
            dynamic_evolution=arc_data['dynamic_evolution']
        )
    
    def _scale_arc_to_duration(self, original_arc: PerformanceArc, target_duration: float) -> PerformanceArc:
        """Scale the performance arc to a new duration"""
        scale_factor = target_duration / original_arc.total_duration
        
        # Scale phases
        scaled_phases = []
        for phase in original_arc.phases:
            scaled_phase = MusicalPhase(
                start_time=phase.start_time * scale_factor,
                end_time=phase.end_time * scale_factor,
                phase_type=phase.phase_type,
                engagement_level=phase.engagement_level,
                instrument_roles=phase.instrument_roles,
                musical_density=phase.musical_density,
                dynamic_level=phase.dynamic_level,
                silence_ratio=phase.silence_ratio
            )
            scaled_phases.append(scaled_phase)
        
        # Scale engagement curve
        original_curve_length = len(original_arc.overall_engagement_curve)
        scaled_curve_length = int(original_curve_length * scale_factor)
        scaled_engagement_curve = np.interp(
            np.linspace(0, original_curve_length - 1, scaled_curve_length),
            np.arange(original_curve_length),
            original_arc.overall_engagement_curve
        ).tolist()
        
        # Scale dynamic evolution
        scaled_dynamic_evolution = np.interp(
            np.linspace(0, original_curve_length - 1, scaled_curve_length),
            np.arange(original_curve_length),
            original_arc.dynamic_evolution
        ).tolist()
        
        # Scale instrument evolution
        scaled_instrument_evolution = {}
        for instrument, evolution in original_arc.instrument_evolution.items():
            scaled_evolution = np.interp(
                np.linspace(0, len(evolution) - 1, int(len(evolution) * scale_factor)),
                np.arange(len(evolution)),
                evolution
            ).tolist()
            scaled_instrument_evolution[instrument] = scaled_evolution
        
        # Scale silence patterns
        scaled_silence_patterns = []
        for start_time, duration in original_arc.silence_patterns:
            scaled_silence_patterns.append((
                start_time * scale_factor,
                duration * scale_factor
            ))
        
        # Scale theme development
        scaled_theme_development = []
        for theme in original_arc.theme_development:
            scaled_theme = {
                'start_time': theme['start_time'] * scale_factor,
                'end_time': theme['end_time'] * scale_factor,
                'harmonic_complexity': theme['harmonic_complexity'],
                'tonal_stability': theme['tonal_stability'],
                'chroma_profile': theme['chroma_profile']
            }
            scaled_theme_development.append(scaled_theme)
        
        return PerformanceArc(
            total_duration=target_duration,
            phases=scaled_phases,
            overall_engagement_curve=scaled_engagement_curve,
            instrument_evolution=scaled_instrument_evolution,
            silence_patterns=scaled_silence_patterns,
            theme_development=scaled_theme_development,
            dynamic_evolution=scaled_dynamic_evolution
        )
    
    def _initialize_performance_state(self):
        """Initialize the performance state"""
        self.performance_state = PerformanceState(
            start_time=time.time(),
            current_time=0.0,
            total_duration=self.config.duration_minutes * 60,
            current_phase=None,
            engagement_level=0.0,
            instrument_roles={'drums': 'accompaniment', 'piano': 'accompaniment', 'bass': 'accompaniment'},
            silence_mode=False,
            last_activity_time=time.time(),
            musical_momentum=0.0
        )
    
    def start_performance(self):
        """Start a new performance session"""
        print(f"🎵 Starting performance session: {self.config.duration_minutes} minutes")
        print(f"📊 Performance arc: {len(self.scaled_arc.phases)} phases")
        print(f"🎚️ Engagement profile: {self.config.engagement_profile}")
        
        self._initialize_performance_state()
        self._update_current_phase()
    
    def is_complete(self) -> bool:
        """Check if the performance duration has been reached"""
        if not self.performance_state:
            return False
        # current_time is already elapsed time (set in update_performance_state)
        return self.performance_state.current_time >= self.performance_state.total_duration
    
    def get_time_remaining(self) -> float:
        """Get remaining time in seconds"""
        if not self.performance_state:
            return 0.0
        # current_time is already elapsed time (set in update_performance_state)
        remaining = self.performance_state.total_duration - self.performance_state.current_time
        return max(0.0, remaining)
    
    def get_fade_out_factor(self, fade_duration: float = 60.0) -> float:
        """
        Get fade-out multiplier for graceful ending
        
        Returns 1.0 until final fade_duration seconds, then smoothly decreases to 0.0
        
        Args:
            fade_duration: Duration of fade-out in seconds (default: 60s)
        
        Returns:
            Float from 1.0 (full activity) to 0.0 (complete fade)
        """
        remaining = self.get_time_remaining()
        if remaining > fade_duration:
            return 1.0
        elif remaining <= 0.0:
            return 0.0
        else:
            # Smooth exponential fade
            fade_progress = remaining / fade_duration
            return fade_progress ** 2  # Square for smoother fade
    
    def update_performance_state(self, human_activity: bool = False, instrument_detected: str = None):
        """Update the performance state based on current time and human activity"""
        current_time = time.time()
        elapsed_time = current_time - self.performance_state.start_time
        
        # Update current time
        self.performance_state.current_time = elapsed_time
        
        # Update last activity time if human is active
        if human_activity:
            self.performance_state.last_activity_time = current_time
        
        # Update current phase
        self._update_current_phase()
        
        # Update engagement level based on phase and human activity
        self._update_engagement_level(human_activity)
        
        # Update instrument roles based on detected instrument
        if instrument_detected:
            self._update_instrument_roles(instrument_detected)
        
        # Update silence mode
        self._update_silence_mode()
        
        # Update musical momentum
        self._update_musical_momentum(human_activity)
    
    def _update_current_phase(self):
        """Update the current phase based on elapsed time"""
        current_time = self.performance_state.current_time
        
        for phase in self.scaled_arc.phases:
            if phase.start_time <= current_time < phase.end_time:
                self.performance_state.current_phase = phase
                return
        
        # If we're past the last phase, use the last phase
        if self.scaled_arc.phases:
            self.performance_state.current_phase = self.scaled_arc.phases[-1]
    
    def _update_engagement_level(self, human_activity: bool):
        """Update engagement level based on phase and human activity"""
        if not self.performance_state.current_phase:
            return
        
        base_engagement = self.performance_state.current_phase.engagement_level
        
        # Adjust based on engagement profile
        if self.config.engagement_profile == 'conservative':
            engagement_multiplier = 0.7
        elif self.config.engagement_profile == 'balanced':
            engagement_multiplier = 1.0
        else:  # experimental
            engagement_multiplier = 1.3
        
        # Adjust based on human activity
        if human_activity:
            activity_multiplier = 1.2
        else:
            activity_multiplier = 0.8
        
        self.performance_state.engagement_level = min(1.0, 
            base_engagement * engagement_multiplier * activity_multiplier)
    
    def _update_instrument_roles(self, instrument_detected: str):
        """Update instrument roles based on detected instrument"""
        if not self.performance_state.current_phase:
            return
        
        # Get roles from current phase
        phase_roles = self.performance_state.current_phase.instrument_roles
        
        # Update the detected instrument's role
        if instrument_detected in phase_roles:
            self.performance_state.instrument_roles[instrument_detected] = phase_roles[instrument_detected]
        
        # Store the detected instrument for behavior decisions
        self.performance_state.detected_instrument = instrument_detected
    
    def _update_silence_mode(self):
        """Update silence mode based on time since last activity and strategic silence patterns"""
        current_time = time.time()
        time_since_activity = current_time - self.performance_state.last_activity_time
        
        # Get current phase for strategic silence
        if self.performance_state.current_phase:
            phase_silence_ratio = self.performance_state.current_phase.silence_ratio
            phase_type = self.performance_state.current_phase.phase_type
            
            # Strategic silence based on phase
            if phase_type == 'intro':
                # More tolerant of silence in intro
                silence_threshold = self.config.silence_tolerance * 1.5
            elif phase_type == 'climax':
                # Less tolerant of silence in climax
                silence_threshold = self.config.silence_tolerance * 0.7
            else:
                # Normal threshold for other phases
                silence_threshold = self.config.silence_tolerance
            
            # Adjust threshold based on phase silence ratio
            silence_threshold *= (1.0 + phase_silence_ratio)
        else:
            silence_threshold = self.config.silence_tolerance
        
        # Check if we should be in silence mode
        if time_since_activity > silence_threshold:
            self.performance_state.silence_mode = True
        else:
            self.performance_state.silence_mode = False
    
    def _update_musical_momentum(self, human_activity: bool):
        """Update musical momentum based on activity and phase"""
        if not self.performance_state.current_phase:
            return
        
        # Base momentum from phase
        base_momentum = self.performance_state.current_phase.musical_density
        
        # Adjust based on human activity
        if human_activity:
            momentum_adjustment = 0.1
        else:
            momentum_adjustment = -0.05
        
        # Update momentum with adaptation rate
        self.performance_state.musical_momentum = max(0.0, min(1.0,
            self.performance_state.musical_momentum + 
            (base_momentum - self.performance_state.musical_momentum) * self.config.adaptation_rate +
            momentum_adjustment
        ))
    
    def get_performance_guidance(self) -> Dict[str, Any]:
        """Get current performance guidance based on state"""
        if not self.performance_state or not self.performance_state.current_phase:
            return {
                'should_respond': False,
                'engagement_level': 0.0,
                'behavior_mode': 'wait',
                'confidence_threshold': 0.8,
                'silence_mode': True
            }
        
        # Determine if we should respond
        should_respond = not self.performance_state.silence_mode
        
        # Strategic re-entry logic - much more conservative
        if self.performance_state.silence_mode:
            # Check if we should re-enter based on momentum and phase
            if self.performance_state.musical_momentum > 0.7:
                # Very high momentum - small chance to re-enter
                should_respond = random.random() < 0.05  # 5% chance
            elif self.performance_state.musical_momentum > 0.5:
                # High momentum - very small chance
                should_respond = random.random() < 0.02  # 2% chance
            elif self.performance_state.musical_momentum > 0.3:
                # Medium momentum - minimal chance
                should_respond = random.random() < 0.01  # 1% chance
            else:
                # Low momentum - almost never re-enter
                should_respond = random.random() < 0.001  # 0.1% chance
        
        # Determine behavior mode based on phase and engagement
        phase = self.performance_state.current_phase
        if phase.phase_type == 'intro':
            behavior_mode = 'imitate'
            confidence_threshold = 0.8
        elif phase.phase_type == 'development':
            behavior_mode = 'contrast'
            confidence_threshold = 0.7
        elif phase.phase_type == 'climax':
            behavior_mode = 'lead'
            confidence_threshold = 0.6
        else:  # resolution
            behavior_mode = 'imitate'
            confidence_threshold = 0.8
        
        # Adjust confidence threshold based on engagement
        confidence_threshold *= (1.0 - self.performance_state.engagement_level * 0.3)
        
        return {
            'should_respond': should_respond,
            'engagement_level': self.performance_state.engagement_level,
            'behavior_mode': behavior_mode,
            'confidence_threshold': confidence_threshold,
            'silence_mode': self.performance_state.silence_mode,
            'current_phase': phase.phase_type,
            'instrument_roles': self.performance_state.instrument_roles,
            'musical_momentum': self.performance_state.musical_momentum,
            'time_remaining': self.performance_state.total_duration - self.performance_state.current_time,
            'detected_instrument': self.performance_state.detected_instrument
        }
    
    def get_performance_progress(self) -> Dict[str, Any]:
        """Get performance progress information"""
        if not self.performance_state:
            return {}
        
        progress_percent = (self.performance_state.current_time / self.performance_state.total_duration) * 100
        
        return {
            'elapsed_time': self.performance_state.current_time,
            'total_duration': self.performance_state.total_duration,
            'progress_percent': progress_percent,
            'current_phase': self.performance_state.current_phase.phase_type if self.performance_state.current_phase else 'unknown',
            'engagement_level': self.performance_state.engagement_level,
            'silence_mode': self.performance_state.silence_mode,
            'musical_momentum': self.performance_state.musical_momentum
        }
    
    def is_performance_complete(self) -> bool:
        """Check if the performance is complete"""
        if not self.performance_state:
            return True
        
        return self.performance_state.current_time >= self.performance_state.total_duration

def main():
    """Test the performance timeline manager"""
    # Create test configuration
    config = PerformanceConfig(
        duration_minutes=20,  # 20-minute performance
        arc_file_path="ai_learning_data/itzama_performance_arc.json",
        engagement_profile="balanced",
        silence_tolerance=5.0,  # 5 seconds
        adaptation_rate=0.1
    )
    
    # Create timeline manager
    timeline_manager = PerformanceTimelineManager(config)
    
    # Start performance
    timeline_manager.start_performance()
    
    # Simulate performance updates
    print("\n🎵 Simulating performance updates...")
    
    for i in range(10):
        # Simulate some human activity
        human_activity = i % 3 == 0  # Every 3rd update
        instrument_detected = ['piano', 'drums', 'bass'][i % 3]
        
        # Update performance state
        timeline_manager.update_performance_state(
            human_activity=human_activity,
            instrument_detected=instrument_detected
        )
        
        # Get performance guidance
        guidance = timeline_manager.get_performance_guidance()
        progress = timeline_manager.get_performance_progress()
        
        print(f"\n⏱️  Update {i+1}:")
        print(f"   Time: {progress['elapsed_time']:.1f}s / {progress['total_duration']:.1f}s ({progress['progress_percent']:.1f}%)")
        print(f"   Phase: {progress['current_phase']}")
        print(f"   Engagement: {progress['engagement_level']:.2f}")
        print(f"   Should respond: {guidance['should_respond']}")
        print(f"   Behavior mode: {guidance['behavior_mode']}")
        print(f"   Confidence threshold: {guidance['confidence_threshold']:.2f}")
        print(f"   Silence mode: {guidance['silence_mode']}")
        
        # Simulate time passing
        time.sleep(0.1)
    
    print(f"\n✅ Performance timeline manager test complete!")

if __name__ == "__main__":
    main()
