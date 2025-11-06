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
    # Override fields (None = use computed values)
    engagement_level_override: Optional[float] = None
    musical_momentum_override: Optional[float] = None
    confidence_threshold_override: Optional[float] = None
    behavior_mode_override: Optional[str] = None

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
        """Initialize the performance timeline manager"""
        self.config = config
        self.performance_arc = None
        self.scaled_arc = None
        self.performance_state = None
        
        # Load and scale the performance arc (if file exists)
        if self.config.arc_file_path and os.path.exists(self.config.arc_file_path):
            self._load_and_scale_arc()
        else:
            if self.config.arc_file_path:
                print(f"⚠️  Performance arc file not found: {self.config.arc_file_path}")
            print(f"📊 Using simple duration-based timeline: {self.config.duration_minutes} minutes")
        
        # Initialize the performance state (works with or without arc)
        self._initialize_performance_state()
    
    def _load_and_scale_arc(self):
        """Load the performance arc and scale it to the desired duration"""
        print(f"📂 Loading performance arc from: {self.config.arc_file_path}")
        
        if not os.path.exists(self.config.arc_file_path):
            print(f"⚠️  Arc file not found, skipping arc load")
            return
    
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
    
    def initialize_from_training_audio(self, audio_file: str, section_duration: float = 60.0):
        """
        Learn arc structure from long training recordings (e.g., Daybreak.wav).
        
        This method analyzes long-form improvisation using Brandtsegg rhythm ratios
        to detect natural section boundaries and tempo changes. It then maps these
        sections to arc phases based on tempo trajectory and rhythmic complexity.
        
        Perfect for recordings like Daybreak.wav (19 min improvisation) where tempo
        varies across sections instead of being globally constant.
        
        Args:
            audio_file: Path to training audio file (10-20 min improvisation)
            section_duration: Analysis window size in seconds (default 60s)
        """
        from rhythmic_engine.audio_file_learning.heavy_rhythmic_analyzer import HeavyRhythmicAnalyzer
        
        print(f"🎭 Learning performance arc structure from: {audio_file}")
        
        # Analyze long-form structure
        analyzer = HeavyRhythmicAnalyzer()
        sections = analyzer.analyze_long_form_improvisation(audio_file, section_duration)
        
        if len(sections) == 0:
            print("⚠️  No sections detected, cannot learn arc structure")
            return
        
        print(f"✅ Detected {len(sections)} sections")
        
        # Map sections to arc phases
        self._map_sections_to_arc_phases(sections)
        
        # Store section data for runtime reference
        self.learned_sections = sections
        
        print(f"🎭 Arc structure learned: {len(sections)} sections mapped to phases")
    
    def _map_sections_to_arc_phases(self, sections: List[Dict]):
        """
        Map detected sections to 5-phase arc structure.
        
        Uses tempo trajectory + rhythmic complexity to infer arc phases:
        - Opening: Low tempo, stable, low complexity
        - Development: Rising tempo/complexity
        - Peak: Highest tempo/density
        - Resolution: Decreasing tempo
        - Closing: Return to opening feel
        
        Args:
            sections: List of section dicts from analyze_long_form_improvisation()
        """
        if len(sections) == 0:
            return
        
        # Extract tempo and complexity trajectories
        tempos = np.array([s['local_tempo'] for s in sections])
        complexities = np.array([s['rhythmic_complexity'] for s in sections])
        densities = np.array([s['onset_density'] for s in sections])
        
        # Normalize to 0-1 range
        tempo_range = tempos.max() - tempos.min()
        if tempo_range < 10:  # Minimal tempo variation
            tempo_normalized = np.ones_like(tempos) * 0.5
        else:
            tempo_normalized = (tempos - tempos.min()) / tempo_range
        
        complexity_range = complexities.max() - complexities.min()
        if complexity_range < 1.0:
            complexity_normalized = np.ones_like(complexities) * 0.5
        else:
            complexity_normalized = (complexities - complexities.min()) / complexity_range
        
        # Combined energy metric (tempo + complexity + density)
        energy = (tempo_normalized + complexity_normalized + densities / densities.max()) / 3.0
        
        # Find peak section (highest energy)
        peak_idx = int(np.argmax(energy))
        
        # Assign phases
        for i, section in enumerate(sections):
            progress = i / len(sections)  # 0.0 to 1.0
            
            if progress < 0.15:
                # Opening: First 15%
                section['arc_phase'] = 'opening'
                section['engagement_level'] = 0.3 + 0.2 * (progress / 0.15)
            elif i < peak_idx:
                # Development: Building toward peak
                development_progress = (i - len(sections) * 0.15) / (peak_idx - len(sections) * 0.15)
                section['arc_phase'] = 'development'
                section['engagement_level'] = 0.5 + 0.3 * development_progress
            elif i == peak_idx:
                # Peak: Highest energy section
                section['arc_phase'] = 'peak'
                section['engagement_level'] = 0.9
            elif progress < 0.85:
                # Resolution: After peak, before closing
                resolution_progress = (i - peak_idx) / (len(sections) * 0.85 - peak_idx)
                section['arc_phase'] = 'resolution'
                section['engagement_level'] = 0.8 - 0.3 * resolution_progress
            else:
                # Closing: Final 15%
                closing_progress = (progress - 0.85) / 0.15
                section['arc_phase'] = 'closing'
                section['engagement_level'] = 0.5 - 0.3 * closing_progress
        
        # Print phase mapping
        print("🎭 Arc phase mapping:")
        for section in sections:
            tempo_change_str = f" (→{section['tempo_change']:.2f}x)" if section['tempo_change'] else ""
            print(f"   {section['start_time']:.0f}s: {section['arc_phase']:12s} "
                  f"| {section['local_tempo']:5.1f} BPM{tempo_change_str:12s} "
                  f"| engagement {section['engagement_level']:.2f}")
    
    def get_section_context(self, elapsed_time: float) -> Optional[Dict]:
        """
        Get learned section context for current performance moment.
        
        Returns section-specific tempo + rhythmic character from training audio analysis.
        Used to adapt behavioral modes to learned section structure.
        
        Args:
            elapsed_time: Current elapsed time in performance (seconds)
            
        Returns:
            Dict with arc_phase, local_tempo, dominant_ratios, onset_density, 
            rhythmic_complexity, engagement_level, section_progress
            OR None if no learned structure available
        """
        if not hasattr(self, 'learned_sections') or not self.learned_sections:
            return None
        
        # Find current section
        for section in self.learned_sections:
            if section['start_time'] <= elapsed_time < section['end_time']:
                section_progress = (elapsed_time - section['start_time']) / (section['end_time'] - section['start_time'])
                
                return {
                    'arc_phase': section['arc_phase'],
                    'local_tempo': section['local_tempo'],
                    'dominant_ratios': section['dominant_ratios'],
                    'onset_density': section['onset_density'],
                    'rhythmic_complexity': section['rhythmic_complexity'],
                    'engagement_level': section['engagement_level'],
                    'section_progress': section_progress,
                    'tempo_change': section['tempo_change']
                }
        
        # Fallback if beyond learned structure
        return None
    
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
        if self.scaled_arc:
            print(f"📊 Performance arc: {len(self.scaled_arc.phases)} phases")
        else:
            print(f"📊 Simple duration-based timeline (no arc phases)")
        print(f"🎚️ Engagement profile: {self.config.engagement_profile}")
        
        # Only initialize if not already initialized (avoid resetting start_time)
        if not self.performance_state:
            self._initialize_performance_state()
        else:
            # Just reset the start time for a new performance run
            self.performance_state.start_time = time.time()
            self.performance_state.current_time = 0.0
            self.performance_state.last_activity_time = time.time()
        self._update_current_phase()
    
    def is_complete(self) -> bool:
        """
        Check if the performance duration has been reached
        
        Returns True when current time exceeds total duration + grace period.
        The grace period allows final notes to finish naturally.
        """
        if not self.performance_state:
            return False
        # Add 3-second grace period after scheduled end to let final notes finish
        grace_period = 3.0
        return self.performance_state.current_time >= (self.performance_state.total_duration + grace_period)
    
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
    
    def get_performance_phase(self) -> str:
        """
        Get current performance phase based on elapsed time
        
        Returns: 'buildup', 'main', or 'ending'
        """
        if not self.performance_state:
            return 'main'
        
        total_duration = self.performance_state.total_duration
        elapsed = self.performance_state.current_time
        progress = elapsed / total_duration if total_duration > 0 else 0.0
        
        # Phase boundaries (configurable percentages)
        buildup_end = 0.15    # First 15% is buildup
        ending_start = 0.80   # Last 20% is ending (fade-out)
        
        if progress < buildup_end:
            return 'buildup'
        elif progress < ending_start:
            return 'main'
        else:
            return 'ending'
    
    def get_activity_multiplier(self) -> float:
        """
        Get activity level multiplier based on current performance phase
        
        Returns:
            Float from 0.0 to 1.0 representing activity level
            - Buildup: 0.3 → 1.0 (gradual increase)
            - Main: 1.0 (full activity)
            - Ending: 1.0 → 0.0 (gradual decrease)
        """
        if not self.performance_state:
            return 1.0
        
        phase = self.get_performance_phase()
        total_duration = self.performance_state.total_duration
        elapsed = self.performance_state.current_time
        progress = elapsed / total_duration if total_duration > 0 else 0.0
        
        if phase == 'buildup':
            # Gradual increase from 0.3 to 1.0 over first 15%
            buildup_progress = progress / 0.15  # 0.0 → 1.0
            return 0.3 + (0.7 * buildup_progress)  # 0.3 → 1.0
        
        elif phase == 'main':
            # Full activity
            return 1.0
        
        else:  # ending
            # Gradual decrease from 1.0 to 0.0 over last 20%
            ending_progress = (progress - 0.80) / 0.20  # 0.0 → 1.0
            fade_factor = 1.0 - ending_progress  # 1.0 → 0.0
            return fade_factor ** 2  # Smooth exponential fade
    
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
        if not self.scaled_arc:
            # No arc loaded - skip phase updates
            return
            
        current_time = self.performance_state.current_time
        
        for phase in self.scaled_arc.phases:
            if phase.start_time <= current_time < phase.end_time:
                self.performance_state.current_phase = phase
                return
        
        # If past all phases, stay on the last phase
        if self.scaled_arc.phases:
            self.performance_state.current_phase = self.scaled_arc.phases[-1]
    
    def _update_engagement_level(self, human_activity: bool):
        """Update engagement level based on phase and human activity"""
        if not self.performance_state.current_phase:
            # No arc - use simple fixed engagement
            self.performance_state.engagement_level = 0.7
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
        """Get current performance guidance based on state and 3-phase arc"""
        if not self.performance_state:
            return {
                'should_respond': False,
                'engagement_level': 0.0,
                'behavior_mode': 'wait',
                'confidence_threshold': 0.8,
                'silence_mode': True,
                'performance_phase': 'main',
                'activity_multiplier': 1.0
            }
        
        # Get 3-phase arc info
        performance_phase = self.get_performance_phase()
        activity_multiplier = self.get_activity_multiplier()
        
        # Determine if we should respond based on activity multiplier and silence mode
        base_should_respond = not self.performance_state.silence_mode if self.performance_state.current_phase else True
        
        # Apply activity multiplier to response probability
        if performance_phase == 'buildup':
            # During buildup, gradually increase response rate
            should_respond = base_should_respond and (random.random() < activity_multiplier)
        elif performance_phase == 'ending':
            # During ending, gradually decrease response rate
            should_respond = base_should_respond and (random.random() < activity_multiplier)
        else:
            # Main phase - full response
            should_respond = base_should_respond
        
        # Strategic re-entry logic during silence - adjusted by phase
        if self.performance_state.silence_mode and self.performance_state.current_phase:
            re_entry_chance = 0.0
            
            if performance_phase == 'buildup':
                # Very conservative during buildup
                re_entry_chance = 0.001 * activity_multiplier
            elif performance_phase == 'ending':
                # Almost never during ending (let it fade)
                re_entry_chance = 0.0
            else:
                # Main phase - use momentum-based logic
                if self.performance_state.musical_momentum > 0.7:
                    re_entry_chance = 0.05  # 5% chance
                elif self.performance_state.musical_momentum > 0.5:
                    re_entry_chance = 0.02  # 2% chance
                elif self.performance_state.musical_momentum > 0.3:
                    re_entry_chance = 0.01  # 1% chance
                else:
                    re_entry_chance = 0.001  # 0.1% chance
            
            should_respond = random.random() < re_entry_chance
        
        # Determine behavior mode based on 3-phase arc and loaded phase
        if self.performance_state.current_phase:
            phase = self.performance_state.current_phase
            # Use loaded arc phases if available
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
        else:
            # Fallback to 3-phase behavior if no arc loaded
            if performance_phase == 'buildup':
                behavior_mode = 'imitate'
                confidence_threshold = 0.8
            elif performance_phase == 'ending':
                behavior_mode = 'imitate'
                confidence_threshold = 0.9
            else:
                behavior_mode = 'contrast'
                confidence_threshold = 0.7
        
        # Adjust engagement level by activity multiplier
        base_engagement = self.performance_state.engagement_level if self.performance_state.current_phase else 0.5
        adjusted_engagement = base_engagement * activity_multiplier
        
        # Apply engagement level override if set
        if self.performance_state.engagement_level_override is not None:
            adjusted_engagement = self.performance_state.engagement_level_override
        
        # Apply behavior mode override if set
        if self.performance_state.behavior_mode_override is not None:
            behavior_mode = self.performance_state.behavior_mode_override
        
        # Adjust confidence threshold based on engagement
        confidence_threshold *= (1.0 - adjusted_engagement * 0.3)
        
        # Apply confidence threshold override if set
        if self.performance_state.confidence_threshold_override is not None:
            confidence_threshold = self.performance_state.confidence_threshold_override
        
        # Get momentum (use override if set)
        current_momentum = self.performance_state.musical_momentum if self.performance_state.current_phase else 0.5
        if self.performance_state.musical_momentum_override is not None:
            current_momentum = self.performance_state.musical_momentum_override
        
        guidance = {
            'should_respond': should_respond,
            'engagement_level': adjusted_engagement,
            'behavior_mode': behavior_mode,
            'confidence_threshold': confidence_threshold,
            'silence_mode': self.performance_state.silence_mode if self.performance_state.current_phase else False,
            'performance_phase': performance_phase,
            'activity_multiplier': activity_multiplier,
            'musical_momentum': current_momentum,
            'time_remaining': self.performance_state.total_duration - self.performance_state.current_time,
            'detected_instrument': self.performance_state.detected_instrument if self.performance_state.current_phase else None
        }
        
        # Add arc-based info if available
        if self.performance_state.current_phase:
            guidance['current_phase'] = self.performance_state.current_phase.phase_type
            guidance['instrument_roles'] = self.performance_state.instrument_roles
        
        return guidance
    
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
    
    # Runtime parameter setters for live performance controls
    
    def set_engagement_profile(self, profile: str):
        """Set engagement profile (conservative/balanced/experimental)"""
        if profile in ['conservative', 'balanced', 'experimental']:
            self.config.engagement_profile = profile
            print(f"🎚️ Engagement profile set to: {profile}")
    
    def set_silence_tolerance(self, tolerance: float):
        """Set silence tolerance in seconds"""
        self.config.silence_tolerance = max(0.0, min(20.0, tolerance))
        print(f"🎚️ Silence tolerance set to: {tolerance:.1f}s")
    
    def set_adaptation_rate(self, rate: float):
        """Set adaptation rate (0.0-1.0)"""
        self.config.adaptation_rate = max(0.0, min(1.0, rate))
        print(f"🎚️ Adaptation rate set to: {rate:.2f}")
    
    def set_engagement_level_override(self, level: Optional[float]):
        """Override engagement level (None to use computed value)"""
        if self.performance_state:
            if level is not None:
                self.performance_state.engagement_level_override = max(0.0, min(1.0, level))
                print(f"🎚️ Engagement level override: {level:.2f}")
            else:
                self.performance_state.engagement_level_override = None
                print("🎚️ Engagement level override cleared (using computed value)")
    
    def set_momentum_override(self, momentum: Optional[float]):
        """Override musical momentum (None to use computed value)"""
        if self.performance_state:
            if momentum is not None:
                self.performance_state.musical_momentum_override = max(0.0, min(1.0, momentum))
                print(f"🎚️ Musical momentum override: {momentum:.2f}")
            else:
                self.performance_state.musical_momentum_override = None
                print("🎚️ Momentum override cleared (using computed value)")
    
    def set_confidence_override(self, confidence: Optional[float]):
        """Override confidence threshold (None to use computed value)"""
        if self.performance_state:
            if confidence is not None:
                self.performance_state.confidence_threshold_override = max(0.6, min(0.9, confidence))
                print(f"🎚️ Confidence threshold override: {confidence:.2f}")
            else:
                self.performance_state.confidence_threshold_override = None
                print("🎚️ Confidence override cleared (using computed value)")
    
    def set_behavior_mode_override(self, mode: Optional[str]):
        """Override behavior mode (None to use computed value)"""
        if self.performance_state:
            if mode and mode in ['imitate', 'contrast', 'lead', 'wait']:
                self.performance_state.behavior_mode_override = mode
                print(f"🎚️ Behavior mode override: {mode}")
            else:
                self.performance_state.behavior_mode_override = None
                print("🎚️ Behavior mode override cleared (using computed value)")

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
