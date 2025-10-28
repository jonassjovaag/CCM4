"""
Real-time Rhythmic Context for Live Performance
Provides tempo, meter, and beat grid awareness to the AI agent
"""

import numpy as np
from typing import List, Tuple, Optional, Deque
from dataclasses import dataclass
from collections import deque
import time


@dataclass
class RhythmicContext:
    """Real-time rhythmic context for AI decision-making"""
    current_tempo: float  # BPM
    meter: Tuple[int, int]  # e.g., (4, 4) for 4/4 time
    beat_position: float  # Current position in measure (0.0 to meter[0])
    beat_grid: List[float]  # Timestamps of recent beats
    next_beat_time: float  # Predicted timestamp of next beat
    syncopation_level: float  # 0.0-1.0, how syncopated is the input
    rhythmic_density: str  # "sparse", "moderate", "dense"
    confidence: float  # Confidence in tempo/meter detection
    timestamp: float  # When this context was created
    last_onset_time: float  # Time of last onset


class RealtimeRhythmicDetector:
    """
    Lightweight real-time rhythmic detection for live performance
    Optimized for low latency (<10ms processing time)
    """
    
    def __init__(self, update_interval: float = 2.0):
        """
        Args:
            update_interval: How often to update tempo/meter estimates (seconds)
        """
        self.update_interval = update_interval
        
        # Onset tracking
        self.onset_times: Deque[float] = deque(maxlen=32)  # Last 32 onsets
        self.ioi_history: Deque[float] = deque(maxlen=16)  # Inter-onset intervals
        
        # Tempo/meter state
        self.current_tempo = 120.0  # Default BPM
        self.current_meter = (4, 4)  # Default 4/4
        self.beat_grid: List[float] = []
        self.last_beat_time = 0.0
        self.beat_phase = 0.0  # Where we are in the current beat (0.0-1.0)
        
        # Detection confidence
        self.tempo_confidence = 0.0
        self.meter_confidence = 0.0
        
        # Update timing
        self.last_update_time = 0.0
        self.last_context: Optional[RhythmicContext] = None
        
        # Activity tracking
        self.recent_activity: Deque[Tuple[float, float]] = deque(maxlen=100)  # (time, rms)
        self.syncopation_buffer: Deque[bool] = deque(maxlen=16)  # Off-beat vs on-beat
        
    def update_from_event(self, event_data: dict, current_time: float) -> RhythmicContext:
        """
        Main update method: analyze event and return rhythmic context
        
        Args:
            event_data: Audio event data (onset, ioi, rms_db, etc.)
            current_time: Current timestamp
            
        Returns:
            RhythmicContext with current rhythmic information
        """
        # Track activity
        rms_db = event_data.get('rms_db', -80.0)
        self.recent_activity.append((current_time, rms_db))
        
        # Track onsets for tempo detection
        if event_data.get('onset', False):
            self.onset_times.append(current_time)
            
            # Calculate IOI
            if len(self.onset_times) >= 2:
                ioi = current_time - self.onset_times[-2]
                self.ioi_history.append(ioi)
        
        # Periodically update tempo/meter estimates
        if current_time - self.last_update_time >= self.update_interval:
            self._update_tempo_meter_estimates()
            self.last_update_time = current_time
        
        # Update beat grid and position
        self._update_beat_grid(current_time)
        beat_position = self._calculate_beat_position(current_time)
        
        # Calculate syncopation level
        syncopation_level = self._calculate_syncopation()
        
        # Determine rhythmic density
        density = self._calculate_rhythmic_density()
        
        # Predict next beat
        next_beat_time = self._predict_next_beat(current_time)
        
        # Create context
        context = RhythmicContext(
            current_tempo=self.current_tempo,
            meter=self.current_meter,
            beat_position=beat_position,
            beat_grid=self.beat_grid[-8:],  # Last 8 beats
            next_beat_time=next_beat_time,
            syncopation_level=syncopation_level,
            rhythmic_density=density,
            confidence=min(self.tempo_confidence, self.meter_confidence),
            timestamp=current_time,
            last_onset_time=self.onset_times[-1] if self.onset_times else 0.0
        )
        
        self.last_context = context
        return context
    
    def _update_tempo_meter_estimates(self):
        """Update tempo and meter estimates from onset history"""
        if len(self.ioi_history) < 4:
            # Not enough data
            self.tempo_confidence = 0.0
            return
        
        # Estimate tempo from median IOI
        ioi_array = np.array(list(self.ioi_history))
        median_ioi = np.median(ioi_array)
        
        if median_ioi > 0:
            # Convert IOI to BPM
            estimated_tempo = 60.0 / median_ioi
            
            # Clamp to reasonable range (40-200 BPM)
            estimated_tempo = max(40.0, min(200.0, estimated_tempo))
            
            # Smooth tempo estimate
            alpha = 0.3
            self.current_tempo = alpha * estimated_tempo + (1 - alpha) * self.current_tempo
            
            # Cap tempo to 100 BPM max (live detection issues)
            self.current_tempo = min(self.current_tempo, 100.0)
            
            # Calculate confidence based on IOI consistency
            ioi_std = np.std(ioi_array)
            ioi_mean = np.mean(ioi_array)
            cv = ioi_std / (ioi_mean + 1e-6)  # Coefficient of variation
            self.tempo_confidence = max(0.0, 1.0 - min(1.0, cv * 2))
        
        # Simple meter detection (4/4 vs 3/4 vs 6/8)
        # For now, stick with 4/4 (most common)
        # TODO: Implement proper meter detection from accent patterns
        self.current_meter = (4, 4)
        self.meter_confidence = 0.5  # Moderate confidence
    
    def _update_beat_grid(self, current_time: float):
        """Update the beat grid based on current tempo"""
        beat_duration = 60.0 / self.current_tempo
        
        # Initialize or extend beat grid
        if not self.beat_grid:
            # Start beat grid from first onset or now
            start_time = self.onset_times[0] if self.onset_times else current_time
            self.beat_grid = [start_time]
            self.last_beat_time = start_time
        
        # Add beats up to current time
        while self.last_beat_time + beat_duration <= current_time:
            self.last_beat_time += beat_duration
            self.beat_grid.append(self.last_beat_time)
        
        # Keep only recent beats (last 16 beats = 4 measures in 4/4)
        if len(self.beat_grid) > 16:
            self.beat_grid = self.beat_grid[-16:]
    
    def _calculate_beat_position(self, current_time: float) -> float:
        """Calculate position within current measure (0.0 to meter[0])"""
        if not self.beat_grid:
            return 0.0
        
        beat_duration = 60.0 / self.current_tempo
        time_since_last_beat = current_time - self.last_beat_time
        
        # Calculate beat phase (0.0-1.0 within current beat)
        self.beat_phase = (time_since_last_beat / beat_duration) % 1.0
        
        # Calculate measure position (0.0 to 4.0 for 4/4)
        beats_per_measure = self.current_meter[0]
        num_beats_in_grid = len(self.beat_grid)
        beat_in_measure = num_beats_in_grid % beats_per_measure
        
        # Add fractional beat position
        measure_position = beat_in_measure + self.beat_phase
        
        return measure_position
    
    def _predict_next_beat(self, current_time: float) -> float:
        """Predict timestamp of next beat"""
        beat_duration = 60.0 / self.current_tempo
        return self.last_beat_time + beat_duration
    
    def _calculate_syncopation(self) -> float:
        """Calculate syncopation level based on off-beat onsets"""
        if len(self.onset_times) < 4 or not self.beat_grid:
            return 0.0
        
        # Check recent onsets against beat grid
        on_beat_count = 0
        off_beat_count = 0
        beat_threshold = 0.1  # 10% of beat duration
        beat_duration = 60.0 / self.current_tempo
        
        for onset_time in list(self.onset_times)[-8:]:  # Check last 8 onsets
            # Find nearest beat
            if self.beat_grid:
                time_to_nearest_beat = min(abs(onset_time - beat) for beat in self.beat_grid)
                
                if time_to_nearest_beat < beat_duration * beat_threshold:
                    on_beat_count += 1
                else:
                    off_beat_count += 1
        
        total = on_beat_count + off_beat_count
        if total == 0:
            return 0.0
        
        return off_beat_count / total
    
    def _calculate_rhythmic_density(self) -> str:
        """Calculate rhythmic density from recent activity"""
        if len(self.onset_times) < 2:
            return "sparse"
        
        # Count onsets in last 4 seconds
        current_time = time.time()
        recent_onsets = sum(1 for t in self.onset_times if current_time - t < 4.0)
        
        # Classify density
        if recent_onsets < 4:
            return "sparse"
        elif recent_onsets < 12:
            return "moderate"
        else:
            return "dense"
    
    def quantize_to_beat(self, target_time: float, mode: str = 'nearest') -> float:
        """
        Quantize a time to the nearest beat
        
        Args:
            target_time: Target timestamp
            mode: 'nearest', 'before', 'after', 'off_beat'
            
        Returns:
            Quantized timestamp
        """
        if not self.beat_grid:
            return target_time
        
        beat_duration = 60.0 / self.current_tempo
        
        if mode == 'nearest':
            # Snap to nearest beat
            nearest_beat = min(self.beat_grid, key=lambda b: abs(b - target_time))
            return nearest_beat
        
        elif mode == 'before':
            # Snap to beat before target
            before_beats = [b for b in self.beat_grid if b <= target_time]
            return before_beats[-1] if before_beats else target_time
        
        elif mode == 'after':
            # Snap to beat after target
            after_beats = [b for b in self.beat_grid if b > target_time]
            return after_beats[0] if after_beats else target_time + beat_duration
        
        elif mode == 'off_beat':
            # Snap to off-beat (halfway between beats)
            nearest_beat = min(self.beat_grid, key=lambda b: abs(b - target_time))
            return nearest_beat + beat_duration / 2
        
        return target_time
    
    def get_beat_strength(self, beat_position: float) -> float:
        """
        Get metrical strength of a beat position (0.0-1.0)
        Strong beats (1, 3) have higher strength than weak beats (2, 4)
        """
        beat_num = int(beat_position) % self.current_meter[0]
        
        # 4/4 meter hierarchy
        if self.current_meter == (4, 4):
            strengths = [1.0, 0.3, 0.6, 0.3]  # Beat 1 strongest, 3 medium, 2,4 weak
            return strengths[beat_num]
        
        # 3/4 meter hierarchy
        elif self.current_meter == (3, 4):
            strengths = [1.0, 0.4, 0.4]  # Beat 1 strongest
            return strengths[beat_num]
        
        # Default: first beat strong, others weak
        return 1.0 if beat_num == 0 else 0.5

