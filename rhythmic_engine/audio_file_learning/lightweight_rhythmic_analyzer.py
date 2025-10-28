#!/usr/bin/env python3
"""
Lightweight Rhythmic Analyzer
Real-time rhythmic analysis for live performance

This module provides fast, lightweight rhythmic analysis including:
- Real-time onset detection
- Tempo tracking and estimation
- Beat position calculation
- Rhythmic density monitoring
"""

import numpy as np
import librosa
from typing import Dict, Optional
from dataclasses import dataclass
import time

@dataclass
class LiveRhythmicContext:
    """Real-time rhythmic context for live performance"""
    onset_detected: bool
    tempo: float
    beat_position: float
    rhythmic_density: float
    time_since_last_onset: float
    confidence: float

class LightweightRhythmicAnalyzer:
    """
    Lightweight rhythmic analysis for real-time performance
    
    Provides fast, efficient rhythmic analysis suitable for live use:
    - Real-time onset detection
    - Tempo tracking and estimation
    - Beat position calculation
    - Rhythmic density monitoring
    """
    
    def __init__(self, sample_rate: int = 22050, hop_length: int = 512):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        
        # Real-time tracking variables
        self.current_tempo = 120.0
        self.beat_position = 0.0
        self.rhythmic_density = 0.5
        self.last_onset_time = 0.0
        self.onset_history = []
        
        # Analysis parameters
        self.onset_threshold = 0.3
        self.tempo_adaptation_rate = 0.1
        self.density_window = 5.0  # seconds
        self.max_onset_history = 50
        
        # Tempo estimation
        self.tempo_history = []
        self.tempo_confidence = 0.5
        
    def analyze_live_rhythm(self, audio_frame: np.ndarray) -> LiveRhythmicContext:
        """
        Analyze rhythm from a single audio frame
        
        Args:
            audio_frame: Audio frame to analyze
            
        Returns:
            LiveRhythmicContext: Current rhythmic context
        """
        current_time = time.time()
        
        # Detect onset
        onset_detected = self._detect_onset(audio_frame)
        
        # Update tempo if onset detected
        if onset_detected:
            self._update_tempo_estimation(current_time)
            self.last_onset_time = current_time
        
        # Update beat position
        self._update_beat_position(current_time)
        
        # Update rhythmic density
        self._update_rhythmic_density(current_time)
        
        # Calculate confidence
        confidence = self._calculate_confidence()
        
        return LiveRhythmicContext(
            onset_detected=onset_detected,
            tempo=self.current_tempo,
            beat_position=self.beat_position,
            rhythmic_density=self.rhythmic_density,
            time_since_last_onset=current_time - self.last_onset_time,
            confidence=confidence
        )
    
    def _detect_onset(self, audio_frame: np.ndarray) -> bool:
        """Detect onset in audio frame"""
        try:
            # Simple onset detection using RMS change
            rms = np.sqrt(np.mean(audio_frame**2))
            
            # Compare with previous frame (simplified)
            if len(self.onset_history) > 0:
                prev_rms = self.onset_history[-1]
                rms_change = abs(rms - prev_rms) / (prev_rms + 1e-6)
                
                if rms_change > self.onset_threshold:
                    self.onset_history.append(rms)
                    return True
            
            self.onset_history.append(rms)
            
            # Keep history manageable
            if len(self.onset_history) > self.max_onset_history:
                self.onset_history.pop(0)
            
            return False
            
        except Exception as e:
            print(f"Error in onset detection: {e}")
            return False
    
    def _update_tempo_estimation(self, current_time: float):
        """Update tempo estimation based on onset timing"""
        if self.last_onset_time > 0:
            # Calculate interval since last onset
            interval = current_time - self.last_onset_time
            
            # Only consider reasonable intervals (0.1 to 2.0 seconds)
            if 0.1 <= interval <= 2.0:
                # Convert interval to BPM
                estimated_tempo = 60.0 / interval
                
                # Clamp to reasonable range
                estimated_tempo = max(60.0, min(200.0, estimated_tempo))
                
                # Update tempo with adaptation
                self.current_tempo = (
                    self.current_tempo * (1 - self.tempo_adaptation_rate) +
                    estimated_tempo * self.tempo_adaptation_rate
                )
                
                # Store in history
                self.tempo_history.append(estimated_tempo)
                if len(self.tempo_history) > 20:
                    self.tempo_history.pop(0)
                
                # Update confidence based on consistency
                self._update_tempo_confidence()
    
    def _update_beat_position(self, current_time: float):
        """Update current beat position"""
        if self.current_tempo > 0:
            beat_interval = 60.0 / self.current_tempo
            
            # Calculate beat position (0.0 to 1.0)
            if self.last_onset_time > 0:
                time_since_beat = (current_time - self.last_onset_time) % beat_interval
                self.beat_position = time_since_beat / beat_interval
            else:
                self.beat_position = 0.0
    
    def _update_rhythmic_density(self, current_time: float):
        """Update rhythmic density over time"""
        # Count onsets in recent window
        window_start = current_time - self.density_window
        recent_onsets = [t for t in self.onset_history if t > window_start]
        
        # Calculate density (onsets per second)
        if self.density_window > 0:
            self.rhythmic_density = len(recent_onsets) / self.density_window
        else:
            self.rhythmic_density = 0.0
        
        # Normalize to 0-1 range
        self.rhythmic_density = min(self.rhythmic_density, 2.0) / 2.0
    
    def _update_tempo_confidence(self):
        """Update confidence in tempo estimation"""
        if len(self.tempo_history) < 2:
            self.tempo_confidence = 0.5
            return
        
        # Calculate consistency of tempo estimates
        tempos = np.array(self.tempo_history)
        tempo_std = np.std(tempos)
        tempo_mean = np.mean(tempos)
        
        # Confidence based on consistency (lower std = higher confidence)
        if tempo_mean > 0:
            consistency = 1.0 - (tempo_std / tempo_mean)
            self.tempo_confidence = max(0.0, min(1.0, consistency))
        else:
            self.tempo_confidence = 0.5
    
    def _calculate_confidence(self) -> float:
        """Calculate overall confidence in rhythmic analysis"""
        # Combine tempo confidence and density stability
        density_confidence = 0.5  # Simplified for now
        
        overall_confidence = (self.tempo_confidence * 0.7 + density_confidence * 0.3)
        
        return max(0.0, min(1.0, overall_confidence))
    
    def get_rhythmic_state(self) -> Dict:
        """Get current rhythmic state for debugging"""
        return {
            'tempo': self.current_tempo,
            'beat_position': self.beat_position,
            'rhythmic_density': self.rhythmic_density,
            'tempo_confidence': self.tempo_confidence,
            'time_since_last_onset': time.time() - self.last_onset_time,
            'onset_history_length': len(self.onset_history)
        }
    
    def reset(self):
        """Reset analyzer state"""
        self.current_tempo = 120.0
        self.beat_position = 0.0
        self.rhythmic_density = 0.5
        self.last_onset_time = 0.0
        self.onset_history = []
        self.tempo_history = []
        self.tempo_confidence = 0.5

def main():
    """Test the lightweight rhythmic analyzer"""
    analyzer = LightweightRhythmicAnalyzer()
    
    print("🥁 Testing lightweight rhythmic analyzer...")
    
    # Simulate audio frames
    for i in range(10):
        # Create synthetic audio frame with occasional onsets
        if i % 3 == 0:  # Simulate onset every 3rd frame
            audio_frame = np.random.randn(512) * 0.5
        else:
            audio_frame = np.random.randn(512) * 0.1
        
        context = analyzer.analyze_live_rhythm(audio_frame)
        
        print(f"Frame {i+1}: "
              f"Onset: {context.onset_detected}, "
              f"Tempo: {context.tempo:.1f}, "
              f"Beat: {context.beat_position:.2f}, "
              f"Density: {context.rhythmic_density:.2f}, "
              f"Conf: {context.confidence:.2f}")
    
    print(f"\nFinal state: {analyzer.get_rhythmic_state()}")

if __name__ == "__main__":
    main()
