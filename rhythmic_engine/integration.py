#!/usr/bin/env python3
"""
Rhythmic Engine Integration
Integration of rhythmic analysis with existing MusicHal 9000 system

This module provides:
- Integration with existing harmonic system
- Combined decision making
- Training pipeline integration
- Live performance integration
"""

import numpy as np
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Import existing components
from agent.behaviors import MusicalDecision, BehaviorMode
from memory.polyphonic_audio_oracle_mps import PolyphonicAudioOracleMPS

# Import new rhythmic components
from .audio_file_learning.heavy_rhythmic_analyzer import HeavyRhythmicAnalyzer, RhythmicAnalysis
from .audio_file_learning.lightweight_rhythmic_analyzer import LightweightRhythmicAnalyzer, LiveRhythmicContext
from .memory.rhythm_oracle import RhythmOracle
from .agent.rhythmic_behavior_engine import RhythmicBehaviorEngine, RhythmicDecision

@dataclass
class CombinedDecision:
    """Combined harmonic and rhythmic decision"""
    harmonic_decision: MusicalDecision
    rhythmic_decision: RhythmicDecision
    combined_confidence: float
    final_decision: MusicalDecision

class RhythmicEngineIntegration:
    """
    Integration of rhythmic engine with existing MusicHal 9000 system
    
    Provides seamless integration between:
    - Harmonic analysis (existing)
    - Rhythmic analysis (new)
    - Combined decision making
    - Training pipeline
    """
    
    def __init__(self):
        # Existing harmonic components
        self.harmonic_memory = None  # Will be set by main system
        self.harmonic_agent = None  # Will be set by main system
        
        # New rhythmic components
        self.heavy_rhythmic_analyzer = HeavyRhythmicAnalyzer()
        self.lightweight_rhythmic_analyzer = LightweightRhythmicAnalyzer()
        self.rhythm_oracle = RhythmOracle()
        self.rhythmic_agent = RhythmicBehaviorEngine()
        
        # Integration state
        self.rhythmic_enabled = True
        self.combined_mode = True
        
    def set_harmonic_components(self, harmonic_memory, harmonic_agent):
        """Set references to existing harmonic components"""
        self.harmonic_memory = harmonic_memory
        self.harmonic_agent = harmonic_agent
    
    def analyze_audio_file_rhythm(self, audio_file: str) -> RhythmicAnalysis:
        """
        Perform heavy rhythmic analysis on audio file
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            RhythmicAnalysis: Complete rhythmic analysis
        """
        return self.heavy_rhythmic_analyzer.analyze_rhythmic_structure(audio_file)
    
    def analyze_live_rhythm(self, audio_frame: np.ndarray) -> LiveRhythmicContext:
        """
        Perform lightweight rhythmic analysis on live audio
        
        Args:
            audio_frame: Audio frame to analyze
            
        Returns:
            LiveRhythmicContext: Current rhythmic context
        """
        return self.lightweight_rhythmic_analyzer.analyze_live_rhythm(audio_frame)
    
    def make_combined_decision(self, current_event: Dict, 
                              harmonic_context: Dict, 
                              rhythmic_context: LiveRhythmicContext) -> CombinedDecision:
        """
        Make combined harmonic and rhythmic decision
        
        Args:
            current_event: Current audio event
            harmonic_context: Harmonic context from existing system
            rhythmic_context: Rhythmic context from rhythmic engine
            
        Returns:
            CombinedDecision: Combined decision
        """
        # Get harmonic decision (existing system)
        if self.harmonic_agent and self.harmonic_memory:
            harmonic_decisions = self.harmonic_agent.decide_behavior(
                current_event, self.harmonic_memory, self.harmonic_memory
            )
            harmonic_decision = harmonic_decisions[0] if harmonic_decisions else None
        else:
            harmonic_decision = None
        
        # Get rhythmic decision (new system)
        rhythmic_context_dict = {
            'tempo': rhythmic_context.tempo,
            'rhythmic_density': rhythmic_context.rhythmic_density,
            'beat_position': rhythmic_context.beat_position,
            'confidence': rhythmic_context.confidence
        }
        
        learned_patterns = self._get_learned_patterns()
        rhythmic_decision = self.rhythmic_agent.decide_rhythmic_response(
            rhythmic_context_dict, learned_patterns
        )
        
        # Combine decisions
        combined_decision = self._combine_decisions(
            harmonic_decision, rhythmic_decision, rhythmic_context
        )
        
        return combined_decision
    
    def train_from_rhythmic_analysis(self, rhythmic_analysis: RhythmicAnalysis):
        """
        Train rhythmic system from analysis results
        
        Args:
            rhythmic_analysis: Results from heavy rhythmic analysis
        """
        # Extract patterns and add to rhythm oracle
        for pattern in rhythmic_analysis.patterns:
            pattern_data = {
                'tempo': pattern.tempo,
                'density': pattern.density,
                'syncopation': pattern.syncopation,
                'meter': pattern.meter,
                'pattern_type': pattern.pattern_type,
                'confidence': pattern.confidence,
                'context': {
                    'start_time': pattern.start_time,
                    'end_time': pattern.end_time
                }
            }
            
            self.rhythm_oracle.add_rhythmic_pattern(pattern_data)
        
        print(f"Trained rhythmic system with {len(rhythmic_analysis.patterns)} patterns")
    
    def _combine_decisions(self, harmonic_decision: Optional[MusicalDecision], 
                          rhythmic_decision: RhythmicDecision,
                          rhythmic_context: LiveRhythmicContext) -> CombinedDecision:
        """Combine harmonic and rhythmic decisions"""
        
        if harmonic_decision is None:
            # No harmonic decision - use rhythmic decision only
            final_decision = self._create_final_decision_from_rhythm(rhythmic_decision)
            combined_confidence = rhythmic_decision.confidence
        else:
            # Combine both decisions
            final_decision = self._merge_decisions(harmonic_decision, rhythmic_decision)
            combined_confidence = (harmonic_decision.confidence + rhythmic_decision.confidence) / 2
        
        return CombinedDecision(
            harmonic_decision=harmonic_decision,
            rhythmic_decision=rhythmic_decision,
            combined_confidence=combined_confidence,
            final_decision=final_decision
        )
    
    def _create_final_decision_from_rhythm(self, rhythmic_decision: RhythmicDecision) -> MusicalDecision:
        """Create final decision from rhythmic decision only"""
        
        if not rhythmic_decision.should_play:
            # Create silent decision
            return MusicalDecision(
                note=0,
                velocity=0,
                duration=0.0,
                attack_time=0.0,
                release_time=0.0,
                voice_type="melodic",
                confidence=rhythmic_decision.confidence
            )
        
        # Create playing decision based on rhythmic context
        # This is simplified - in a real implementation, you'd use more sophisticated mapping
        
        # Map rhythmic density to velocity
        velocity = int(60 + rhythmic_decision.density * 60)  # 60-120 range
        
        # Map pattern type to duration
        duration_map = {
            'dense': 0.3,
            'moderate': 0.5,
            'sparse': 0.8,
            'syncopated': 0.4,
            'complex': 0.6
        }
        duration = duration_map.get(rhythmic_decision.pattern_type, 0.5)
        
        # Use timing for attack time
        attack_time = rhythmic_decision.timing * 0.1  # Scale timing
        
        return MusicalDecision(
            note=60,  # Middle C - simplified
            velocity=velocity,
            duration=duration,
            attack_time=attack_time,
            release_time=duration * 0.3,
            voice_type="melodic",
            confidence=rhythmic_decision.confidence
        )
    
    def _merge_decisions(self, harmonic_decision: MusicalDecision, 
                        rhythmic_decision: RhythmicDecision) -> MusicalDecision:
        """Merge harmonic and rhythmic decisions"""
        
        # Start with harmonic decision
        final_decision = MusicalDecision(
            note=harmonic_decision.note,
            velocity=harmonic_decision.velocity,
            duration=harmonic_decision.duration,
            attack_time=harmonic_decision.attack_time,
            release_time=harmonic_decision.release_time,
            voice_type=harmonic_decision.voice_type,
            confidence=harmonic_decision.confidence
        )
        
        # Apply rhythmic modifications
        if rhythmic_decision.should_play:
            # Adjust timing based on rhythmic decision
            final_decision.attack_time = rhythmic_decision.timing * 0.1
            
            # Adjust duration based on rhythmic density
            if rhythmic_decision.density < 0.3:
                final_decision.duration *= 1.5  # Longer for sparse
            elif rhythmic_decision.density > 0.7:
                final_decision.duration *= 0.7  # Shorter for dense
            
            # Adjust velocity based on rhythmic confidence
            velocity_factor = 0.8 + (rhythmic_decision.confidence * 0.4)
            final_decision.velocity = int(final_decision.velocity * velocity_factor)
            final_decision.velocity = max(20, min(127, final_decision.velocity))
        else:
            # Rhythmic decision says don't play
            final_decision.velocity = 0
            final_decision.duration = 0.0
        
        return final_decision
    
    def _get_learned_patterns(self) -> List[Dict]:
        """Get learned patterns from rhythm oracle"""
        # Convert stored patterns to format expected by behavior engine
        patterns = []
        
        for pattern in self.rhythm_oracle.rhythmic_patterns:
            pattern_dict = {
                'tempo': pattern.tempo,
                'density': pattern.density,
                'syncopation': pattern.syncopation,
                'pattern_type': pattern.pattern_type,
                'confidence': pattern.confidence
            }
            patterns.append(pattern_dict)
        
        return patterns
    
    def get_rhythmic_statistics(self) -> Dict:
        """Get rhythmic system statistics"""
        return {
            'rhythmic_enabled': self.rhythmic_enabled,
            'combined_mode': self.combined_mode,
            'rhythm_oracle_stats': self.rhythm_oracle.get_rhythmic_statistics(),
            'rhythmic_agent_state': self.rhythmic_agent.get_behavior_state()
        }
    
    def save_rhythmic_data(self, filepath: str):
        """Save rhythmic system data"""
        self.rhythm_oracle.save_patterns(filepath)
    
    def load_rhythmic_data(self, filepath: str):
        """Load rhythmic system data"""
        self.rhythm_oracle.load_patterns(filepath)

def main():
    """Test the rhythmic engine integration"""
    integration = RhythmicEngineIntegration()
    
    print("🥁 Testing Rhythmic Engine Integration...")
    
    # Test heavy analysis
    audio_file = "input_audio/Grab-a-hold.mp3"
    if os.path.exists(audio_file):
        print(f"Testing heavy analysis with: {audio_file}")
        analysis = integration.analyze_audio_file_rhythm(audio_file)
        
        print(f"Analysis results:")
        print(f"  Tempo: {analysis.tempo:.1f} BPM")
        print(f"  Patterns: {len(analysis.patterns)}")
        print(f"  Syncopation: {analysis.syncopation_score:.3f}")
        
        # Train from analysis
        integration.train_from_rhythmic_analysis(analysis)
    
    # Test live analysis
    print("\nTesting live analysis...")
    test_frame = np.random.randn(512) * 0.1
    context = integration.analyze_live_rhythm(test_frame)
    
    print(f"Live context:")
    print(f"  Tempo: {context.tempo:.1f} BPM")
    print(f"  Density: {context.rhythmic_density:.2f}")
    print(f"  Beat position: {context.beat_position:.2f}")
    
    # Test combined decision
    print("\nTesting combined decision...")
    current_event = {'rms_db': -20.0, 'f0': 440.0}
    harmonic_context = {'chord': 'C', 'key': 'C'}
    
    combined_decision = integration.make_combined_decision(
        current_event, harmonic_context, context
    )
    
    print(f"Combined decision:")
    print(f"  Harmonic: {combined_decision.harmonic_decision}")
    print(f"  Rhythmic: {combined_decision.rhythmic_decision}")
    print(f"  Final: {combined_decision.final_decision}")
    print(f"  Confidence: {combined_decision.combined_confidence:.2f}")
    
    # Show statistics
    stats = integration.get_rhythmic_statistics()
    print(f"\nStatistics: {stats}")

if __name__ == "__main__":
    import os
    main()
