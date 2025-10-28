#!/usr/bin/env python3
"""
Rhythmic Behavior Engine
AI decision making for rhythmic responses

This module provides:
- Rhythmic decision making for AI agent
- Silence/activity prediction based on rhythm
- Rhythmic pattern matching and generation
- Integration with existing behavior system
"""

import numpy as np
import random
import time
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

class RhythmicMode(Enum):
    """Rhythmic behavior modes"""
    SYNC = "sync"           # Sync with current rhythm
    CONTRAST = "contrast"   # Contrast with current rhythm
    LEAD = "lead"           # Lead rhythmic development
    SILENT = "silent"       # Maintain silence

@dataclass
class RhythmicDecision:
    """Rhythmic decision for AI agent"""
    should_play: bool
    timing: float  # When to play (beat position)
    density: float  # Rhythmic density (0.0 to 1.0)
    pattern_type: str
    confidence: float
    reasoning: str

class RhythmicBehaviorEngine:
    """
    Rhythmic decision making for AI agent
    
    Makes decisions about when to play, when to be silent,
    and what rhythmic patterns to use based on:
    - Current rhythmic context
    - Learned rhythmic patterns
    - Musical behavior mode
    """
    
    def __init__(self):
        self.current_mode = RhythmicMode.SYNC
        self.mode_history = []
        
        # Decision parameters
        self.silence_probability = 0.3
        self.density_matching_weight = 0.7
        self.tempo_matching_weight = 0.8
        
        # Timing controls
        self.min_decision_interval = 0.5  # seconds
        self.max_decision_interval = 3.0  # seconds
        self.last_decision_time = time.time()
        
        # Rhythmic preferences
        self.preferred_density_range = (0.3, 0.8)
        self.tempo_tolerance = 0.15  # 15% tolerance
        
    def decide_rhythmic_response(self, current_context: Dict, 
                                learned_patterns: List[Dict]) -> RhythmicDecision:
        """
        Make rhythmic decision based on current context
        
        Args:
            current_context: Current rhythmic context
            learned_patterns: Available learned patterns
            
        Returns:
            RhythmicDecision: Rhythmic decision
        """
        current_time = time.time()
        
        # Check if we should make a decision now
        if not self._should_make_decision(current_time):
            return self._create_silent_decision("Too soon for decision")
        
        # Determine behavior mode
        self._update_behavior_mode(current_context)
        
        # Make decision based on mode
        if self.current_mode == RhythmicMode.SYNC:
            decision = self._make_sync_decision(current_context, learned_patterns)
        elif self.current_mode == RhythmicMode.CONTRAST:
            decision = self._make_contrast_decision(current_context, learned_patterns)
        elif self.current_mode == RhythmicMode.LEAD:
            decision = self._make_lead_decision(current_context, learned_patterns)
        else:
            decision = self._create_silent_decision("Silent mode")
        
        # Update decision history
        self.last_decision_time = current_time
        self.mode_history.append(self.current_mode)
        
        # Keep history manageable
        if len(self.mode_history) > 100:
            self.mode_history.pop(0)
        
        return decision
    
    def _should_make_decision(self, current_time: float) -> bool:
        """Check if we should make a decision now"""
        time_since_last = current_time - self.last_decision_time
        
        # Check minimum interval
        if time_since_last < self.min_decision_interval:
            return False
        
        # Random chance based on interval
        if time_since_last > self.max_decision_interval:
            return True
        
        # Probability increases with time
        probability = (time_since_last - self.min_decision_interval) / \
                     (self.max_decision_interval - self.min_decision_interval)
        
        return random.random() < probability
    
    def _update_behavior_mode(self, current_context: Dict):
        """Update behavior mode based on context"""
        # Simple mode switching logic
        # In a real implementation, this would be more sophisticated
        
        if random.random() < 0.1:  # 10% chance to change mode
            modes = [RhythmicMode.SYNC, RhythmicMode.CONTRAST, RhythmicMode.LEAD]
            self.current_mode = random.choice(modes)
    
    def _make_sync_decision(self, current_context: Dict, learned_patterns: List[Dict]) -> RhythmicDecision:
        """Make decision to sync with current rhythm"""
        
        # Check if we should play based on current density
        current_density = current_context.get('rhythmic_density', 0.5)
        should_play = current_density > 0.3 and random.random() > self.silence_probability
        
        if not should_play:
            return self._create_silent_decision("Low density - staying silent")
        
        # Find similar patterns to match
        similar_patterns = self._find_similar_patterns(current_context, learned_patterns)
        
        if similar_patterns:
            # Use similar pattern
            chosen_pattern = random.choice(similar_patterns[:3])  # Top 3
            timing = self._calculate_sync_timing(current_context)
            density = chosen_pattern.get('density', current_density)
            pattern_type = chosen_pattern.get('pattern_type', 'sync')
            confidence = 0.8
            reasoning = f"Syncing with similar pattern: {pattern_type}"
        else:
            # Create new sync pattern
            timing = self._calculate_sync_timing(current_context)
            density = current_density * 0.9  # Slightly less dense
            pattern_type = 'sync'
            confidence = 0.6
            reasoning = "Creating sync pattern"
        
        return RhythmicDecision(
            should_play=True,
            timing=timing,
            density=density,
            pattern_type=pattern_type,
            confidence=confidence,
            reasoning=reasoning
        )
    
    def _make_contrast_decision(self, current_context: Dict, learned_patterns: List[Dict]) -> RhythmicDecision:
        """Make decision to contrast with current rhythm"""
        
        # Contrast by playing when density is low, or vice versa
        current_density = current_context.get('rhythmic_density', 0.5)
        
        if current_density > 0.7:
            # High density - play sparse
            should_play = random.random() < 0.6
            density = 0.2
            pattern_type = 'sparse'
            reasoning = "Contrasting high density with sparse pattern"
        else:
            # Low density - play dense
            should_play = random.random() < 0.8
            density = 0.8
            pattern_type = 'dense'
            reasoning = "Contrasting low density with dense pattern"
        
        if not should_play:
            return self._create_silent_decision("Contrast mode - strategic silence")
        
        timing = self._calculate_contrast_timing(current_context)
        confidence = 0.7
        
        return RhythmicDecision(
            should_play=True,
            timing=timing,
            density=density,
            pattern_type=pattern_type,
            confidence=confidence,
            reasoning=reasoning
        )
    
    def _make_lead_decision(self, current_context: Dict, learned_patterns: List[Dict]) -> RhythmicDecision:
        """Make decision to lead rhythmic development"""
        
        # Lead by introducing new patterns or changing tempo
        should_play = random.random() < 0.9  # High probability to play
        
        if not should_play:
            return self._create_silent_decision("Lead mode - building tension")
        
        # Choose interesting pattern
        interesting_patterns = [p for p in learned_patterns 
                              if p.get('pattern_type') in ['syncopated', 'complex', 'dense']]
        
        if interesting_patterns:
            chosen_pattern = random.choice(interesting_patterns)
            density = chosen_pattern.get('density', 0.7)
            pattern_type = chosen_pattern.get('pattern_type', 'lead')
            reasoning = f"Leading with interesting pattern: {pattern_type}"
        else:
            density = 0.7
            pattern_type = 'lead'
            reasoning = "Leading with new pattern"
        
        timing = self._calculate_lead_timing(current_context)
        confidence = 0.8
        
        return RhythmicDecision(
            should_play=True,
            timing=timing,
            density=density,
            pattern_type=pattern_type,
            confidence=confidence,
            reasoning=reasoning
        )
    
    def _create_silent_decision(self, reasoning: str) -> RhythmicDecision:
        """Create a decision to remain silent"""
        return RhythmicDecision(
            should_play=False,
            timing=0.0,
            density=0.0,
            pattern_type='silent',
            confidence=0.9,
            reasoning=reasoning
        )
    
    def _find_similar_patterns(self, current_context: Dict, learned_patterns: List[Dict]) -> List[Dict]:
        """Find patterns similar to current context"""
        similar_patterns = []
        
        current_tempo = current_context.get('tempo', 120.0)
        current_density = current_context.get('rhythmic_density', 0.5)
        
        for pattern in learned_patterns:
            pattern_tempo = pattern.get('tempo', 120.0)
            pattern_density = pattern.get('density', 0.5)
            
            # Calculate similarity
            tempo_sim = 1.0 - abs(current_tempo - pattern_tempo) / max(current_tempo, pattern_tempo)
            density_sim = 1.0 - abs(current_density - pattern_density)
            
            similarity = (tempo_sim * self.tempo_matching_weight + 
                         density_sim * self.density_matching_weight)
            
            if similarity > 0.6:  # Threshold for similarity
                similar_patterns.append(pattern)
        
        return similar_patterns
    
    def _calculate_sync_timing(self, current_context: Dict) -> float:
        """Calculate timing for sync decision"""
        beat_position = current_context.get('beat_position', 0.0)
        
        # Play on strong beats (0.0, 0.5) or slightly before
        if beat_position < 0.1 or 0.4 < beat_position < 0.6:
            return beat_position + 0.05  # Slightly ahead
        else:
            return 0.0  # Next strong beat
    
    def _calculate_contrast_timing(self, current_context: Dict) -> float:
        """Calculate timing for contrast decision"""
        beat_position = current_context.get('beat_position', 0.0)
        
        # Play on weak beats (0.25, 0.75) or off-beat
        if 0.2 < beat_position < 0.3 or 0.7 < beat_position < 0.8:
            return beat_position + 0.1
        else:
            return 0.25  # Weak beat
    
    def _calculate_lead_timing(self, current_context: Dict) -> float:
        """Calculate timing for lead decision"""
        beat_position = current_context.get('beat_position', 0.0)
        
        # Lead by playing slightly ahead of beat
        return beat_position + 0.1
    
    def get_behavior_state(self) -> Dict:
        """Get current behavior state for debugging"""
        return {
            'current_mode': self.current_mode.value,
            'mode_history_length': len(self.mode_history),
            'last_decision_time': self.last_decision_time,
            'silence_probability': self.silence_probability
        }

def main():
    """Test the rhythmic behavior engine"""
    engine = RhythmicBehaviorEngine()
    
    print("🥁 Testing Rhythmic Behavior Engine...")
    
    # Test contexts
    test_contexts = [
        {
            'tempo': 120.0,
            'rhythmic_density': 0.8,
            'beat_position': 0.0,
            'confidence': 0.9
        },
        {
            'tempo': 140.0,
            'rhythmic_density': 0.3,
            'beat_position': 0.5,
            'confidence': 0.7
        },
        {
            'tempo': 100.0,
            'rhythmic_density': 0.6,
            'beat_position': 0.25,
            'confidence': 0.8
        }
    ]
    
    # Test patterns
    test_patterns = [
        {
            'tempo': 120.0,
            'density': 0.7,
            'pattern_type': 'dense'
        },
        {
            'tempo': 140.0,
            'density': 0.4,
            'pattern_type': 'syncopated'
        },
        {
            'tempo': 100.0,
            'density': 0.8,
            'pattern_type': 'complex'
        }
    ]
    
    # Test decisions
    for i, context in enumerate(test_contexts):
        print(f"\nTest {i+1}: Context = {context}")
        
        decision = engine.decide_rhythmic_response(context, test_patterns)
        
        print(f"  Decision: Play={decision.should_play}, "
              f"Timing={decision.timing:.2f}, "
              f"Density={decision.density:.2f}, "
              f"Type={decision.pattern_type}, "
              f"Confidence={decision.confidence:.2f}")
        print(f"  Reasoning: {decision.reasoning}")
    
    print(f"\nBehavior state: {engine.get_behavior_state()}")

if __name__ == "__main__":
    main()
