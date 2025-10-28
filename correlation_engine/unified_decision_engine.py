"""
Unified Decision-Making Engine

This module implements AI decision-making that considers both harmonic and rhythmic
context, creating musically intelligent responses based on cross-modal patterns.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum
import random


class MusicalContext(Enum):
    """Musical context types for decision making"""
    VERSE = "verse"
    CHORUS = "chorus"
    BRIDGE = "bridge"
    SOLO = "solo"
    INTRO = "intro"
    OUTRO = "outro"
    TRANSITION = "transition"
    UNKNOWN = "unknown"


class ResponseMode(Enum):
    """Response modes for AI decisions"""
    IMITATE = "imitate"      # Follow the input pattern
    CONTRAST = "contrast"    # Create contrast to input
    LEAD = "lead"           # Lead in a new direction
    SUPPORT = "support"     # Support the harmonic/rhythmic structure
    EXPLORE = "explore"     # Explore new harmonic/rhythmic territory


@dataclass
class UnifiedMusicalDecision:
    """A musical decision that considers both harmonic and rhythmic context"""
    harmonic_decision: Dict[str, Any]  # Chord, key, harmonic direction
    rhythmic_decision: Dict[str, Any]  # Tempo, rhythm, timing
    joint_decision: Dict[str, Any]     # Cross-modal decision factors
    confidence: float
    context: MusicalContext
    response_mode: ResponseMode
    reasoning: str


@dataclass
class CrossModalContext:
    """Context that combines harmonic and rhythmic information"""
    harmonic_context: Dict[str, Any]
    rhythmic_context: Dict[str, Any]
    correlation_patterns: List[Dict[str, Any]]
    temporal_alignment: Dict[str, Any]
    musical_context: MusicalContext


class UnifiedDecisionEngine:
    """
    Unified decision-making engine that considers both harmonic and rhythmic
    context to make musically intelligent decisions
    """
    
    def __init__(self):
        self.correlation_patterns: Dict[str, Any] = {}
        self.temporal_alignments: List[Dict[str, Any]] = []
        self.musical_contexts: Dict[str, MusicalContext] = {}
        
        # Decision weights for different factors
        self.harmonic_weight = 0.4
        self.rhythmic_weight = 0.4
        self.correlation_weight = 0.2
        
        # Response mode probabilities
        self.response_mode_weights = {
            ResponseMode.IMITATE: 0.3,
            ResponseMode.CONTRAST: 0.2,
            ResponseMode.LEAD: 0.2,
            ResponseMode.SUPPORT: 0.2,
            ResponseMode.EXPLORE: 0.1
        }
    
    def load_correlation_patterns(self, data: Dict[str, Any]):
        """
        Load pre-computed correlation patterns from training
        
        Args:
            data: Dictionary with 'patterns' and 'temporal_alignments'
        """
        self.correlation_patterns = {
            p['pattern_id']: p for p in data.get('patterns', [])
        }
        self.temporal_alignments = data.get('temporal_alignments', [])
        
        print(f"🔗 Loaded {len(self.correlation_patterns)} correlation patterns")
    
    def make_unified_decision(self, 
                            harmonic_input: Dict[str, Any],
                            rhythmic_input: Dict[str, Any],
                            correlation_context: Optional[CrossModalContext] = None) -> UnifiedMusicalDecision:
        """
        Make a unified musical decision considering both harmonic and rhythmic context
        
        Args:
            harmonic_input: Current harmonic context
            rhythmic_input: Current rhythmic context
            correlation_context: Cross-modal correlation context
            
        Returns:
            UnifiedMusicalDecision with both harmonic and rhythmic decisions
        """
        
        # Analyze the input context
        input_context = self._analyze_input_context(harmonic_input, rhythmic_input)
        
        # Determine musical context
        musical_context = self._determine_musical_context(input_context, correlation_context)
        
        # Select response mode
        response_mode = self._select_response_mode(input_context, musical_context)
        
        # Make harmonic decision
        harmonic_decision = self._make_harmonic_decision(harmonic_input, input_context, response_mode)
        
        # Make rhythmic decision
        rhythmic_decision = self._make_rhythmic_decision(rhythmic_input, input_context, response_mode)
        
        # Create joint decision considering correlations
        joint_decision = self._create_joint_decision(
            harmonic_decision, rhythmic_decision, input_context, correlation_context
        )
        
        # Calculate overall confidence
        confidence = self._calculate_decision_confidence(
            harmonic_decision, rhythmic_decision, joint_decision, input_context
        )
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            harmonic_decision, rhythmic_decision, joint_decision, 
            musical_context, response_mode, input_context
        )
        
        return UnifiedMusicalDecision(
            harmonic_decision=harmonic_decision,
            rhythmic_decision=rhythmic_decision,
            joint_decision=joint_decision,
            confidence=confidence,
            context=musical_context,
            response_mode=response_mode,
            reasoning=reasoning
        )
    
    def _analyze_input_context(self, 
                             harmonic_input: Dict[str, Any],
                             rhythmic_input: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the combined input context"""
        
        context = {
            'harmonic_stability': self._calculate_harmonic_stability(harmonic_input),
            'rhythmic_stability': self._calculate_rhythmic_stability(rhythmic_input),
            'harmonic_complexity': self._calculate_harmonic_complexity(harmonic_input),
            'rhythmic_complexity': self._calculate_rhythmic_complexity(rhythmic_input),
            'overall_activity': self._calculate_overall_activity(harmonic_input, rhythmic_input),
            'tension_level': self._calculate_tension_level(harmonic_input, rhythmic_input)
        }
        
        return context
    
    def _calculate_harmonic_stability(self, harmonic_input: Dict[str, Any]) -> float:
        """Calculate harmonic stability (0 = unstable, 1 = stable)"""
        if not harmonic_input:
            return 0.5
        
        # Factors that increase stability
        key_stability = harmonic_input.get('key_stability', 0.5)
        chord_simplicity = 1.0 - harmonic_input.get('chord_diversity', 0.5)
        tension_level = 1.0 - harmonic_input.get('harmonic_tension', 0.5)
        
        stability = (key_stability + chord_simplicity + tension_level) / 3.0
        return max(0.0, min(1.0, stability))
    
    def _calculate_rhythmic_stability(self, rhythmic_input: Dict[str, Any]) -> float:
        """Calculate rhythmic stability (0 = unstable, 1 = stable)"""
        if not rhythmic_input:
            return 0.5
        
        # Factors that increase stability
        tempo_stability = rhythmic_input.get('tempo_stability', 0.5)
        syncopation_level = 1.0 - min(rhythmic_input.get('syncopation', 0.5), 1.0)
        density_consistency = 1.0 - abs(rhythmic_input.get('rhythmic_density', 0.5) - 0.5) * 2
        
        stability = (tempo_stability + syncopation_level + density_consistency) / 3.0
        return max(0.0, min(1.0, stability))
    
    def _calculate_harmonic_complexity(self, harmonic_input: Dict[str, Any]) -> float:
        """Calculate harmonic complexity (0 = simple, 1 = complex)"""
        if not harmonic_input:
            return 0.5
        
        chord_diversity = harmonic_input.get('chord_diversity', 0.5)
        tension_level = harmonic_input.get('harmonic_tension', 0.5)
        change_rate = harmonic_input.get('chord_change_rate', 0.5)
        
        complexity = (chord_diversity + tension_level + change_rate) / 3.0
        return max(0.0, min(1.0, complexity))
    
    def _calculate_rhythmic_complexity(self, rhythmic_input: Dict[str, Any]) -> float:
        """Calculate rhythmic complexity (0 = simple, 1 = complex)"""
        if not rhythmic_input:
            return 0.5
        
        syncopation = min(rhythmic_input.get('syncopation', 0.5), 1.0)
        density = rhythmic_input.get('rhythmic_density', 0.5)
        complexity = rhythmic_input.get('rhythmic_complexity', 0.5)
        
        complexity_score = (syncopation + density + complexity) / 3.0
        return max(0.0, min(1.0, complexity_score))
    
    def _calculate_overall_activity(self, 
                                 harmonic_input: Dict[str, Any],
                                 rhythmic_input: Dict[str, Any]) -> float:
        """Calculate overall musical activity level"""
        
        harmonic_activity = 0.0
        if harmonic_input:
            harmonic_activity = (
                harmonic_input.get('chord_change_rate', 0.5) +
                harmonic_input.get('event_count', 0) / 10.0
            ) / 2.0
        
        rhythmic_activity = 0.0
        if rhythmic_input:
            rhythmic_activity = (
                rhythmic_input.get('rhythmic_density', 0.5) +
                rhythmic_input.get('event_count', 0) / 10.0
            ) / 2.0
        
        return (harmonic_activity + rhythmic_activity) / 2.0
    
    def _calculate_tension_level(self, 
                              harmonic_input: Dict[str, Any],
                              rhythmic_input: Dict[str, Any]) -> float:
        """Calculate overall tension level"""
        
        harmonic_tension = harmonic_input.get('harmonic_tension', 0.5) if harmonic_input else 0.5
        rhythmic_tension = min(rhythmic_input.get('syncopation', 0.5), 1.0) if rhythmic_input else 0.5
        
        return (harmonic_tension + rhythmic_tension) / 2.0
    
    def _determine_musical_context(self, 
                                 input_context: Dict[str, Any],
                                 correlation_context: Optional[CrossModalContext]) -> MusicalContext:
        """Determine the musical context based on input and correlation patterns"""
        
        if correlation_context and correlation_context.musical_context != MusicalContext.UNKNOWN:
            return correlation_context.musical_context
        
        # Use heuristics to determine context
        stability = (input_context['harmonic_stability'] + input_context['rhythmic_stability']) / 2
        complexity = (input_context['harmonic_complexity'] + input_context['rhythmic_complexity']) / 2
        activity = input_context['overall_activity']
        tension = input_context['tension_level']
        
        # Simple context determination heuristics
        if tension > 0.7 and complexity > 0.6:
            return MusicalContext.BRIDGE
        elif stability > 0.7 and activity > 0.6:
            return MusicalContext.CHORUS
        elif stability > 0.5 and activity < 0.4:
            return MusicalContext.VERSE
        elif complexity > 0.7:
            return MusicalContext.SOLO
        else:
            return MusicalContext.UNKNOWN
    
    def _select_response_mode(self, 
                            input_context: Dict[str, Any],
                            musical_context: MusicalContext) -> ResponseMode:
        """Select response mode based on context"""
        
        # Adjust weights based on context
        weights = self.response_mode_weights.copy()
        
        # Context-based adjustments
        if musical_context == MusicalContext.CHORUS:
            weights[ResponseMode.SUPPORT] += 0.2
            weights[ResponseMode.IMITATE] += 0.1
        elif musical_context == MusicalContext.BRIDGE:
            weights[ResponseMode.CONTRAST] += 0.2
            weights[ResponseMode.EXPLORE] += 0.1
        elif musical_context == MusicalContext.SOLO:
            weights[ResponseMode.LEAD] += 0.2
            weights[ResponseMode.EXPLORE] += 0.1
        
        # Stability-based adjustments
        stability = (input_context['harmonic_stability'] + input_context['rhythmic_stability']) / 2
        if stability < 0.3:
            weights[ResponseMode.SUPPORT] += 0.2
        elif stability > 0.7:
            weights[ResponseMode.CONTRAST] += 0.1
        
        # Select mode based on weights
        modes = list(weights.keys())
        probabilities = list(weights.values())
        probabilities = np.array(probabilities)
        probabilities = probabilities / np.sum(probabilities)  # Normalize
        
        return np.random.choice(modes, p=probabilities)
    
    def _make_harmonic_decision(self, 
                             harmonic_input: Dict[str, Any],
                             input_context: Dict[str, Any],
                             response_mode: ResponseMode) -> Dict[str, Any]:
        """Make harmonic decision based on context and response mode"""
        
        decision = {
            'chord_progression': [],
            'key_change': None,
            'harmonic_direction': 'stable',
            'tension_change': 0.0,
            'complexity_change': 0.0
        }
        
        if not harmonic_input:
            return decision
        
        current_chord = harmonic_input.get('primary_chord', 'C')
        current_key = harmonic_input.get('key', 'C')
        current_tension = harmonic_input.get('harmonic_tension', 0.5)
        
        # Response mode-based decisions
        if response_mode == ResponseMode.IMITATE:
            decision['chord_progression'] = [current_chord]
            decision['harmonic_direction'] = 'stable'
            decision['tension_change'] = 0.0
            
        elif response_mode == ResponseMode.CONTRAST:
            # Create harmonic contrast
            decision['chord_progression'] = self._get_contrast_chords(current_chord)
            decision['harmonic_direction'] = 'contrast'
            decision['tension_change'] = -current_tension * 0.5  # Reduce tension
            
        elif response_mode == ResponseMode.LEAD:
            # Lead harmonically
            decision['chord_progression'] = self._get_leading_chords(current_chord)
            decision['harmonic_direction'] = 'progressive'
            decision['tension_change'] = 0.2  # Increase tension
            
        elif response_mode == ResponseMode.SUPPORT:
            # Support current harmony
            decision['chord_progression'] = [current_chord]
            decision['harmonic_direction'] = 'supportive'
            decision['tension_change'] = -0.1  # Slight tension reduction
            
        elif response_mode == ResponseMode.EXPLORE:
            # Explore new harmonic territory
            decision['chord_progression'] = self._get_exploratory_chords(current_chord)
            decision['harmonic_direction'] = 'exploratory'
            decision['tension_change'] = 0.3  # Increase tension for exploration
        
        return decision
    
    def _make_rhythmic_decision(self, 
                              rhythmic_input: Dict[str, Any],
                              input_context: Dict[str, Any],
                              response_mode: ResponseMode) -> Dict[str, Any]:
        """Make rhythmic decision based on context and response mode"""
        
        decision = {
            'tempo_change': 0.0,
            'rhythmic_pattern': 'stable',
            'syncopation_change': 0.0,
            'density_change': 0.0,
            'rhythmic_direction': 'stable'
        }
        
        if not rhythmic_input:
            return decision
        
        current_tempo = rhythmic_input.get('tempo', 120)
        current_syncopation = rhythmic_input.get('syncopation', 0.5)
        current_density = rhythmic_input.get('rhythmic_density', 0.5)
        
        # Response mode-based decisions
        if response_mode == ResponseMode.IMITATE:
            decision['rhythmic_pattern'] = 'imitate'
            decision['rhythmic_direction'] = 'stable'
            
        elif response_mode == ResponseMode.CONTRAST:
            # Create rhythmic contrast
            decision['tempo_change'] = -current_tempo * 0.1  # Slow down
            decision['syncopation_change'] = -current_syncopation * 0.3  # Reduce syncopation
            decision['rhythmic_direction'] = 'contrast'
            
        elif response_mode == ResponseMode.LEAD:
            # Lead rhythmically
            decision['tempo_change'] = current_tempo * 0.05  # Slight speed up
            decision['syncopation_change'] = 0.2  # Increase syncopation
            decision['rhythmic_direction'] = 'progressive'
            
        elif response_mode == ResponseMode.SUPPORT:
            # Support current rhythm
            decision['rhythmic_pattern'] = 'support'
            decision['rhythmic_direction'] = 'supportive'
            
        elif response_mode == ResponseMode.EXPLORE:
            # Explore new rhythmic territory
            decision['tempo_change'] = current_tempo * 0.1  # More tempo change
            decision['syncopation_change'] = 0.3  # More syncopation
            decision['density_change'] = 0.2  # Increase density
            decision['rhythmic_direction'] = 'exploratory'
        
        return decision
    
    def _create_joint_decision(self, 
                             harmonic_decision: Dict[str, Any],
                             rhythmic_decision: Dict[str, Any],
                             input_context: Dict[str, Any],
                             correlation_context: Optional[CrossModalContext]) -> Dict[str, Any]:
        """Create joint decision considering harmonic-rhythmic correlations"""
        
        joint_decision = {
            'harmonic_rhythmic_coupling': 0.0,
            'temporal_alignment': 'aligned',
            'cross_modal_consistency': 0.5,
            'joint_direction': 'stable'
        }
        
        # Calculate harmonic-rhythmic coupling
        harmonic_tension_change = harmonic_decision.get('tension_change', 0.0)
        rhythmic_syncopation_change = rhythmic_decision.get('syncopation_change', 0.0)
        
        # Positive coupling: tension and syncopation move together
        coupling = abs(harmonic_tension_change + rhythmic_syncopation_change) / 2.0
        joint_decision['harmonic_rhythmic_coupling'] = coupling
        
        # Determine temporal alignment
        harmonic_direction = harmonic_decision.get('harmonic_direction', 'stable')
        rhythmic_direction = rhythmic_decision.get('rhythmic_direction', 'stable')
        
        if harmonic_direction == rhythmic_direction:
            joint_decision['temporal_alignment'] = 'aligned'
            joint_decision['cross_modal_consistency'] = 0.8
        elif harmonic_direction == 'contrast' and rhythmic_direction == 'contrast':
            joint_decision['temporal_alignment'] = 'aligned'
            joint_decision['cross_modal_consistency'] = 0.7
        else:
            joint_decision['temporal_alignment'] = 'mixed'
            joint_decision['cross_modal_consistency'] = 0.4
        
        # Determine joint direction
        if harmonic_direction == 'progressive' and rhythmic_direction == 'progressive':
            joint_decision['joint_direction'] = 'progressive'
        elif harmonic_direction == 'contrast' and rhythmic_direction == 'contrast':
            joint_decision['joint_direction'] = 'contrast'
        elif harmonic_direction == 'exploratory' or rhythmic_direction == 'exploratory':
            joint_decision['joint_direction'] = 'exploratory'
        else:
            joint_decision['joint_direction'] = 'stable'
        
        return joint_decision
    
    def _calculate_decision_confidence(self, 
                                     harmonic_decision: Dict[str, Any],
                                     rhythmic_decision: Dict[str, Any],
                                     joint_decision: Dict[str, Any],
                                     input_context: Dict[str, Any]) -> float:
        """Calculate confidence in the unified decision"""
        
        # Base confidence from input stability
        stability = (input_context['harmonic_stability'] + input_context['rhythmic_stability']) / 2
        base_confidence = stability
        
        # Adjust for cross-modal consistency
        consistency = joint_decision.get('cross_modal_consistency', 0.5)
        consistency_factor = consistency
        
        # Adjust for coupling strength
        coupling = joint_decision.get('harmonic_rhythmic_coupling', 0.0)
        coupling_factor = min(coupling * 2, 1.0)  # Scale coupling to 0-1
        
        # Calculate final confidence
        confidence = (base_confidence * 0.4 + consistency_factor * 0.4 + coupling_factor * 0.2)
        
        return max(0.0, min(1.0, confidence))
    
    def _generate_reasoning(self, 
                          harmonic_decision: Dict[str, Any],
                          rhythmic_decision: Dict[str, Any],
                          joint_decision: Dict[str, Any],
                          musical_context: MusicalContext,
                          response_mode: ResponseMode,
                          input_context: Dict[str, Any]) -> str:
        """Generate reasoning for the decision"""
        
        reasoning_parts = []
        
        # Context reasoning
        reasoning_parts.append(f"Musical context: {musical_context.value}")
        reasoning_parts.append(f"Response mode: {response_mode.value}")
        
        # Harmonic reasoning
        harmonic_direction = harmonic_decision.get('harmonic_direction', 'stable')
        tension_change = harmonic_decision.get('tension_change', 0.0)
        if tension_change != 0:
            direction = "increase" if tension_change > 0 else "decrease"
            reasoning_parts.append(f"Harmonic tension {direction}s by {abs(tension_change):.2f}")
        
        # Rhythmic reasoning
        rhythmic_direction = rhythmic_decision.get('rhythmic_direction', 'stable')
        tempo_change = rhythmic_decision.get('tempo_change', 0.0)
        if tempo_change != 0:
            direction = "increase" if tempo_change > 0 else "decrease"
            reasoning_parts.append(f"Tempo {direction}s by {abs(tempo_change):.1f} BPM")
        
        # Joint reasoning
        joint_direction = joint_decision.get('joint_direction', 'stable')
        coupling = joint_decision.get('harmonic_rhythmic_coupling', 0.0)
        if coupling > 0.3:
            reasoning_parts.append(f"Strong harmonic-rhythmic coupling ({coupling:.2f})")
        
        return "; ".join(reasoning_parts)
    
    def _get_contrast_chords(self, current_chord: str) -> List[str]:
        """Get chords that contrast with the current chord"""
        # Simple contrast logic - could be enhanced with music theory
        contrast_map = {
            'C': ['F', 'G'],
            'F': ['C', 'Bb'],
            'G': ['C', 'D'],
            'D': ['G', 'A'],
            'A': ['D', 'E'],
            'E': ['A', 'B'],
            'B': ['E', 'F#'],
            'F#': ['B', 'C#'],
            'C#': ['F#', 'G#'],
            'G#': ['C#', 'D#'],
            'D#': ['G#', 'A#'],
            'A#': ['D#', 'F']
        }
        return contrast_map.get(current_chord, [current_chord])
    
    def _get_leading_chords(self, current_chord: str) -> List[str]:
        """Get chords that lead harmonically"""
        # Simple leading chord logic
        leading_map = {
            'C': ['F', 'Am'],
            'F': ['Bb', 'Dm'],
            'G': ['C', 'Em'],
            'D': ['G', 'Bm'],
            'A': ['D', 'F#m'],
            'E': ['A', 'C#m'],
            'B': ['E', 'G#m'],
            'F#': ['B', 'D#m'],
            'C#': ['F#', 'A#m'],
            'G#': ['C#', 'Fm'],
            'D#': ['G#', 'Cm'],
            'A#': ['D#', 'Gm']
        }
        return leading_map.get(current_chord, [current_chord])
    
    def _get_exploratory_chords(self, current_chord: str) -> List[str]:
        """Get chords for harmonic exploration"""
        # Simple exploratory chord logic
        exploratory_map = {
            'C': ['F#', 'Bb', 'Dm'],
            'F': ['B', 'Eb', 'Gm'],
            'G': ['C#', 'F', 'Am'],
            'D': ['G#', 'Bb', 'Bm'],
            'A': ['D#', 'Eb', 'C#m'],
            'E': ['A#', 'Ab', 'G#m'],
            'B': ['F', 'Gb', 'D#m'],
            'F#': ['C', 'Fb', 'A#m'],
            'C#': ['G', 'Cb', 'Fm'],
            'G#': ['D', 'Gb', 'Cm'],
            'D#': ['A', 'Db', 'Gm'],
            'A#': ['E', 'Ab', 'Dm']
        }
        return exploratory_map.get(current_chord, [current_chord])


def main():
    """Test the unified decision engine"""
    print("🔄 Testing Unified Decision Engine...")
    
    # Create mock input data
    harmonic_input = {
        'primary_chord': 'C',
        'key': 'C',
        'harmonic_tension': 0.6,
        'chord_diversity': 0.4,
        'key_stability': 0.8,
        'chord_change_rate': 0.3,
        'event_count': 5
    }
    
    rhythmic_input = {
        'tempo': 120,
        'syncopation': 0.4,
        'rhythmic_density': 0.6,
        'tempo_stability': 0.7,
        'rhythmic_complexity': 0.5,
        'event_count': 8
    }
    
    # Create decision engine
    engine = UnifiedDecisionEngine()
    
    # Make unified decision
    decision = engine.make_unified_decision(harmonic_input, rhythmic_input)
    
    print(f"✅ Unified Decision Made!")
    print(f"   Context: {decision.context.value}")
    print(f"   Response Mode: {decision.response_mode.value}")
    print(f"   Confidence: {decision.confidence:.3f}")
    print(f"   Reasoning: {decision.reasoning}")
    
    print(f"\n🎵 Harmonic Decision:")
    for key, value in decision.harmonic_decision.items():
        print(f"   {key}: {value}")
    
    print(f"\n🥁 Rhythmic Decision:")
    for key, value in decision.rhythmic_decision.items():
        print(f"   {key}: {value}")
    
    print(f"\n🔗 Joint Decision:")
    for key, value in decision.joint_decision.items():
        print(f"   {key}: {value}")


if __name__ == "__main__":
    main()
