#!/usr/bin/env python3
"""
Musical Decision Explainer
Provides transparency into AI musical decision-making

This module creates human-readable explanations for why specific musical
responses were generated, building trust through visibility into the
decision process.

@author: Jonas Sjøvaag
@date: 2025-10-24
"""

import time
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class DecisionExplanation:
    """Complete explanation of a musical decision"""
    timestamp: float
    mode: str
    voice_type: str
    
    # Input context
    trigger_event: Dict
    recent_context: Dict
    
    # Request parameters used
    request_params: Dict
    
    # Pattern matching
    pattern_match_score: Optional[float]
    pattern_source: Optional[str]
    
    # Generated output
    generated_notes: List[int]
    generated_durations: List[float]
    
    # Reasoning
    reasoning: str
    confidence: float


class MusicalDecisionExplainer:
    """
    Explains musical decisions in human-readable format
    
    Tracks the decision-making process from input to output,
    providing transparency and verification that the system
    is responding to user input.
    """
    
    def __init__(self, enable_console_output: bool = False):
        """
        Initialize decision explainer
        
        Args:
            enable_console_output: If True, print explanations to console
        """
        self.enable_console_output = enable_console_output
        self.explanation_history = []
        self.max_history = 100  # Keep last 100 explanations
    
    def explain_generation(self,
                          mode: str,
                          voice_type: str,
                          trigger_event: Dict,
                          recent_context: Dict,
                          request_params: Dict,
                          generated_notes: List[int],
                          generated_durations: List[float],
                          pattern_match_score: Optional[float] = None,
                          pattern_source: Optional[str] = None,
                          confidence: float = 1.0) -> DecisionExplanation:
        """
        Create detailed explanation of a musical decision
        
        Args:
            mode: Behavior mode (SHADOW, MIRROR, COUPLE, etc.)
            voice_type: "melodic" or "bass"
            trigger_event: Event data that triggered this response
            recent_context: Recent musical context (tokens, consonance, etc.)
            request_params: Request mask parameters used
            generated_notes: MIDI notes generated
            generated_durations: Note durations
            pattern_match_score: Similarity score if pattern was matched
            pattern_source: Source of matched pattern (e.g., "training bar 23")
            confidence: Decision confidence (0.0-1.0)
            
        Returns:
            DecisionExplanation object
        """
        # Create explanation
        explanation = DecisionExplanation(
            timestamp=time.time(),
            mode=mode,
            voice_type=voice_type,
            trigger_event=trigger_event,
            recent_context=recent_context,
            request_params=request_params,
            pattern_match_score=pattern_match_score,
            pattern_source=pattern_source,
            generated_notes=generated_notes,
            generated_durations=generated_durations,
            reasoning=self._build_reasoning(
                mode, voice_type, trigger_event, recent_context,
                request_params, generated_notes, pattern_match_score
            ),
            confidence=confidence
        )
        
        # Store in history
        self.explanation_history.append(explanation)
        if len(self.explanation_history) > self.max_history:
            self.explanation_history.pop(0)
        
        # Print if enabled
        if self.enable_console_output:
            self._print_explanation(explanation)
        
        return explanation
    
    def _build_reasoning(self,
                        mode: str,
                        voice_type: str,
                        trigger_event: Dict,
                        recent_context: Dict,
                        request_params: Dict,
                        generated_notes: List[int],
                        pattern_match_score: Optional[float]) -> str:
        """Build human-readable reasoning text"""
        
        lines = []
        
        # Mode-specific reasoning
        if mode.upper() == 'SHADOW':
            lines.append("Close imitation of your input")
        elif mode.upper() == 'MIRROR':
            lines.append("Complementary variation with phrase awareness")
        elif mode.upper() == 'COUPLE':
            lines.append("Independent dialogue, harmonically aware")
        else:
            lines.append(f"{mode} mode response")
        
        # Input trigger
        trigger_midi = trigger_event.get('midi', 0)
        trigger_note_name = self._midi_to_note_name(trigger_midi)
        lines.append(f"Triggered by your {trigger_note_name} (MIDI {trigger_midi})")
        
        # Context used
        context_parts = []
        if 'gesture_tokens' in recent_context and recent_context['gesture_tokens']:
            tokens = recent_context['gesture_tokens']
            context_parts.append(f"gesture_token={tokens[-1]}")
        if 'avg_consonance' in recent_context:
            cons = recent_context['avg_consonance']
            context_parts.append(f"consonance={cons:.2f}")
        if 'melodic_tendency' in recent_context:
            tend = recent_context['melodic_tendency']
            direction = "ascending" if tend > 0 else "descending" if tend < 0 else "static"
            context_parts.append(f"melody: {direction}")
        
        if context_parts:
            lines.append(f"Context: {', '.join(context_parts)}")
        
        # Request parameters
        if request_params:
            req_desc = self._describe_request(request_params)
            if req_desc:
                lines.append(f"Request: {req_desc}")
        
        # Pattern matching
        if pattern_match_score is not None:
            lines.append(f"Pattern match: {pattern_match_score:.0%} similarity")
        
        # Generated output
        if generated_notes:
            note_names = [self._midi_to_note_name(n) for n in generated_notes[:5]]
            if len(generated_notes) > 5:
                note_names.append("...")
            lines.append(f"Generated: {' → '.join(note_names)}")
            
            # Describe melodic shape
            if len(generated_notes) >= 2:
                shape = self._describe_melodic_shape(generated_notes)
                lines.append(f"Shape: {shape}")
        
        return "; ".join(lines)
    
    def _describe_request(self, request_params: Dict) -> str:
        """Describe request parameters in human terms"""
        parts = []
        
        # Handle multi-parameter requests
        if 'primary' in request_params:
            primary = request_params['primary']
            parts.append(self._describe_single_request(primary, prefix=""))
            
            if 'secondary' in request_params:
                secondary = request_params['secondary']
                parts.append(self._describe_single_request(secondary, prefix="also "))
        else:
            # Single request
            parts.append(self._describe_single_request(request_params, prefix=""))
        
        return ", ".join(parts)
    
    def _describe_single_request(self, request: Dict, prefix: str = "") -> str:
        """Describe a single request parameter"""
        param = request.get('parameter', 'unknown')
        req_type = request.get('type', '==')
        value = request.get('value')
        
        if value is None:
            return f"{prefix}{param}"
        
        # Describe based on type
        if req_type == '==':
            return f"{prefix}match {param}={value}"
        elif req_type == '>':
            return f"{prefix}{param} > {value}"
        elif req_type == '<':
            return f"{prefix}{param} < {value}"
        elif req_type == 'gradient':
            direction = "high" if value > 0 else "low"
            return f"{prefix}favor {direction} {param}"
        else:
            return f"{prefix}{param} ({req_type})"
    
    def _describe_melodic_shape(self, notes: List[int]) -> str:
        """Describe the overall shape of a melodic line"""
        if len(notes) < 2:
            return "single note"
        
        intervals = [notes[i+1] - notes[i] for i in range(len(notes)-1)]
        
        # Count direction changes
        ups = sum(1 for i in intervals if i > 0)
        downs = sum(1 for i in intervals if i < 0)
        
        # Overall range
        note_range = max(notes) - min(notes)
        
        # Describe
        if ups > downs * 2:
            shape = "ascending"
        elif downs > ups * 2:
            shape = "descending"
        elif ups + downs > 0:
            shape = "meandering"
        else:
            shape = "static"
        
        if note_range > 12:
            shape += ", wide range"
        elif note_range < 5:
            shape += ", narrow"
        
        return shape
    
    def _midi_to_note_name(self, midi: int) -> str:
        """Convert MIDI number to note name"""
        if midi <= 0:
            return "silence"
        
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        octave = (midi // 12) - 1
        note = note_names[midi % 12]
        return f"{note}{octave}"
    
    def _print_explanation(self, explanation: DecisionExplanation):
        """Print explanation to console in formatted style"""
        print(f"\n🎭 {explanation.mode.upper()} mode ({explanation.voice_type}):")
        print(f"   {explanation.reasoning}")
        print(f"   Confidence: {explanation.confidence:.0%}")
    
    def get_recent_explanations(self, n: int = 10) -> List[DecisionExplanation]:
        """Get the N most recent explanations"""
        return self.explanation_history[-n:]
    
    def export_to_dict(self, explanation: DecisionExplanation) -> Dict:
        """Export explanation to dictionary for logging"""
        return {
            'timestamp': explanation.timestamp,
            'mode': explanation.mode,
            'voice_type': explanation.voice_type,
            'trigger_midi': explanation.trigger_event.get('midi', 0),
            'request_params': str(explanation.request_params),
            'pattern_match_score': explanation.pattern_match_score,
            'generated_notes': explanation.generated_notes,
            'reasoning': explanation.reasoning,
            'confidence': explanation.confidence
        }


def test_decision_explainer():
    """Test the decision explainer"""
    print("🧪 Testing MusicalDecisionExplainer...")
    
    explainer = MusicalDecisionExplainer(enable_console_output=True)
    
    # Test 1: SHADOW mode
    print("\n--- Test 1: SHADOW Mode ---")
    trigger = {'midi': 65, 'f0': 349.2, 'rms': -20.0}
    context = {
        'gesture_tokens': [142, 156, 142],
        'avg_consonance': 0.73,
        'melodic_tendency': 2.5
    }
    request = {
        'primary': {
            'parameter': 'gesture_token',
            'type': '==',
            'value': 142,
            'weight': 0.95
        }
    }
    
    explainer.explain_generation(
        mode='SHADOW',
        voice_type='melodic',
        trigger_event=trigger,
        recent_context=context,
        request_params=request,
        generated_notes=[65, 67, 69, 70],
        generated_durations=[0.5, 0.5, 0.5, 1.0],
        pattern_match_score=0.87,
        confidence=0.9
    )
    
    # Test 2: COUPLE mode
    print("\n--- Test 2: COUPLE Mode ---")
    trigger2 = {'midi': 72, 'f0': 523.3, 'rms': -18.0}
    context2 = {
        'avg_consonance': 0.45,
        'melodic_tendency': -3.2,
        'rhythmic_tendency': 1.8
    }
    request2 = {
        'primary': {
            'parameter': 'consonance',
            'type': '>',
            'value': 0.7,
            'weight': 0.3
        },
        'secondary': {
            'parameter': 'midi_relative',
            'type': 'gradient',
            'value': 2.0,
            'weight': 0.4
        }
    }
    
    explainer.explain_generation(
        mode='COUPLE',
        voice_type='bass',
        trigger_event=trigger2,
        recent_context=context2,
        request_params=request2,
        generated_notes=[48, 50, 52, 53, 55],
        generated_durations=[1.0, 0.5, 0.5, 0.5, 1.0],
        pattern_match_score=0.34,
        confidence=0.75
    )
    
    print(f"\n✅ Generated {len(explainer.explanation_history)} explanations")
    print(f"   Recent explanations: {len(explainer.get_recent_explanations(5))}")


if __name__ == "__main__":
    test_decision_explainer()

