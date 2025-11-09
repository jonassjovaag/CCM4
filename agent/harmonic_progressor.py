"""
Harmonic Progression Intelligence for MusicHal 9000

This module implements learned harmonic progression using transition probabilities
discovered during training. Combines learned patterns with behavioral mode personality.

Architecture:
- Load transition graph from training (JSON file)
- Select next chord based on learned probabilities
- Modulate choices using behavioral modes:
  * SHADOW: Stay with detected chord (high stability)
  * MIRROR: Choose complement/contrast (low probability transitions)
  * COUPLE: Follow learned progressions (weighted random)

Usage:
    progressor = HarmonicProgressor("JSON/model_harmonic_transitions.json")
    next_chord = progressor.choose_next_chord(
        current_chord="Dm",
        behavioral_mode="COUPLE"
    )
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ChordTransition:
    """Represents a learned chord transition"""
    from_chord: str
    to_chord: str
    probability: float
    count: int
    from_chord_occurrences: int


@dataclass
class ChordStatistics:
    """Statistics for a specific chord"""
    name: str
    frequency: int
    avg_duration: float
    std_duration: float
    min_duration: float
    max_duration: float


class HarmonicProgressor:
    """
    Learned harmonic progression engine
    
    Uses transition probabilities discovered during training to make
    intelligent chord progression choices, modulated by behavioral modes.
    """
    
    def __init__(self, transition_graph_file: Optional[str] = None):
        """
        Initialize harmonic progressor
        
        Args:
            transition_graph_file: Path to JSON file with transition graph
                                  If None, progressor is disabled
        """
        self.enabled = False
        self.transitions: Dict[str, ChordTransition] = {}
        self.chord_stats: Dict[str, ChordStatistics] = {}
        self.transition_matrix: Dict[str, List[Tuple[str, float]]] = {}
        
        if transition_graph_file and Path(transition_graph_file).exists():
            self._load_transition_graph(transition_graph_file)
            self.enabled = True
        else:
            if transition_graph_file:
                print(f"⚠️  Harmonic transition file not found: {transition_graph_file}")
            print("   HarmonicProgressor disabled - will use detected chords only")
    
    def _load_transition_graph(self, file_path: str):
        """Load and parse transition graph from JSON"""
        try:
            with open(file_path, 'r') as f:
                graph = json.load(f)
            
            # Parse transitions
            for trans_key, trans_data in graph.get('transitions', {}).items():
                from_chord, to_chord = trans_key.split('->')
                
                self.transitions[trans_key] = ChordTransition(
                    from_chord=from_chord,
                    to_chord=to_chord,
                    probability=trans_data['probability'],
                    count=trans_data['count'],
                    from_chord_occurrences=trans_data['from_chord_occurrences']
                )
            
            # Parse chord statistics
            chord_freqs = graph.get('chord_frequencies', {})
            chord_durations = graph.get('chord_durations', {})
            
            for chord_name, freq in chord_freqs.items():
                duration_data = chord_durations.get(chord_name, {})
                
                self.chord_stats[chord_name] = ChordStatistics(
                    name=chord_name,
                    frequency=freq,
                    avg_duration=duration_data.get('average', 2.0),
                    std_duration=duration_data.get('std', 0.5),
                    min_duration=duration_data.get('min', 0.5),
                    max_duration=duration_data.get('max', 5.0)
                )
            
            # Build transition matrix for fast lookup
            self._build_transition_matrix()
            
            # Print summary
            total_chords = graph.get('total_chords', 0)
            unique_chords = graph.get('unique_chords', 0)
            total_transitions = graph.get('total_transitions', 0)
            
            print(f"✅ Harmonic transition graph loaded:")
            print(f"   File: {file_path}")
            print(f"   Total chord events: {total_chords}")
            print(f"   Unique chords: {unique_chords}")
            print(f"   Total transitions: {total_transitions}")
            print(f"   Loaded {len(self.transitions)} transition patterns")
            
        except Exception as e:
            print(f"❌ Failed to load harmonic transition graph: {e}")
            self.enabled = False
    
    def _build_transition_matrix(self):
        """Build fast lookup matrix for transitions from each chord"""
        for trans_key, transition in self.transitions.items():
            from_chord = transition.from_chord
            
            if from_chord not in self.transition_matrix:
                self.transition_matrix[from_chord] = []
            
            self.transition_matrix[from_chord].append(
                (transition.to_chord, transition.probability)
            )
        
        # Sort by probability (descending) for each chord
        for chord in self.transition_matrix:
            self.transition_matrix[chord].sort(key=lambda x: x[1], reverse=True)
    
    def get_possible_transitions(self, from_chord: str) -> List[Tuple[str, float]]:
        """
        Get all possible transitions from a chord with probabilities
        
        Args:
            from_chord: Starting chord name
            
        Returns:
            List of (to_chord, probability) tuples, sorted by probability
        """
        return self.transition_matrix.get(from_chord, [])
    
    def choose_next_chord(self, 
                         current_chord: str, 
                         behavioral_mode: str,
                         temperature: float = 1.0) -> str:
        """
        Choose next chord based on learned transitions + behavioral mode
        
        Args:
            current_chord: Current detected chord (e.g., "Dm")
            behavioral_mode: "SHADOW", "MIRROR", or "COUPLE"
            temperature: Randomness in selection (0.0 = deterministic, 2.0 = very random)
            
        Returns:
            Next chord name
        """
        if not self.enabled:
            # Progressor disabled - return current chord
            return current_chord
        
        # Get possible transitions
        possible_transitions = self.get_possible_transitions(current_chord)
        
        if not possible_transitions:
            # No learned transitions for this chord
            return current_chord
        
        # Apply behavioral mode logic
        if behavioral_mode == 'SHADOW':
            # SHADOW: High stability, stay with current chord
            # Occasionally move to most common transition
            if np.random.random() < 0.2:  # 20% chance to move
                return possible_transitions[0][0]  # Most probable
            else:
                return current_chord
        
        elif behavioral_mode == 'MIRROR':
            # MIRROR: Contrast/complement, choose unexpected transitions
            # Favor low-probability transitions (contrasting harmonies)
            if len(possible_transitions) > 1:
                # Weight inversely to probability (lower prob = higher weight)
                chords, probs = zip(*possible_transitions)
                inverse_probs = 1.0 - np.array(probs)
                inverse_probs = inverse_probs / np.sum(inverse_probs)  # Normalize
                
                # Apply temperature
                if temperature != 1.0:
                    inverse_probs = np.power(inverse_probs, 1.0 / temperature)
                    inverse_probs = inverse_probs / np.sum(inverse_probs)
                
                return np.random.choice(chords, p=inverse_probs)
            else:
                return possible_transitions[0][0]
        
        elif behavioral_mode == 'COUPLE':
            # COUPLE: Follow learned progressions (weighted random)
            chords, probs = zip(*possible_transitions)
            probs = np.array(probs)
            
            # Apply temperature
            if temperature != 1.0:
                probs = np.power(probs, 1.0 / temperature)
                probs = probs / np.sum(probs)  # Re-normalize
            
            return np.random.choice(chords, p=probs)
        
        else:
            # Unknown mode - use COUPLE as default
            chords, probs = zip(*possible_transitions)
            return np.random.choice(chords, p=np.array(probs) / np.sum(probs))
    
    def get_chord_duration_estimate(self, chord_name: str) -> float:
        """
        Estimate how long to stay on a chord based on learned data
        
        Args:
            chord_name: Chord name
            
        Returns:
            Estimated duration in seconds
        """
        if chord_name in self.chord_stats:
            stats = self.chord_stats[chord_name]
            # Return average duration with some randomness
            return np.random.normal(stats.avg_duration, stats.std_duration / 2)
        else:
            # Default: 2 seconds with variation
            return np.random.uniform(1.5, 3.0)
    
    def get_most_common_chord(self) -> str:
        """Get the most frequently occurring chord from training"""
        if not self.chord_stats:
            return "C"  # Default
        
        return max(self.chord_stats.items(), key=lambda x: x[1].frequency)[0]
    
    def get_statistics_summary(self) -> Dict[str, Any]:
        """Get summary statistics about loaded transitions"""
        if not self.enabled:
            return {'enabled': False}
        
        return {
            'enabled': True,
            'total_chords': len(self.chord_stats),
            'total_transitions': len(self.transitions),
            'most_common_chord': self.get_most_common_chord(),
            'chords_with_transitions': len(self.transition_matrix),
            'avg_transitions_per_chord': len(self.transitions) / max(len(self.transition_matrix), 1)
        }
    
    def explain_choice(self, 
                      current_chord: str, 
                      chosen_chord: str, 
                      behavioral_mode: str) -> str:
        """
        Generate human-readable explanation of chord choice
        
        Args:
            current_chord: Current chord
            chosen_chord: Chosen next chord
            behavioral_mode: Behavioral mode used
            
        Returns:
            Explanation string for logging/transparency
        """
        if not self.enabled:
            return f"Harmonic progressor disabled - using detected chord: {current_chord}"
        
        if current_chord == chosen_chord:
            return f"SHADOW mode: Staying on {current_chord} (stability)"
        
        trans_key = f"{current_chord}->{chosen_chord}"
        if trans_key in self.transitions:
            trans = self.transitions[trans_key]
            return (f"{behavioral_mode} mode: {current_chord}→{chosen_chord} "
                   f"(learned: {trans.probability:.1%}, count: {trans.count})")
        else:
            return (f"{behavioral_mode} mode: {current_chord}→{chosen_chord} "
                   f"(exploring - not in learned transitions)")


# Example usage and testing
if __name__ == "__main__":
    print("🎼 HarmonicProgressor Test\n")
    
    # Try to find a transition graph file
    import glob
    transition_files = glob.glob("JSON/*_harmonic_transitions.json")
    
    if transition_files:
        print(f"Found transition graphs: {len(transition_files)}")
        test_file = transition_files[0]
        print(f"Testing with: {test_file}\n")
        
        # Initialize
        progressor = HarmonicProgressor(test_file)
        
        if progressor.enabled:
            # Print statistics
            stats = progressor.get_statistics_summary()
            print(f"\n📊 Statistics:")
            for key, value in stats.items():
                print(f"   {key}: {value}")
            
            # Test chord choices
            print(f"\n🎵 Testing chord progression choices:")
            test_chord = progressor.get_most_common_chord()
            print(f"   Starting chord: {test_chord}")
            
            for mode in ['SHADOW', 'MIRROR', 'COUPLE']:
                choices = []
                for _ in range(10):
                    next_chord = progressor.choose_next_chord(test_chord, mode)
                    choices.append(next_chord)
                
                from collections import Counter
                distribution = Counter(choices)
                print(f"\n   {mode} mode (10 trials from {test_chord}):")
                for chord, count in distribution.most_common():
                    print(f"      {chord}: {count}/10 times")
                    explanation = progressor.explain_choice(test_chord, chord, mode)
                    print(f"         {explanation}")
    else:
        print("❌ No harmonic transition graphs found in JSON/ directory")
        print("   Run Chandra_trainer.py first to generate transition data")
