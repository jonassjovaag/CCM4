#!/usr/bin/env python3
"""
RhythmOracle
Rhythmic pattern memory and retrieval system

This module provides:
- Rhythmic pattern storage and retrieval
- Pattern similarity matching
- Rhythmic context prediction
- Pattern transition learning
"""

import numpy as np
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import time

@dataclass
class RhythmicPattern:
    """Stored rhythmic pattern"""
    pattern_id: str
    tempo: float
    density: float
    syncopation: float
    meter: str
    pattern_type: str
    confidence: float
    context: Dict
    timestamp: float

@dataclass
class PatternTransition:
    """Transition between rhythmic patterns"""
    from_pattern: str
    to_pattern: str
    frequency: int
    context: Dict

class RhythmOracle:
    """
    Rhythmic pattern memory and retrieval system
    
    Similar to AudioOracle but specialized for rhythmic patterns:
    - Stores and retrieves rhythmic patterns
    - Learns pattern transitions
    - Provides rhythmic context prediction
    """
    
    def __init__(self):
        self.rhythmic_patterns = []
        self.pattern_transitions = {}
        self.pattern_frequency = {}
        self.context_history = []
        
        # Pattern matching parameters
        self.tempo_tolerance = 0.1  # 10% tolerance
        self.density_tolerance = 0.2
        self.syncopation_tolerance = 0.3
        
    def add_rhythmic_pattern(self, pattern_data: Dict) -> str:
        """
        Add a rhythmic pattern to memory
        
        Args:
            pattern_data: Dictionary containing pattern information
            
        Returns:
            str: Pattern ID
        """
        pattern_id = f"pattern_{len(self.rhythmic_patterns)}"
        
        pattern = RhythmicPattern(
            pattern_id=pattern_id,
            tempo=pattern_data.get('tempo', 120.0),
            density=pattern_data.get('density', 0.5),
            syncopation=pattern_data.get('syncopation', 0.0),
            meter=pattern_data.get('meter', '4/4'),
            pattern_type=pattern_data.get('pattern_type', 'unknown'),
            confidence=pattern_data.get('confidence', 0.5),
            context=pattern_data.get('context', {}),
            timestamp=time.time()
        )
        
        self.rhythmic_patterns.append(pattern)
        self.pattern_frequency[pattern_id] = 1
        
        # Update transitions if we have previous patterns
        if len(self.rhythmic_patterns) > 1:
            prev_pattern = self.rhythmic_patterns[-2]
            self._update_transition(prev_pattern.pattern_id, pattern_id)
        
        return pattern_id
    
    def find_similar_patterns(self, query_pattern: Dict, threshold: float = 0.7) -> List[Tuple[RhythmicPattern, float]]:
        """
        Find similar rhythmic patterns
        
        Args:
            query_pattern: Pattern to match against
            threshold: Similarity threshold
            
        Returns:
            List of (pattern, similarity) tuples
        """
        similar_patterns = []
        
        for pattern in self.rhythmic_patterns:
            similarity = self._calculate_pattern_similarity(query_pattern, pattern)
            
            if similarity >= threshold:
                similar_patterns.append((pattern, similarity))
        
        # Sort by similarity (highest first)
        similar_patterns.sort(key=lambda x: x[1], reverse=True)
        
        return similar_patterns
    
    def predict_next_pattern(self, current_context: Dict) -> Optional[RhythmicPattern]:
        """
        Predict likely next rhythmic pattern based on context
        
        Args:
            current_context: Current rhythmic context
            
        Returns:
            Predicted pattern or None
        """
        if not self.rhythmic_patterns:
            return None
        
        # Find most similar current pattern
        current_pattern = self._find_most_similar_pattern(current_context)
        
        if current_pattern is None:
            return None
        
        # Look for transitions from current pattern
        transitions = self.pattern_transitions.get(current_pattern.pattern_id, {})
        
        if not transitions:
            return None
        
        # Find most frequent transition
        most_frequent = max(transitions.items(), key=lambda x: x[1]['frequency'])
        next_pattern_id = most_frequent[0]
        
        # Find the pattern
        for pattern in self.rhythmic_patterns:
            if pattern.pattern_id == next_pattern_id:
                return pattern
        
        return None
    
    def get_rhythmic_statistics(self) -> Dict:
        """Get statistics about stored patterns"""
        if not self.rhythmic_patterns:
            return {
                'total_patterns': 0,
                'avg_tempo': 120.0,
                'avg_density': 0.5,
                'avg_syncopation': 0.0,
                'pattern_types': {},
                'total_transitions': 0
            }
        
        tempos = [p.tempo for p in self.rhythmic_patterns]
        densities = [p.density for p in self.rhythmic_patterns]
        syncopations = [p.syncopation for p in self.rhythmic_patterns]
        
        # Count pattern types
        pattern_types = {}
        for pattern in self.rhythmic_patterns:
            pattern_type = pattern.pattern_type
            pattern_types[pattern_type] = pattern_types.get(pattern_type, 0) + 1
        
        # Count total transitions
        total_transitions = sum(
            sum(trans['frequency'] for trans in transitions.values())
            for transitions in self.pattern_transitions.values()
        )
        
        return {
            'total_patterns': len(self.rhythmic_patterns),
            'avg_tempo': np.mean(tempos),
            'avg_density': np.mean(densities),
            'avg_syncopation': np.mean(syncopations),
            'pattern_types': pattern_types,
            'total_transitions': total_transitions
        }
    
    def _calculate_pattern_similarity(self, query: Dict, pattern: RhythmicPattern) -> float:
        """Calculate similarity between query and stored pattern"""
        
        # Tempo similarity - use percentage-based tolerance (allows 50% tempo variation)
        query_tempo = query.get('tempo', 120.0)
        pattern_tempo = pattern.tempo
        tempo_ratio = max(query_tempo, pattern_tempo) / min(query_tempo, pattern_tempo)
        # Map ratio to similarity: 1.0 (same tempo) → 1.0, 1.5 (50% faster) → 0.5, 2.0 (double) → 0.0
        tempo_sim = max(0.0, 1.0 - (tempo_ratio - 1.0))
        
        # Density similarity
        density_sim = 1.0 - abs(query.get('density', 0.5) - pattern.density)
        density_sim = max(0.0, density_sim)
        
        # Syncopation similarity
        syncopation_sim = 1.0 - abs(query.get('syncopation', 0.0) - pattern.syncopation)
        syncopation_sim = max(0.0, syncopation_sim)
        
        # Pattern type similarity
        type_sim = 1.0 if query.get('pattern_type', 'unknown') == pattern.pattern_type else 0.0
        
        # Weighted combination - REDUCED tempo weight, increased density/syncopation
        # This makes it more forgiving of tempo differences while preserving rhythmic feel
        similarity = (
            tempo_sim * 0.2 +       # Reduced from 0.3 - tempo less critical
            density_sim * 0.4 +     # Increased from 0.3 - note density more important
            syncopation_sim * 0.3 + # Increased from 0.2 - rhythmic feel crucial
            type_sim * 0.1          # Reduced from 0.2 - pattern type less strict
        )
        
        return similarity
    
    def _find_most_similar_pattern(self, context: Dict) -> Optional[RhythmicPattern]:
        """Find the most similar pattern to current context"""
        if not self.rhythmic_patterns:
            return None
        
        best_pattern = None
        best_similarity = 0.0
        
        for pattern in self.rhythmic_patterns:
            similarity = self._calculate_pattern_similarity(context, pattern)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_pattern = pattern
        
        return best_pattern if best_similarity > 0.5 else None
    
    def _update_transition(self, from_pattern_id: str, to_pattern_id: str):
        """Update pattern transition statistics"""
        if from_pattern_id not in self.pattern_transitions:
            self.pattern_transitions[from_pattern_id] = {}
        
        if to_pattern_id not in self.pattern_transitions[from_pattern_id]:
            self.pattern_transitions[from_pattern_id][to_pattern_id] = {
                'frequency': 0,
                'context': {}
            }
        
        # Increment frequency
        self.pattern_transitions[from_pattern_id][to_pattern_id]['frequency'] += 1
        
        # Update frequency count
        self.pattern_frequency[to_pattern_id] = self.pattern_frequency.get(to_pattern_id, 0) + 1
    
    def save_patterns(self, filepath: str):
        """Save patterns to JSON file"""
        data = {
            'patterns': [asdict(p) for p in self.rhythmic_patterns],
            'transitions': self.pattern_transitions,
            'frequency': self.pattern_frequency,
            'timestamp': time.time()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_patterns(self, filepath: str):
        """Load patterns from JSON file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Reconstruct patterns
            self.rhythmic_patterns = []
            for pattern_dict in data.get('patterns', []):
                pattern = RhythmicPattern(**pattern_dict)
                self.rhythmic_patterns.append(pattern)
            
            # Restore transitions and frequency
            self.pattern_transitions = data.get('transitions', {})
            self.pattern_frequency = data.get('frequency', {})
            
            print(f"Loaded {len(self.rhythmic_patterns)} rhythmic patterns from {filepath}")
            
        except Exception as e:
            print(f"Error loading patterns: {e}")

def main():
    """Test the RhythmOracle"""
    oracle = RhythmOracle()
    
    print("🥁 Testing RhythmOracle...")
    
    # Add some test patterns
    test_patterns = [
        {
            'tempo': 120.0,
            'density': 0.8,
            'syncopation': 0.2,
            'pattern_type': 'dense',
            'confidence': 0.9
        },
        {
            'tempo': 140.0,
            'density': 0.4,
            'syncopation': 0.6,
            'pattern_type': 'syncopated',
            'confidence': 0.8
        },
        {
            'tempo': 100.0,
            'density': 0.2,
            'syncopation': 0.1,
            'pattern_type': 'sparse',
            'confidence': 0.7
        }
    ]
    
    # Add patterns
    for pattern_data in test_patterns:
        pattern_id = oracle.add_rhythmic_pattern(pattern_data)
        print(f"Added pattern: {pattern_id}")
    
    # Test similarity search
    query = {
        'tempo': 125.0,
        'density': 0.7,
        'syncopation': 0.3,
        'pattern_type': 'dense'
    }
    
    similar = oracle.find_similar_patterns(query, threshold=0.5)
    print(f"\nSimilar patterns to query: {len(similar)}")
    for pattern, similarity in similar:
        print(f"  {pattern.pattern_id}: {similarity:.3f}")
    
    # Test prediction
    current_context = {
        'tempo': 120.0,
        'density': 0.8,
        'syncopation': 0.2
    }
    
    predicted = oracle.predict_next_pattern(current_context)
    if predicted:
        print(f"\nPredicted next pattern: {predicted.pattern_id}")
    else:
        print("\nNo prediction available")
    
    # Show statistics
    stats = oracle.get_rhythmic_statistics()
    print(f"\nStatistics: {stats}")

if __name__ == "__main__":
    main()
