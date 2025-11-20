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
    """Stored rhythmic pattern (tempo-independent)"""
    pattern_id: str
    duration_pattern: List[int]  # Tempo-free rational durations (e.g., [2,1,1,2])
    density: float  # Events per beat (tempo-independent)
    syncopation: float  # Syncopation score (tempo-independent)
    pulse: int  # Pulse subdivision (2, 3, 4)
    complexity: float  # Barlow indigestability
    meter: str  # Time signature
    pattern_type: str  # Pattern classification
    confidence: float  # Analysis confidence
    context: Dict  # Additional metadata
    timestamp: float  # When pattern was created
    
    def to_absolute_timing(self, tempo: float, start_time: float = 0.0) -> List[float]:
        """
        Convert duration pattern to absolute onset times at given tempo
        
        Args:
            tempo: Target tempo in BPM
            start_time: Starting timestamp
            
        Returns:
            List of absolute onset times
        """
        beat_duration = 60.0 / tempo
        onsets = [start_time]
        
        for duration in self.duration_pattern:
            next_onset = onsets[-1] + (beat_duration * duration)
            onsets.append(next_onset)
        
        # Return all but first (which is start_time)
        return onsets[1:]

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
            pattern_data: Dictionary containing pattern information:
                - duration_pattern: List[int] - Tempo-free rational durations
                - density: float - Events per beat
                - syncopation: float - Syncopation score
                - pulse: int - Pulse subdivision (2, 3, 4)
                - complexity: float - Barlow indigestability
                - meter: str (optional) - Time signature
                - pattern_type: str (optional) - Pattern classification
                - confidence: float (optional) - Analysis confidence
                - context: Dict (optional) - Additional metadata
            
        Returns:
            str: Pattern ID
        """
        pattern_id = f"pattern_{len(self.rhythmic_patterns)}"
        
        pattern = RhythmicPattern(
            pattern_id=pattern_id,
            duration_pattern=pattern_data.get('duration_pattern', [2, 2, 2, 2]),
            density=pattern_data.get('density', 0.5),
            syncopation=pattern_data.get('syncopation', 0.0),
            pulse=pattern_data.get('pulse', 4),
            complexity=pattern_data.get('complexity', 0.0),
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
    
    def find_similar_patterns(self, query_pattern: Dict, threshold: float = 0.5) -> List[Tuple[RhythmicPattern, float]]:
        """
        Find similar rhythmic patterns
        
        Args:
            query_pattern: Pattern to match against
            threshold: Similarity threshold (default 0.5 for more lenient matching)
            
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
        """Get statistics about stored patterns (tempo-independent)"""
        if not self.rhythmic_patterns:
            return {
                'total_patterns': 0,
                'avg_tempo': 120.0,  # Default tempo for backward compatibility
                'avg_density': 0.5,
                'avg_syncopation': 0.0,
                'avg_complexity': 0.0,
                'pattern_types': {},
                'total_transitions': 0
            }
        
        densities = [p.density for p in self.rhythmic_patterns]
        syncopations = [p.syncopation for p in self.rhythmic_patterns]
        complexities = [p.complexity for p in self.rhythmic_patterns]
        # Note: tempo is now tempo-independent, but we include avg for backward compatibility
        tempos = [p.tempo for p in self.rhythmic_patterns if hasattr(p, 'tempo') and p.tempo > 0]
        avg_tempo = np.mean(tempos) if tempos else 120.0
        
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
            'avg_tempo': avg_tempo,  # Include for backward compatibility
            'avg_density': np.mean(densities),
            'avg_syncopation': np.mean(syncopations),
            'avg_complexity': np.mean(complexities),
            'pattern_types': pattern_types,
            'total_transitions': total_transitions
        }
    
    def _calculate_pattern_similarity(self, query: Dict, pattern: RhythmicPattern) -> float:
        """
        Calculate similarity between query and stored pattern
        
        Now tempo-independent: matches on duration pattern structure, density,
        syncopation, and pulse instead of absolute tempo.
        
        Args:
            query: Query dict with duration_pattern, density, syncopation, pulse
            pattern: Stored RhythmicPattern
            
        Returns:
            Similarity score (0.0-1.0)
        """
        
        # Duration pattern similarity - compare ratio sequences
        query_pattern = query.get('duration_pattern', [2, 2, 2, 2])
        pattern_pattern = pattern.duration_pattern
        
        # Simple correlation-based similarity for duration patterns
        # (Could use DTW for variable-length patterns, but start simple)
        if len(query_pattern) == len(pattern_pattern):
            # Same length - use normalized correlation
            query_arr = np.array(query_pattern, dtype=float)
            pattern_arr = np.array(pattern_pattern, dtype=float)
            
            # Normalize both to unit vectors
            query_norm = query_arr / np.linalg.norm(query_arr) if np.linalg.norm(query_arr) > 0 else query_arr
            pattern_norm = pattern_arr / np.linalg.norm(pattern_arr) if np.linalg.norm(pattern_arr) > 0 else pattern_arr
            
            # Cosine similarity
            duration_sim = float(np.dot(query_norm, pattern_norm))
            duration_sim = max(0.0, duration_sim)  # Clamp to [0, 1]
        else:
            # Different length - use simpler metric (could improve with DTW)
            # For now: penalize length difference, compare normalized patterns
            len_ratio = min(len(query_pattern), len(pattern_pattern)) / max(len(query_pattern), len(pattern_pattern))
            
            # Truncate to shortest length for comparison
            min_len = min(len(query_pattern), len(pattern_pattern))
            query_arr = np.array(query_pattern[:min_len], dtype=float)
            pattern_arr = np.array(pattern_pattern[:min_len], dtype=float)
            
            query_norm = query_arr / np.linalg.norm(query_arr) if np.linalg.norm(query_arr) > 0 else query_arr
            pattern_norm = pattern_arr / np.linalg.norm(pattern_arr) if np.linalg.norm(pattern_arr) > 0 else pattern_arr
            
            partial_sim = float(np.dot(query_norm, pattern_norm))
            duration_sim = max(0.0, partial_sim * len_ratio)
        
        # Density similarity - FIXED to normalize both to events-per-beat scale
        # Problem: Query often has normalized 0-1 values, patterns have absolute events/second
        # Solution: Normalize both to events-per-beat using tempo context
        query_density = query.get('density', 0.5)
        pattern_density = pattern.density

        # Normalize to events-per-beat scale (0.25 to 4.0 typical range)
        # If query density < 1.5, assume it's normalized 0-1 and scale up
        # If pattern density > 2.0, assume it's events/second and scale down
        normalized_query = query_density
        normalized_pattern = pattern_density

        # Detect and normalize query (0-1 scale -> events-per-beat)
        if query_density <= 1.0:
            # Likely normalized 0-1, scale to 0.25-4.0 events-per-beat
            normalized_query = 0.25 + query_density * 3.75

        # Detect and normalize pattern (events/second -> events-per-beat at ~120 BPM)
        if pattern_density > 4.0:
            # Likely events/second, convert to events-per-beat (assume 120 BPM = 2 beats/sec)
            normalized_pattern = pattern_density / 2.0

        # Use ratio-based similarity on normalized values
        if normalized_pattern > 0 and normalized_query > 0:
            density_ratio = min(normalized_query, normalized_pattern) / max(normalized_query, normalized_pattern)
            density_sim = density_ratio  # 1.0 = same density, 0.0 = very different
        else:
            density_sim = 0.0
        
        # Syncopation similarity - tempo-independent rhythmic character
        syncopation_sim = 1.0 - abs(query.get('syncopation', 0.0) - pattern.syncopation)
        syncopation_sim = max(0.0, syncopation_sim)
        
        # Pulse similarity - matches subdivision feel (2, 3, 4)
        pulse_sim = 1.0 if query.get('pulse', 4) == pattern.pulse else 0.5
        
        # Pattern type similarity (optional)
        type_sim = 1.0 if query.get('pattern_type', 'unknown') == pattern.pattern_type else 0.0
        
        # Weighted combination - TEMPO-FREE
        # Emphasize duration pattern structure and rhythmic feel
        similarity = (
            duration_sim * 0.4 +     # Duration pattern structure (highest weight)
            syncopation_sim * 0.25 + # Off-beat character crucial
            density_sim * 0.2 +      # Note density important
            pulse_sim * 0.1 +        # Pulse subdivision feel
            type_sim * 0.05          # Pattern type least important
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
        
        return best_pattern if best_similarity > 0.3 else None  # Lowered from 0.5 to match find_similar_patterns threshold
    
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
    
    def to_dict(self) -> Dict:
        """
        Serialize RhythmOracle to dictionary for JSON serialization.
        
        Returns:
            Dict with patterns, transitions, frequency
        """
        return {
            'patterns': [asdict(p) for p in self.rhythmic_patterns],
            'transitions': self.pattern_transitions,
            'frequency': self.pattern_frequency,
            'timestamp': time.time()
        }
    
    def save_patterns(self, filepath: str):
        """Save patterns to JSON file"""
        data = self.to_dict()
        
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
    
    def load_from_dict(self, data: Dict):
        """
        Load state from dictionary (e.g. from embedded model data).
        
        Args:
            data: Dictionary containing patterns, transitions, frequency
        """
        try:
            # Reconstruct patterns
            self.rhythmic_patterns = []
            for pattern_dict in data.get('patterns', []):
                # Handle potential key mismatches or missing fields if schema changed
                try:
                    pattern = RhythmicPattern(**pattern_dict)
                    self.rhythmic_patterns.append(pattern)
                except TypeError as e:
                    print(f"⚠️ Skipping invalid pattern dict: {e}")
            
            # Restore transitions and frequency
            # Convert string keys back to tuples for transitions if needed
            # JSON keys are always strings, but transitions dict uses tuple keys in memory?
            # Let's check how it's saved. to_dict saves self.pattern_transitions directly.
            # If keys are tuples, json.dump will fail or convert them to strings.
            # Python's json.dump converts tuple keys to string representation "('a', 'b')" if skipkeys=False? 
            # No, it raises TypeError unless keys are strings, numbers, etc.
            # So to_dict probably needs to handle tuple keys if they exist.
            
            self.pattern_transitions = data.get('transitions', {})
            self.pattern_frequency = data.get('frequency', {})
            
            print(f"Loaded {len(self.rhythmic_patterns)} rhythmic patterns from dictionary")
            
        except Exception as e:
            print(f"Error loading from dict: {e}")

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
