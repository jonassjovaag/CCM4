#!/usr/bin/env python3
"""
Temporal Reconciliation Engine
Ported from Brandtsegg/Formo rhythm_ratio_analyzer

Maintains rhythmic coherence across phrases by reconciling tempo relationships.
Handles cases where the human plays at different metric levels (e.g., quarter notes vs eighth notes).

@author: Øyvind Brandtsegg, Daniel Formo (original)
@adapted_by: Jonas Sjøvaag (CCM3 integration)
@license: GPL
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque


class ReconciliationEngine:
    """
    Engine for reconciling tempo across consecutive phrase analyses
    
    When analyzing consecutive musical phrases, the perceived tempo might differ
    (e.g., human switches from quarter notes to eighth notes). This engine detects
    such relationships and maintains coherent metric interpretation.
    """
    
    def __init__(self, tolerance: float = 0.15, max_history: int = 3):
        """
        Initialize reconciliation engine
        
        Args:
            tolerance: Tolerance for tempo matching (0.15 = 15%)
            max_history: Maximum number of phrases to keep in history
        """
        self.tolerance = tolerance
        self.max_history = max_history
        
        # Phrase history: stores recent phrase analyses
        self.phrase_history = deque(maxlen=max_history)
        
        # Tempo factors: valid integer relationships between tempi
        # Format: [factor_for_prev, factor_for_new]
        self.tempo_factors = [
            [1, 1],   # Same tempo
            [1, 2], [1, 4], [1, 8],  # New tempo is faster
            [1, 3], [1, 6], [1, 12],  # Triplet relationships
            [2, 1], [4, 1], [8, 1],  # New tempo is slower
            [2, 3], [2, 6], [2, 12],  # Complex relationships
            [3, 1], [6, 1], [12, 1],
            [3, 2], [3, 4], [3, 8],
            [4, 3], [5, 4], [4, 5]  # Include 5:4 for expressive timing
        ]
    
    def reconcile_tempi_singles(self, 
                                prev_tempo: float, 
                                new_tempo: float) -> Tuple[float, float]:
        """
        Reconcile two tempi by finding integer factor relationship
        
        Args:
            prev_tempo: Previous phrase tempo (BPM)
            new_tempo: New phrase tempo (BPM)
            
        Returns:
            (tempo_factor, deviation) tuple where:
                - tempo_factor: Integer multiplier to reconcile tempi
                - deviation: Normalized deviation from perfect match
        """
        reconcile_factors = []
        
        for tf in self.tempo_factors:
            # Check if tempi match within tolerance
            near_match = np.isclose(
                prev_tempo * tf[0], 
                new_tempo * tf[1], 
                rtol=self.tolerance
            )
            
            if near_match:
                reconcile_factor = tf[0] / tf[1]
                unquantized = new_tempo / prev_tempo
                deviation = abs(1 - reconcile_factor / unquantized)
                reconcile_factors.append([reconcile_factor, deviation])
        
        if len(reconcile_factors) == 0:
            # Cannot be reconciled - return identity
            print(f"⚠️  Cannot reconcile tempi: {prev_tempo:.1f} BPM → {new_tempo:.1f} BPM")
            return 1.0, 1.0
        
        # Sort by deviation (lowest first)
        reconcile_factors = np.array(reconcile_factors)
        reconcile_factors = reconcile_factors[reconcile_factors[:, 1].argsort()]
        
        # Return best match
        best_factor = reconcile_factors[0][0]
        best_deviation = reconcile_factors[0][1]
        
        return float(best_factor), float(best_deviation)
    
    def reconcile_new_phrase(self, new_analysis: Dict) -> Dict:
        """
        Reconcile new phrase analysis with phrase history
        
        Args:
            new_analysis: New phrase analysis dict with:
                - duration_pattern: Integer duration pattern
                - tempo: Subdiv tempo in BPM
                - pulse: Pulse subdivision
                - complexity: Complexity score
                - deviations: Deviation array
                - confidence: Analysis confidence
                
        Returns:
            Reconciled analysis dict with added fields:
                - tempo_factor: Multiplier applied to duration pattern
                - reconciled: Whether reconciliation occurred
                - prev_tempo: Previous phrase tempo (if reconciled)
        """
        reconciled_analysis = new_analysis.copy()
        reconciled_analysis['tempo_factor'] = 1.0
        reconciled_analysis['reconciled'] = False
        
        # If no history, just store and return
        if len(self.phrase_history) == 0:
            self.phrase_history.append(new_analysis)
            return reconciled_analysis
        
        # Get previous phrase tempo
        prev_analysis = self.phrase_history[-1]
        prev_tempo = prev_analysis['tempo']
        new_tempo = new_analysis['tempo']
        
        # Attempt reconciliation
        tempo_factor, deviation = self.reconcile_tempi_singles(prev_tempo, new_tempo)
        
        # Check if reconciliation is meaningful (not identity)
        if abs(tempo_factor - 1.0) > 0.01:  # More than 1% difference
            print(f"🔄 Tempo reconciliation:")
            print(f"   Previous tempo: {prev_tempo:.1f} BPM")
            print(f"   New tempo: {new_tempo:.1f} BPM")
            print(f"   Tempo factor: {tempo_factor:.2f}")
            print(f"   Deviation: {deviation:.3f}")
            
            # Apply reconciliation
            reconciled_analysis['tempo_factor'] = tempo_factor
            reconciled_analysis['reconciled'] = True
            reconciled_analysis['prev_tempo'] = prev_tempo
            
            # Adjust duration pattern if needed
            if tempo_factor != 1.0:
                adjusted_pattern = [
                    int(d * tempo_factor) 
                    for d in new_analysis['duration_pattern']
                ]
                reconciled_analysis['duration_pattern'] = adjusted_pattern
                print(f"   Adjusted pattern: {new_analysis['duration_pattern']} → {adjusted_pattern}")
        
        # Add to history
        self.phrase_history.append(reconciled_analysis)
        
        return reconciled_analysis
    
    def get_current_tempo(self) -> Optional[float]:
        """
        Get current reconciled tempo
        
        Returns:
            Current tempo in BPM, or None if no history
        """
        if len(self.phrase_history) == 0:
            return None
        return self.phrase_history[-1]['tempo']
    
    def get_average_tempo(self) -> Optional[float]:
        """
        Get average tempo across phrase history
        
        Returns:
            Average tempo in BPM, or None if no history
        """
        if len(self.phrase_history) == 0:
            return None
        
        tempos = [phrase['tempo'] for phrase in self.phrase_history]
        return float(np.mean(tempos))
    
    def clear_history(self):
        """Clear phrase history"""
        self.phrase_history.clear()
    
    def get_statistics(self) -> Dict:
        """
        Get reconciliation statistics
        
        Returns:
            Dictionary with statistics about reconciliation history
        """
        if len(self.phrase_history) == 0:
            return {
                'num_phrases': 0,
                'reconciliations': 0,
                'avg_tempo': None,
                'tempo_range': None,
                'avg_tempo_factor': None
            }
        
        num_reconciled = sum(
            1 for phrase in self.phrase_history 
            if phrase.get('reconciled', False)
        )
        
        tempos = [phrase['tempo'] for phrase in self.phrase_history]
        tempo_factors = [
            phrase.get('tempo_factor', 1.0) 
            for phrase in self.phrase_history
        ]
        
        return {
            'num_phrases': len(self.phrase_history),
            'reconciliations': num_reconciled,
            'avg_tempo': float(np.mean(tempos)),
            'tempo_range': (float(np.min(tempos)), float(np.max(tempos))),
            'avg_tempo_factor': float(np.mean(tempo_factors))
        }


def test_reconciliation_engine():
    """Test the reconciliation engine"""
    print("🔄 Testing ReconciliationEngine...")
    
    engine = ReconciliationEngine(tolerance=0.15, max_history=3)
    
    # Test case 1: Same tempo
    print("\n📍 Test 1: Same tempo (120 → 120 BPM)")
    analysis1 = {
        'duration_pattern': [2, 1, 1, 2],
        'tempo': 120.0,
        'pulse': 4,
        'complexity': 5.0,
        'deviations': [0.01, -0.02, 0.0, 0.01],
        'confidence': 0.9
    }
    result1 = engine.reconcile_new_phrase(analysis1)
    print(f"   Reconciled: {result1['reconciled']}")
    print(f"   Tempo factor: {result1['tempo_factor']}")
    
    # Test case 2: Double tempo (120 → 240 BPM)
    print("\n📍 Test 2: Double tempo (120 → 240 BPM)")
    analysis2 = {
        'duration_pattern': [1, 1, 1, 1],
        'tempo': 240.0,
        'pulse': 4,
        'complexity': 3.0,
        'deviations': [0.0, 0.0, 0.01, -0.01],
        'confidence': 0.85
    }
    result2 = engine.reconcile_new_phrase(analysis2)
    print(f"   Reconciled: {result2['reconciled']}")
    print(f"   Tempo factor: {result2['tempo_factor']}")
    print(f"   Adjusted pattern: {result2['duration_pattern']}")
    
    # Test case 3: Triplet relationship (120 → 180 BPM)
    print("\n📍 Test 3: Triplet relationship (120 → 180 BPM)")
    analysis3 = {
        'duration_pattern': [3, 3, 3],
        'tempo': 180.0,
        'pulse': 3,
        'complexity': 4.0,
        'deviations': [0.02, -0.01, 0.0],
        'confidence': 0.8
    }
    result3 = engine.reconcile_new_phrase(analysis3)
    print(f"   Reconciled: {result3['reconciled']}")
    print(f"   Tempo factor: {result3['tempo_factor']}")
    
    # Test case 4: Cannot reconcile (120 → 155 BPM)
    print("\n📍 Test 4: Cannot reconcile (120 → 155 BPM)")
    analysis4 = {
        'duration_pattern': [2, 2, 1],
        'tempo': 155.0,
        'pulse': 4,
        'complexity': 4.5,
        'deviations': [0.0, 0.01, -0.02],
        'confidence': 0.75
    }
    result4 = engine.reconcile_new_phrase(analysis4)
    print(f"   Reconciled: {result4['reconciled']}")
    print(f"   Tempo factor: {result4['tempo_factor']}")
    
    # Show statistics
    print("\n📊 Reconciliation Statistics:")
    stats = engine.get_statistics()
    print(f"   Phrases analyzed: {stats['num_phrases']}")
    print(f"   Reconciliations: {stats['reconciliations']}")
    print(f"   Average tempo: {stats['avg_tempo']:.1f} BPM")
    print(f"   Tempo range: {stats['tempo_range'][0]:.1f}-{stats['tempo_range'][1]:.1f} BPM")
    print(f"   Average tempo factor: {stats['avg_tempo_factor']:.2f}")
    
    print("\n✅ ReconciliationEngine tests complete!")


if __name__ == "__main__":
    test_reconciliation_engine()

