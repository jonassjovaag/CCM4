#!/usr/bin/env python3
"""
Test: Does Pulse 4 Template Add Value Over Pulse 2?

This test compares Brandtsegg's original 2-template approach (pulse 2, pulse 3)
with the enhanced 3-template approach (pulse 4, pulse 2, pulse 3) to determine:

1. Does pulse 2 already handle 4/4 time adequately?
2. Does pulse 4 provide different/better detection for 4/4 patterns?
3. Is pulse 4 redundant (just "pulse 2 × 2")?

Test approach:
- Run identical patterns through both implementations
- Compare detected pulse values and confidence scores
- Show detailed scoring breakdown to understand template matching
"""

import numpy as np
from typing import Dict, List, Tuple


class PulseDetectionTester:
    """Test pulse detection with 2-template vs 3-template approaches"""
    
    def __init__(self):
        # Barlow indispensability templates
        self.indis_2 = np.array([1, 0])  # Duple meter
        self.indis_3 = np.array([2, 0, 1])  # Triple meter
        self.indis_4 = np.array([3, 0, 1, 0])  # Quadruple meter
        
        # Normalize
        self.indis_3 = self.indis_3 / np.max(self.indis_3)
        self.indis_4 = self.indis_4 / np.max(self.indis_4)
    
    def analyze_with_2_templates(self, onsets: List[float]) -> Dict:
        """Original Brandtsegg approach: pulse 2 and pulse 3 only"""
        indis_all = [self.indis_3, self.indis_2]  # Original ordering (pulse 3 first)
        indis_scores = np.array([
            [3, 0., 0., 0],  # pulse 3
            [2, 0., 0., 0]   # pulse 2
        ])
        
        result = self._detect_pulse(onsets, indis_all, indis_scores)
        result['templates'] = '2-template (original Brandtsegg)'
        return result
    
    def analyze_with_2_templates_reordered(self, onsets: List[float]) -> Dict:
        """Brandtsegg with fixed ordering: pulse 2 before pulse 3"""
        indis_all = [self.indis_2, self.indis_3]  # Fixed ordering (pulse 2 first)
        indis_scores = np.array([
            [2, 0., 0., 0],  # pulse 2
            [3, 0., 0., 0]   # pulse 3
        ])
        
        result = self._detect_pulse(onsets, indis_all, indis_scores)
        result['templates'] = '2-template (reordered: pulse 2 first)'
        return result
    
    def analyze_with_3_templates(self, onsets: List[float]) -> Dict:
        """Enhanced approach: pulse 4, pulse 2, pulse 3"""
        indis_all = [self.indis_4, self.indis_2, self.indis_3]  # 3 templates
        indis_scores = np.array([
            [4, 0., 0., 0],  # pulse 4
            [2, 0., 0., 0],  # pulse 2
            [3, 0., 0., 0]   # pulse 3
        ])
        
        result = self._detect_pulse(onsets, indis_all, indis_scores)
        result['templates'] = '3-template (with pulse 4)'
        return result
    
    def _detect_pulse(self, onsets: List[float], indis_all: List[np.ndarray], 
                      indis_scores: np.ndarray) -> Dict:
        """Core pulse detection logic (from ratio_analyzer.py)"""
        
        if len(onsets) < 2:
            return {'pulse': 2, 'score': 0.0, 'confidence': 0.0, 'details': 'Too few onsets'}
        
        # Calculate inter-onset intervals
        intervals = np.diff(onsets)
        
        # Create duration pattern (simplified - just using intervals)
        dur_pattern = intervals / np.min(intervals)  # Normalize to smallest interval
        dur_pattern = np.round(dur_pattern).astype(int)
        
        # Check if all durations are equal (uniform pattern)
        all_equal = len(set(dur_pattern)) == 1
        pattern_length = len(dur_pattern)
        
        # Test each indispensability template
        for idx, indis_template in enumerate(indis_all):
            pulse_length = len(indis_template)
            
            # Create trigger sequence (simplified)
            # In real implementation, this comes from duration pattern subdivision
            # For testing, we'll use pattern length modulo pulse length
            matches = pattern_length % pulse_length == 0
            
            if matches:
                # Score based on pattern length divisibility
                score = pattern_length / pulse_length
                indis_scores[idx, 1] = score
                indis_scores[idx, 2] = 1.0 if all_equal else 0.8  # Confidence
        
        # Find best match
        max_score = np.max(indis_scores[:, 1])
        
        if max_score == 0:
            # No matches - apply heuristic for uniform patterns
            if all_equal:
                if pattern_length % 3 == 0:
                    if pattern_length % 12 == 0:
                        pulse = 3
                    elif pattern_length % 4 != 0:
                        pulse = 3
                    else:
                        pulse = 4 if len(indis_all) > 2 else 2  # Use pulse 4 if available
                elif pattern_length % 4 == 0 and pattern_length >= 4:
                    pulse = 4 if len(indis_all) > 2 else 2  # Use pulse 4 if available
                else:
                    pulse = 2
                confidence = 0.5  # Low confidence for heuristic
            else:
                pulse = 2  # Default fallback
                confidence = 0.3
        else:
            # Find template(s) with max score
            tied_indices = [i for i in range(len(indis_scores)) 
                           if indis_scores[i, 1] == max_score]
            
            if len(tied_indices) > 1:
                # Multiple templates tied - use first (preference order)
                best_idx = tied_indices[0]
                confidence = 0.6  # Medium confidence for ties
            else:
                best_idx = tied_indices[0]
                confidence = indis_scores[best_idx, 2]
            
            # Map index to pulse value
            pulse = int(indis_scores[best_idx, 0])
        
        return {
            'pulse': pulse,
            'score': max_score,
            'confidence': confidence,
            'pattern_length': pattern_length,
            'all_equal': all_equal,
            'scoring_details': indis_scores.copy(),
            'details': f"Pattern length {pattern_length}, equal={all_equal}"
        }


def print_comparison(pattern_name: str, onsets: List[float], tester: PulseDetectionTester):
    """Print detailed comparison of all three approaches"""
    
    print(f"\n{'='*80}")
    print(f"Pattern: {pattern_name}")
    print(f"Onsets: {onsets}")
    print(f"Pattern length: {len(onsets)-1} intervals")
    print(f"{'='*80}")
    
    # Test all three approaches
    result_original = tester.analyze_with_2_templates(onsets)
    result_reordered = tester.analyze_with_2_templates_reordered(onsets)
    result_3template = tester.analyze_with_3_templates(onsets)
    
    results = [result_original, result_reordered, result_3template]
    
    for result in results:
        print(f"\n{result['templates']}:")
        print(f"  Detected pulse: {result['pulse']}")
        print(f"  Score: {result['score']:.3f}")
        print(f"  Confidence: {result['confidence']:.3f}")
        print(f"  Details: {result['details']}")
        
        # Show scoring breakdown
        print(f"  Scoring breakdown:")
        scores = result['scoring_details']
        for i in range(len(scores)):
            pulse_num = int(scores[i, 0])
            score = scores[i, 1]
            conf = scores[i, 2]
            print(f"    Pulse {pulse_num}: score={score:.3f}, conf={conf:.3f}")
    
    # Highlight differences
    pulses = [r['pulse'] for r in results]
    if len(set(pulses)) > 1:
        print(f"\n⚠️  DIFFERENT RESULTS!")
        print(f"   Original (pulse 3 first): pulse {result_original['pulse']}")
        print(f"   Reordered (pulse 2 first): pulse {result_reordered['pulse']}")
        print(f"   3-template (with pulse 4): pulse {result_3template['pulse']}")
    else:
        print(f"\n✅ All approaches agree: pulse {pulses[0]}")


def main():
    """Run comprehensive pulse detection tests"""
    
    print("=" * 80)
    print("PULSE DETECTION COMPARISON TEST")
    print("Does pulse 4 template add value over pulse 2?")
    print("=" * 80)
    
    tester = PulseDetectionTester()
    
    # Test patterns
    test_cases = {
        "4/4 straight 8-beat (120 BPM)": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
        
        "2/4 simple (4 beats)": [0.0, 0.5, 1.0, 1.5],
        
        "Triplet pattern (9 beats)": [0.0, 0.333, 0.667, 1.0, 1.333, 1.667, 2.0, 2.333, 2.667],
        
        "16th notes in 4/4 (16 beats)": [i * 0.25 for i in range(16)],
        
        "4/4 two bars (16 beats)": [i * 0.5 for i in range(16)],
        
        "12 equal beats (3×4 or 4×3?)": [i * 0.5 for i in range(12)],
        
        "6 equal beats (2×3 or 3×2?)": [i * 0.5 for i in range(6)],
        
        "Itzama-like: 8 equal beats": [i * 0.5 for i in range(8)],
    }
    
    for pattern_name, onsets in test_cases.items():
        print_comparison(pattern_name, onsets, tester)
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY & CONCLUSIONS")
    print("=" * 80)
    print("""
Key Questions to Answer:

1. Does pulse 2 already handle 4/4 patterns adequately?
   → Look for 4/4 and 16th-note patterns above
   
2. Does pulse 4 provide different detection than pulse 2?
   → Compare "2-template (reordered)" vs "3-template" results
   
3. Is the main problem just the ordering (pulse 3 before pulse 2)?
   → Compare "Original" vs "Reordered" results
   
4. Should we keep pulse 4, or is it redundant?
   → Based on whether pulse 4 results differ meaningfully from pulse 2

Musical Context:
- Itzama.wav is "very straight, very 8-beat" (electronic beat music)
- Should detect pulse 2 or pulse 4 (duple/quadruple), NOT pulse 3 (triple)
- If pulse 2 handles this correctly, pulse 4 may be unnecessary complexity
- If pulse 4 provides better hierarchical understanding, keep it
""")


if __name__ == "__main__":
    main()
