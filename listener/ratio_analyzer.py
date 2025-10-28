#!/usr/bin/env python3
"""
Frequency Ratio Analyzer - Mathematical Approach to Chord Analysis
===================================================================

Instead of relying on note name matching, this module analyzes chords
based on the mathematical ratios between detected frequencies.

This approach is grounded in psychoacoustic research:
- Simple integer ratios (2:3, 4:5, etc.) are perceived as consonant
- Complex ratios are perceived as dissonant
- Based on Helmholtz (1877), Shapira Lots & Stone (2008), and others

Example:
    >>> analyzer = FrequencyRatioAnalyzer()
    >>> freqs = [261.63, 329.63, 392.00]  # C, E, G
    >>> result = analyzer.analyze_frequencies(freqs)
    >>> print(result['chord_match']['type'])  # 'major'
    >>> print(result['consonance_score'])     # ~0.90
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from fractions import Fraction


@dataclass
class IntervalAnalysis:
    """Analysis of a single interval between two frequencies"""
    freq1: float
    freq2: float
    ratio: float
    simplified: Tuple[int, int]  # Simplified integer ratio (num, den)
    interval_name: str
    consonance: float  # 0-1, higher = more consonant
    cents: float  # Interval in cents


@dataclass
class ChordAnalysis:
    """Complete analysis of a chord based on frequency ratios"""
    frequencies: List[float]
    ratios: List[float]  # Ratios from fundamental
    simplified_ratios: List[Tuple[int, int]]
    chord_match: Dict  # Best matching chord type
    consonance_score: float  # Overall consonance (0-1)
    intervals: List[IntervalAnalysis]
    fundamental: float
    root_position_equivalent: Optional[str]  # e.g., "C major" if detectable


class FrequencyRatioAnalyzer:
    """
    Analyzes detected frequencies and converts them to 
    mathematical ratios for chord identification
    """
    
    # Ideal chord ratios (in relation to fundamental)
    # Format: chord_name -> list of ratios from fundamental
    CHORD_RATIOS = {
        # Triads
        'major': [1.0, 1.25, 1.5],           # 4:5:6 (Major triad)
        'minor': [1.0, 1.2, 1.5],            # 5:6:7.5 or 10:12:15
        'diminished': [1.0, 1.189, 1.414],   # Approx (tritone is irrational in 12-TET)
        'augmented': [1.0, 1.25, 1.5625],    # 16:20:25
        'sus4': [1.0, 1.333, 1.5],           # 3:4:4.5 or 6:8:9
        'sus2': [1.0, 1.125, 1.5],           # 8:9:12
        
        # Seventh chords (4 notes)
        'maj7': [1.0, 1.25, 1.5, 1.875],     # 8:10:12:15
        'min7': [1.0, 1.2, 1.5, 1.8],        # 10:12:15:18
        'dom7': [1.0, 1.25, 1.5, 1.778],     # 9:11.25:13.5:16 (approx)
        'm7b5': [1.0, 1.189, 1.414, 1.778],  # Half-diminished
        'dim7': [1.0, 1.189, 1.414, 1.682],  # Fully diminished
        
        # Extended chords (5 notes)
        'maj9': [1.0, 1.25, 1.5, 1.875, 2.25],   # With 9th
        'min9': [1.0, 1.2, 1.5, 1.8, 2.25],
        'dom9': [1.0, 1.25, 1.5, 1.778, 2.25],
        'dim7': [1.0, 1.189, 1.414, 1.682],  # Fully diminished
    }
    
    # Interval ratios with consonance scores
    # Based on Helmholtz (1877) ordering and neural synchronization theory
    # Format: (numerator, denominator) -> (interval_name, consonance_score)
    INTERVAL_RATIOS = {
        # Absolute consonances
        (1, 1): ('unison', 1.00),
        (2, 1): ('octave', 0.98),
        (1, 2): ('octave_down', 0.98),
        
        # Perfect consonances
        (3, 2): ('fifth', 0.95),
        (2, 3): ('fourth', 0.90),
        (4, 3): ('fourth', 0.90),
        (3, 4): ('fifth_down', 0.95),
        
        # Medial consonances
        (5, 3): ('major_sixth', 0.85),
        (3, 5): ('major_sixth_down', 0.85),
        (5, 4): ('major_third', 0.85),
        (4, 5): ('major_third_down', 0.85),
        
        # Imperfect consonances
        (6, 5): ('minor_third', 0.82),
        (5, 6): ('minor_third_down', 0.82),
        (8, 5): ('minor_sixth', 0.80),
        (5, 8): ('minor_sixth_down', 0.80),
        
        # Dissonances
        (9, 8): ('major_second', 0.60),
        (8, 9): ('major_second_down', 0.60),
        (15, 8): ('major_seventh', 0.55),
        (8, 15): ('major_seventh_down', 0.55),
        (16, 9): ('minor_seventh', 0.50),
        (9, 16): ('minor_seventh_down', 0.50),
        (16, 15): ('minor_second', 0.30),
        (15, 16): ('minor_second_down', 0.30),
        (45, 32): ('tritone', 0.25),
        (32, 45): ('tritone_inv', 0.25),
    }
    
    def __init__(self, tolerance: float = 0.06, max_denominator: int = 50):
        """
        Initialize ratio analyzer
        
        Args:
            tolerance: Tolerance for ratio matching (default 3%)
            max_denominator: Maximum denominator for ratio simplification
        """
        self.tolerance = tolerance
        self.max_denominator = max_denominator
    
    def analyze_frequencies(self, frequencies: List[float], 
                           octave_invariant: bool = True) -> Optional[ChordAnalysis]:
        """
        Main analysis function
        
        Args:
            frequencies: List of detected fundamental frequencies (Hz)
            octave_invariant: If True, normalize all frequencies to same octave for analysis
            
        Returns:
            ChordAnalysis object with complete analysis, or None if < 2 frequencies
        """
        if len(frequencies) < 2:
            return None
        
        # Sort frequencies (lowest to highest)
        freqs = sorted([f for f in frequencies if f > 0])
        
        if len(freqs) < 2:
            return None
        
        # Optionally normalize to same octave (fold octaves)
        # This makes chord analysis invariant to octave (C-E-G is same as C-E-G in any octave)
        if octave_invariant:
            freqs_normalized = self._normalize_to_octave(freqs)
        else:
            freqs_normalized = freqs
        
        # Calculate ratios from fundamental (lowest frequency)
        fundamental = freqs_normalized[0]
        ratios = [f / fundamental for f in freqs_normalized]
        
        # Simplify to small integer ratios
        simplified = [self._simplify_ratio(r) for r in ratios]
        
        # Find best chord match
        chord_match = self._match_chord(ratios, len(freqs_normalized))
        
        # Calculate overall consonance score (use original freqs for this)
        consonance = self._calculate_consonance(freqs)
        
        # Analyze all pairwise intervals (use normalized for chord structure)
        intervals = self._analyze_intervals(freqs_normalized)
        
        # Try to identify root position equivalent (if possible)
        root_equiv = self._identify_root_position(freqs, chord_match)
        
        return ChordAnalysis(
            frequencies=freqs,  # Store original frequencies
            ratios=ratios,
            simplified_ratios=simplified,
            chord_match=chord_match,
            consonance_score=consonance,
            intervals=intervals,
            fundamental=fundamental,
            root_position_equivalent=root_equiv
        )
    
    def _normalize_to_octave(self, frequencies: List[float]) -> List[float]:
        """
        Normalize all frequencies to same octave range
        This makes chord analysis octave-invariant
        
        Args:
            frequencies: List of frequencies (sorted)
            
        Returns:
            Frequencies normalized to single octave starting from lowest
        """
        if not frequencies:
            return []
        
        base = frequencies[0]
        normalized = [base]
        
        for freq in frequencies[1:]:
            # Reduce to within one octave of base
            f_norm = freq
            while f_norm > base * 2.0:
                f_norm = f_norm / 2.0
            normalized.append(f_norm)
        
        return sorted(normalized)
    
    def _simplify_ratio(self, ratio: float) -> Tuple[int, int]:
        """
        Convert decimal ratio to simple integer ratio
        Uses Python's Fraction with limited denominator
        
        Args:
            ratio: Decimal ratio (e.g., 1.5)
            
        Returns:
            Tuple of (numerator, denominator) e.g., (3, 2)
        """
        # Use fractions library to find best approximation
        frac = Fraction(ratio).limit_denominator(self.max_denominator)
        return (frac.numerator, frac.denominator)
    
    def _match_chord(self, ratios: List[float], num_notes: int) -> Dict:
        """
        Find best matching chord type based on ratios
        
        Args:
            ratios: List of ratios from fundamental
            num_notes: Number of notes in chord
            
        Returns:
            Dictionary with:
                - type: chord type name (e.g., 'major')
                - confidence: match confidence (0-1)
                - match_error: average ratio error
                - description: human-readable description
        """
        best_match = None
        best_score = 0
        best_error = float('inf')
        
        # Only compare with chords of same number of notes
        for chord_name, ideal_ratios in self.CHORD_RATIOS.items():
            if len(ideal_ratios) != num_notes:
                continue
            
            # Calculate match score
            score, error = self._ratio_match_score(ratios, ideal_ratios)
            
            if score > best_score:
                best_score = score
                best_match = chord_name
                best_error = error
        
        # If no good match found
        if best_match is None or best_score < 0.5:
            return {
                'type': 'unknown',
                'confidence': 0.0,
                'match_error': 1.0,
                'description': f'Unknown chord ({num_notes} notes)'
            }
        
        return {
            'type': best_match,
            'confidence': best_score,
            'match_error': best_error,
            'description': self._describe_chord(best_match, ratios)
        }
    
    def _ratio_match_score(self, detected: List[float], 
                          ideal: List[float]) -> Tuple[float, float]:
        """
        Calculate how well detected ratios match ideal ratios
        
        Returns:
            (score, error) where:
                score: 0-1, higher is better match
                error: average relative error
        """
        if len(detected) != len(ideal):
            return (0.0, 1.0)
        
        errors = []
        for d, i in zip(detected, ideal):
            relative_error = abs(d - i) / i
            errors.append(relative_error)
        
        avg_error = np.mean(errors)
        
        # Convert error to score (exponential decay)
        # error=0 -> score=1, error=tolerance -> score≈0.5
        score = np.exp(-avg_error / self.tolerance)
        
        return (score, avg_error)
    
    def _calculate_consonance(self, frequencies: List[float]) -> float:
        """
        Calculate overall consonance score for the chord
        Based on simplicity of all pairwise frequency ratios
        
        Higher score = more consonant (1.0 = perfect consonance)
        Lower score = more dissonant (0.0 = maximum dissonance)
        
        Args:
            frequencies: List of frequencies
            
        Returns:
            Consonance score (0-1)
        """
        if len(frequencies) < 2:
            return 1.0
        
        consonance_scores = []
        
        # Analyze all pairwise intervals
        for i in range(len(frequencies)):
            for j in range(i + 1, len(frequencies)):
                ratio = frequencies[j] / frequencies[i]
                score = self._interval_consonance(ratio)
                consonance_scores.append(score)
        
        # Return average consonance
        return float(np.mean(consonance_scores))
    
    def _interval_consonance(self, ratio: float) -> float:
        """
        Calculate consonance score for a single interval ratio
        Based on Helmholtz ordering and neural synchronization theory
        
        Args:
            ratio: Frequency ratio (higher / lower)
            
        Returns:
            Consonance score (0-1)
        """
        # Normalize ratio to be >= 1.0
        if ratio < 1.0:
            ratio = 1.0 / ratio
        
        # Check octave equivalence (reduce to within one octave)
        while ratio > 2.0:
            ratio = ratio / 2.0
        
        # Find closest matching interval ratio
        best_consonance = 0.4  # Default for unmatched ratios
        
        for (num, den), (name, consonance) in self.INTERVAL_RATIOS.items():
            ideal_ratio = num / den
            
            # Normalize ideal ratio similarly
            if ideal_ratio < 1.0:
                ideal_ratio = 1.0 / ideal_ratio
            while ideal_ratio > 2.0:
                ideal_ratio = ideal_ratio / 2.0
            
            # Calculate relative error
            error = abs(ratio - ideal_ratio) / ideal_ratio
            
            if error < self.tolerance:
                # Match found - return consonance score
                # adjusted by how close we are to ideal
                match_quality = 1.0 - (error / self.tolerance)
                current_consonance = consonance * match_quality
                
                if current_consonance > best_consonance:
                    best_consonance = current_consonance
        
        return best_consonance
    
    def _analyze_intervals(self, frequencies: List[float]) -> List[IntervalAnalysis]:
        """
        Analyze all pairwise intervals in the chord
        
        Args:
            frequencies: Sorted list of frequencies
            
        Returns:
            List of IntervalAnalysis objects
        """
        intervals = []
        
        for i in range(len(frequencies)):
            for j in range(i + 1, len(frequencies)):
                f1, f2 = frequencies[i], frequencies[j]
                ratio = f2 / f1
                
                # Simplify ratio
                simplified = self._simplify_ratio(ratio)
                
                # Get interval name and consonance
                interval_name = self._get_interval_name(simplified)
                consonance = self._interval_consonance(ratio)
                
                # Calculate cents
                cents = 1200 * np.log2(ratio)
                
                intervals.append(IntervalAnalysis(
                    freq1=f1,
                    freq2=f2,
                    ratio=ratio,
                    simplified=simplified,
                    interval_name=interval_name,
                    consonance=consonance,
                    cents=cents
                ))
        
        return intervals
    
    def _get_interval_name(self, simplified: Tuple[int, int]) -> str:
        """
        Get interval name from simplified ratio
        
        Args:
            simplified: (numerator, denominator) tuple
            
        Returns:
            Interval name string
        """
        # Check if it matches a known interval
        if simplified in self.INTERVAL_RATIOS:
            return self.INTERVAL_RATIOS[simplified][0]
        
        # Try inverted ratio
        inverted = (simplified[1], simplified[0])
        if inverted in self.INTERVAL_RATIOS:
            name = self.INTERVAL_RATIOS[inverted][0]
            return f"{name}_inv"
        
        # Unknown interval - describe by ratio
        return f"interval_{simplified[0]}:{simplified[1]}"
    
    def _describe_chord(self, chord_type: str, ratios: List[float]) -> str:
        """
        Generate human-readable description of chord
        
        Args:
            chord_type: Type identifier (e.g., 'major')
            ratios: Detected ratios
            
        Returns:
            Description string
        """
        descriptions = {
            'major': 'Major triad (4:5:6 ratio)',
            'minor': 'Minor triad (10:12:15 ratio)',
            'diminished': 'Diminished triad',
            'augmented': 'Augmented triad (16:20:25 ratio)',
            'sus4': 'Suspended 4th (3:4:4.5 ratio)',
            'sus2': 'Suspended 2nd (8:9:12 ratio)',
            'maj7': 'Major 7th (8:10:12:15 ratio)',
            'min7': 'Minor 7th (10:12:15:18 ratio)',
            'dom7': 'Dominant 7th',
            'm7b5': 'Half-diminished 7th',
            'dim7': 'Diminished 7th',
            'maj9': 'Major 9th',
            'min9': 'Minor 9th',
            'dom9': 'Dominant 9th',
        }
        
        base_desc = descriptions.get(chord_type, f'{chord_type} chord')
        
        # Add ratio information
        ratio_str = ':'.join([f"{r:.2f}" for r in ratios])
        
        return f"{base_desc} [ratios: {ratio_str}]"
    
    def _identify_root_position(self, frequencies: List[float], 
                                chord_match: Dict) -> Optional[str]:
        """
        Try to identify the root position equivalent of the chord
        (e.g., "C major" if we can determine the root note)
        
        This is a bridge between ratio-based and traditional analysis
        
        Args:
            frequencies: Detected frequencies
            chord_match: Chord match result
            
        Returns:
            String like "C major" or None if not identifiable
        """
        # This requires mapping frequencies to note names
        # For now, return None - can be implemented later
        # if we want to bridge to traditional notation
        return None
    
    def ratio_to_cents(self, ratio: float) -> float:
        """
        Convert frequency ratio to cents
        
        Args:
            ratio: Frequency ratio
            
        Returns:
            Interval in cents
        """
        return 1200 * np.log2(ratio)
    
    def cents_to_ratio(self, cents: float) -> float:
        """
        Convert cents to frequency ratio
        
        Args:
            cents: Interval in cents
            
        Returns:
            Frequency ratio
        """
        return 2 ** (cents / 1200)


def demo():
    """Demonstration of ratio analysis"""
    analyzer = FrequencyRatioAnalyzer()
    
    print("=" * 70)
    print("Frequency Ratio Analyzer - Demo")
    print("=" * 70)
    
    # Test 1: Perfect C major triad (4:5:6)
    print("\n1. C Major Triad (C4-E4-G4)")
    print("-" * 70)
    c_major = [261.63, 329.63, 392.00]  # C, E, G
    result = analyzer.analyze_frequencies(c_major)
    
    if result:
        print(f"   Detected frequencies: {result.frequencies}")
        print(f"   Ratios from fundamental: {[f'{r:.3f}' for r in result.ratios]}")
        print(f"   Simplified ratios: {result.simplified_ratios}")
        print(f"   Chord type: {result.chord_match['type']} "
              f"(confidence: {result.chord_match['confidence']:.2%})")
        print(f"   Consonance score: {result.consonance_score:.3f}")
        print(f"   Description: {result.chord_match['description']}")
    
    # Test 2: C minor triad
    print("\n2. C Minor Triad (C4-Eb4-G4)")
    print("-" * 70)
    c_minor = [261.63, 311.13, 392.00]  # C, Eb, G
    result = analyzer.analyze_frequencies(c_minor)
    
    if result:
        print(f"   Chord type: {result.chord_match['type']} "
              f"(confidence: {result.chord_match['confidence']:.2%})")
        print(f"   Consonance score: {result.consonance_score:.3f}")
    
    # Test 3: Diminished triad (less consonant)
    print("\n3. C Diminished Triad (C4-Eb4-Gb4)")
    print("-" * 70)
    c_dim = [261.63, 311.13, 369.99]  # C, Eb, Gb
    result = analyzer.analyze_frequencies(c_dim)
    
    if result:
        print(f"   Chord type: {result.chord_match['type']} "
              f"(confidence: {result.chord_match['confidence']:.2%})")
        print(f"   Consonance score: {result.consonance_score:.3f}")
        print("\n   Interval analysis:")
        for interval in result.intervals:
            print(f"      {interval.freq1:.1f} Hz -> {interval.freq2:.1f} Hz: "
                  f"{interval.interval_name} "
                  f"(ratio {interval.simplified[0]}:{interval.simplified[1]}, "
                  f"consonance: {interval.consonance:.2f})")
    
    # Test 4: Dominant 7th
    print("\n4. C Dominant 7th (C4-E4-G4-Bb4)")
    print("-" * 70)
    c_dom7 = [261.63, 329.63, 392.00, 466.16]  # C, E, G, Bb
    result = analyzer.analyze_frequencies(c_dom7)
    
    if result:
        print(f"   Chord type: {result.chord_match['type']} "
              f"(confidence: {result.chord_match['confidence']:.2%})")
        print(f"   Consonance score: {result.consonance_score:.3f}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    demo()

