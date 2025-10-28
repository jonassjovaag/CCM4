#!/usr/bin/env python3
"""
Rational Rhythm Ratio Analyzer
Ported from Brandtsegg/Formo rhythm_ratio_analyzer

This module analyzes rhythmic sequences by finding the simplest rational
relationships between delta times, using competing theories and evidence-based ranking.

Core concept: For a sequence of timestamps, generate N-1 competing theories
where each delta time serves as a reference unit. Evaluate theories using:
- Deviation (approximation error)
- Complexity (Barlow indigestability)
- Autocorrelation (pattern repetition)

@author: Øyvind Brandtsegg, Daniel Formo (original)
@adapted_by: Jonas Sjøvaag (CCM3 integration)
@license: GPL
"""

import numpy as np
import math
from typing import Tuple, List, Dict, Optional

np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)


class RatioAnalyzer:
    """
    Rational rhythm analysis engine
    
    Analyzes timing sequences to find simplest rational relationships,
    providing structural rhythm understanding beyond statistical features.
    """
    
    def __init__(self, 
                 complexity_weight: float = 0.5,
                 deviation_weight: float = 0.5,
                 simplify: bool = True,
                 div_limit: int = 4):
        """
        Initialize ratio analyzer
        
        Args:
            complexity_weight: Weight for complexity in evaluation (0.0-1.0)
            deviation_weight: Weight for deviation in evaluation (0.0-1.0)
            simplify: Whether to simplify duration patterns
            div_limit: Division limit for rational approximation (2, 4, 8, 16)
        """
        self.weights = [complexity_weight, deviation_weight]
        self.simplify = simplify
        self.div_limit = div_limit
    
    def set_precision(self, precision: float):
        """
        Set balance between complexity and deviation
        
        Args:
            precision: 0.0 (prefer simple patterns) to 1.0 (prefer accurate patterns)
        """
        complexity = precision
        deviation = 1 - precision
        self.weights = [complexity, deviation]
    
    def rational_approx(self, n: float, div_limit: Optional[int] = None) -> Tuple[int, int, float]:
        """
        Find simplest rational approximation of float
        
        Args:
            n: Float to approximate
            div_limit: Division limit (2, 4, 8, 16)
            
        Returns:
            (numerator, denominator, deviation)
        """
        if div_limit is None:
            div_limit = self.div_limit
            
        # Faster rational approx for 2 and 3 divisions
        if div_limit == 2:
            fact = np.array([2])
            threshold = 0.2501  # Finer resolution with small n
        else:
            fact = np.array([3, 4])
            threshold = 0.208333
        
        if div_limit == 8:
            fact *= 2
            threshold /= 2
        if div_limit == 16:
            fact *= 4
            threshold /= 4
        
        # Adjust threshold for small values
        while n < threshold:
            fact *= 2
            threshold /= 2
            if threshold < 0.011:
                break
        
        # Handle very small values
        if n < (1/64):
            num = 1
            denom = 64
        else:
            res = n * fact
            dev = np.abs(np.round(res) - res) * (1/fact)  # Adjust deviation by factor
            num = round(res[np.argmin(dev)])
            denom = fact[np.argmin(dev)]
        
        # Simplify using GCD
        gcd = np.gcd(num, denom)
        num //= gcd
        denom //= gcd
        
        deviation = (n - (num/denom))
        return int(num), int(denom), float(deviation)
    
    def ratio_to_each(self, timeseries: List[float], mode: str = 'connect', div_limit: Optional[int] = None) -> np.ndarray:
        """
        Calculate ratios of delta times to each other delta time
        
        Generates competing theories where each delta time serves as reference.
        In 'connect' mode, also considers combinations of neighboring deltas.
        
        Args:
            timeseries: List of timestamps
            mode: 'connect' to include neighbor combinations, else simple
            div_limit: Division limit for rational approximation (uses self.div_limit if None)
            
        Returns:
            Array of shape (num_theories, num_deltas, 5)
            where last dim is [numerator, denominator, deviation, delta, ref_delta]
        """
        if div_limit is None:
            div_limit = self.div_limit
        t_series_len = len(timeseries) - 1
        ref_deltas = []
        deltas = []
        
        # Build reference deltas and delta list
        for i in range(t_series_len):
            delta = timeseries[i+1] - timeseries[i]
            ref_deltas.append(delta)
            deltas.append(delta)
            
            # Also include combinations of neighboring deltas
            if (mode == 'connect') and (i < (t_series_len - 1)):
                ref_deltas.append(timeseries[i+2] - timeseries[i])
        
        ref_deltas_len = len(ref_deltas)
        ratios = np.zeros((ref_deltas_len, t_series_len, 5))
        
        # For each reference delta, find ratios to all deltas
        for i in range(ref_deltas_len):
            ref_delta = ref_deltas[i]
            for j in range(t_series_len):
                delta = deltas[j]
                ratio = delta / ref_delta
                numerator, denom, deviation = self.rational_approx(ratio, div_limit)
                ratios[i, j] = [numerator, denom, deviation, delta, ref_delta]
        
        return ratios
    
    def make_commondiv_ratios_single(self, ratio_sequence: np.ndarray, 
                                     commondiv: str = 'auto') -> np.ndarray:
        """
        Set all ratios in a suggestion to common denominator
        
        Args:
            ratio_sequence: Array with columns [numerator, denominator, ...]
            commondiv: 'auto' or integer denominator
            
        Returns:
            Ratio sequence with common denominator
        """
        d = ratio_sequence[:, 1].astype(int)
        
        if commondiv == 'auto':
            d_ = np.unique(d)
            commondiv = math.lcm(*d_)
        
        f = commondiv / d[:]
        ratio_sequence[:, 0] *= f
        ratio_sequence[:, 1] *= f
        
        return ratio_sequence
    
    def make_duration_pattern(self, ratio_sequence: np.ndarray) -> np.ndarray:
        """
        Convert ratio sequence to duration pattern (integer durations)
        
        Args:
            ratio_sequence: Ratio array from ratio_to_each
            
        Returns:
            Integer duration pattern (e.g., [2, 1, 1, 2])
        """
        ratio_sequence = self.make_commondiv_ratios_single(ratio_sequence)
        n = ratio_sequence[:, 0]
        return n
    
    def make_box_notation(self, dur_pattern: np.ndarray) -> List[int]:
        """
        Convert duration pattern to binary trigger sequence
        
        1 = transient, 0 = space
        e.g., [6, 3, 3] becomes [1,0,0,0,0,0, 1,0,0, 1,0,0]
        
        Args:
            dur_pattern: Integer duration pattern
            
        Returns:
            Binary trigger sequence
        """
        box_notation = []
        for num in dur_pattern:
            box_notation.append(1)
            for i in range(int(num) - 1):
                box_notation.append(0)
        box_notation.append(1)  # Terminator after last duration
        return box_notation
    
    def prime_factorization(self, n: int) -> Dict[int, int]:
        """
        Prime factorization as dictionary {prime: multiplicity}
        
        Args:
            n: Integer to factorize
            
        Returns:
            Dict mapping primes to their multiplicity
        """
        prime_factors = {}
        i = 2
        while i**2 <= n:
            if n % i:
                i += 1
            else:
                n /= i
                prime_factors[i] = prime_factors.get(i, 0) + 1
        
        if n > 1:
            prime_factors[int(n)] = prime_factors.get(int(n), 0) + 1
        
        return prime_factors
    
    def indigestability(self, n: int) -> float:
        """
        Barlow's indigestability measure for complexity
        
        Lower values = simpler/more consonant
        
        Args:
            n: Integer to measure
            
        Returns:
            Indigestability score
        """
        d = self.prime_factorization(n)
        b = 0.0
        for p, mult in d.items():
            b += (mult * ((p - 1)**2) / p)
        return b * 2
    
    def suavitatis(self, n: int) -> float:
        """
        Euler's gradus suavitatis for complexity
        
        Args:
            n: Integer to measure
            
        Returns:
            Suavitatis score
        """
        d = self.prime_factorization(n)
        s = 0
        for p, mult in d.items():
            s += mult * (p - 1)
        return s
    
    def dur_pattern_height(self, d: np.ndarray) -> float:
        """
        Complexity measure of duration pattern using Barlow indigestability
        
        Args:
            d: Duration pattern array
            
        Returns:
            Total complexity score
        """
        indigest = 0.0
        for n in d:
            indigest += self.indigestability(int(n))
        return indigest
    
    def fit_tempo_from_dur_pattern(self, dur_pattern: np.ndarray, t: np.ndarray) -> float:
        """
        Estimate tempo from duration pattern and time series
        
        Args:
            dur_pattern: Integer duration pattern
            t: Original timestamp series
            
        Returns:
            Tempo in BPM (subdiv tempo)
        """
        length = len(dur_pattern)
        t_diff = np.diff(t)
        tempi = 0.0
        num_iterations = 0
        subsize = 1
        
        # Check all subsequences
        while subsize <= length:
            for i in range(length - subsize + 1):
                num_iterations += 1
                sub_dur = dur_pattern[i:i+subsize]
                sub_time = t_diff[i:i+subsize]
                temp = (np.sum(sub_time) / np.sum(sub_dur))
                tempi += temp
            subsize += 1
        
        tempo = 60 / (tempi / num_iterations)
        return tempo
    
    def get_dur_pattern_deviations(self, dur_pattern: np.ndarray, 
                                   t: np.ndarray, tempo: float) -> np.ndarray:
        """
        Calculate timing deviations from quantized pattern
        
        Args:
            dur_pattern: Integer duration pattern
            t: Original timestamp series
            tempo: Tempo in BPM
            
        Returns:
            Deviation array (fraction of delta time)
        """
        step_size = 60 / tempo
        t_quantized = [t[0]]
        
        for i in range(1, len(dur_pattern) + 1):
            t_quantized.append(t_quantized[i-1] + (dur_pattern[i-1] * step_size))
        
        t_dev = t - t_quantized
        dur_dev = (t_dev[1:] / np.diff(t_quantized))
        
        return dur_dev
    
    def get_deviation_polarity(self, deviations: np.ndarray, threshold: float = 0.05) -> np.ndarray:
        """
        Quantize deviations to -1, 0, or 1
        
        Args:
            deviations: Deviation array
            threshold: Threshold for "on-time" classification
            
        Returns:
            Array of -1 (early), 0 (on-time), 1 (late)
        """
        pos = np.greater(deviations, threshold) * 1
        neg = np.less(deviations, -threshold) * -1
        deviations_polarity = pos + neg
        return deviations_polarity
    
    def simplify_dur_pattern(self, dur_pattern: np.ndarray, deviation: np.ndarray) -> List[int]:
        """
        Simplify duration pattern using deviation polarity
        
        Corrects for odd combinations (e.g., triplets + 8ths mixed)
        
        Args:
            dur_pattern: Integer duration pattern
            deviation: Deviation array
            
        Returns:
            Simplified duration pattern
        """
        dur_pattern_2 = np.copy(dur_pattern)
        
        if np.min(dur_pattern) > 1:
            dur_pattern_2 = dur_pattern_2 / 2
            pos = np.greater(deviation, 0.0001) * 0.1
            neg = np.less(deviation, -0.0001) * -0.1
            deviations_polarity = pos + neg
            dur_pattern_2 += deviations_polarity
            dur_pattern_2 = np.round(dur_pattern_2)
        
        return np.round(dur_pattern_2).astype('int').tolist()
    
    def indispensability_subdiv(self, trigger_seq: List[int]) -> Tuple[int, int]:
        """
        Find pattern subdivision based on Barlow indispensability
        
        Args:
            trigger_seq: Binary trigger sequence
            
        Returns:
            (subdivision, position) tuple
        """
        indis_2 = np.array([1, 0])
        indis_3 = np.array([2, 0, 1])
        indis_3 = (indis_3 / np.max(indis_3))
        
        # All indispensabilities (in increasing order of preference)
        indis_all = [indis_3, indis_2]
        for i in range(len(indis_all)):
            # Tile until long enough
            indis_all[i] = np.tile(indis_all[i], 
                                  int(np.ceil(len(trigger_seq) / len(indis_all[i])) + 1))
        
        # Score table: [length, max_score, confidence, rotation]
        indis_scores = np.array([[3, 0., 0., 0],
                                [2, 0., 0., 0]])
        
        for i in range(len(indis_all)):
            subscores = np.zeros(int(indis_scores[i][0]))
            for j in range(int(indis_scores[i][0])):
                subscore = np.sum(trigger_seq * indis_all[i][j:len(trigger_seq)+j])
                subscores[j] = subscore
            
            indis_scores[i, 1] = np.max(subscores)
            minimum = np.min(subscores)
            if minimum == 0:
                minimum = 1
            indis_scores[i, 2] = np.max(subscores) / minimum
            
            # Find rotation with max score (prefer least rotation)
            found_max = False
            for j in np.argsort(subscores):
                if (subscores[j] == np.max(subscores)) and not found_max:
                    indis_scores[i, 3] = j
                    found_max = True
        
        # Rank by score
        ranked = np.argsort(indis_scores[:, 1])
        subdiv = indis_scores[ranked[-1], 0]
        position = indis_scores[ranked[-1], 3]
        
        # Handle ties using confidence
        test_best = 2
        while test_best <= len(indis_scores):
            if indis_scores[ranked[-test_best], 1] == indis_scores[ranked[-1], 1]:
                if indis_scores[ranked[-test_best], 2] > indis_scores[ranked[-1], 2]:
                    subdiv = indis_scores[ranked[-test_best]][0]
                    position = indis_scores[ranked[-test_best]][3]
            test_best += 1
            if test_best > len(indis_scores):
                break
        
        return int(subdiv), int(position)
    
    def normalize_and_add_scores(self, scores: List[np.ndarray], 
                                 weights: List[float],
                                 invert: Optional[List[int]] = None) -> np.ndarray:
        """
        Normalize and combine multiple score lists
        
        Args:
            scores: List of score arrays
            weights: Weight for each score list
            invert: List of 0/1 indicating whether to invert each score
            
        Returns:
            Combined score array
        """
        scoresum = np.zeros(len(scores[0]))
        
        if invert is None:
            invert = np.zeros(len(weights))
        
        for i in range(len(scores)):
            smax = max(scores[i])
            smin = min(scores[i])
            
            if smax == smin:
                s = 1
            else:
                s = (np.array(scores[i]) - smin) / (smax - smin)
            
            if invert[i] > 0:
                s = np.subtract(1, s)
            
            scoresum += s * weights[i]
        
        return scoresum
    
    def dur_pattern_suggestions(self, t: np.ndarray) -> Tuple[List, List, List]:
        """
        Analyze time sequence and generate duration pattern suggestions
        
        Args:
            t: Timestamp array
            
        Returns:
            (duration_patterns, deviations, tempi) tuple
        """
        timedata = t.tolist()
        ratios = self.ratio_to_each(timedata, div_limit=self.div_limit)
        
        duration_patterns = []
        deviations = []
        tempi = []
        
        for i in range(len(ratios)):
            dur_pattern = self.make_duration_pattern(ratios[i]).astype('int').tolist()
            
            if dur_pattern not in duration_patterns:
                tempo = self.fit_tempo_from_dur_pattern(np.array(dur_pattern), t)
                duration_patterns.append(dur_pattern)
                tempi.append(tempo)
                dev = self.get_dur_pattern_deviations(np.array(dur_pattern), t, tempo)
                deviations.append(dev)
                
                # Try simplified version
                if self.simplify:
                    d2 = self.simplify_dur_pattern(np.array(dur_pattern), dev)
                    d2_half = (np.array(d2) / 2).astype('int').tolist()
                    
                    if (d2 not in duration_patterns) and (d2_half not in duration_patterns):
                        tempo = self.fit_tempo_from_dur_pattern(np.array(d2), t)
                        duration_patterns.append(d2)
                        tempi.append(tempo)
                        dev = self.get_dur_pattern_deviations(np.array(d2), t, tempo)
                        deviations.append(dev)
        
        return duration_patterns, deviations, tempi
    
    def evaluate(self, duration_patterns: List, deviations: List, weights: List[float]) -> np.ndarray:
        """
        Evaluate fitness of duration pattern suggestions
        
        Args:
            duration_patterns: List of duration pattern suggestions
            deviations: List of deviation arrays
            weights: [complexity_weight, deviation_weight]
            
        Returns:
            Score array
        """
        heights = []
        for d in duration_patterns:
            height = self.dur_pattern_height(np.array(d))
            heights.append(height)
        
        scoresum = self.normalize_and_add_scores([deviations, heights], weights)
        return scoresum
    
    def analyze(self, t: np.ndarray) -> Dict:
        """
        Main analysis entry point
        
        Analyzes time sequence and returns best rational interpretation.
        
        Args:
            t: Timestamp array (in seconds)
            
        Returns:
            Dictionary with:
                - duration_pattern: Best integer duration pattern
                - tempo: Subdiv tempo in BPM
                - pulse: Pulse subdivision
                - pulse_position: Phase offset
                - complexity: Barlow indigestability score
                - deviations: Per-event timing deviations
                - deviation_polarity: Quantized deviations (-1, 0, 1)
                - confidence: Analysis confidence (inverse of best score)
                - all_suggestions: All competing theories with scores
        """
        # Generate competing theories
        duration_patterns, deviations, tempi = self.dur_pattern_suggestions(t)
        
        # Calculate deviation sums
        devsums = []
        for dev in deviations:
            devsum = np.sum(np.abs(dev))
            devsums.append(devsum)
        
        # Evaluate and rank
        scores = self.evaluate(duration_patterns, devsums, self.weights)
        
        # Best theory
        best = np.argsort(scores)[0]
        best_dur_pattern = duration_patterns[best]
        best_tempo = tempi[best]
        best_deviations = deviations[best]
        
        # Calculate pulse from trigger sequence
        trigger_seq = self.make_box_notation(np.array(best_dur_pattern))
        pulse, pulsepos = self.indispensability_subdiv(trigger_seq)
        
        # Calculate complexity
        complexity = self.dur_pattern_height(np.array(best_dur_pattern))
        
        # Deviation polarity
        dev_polarity = self.get_deviation_polarity(np.array(best_deviations))
        
        # Confidence (inverse of normalized score, clamped)
        confidence = 1.0 - min(scores[best], 1.0)
        
        return {
            'duration_pattern': best_dur_pattern,
            'tempo': float(best_tempo),
            'pulse': int(pulse),
            'pulse_position': int(pulsepos),
            'complexity': float(complexity),
            'deviations': best_deviations.tolist(),
            'deviation_polarity': dev_polarity.tolist(),
            'confidence': float(confidence),
            'all_suggestions': [
                {
                    'duration_pattern': duration_patterns[i],
                    'tempo': tempi[i],
                    'deviation_sum': devsums[i],
                    'score': float(scores[i])
                }
                for i in range(len(duration_patterns))
            ]
        }


def test_analyzer():
    """Test the ratio analyzer"""
    print("🎵 Testing RatioAnalyzer...")
    
    analyzer = RatioAnalyzer(complexity_weight=0.5, deviation_weight=0.5)
    
    # Test case: Simple 4/4 pattern
    timeseries = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    
    print(f"\nAnalyzing timeseries: {timeseries}")
    result = analyzer.analyze(timeseries)
    
    print(f"\n✅ Best interpretation:")
    print(f"   Duration pattern: {result['duration_pattern']}")
    print(f"   Tempo: {result['tempo']:.1f} BPM")
    print(f"   Pulse: {result['pulse']}")
    print(f"   Complexity: {result['complexity']:.2f}")
    print(f"   Confidence: {result['confidence']:.2f}")
    print(f"   Deviations: {result['deviations']}")
    print(f"   Deviation polarity: {result['deviation_polarity']}")
    
    print(f"\n📊 Alternative interpretations: {len(result['all_suggestions'])}")
    for i, sug in enumerate(result['all_suggestions'][:3]):
        print(f"   {i+1}. {sug['duration_pattern']} @ {sug['tempo']:.1f} BPM (score: {sug['score']:.3f})")


if __name__ == "__main__":
    test_analyzer()

