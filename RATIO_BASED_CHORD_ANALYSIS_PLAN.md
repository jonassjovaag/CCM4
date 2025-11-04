# Ratio-Based Chord Analysis System - Implementation Plan

## Executive Summary

This document outlines a plan to improve the chord validation system by shifting from **descriptive labeling** (e.g., "this is E major because we detect E, G#, B") to **mathematical frequency ratio analysis** (e.g., "this is a major chord because the detected frequency ratios match 4:5:6").

This approach is grounded in psychoacoustic research on consonance/dissonance and provides a more fundamental, physics-based understanding of harmonic structures.

---

## Background Research

### Key References

1. **rsif20080143.txt** - "Perception of musical consonance and dissonance: an outcome of neural synchronization" (Shapira Lots & Stone, 2008)
   - Explains why simple frequency ratios (2:3, 3:4, 4:5, etc.) are perceived as consonant
   - Based on neural synchronization theory, not just Helmholtz's beating theory
   - Provides ordering of consonance from most to least consonant

2. **2106.08479v1.txt** - "Tonal Frequencies, Consonance, Dissonance: A Math-Bio Intersection" (Mathew, 2021)
   - Mathematical model for determining tonal frequencies
   - Analysis of chord consonance based on intersection patterns of sinusoidal waves
   - Ranking of triads: Major (0.054s) > Sus4 (0.081s) > Sus2 (0.109s) > Minor (0.21s) > Aug (1.41s) > Dim (1.51s)

3. **fpsyg-09-00381.txt** - "Computational Approach to Musical Consonance and Dissonance" (Trulla et al., 2018)
   - Recurrence Quantification Analysis (RQA) approach
   - Second-order beats and neural processing
   - Demonstrates link between consonance and dynamical features of signals

### Interval Ratios Table (from Helmholtz 1877)

| Interval Name    | Ratio    | ΔΩ    | Consonance Level       |
|------------------|----------|-------|------------------------|
| Unison           | 1:1      | 0.075 | Absolute consonance    |
| Octave           | 1:2      | 0.023 | Absolute consonance    |
| Fifth            | 2:3      | 0.022 | Perfect consonance     |
| Fourth           | 3:4      | 0.012 | Perfect consonance     |
| Major Sixth      | 3:5      | 0.010 | Medial consonance      |
| Major Third      | 4:5      | 0.010 | Medial consonance      |
| Minor Third      | 5:6      | 0.010 | Imperfect consonance   |
| Minor Sixth      | 5:8      | 0.007 | Imperfect consonance   |
| Major Second     | 8:9      | 0.006 | Dissonance             |
| Major Seventh    | 8:15     | 0.005 | Dissonance             |
| Minor Seventh    | 9:16     | 0.003 | Dissonance             |
| Minor Second     | 15:16    | —     | Dissonance             |
| Tritone          | 32:45    | —     | Dissonance             |

---

## Current System Analysis

### Current Approach (simple_chord_validator.py)

```python
# Current method:
1. Play MIDI chord (e.g., [60, 64, 67] = C E G)
2. Collect detected pitches via YIN
3. Convert to MIDI note numbers
4. Compare: "Did we detect C, E, and G?"
5. Conclusion: "This is C major"
```

**Limitations:**
- Relies on note name matching (brittle)
- Doesn't understand WHY a chord sounds the way it does
- Doesn't measure consonance/dissonance quantitatively
- Doesn't work well with detuned/microtonal/non-Western harmonies
- Ignores the fundamental physics of sound perception

---

## Proposed Ratio-Based System

### Core Concept

Instead of saying "this is C major because we detected C, E, G", we analyze:

```python
# New method:
1. Detect fundamental frequencies: [261.63 Hz, 329.63 Hz, 392.00 Hz]
2. Calculate ratios from lowest frequency (fundamental):
   - 261.63 : 329.63 : 392.00
   - Normalized: 1.0 : 1.260 : 1.498
   - Simplified to small integers: 4 : 5 : 6
3. Compare to known chord ratio patterns:
   - Major triad = 4:5:6 ✓
   - Minor triad = 10:12:15
   - Diminished = 10:12:14.3
4. Calculate consonance score based on ratio simplicity
5. Conclusion: "This is a major triad (ratio 4:5:6) with consonance score 0.95"
```

---

## Implementation Architecture

### Phase 1: Frequency Ratio Analyzer Module

Create `listener/ratio_analyzer.py`:

```python
class FrequencyRatioAnalyzer:
    """
    Analyzes detected frequencies and converts them to 
    mathematical ratios for chord identification
    """
    
    def __init__(self, tolerance: float = 0.02):
        self.tolerance = tolerance  # 2% tolerance for ratio matching
        
        # Define ideal chord ratios
        self.chord_ratios = {
            'major': [4, 5, 6],           # Major triad: perfect 4:5:6
            'minor': [10, 12, 15],        # Minor triad: 10:12:15
            'dim': [10, 12, 14.3],        # Diminished (approx, tritone)
            'aug': [16, 20, 25],          # Augmented
            'sus4': [3, 4, 6],            # Suspended 4th
            'sus2': [8, 9, 12],           # Suspended 2nd
            'maj7': [8, 10, 12, 15],      # Major 7th
            'min7': [10, 12, 15, 18],     # Minor 7th
            'dom7': [4, 5, 6, 7],         # Dominant 7th
        }
        
        # Interval ratios for consonance analysis
        self.interval_ratios = {
            (1, 1): ('unison', 1.0),      # Perfect consonance
            (1, 2): ('octave', 0.98),     # Perfect consonance
            (2, 3): ('fifth', 0.95),      # Perfect consonance
            (3, 4): ('fourth', 0.90),     # Perfect consonance
            (4, 5): ('major_third', 0.85),
            (5, 6): ('minor_third', 0.82),
            (3, 5): ('major_sixth', 0.80),
            (5, 8): ('minor_sixth', 0.75),
            (8, 9): ('major_second', 0.60),   # Dissonance starts
            (15, 16): ('minor_second', 0.30),  # Strong dissonance
            (32, 45): ('tritone', 0.25),       # Maximum dissonance
        }
    
    def analyze_frequencies(self, frequencies: List[float]) -> dict:
        """
        Main analysis function
        
        Args:
            frequencies: List of detected fundamental frequencies (Hz)
            
        Returns:
            Dictionary with:
            - ratios: Computed frequency ratios
            - simplified_ratios: Integer approximation
            - chord_match: Best matching chord type
            - consonance_score: Overall consonance (0-1)
            - interval_analysis: Detailed interval breakdown
        """
        if len(frequencies) < 2:
            return None
        
        # Sort frequencies
        freqs = sorted(frequencies)
        
        # Calculate ratios from fundamental (lowest frequency)
        fundamental = freqs[0]
        ratios = [f / fundamental for f in freqs]
        
        # Simplify to small integers (within tolerance)
        simplified = self._simplify_ratios(ratios)
        
        # Find best chord match
        chord_match = self._match_chord(simplified)
        
        # Calculate consonance score
        consonance = self._calculate_consonance(freqs)
        
        # Analyze all intervals
        intervals = self._analyze_intervals(freqs)
        
        return {
            'frequencies': freqs,
            'ratios': ratios,
            'simplified_ratios': simplified,
            'chord_match': chord_match,
            'consonance_score': consonance,
            'intervals': intervals,
            'fundamental': fundamental
        }
    
    def _simplify_ratios(self, ratios: List[float]) -> List[int]:
        """
        Convert decimal ratios to simple integer ratios
        Uses continued fractions algorithm
        """
        # Implementation using Stern-Brocot tree or continued fractions
        # to find best small-integer approximations
        pass
    
    def _match_chord(self, simplified_ratios: List[int]) -> dict:
        """
        Find best matching chord type based on ratios
        
        Returns:
            {
                'type': 'major',
                'confidence': 0.95,
                'match_error': 0.02
            }
        """
        best_match = None
        best_score = 0
        
        for chord_name, ideal_ratios in self.chord_ratios.items():
            if len(simplified_ratios) != len(ideal_ratios):
                continue
            
            # Calculate match score
            score = self._ratio_match_score(simplified_ratios, ideal_ratios)
            
            if score > best_score:
                best_score = score
                best_match = chord_name
        
        return {
            'type': best_match,
            'confidence': best_score,
            'description': self._describe_chord(best_match, simplified_ratios)
        }
    
    def _calculate_consonance(self, frequencies: List[float]) -> float:
        """
        Calculate overall consonance score for the chord
        Based on simplicity of all pairwise frequency ratios
        
        Higher score = more consonant
        """
        if len(frequencies) < 2:
            return 1.0
        
        consonance_scores = []
        
        # Analyze all pairs
        for i in range(len(frequencies)):
            for j in range(i + 1, len(frequencies)):
                ratio = frequencies[j] / frequencies[i]
                score = self._interval_consonance(ratio)
                consonance_scores.append(score)
        
        # Return average consonance
        return np.mean(consonance_scores)
    
    def _interval_consonance(self, ratio: float) -> float:
        """
        Calculate consonance score for a single interval ratio
        Based on Helmholtz ordering and neural synchronization theory
        """
        # Find closest simple ratio
        best_match_score = 0
        
        for (num, den), (name, consonance) in self.interval_ratios.items():
            ideal_ratio = num / den
            error = abs(ratio - ideal_ratio) / ideal_ratio
            
            if error < self.tolerance:
                # Match found - return consonance score
                # adjusted by how close we are to ideal
                match_quality = 1.0 - (error / self.tolerance)
                return consonance * match_quality
        
        # No simple ratio found - dissonant
        return 0.4  # Base dissonance for complex ratios
    
    def _analyze_intervals(self, frequencies: List[float]) -> List[dict]:
        """
        Analyze all pairwise intervals in the chord
        """
        intervals = []
        
        for i in range(len(frequencies)):
            for j in range(i + 1, len(frequencies)):
                ratio = frequencies[j] / frequencies[i]
                simplified = self._simplify_ratio_pair(frequencies[j], frequencies[i])
                
                intervals.append({
                    'freq1': frequencies[i],
                    'freq2': frequencies[j],
                    'ratio': ratio,
                    'simplified': simplified,
                    'interval_name': self._get_interval_name(simplified),
                    'consonance': self._interval_consonance(ratio)
                })
        
        return intervals
```

### Phase 2: Enhanced Chord Validator

Upgrade `simple_chord_validator.py` to use ratio analysis:

```python
class RatioBasedChordValidator:
    """
    Chord validator using mathematical frequency ratios
    instead of note name matching
    """
    
    def __init__(self, ...):
        # Existing initialization
        self.ratio_analyzer = FrequencyRatioAnalyzer()
        self.frequency_buffer = []
        ...
    
    def validate_chord(self, sent_midi: List[int], chord_name: str) -> dict:
        """
        Validate chord using ratio analysis
        
        Returns comprehensive analysis instead of just True/False
        """
        # Play chord and collect frequencies (not just MIDI notes)
        frequencies = self._collect_frequencies(sent_midi)
        
        # Analyze using ratios
        analysis = self.ratio_analyzer.analyze_frequencies(frequencies)
        
        # Calculate validation metrics
        validation = {
            'sent_chord': chord_name,
            'sent_midi': sent_midi,
            'detected_frequencies': frequencies,
            'ratio_analysis': analysis,
            'consonance_score': analysis['consonance_score'],
            'detected_chord_type': analysis['chord_match']['type'],
            'match_confidence': analysis['chord_match']['confidence'],
            'validation_passed': self._validate_match(chord_name, analysis)
        }
        
        return validation
    
    def _collect_frequencies(self, midi_notes: List[int]) -> List[float]:
        """
        Collect fundamental frequencies instead of MIDI note numbers
        """
        self.frequency_buffer = []
        self.collecting = True
        
        # Play chord
        for note in midi_notes:
            self.midi_port.send(mido.Message('note_on', ...))
        
        time.sleep(self.chord_duration)
        
        # Stop chord
        for note in midi_notes:
            self.midi_port.send(mido.Message('note_off', ...))
        
        self.collecting = False
        
        # Return unique fundamental frequencies (clustered)
        return self._cluster_frequencies(self.frequency_buffer)
    
    def _on_pitch_event(self, event: Event):
        """
        Collect fundamental frequencies (not MIDI notes)
        """
        if not self.collecting:
            return
        
        if event.rms_db > -50 and event.f0 > 0:
            # Store actual frequency, not MIDI note
            self.frequency_buffer.append(event.f0)
    
    def _cluster_frequencies(self, raw_freqs: List[float]) -> List[float]:
        """
        Cluster detected frequencies to find stable fundamentals
        Accounts for vibrato, tuning variations, etc.
        """
        # Use simple clustering to find stable frequencies
        # Return one representative frequency per detected note
        pass
```

### Phase 3: Integration with Autonomous Trainer

Update `autonomous_chord_trainer.py`:

```python
class AutonomousChordTrainer:
    
    def __init__(self, ...):
        # Add ratio analyzer
        self.ratio_analyzer = FrequencyRatioAnalyzer()
        
        # Store frequency data alongside MIDI data
        self.frequency_training_data = []
        ...
    
    def _extract_features(self, event: Event) -> np.ndarray:
        """
        Extract features including ratio-based features
        """
        # Existing features
        chroma = self.harmonic_detector.get_chroma_vector()
        
        # NEW: Add ratio-based features
        if len(self.current_frequencies) >= 2:
            ratio_analysis = self.ratio_analyzer.analyze_frequencies(
                self.current_frequencies
            )
            
            ratio_features = [
                ratio_analysis['consonance_score'],
                *ratio_analysis['ratios'][:4],  # Up to 4 notes
                len(ratio_analysis['intervals']),
            ]
        else:
            ratio_features = [0.0] * 10
        
        # Combine features
        features = np.concatenate([
            chroma,
            ratio_features,
            # ... other existing features
        ])
        
        return features
    
    def _validate_prediction(self, ground_truth: ChordInversion) -> bool:
        """
        Validate using both note-matching AND ratio analysis
        """
        # Get detected frequencies
        detected_freqs = self._get_current_frequencies()
        
        # Ratio analysis
        ratio_result = self.ratio_analyzer.analyze_frequencies(detected_freqs)
        
        # Expected chord type from ground truth
        expected_type = self._chord_quality_to_type(ground_truth.quality)
        detected_type = ratio_result['chord_match']['type']
        
        # Validation passes if:
        # 1. Ratio-based chord type matches expected type
        # 2. Consonance score is appropriate for chord type
        # 3. Individual intervals match expected intervals
        
        type_match = (detected_type == expected_type)
        consonance_ok = ratio_result['consonance_score'] > self.consonance_threshold
        
        return type_match and consonance_ok
```

---

## Benefits of Ratio-Based Approach

### 1. **Fundamental Understanding**
- Analyzes WHY a chord sounds the way it does (physics/psychoacoustics)
- Not dependent on Western note naming conventions
- Works with any tuning system (just intonation, equal temperament, microtonal)

### 2. **Quantitative Metrics**
- Consonance score: objective measure of harmonic stability
- Interval analysis: understand which intervals contribute to overall sound
- Match confidence: how well detected ratios match ideal ratios

### 3. **Robustness**
- Less sensitive to tuning variations (slight detuning doesn't break matching)
- Works with detuned instruments, stretched tuning, etc.
- Handles harmonics and overtones more gracefully

### 4. **Educational Value**
- Provides insight into why certain chords sound "stable" vs "unstable"
- Can be used to teach music theory from first principles
- Links perception to mathematical properties

### 5. **Extensibility**
- Can analyze non-traditional chords and microtonal harmonies
- Can measure "chord tension" quantitatively
- Can guide composition by quantifying harmonic relationships

---

## Implementation Phases

### Phase 1: Core Ratio Analyzer (Week 1-2)
- [ ] Create `listener/ratio_analyzer.py`
- [ ] Implement ratio simplification algorithm
- [ ] Implement consonance calculation
- [ ] Test with known frequency combinations
- [ ] Document ratio database for common chords

### Phase 2: Enhanced Validator (Week 2-3)
- [ ] Upgrade `simple_chord_validator.py`
- [ ] Implement frequency collection (not just MIDI)
- [ ] Add ratio-based validation metrics
- [ ] Create comprehensive test suite
- [ ] Compare results with traditional method

### Phase 3: Integration with Trainer (Week 3-4)
- [ ] Add ratio features to `autonomous_chord_trainer.py`
- [ ] Update ML model to use ratio features
- [ ] Implement dual validation (notes + ratios)
- [ ] Train and evaluate performance
- [ ] Document improvements

### Phase 4: Visualization & Analysis Tools (Week 4-5)
- [ ] Create visualization of frequency ratios
- [ ] Show consonance scores graphically
- [ ] Display interval analysis
- [ ] Add real-time ratio monitoring
- [ ] Create educational demos

---

## Testing Strategy

### Test Cases

1. **Perfect Ratios**
   - Generate pure sine waves with exact integer ratios
   - Verify correct identification and consonance scores

2. **Realistic Chords**
   - Use actual instrument recordings
   - Compare ratio analysis with human perception

3. **Edge Cases**
   - Detuned instruments (±10 cents)
   - Microtonal intervals
   - Very low/high registers
   - Chords with overtones/harmonics

4. **Comparison Study**
   - Compare accuracy: ratio-based vs note-based
   - Measure robustness to noise and tuning variations
   - Evaluate computational efficiency

---

## Validation Metrics

### Quantitative Metrics

1. **Chord Type Accuracy**: % of chords correctly identified by type
2. **Consonance Correlation**: Correlation with human-rated consonance
3. **Robustness**: Performance with ±5, ±10, ±20 cent detuning
4. **Speed**: Computation time per analysis

### Qualitative Metrics

1. **Interpretability**: Can musicians understand the analysis?
2. **Usefulness**: Does it provide actionable insights?
3. **Consistency**: Does it give consistent results for similar inputs?

---

## Future Extensions

### 1. Timbre-Aware Analysis
- Account for spectral envelope
- Analyze inharmonicity in real instruments
- Weight ratios by harmonic energy

### 2. Temporal Analysis
- Track ratio stability over time
- Analyze chord transitions and voice leading
- Measure "roughness" from beating

### 3. Cultural/Stylistic Context
- Learn genre-specific consonance preferences
- Adapt to different musical traditions
- Personalize to individual perception

### 4. Real-Time Performance
- Live ratio visualization
- Consonance meter for performers
- Tuning guidance based on target ratios

---

## References

1. Shapira Lots, I., & Stone, L. (2008). Perception of musical consonance and dissonance: an outcome of neural synchronization. *Journal of the Royal Society Interface*, 5(29), 1429-1434.

2. Mathew, S. (2021). Tonal Frequencies, Consonance, Dissonance: A Math-Bio Intersection. Preprint.

3. Trulla, L. L., Di Stefano, N., & Giuliani, A. (2018). Computational approach to musical consonance and dissonance. *Frontiers in Psychology*, 9, 381.

4. Helmholtz, H. V. (1877). *On the Sensations of Tone*. Dover Publications.

5. Roederer, J. G. (1975). *Introduction to the Physics and Psychophysics of Music*. Springer-Verlag.

---

## Conclusion

This ratio-based approach represents a fundamental shift from descriptive labeling to mathematical analysis of harmonic relationships. By grounding chord analysis in the physics of sound and the psychophysics of perception, we create a more robust, interpretable, and extensible system that can:

1. Identify chords based on their mathematical properties
2. Quantify consonance and dissonance objectively
3. Work across different tuning systems and musical traditions
4. Provide educational insights into why music sounds the way it does

The implementation will proceed in phases, maintaining backward compatibility while gradually introducing ratio-based features. The system will be thoroughly tested and validated against both traditional methods and human perception studies.

