# Music Theory Enhancement Plan for CCM3
## Comprehensive Roadmap to Advanced Musical Analysis

---

## Priority 1: Extended Chord Vocabulary (HIGH IMPACT, MEDIUM EFFORT)

### Current State
- Only detects: major, minor, dominant 7th, minor 7th
- Missing: sus, add9, dim, aug, 9ths, 11ths, 13ths, altered dominants

### Enhancement Plan

#### Phase 1.1: Add Common Jazz Chords (2-3 days)
**Location**: `hybrid_training/real_chord_detector.py`

**New chord templates to add:**
```python
chord_templates = {
    # Current (4 types × 12 roots = 48 chords)
    'major', 'minor', 'dom7', 'min7',
    
    # Add Phase 1 (13 new types × 12 roots = 156 new chords)
    'sus2':        [root, 2, 7],           # Suspended 2nd: 1-2-5
    'sus4':        [root, 5, 7],           # Suspended 4th: 1-4-5
    'maj7':        [root, 4, 7, 11],       # Major 7th: 1-3-5-7
    'min7b5':      [root, 3, 6, 10],       # Half-diminished: 1-♭3-♭5-♭7
    'dim7':        [root, 3, 6, 9],        # Diminished 7th: 1-♭3-♭5-♭♭7
    'aug':         [root, 4, 8],           # Augmented: 1-3-♯5
    'add9':        [root, 2, 4, 7],        # Add9: 1-3-5-9
    '6':           [root, 4, 7, 9],        # Major 6th: 1-3-5-6
    'min6':        [root, 3, 7, 9],        # Minor 6th: 1-♭3-5-6
    '9':           [root, 4, 7, 10, 2],    # Dominant 9th: 1-3-5-♭7-9
    'maj9':        [root, 4, 7, 11, 2],    # Major 9th: 1-3-5-7-9
    'min9':        [root, 3, 7, 10, 2],    # Minor 9th: 1-♭3-5-♭7-9
    '7#9':         [root, 4, 7, 10, 3],    # Hendrix chord: 1-3-5-♭7-♯9
}
```

**Implementation steps:**
1. Extend `_create_chord_templates()` method
2. Update chord detection to handle 4-5 note chords (currently only 3-4)
3. Add chord quality scoring (major vs minor 3rd, perfect vs dim/aug 5th)
4. Test on jazz recordings (Coltrane, Bill Evans, Herbie Hancock)

**Expected outcome**: 
- Detect 17 chord types instead of 4
- Total: 204 chords (17 types × 12 roots)
- Jazz chord recognition: 85%+ accuracy

#### Phase 1.2: Add Extended Chords (2-3 days)
**11th and 13th chords:**
```python
'11':          [root, 4, 7, 10, 2, 5],     # Dominant 11th
'maj11':       [root, 4, 7, 11, 2, 5],     # Major 11th
'min11':       [root, 3, 7, 10, 2, 5],     # Minor 11th
'13':          [root, 4, 7, 10, 2, 5, 9],  # Dominant 13th
'maj13':       [root, 4, 7, 11, 2, 5, 9],  # Major 13th
'min13':       [root, 3, 7, 10, 2, 5, 9],  # Minor 13th
```

**Challenge**: 6-7 note chords are often voiced with omissions
**Solution**: Template matching with "optional note" flags

**Implementation:**
```python
def _create_extended_chord_template(root, chord_type, required_notes, optional_notes):
    """
    Match chord even if optional notes are missing
    Example: 13th chord requires 1-3-7, but 5-9-13 can be omitted
    """
    template = {
        'required': required_notes,  # Must be present
        'optional': optional_notes,  # Nice to have
        'weight': importance_weight
    }
    return template
```

#### Phase 1.3: Add Altered Dominants (1-2 days)
**Altered chords (jazz essentials):**
```python
'7b9':         [root, 4, 7, 10, 1],    # Dominant ♭9
'7#9':         [root, 4, 7, 10, 3],    # Dominant ♯9 (Hendrix)
'7b5':         [root, 4, 6, 10],       # Dominant ♭5
'7#5':         [root, 4, 8, 10],       # Dominant ♯5
'7alt':        [root, 4, 6/8, 10, 1/3], # Altered (♭9,♯9,♭5,♯5)
'7#11':        [root, 4, 7, 10, 6],    # Lydian dominant
```

**Test cases:**
- John Coltrane "Giant Steps" (altered dominants)
- Herbie Hancock "Dolphin Dance" (lydian dominant)
- Bill Evans "Waltz for Debby" (sus chords, add9)

---

## Priority 2: Voice Leading Analysis (HIGH IMPACT, HIGH EFFORT)

### Current State
- No tracking of melodic motion between chords
- No detection of parallel 5ths/octaves
- No analysis of voice independence

### Enhancement Plan

#### Phase 2.1: Chord Voicing Detection (3-4 days)
**Location**: New file `hybrid_training/voice_leading_analyzer.py`

**Core concept**: Track individual voices across time

```python
class VoiceLeadingAnalyzer:
    """
    Analyzes melodic motion between chords
    Based on Tymoczko (2011) "A Geometry of Chords"
    """
    
    def __init__(self):
        self.voices = {
            'soprano': [],  # Top voice
            'alto': [],     # 2nd voice
            'tenor': [],    # 3rd voice
            'bass': []      # Bottom voice
        }
    
    def analyze_voice_leading(self, chord_progression, voicings):
        """
        Analyze voice leading between consecutive chords
        
        Returns:
        - smoothness_score: Average interval size (smaller = smoother)
        - parallel_fifths: List of parallel 5th violations
        - parallel_octaves: List of parallel octave violations
        - contrary_motion: Instances of contrary motion (good)
        - voice_crossings: Instances of voice crossing
        """
        pass
```

**Implementation steps:**

1. **Separate chord tones by frequency** (assign to soprano/alto/tenor/bass)
```python
def _assign_voices(self, pitch_classes, frequencies):
    """
    Assign detected pitches to voices based on frequency
    Soprano = highest, Bass = lowest
    """
    sorted_pitches = sorted(zip(frequencies, pitch_classes), reverse=True)
    return {
        'soprano': sorted_pitches[0] if len(sorted_pitches) > 0 else None,
        'alto': sorted_pitches[1] if len(sorted_pitches) > 1 else None,
        'tenor': sorted_pitches[2] if len(sorted_pitches) > 2 else None,
        'bass': sorted_pitches[3] if len(sorted_pitches) > 3 else None,
    }
```

2. **Track voice motion**
```python
def _calculate_voice_motion(self, prev_voices, current_voices):
    """
    Calculate interval motion for each voice
    Returns: dict of {voice: interval_in_semitones}
    """
    motion = {}
    for voice_name in ['soprano', 'alto', 'tenor', 'bass']:
        if prev_voices[voice_name] and current_voices[voice_name]:
            interval = current_voices[voice_name] - prev_voices[voice_name]
            motion[voice_name] = interval
    return motion
```

3. **Detect parallel motion violations**
```python
def _detect_parallel_motion(self, prev_voices, current_voices, motion):
    """
    Detect parallel 5ths and octaves (voice leading violations)
    
    Parallel 5th: Two voices move by same interval AND form perfect 5th
    Parallel octave: Two voices move by same interval AND form octave
    """
    violations = []
    
    for voice1, voice2 in combinations(['soprano', 'alto', 'tenor', 'bass'], 2):
        # Check if both voices moved by same interval (parallel motion)
        if motion[voice1] == motion[voice2]:
            # Check interval between voices
            prev_interval = (prev_voices[voice1] - prev_voices[voice2]) % 12
            curr_interval = (current_voices[voice1] - current_voices[voice2]) % 12
            
            if prev_interval == 7 and curr_interval == 7:  # Parallel 5ths
                violations.append(('parallel_5th', voice1, voice2))
            elif prev_interval == 0 and curr_interval == 0:  # Parallel octaves
                violations.append(('parallel_octave', voice1, voice2))
    
    return violations
```

4. **Calculate smoothness score**
```python
def _calculate_smoothness(self, motion_history):
    """
    Smoothness = average absolute interval size
    Smooth voice leading: small intervals (stepwise motion)
    Rough voice leading: large intervals (leaps)
    
    Good: 0-2 semitones average (stepwise)
    OK: 2-4 semitones average
    Rough: >4 semitones average
    """
    intervals = [abs(m) for m in motion_history.values()]
    return np.mean(intervals) if intervals else 0.0
```

**Test cases:**
- Bach chorales (should show smooth voice leading, no parallel 5ths)
- Modern pop (may have parallel 5ths - stylistically acceptable)
- Jazz voicings (often use voice leading for chord transitions)

#### Phase 2.2: Bass Line Analysis (2-3 days)

**Track bass motion and chord inversions:**

```python
class BassLineAnalyzer:
    """
    Analyzes bass line and chord inversions
    """
    
    def analyze_bass_line(self, chord_progression, bass_notes):
        """
        Determine:
        - Root position vs inversions
        - Bass line contour (ascending, descending, static)
        - Bass line intervals (stepwise, leaps)
        - Pedal points (sustained bass notes)
        """
        results = {
            'inversions': [],      # List of (chord, inversion_type)
            'bass_contour': [],    # Direction of bass motion
            'pedal_points': [],    # Sustained bass notes
            'walking_bass': False  # Characteristic stepwise motion
        }
        
        for i, (chord, bass) in enumerate(zip(chord_progression, bass_notes)):
            # Determine inversion
            root = chord.root
            if bass == root:
                results['inversions'].append('root_position')
            elif bass == (root + 4) % 12 or bass == (root + 3) % 12:
                results['inversions'].append('first_inversion')  # 3rd in bass
            elif bass == (root + 7) % 12:
                results['inversions'].append('second_inversion')  # 5th in bass
            else:
                results['inversions'].append('slash_chord')  # Non-chord tone in bass
        
        return results
```

**Applications:**
- Jazz walking bass detection
- Classical bass line analysis
- Pedal point detection (organ music, Pink Floyd)
- Slash chord detection (C/E, G/B)

---

## Priority 3: Modulation Detection (MEDIUM IMPACT, MEDIUM EFFORT)

### Current State
- Key changes not tracked
- No pivot chord analysis
- No tonicization detection

### Enhancement Plan

#### Phase 3.1: Key Change Detection (2-3 days)
**Location**: Extend `real_chord_detector.py`

**Strategy**: Windowed key analysis with confidence scores

```python
class ModulationDetector:
    """
    Detects key changes (modulations) in music
    Based on Temperley (1999) "What's Key for Key?"
    """
    
    def __init__(self, window_size=8):  # 8 chords per analysis window
        self.window_size = window_size
        self.key_profiles = self._create_key_profiles()  # Krumhansl-Kessler
    
    def detect_modulations(self, chord_progression, timestamps):
        """
        Sliding window analysis to detect key changes
        
        Returns:
        - key_timeline: [(timestamp, key, confidence)]
        - modulation_points: [(timestamp, from_key, to_key, pivot_chord)]
        """
        modulations = []
        current_key = None
        
        for i in range(len(chord_progression) - self.window_size):
            window = chord_progression[i:i+self.window_size]
            key, confidence = self._estimate_key(window)
            
            # Detect key change
            if current_key and key != current_key and confidence > 0.7:
                pivot = self._find_pivot_chord(
                    chord_progression[i-2:i+2], 
                    current_key, 
                    key
                )
                modulations.append({
                    'timestamp': timestamps[i],
                    'from_key': current_key,
                    'to_key': key,
                    'pivot_chord': pivot,
                    'confidence': confidence
                })
            
            current_key = key
        
        return modulations
```

**Pivot chord analysis:**
```python
def _find_pivot_chord(self, transition_chords, from_key, to_key):
    """
    Find chord that functions in both keys (pivot chord)
    
    Example: Modulation from C major to G major
    - Am chord = vi in C major, ii in G major (good pivot)
    - Em chord = iii in C major, vi in G major (good pivot)
    """
    pivot_candidates = []
    
    for chord in transition_chords:
        from_function = self._analyze_function(chord, from_key)
        to_function = self._analyze_function(chord, to_key)
        
        # Chord functions in both keys = potential pivot
        if from_function and to_function:
            pivot_candidates.append({
                'chord': chord,
                'from_function': from_function,  # e.g., "vi"
                'to_function': to_function,      # e.g., "ii"
                'strength': self._pivot_strength(from_function, to_function)
            })
    
    return max(pivot_candidates, key=lambda x: x['strength']) if pivot_candidates else None
```

#### Phase 3.2: Tonicization Detection (2 days)

**Tonicization**: Temporary key center (not full modulation)

```python
def detect_tonicization(self, chord_progression):
    """
    Detect temporary key centers (tonicization)
    
    Pattern: Secondary dominant → target chord
    Example: D7 → G in key of C = V7/V → V (tonicizes G)
    """
    tonicizations = []
    
    for i in range(len(chord_progression) - 1):
        current = chord_progression[i]
        next_chord = chord_progression[i+1]
        
        # Check if current chord is dominant 7th
        if current.quality == 'dom7':
            # Check if it resolves up a 4th (dominant resolution)
            if (next_chord.root - current.root) % 12 == 5:
                tonicizations.append({
                    'position': i,
                    'secondary_dominant': current,
                    'target': next_chord,
                    'roman_numeral': self._to_secondary_dominant_notation(current, next_chord)
                })
    
    return tonicizations
```

**Test cases:**
- Bach chorales (frequent tonicizations)
- Beatles songs (secondary dominants)
- Jazz standards (ii-V-I in multiple keys)

---

## Priority 4: Advanced Rhythm Theory (MEDIUM IMPACT, HIGH EFFORT)

### Current State
- Basic meter detection (4/4, 3/4, 6/8)
- No complex meters
- No polyrhythm detection

### Enhancement Plan

#### Phase 4.1: Complex Meter Detection (3-4 days)
**Location**: `rhythmic_engine/heavy_rhythmic_analyzer.py`

**Add meters:**
- 5/4 (Take Five)
- 7/8 (Money by Pink Floyd)
- Mixed meters (changing time signatures)
- Compound meters (6/8, 9/8, 12/8)

```python
class ComplexMeterDetector:
    """
    Detects complex and changing time signatures
    """
    
    def detect_complex_meter(self, onset_times, tempo):
        """
        Analyze inter-onset intervals to detect meter
        
        Strategies:
        1. Auto-correlation of onset patterns
        2. Hierarchical beat structure analysis
        3. Accent pattern recognition
        """
        ioi = np.diff(onset_times)  # Inter-onset intervals
        
        # Test for common complex meters
        meters_to_test = [
            (5, 4),   # 5/4 time
            (7, 8),   # 7/8 time
            (7, 4),   # 7/4 time
            (11, 8),  # 11/8 time
            (9, 8),   # 9/8 time
            (12, 8),  # 12/8 time
        ]
        
        best_fit = None
        best_score = 0
        
        for numerator, denominator in meters_to_test:
            score = self._test_meter_fit(ioi, numerator, denominator, tempo)
            if score > best_score:
                best_score = score
                best_fit = (numerator, denominator)
        
        return best_fit, best_score
```

#### Phase 4.2: Polyrhythm Detection (4-5 days)

**Polyrhythm**: Multiple rhythmic layers with different patterns

```python
class PolyrhythmDetector:
    """
    Detects polyrhythms and polymeter
    Examples: 3 against 4, 5 against 4, African drumming
    """
    
    def detect_polyrhythm(self, onset_streams):
        """
        Analyze multiple rhythmic streams simultaneously
        
        Input: List of onset streams (one per instrument/voice)
        Output: Polyrhythmic relationships
        
        Example: Drums in 4/4, Piano in 3/4 = polymeter
        """
        ratios = []
        
        for stream1, stream2 in combinations(onset_streams, 2):
            ratio = self._calculate_rhythmic_ratio(stream1, stream2)
            if ratio and self._is_polyrhythm(ratio):
                ratios.append({
                    'stream1': stream1,
                    'stream2': stream2,
                    'ratio': ratio,  # e.g., "3:4"
                    'confidence': self._polyrhythm_confidence(stream1, stream2)
                })
        
        return ratios
```

**Test cases:**
- Dave Brubeck "Take Five" (5/4)
- Pink Floyd "Money" (7/8)
- Tool "Schism" (complex changing meters)
- African drumming patterns (polyrhythms)

---

## Priority 5: Counterpoint Analysis (LOW IMPACT, VERY HIGH EFFORT)

### Current State
- No polyphonic voice analysis
- No fugue/canon detection

### Enhancement Plan

#### Phase 5.1: Voice Separation (5-7 days)
**Location**: New file `hybrid_training/counterpoint_analyzer.py`

**Challenge**: Separate multiple melodic lines from polyphonic audio

**Approach**: Source separation + melodic tracking

```python
class CounterpointAnalyzer:
    """
    Analyzes contrapuntal music (multiple independent melodic lines)
    Based on Fux (1725) "Gradus ad Parnassum"
    """
    
    def __init__(self):
        self.source_separator = self._init_source_separation()  # Spleeter or Demucs
        self.melodic_tracker = MelodicLineTracker()
    
    def analyze_counterpoint(self, audio_file):
        """
        1. Separate audio into stems
        2. Track melodic lines
        3. Analyze contrapuntal relationships
        """
        # Separate sources
        stems = self.source_separator.separate(audio_file)
        
        # Track melodic lines
        melodic_lines = []
        for stem in stems:
            melody = self.melodic_tracker.extract_melody(stem)
            melodic_lines.append(melody)
        
        # Analyze counterpoint
        analysis = {
            'num_voices': len(melodic_lines),
            'species': self._detect_species(melodic_lines),
            'intervals': self._analyze_intervals(melodic_lines),
            'independence': self._measure_voice_independence(melodic_lines),
            'imitation': self._detect_imitation(melodic_lines)
        }
        
        return analysis
```

#### Phase 5.2: Fugue Detection (3-4 days)

**Fugue characteristics:**
- Subject (main theme)
- Answer (transposed subject)
- Exposition (all voices enter with subject)
- Episodes (developmental sections)

```python
def detect_fugue(self, melodic_lines):
    """
    Detect fugal structure
    
    1. Find subject (recurring melodic pattern)
    2. Find answer (transposed subject, usually up a 5th)
    3. Track subject entries across voices
    4. Identify exposition, episodes, stretto
    """
    subject = self._find_subject(melodic_lines[0])  # Usually in first voice
    
    if not subject:
        return None
    
    # Find answer (subject transposed by perfect 5th or 4th)
    answer = self._find_answer(melodic_lines, subject)
    
    # Track all subject/answer entries
    entries = self._track_entries(melodic_lines, subject, answer)
    
    if len(entries) >= len(melodic_lines):  # All voices have subject
        return {
            'type': 'fugue',
            'subject': subject,
            'answer': answer,
            'entries': entries,
            'exposition_length': self._calculate_exposition(entries),
            'episodes': self._find_episodes(entries)
        }
    
    return None
```

**Test cases:**
- Bach fugues (The Well-Tempered Clavier)
- Bach inventions (2-voice counterpoint)
- Palestrina masses (Renaissance counterpoint)

---

## Implementation Timeline

### Sprint 1: Extended Chords (1 week)
- ✅ Add 13 common jazz chord types
- ✅ Test on jazz recordings
- ✅ Update documentation

### Sprint 2: Voice Leading Basics (1 week)
- ✅ Voice assignment and tracking
- ✅ Smoothness calculation
- ✅ Parallel motion detection
- ✅ Test on Bach chorales

### Sprint 3: Bass Line Analysis (3-4 days)
- ✅ Inversion detection
- ✅ Bass contour analysis
- ✅ Pedal point detection

### Sprint 4: Modulation Detection (1 week)
- ✅ Sliding window key analysis
- ✅ Pivot chord detection
- ✅ Tonicization detection
- ✅ Test on classical/Beatles repertoire

### Sprint 5: Complex Rhythm (1 week)
- ✅ Complex meter detection (5/4, 7/8, etc.)
- ✅ Accent pattern analysis
- ✅ Test on prog rock, jazz

### Sprint 6: Polyrhythm (1 week)
- ✅ Multi-stream rhythm analysis
- ✅ Ratio detection
- ✅ Test on African music, Tool

### Sprint 7: Counterpoint (2-3 weeks)
- ✅ Voice separation
- ✅ Melodic line tracking
- ✅ Species counterpoint analysis
- ✅ Fugue detection
- ✅ Test on Bach

**Total estimated time: 8-10 weeks**

---

## Testing Strategy

### Unit Tests
```python
# test_extended_chords.py
def test_sus4_detection():
    chroma = create_sus4_chroma([0, 5, 7])  # C-F-G
    chord = detector.detect_chord(chroma)
    assert chord.quality == 'sus4'

def test_altered_dominant():
    chroma = create_altered_chroma([0, 4, 6, 10, 1])  # C-E-Gb-Bb-Db
    chord = detector.detect_chord(chroma)
    assert chord.quality == '7alt'
```

### Integration Tests
```python
# test_voice_leading.py
def test_bach_chorale():
    """Bach chorales should have smooth voice leading"""
    audio = load_audio("bach_chorale.wav")
    analysis = voice_analyzer.analyze(audio)
    
    assert analysis['smoothness'] < 2.0  # Mostly stepwise motion
    assert len(analysis['parallel_fifths']) == 0  # No violations
    assert len(analysis['parallel_octaves']) == 0
```

### Real-World Tests
1. **Jazz Standards**: Test chord detection on "Giant Steps", "All the Things You Are"
2. **Classical**: Test voice leading on Bach chorales, Mozart piano sonatas
3. **Contemporary**: Test on Beatles, Radiohead (complex chords, modulations)
4. **Experimental**: Test on Tool, Meshuggah (complex meters, polyrhythms)

---

## Prioritization Rationale

### Why Extended Chords First?
- **High impact**: Jazz music unplayable without sus, 9ths, altered chords
- **Medium effort**: Template matching already implemented
- **Immediate value**: Chandra_trainer will learn better chord progressions

### Why Voice Leading Second?
- **High impact**: Distinguishes good vs bad harmonic motion
- **Enables**: Better MIDI output from MusicHal_9000 (smoother chord transitions)
- **Research value**: Novel contribution to AI music systems

### Why Modulation Third?
- **Medium impact**: Important for long-form music analysis
- **Moderate effort**: Key detection already exists, extend to windowed analysis
- **Musical value**: Understand song structure better

### Why Complex Rhythm Fourth?
- **Medium impact**: Covers prog rock, jazz, world music
- **High effort**: Requires sophisticated beat tracking
- **Specialized**: Not critical for most popular music

### Why Counterpoint Last?
- **Low impact**: Applies to limited repertoire (Bach, classical)
- **Very high effort**: Requires source separation, melodic tracking
- **Research**: More academic than practical for performance system

---

## Success Metrics

### Chord Detection
- **Accuracy**: 90%+ on jazz standards (currently ~70%)
- **Coverage**: 17+ chord types (currently 4)
- **Test corpus**: 100 jazz/pop songs

### Voice Leading
- **Bach chorales**: 0 false positives for parallel 5ths/octaves
- **Smoothness**: Correlation >0.8 with expert analysis
- **Test corpus**: 50 Bach chorales, 50 pop songs

### Modulation
- **Detection**: 85%+ accuracy on known modulations
- **False positives**: <10% (avoid over-detection)
- **Test corpus**: Beatles catalog, Mozart sonatas

### Complex Rhythm
- **Meter**: 90%+ accuracy on 5/4, 7/8, 9/8
- **Test corpus**: Prog rock (Tool, Dream Theater), jazz (Brubeck)

### Counterpoint
- **Voice separation**: 80%+ F1 score
- **Fugue detection**: 85%+ precision on Bach fugues
- **Test corpus**: Bach Well-Tempered Clavier

---

## Dependencies and Prerequisites

### New Python Packages
```python
# For source separation (counterpoint)
pip install spleeter  # or demucs

# For advanced signal processing
pip install essentia  # Music analysis library

# For music theory utilities
pip install music21  # Symbolic music analysis
```

### New Data Structures
```python
@dataclass
class ExtendedChord:
    root: int
    quality: str  # 'maj7', 'min9', '7alt', etc.
    notes: List[int]
    confidence: float
    inversion: str  # 'root', 'first', 'second', 'slash'
    voicing: List[int]  # Actual frequencies
    extensions: List[str]  # ['9th', '13th', etc.]

@dataclass
class VoiceLeading:
    voices: Dict[str, List[int]]  # soprano, alto, tenor, bass
    motion: Dict[str, int]  # interval motion per voice
    smoothness: float
    parallel_fifths: List[Tuple]
    parallel_octaves: List[Tuple]
    contrary_motion: int
```

---

## Documentation Updates

Each enhancement phase should include:
1. **Code documentation**: Docstrings, type hints, examples
2. **Theory documentation**: Update `music_theory_foundation.md`
3. **Testing documentation**: Test cases, expected results
4. **User guide**: How to interpret new analysis results
5. **Research notes**: Academic references, algorithms used

---

## Questions for Discussion

1. **Priority order**: Agree with prioritization? Any changes?
2. **Scope**: All enhancements or focus on top 2-3?
3. **Timeline**: 8-10 weeks realistic? Need to accelerate?
4. **Resources**: Need additional libraries/datasets/expertise?
5. **Integration**: How to integrate into existing Chandra_trainer workflow?
6. **Testing**: What test corpus should we prioritize?

---

**Next Steps**: 
1. Review and approve plan
2. Set up Git branch: `music-theory-enhancements`
3. Begin Sprint 1: Extended Chord Vocabulary
4. Create test corpus of jazz recordings
5. Implement first 5 chord types and test

---

This plan transforms CCM3 from a basic music theory system to a comprehensive analytical framework approaching professional-level music analysis software.


