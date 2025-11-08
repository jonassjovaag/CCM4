# Harmonic Generation Architecture Analysis
**Date:** 2025-01-30  
**Investigation:** Complete data flow trace of 768D perceptual features → MIDI output

---

## Executive Summary

**CRITICAL FINDING**: The system has **NO harmonic translation layer** between perceptual understanding (768D Wav2Vec) and MIDI output. Current mechanism is **memory retrieval** (playback of stored training MIDI) with **NO harmonic filtering/transformation**.

### What Works ✅
- **Perceptual Understanding**: Wav2Vec 768D fingerprints correctly identify timbre/texture
- **Harmonic Analysis**: RealtimeHarmonicDetector + Brandtsegg ratios understand chord/consonance
- **Musical Decision**: Behavioral modes (SHADOW/MIRROR/COUPLE) decide WHAT to do
- **Rhythm Intelligence**: RhythmOracle learned patterns, correlation engine

### What's Missing ❌
- **Harmonic Translation**: No component converts "respond with consonance 0.85" → specific MIDI notes
- **Scale Constraints on Retrieval**: `_apply_scale_constraint()` only works on fallback algorithmic generation
- **Harmonic Context Usage**: `current_chord`, `scale_degrees` stored but NOT used to filter retrieved MIDI

---

## Architecture Flow (Current State)

### 1. INPUT ANALYSIS (Works Correctly)

```
Audio Input 
  ├─→ Wav2Vec Encoder → 768D embedding → Gesture Token (perceptual fingerprint)
  ├─→ Brandtsegg Ratio Analyzer → Consonance, Frequency Ratios
  ├─→ RealtimeHarmonicDetector → Current Chord, Key, Scale Degrees
  └─→ RhythmOracle → Rhythm Patterns
```

**Status**: ✅ All components functional and tested

### 2. MUSICAL DECISION (Works Correctly)

```
Behavioral Mode (SHADOW/MIRROR/COUPLE)
  ├─→ _build_shadowing_request() → gesture_token match, consonance target
  ├─→ _build_mirroring_request() → inverted consonance
  ├─→ _build_coupling_request() → hybrid parameters
  └─→ Request Parameters: {gesture_token, consonance, rhythmic_phrasing, weight}
```

**Status**: ✅ Request masking parameters correctly assembled

### 3. MEMORY RETRIEVAL (Works, But No Harmonic Filtering)

```
AudioOracle.generate_with_request(request, current_context)
  ├─→ Filter frames by: gesture_token, consonance, rhythmic_phrasing
  ├─→ Apply temperature sampling
  ├─→ Return frame IDs [342, 891, 1205, ...]
  └─→ phrase_generator extracts MIDI from audio_frames['midi']
```

**Status**: ⚠️ Filtering works BUT retrieved MIDI notes are scattered

**Evidence from test_predictor_comparison.py:**
```
Token 27 (most frequent): 469 frames
  └─→ Maps to 17 different MIDI notes (range 29-79)
  └─→ Variance: 132.66 (out of MIDI range 0-127)

Consonance bin 0.7-0.8: 847 frames  
  └─→ Maps to 50+ different MIDI notes
  └─→ Variance: 179.20

Result: ❌ Neither gesture_token NOR consonance predict MIDI notes
```

### 4. MIDI OUTPUT (No Harmonic Translation)

**Current behavior (phrase_generator.py lines 1120-1200):**

```python
if oracle_notes and len(oracle_notes) > 0:
    # Use learned patterns!
    phrase_length = min(len(oracle_notes), phrase_length)
    notes = oracle_notes[:phrase_length]  # ← RAW MIDI, NO FILTERING
    
    # Generate timing/velocity/duration
    # ...
    
    return MusicalPhrase(notes=notes, ...)  # ← OUTPUT AS-IS
```

**NO application of:**
- ❌ `_apply_scale_constraint()` to snap to `scale_degrees`
- ❌ `current_chord` to constrain pitch classes
- ❌ `current_key` to filter accidentals
- ❌ Harmonic transposition to match input context

**Scale constraints ONLY used in fallback** (lines 1240-1290):

```python
# Fallback algorithmic generation (user reports: "random and shit")
note = random.randint(min_note, max_note)
# ...
note = self._apply_scale_constraint(note)  # ← ONLY HERE
```

---

## Quantitative Evidence

### Test 1: Fast Translation Analysis
**File:** `tests/test_fast_translation_analysis.py`  
**Model:** Itzama.json (5000 states, 60 gesture tokens)

**Findings:**
- Gesture Token 27 (most frequent):
  - 469 frames (9.4% of all data)
  - 17 unique MIDI notes
  - Range: 29-79 (4+ octaves)
  - Variance: 132.66
  
- Mapping consistency:
  - Single MIDI note: 3/60 tokens (5.0%)
  - Low variance (<5): 5/60 tokens (8.3%)
  - **92% of tokens have high MIDI variance**

### Test 2: Predictor Comparison
**File:** `tests/test_predictor_comparison.py`

**Gesture Token as Predictor:**
- Mean variance: 162.29
- Median variance: 132.05
- Interpretation: Token 27 could be C4, G5, or Bb2 equally likely

**Consonance as Predictor:**
- Mean variance: 158.01
- Median variance: 179.20
- Pearson correlation (consonance ↔ MIDI): 0.164
- Interpretation: High consonance (0.9) does NOT predict specific pitch

**Variance Reduction:**
- Consonance improves gesture_token by only 2.6%
- **Conclusion**: ⚠️ NEITHER IS GOOD PREDICTOR

---

## Why This Happens (Root Cause Analysis)

### 1. Wav2Vec Was Never Meant for Pitch
**Design purpose**: Phoneme recognition in speech (timbre/texture, NOT pitch)

**Evidence:**
- Token 27 maps to MIDI 29-79 because it represents a **perceptual quality** (e.g., "bright sustained sound")
- Same perceptual quality can occur at ANY pitch
- **This is CORRECT behavior** - Wav2Vec is working as designed

### 2. Consonance Measures Intervals, Not Absolute Pitch
**Brandtsegg ratios**: Measure harmonic relationships between frequencies

**Evidence:**
- C major (C-E-G) has consonance ~0.92
- G major (G-B-D) has consonance ~0.92
- **Same consonance, different MIDI notes**
- Consonance is pitch-class-set-invariant

### 3. AudioOracle Is a Perceptual Memory, Not Harmonic Memory
**Factor Oracle design**: Learn sequential patterns in feature space

**Current usage:**
- State transitions based on 15D features (includes gesture_token, consonance)
- MIDI is stored alongside features but NOT learned
- Retrieval: "Find moments that sounded similar perceptually"
- **NOT**: "Find moments with similar harmonic function"

---

## Existing Components (Underutilized)

### 1. RealtimeHarmonicDetector (harmonic_context.py)
**Capabilities:**
- Chroma-based chord detection
- Major/minor/7th chord templates
- Key signature estimation
- Temporal smoothing

**Current usage:**
```python
# phrase_generator.py line 991
self.current_chord = harmonic_context.get('current_chord', 'C')
self.current_key = key_name.split('_')[0]
```

**Usage in generation:** ❌ STORED BUT NOT USED

### 2. Scale Constraint System
**Capabilities:**
- `_apply_scale_constraint(note, scale_degrees)` - snap to diatonic notes
- `scale_degrees` calculated from `current_key` + mode (major/minor/dorian)
- Wrapping logic for octave boundaries

**Current usage:** ✅ Works in fallback generation  
**Missing usage:** ❌ NOT applied to AudioOracle retrieved notes

### 3. Phrase Memory System
**Capabilities:**
- Stores 20 motifs with variations
- Thematic recall every 30-60s
- Transpose/invert/augment/diminish transformations

**Current usage:** ✅ Functional  
**Harmonic awareness:** ⚠️ Transformations are INTERVAL-BASED, not scale-aware

---

## Design Gap: The Missing Translation Layer

### Current Pipeline
```
Decision: "Respond with high consonance, similar gesture to Token 27"
    ↓
AudioOracle: "Here are 5 frames matching Token 27 + consonance >0.8"
    ↓
Frames: [ID 342, 891, 1205, 1876, 2431]
    ↓
MIDI Extraction: [67, 29, 74, 51, 62]  ← SCATTERED ACROSS RANGE
    ↓
Output: [67, 29, 74, 51, 62]  ← NO FILTERING/TRANSFORMATION
```

**Result:** User hears "random and shit" because:
- MIDI jumps 4 octaves (29 → 74)
- No harmonic relationship to input chord
- No scale coherence

### Required Translation Layer

**Option 1: Post-Retrieval Filtering + Transformation**
```
AudioOracle retrieves MIDI: [67, 29, 74, 51, 62]
    ↓
HarmonicTranslator:
  - Input context: current_chord="Dm", current_key="F_major"
  - Filter: Keep only notes in F major scale
  - Transpose: Shift outliers to match chord tones
  - Constrain: Apply _apply_scale_constraint() to each note
    ↓
Output: [62, 65, 67, 60, 62]  ← COHERENT MELODY IN F MAJOR
```

**Option 2: Harmonic-Aware Retrieval**
```
Decision: "Respond with consonance 0.85, mid register, gesture Token 27"
    ↓
HarmonicGenerator:
  - Uses current_chord, scale_degrees, consonance target
  - Queries AudioOracle for INTERVALS not absolute MIDI
  - Constructs melody using retrieved intervals + current harmonic context
  - Applies scale constraints in real-time
    ↓
Output: Melody in correct key/mode with learned gesture quality
```

**Option 3: Hybrid (Most Promising)**
```
1. AudioOracle retrieves frames matching perceptual constraints
2. Extract INTERVALS from retrieved MIDI (melodic motion, not absolute pitch)
3. Apply intervals to current_chord root note
4. Constrain with _apply_scale_constraint(note, scale_degrees)
5. Verify consonance of resulting notes against current_chord
6. Output harmonically coherent melody with learned gesture quality
```

---

## Recommendations

### Immediate Actions

1. **Test Interval-Based Retrieval** (Quick Validation)
   - Modify MIDI extraction to store intervals instead of absolute notes
   - Test: Does Token 27 have consistent INTERVAL patterns despite scattered MIDI?
   - Expected: Intervals might be coherent (e.g., +2, -1, +4) even when MIDI varies

2. **Enable Scale Constraints on Retrieved Notes** (Simple Fix)
   - Add `_apply_scale_constraint()` call after oracle_notes extraction (line 1140)
   - Use `self.scale_degrees` (already calculated from harmonic_context)
   - Test: Does this improve coherence without destroying gesture quality?

3. **Measure Harmonic Coherence** (Validation Metric)
   - Create test: Input C major audio → measure output consonance
   - Expected: If input consonance=0.92, output should be >0.7 (currently likely <0.4)
   - Quantify improvement after translation layer added

### Design Phase (New Component)

4. **Build HarmonicTranslationLayer** (New Class)
   ```python
   class HarmonicTranslationLayer:
       def translate(self, 
                    retrieved_midi: List[int],
                    harmonic_context: Dict,
                    request: Dict) -> List[int]:
           """
           Translate retrieved MIDI notes to harmonic context
           
           Args:
               retrieved_midi: Raw MIDI from AudioOracle
               harmonic_context: {current_chord, current_key, scale_degrees}
               request: {consonance_target, register, gesture_token}
               
           Returns:
               Harmonically coherent MIDI notes
           """
           # Option A: Extract intervals, apply to chord root
           # Option B: Filter by scale, transpose outliers
           # Option C: Hybrid approach
           pass
   ```

5. **Integration Points**
   - Insert between AudioOracle retrieval and MusicalPhrase construction
   - Use existing `harmonic_context` (already passed to generate_phrase)
   - Preserve rhythmic_phrasing from RhythmOracle (timing separate from pitch)

### Validation Phase

6. **Test with Synthetic Ground Truth**
   - Use existing 15 test audio files (C major, D minor, etc.)
   - Expected: C major input → C major output (or at least high consonance)
   - Measure: MIDI note accuracy, consonance correlation, scale adherence

7. **A/B Test with User**
   - Current system vs. translation layer
   - Subjective: "Does it feel less random and shit?"
   - Objective: Log consonance variance, scale violations

---

## File Locations (For Implementation)

### Core Generation
- `agent/phrase_generator.py` - Line 1120-1200 (AudioOracle note usage)
  - **TODO**: Add harmonic translation after line 1140
  
- `agent/phrase_generator.py` - Line 461-500 (`_apply_scale_constraint`)
  - **READY**: Can be called on retrieved notes immediately

### Harmonic Analysis
- `listener/harmonic_context.py` - RealtimeHarmonicDetector class
  - **AVAILABLE**: Already provides current_chord, current_key, scale_degrees
  
- `listener/ratio_analyzer.py` - Brandtsegg consonance calculation
  - **AVAILABLE**: Can validate output consonance

### Memory System
- `memory/polyphonic_audio_oracle.py` - Line 595-800 (generate_with_request)
  - **WORKING**: Request masking filters frames correctly
  - **NO CHANGES NEEDED**: Problem is in post-retrieval processing

### Testing Infrastructure
- `tests/test_fast_translation_analysis.py` - Analyzes existing model
- `tests/test_predictor_comparison.py` - Compares predictor variance
- `tests/test_audio/` - 15 synthetic samples with ground truth
  - **READY**: Use for validation testing

---

## Next Steps (Prioritized)

### Phase 1: Quick Validation (1-2 hours)
1. ✅ **DONE**: Trace existing code flow (this document)
2. **Test interval-based analysis**: Do gesture tokens have consistent intervals?
3. **Simple fix**: Add `_apply_scale_constraint()` to retrieved notes
4. **Measure**: Does consonance improve? Does gesture quality degrade?

### Phase 2: Design (2-4 hours)
5. **Choose approach**: Option 1 (filter), Option 2 (intervals), or Option 3 (hybrid)
6. **Create HarmonicTranslationLayer class** (new file: `agent/harmonic_translator.py`)
7. **Unit tests**: Test translation with known input/output

### Phase 3: Integration (2-3 hours)
8. **Modify phrase_generator.py**: Insert translation layer
9. **Preserve rhythmic intelligence**: Don't break RhythmOracle timing
10. **End-to-end test**: Full pipeline with synthetic audio

### Phase 4: Validation (1-2 hours)
11. **Ground truth test**: 15 synthetic samples → measure accuracy
12. **Live performance test**: User plays → subjective evaluation
13. **Log analysis**: Compare consonance variance before/after

**Total estimated time**: 6-11 hours of focused work

---

## Conclusion

The system's perceptual understanding (768D Wav2Vec) and harmonic analysis (RealtimeHarmonicDetector) work correctly. The problem is in the **translation layer** - specifically, the missing step that converts:

**Musical decision** ("respond with high consonance, similar gesture")  
→ **Specific MIDI notes** (in correct key/scale/register)

Current system does **memory playback** (retrieve whatever MIDI was stored during training). This produces scattered, incoherent output because:
- Gesture tokens are perceptual (timbre), not harmonic (pitch)
- Consonance is relative (intervals), not absolute (specific notes)
- No filtering/transformation applied to retrieved MIDI

**Solution exists in codebase but is disconnected:**
- ✅ `harmonic_context` provides current_chord, current_key, scale_degrees
- ✅ `_apply_scale_constraint()` can snap notes to scale
- ❌ These are NOT applied to AudioOracle retrieved notes

**Path forward**: Build HarmonicTranslationLayer that bridges perceptual memory → harmonic output using existing components.

---

**Status**: Analysis complete, ready for implementation phase.
