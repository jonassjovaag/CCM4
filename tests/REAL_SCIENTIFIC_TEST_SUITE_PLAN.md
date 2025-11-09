# Real Scientific Test Suite - Double-Blind Ground Truth Validation

## What's Different From Previous "Tests"

**Previous tests (Phase 1-2):**
- Analyzed static serialized model files
- Computed statistics on existing data
- Validated utilities work without crashing
- **No actual audio → MIDI pipeline testing**

**This test suite:**
- **Ground truth input**: Known audio with documented properties
- **Measured output**: Actual system response at each pipeline stage
- **Quantified delta**: Exact difference between expected and actual
- **Iterative improvement**: Adjust parameters, retest, measure improvement

---

## Test Audio Specification

### Option 1: Use Existing `generate_synthetic_chord_dataset.py`

**Modify to create test sequence:**
```python
# Ground truth test sequence (8 seconds):
# 0.0-2.0s: C major (C4-E4-G4)
# 2.0-4.0s: D minor (D4-F4-A4)  
# 4.0-8.0s: G major lydian phrase (G4-B4-D5-F#5-A5)
```

**Known properties:**
- Exact MIDI notes at each timestamp
- Expected consonance (major/minor ≈0.85-0.95, lydian ≈0.90)
- Onset times (controlled)
- NO gesture tokens (can't predict Wav2Vec output without running it)

### Option 2: Record Real Performance + Manual Labeling

User provides:
- `test_phrase.wav` (actual played C→Dm→G Lydian)
- `test_phrase_labels.json`:
```json
{
  "segments": [
    {"start": 0.0, "end": 2.1, "chord": "C major", "notes": [60, 64, 67], "expected_consonance": 0.90},
    {"start": 2.1, "end": 4.3, "chord": "D minor", "notes": [62, 65, 69], "expected_consonance": 0.85},
    {"start": 4.3, "end": 8.0, "chord": "G Lydian phrase", "notes": [67, 71, 74, 78, 81], "expected_consonance": 0.88}
  ]
}
```

**Advantages**: Real audio complexity, real timbral variations  
**Disadvantages**: Manual labeling required, subjective boundaries

---

## Pipeline Test Structure

### Test 1: Feature Extraction Accuracy
**File**: `tests/test_audio_feature_extraction.py`

**Input**: `test_phrase.wav` (or synthetic)

**Process**:
1. Load audio through `listener/jhs_listener_core.py`
2. Run onset detection
3. Extract features (Wav2Vec, Brandtsegg ratios, MIDI)

**Measurements**:
- **Onset timing delta**: `|detected_onset_time - expected_onset_time|` (tolerance: ±50ms)
- **MIDI note accuracy**: `% of detected notes matching ground truth`
- **Consonance correlation**: `correlation(detected_consonance, expected_consonance)`
- **Gesture token stability**: Do consecutive frames in same chord produce similar tokens?

**Pass criteria** (to be determined through iteration):
- Onset timing: 90% within ±50ms
- MIDI accuracy: 80% exact match (complex chords have ambiguity)
- Consonance correlation: r > 0.7
- Gesture token variance within chord: σ < threshold (TBD)

**Output**: `feature_extraction_results.json` with per-frame deltas

---

### Test 2: Training Pipeline Validation
**File**: `tests/test_audio_training_pipeline.py`

**Input**: `test_phrase.wav` + ground truth labels

**Process**:
1. Run `Chandra_trainer.py --file test_phrase.wav --output test_model.json`
2. Load resulting model
3. Analyze structure

**Measurements**:
- **State count**: Are there distinct states for C major, D minor, G Lydian regions?
- **Feature clustering**: Do frames from same chord cluster together (DBSCAN or visual inspection)?
- **Transition sequence**: Does state graph follow C→D→G progression?
- **Suffix link quality**: Do suffix links connect states with similar consonance/harmony?

**Pass criteria** (iterative):
- Minimum 3 distinct harmonic clusters detected
- Transitions reflect input sequence order (some tolerance for repetitions)
- Suffix link distance threshold < 0.15 connects genuinely similar patterns

**Output**: `training_pipeline_results.json` with cluster analysis, transition graph

---

### Test 3: Live Generation Response
**File**: `tests/test_audio_live_generation.py`

**Input**: Trained model from Test 2 + simulated audio frames from `test_phrase.wav`

**Process**:
1. Load model into `agent/phrase_generator.py`
2. Feed audio frames sequentially (simulate live input)
3. Capture generated MIDI output
4. Measure request masking effectiveness

**Measurements**:
- **Response latency**: Time from audio frame → MIDI output (target <50ms)
- **Harmonic coherence**: Does output consonance track input consonance?
- **Request masking**: When we set `request['consonance']=0.9`, does output match?
- **Behavioral mode consistency**: Do modes persist 30-90s as designed?

**Pass criteria** (iterative):
- Latency: 95% of frames <50ms
- Consonance correlation with input: r > 0.5 (allows for contrast behavior)
- Request masking: When constrained, 70% of outputs satisfy constraint
- Mode duration: Measured durations within 30-90s range

**Output**: `live_generation_results.json` with frame-by-frame latency, coherence metrics

---

### Test 4: Viewport Translation Accuracy
**File**: `tests/test_audio_viewport_translation.py`

**Input**: Audio frames with known harmony (C major, D minor, G Lydian)

**Process**:
1. Extract features from test audio
2. Pass through chord name mapping (Music Theory Transformer or ratio analyzer)
3. Compare to ground truth labels

**Measurements**:
- **Precision**: Of labeled chords, % correct
- **Recall**: Of ground truth chords, % detected (vs "---" ambiguous)
- **Confusion matrix**: Which chords get misidentified as what?

**Example**:
- C major frames → "C major": 85% correct, "---": 10%, "Csus4": 5%
- D minor frames → "D minor": 70%, "---": 25%, "Dm7": 5%
- G Lydian → "G major": 60%, "---": 30%, "Gmaj7": 10%

**Pass criteria** (iterative):
- Precision for major/minor triads: >70%
- Recall (not "---"): >60%
- No catastrophic failures (dim7 for major chord)

**Output**: `viewport_translation_results.json` with confusion matrix, per-chord accuracy

---

### Test 5: End-to-End Integration
**File**: `tests/test_audio_end_to_end.py`

**Input**: `test_phrase.wav` (fresh, no prior training on it)

**Process**:
1. Train model from audio
2. Simulate live performance with same audio
3. Measure complete pipeline

**Measurements**:
- All metrics from Tests 1-4 combined
- **Musical coherence**: Subjective evaluation guide ("Does output respond sensibly?")
- **System stability**: No crashes, no memory leaks, consistent performance

**Output**: `end_to_end_results.json` + human evaluation form

---

## Iteration Framework

### Workflow:
```bash
# 1. Generate/label test audio
python tests/generate_test_audio.py --output tests/test_audio/test_phrase.wav

# 2. Run all tests
python tests/run_audio_tests.py

# 3. Review results
cat tests/test_output/audio_test_summary.json

# 4. Adjust parameters (e.g., distance threshold, consonance mapping)
# Edit config or production code

# 5. Rerun tests
python tests/run_audio_tests.py

# 6. Compare results
python tests/compare_test_runs.py --baseline run_001 --current run_002
```

### Version Control:
- Each test run saved with timestamp: `run_20251108_143052/`
- Results JSON includes all parameters used
- Git commit before parameter changes
- Track improvement over iterations

---

## What We're Actually Measuring

### NOT:
- "Does it work?" (yes/no)
- "Is the code correct?" (assumes we know what correct is)

### YES:
- **Accuracy**: How close is output to known input? (quantified)
- **Consistency**: Do repeated tests give same results? (variance)
- **Sensitivity**: If we change input, does output change appropriately?
- **Boundary behavior**: What happens with edge cases (silence, dissonance, ambiguous chords)?

---

## Honest Limitations

### What this CANNOT test:
- **Musical quality** (subjective, requires human evaluation)
- **Live performance variability** (human timing, dynamics, timbre changes)
- **Long-term coherence** (15-minute performance arcs)
- **Creativity/surprise** (can't have ground truth for emergent behavior)

### What this CAN test:
- **Technical accuracy** (MIDI notes, timing, consonance values)
- **Algorithmic behavior** (distance thresholds, state transitions, request masking)
- **Regression prevention** (ensure changes don't break working features)
- **Parameter tuning** (find optimal thresholds through iteration)

---

## Implementation Priority

1. ✅ **Generate test audio** (synthetic C→Dm→G with known properties)
2. **Test 1: Feature extraction** (does listener pipeline detect what we expect?)
3. **Test 2: Training pipeline** (does model structure reflect input?)
4. **Test 4: Viewport translation** (can we name chords correctly?)
5. **Test 3: Live generation** (does it respond coherently?)
6. **Test 5: End-to-end** (full integration check)

### Rationale:
Start with feature extraction (foundation), then training (model quality), then translation (debugging aid), then generation (most complex). End-to-end comes last as integration verification.

---

## Next Steps

**User decision needed:**
1. Use synthetic audio (clean, controlled) or real recording (complex, labeled)?
2. What tolerance for "success"? (80% accuracy? 90%? 70%?)
3. Which test to implement first?

**Agent will:**
1. Build test audio generator (or label existing recording)
2. Implement Test 1 (feature extraction)
3. Run, measure delta, document results
4. Iterate based on findings

This is actual scientific testing: falsifiable hypotheses, quantified measurements, documented iteration.
