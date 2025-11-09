# Scientific Test Suite for MusicHal 9000
**Purpose:** Trace and verify complete data flow from audio input to MIDI output

## Data Flow Architecture

```
TRAINING PIPELINE:
1. Itzama.wav (audio file)
   ↓
2. PolyphonicAudioProcessor.extract_onset_events()
   → timestamps, f0 (Hz), amplitude, duration
   ↓
3. HybridDetector (Wav2Vec + Brandtsegg)
   → gesture_token (0-63), consonance (0-1), frequency_ratios
   ↓
4. Music Theory Transformer
   → chord_label ("major"/"minor"/etc for display), harmonic analysis
   ↓
5. Event merging: merge hierarchical segments with audio events
   → chord_name_display (for humans) = chord_label
   → gesture_token (for AI learning) = quantized Wav2Vec embedding
   ↓
6. AudioOracle.train()
   → Builds graph: states, transitions, suffix_links
   → Each state contains AudioFrame with audio_data dict
   ↓
7. Serialization
   → Saves as dict (not object!) to JSON/pickle
   → audio_frames[frame_id] = {audio_data: {pitch_hz, gesture_token, consonance, chord_name_display, ...}}

LIVE PERFORMANCE PIPELINE:
1. Microphone input
   ↓
2. DriftListener.process_audio()
   → onset detection, f0 extraction
   ↓
3. HybridDetector
   → gesture_token, consonance (same as training)
   ↓
4. MemoryBuffer (180s ring buffer)
   → Recent musical context
   ↓
5. AIAgent.decide()
   → Behavioral mode (imitate/contrast/lead)
   → Request formation: {gesture_token: X, consonance: Y, rhythm_ratio: Z}
   ↓
6. PhraseGenerator.generate_phrase()
   → Calls AudioOracle.generate_with_request(request)
   → Returns frame_ids matching request constraints
   ↓
7. Note Extraction (phrase_generator.py:1092)
   → if frame_id in audio_oracle.audio_frames:  # ← THE BUG FIX
   →   audio_data = frame.audio_data
   →   midi = audio_data['pitch_hz'] → frequency_to_midi() conversion
   ↓
8. MIDI Output
   → Sends note to IAC Driver
   ↓
9. Viewport Display (parallel)
   → Shows chord_name_display for human understanding
   → "major", "minor", etc.
```

## Test Suite Structure

### Layer 1: Training Data Validation
**File:** `tests/test_training_data_extraction.py`
- [ ] Verify onset extraction from Itzama.wav produces expected timestamps
- [ ] Confirm f0 (Hz) values are in musically plausible range (50-2000 Hz)
- [ ] Validate amplitude/duration are non-zero
- [ ] Test edge cases: silence, sustained notes, rapid onsets

### Layer 2: Feature Extraction Verification
**File:** `tests/test_feature_extraction.py`
- [ ] **Wav2Vec gesture tokens:**
  - Load Itzama.wav sample
  - Extract gesture tokens
  - Verify token range (0-63 for harmonic vocabulary)
  - Check token distribution (should not collapse to single value)
- [ ] **Brandtsegg ratio analysis:**
  - Test known chords (C major triad → consonance ~0.9)
  - Test dissonance (chromatic cluster → consonance ~0.2)
  - Verify frequency_ratios are sorted, normalized
- [ ] **Rhythm analysis:**
  - Beat tracking accuracy
  - Syncopation detection
  - rhythm_ratio calculation

### Layer 3: AudioOracle Training Verification
**File:** `tests/test_audiooracle_training.py`
- [ ] **Graph construction:**
  - Count states (should match number of training events)
  - Count transitions (forward links between states)
  - Count suffix_links (pattern repetitions)
  - Verify suffix_link distance threshold (0.15 Euclidean)
- [ ] **State content:**
  - Each state contains AudioFrame
  - AudioFrame.audio_data has required fields: pitch_hz, gesture_token, consonance
  - Optional fields: chord_name_display, dual_chroma, transformer_insights
- [ ] **Serialization integrity:**
  - Train oracle → serialize → deserialize → verify identical structure
  - Test both JSON and pickle formats
  - Verify no data loss in conversion

### Layer 4: Request Masking & Generation
**File:** `tests/test_request_masking.py`
- [ ] **Request formation:**
  - Given input features (gesture_token=42, consonance=0.8)
  - Verify request dict structure
  - Test multi-constraint requests (gesture + consonance + rhythm)
- [ ] **Oracle query:**
  - Load trained model
  - Generate with request: `oracle.generate_with_request(request)`
  - Verify returned frame_ids match constraints
  - Test temperature parameter (0.0 = deterministic, 1.0 = stochastic)
- [ ] **Constraint filtering:**
  - Request consonance > 0.7 → all returned frames should have consonance > 0.7
  - Request specific gesture_token → frames should match token
  - Test edge cases: no matching states, overly restrictive constraints

### Layer 5: Note Extraction & MIDI Conversion
**File:** `tests/test_note_extraction.py` (REFACTOR EXISTING)
- [x] Dictionary membership check: `frame_id in audio_frames` (FIXED)
- [x] MIDI field fallbacks: pitch_hz → f0 → midi → midi_note
- [ ] **Frequency to MIDI conversion accuracy:**
  - 440 Hz → MIDI 69 (A4)
  - 220 Hz → MIDI 57 (A3)
  - Test edge cases: very low/high frequencies
- [ ] **Velocity/duration extraction:**
  - amplitude → velocity mapping
  - duration field → note duration
- [ ] **Multi-note handling:**
  - Polyphonic frames (multiple simultaneous notes)
  - Sequential phrase generation

### Layer 6: Viewport Translation
**File:** `tests/test_viewport_translation.py` (REFACTOR EXISTING)
- [ ] **Chord name display:**
  - Training: chord_name_display set from Music Theory Transformer chord_label
  - Live: How does chord_name_display get populated? (NOT from AudioOracle!)
  - Test: Given gesture_token X, what chord name should display?
- [ ] **Dual vocabulary mapping:**
  - gesture_token (machine) ↔ chord_name (human)
  - Test harmonic vocabulary (0-63) maps to chord types
  - Test percussive vocabulary (0-63) maps to drum patterns
- [ ] **Ratio to chord name translation:**
  - `ratio_to_chord_name(frequency_ratios)` function
  - Test known intervals: [1.0, 1.25, 1.5] → "major triad"

### Layer 7: End-to-End Integration
**File:** `tests/test_end_to_end_integration.py` (REFACTOR EXISTING)
- [ ] **Simulate training:**
  - Load audio sample (10 seconds of Itzama.wav)
  - Run full training pipeline
  - Verify model structure
- [ ] **Simulate live performance:**
  - Load trained model (as dict)
  - Create mock audio input event
  - Extract features (gesture_token, consonance)
  - Form request
  - Query oracle
  - Extract notes
  - Verify MIDI output data structure
- [ ] **Compare training vs. live:**
  - Same gesture_token in training and live should match
  - Consonance values should be consistent
  - Request masking should work identically

### Layer 8: Model Inspection & Debugging
**File:** `tests/test_model_inspection.py` (REFACTOR EXISTING)
- [x] Load serialized model (dict format)
- [x] Count frames, states, transitions
- [ ] **Statistical analysis:**
  - gesture_token distribution (histogram)
  - consonance distribution (mean, std, range)
  - chord_name_display frequencies
- [ ] **Graph connectivity:**
  - Strongly connected components
  - Dead-end states (no outgoing transitions)
  - Suffix link coverage (% states with suffix links)
- [ ] **Feature validation:**
  - All frames have required fields
  - No NaN/Inf values
  - Gesture tokens in valid range

### Layer 9: Performance Benchmarking
**File:** `tests/test_performance_benchmarks.py` (NEW)
- [ ] **Latency measurements:**
  - Feature extraction time (<10ms target)
  - Oracle query time (<30ms target)
  - End-to-end latency (<50ms target)
- [ ] **Memory usage:**
  - Model loading (should fit in RAM)
  - Live performance memory footprint
- [ ] **GPU vs. CPU:**
  - MPS-accelerated oracle vs. CPU
  - Wav2Vec inference time

### Layer 10: Regression Testing
**File:** `tests/test_regressions.py` (NEW)
- [ ] **Known bugs:**
  - frame_id dictionary lookup (fixed, verify stays fixed)
  - Temporal smoothing chord flicker (verify no regression)
  - Gesture token collapse (verify distribution maintained)
- [ ] **Reference outputs:**
  - Save "golden" model from known-good training run
  - Verify new training produces similar structure
  - Alert on major deviations

## Test Data Requirements

### Training Audio Samples
- `test_audio/simple_major_chord.wav` - Sustained C major triad (440 Hz A + 523 Hz C + 659 Hz E)
- `test_audio/simple_minor_chord.wav` - Sustained A minor triad
- `test_audio/chromatic_cluster.wav` - Dissonant cluster (low consonance test)
- `test_audio/itzama_10sec.wav` - First 10 seconds of Itzama.wav (real training data subset)
- `test_audio/silence_1sec.wav` - Pure silence (edge case)
- `test_audio/rapid_onsets.wav` - Drum pattern (temporal smoothing test)

### Reference Models
- `test_models/minimal_model.pkl.gz` - 100-event trained model (fast loading)
- `test_models/itzama_full.pkl.gz` - Full Itzama model (5000 events) - use existing
- `test_models/golden_reference.pkl.gz` - Known-good baseline for regression testing

### Expected Outputs
- `test_expected/major_chord_features.json` - Expected gesture_token, consonance for C major
- `test_expected/itzama_10sec_stats.json` - Expected statistics from 10-second sample
- `test_expected/oracle_query_results.json` - Expected frame_ids for standard requests

## Execution Strategy

### Phase 1: Foundation (Current State)
- [x] test_model_inspection.py - Basic structure validation
- [x] test_note_extraction.py - Bug fix verification
- [ ] Refactor both to be scientific, not just "does it crash"

### Phase 2: Feature Pipeline
- [ ] test_training_data_extraction.py
- [ ] test_feature_extraction.py
- [ ] Create test audio samples

### Phase 3: Core Intelligence
- [ ] test_audiooracle_training.py
- [ ] test_request_masking.py
- [ ] Create reference models

### Phase 4: Output Layer
- [ ] Refactor test_viewport_translation.py
- [ ] test_end_to_end_integration.py refactor

### Phase 5: Quality Assurance
- [ ] test_performance_benchmarks.py
- [ ] test_regressions.py
- [ ] Document findings in scientific paper format

## Success Criteria

A **scientifically rigorous** test suite should:

1. ✅ **Trace every data transformation** from audio file to MIDI output
2. ✅ **Verify expected values** at each stage (not just "no crash")
3. ✅ **Isolate components** (test feature extraction independent of oracle)
4. ✅ **Use real data** (Itzama.wav samples) alongside synthetic tests
5. ✅ **Measure performance** (latency, memory, accuracy)
6. ✅ **Prevent regressions** (golden reference comparisons)
7. ✅ **Document findings** (what we learned about the system)

## Research Questions to Answer

1. **How many unique gesture tokens does Itzama.wav produce?**
   - Expected: 50-63 (most of vocabulary used)
   - Actual: TBD from test results

2. **What is the consonance distribution in training data?**
   - Expected: Bimodal (consonant vs. dissonant moments)
   - Actual: TBD

3. **How dense is the AudioOracle graph?**
   - States: 5000
   - Transitions: Expected ~7000-10000 (avg 1.5 per state)
   - Suffix links: Expected ~2500-3500 (50-70% of states)

4. **What is the mapping between gesture tokens and chord names?**
   - Is it consistent? One-to-many? Many-to-one?
   - Can we visualize this mapping?

5. **Why is chord_name_display 'unknown' in some frames?**
   - Music Theory Transformer failed to analyze?
   - Edge case handling?

6. **Does request masking actually constrain generation?**
   - Request consonance > 0.8 → verify all returned frames meet this
   - Measure constraint satisfaction rate

7. **What is the end-to-end latency distribution?**
   - Mean, median, 95th percentile, max
   - Identify bottlenecks

## Documentation Output

After completing test suite, produce:

1. **Test Results Report** (`tests/RESULTS.md`)
   - Summary statistics from all tests
   - Performance metrics
   - Regression status

2. **Data Flow Diagram** (`tests/data_flow_verified.svg`)
   - Visual representation of tested pipeline
   - Color-coded: green = tested, yellow = partial, red = untested

3. **Scientific Findings** (`tests/FINDINGS.md`)
   - Answers to research questions
   - Unexpected discoveries
   - Recommendations for system improvements

4. **Integration Guide** (`tests/INTEGRATION.md`)
   - How to run tests in CI/CD
   - Expected test duration
   - Dependencies and setup

---

**Next Step:** Start with Phase 1 refactoring. Make test_model_inspection.py and test_note_extraction.py scientifically rigorous with statistical analysis and documented expectations.
