# Dual Vocabulary Implementation - Step 1: DualPerceptionModule Updates

## ✅ Completed Changes

### 1. DualPerceptionResult Dataclass Enhancement

**File**: `listener/dual_perception.py` (lines 57-93)

**Changes**:
- Added `harmonic_token: Optional[int]` field for harmonic vocabulary tokens
- Added `percussive_token: Optional[int]` field for percussive vocabulary tokens
- Marked legacy `gesture_token` field with comment
- Updated `to_dict()` method to include both new token fields
- Fixed type annotations to use `Optional[]` for nullable fields
- Added `__post_init__()` to handle default values for mutable fields

**Purpose**: Enable storing both harmonic and percussive gesture tokens per audio event, allowing the system to query appropriate vocabulary based on detected content type.

### 2. DualPerceptionModule Constructor Updates

**File**: `listener/dual_perception.py` (lines 122-182)

**Changes**:
- Added `enable_dual_vocabulary: bool = False` parameter
- Added conditional initialization:
  - **Dual mode**: Creates `harmonic_quantizer` and `percussive_quantizer` (64 tokens each)
  - **Traditional mode**: Creates single `quantizer` (backward compatible)
- Added `self.enable_dual_vocabulary` flag for runtime checking
- Updated initialization logging to show dual vocabulary status

**Architecture**:
```python
if enable_dual_vocabulary:
    # Dual mode for percussion-aware listening
    self.harmonic_quantizer = SymbolicQuantizer(64)
    self.percussive_quantizer = SymbolicQuantizer(64)
    self.quantizer = None  # Not used
else:
    # Traditional mode (backward compatible)
    self.quantizer = SymbolicQuantizer(64)
    self.harmonic_quantizer = None
    self.percussive_quantizer = None
```

### 3. Content Detection Method

**File**: `listener/dual_perception.py` (lines 283-330)

**New Method**: `detect_content_type(audio, sr) -> Tuple[str, float, float]`

**Implementation**:
```python
# Apply HPSS to separate harmonic and percussive components
y_harmonic, y_percussive = librosa.effects.hpss(
    audio,
    kernel_size=31,  # Larger = better separation
    power=2.0,
    mask=True
)

# Calculate energy ratios
harmonic_energy = np.sum(y_harmonic ** 2)
percussive_energy = np.sum(y_percussive ** 2)
total_energy = harmonic_energy + percussive_energy

harmonic_ratio = harmonic_energy / total_energy
percussive_ratio = percussive_energy / total_energy

# Classify (threshold: 0.7 = 70% energy in one component)
if percussive_ratio > 0.7:
    return "percussive", harmonic_ratio, percussive_ratio
elif harmonic_ratio > 0.7:
    return "harmonic", harmonic_ratio, percussive_ratio
else:
    return "hybrid", harmonic_ratio, percussive_ratio
```

**Returns**:
- `content_type`: "harmonic", "percussive", or "hybrid"
- `harmonic_ratio`: 0.0-1.0 (proportion of harmonic energy)
- `percussive_ratio`: 0.0-1.0 (proportion of percussive energy)

**Use Case**:
```python
# In MusicHal_9000.py
content_type, harm_ratio, perc_ratio = self.dual_perception.detect_content_type(audio, sr)

if content_type == "percussive":
    # User playing drums → AI responds with harmony
    request = {'percussive_token': perc_tok, 'response_mode': 'harmonic'}
    print(f"🥁 Drums detected ({perc_ratio:.1%}) → generating harmony")
```

### 4. Training Method Updates

**File**: `listener/dual_perception.py` (lines 332-378)

**Method**: `train_gesture_vocabulary(features_list, vocabulary_type="single")`

**Changes**:
- Added `vocabulary_type` parameter: "single", "harmonic", or "percussive"
- Selects appropriate quantizer based on vocabulary type
- Maintains backward compatibility (default "single" uses legacy quantizer)

**Usage**:
```python
# Traditional training
perception.train_gesture_vocabulary(all_features, "single")

# Dual vocabulary training
perception.train_gesture_vocabulary(harmonic_features, "harmonic")
perception.train_gesture_vocabulary(percussive_features, "percussive")
```

### 5. Save/Load Method Updates

**File**: `listener/dual_perception.py` (lines 380-411)

**Methods**: `save_vocabulary()` and `load_vocabulary()`

**Changes**:
- Added `vocabulary_type` parameter to both methods
- Routes to appropriate quantizer based on type
- Error handling for uninitialized quantizers

**Usage**:
```python
# Save dual vocabularies
perception.save_vocabulary("harmonic_vocab.joblib", "harmonic")
perception.save_vocabulary("percussive_vocab.joblib", "percussive")

# Load dual vocabularies
perception.load_vocabulary("harmonic_vocab.joblib", "harmonic")
perception.load_vocabulary("percussive_vocab.joblib", "percussive")
```

## Architecture: How It Works

### Training Phase (Chandra_trainer.py - NEXT STEP)

1. Load full-band audio (e.g., Daybreak.wav with guitar + drums)
2. Apply HPSS to separate sources:
   - `audio_harmonic` = guitar, bass, vocals, sustained tones
   - `audio_percussive` = drums, hi-hats, transients
3. Extract segments from both sources
4. Process harmonic segments → Wav2Vec → `harmonic_features[]`
5. Process percussive segments → Wav2Vec → `percussive_features[]`
6. Train two vocabularies:
   ```python
   perception.train_gesture_vocabulary(harmonic_features, "harmonic")
   perception.train_gesture_vocabulary(percussive_features, "percussive")
   ```
7. For each event, assign BOTH tokens:
   ```python
   event['harmonic_token'] = harmonic_quantizer.transform(event_features)
   event['percussive_token'] = percussive_quantizer.transform(event_features)
   ```
8. AudioOracle learns correlations between harmonic and percussive patterns

### Performance Phase (MusicHal_9000.py - STEP 3)

1. Audio input → detect content type:
   ```python
   content_type, h_ratio, p_ratio = detect_content_type(audio)
   ```
2. Build adaptive request based on content:
   ```python
   if content_type == "percussive":
       # User plays drums → query "what harmonic patterns co-occurred with this drum pattern?"
       request = {
           'percussive_token': current_perc_token,
           'response_mode': 'harmonic'
       }
   elif content_type == "harmonic":
       # User plays guitar → query "what rhythmic patterns co-occurred with this harmony?"
       request = {
           'harmonic_token': current_harm_token,
           'response_mode': 'percussive'
       }
   ```
3. AudioOracle queries learned correlations
4. AI responds with contextually appropriate content

## Backward Compatibility

✅ **Fully backward compatible**:
- `enable_dual_vocabulary=False` (default) → traditional single-vocabulary mode
- All existing code continues to work
- Legacy `gesture_token` field preserved
- No breaking changes to existing models

## Next Steps

### STEP 2: Update Chandra_trainer.py

- [ ] Add `--dual-vocabulary` command-line flag
- [ ] Implement HPSS separation in Step 4b (Dual Perception)
- [ ] Extract harmonic and percussive segments
- [ ] Train both vocabularies
- [ ] Assign dual tokens to events
- [ ] Save both vocabularies to separate files

### STEP 3: Update AudioOracle

- [ ] Modify `add()` to accept `harmonic_token` and `percussive_token` parameters
- [ ] Store both tokens in state metadata
- [ ] Modify `generate_with_request()` to support `response_mode` filtering

### STEP 4: Update MusicHal_9000.py

- [ ] Add content detection in `process_event()`
- [ ] Build adaptive requests based on detected content type
- [ ] Print debug info showing detection and response strategy

### STEP 5: Testing

- [ ] Train on Daybreak.wav with dual vocabulary
- [ ] Test with drums-only input
- [ ] Test with guitar-only input
- [ ] Test with hybrid input
- [ ] Verify contextually appropriate responses

## Research Context

This addresses user feedback: "I have tested this script when playing drums only...let's explore a version...where the input might work better with this kind of input."

**Problem**: Current system assumes harmonic content (ratio analysis, chroma extraction). With drums-only input, only Wav2Vec gesture tokens capture meaningful patterns—other features become noise.

**Solution**: Dual-vocabulary architecture that:
1. Learns separate vocabularies for harmonic and percussive content
2. Preserves musical relationships (kick+root, snare+chord-change correlations)
3. Detects content type at runtime
4. Responds appropriately: drums → harmony, guitar → rhythm

**Artistic Research Goal**: Enable trustworthy musical partnership regardless of input type, maintaining coherence and intentionality through learned correlations.

---

**Implementation Status**: ✅ Step 1 Complete (DualPerceptionModule updates)
**Next**: Step 2 (Chandra_trainer.py modifications)
