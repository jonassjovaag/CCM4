# 🔴 CRITICAL BUG REPORT: Feature Extraction Failure

## Summary
**100% of training data has NO audio features** (f0, midi, rms_db, etc.)

## Root Cause Chain

1. **Chandra_trainer.py** uses hierarchical sampling via `SimpleHierarchicalAnalyzer`
2. **SimpleHierarchicalAnalyzer** returns `SampledEvent` objects from `SmartSampler`
3. **SampledEvent** dataclass only contains:
   - `event_id`: str
   - `time`: float
   - `features`: np.ndarray (raw feature vector)
   - `significance_score`: float
   - `sampling_reason`: str

4. **Missing fields**: f0, midi, rms_db, centroid, polyphonic_pitches, etc.

5. **Chain continues:**
   - `SmartSampler` extracts librosa features into numpy array
   - Creates `SampledEvent` with numpy array but NO individual fields
   - Chandra_trainer converts `SampledEvent` → Dict
   - Dict only has metadata: `{t, features, significance_score, sampling_reason, chord, stream_id, ...}`
   - **NO f0, midi, rms_db fields!**

6. **PolyphonicAudioOracle.extract_polyphonic_features()**:
   ```python
   features = [
       float(event_data.get('rms_db', -20.0)),  # ← Always -20.0 (default)!
       float(event_data.get('f0', 440.0)),      # ← Always 440.0 (default)!
       ...
   ]
   ```

7. **Result**: ALL audio_frames have identical fallback values
   - f0 = 440.0 Hz (default)
   - midi = 69 (A4, default)
   - rms_db = -20.0 dB (default)

## Evidence

### Synthetic Test Audio (C-D-E-F-G-A-B-C melody)
- **Expected**: MIDI [60, 62, 64, 65, 67, 69, 71, 72]
- **Found**: ALL frames = default values
- **Result**: 100% repetition of default note

### Itzama.wav (100 BPM with melody)
- **Expected**: Varied pitches
- **Found**: ALL frames = MIDI 60
- **Result**: AudioOracle learns NO melodic patterns

### Nineteen.wav (53 BPM)
- **Expected**: Varied pitches
- **Found**: ALL frames = MIDI 60
- **Result**: Same problem

## Fix Required

### Option 1: Enhance SampledEvent Dataclass
```python
@dataclass
class SampledEvent:
    # Existing fields
    event_id: str
    time: float
    features: np.ndarray
    significance_score: float
    sampling_reason: str
    
    # ADD these fields:
    f0: float = 440.0
    midi: int = 69
    rms_db: float = -20.0
    centroid: float = 2000.0
    rolloff: float = 3000.0
    bandwidth: float = 1000.0
    # ... etc for all audio features
```

### Option 2: Smart Feature Extraction in SmartSampler
When creating SampledEvent, also extract individual features:
```python
def _create_sampled_event(self, time, features_array):
    # Extract individual features from array
    # (Need to know the feature order/indices)
    f0 = self._extract_f0_from_features(features_array)
    midi = self._freq_to_midi(f0)
    rms_db = features_array[some_index]
    
    return SampledEvent(
        ...,
        f0=f0,
        midi=midi,
        rms_db=rms_db,
        ...
    )
```

### Option 3: Use Original Processor for Ground Truth
- Run `PolyphonicAudioProcessor.extract_features()` FIRST
- Save all events with full audio features
- THEN run hierarchical sampling to SELECT which events to keep
- This way selected events have all original audio features

## Recommendation
**Option 3** is cleanest and most robust:
1. Extract ALL events with full features (PolyphonicAudioProcessor)
2. Run hierarchical analysis for sampling scores
3. Select top N events based on scores
4. Keep original audio features intact

This preserves data integrity and doesn't require changing dataclasses.

