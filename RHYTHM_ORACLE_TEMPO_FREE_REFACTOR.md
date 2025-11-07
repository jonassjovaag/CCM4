# RhythmOracle Tempo-Free Refactor - COMPLETE ✅

## Summary

Successfully refactored RhythmOracle to use **Brandtsegg ratio-based rhythm analysis** instead of absolute tempo matching. This enables true tempo-free improvisation (80-200+ BPM) while preserving rhythmic character and pattern matching.

## Problem Identified

**Root Cause**: Architectural contradiction between two systems built in parallel:
- `listener/ratio_analyzer.py` (CORRECT) - Extracts tempo-independent interval ratios using Brandtsegg method
- `rhythmic_engine/memory/rhythm_oracle.py` (WRONG) - Stored absolute tempo/onsets, penalized tempo differences

**Symptom**: RhythmOracle returned 0 patterns during performance because:
- Training tempo: 99.4 BPM
- Performance tempo: 140-170 BPM (40-70% faster)
- Tempo penalty made similarity too low → no matches

**User Insight**: "but was this strict rhythm really the plan? Isnt that the reason for using the Brandtsegg rhythm ratio idea?"

## Solution Implemented

### 1. RhythmPattern Class Refactor ✅

**File**: `rhythmic_engine/memory/rhythm_oracle.py`

**Changes**:
```python
# BEFORE (tempo-dependent)
class RhythmicPattern:
    pattern_id: str
    tempo: float  # Absolute BPM - WRONG
    density: float
    syncopation: float
    # ...

# AFTER (tempo-free)
class RhythmicPattern:
    pattern_id: str
    duration_pattern: List[int]  # [2, 1, 1, 2] - tempo-free ratios
    density: float
    syncopation: float
    pulse: int  # 2, 3, or 4
    complexity: float  # Barlow indigestability
    # ...
    
    def to_absolute_timing(self, tempo: float, start_time: float = 0.0) -> List[float]:
        """Convert ratios to playback timing at any tempo"""
        beat_duration = 60.0 / tempo
        onsets = [start_time]
        for duration in self.duration_pattern:
            onsets.append(onsets[-1] + (beat_duration * duration))
        return onsets[1:]
```

**Key Feature**: Patterns stored in tempo-independent format, scaled to playback tempo on-demand.

### 2. Pattern Similarity Calculation ✅

**Changes**:
```python
# BEFORE (tempo penalty)
tempo_sim = 1.0 - abs(query_tempo - pattern_tempo) / pattern_tempo
similarity = tempo_sim * 0.2 + density_sim * 0.4 + syncopation_sim * 0.3

# AFTER (ratio-based matching)
# Cosine similarity of duration patterns
duration_sim = np.dot(query_pattern_norm, pattern_pattern_norm)
similarity = (
    duration_sim * 0.4 +      # Duration structure highest weight
    syncopation_sim * 0.25 +  # Off-beat character
    density_sim * 0.2 +       # Note density
    pulse_sim * 0.1 +         # Subdivision feel (2, 3, 4)
    type_sim * 0.05           # Pattern type
)
```

**Result**: Patterns match on rhythmic **character** (syncopation, density, pulse) rather than absolute tempo.

### 3. Training Extraction ✅

**File**: `Chandra_trainer.py::_train_rhythm_oracle()`

**Changes**:
```python
# For each detected rhythmic pattern:
if len(pattern_onsets) >= 3:
    # Use existing Brandtsegg ratio analyzer
    rational = self.rhythmic_analyzer.analyze_rational_structure(pattern_onsets)
    if rational:
        duration_pattern = rational['duration_pattern']  # e.g., [2, 1, 1, 2]
        pulse = rational['pulse']  # e.g., 4
        complexity = rational['complexity']  # Barlow score

pattern_data = {
    'duration_pattern': duration_pattern,  # NOT tempo
    'density': pattern.density,
    'syncopation': pattern.syncopation,
    'pulse': pulse,
    'complexity': complexity,
    # ...
}
self.rhythm_oracle.add_rhythmic_pattern(pattern_data)
```

**Fallback**: If rational analysis fails (< 3 onsets), creates simple interval-based pattern.

### 4. Performance Query Logic ✅

**File**: `agent/phrase_generator.py::_get_rhythmic_phrasing_from_oracle()`

**Changes**:
```python
# Build tempo-free query from recent input
recent_onsets = [e.get('t') for e in recent_events]
intervals = np.diff(recent_onsets)
duration_pattern = [int(round(interval / min(intervals))) for interval in intervals]

query = {
    'duration_pattern': duration_pattern,  # NOT tempo
    'density': avg_density,
    'syncopation': avg_syncopation,
    'pulse': median_pulse
}

# Query RhythmOracle (tempo-free matching)
similar_patterns = self.rhythm_oracle.find_similar_patterns(query, threshold=0.6)

# Estimate CURRENT tempo from recent playing
avg_interval = np.mean(np.diff(recent_onsets))
current_tempo = 60.0 / avg_interval  # Clamped to 60-200 BPM

# Scale pattern to current tempo for playback
absolute_onsets = best_pattern.to_absolute_timing(current_tempo)

return {
    'duration_pattern': best_pattern.duration_pattern,
    'absolute_onsets': absolute_onsets,  # Ready for MIDI
    'current_tempo': current_tempo,
    # ...
}
```

**Result**: Pattern matching independent of tempo, playback adapts to current performance tempo.

## Architecture Alignment

**Before** (Contradiction):
```
ratio_analyzer.py (tempo-free) ----X----> RhythmOracle (tempo-dependent)
         ↓                                        ↓
  Duration patterns                        Absolute tempo
  Syncopation                              Tempo penalty
  Pulse subdivision                        No matches
```

**After** (Unified):
```
ratio_analyzer.py (tempo-free) --------> RhythmOracle (tempo-free)
         ↓                                        ↓
  Duration patterns  ──────────────────>  Duration patterns
  Syncopation       ──────────────────>  Syncopation
  Pulse subdivision ──────────────────>  Pulse subdivision
                                                  ↓
                                         to_absolute_timing(current_tempo)
                                                  ↓
                                            MIDI playback
```

## Files Modified

1. **rhythmic_engine/memory/rhythm_oracle.py** (75 lines changed)
   - RhythmicPattern dataclass: removed `tempo`, added `duration_pattern`, `pulse`, `complexity`
   - Added `to_absolute_timing()` method
   - `_calculate_pattern_similarity()`: removed tempo penalty, added ratio cosine similarity
   - `get_rhythmic_statistics()`: removed tempo averaging
   - `add_rhythmic_pattern()`: updated to accept new fields

2. **Chandra_trainer.py** (`_train_rhythm_oracle()` method, 80 lines)
   - Extract duration patterns using `analyze_rational_structure()`
   - Fallback to interval-based patterns if rational analysis unavailable
   - Statistics reporting for rational vs fallback patterns

3. **agent/phrase_generator.py** (110 lines changed)
   - `_get_rhythmic_phrasing_from_oracle()`: query with duration patterns, estimate current tempo
   - `_apply_rhythmic_phrasing_to_timing()`: use `absolute_onsets` from pattern, scale to tempo
   - Added traceback printing for debugging

## Testing Required

**Next Step**: Todo #5 - Test with tempo variation

1. **Retrain** on Itzama.wav to generate tempo-free RhythmOracle patterns:
   ```bash
   python Chandra_trainer.py --file "input_audio/Itzama.wav" --output "JSON/Itzama_tempo_free_test.json"
   ```

2. **Test performance** at different tempos:
   - Run 5-minute performance session
   - Vary playing tempo: 80 → 120 → 160 → 100 BPM
   - Watch terminal for: `🥁 RhythmOracle returned X patterns` (should be > 0)
   - Verify rhythm character preserved across tempo changes

3. **Success criteria**:
   - RhythmOracle returns patterns at ALL tempos (not just training tempo)
   - Rhythmic feel (syncopation, swing, density) matches across tempo changes
   - System responds organically to tempo shifts

## Expected Behavior

**Before Fix**:
```
Training: 99 BPM
Performance at 140 BPM:
  🥁 RhythmOracle returned 0 patterns  ← NO MATCHES (tempo penalty too high)
  Falling back to random generation
```

**After Fix**:
```
Training: 99 BPM (duration patterns: [2,1,1,2], [3,3,2], etc.)
Performance at 80 BPM:
  🥁 RhythmOracle returned 5 patterns  ← MATCHES (duration similarity)
  Using pattern [2,1,1,2] scaled to 80 BPM
Performance at 160 BPM:
  🥁 RhythmOracle returned 5 patterns  ← SAME PATTERNS (tempo-free)
  Using pattern [2,1,1,2] scaled to 160 BPM
```

## Technical Notes

### Cosine Similarity for Duration Patterns

```python
# Normalize to unit vectors
query_norm = query_pattern / ||query_pattern||
pattern_norm = pattern_pattern / ||pattern_pattern||

# Cosine similarity (1.0 = identical, 0.0 = orthogonal)
similarity = query_norm · pattern_norm
```

**Handles**:
- Same length patterns: direct comparison
- Different length patterns: truncate to shortest, penalize length difference

**Future improvement**: Dynamic Time Warping (DTW) for variable-length sequence alignment.

### Tempo Estimation Algorithm

```python
recent_onsets = [t1, t2, t3, t4, ...]  # Last 10 events
intervals = [t2-t1, t3-t2, t4-t3, ...]
avg_interval = mean(intervals)  # Average inter-onset interval
tempo = 60.0 / avg_interval  # Convert to BPM
tempo = clamp(tempo, 60, 200)  # Reasonable range
```

**Emergent tempo**: Calculated from actual performance, not fixed/tracked.

### Brandtsegg Rational Analysis

From `rhythmic_engine/ratio_analyzer.py`:
- **Duration pattern**: Integer ratios (e.g., [2,1,1,2] = "long, short, short, long")
- **Pulse**: Subdivision (2=duple, 3=triple, 4=quadruple)
- **Complexity**: Barlow indigestability (simpler = more consonant)
- **Deviations**: Timing errors from quantized pattern

**Already implemented** - we just connected it to RhythmOracle!

## Philosophical Alignment

**User's original intent**: "tempo will vary based on the response from the script, my mood, and so on"

**System design**: 
- Learns rhythmic **character** (syncopation, swing, density)
- Adapts to **emergent tempo** from interaction
- Preserves **musical feel** across tempo changes

**Not**: Fixed tempo tracking, metronome-based rhythm, strict tempo matching.

**This fix aligns implementation with artistic research goals.**

---

**Status**: Implementation complete. Ready for testing (Todo #5).
**Author**: AI Agent (GitHub Copilot)
**Date**: November 6, 2024
**Context**: MusicHal 9000 - AI Musical Partner System (PhD Project)
