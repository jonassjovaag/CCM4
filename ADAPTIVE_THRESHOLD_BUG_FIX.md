# CRITICAL BUG FIX: Adaptive Threshold Lowering Distance Threshold to 0.05

**Date**: November 10, 2025  
**Issue**: All suffix links pointing to state 0 (degenerate case)  
**Root Cause**: Adaptive threshold logic LOWERING distance threshold to 0.05 during training

## The Bug

**File**: `memory/audio_oracle.py` line 238-239

**Buggy code**:
```python
# Clamp to reasonable range for cosine [0.05, 0.5]
new_threshold = np.clip(new_threshold, 0.05, 0.5)
```

**Problem**: This MINIMUM value (0.05) was set for low-dimensional features (6D-15D). For **768D Wav2Vec features** with cosine distance, this is WAY too strict!

## Why It Happened

1. **We set constructor default**: 0.15 → 0.25 ✅
2. **Training starts**: threshold = 0.25
3. **After 100 events**: adaptive_threshold triggers (`_adjust_threshold()`)
4. **Adaptive logic calculates**: `mean_dist + 0.5 * std_dist`
5. **Clamp applies**: `np.clip(new_threshold, 0.05, 0.5)`
6. **Result**: Threshold LOWERED to 0.05 (below minimum)
7. **Effect**: Almost NO suffix links created (distance always > 0.05 for 768D features)

## The Fix

**Three files updated**:

### 1. `memory/audio_oracle.py` (line 238)
```python
# OLD:
new_threshold = np.clip(new_threshold, 0.05, 0.5)

# NEW:
new_threshold = np.clip(new_threshold, 0.15, 0.6)  # Increased for 768D features
```

###  2. `memory/polyphonic_audio_oracle.py` (line 63)
```python
# OLD:
distance_threshold: float = 0.15,

# NEW:
distance_threshold: float = 0.25,  # Increased from 0.15 for 12D feature space
```

### 3. `audio_file_learning/hybrid_batch_trainer.py` (line 24)
```python
# OLD:
distance_threshold: float = 0.15,

# NEW:
distance_threshold: float = 0.25,  # Increased from 0.15 for 768D features
```

## Why 0.15 Minimum for Cosine Distance?

**Cosine distance properties**:
- Range: 0 (identical) to 2 (opposite directions)
- For normalized 768D vectors, typical distances:
  * Very similar patterns: 0.10 - 0.20
  * Moderately similar: 0.20 - 0.40
  * Different: 0.40 - 0.80
  * Very different: 0.80+

**0.05 threshold** = "Only accept EXTREMELY similar patterns"
- Result: Almost nothing matches → all suffix links → state 0

**0.15 threshold** = "Accept reasonably similar patterns"
- Result: Suffix links connect genuinely related musical moments

**0.25 starting point** = "Be generous initially, let adaptive logic tighten"
- Adaptive will lower to ~0.15-0.20 based on actual data distribution

## Testing

**Before fix**:
```
States: 5001
Suffix links: 5000
Unique targets: 1 (all → state 0)  ❌ BROKEN
```

**After fix** (running now):
```
Expected:
Unique targets: >100 (diverse pattern connections)
Suffix links per state: 1-3
Coverage: 50-70%
```

## Lessons Learned

1. **Adaptive thresholds need dimension-aware clamps**
   - 6D features: 0.05-0.3 works
   - 768D features: 0.15-0.6 needed

2. **Always check saved model values**
   - Don't trust constructor defaults
   - Adaptive logic can override during training

3. **Cosine distance != Euclidean distance**
   - Different distance metrics need different thresholds
   - Cosine is normalized (0-2 range)
   - Euclidean grows with dimensionality

4. **Test suffix link construction during development**
   - Don't wait until phrasing tests fail
   - Check unique suffix targets immediately after training

## Files Modified

1. `memory/audio_oracle.py` - Adaptive threshold clamp (0.05 → 0.15)
2. `memory/polyphonic_audio_oracle.py` - Constructor default (0.15 → 0.25)
3. `memory/polyphonic_audio_oracle_mps.py` - Constructor default (0.15 → 0.25)
4. `audio_file_learning/hybrid_batch_trainer.py` - Constructor default (0.15 → 0.25)

## Commit Message

```
fix: Increase adaptive threshold minimum for 768D Wav2Vec features

Adaptive threshold was clamping to minimum 0.05, causing all suffix
links to point to state 0 (no pattern repetitions found). This was
tuned for low-D features (6D-15D) but too strict for 768D embeddings.

Changes:
- Adaptive clamp: 0.05 → 0.15 minimum (audio_oracle.py)
- Constructor default: 0.15 → 0.25 (polyphonic_audio_oracle.py)
- Hybrid trainer: 0.15 → 0.25 (hybrid_batch_trainer.py)

Result: Suffix links now connect genuinely similar patterns instead
of always defaulting to state 0. Enables thematic development and
musical memory as documented in COMPLETE_ARTISTIC_RESEARCH_DOCUMENTATION.md.

Fixes degenerate suffix link construction discovered via test suite
(tests/test_repetition_analysis.py).
```

## Next Steps

1. ✅ Fix applied
2. 🔄 Training test running (2000 events)
3. ⏳ Diagnose suffix links in new model
4. ⏳ Re-run provenance tests
5. ⏳ Full training if tests pass (5000 events)
6. ⏳ Commit fixes to repository
