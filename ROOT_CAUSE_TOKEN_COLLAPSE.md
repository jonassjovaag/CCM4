# TOKEN DIVERSITY COLLAPSE - ROOT CAUSE FOUND

**Date**: November 21, 2025  
**Problem**: Live extraction shows only 8-36% token diversity vs 100% during training  
**Status**: ✅ ROOT CAUSE IDENTIFIED

## The Problem

Training achieved perfect 64/64 token diversity (100%) with even distribution.  
Live performance collapsed to 8-36% diversity depending on configuration.

## Root Cause Discovery Process

### Hypotheses Tested (All REJECTED):
1. ❌ Model mismatch (Wav2Vec vs MERT) - both use MERT-v1-95M
2. ❌ Vocabulary corruption - all 64 tokens trained correctly
3. ❌ Musical mismatch - even exact training audio showed low diversity
4. ❌ Gesture smoothing - no difference with/without
5. ❌ Dense sampling solves it - actually made it WORSE (7/64 tokens!)

### The Breakthrough

**Sample Count Analysis**:
- Training vocabulary: 49,865 samples
- Expected with 350ms/no overlap: 991 segments
- MERT outputs: **~26 frames per 350ms segment**
- Calculation: 991 segments × 26 frames = 25,792 frames
- 25,792 × 2 (harmonic + percussive HPSS sources) = **51,584**
- Actual: 49,865 (96% match - difference likely edge effects)

**MERT Frame Output Test**:
```
100ms  →  7 frames (70 frames/sec)
200ms  → 14 frames (70 frames/sec)  
350ms  → 26 frames (74 frames/sec)
500ms  → 37 frames (74 frames/sec)
1000ms → 74 frames (74 frames/sec)
```

MERT consistently outputs ~74 frames/sec. Current code does `.mean(dim=1)` which averages all time steps into a single 768D vector.

## Root Cause

**Training**: Extracts ALL MERT frames (26 per segment) from both harmonic and percussive sources
- 991 segments × 26 frames × 2 sources ≈ 51,584 feature vectors
- Each frame gets quantized separately → 49,865 diverse tokens

**Live**: Averages all frames to single vector per segment  
- 991 segments × 1 averaged vector = 991 feature vectors
- Massive information loss (26x less data per segment)
- Temporal variation within 350ms completely lost

**Code Location**: `listener/wav2vec_perception.py` line 204:
```python
# Current (WRONG for live):
features = hidden_states.mean(dim=1).squeeze().cpu().numpy()

# Should be (match training):
# Extract ALL frames: hidden_states[0].cpu().numpy()  # Shape: [time_steps, 768]
```

## Verification Test

Extracted all frames (no averaging) from 60s of General_idea.wav:

| Configuration | Unique Tokens | Top Token % | Top 3 % |
|--------------|---------------|-------------|---------|
| **Training** | **64/64 (100%)** | **3.5%** | - |
| Live (averaged) | 23/64 (35.9%) | 13.5% | 33.9% |
| **Live (all frames)** | **48/64 (75.0%)** | **7.4%** | **20.4%** |

**Result**: Extracting all frames DOUBLES diversity (35.9% → 75.0%)!

Still not perfect 100%, but:
- Only tested 60s vs full 347s training audio
- Much closer to training behavior
- Confirms the hypothesis

## The Fix

### Option 1: Make live extraction match training (recommended)
Modify `Wav2VecMusicEncoder.encode()` to return all frames instead of averaging:

```python
def encode(self, audio, sr, timestamp, return_all_frames=True):
    # ... existing preprocessing ...
    
    with torch.no_grad():
        outputs = self.model(**inputs)
    
    hidden_states = outputs.last_hidden_state  # [batch=1, time_steps, 768]
    
    if return_all_frames:
        # Return ALL frames (match training)
        all_frames = hidden_states[0].cpu().numpy()  # [time_steps, 768]
        # Return list of Wav2VecFeatures objects, one per frame
        return [Wav2VecFeatures(features=frame, ...) for frame in all_frames]
    else:
        # Legacy averaging behavior
        features = hidden_states.mean(dim=1).squeeze().cpu().numpy()
        return Wav2VecFeatures(features=features, ...)
```

**Implications**:
- Each 350ms audio segment → 26 tokens instead of 1
- Agent must handle multiple tokens per time step
- Memory buffer stores 26x more tokens
- Gesture smoothing needs redesign
- MIDI output timing needs adjustment

### Option 2: Retrain vocabulary with averaged features
Re-run training with averaging enabled. This would be consistent but loses temporal detail that was learned.

**Recommendation**: Option 1 - make live match training. The temporal variation is musically meaningful.

## Why Dense Sampling Made It Worse

Earlier test: 350ms window with 7ms hop → only 7/64 tokens (10.9%), token 16 at 82.6%

**Explanation**: Overlapping windows with averaging meant each segment contained mostly the same audio, producing nearly identical averaged features → all map to same token. Averaging destroys temporal variation, overlap amplifies the problem.

## Next Steps

1. ✅ Root cause confirmed via all-frames test
2. ⏳ Implement `return_all_frames` mode in Wav2VecMusicEncoder
3. ⏳ Update DualPerceptionModule to handle frame lists
4. ⏳ Modify agent to process multiple tokens per segment
5. ⏳ Test live performance with all-frames extraction
6. ⏳ Compare diversity with training baseline

## Files Created During Investigation

- `test_mert_frames.py` - Initial frame count test (26 frames per 350ms)
- `test_dense_sampling.py` - Dense sampling paradox test
- `test_mert_frame_scaling.py` - Frame rate scaling verification
- `check_vocab_totals.py` - Vocabulary sample count analysis
- `test_all_frames_extraction.py` - **Breakthrough test** (48/64 tokens!)
- `diagnose_simple.py`, `diagnose_diversity_collapse.py` - Earlier diagnostic tests

## Key Learning

**Never assume preprocessing matches between training and inference!**

The `.mean(dim=1)` averaging was added for computational efficiency during live performance, assuming it wouldn't matter. But the training code was extracting and quantizing ALL frames, making that temporal detail part of the learned vocabulary. The mismatch caused catastrophic diversity collapse.

Always verify: "Is the exact same preprocessing applied during training and inference?"
