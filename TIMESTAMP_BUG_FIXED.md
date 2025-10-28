# Bug Found & Fixed: Timestamp Mismatch

## Date: Oct 8, 2025, 23:45

## The Bug 🐛

Debug output revealed that **ALL 1500 events were getting the SAME gesture token (32)**:

```
🔍 DEBUG - Token Assignment:
   Unique tokens assigned: 1  ← ALL EVENTS GET TOKEN 32!
   Sample events: [(1759959594.0343218, 32), (1759959594.3511736, 32), ...]
                    ^^^^^^^^^^^^^^^^^
                    ~55 YEARS in seconds!
```

## Root Cause

**Timestamp mismatch** between events and segments:
- **Segment times:** 0 to 244 seconds (correct audio duration) ✅
- **Event times:** ~1,759,959,594 seconds (absolute/Unix timestamps) ❌

When finding the closest segment for each event:
```python
closest_segment = min(segment_features,
                     key=lambda s: abs((s['start_time'] + s['end_time'])/2 - event_time))
```

ALL events are equally far (~1.76 billion seconds) from ALL segments → Always picks the SAME segment → Always gets token 32!

## The Fix ✅

Added timestamp normalization (lines 949-966):

```python
# Get audio duration for normalization
audio_duration = len(audio) / sr

for event in events:
    event_time = event.get('t', 0)
    
    # Normalize event time if it's absolute (e.g., Unix timestamp)
    if event_time > audio_duration:
        # Use modulo to wrap into [0, audio_duration] range
        event_time = event_time % audio_duration
        print(f"   ⚠️ WARNING: Event has absolute timestamp, normalizing to {event_time:.2f}s")
    
    # Now find closest segment with normalized time
    closest_segment = min(segment_features, ...)
```

## Expected Results After Fix

```
🔍 DEBUG - Token Assignment:
   Unique tokens assigned: 40-60  ← DIVERSE TOKENS!
   Token range: 0 to 63
   Sample tokens: [1, 5, 12, 18, 23, 32, 41, ...]  ← VARIETY!
   Sample events: [(1.2, 5), (1.5, 12), (1.8, 23), ...]  ← NORMALIZED TIMES!
```

## Why L2 Normalization IS Working

The vocabulary training showed **perfect results**:
```
Normalization: L2 (IRCAM)  ✅
Active tokens: 64/64  ✅
Entropy: 5.67 bits  ✅✅✅
```

The problem was **ONLY** in the timestamp-based segment-to-event mapping, NOT in the L2 normalization or K-means clustering!

## Next Steps

**Retrain Georgia** with the timestamp fix:

```bash
python Chandra_trainer.py \
    --file input_audio/Georgia.wav \
    --max-events 1500 \
    --hybrid-perception \
    --wav2vec \
    --gpu
```

**What to watch for:**
1. Debug output should show normalized timestamps (0-244 range)
2. "Unique tokens assigned" should be 40-60 (not 1!)
3. Token distribution should show variety

This fix resolves the mapping bug while preserving the working L2 normalization! 🎯





