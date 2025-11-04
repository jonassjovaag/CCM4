# Training Analysis: L2 Normalization Success BUT Mapping Failure

## Date: Oct 8, 2025, 23:25

## What Worked ✅

**L2 Normalization FIX SUCCESS!**
```
📚 Learning musical vocabulary (64 classes)...
   Normalization: L2 (IRCAM)  ✅
   Active tokens: 64/64  ✅
   Entropy: 5.67 bits  ✅✅✅ (PERFECT!)
```

The L2 normalization fix completely solved the K-means clustering problem. The quantizer learned a perfect vocabulary with all 64 tokens active and excellent entropy distribution.

## What Failed ❌

**Gesture Tokens NOT Saved to Model!**

Model inspection revealed:
```
🎯 Audio Frames: 1500 total
   Frames with gesture_token: 0  ❌❌❌
```

### The Problem Chain

1. **698 segments extracted** → Wav2Vec features computed ✅
2. **Vocabulary trained** → 64 tokens, 5.67 bits entropy ✅
3. **Events augmented** → `gesture_token` field added to events ✅
4. **AudioOracle trained** → Events passed to training ✅
5. **Model saved** → `gesture_token` field **MISSING** from audio_frames! ❌

### Evidence

**Training output says:**
```
🤖 MACHINE PERCEPTION:
   • Gesture tokens: 1 unique patterns  ← This is WRONG!
```

**JSON model shows:**
- `audio_frames`: 1500 total
- Frames with `gesture_token` field: **0**  
- Frames with `features` field: Unknown (need to check)

### Root Cause

The `gesture_token` and possibly `features` fields are being:
1. **Never passed to AudioOracle**, OR
2. **Stripped out during AudioOracle training**, OR
3. **Not saved during model serialization**

## Code Analysis

### Where tokens ARE set (✅):

**`Chandra_trainer.py` lines 928-967:**
```python
# Map segments to events and augment with DUAL representations
for event in events:
    # Extract gesture token for this segment
    gesture_token = int(self.dual_perception.quantizer.transform(...)[0])
    
    # === MACHINE REPRESENTATION ===
    event['gesture_token'] = gesture_token  ← Set here!
    event['features'] = closest_segment['wav2vec_features'].tolist()  ← Set here!
    event['consonance'] = closest_segment['consonance']
    ...
```

### Where tokens might get LOST (❌):

1. **`_enhance_events_with_insights()` (lines 1009+)**
   - Creates `enhanced_event = event.copy()`
   - Might not preserve `gesture_token`?

2. **AudioOracle training**
   - May filter which fields get saved
   - Check: `PolyphonicAudioOracle.add()` method

3. **Model serialization**
   - Check: `save()` method in AudioOracle
   - May only save specific fields

## The "1 unique pattern" Mystery

Training output claims "1 unique pattern" but this contradicts the fact that:
- 64 tokens were learned
- Entropy is 5.67 bits (excellent)
- Each of 698 segments got a different token (should have!)

**Hypothesis:** The counting logic at line 970 is correct:
```python
unique_tokens = len(set(e.get('gesture_token') for e in events if e.get('gesture_token') is not None))
```

But somehow all 1500 events are getting `None` or the same token when mapped.

## Next Steps to Debug

### 1. Add debug logging to `_augment_with_dual_features()`

After line 967, add:
```python
# DEBUG: Check what tokens were actually assigned
tokens_assigned = [e.get('gesture_token') for e in events]
unique_assigned = set(t for t in tokens_assigned if t is not None)
print(f"   🔍 DEBUG: Assigned {len(unique_assigned)} unique tokens to {len(events)} events")
print(f"   🔍 DEBUG: Token distribution: {sorted(unique_assigned)[:10]}...")
print(f"   🔍 DEBUG: Sample events with tokens: {[(e.get('t'), e.get('gesture_token')) for e in events[:5]]}")
```

### 2. Check if tokens survive `_enhance_events_with_insights()`

Add logging before/after enhancement to see if fields get lost.

### 3. Check AudioOracle field preservation

Inspect what fields AudioOracle actually stores in `audio_frames`.

### 4. Check model serialization

See if `gesture_token` gets filtered out during JSON save.

## Immediate Question for User

**The L2 normalization IS working** (5.67 bits entropy proves it), but the gesture tokens aren't making it to the final model.

Options:
1. **Add debug logging** and retrain to see where tokens get lost
2. **Inspect the AudioOracle code** to see how it handles custom fields
3. **Try a minimal test** with just a few events to isolate the issue

What would you like to do?

---

**Summary:** The core L2 norm fix works perfectly! But there's a secondary bug in how gesture tokens are stored/serialized. The vocabulary training is now correct - we just need to ensure the tokens actually get saved to the model.





