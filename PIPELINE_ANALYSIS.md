# Pipeline Analysis: What's Working & What's Not

## Date: Oct 8, 2025, 23:58

## Summary: Mixed Success

### ✅ What's Working PERFECTLY

1. **L2 Normalization** 
   ```
   Normalization: L2 (IRCAM)
   Active tokens: 64/64
   Entropy: 5.67 bits
   ```
   - K-means clustering is working flawlessly
   - All 64 gesture tokens learned with excellent diversity

2. **Timestamp Normalization**
   ```
   Unique tokens assigned: 64
   Token range: 0 to 63
   ```
   - Events are successfully mapped to segments
   - All 64 unique tokens are being assigned

3. **Token Persistence Through Enhancement**
   ```
   Before enhancement: 64 unique tokens
   After enhancement: 64 unique tokens
   ```
   - Gesture tokens survive the enhancement step

### ❌ What's BROKEN

**Gesture tokens are NOT being saved to the model file!**

```
Training log shows: 64 unique tokens ✅
Model file contains: 0 frames with 'gesture_token' field ❌
```

## Root Cause Analysis

### The DataFlow

1. **Step 4b: Dual Perception** (Chandra_trainer.py line 945)
   ```python
   event['gesture_token'] = gesture_token  # ✅ Set here
   ```

2. **Step 8: Enhancement** (line 1021-1050)
   ```python
   enhanced_event = event.copy()  # ✅ Copied here
   # gesture_token is in enhanced_event
   ```

3. **AudioOracle Training** (hybrid_trainer.train_from_events)
   ```python
   # Events passed to AudioOracle with gesture_token field
   ```

4. **AudioOracle Add** (memory/audio_oracle.py or polyphonic_audio_oracle.py)
   ```python
   frame = AudioFrame(
       timestamp=event['t'],
       features=feature_vector,
       audio_data=???  # ❓ Does this include gesture_token?
   )
   ```

5. **Model Serialization**
   ```python
   # AudioFrame serialized to JSON
   # audio_data dict should contain all event fields
   ```

### The Problem

**AudioOracle is filtering which fields go into `audio_data`!**

The `AudioFrame` dataclass stores:
- `timestamp`: From event['t']
- `features`: The 768D Wav2Vec features
- `audio_data`: A filtered subset of the event dict

The `audio_data` dict probably only includes standard fields like `f0`, `midi`, `rms_db`, etc., and **excludes** custom fields like `gesture_token`.

## What We Need to Check

1. **Where does AudioOracle create AudioFrame objects?**
   - Which event fields are copied to `audio_data`?
   - Is there a whitelist of allowed fields?

2. **How are frames serialized to JSON?**
   - Does `audio_data` dict get saved completely?
   - Or is there another filtering step?

## Two Possible Fixes

### Option 1: Modify AudioOracle to preserve all fields
```python
# In audio_oracle.py or hybrid_trainer.py
def add_event(self, event):
    frame = AudioFrame(
        timestamp=event['t'],
        features=self._extract_features(event),
        audio_data=event  # ✅ Save ENTIRE event dict
    )
```

### Option 2: Add gesture_token to the whitelist
```python
# Find where audio_data is created
audio_data = {
    't': event.get('t'),
    'f0': event.get('f0'),
    'midi': event.get('midi'),
    # ... other fields ...
    'gesture_token': event.get('gesture_token'),  # ✅ Add this
}
```

## Impact Assessment

### Current State
- ✅ Training works: 64 unique tokens computed
- ✅ Tokens persist through pipeline
- ❌ Tokens don't reach MusicHal (not saved)
- ❌ MusicHal can't use gesture information

### After Fix
- ✅ Tokens saved to model
- ✅ MusicHal can access gesture_token for each frame
- ✅ Can make decisions based on gesture patterns
- ✅ Full IRCAM-style pattern learning

## Next Steps

1. **Find where AudioFrame.audio_data is populated**
2. **Ensure gesture_token is included**
3. **Verify it appears in saved JSON**
4. **Test MusicHal with gesture-aware model**

---

**The good news:** The hard parts (L2 norm, quantization, token assignment) are all working! We just need to preserve the tokens through serialization. 🎯





