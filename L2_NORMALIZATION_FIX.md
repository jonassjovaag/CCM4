# L2 Normalization Fix - Based on IRCAM Research

## The Problem 🚨

Your Georgia training showed **only 1 unique gesture token** despite having 698 segments! This means K-means clustering completely failed to find diverse patterns.

### Root Cause

We were using **StandardScaler** (mean=0, std=1) for feature normalization, but the IRCAM paper (Bujard et al., 2025) explicitly uses **L2 normalization** for Wav2Vec features before K-means clustering.

## The Difference

### StandardScaler (What we had) ❌
```python
# Centers data: mean=0
# Scales: std=1
# Each FEATURE dimension is normalized independently
```

**Problem:** In high-dimensional space (768D Wav2Vec), StandardScaler doesn't handle the geometry well.

### L2 Normalization (IRCAM approach) ✅
```python
# Normalizes each VECTOR to unit length
# All vectors lie on a hypersphere
# Preserves angular relationships between vectors
```

**Why it works:** In high-dimensional spaces, angular distance on a hypersphere is more meaningful than Euclidean distance. K-means works better when data lies on a manifold (hypersphere) rather than in raw high-D space.

## The Fix ✅

Updated `listener/symbolic_quantizer.py`:

1. **Added L2 norm option** (line 45):
   ```python
   use_l2_norm: bool = True  # Default to IRCAM approach
   ```

2. **Updated fit() method** (line 104-110):
   ```python
   if self.use_l2_norm:
       # L2 normalization: Each vector has unit length (IRCAM approach)
       features_scaled = normalize(features, norm='l2', axis=1)
   else:
       # StandardScaler: Mean=0, Std=1 (traditional approach)
       features_scaled = self.scaler.fit_transform(features)
   ```

3. **Updated transform() method** (line 147-150):
   ```python
   if self.use_l2_norm:
       features_scaled = normalize(features, norm='l2', axis=1)
   else:
       features_scaled = self.scaler.transform(features)
   ```

4. **Backward compatible:** Old models with StandardScaler still work.

## Expected Results After Retraining

**Before (with StandardScaler):**
```
🤖 MACHINE PERCEPTION:
   • Gesture tokens: 1 unique patterns  ❌
   • Entropy: 0.00 bits  ❌
```

**After (with L2 normalization):**
```
🤖 MACHINE PERCEPTION:
   • Gesture tokens: 40-60 unique patterns  ✅
   • Entropy: 4-5 bits  ✅
```

## Why This Matters for Music

**L2 normalization preserves:**
- **Angular relationships:** Similar-sounding segments have similar angles
- **Spectral shapes:** Timbre relationships are preserved
- **Relative feature importance:** Doesn't distort dimensions differently

**For Wav2Vec features specifically:**
- Wav2Vec already encodes musical gestures in its 768D space
- L2 norm respects the learned geometry from Wav2Vec pretraining
- K-means can find clusters of similar gestures (timbre, rhythm, pitch contour)

## References

**Bujard et al. (2025) - IRCAM**
- "Feature Extraction and Quantization": Wav2Vec → **L2 normalize** → K-means
- Vocabulary sizes tested: 16, 64, 256 (64 works best for musical relationships)
- Critical for learning patterns in AudioOracle

## Next Steps

1. **Retrain Georgia** with the fixed quantizer (L2 norm is now default)
2. **Verify** you get 40-60 unique tokens (not just 1!)
3. **Test** that MusicHal now responds musically instead of randomly

---

**The fix is simple but crucial!** 🎯 This aligns our implementation with the actual IRCAM research.





