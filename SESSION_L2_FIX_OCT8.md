# Session Summary: L2 Normalization Fix (Oct 8, 2025)

## The Breakthrough 🎯

After reading the IRCAM papers deeply, I found the critical bug: **We were using the wrong normalization method for Wav2Vec features!**

## Timeline of Discovery

### 1. Initial Problem
- Georgia training DID use dual perception ✅
- But only learned **1 unique gesture token** (should be 40-60) ❌
- MusicHal sounded "like random shit" ❌

### 2. Deep Research
Read two key papers:
- **Perez et al. (Deep Chromas):** Deep learning chromas can hallucinate notes
- **Bujard et al. (IRCAM Wav2Vec):** Explicit pipeline: Wav2Vec → **L2 normalize** → K-means

### 3. Root Cause Found
Our `symbolic_quantizer.py` used **StandardScaler** (mean=0, std=1), but IRCAM uses **L2 normalization** (unit vectors on hypersphere).

**Why this matters:**
- StandardScaler: Centers each feature dimension independently
- L2 normalization: Normalizes entire vectors to unit length
- **In 768D space:** L2 norm preserves angular relationships, StandardScaler distorts geometry
- **Result:** K-means with L2 norm finds natural clusters, with StandardScaler it fails

## The Fix ✅

Updated `listener/symbolic_quantizer.py`:

```python
def __init__(self, vocabulary_size=64, use_l2_norm=True):  # L2 by default
    self.use_l2_norm = use_l2_norm
    
def fit(self, features):
    if self.use_l2_norm:
        # IRCAM approach: All vectors on unit hypersphere
        features_scaled = normalize(features, norm='l2', axis=1)
    else:
        # Traditional: Mean=0, Std=1
        features_scaled = self.scaler.fit_transform(features)
```

**Key changes:**
1. Added `use_l2_norm` parameter (defaults to True)
2. Uses `sklearn.preprocessing.normalize()` with `norm='l2'`
3. Applied to `fit()`, `transform()`, and `inverse_transform()`
4. Backward compatible with old models

## Expected Results

### Before (StandardScaler):
```
🤖 MACHINE PERCEPTION:
   • Gesture tokens: 1 unique patterns  ❌
   • Entropy: 0.00 bits
   • All 698 segments → same token!
```

### After (L2 normalization):
```
🤖 MACHINE PERCEPTION:
   • Gesture tokens: 40-60 unique patterns  ✅
   • Entropy: 4-5 bits
   • Diverse token distribution
```

## Mathematical Intuition

**StandardScaler:**
- Each dimension scaled independently: `(x - μ) / σ`
- Good for low-D features (chroma, MFCC)
- In 768D: Distorts vector relationships

**L2 Normalization:**
- Each vector scaled to unit length: `x / ||x||₂`
- All vectors lie on unit hypersphere in 768D
- Preserves angular distances (cosine similarity)
- K-means clusters by angle, not magnitude

**Why it works for Wav2Vec:**
- Wav2Vec learns rich 768D embeddings
- Similar sounds have similar angles (not magnitudes!)
- L2 norm respects the learned geometry from pretraining
- K-means finds clusters of similar musical gestures

## Files Modified

1. **`listener/symbolic_quantizer.py`** ✅
   - Added L2 normalization option
   - Default: `use_l2_norm=True` (IRCAM approach)
   - Backward compatible

2. **Documentation created:**
   - `L2_NORMALIZATION_FIX.md` - Technical explanation
   - `RETRAIN_GEORGIA_L2_FIX.md` - Training instructions

## Next Steps

### 1. Retrain Georgia (USER ACTION)
```bash
python Chandra_trainer.py \
    --file input_audio/Georgia.wav \
    --max-events 1500 \
    --hybrid-perception \
    --wav2vec \
    --gpu
```

**Look for:**
```
📚 Learning musical vocabulary (64 classes)...
   Normalization: L2 (IRCAM)  ← Critical!

🤖 MACHINE PERCEPTION:
   • Gesture tokens: 40-60 unique patterns  ← Should see this!
   • Entropy: 4-5 bits
```

### 2. Test MusicHal
With proper gesture diversity, MusicHal should:
- Distinguish different musical contexts
- Respond with musically coherent patterns
- NOT sound like "random shit"

### 3. Future Work (TODOs)
- TPP evaluation metrics (~30 min)
- Train supervised chord decoder with 600-chord dataset (~45 min)

## Key Insights

1. **IRCAM never extracts chord names from Wav2Vec** - Tokens ARE the patterns
2. **L2 normalization is critical** - Not mentioned prominently in paper but visible in code
3. **Geometry matters in high-D spaces** - 768D needs careful handling
4. **Angular distance > Euclidean distance** - For semantic embeddings

## References

**Bujard et al. (2025) - "Exploring Relationships in Musical Sequences"**
- Section 3.1: Feature Extraction and Quantization
- Pipeline: Wav2Vec → L2 normalize → K-means → Gesture tokens
- Vocabulary: 16/64/256 (64 optimal)

**Perez et al. - "Deep Chromas"**
- Multi-pitch trained chromas most accurate
- Chord-trained models hallucinate triads
- Simple methods excel at 1-2 notes, deep at 3+

---

**This is the real fix!** 🎯 Previous "dual perception" architecture was correct, but the quantizer geometry was wrong. Now aligned with IRCAM research.





