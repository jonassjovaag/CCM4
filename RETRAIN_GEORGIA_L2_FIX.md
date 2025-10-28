# Retrain Georgia with L2 Normalization Fix - Instructions

## Update: Critical Bug Fixed! 🎯

**Problem:** Previous training only learned **1 unique gesture token** because we used StandardScaler instead of L2 normalization.

**Fix:** Updated `symbolic_quantizer.py` to use **L2 normalization** (IRCAM approach) by default.

**Expected:** Should now learn **40-60 unique gesture tokens** instead of 1!

## What Was Wrong

Your Georgia model (`JSON/Georgia_081025_2259.json`) was trained with the dual perception module, but:
- ❌ K-means quantizer used StandardScaler (wrong for 768D Wav2Vec features)
- ❌ Only 1 unique token learned → No pattern diversity
- ❌ MusicHal couldn't distinguish different musical gestures → Random output

## The Solution

**We've now fixed the quantizer** to use L2 normalization (aligns with IRCAM paper):
- ✅ Wav2Vec features (768D) → L2 normalize → K-means
- ✅ All vectors lie on unit hypersphere (better geometry for clustering)
- ✅ Angular relationships preserved → Better musical pattern discovery

See `L2_NORMALIZATION_FIX.md` for technical details.

## Retrain Command

```bash
cd /Users/jonashsj/Jottacloud/PhD\ -\ UiA/CCM3/CCM3

# Train with fixed quantizer (L2 norm is now default)
python Chandra_trainer.py \
    --file input_audio/Georgia.wav \
    --max-events 1500 \
    --hybrid-perception \
    --wav2vec \
    --gpu
```

## What to Look For

### ✅ Success Indicators:

```
🔬 Initializing Dual Perception Module...
   Normalization: L2 (IRCAM)  ← LOOK FOR THIS!

🤖 MACHINE PERCEPTION (What AI learns):
   • Gesture tokens: 40-60 unique patterns  ← NOT "1"!
   • Entropy: 4-5 bits  ← NOT "0.00"!

👤 HUMAN INTERFACE (What humans see):
   • Chord labels: Cmaj, G7, Am, etc.  ← Should show variety!
```

### ❌ Signs of Failure (same as before):

```
   • Gesture tokens: 1 unique patterns  ← STILL BROKEN!
   • Entropy: 0.00 bits
   • Chord labels: ["C", "C", "C"]
```

If you still see only 1 token, something else is wrong and we need to investigate further.

## After Training

Test MusicHal_9000 with the new model:

```bash
# Start MusicHal with new Georgia model
python MusicHal_9000.py --model JSON/Georgia_MMDD_HHMM_model.json
```

**Expected behavior:** Should respond to your playing with musically sensible gestures (not random noise).

## Technical Details

**IRCAM Research (Bujard et al., 2025):**
- Wav2Vec (768D) → **L2 normalize** → K-means → Gesture tokens
- Vocabulary: 16/64/256 (64 optimal for learning relationships)
- Preserves angular distance in high-dimensional space

**Our Fix:**
- Updated `listener/symbolic_quantizer.py` to use L2 norm by default
- Backward compatible with old models (uses StandardScaler if needed)
- Training output now shows which normalization method is used

---

**Ready to retrain!** The fix ensures proper pattern discovery from Wav2Vec features. 🎵





