# Retrain Georgia with Dual Perception - Instructions

## The Problem

Your Georgia model (`JSON/Georgia_081025_1828.json`) was trained with the **OLD version** of Chandra_trainer.py:
- ❌ Used old hybrid_perception (not dual_perception)
- ❌ Chord extraction tried to read first 12 dims of Wav2Vec as chroma → Got "C C C" everywhere
- ❌ 79k patterns learned but mislabeled → MusicHal plays random sounds

## The Solution

We've now fixed Chandra_trainer.py to use **proper dual perception**:
- ✅ Machine logic: Wav2Vec gesture tokens (0-63) + mathematical ratios
- ✅ Human interface: Chord names for display only (translated from ratios, not Wav2Vec)
- ✅ No more trying to extract chords from Wav2Vec features!

## Retrain Command

```bash
cd /Users/jonashsj/Jottacloud/PhD\ -\ UiA/CCM3/CCM3

python Chandra_trainer.py \
    --file input_audio/Georgia.wav \
    --hybrid-perception \
    --wav2vec \
    --gpu
```

## What You Should See (New Output)

Look for these lines during initialization:

```
🎵 Dual perception enabled:
   Machine logic: facebook/wav2vec2-base → gesture tokens (0-63)
   Machine logic: Ratio analysis → consonance + frequency ratios
   Human interface: Chord names for display only
   ✨ Tokens ARE the patterns, not chord names!
```

And during training:

```
🔬 Step 4b: Dual Perception Feature Extraction (Wav2Vec + Ratios)...
   Using temporal segmentation: XXX segments (350.0ms each)
   Machine representation: Gesture tokens + Ratios + Consonance
   Human translation: Chord labels from ratio analysis
✅ Dual perception features extracted in X.XXs
```

## What Changed in the Model

### OLD Model (Georgia_081025_1828.json):
```json
{
  "sample_harmonic_patterns": ["C", "C", "C", "C", "C"],  // ❌ Wrong!
  "feature_dimensions": 768,
  "patterns": 79765
}
```

### NEW Model (After retraining):
```json
{
  "sample_harmonic_patterns": ["Cmaj", "G7", "Fmaj", "Dm7"],  // ✅ From ratio analysis!
  "sample_gesture_tokens": [12, 45, 3, 58],  // ✅ Pure machine patterns!
  "sample_ratios": [[1.0, 1.25, 1.5], [1.0, 1.12, 1.4, 1.68]],  // ✅ Mathematical truth!
  "feature_dimensions": 768,
  "patterns": 79765
}
```

## The Pipeline Flow (After Retrain)

```
┌─────────────────────────────────────────────────┐
│ chord_ground_truth_trainer_hybrid.py            │
│ (600 chord dataset → Supervised chord decoder)  │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│ Chandra_trainer.py (FIXED!)                     │
│ Machine: Token 42 → Token 87 when ratio=[1.5]  │
│ Human: "Cmaj → G7" (from ratio analyzer)       │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│ MusicHal_9000.py                                │
│ Uses: gesture tokens + ratios (machine logic)  │
│ Displays: "Cmaj" (human interface)             │
│ Result: Musical sense!                          │
└─────────────────────────────────────────────────┘
```

## Expected Training Time

- **Step 1:** Hierarchical Analysis (~5s)
- **Step 2:** Rhythmic Analysis (~10s)
- **Step 3:** Adaptive Sampling (~5s)
- **Step 4a:** Polyphonic Analysis (~30s)
- **Step 4b:** Dual Perception (~60s) ← New step!
- **Step 5:** AudioOracle Training (~20s)
- **Step 6:** Save Model (~5s)
- **Total:** ~2-3 minutes

## After Retraining

1. **Test with MusicHal_9000:**
   ```bash
   python MusicHal_9000.py --model JSON/Georgia_081025_1828.json
   ```

2. **What you should hear:**
   - Proper harmonic responses (not random!)
   - Gesture-aware continuations
   - Ratio-informed consonance

3. **What you should see in the terminal:**
   ```
   🎵 Perceived: Gesture Token 42, Consonance 0.85, Ratio [1.0, 1.25, 1.5]
   🎸 Human translation: Cmaj7
   🤖 Prediction: Token 87 (consonance weight: 0.8)
   ```

## Troubleshooting

### If you still see "C C C":
- Check that you're using the **NEW** Chandra_trainer.py (modified Oct 8 22:35)
- Verify you see "🎵 Dual perception enabled" during init
- Verify Step 4b says "Dual Perception Feature Extraction"

### If training fails:
- Check `listener/dual_perception.py` exists
- Check imports work: `python3 -c "from listener.dual_perception import DualPerceptionModule"`
- Check MPS availability: `python3 -c "import torch; print(torch.backends.mps.is_available())"`

### If MusicHal still sounds random:
- The model might need **more diverse training data** (not just Georgia)
- Try training on multiple songs
- Consider the supervised chord decoder (chord_ground_truth_trainer_hybrid.py)

## Next Steps After Successful Retrain

1. ✅ Verify Georgia model has proper chord detection
2. 🎯 Test MusicHal_9000 live performance  
3. 📊 Implement TPP evaluation metrics
4. 🎓 Train supervised chord decoder with 600-chord dataset
5. 🎸 Integrate supervised decoder with MusicHal for even better musicality

---

**The key insight:** The machine should work in token space + ratios, NOT chord names. Chord names are just for human understanding!





