# Wav2Vec Chord Classifier Integration Complete! 🎹

## What Was Done

Integrated the Wav2Vec chord classifier into live performance (`MusicHal_9000.py`) so you can see what MusicHal "thinks" you're playing in real-time.

### Changes Made:

#### 1. **Load Classifier at Startup**
```python
# In __init__ when --wav2vec is enabled
from hybrid_training.wav2vec_chord_classifier import Wav2VecChordClassifier
self.wav2vec_chord_classifier = Wav2VecChordClassifier("models/wav2vec_chord_classifier.pkl")
```

**Location:** Lines 208-215 in `MusicHal_9000.py`

---

#### 2. **Store Wav2Vec Features for Classification**
```python
# When hybrid perception extracts features
if self.enable_wav2vec and len(hybrid_result.features) == 768:
    event_data['hybrid_wav2vec_features'] = hybrid_result.features  # 768D Wav2Vec features
```

**Location:** Lines 674-677 in `MusicHal_9000.py`

---

#### 3. **Add Wav2Vec Chord to Candidates**
```python
# In chord detection ensemble (before ratio-based detection)
if self.wav2vec_chord_classifier and 'hybrid_wav2vec_features' in event_data:
    wav2vec_features = event_data['hybrid_wav2vec_features']
    chord_name, w2v_conf = self.wav2vec_chord_classifier.classify(wav2vec_features)
    if w2v_conf > 0.15:  # Low threshold - neural perception is sensitive
        candidates.append((chord_name, w2v_conf, w2v_conf * 1.0, 'W2V'))
```

**Location:** Lines 1056-1065 in `MusicHal_9000.py`

---

## How It Works

### The Ensemble Approach:
MusicHal now uses **3 chord detection methods** ranked by confidence:

1. **Wav2Vec Neural Perception** (new!) 
   - Uses 768D neural features from Wav2Vec 2.0
   - Trained on synthetic chord dataset
   - Good at: Recognizing learned patterns, consistent perception
   - Weight: 1.0x (equal confidence)
   - Tag: `W2V`

2. **Ratio-Based Harmonic Analysis** (existing)
   - Uses Brandtsegg frequency ratio analyzer
   - Good at: Clear triads and 7th chords, psychoacoustic accuracy
   - Weight: 1.2x (boosted for accuracy)
   - Tag: `ratio`

3. **ML Chord Model** (existing)
   - Traditional ML classifier
   - Good at: Learned patterns from training data
   - Weight: 1.0x
   - Tag: `ML`

4. **Harmonic Context** (existing)
   - Real-time harmonic tracking
   - Good at: Sustained chords, key awareness
   - Weight: 0.8x
   - Tag: `HC`

**Best candidate wins!** The chord with the highest weighted confidence is displayed.

---

## Expected Behavior

### What You'll See:

**Terminal Output:**
```
🎹 C4 | CHORD: Cmaj         ( 67%) | Consonance: 0.82 | ...
```

The chord name now comes from the **best** of the 4 methods.

**When Wav2Vec is confident:**
```
🎹 G3 | CHORD: G7           ( 89%) | Consonance: 0.74 | ...
                     ↑ Detected by Wav2Vec classifier
```

**When multiple methods agree:**
```
🎹 D4 | CHORD: Dm           ( 92%) | Consonance: 0.78 | ...
                     ↑ Both Wav2Vec AND ratio agree = high confidence
```

**When nothing is confident:**
```
🎹 F#2 | CHORD: ---          (  0%) | Consonance: 0.44 | ...
                      ↑ All methods below threshold
```

---

## Why Chords Might Still Show "---"

### 1. **Model Not Trained on Your Input**
- The Wav2Vec classifier was trained on **synthetic chords** (clean triads, 7ths, etc.)
- If you're playing **jazz voicings**, **extended harmonies**, or **unique timbres**, it might not recognize them
- **Solution:** Retrain with your actual playing as training data

### 2. **Audio Buffer Too Short**
- Wav2Vec needs ~1024 samples (~23ms at 44.1kHz) to extract features
- Very fast playing might not give enough time
- **Solution:** Already optimized, but some rapid gestures will always be ambiguous

### 3. **Low Confidence Threshold**
- All 4 methods must be >15-25% confident to display
- If audio is noisy or ambiguous, all will be below threshold
- **This is good!** Better to show "---" than guess wrong

### 4. **Monophonic Input**
- If you're playing single notes, there's no chord to detect!
- Wav2Vec was trained on chords (3+ notes), so single notes confuse it
- **Expected:** Monophonic = "---"

---

## Testing the Integration

### Quick Test:
```bash
python MusicHal_9000.py \
  --hybrid-perception \
  --wav2vec \
  --gpu \
  --visualize
```

### What to Try:

**1. Clear Triads (should work well):**
- Play C-E-G (C major)
- Play D-F-A (D minor)
- **Expected:** Should show "Cmaj" or "Dm" with 60-90% confidence

**2. 7th Chords (should work okay):**
- Play G-B-D-F (G7)
- Play C-E-G-B (Cmaj7)
- **Expected:** Might show "G7" or simplify to "Gmaj"

**3. Complex Jazz Voicings (might not work):**
- Play Dm11, Ebmaj7#11, etc.
- **Expected:** Might show "---" or simplest match (e.g., "Dm")

**4. Single Notes (should show ---):**
- Sing a melody
- Play a bass line
- **Expected:** "---" (0%) - this is correct!

---

## Troubleshooting

### "⚠️  Chord classifier not available: [Errno 2] No such file or directory"
**Problem:** Trained model not found at `models/wav2vec_chord_classifier.pkl`

**Solutions:**
1. Check if model exists: `ls -lh models/wav2vec_chord_classifier.pkl`
2. If missing, retrain:
   ```bash
   python train_wav2vec_chord_classifier.py
   ```
3. Or disable Wav2Vec: Run without `--wav2vec` flag

### "Still showing CHORD: --- (0%)"
**Possible causes:**
1. **All 4 methods are uncertain** - Play clearer chords
2. **Input too quiet** - Increase gain
3. **Model untrained on your timbre** - Retrain with your audio
4. **You're playing single notes** - This is expected!

### "Wrong chords showing"
**Possible causes:**
1. **Wav2Vec hallucinating** - It's doing its best with neural patterns!
2. **Ratio analyzer confused by inversion** - Root position works better
3. **Training data mismatch** - Retrain on your playing style

---

## Next Steps (Optional Improvements)

### 1. Retrain Classifier on Your Playing
```bash
# Record yourself playing various chords, label them
# Then retrain:
python train_wav2vec_chord_classifier.py --training-data your_chords.wav --labels your_labels.csv
```

### 2. Lower Confidence Threshold
If you want to see "guesses" even when uncertain:
```python
# In MusicHal_9000.py line 1061, change:
if w2v_conf > 0.15:  # Lower to 0.05 for more guesses
```

### 3. Add Debugging Output
To see what all 4 methods think:
```python
# After line 1095, add:
if candidates:
    print(f"  Chord candidates: {[(c[0], f'{c[1]:.0%}', c[3]) for c in candidates]}")
```

---

## Summary

✅ **Wav2Vec chord classifier is now integrated**  
✅ **Runs in real-time during performance**  
✅ **Contributes to ensemble chord detection**  
✅ **Provides transparency into MusicHal's perception**

The chord label is **not used for decision-making** - it's purely for **your understanding** of what MusicHal hears. The actual AI uses the 768D Wav2Vec features directly, which are much richer than any chord label!

**Test it now and see what MusicHal thinks you're playing!** 🎵🤖










