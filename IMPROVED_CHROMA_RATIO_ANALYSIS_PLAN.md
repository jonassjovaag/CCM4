# Improved Chroma-Based Ratio Analysis - Research-Informed Plan

## Summary of New Research

Based on 4 new papers added to References:

### 1. Sparse Chroma Estimation (Kronvall et al., 2015)
**Key Insights:**
- Standard FFT-based chroma suffers from **tone ambiguity**
- **Block sparse reconstruction** improves accuracy by accounting for harmonic structure
- Exploits the fact that musical tones have harmonic peaks at integer multiples
- Uses convex optimization (ADMM) for sparse chroma extraction

**Relevance:** Our current implementation uses librosa's basic chroma_stft, which is FFT-based and prone to ambiguity

### 2. Sparse Chroma for Non-Stationary Signals (Juhlin et al., 2015)
**Key Insights:**
- Time-varying envelopes require **amplitude modulation** in dictionary
- Better for percussive instruments (trumpet, trombone) with sudden bursts
- Can model longer frames while retaining time-localized information
- Spline-based amplitude modulation improves modeling

**Relevance:** Our chord playback has attack transients that could confuse standard chroma

### 3. Learning Optimal Features (Joder et al., 2013)
**Key Insights:**
- Template-based matching can be **learned** rather than heuristic
- Symmetric Kullback-Leibler divergence outperforms other distance functions
- CQT-based representation + spectrogram both work well
- Learned mappings outperform classic "canonical mapping"

**Relevance:** We're using heuristic templates in ratio_analyzer - could be learned

### 4. Temporal Correlation SVM (Rao et al., 2016)
**Key Insights:**
- **Robust PCA (RPCA)** separates vocals from accompaniment
- Low-rank matrix = musical accompaniment, sparse matrix = vocals
- Temporal correlation among chroma features improves accuracy
- Enhanced LPCP (Logarithmic PCP) from low-rank component

**Relevance:** We're getting harmonic noise - RPCA could clean this up

---

## Current Issues in Our Implementation

### Issue 1: Detecting Extra Harmonics
**Current output:**
```
Pitch classes detected: [ 0  1  3  4  7 11]  # C, C#, D#, E, G, B
   C: 0.991    ✓ (correct)
   C#: 0.287   ✗ (harmonic noise)
   D#: 0.311   ✗ (harmonic noise)
   E: 0.595    ✓ (correct)
   G: 0.300    ✓ (correct)
   B: 0.365    ✗ (harmonic noise)
```

**Root cause:** Basic FFT chroma doesn't account for harmonic structure properly

### Issue 2: Simple Thresholding
**Current approach:**
```python
threshold = 0.4 * np.max(chroma_mean)  # Simple fixed threshold
```

**Problem:** Can't distinguish between fundamental pitches and their harmonics

### Issue 3: No Temporal Smoothing
We collect 2.5 seconds of audio but don't exploit temporal correlation

### Issue 4: Octave-Folded Frequencies
We convert pitch classes to frequencies in C4 octave, losing octave information

---

## Proposed Improvements (Priority Order)

### Phase 1: Harmonic-Aware Chroma (HIGH PRIORITY)

**Implement sparse/harmonic chroma extraction:**

```python
class HarmonicAwareChroma:
    """
    Improved chroma extraction accounting for harmonic structure
    Based on Kronvall et al. (2015) and Juhlin et al. (2015)
    """
    
    def __init__(self):
        # Harmonic weights: fundamental gets highest weight
        # Harmonics get progressively lower weights
        self.harmonic_weights = {
            1: 1.0,      # Fundamental
            2: 0.3,      # Octave (important but suppress)
            3: 0.15,     # Fifth (2:3 ratio)
            4: 0.1,      # 2nd octave
            5: 0.05,     # Major third region
        }
    
    def extract_chroma(self, audio, sr=44100):
        """
        Extract chroma with harmonic awareness
        
        Strategy:
        1. Get full spectrum
        2. For each pitch class, identify fundamental vs harmonics
        3. Weight fundamentals higher than harmonics
        4. Use sparse reconstruction to resolve ambiguity
        """
        
        # Use CQT (Constant-Q Transform) instead of STFT
        # CQT has logarithmic frequency spacing (like musical notes!)
        cqt = librosa.cqt(audio, sr=sr, n_bins=84, bins_per_octave=12)
        cqt_mag = np.abs(cqt)
        
        # Identify harmonic relationships
        chroma_harmonic = np.zeros(12)
        
        for pitch_class in range(12):
            # Collect energy from this pitch class across octaves
            fundamental_energy = 0
            harmonic_energy = 0
            
            for octave in range(7):  # 7 octaves
                bin_idx = pitch_class + octave * 12
                if bin_idx < cqt_mag.shape[0]:
                    energy = np.mean(cqt_mag[bin_idx, :])
                    
                    # Is this a fundamental or harmonic?
                    if octave == 0:  # Lowest octave - likely fundamental
                        fundamental_energy += energy * self.harmonic_weights[1]
                    else:
                        # Could be harmonic of lower note
                        # e.g., G4 could be 3rd harmonic of C3
                        weight = self.harmonic_weights.get(octave + 1, 0.01)
                        harmonic_energy += energy * weight
            
            # Total weighted energy for this pitch class
            chroma_harmonic[pitch_class] = fundamental_energy + harmonic_energy * 0.2
        
        return chroma_harmonic / (np.sum(chroma_harmonic) + 1e-9)
```

**Expected improvement:** Reduce false pitch class detections by 50-70%

### Phase 2: Temporal Correlation (MEDIUM PRIORITY)

**Use temporal smoothing inspired by Rao et al. (2016):**

```python
def extract_chroma_temporal(self, audio, sr=44100, hop_length=512):
    """
    Extract chroma with temporal correlation
    
    Instead of averaging over time, preserve temporal structure
    then use correlation to identify stable chord tones
    """
    
    # Get time-series of chroma
    chroma_time = librosa.feature.chroma_cqt(
        y=audio,
        sr=sr,
        hop_length=hop_length
    )  # Shape: (12, n_frames)
    
    # For each pitch class, measure temporal stability
    chroma_stability = np.zeros(12)
    
    for pc in range(12):
        time_series = chroma_time[pc, :]
        
        # Stable tones have:
        # 1. High mean energy
        # 2. Low variance
        # 3. Consistent presence across frames
        
        mean_energy = np.mean(time_series)
        consistency = np.sum(time_series > 0.2 * np.max(time_series)) / len(time_series)
        variance = np.var(time_series)
        
        # Stability score: high mean, low variance, high consistency
        chroma_stability[pc] = mean_energy * consistency / (variance + 0.01)
    
    # Normalize
    return chroma_stability / (np.sum(chroma_stability) + 1e-9)
```

**Expected improvement:** Better distinguish between sustained chord tones and transients

### Phase 3: Robust PCA for Noise Reduction (MEDIUM PRIORITY)

**Implement RPCA as in Rao et al. (2016):**

```python
def apply_rpca(self, audio, sr=44100):
    """
    Use Robust PCA to separate:
    - Low-rank component = sustained harmonic content (what we want)
    - Sparse component = transients, noise, percussive (suppress)
    
    This is especially useful if there's background noise or artifacts
    """
    
    # Convert to spectrogram
    S = librosa.stft(audio)
    S_mag = np.abs(S)
    
    # Apply RPCA
    from sklearn.decomposition import PCA
    # (Simplified - real RPCA needs iterative optimization)
    
    # Low-rank approximation
    pca = PCA(n_components=min(20, S_mag.shape[0] // 2))
    S_lowrank = pca.inverse_transform(pca.fit_transform(S_mag.T)).T
    
    # Sparse component = residual
    S_sparse = S_mag - S_lowrank
    
    # Use low-rank for chroma (sustained harmonic content)
    return librosa.istft(S_lowrank * np.exp(1j * np.angle(S)))
```

**Expected improvement:** Cleaner signal with less noise/artifacts

### Phase 4: Learned Templates (LOW PRIORITY - FUTURE)

**Implement learned chord templates as in Joder et al. (2013):**

Instead of fixed ratio templates (4:5:6 for major), learn optimal templates from data.

This is more complex and requires training data, so defer to future work.

---

## Implementation Strategy

### Step 1: Add Harmonic-Aware Chroma (This Week)

**File:** `listener/harmonic_chroma.py`

```python
class HarmonicAwareChromaExtractor:
    """
    Extract chroma with harmonic structure awareness
    Reduces ambiguity between fundamentals and harmonics
    """
    
    def extract(self, audio, sr=44100):
        # CQT-based with harmonic weighting
        # Returns clean 12-dimensional chroma
        pass
```

**Integration:** Update `ratio_based_chord_validator.py`:
```python
from listener.harmonic_chroma import HarmonicAwareChromaExtractor

# In __init__:
self.chroma_extractor = HarmonicAwareChromaExtractor()

# In validate_chord:
chroma_clean = self.chroma_extractor.extract(audio_array, self.sample_rate)
detected_freqs = self._chroma_to_frequencies(chroma_clean, threshold=0.3)
```

### Step 2: Add Temporal Analysis (Next Week)

Extend `HarmonicAwareChromaExtractor` with temporal correlation method.

### Step 3: Add RPCA (Optional, If Needed)

Only if we're still getting noise after Steps 1-2.

---

## Expected Results

### Before (Current):
```
Detected: C, C#, D#, E, G, B (6 notes)
Expected: C, E, G (3 notes)
Match: 50% ✗
```

### After Phase 1:
```
Detected: C, E, G (3 notes)
Expected: C, E, G (3 notes)
Match: 100% ✓
```

### After Phase 2:
```
Detected: C, E, G (high confidence, stable)
Transients suppressed
Match: 100% ✓ with higher confidence
```

---

## Integration with Ratio Analysis

Once we have clean chroma (only fundamentals, no harmonics):

1. **Convert chroma to frequencies** (already doing this)
2. **Calculate ratios** (already doing this with `ratio_analyzer.py`)
3. **Match to chord templates** (already doing this)

The ratio analysis part is good! The issue was in the chroma extraction feeding into it.

**Flow:**
```
Audio → Harmonic-Aware Chroma → Pitch Classes → Frequencies → Ratios → Chord Type
         [NEW: Phase 1-2]        [CLEAN]         [CLEAN]      [WORKS]   [WORKS]
```

---

## Testing Plan

### Test 1: Basic Triads
```python
test_chords = [
    ([60, 64, 67], "C major", [0, 4, 7]),      # Should detect ONLY C, E, G
    ([60, 63, 67], "C minor", [0, 3, 7]),      # Should detect ONLY C, Eb, G
    ([60, 64, 68], "C aug", [0, 4, 8]),        # Should detect ONLY C, E, G#
]
```

**Success criteria:** Detect exactly 3 pitch classes matching expected

### Test 2: Seventh Chords
```python
test_chords = [
    ([60, 64, 67, 70], "C7", [0, 4, 7, 10]),   # 4 notes
    ([60, 64, 67, 71], "Cmaj7", [0, 4, 7, 11]), # 4 notes
]
```

**Success criteria:** Detect exactly 4 pitch classes

### Test 3: Inversions
Use the 6 inversions from `autonomous_chord_trainer.py` for each chord type.

**Success criteria:** Same chord type detected regardless of inversion

---

## Key References for Implementation

### Harmonic Structure
- Kronvall et al. (2015): Block sparse reconstruction
- Formula for harmonic weighting: `weight[n] = 1 / n` where n is harmonic number

### CQT vs STFT
- Use `librosa.cqt()` instead of `librosa.stft()`
- CQT = Constant-Q Transform (logarithmic frequency spacing)
- Better for music because notes are logarithmically spaced

### Temporal Correlation
- Rao et al. (2016): Temporal correlation SVM
- Look at consistency across frames, not just mean energy

### RPCA
- Rao et al. (2016): Robust PCA for voice separation
- Low-rank = accompaniment, Sparse = vocals/noise
- Can use `sklearn` or implement iterative optimization

---

## Timeline

- **Week 1:** Implement harmonic-aware chroma extraction
- **Week 2:** Add temporal correlation analysis
- **Week 3:** Test with full chord vocabulary (inversions, 7ths, extended)
- **Week 4:** Integrate with autonomous trainer, evaluate improvements

---

## Success Metrics

### Current Performance:
- Detection accuracy: ~50% (detecting 6 notes instead of 3)
- False positives: 50% (3 extra notes)
- Chord type recognition: 0% (unknown due to extra notes)

### Target Performance After Phase 1:
- Detection accuracy: ≥90% (detect correct number of notes)
- False positives: ≤10% (maybe 1 extra note occasionally)
- Chord type recognition: ≥80% (correctly identify major/minor/etc.)

### Target Performance After Phase 2:
- Detection accuracy: ≥95%
- False positives: ≤5%
- Chord type recognition: ≥90%
- Inversion invariance: 100% (same chord type regardless of inversion)

---

## Conclusion

The core issue is **chroma extraction, not ratio analysis**. The ratio-based approach is sound, but it needs clean input. By implementing harmonic-aware chroma extraction based on recent research (Kronvall et al., Juhlin et al., Rao et al.), we can dramatically improve accuracy while keeping the mathematically-grounded ratio analysis intact.

The research shows us that:
1. **Block sparse reconstruction** resolves tone ambiguity
2. **Harmonic weighting** distinguishes fundamentals from overtones
3. **Temporal correlation** improves stability
4. **CQT** is better than FFT for musical signals

This is the path forward for a production-ready ratio-based chord validator!

