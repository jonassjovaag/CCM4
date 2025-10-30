# CCM4 System Status Report
**Date**: October 28, 2025  
**Branch**: refactoring

## 🔴 CRITICAL ISSUES (Blocking Core Functionality)

### 1. **Dtype Mismatch Bug - PARTIALLY FIXED** ⚠️
**Status**: Monkey-patch applied but still failing  
**Location**: `listener/symbolic_quantizer.py` line 150  
**Error**: `ValueError: Buffer dtype mismatch, expected 'const float' but got 'double'`

**What's Happening**:
- sklearn's `KMeans.predict()` expects float32 data
- sklearn's `normalize()` function returns float64
- The pickled quantizer (`JSON/polyphonic_audio_oracle_training_quantizer.joblib`) contains OLD code
- Monkey-patch was applied to convert dtypes, but STILL getting errors

**Root Cause**: The KMeans cluster centers themselves might be float64 in the pickled model

**Impact**:
- ❌ Gesture tokens NOT being extracted (`recent_tokens=[]` always)
- ❌ Consonance stuck at default 0.50
- ❌ Oracle falls back to random note generation instead of pattern-based
- ❌ No meaningful pattern matching
- ❌ Request parameters show None values

**Latest Fix Attempt** (lines 1868-1903 in MusicHal_9000.py):
```python
# Added conversion of kmeans cluster_centers_ to float32
if hasattr(quantizer.kmeans, 'cluster_centers_'):
    if quantizer.kmeans.cluster_centers_.dtype != np.float32:
        quantizer.kmeans.cluster_centers_ = quantizer.kmeans.cluster_centers_.astype(np.float32)
```

**Next Steps**:
1. Test if converting cluster centers fixes the issue
2. If not: Retrain the quantizer from scratch with float32 throughout
3. Alternative: Use CPU-only mode which may be more dtype-tolerant

---

### 2. **Chord Detection Showing 0% Confidence** 🔴
**Status**: Broken  
**Display**: `CHORD: --- (0%)`  

**What's Happening**:
- System reports `📝 No ML chord model found - using harmonic detection only`
- Missing file: `models/wav2vec_chord_classifier.pt`
- Harmonic fallback detection is running but returning 0% confidence

**Possible Causes**:
1. **Ratio analyzer** not finding valid chord patterns
2. **Chroma features** not being extracted correctly (blocked by dtype bug?)
3. **No audio input** - system might not be receiving actual audio
4. Fallback chord detection logic has bugs

**Related Systems**:
- `listener/ratio_analyzer.py` (Brandtsegg integration) 
- `listener/harmonic_chroma.py`
- `hybrid_training/wav2vec_chord_classifier.py` (missing trained model)

**Impact**:
- No chord-aware accompaniment
- No harmonic context for Oracle queries
- Reduced musical intelligence

---

## 🟡 SECONDARY ISSUES (Non-Critical)

### 3. **Visualization System Disabled**
**Status**: PyQt5 not available in current environment  
**Message**: `⚠️ PyQt5 not available, visualization disabled`

**What This Means**:
- Cannot see the 5 viewport windows
- Cannot debug visually what's happening
- Request parameters, pattern matching, audio analysis viewports not visible

**Fix**: Need to either:
- Install PyQt5 in the current Python environment, OR
- Switch to an environment that has PyQt5

---

### 4. **Oracle Context Parameter** ✅ FIXED
**Status**: Fixed in code, pending test  
**What Was Done**:
- Changed all 4 Oracle calls from `length=` to `max_length=`
- Added `recent_tokens = self._get_recent_human_tokens(n=3)` before each call
- Now passing `current_context=recent_tokens` as first parameter

**Files Modified**:
- `agent/phrase_generator.py` lines 500-920
- All 4 phrase generation methods updated: buildup, peak, release, contemplation

**Current Behavior**:
- Oracle queries show: `🎯 Shadow: recent_tokens=[], consonance=0.50, barlow=5.0`
- Empty `recent_tokens` because gesture token extraction is broken (dtype bug)
- Once dtype is fixed, this should work

---

## 🟢 WORKING SYSTEMS

### System Initialization ✅
- ✅ AudioOracle model loads (4960 frames, 39,355 patterns)
- ✅ Wav2Vec encoder loads (facebook/wav2vec2-base)
- ✅ MPS/GPU acceleration enabled
- ✅ MPE MIDI output working (Melody + Bass channels)
- ✅ Brandtsegg rhythm analyzer initialized
- ✅ Hybrid perception module initialized
- ✅ Behavioral modes working (IMITATE, SHADOW, EXPLORE, etc.)

### Audio Input ✅
- ✅ System is receiving audio events
- ✅ Pitch detection working (showing notes: F1, F#1, E3, B2, etc.)
- ✅ Event counter incrementing (Events: 12, 54, 82, etc.)

### MIDI Output ✅
- ✅ MIDI notes being sent: `🎵 MIDI sent: note_on channel=2 note=78 velocity=81`
- ✅ Bass voice selection logic working
- ✅ Timing between melody/bass alternation functional

### Learning Data ✅
- ✅ 1763 musical moments loaded from memory
- ✅ AudioOracle pre-trained with 4960 audio frames
- ✅ GPT-OSS behavioral insights loaded (silence strategy, role development)

---

## 📊 SYSTEM FUNCTIONALITY MATRIX

| Component | Status | Functionality |
|-----------|--------|---------------|
| **Audio Input** | 🟢 Working | Receiving audio, detecting pitches |
| **Pitch Detection** | 🟢 Working | F0 tracking functional |
| **MIDI Output** | 🟢 Working | Sending notes to IAC Driver |
| **Wav2Vec Features** | 🟡 Partial | Extracting 768D vectors but dtype issues |
| **Gesture Tokens** | 🔴 Broken | Not extracting (dtype mismatch) |
| **Consonance** | 🔴 Broken | Stuck at 0.50 default |
| **Chord Detection** | 🔴 Broken | Always 0% confidence |
| **Oracle Pattern Gen** | 🔴 Broken | Falling back to random (no tokens) |
| **Ratio Analysis** | ❓ Unknown | Brandtsegg code present but not verified |
| **Behavioral Modes** | 🟢 Working | Mode switching, requests building correctly |
| **Visualization** | 🔴 Disabled | PyQt5 not available |
| **Learning/Memory** | 🟢 Working | Loading/saving musical moments |

---

## 🎯 WHAT'S ACTUALLY WORKING RIGHT NOW

1. **Basic note generation**: System generates MIDI notes and sends them
2. **Voice alternation**: Switches between melodic and bass voices
3. **Mode system**: Behavioral modes (IMITATE, SHADOW, etc.) are active
4. **Audio perception**: Receiving audio input and detecting pitches
5. **Pre-trained memory**: AudioOracle has 4960 learned patterns available

---

## 🔥 WHAT'S BROKEN RIGHT NOW

1. **Pattern-based generation**: Oracle generates random notes instead of using learned patterns (because no gesture tokens)
2. **Harmonic awareness**: No chord detection, consonance stuck at 0.50
3. **Musical intelligence**: System can't adapt to harmonic context
4. **Gesture vocabulary**: Symbolic quantization failing due to dtype mismatch
5. **Visual feedback**: No visualization windows

---

## 🛠️ IMMEDIATE ACTION ITEMS (Priority Order)

### Priority 1: Fix Dtype Bug (CRITICAL)
**Test the latest fix**:
```bash
python MusicHal_9000.py --wav2vec --hybrid-perception --gpu --performance-duration 1
```

**Look for**:
- ❌ Still seeing: `Buffer dtype mismatch, expected 'const float' but got 'double'`?
- ✅ OR: `🎵 Gesture token: 42` (some actual token number)?

**If still broken**:
- Option A: Retrain quantizer with float32 throughout
- Option B: Test without GPU (`--no-gpu`) to see if CPU is more tolerant
- Option C: Directly fix the pickled model file

---

### Priority 2: Debug Chord Detection
**Once dtype is fixed**, investigate why chords are 0%:
1. Check if ratio analyzer is being called
2. Verify chroma features are valid
3. Add debug logging to chord detection pipeline
4. Test with known chord progressions

---

### Priority 3: Enable Visualization
**Install PyQt5**:
```bash
pip install PyQt5
```
Then can visually verify:
- Pattern matching viewport
- Request parameters viewport  
- Audio analysis viewport
- Phrase memory viewport
- Performance timeline viewport

---

## 📈 SYSTEM ARCHITECTURE (Quick Ref)

```
Audio Input
    ↓
Hybrid Perception Module
    ├─ Wav2Vec Encoder (768D features)
    ├─ Ratio Analyzer (Brandtsegg)
    ├─ Harmonic Chroma Extractor
    └─ Symbolic Quantizer ← BROKEN (dtype bug)
         ↓
    Gesture Token ← NOT WORKING
         ↓
AudioOracle Pattern Memory
    ├─ 4960 pre-trained frames
    ├─ 39,355 learned patterns
    └─ Suffix link traversal
         ↓
Request-Based Generation
    ├─ Shadow (imitation)
    ├─ Mirror (variation)
    ├─ Explore (novelty)
    └─ Challenge (tension)
         ↓
Phrase Generator
    ├─ Buildup
    ├─ Peak
    ├─ Release
    └─ Contemplation
         ↓
MIDI Output (MPE)
```

---

## 💡 DIAGNOSIS SUMMARY

**The Core Problem**: A dtype mismatch in the machine learning pipeline is blocking the entire gesture token extraction system, which cascades into:
- No pattern-based generation
- No harmonic context
- No meaningful Oracle queries
- System operating in "random fallback" mode

**The Symptom You're Seeing**: Chords at 0% is a SECONDARY symptom. The PRIMARY issue is the dtype bug preventing gesture tokens from being extracted.

**The Good News**: 
- System loads successfully
- Basic functionality works (audio in, MIDI out)
- All the sophisticated code is there and ready
- Pre-trained models are loaded
- Just need to fix the dtype conversion issue

**The Bad News**:
- Until dtype is fixed, system is essentially operating in "demo mode" without its main intelligence
- Multiple attempted fixes haven't worked yet
- May need to retrain the quantizer model from scratch

---

## 🔬 TESTING RECOMMENDATIONS

1. **Quick Test** (1 minute):
   ```bash
   python MusicHal_9000.py --wav2vec --hybrid-perception --gpu --performance-duration 1 2>&1 | grep -E "(dtype|Gesture token|consonance|CHORD)"
   ```

2. **With Visualization** (if PyQt5 installed):
   ```bash
   python MusicHal_9000.py --wav2vec --hybrid-perception --gpu --visualize --performance-duration 2
   ```

3. **CPU Fallback Test** (to see if GPU dtype strict):
   ```bash
   python MusicHal_9000.py --wav2vec --hybrid-perception --performance-duration 1
   ```

---

## 📝 FILES MODIFIED IN THIS SESSION

1. **MusicHal_9000.py**:
   - Lines 1855-1903: Added comprehensive dtype monkey-patch
   - Lines 1786: Fixed `model_loaded` initialization bug
   
2. **agent/phrase_generator.py**:
   - Lines 500-920: Fixed Oracle calls (4 locations)
   - Added `recent_tokens` parameter passing
   - Changed `length=` to `max_length=`

3. **listener/symbolic_quantizer.py**:
   - Lines 143-167: Added dtype conversions (NOT executing - pickled code)

---

## 🎓 KEY LEARNINGS

1. **Pickled models contain embedded code** - changes to source files don't affect loaded models
2. **Dtype strictness varies** - sklearn KMeans is very strict about float32 vs float64
3. **Cascading failures** - one low-level bug (dtype) breaks multiple high-level features
4. **Monkey-patching is tricky** - need to patch the RIGHT object at the RIGHT time

---

**END OF REPORT**

Next step: Test the latest dtype fix (cluster centers conversion) and report results.
