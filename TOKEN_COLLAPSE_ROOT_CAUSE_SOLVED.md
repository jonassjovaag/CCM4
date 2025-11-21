# Token Collapse Root Cause - SOLVED

## Problem Summary

During singing test (97 events, A#2 → G3 melodic line):
- **All input mapped to token 16** (gesture diversity 0.17)
- System technically functional but perceptually blind
- RhythmOracle returned 0 patterns (separate issue)
- AudioOracle interval extraction crashed (separate issue)

## Root Cause: MODEL MISMATCH

**Training environment:**
- Vocabulary trained with `facebook/wav2vec2-base` (default in legacy Chandra_trainer.py)
- 500 frames → 51/64 tokens active (0.80 diversity - EXCELLENT!)
- Token 16 appeared only 4.6% of the time during training

**Live performance environment:**
- MusicHal defaults to `m-a-p/MERT-v1-95M` (line 3525 in MusicHal_9000.py)
- MERT is music-optimized transformer (better for music than general Wav2Vec)
- BUT vocabulary was NOT trained on MERT features!

**Feature distribution mismatch:**
- Wav2Vec (facebook/wav2vec2-base): General audio embeddings
- MERT (m-a-p/MERT-v1-95M): Music-specific embeddings, different learned representations
- Same 768D output dimension BUT different statistical distributions
- PCA trained on Wav2Vec features collapses when given MERT features

**Evidence:**
1. Vocabulary diagnostics showed 64/64 tokens active, good cluster separation (min 0.20)
2. Reapplying trained vocabulary to training features: 51/64 tokens (0.80 diversity) ✅
3. Random Gaussian features: collapse to token 7 (not musical features)
4. MERT features during live performance: collapse to token 16 (outside training distribution)

## Solutions

### Option A: Retrain Vocabulary with MERT (RECOMMENDED)

**Why:** MERT is superior for music (trained on 160K hours of music, understands harmony/rhythm/timbre)

**Steps:**
1. Retrain vocabulary using MERT-v1-95M encoder:
   ```bash
   python Chandra_trainer.py \
       --file input_audio/General_idea.wav \
       --output JSON/General_idea.json \
       --wav2vec-model m-a-p/MERT-v1-95M \
       --max-events 10000
   ```

2. This will create new vocabularies:
   - `input_audio/General_idea_harmonic_vocab.joblib` (MERT features)
   - `input_audio/General_idea_percussive_vocab.joblib` (MERT features)

3. Live performance will automatically use MERT (already default)

**Expected result:** Token diversity during live performance should match training (0.80, 51/64 tokens active)

### Option B: Use Wav2Vec During Live Performance (QUICK FIX)

**Why:** Match training environment without retraining (faster but suboptimal)

**Steps:**
1. Run MusicHal with matching model:
   ```bash
   python MusicHal_9000.py \
       --wav2vec-model facebook/wav2vec2-base \
       --performance-duration 10
   ```

2. No retraining needed - uses existing vocabulary

**Expected result:** Token diversity should improve (0.80 instead of 0.17)

**Downsides:**
- Wav2Vec is general audio, not music-optimized
- Missing MERT's music-aware representations
- Chord detection, genre understanding will be weaker

## Secondary Issues (Also Found)

### 1. RhythmOracle Density Mismatch
- **Problem:** Query density 0.5 vs trained density 4.2-14.4 onsets/sec (8-28x difference)
- **Impact:** Pattern similarity matching returns 0 results
- **Solution:** Normalize density in query or adjust similarity threshold

### 2. AudioOracle Interval Extraction Crash
- **Problem:** "NoneType has no len()" when Oracle returns frame_id
- **Impact:** Falls back to random generation instead of using learned patterns
- **Solution:** Add null check before accessing audio_frames[frame_id] data

## Implementation Priority

1. **FIX MODEL MISMATCH** (this document) - CRITICAL
   - Either retrain with MERT OR run live with Wav2Vec
   - Restores perceptual diversity from 0.17 to 0.80

2. **Fix RhythmOracle density** - HIGH
   - Normalize density before similarity check
   - Enables rhythmic intelligence

3. **Fix interval extraction** - HIGH  
   - Add null check in phrase_generator.py
   - Enables learned pattern generation

## Test Plan

1. Retrain vocabulary with MERT
2. Run singing test again (same A#2 → G3 melody)
3. Expected outcome:
   - Gesture diversity: 0.70-0.80 (instead of 0.17)
   - Unique tokens: 40-50 (instead of 6)
   - Pattern matching: Active (instead of fallback to random)
   - Musical coherence: HIGH (system can distinguish gestures)

## Files Modified

- `diagnose_performance_issues.py` - Diagnostic script (vocabulary health, frame availability, rhythm patterns)
- `test_live_token_extraction.py` - Live token extraction simulation
- `root_cause_token16_analysis.py` - Deep analysis of token 16 characteristics
- `listener/dual_perception.py` - Added debug logging (lines 244-256)

## Key Learnings

1. **Vocabulary health != Live performance success**
   - Vocabulary can be perfectly trained on one model's features
   - But completely fail on another model's features
   - Always verify: training model == live model

2. **Random Gaussian features are NOT a valid test**
   - They don't match any real audio encoder distribution
   - Use real training features for validation

3. **Token collapse has multiple signatures**
   - Random features → token 7 (not in training distribution)
   - MERT features → token 16 (partially overlaps but wrong scale)
   - Training features → 51 diverse tokens (correct)

4. **Model naming can be misleading**
   - Both Wav2Vec and MERT output 768D features
   - But distributions are incompatible
   - Must track which model was used during training

## Recommendation

**RETRAIN WITH MERT** (Option A). Reasons:
1. MERT is designed for music (Wav2Vec is general audio)
2. Future-proof: MERT is the better choice long-term
3. Enables music-aware perceptual features
4. Only requires one training run (~30 min for General_idea.wav)

After retraining, test with singing again. If diversity improves to 0.70-0.80 and RhythmOracle still returns 0, then move to fixing the density normalization issue.
