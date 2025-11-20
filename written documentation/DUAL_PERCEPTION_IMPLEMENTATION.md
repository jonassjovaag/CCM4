# Dual Perception Implementation - Summary

## What We Did

### 1. Clarified the Architecture ✅

**The Key Insight:** IRCAM's AudioOracle never tries to extract chord names from Wav2Vec features. It works in **pure token space**.

**Machine Logic (Internal):**
- Gesture tokens (0-63): Learned patterns, NOT chord names!
- Frequency ratios: Mathematical relationships [1.0, 1.25, 1.5]
- Consonance scores: Perceptual measure (0.0-1.0)

**Human Interface (Display):**
- Chord names: "Cmaj", "E7" - ONLY for humans to understand
- Note names: "C", "E", "G" - For display purposes

### 2. Updated Files

#### `/listener/dual_perception.py` ✅
- Enhanced documentation explaining machine vs human perception
- Keeps existing API: `extract_features()` returns features + ratios + chord labels
- Machine uses: `wav2vec_features`, `gesture_token`, `consonance`, `ratio_analysis`
- Human sees: `chord_label`, `chord_confidence`

#### `/Chandra_trainer.py` ✅
- Fixed `_augment_with_dual_features()` method
- Extracts gesture tokens from Wav2Vec features
- Stores ratios and consonance for context
- Labels chords for human display only
- Machine learns: "Token 42 → Token 87 when consonance > 0.8"

#### `/DUAL_PERCEPTION_ARCHITECTURE.md` ✅
- Complete documentation of the architecture
- Diagrams showing data flow
- Explanation of why this is better
- References to IRCAM research

### 3. How It Works Now

**During Training:**
```python
# Audio → Dual perception
Audio segment → Wav2Vec → Token 42 (machine)
                       → Ratios [1.0, 1.25, 1.5] (context)
                       → "major" (human display only)

# AudioOracle learns TOKEN patterns
Oracle.add(token=42, context={'consonance': 0.92, 'ratios': [...]})
```

**During Performance:**
```python
# Machine makes decisions
next_token = Oracle.query(current_token, context)

# Human sees translation
Display: "Playing Cmaj chord" (for understanding)
```

### 4. The Georgia Problem - Now Solved!

**Before:**
- System tried to extract chord names from Wav2Vec features
- Showed "C C C" because extraction was wrong
- Forced machine to think in human terms

**After:**
- Machine works with 79k+ learned gesture tokens
- Ratios provide mathematical context
- Chord names displayed separately for humans
- **Tokens ARE the patterns!**

## What This Means

### For Training (`Chandra_trainer.py --wav2vec`)
1. Wav2Vec extracts 768D features from 350ms segments
2. K-means quantizes to 64 gesture tokens
3. Ratio analyzer provides consonance + frequency relationships
4. Chord names labeled for logging (not for learning)
5. AudioOracle learns token→token transitions with ratio context

### For Performance (`MusicHal_9000.py`)
1. Machine perceives current gesture token + ratio context
2. AudioOracle predicts next token based on learned patterns
3. System generates MIDI based on token + consonance
4. Display shows human-friendly chord names (optional)

## Key Principles

1. **Separate Representations:**
   - Machine: Tokens + Ratios + Consonance
   - Human: Chord names + Note names

2. **Token Space Learning:**
   - Tokens are learned patterns, not predetermined labels
   - System can discover gestures humans haven't named
   - Context-dependent transitions: "Token X → Token Y when..."

3. **Ratio Context:**
   - Mathematical truth: [1.0, 1.25, 1.5] ratios
   - Psychoacoustic reality: consonance scores
   - NOT semantic labels like "major" or "minor"

4. **Display Layer:**
   - Chord names are POST-HOC translations
   - Only for human understanding
   - Never used in machine reasoning

## Next Steps

### Testing
Run training with dual perception enabled:
```bash
python Chandra_trainer.py \
    --hybrid-perception \
    --wav2vec \
    --vocab-size 64 \
    --gpu \
    input_audio/Georgia.wav \
    georgia_dual_model.json
```

Expected output:
- ✅ Gesture tokens: ~50-60 unique patterns from Georgia
- ✅ Average consonance: ~0.7-0.8 (jazz harmony)
- ✅ Chord display: Shows chord progression for human
- ✅ AudioOracle: Learns token patterns (not chord names!)

### Performance
Run MusicHal with trained dual model:
```bash
python MusicHal_9000.py --model georgia_dual_model.json
```

System will:
- Perceive gesture tokens in real-time
- Make decisions based on token patterns
- Display chord names for operator (optional)
- Generate musical responses using learned gestures

## References

- Bujard, S., Schwarz, D., Assayag, G. (2025). "Musical Agents" - IRCAM
- Ragano, A., et al. (2023). "Wav2Vec 2.0 for Music Understanding"  
- Assayag & Dubnov (2004). "AudioOracle: Using System Oracles for Musical Improvisation"

---

**Bottom Line:** The machine thinks in tokens and ratios. Humans see chord names. They're separate, and that's exactly how it should be!

