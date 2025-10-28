# Dual Perception Architecture

## The Key Insight

**The machine should NOT think in chord names!**

IRCAM's AudioOracle works in **pure token space** - it never tries to extract chord names like "Cmaj" or "E7". Instead:

- **Gesture tokens** (0-63) ARE the learned musical patterns
- **Mathematical ratios** provide psychoacoustic context
- **Chord names** are ONLY for humans to understand what's happening

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      AUDIO INPUT                             │
└────────────────┬────────────────────────────────────────────┘
                 │
        ┌────────┴────────┐
        │                 │
        ▼                 ▼
┌───────────────┐  ┌──────────────────┐
│   Wav2Vec     │  │ Ratio Analyzer   │
│   Encoder     │  │ (Mathematical)   │
└───────┬───────┘  └────────┬─────────┘
        │                   │
        ▼                   ▼
   768D features      [1.0, 1.25, 1.5]
        │              Consonance: 0.92
        │                   │
        ▼                   │
   ┌──────────┐             │
   │ Quantizer│             │
   └─────┬────┘             │
         │                  │
         ▼                  │
      Token 42              │
         │                  │
         └────────┬─────────┘
                  │
      ┌───────────▼───────────┐
      │   MACHINE PERCEPTION  │
      │  (What AI works with) │
      │                       │
      │  • Token: 42          │
      │  • Ratios: [1.0, ...] │
      │  • Consonance: 0.92   │
      └───────────┬───────────┘
                  │
                  │ AudioOracle learns:
                  │ "Token 42 → Token 87"
                  │ "when consonance > 0.8"
                  │
         ┌────────▼────────┐
         │  TRANSLATION    │
         │  (Human only)   │
         └────────┬────────┘
                  │
                  ▼
         ┌───────────────────┐
         │ HUMAN INTERFACE   │
         │                   │
         │ Display: "Cmaj"   │
         │ Notes: "C, E, G"  │
         └───────────────────┘
```

## Why This Matters

### The Old (Wrong) Way
```python
# ❌ BAD: Forcing machine to use human labels
Audio → Extract chord name "Cmaj" → Learn "Cmaj follows Fmaj"
```

Problems:
- Forced to discretize into named chords
- Can't capture microtonal or spectral qualities
- Loses nuance of actual musical gestures
- Limited to Western harmony vocabulary

### The New (Correct) Way
```python
# ✅ GOOD: Machine works in token space
Audio → Wav2Vec → Token 42 → Learn "Token 42 → Token 87 when consonance > 0.8"
                → Ratios [1.0, 1.25, 1.5] for context
```

Benefits:
- Discovers patterns beyond named chords
- Learns context-dependent transitions
- Captures subtle variations
- Not limited by human vocabulary

## Implementation

### During Training (Chandra_trainer.py)

```python
# Extract dual perception
result = dual_perception.extract_features(audio, sr, timestamp, f0)

# MACHINE representation (what gets stored and learned)
event['gesture_token'] = result.gesture_token  # 0-63
event['features'] = result.wav2vec_features  # 768D
event['frequency_ratios'] = result.ratio_analysis.ratios  # [1.0, 1.25, 1.5]
event['consonance'] = result.consonance  # 0.92

# HUMAN translation (ONLY for logging/display)
event['chord_name_display'] = result.chord_label  # "major"
event['chord_confidence'] = result.chord_confidence  # 0.95
```

### During Performance (MusicHal_9000.py)

```python
# Machine makes decisions based on TOKENS + RATIOS
next_token = audio_oracle.query(current_token, context={'consonance': 0.8})

# Human sees translation
print(f"Playing gesture token {next_token}")  # For debugging
print(f"(Human: This is a {chord_name} chord)")  # For understanding
```

## The Georgia Problem Solved

**Before:** System showed "C C C" because it was trying to extract chord NAMES from Wav2Vec features

**After:** System works with:
- **79,000+ gesture tokens** learned from audio
- **Ratio context** for musical relationships  
- **Chord names** displayed separately for humans

The tokens ARE the patterns! They don't need names.

## Key Principles

1. **Machine Logic:** Tokens + Ratios + Consonance
2. **Human Interface:** Chord names for display
3. **Separation:** Machine never sees "Cmaj", only Token 42
4. **Context:** Ratios provide mathematical context for token transitions
5. **Discovery:** Machine can learn patterns humans haven't named

## References

- Bujard et al. (2025) - IRCAM Musical Agents and AudioOracle
- Ragano et al. (2023) - Wav2Vec 2.0 for music representation
- Assayag & Dubnov (2004) - AudioOracle and symbolic learning

---

**Bottom Line:** Let the machine think like a machine. Let humans see human labels. Keep them separate.

