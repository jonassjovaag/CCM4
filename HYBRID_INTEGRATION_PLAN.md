# Hybrid Integration Plan - Best of Both Worlds

## Combining Our Work + IRCAM Research (Bujard et al. 2025)

### What We're Building:

**Next-generation perception system** that combines:
1. **Our ratio-based analysis** (interpretable, psycho acoustic)
2. **Our harmonic-aware chroma** (CQT + temporal correlation)
3. **IRCAM's symbolic quantization** (efficient, learnable)
4. **IRCAM's temporal segmentation** (musical gesture capture)

---

## Architecture Overview

```
Audio Input
    ↓
┌─────────────────────────────────────────────────────────┐
│  HYBRID PERCEPTION MODULE (NEW!)                        │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  [Harmonic-Aware Chroma]  (Our contribution)            │
│     • CQT-based (log frequency spacing)                 │
│     • Harmonic weighting (suppress overtones)           │
│     • Temporal correlation                              │
│     → 12D chroma vector                                 │
│                                                           │
│  [Ratio Analyzer]  (Our contribution)                   │
│     • Frequency ratios (4:5:6 for major)               │
│     • Consonance scoring (Helmholtz 1877)              │
│     • Interval analysis                                 │
│     → 10D ratio features                                │
│                                                           │
│  [Spectral Features]  (Standard)                        │
│     • RMS, centroid, rolloff, etc.                     │
│     → 6D spectral features                              │
│                                                           │
│  CONCATENATE → 28D feature vector                       │
│                                                           │
│  [Vector Quantization]  (IRCAM contribution)            │
│     • K-means (vocabulary = 64 classes)                 │
│     • Creates symbolic "musical alphabet"               │
│     → Single token ID (0-63)                            │
│                                                           │
└─────────────────────────────────────────────────────────┘
    ↓
Dual Output:
  1. Continuous features (28D) → ML Classifier
  2. Symbolic token (1D) → AudioOracle/Memory
```

---

## Implementation Status

### ✅ Completed:

1. **`listener/ratio_analyzer.py`**
   - Frequency ratio calculation
   - Consonance scoring
   - Chord type matching by ratios

2. **`listener/harmonic_chroma.py`**
   - CQT-based chroma
   - Harmonic weighting
   - Temporal correlation
   - Guided peak search

3. **`listener/symbolic_quantizer.py`**
   - K-means vector quantization
   - Vocabulary learning (16-256 classes)
   - Token encoding/decoding
   - Codebook statistics

4. **`listener/hybrid_perception.py`**
   - Integrates all three modules
   - Produces dual output (continuous + symbolic)
   - Ready for integration

### 🔄 In Progress:

5. **Integration into Chandra_trainer**
6. **Integration into MusicHal_9000**
7. **Temporal segmentation (250-500ms windows)**
8. **TPP evaluation metrics**

---

## Integration Points

### 1. Chandra_trainer Enhancement

**Current flow**:
```python
Audio file → Extract events → Train AudioOracle
```

**Enhanced flow**:
```python
Audio file → Extract events with hybrid perception
           → Features: 28D (chroma + ratio + spectral)
           → Symbolic tokens: For AudioOracle efficiency
           → Train vocabulary (K-means codebook)
           → Train AudioOracle on symbolic tokens
           → Save: model.json + vocabulary.pkl
```

**Changes needed**:
- Add hybrid perception to event extraction
- Learn symbolic vocabulary from all events
- Store both continuous features + symbolic tokens
- Use tokens for more efficient AudioOracle

### 2. MusicHal_9000 Enhancement

**Current flow**:
```python
Live audio → DriftListener (YIN + spectral)
           → AudioOracle pattern matching
           → AI Agent decision
           → MIDI output
```

**Enhanced flow**:
```python
Live audio → DriftListener (existing)
           → Hybrid Perception (NEW!)
              ├─ Chroma features
              ├─ Ratio features (consonance!)
              ├─ Symbolic token
              └─ Musical insights
           → AudioOracle pattern matching (using tokens)
           → AI Agent decision (with consonance awareness)
           → MIDI output (harmonically informed)
```

**Benefits**:
- Richer feature representation
- More efficient memory (symbolic tokens)
- Consonance-aware decision making
- Interpretable musical parameters

### 3. Ratio-Based Chord Trainer

**Already implemented** - feeds into the hybrid perception!

The chord trainer validates that our ratio approach works (100% success!),
and these features now get integrated into the full system.

---

## Key Improvements from IRCAM Paper

### 1. Symbolic Vocabulary (Implemented ✓)

**From paper**: Vocabulary size 16-64 optimal for learning

**Our implementation**:
- 64-class vocabulary (sweet spot)
- K-means clustering
- Entropy tracking
- Token frequency analysis

**Integration**:
```python
# In Chandra_trainer:
perception = HybridPerceptionModule(vocabulary_size=64)

# Extract features
for event in audio_events:
    result = perception.extract_features(event.audio)
    
    # Store both:
    continuous_features = result.features  # For ML
    symbolic_token = result.symbolic_token  # For AudioOracle
```

### 2. Temporal Segmentation

**From paper**: 250-500ms segments better than frame-by-frame

**Current**: Frame-by-frame (256 samples = 6ms)

**Enhancement**:
```python
class TemporalSegmenter:
    """Segment audio into musical gestures"""
    
    def segment(self, audio, segment_duration_ms=350):
        """
        Segment audio into uniform windows
        
        Args:
            segment_duration_ms: 250ms (improvisation) or 500ms (structured)
        """
        # Paper found 350ms works well for both!
```

### 3. Learned Relationships

**From paper**: Transformer learns Track A → Track B mappings

**Could add to MusicHal_9000**:
```python
class RelationshipLearner:
    """Learn 'when performer plays X, respond with Y'"""
    
    def train_on_duos(self, performer_track, ai_track):
        """
        Learn from recordings of performer + AI
        Similar to paper's paired tracks approach
        """
```

---

## Immediate Next Steps (Implementing Now)

1. ✅ Create `symbolic_quantizer.py` (DONE)
2. ✅ Create `hybrid_perception.py` (DONE)
3. ⏳ Add to Chandra_trainer (IN PROGRESS)
4. ⏳ Add to MusicHal_9000 (NEXT)
5. ⏳ Add temporal segmentation (NEXT)

---

## Expected Improvements

### Chandra_trainer:
- **Before**: AudioOracle with 15D continuous features
- **After**: AudioOracle with 64-class symbolic tokens + 28D features for validation
- **Benefit**: More efficient memory, clearer patterns

### MusicHal_9000:
- **Before**: Pattern matching with hand-crafted features
- **After**: Pattern matching + ratio analysis + consonance-aware decisions
- **Benefit**: More musical responses, interpretable behavior

### Chord Analyzer:
- **Before**: Standalone validation tool
- **After**: Integrated into training pipeline, provides ratio features
- **Benefit**: Unified system with mathematical grounding

---

## Files Created/Modified

### New Files:
1. `listener/symbolic_quantizer.py` (285 lines) ✅
2. `listener/hybrid_perception.py` (250 lines) ✅
3. `HYBRID_INTEGRATION_PLAN.md` (this file)

### Files to Modify:
4. `Chandra_trainer.py` - Add hybrid perception
5. `MusicHal_9000.py` - Add hybrid perception
6. `memory/polyphonic_audio_oracle.py` - Add symbolic token support

---

## Testing Strategy

1. **Unit test** each module (symbolic_quantizer, hybrid_perception) ✓
2. **Integration test** with Chandra_trainer on small audio file
3. **Validation** with ratio_based_chord_validator results (588 chords!)
4. **Live test** with MusicHal_9000 in performance
5. **Comparison**: Old vs new system accuracy/efficiency

---

## Timeline

- **Phase 1** (Today): Core modules + Chandra integration
- **Phase 2** (Tomorrow): MusicHal integration + testing
- **Phase 3** (This week): Temporal segmentation + optimization
- **Phase 4** (Next week): Evaluation + refinement

---

**Status**: 🟢 Active implementation in progress!






























