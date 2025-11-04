# Ratio Integration - Quick Reference

## 🚀 Quick Start (After Georgia Training)

### 1. Validate Model
```bash
python test_musichal_requests.py --model-path JSON/Georgia_<timestamp>_model.json
```

### 2. Run MusicHal
```bash
python MusicHal_9000.py --input-device 5 --hybrid-perception --wav2vec --gpu
```

### 3. Look For
```
🎯 Using request-based generation: mode=shadow, parameter=gesture_token
```

---

## 🎭 Behavior Modes

### SHADOW (MIDI CC 16: 0-42)
**Intent:** Close imitation, echo gestures  
**Request:** `gesture_token == <last_token>`  
**Weight:** 0.8  
**Feel:** Following, shadowing

### MIRROR (MIDI CC 16: 43-85)
**Intent:** Complementary, opposite direction  
**Request:** `midi_relative gradient <-2.0 or +2.0>`  
**Weight:** 0.7  
**Feel:** Answering, call-response

### COUPLE (MIDI CC 16: 86-127)
**Intent:** Harmonic support, consonance  
**Request:** `consonance > 0.7`  
**Weight:** 0.9  
**Feel:** Supporting, accompanying

---

## 📊 New Event Fields

### Ratio Analysis
- `rhythm_ratio` - Duration in rational units (1, 2, 3, etc.)
- `rhythm_subdiv_tempo` - Subdivision tempo (BPM)
- `deviation` - Timing deviation (-0.05 to +0.05)
- `deviation_polarity` - Quantized: -1, 0, 1

### Reconciliation
- `tempo_factor` - Multiplier (1.0, 2.0, 0.5, etc.)
- `tempo_reconciled` - Boolean
- `prev_tempo` - Previous phrase tempo

### Relative Parameters
- `midi_relative` - Interval in semitones
- `velocity_relative` - Velocity change
- `ioi_relative` - IOI ratio

---

## 🔧 Quick Tuning

### Too Deterministic?
```python
# In agent/phrase_generator.py
# Line 186, 202, 212: Lower weights
'weight': 0.6  # Was 0.8
```

### Too Random?
```python
# In agent/phrase_generator.py
# Line 186, 202, 212: Raise weights
'weight': 0.9  # Was 0.7
```

### Temperature Adjustment
```python
# In agent/phrase_generator.py, line 321
temperature=0.5,  # Lower = focused, higher = exploratory
```

---

## 🐛 Common Issues

### "No request-based generation messages"
→ Check oracle loaded, model has audio_frames

### "Modes feel same"
→ Increase weight differences, check event tracking

### "No matching frames"
→ Lower request weights, verify model fields exist

---

## 📈 Success Metrics

### Target Values
- **Shadowing:** >60% token match
- **Mirroring:** Negative interval correlation
- **Coupling:** >70% consonance
- **Latency:** <10ms overhead

---

## 📂 Key Files

### Implementation
- `rhythmic_engine/ratio_analyzer.py`
- `memory/request_mask.py`
- `rhythmic_engine/tempo_reconciliation.py`
- `agent/phrase_generator.py` (modified)
- `MusicHal_9000.py` (modified)

### Testing
- `test_musichal_requests.py`
- `test_ratio_analysis_integration.py`

### Documentation
- `IMPLEMENTATION_STATUS.md` - Current status
- `MUSICHAL_REQUEST_INTEGRATION_COMPLETE.md` - Full guide
- `BRANDTSEGG_INTEGRATION_FINAL.md` - Complete overview

---

## ⚡ Emergency Reset

### If something breaks:
```bash
# Fallback to standard generation
# Edit agent/phrase_generator.py, line 316:
if False and request and hasattr(...):  # Disable requests
```

### Restore original behavior:
```bash
git diff agent/phrase_generator.py  # See changes
git checkout agent/phrase_generator.py  # Revert if needed
```

---

**READY TO TEST!** 🎵

