# Rhythmic Awareness Implementation - Progress Report

## 🎯 Goal
Implement rhythmic awareness following the same pattern as harmonic awareness:
1. **Detect** rhythmic context (tempo, meter, beat grid)
2. **Pass** through the system (Event → Agent → Mapper)
3. **Quantize** timing decisions to beat grid
4. **Apply** mode-specific rhythmic intelligence

---

## ✅ Completed (Step 1: Listener Integration)

### 1. Created `listener/rhythmic_context.py`
**Status:** ✅ COMPLETE

**What it does:**
- `RhythmicContext`: Dataclass with tempo, meter, beat position, density, syncopation
- `RealtimeRhythmicDetector`: Real-time tempo/beat detection from onsets

**Key features:**
- Tempo estimation from IOI (inter-onset intervals)
- Beat grid generation and maintenance
- Beat position tracking (where in measure: 0.0-4.0 for 4/4)
- Syncopation detection (on-beat vs off-beat onsets)
- Rhythmic density classification (sparse/moderate/dense)
- Beat quantization methods (nearest, before, after, off_beat)
- Beat strength calculation (strong vs weak beats)

### 2. Updated `listener/jhs_listener_core.py`
**Status:** ✅ COMPLETE

**Changes:**
- Added `RhythmicContext` import
- Added `rhythmic_context` field to `Event` class
- Initialized `RealtimeRhythmicDetector` in `__init__`
- Update rhythmic context every event (uses onset/ioi data)
- Added rhythmic context to `Event.to_dict()`
- Debug logging for rhythmic context (every 20th event)

**Example output:**
```
🥁 Rhythmic: 120.0 BPM, 4/4, beat 2.35, density=moderate
```

---

## 🚧 Next Steps (Steps 2-4)

### Step 2: Update BehaviorEngine with Rhythmic Awareness
**File:** `agent/behaviors.py`  
**Status:** ⏳ TODO

**What needs to be done:**
```python
def _generate_imitate_params(self, ...):
    # Add rhythmic information
    if rhythmic_context:
        params['current_tempo'] = rhythmic_context.current_tempo
        params['beat_position'] = rhythmic_context.beat_position
        params['next_beat_time'] = rhythmic_context.next_beat_time
        params['syncopation_level'] = rhythmic_context.syncopation_level
        params['rhythmic_density'] = rhythmic_context.rhythmic_density
```

### Step 3: Update FeatureMapper with Rhythmic Quantization
**File:** `mapping/feature_mapper.py`  
**Status:** ⏳ TODO

**What needs to be added:**

1. **Extract rhythmic context:**
```python
def _extract_rhythmic_context(self, musical_params):
    return {
        'tempo': musical_params.get('current_tempo', 120.0),
        'beat_position': musical_params.get('beat_position', 0.0),
        'next_beat_time': musical_params.get('next_beat_time'),
        ...
    }
```

2. **Quantize timing to beat:**
```python
def _quantize_timing_to_beat(self, target_time, rhythmic_context, mode):
    """Quantize note timing to beat grid based on mode"""
    beat_grid = rhythmic_context.get('beat_grid', [])
    
    if mode == 'imitate':
        # Play ON the beat
        return nearest_beat(target_time, beat_grid)
    
    elif mode == 'contrast':
        # Play OFF-beat (syncopated)
        return between_beats(target_time, beat_grid)
    
    elif mode == 'lead':
        # Anticipate the beat (slightly early)
        return before_beat(target_time, beat_grid, offset=0.05)
```

3. **Apply to duration/timing:**
```python
# Instead of random duration
duration = self._map_ioi_to_duration(ioi, voice_type)  # ❌ Current

# Use beat-aware duration
duration = self._quantize_duration_to_beat(ioi, rhythmic_context)  # ✅ Proposed
```

### Step 4: Testing
**File:** `test_rhythmic_awareness.py`  
**Status:** ⏳ TODO

**Tests needed:**
1. Tempo detection accuracy
2. Beat quantization (on-beat vs off-beat)
3. Mode-specific timing (imitate/contrast/lead)
4. Syncopation handling
5. Rhythmic density adaptation

---

## 📊 Architecture (Parallel to Harmonic)

| Component | Harmonic | Rhythmic | Status |
|-----------|----------|----------|--------|
| **Context Detection** | Chroma → Chord/Key | Onsets → Tempo/Beat | ✅ Done |
| **Context Dataclass** | `HarmonicContext` | `RhythmicContext` | ✅ Done |
| **Listener Integration** | `harmonic_context` in Event | `rhythmic_context` in Event | ✅ Done |
| **BehaviorEngine** | Adds chord/scale to params | **TODO: Add tempo/beat to params** | ⏳ Next |
| **FeatureMapper** | Quantize pitch to scale | **TODO: Quantize timing to beat** | ⏳ Next |
| **Testing** | `test_featuremapper_harmonic.py` | **TODO: `test_rhythmic_awareness.py`** | ⏳ Next |

---

## 🎼 Example: How It Will Work

### Input Event:
```
Time: 5.27 seconds
Onset: True
IOI: 0.48 seconds
```

### Rhythmic Context (Detected):
```
Tempo: 125 BPM
Meter: 4/4
Beat Position: 2.75 (between beat 3 and 4)
Next Beat: 5.32 seconds
Density: moderate
```

### Mode-Specific Behavior:

**Imitate Mode:**
```python
Target: 5.27s
Quantized: 5.32s (next beat)
Result: Plays ON beat 4 ✅
```

**Contrast Mode:**
```python
Target: 5.27s
Quantized: 5.14s (off-beat, halfway between beats 3 and 4)
Result: Plays syncopated ✅
```

**Lead Mode:**
```python
Target: 5.27s
Quantized: 5.27s (slight anticipation of beat 4)
Result: Plays slightly before beat ✅
```

---

## 🔄 Data Flow (When Complete)

```
Audio Input → DriftListener
            ↓
   Extract: onsets, IOI
            ↓
   Detect: tempo, meter, beat grid
            ↓
   RhythmicContext (120 BPM, 4/4, beat 2.5)
            ↓
   Event.rhythmic_context
            ↓
   AIAgent → BehaviorEngine
            ↓
   musical_params.tempo/beat_position
            ↓
   FeatureMapper
            ↓
   Quantize timing to beat
            ↓
   MIDI with beat-aware timing ✅
```

---

## 💡 Key Benefits

### Before (No Rhythmic Awareness):
- Random timing
- No relationship to detected tempo
- Ignores meter and beat structure
- Can play awkwardly off-beat unintentionally

### After (With Rhythmic Awareness):
- Timing quantized to beat grid
- Respects detected tempo and meter
- Mode-specific rhythmic behavior:
  - **Imitate**: On-beat (locked)
  - **Contrast**: Syncopated (intentional off-beat)
  - **Lead**: Anticipatory (pushing the beat)
- Musically intelligent timing decisions

---

## 📝 Summary

**Completed:** Listener integration with real-time tempo/beat detection ✅  
**Next:** BehaviorEngine rhythmic params (10 min) → FeatureMapper quantization (20 min) → Testing (15 min)  
**Total Remaining:** ~45 minutes of implementation

**The foundation is in place! Now we need to wire the rhythmic context through the decision pipeline, just like we did with harmonic context.**

