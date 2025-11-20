# Three-Phase Performance Arc Implementation - COMPLETE

## Summary

Implemented a clean, percentage-based 3-phase performance arc that works for ANY duration:

```
┌─────────────────────────────────────────────────────────────┐
│  BUILDUP (15%)  │      MAIN (65%)      │   ENDING (20%)    │
│   0.3 → 1.0     │        1.0           │     1.0 → 0.0     │
│   sparse → full │   full activity      │  gradual fade     │
└─────────────────────────────────────────────────────────────┘
```

## Architecture

### 1. Timeline Manager (`performance_timeline_manager.py`)

Added three new methods:

#### `get_performance_phase() -> str`
Returns current phase: `'buildup'`, `'main'`, or `'ending'`

- **Buildup**: First 15% of performance (0.0 - 0.15 progress)
- **Main**: Middle 65% (0.15 - 0.80 progress)  
- **Ending**: Final 20% (0.80 - 1.0 progress)

#### `get_activity_multiplier() -> float`
Returns activity level based on current phase:

- **Buildup**: Gradual increase from 0.3 → 1.0
  - Formula: `0.3 + (0.7 * buildup_progress)`
  - Creates smooth introduction of AI partner
  
- **Main**: Full activity at 1.0
  - Complete musical engagement
  
- **Ending**: Smooth exponential fade from 1.0 → 0.0
  - Formula: `(1.0 - ending_progress) ** 2`
  - Graceful withdrawal, not abrupt cutoff

#### Updated `get_performance_guidance() -> Dict`
Now includes:
- `'performance_phase'`: Current 3-phase arc phase
- `'activity_multiplier'`: 0.0-1.0 activity level
- Adjusted `should_respond` probability based on phase
- Near-zero re-entry during ending (let it fade)

### 2. Data Flow Through System

```
MusicHal_9000.py::process_audio_event()
    ↓
    Gets activity_multiplier from timeline_manager.get_performance_guidance()
    ↓
ai_agent.py::process_event(..., activity_multiplier)
    ↓
behaviors.py::decide_behavior(..., activity_multiplier)
    ↓
behaviors.py::_generate_decision(..., activity_multiplier)
    ↓
phrase_generator.py::generate_phrase(..., activity_multiplier)
    ↓
phrase_generator.py::_generate_buildup_phrase(..., activity_multiplier)
    (and _generate_peak_phrase, _generate_release_phrase, _generate_contemplation_phrase)
```

### 3. Phrase Length Scaling

In `_generate_buildup_phrase()`:

```python
# Base phrase length (before scaling)
if voice_type == "melodic":
    base_length = random.randint(2, 4)  # 2-4 notes
else:
    base_length = random.randint(1, 2)  # 1-2 notes (bass)

# Scale by activity multiplier
phrase_length = max(1, int(base_length * activity_multiplier))
```

**Result**:
- **Early buildup** (activity = 0.3): Melody 0-1 notes, Bass 0-1 notes (very sparse)
- **Mid buildup** (activity = 0.65): Melody 1-2 notes, Bass 1 note (building)
- **Main phase** (activity = 1.0): Melody 2-4 notes, Bass 1-2 notes (full)
- **Ending** (activity = 0.5): Melody 1-2 notes, Bass 0-1 notes (fading)

### 4. Response Probability Modulation

In `get_performance_guidance()`:

```python
if performance_phase == 'buildup':
    # Gradually increase response rate
    should_respond = base_should_respond and (random.random() < activity_multiplier)
elif performance_phase == 'ending':
    # Gradually decrease response rate
    should_respond = base_should_respond and (random.random() < activity_multiplier)
else:
    # Main phase - full response
    should_respond = base_should_respond
```

**Result**:
- **Early buildup**: 30% chance to respond (sparse, listening)
- **Late buildup**: 100% chance (fully engaged)
- **Main**: 100% chance (active partner)
- **Ending**: Decreasing chance → 0% (graceful fade)

### 5. Silence Mode During Ending

```python
if performance_phase == 'buildup':
    # Very conservative re-entry during buildup
    re_entry_chance = 0.001 * activity_multiplier
elif performance_phase == 'ending':
    # Almost never re-enter during ending (let it fade)
    re_entry_chance = 0.0
else:
    # Main phase - use momentum-based logic
    re_entry_chance = 0.01 to 0.05  # Based on musical momentum
```

**Result**: Once the ending phase starts fading and silence kicks in, AI stays silent. Clean, graceful ending.

## Configurable Parameters

In `get_performance_phase()`:

```python
buildup_end = 0.15    # First 15% is buildup
ending_start = 0.80   # Last 20% is ending
```

Easy to adjust:
- More gradual buildup: `buildup_end = 0.25` (25%)
- Longer ending: `ending_start = 0.70` (30% ending)
- Aggressive start: `buildup_end = 0.05` (5% buildup)

In `get_activity_multiplier()`:

```python
# Buildup starts at 0.3 (30% activity minimum)
return 0.3 + (0.7 * buildup_progress)
```

Change `0.3` to start even sparser (0.1) or fuller (0.5).

## Integration with Existing Arc System

The 3-phase system works **alongside** the loaded performance arc:

- If performance arc is loaded: Uses arc's `intro/development/climax/resolution` for behavior mode selection
- 3-phase arc **always** provides `activity_multiplier` and `performance_phase`
- Best of both worlds: Arc defines personality, 3-phase defines energy curve

## Testing

To test different durations:

```bash
# 3-minute performance
python MusicHal_9000.py --duration 3

# 10-minute performance
python MusicHal_9000.py --duration 10

# 1-minute quick test
python MusicHal_9000.py --duration 1
```

Expected behavior:
- **First 15%**: Sparse responses, short phrases, listening mode
- **Middle 65%**: Full engagement, normal phrase lengths
- **Final 20%**: Gradually decreasing responses, shorter phrases, smooth fade to silence

## Performance Characteristics

### Buildup Phase (First 15%)
- AI starts tentative (30% activity)
- Short phrases (1-2 notes melody, 0-1 bass)
- Low response probability (30% at start)
- Imitate mode (following human)
- "Getting to know you" character

### Main Phase (Middle 65%)
- Full AI engagement (100% activity)
- Normal phrases (2-4 notes melody, 1-2 bass)
- Full response probability
- Mode from loaded arc (contrast, lead, etc.)
- "Equal partner" character

### Ending Phase (Final 20%)
- Smooth fade (100% → 0% activity)
- Progressively shorter phrases
- Decreasing response probability
- No re-entry from silence (stays quiet once faded)
- "Graceful goodbye" character

## Musical Result

The AI partner now:
1. ✅ **Builds gradually** from sparse listener to active partner
2. ✅ **Maintains full engagement** during main performance
3. ✅ **Fades gracefully** without abrupt cutoff
4. ✅ **Works for any duration** (1 minute to 15+ minutes)
5. ✅ **Respects loaded arc** for personality while adding energy curve

## Code Changes Summary

**Modified Files**:
- `performance_timeline_manager.py`: Added `get_performance_phase()`, `get_activity_multiplier()`, updated `get_performance_guidance()`
- `agent/phrase_generator.py`: Added `activity_multiplier` parameter to `generate_phrase()` and all phrase generation methods
- `agent/behaviors.py`: Added `activity_multiplier` parameter to `decide_behavior()` and `_generate_decision()`
- `agent/ai_agent.py`: Added `activity_multiplier` parameter to `process_event()`
- `MusicHal_9000.py`: Extract `activity_multiplier` from guidance and pass through to `process_event()`

**No Breaking Changes**: All new parameters have defaults (`activity_multiplier=1.0`). Existing code works unchanged.

## Next Steps (Optional Enhancements)

1. **Velocity scaling**: Multiply velocities by `activity_multiplier` for dynamic buildup/fadeout
2. **Phrase timing**: Stretch timings during buildup (`timing *= (1.0 + (1.0 - activity_multiplier))`)
3. **Register shift**: Lower register during buildup, normal during main, higher during ending
4. **Visualization**: Show 3-phase arc in Qt viewport (buildup/main/ending zones)
5. **User configuration**: Command-line flags `--buildup-percent 20 --ending-percent 30`

---

**Status**: ✅ COMPLETE - Ready for testing

**Test Command**: `python MusicHal_9000.py --duration 5 --enable-rhythmic`
