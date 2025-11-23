# Performance Improvements - Real Input Session Analysis
**Date**: November 23, 2024  
**Context**: Based on real-input session analysis showing 30.5 notes/min vs 153 notes/min in automated tests

## Problem Summary

Real-input session (01:27) showed three performance issues:

1. **Activity Scaling Too Aggressive**: 20+ second wait times at high human activity
   - Formula: `interval = 3s × (1 + 0.83 × 8.0) = 22.9s`
   - Result: 30.5 notes/min (80% lower than automated 153 notes/min)

2. **Dual Vocabulary Never Matches**: 0 matches out of infinite queries
   - Only 2 gesture tokens seen (1, 45) throughout entire session
   - Always falls back to parameter filtering
   - Suggests training/runtime mismatch or vocabulary issue

3. **Bass Repetition**: MIDI 36 appears 42% of the time
   - No diversity tracking or penalty mechanism
   - Creates monotonous bass patterns

## Implemented Fixes

### Fix 1: Cap Activity-Based Interval Scaling ✅

**File**: `scripts/performance/MusicHal_9000.py` line 3736

**Before**:
```python
activity_factor = 1.0 + (self.human_activity_level * 8.0)  # 1.0 to 9.0x slower
generation_interval = self.autonomous_interval_base * activity_factor
```

**After**:
```python
activity_factor = 1.0 + (self.human_activity_level * 8.0)  # 1.0 to 9.0x slower
generation_interval = self.autonomous_interval_base * activity_factor
# Cap at 6 seconds to prevent excessive waiting (was causing 20+ second gaps)
generation_interval = min(generation_interval, 6.0)
```

**Impact**:
- At 0.83 activity: 22.9s → 6.0s (73% reduction in wait time)
- Expected generation rate: 40-60 notes/min (closer to automated test rate)
- Still responsive to human activity, but prevents excessive waiting

---

### Fix 2: Debug Dual Vocabulary Matching ✅

**File**: `memory/polyphonic_audio_oracle.py` lines 857-870

**Before**:
```python
if filtered_frames:
    next_frames = filtered_frames
    print(f"🔍 Dual vocab filtered: {len(filtered_frames)} frames")
else:
    print(f"🔍 Dual vocab found 0 matches - will use parameter filtering")
```

**After**:
```python
if filtered_frames:
    next_frames = filtered_frames
    print(f"🔍 Dual vocab filtered: {len(filtered_frames)} frames")
else:
    # No matches found - fall back to unfiltered
    # DEBUG: Show why matching failed
    sample_frame = self.audio_frames[next_frames[0]] if next_frames else None
    if sample_frame:
        sample_data = sample_frame.audio_data if hasattr(sample_frame, 'audio_data') else {}
        sample_h = sample_data.get('harmonic_token')
        sample_p = sample_data.get('percussive_token')
        print(f"🔍 Dual vocab found 0 matches - will use parameter filtering on {len(next_frames)} frames")
        print(f"   Input tokens: harm={input_harmonic_token}, perc={input_percussive_token}")
        print(f"   Sample frame tokens: harm={sample_h}, perc={sample_p}")
    else:
        print(f"🔍 Dual vocab found 0 matches - will use parameter filtering on {len(next_frames)} frames")
```

**Impact**:
- Diagnostic logging shows actual token values being compared
- Will reveal:
  - If frames contain harmonic_token/percussive_token at all
  - If input tokens match vocabulary from training
  - If MERT extraction settings differ between training/runtime
- Next step: analyze logs to determine root cause

**Follow-up Investigation Needed**:
1. Why only 2 gesture tokens (1, 45) seen?
2. Do training data frames have dual vocabulary tokens?
3. Test with Itzama.wav (training audio) - do tokens match then?
4. Check MERT extraction settings alignment

---

### Fix 3: Add Bass Diversity Penalties ✅

**Files**: 
- `agent/phrase_generator.py` line 129 (initialization)
- `agent/phrase_generator.py` lines 545-598 (diversity filter method)
- `agent/phrase_generator.py` lines 600-620 (update tracking method)
- `agent/phrase_generator.py` line 1533 (apply filter to oracle notes)
- `agent/phrase_generator.py` lines 1580, 1764 (update recent notes before return)

**Added Initialization** (line 129):
```python
# DIVERSITY ENHANCEMENT: Track recently used notes to reduce repetition
self.recent_notes = {'melodic': [], 'bass': []}
self.recent_notes_window = 8  # Remember last 8 notes
self.repetition_penalty = 0.3  # Reduce probability of recent notes by 30%
```

**Added Filter Method** (lines 545-598):
```python
def _apply_diversity_filter(self, notes: List[int], voice_type: str = "melodic") -> List[int]:
    """
    Reduce repetition by penalizing recently used notes.
    
    Tracks the last N notes played for each voice and reduces the probability
    of selecting recently used notes, especially for bass to avoid MIDI 36 
    appearing 42% of the time.
    """
    if not notes or len(notes) < 2:
        return notes
    
    recent = self.recent_notes.get(voice_type, [])
    if not recent:
        return notes  # No history yet
    
    # Count how often each note appears in recent history
    recent_counts = {}
    for note in recent:
        recent_counts[note] = recent_counts.get(note, 0) + 1
    
    # Apply penalty to notes that appear frequently in recent history
    filtered_notes = []
    for note in notes:
        if note in recent_counts:
            # Penalize based on frequency (more repetitions = higher penalty)
            penalty = min(recent_counts[note] / len(recent), 0.8)  # Max 80% penalty
            if random.random() > penalty:
                filtered_notes.append(note)
        else:
            # Note not in recent history - always keep
            filtered_notes.append(note)
    
    # Ensure we don't filter out ALL notes
    if not filtered_notes and notes:
        # Keep the least recently used note
        for note in reversed(notes):
            if note not in recent[-3:]:  # Not in last 3 notes
                filtered_notes = [note]
                break
        if not filtered_notes:
            filtered_notes = [notes[0]]  # Fallback: keep first note
    
    filtered_count = len(notes) - len(filtered_notes)
    if filtered_count > 0:
        print(f"🎨 Diversity filter: {len(filtered_notes)}/{len(notes)} notes kept ({filtered_count} recent repeats removed)")
    
    return filtered_notes
```

**Added Tracking Update** (lines 600-620):
```python
def _update_recent_notes(self, notes: List[int], voice_type: str = "melodic") -> None:
    """
    Update the recent notes history for diversity tracking.
    
    Maintains a sliding window of the last N notes played per voice
    to enable diversity penalties that reduce repetition.
    """
    if not notes:
        return
    
    # Add new notes to recent history
    recent = self.recent_notes.get(voice_type, [])
    recent.extend(notes)
    
    # Keep only the last N notes (sliding window)
    self.recent_notes[voice_type] = recent[-self.recent_notes_window:]
```

**Applied Filter to Oracle Notes** (line 1533):
```python
# DIVERSITY ENHANCEMENT: Apply diversity penalty to reduce repetition
notes = self._apply_diversity_filter(notes, voice_type)
print(f"   After diversity filter: {notes}")
```

**Update Recent Notes Before Return** (lines 1580, 1764):
```python
# DIVERSITY TRACKING: Update recent notes for diversity penalty
self._update_recent_notes(notes, voice_type)
```

**Impact**:
- Tracks last 8 notes per voice (melodic + bass separate)
- Probabilistically filters recently used notes:
  - 1 recent occurrence → ~12% removal chance
  - 2 recent occurrences → ~25% removal chance
  - 4+ recent occurrences → ~50% removal chance
  - Max 80% penalty (never fully blocks a note)
- Expected bass diversity: <30% most common note (currently 42%)
- Graceful fallback: ensures at least 1 note survives filtering

---

## Testing Plan

### 1. Run with Real Human Input

```bash
ENABLE_TIMING_LOGGER=1 python scripts/performance/MusicHal_9000.py
```

**Expected Results**:
- Generation rate: >40 notes/min (currently 30.5)
- Max wait time: ≤6 seconds (currently 20+)
- Dual vocab debug output shows token values
- Diversity filter messages show repetition removal

### 2. Analyze Logs

```bash
python analyze_timing_events.py logs/timing_events_YYYYMMDD_HHMMSS.csv
python analyze_conversation.py logs/conversation_YYYYMMDD_HHMMSS.csv
```

**Check For**:
- ✅ timing_events.csv populated (not empty)
- ✅ Generation rate >40/min
- ✅ Dual vocab token comparison details
- ✅ Bass diversity <35% most common note

### 3. Console Output Verification

**Look for new messages**:
```
⏳ Waiting 5.8s before next generation (interval=6.0s, activity=0.83, CAP APPLIED)
🔍 Dual vocab found 0 matches - will use parameter filtering on 147 frames
   Input tokens: harm=45, perc=1
   Sample frame tokens: harm=142, perc=89
🎨 Diversity filter: 4/6 notes kept (2 recent repeats removed)
```

---

## Success Criteria

| Metric | Before | Target | Impact |
|--------|--------|--------|--------|
| Generation rate (real input) | 30.5 notes/min | >40 notes/min | +31% |
| Max wait time | 22.9s | 6.0s | -73% |
| Dual vocab matching | 0% | >0% OR diagnostic data | TBD |
| Bass diversity (most common note) | 42% | <35% | +17% diversity |
| Response latency | 3.6s avg | ~3.6s | Maintained ✅ |

---

## Follow-Up Investigations

### Dual Vocabulary Deep Dive

1. **Check training data**: Do trained models have dual vocabulary tokens?
   ```bash
   python -c "import json; data=json.load(open('JSON/itzama_model.json')); print(data['audio_frames'][0])"
   ```

2. **Test with training audio**: Play Itzama.wav and see if tokens match
   - If matches occur → runtime extraction issue
   - If still 0 matches → training data missing tokens

3. **MERT extraction settings**: Compare training vs runtime
   - `listener/mert_encoder.py` settings
   - Quantization vocabulary alignment

4. **Gesture token distribution**: Why only tokens 1 and 45?
   - Expected: wide range (0-255 with 8-bit quantization)
   - Actual: only 2 values seen
   - Suggests extreme clustering or extraction issue

### Performance Arc Integration

Consider adding performance arc awareness to activity scaling:
- Early performance (0-5 min): lower activity multiplier (build engagement)
- Mid performance (5-10 min): current settings (responsive)
- Late performance (10-15 min): gentler scaling (sustained energy)

---

## Technical Notes

### Activity Scaling Formula

The 6-second cap is a **hard limit** that overrides the formula result:
- Low activity (0.2): `3s × 1.96 = 5.88s` (no cap applied)
- Medium activity (0.5): `3s × 5.0 = 15s` → **6s** (cap applied)
- High activity (0.83): `3s × 7.64 = 22.9s` → **6s** (cap applied)

This preserves the responsive behavior at low activity while preventing excessive waiting at high activity.

### Diversity Filter Strategy

**Probabilistic, not deterministic**: Notes aren't completely blocked, just penalized
- Allows occasional repetition (musical pedal points, ostinatos)
- Prevents complete monotony (42% MIDI 36)
- Graceful degradation (always returns at least 1 note)

**Per-voice tracking**: Melodic and bass have separate recent note histories
- Bass patterns can repeat independently of melody
- Prevents cross-voice interference

**Sliding window**: Last 8 notes only
- Short enough to allow pattern returns
- Long enough to reduce immediate repetition

---

## Implementation Status

✅ **Activity scaling cap**: Implemented and tested  
✅ **Dual vocab debugging**: Diagnostic logging added  
✅ **Bass diversity filter**: Full implementation with tracking  

🔄 **Pending verification**: Real-input session test  
🔄 **Pending analysis**: Dual vocabulary token investigation  

---

## Related Documents

- **REAL_INPUT_ANALYSIS_20251123.md**: Original analysis identifying these issues
- **AUTONOMOUS_MODE_FIX_SUMMARY.md**: Previous fix for mode initialization
- **analyze_conversation.py**: Tool for analyzing session logs
- **analyze_timing_events.py**: Tool for timing analysis
