# Autonomous Generation Implementation (Pre-Somax Branch)

**Branch**: `fix/autonomous-generation-no-somax`  
**Base Commit**: `24600ef` (Fix performance arc corruption - before SomaxBridge)  
**Date**: November 22, 2025  
**Status**: ✅ Implementation Complete, Tests Passing

## Goal

Test hypothesis: **Is SomaxBridge Factor Oracle navigation causing catastrophic bass C2 repetition (54.4%)?**

By reverting to pre-Somax code and implementing autonomous generation directly in PhraseGenerator (without SomaxBridge layer), we can isolate whether:
- ✅ SomaxBridge navigation causes repetition → Diversity improves
- ❌ Harmonic translation/Oracle training causes repetition → Repetition persists

## Problem Context

After BREAKTHROUGH session (53.2 notes/min continuous generation achieved), analysis revealed:
- **Bass C2 repetition**: 37 out of 68 bass notes = **54.4%** (catastrophic)
- **Consecutive sequences**: 11, 16, and 6 identical C2 notes in a row
- **Melody diversity better**: 18 unique notes (38% variety) vs bass 16 unique (23%)
- **Top bass note**: C2 @ 54.4%, next is C3 @ 8.8% (massive drop-off)

User hypothesis: "the somax2 path was wrong all along" - suspect SomaxBridge architecture.

## Implementation Details

### 1. PhraseGenerator (`agent/phrase_generator.py`)

**Autonomous State Variables** (lines 120-127):
```python
self.autonomous_mode = False
self.phrase_complete = {'melodic': False, 'bass': False}
self.notes_in_phrase = {'melodic': 0, 'bass': 0}
self.target_phrase_length = {'melodic': 6, 'bass': 10}
self.phrase_complete_time = {'melodic': 0.0, 'bass': 0.0}
self.last_generation_time = {'melodic': 0.0, 'bass': 0.0}
self.autonomous_interval = 0.5  # Generation attempt interval
```

**Methods Added** (after `generate_phrase()`, ~line 1085):

- **`should_respond(voice) -> bool`**: Check if voice can generate
  - Returns `True` if phrase not complete OR past 2s auto-reset
  - Returns `False` during pause period
  - Enforces 0.5s minimum interval between notes
  
- **`mark_note_generated(voice)`**: Track note generation
  - Increments `notes_in_phrase` counter
  - Marks `phrase_complete=True` when target length reached
  - Triggers 2s pause period
  
- **`set_autonomous_mode(enabled)`**: Enable/disable autonomous mode
  - Sets `self.autonomous_mode` flag
  - Prints status for transparency

### 2. AIAgent (`agent/ai_agent.py`)

**Scheduler Bypass** (lines 70-77):
```python
# AUTONOMOUS MODE: Skip scheduler check if autonomous generation enabled
if not (hasattr(self.behavior_engine, 'phrase_generator') and 
        self.behavior_engine.phrase_generator.autonomous_mode):
    # Check if we should make a decision (human-reactive mode)
    if not self.scheduler.should_make_decision():
        return []
```

In autonomous mode, skip density/initiative scheduler checks - PhraseGenerator handles timing.

### 3. BehaviorEngine (`agent/behaviors.py`)

**Use `should_respond()` for Voice Selection** (lines 1981-1990):
```python
# AUTONOMOUS MODE: Use PhraseGenerator's should_respond() if enabled
if self.phrase_generator.autonomous_mode:
    melody_should_play = self.phrase_generator.should_respond('melodic')
    bass_should_play = self.phrase_generator.should_respond('bass')
    melody_reasoning = "autonomous (phrase tracking)"
    bass_reasoning = "autonomous (phrase tracking)"
else:
    # Check episode states (hierarchical level 3)
    melody_should_play, melody_reasoning = self.melody_episode_manager.should_generate_phrase()
    bass_should_play, bass_reasoning = self.bass_episode_manager.should_generate_phrase()
```

**Track Note Generation** (after decision appended, ~line 2103):
```python
# AUTONOMOUS MODE: Track note generation for phrase completion
if self.phrase_generator.autonomous_mode:
    self.phrase_generator.mark_note_generated(voice_type)
```

### 4. MusicHal_9000 (`scripts/performance/MusicHal_9000.py`)

**Fast Outer Loop** (lines 1015-1030 in `_calculate_adaptive_interval()`):
```python
# AUTONOMOUS MODE: Fast outer loop (0.1s polling)
if (hasattr(self, 'ai_agent') and self.ai_agent and 
    hasattr(self.ai_agent, 'behavior_engine') and 
    hasattr(self.ai_agent.behavior_engine, 'phrase_generator') and
    self.ai_agent.behavior_engine.phrase_generator.autonomous_mode):
    return 0.1  # Fast polling - phrase_generator handles timing
```

**Enable Autonomous Mode** (lines 3951-3957):
```python
# Enable autonomous mode in PhraseGenerator
if not args.no_autonomous:
    if (hasattr(drift_ai, 'ai_agent') and drift_ai.ai_agent and 
        hasattr(drift_ai.ai_agent, 'behavior_engine') and 
        hasattr(drift_ai.ai_agent.behavior_engine, 'phrase_generator')):
        drift_ai.ai_agent.behavior_engine.phrase_generator.set_autonomous_mode(True)
```

## Unit Tests

**File**: `test_autonomous_pre_somax.py`

### Test Results (All Passing ✅)

1. **Autonomous State Tracking**:
   - ✅ Mode starts disabled, enables correctly
   - ✅ Per-voice state initialized (melodic/bass separate)
   - ✅ Phrase completion after target lengths (6 melody, 10 bass)
   - ✅ Pause period enforced (no generation while phrase_complete=True)
   - ✅ Voice independence (melody pauses, bass continues)
   - ✅ Auto-reset after 2s (state cleared, ready to generate)

2. **Minimum Generation Interval**:
   - ✅ First call ready
   - ✅ Immediate second call blocked (interval not passed)
   - ✅ Ready after 0.5s interval

3. **Voice Independence**:
   - ✅ Independent note counters (melody 3/6, bass 8/10)
   - ✅ Staggered phrase completion (melody completes first)
   - ✅ Different completion times allow different reset times

**Console Output**:
```
SUCCESS: All autonomous generation tests passed!
Ready for 2-minute diagnostic test.
```

## Architecture Comparison

### Before (Somax Branch)
```
Listener → AIAgent → BehaviorEngine → SomaxBridge → Factor Oracle Navigation
                                     ↓
                                     PhraseGenerator (phrase memory only)
```

### After (Pre-Somax Branch)
```
Listener → AIAgent → BehaviorEngine → PhraseGenerator (direct)
           (scheduler bypassed)      ↓
                                     Per-voice phrase tracking
                                     Autonomous timing control
```

**Key Difference**: No SomaxBridge Factor Oracle navigation layer. PhraseGenerator handles all generation logic directly.

## Proven Timing Features Preserved

From BREAKTHROUGH session (November 22, 2025):
- ✅ **Per-voice phrase tracking**: Separate state for melodic/bass
- ✅ **Fast outer loop**: 0.1s polling interval
- ✅ **Scheduler bypass**: Skip density/initiative checks
- ✅ **Auto-reset**: 2s pause after phrase completion
- ✅ **Independent voices**: Different phrase lengths → staggered pauses

## Next Steps

### 1. Run 2-Minute Diagnostic Test

```bash
python MusicHal_9000.py --enable-meld --performance-duration 2
```

**Expected Behavior**:
- Continuous generation (>50 notes/min)
- Per-voice independence maintained
- Auto-reset pauses visible

### 2. Analyze Bass Note Distribution

Parse `logs/midi_output_YYYYMMDD_HHMMSS.csv`:
- Count unique bass notes
- Calculate percentage for most frequent note
- Compare to Somax baseline (54.4% C2)

**Success Criteria**:
- ✅ Bass diversity improved (<30% any single note, currently 54.4%)
- ✅ No consecutive sequences >3 identical notes
- ✅ >30 unique notes total (currently 29)

### 3. Decision Point

**If diversity improves**:
- ✅ SomaxBridge Factor Oracle was the problem
- → Keep PhraseGenerator-only autonomous mode
- → Document SomaxBridge as unsuitable for autonomous generation

**If diversity persists**:
- ❌ Deeper issue (harmonic translation or oracle training)
- → Investigate `harmonic_translator.py` bass interval conversion
- → Check oracle training data for bass pattern diversity
- → Consider expanding `bass_range`, relaxing `scale_constraint`

## Files Modified

1. `agent/phrase_generator.py` (+76 lines)
   - Added autonomous state variables
   - Added `should_respond()`, `mark_note_generated()`, `set_autonomous_mode()`
   
2. `agent/ai_agent.py` (+6 lines)
   - Added scheduler bypass for autonomous mode
   
3. `agent/behaviors.py` (+13 lines)
   - Use `should_respond()` for voice selection
   - Track note generation with `mark_note_generated()`
   
4. `scripts/performance/MusicHal_9000.py` (+17 lines)
   - Fast outer loop (0.1s) when autonomous active
   - Enable autonomous mode in PhraseGenerator

**Total**: 4 files, 112 insertions, 6 deletions

## Git History

**Branch Created**: `git checkout -b fix/autonomous-generation-no-somax 24600ef`

**Commits**:
1. `768db5a` - Add autonomous generation to PhraseGenerator (pre-Somax branch)

**Test File**: `test_autonomous_pre_somax.py` (new file, not committed yet)

## Research Context

This is part of Jonas Sjøvaag's PhD artistic research at University of Agder. The goal is to build a musical partner trustworthy enough to improvise with, achieved through:
- **Transparency**: Visible reasoning (logs show why decisions were made)
- **Coherence**: Memory-based generation (thematic development)
- **Personality**: Behavioral modes with consistent character

The current work isolates whether SomaxBridge's Factor Oracle navigation layer is causing quality issues (bass C2 repetition), or if the problem lies in deeper layers (harmonic translation, oracle training data).

This is a **controlled experiment approach**: Same timing architecture, different generation path.

---

**Status**: ✅ Ready for diagnostic testing  
**Next Command**: `python MusicHal_9000.py --enable-meld --performance-duration 2`
