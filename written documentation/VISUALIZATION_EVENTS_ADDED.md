# Visualization Event Emissions - Added to MusicHal_9000.py

## ✅ Events Successfully Integrated

### 1. Audio Analysis Viewport (ACTIVE)
**Location:** `_on_audio_event()` method (line ~685-705)

**What it shows:**
- Real-time waveform display from audio buffer
- Onset detection (orange highlight when detected)
- Consonance value (color-coded: green=consonant, red=dissonant)
- Rhythm ratio (placeholder - actual ratio would need deeper extraction)

**Emits when:**
- Every audio event is processed
- Raw audio buffer is available
- Hybrid perception features are extracted

```python
self.visualization_manager.emit_audio_analysis(
    waveform=audio_buffer,
    onset=event_data.get('onset', False),
    ratio=ratio,
    consonance=event_data.get('hybrid_consonance', None),
    timestamp=current_time
)
```

---

### 2. Timeline Viewport (ACTIVE)
**Locations:**
- Human input: `_on_audio_event()` (line ~705)
- AI response: `_main_loop()` (lines ~1425, ~1449)

**What it shows:**
- Scrolling 5-minute timeline
- Human input events (purple triangles)
- AI response events (green circles)
- Mode changes (blue rectangles) - NOT YET IMPLEMENTED
- Thematic recalls (yellow stars) - NOT YET IMPLEMENTED

**Emits when:**
- Human plays a note (audio event processed)
- AI sends a MIDI note (both MPE and standard MIDI paths)

```python
# Human input
self.visualization_manager.emit_timeline_update('human_input', timestamp=current_time)

# AI response
self.visualization_manager.emit_timeline_update('response', timestamp=current_time)
```

---

### 3. Pattern Matching Viewport (PARTIALLY ACTIVE)
**Location:** `_on_audio_event()` method (line ~665-673)

**What it shows:**
- Current gesture token (large display)
- Pattern match score (0-100%, color-coded progress bar)
- AudioOracle state matched (currently shows 0 - needs actual state ID)
- Recent token history (scrolling list)

**Emits when:**
- Gesture token is extracted from Wav2Vec features
- Hybrid perception is enabled

**Current limitation:**
- Match score is estimated from consonance (placeholder)
- State ID is hardcoded to 0 (needs actual AudioOracle state from generation)
- Actual pattern matching happens in `agent/phrase_generator.py` (separate file)

```python
self.visualization_manager.emit_pattern_match(
    score=match_score,  # Currently: consonance * 100
    state_id=0,  # TODO: Get actual AudioOracle state
    gesture_token=hybrid_result.symbolic_token,
    context={'consonance': hybrid_result.consonance}
)
```

**To make fully functional:**
- Need to access AudioOracle state during generation
- Need actual pattern match score from `AudioOracle.generate_with_request()`
- Would require modifying `agent/phrase_generator.py` to pass state_id back

---

## ⏳ Events NOT Yet Integrated

### 4. Request Parameters Viewport (NOT ACTIVE)
**What it should show:**
- Current behavior mode (SHADOW, MIRROR, COUPLE, etc.)
- Mode duration countdown
- Request structure (primary/secondary/tertiary parameters)
- Temperature setting

**Where to add:**
- In `agent/behaviors.py` when mode changes
- In `agent/phrase_generator.py` when requests are built

**Required changes:**
```python
# In agent/behaviors.py, when mode changes:
if drift_ai.visualization_manager:
    drift_ai.visualization_manager.emit_mode_change(
        mode=new_mode,
        duration=mode_duration,
        request_params=request_structure,
        temperature=current_temperature
    )
    
    # Also emit timeline event
    drift_ai.visualization_manager.emit_timeline_update('mode_change', mode=new_mode)
```

**Challenge:**
- `agent/behaviors.py` doesn't have direct access to MusicHal instance
- Would need to pass `visualization_manager` through to agent initialization
- Or use a callback/observer pattern

---

### 5. Phrase Memory Viewport (NOT ACTIVE)
**What it should show:**
- Stored motifs (list)
- Current theme
- Recall probability indicator
- Recent memory events (store/recall/variation)

**Where to add:**
- In `agent/phrase_memory.py` when motifs are stored/recalled/varied

**Required changes:**
```python
# In agent/phrase_memory.py, when storing motif:
if self.visualization_manager:
    self.visualization_manager.emit_phrase_memory(
        action='store',
        motif=motif_notes,
        timestamp=current_time
    )

# When recalling:
if self.visualization_manager:
    self.visualization_manager.emit_phrase_memory(
        action='recall',
        motif=motif_notes,
        timestamp=current_time
    )

# When applying variation:
if self.visualization_manager:
    self.visualization_manager.emit_phrase_memory(
        action='variation',
        motif=varied_notes,
        variation_type='transpose',  # or 'invert', 'retrograde', etc.
        timestamp=current_time
    )
```

**Challenge:**
- Same as Request Parameters - `phrase_memory.py` doesn't have access to MusicHal instance
- Need to pass visualization_manager through initialization chain

---

## 📊 Current Functionality Status

| Viewport | Status | Update Frequency | Notes |
|----------|--------|------------------|-------|
| Audio Analysis | ✅ Working | Every audio event (~50ms) | Waveform, onset, consonance display |
| Timeline | ✅ Working | On events (human input/AI response) | Shows interaction over time |
| Pattern Matching | ⚠️ Partial | Every audio event (~50ms) | Token works, score/state are placeholders |
| Request Parameters | ❌ Not Connected | N/A | Needs agent integration |
| Phrase Memory | ❌ Not Connected | N/A | Needs agent integration |

**Overall:** 2.5 out of 5 viewports functional (50% complete)

---

## 🚀 Next Steps to Complete Integration

### Quick Wins (30 minutes)

1. **Pass visualization_manager to agent initialization**
   ```python
   # In EnhancedDriftEngineAI.__init__:
   self.ai_agent = AIAgent(visualization_manager=self.visualization_manager)
   ```

2. **Add mode change emission in behaviors.py**
   - Store reference to visualization_manager in BehaviorEngine
   - Emit mode_change event when mode switches

3. **Add phrase memory emission in phrase_memory.py**
   - Store reference to visualization_manager
   - Emit store/recall/variation events

### More Complex (2-3 hours)

4. **Get actual AudioOracle state during generation**
   - Modify `AudioOracle.generate_with_request()` to return state_id
   - Pass state_id through to visualization emission
   - Update pattern match score to use actual match quality

5. **Request structure visualization**
   - Extract full request structure from phrase_generator
   - Include weights for primary/secondary/tertiary parameters
   - Show parameter types and values

---

## 🧪 Testing Current Implementation

### Run MusicHal with Visualization:

```bash
python MusicHal_9000.py \
  --input-device 0 \
  --hybrid-perception \
  --wav2vec \
  --gpu \
  --visualize
```

### Expected Behavior:

1. **Audio Analysis Viewport** - Updates in real-time with your playing
   - Waveform should animate with audio input
   - Orange "Onset: DETECTED" when you hit a note
   - Consonance value changes with harmonic content

2. **Timeline Viewport** - Shows events over time
   - Purple triangles appear when you play (human input)
   - Green circles appear when AI responds
   - Scrolls right to left, "NOW" indicator on right edge

3. **Pattern Matching Viewport** - Shows current gesture analysis
   - Large number display updates with each gesture token
   - Score bar shows estimated match quality (based on consonance)
   - Token history scrolls in bottom panel

4. **Request Parameters Viewport** - Will be empty/static
   - Shows "---" for mode
   - No countdown or parameters

5. **Phrase Memory Viewport** - Will be empty/static
   - Shows "Current Theme: None"
   - No stored motifs or events

---

## 📝 Code Locations Summary

**Files modified:**
- `MusicHal_9000.py` - Added 3 emission points (~30 lines total)

**Emission locations:**
1. Line ~665-673: Pattern matching (gesture token)
2. Line ~685-705: Audio analysis + human input timeline
3. Line ~1425: AI response timeline (MPE path)
4. Line ~1449: AI response timeline (standard MIDI path)

**Files that still need modification:**
- `agent/behaviors.py` - Add mode change emissions
- `agent/phrase_memory.py` - Add motif store/recall/variation emissions  
- `agent/phrase_generator.py` - Add request structure emissions

---

## 💡 Design Notes

### Why Some Viewports Aren't Connected

The agent subsystem (`agent/behaviors.py`, `agent/phrase_memory.py`, `agent/phrase_generator.py`) is designed to be independent of the main MusicHal class. This is good architecture for separation of concerns, but makes it harder to add cross-cutting features like visualization.

**Solutions:**
1. **Dependency Injection** - Pass visualization_manager through constructors
2. **Observer Pattern** - Agent emits events, MusicHal forwards to visualization
3. **Global Reference** - Store visualization_manager in a shared context (not recommended)

**Recommendation:** Use dependency injection (Option 1) - cleanest and most explicit.

---

## ✅ Summary

**What works:**
- Real-time waveform display
- Onset detection visualization
- Consonance tracking
- Timeline of human/AI interaction
- Gesture token extraction display

**What's missing:**
- Behavior mode display and countdown
- Request parameter structure
- Phrase memory operations
- Actual AudioOracle pattern match scores
- Thematic recall events on timeline

**Effort to complete:**
- Quick implementation (30 min): Add mode + phrase memory emissions
- Full implementation (2-3 hours): Include actual pattern match scoring + request structure display

The current implementation gives you **visual feedback on the core perception and interaction loop**, which is valuable for understanding system behavior and useful for documentation/recording purposes.

