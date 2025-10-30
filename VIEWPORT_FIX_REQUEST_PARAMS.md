# Viewport Fix: Request Parameters Now Displaying

## Problem Diagnosis

The request parameters viewport was showing "no parameters" because of an **architectural mismatch**:

1. **Original Design**: Visualization events were emitted from `phrase_generator.generate_phrase()`
2. **Actual Runtime**: MusicHal uses `ai_agent.process_event()` → `behavior_engine.decide_behavior()` → `_generate_single_decision()` path
3. **Root Cause**: `phrase_generator` is only created when `rhythm_oracle` is passed to `AIAgent`, but MusicHal_9000.py line 298 creates `AIAgent` without `rhythm_oracle`, so `phrase_generator` stays `None`

## Solution Implemented

**Emit visualization events from the actual decision-making location** - the three decision methods in `BehaviorEngine`:

- `_imitate_decision()` (line ~846)
- `_contrast_decision()` (line ~903)
- `_lead_decision()` (line ~949)

### Changes Made

**File: `agent/behaviors.py`**

1. **Added helper method** `_extract_request_params_from_context()` (line ~968):
   - Extracts harmonic context (chord, key, stability/consonance)
   - Adds mode-specific behavioral parameters (similarity/contrast/innovation)
   - Returns list of parameter dicts with required structure: `{'parameter', 'type', 'value', 'weight'}`

2. **Modified decision methods** to emit visualization events:
   - Each method now checks if `visualization_manager` exists
   - Extracts request parameters from `harmonic_context`
   - Calls `visualization_manager.update_request_params(mode, request_params)`
   - Added debug logging: `🎨 VIZ DEBUG: Emitting request params from _*_decision`

### Test Results

**File: `test_decision_visualization.py`**

```
✅ SUCCESS: All three decision methods emitted visualization events
✅ Call 1: 4 parameters, all valid structure (chord, key, consonance, similarity)
✅ Call 2: 4 parameters, all valid structure (chord, key, consonance, contrast)
✅ Call 3: 4 parameters, all valid structure (chord, key, consonance, innovation)
```

Each decision emits:
- Harmonic parameters: chord (Cmaj7), key (C major), consonance (0.85)
- Behavioral parameter: mode-specific (similarity/contrast/innovation)

## Next Steps

1. **Test in live performance**: Run MusicHal_9000.py and verify request parameters viewport updates
2. **Check phrase memory viewport**: Emissions were already added to `phrase_generator.py`, but may need similar fix
3. **Verify timeline viewport**: Human input events are emitting, need to check if they display

## Technical Notes

- **Decision Flow**: `ai_agent.process_event()` → `behavior_engine.decide_behavior()` → `_generate_decision()` → `_generate_single_decision()` → `_imitate/_contrast/_lead_decision()`
- **Visualization Thread Safety**: Events use `Qt.QueuedConnection` for cross-thread signals
- **Parameter Format**: Viewport expects list of dicts with `{'parameter': str, 'type': str, 'value': str, 'weight': float}`
- **Fallback Path**: System uses fallback path when `phrase_generator` is None (which it always is in current config)

## Architectural Insight

This revealed that MusicHal has **two parallel generation paths**:

1. **NEW path** (designed but unused): `phrase_generator.generate_phrase()` - creates full musical phrases with rhythmic structure
2. **OLD path** (actually used): `_generate_single_decision()` - creates individual note decisions

The viewports were designed for the NEW path but the system runs the OLD path. This fix bridges that gap by emitting events from the OLD path.

**Future consideration**: Either fully enable phrase_generator (pass rhythm_oracle to AIAgent) OR remove unused phrase_generator code to simplify architecture.
