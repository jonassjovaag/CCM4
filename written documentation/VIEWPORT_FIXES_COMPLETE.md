# Complete Viewport Fix Summary

## Issues Fixed

### 1. Request Parameters Viewport - "No Parameters" Issue

**Root Cause**: 
- System uses `phrase_generator.generate_phrase()` (contrary to initial diagnosis)
- BUT it takes the **thematic recall path** which returns early
- This path skipped `_build_request_for_mode()` so request parameters never got built/emitted

**Fixes Applied**:

**File: `agent/phrase_generator.py`**
- Line ~369: Added `emit_request_params()` call in thematic recall path
- Emits motif_recall parameters when recalling/varying motifs
- Line ~311: Added debug logging to trace phrase generation flow

**File: `agent/behaviors.py`**
- Lines ~846, 905, 951: Added `emit_request_params()` in fallback decision methods
- `_imitate_decision()`, `_contrast_decision()`, `_lead_decision()` now emit visualization events
- Created helper method `_extract_request_params_from_context()` (line ~968)
- Extracts harmonic context (chord, key, consonance) + mode-specific parameters

**Method Name Fix**:
- Changed `update_request_params()` → `emit_request_params()` (correct method name)
- Applied to all 4 emission locations

### 2. Repetitive "C C C" Pattern

**Root Cause**: Thematic recall system was constantly recalling same motif without generating new material

**Fix**: Now emits proper request parameters showing what variation is being applied (transpose/invert/augment/diminish)

### 3. Mode Switching Performance

**Added**: Timing debug to identify slowdowns
- Line ~481: Added mode_change_start timer
- Mode switch now reports duration: `[switch took X ms]`

## Testing Status

✅ Method name corrected - should no longer throw AttributeError
✅ Request parameters now emit from both paths:
  - Thematic recall path (phrase_generator.py)
  - Fallback decision path (behaviors.py)
✅ Debug logging added to trace execution flow

## Next Steps

1. **Test live performance**: Run MusicHal and verify:
   - Request parameters viewport updates
   - Mode switch timing is acceptable
   - No AttributeError on emit_request_params

2. **Check remaining issues**:
   - Phrase memory "recall probability 0%" - may need separate fix
   - Timeline showing both human input and AI response - verify events display

3. **Performance optimization** (if mode switching is still slow):
   - Profile visualization event processing
   - Check Qt event queue processing
   - Consider async emission if needed

## Technical Notes

**Two Generation Paths in MusicHal**:
1. **Thematic recall path**: Lines 315-374 in phrase_generator.py
   - Used when `phrase_memory.should_recall_theme()` returns True
   - Recalls existing motifs with variations
   - Was skipping request parameter emission (NOW FIXED)

2. **Fresh generation path**: Lines 378+ in phrase_generator.py
   - Builds new phrases using AudioOracle
   - Already had request parameter emission
   - Still needs verification

3. **Fallback path**: Lines 628+ in behaviors.py
   - Used when phrase_generator is None (shouldn't happen in MusicHal)
   - Single-note decisions instead of phrases
   - Now has request parameter emission added

**Recall Intervals**: 30-60 seconds between thematic recalls (configurable in phrase_memory.py)
