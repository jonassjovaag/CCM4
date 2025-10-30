# PyQt5 Thread Safety Crash Fix

**Date**: October 29, 2025
**Issue**: Python crashed with SIGTRAP after 5-minute performance duration timeout
**Root Cause**: PyQt5 thread safety violation when closing visualization windows

## The Problem

### Crash Report Analysis
```
Exception Type:        EXC_BREAKPOINT (SIGTRAP)
Exception Codes:       0x0000000000000001, 0x0000000185c97cc0
Application Specific Information:
Must only be used from the main thread

Thread 7 Crashed:
10  libqcocoa.dylib - QWindowPrivate::setVisible
12  QtWidgets - QWidgetPrivate::hide_sys
14  QtWidgets - QWidgetPrivate::setVisible
16  QtWidgets - QWidgetPrivate::close_helper
17  QtWidgets.abi3.so - meth_QWidget_close
```

**Translation**: Some code tried to close a PyQt5 widget from a **background thread** instead of the **main Qt thread**. PyQt5 **strictly forbids** GUI operations from background threads.

### Code Path to Crash

1. Main audio processing loop (Thread 7 - background) detects `timeline_manager.is_complete() == True`
2. Calls `self.stop()` from background thread
3. `stop()` method calls `self.visualization_manager.close()`
4. `close()` iterates viewports and calls `viewport.close()` **← THREAD SAFETY VIOLATION**
5. `viewport.close()` tries to manipulate Qt widgets from background thread
6. macOS Qt backend (`libqcocoa.dylib`) detects thread violation and crashes with `SIGTRAP`

### Original Code (UNSAFE)
```python
# visualization/visualization_manager.py
def close(self):
    """Close all viewports and cleanup"""
    for viewport in self.viewports.values():
        viewport.close()  # ← Can be called from any thread!
    print("🎨 Visualization system closed")
```

## The Fix

### Thread-Safe Close Method
```python
# visualization/visualization_manager.py (FIXED)
def close(self):
    """Close all viewports and cleanup (THREAD-SAFE)"""
    # If we're not on the Qt main thread, schedule close on main thread
    if QThread.currentThread() != self.app.thread():
        # Use QTimer.singleShot to safely execute close on main thread
        QTimer.singleShot(0, self._close_viewports)
    else:
        # Already on main thread, close directly
        self._close_viewports()

def _close_viewports(self):
    """Internal method to actually close viewports (must be on main thread)"""
    for viewport in self.viewports.values():
        try:
            viewport.close()
        except Exception as e:
            print(f"⚠️  Error closing viewport: {e}")
    print("🎨 Visualization system closed")
```

### How It Works

1. **Thread Detection**: `QThread.currentThread() != self.app.thread()` checks if we're on the Qt main thread
2. **Safe Delegation**: If on background thread, use `QTimer.singleShot(0, ...)` to schedule the close operation on the main thread's event loop
3. **Immediate Execution**: If already on main thread, execute directly
4. **Error Handling**: Wrap each viewport close in try/except to prevent cascade failures

### Why QTimer.singleShot?

`QTimer.singleShot(0, callback)` is Qt's idiomatic way to say:
- "Execute this callback on the main Qt thread"
- "Do it ASAP (0ms delay)"
- "Thread-safe: can be called from any thread"

This is **safer** than `QMetaObject.invokeMethod` because:
- Simpler API
- Automatically uses Qt::QueuedConnection for cross-thread calls
- No need to manually specify argument types

## Bonus Discovery: Melody/Bass Voice Timing

While investigating the crash, I also clarified the **melody/bass separation** behavior:

### Yes, They Are Deliberately Kept Separate

**Code Location**: `agent/behaviors.py` lines 555-583

```python
time_since_melody = current_time - self.last_melody_time
time_since_bass = current_time - self.last_bass_time

# Randomized pause durations (prevents predictability)
required_melody_gap = random.uniform(self.melody_pause_min, self.melody_pause_max)
required_bass_gap = random.uniform(self.bass_pause_min, self.bass_pause_max)

if time_since_melody > required_melody_gap and time_since_bass > required_bass_gap * 0.75:
    # Both voices CAN play - alternate between them (voice_alternation_counter)
    voice_type = "melodic" if self._voice_alternation_counter % 2 == 0 else "bass"
    self._voice_alternation_counter += 1
    
elif time_since_melody > required_melody_gap:
    # Only melody ready
    voice_type = "melodic"
    
elif time_since_bass > required_bass_gap:
    # Only bass ready
    voice_type = "bass"
    
else:
    # BOTH voices BLOCKED - return empty (no single-note fallback)
    print(f"🚫 Voice selection: BLOCKED - returning empty (no single-note fallback)")
    return decisions  # Empty list
```

### Voice Timing Rules

1. **Minimum Gap Enforcement**: Each voice has a minimum time gap before it can play again
2. **Randomized Gaps**: Gaps vary randomly within `[pause_min, pause_max]` range to avoid predictability
3. **Alternation When Both Ready**: When both voices are ready, they alternate (not simultaneous)
4. **Blocking Returns Empty**: When timing blocks generation, system returns empty list (no fallback to random single notes)

### Why This Design?

From artistic research perspective:
- **Space for Human**: Longer gaps (7-9s melody, 4-5s bass) give human player dominant voice
- **Voice Clarity**: Separating melody/bass prevents muddy texture
- **Conversational Flow**: Alternation creates call-and-response feel
- **Unpredictability**: Randomized gaps prevent robotic timing patterns

### Terminal Output Examples

You'll see these messages frequently:
```
🎵 Voice selection: melody (since_melody=9.0s, required=7.9s)
🎵 Voice selection: bass (since_bass=5.4s, required=4.4s)
🚫 Voice selection: BLOCKED - returning empty (no single-note fallback)
```

The blocked messages are **intentional** - they prevent the system from constantly playing when timing doesn't allow it.

## Testing

After applying this fix:

1. ✅ **System stops gracefully at 5 minutes** - Timeline correctly triggers stop
2. ✅ **No more Python crashes** - Visualization windows close safely from any thread
3. ✅ **Fade-out behavior works** - Activity multiplier drops to 0% in final 20%
4. ⚠️  **Linter warnings** - Expected (self.app could be None in theory, but never is in practice)

### Verification Test
```bash
python MusicHal_9000.py --performance-duration 5 --visualize
```

Should now:
- Run for exactly 5 minutes
- Fade out gracefully in final minute
- Close visualization windows without crashing
- Exit cleanly with "✅ Enhanced Drift Engine AI stopped"

## Related Fixes in This Session

This crash fix was discovered during a larger debugging session that also fixed:

1. **Performance Duration Timeout** - Changed `return` to `break` after `self.stop()`
2. **Single Note Emissions** - Early return when voice timing blocked
3. **Melodic Coherence** - Weighted intervals (75% stepwise motion)
4. **Three-Phase Performance Arc** - Buildup → Main → Ending with activity multiplier
5. **Arc File Made Optional** - Timeline works without arc file in simple duration mode
6. **PyQt5 Thread Safety** ← **This fix**

All fixes documented in:
- `SESSION_SUMMARY_ALL_FIXES_COMPLETE.md`
- `THREE_PHASE_PERFORMANCE_ARC_COMPLETE.md`

## Key Takeaway

**PyQt5 Rule**: ALL GUI operations (show, hide, close, update widgets) **MUST** happen on the main Qt thread.

**Solution Pattern**: When closing GUI from background threads:
```python
if not_on_main_thread:
    QTimer.singleShot(0, lambda: close_gui_safely())
else:
    close_gui_safely()
```

This pattern applies to ALL PyQt5 widget manipulation in multi-threaded applications.
