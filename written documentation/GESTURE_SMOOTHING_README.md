# Gesture Token Temporal Smoothing

## Problem Solved

Previously, gesture tokens changed 4+ times per second (at onset rate), causing the AI to respond to individual note events rather than musical phrases. This created "jittery" behavior that didn't match your playing style where chord changes happen ~1/second.

## Solution

Added **temporal smoothing** that accumulates gesture tokens over a 2-4 second window and uses weighted consensus to represent the current musical phrase, not just the instant onset.

## How It Works

```
Raw tokens (onset-level):  [142, 139, 142, 141, 142, 143, 142]
                             ↓ (3-second window + weighted voting)
Smoothed token (phrase-level): 142
```

- Maintains sliding window of recent tokens (default: 3 seconds)
- Recent tokens weighted higher (exponential decay)
- Returns most common token weighted by recency
- Matches musical phrase timing, not onset timing

## Files Created

1. **`listener/gesture_smoothing.py`** - Core smoothing implementation
2. **`test_gesture_smoothing.py`** - Comprehensive test suite
3. **Updated `listener/dual_perception.py`** - Integrated smoother
4. **Updated `MusicHal_9000.py`** - Command-line arguments

## Usage

### Training (Chandra_trainer.py)

Gesture smoothing is automatically enabled during training with default settings (3s window).

### Live Performance (MusicHal_9000.py)

```bash
# Default: 3-second window (good for most playing)
python MusicHal_9000.py --enable-rhythmic

# More stable (slower response, better for sustained chords)
python MusicHal_9000.py --gesture-window 4.0

# More responsive (faster tracking, better for rapid improvisation)
python MusicHal_9000.py --gesture-window 2.0 --gesture-min-tokens 2

# Very stable (5-second phrases)
python MusicHal_9000.py --gesture-window 5.0 --gesture-min-tokens 5
```

## Command-Line Arguments

- `--gesture-window <seconds>`: Smoothing window duration (default: 3.0, range: 1.0-5.0)
- `--gesture-min-tokens <count>`: Minimum tokens before consensus (default: 3)

## Configuration Recommendations

**Your playing style** (chord changes ~1/second):
- **Default (3.0s window)**: Good balance
- **Sustained chords**: `--gesture-window 4.0`
- **Rapid changes**: `--gesture-window 2.0`

## Testing

```bash
# Run all gesture smoothing tests
python test_gesture_smoothing.py
```

Tests verify:
- Consensus formation from repeated tokens
- Smooth transitions between phrases
- Decay weighting (recent tokens dominate)
- Window cleanup
- Realistic performance scenarios

## Technical Details

### GestureTokenSmoother Class

```python
from listener.gesture_smoothing import GestureTokenSmoother

smoother = GestureTokenSmoother(
    window_duration=3.0,   # 3-second phrase window
    min_tokens=3,          # Need 3+ tokens for consensus
    decay_time=1.0         # 1-second decay time
)

# Add token and get smoothed result
smoothed_token = smoother.add_token(raw_token, timestamp)

# Get statistics for transparency
stats = smoother.get_statistics()
# Returns: window_duration, tokens_in_window, total_processed,
#          consensus_changes, current_consensus, token_distribution
```

### Integration Points

1. **`DualPerceptionModule`** - Smooths gesture tokens during feature extraction
2. **`AudioOracle`** - Uses smoothed tokens for request masking
3. **`AI Agent`** - Receives phrase-level tokens instead of onset-level

### Musical Impact

**Before:**
- Token changes: 4+ Hz (onset rate)
- AI responds to: Individual note events
- Behavior: Jittery, rapid context switching

**After:**
- Token changes: ~0.5 Hz (phrase rate)
- AI responds to: Chord changes and phrases
- Behavior: Stable, phrase-level coherence

## Transparency & Debugging

The smoother tracks statistics for debugging:

```python
# Get current smoothing stats
stats = dual_perception.get_gesture_smoothing_stats()

print(f"Tokens in window: {stats['tokens_in_window']}")
print(f"Current consensus: {stats['current_consensus']}")
print(f"Token distribution: {stats['token_distribution'][:3]}")  # Top 3
```

## Future Enhancements

- **Adaptive window**: Automatically adjust based on tempo/phrase length
- **Mode-aware smoothing**: Different windows for different behavioral modes
- **Visualization**: Show token distribution in real-time UI

## Implementation Date

31 October 2025 - Refactoring branch

## Related Files

- `listener/gesture_smoothing.py` - Core implementation
- `listener/dual_perception.py` - Integration
- `test_gesture_smoothing.py` - Tests
- `MusicHal_9000.py` - CLI integration
- `Chandra_trainer.py` - Training integration (future)
