# Performance Arc CLI Configuration - Complete

## Summary

Successfully implemented command-line configuration for performance arc file paths in `MusicHal_9000.py`. Users can now specify custom performance arc files when running timed performances.

## Changes Made

### 1. Command-Line Argument (`main()` function)

**Location**: Line ~2836  
**Added**:
```python
parser.add_argument('--performance-arc', type=str, 
                   default='ai_learning_data/itzama_performance_arc.json',
                   help='Path to performance arc JSON file (default: itzama_performance_arc.json)')
```

### 2. Constructor Parameter (`EnhancedDriftEngineAI.__init__()`)

**Location**: Line ~181  
**Added parameter**:
```python
def __init__(self, midi_port: Optional[str] = None, input_device: Optional[int] = None, 
             enable_rhythmic: bool = True, enable_mpe: bool = True, performance_duration: int = 0,
             performance_arc_path: Optional[str] = None,  # NEW
             enable_hybrid_perception: bool = False, ...
```

**Stored as instance variable** (Line ~336):
```python
self.performance_arc_path = performance_arc_path
```

### 3. Timeline Manager Initialization

**Location**: Line ~420  
**Updated call**:
```python
if self.performance_duration > 0:
    self._initialize_timeline_manager(self.performance_arc_path)
```

**Method signature** (Line ~474):
```python
def _initialize_timeline_manager(self, arc_file_path: Optional[str] = None):
    """Initialize performance timeline manager"""
    try:
        config = PerformanceConfig(
            duration_minutes=self.performance_duration,
            arc_file_path=arc_file_path or "ai_learning_data/itzama_performance_arc.json",  # Fallback
            engagement_profile="balanced",
            silence_tolerance=5.0,
            adaptation_rate=0.1
        )
        self.timeline_manager = PerformanceTimelineManager(config)
        print(f"🎭 Performance timeline initialized: {self.performance_duration} minutes")
    except Exception as e:
        print(f"⚠️ Failed to initialize timeline manager: {e}")
        self.timeline_manager = None
```

### 4. Main Function Integration

**Location**: Line ~2859  
**Updated instantiation**:
```python
drift_ai = EnhancedDriftEngineAI(
    midi_port=args.midi_port,
    input_device=args.input_device,
    enable_rhythmic=not args.no_rhythmic,
    enable_mpe=not args.no_mpe,
    performance_duration=args.performance_duration,
    performance_arc_path=args.performance_arc,  # NEW - passes CLI argument
    enable_hybrid_perception=not args.no_hybrid_perception,
    ...
)
```

## Usage Examples

### Default Itzama Arc
```bash
python MusicHal_9000.py --performance-duration 5
```
Uses: `ai_learning_data/itzama_performance_arc.json`

### Custom Subtle_southern Arc
```bash
python MusicHal_9000.py --performance-duration 18 \
    --performance-arc ai_learning_data/subtle_southern_performance_arc.json
```
Uses: `ai_learning_data/subtle_southern_performance_arc.json` (1109s / 18.5 minutes)

### Relative or Absolute Paths
```bash
# Relative path
python MusicHal_9000.py --performance-duration 10 \
    --performance-arc JSON/custom_arc.json

# Absolute path
python MusicHal_9000.py --performance-duration 5 \
    --performance-arc /Users/jonashsj/custom_arcs/experimental_arc.json
```

## Integration with Training Pipeline

### Automatic Arc Generation (`Chandra_trainer.py`)

**Location**: Lines ~827–845  
During training, performance arcs are automatically saved:

```python
# Extract base filename for arc file
arc_filename = os.path.splitext(os.path.basename(audio_file))[0]
arc_file_path = os.path.join("ai_learning_data", f"{arc_filename}_performance_arc.json")

# Serialize performance arc
arc_dict = performance_arc.to_dict()
with open(arc_file_path, 'w') as f:
    json.dump(arc_dict, f, indent=2)
print(f"✅ Performance arc saved: {arc_file_path}")
print(f"   Duration: {arc_dict.get('total_duration_seconds', 0):.1f}s")
print(f"   Phases: {len(arc_dict.get('phases', []))}")
```

### Training → Performance Workflow

1. **Train on audio file**:
   ```bash
   python Chandra_trainer.py --file input_audio/Subtle_southern.wav \
       --output JSON/subtle_southern_training.json
   ```
   
2. **Auto-generated files**:
   - `JSON/Subtle_southern_{timestamp}_training_model.json` (AudioOracle)
   - `JSON/Subtle_southern_{timestamp}_training_rhythm_oracle.json` (RhythmOracle)
   - `ai_learning_data/subtle_southern_performance_arc.json` ✨ (NEW)

3. **Perform with trained arc**:
   ```bash
   python MusicHal_9000.py --performance-duration 18 \
       --performance-arc ai_learning_data/subtle_southern_performance_arc.json
   ```

## Performance Arc Structure

**Example**: `ai_learning_data/subtle_southern_performance_arc.json`

```json
{
  "total_duration_seconds": 1109.0,
  "phases": [
    {
      "start_time": 0.0,
      "end_time": 23.456,
      "dominant_quality": "consonant",
      "intensity": "moderate",
      "density": 0.6,
      "melodic_range": 12,
      "rhythmic_complexity": 0.45
    },
    // ... 33 more phases
  ],
  "engagement_curve": [[0.0, 0.5], [1.2, 0.7], ...],  // 95522 points
  "silence_patterns": [
    {
      "start_time": 45.3,
      "duration": 2.8,
      "type": "phrase_boundary"
    },
    // ... 5 more patterns
  ],
  "instrument_evolution": [...],
  "dynamic_evolution": [...]
}
```

## Timeline Manager Behavior

**With arc file specified**:
- Loads phases, engagement curves, silence patterns from JSON
- Scales to requested `--performance-duration` (proportionally adjusts timings)
- Guides behavioral modes through performance structure
- Applies fade-out during final minute

**Without arc file** (no `--performance-duration` or missing arc):
- Timeline manager not initialized (`None`)
- Performance runs indefinitely (Ctrl+C to stop)
- Behavioral modes use default heuristics (no structured arc guidance)

## Technical Notes

### Type Safety
- `performance_arc_path: Optional[str] = None` allows `None` or string path
- Fallback chain: CLI arg → instance variable → default path
- No type errors with `Optional[str]` hint

### Backward Compatibility
- If `--performance-arc` not specified, uses default `itzama_performance_arc.json`
- If `--performance-duration` is 0 (default), timeline manager not initialized
- Existing behavior preserved for indefinite-duration performances

### Error Handling
- If arc file not found: warning printed, timeline manager set to `None`
- Performance continues without structured arc (graceful degradation)
- Check logs for "⚠️ Failed to initialize timeline manager" if issues occur

## Related Documentation

- **Performance Arc Analysis**: `performance_arc_analyzer.py`
- **Performance Duration Bug Fix**: Line ~348 in `performance_timeline_manager.py` (fixed `start_time` reset bug)
- **Training Integration**: `Chandra_trainer.py` lines ~574 (analysis) and ~827–845 (serialization)
- **Three-Pipeline Architecture**: See `.github/copilot-instructions.md` for full context

## Testing

```bash
# Quick test: 5-minute performance with default arc
python MusicHal_9000.py --performance-duration 5

# Full test: 18-minute performance with custom arc
python MusicHal_9000.py --performance-duration 18 \
    --performance-arc ai_learning_data/subtle_southern_performance_arc.json \
    --enable-rhythmic

# Verify arc loading
grep "🎭 Performance timeline initialized" logs/latest.log
grep "Performance arc loaded" logs/latest.log
```

## Future Enhancements

- [ ] Add `--list-arcs` command to show available arc files in `ai_learning_data/`
- [ ] Add arc validation on load (check phases, duration, required fields)
- [ ] Support arc interpolation/blending (mix multiple arcs)
- [ ] Add `--arc-scaling` mode: `strict` (exact durations) vs. `adaptive` (stretch/compress)
- [ ] Generate arc previews during training (visualization of phases/engagement)

---

**Status**: ✅ Complete and tested  
**Date**: 2025-01-05  
**Branch**: `refactoring`  
**Related Issues**: Performance duration bug (fixed), arc auto-save (added), CLI flexibility (implemented)
