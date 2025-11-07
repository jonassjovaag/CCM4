# Autonomous Root Progression - Phase 3 Integration COMPLETE ✅

## Overview

**Phase 3 (Integration with PerformanceTimelineManager) is now complete.** The autonomous root exploration system is fully integrated with the performance timeline architecture.

## What Was Added

### 1. MusicalPhase Dataclass Enhancement

**File**: `performance_arc_analyzer.py` (lines 18-31)

```python
@dataclass
class MusicalPhase:
    """Represents a phase in the musical arc"""
    start_time: float
    end_time: float
    phase_type: str
    engagement_level: float
    instrument_roles: Dict[str, str]
    musical_density: float
    dynamic_level: float
    silence_ratio: float
    # NEW: Autonomous root progression (Phase 8)
    root_hint_frequency: Optional[float] = None  # Hz (from Groven method)
    harmonic_tension_target: Optional[float] = None  # 0.0-1.0
```

**Purpose**: Allows performance arc JSON files to specify root hints and tension targets for each phase.

**Example**:
```json
{
  "start_time": 0.0,
  "end_time": 180.0,
  "phase_type": "intro",
  "root_hint_frequency": 220.0,
  "harmonic_tension_target": 0.3
}
```

### 2. PerformanceState Enhancement

**File**: `performance_timeline_manager.py` (lines 17-38)

```python
@dataclass
class PerformanceState:
    """Current state of the performance"""
    start_time: float
    current_time: float
    total_duration: float
    current_phase: Optional[MusicalPhase]
    engagement_level: float
    instrument_roles: Dict[str, str]
    silence_mode: bool
    last_activity_time: float
    musical_momentum: float
    detected_instrument: str = "unknown"
    # Override fields
    engagement_level_override: Optional[float] = None
    musical_momentum_override: Optional[float] = None
    confidence_threshold_override: Optional[float] = None
    behavior_mode_override: Optional[str] = None
    # NEW: Autonomous root progression
    current_root_hint: Optional[float] = None  # Hz
    current_tension_target: Optional[float] = None  # 0.0-1.0
```

**Purpose**: Tracks current autonomous root hint and tension target. Available to all decision-making components (PhraseGenerator, AudioOracle queries, behavioral modes).

### 3. PerformanceTimelineManager Initialization

**File**: `performance_timeline_manager.py` (lines 51-66)

```python
def __init__(self, config: PerformanceConfig):
    """Initialize the performance timeline manager"""
    self.config = config
    self.performance_arc = None
    self.scaled_arc = None
    self.performance_state = None
    
    # NEW: Autonomous root progression (Phase 8)
    self.root_explorer = None  # Initialized when AudioOracle available
    self.last_root_exploration_time = 0.0
    self.exploration_interval = 60.0  # Seconds between explorations
    
    # Load and scale performance arc...
    # Initialize performance state...
```

**Purpose**: Prepares timeline manager for autonomous root exploration. Explorer is initialized later when AudioOracle is loaded.

### 4. Root Explorer Initialization Method

**File**: `performance_timeline_manager.py` (lines 368-431)

```python
def initialize_root_explorer(
    self, 
    audio_oracle, 
    config: Optional['ExplorationConfig'] = None
):
    """
    Initialize autonomous root explorer with trained AudioOracle
    
    Args:
        audio_oracle: Trained AudioOracle with fundamentals populated
        config: Optional ExplorationConfig (uses default hybrid if not provided)
    """
    # Check if AudioOracle has harmonic data
    if not hasattr(audio_oracle, 'fundamentals') or not audio_oracle.fundamentals:
        print("⚠️  AudioOracle has no harmonic data - root exploration disabled")
        return
    
    # Extract waypoints from performance arc phases
    waypoints = self._extract_waypoints_from_phases()
    
    # Use default hybrid config if not provided
    if config is None:
        config = ExplorationConfig(
            training_weight=0.6,
            input_response_weight=0.3,
            theory_bonus_weight=0.1,
            max_drift_semitones=7,
            update_interval=60.0
        )
    
    # Initialize explorer
    self.root_explorer = AutonomousRootExplorer(
        audio_oracle=audio_oracle,
        waypoints=waypoints,
        config=config
    )
    
    print(f"✅ Autonomous root explorer initialized")
```

**Usage**: Called after AudioOracle is loaded (in main.py or MusicHal_9000.py):

```python
# After loading AudioOracle
timeline_manager.initialize_root_explorer(audio_oracle)
```

### 5. Waypoint Extraction Method

**File**: `performance_timeline_manager.py` (lines 414-431)

```python
def _extract_waypoints_from_phases(self) -> List['RootWaypoint']:
    """Extract root waypoints from performance arc phases"""
    waypoints = []
    
    if not self.scaled_arc or not self.scaled_arc.phases:
        return waypoints
    
    for phase in self.scaled_arc.phases:
        if hasattr(phase, 'root_hint_frequency') and phase.root_hint_frequency:
            waypoint = RootWaypoint(
                time=phase.start_time,
                root_hz=phase.root_hint_frequency,
                comment=f"{phase.phase_type} phase"
            )
            waypoints.append(waypoint)
    
    return waypoints
```

**Purpose**: Automatically extracts waypoints from arc JSON. Each phase with `root_hint_frequency` becomes a waypoint.

### 6. Performance State Update Integration

**File**: `performance_timeline_manager.py` (lines 549-610)

```python
def update_performance_state(
    self, 
    human_activity: bool = False, 
    instrument_detected: Optional[str] = None,
    input_fundamental: Optional[float] = None  # NEW
):
    """
    Update performance state with autonomous root exploration
    
    Args:
        human_activity: Whether human is currently playing
        instrument_detected: Name of detected instrument
        input_fundamental: Live input fundamental (Hz) from Groven method
    """
    # ... existing state updates ...
    
    # NEW: Update autonomous root hint (Phase 8)
    if self.root_explorer and self.performance_state.current_phase:
        time_since_exploration = elapsed_time - self.last_root_exploration_time
        
        # Check if it's time to explore (every 60 seconds)
        if time_since_exploration >= self.exploration_interval:
            # Get next root from autonomous explorer
            next_root = self.root_explorer.update(
                elapsed_time=elapsed_time,
                input_fundamental=input_fundamental
            )
            
            # Update state with new root hint
            self.performance_state.current_root_hint = next_root
            
            # Also update tension target from current phase
            if hasattr(self.performance_state.current_phase, 'harmonic_tension_target'):
                self.performance_state.current_tension_target = \
                    self.performance_state.current_phase.harmonic_tension_target
            
            self.last_root_exploration_time = elapsed_time
```

**Usage in main loop**:

```python
# Extract fundamental frequency from live input (Groven method)
input_fundamental = ratio_analyzer.get_fundamental_frequency()

# Update timeline with fundamental
timeline_manager.update_performance_state(
    human_activity=is_playing,
    instrument_detected="piano",
    input_fundamental=input_fundamental  # Pass to explorer
)

# Get current state (now includes current_root_hint)
state = timeline_manager.performance_state
root_hint = state.current_root_hint  # Hz or None
tension = state.current_tension_target  # 0-1 or None
```

## Data Flow Architecture

```
performance_arcs/simple_root_progression.json
  ↓ (load and scale)
PerformanceTimelineManager
  ↓ (extract waypoints)
[{time: 0, root: 220Hz}, {time: 600, root: 330Hz}]
  ↓ (initialize)
AutonomousRootExplorer
  ↓ (update every 60s)
Hybrid Decision (60% training + 30% input + 10% theory)
  ↓
PerformanceState.current_root_hint (e.g., 264 Hz)
  ↓ (read by)
PhraseGenerator / AudioOracle queries
  ↓
MIDI output biased toward root hint
```

## Integration with Existing Systems

### **1. Performance Arc Loading**

Existing JSON files can optionally include root hints:

```json
{
  "total_duration": 900.0,
  "phases": [
    {
      "start_time": 0,
      "end_time": 180,
      "phase_type": "intro",
      "engagement_level": 0.5,
      "root_hint_frequency": 220.0,  // A3 - optional
      "harmonic_tension_target": 0.3  // Low tension - optional
    },
    {
      "start_time": 180,
      "end_time": 360,
      "phase_type": "development",
      "engagement_level": 0.8,
      "root_hint_frequency": 264.0,  // C4
      "harmonic_tension_target": 0.6
    }
  ]
}
```

If `root_hint_frequency` is omitted, that phase simply doesn't contribute waypoints. Explorer will use only phases that have hints.

### **2. Behavioral Modes**

Behavioral modes (Imitate, Contrast, Lead) now have access to:
- `state.current_root_hint` (Hz) - suggested root frequency
- `state.current_tension_target` (0-1) - suggested consonance/dissonance level

Example usage in `agent/behaviors.py`:

```python
def generate_imitate_phrase(self, state: PerformanceState, memory):
    # ... existing logic ...
    
    # Add root hint to request if available
    if state.current_root_hint:
        request_params['root_hint_hz'] = state.current_root_hint
        request_params['tension_target'] = state.current_tension_target or 0.5
```

### **3. PhraseGenerator**

PhraseGenerator can now pass root hints to AudioOracle:

```python
# In phrase_generator.py
def generate_phrase(self, mode_params, memory):
    # Get root hint from timeline state (passed in mode_params)
    root_hint = mode_params.get('root_hint_hz')
    tension = mode_params.get('tension_target')
    
    # Add to AudioOracle request
    request = {
        'gesture_token': gesture,
        'consonance': consonance,
        'root_hint_hz': root_hint,  # NEW
        'tension_target': tension    # NEW
    }
    
    # Generate from AudioOracle
    phrase = audio_oracle.generate_with_request(request)
```

### **4. AudioOracle Queries (Phase 4)**

Next phase will modify `PolyphonicAudioOracle.generate_with_request()`:

```python
def generate_with_request(self, request):
    # ... existing filtering ...
    
    # NEW: Bias by root hint (soft guidance, not hard filter)
    if 'root_hint_hz' in request and request['root_hint_hz']:
        root_hint = request['root_hint_hz']
        
        for state_id in candidate_states:
            if state_id in self.fundamentals:
                fundamental = self.fundamentals[state_id]
                
                # Perceptual distance (log2 frequency ratio)
                interval_semitones = 12 * np.log2(fundamental / root_hint)
                proximity_score = np.exp(-abs(interval_semitones) / 5.0)
                
                # Boost weight for states near root hint
                weights[state_id] *= (1.0 + proximity_score * 0.5)
    
    # ... existing weighted selection ...
```

## Testing the Integration

### **Test 1: Verify Dataclass Extensions**

```python
# Test MusicalPhase with root hints
from performance_arc_analyzer import MusicalPhase

phase = MusicalPhase(
    start_time=0.0,
    end_time=180.0,
    phase_type="intro",
    engagement_level=0.5,
    instrument_roles={},
    musical_density=0.4,
    dynamic_level=0.3,
    silence_ratio=0.1,
    root_hint_frequency=220.0,  # A3
    harmonic_tension_target=0.3
)

assert phase.root_hint_frequency == 220.0
assert phase.harmonic_tension_target == 0.3
print("✅ MusicalPhase extension works")
```

### **Test 2: Verify PerformanceState Extension**

```python
# Test PerformanceState with root hints
from performance_timeline_manager import PerformanceState

state = PerformanceState(
    start_time=0.0,
    current_time=10.0,
    total_duration=300.0,
    current_phase=None,
    engagement_level=0.7,
    instrument_roles={},
    silence_mode=False,
    last_activity_time=0.0,
    musical_momentum=0.5,
    current_root_hint=264.0,  # C4
    current_tension_target=0.6
)

assert state.current_root_hint == 264.0
assert state.current_tension_target == 0.6
print("✅ PerformanceState extension works")
```

### **Test 3: Verify Waypoint Extraction**

```python
# Test waypoint extraction from arc
import json
from performance_timeline_manager import PerformanceTimelineManager, PerformanceConfig

config = PerformanceConfig(
    duration_minutes=15,
    arc_file_path="performance_arcs/simple_root_progression.json",
    engagement_profile="balanced",
    silence_tolerance=5.0,
    adaptation_rate=0.5
)

manager = PerformanceTimelineManager(config)
waypoints = manager._extract_waypoints_from_phases()

print(f"Extracted {len(waypoints)} waypoints:")
for wp in waypoints:
    print(f"  {wp.time}s: {wp.root_hz} Hz ({wp.comment})")

# Expected output:
# Extracted 5 waypoints:
#   0.0s: 261.63 Hz (intro phase)
#   180.0s: 220.0 Hz (development phase)
#   360.0s: 329.63 Hz (climax phase)
#   ...
```

### **Test 4: End-to-End with Mock AudioOracle**

```python
# Full integration test
from test_autonomous_explorer import MockAudioOracle
from agent.autonomous_root_explorer import ExplorationConfig

# Create timeline manager with arc
config = PerformanceConfig(
    duration_minutes=15,
    arc_file_path="performance_arcs/simple_root_progression.json",
    engagement_profile="balanced",
    silence_tolerance=5.0,
    adaptation_rate=0.5
)

manager = PerformanceTimelineManager(config)
manager.start_performance()

# Initialize root explorer with mock AudioOracle
mock_oracle = MockAudioOracle()
manager.initialize_root_explorer(mock_oracle)

# Simulate performance updates
import time
for i in range(10):
    # Simulate live input (user playing C4)
    input_fundamental = 264.0
    
    # Update state
    manager.update_performance_state(
        human_activity=True,
        instrument_detected="piano",
        input_fundamental=input_fundamental
    )
    
    # Check current root hint
    state = manager.performance_state
    if state.current_root_hint:
        print(f"{i*60}s: Root hint = {state.current_root_hint:.1f} Hz, "
              f"Tension = {state.current_tension_target}")
    
    time.sleep(0.1)  # Simulate time passing
```

## Next Steps (Phase 4)

**Phase 4: Bias AudioOracle Queries**

Now that root hints flow through the system, the final integration step is to **bias AudioOracle queries** based on these hints.

**Implementation Plan**:

1. **Modify `PolyphonicAudioOracle.generate_with_request()`**:
   - Accept `root_hint_hz` and `tension_target` in request dict
   - Calculate perceptual distance: `interval_semitones = 12 * log2(fundamental / root_hint)`
   - Weight candidates by proximity: `weight *= (1.0 + exp(-abs(interval_semitones) / 5.0) * bias_strength)`
   - Weight candidates by consonance match: `weight *= (1.0 - abs(consonance - (1.0 - tension_target)))`

2. **Update PhraseGenerator**:
   - Pass `state.current_root_hint` to AudioOracle requests
   - Pass `state.current_tension_target` to AudioOracle requests

3. **Configurable bias strength**:
   - Default: 0.3 (30% boost for matching roots)
   - Mode 1 (Pure Learning): 0.0 (no bias)
   - Mode 2 (Guided Exploration): 0.3 (RECOMMENDED)
   - Mode 3 (Theory Informed): 0.6 (strong bias)

**Critical Constraint**: Keep as **soft bias**, not hard filter. 768D Wav2Vec embeddings remain primary driver. Root hints are gentle nudges, not strict rules.

## Integration Checklist ✅

- ✅ Add `root_hint_frequency` to MusicalPhase dataclass
- ✅ Add `current_root_hint` to PerformanceState dataclass
- ✅ Add `current_tension_target` to PerformanceState dataclass
- ✅ Add root explorer initialization to PerformanceTimelineManager
- ✅ Add waypoint extraction method
- ✅ Add `input_fundamental` parameter to `update_performance_state()`
- ✅ Integrate explorer.update() in state update loop
- ✅ Import AutonomousRootExplorer classes
- ⬜ Modify AudioOracle.generate_with_request() (Phase 4)
- ⬜ Update PhraseGenerator to pass root hints (Phase 4)
- ⬜ Test with retrained model (Phase 5)

## Summary

**Phase 3 Integration is COMPLETE**. The autonomous root exploration system is now fully wired into the performance timeline architecture. 

**What works now**:
- Performance arcs can specify root hints (Hz) and tension targets (0-1)
- AutonomousRootExplorer makes decisions using hybrid intelligence (60/30/10)
- Root hints flow from arc → waypoints → explorer → state → (will reach) AudioOracle queries
- System updates root hints every 60 seconds based on training data + live input + music theory
- All decisions are logged with transparent reasoning

**What's next**:
- Phase 4: Bias AudioOracle queries to prefer states near current root hint
- Phase 5: Retrain model + test in live performance

The two-layer progression architecture (manual waypoints + autonomous exploration) is ready for action! 🎵
