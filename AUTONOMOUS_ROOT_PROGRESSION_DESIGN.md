# Autonomous Root Progression Design

## Your Vision: Two-Layer Harmonic Control

### Layer 1: Manual Waypoints (Your Input)
```
0 min  → 220 Hz (A3)    "Start here"
10 min → 330 Hz (E4)    "Shift to here"  
20 min → 440 Hz (A4)    "Move higher"
30 min → 261 Hz (C4)    "Return to tonic area"
```

### Layer 2: Autonomous Exploration (System Decides)
While anchored around your waypoint (e.g., 220 Hz), system discovers:
- Relative relationships (minor/major tonic, subdominant, dominant)
- Chromatic neighbors
- Patterns from training data
- Responses to live input

**Example (220 Hz waypoint = A3)**:
```
Minutes 0-10 (anchored to A3/220 Hz):
  0-2 min:  220 Hz (A3)  → stable on waypoint
  2-4 min:  264 Hz (C4)  → discovered relative major (III)
  4-6 min:  293 Hz (D4)  → moved to subdominant (IV)  
  6-8 min:  247 Hz (B3)  → chromatic neighbor exploration
  8-10 min: 220 Hz (A3)  → return to waypoint (preparing for shift)
  
Minutes 10-20 (transition + anchor to E4/330 Hz):
  10-12 min: 275 Hz      → gradual transition (halfway)
  12-14 min: 330 Hz (E4) → arrive at new waypoint
  14-16 min: 392 Hz (G4) → discover minor third above
  ...
```

## Current System Capabilities

### ✅ Already Captured During Training
From `Chandra_trainer.py` line 1674:
```python
event['fundamental_freq'] = float(closest_segment['ratio_analysis'].fundamental)
event['consonance'] = closest_segment['consonance']
```

**What this means**:
- Every training event has a detected fundamental frequency (Groven method)
- System knows "Token 42 happened when fundamental was 220 Hz with consonance 0.8"
- Patterns encode implicit harmonic relationships

### ❌ Not Currently Stored in Model
Audio frames in trained model contain:
- `features` (15D vector)
- `audio_data` (raw waveform)
- `timestamp`, `frame_id`

**Missing**: `fundamental_freq`, `consonance`

### 🔍 What We Need to Enable Your Vision

## Design Proposal: Three-Component System

### Component 1: Harmonic Data Storage (TRAINING)

**Modify Chandra_trainer.py to store harmonic info in AudioOracle:**

```python
# During training, after detecting fundamental:
if 'fundamental_freq' in event and event['fundamental_freq'] > 0:
    audio_oracle.fundamentals[state_id] = event['fundamental_freq']
    audio_oracle.consonances[state_id] = event['consonance']
```

**Add to AudioOracle serialization (save to JSON):**
```python
# In polyphonic_audio_oracle.py::to_dict()
data['fundamentals'] = self.fundamentals  # Dict[state_id -> Hz]
data['consonances'] = self.consonances    # Dict[state_id -> 0-1]
```

This creates a **harmonic map** of the training data:
- "State 142: fundamental 220 Hz, consonance 0.85"
- "State 857: fundamental 264 Hz, consonance 0.72"

### Component 2: Waypoint System (PERFORMANCE ARC)

**Two-tier structure in performance arc JSON:**

```json
{
  "waypoints": [
    {
      "time": 0.0,
      "root_hz": 220.0,
      "comment": "Start on A3"
    },
    {
      "time": 600.0,
      "root_hz": 330.0,
      "comment": "10 min: shift to E4"
    },
    {
      "time": 1200.0,
      "root_hz": 440.0,
      "comment": "20 min: move to A4"
    }
  ],
  "exploration_config": {
    "max_drift_semitones": 7,
    "return_probability": 0.3,
    "transition_duration": 120.0,
    "learning_weight": 0.5,
    "input_response_weight": 0.3
  }
}
```

### Component 3: Autonomous Root Discovery (LIVE PERFORMANCE)

**Algorithm runs every 30-60 seconds:**

```python
class AutonomousRootExplorer:
    def __init__(self, audio_oracle, waypoints, config):
        self.audio_oracle = audio_oracle
        self.waypoints = waypoints
        self.config = config
        self.current_target_root = None
        self.exploration_history = []
    
    def update(self, elapsed_time, recent_input_fundamental=None):
        # 1. Get current waypoint + next waypoint
        current_waypoint = self._get_waypoint_at_time(elapsed_time)
        next_waypoint = self._get_next_waypoint(elapsed_time)
        
        # 2. Calculate transition phase (0.0 = at current, 1.0 = at next)
        transition_phase = self._calculate_transition_phase(
            elapsed_time, 
            current_waypoint, 
            next_waypoint
        )
        
        # 3. Decide: explore or transition?
        if transition_phase < 0.1:  # Not transitioning yet
            # EXPLORE around current waypoint
            new_root = self._explore_harmonically(
                current_waypoint['root_hz'],
                recent_input_fundamental
            )
        elif transition_phase > 0.9:  # Almost at next waypoint
            # STABILIZE on next waypoint
            new_root = next_waypoint['root_hz']
        else:  # Mid-transition
            # INTERPOLATE between waypoints
            new_root = self._interpolate_roots(
                current_waypoint['root_hz'],
                next_waypoint['root_hz'],
                transition_phase
            )
        
        self.current_target_root = new_root
        return new_root
    
    def _explore_harmonically(self, anchor_root, input_fundamental):
        """
        Discover interesting roots near anchor, informed by:
        1. Training data patterns (what fundamentals appear in learned states)
        2. Live input (what you're playing now)
        3. Musical theory (common intervals: P4, P5, m3, M3)
        """
        
        # Find states in AudioOracle with fundamentals near anchor
        candidate_states = []
        for state_id, fundamental in self.audio_oracle.fundamentals.items():
            # Within max_drift_semitones of anchor
            semitone_distance = 12 * np.log2(fundamental / anchor_root)
            if abs(semitone_distance) <= self.config['max_drift_semitones']:
                candidate_states.append({
                    'state_id': state_id,
                    'fundamental': fundamental,
                    'consonance': self.audio_oracle.consonances[state_id],
                    'semitone_distance': semitone_distance
                })
        
        if not candidate_states:
            return anchor_root  # Stay at anchor
        
        # Score each candidate
        scores = []
        for candidate in candidate_states:
            score = 0.0
            
            # 1. Prefer roots that appear frequently in training
            state_frequency = self._get_state_frequency(candidate['state_id'])
            score += state_frequency * self.config['learning_weight']
            
            # 2. Respond to live input (if you're playing E, bias toward E)
            if input_fundamental:
                input_distance = abs(
                    12 * np.log2(candidate['fundamental'] / input_fundamental)
                )
                input_match = np.exp(-input_distance / 3.0)  # Decay
                score += input_match * self.config['input_response_weight']
            
            # 3. Prefer consonant states
            score += candidate['consonance'] * 0.2
            
            # 4. Slight bias toward returning to anchor
            return_bias = np.exp(-abs(candidate['semitone_distance']) / 5.0)
            score += return_bias * self.config['return_probability']
            
            # 5. Prefer common intervals (P4=5, P5=7, m3=3, M3=4)
            common_intervals = [0, 3, 4, 5, 7, 12]  # Unison, m3, M3, P4, P5, 8ve
            closest_interval = min(common_intervals, 
                                   key=lambda x: abs(x - abs(candidate['semitone_distance'])))
            if abs(closest_interval - abs(candidate['semitone_distance'])) < 0.5:
                score += 0.3  # Boost if close to common interval
            
            scores.append(score)
        
        # Choose weighted random (not always highest - adds variety)
        weights = np.array(scores)
        weights = np.exp(weights) / np.sum(np.exp(scores))  # Softmax
        chosen_idx = np.random.choice(len(candidate_states), p=weights)
        
        chosen_root = candidate_states[chosen_idx]['fundamental']
        
        # Log for transparency
        self.exploration_history.append({
            'time': time.time(),
            'anchor': anchor_root,
            'chosen': chosen_root,
            'interval': 12 * np.log2(chosen_root / anchor_root),
            'input_fundamental': input_fundamental,
            'reason': 'exploration'
        })
        
        return chosen_root
```

## How It Works in Practice

### Example: 60-Minute Concert

**Your input (simple!):**
```json
{
  "waypoints": [
    {"time": 0, "root_hz": 220.0},    // A3
    {"time": 600, "root_hz": 330.0},  // E4 at 10 min
    {"time": 1200, "root_hz": 440.0}, // A4 at 20 min  
    {"time": 1800, "root_hz": 294.0}, // D4 at 30 min
    {"time": 2400, "root_hz": 220.0}, // A3 at 40 min
    {"time": 3000, "root_hz": 220.0}, // A3 at 50 min
    {"time": 3600, "root_hz": 220.0}  // A3 at 60 min (resolution)
  ],
  "exploration_config": {
    "max_drift_semitones": 7,
    "return_probability": 0.3
  }
}
```

**What system does (autonomous):**

**Minutes 0-10 (Waypoint: A3/220 Hz)**
```
0:00  → 220 Hz (A3)     [at waypoint]
1:30  → 264 Hz (C4)     [discovered: +3 semitones, major third, consonance 0.85]
3:00  → 294 Hz (D4)     [explored: +5 semitones, perfect fourth]
4:30  → 247 Hz (B3)     [response to input: you played B]
6:00  → 220 Hz (A3)     [returned to anchor: probability triggered]
7:30  → 196 Hz (G3)     [explored downward: -2 semitones]
9:00  → 231 Hz (A#3)    [preparing transition: halfway to E]
```

**Minutes 10-12 (Transition to E4)**
```
10:00 → 275 Hz          [interpolating: 0.5 * (220 + 330)]
11:00 → 303 Hz          [interpolating: 0.75 * (220 + 330)]
12:00 → 330 Hz (E4)     [arrived at waypoint]
```

**Minutes 12-20 (Waypoint: E4/330 Hz)**
```
12:00 → 330 Hz (E4)     [at waypoint]
13:30 → 392 Hz (G4)     [discovered: +3 semitones above E]
15:00 → 440 Hz (A4)     [explored: +5 semitones, P4 above E]
... (continues exploring around E)
```

## Advantages of This Design

### ✅ You Control the Arc
- Simple waypoint list defines overall harmonic journey
- Transition timing explicit (happens over 2 minutes before waypoint)
- No complex programming needed

### ✅ System Has Autonomy
- Discovers roots based on **what it learned** (training data fundamentals)
- Responds to **what you play** (live input fundamental)
- Uses **musical theory** (common intervals, consonance)
- Creates **variety** (weighted random, not deterministic)

### ✅ Perceptual Foundation Preserved
- All in Hz, not symbolic chord names
- Biases AudioOracle queries (soft guidance)
- 768D Wav2Vec still primary decision-maker
- Root is contextual hint, not constraint

### ✅ Transparent & Debuggable
- Exploration history logs every decision
- Can see: "At 3:45, chose 264 Hz because training data showed high frequency + you played C"
- Fits artistic research transparency goals

## Implementation Checklist

### Phase 1: Store Harmonic Data (TRAINING)
- [ ] Add `fundamentals` dict to PolyphonicAudioOracle
- [ ] Add `consonances` dict to PolyphonicAudioOracle  
- [ ] Store during training (Chandra_trainer.py line ~1674)
- [ ] Serialize to JSON (polyphonic_audio_oracle.py::to_dict())
- [ ] Deserialize from JSON (polyphonic_audio_oracle.py::from_dict())
- [ ] Retrain model to capture harmonic map

### Phase 2: Waypoint System (PERFORMANCE ARC)
- [ ] Create waypoint JSON schema
- [ ] Add `AutonomousRootExplorer` class
- [ ] Integrate with PerformanceTimelineManager
- [ ] Pass current_target_root to PerformanceState

### Phase 3: AudioOracle Integration (LIVE GENERATION)
- [ ] Modify generate_with_request() to accept root_hint_hz
- [ ] Bias state selection toward matching fundamental
- [ ] Weight by consonance if tension_target specified
- [ ] Maintain 768D gesture tokens as primary

### Phase 4: Live Input Listening (RESPONSE)
- [ ] Extract fundamental from live audio input
- [ ] Pass to AutonomousRootExplorer
- [ ] Bias exploration toward what you're playing
- [ ] Balance: 50% training, 30% input, 20% theory

### Phase 5: Testing & Refinement
- [ ] Create test waypoint file (simple A→E→A progression)
- [ ] Run 15-minute test performance
- [ ] Log all root decisions with reasoning
- [ ] Verify: exploration stays within max_drift_semitones
- [ ] Verify: transitions happen smoothly
- [ ] Adjust weights based on musical results

## Open Questions for Discussion

1. **Exploration frequency**: How often should system reconsider root?
   - Every phrase (6-10 seconds)?
   - Every mode change (30-90 seconds)?
   - Fixed interval (60 seconds)?

2. **Transition style**: 
   - Linear interpolation (smooth glide)?
   - Stepwise (chromatic neighbors)?
   - Sudden jump at boundary?

3. **Input response**: 
   - Should system **follow** your root immediately?
   - Or **complement** (play different root for tension)?
   - Weight adjustable in config?

4. **Drift limits**:
   - 7 semitones (perfect fifth) reasonable?
   - Or smaller (3-4 semitones = major third)?

5. **Return probability**:
   - Should system always return to waypoint before transition?
   - Or allow direct jump from explored root to next waypoint?

6. **Training data bias**:
   - If Itzama.wav mostly uses A minor, system will prefer A-C-D-E-F
   - Is this desired (learned style) or limiting (needs broader training)?

## Next Steps

**Before implementation, we should discuss:**

1. Does this autonomous exploration match your vision?
2. What parameters feel right? (drift limits, transition duration, etc.)
3. How should system balance: training data vs. live input vs. music theory?
4. Do you want **full autonomy** or **semi-guided** (e.g., "prefer minor thirds")?

Once we align on design, implementation is straightforward - the architecture supports it!
