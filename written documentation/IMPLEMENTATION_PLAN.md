# CCM3 Implementation Plan
**Based on Deep Research Analysis (Oct 15, 2024)**  
**Source:** IRCAM papers (2022-2023), AudioOracle, Wav2Vec, Consonance theory

---

## System Overview

### Chandra_trainer.py (Offline Learning)
- Learns from audio files (Georgia.wav, etc.)
- Extracts features → gesture tokens → AudioOracle
- Saves models to JSON
- **No microphone feedback issues**

### MusicHal_9000.py (Live Performance)
- Loads trained models
- Real-time microphone input
- Generates MIDI output
- **Has microphone feedback issues**

---

## Phase 1: CRITICAL FIXES (Stop Bad Learning)

### 1.1 Self-Output Filtering - MusicHal ONLY
**File:** `MusicHal_9000.py`  
**Lines:** 371-463 (`_on_audio_event` method)  
**Time:** 30 minutes  
**Priority:** CRITICAL

**Problem:**
- MusicHal sends MIDI → Synth plays → Microphone hears it → Learns from own output
- ~30% of events (1,173 notes / 3,912 events) are self-generated
- Degrades AudioOracle learning quality

**Solution:**
```python
# Add to __init__:
self.own_output_tracker = {
    'recent_notes': [],  # List of (note, timestamp, velocity)
    'max_age_seconds': 2.0,
    'enabled': True
}

# Add new method:
def _is_own_output(self, event_data, current_time):
    """Check if detected audio matches recently sent MIDI"""
    if not self.own_output_tracker['enabled']:
        return False
    
    # Clean old entries
    self.own_output_tracker['recent_notes'] = [
        (note, t, vel) for note, t, vel in self.own_output_tracker['recent_notes']
        if current_time - t < self.own_output_tracker['max_age_seconds']
    ]
    
    # Check if detected MIDI matches recent output
    detected_midi = event_data.get('midi', 0)
    for note, sent_time, vel in self.own_output_tracker['recent_notes']:
        if note == detected_midi and (current_time - sent_time) < 1.0:
            return True
    
    return False

# Modify _on_audio_event (around line 386):
def _on_audio_event(self, *args):
    # ... existing code ...
    
    # NEW: Check if this is our own output
    if self._is_own_output(event_data, current_time):
        self._update_status_bar(event, event_data)  # Still show it
        return  # Don't learn from it
    
    # Only human input reaches here
    self.memory_buffer.add_moment(event_data)  # Line 459
    self.clustering.add_sequence([event_data])  # Line 463

# Track sent notes (in all MIDI sending code, ~lines 1098-1110):
self.own_output_tracker['recent_notes'].append(
    (midi_params['note'], time.time(), midi_params.get('velocity', 64))
)
```

**NOT needed in Chandra:** Offline learning from files, no feedback loop

---

### 1.2 Gesture Token Extraction - BOTH SYSTEMS

#### A) Chandra_trainer.py - VERIFY ONLY
**Status:** ✅ Already working  
**Evidence:** Georgia model has 50/50 frames with gesture_token  
**Action:** None needed, just verify it continues working

**Verification checklist:**
- [ ] `symbolic_quantizer.py` uses L2 normalization before K-means
- [ ] K-means vocabulary_size = 64
- [ ] Gesture tokens saved to JSON via `_sanitize_audio_data()`
- [ ] Model JSON contains gesture_token field in audio_data

---

#### B) MusicHal_9000.py - FIX NEEDED
**File:** `MusicHal_9000.py`  
**Lines:** ~405-444 (`_on_audio_event` method)  
**Time:** 15 minutes  
**Priority:** CRITICAL

**Problem:**
- Dual perception extracts Wav2Vec features during training
- But live mode doesn't extract gesture tokens from incoming audio
- AudioOracle can't use learned patterns without tokens!

**Solution:**
```python
# In _on_audio_event, around line 405 (after hybrid_perception block):

# NEW: Extract dual perception features (gesture tokens)
if hasattr(self, 'dual_perception') and self.dual_perception:
    try:
        # Get raw audio buffer from event
        audio_buffer = event.raw_audio if hasattr(event, 'raw_audio') and event.raw_audio is not None else None
        
        if audio_buffer is not None and len(audio_buffer) > 0:
            # Extract Wav2Vec features and gesture token
            dual_result = self.dual_perception.extract_features(
                audio_buffer, 
                self.listener.sr, 
                current_time
            )
            
            # Add to event data for AudioOracle
            event_data['gesture_token'] = dual_result.gesture_token
            event_data['wav2vec_features'] = dual_result.wav2vec_features.tolist()
            
            # Debug log (can remove later)
            if self.stats['events_processed'] % 100 == 0:
                print(f"🎵 Gesture token: {dual_result.gesture_token}")
    
    except Exception as e:
        # Log first few errors only
        if not hasattr(self, '_dual_error_count'):
            self._dual_error_count = 0
        if self._dual_error_count < 3:
            print(f"⚠️ Dual perception error: {e}")
            self._dual_error_count += 1
```

**Verify dual_perception initialization:**
- Check if `self.dual_perception` is initialized in `__init__`
- Should be created when `--hybrid-perception --wav2vec` flags used

---

## Phase 2: MUSICAL INTELLIGENCE (Pause & Phrase Detection)

### 2.1 Pause Detection - MusicHal ONLY
**File:** `MusicHal_9000.py` (new class + integration)  
**Time:** 45 minutes  
**Priority:** HIGH

**Problem:**
- No recognition of musical pauses/breathing
- Plays too continuously, doesn't give space
- From NIME2023: "Give space to the machine" required pause awareness

**Solution:**

```python
# Add new class (around line 100, after imports):

class MusicalPauseDetector:
    """Detects musical pauses based on onset gaps and RMS level"""
    
    def __init__(self):
        self.onset_times = []
        self.silence_threshold_db = -50
        self.pause_duration_threshold = 0.5  # seconds
        self.max_history = 10.0  # Keep last 10s of onsets
    
    def add_onset(self, timestamp):
        """Track an onset event"""
        self.onset_times.append(timestamp)
        # Clean old onsets
        self.onset_times = [t for t in self.onset_times 
                           if timestamp - t < self.max_history]
    
    def detect_pause(self, current_time, rms_db):
        """Returns True if currently in a musical pause"""
        if not self.onset_times:
            return True  # No recent onsets = pause
        
        time_since_last_onset = current_time - self.onset_times[-1]
        is_quiet = rms_db < self.silence_threshold_db
        
        return (time_since_last_onset > self.pause_duration_threshold 
                and is_quiet)
    
    def time_since_last_onset(self, current_time):
        """How long since last onset"""
        if not self.onset_times:
            return 999.0
        return current_time - self.onset_times[-1]


# In __init__ (around line 150):
self.pause_detector = MusicalPauseDetector()


# In _on_audio_event (around line 446):
# Detect onsets
if event_data.get('onset', False):
    self.pause_detector.add_onset(current_time)

# Store pause state for use in autonomous generation
self.in_pause = self.pause_detector.detect_pause(
    current_time, 
    event_data.get('rms_db', -80)
)
```

**NOT needed in Chandra:** Offline analysis doesn't need real-time pause detection

---

### 2.2 Phrase Boundary Detection - MusicHal ONLY
**File:** `MusicHal_9000.py` (new class + integration)  
**Time:** 1 hour  
**Priority:** HIGH

**Problem:**
- No understanding of phrase structure
- Responds mid-phrase, interrupts musical flow
- From NIME2023: Better to respond at phrase boundaries

**Solution:**

```python
# Add new class (after MusicalPauseDetector):

class PhraseBoundaryDetector:
    """Detects phrase boundaries based on onset density"""
    
    def __init__(self, window_size=2.0):
        self.window_size = window_size  # Analysis window in seconds
        self.onset_times = []
        self.density_threshold_high = 2.0  # onsets/second for "in phrase"
        self.density_threshold_low = 0.5   # onsets/second for "boundary"
        self.max_history = 10.0
    
    def add_onset(self, timestamp):
        """Track an onset event"""
        self.onset_times.append(timestamp)
        # Clean old onsets
        self.onset_times = [t for t in self.onset_times 
                           if timestamp - t < self.max_history]
    
    def get_onset_density(self, current_time):
        """Calculate onsets per second in recent window"""
        recent = [t for t in self.onset_times 
                 if current_time - t < self.window_size]
        return len(recent) / self.window_size
    
    def is_phrase_boundary(self, current_time):
        """Returns True if at a phrase boundary"""
        density = self.get_onset_density(current_time)
        return density < self.density_threshold_low
    
    def is_in_phrase(self, current_time):
        """Returns True if actively in a phrase"""
        density = self.get_onset_density(current_time)
        return density > self.density_threshold_high
    
    def get_state(self, current_time):
        """Returns 'in_phrase', 'boundary', or 'silence'"""
        density = self.get_onset_density(current_time)
        if density > self.density_threshold_high:
            return 'in_phrase'
        elif density > self.density_threshold_low:
            return 'transition'
        else:
            return 'boundary'


# In __init__:
self.phrase_detector = PhraseBoundaryDetector()


# In _on_audio_event:
if event_data.get('onset', False):
    self.phrase_detector.add_onset(current_time)

self.phrase_state = self.phrase_detector.get_state(current_time)
```

**NOT needed in Chandra:** Offline learning doesn't need real-time phrase detection

---

### 2.3 Adaptive Autonomous Interval - MusicHal ONLY
**File:** `MusicHal_9000.py`  
**Lines:** 1064-1065 (`_autonomous_generation_tick` calls)  
**Time:** 30 minutes  
**Priority:** MEDIUM

**Problem:**
- Fixed 3-second autonomous interval regardless of musical context
- Should respond faster in pauses, slower during human phrases

**Current code:**
```python
# Line ~1064
if self.autonomous_generation_enabled:
    self._autonomous_generation_tick(current_time)
```

**Solution:**

```python
# In __init__ (around line 167):
self.autonomous_interval_base = 3.0  # Base interval
self.last_autonomous_time = 0

# Add new method:
def _calculate_adaptive_interval(self):
    """Calculate autonomous generation interval based on musical context"""
    
    # Fast response in pauses (fill silence)
    if self.in_pause:
        return 1.0
    
    # Slow during active phrases (give space)
    if self.phrase_state == 'in_phrase':
        return 5.0
    
    # Medium at boundaries (good time to respond)
    if self.phrase_state == 'boundary':
        return 2.0
    
    # Default
    return self.autonomous_interval_base

# Modify _main_loop (around line 1064):
if self.autonomous_generation_enabled:
    adaptive_interval = self._calculate_adaptive_interval()
    
    if current_time - self.last_autonomous_time >= adaptive_interval:
        self._autonomous_generation_tick(current_time)
        self.last_autonomous_time = current_time
```

**NOT needed in Chandra:** No real-time generation

---

## Phase 3: BEHAVIOR MODES (User Control)

### 3.1 Behavior Mode Implementation - MusicHal ONLY
**Files:** `agent/behaviors.py` (expand), `MusicHal_9000.py` (integrate)  
**Time:** 2 hours  
**Priority:** HIGH

**Problem:**
- No explicit behavior modes (shadow/mirror/couple)
- User can't control interaction style
- From IRCAM 2023: Core best practice for mixed-initiative

**Solution:**

```python
# In agent/behaviors.py - ADD these modes:

class InteractionMode(Enum):
    """IRCAM-style interaction modes"""
    SHADOW = "shadow"    # Close imitation, quick response
    MIRROR = "mirror"    # Similar with variation, phrase-aware
    COUPLE = "couple"    # Independent but complementary
    # Keep existing modes too...

# Add new class:
class BehaviorController:
    """Manages interaction modes and response timing"""
    
    def __init__(self):
        self.mode = InteractionMode.MIRROR  # Default
        
        # Mode parameters
        self.mode_params = {
            InteractionMode.SHADOW: {
                'similarity_threshold': 0.8,
                'response_delay': 0.2,
                'volume_factor': 0.7  # Quieter than human
            },
            InteractionMode.MIRROR: {
                'similarity_threshold': 0.5,
                'response_delay': 'phrase_aware',  # Special handling
                'volume_factor': 0.8
            },
            InteractionMode.COUPLE: {
                'similarity_threshold': 0.2,
                'response_delay': 2.0,
                'volume_factor': 1.0  # Equal volume
            }
        }
    
    def set_mode(self, mode: InteractionMode):
        """Change interaction mode"""
        self.mode = mode
        print(f"🎭 Behavior mode: {mode.value}")
    
    def get_similarity_threshold(self):
        """Get pattern matching threshold for current mode"""
        return self.mode_params[self.mode]['similarity_threshold']
    
    def get_response_delay(self, phrase_detector):
        """Calculate response delay based on mode and phrase state"""
        delay = self.mode_params[self.mode]['response_delay']
        
        if delay == 'phrase_aware':
            # Wait for phrase boundary in MIRROR mode
            if phrase_detector.is_in_phrase():
                return None  # Don't respond yet
            else:
                return 0.1  # Respond quickly at boundary
        else:
            return delay
    
    def get_volume_factor(self):
        """Get volume scaling for current mode"""
        return self.mode_params[self.mode]['volume_factor']
    
    def filter_pattern_matches(self, matches, similarity_scores):
        """Filter AudioOracle matches based on mode similarity threshold"""
        threshold = self.get_similarity_threshold()
        filtered = []
        for match, score in zip(matches, similarity_scores):
            if score >= threshold:
                filtered.append(match)
        return filtered


# In MusicHal_9000.py __init__:
from agent.behaviors import BehaviorController, InteractionMode

self.behavior_controller = BehaviorController()


# Add MIDI CC control for mode switching:
def _handle_mode_change_cc(self, cc_value):
    """Switch behavior mode via MIDI CC"""
    if cc_value < 42:
        self.behavior_controller.set_mode(InteractionMode.SHADOW)
    elif cc_value < 85:
        self.behavior_controller.set_mode(InteractionMode.MIRROR)
    else:
        self.behavior_controller.set_mode(InteractionMode.COUPLE)


# Integrate with autonomous generation:
def _autonomous_generation_tick(self, current_time):
    # Check if we should respond based on mode
    delay = self.behavior_controller.get_response_delay(self.phrase_detector)
    
    if delay is None:
        return  # Not time to respond yet (waiting for phrase boundary)
    
    # ... existing generation code ...
    
    # Apply volume factor to generated notes
    volume_factor = self.behavior_controller.get_volume_factor()
    midi_params['velocity'] = int(midi_params['velocity'] * volume_factor)
```

**NOT needed in Chandra:** No live interaction

---

## Phase 4: REFINEMENTS

### 4.1 Temporal Smoothing for Chroma - BOTH SYSTEMS
**File:** `listener/hybrid_perception.py`  
**Time:** 20 minutes  
**Priority:** MEDIUM

**Problem:**
- Frame-by-frame jitter in chord detection
- Spurious chord changes

**Solution:**

```python
# In hybrid_perception.py, after chroma extraction:

from scipy.ndimage import median_filter

# Apply temporal smoothing
chroma_smoothed = median_filter(chroma, size=(1, 5))  # Smooth over 5 frames
```

**Applies to:** Both Chandra (offline) and MusicHal (live)

---

### 4.2 Variable Markov Oracle Adaptive Thresholds - BOTH SYSTEMS
**File:** `memory/polyphonic_audio_oracle.py`  
**Time:** 1 hour  
**Priority:** LOW (future improvement)

**Problem:**
- Fixed distance threshold for pattern matching
- VMO paper shows adaptive is better

**Solution:**

```python
# In PolyphonicAudioOracle class:

def _calculate_adaptive_threshold(self, recent_features, base_threshold=0.3):
    """Calculate adaptive threshold based on local feature variance (VMO)"""
    if len(recent_features) < 10:
        return base_threshold
    
    # Calculate variance in recent window
    variance = np.var(recent_features, axis=0)
    mean_variance = np.mean(variance)
    
    # Scale threshold based on variance
    # High variance → higher threshold (be more selective)
    # Low variance → lower threshold (be more lenient)
    return base_threshold * (1 + mean_variance)
```

**Applies to:** Both systems (core AudioOracle improvement)

---

## Testing & Validation

### After Phase 1 (Critical Fixes):
- [ ] Run MusicHal, verify no self-learning (Events counter shouldn't jump from MIDI sends)
- [ ] Check logs for gesture_token values appearing
- [ ] Verify AudioOracle uses tokens (check phrase context logs)

### After Phase 2 (Musical Intelligence):
- [ ] Play with pauses, verify machine doesn't fill every gap
- [ ] Play continuous phrases, verify machine waits for boundaries
- [ ] Check autonomous interval adapts (log interval values)

### After Phase 3 (Behavior Modes):
- [ ] Test SHADOW mode: Quick imitation
- [ ] Test MIRROR mode: Phrase-aware responses
- [ ] Test COUPLE mode: Independent playing
- [ ] Verify MIDI CC switches modes

### Overall Success Criteria:
1. ✅ No feedback loops (100% human input learned)
2. ✅ Gesture tokens used in live performance
3. ✅ Musical pauses recognized and respected
4. ✅ Phrase boundaries detected
5. ✅ Three behavior modes working
6. ✅ Adaptive timing based on context
7. ✅ User reports "much better" musical coherence

---

## File Reference Quick Guide

### Core Files to Modify:

**MusicHal_9000.py**
- Lines 371-463: `_on_audio_event` (self-filter + gesture tokens)
- Lines ~100: Add MusicalPauseDetector and PhraseBoundaryDetector classes
- Lines ~150: Initialize detectors in `__init__`
- Lines 1064: Adaptive autonomous interval

**agent/behaviors.py**
- Add BehaviorController class
- Add InteractionMode enum

**listener/hybrid_perception.py**
- Add temporal smoothing to chroma

**memory/polyphonic_audio_oracle.py**
- (Future) Add adaptive thresholds

### Files to Verify (No Changes):

**Chandra_trainer.py**
- Gesture tokens already working
- Just verify continued functionality

**listener/symbolic_quantizer.py**
- K-means with L2 normalization working
- No changes needed

**listener/dual_perception.py**
- Wav2Vec extraction working
- No changes needed

---

## Implementation Order

### Session 1 (1 hour): Critical Fixes
1. Self-output filtering in MusicHal
2. Gesture token extraction in MusicHal live mode
3. Test both fixes

### Session 2 (2 hours): Musical Intelligence
1. Add MusicalPauseDetector class
2. Add PhraseBoundaryDetector class
3. Integrate onset tracking
4. Adaptive autonomous interval
5. Test pause and phrase detection

### Session 3 (2 hours): Behavior Modes
1. Add BehaviorController to agent/behaviors.py
2. Integrate with MusicHal
3. Add MIDI CC control
4. Test all three modes

### Session 4 (1 hour): Refinements & Testing
1. Temporal smoothing for chroma
2. Final integration testing
3. Performance tuning

**Total estimated time: 6 hours across 4 sessions**

---

## Notes

- **Chandra is mostly correct** - focus on MusicHal
- **Self-output filtering is CRITICAL** - do this first
- **Gesture tokens** must work in live mode for AudioOracle to be useful
- **Pause/phrase detection** addresses NIME2023 core finding
- **Behavior modes** are IRCAM 2023 best practice
- All changes are **additive** - won't break existing functionality

---

**Last Updated:** October 15, 2024  
**Based on:** RESEARCH_ANALYSIS_REPORT.md





