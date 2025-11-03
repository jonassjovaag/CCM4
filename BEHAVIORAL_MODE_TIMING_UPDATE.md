# Behavioral Mode Timing & Variety Update

## Problem Identified

After analyzing the 5-minute test session logs, we discovered:

### Mode Distribution Issues:
- **Shadow mode: 62.8%** (118/188 events) - dominated performance
- **Contrast mode: 30.3%** (57/188 events)
- **Lead mode: 4.8%** (9/188 events)
- **Imitate mode: 2.1%** (4/188 events)

### Transition Problems:
- Only **~4 mode transitions** in 5 minutes
- Modes persisting 60-90 seconds each
- Insufficient variety between behavioral personalities
- User feedback: "I would like more pause between modes, and also more variety between them"

## Root Causes

1. **Mode duration too long**: `min_mode_duration = 30s`, `max_mode_duration = 90s`
2. **Weak transition probability**: Only 30% chance to change after minimum duration
3. **Parameter contrast needs enhancement**: Modes felt too similar musically

## Changes Implemented

### 1. Mode Duration Adjustment (agent/behaviors.py lines 266-269)

**BEFORE:**
```python
self.min_mode_duration = 30.0  # seconds (stay in mode longer)
self.max_mode_duration = 90.0  # seconds (sustained identity)
self.current_mode_duration = random.uniform(30.0, 90.0)
```

**AFTER:**
```python
self.min_mode_duration = 15.0  # seconds (shorter for more transitions)
self.max_mode_duration = 45.0  # seconds (prevent single-mode dominance)
self.current_mode_duration = random.uniform(15.0, 45.0)
```

**Impact:** 
- Modes now persist **15-45 seconds** instead of 30-90 seconds
- Expect **6-12 mode transitions** in 5 minutes instead of 3-4
- More dynamic AI personality changes while maintaining coherence

### 2. Transition Probability Increase (agent/behaviors.py lines 433-437)

**BEFORE:**
```python
if mode_duration >= self.min_mode_duration:
    return random.random() < 0.3  # 30% chance
```

**AFTER:**
```python
if mode_duration >= self.min_mode_duration:
    return random.random() < 0.5  # 50% chance (more variety)
```

**Impact:**
- 50% chance (up from 30%) to transition after minimum duration
- Creates more dynamic mode changes while maintaining musical coherence

### 3. Enhanced Mode Parameter Contrast

#### SHADOW Mode (lines 48-57) - The Follower
**Changes:**
- `temperature`: 0.3 → **0.2** (ultra predictable)
- `phrase_variation`: 0.05 → **0.02** (minimal variation)
- `response_delay`: 0.1 → **0.05** (ultra immediate)
- `volume_factor`: 0.6 → **0.5** (much quieter shadow)
- `note_density`: 1.5 → **1.8** (dense chatter)

**Musical Character:** Quiet, immediate, predictable follower with dense note texture

#### COUPLE Mode (lines 68-77) - The Independent Partner
**Changes:**
- `request_weight`: 0.2 → **0.1** (almost independent)
- `temperature`: 1.8 → **2.0** (ultra wild exploration)
- `phrase_variation`: 0.9 → **0.95** (maximum variation)
- `response_delay`: 3.0 → **4.0** (very long delay)
- `volume_factor`: 1.2 → **1.3** (louder presence)
- `note_density`: 0.6 → **0.5** (sparser for more space)

**Musical Character:** Loud, independent, wildly exploratory with sparse texture

#### CONTRAST Mode (lines 90-99) - The Counterpoint
**Changes:**
- `similarity_threshold`: 0.15 → **0.1** (more independent)
- `request_weight`: 0.3 → **0.2** (lower following)
- `temperature`: 1.5 → **1.6** (higher exploration)
- `phrase_variation`: 0.85 → **0.9** (maximum variation)
- `response_delay`: 1.5 → **2.0** (longer delay)
- `volume_factor`: 1.0 → **1.1** (louder)
- `note_density`: 0.7 → **0.6** (sparser)

**Musical Character:** Bass counterpoint with strong independence and dramatic contrast

## Expected Results

### Mode Balance:
- **Target distribution**: 20-30% each mode (instead of 62% shadow dominance)
- **More transitions**: 6-12 mode changes per 5 minutes
- **Clearer personality shifts**: Listeners should perceive distinct AI behaviors

### Musical Variety:
- **SHADOW → COUPLE transition**: Quiet follower → loud independent partner (dramatic shift)
- **SHADOW → CONTRAST transition**: High register chatter → bass counterpoint (textural shift)
- **Parameter ranges expanded**: Temperature (0.2-2.0), volume (0.5-1.3), density (0.5-1.8)

### Timing Dynamics:
- **15-45 second mode duration** provides:
  - Minimum 15s: Establishes recognizable personality
  - Maximum 45s: Prevents boredom, allows phrase completion
  - Random duration: Prevents mechanical predictability
  - 50% transition probability: Creates natural variety without flickering

## Testing Instructions

1. **Run new 5-minute test session:**
   ```bash
   python MusicHal_9000.py --enable-rhythmic
   ```

2. **Analyze logs for mode distribution:**
   ```bash
   python analyze_latest.py
   ```

3. **Check for:**
   - More balanced mode distribution (no single mode > 40%)
   - 6-12 mode transitions in 5 minutes
   - Clearer musical personality shifts
   - Listeners should easily identify "quiet follower" vs "loud independent" moments

## Musical Trade-offs

### Coherence vs. Variety Balance:
- **Original (30-90s)**: High coherence, low variety, can feel monotonous
- **New (15-45s)**: Balanced coherence/variety, more dynamic personality
- **Risk**: If too short (<10s), AI behavior becomes incoherent/flickering

### Design Principles Maintained:
- **Sticky modes** (IRCAM research): Still present at 15-45s minimum
- **Recognizable personality**: 15s establishes character before transition
- **Phrase completion**: 45s maximum allows musical phrases to develop
- **Random duration**: Avoids mechanical metronome-like transitions

## Research Context

This adjustment responds to real-world artistic research feedback. The original 30-90s duration was based on IRCAM research (Thelle & Wærstad, 2023) for "recognizable personality." However, in practice with piano improvisation, this created:

1. **Single-mode dominance** (shadow 62.8%)
2. **Insufficient variety** for engaging musical partnership
3. **User experience**: "more pause between modes, and also more variety between them"

The new 15-45s range balances:
- **Artistic goal**: Dynamic, engaging AI musical partner
- **Research validity**: Still maintains sticky behavioral coherence
- **User preference**: More variety and clearer mode transitions

## Documentation Updates

Updated files:
- `agent/behaviors.py`: Lines 48-57 (SHADOW params), 68-77 (COUPLE params), 90-99 (CONTRAST params), 266-269 (mode duration), 433-437 (transition probability)

Related documentation:
- `COMPLETE_ARTISTIC_RESEARCH_DOCUMENTATION.md`: Should be updated to reflect this timing adjustment as part of iterative practice-based research
- `AUTONOMOUS_BEHAVIOR_ARCHITECTURE.md`: May need update with new timing ranges

---

**Quick Summary:**
- Mode duration: **30-90s → 15-45s** (2x faster transitions)
- Transition probability: **30% → 50%** (more variety)
- Parameter contrast: **Enhanced** (temperature 0.2-2.0, volume 0.5-1.3, density 0.5-1.8)
- Expected: **6-12 transitions per 5 minutes** instead of 3-4
- Goal: **More dynamic AI personality with clearer behavioral shifts**
