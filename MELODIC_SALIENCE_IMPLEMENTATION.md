# 🎵 Melodic Salience Detection - Implementation Complete

## Overview
Added intelligent melodic voice extraction from polyphonic audio (e.g., piano with melody + accompaniment).

## What Was Added

### 1. **MelodicSalienceDetector Class**
Location: `audio_file_learning/polyphonic_processor.py` (lines 46-158)

**Purpose**: Identifies which voice in polyphonic audio is the melody using multiple heuristics.

**Heuristics (weighted scoring system)**:
1. **Highest Pitch** (weight: 3.0)
   - Top note is often melody in piano music
   
2. **Loudest Pitch** (weight: 2.0)
   - Melody is often emphasized dynamically
   
3. **Pitch Continuity** (weight: 4.0) - **MOST IMPORTANT**
   - Melody moves smoothly (steps/small leaps)
   - Harmony jumps around
   - Tracks last melodic note for continuity
   
4. **Spectral Brightness** (weight: 1.5)
   - Melody often has brighter timbre
   - Bonus for pitches C4 and above when centroid > 3kHz
   
5. **Avoid Bass Register** (weight: -2.0)
   - Lowest note is likely bass, not melody
   - Only penalized if there are 3+ voices

### 2. **Integration with PolyphonicAudioProcessor**
Location: `audio_file_learning/polyphonic_processor.py` (lines 634-653)

**Changes**:
- Melodic detector initialized in `__init__`
- Called during event creation in `create_event_from_multi_pitch`
- **CRITICAL**: Main `event.f0` and `event.midi` now point to the **melodic voice**, not the loudest voice

**New Event Attributes**:
```python
event.melodic_voice_idx  # Index of melodic voice in polyphonic arrays
event.voice_roles        # ["melody", "harmony", "bass"] for each voice
event.melodic_pitch      # F0 of melodic voice
event.melodic_midi       # MIDI of melodic voice
event.is_melody          # Boolean: is this event melodic?
```

### 3. **Enhanced MultiPitchFrame Dataclass**
Location: `audio_file_learning/polyphonic_processor.py` (lines 42-43)

**New Fields**:
```python
melodic_voice: Optional[int]        # Index of melodic voice
voice_roles: Optional[List[str]]    # Role tags for each voice
```

## How It Works

### Example: Piano Playing Melody + Chords

**Input Frame**:
- Pitch 1: 130 Hz (C3) - Bass note, loud
- Pitch 2: 262 Hz (C4) - Chord tone, medium
- Pitch 3: 392 Hz (G4) - Melody note, smooth continuation from previous frame

**Scoring**:
```
Pitch 1 (C3):
  + 2.0 (loudest)
  - 2.0 (lowest/bass)
  = 0.0

Pitch 2 (C4):
  + 0.0 (not highest, not loudest)
  = 0.0

Pitch 3 (G4):
  + 3.0 (highest)
  + 4.0 (smooth continuation from previous melody note)
  + 1.5 (bright register)
  = 8.5 ✅ WINNER
```

**Result**: Pitch 3 (G4) identified as melody

### Continuity Tracking

The detector maintains a **pitch history** (last 10 melodic notes) to:
- Detect smooth melodic contours
- Penalize large leaps (more likely harmony)
- Reward stepwise motion (very melodic)

**Interval Scoring**:
- 0-2 semitones (step): +4.0 points
- 3-5 semitones (small leap): +2.0 points
- 6-12 semitones (octave): +1.0 points
- >12 semitones (large leap): 0 points

## Impact on Training

### Before Melodic Salience:
```
❌ event.midi = loudest_pitch (often bass or chord root)
❌ Melody buried in polyphonic data
❌ Training learns bass/harmony, not melody
```

### After Melodic Salience:
```
✅ event.midi = melodic_pitch (intelligent extraction)
✅ Melody explicitly identified and tagged
✅ Training learns actual melodic patterns
✅ event.voice_roles = ["melody", "harmony", "bass"]
```

## Next Steps

### 2. Enhance GPT-OSS Analysis
Add melodic analysis prompts to extract:
- Melodic phrase boundaries
- Melodic contour characteristics
- Melody vs. accompaniment patterns

### 3. Create Melodic Event Stream
Filter training events by `event.is_melody == True` to create a pure melodic training set.

## Testing

To test the melodic salience detection:

```bash
python Chandra_trainer.py --file input_audio/Curious_child.wav --training-events 1000
```

Then analyze the model:

```python
import json
model = json.load(open('JSON/Curious_child_[timestamp]_model.json'))

melodic_events = []
for i in range(len(model['audio_frames'])):
    frame = model['audio_frames'][str(i)]
    audio_data = frame['audio_data']
    if audio_data.get('is_melody'):
        melodic_events.append(audio_data)

print(f"Melodic events: {len(melodic_events)}")
print(f"Melodic MIDI range: {min(e['midi'] for e in melodic_events)} to {max(e['midi'] for e in melodic_events)}")
```

## Expected Results

For piano instrumental with clear melody:
- **Before**: 1.2% melodic content (35 events)
- **After**: 30-50% melodic content (900-1500 events)
- **Melodic range**: Should shift from bass-heavy to mid-high register

---
*Implementation Date: October 4, 2025*
*Status: Ready for Testing*
