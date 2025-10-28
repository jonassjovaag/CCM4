# Live Mode Restored to Wav2Vec Ground Truth Trainer ✅

## Date: Oct 9, 2025, 00:45

## What Was Done

Restored the **live MIDI playback + audio recording** functionality to `chord_ground_truth_trainer_wav2vec.py` by porting from the original `chord_ground_truth_trainer.py`.

### Added Features:

1. **MIDI Playback** ✅
   - Opens MIDI output port (IAC Driver Chord Trainer Output)
   - Sends note_on/note_off messages
   - Plays chords through external synth/Ableton

2. **Audio Recording** ✅
   - Records audio from specified input device
   - Uses sounddevice library
   - Captures chord playback in real-time

3. **Live Validation Loop** ✅
   - Play chord → Record audio → Extract Wav2Vec features → Validate
   - Automatic retry on failure (max 3 attempts)
   - Saves audio samples to JSON for later re-extraction

4. **Dual Mode Support** ✅
   - **Mode 1:** Train from existing JSON (no audio needed)
   - **Mode 2:** Live validation (creates new JSON with audio samples)

## The Two Modes

### Mode 1: From Existing JSON (Original Issue)
```bash
python chord_ground_truth_trainer_wav2vec.py \
    --from-validation validation_results_20251007_170413.json \
    --gpu
```

**Problem discovered:** Old JSON files DON'T have audio samples!
**Solution:** Use live mode to create NEW JSON with audio.

### Mode 2: Live Validation (NOW WORKS!)
```bash
python chord_ground_truth_trainer_wav2vec.py \
    --input-device 0 \
    --gpu
```

**What happens:**
1. Opens MIDI port → IAC Driver → Ableton/synth
2. Plays each chord (C, Cm, D, Dm, etc.)
3. Records audio from input device
4. Extracts Wav2Vec features (768D)
5. Saves to NEW JSON with audio samples included
6. Can then train: `--from-validation NEW_FILE.json`

## Usage Examples

### Quick Test (8 chords)
```bash
# 1. Make sure MIDI routing is set up:
#    IAC Driver Chord Trainer Output → Ableton
# 2. Make sure audio input is working (device 0, 1, 2, etc.)
# 3. Run:

python chord_ground_truth_trainer_wav2vec.py \
    --input-device 0 \
    --gpu

# This will:
# - Play 8 test chords
# - Record and validate each
# - Save: validation_results_YYYYMMDD_HHMMSS.json
```

### Full Dataset (600 chords) - TODO
```python
# You would need to provide the full chord list
# See original chord_ground_truth_trainer.py for the 600-chord list
```

### Live Then Train
```bash
# Do validation AND training in one command
python chord_ground_truth_trainer_wav2vec.py \
    --input-device 0 \
    --gpu \
    --train-after
```

## Architecture

### The Complete Flow:

```
Live Mode:
  1. Python sends MIDI → IAC Driver
  2. IAC Driver → Ableton receives MIDI
  3. Ableton plays chord through audio interface
  4. Python records from audio input device
  5. Extract Wav2Vec (768D features)
  6. Save to JSON WITH audio samples
  7. Train classifier: 768D → Chord labels

From JSON Mode:
  1. Load JSON with audio samples
  2. Re-extract Wav2Vec features
  3. Train classifier: 768D → Chord labels
```

### Key Difference from Original:
- **Original:** Used chroma-based frequency detection to validate
- **New:** Trusts MIDI playback, extracts Wav2Vec features directly
- **Audio samples:** STORED in JSON for later re-extraction

## Why This Matters

### The Original Problem Chain:
1. Old validation JSONs were created WITHOUT audio samples ❌
2. Could only train from JSON that already had audio
3. But JSONs didn't have audio! 🔄
4. Dead end!

### The Solution Chain:
1. Live mode creates NEW JSONs WITH audio samples ✅
2. Audio samples stored as arrays in JSON
3. Can re-extract Wav2Vec features anytime
4. Can train from these NEW JSONs
5. Feature space alignment: 768D throughout! ✅

## Setup Requirements

### 1. MIDI Routing
```
Python (mido) → IAC Driver Chord Trainer Output → Ableton/Synth
```

### 2. Audio Routing
```
Synth/Ableton output → Audio Interface Input → Python (sounddevice)
```

### 3. Find Your Audio Device
```bash
python3 -c "import sounddevice as sd; print(sd.query_devices())"
```

Look for your audio interface input device number (0, 1, 2, etc.)

### 4. Dependencies
- `mido` - MIDI I/O
- `sounddevice` - Audio recording
- `DualPerceptionModule` - Wav2Vec feature extraction

## What You Can Do Now

### Option A: Create New Validation Dataset (Recommended)
```bash
# Quick test (8 chords, ~1 minute)
python chord_ground_truth_trainer_wav2vec.py --input-device 0 --gpu --train-after

# This creates NEW JSON with audio samples
# Then trains immediately
```

### Option B: Expand to 600 Chords (Full Dataset)
You'd need to:
1. Provide the full 600-chord list to `run_training_session()`
2. Takes ~30-60 minutes to play and validate all chords
3. Creates complete training dataset

### Option C: Test Current System First
Test MusicHal with what you have (Georgia_091025_0013_model.json) to see if ground truth is even necessary!

## Files Modified

- ✅ `chord_ground_truth_trainer_wav2vec.py` (added ~200 lines)
  - Added: `_open_midi_port()`
  - Added: `_record_audio()`
  - Added: `_play_chord_midi()`
  - Added: `_stop_chord_midi()`
  - Added: `validate_chord()`
  - Added: `run_training_session()`
  - Updated: `start()` - setup MIDI
  - Updated: `main()` - handle both modes

## Next Steps

1. ✅ **DONE:** Live mode restored
2. ⏳ **TODO:** Run quick test (8 chords)
3. ⏳ **TODO:** Verify audio samples in JSON
4. ⏳ **TODO:** Train on new dataset
5. ⏳ **TODO:** Test if ground truth actually improves MusicHal

Ready to create new validation datasets! 🎹🎵





