# Minimal Ground Truth Test Guide

## Quick Start: Test with 2 Chords Only

**What it tests:** C major + C minor (each with 3 inversions) = 6 total validations

**Time:** ~1-2 minutes

## Setup

### 1. Find Your Audio Input Device
```bash
python3 -c "import sounddevice as sd; print(sd.query_devices())"
```

Look for your audio interface INPUT. Note the device number (e.g., 0, 1, 2).

### 2. MIDI Routing
Make sure:
- **IAC Driver Chord Trainer Output** is visible in Audio MIDI Setup
- Ableton (or your synth) is receiving from IAC Driver
- A piano/synth is loaded and can play

### 3. Audio Routing
Make sure:
- Ableton/synth output goes to your audio interface
- Audio interface input is the device number you found in step 1
- Volume is audible but not clipping

## Run the Test

```bash
python test_ground_truth_minimal.py --input-device 0 --gpu
```

Replace `0` with your actual audio input device number.

## What Happens

```
[1/6] Validating: C (inv=0, oct=0)
   🎹 Sending MIDI: [60, 64, 67] (notes: C)
   ✅ VALIDATED! Wav2Vec features extracted
   📊 Gesture token: 42
   📊 Consonance: 0.87

[2/6] Validating: C (inv=1, oct=0)
   🎹 Sending MIDI: [64, 67, 72] (notes: C)
   ✅ VALIDATED! Wav2Vec features extracted
   📊 Gesture token: 15
   📊 Consonance: 0.85

... (continues for all 6 chords)

🎯 Session Summary:
   Total attempts: 6
   Successful validations: 6
   Failed validations: 0
   Success rate: 100.0%

💾 Validation results saved: validation_results_20251009_HHMMSS.json
   You can now train with: --from-validation validation_results_20251009_HHMMSS.json
```

## What Gets Saved

The JSON file will contain:
- All 6 chord validations
- **Audio samples** for each chord (ready for re-extraction!)
- Wav2Vec features (768D)
- Gesture tokens
- Consonance scores

## Next Steps

### Option 1: Train Immediately
```bash
python test_ground_truth_minimal.py --input-device 0 --gpu --train-after
```

**Note:** Won't actually train because we need 10+ samples. But it will validate the pipeline!

### Option 2: Expand to More Chords
Edit `test_ground_truth_minimal.py` and add more chords to the list:
```python
MINIMAL_CHORD_LIST = [
    # ... existing C and Cm ...
    
    # Add D major
    ("D", "", [62, 66, 69], 0, 0),
    ("D", "", [66, 69, 74], 1, 0),
    ("D", "", [69, 74, 78], 2, 0),
    
    # Add D minor
    ("Dm", "m", [62, 65, 69], 0, 0),
    ("Dm", "m", [65, 69, 74], 1, 0),
    ("Dm", "m", [69, 74, 77], 2, 0),
]
```

Now you'd have 12 chords → enough to train!

### Option 3: Verify JSON Has Audio
```bash
python3 << 'EOF'
import json
import numpy as np

data = json.load(open('validation_results_XXXXXX_XXXXXX.json'))
print(f"Chords validated: {len(data)}")

if len(data) > 0:
    first = data[0]
    audio_samples = first['audio_features'].get('audio_samples', [])
    print(f"First chord has audio: {'YES' if audio_samples else 'NO'}")
    print(f"Audio length: {len(audio_samples)} samples")
    print(f"Duration: {len(audio_samples) / 44100:.2f} seconds")
EOF
```

## Troubleshooting

### "❌ Port not found: IAC Driver Chord Trainer Output"
- Open **Audio MIDI Setup** (macOS)
- Enable IAC Driver
- Create port named "Chord Trainer Output"

### "No audio recorded" or very quiet
- Check Ableton output routing
- Increase volume (but don't clip!)
- Verify input device number is correct

### "Failed to extract Wav2Vec features"
- Check if audio is actually being recorded
- Try CPU mode (remove `--gpu`)
- Check if audio samples are not all zeros

## Success Criteria

✅ All 6 chords validated
✅ JSON file created
✅ JSON contains audio_samples arrays
✅ Wav2Vec features extracted (768D)
✅ Gesture tokens assigned (0-63)

Ready to test! 🎹





