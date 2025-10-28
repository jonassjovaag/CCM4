# Session Complete: Minimal Test Ready! 🎯

## What We Accomplished

1. ✅ **Restored live MIDI playback mode** to Wav2Vec ground truth trainer
2. ✅ **Created minimal test script** - Just 2 chords (C, Cm) with inversions = 6 validations
3. ✅ **Complete setup guide** with troubleshooting

## The Minimal Test

**File:** `test_ground_truth_minimal.py`

**What it tests:**
- C major: Root, 1st inversion, 2nd inversion
- C minor: Root, 1st inversion, 2nd inversion
- **Total:** 6 chords (~1-2 minutes)

**Why this is perfect:**
- Quick validation that everything works
- Tests MIDI routing
- Tests audio recording
- Tests Wav2Vec extraction
- Creates JSON with audio samples
- Minimal time investment!

## How to Run

### 1. Find Audio Device
```bash
python3 -c "import sounddevice as sd; print(sd.query_devices())"
```

### 2. Run Test
```bash
python test_ground_truth_minimal.py --input-device 0 --gpu
```

Replace `0` with your audio input device number.

### 3. What You'll See
```
[1/6] Validating: C (inv=0, oct=0)
   🎹 Sending MIDI: [60, 64, 67]
   ✅ VALIDATED! Wav2Vec features extracted
   📊 Gesture token: 42
   📊 Consonance: 0.87

... (continues for all 6 chords)

💾 Validation results saved: validation_results_YYYYMMDD_HHMMSS.json
```

## Success Criteria

After the test, you should have:
- ✅ JSON file created
- ✅ 6 chord validations
- ✅ Audio samples stored in JSON
- ✅ Wav2Vec features (768D) extracted
- ✅ Gesture tokens assigned (0-63)

## Next Steps After Minimal Test

### If It Works:
1. **Expand the chord list** (add D, Dm, E, Em, etc.)
2. Get to 10+ chords → Can train classifier
3. Or go straight to full 600-chord dataset

### If It Fails:
- Check MIDI routing (IAC Driver → Ableton)
- Check audio routing (Ableton → Audio Interface Input)
- Check device number is correct
- Check volume levels
- See MINIMAL_TEST_GUIDE.md for troubleshooting

## Files Created

1. ✅ `test_ground_truth_minimal.py` - Minimal test script (6 chords)
2. ✅ `MINIMAL_TEST_GUIDE.md` - Complete setup & troubleshooting guide
3. ✅ `LIVE_MODE_RESTORED.md` - Technical documentation
4. ✅ `chord_ground_truth_trainer_wav2vec.py` - Full trainer with live mode

## The Complete Architecture (When Done)

```
Minimal Test (6 chords) → JSON with audio
                            ↓
                       Verify it works
                            ↓
                       Expand to 20+ chords
                            ↓
                       Train classifier (768D → Chord labels)
                            ↓
                       models/chord_model_wav2vec.pkl
                            ↓
                       Connect to Chandra_trainer
                            ↓
                       Better chord predictions
                            ↓
                       MusicHal plays musically!
```

## Ready to Test! 🎹

Run the minimal test whenever you're ready. It's just 6 chords and will verify the entire pipeline works!





