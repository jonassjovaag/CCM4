# Troubleshooting: Ratio-Based Chord Validator

## Issue: "Clustered into 0 stable frequencies"

### Problem
The validator was detecting frequency samples (e.g., 29 samples) but clustering them into 0 stable frequencies, causing the analysis to fail.

### Root Causes
1. **Too strict clustering tolerance** (3% → now 5%)
2. **Too strict minimum cluster size** (5 samples → now 3 samples)
3. **Too strict RMS threshold** (-50 dB → now -60 dB)
4. **Missing error handling** for failed analyses

### Fixes Applied

#### 1. More Lenient Clustering
```python
# Before:
tolerance = 0.03  # 3%
if len(cluster) >= 5:  # Required 5 detections

# After:
tolerance = 0.05  # 5%
if len(cluster) >= 3:  # Required 3 detections
```

#### 2. More Lenient Pitch Detection
```python
# Before:
if event.rms_db > -50 and event.f0 > 0:

# After:
if event.rms_db > -60 and 50 < event.f0 < 2000:
```

#### 3. Better Diagnostics
Added detailed output:
- Raw frequency range
- Frequency distribution histogram (10 Hz bins)
- Cluster information (samples per cluster)
- Rejected clusters (too few samples)

#### 4. Fixed KeyError
Added proper error handling for failed analyses in `run_test()`:
```python
if 'ratio_analysis' in result and result['validation_passed']:
    # Use ratio_analysis
else:
    # Handle failure gracefully
```

## How to Test

### 1. Basic Test (No Audio Required)
```bash
python listener/ratio_analyzer.py
```
This runs the demo with synthetic frequencies - should work perfectly.

### 2. Full Validation Test (Requires Audio Setup)
```bash
python ratio_based_chord_validator.py --input-device 0
```

### Expected Output (Success)
```
🎹 Validating: C major
   Sending MIDI: [60, 64, 67] (['C4', 'E4', 'G4'])
   Expected frequencies: ['261.63 Hz', '329.63 Hz', '392.00 Hz']
   Detected 29 frequency samples
   Raw frequency range: 250.00 - 400.00 Hz
   Frequency distribution (bins of 10 Hz):
      ~260 Hz: 10 samples
      ~330 Hz: 10 samples
      ~390 Hz: 9 samples
      Cluster 1: 10 samples, mean = 261.50 Hz
      Cluster 2: 10 samples, mean = 329.80 Hz
      Cluster 3: 9 samples, mean = 391.20 Hz
   Clustered into 3 stable frequencies:
      261.50 Hz (C4)
      329.80 Hz (E4)
      391.20 Hz (G4)

   📊 Ratio Analysis:
      Fundamental: 261.50 Hz
      Ratios: ['1.000', '1.261', '1.496']
      Simplified: [(1, 1), (63, 50), (3, 2)]
      Chord type: major (confidence: 90.4%)
      Consonance: 0.705
   ✅ VALIDATED!
```

## Common Issues & Solutions

### Issue 1: Still getting 0 stable frequencies

**Diagnosis**: The detected frequencies might be too scattered.

**Solutions**:
1. Check audio routing: Make sure MIDI → Ableton → Audio Input is working
2. Increase chord duration: `--duration 3.0` (give more time to collect samples)
3. Lower RMS threshold further (edit code: change -60 to -70)
4. Check input device: Try different `--input-device` values

**Debug steps**:
```python
# Add this to _on_pitch_event to see what's being detected:
print(f"Detected: f0={event.f0:.1f} Hz, rms={event.rms_db:.1f} dB")
```

### Issue 2: Wrong frequencies detected

**Diagnosis**: YIN might be detecting harmonics instead of fundamentals.

**Solutions**:
1. Adjust YIN threshold in DriftListener
2. Use lower register chords (octave down)
3. Check for background noise

### Issue 3: Correct frequencies but wrong chord type

**Diagnosis**: Ratio matching might need tuning.

**Solutions**:
1. Check the ideal ratios in `listener/ratio_analyzer.py`
2. Adjust tolerance: `FrequencyRatioAnalyzer(tolerance=0.05)`
3. Review interval analysis to see what's being detected

### Issue 4: No MIDI output

**Diagnosis**: MIDI port not configured.

**Solutions**:
1. Create IAC Driver port: "IAC Driver Chord Trainer Output"
2. Route in Ableton: Set input to this IAC port
3. Check available ports:
```python
import mido
print(mido.get_output_names())
```

## Testing Without Full Audio Setup

If you don't have the full audio routing set up, you can test the core functionality:

### Test 1: Ratio Analyzer (No Audio)
```bash
python listener/ratio_analyzer.py
```

### Test 2: Modified Validator (Synthetic Frequencies)
Create a test script:
```python
from listener.ratio_analyzer import FrequencyRatioAnalyzer

analyzer = FrequencyRatioAnalyzer()

# Test C major
c_major = [261.63, 329.63, 392.00]
result = analyzer.analyze_frequencies(c_major)

print(f"Type: {result.chord_match['type']}")
print(f"Confidence: {result.chord_match['confidence']:.1%}")
print(f"Consonance: {result.consonance_score:.3f}")
```

## Understanding the Output

### Frequency Distribution
Shows which frequency ranges have the most detections:
```
~260 Hz: 10 samples  ← C4 region
~330 Hz: 10 samples  ← E4 region
~390 Hz: 9 samples   ← G4 region
```

### Cluster Information
```
Cluster 1: 10 samples, mean = 261.50 Hz  ← Accepted (≥3 samples)
Cluster 2: 2 samples (rejected)          ← Rejected (<3 samples)
```

### Ratio Analysis
```
Ratios: [1.000, 1.261, 1.496]
       ↓      ↓      ↓
      4  :   5  :   6     ← Major triad signature!
```

## Performance Tips

### 1. Optimize Chord Duration
- Too short: Not enough samples
- Too long: Unnecessary wait
- Recommended: 2.0 - 3.0 seconds

### 2. Optimize Clustering
Current settings work for most cases, but you can tune:
```python
tolerance = 0.05        # How close frequencies must be (5%)
min_cluster_size = 3    # Minimum samples to accept cluster
```

### 3. Optimize RMS Threshold
```python
if event.rms_db > -60:  # Current threshold
# Lower = more lenient (catches quiet notes)
# Higher = more strict (only loud notes)
```

## Next Steps

1. **Test with real audio** to see the new diagnostic output
2. **Adjust parameters** based on actual results
3. **Integrate with autonomous trainer** once validation is stable
4. **Create visualizations** of frequency distributions and ratios

## Need More Help?

Check these files:
- `listener/ratio_analyzer.py` - Core ratio analysis logic
- `ratio_based_chord_validator.py` - Validation system
- `RATIO_BASED_ANALYSIS_SUMMARY.md` - Complete documentation
- `RATIO_BASED_CHORD_ANALYSIS_PLAN.md` - Implementation plan

