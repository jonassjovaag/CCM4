# Root Note Frequencies Reference Guide

## How to Create Performance Arcs with Root Hints

### Quick Start: Manual JSON Creation

1. **Create a JSON file** in `performance_arcs/` directory
2. **Define phases** with `root_hint_frequency` (in Hz) and `harmonic_tension_target` (0.0-1.0)
3. **Run MusicHal** with: `python MusicHal_9000.py --performance-arc-path performance_arcs/your_arc.json`

### Root Note Frequencies (Hz)

Use these frequencies in your `root_hint_frequency` field:

#### Common Roots in C Major Scale:
```
C3  = 130.81 Hz   (low tonic)
D3  = 146.83 Hz   (supertonic)
E3  = 164.81 Hz   (mediant)
F3  = 174.61 Hz   (subdominant)
G3  = 196.00 Hz   (dominant)
A3  = 220.00 Hz   (submediant - relative minor root)
B3  = 246.94 Hz   (leading tone)

C4  = 261.63 Hz   (middle tonic) ← Most common
D4  = 293.66 Hz
E4  = 329.63 Hz   (mediant)
F4  = 349.23 Hz   (subdominant)
G4  = 392.00 Hz   (dominant)
A4  = 440.00 Hz   (submediant)
B4  = 493.88 Hz

C5  = 523.25 Hz   (high tonic)
```

### Example Progressions

#### Classic I-vi-iii-I Progression (15 min):
```json
{
  "phases": [
    {
      "start_time": 0.0,
      "end_time": 300.0,
      "root_hint_frequency": 261.63,  // C4 - Tonic
      "harmonic_tension_target": 0.2   // Low tension
    },
    {
      "start_time": 300.0,
      "end_time": 600.0,
      "root_hint_frequency": 220.0,    // A3 - Submediant (vi)
      "harmonic_tension_target": 0.5   // Medium tension
    },
    {
      "start_time": 600.0,
      "end_time": 900.0,
      "root_hint_frequency": 329.63,   // E4 - Mediant (iii)
      "harmonic_tension_target": 0.8   // High tension (climax)
    },
    {
      "start_time": 900.0,
      "end_time": 1200.0,
      "root_hint_frequency": 261.63,   // C4 - Tonic (return)
      "harmonic_tension_target": 0.1   // Resolution
    }
  ]
}
```

#### I-IV-V-I "Folk Progression" (10 min):
```json
{
  "phases": [
    {
      "start_time": 0.0,
      "end_time": 150.0,
      "root_hint_frequency": 261.63,   // C4 - Tonic (I)
      "harmonic_tension_target": 0.2
    },
    {
      "start_time": 150.0,
      "end_time": 300.0,
      "root_hint_frequency": 349.23,   // F4 - Subdominant (IV)
      "harmonic_tension_target": 0.4
    },
    {
      "start_time": 300.0,
      "end_time": 450.0,
      "root_hint_frequency": 392.00,   // G4 - Dominant (V)
      "harmonic_tension_target": 0.7
    },
    {
      "start_time": 450.0,
      "end_time": 600.0,
      "root_hint_frequency": 261.63,   // C4 - Tonic (I)
      "harmonic_tension_target": 0.1
    }
  ]
}
```

### Harmonic Tension Target Guidelines

- **0.0 - 0.3**: Very consonant, stable, restful (intro/outro)
- **0.3 - 0.5**: Balanced, forward motion
- **0.5 - 0.7**: Increased tension, building energy
- **0.7 - 1.0**: High tension, climax, dramatic moments

### Phase Timing Control

**Q: How do we control when the change from one root to another is made?**

**A: Time-based boundaries in the JSON file**

The system uses **hard phase boundaries** based on elapsed time:

```json
{
  "phases": [
    {
      "start_time": 0.0,      // Starts at 0 seconds
      "end_time": 180.0,      // Ends at 180 seconds (3 minutes)
      "root_hint_frequency": 261.63
    },
    {
      "start_time": 180.0,    // Starts exactly at 180 seconds
      "end_time": 360.0,      // Ends at 360 seconds (6 minutes total)
      "root_hint_frequency": 220.0
    }
  ]
}
```

**The transition happens INSTANTLY at the phase boundary** (e.g., at exactly 180.0 seconds).

### Optional: Smooth Transitions

If you want **gradual root changes** instead of instant jumps, you can:

1. **Use overlapping phases** with different weights:
```json
// Phase 1: C dominant (0-150s)
{
  "start_time": 0.0,
  "end_time": 150.0,
  "root_hint_frequency": 261.63
},
// Transition zone: C fading, A emerging (150-180s)
{
  "start_time": 150.0,
  "end_time": 180.0,
  "root_hint_frequency": 240.0,  // Midpoint between C (261.63) and A (220.0)
  "harmonic_tension_target": 0.4  // Transitional tension
},
// Phase 2: A dominant (180-300s)
{
  "start_time": 180.0,
  "end_time": 300.0,
  "root_hint_frequency": 220.0
}
```

2. **Use shorter phases** for more frequent, subtle changes:
```json
// Instead of one 300s phase, use three 100s phases
// with gradually shifting roots (C → B → A)
{
  "start_time": 0.0,
  "end_time": 100.0,
  "root_hint_frequency": 261.63  // C4
},
{
  "start_time": 100.0,
  "end_time": 200.0,
  "root_hint_frequency": 246.94  // B3 (between C and A)
},
{
  "start_time": 200.0,
  "end_time": 300.0,
  "root_hint_frequency": 220.0   // A3
}
```

### Usage

**Run MusicHal with your custom arc:**
```bash
python MusicHal_9000.py --performance-arc-path performance_arcs/simple_root_progression.json --duration 15
```

**Or let the system find the most recent arc automatically:**
```bash
python MusicHal_9000.py --duration 15
# Searches for *_performance_arc.json in ai_learning_data/
```

### Advanced: Extract Arc from Audio

You can also **analyze existing recordings** to extract natural harmonic progressions:

```bash
python performance_arc_analyzer.py
# Analyzes input_audio/Itzama.wav
# Creates ai_learning_data/itzama_performance_arc.json
```

Then manually add `root_hint_frequency` fields to each phase based on your musical intent.

### Important Notes

1. **Perceptual, not symbolic**: Root hints are frequencies (Hz), not chord names
2. **Soft guidance**: The system biases towards the root, doesn't force it
3. **768D still primary**: Wav2Vec gesture tokens drive the core decisions
4. **Groven method**: Harmonic analysis uses frequency ratio analyzer (NOT Brandtsegg)
5. **Brandtsegg is for rhythm**: Duration ratios, not harmony

### Example: 5-Minute Simple Performance

See `performance_arcs/simple_root_progression.json` for a complete example with:
- 5 phases over 15 minutes
- Progression: C → A → E → B → C (tonic → vi → iii → vii → tonic)
- Tension arc: 0.2 → 0.5 → 0.8 → 0.4 → 0.1
