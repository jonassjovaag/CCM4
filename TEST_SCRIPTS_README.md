# Chord Detection Test Scripts

This collection of test scripts helps isolate and analyze the harmonic detection and bass response capabilities of the MusicHal_9000 system.

## Test Scripts

### 1. `test_simple_chord_detection.py` (Recommended to start with)
**Purpose**: Test basic harmonic detection accuracy
**Focus**: Detects chord changes in the progression C - F - Gm - Bb - Cm - C
**Output**: Reports on chord detection accuracy and confidence

**Usage**:
```bash
python test_simple_chord_detection.py
```

**What it does**:
- Listens to your piano input
- Detects chord changes using the harmonic context system
- Compares detected chords to expected progression
- Generates a detailed report in `logs/simple_chord_test_YYYYMMDD_HHMMSS.json`

### 2. `test_bass_response.py`
**Purpose**: Test bass response to chord changes
**Focus**: Generates bass responses when chord changes are detected
**Output**: MIDI bass output + response analysis

**Usage**:
```bash
python test_bass_response.py
```

**What it does**:
- Detects chord changes (same as simple test)
- Generates bass responses using the phrase generator
- Sends MIDI bass notes to "IAC Driver Bass"
- Records all bass responses with timing and context

### 3. `test_chord_detection.py` (Full system test)
**Purpose**: Comprehensive test with full AudioOracle integration
**Focus**: Complete system test with learning and response
**Output**: Full system analysis

**Usage**:
```bash
python test_chord_detection.py
```

## Expected Chord Progression

Play this progression on your piano (you can play notes individually, not necessarily as full chords):

1. **C major** (C-E-G) - Root: C
2. **F major** (F-A-C) - Root: F  
3. **G minor** (G-Bb-D) - Root: G
4. **Bb major** (Bb-D-F) - Root: Bb
5. **C minor** (C-Eb-G) - Root: C
6. **C major** (C-E-G) - Root: C (return)

## Setup Requirements

1. **Audio Input**: Piano connected to your computer
2. **MIDI Output**: IAC Driver Bass configured in Ableton Live
3. **Python Environment**: All dependencies installed

## Test Duration

Each test runs for 60 seconds by default. You can stop early with Ctrl+C.

## Output Files

All tests generate detailed JSON reports in the `logs/` directory:
- `simple_chord_test_YYYYMMDD_HHMMSS.json`
- `bass_response_test_YYYYMMDD_HHMMSS.json` 
- `chord_detection_test_YYYYMMDD_HHMMSS.json`

## Analysis

The reports include:
- Detected chord changes with timestamps
- Confidence scores for each detection
- Expected vs actual chord comparisons
- Bass response analysis (for bass tests)
- Overall accuracy metrics

## Troubleshooting

- **No chord detection**: Check audio input levels and harmonic context settings
- **No bass output**: Verify IAC Driver Bass is configured in Ableton Live
- **Import errors**: Ensure all dependencies are installed and paths are correct


