# Musical AI Training Pipeline Guide

## Overview

This document outlines the complete training pipeline for the Musical AI system, from raw audio processing to live performance agents.

## Pipeline Architecture

```
Audio Files → Chandra_trainer.py → AudioOracle Model
     ↓
Live Audio → Interactive Chord Trainer → ML Chord Model
     ↓
Live Performance → MusicHal_9000 / MusicHal_bass → MPE MIDI Output
```

## Stage 1: Audio Oracle Training (Chandra_trainer.py)

### Purpose
Trains the core AudioOracle model on polyphonic audio to learn musical patterns, melodic salience, and harmonic relationships.

### Process
1. **Input**: Audio files (WAV, MP3, etc.)
2. **Processing**: 
   - Polyphonic audio analysis with melodic salience detection
   - Hierarchical sampling for pattern recognition
   - Feature extraction (F0, RMS, spectral features, etc.)
3. **Output**: `ai_learning_data/audio_oracle_model.json`

### Usage
```bash
python Chandra_trainer.py --input-dir /path/to/audio/files
```

### Key Features
- **Melodic Salience Detection**: Identifies the melodic voice in polyphonic audio
- **Hierarchical Sampling**: Learns patterns at multiple time scales
- **Feature Preservation**: Maintains all audio features during training
- **Auto-save**: Timestamped model files in `JSON/` directory

## Stage 2: Chord Detection Training (test_interactive_chord_trainer.py)

### Purpose
Trains an ML model for accurate chord detection using supervised learning with user-labeled data.

### Process
1. **Input**: Live audio from microphone + user chord labels
2. **Processing**:
   - Real-time harmonic analysis
   - Feature extraction (chroma, confidence, stability, RMS, F0)
   - User labeling of chords
3. **Output**: `models/chord_model.pkl`, `models/chord_scaler.pkl`

### Usage
```bash
python test_interactive_chord_trainer.py
```

### Key Features
- **Continuous Learning**: Appends to existing model, doesn't start fresh
- **Full Jazz Vocabulary**: Supports major, minor, 7th, 9th, 11th, 13th, dim, aug, sus chords
- **Real-time Training**: Learn as you play
- **Persistent Storage**: Model improves over time

### Chord Vocabulary
- **Basic**: C, Cm, D, Dm, E, Em, F, Fm, G, Gm, A, Am, B, Bm
- **Chromatic**: C#, D#, F#, G#, A# (and flats)
- **7th Chords**: C7, Cm7, Cmaj7, Cdom7, etc.
- **Extensions**: C9, Cm9, Cmaj9, C11, C13, etc.
- **Altered**: C7alt, C7b9, C7#9, C7b5, C7#5
- **Other**: Cdim, Caug, Csus2, Csus4, Cm7b5

## Stage 3: Live Performance Agents

### MusicHal_9000.py
Full-featured AI musical agent with melodic and bass responses.

**Features**:
- Uses AudioOracle for pattern-based generation
- Harmonic and rhythmic analysis
- Cross-modal correlation
- MPE MIDI output to multiple channels
- Autonomous generation with activity tracking

**MIDI Outputs**:
- `IAC Driver Melody Channel` - Melodic responses
- `IAC Driver Bass` - Bass responses  
- `IAC Driver Rhythm Channel` - Rhythmic elements

### MusicHal_bass.py
Bass-only version of MusicHal_9000 for focused bass interaction.

**Features**:
- Identical AI intelligence to MusicHal_9000
- Only outputs to `IAC Driver Bass`
- Same training, memory, and decision-making
- Perfect for bass-focused musical interaction

## Integration with ML Chord Detection

### test_ml_chord_bass_response.py
Demonstrates ML chord detection with bass responses.

**Features**:
- Uses trained ML model for chord detection
- Real-time bass responses via MPE MIDI
- Confidence tracking and stability analysis
- Integration potential with main agents

## Recommended Workflow

### Initial Setup
1. **Train AudioOracle**: Run `Chandra_trainer.py` on your audio library
2. **Train Chord Detection**: Use `test_interactive_chord_trainer.py` to build chord vocabulary
3. **Test Integration**: Verify with `test_ml_chord_bass_response.py`

### Daily Usage
1. **Start MusicHal_9000**: For full melodic + bass interaction
2. **Or Start MusicHal_bass**: For bass-only interaction
3. **Continue Training**: Periodically use chord trainer to improve accuracy

### Continuous Improvement
1. **Add Audio Files**: Expand AudioOracle training data
2. **Label More Chords**: Improve ML chord detection
3. **Adjust Parameters**: Fine-tune response frequencies and timing

## File Structure

```
CCM3/
├── Chandra_trainer.py              # Stage 1: AudioOracle training
├── test_interactive_chord_trainer.py  # Stage 2: Chord ML training
├── MusicHal_9000.py               # Stage 3: Full performance agent
├── MusicHal_bass.py               # Stage 3: Bass-only agent
├── test_ml_chord_bass_response.py # Integration test
├── ai_learning_data/              # AudioOracle models
│   └── audio_oracle_model.json
├── models/                        # ML chord models
│   ├── chord_model.pkl
│   ├── chord_scaler.pkl
│   └── chord_metadata.json
└── JSON/                          # Timestamped training outputs
    └── *_model.json
```

## Key Benefits

1. **Modular Design**: Each stage can be improved independently
2. **Continuous Learning**: Models improve over time
3. **Real-time Performance**: Low-latency live interaction
4. **Musical Intelligence**: Combines pattern learning with harmonic analysis
5. **Flexible Output**: Multiple agents for different use cases

## Future Enhancements

1. **Sheet Music Integration**: Add partitura for score-based training
2. **Multi-instrument Support**: Extend beyond piano
3. **Style Transfer**: Learn different musical styles
4. **Collaborative Learning**: Multiple users training the same model
5. **Real-time Model Updates**: Continuous learning during performance

## Troubleshooting

### Common Issues
1. **MIDI Not Working**: Check IAC Driver setup in Audio MIDI Setup
2. **Low Chord Accuracy**: Train more samples with chord trainer
3. **Repetitive Responses**: Adjust timing parameters in agents
4. **Model Loading Errors**: Check file paths and permissions

### Performance Tips
1. **Use SSD Storage**: Faster model loading/saving
2. **Adequate RAM**: 8GB+ recommended for large models
3. **Low-latency Audio**: Use dedicated audio interface
4. **Regular Training**: Keep models updated with new data


