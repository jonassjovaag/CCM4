# Environment Setup for MusicHal 9000

## Quick Start

```bash
# 1. Setup environment (one time)
bash setup_environment.sh

# 2. Activate environment (each session)
source CCM3/bin/activate

# 3. Train a model
python Chandra_trainer.py --file input_audio/Itzama.wav --max-events 5000

# 4. Run live performance
python MusicHal_9000.py
```

## Manual Setup

If the setup script doesn't work:

```bash
# Create virtual environment
python3.10 -m venv CCM3

# Activate it
source CCM3/bin/activate  # Linux/Mac
# or
CCM3\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

## Dependencies Summary

- **Total packages:** 123
- **Key dependencies:**
  - `torch` - Neural networks + Apple Silicon GPU support
  - `transformers` - Wav2Vec 2.0 for gesture tokens
  - `librosa` - Audio analysis and HPSS separation
  - `scikit-learn` - K-means clustering for vocabularies
  - `PyQt6` - GUI for visualization system
  - `mido` - MIDI input/output

## System Requirements

- **Python:** 3.10+
- **OS:** macOS (Apple Silicon optimized), Linux, Windows
- **RAM:** 8GB+ recommended for training
- **Storage:** 2GB for dependencies + models

## Troubleshooting

If you get import errors:
1. Make sure CCM3 environment is activated: `source CCM3/bin/activate`
2. Check Python version: `python --version` (should be 3.10+)
3. Reinstall requirements: `pip install -r requirements.txt --force-reinstall`

For Apple Silicon Macs, MPS (GPU) acceleration is automatically enabled for Wav2Vec processing.