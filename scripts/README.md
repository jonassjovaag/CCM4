# Scripts Directory

Executable scripts organized by purpose.

## Directory Structure

- **train/** - Training scripts (train models from audio)
- **analysis/** - Analysis and diagnostic scripts
- **utils/** - Utility scripts (conversion, checking, fixing)
- **demo/** - Demonstration scripts
- **performance/** - Live performance and daemon scripts
- **testing/** - Test and validation scripts

## Entry Points

**Main training entry point:**
```bash
python scripts/train/train_modular.py audio_file.wav output.json
```

**Live performance:**
```bash
python scripts/performance/live_performance.py
```

## Legacy Scripts

Some scripts may be outdated or superseded by newer implementations.
Check the script's docstring for current status.
