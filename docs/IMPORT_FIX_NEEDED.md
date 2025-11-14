# Import Path Fixes Required

## Issue

After moving files to `scripts/` directories, the import paths need updating to work with the new `musichal/` package structure.

## Fixed ✅

### Training Scripts
✅ **`scripts/train/train_modular.py`** - Now works!

### Performance Scripts
✅ **`scripts/performance/MusicHal_9000.py`** - Now works!
✅ **`scripts/performance/MusicHal_bass.py`** - Now works!
✅ **`scripts/performance/main.py`** - Now works!
✅ **`scripts/performance/performance_timeline_manager.py`** - Fixed (dependency)

## Still Need Fixing

The following scripts in `scripts/train/` may have old import paths:

### Other Training Scripts
- `scripts/train/train_hybrid.py`
- `scripts/train/train_hybrid_enhanced.py`
- `scripts/train/*` (other training files)

## How to Run Fixed Scripts

### ✅ Working: train_modular.py

```bash
# From CCM4 directory
cd "K:\Scripts and web projects with Claude\CCM4"

# Run training
python scripts/train/train_modular.py audio.wav output.json

# With profile
python scripts/train/train_modular.py audio.wav output.json --profile quick_test

# Get help
python scripts/train/train_modular.py --help
```

### ✅ Working: Performance Scripts

```bash
# From CCM4 directory
cd "K:\Scripts and web projects with Claude\CCM4"

# Main performance script (full features)
python scripts/performance/MusicHal_9000.py --help

# Bass-only version
python scripts/performance/MusicHal_bass.py --help

# Simplified main
python scripts/performance/main.py --help
```

## Workaround for Other Scripts

Until all scripts are updated, you can use the old files (they're still there for backward compatibility):

### Option 1: Use Legacy Trainer
```bash
# Old monolithic trainer still works
python legacy/Chandra_trainer.py --file audio.wav --output output.json
```

### Option 2: Use Original Files (not moved)
The `core/` and `training_pipeline/` directories are still intact:
```bash
# Original imports still work from project root
cd "K:\Scripts and web projects with Claude\CCM4"
python -c "from core.config_manager import ConfigManager; print('Works!')"
```

## Standard Import Pattern

All scripts in `scripts/` should use this pattern:

```python
#!/usr/bin/env python3
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Then import from musichal package
from musichal.core import ConfigManager
from musichal.training import TrainingOrchestrator
```

## Summary

**For now, use:**
- ✅ `scripts/train/train_modular.py` - Fixed and working
- ✅ `legacy/Chandra_trainer.py` - Still works (old monolithic trainer)

**Performance scripts need fixing before they'll work from new locations.**
