# Legacy Code

This directory contains deprecated code that has been superseded by newer implementations.

## Contents

### Chandra_trainer.py (DEPRECATED)
**Status:** Replaced by `scripts/train/train_modular.py`
**Size:** 2,413 lines
**Issue:** Monolithic script, hard to maintain
**Replacement:** Modular training pipeline in `musichal/training/pipeline/`

**Migration:**
```bash
# Old (don't use):
python Chandra_trainer.py audio.wav output.json

# New (use this):
python scripts/train/train_modular.py audio.wav output.json
```

### CCM3/ (DEPRECATED)
**Status:** Previous version of the system
**Issue:** Outdated architecture
**Replacement:** Current CCM4 system with modular architecture

## Why Keep Legacy Code?

- **Reference:** May contain useful algorithms or approaches
- **Historical:** Documents system evolution
- **Recovery:** Backup if new system has issues (unlikely)

## Do NOT Use

This code is not maintained and may not work with current dependencies.
Use the current system in `musichal/` and `scripts/` instead.
