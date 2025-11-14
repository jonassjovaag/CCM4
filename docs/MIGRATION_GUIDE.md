# Migration Guide: Using Enhanced Data Safety Features

## Overview

Phase 1 of the refactoring has introduced robust data safety infrastructure. This guide explains how to migrate existing code to use the new features.

## What's New

### 1. Atomic File Writer
- Prevents data loss from partial writes
- Atomic rename operations
- Automatic retry on transient failures
- Checksum verification

### 2. Backup Manager
- Versioned backups with timestamps
- Configurable retention policies
- Easy restore functionality
- Automatic cleanup of old backups

### 3. Data Validator
- JSON schema validation
- Data quality checks
- Default value detection
- Detailed error reporting

### 4. Enhanced Save/Load
- Combines all safety features
- Drop-in replacements for existing methods
- Automatic validation and backups
- Fallback to backup on load failure

---

## Migration Strategies

### Strategy 1: Quick Win - Use Enhanced Functions Directly

**Before:**
```python
import json

def save_to_file(self, filepath: str) -> bool:
    try:
        with open(filepath, 'w') as f:
            json.dump(self.data, f)
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False
```

**After:**
```python
from core.data_safety import enhanced_save_json

def save_to_file(self, filepath: str) -> bool:
    return enhanced_save_json(
        self.data,
        filepath,
        description="Saving oracle state"
    )
```

**Benefits:**
- ✅ Atomic writes
- ✅ Automatic backups
- ✅ Retry logic
- ✅ Proper error logging

---

### Strategy 2: Use AtomicFileWriter for Custom Logic

If you need custom serialization logic but want atomic writes:

**Before:**
```python
def save_to_file(self, filepath: str) -> bool:
    temp_file = filepath + ".tmp"
    try:
        with open(temp_file, 'w') as f:
            json.dump(self.data, f, cls=NumpyEncoder)
        shutil.move(temp_file, filepath)
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False
```

**After:**
```python
from core.data_safety import AtomicFileWriter

def save_to_file(self, filepath: str) -> bool:
    writer = AtomicFileWriter()
    return writer.write_json(
        self.data,
        filepath,
        encoder_cls=NumpyEncoder,
        create_backup=True
    )
```

**Benefits:**
- ✅ Handles temp file cleanup
- ✅ Works on Windows and Unix
- ✅ Verifies write succeeded
- ✅ Creates backups automatically

---

### Strategy 3: Add Validation to Existing Code

Validate data before saving to catch corruption early:

```python
from core.data_safety import DataValidator, enhanced_save_json

validator = DataValidator()

def save_to_file(self, filepath: str) -> bool:
    # Validate before saving
    errors = validator.validate(
        self.get_save_data(),
        "audio_oracle_schema"
    )

    if errors:
        logger.error(f"Validation failed: {errors}")
        return False

    # Save with safety features
    return enhanced_save_json(
        self.get_save_data(),
        filepath,
        encoder_cls=NumpyEncoder
    )
```

---

### Strategy 4: Add Data Quality Checks

Detect issues like default values or missing features:

```python
from core.data_safety import DataValidator

validator = DataValidator()

def save_to_file(self, filepath: str) -> bool:
    data = self.get_save_data()

    # Check data quality
    warnings = validator.check_data_quality(data, "audio_oracle")
    for warning in warnings:
        logger.warning(f"Data quality issue: {warning}")

    # Save anyway (warnings don't prevent save)
    return enhanced_save_json(data, filepath)
```

---

### Strategy 5: Use EnhancedSaveLoad Class

For classes that need consistent save/load behavior:

```python
from core.data_safety import EnhancedSaveLoad

class MyOracle:
    def __init__(self):
        self.save_load_handler = EnhancedSaveLoad(
            validate=True,
            create_backups=True,
            max_retries=3
        )

    def save_to_file(self, filepath: str) -> bool:
        return self.save_load_handler.save_json(
            self.get_save_data(),
            filepath,
            encoder_cls=NumpyEncoder,
            schema_name="audio_oracle_schema",
            description="Oracle checkpoint"
        )

    def load_from_file(self, filepath: str) -> bool:
        data = self.save_load_handler.load_json(
            filepath,
            schema_name="audio_oracle_schema",
            fallback_to_backup=True
        )

        if data is None:
            return False

        self.restore_from_data(data)
        return True
```

---

## File-by-File Migration Priority

### High Priority (Critical Data Paths)

1. **`memory/polyphonic_audio_oracle_mps.py`**
   - Saves trained AudioOracle models
   - Recommendation: Strategy 5 (full EnhancedSaveLoad)

2. **`memory/audio_oracle.py`**
   - Core AudioOracle implementation
   - Recommendation: Strategy 5 (full EnhancedSaveLoad)

3. **`memory/memory_buffer.py`**
   - Real-time memory buffer
   - Recommendation: Strategy 2 (AtomicFileWriter with custom logic)

### Medium Priority (Important but Less Frequent)

4. **`rhythmic_engine/memory/rhythm_oracle.py`**
   - Rhythmic pattern storage
   - Recommendation: Strategy 3 (add validation)

5. **`Chandra_trainer.py`**
   - Training pipeline output
   - Recommendation: Strategy 3 (add validation)

### Low Priority (Can Wait)

6. **Configuration files, logs, temporary data**
   - Recommendation: Strategy 1 (quick win) when convenient

---

## Testing After Migration

After migrating a save/load method, test:

### 1. Normal Operation
```python
# Save
success = oracle.save_to_file("test.json")
assert success, "Save failed"

# Load
success = oracle.load_from_file("test.json")
assert success, "Load failed"
```

### 2. Backup Creation
```python
from pathlib import Path

# Save twice (second should create backup)
oracle.save_to_file("test.json")
oracle.save_to_file("test.json")

# Check backup exists
backups = list(Path("test.json").parent.glob("*.bak"))
assert len(backups) > 0, "No backup created"
```

### 3. Recovery from Corruption
```python
# Save good data
oracle.save_to_file("test.json")

# Corrupt file
with open("test.json", 'w') as f:
    f.write("{invalid json")

# Load should recover from backup
success = oracle.load_from_file("test.json")
assert success, "Failed to recover from backup"
```

### 4. Validation Errors
```python
# Try to save invalid data
invalid_data = {"training_successful": "yes"}  # Should be boolean

validator = DataValidator()
errors = validator.validate(invalid_data, "training_results_schema")
assert len(errors) > 0, "Validation should fail"
```

---

## Rollback Strategy

If migration causes issues, easy to rollback:

### Option 1: Revert Individual File
```bash
git checkout HEAD -- memory/audio_oracle.py
```

### Option 2: Disable New Features Temporarily
```python
from core.data_safety import EnhancedSaveLoad

# Disable validation and backups
handler = EnhancedSaveLoad(
    validate=False,        # Disable validation
    create_backups=False,  # Disable backups
    max_retries=1          # Single attempt only
)
```

### Option 3: Use Backups
```python
from core.data_safety import restore_backup

# Restore from pre-refactoring backup
restore_backup("pre_refactor_20251113_124720", "JSON")
```

---

## Common Issues and Solutions

### Issue 1: "Schema not found"
**Solution:** Check schema file exists in `schemas/` directory
```bash
ls schemas/*.json
```

### Issue 2: "Validation failed" for existing files
**Solution:** Validation is strict. Either:
1. Fix the data to match schema
2. Update schema to match actual data format
3. Disable validation temporarily

### Issue 3: Too many backups filling disk
**Solution:** Configure retention policy
```python
from core.data_safety import BackupManager

manager = BackupManager(
    max_backups=5,      # Keep only 5 most recent
    max_age_days=7      # Delete backups older than 7 days
)
```

### Issue 4: Performance regression
**Solution:** Profile and optimize
```python
import time

start = time.time()
oracle.save_to_file("test.json")
duration = time.time() - start

print(f"Save took {duration:.2f}s")
```

If too slow, consider:
- Disabling checksum verification
- Reducing retry attempts
- Disabling backups for non-critical files

---

## Examples

### Example 1: Migrate MemoryBuffer

**Before (memory/memory_buffer.py):**
```python
def save_to_file(self, filepath: str) -> bool:
    try:
        save_data = {'moments': [], ...}
        with open(filepath, 'w') as f:
            json.dump(save_data, f, cls=NumpyEncoder)
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False
```

**After:**
```python
from core.data_safety import enhanced_save_json

def save_to_file(self, filepath: str) -> bool:
    save_data = {
        'moments': [self._moment_to_dict(m) for m in self.buffer],
        'feature_stats': self.feature_stats,
        'max_duration_seconds': self.max_duration_seconds,
        'feature_dimensions': self.feature_dimensions,
        'save_timestamp': time.time()
    }

    return enhanced_save_json(
        save_data,
        filepath,
        encoder_cls=NumpyEncoder,
        description="Memory buffer snapshot"
    )
```

### Example 2: Migrate AudioOracle with Validation

```python
from core.data_safety import EnhancedSaveLoad, DataValidator

class PolyphonicAudioOracle:
    def __init__(self):
        self.saver = EnhancedSaveLoad(validate=True, create_backups=True)
        self.validator = DataValidator()

    def save_to_file(self, filepath: str) -> bool:
        data = self.get_serializable_state()

        # Quality check before saving
        warnings = self.validator.check_data_quality(data, "audio_oracle")
        if warnings:
            logger.warning(f"Data quality issues detected: {warnings}")

        # Save with full safety features
        return self.saver.save_json(
            data,
            filepath,
            encoder_cls=NumpyEncoder,
            schema_name="audio_oracle_schema",
            description=f"AudioOracle ({self.size} states)"
        )
```

---

## Next Steps

1. ✅ Phase 1.3 complete - Enhanced save/load infrastructure ready
2. 🔄 Phase 1.4 - Add training metadata to JSON files
3. 🔄 Phase 2 - Code organization and modularization
4. 🔄 Phase 3 - Performance optimization

## Questions?

Check the following for more info:
- `core/data_safety/README.md` - Detailed API documentation
- `tests/test_data_safety.py` - Working examples
- `tests/test_data_validator.py` - Validation examples

---

**Generated:** 2025-11-13
**Refactoring Phase:** 1.3
**Status:** Ready for migration
