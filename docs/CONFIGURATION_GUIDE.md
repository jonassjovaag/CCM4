# Configuration System Guide

## Overview

The centralized configuration system provides a flexible, maintainable way to manage all system parameters. No more hardcoded values scattered throughout the codebase!

## Features

- ✅ **YAML-based** - Human-readable configuration files
- ✅ **Profile system** - Pre-configured profiles for common use cases
- ✅ **Inheritance** - Profiles inherit from default config
- ✅ **Dot notation** - Easy nested access (`config.get('audio_oracle.distance_threshold')`)
- ✅ **Runtime overrides** - Override values via CLI or code
- ✅ **Validation** - Catch invalid configurations early
- ✅ **Type-safe** - Clear structure with defaults

---

## Quick Start

### Load Default Configuration

```python
from core.config_manager import get_config

# Get singleton instance with default config
config = get_config()

# Access values
threshold = config.get('audio_oracle.distance_threshold')
print(f"Distance threshold: {threshold}")
```

### Load a Profile

```python
from core.config_manager import load_config

# Load quick_test profile (inherits from default + overrides)
config = load_config(profile='quick_test')

max_events = config.get('audio_oracle.training.max_events')
print(f"Max events: {max_events}")  # 1000 (overridden)
```

### Override Values

```python
# Load with runtime overrides
config = load_config(
    profile='live_performance',
    overrides={
        'audio_oracle.distance_threshold': 0.2,
        'memory_buffer.max_duration_seconds': 240.0
    }
)
```

---

## Available Profiles

### 1. **Default** (no profile specified)
- Balanced settings for general use
- All features enabled
- Moderate performance

**Use when:** Standard training or testing

### 2. **quick_test**
- Fast iteration for development
- Minimal events (1000)
- Reduced logging
- No backups or metadata

**Use when:** Rapid prototyping, debugging

**Example:**
```bash
python Chandra_trainer.py --profile quick_test audio.wav
```

### 3. **full_training**
- Comprehensive training with all features
- Extended events (20,000)
- All analysis components enabled
- Full backups and metadata

**Use when:** Final training runs, production models

**Example:**
```bash
python Chandra_trainer.py --profile full_training curious_child.wav
```

### 4. **live_performance**
- Optimized for real-time performance
- Low latency settings
- Reduced logging overhead
- Hardware acceleration enabled

**Use when:** Live performances, real-time interaction

**Example:**
```bash
python main.py --profile live_performance
```

---

## Configuration Structure

```yaml
# config/default_config.yaml

system:
  name: "MusicHal 9000"
  version: "2.0"

audio_oracle:
  distance_threshold: 0.15
  distance_function: "euclidean"
  feature_dimensions: 15
  adaptive_threshold: true

  training:
    max_events: 15000
    hierarchical_enabled: true

memory_buffer:
  max_duration_seconds: 180.0
  feature_dimensions: 4

ai_agent:
  default_behavior: "imitate"
  autonomous_when_silent: true

midi:
  port: "IAC Driver Melody Channel"
  channel: 1

# ... and more
```

---

## Usage Patterns

### Pattern 1: Simple Access

```python
from core.config_manager import get_config

config = get_config()

# Get single value
threshold = config.get('audio_oracle.distance_threshold')

# Get with default
buffer_size = config.get('audio.buffer_size', 512)

# Get entire section
oracle_config = config.get_section('audio_oracle')
print(oracle_config['distance_threshold'])
```

### Pattern 2: Initialize Components with Config

**Before (hardcoded):**
```python
oracle = PolyphonicAudioOracleMPS(
    distance_threshold=0.15,
    distance_function='euclidean',
    feature_dimensions=15,
    adaptive_threshold=True
)
```

**After (config-driven):**
```python
from core.config_manager import get_config

config = get_config()
oracle_config = config.get_section('audio_oracle')

oracle = PolyphonicAudioOracleMPS(
    distance_threshold=oracle_config['distance_threshold'],
    distance_function=oracle_config['distance_function'],
    feature_dimensions=oracle_config['feature_dimensions'],
    adaptive_threshold=oracle_config['adaptive_threshold']
)
```

### Pattern 3: Runtime Configuration

```python
from core.config_manager import ConfigManager

config = ConfigManager()
config.load(profile='live_performance')

# Adjust for specific performance conditions
if venue_size == 'large':
    config.set('ai_agent.density.max_notes_per_second', 4.0)
elif venue_size == 'intimate':
    config.set('ai_agent.density.max_notes_per_second', 2.0)

# Use adjusted config
max_density = config.get('ai_agent.density.max_notes_per_second')
```

### Pattern 4: Command-Line Integration

```python
import argparse
from core.config_manager import load_config

parser = argparse.ArgumentParser()
parser.add_argument('--profile', default=None)
parser.add_argument('--threshold', type=float, default=None)
parser.add_argument('--max-events', type=int, default=None)
args = parser.parse_args()

# Build overrides from CLI args
overrides = {}
if args.threshold:
    overrides['audio_oracle.distance_threshold'] = args.threshold
if args.max_events:
    overrides['audio_oracle.training.max_events'] = args.max_events

# Load config with CLI overrides
config = load_config(profile=args.profile, overrides=overrides)
```

**Usage:**
```bash
python main.py --profile live_performance --threshold 0.2
python Chandra_trainer.py --profile quick_test --max-events 5000
```

---

## Creating Custom Profiles

### Method 1: Create New Profile File

Create `config/profiles/my_profile.yaml`:

```yaml
# My Custom Profile
# Only specify what differs from default

audio_oracle:
  distance_threshold: 0.18  # Custom threshold
  training:
    max_events: 12000  # Medium training size

logging:
  log_level: "DEBUG"  # Verbose logging

# All other values inherited from default
```

Load it:
```python
config = load_config(profile='my_profile')
```

### Method 2: Save Current Configuration

```python
from core.config_manager import get_config

config = get_config()

# Modify as needed
config.set('audio_oracle.distance_threshold', 0.18)
config.set('logging.log_level', 'DEBUG')

# Save as new profile
config.save('config/profiles/my_profile.yaml')
```

---

## Advanced Features

### Environment Variable Substitution (Future)

```yaml
midi:
  port: "${MIDI_PORT:IAC Driver Melody Channel}"
  # Uses $MIDI_PORT env var, or default if not set
```

### Validation

```python
config = get_config()

# Validate configuration
if not config.validate():
    print("Configuration is invalid!")
    sys.exit(1)
```

### List Available Profiles

```python
config = ConfigManager()
profiles = config.list_profiles()
print(f"Available profiles: {', '.join(profiles)}")
```

### Export Configuration

```python
# Get complete config as dictionary
config_dict = config.to_dict()

# Save for documentation
import json
with open('current_config.json', 'w') as f:
    json.dump(config_dict, f, indent=2)
```

---

## Migration Guide

### Migrating Existing Code

**Step 1:** Identify hardcoded parameters
```python
# Old code with hardcoded values
oracle = PolyphonicAudioOracleMPS(
    distance_threshold=0.15,  # Hardcoded!
    feature_dimensions=15      # Hardcoded!
)
```

**Step 2:** Load config at module/class init
```python
from core.config_manager import get_config

class MyClass:
    def __init__(self):
        self.config = get_config()
        self.oracle = self._create_oracle()

    def _create_oracle(self):
        oracle_config = self.config.get_section('audio_oracle')
        return PolyphonicAudioOracleMPS(
            distance_threshold=oracle_config['distance_threshold'],
            feature_dimensions=oracle_config['feature_dimensions']
        )
```

**Step 3:** Test with different profiles
```python
# Test with quick_test profile
config = load_config(profile='quick_test')
# Verify behavior changes as expected
```

### Gradual Migration Strategy

1. **Phase 1:** Add config system (✅ done)
2. **Phase 2:** Migrate main.py and Chandra_trainer.py
3. **Phase 3:** Migrate core modules (oracle, memory buffer, agent)
4. **Phase 4:** Migrate remaining modules
5. **Phase 5:** Remove all hardcoded values

---

## Configuration Reference

### All Available Sections

- `system` - System metadata
- `audio` - Audio input settings
- `audio_oracle` - AudioOracle configuration
  - `training` - Training-specific settings
- `memory_buffer` - Memory buffer settings
- `rhythm_oracle` - RhythmOracle settings
- `ai_agent` - AI agent behavior
  - `behavior_weights` - Behavior mode weights
  - `density` - Note density control
- `feature_mapper` - MIDI mapping settings
- `midi` - MIDI output configuration
- `performance_arc` - Performance structure
  - `phases` - Performance phases
- `logging` - Logging configuration
  - `performance_logging` - Performance-specific logs
- `persistence` - Data storage settings
  - `auto_save` - Auto-save configuration
  - `backup` - Backup settings
- `data_safety` - Data safety features (Phase 1)
  - `validation` - Validation settings
- `feature_extraction` - Feature extraction settings
  - `wav2vec`, `frequency_analysis`, etc.
- `hierarchical_analysis` - Multi-timescale analysis
- `training` - Training pipeline settings
  - `components` - Which analysis components to enable
- `visualization` - Visualization settings
- `performance` - Performance optimization
- `debug` - Debug/development settings

See `config/default_config.yaml` for complete reference with comments.

---

## Troubleshooting

### Configuration Not Found

```
ConfigurationError: Default config not found
```

**Solution:** Ensure `config/default_config.yaml` exists in project root.

### Profile Not Found

```
ConfigurationError: Profile not found: my_profile.yaml
```

**Solution:** Check `config/profiles/` directory. Profile files must be `.yaml`.

### Invalid Configuration

```python
if not config.validate():
    # Configuration is invalid
```

**Solution:** Check validation errors in logs. Common issues:
- Invalid distance_threshold (must be > 0)
- Missing required sections

### Values Not Overriding

**Issue:** Profile values not overriding defaults

**Solution:** Ensure profile inheritance is correct. Use dot notation:
```yaml
audio_oracle:
  distance_threshold: 0.2  # Correct

audio_oracle.distance_threshold: 0.2  # Wrong!
```

---

## Best Practices

1. **Use profiles for different scenarios**
   - Don't modify default config
   - Create profiles for specific use cases

2. **Document custom values**
   - Add comments explaining why values differ from default

3. **Version control your profiles**
   - Commit custom profiles to git
   - Document which profile to use for which purpose

4. **Validate before running**
   - Always call `config.validate()` in production

5. **Use dot notation consistently**
   - `config.get('audio_oracle.distance_threshold')`
   - Not `config._config['audio_oracle']['distance_threshold']`

6. **Keep defaults sensible**
   - Default config should work for most cases
   - Profiles for specialized needs only

---

## Future Enhancements

Planned for future phases:

- [ ] Environment variable substitution
- [ ] Config schema validation (JSON Schema)
- [ ] Config diff tool (compare profiles)
- [ ] Config migration tool (v1 → v2)
- [ ] Web UI for config editing
- [ ] Real-time config reload (hot reload)

---

## Examples

### Example 1: Training Script

```python
#!/usr/bin/env python3
from core.config_manager import load_config
from Chandra_trainer import train_model

def main():
    # Load full training profile
    config = load_config(profile='full_training')

    # Train with configured parameters
    train_model(
        audio_file='curious_child.wav',
        max_events=config.get('audio_oracle.training.max_events'),
        output_dir=config.get('training.output_dir')
    )

if __name__ == '__main__':
    main()
```

### Example 2: Live Performance

```python
#!/usr/bin/env python3
from core.config_manager import load_config
from main import DriftEngineAI

def main():
    # Load live performance profile
    config = load_config(profile='live_performance')

    # Create engine with configured parameters
    engine = DriftEngineAI(
        midi_port=config.get('midi.port'),
        performance_duration=config.get('performance_arc.default_duration_minutes')
    )

    engine.run()

if __name__ == '__main__':
    main()
```

---

**Generated:** 2025-11-13
**Phase:** 2.1 - Centralized Configuration
**Status:** Complete and tested
