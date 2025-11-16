# MusicHal 9000 - AI Musical Co-Improviser

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-57%2F57%20passing-brightgreen.svg)](tests/)
[![Code Style](https://img.shields.io/badge/code%20style-professional-blue.svg)](musichal/)

A real-time AI musical partner that learns from audio and responds with intelligent MIDI improvisation. Features **Factor Oracle learning**, **Wav2Vec neural encoding**, **hierarchical pattern recognition**, and **type-safe architecture**.

---

## 🎯 Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/musichal.git
cd musichal

# Install package in development mode
pip install -e .

# Or install dependencies only
pip install -r requirements.txt
```

### Train Your First Model
```bash
# Using the modular training pipeline (recommended)
python scripts/train/train_modular.py input_audio/your_audio.wav output/model.json

# With configuration profile
python scripts/train/train_modular.py audio.wav output.json --profile quick_test
```

### Live Performance
```bash
# Run live performance system
python scripts/performance/MusicHal_9000.py

# Or traditional entry point
python scripts/performance/main.py
```

---

## ✨ Key Features

### 🎵 **Musical Intelligence**
- **Factor Oracle Algorithm**: Learns musical patterns and structures
- **MERT-v1-95M Encoding**: Music-optimized transformer for deep audio understanding ⭐ **NEW**
- **CLAP Style Detection**: Automatic behavioral mode selection via audio-text alignment ⭐ **NEW**
- **Wav2Vec 2.0 Encoding**: 768D neural audio representations (legacy support)
- **Hierarchical Analysis**: Multi-timescale musical understanding
- **Gesture Tokenization**: 64-token discrete musical vocabulary

### 🏗️ **Professional Architecture**
- **Type-Safe Models**: Pydantic validation throughout
- **Modular Pipeline**: 5-stage training architecture
- **Data Safety**: Atomic writes, backups, checksums
- **Configuration Profiles**: YAML-based, environment-aware

### 🔬 **Advanced Features**
- **Dual Perception**: Wav2Vec + traditional features
- **Rhythmic Oracle**: Tempo, syncopation, beat detection
- **Harmonic Analysis**: Chord detection, key estimation
- **Performance Arcs**: Dynamic tension and release

---

## 📁 Project Structure

```
musichal/                    # Main package
├── core/                    # Core infrastructure
│   ├── data_safety/        # Atomic writes, backups, validation
│   ├── models/             # Pydantic data models
│   ├── config_manager.py   # YAML configuration
│   └── metadata_manager.py # Reproducibility tracking
└── training/               # Training system
    └── pipeline/           # Modular 5-stage pipeline
        ├── stages/
        └── orchestrators/

scripts/                     # Executable scripts
├── train/                  # Training scripts (15 files)
│   ├── train_modular.py   # Main entry point ⭐
│   └── ...
├── performance/            # Live performance (5 files)
│   ├── MusicHal_9000.py   # Main performance entry ⭐
│   └── main.py
├── analysis/               # Analysis tools (11 files)
├── utils/                  # Utilities (18 files)
├── demo/                   # Demos (3 files)
└── testing/                # Test scripts (49 files)

config/                      # Configuration files
├── default_config.yaml     # Default configuration
└── profiles/               # Configuration profiles
    ├── quick_test.yaml
    ├── full_training.yaml
    └── live_performance.yaml

tests/                       # Test suite (57 tests, 100% passing)
docs/                        # Documentation
data/                        # Data directories
├── input_audio/
└── training_output/
```

---

## 🚀 Usage

### Training Models

#### Modular Pipeline (Recommended)
```bash
# Standard training
python scripts/train/train_modular.py audio.wav output.json

# Quick test profile (1000 events)
python scripts/train/train_modular.py audio.wav output.json --profile quick_test

# Full training profile (20000 events)
python scripts/train/train_modular.py audio.wav output.json --profile full_training

# Custom configuration
python scripts/train/train_modular.py audio.wav output.json --config custom_config.yaml
```

#### Legacy Training
```bash
# Hybrid training with transformer
python scripts/train/train_hybrid.py --file audio.mp3 --output model.json --transformer

# Chord-based training
python scripts/train/chord_ground_truth_trainer.py --file audio.wav --output chords.json
```

### Python API

```python
from musichal.core import ConfigManager, AudioEvent, TrainingResult
from musichal.training import TrainingOrchestrator

# Load configuration
config = ConfigManager()
config.load(profile='quick_test')

# Customize settings
config.set('audio_oracle.distance_threshold', 0.2)

# Run training
orchestrator = TrainingOrchestrator(config)
result = orchestrator.run('audio.wav', 'output.json')

# Validate result
is_valid, issues = result.validate_training()
print(f"Training valid: {is_valid}")
```

---

## ⚙️ Configuration

### Configuration Profiles

**quick_test** - Fast iteration (1000 events, minimal features)
```bash
python scripts/train/train_modular.py audio.wav output.json --profile quick_test
```

**full_training** - Production quality (20000 events, all features)
```bash
python scripts/train/train_modular.py audio.wav output.json --profile full_training
```

**live_performance** - Low latency (<50ms response)
```bash
python scripts/performance/MusicHal_9000.py --profile live_performance
```

### Custom Configuration

Create `my_config.yaml`:
```yaml
audio_oracle:
  distance_threshold: 1.5
  max_pattern_length: 50

feature_extraction:
  sample_rate: 44100
  hop_length: 512

training:
  max_events: 10000
```

Use it:
```bash
python scripts/train/train_modular.py audio.wav output.json --config my_config.yaml
```

---

## 🔬 Architecture

### Training Pipeline (5 Stages)

1. **Audio Extraction** - Extract features from audio
2. **Feature Analysis** - Wav2Vec + dual perception
3. **Hierarchical Sampling** - Multi-timescale significance filtering
4. **Oracle Training** - Factor Oracle + Rhythm Oracle
5. **Validation** - Model validation and output generation

### Data Models (Pydantic)

```python
# Audio events with validation
event = AudioEvent(
    timestamp=1.5,
    features=AudioEventFeatures(
        f0=440.0,        # Validated: 0-10000 Hz
        rms_db=-30.0,    # Validated: -120 to 0 dB
        gesture_token=42 # Validated: 0-63
    )
)

# Training results with metadata
result = TrainingResult(
    metadata=TrainingMetadata(
        audio_file='test.wav',
        git_commit='f81aa4a',
        python_version='3.13.0'
    ),
    training_successful=True,
    events_processed=1000,
    audio_oracle_stats=oracle_stats
)
```

### Data Safety

- **Atomic Writes**: Temp file + rename (no corruption)
- **Versioned Backups**: Time-stamped with checksums
- **Schema Validation**: JSON schema enforcement
- **Metadata Tracking**: Git commit, system info, parameters

---

## 📊 Performance

| Operation | Latency | Throughput | GPU |
|-----------|---------|------------|-----|
| Live Performance | <50ms | Real-time | No |
| Training (Quick) | 2-5 min | 10-20 events/sec | Optional |
| Training (Full) | 10-30 min | 10-20 events/sec | Optional |
| Feature Extraction | - | 100+ events/sec | Yes (MPS) |

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test category
pytest tests/test_models.py -v
pytest tests/test_config_manager.py -v

# Quick test run
pytest tests/ -q

# With coverage
pytest tests/ --cov=musichal --cov-report=html
```

**Current Status:** 57/57 tests passing (100%) ✓

---

## 📖 Documentation

### Core Documentation
- **[RECENT_UPDATES.md](RECENT_UPDATES.md)** - November 2025 changelog (MERT, CLAP, modular pipeline)
- **[MERT_INTEGRATION.md](MERT_INTEGRATION.md)** - Music-optimized transformer encoding ⭐ **NEW**
- **[CLAP_STYLE_DETECTION.md](CLAP_STYLE_DETECTION.md)** - Automatic behavioral mode selection ⭐ **NEW**
- **[COMPLETE_ARTISTIC_RESEARCH_DOCUMENTATION.md](COMPLETE_ARTISTIC_RESEARCH_DOCUMENTATION.md)** - Full system context

### Architecture & Refactoring
- [Refactoring Complete Summary](docs/REFACTORING_COMPLETE.md) - Complete refactoring journey
- [Phase 2.4 Completion](docs/phase_2_4_completion.md) - Project structure improvements
- [Phase 2.3 Completion](docs/phase_2_3_completion.md) - Pydantic models
- [Scripts README](scripts/README.md) - Scripts organization guide
- [Legacy Code](legacy/README.md) - Deprecated code notices

---

## 🛠️ Development

### Package Structure
```python
# Import from package
from musichal.core import (
    ConfigManager,          # Configuration management
    AudioEvent,            # Audio event models
    TrainingResult,        # Training output models
    AtomicFileWriter,      # Safe file operations
    BackupManager,         # Backup management
    DataValidator          # Schema validation
)

from musichal.training import (
    TrainingOrchestrator,  # Pipeline orchestration
    PipelineStage,         # Stage base class
    StageResult            # Stage output
)
```

### Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Make changes with tests
4. Run test suite (`pytest tests/`)
5. Commit changes (`git commit -m 'Add amazing feature'`)
6. Push to branch (`git push origin feature/amazing-feature`)
7. Open Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Factor Oracle algorithm (Assayag et al.)
- Wav2Vec 2.0 (Meta AI)
- Pydantic validation framework
- PyTorch and TensorFlow ecosystems

---

## 📧 Contact

- **Issues**: [GitHub Issues](https://github.com/yourusername/musichal/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/musichal/discussions)

---

**Made with ♥ for musicians and AI enthusiasts**
