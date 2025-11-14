# MusicHal 9000 - Project Structure

```
CCM4/
│
├── 📦 musichal/                      # Main Python Package
│   ├── __init__.py                  # Package initialization
│   │
│   ├── core/                        # Core Infrastructure
│   │   ├── __init__.py              # Exports: ConfigManager, models, data safety
│   │   ├── config_manager.py        # YAML configuration system (Phase 2.1)
│   │   ├── metadata_manager.py      # Reproducibility tracking (Phase 1)
│   │   ├── engine.py                # Frequency engine
│   │   ├── manager.py               # Harmonic manager
│   │   ├── rhythm.py                # Rhythm engine
│   │   ├── voice.py                 # Voice synthesis
│   │   │
│   │   ├── data_safety/             # Data Safety System (Phase 1)
│   │   │   ├── __init__.py
│   │   │   ├── atomic_file_writer.py    # Atomic file operations
│   │   │   ├── backup_manager.py        # Versioned backups
│   │   │   ├── data_validator.py        # Schema validation
│   │   │   └── enhanced_save_load.py    # Safe I/O wrapper
│   │   │
│   │   └── models/                  # Pydantic Data Models (Phase 2.3)
│   │       ├── __init__.py
│   │       ├── audio_event.py           # AudioEvent, AudioEventFeatures
│   │       ├── musical_moment.py        # MusicalMoment (memory buffer)
│   │       ├── oracle_state.py          # AudioOracleStats, RhythmOracleStats
│   │       ├── performance_context.py   # PerformanceContext, BehaviorMode
│   │       └── training_result.py       # TrainingResult, TrainingMetadata
│   │
│   └── training/                    # Training System
│       ├── __init__.py              # Exports: TrainingOrchestrator
│       │
│       └── pipeline/                # Modular Training Pipeline (Phase 2.2)
│           ├── __init__.py
│           │
│           ├── stages/              # 5 Pipeline Stages
│           │   ├── __init__.py
│           │   ├── base_stage.py            # PipelineStage, StageResult
│           │   ├── audio_extraction_stage.py    # Stage 1: Extract features
│           │   ├── feature_analysis_stage.py    # Stage 2: Wav2Vec + perception
│           │   ├── hierarchical_sampling_stage.py  # Stage 3: Significance filter
│           │   ├── oracle_training_stage.py     # Stage 4: Train oracles
│           │   └── validation_stage.py          # Stage 5: Validate & output
│           │
│           └── orchestrators/       # Pipeline Orchestration
│               ├── __init__.py
│               └── training_orchestrator.py  # Main orchestrator
│
├── 📜 scripts/                       # Executable Scripts (108 files)
│   ├── README.md                    # Scripts documentation
│   │
│   ├── train/                       # Training Scripts (15 files)
│   │   ├── train_modular.py         # ⭐ Main training entry point
│   │   ├── train_hybrid.py
│   │   ├── train_hybrid_enhanced.py
│   │   ├── train_wav2vec_chord_classifier.py
│   │   ├── autonomous_chord_trainer.py
│   │   ├── chord_ground_truth_trainer.py
│   │   ├── chord_ground_truth_trainer_hybrid.py
│   │   ├── chord_ground_truth_trainer_wav2vec.py
│   │   ├── complete_ground_truth_dataset.py
│   │   ├── generate_full_chord_dataset.py
│   │   ├── generate_synthetic_chord_dataset.py
│   │   ├── learn_polyphonic_hybrid.py
│   │   └── learn_polyphonic_mps.py
│   │
│   ├── performance/                 # Live Performance (5 files)
│   │   ├── MusicHal_9000.py         # ⭐ Main performance entry
│   │   ├── MusicHal_bass.py
│   │   ├── main.py
│   │   ├── performance_simulator.py
│   │   └── performance_timeline_manager.py
│   │
│   ├── analysis/                    # Analysis Tools (11 files)
│   │   ├── analyze_conversation_log.py
│   │   ├── analyze_feature_collapse.py
│   │   ├── analyze_gesture_training_data.py
│   │   ├── analyze_harmonic_distribution.py
│   │   ├── analyze_itzama_run.py
│   │   ├── analyze_latest.py
│   │   ├── analyze_log.py
│   │   ├── diagnose_audiooracle.py
│   │   ├── diagnose_gesture_failure.py
│   │   ├── diagnose_rhythmic_integration.py
│   │   └── performance_arc_analyzer.py
│   │
│   ├── utils/                       # Utility Scripts (18 files)
│   │   ├── check_audio.py
│   │   ├── check_model_fields.py
│   │   ├── check_new_model_harmonics.py
│   │   ├── check_quantizer.py
│   │   ├── verify_distance_fix.py
│   │   ├── verify_dual_perception_ready.py
│   │   ├── validate_autonomous_trainer.py
│   │   ├── fix_correlation_analysis.py
│   │   ├── convert_model_to_pickle.py
│   │   ├── convert_pdfs.py
│   │   ├── convert_pdfs_docling.py
│   │   ├── regenerate_harmonic_transitions.py
│   │   ├── ccm3_venv_manager.py
│   │   ├── simple_chord_validator.py
│   │   ├── simple_hierarchical_integration.py
│   │   ├── ratio_based_chord_validator.py
│   │   ├── temporal_smoothing_optimization.py
│   │   ├── hierarchical_integration.py
│   │   ├── gpt_oss_client.py
│   │   └── gpt_reflection_engine.py
│   │
│   ├── demo/                        # Demo Scripts (3 files)
│   │   ├── demo_factor_oracle_advantages.py
│   │   ├── quick_gesture_check.py
│   │   └── quick_test.py
│   │
│   └── testing/                     # Test Scripts (49 files)
│       ├── test_*.py (48 files)
│       └── longer_test.py
│
├── ⚙️ config/                        # Configuration Files
│   ├── default_config.yaml          # Default configuration (230 lines)
│   │
│   └── profiles/                    # Configuration Profiles
│       ├── quick_test.yaml          # Fast testing (1000 events)
│       ├── full_training.yaml       # Production (20000 events)
│       └── live_performance.yaml    # Low latency (<50ms)
│
├── 🧪 tests/                         # Test Suite (57 tests, 100% passing)
│   ├── test_config_manager.py       # 11 tests
│   ├── test_data_safety.py          # 2 tests
│   ├── test_data_validator.py       # 3 tests
│   ├── test_metadata_manager.py     # 5 tests
│   ├── test_models.py               # 19 tests (Pydantic models)
│   ├── test_validation_integration.py  # 5 tests
│   └── ... (other tests)
│
├── 📋 schemas/                       # JSON Validation Schemas
│   ├── audio_oracle_schema.json     # AudioOracle data schema
│   ├── rhythm_oracle_schema.json    # RhythmOracle data schema
│   └── training_results_schema.json # Training output schema
│
├── 📚 docs/                          # Documentation
│   ├── REFACTORING_COMPLETE.md      # Complete refactoring summary
│   ├── phase_2_4_completion.md      # Project structure (Phase 2.4)
│   ├── phase_2_4a_completion.md     # Package creation (Phase 2.4a)
│   ├── phase_2_4_structure_plan.md  # Planning document
│   ├── phase_2_3_completion.md      # Pydantic models (Phase 2.3)
│   └── ... (other documentation)
│
├── 🗄️ legacy/                        # Deprecated Code (DO NOT USE)
│   ├── README.md                    # Deprecation notices
│   ├── Chandra_trainer.py           # Old 2,413-line monolithic trainer
│   │                                # → Replaced by train_modular.py
│   └── CCM3/                        # Previous system version
│
├── 💾 backups/                       # Versioned Backups (Phase 0)
│   └── pre_refactor_20251113_124720/  # Complete pre-refactoring backup
│       ├── JSON/                    # 183 JSON files (208 MB)
│       ├── checksums.json           # MD5 checksums
│       └── ...
│
├── 🛠️ tools/                         # Development Tools
│   ├── generate_backup_checksums.py # Backup verification tool
│   └── data_integrity_audit.py      # Data quality checker
│
├── 📁 Other Directories/             # Additional Components
│   ├── adaptive_sampling/           # Smart sampling utilities
│   ├── agent/                       # AI agent logic
│   ├── analysis/                    # Analysis modules
│   ├── audio_file_learning/         # Audio processing
│   ├── controllers/                 # MIDI controllers
│   ├── correlation_engine/          # Correlation analysis
│   ├── fft_analyzer/                # FFT analysis
│   ├── generators/                  # Sound generators
│   ├── hierarchical_analysis/       # Multi-timescale analysis
│   ├── hybrid_training/             # Hybrid training methods
│   ├── input_audio/                 # Audio input files
│   ├── JSON/                        # Training data output
│   ├── listener/                    # Audio listening
│   ├── mapping/                     # Mapping utilities
│   ├── memory/                      # Memory buffer
│   ├── midi_io/                     # MIDI I/O
│   ├── perceptual_filtering/        # Perceptual filters
│   ├── performance_arcs/            # Performance arc generation
│   ├── predictive_processing/       # Predictive models
│   ├── rhythmic_engine/             # Rhythm analysis
│   └── visualization/               # Visualization tools
│
├── 📄 Root Files
│   ├── setup.py                     # Package installation
│   ├── requirements.txt             # Dependencies
│   ├── README.md                    # Project README (updated)
│   ├── CHANGELOG.md                 # Version history
│   ├── PROJECT_STRUCTURE.md         # This file
│   └── .gitignore                   # Git ignore patterns
│
└── 🔧 Hidden/Config Files
    ├── .git/                        # Git repository
    ├── .github/                     # GitHub configuration
    ├── .claude/                     # Claude configuration
    └── .pytest_cache/               # Pytest cache
```

## 📊 Statistics

### Code Organization
- **Root .py files**: 1 (setup.py)
- **Package files**: ~50 files in `musichal/`
- **Scripts**: 108 files organized in 6 categories
- **Tests**: 57 tests (100% passing)
- **Documentation**: 8+ markdown files

### Key Entry Points

#### Training
```bash
python scripts/train/train_modular.py audio.wav output.json
```

#### Performance
```bash
python scripts/performance/MusicHal_9000.py
```

#### Python API
```python
from musichal.core import ConfigManager
from musichal.training import TrainingOrchestrator
```

## 🎯 Navigation Guide

### For Users
- **Start here**: `README.md`
- **Training**: `scripts/train/train_modular.py`
- **Performance**: `scripts/performance/MusicHal_9000.py`
- **Configuration**: `config/default_config.yaml`

### For Developers
- **Package code**: `musichal/`
- **Tests**: `tests/`
- **Documentation**: `docs/`
- **Schemas**: `schemas/`

### For Contributors
- **Setup**: `setup.py`, `requirements.txt`
- **Contributing guide**: `README.md` (Development section)
- **Changelog**: `CHANGELOG.md`
- **Project structure**: This file

## ✨ Highlights

### Professional Package (`musichal/`)
- Type-safe Pydantic models
- Modular 5-stage pipeline
- Data safety infrastructure
- Configuration system

### Organized Scripts (`scripts/`)
- 15 training scripts
- 5 performance scripts
- 11 analysis tools
- 18 utilities
- 3 demos
- 49 test scripts

### Complete Safety (`backups/`, `tools/`)
- Pre-refactoring backup (208 MB)
- MD5 checksum verification
- Data integrity tools

### Legacy Code (`legacy/`)
- Deprecated code separated
- Clear migration guides
- Historical reference

---

**Last Updated**: 2025-11-13
**Version**: 2.0.0
**Status**: Production Ready ✓
