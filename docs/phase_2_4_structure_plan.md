# Phase 2.4: Project Structure Improvements

## Current State Analysis

### Directory Structure (35+ directories)
```
CCM4/
├── adaptive_sampling/         # Smart sampling utilities
├── agent/                     # AI agent logic (11 files)
├── analysis/                  # Analysis scripts
├── audio_file_learning/       # Audio file processing (13 files)
├── backups/                   # Versioned backups
├── CCM3/                      # Legacy CCM3 code
├── config/                    # YAML configurations (Phase 2.1)
├── controllers/               # MIDI and performance controllers
├── core/                      # Core infrastructure (Phase 1 & 2.3)
│   ├── data_safety/          # Phase 1 data safety
│   └── models/               # Phase 2.3 Pydantic models
├── correlation_engine/        # Correlation analysis
├── docs/                      # Documentation
├── fft_analyzer/             # FFT analysis
├── generators/               # Sound generators
├── hierarchical_analysis/    # Multi-timescale analysis
├── hybrid_training/          # Hybrid training methods
├── input_audio/              # Audio input files
├── JSON/                     # Training data output
├── listener/                 # Audio listening
├── mapping/                  # Mapping utilities
├── memory/                   # Memory buffer
├── midi_io/                  # MIDI I/O
├── perceptual_filtering/     # Perceptual filters
├── performance_arcs/         # Performance arc generation
├── predictive_processing/    # Predictive models
├── rhythmic_engine/          # Rhythm analysis
├── schemas/                  # JSON schemas (Phase 1)
├── tests/                    # Test suite (57 tests)
├── tools/                    # Utility tools (Phase 0)
├── training_pipeline/        # Modular pipeline (Phase 2.2)
├── tmp_epub/                 # Temporary files
└── visualization/            # Visualization tools
```

### Root-Level Python Files (50+ files)
```
analyze_*.py (8 analysis scripts)
autonomous_chord_trainer.py
ccm3_venv_manager.py
Chandra_trainer.py (2,413 lines - LEGACY, replaced by train_modular.py)
check_*.py (4 validation scripts)
chord_ground_truth_trainer*.py (3 variants)
complete_ground_truth_dataset.py
convert_*.py (2 conversion scripts)
create_*.py (4 creation scripts)
daemon_*.py (3 daemon processes)
demo_*.py (2 demo scripts)
export_*.py (2 export scripts)
fix_*.py (3 fix scripts)
inspect_*.py (5 inspection scripts)
list_*.py (2 list scripts)
live_*.py (2 live performance scripts)
main.py
prepare_*.py (2 preparation scripts)
rebuild_*.py (2 rebuild scripts)
retrain_*.py (2 retraining scripts)
run_*.py (4 runner scripts)
test_*.py (3 standalone tests)
train_*.py (4 training scripts)
view_*.py (2 viewing scripts)
+ many more...
```

---

## Problems Identified

### 1. **Root Directory Clutter**
- **50+ Python files** in root directory
- Difficult to find entry points
- No clear organization
- Mix of scripts, utilities, and main programs

### 2. **Inconsistent Naming**
- `analyze_*.py`, `check_*.py`, `fix_*.py` - should be in utilities
- `Chandra_trainer.py` - PascalCase vs snake_case
- Multiple variants of same functionality (e.g., `*_mps.py`, `*_hybrid.py`)

### 3. **Unclear Entry Points**
- `main.py` vs `live_performance.py` vs `daemon_main.py`
- Which is the current entry point?
- No clear separation of concerns

### 4. **Legacy Code**
- `CCM3/` directory (old version)
- `Chandra_trainer.py` (2,413 lines, replaced by `train_modular.py`)
- Multiple outdated variants

### 5. **Temporary Files**
- `tmp_epub/` in version control
- Should be in `.gitignore`

### 6. **Package Structure**
- Some directories are packages (`__init__.py`), others aren't
- Inconsistent import paths
- No clear top-level package

---

## Proposed Structure

### New Directory Organization
```
CCM4/
├── README.md                  # Project overview
├── setup.py                   # Package installation
├── requirements.txt           # Dependencies
├── .gitignore                 # Ignore patterns
│
├── musichal/                  # Main package (NEW)
│   ├── __init__.py
│   ├── core/                  # Core infrastructure
│   │   ├── __init__.py
│   │   ├── data_safety/       # Atomic writes, backups (Phase 1)
│   │   ├── models/            # Pydantic models (Phase 2.3)
│   │   ├── config.py          # ConfigManager (Phase 2.1)
│   │   └── metadata.py        # MetadataManager
│   │
│   ├── training/              # Training system
│   │   ├── __init__.py
│   │   ├── pipeline/          # Modular pipeline (Phase 2.2)
│   │   │   ├── stages/
│   │   │   └── orchestrator.py
│   │   ├── processors/        # Audio processors
│   │   │   ├── polyphonic.py
│   │   │   ├── wav2vec.py
│   │   │   └── rhythmic.py
│   │   └── samplers/          # Sampling strategies
│   │       ├── hierarchical.py
│   │       └── adaptive.py
│   │
│   ├── agent/                 # AI Agent
│   │   ├── __init__.py
│   │   ├── ai_agent.py
│   │   ├── behaviors.py
│   │   ├── harmonic/
│   │   │   ├── translator.py
│   │   │   ├── progressor.py
│   │   │   └── interval_extractor.py
│   │   ├── phrase/
│   │   │   ├── generator.py
│   │   │   └── memory.py
│   │   └── decision/
│   │       ├── scheduler.py
│   │       ├── explainer.py
│   │       └── density.py
│   │
│   ├── performance/           # Live performance
│   │   ├── __init__.py
│   │   ├── controllers/       # MIDI & state controllers
│   │   ├── arcs/              # Performance arcs
│   │   └── listener/          # Audio input
│   │
│   ├── analysis/              # Analysis tools
│   │   ├── __init__.py
│   │   ├── hierarchical/
│   │   ├── rhythmic/
│   │   ├── harmonic/
│   │   ├── perceptual/
│   │   └── correlation/
│   │
│   ├── memory/                # Memory systems
│   │   ├── __init__.py
│   │   ├── buffer.py
│   │   └── predictive.py
│   │
│   └── io/                    # Input/Output
│       ├── __init__.py
│       ├── midi.py
│       ├── audio.py
│       └── generators/
│
├── scripts/                   # Executable scripts (NEW)
│   ├── train/
│   │   ├── train_modular.py   # Main training entry point
│   │   ├── train_batch.py
│   │   └── train_from_saved_events.py
│   ├── performance/
│   │   ├── live_performance.py
│   │   └── daemon_main.py
│   ├── analysis/
│   │   ├── analyze_latest.py
│   │   ├── analyze_harmonics.py
│   │   └── analyze_features.py
│   ├── utils/
│   │   ├── check_model.py
│   │   ├── inspect_oracle.py
│   │   ├── fix_json.py
│   │   └── convert_data.py
│   └── demo/
│       ├── demo_agent.py
│       └── demo_translation.py
│
├── config/                    # Configuration files
│   ├── default_config.yaml
│   └── profiles/
│
├── tests/                     # Test suite (57 tests)
│   ├── unit/
│   ├── integration/
│   └── end_to_end/
│
├── docs/                      # Documentation
│   ├── architecture/
│   ├── api/
│   └── guides/
│
├── tools/                     # Development tools
│   ├── generate_backup_checksums.py
│   └── data_integrity_audit.py
│
├── data/                      # Data directories (NEW)
│   ├── input_audio/
│   ├── training_output/       # Replaces JSON/
│   └── models/                # Trained models
│
├── schemas/                   # JSON schemas
└── legacy/                    # Deprecated code (NEW)
    ├── CCM3/
    └── Chandra_trainer.py
```

---

## Migration Plan

### Step 1: Create Package Structure
1. Create `musichal/` top-level package
2. Move `core/` into `musichal/core/`
3. Move `training_pipeline/` to `musichal/training/pipeline/`
4. Create sub-packages with proper `__init__.py`

### Step 2: Organize Root Scripts
1. Create `scripts/` directory with subdirectories
2. Move analysis scripts to `scripts/analysis/`
3. Move training scripts to `scripts/train/`
4. Move utility scripts to `scripts/utils/`
5. Update imports in all moved scripts

### Step 3: Consolidate Related Code
1. Move audio processors to `musichal/training/processors/`
2. Move samplers to `musichal/training/samplers/`
3. Reorganize agent code into sub-packages
4. Consolidate analysis modules

### Step 4: Clean Up Legacy
1. Move `CCM3/` to `legacy/CCM3/`
2. Move `Chandra_trainer.py` to `legacy/`
3. Add `legacy/README.md` explaining deprecation
4. Update `.gitignore` for temporary files

### Step 5: Rename Data Directories
1. Rename `JSON/` to `data/training_output/`
2. Move `input_audio/` to `data/input_audio/`
3. Create `data/models/` for saved models
4. Update all path references

### Step 6: Create Entry Points
1. Create `musichal/cli.py` for command-line interface
2. Add `setup.py` with console_scripts entry points
3. Document standard entry points in README

### Step 7: Update Documentation
1. Add architecture overview
2. Document import paths
3. Create migration guide for existing code
4. Update all file references

---

## Benefits

### 1. **Clarity**
- Clear entry points via `scripts/`
- Logical grouping of related code
- Easy to find functionality

### 2. **Maintainability**
- Package structure enables proper imports
- Related code grouped together
- Clear separation of concerns

### 3. **Professionalism**
- Standard Python package layout
- Installable via `pip install -e .`
- Follows community conventions

### 4. **Discoverability**
- New contributors can navigate easily
- Clear distinction between library code and scripts
- Documented structure

### 5. **Testing**
- Test organization mirrors package structure
- Easy to find related tests
- Clear unit/integration/e2e separation

---

## Implementation

### Phase 2.4a: Create Package Structure (CURRENT)
- Create `musichal/` package
- Move core components
- Update imports

### Phase 2.4b: Organize Scripts
- Create `scripts/` directory
- Categorize and move scripts
- Update paths

### Phase 2.4c: Clean Legacy
- Move deprecated code to `legacy/`
- Document what's deprecated
- Update README

### Phase 2.4d: Documentation
- Architecture diagrams
- Import guide
- Migration documentation

---

## Success Criteria

- ✓ All 57 tests still pass
- ✓ Zero data loss (backups intact)
- ✓ Clear entry points documented
- ✓ Installable package (`pip install -e .`)
- ✓ No more than 10 files in root directory
- ✓ All Python code in packages
- ✓ Legacy code clearly separated

---

## Next Steps

Start with **Phase 2.4a: Create Package Structure**:
1. Create `musichal/` package directory
2. Move `core/` to `musichal/core/`
3. Move `training_pipeline/` to `musichal/training/pipeline/`
4. Update all imports
5. Run tests to verify nothing broke
