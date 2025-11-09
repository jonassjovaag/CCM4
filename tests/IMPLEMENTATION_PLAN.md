# Safe Test Suite Implementation Plan
**Critical Rule:** ZERO modifications to production code (MusicHal_9000.py, Chandra_trainer.py, agent/, memory/, listener/, etc.)

## Safety Guarantees

### What We Will NOT Touch
- ❌ MusicHal_9000.py
- ❌ Chandra_trainer.py
- ❌ agent/phrase_generator.py (bug already fixed, leave it alone)
- ❌ memory/polyphonic_audio_oracle.py
- ❌ listener/hybrid_detector.py
- ❌ Any other production code

### What We WILL Create (tests/ directory only)
- ✅ tests/*.py (test files)
- ✅ tests/test_audio/ (test audio samples)
- ✅ tests/test_models/ (reference models for testing)
- ✅ tests/test_expected/ (expected outputs)
- ✅ tests/fixtures/ (reusable test data)
- ✅ tests/utils/ (test-only helper functions)
- ✅ tests/*.md (documentation)

### Isolation Strategy
All tests will:
1. **Import** production code (read-only)
2. **Load** existing trained models (read-only)
3. **Create** test data in tests/ directory only
4. **Never modify** any file outside tests/
5. **Use temporary directories** for any test outputs

## Implementation Phases

### Phase 1: Test Infrastructure (Foundation)
**Goal:** Create reusable test utilities without touching production code

#### Step 1.1: Test Utilities Module
**File:** `tests/utils/test_helpers.py`
```python
"""
Test utilities for MusicHal 9000 test suite
These are ONLY used by tests, never by production code
"""

import os
import tempfile
import gzip
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple

def get_test_root() -> Path:
    """Get absolute path to tests/ directory"""
    return Path(__file__).parent.parent

def get_project_root() -> Path:
    """Get absolute path to project root"""
    return Path(__file__).parent.parent.parent

def load_test_model(model_name: str) -> Dict:
    """Load a model from tests/test_models/ directory"""
    # Reads existing trained models WITHOUT modifying them
    pass

def create_synthetic_audio_data(...) -> Dict:
    """Generate synthetic audio event data for testing"""
    # Pure test data generation, doesn't touch production
    pass

def assert_feature_range(...):
    """Validate features are in expected ranges"""
    pass

def compute_model_statistics(model_data: Dict) -> Dict:
    """Extract statistics from loaded model"""
    # Statistical analysis helper
    pass
```

#### Step 1.2: Test Fixtures
**File:** `tests/fixtures/audio_events.py`
```python
"""
Synthetic audio event fixtures for testing
These are predictable, known-good events for validation
"""

def c_major_chord_event() -> Dict:
    """Returns synthetic event representing C major triad"""
    return {
        't': 0.0,
        'pitch_hz': 261.63,  # C4
        'amplitude': 0.8,
        'duration': 0.5,
        'gesture_token': 42,  # Arbitrary but consistent
        'consonance': 0.95,  # High consonance for major triad
        'frequency_ratios': [1.0, 1.25, 1.5],  # Perfect major intervals
        'chord_name_display': 'major'
    }

def a_minor_chord_event() -> Dict:
    """Returns synthetic event representing A minor triad"""
    # ...

def dissonant_cluster_event() -> Dict:
    """Returns synthetic event representing chromatic cluster"""
    # Low consonance, tight frequency ratios
    # ...
```

**File:** `tests/fixtures/models.py`
```python
"""
Minimal model fixtures for fast testing
"""

def minimal_audiooracle_dict() -> Dict:
    """
    Returns minimal valid AudioOracle structure (dict format)
    Just enough to test serialization/deserialization
    """
    return {
        'distance_threshold': 0.15,
        'distance_function': 'euclidean',
        'feature_dimensions': 15,
        'audio_frames': {
            0: {'audio_data': {...}, 'features': [...], ...},
            1: {...},
        },
        'states': {0: [], 1: []},
        'transitions': {0: [1]},
        'suffix_links': {}
    }
```

#### Step 1.3: Test Configuration
**File:** `tests/config.py`
```python
"""
Test configuration (paths, thresholds, expected values)
"""

from pathlib import Path

# Paths (all within tests/ directory)
TEST_ROOT = Path(__file__).parent
TEST_AUDIO_DIR = TEST_ROOT / "test_audio"
TEST_MODELS_DIR = TEST_ROOT / "test_models"
TEST_EXPECTED_DIR = TEST_ROOT / "test_expected"
TEST_OUTPUT_DIR = TEST_ROOT / "test_output"  # Temporary outputs

# Reference model (existing trained model - read-only)
REFERENCE_MODEL_PATH = TEST_ROOT.parent / "JSON" / "Itzama_071125_2130_training_model.pkl.gz"

# Expected ranges (from domain knowledge)
EXPECTED_RANGES = {
    'pitch_hz': (50.0, 2000.0),
    'consonance': (0.0, 1.0),
    'gesture_token': (0, 63),  # Harmonic vocabulary
    'amplitude': (0.0, 1.0),
}

# Performance targets
LATENCY_TARGETS = {
    'feature_extraction_ms': 10.0,
    'oracle_query_ms': 30.0,
    'end_to_end_ms': 50.0,
}

# Statistical thresholds
STATISTICAL_THRESHOLDS = {
    'min_unique_tokens': 20,  # Expect at least 20 unique gesture tokens
    'consonance_std_min': 0.1,  # Should have variation, not collapsed
}
```

### Phase 2: Refactor Existing Tests (Make Scientific)
**Goal:** Improve test_model_inspection.py and test_note_extraction.py without changing what they import

#### Step 2.1: Refactor test_model_inspection.py
**Changes:**
- Add statistical analysis using test_helpers
- Document expected vs. actual values
- Create assertions with meaningful error messages
- Output summary statistics to JSON file in tests/test_output/

**Example structure:**
```python
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import production code (READ-ONLY)
from memory.polyphonic_audio_oracle import PolyphonicAudioOracle

# Import test utilities
from utils.test_helpers import load_test_model, compute_model_statistics
from config import REFERENCE_MODEL_PATH, EXPECTED_RANGES, STATISTICAL_THRESHOLDS

def test_model_structure():
    """Test 1: Validate model structure"""
    model_data = load_test_model(REFERENCE_MODEL_PATH)
    
    # Statistical analysis
    stats = compute_model_statistics(model_data)
    
    # Assertions with research context
    assert stats['unique_gesture_tokens'] >= STATISTICAL_THRESHOLDS['min_unique_tokens'], \
        f"Gesture tokens may have collapsed: only {stats['unique_gesture_tokens']} unique values"
    
    # Save results for research documentation
    save_test_results('model_inspection', stats)
```

#### Step 2.2: Refactor test_note_extraction.py
**Changes:**
- Test with multiple known events (use fixtures)
- Verify MIDI conversion accuracy with known frequencies
- Test all fallback paths (pitch_hz → f0 → midi → midi_note)
- Measure extraction success rate

### Phase 3: New Test Modules (One at a time)
**Goal:** Build comprehensive coverage incrementally

#### Step 3.1: Feature Extraction Tests
**File:** `tests/test_feature_extraction.py`
**Dependencies:** 
- Imports: listener/hybrid_detector.py (read-only)
- Test data: fixtures/audio_events.py

**Tests:**
1. Gesture token extraction from known audio
2. Consonance calculation accuracy
3. Frequency ratio normalization
4. Edge cases (silence, single note, noise)

**Safety:** Only uses synthetic data from fixtures, never modifies production code

#### Step 3.2: AudioOracle Training Tests
**File:** `tests/test_audiooracle_training.py`
**Dependencies:**
- Imports: memory/polyphonic_audio_oracle.py (read-only)
- Test data: fixtures/models.py, fixtures/audio_events.py

**Tests:**
1. Train on minimal synthetic dataset
2. Verify graph structure (states, transitions, suffix_links)
3. Test serialization round-trip
4. Validate distance calculations

**Safety:** Creates temporary AudioOracle instances, trains on synthetic data, discards after test

#### Step 3.3: Request Masking Tests
**File:** `tests/test_request_masking.py`
**Dependencies:**
- Imports: memory/polyphonic_audio_oracle.py (read-only)
- Test data: Load existing trained model (read-only)

**Tests:**
1. Form requests with various constraints
2. Query oracle, verify returned frames match constraints
3. Test multi-constraint requests
4. Measure constraint satisfaction rate

**Safety:** Loads existing model, queries it (read-only operation), never modifies

#### Step 3.4: Viewport Translation Tests
**File:** `tests/test_viewport_translation.py` (major refactor)
**Dependencies:**
- Imports: listener/dual_perception.py ratio_to_chord_name (read-only)
- Test data: fixtures/audio_events.py

**Tests:**
1. Test ratio_to_chord_name() with known intervals
2. Verify gesture_token → chord_name_display mapping
3. Test dual vocabulary consistency
4. Identify unknown/edge cases

**Safety:** Pure function testing, no state modification

#### Step 3.5: End-to-End Integration Tests
**File:** `tests/test_end_to_end_integration.py` (major refactor)
**Dependencies:**
- Imports: Multiple production modules (all read-only)
- Test data: Synthetic audio events simulating live performance

**Tests:**
1. Simulate audio input → feature extraction → request formation
2. Load model → query oracle → extract notes → verify MIDI output format
3. Compare training vs. live data flow
4. Measure end-to-end latency

**Safety:** Uses temporary directories for any outputs, never touches production models

### Phase 4: Test Audio Generation (Optional)
**Goal:** Create synthetic test audio files

#### Step 4.1: Audio Synthesis Script
**File:** `tests/utils/generate_test_audio.py`
**Purpose:** Generate test_audio/*.wav files with known properties

```python
"""
Generate synthetic audio files for testing
Run once to create test_audio/ directory contents
"""

import numpy as np
import soundfile as sf

def generate_major_chord():
    """Generate 1-second C major triad"""
    # Sine waves at 261.63 Hz (C4), 329.63 Hz (E4), 392.00 Hz (G4)
    # Save to tests/test_audio/c_major_chord.wav
    pass

def generate_chromatic_cluster():
    """Generate dissonant cluster"""
    pass

if __name__ == "__main__":
    # Create all test audio files
    generate_major_chord()
    generate_chromatic_cluster()
    # ...
```

**Safety:** Creates files only in tests/test_audio/, never touches production audio

### Phase 5: Documentation & Results
**Goal:** Document findings

#### Step 5.1: Automated Results Collection
**File:** `tests/utils/results_collector.py`
```python
"""
Collect and aggregate test results for scientific reporting
"""

def collect_all_test_results() -> Dict:
    """Read all test_output/*.json files and aggregate"""
    pass

def generate_statistics_report() -> str:
    """Create markdown report with statistics"""
    pass
```

#### Step 5.2: Results Documentation
**File:** `tests/RESULTS.md` (auto-generated)
- Summary statistics from all tests
- Answers to research questions
- Performance metrics

## Implementation Order

### Week 1: Foundation
1. Create tests/utils/test_helpers.py
2. Create tests/fixtures/audio_events.py
3. Create tests/fixtures/models.py
4. Create tests/config.py
5. Verify imports work, no production code touched

### Week 2: Refactoring
6. Refactor test_model_inspection.py (use new utilities)
7. Refactor test_note_extraction.py (add fixtures, statistics)
8. Verify existing model still loads correctly

### Week 3: New Tests (Part 1)
9. Create test_feature_extraction.py
10. Create test_audiooracle_training.py
11. Run and verify, collect results

### Week 4: New Tests (Part 2)
12. Create test_request_masking.py
13. Refactor test_viewport_translation.py
14. Refactor test_end_to_end_integration.py

### Week 5: Polish
15. Generate test audio files (optional)
16. Create results collection system
17. Generate RESULTS.md
18. Create data flow diagram

## Safety Checklist (Before Each Step)

Before implementing any step, verify:
- [ ] No imports will cause side effects in production code
- [ ] All file creation is within tests/ directory
- [ ] All file reading is read-only (no writes to production files)
- [ ] Test can run without affecting MusicHal_9000.py or Chandra_trainer.py
- [ ] If test fails, production code remains unaffected

## Rollback Plan

If anything goes wrong:
1. All changes are in tests/ directory
2. Simply `git checkout tests/` to revert
3. Production code remains untouched
4. Can continue using MusicHal_9000.py and Chandra_trainer.py as before

## Success Metrics

**For each phase:**
- [ ] All tests pass
- [ ] No production code modified
- [ ] Results documented in tests/test_output/
- [ ] Can run `python MusicHal_9000.py` successfully (unchanged)
- [ ] Can run `python Chandra_trainer.py` successfully (unchanged)

**Final success:**
- [ ] Complete test suite runs in <5 minutes
- [ ] All research questions answered (documented in RESULTS.md)
- [ ] Zero regressions in production code
- [ ] Clear separation: tests/ vs. production code

---

**Ready to proceed?** I'll start with Phase 1, Step 1.1 (test_helpers.py) and work incrementally, verifying safety at each step.
