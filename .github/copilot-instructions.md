# MusicHal 9000 - AI Musical Partner System

This is **MusicHal 9000**, an AI musical partner for real-time improvisation. Built as artistic research (PhD project) exploring machine listening, musical memory, and trust through transparency.

## Architecture Overview

### The Three-Pipeline System

1. **Training Pipeline** (`Chandra_trainer.py`) - Learns from audio recordings
   - Analyzes audio → extracts multi-dimensional features → trains AudioOracle graph structure
   - Combines: Wav2Vec perceptual encoding + Brandtsegg rhythm ratios + Music Theory Transformer
   - Outputs: Serialized JSON models (`JSON/` directory) containing learned patterns
   
2. **Live Performance** (`main.py`, `MusicHal_9000.py`) - Real-time musical interaction
   - Listens via audio input → extracts features → queries trained models → generates MIDI output
   - <50ms latency from audio in to MIDI out
   - Uses **sticky behavioral modes** (30-90s duration) for consistent personality

3. **Behavioral Intelligence** (`agent/`, `correlation_engine/`) - Decision-making layer
   - Request masking: multi-parameter conditional generation (consonance + rhythm + gesture tokens)
   - Phrase memory: 20 motifs with variations, recalled every 30-60s for thematic development
   - Performance arc: temporal structure guiding engagement over 5-15 minute performances

### Core Memory: AudioOracle (Factor Oracle)

The heart of the system is **AudioOracle** (`memory/polyphonic_audio_oracle.py`, `memory/audio_oracle_mps.py`):

- **Graph structure**: states (nodes) + transitions (edges) + suffix links (pattern repetitions)
- **Learning**: O(m) incremental - processes features sequentially, creates suffix links to similar states
- **Generation**: traverses graph using request constraints + transition probabilities
- **Training**: load audio file → extract 15D feature vectors → build graph → serialize to JSON
- **Performance**: load JSON → query with constraints → follow transitions → output MIDI

**Critical detail**: Suffix links point to similar past patterns. This enables both continuation (forward transitions) and variation (jumping through suffix links). Distance threshold = 0.15 (Euclidean).

MPS-accelerated version uses Apple Silicon GPU for real-time distance calculations.

### Feature Extraction Philosophy

**Multi-modal perception** (subsymbolic + symbolic + semantic):

1. **MERT-v1-95M** (`listener/mert_encoder.py`): 768D music-optimized neural embeddings → gesture tokens
   - Music-specific transformer pre-trained on 160K hours of music
   - Superior musical understanding vs. general-purpose Wav2Vec
   - Captures harmony, rhythm, timbre, and musical semantics
   - Primary feature extractor for new models

2. **CLAP Style Detection** (`listener/clap_style_detector.py`): Audio-text alignment → behavioral modes
   - LAION `clap-htsat-unfused` model for semantic style understanding
   - Maps musical styles (ballad/rock/jazz/ambient) to behavioral modes (SHADOW/MIRROR/COUPLE)
   - Enables automatic mode selection based on musical context
   - Optional, falls back to manual mode selection

3. **Neural Audio Model** (`listener/wav2vec_perception.py`, `listener/hybrid_detector.py`): Legacy support
   - Originally Wav2Vec 2.0 (768D general audio embeddings)
   - Kept for backward compatibility with older models trained before MERT
   - New models should use MERT (section 1)

4. **Frequency Ratio Analysis** (`listener/ratio_analyzer.py`): harmonic frequency ratios → consonance scores
   - **FrequencyRatioAnalyzer** class analyzes pitch relationships (3:2 perfect fifth, 5:4 major third, etc.)
   - Mathematical harmonic relationships independent of cultural chord naming
   - Outputs: `consonance`, `dissonance`, `ratio_features[3]`, `frequency_ratios[6]`
   - **NOT Brandtsegg** - this is harmonic theory, not rhythmic

5. **Brandtsegg Rhythm Ratios** (`rhythmic_engine/`): onset timing relationships → rhythmic patterns
   - **ONLY for rhythm** - analyzes temporal ratios between onsets, syncopation, tempo-independent patterns
   - **NOT for harmony** - separate from frequency ratio analysis
   - Used by RhythmOracle for rhythmic pattern learning

6. **Rhythm Analysis** (`rhythmic_engine/`): onset detection + tempo tracking + syncopation
   - RhythmOracle learns rhythmic patterns parallel to harmonic AudioOracle
   - Correlation engine discovers harmonic-rhythmic relationships

**CRITICAL DISTINCTION - DO NOT CONFUSE**:
- **Harmonic ratios** = FrequencyRatioAnalyzer = pitch frequency relationships (3:2, 5:4) = MUSIC THEORY
- **Brandtsegg ratios** = Rhythmic timing relationships = onset intervals = RHYTHM ONLY
- These are completely separate systems analyzing different musical dimensions
**Why this matters**: MERT provides music-aware perceptual features that understand musical concepts (chords, keys, genres) without requiring symbolic transcription. CLAP adds semantic layer for style-driven behavioral adaptation. Chord names (Cmaj7, Dm7) are post-hoc symbolic labels—the system learns from perceptual features first, symbolic interpretation second. This aligns with practice-based artistic research goals.

## Development Workflows

### Training a New Model

```bash
# Full pipeline with all analysis layers (recommended)
python Chandra_trainer.py --file "input_audio/recording.wav" --output "JSON/model.json" --max-events 15000

# Quick test (2000 events processing, 100 training)
python Chandra_trainer.py --file "test.wav" --output "test_model.json" --max-events 2000 --training-events 100

# Disable specific analysis layers
python Chandra_trainer.py --file "audio.wav" --output "model.json" --no-transformer --no-rhythmic
```

**Training output**: JSON file with AudioOracle graph (`states`, `transitions`, `suffix_links`, `audio_frames`) + RhythmOracle patterns + transformer insights + GPT-OSS analysis.

**Key parameters**:
- `--max-events`: Limits onset extraction (default 10000)
- `--training-events`: Limits AudioOracle training (speeds up testing)
- `--sampling-strategy`: `balanced`|`structural`|`perceptual` (hierarchical filtering)

### Running Live Performance

```bash
# Enhanced mode (rhythmic + correlation intelligence)
python MusicHal_9000.py --enable-rhythmic

# Traditional mode (harmonic only)
python main.py

# With performance arc (5-minute structured performance)
python main.py --duration 5
```

**Runtime behavior**:
- Auto-loads trained models from `ai_learning_data/`
- Listens to default audio input (change with `--input-device`)
- Outputs MIDI to "IAC Driver Melody Channel" (change with `--midi-port`)
- Logs decisions to `logs/` directory for transparency analysis

### Testing & Debugging

```bash
# Component tests
python test_simple_chord_detection.py  # Chord detection validation
python test_ratio_analysis_integration.py  # Request masking
python test_temporal_smoothing.py  # Anti-flicker system

# End-to-end system tests
python test_end_to_end_system.py
python test_system_components.py
```

**Common debugging pattern**: Check logs in `logs/` directory. Each decision includes reasoning (why this mode, which patterns matched, request parameters applied).

### File Structure Guide

```
├── main.py, MusicHal_9000.py          # Live performance entry points
├── Chandra_trainer.py                 # Training pipeline (THE main trainer)
├── memory/
│   ├── polyphonic_audio_oracle.py     # Core AudioOracle (CPU)
│   ├── polyphonic_audio_oracle_mps.py # MPS GPU-accelerated version
│   └── memory_buffer.py               # Short-term memory (180s)
├── agent/
│   ├── ai_agent.py                    # Decision coordinator
│   ├── behaviors.py                   # Behavioral modes (imitate/contrast/lead)
│   ├── phrase_generator.py            # Long-term motif memory
│   └── decision_explainer.py          # Transparency logging
├── listener/
│   ├── jhs_listener_core.py          # Audio input + onset detection
│   ├── mert_encoder.py               # MERT-v1-95M music embeddings
│   ├── clap_style_detector.py        # CLAP style-based mode selection
│   ├── hybrid_detector.py            # Wav2Vec gesture tokens (legacy)
│   └── ratio_analyzer.py             # Brandtsegg ratio analysis
├── rhythmic_engine/                   # Parallel rhythmic intelligence
│   ├── memory/rhythm_oracle.py       # Rhythmic pattern memory
│   └── agent/rhythmic_behavior_engine.py
├── correlation_engine/                # Harmonic-rhythmic correlation
│   ├── correlation_analyzer.py       # Cross-modal pattern discovery
│   └── unified_decision_engine.py    # Combined decision-making
├── hybrid_training/                   # Music theory transformer
│   └── music_theory_transformer.py   # Deep music analysis
├── JSON/                             # Trained models (serialized)
├── ai_learning_data/                 # Runtime persistent memory
└── logs/                             # Decision transparency logs
```

## Project-Specific Conventions

### 1. Feature Vector Dimensions Matter

- AudioOracle expects **15-dimensional** features (polyphonic)
- MERT outputs **768-dimensional** → quantized to gesture tokens (primary encoder)
- Wav2Vec outputs **768-dimensional** → quantized to gesture tokens (legacy support)
- Ratio analyzer outputs **3D consonance vector + 6D frequency ratios**
- **Never mix dimensions** - each analysis layer has specific shapes

### 2. Sticky Behavioral Modes

Modes persist 30-90 seconds (not rapid flickering). See `agent/behaviors.py`:

```python
# Modes transition on timer + musical context, not every event
- Imitate: shadows input closely (0.7-0.9 similarity)
- Contrast: inverts/diverges (lower similarity threshold)
- Lead: independent patterns (ignores recent input)
```

### 3. Request Masking (Critical Pattern)

All generation uses **request masking** - multi-parameter constraints applied to AudioOracle queries:

```python
request = {
    'gesture_token': 142,              # Wav2Vec quantized token
    'consonance': 0.8,                 # Ratio-based consonance
    'rhythm_ratio': [3, 2],            # Brandtsegg rhythm pattern
    'tension': 0.6,                    # Harmonic tension
    'register': 'mid',                 # Pitch register
    'density': 0.5                     # Note density
}
```

AudioOracle filters states by these constraints before selecting transitions. This enables "play something consonant with this gesture in medium register" instead of pure statistical continuation.

### 4. Serialization Requirements

All trained models serialize to JSON. **Critical**: NumPy arrays → lists, defaultdicts → dicts, avoid circular references. See `_json_serialize_helper()` methods in AudioOracle classes.

### 5. MPS (Apple Silicon GPU) vs CPU

- Training: Force CPU for AudioOracle (compatibility), GPU for Wav2Vec
- Live performance: MPS-accelerated AudioOracle (`PolyphonicAudioOracleMPS`) for real-time queries
- Check: `torch.backends.mps.is_available()` before enabling

### 6. Temporal Smoothing (Anti-Flicker)

Sustained chords create duplicate onset events → flicker. Solution: `core/temporal_smoothing.py` groups events within 300ms windows, deduplicates if feature change < threshold.

**When editing**: If you add new feature extraction, update temporal smoothing to handle those features.

### 7. Documentation Philosophy

This is **artistic research**, not product engineering. Documentation in:
- `COMPLETE_ARTISTIC_RESEARCH_DOCUMENTATION.md` - full context, methodology, reasoning
- `README.md` - practical commands
- Inline comments explain "why" not just "what"

**Avoid**: Generic software patterns docs. **Preserve**: Musical reasoning, research methodology, subjective experience.

## Critical Integration Points

### AudioOracle ↔ Agent

Agent queries AudioOracle with request constraints → AudioOracle returns matching states → Agent applies behavioral mode filtering → Phrase generator adds variations.

**Key files**: `agent/behaviors.py` calls `memory/polyphonic_audio_oracle.py::generate_with_request()`

### Listener → Memory → Agent → Output

1. `listener/jhs_listener_core.py` - onset detection + feature extraction
2. `memory/memory_buffer.py` - 180s ring buffer
3. `agent/ai_agent.py` - decision coordination
4. `midi_io/midi_output.py` - MIDI transmission

**Latency target**: <50ms end-to-end. Profile with `core/logger.py::PerformanceLogger`.

### Hierarchical Analysis (Training Only)

`simple_hierarchical_integration.py` applies perceptual filtering during training:
- Multi-timescale streams (section/phrase/measure)
- Adaptive sampling based on musical significance
- Output: pruned event set for AudioOracle training

**Not used in live performance** - hierarchical structure baked into trained models.

### GPT-OSS Optional Analysis

If `gpt_oss_client.py` configured, adds LLM-based musical intelligence analysis during training. Outputs insights like "this section shows increasing rhythmic complexity" → embedded in trained model metadata.

**Disable with**: `--no-gpt-oss` flag.

## Common Pitfalls

1. **Training without temporal smoothing** → chord flicker → noisy models
2. **Mixing MPS and CPU AudioOracle** → serialization errors
3. **Forgetting to update request mask** when adding new features → constraints ignored
4. **Large max-events on first test** → long training time. Start with 2000 events.
5. **Editing behavioral mode durations** without updating scheduler → unstable personality
6. **Not checking logs/** directory when debugging decisions → flying blind

## Research Context

This is Jonas Sjøvaag's PhD artistic research at University of Agder. Goals:
- Build musical partner trustworthy enough to improvise with
- Achieve trust through transparency (visible reasoning) + coherence (memory) + personality (behavioral modes)
- Practice-based methodology: iterative development guided by musical experience

**Not goals**: Symbolic music theory engine, style transfer, generative composition, multi-user system.

The system learns from specific recordings (Itzama.wav - electronic pop/beat music) to develop localized musical identity. Like musicians develop personality through their practice history, MusicHal's character emerges from training data + architectural constraints.

---

**Quick reference commands**:
- Train: `python Chandra_trainer.py --file audio.wav --output model.json`
- Perform: `python MusicHal_9000.py --enable-rhythmic`
- Test: `python test_end_to_end_system.py`

**When editing**: Preserve the three-pipeline architecture. Training changes should serialize properly. Live performance changes must maintain <50ms latency. Behavioral changes should respect sticky mode durations.

---

## Development Principles

### Code Quality & Maintenance

**Context Before Code**: Always read related files, documentation, and code comments before making changes. Use VS Code's search and outline tools to trace logic paths. Never guess—confirm through inspection and reasoning. This codebase has deep integration points (AudioOracle ↔ Agent ↔ RhythmOracle) that require understanding the full data flow.

**Focused, Minimal Edits**: Keep each edit small, specific, and purposeful. Avoid large refactors unless absolutely necessary. Prefer surgical edits that improve clarity or correctness. Each commit should have a single clear intent. Example: If fixing request masking, don't also refactor behavioral modes.

**Iterative Debugging**: Start from the simplest working state. Use breakpoints, logging (`print` statements), and VS Code's debugger. Validate one step at a time—commit progress incrementally. Check `logs/` directory for decision transparency data when debugging agent behavior.

### Quality Standards

**Avoid Heuristics and Hacks**: Don't rely on "reasonable guesses" when specs are explicit. If a heuristic feels necessary, step back and analyze the real issue. Fix root causes, not surface symptoms. Example: Don't add arbitrary delay constants to fix timing issues—understand the onset detection timing model.

**Maintain Structural Integrity**: Never remove tests or disable checks just to make things compile. Don't sacrifice readability or design for quick fixes. Keep code clean, consistent, and logically structured. The three-pipeline architecture (Training/Performance/Intelligence) must remain distinct.

**Manage Resources Precisely**: Avoid arbitrary buffer sizes or magic numbers. Be explicit about performance and memory requirements. Example: `memory_buffer.py` uses 180-second ring buffer for specific musical memory reasons, not arbitrary choice. Document the "why" for such constants.

### Problem-Solving

**Systematic Debugging**: Use VS Code's debugger, logging, and watch variables extensively. Break complex issues into smaller, reproducible test cases. For audio/MIDI issues, use the test files in `Python test files/` directory. Don't rely solely on visual inspection—use logging to verify data flow.

**Think System-Wide**: Understand how changes affect the entire system. A change to feature extraction in `listener/` affects AudioOracle training in `memory/` and generation in `agent/`. Use refactors to reduce technical debt. Follow idiomatic Python patterns and music-domain naming conventions.

**Focus and Persistence**: Work in focused sessions (20-40 minutes). Don't abandon an approach after a single failure—iterate with insight. Keep a clear mental model of the end goal. Musical coherence issues may require multiple iterations to solve properly.

### Testing & Verification

**Test Components Individually**: Test components before integration. Use unit tests in `tests/` directory. Run component tests (`test_simple_chord_detection.py`, `test_ratio_analysis_integration.py`) before end-to-end tests. Follow deterministic specs—avoid "mostly working" results.

**Documentation and Logging**: Add logging and comments early in development. Document complex logic, especially musical reasoning. Keep inline notes for feature dimensions, distance thresholds, and timing constants. Ensure debug logs are useful, readable, and toggleable (via verbosity flags).

### Project-Specific Guidelines

**Musical Reasoning First**: This is artistic research, not just software. Comments should explain musical intent, not just technical implementation. Example: "30-90s behavioral persistence creates recognizable personality" not just "mode timer = 60s".

**Feature Vectors Are Sacred**: Never change feature dimensions without updating ALL components (extraction → memory → generation → serialization). 15D polyphonic features are the backbone. Test serialization after any feature changes.

**Latency Matters**: Live performance requires <50ms latency. Profile with `PerformanceLogger`. Avoid expensive operations in the hot path (listener → agent → MIDI output). GPU acceleration (MPS) is available but must be properly managed.

**Serialization Discipline**: All models must serialize to JSON. NumPy arrays → lists, defaultdicts → dicts, no circular references. Test load/save cycle after training changes. See `_json_serialize_helper()` pattern in AudioOracle classes.

**Git Workflow**: This is on branch `refactoring`. Small, focused commits with clear messages. Reference the three-pipeline architecture in commit messages when relevant. Document breaking changes in commit body.
