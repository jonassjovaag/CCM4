"""
Test configuration for MusicHal 9000 test suite
Centralizes paths, thresholds, and expected values
"""

from pathlib import Path

# === PATHS (all read-only or within tests/) ===

# Test directories
TEST_ROOT = Path(__file__).parent
PROJECT_ROOT = TEST_ROOT.parent

# Test subdirectories (will be created as needed)
TEST_AUDIO_DIR = TEST_ROOT / "test_audio"
TEST_MODELS_DIR = TEST_ROOT / "test_models"
TEST_EXPECTED_DIR = TEST_ROOT / "test_expected"
TEST_OUTPUT_DIR = TEST_ROOT / "test_output"
TEST_FIXTURES_DIR = TEST_ROOT / "fixtures"

# Reference model (existing trained model - READ-ONLY)
REFERENCE_MODEL_PATH = PROJECT_ROOT / "JSON" / "Itzama_071125_2130_training_model.pkl.gz"

# === EXPECTED RANGES (from domain knowledge) ===

EXPECTED_RANGES = {
    # Audio features
    'pitch_hz': (50.0, 2000.0),          # Musical range from low bass to high soprano
    'amplitude': (0.0, 1.0),             # Normalized amplitude
    'duration': (0.01, 10.0),            # Note duration in seconds
    
    # Feature extraction
    'consonance': (0.0, 1.0),            # Brandtsegg consonance measure
    'gesture_token_harmonic': (0, 63),   # Harmonic vocabulary size
    'gesture_token_percussive': (0, 63), # Percussive vocabulary size
    
    # MIDI
    'midi_note': (0, 127),               # MIDI note range
    'velocity': (0, 127),                # MIDI velocity
    
    # Frequency ratios
    'frequency_ratio': (1.0, 2.0),       # Within octave
}

# === STATISTICAL THRESHOLDS ===

STATISTICAL_THRESHOLDS = {
    # Gesture token diversity
    'min_unique_tokens': 20,             # Expect at least 20 unique tokens (avoid collapse)
    'max_single_token_dominance': 0.5,   # No single token should be >50% of all tokens
    
    # Consonance variation
    'consonance_std_min': 0.05,          # Should have some variation (not collapsed)
    'consonance_mean_min': 0.3,          # Shouldn't be purely dissonant
    'consonance_mean_max': 0.9,          # Shouldn't be purely consonant
    
    # Graph connectivity
    'min_suffix_link_coverage': 0.5,     # At least 50% of states should have suffix links
    'min_avg_transitions_per_state': 1.0, # Each state should have at least 1 transition on average
    
    # Pitch distribution
    'pitch_hz_std_min': 50.0,            # Should have pitch variation
}

# === PERFORMANCE TARGETS ===

LATENCY_TARGETS = {
    # Component-level latency (milliseconds)
    'feature_extraction_ms': 10.0,       # Wav2Vec + Brandtsegg analysis
    'oracle_query_ms': 30.0,             # AudioOracle.generate_with_request()
    'note_extraction_ms': 5.0,           # Extract MIDI from audio_data
    
    # End-to-end latency
    'end_to_end_ms': 50.0,               # Total: audio input → MIDI output
}

# === KNOWN TEST CASES ===

# Frequency → MIDI conversions (for validation)
FREQUENCY_MIDI_PAIRS = [
    (440.0, 69),    # A4
    (220.0, 57),    # A3
    (880.0, 81),    # A5
    (261.63, 60),   # C4
    (523.25, 72),   # C5
]

# Simple chord frequency ratios
KNOWN_CHORD_RATIOS = {
    'major_triad': [1.0, 1.25, 1.5],              # Root, major third, perfect fifth
    'minor_triad': [1.0, 1.2, 1.5],               # Root, minor third, perfect fifth
    'perfect_fifth': [1.0, 1.5],                  # Root, perfect fifth
    'octave': [1.0, 2.0],                         # Root, octave
    'chromatic_cluster': [1.0, 1.06, 1.12, 1.19], # Very dissonant
}

# Expected consonance for known chords
EXPECTED_CONSONANCE = {
    'major_triad': (0.8, 1.0),           # High consonance
    'minor_triad': (0.7, 0.9),           # Moderately high
    'perfect_fifth': (0.85, 1.0),        # Very consonant
    'chromatic_cluster': (0.0, 0.3),     # Dissonant
}

# === SCIENTIFIC RESEARCH QUESTIONS ===

RESEARCH_QUESTIONS = {
    'Q1': 'How many unique gesture tokens does Itzama.wav produce?',
    'Q1_expected': '50-63 (most of vocabulary used)',
    
    'Q2': 'What is the consonance distribution in training data?',
    'Q2_expected': 'Bimodal or wide distribution (0.5-0.9)',
    
    'Q3': 'How dense is the AudioOracle graph?',
    'Q3_expected': 'Avg 1.5-2.5 transitions per state, 50-70% suffix link coverage',
    
    'Q4': 'What is the mapping between gesture tokens and chord names?',
    'Q4_expected': 'Many-to-many (one token can map to multiple chords)',
    
    'Q5': 'Why is chord_name_display "unknown" in some frames?',
    'Q5_expected': 'Music Theory Transformer couldn\'t classify or edge cases',
    
    'Q6': 'Does request masking actually constrain generation?',
    'Q6_expected': '>90% constraint satisfaction rate',
    
    'Q7': 'What is the end-to-end latency distribution?',
    'Q7_expected': 'Mean <50ms, 95th percentile <100ms',
}

# === TEST DATA ===

# Synthetic test events (minimal examples)
SYNTHETIC_C_MAJOR_EVENT = {
    't': 0.0,
    'pitch_hz': 261.63,  # C4
    'amplitude': 0.8,
    'duration': 0.5,
    'gesture_token': 42,
    'consonance': 0.95,
    'frequency_ratios': [1.0, 1.25, 1.5],
    'chord_name_display': 'major'
}

SYNTHETIC_A_MINOR_EVENT = {
    't': 0.5,
    'pitch_hz': 220.0,  # A3
    'amplitude': 0.7,
    'duration': 0.6,
    'gesture_token': 28,
    'consonance': 0.85,
    'frequency_ratios': [1.0, 1.2, 1.5],
    'chord_name_display': 'minor'
}

SYNTHETIC_DISSONANT_EVENT = {
    't': 1.0,
    'pitch_hz': 300.0,
    'amplitude': 0.6,
    'duration': 0.3,
    'gesture_token': 15,
    'consonance': 0.2,
    'frequency_ratios': [1.0, 1.06, 1.12],
    'chord_name_display': 'unknown'
}

# === MODEL STRUCTURE EXPECTATIONS ===

EXPECTED_MODEL_STRUCTURE = {
    'required_keys': [
        'audio_frames',
        'states',
        'transitions',
        'suffix_links',
        'distance_threshold',
        'feature_dimensions',
        'distance_function',
    ],
    'audio_frame_required_fields': [
        # Fields that should exist in audio_data for most frames
        'pitch_hz',
        'gesture_token',
        'consonance',
    ],
    'audio_frame_optional_fields': [
        # Fields that may exist
        'chord_name_display',
        'dual_chroma',
        'transformer_insights',
        'frequency_ratios',
        'fundamental_freq',
    ]
}

# === TOLERANCE VALUES ===

TOLERANCES = {
    'frequency_hz': 1.0,          # ±1 Hz tolerance for frequency comparisons
    'midi_note': 0,               # Exact match for MIDI notes
    'consonance': 0.05,           # ±0.05 tolerance for consonance
    'timestamp': 0.01,            # ±10ms for timestamp comparisons
}
