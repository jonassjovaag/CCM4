"""
Test utilities for MusicHal 9000 test suite
These are ONLY used by tests, never by production code

Purpose: Provide reusable functions for:
- Loading models (read-only)
- Statistical analysis
- Validation assertions
- Test data generation
"""

import sys
import gzip
import pickle
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

# Add project root to path so we can import production modules
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def get_test_root() -> Path:
    """Get absolute path to tests/ directory"""
    return Path(__file__).parent.parent


def get_project_root() -> Path:
    """Get absolute path to project root"""
    return Path(__file__).parent.parent.parent


def load_trained_model(model_path: Optional[Path] = None) -> Dict:
    """
    Load a trained AudioOracle model from pickle/gzip file
    
    Args:
        model_path: Path to model file. If None, loads reference model.
    
    Returns:
        Dict containing model data (serialized AudioOracle state)
    
    Safety: Read-only operation, never modifies the model file
    """
    if model_path is None:
        # Use reference model from JSON/ directory
        project_root = get_project_root()
        model_path = project_root / "JSON" / "Itzama_071125_2130_training_model.pkl.gz"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    with gzip.open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    return model_data


def extract_audio_frames(model_data: Dict) -> Dict:
    """
    Extract audio_frames from model data
    
    Args:
        model_data: Loaded model dict
    
    Returns:
        Dict[int, AudioFrame] or Dict[int, dict] - the audio_frames structure
    """
    if isinstance(model_data, dict):
        return model_data.get('audio_frames', {})
    else:
        # If it's an AudioOracle object (shouldn't be, but handle gracefully)
        return getattr(model_data, 'audio_frames', {})


def compute_model_statistics(model_data: Dict) -> Dict[str, Any]:
    """
    Extract comprehensive statistics from loaded model
    
    Args:
        model_data: Loaded model dict
    
    Returns:
        Dict with statistical measures:
        - frame_count: Number of audio frames
        - unique_gesture_tokens: Number of unique gesture tokens
        - consonance_stats: {mean, std, min, max}
        - pitch_hz_stats: {mean, std, min, max}
        - chord_name_frequencies: Counter of chord_name_display values
        - state_count: Number of oracle states
        - transition_count: Number of oracle transitions
        - suffix_link_count: Number of suffix links
    """
    audio_frames = extract_audio_frames(model_data)
    
    stats = {
        'frame_count': len(audio_frames),
        'state_count': len(model_data.get('states', {})),
        'transition_count': len(model_data.get('transitions', {})),  # transitions are dict of tuples
        'suffix_link_count': len(model_data.get('suffix_links', {})),
    }
    
    # Collect features from all frames
    gesture_tokens = []
    consonances = []
    pitch_hzs = []
    chord_names = []
    
    for frame_id, frame in audio_frames.items():
        # Handle both dict and AudioFrame object formats
        if isinstance(frame, dict):
            audio_data = frame.get('audio_data', frame)
        else:
            audio_data = getattr(frame, 'audio_data', {})
        
        # Collect gesture tokens
        if 'gesture_token' in audio_data and audio_data['gesture_token'] is not None:
            gesture_tokens.append(audio_data['gesture_token'])
        
        # Collect consonance values
        if 'consonance' in audio_data and audio_data['consonance'] is not None:
            consonances.append(float(audio_data['consonance']))
        
        # Collect pitch_hz
        if 'pitch_hz' in audio_data and audio_data['pitch_hz'] is not None:
            pitch_hzs.append(float(audio_data['pitch_hz']))
        
        # Collect chord names
        if 'chord_name_display' in audio_data:
            chord_names.append(audio_data['chord_name_display'])
    
    # Compute statistics
    if gesture_tokens:
        stats['unique_gesture_tokens'] = len(set(gesture_tokens))
        stats['gesture_token_range'] = (min(gesture_tokens), max(gesture_tokens))
        stats['gesture_token_distribution'] = dict(zip(*np.unique(gesture_tokens, return_counts=True)))
    else:
        stats['unique_gesture_tokens'] = 0
        stats['gesture_token_range'] = (None, None)
        stats['gesture_token_distribution'] = {}
    
    if consonances:
        stats['consonance_stats'] = {
            'mean': float(np.mean(consonances)),
            'std': float(np.std(consonances)),
            'min': float(np.min(consonances)),
            'max': float(np.max(consonances)),
            'median': float(np.median(consonances)),
        }
    else:
        stats['consonance_stats'] = {}
    
    if pitch_hzs:
        stats['pitch_hz_stats'] = {
            'mean': float(np.mean(pitch_hzs)),
            'std': float(np.std(pitch_hzs)),
            'min': float(np.min(pitch_hzs)),
            'max': float(np.max(pitch_hzs)),
            'median': float(np.median(pitch_hzs)),
        }
    else:
        stats['pitch_hz_stats'] = {}
    
    if chord_names:
        from collections import Counter
        stats['chord_name_frequencies'] = dict(Counter(chord_names).most_common(20))
    else:
        stats['chord_name_frequencies'] = {}
    
    return stats


def validate_feature_range(value: float, feature_name: str, expected_min: float, expected_max: float) -> Tuple[bool, str]:
    """
    Validate that a feature value is within expected range
    
    Args:
        value: Feature value to validate
        feature_name: Name of feature (for error message)
        expected_min: Minimum expected value
        expected_max: Maximum expected value
    
    Returns:
        (is_valid, error_message) tuple
    """
    if value < expected_min or value > expected_max:
        return False, f"{feature_name} out of range: {value} (expected {expected_min}-{expected_max})"
    return True, ""


def frequency_to_midi(freq_hz: float) -> int:
    """
    Convert frequency in Hz to MIDI note number
    
    Args:
        freq_hz: Frequency in Hz
    
    Returns:
        MIDI note number (0-127)
    
    Formula: MIDI = 69 + 12 * log2(freq / 440)
    """
    import math
    if freq_hz <= 0:
        return 0
    midi = 69 + 12 * math.log2(freq_hz / 440.0)
    return int(round(np.clip(midi, 0, 127)))


def midi_to_frequency(midi_note: int) -> float:
    """
    Convert MIDI note number to frequency in Hz
    
    Args:
        midi_note: MIDI note number (0-127)
    
    Returns:
        Frequency in Hz
    
    Formula: freq = 440 * 2^((MIDI - 69) / 12)
    """
    return 440.0 * (2 ** ((midi_note - 69) / 12))


def assert_model_structure(model_data: Dict):
    """
    Assert that model has required structure
    
    Raises AssertionError with descriptive message if structure is invalid
    """
    required_keys = ['audio_frames', 'states', 'transitions', 'distance_threshold', 'feature_dimensions']
    
    for key in required_keys:
        assert key in model_data, f"Model missing required key: {key}"
    
    assert isinstance(model_data['audio_frames'], dict), "audio_frames must be dict"
    assert len(model_data['audio_frames']) > 0, "audio_frames is empty"
    
    # Validate feature_dimensions
    assert isinstance(model_data['feature_dimensions'], int), "feature_dimensions must be int"
    assert model_data['feature_dimensions'] > 0, "feature_dimensions must be positive"


def save_test_results(test_name: str, results: Dict[str, Any]):
    """
    Save test results to JSON file in tests/test_output/
    
    Args:
        test_name: Name of test (becomes filename)
        results: Dictionary of results to save
    
    Safety: Creates files only in tests/test_output/, never touches production
    """
    import numpy as np
    
    def convert_numpy_types(obj):
        """Convert NumPy types to Python native types for JSON serialization"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {convert_numpy_types(k): convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        return obj
    
    output_dir = get_test_root() / "test_output"
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / f"{test_name}_results.json"
    
    # Convert NumPy types before saving
    results_clean = convert_numpy_types(results)
    
    with open(output_file, 'w') as f:
        json.dump(results_clean, f, indent=2, default=str)
    
    print(f"📊 Test results saved: {output_file}")


def load_test_results(test_name: str) -> Optional[Dict[str, Any]]:
    """
    Load previously saved test results
    
    Args:
        test_name: Name of test
    
    Returns:
        Dict of results, or None if file doesn't exist
    """
    output_dir = get_test_root() / "test_output"
    output_file = output_dir / f"{test_name}_results.json"
    
    if not output_file.exists():
        return None
    
    with open(output_file, 'r') as f:
        return json.load(f)


def print_statistics_summary(stats: Dict[str, Any]):
    """
    Pretty-print statistics dictionary
    
    Args:
        stats: Statistics dict from compute_model_statistics()
    """
    print("\n" + "="*80)
    print("MODEL STATISTICS SUMMARY")
    print("="*80)
    
    print(f"\n📊 Structure:")
    print(f"   Frames: {stats['frame_count']}")
    print(f"   States: {stats['state_count']}")
    print(f"   Transitions: {stats['transition_count']}")
    print(f"   Suffix links: {stats['suffix_link_count']}")
    
    if stats['unique_gesture_tokens'] > 0:
        print(f"\n🎵 Gesture Tokens:")
        print(f"   Unique: {stats['unique_gesture_tokens']}")
        print(f"   Range: {stats['gesture_token_range']}")
        print(f"   Top 10: {list(stats['gesture_token_distribution'].items())[:10]}")
    
    if stats['consonance_stats']:
        cs = stats['consonance_stats']
        print(f"\n🎼 Consonance:")
        print(f"   Mean: {cs['mean']:.3f} ± {cs['std']:.3f}")
        print(f"   Range: [{cs['min']:.3f}, {cs['max']:.3f}]")
        print(f"   Median: {cs['median']:.3f}")
    
    if stats['pitch_hz_stats']:
        ps = stats['pitch_hz_stats']
        print(f"\n🎹 Pitch (Hz):")
        print(f"   Mean: {ps['mean']:.1f} ± {ps['std']:.1f}")
        print(f"   Range: [{ps['min']:.1f}, {ps['max']:.1f}]")
        print(f"   Median: {ps['median']:.1f}")
    
    if stats['chord_name_frequencies']:
        print(f"\n🎸 Chord Names (top 10):")
        for chord, count in list(stats['chord_name_frequencies'].items())[:10]:
            print(f"   {chord}: {count}")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    """
    Test the test helpers themselves
    """
    print("Testing test_helpers.py...")
    
    # Test loading model
    print("\n1. Testing model loading...")
    model = load_trained_model()
    print(f"   ✅ Loaded model with {len(model.get('audio_frames', {}))} frames")
    
    # Test statistics computation
    print("\n2. Testing statistics computation...")
    stats = compute_model_statistics(model)
    print_statistics_summary(stats)
    
    # Test frequency conversion
    print("\n3. Testing frequency/MIDI conversion...")
    test_cases = [
        (440.0, 69),  # A4
        (220.0, 57),  # A3
        (880.0, 81),  # A5
    ]
    for freq, expected_midi in test_cases:
        midi = frequency_to_midi(freq)
        back_to_freq = midi_to_frequency(midi)
        print(f"   {freq} Hz → MIDI {midi} (expected {expected_midi}) → {back_to_freq:.1f} Hz")
        assert midi == expected_midi, f"MIDI conversion failed for {freq} Hz"
    
    # Test saving results
    print("\n4. Testing results saving...")
    save_test_results('test_helpers_selftest', stats)
    loaded = load_test_results('test_helpers_selftest')
    assert loaded is not None, "Failed to load saved results"
    print(f"   ✅ Successfully saved and loaded results")
    
    print("\n✅ All test_helpers tests passed!")
