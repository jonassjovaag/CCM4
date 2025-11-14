#!/usr/bin/env python3
"""
Quick test to verify MERT model is loading correctly
"""

import sys
import numpy as np
from musichal.core.config_manager import ConfigManager
from listener.dual_perception import DualPerceptionModule

def test_mert_loading():
    print("=" * 70)
    print("Testing MERT Model Loading")
    print("=" * 70)

    # Load config with quick_test profile
    config_manager = ConfigManager()
    config_manager.load(profile='quick_test')

    print(f"\nConfig loaded:")
    model_name = config_manager.get('feature_extraction.wav2vec.model', 'NOT FOUND')
    print(f"   Model in config: {model_name}")

    # Initialize DualPerceptionModule (same as feature_analysis_stage.py)
    print(f"\nInitializing Dual Perception Module...")
    print(f"   Passing model_name: {model_name}")

    analyzer = DualPerceptionModule(
        vocabulary_size=config_manager.get('symbolic_vocabulary_size', 64),
        wav2vec_model=model_name,
        use_gpu=config_manager.get('use_gpu', True),
        enable_symbolic=True
    )

    # Test encoding a simple audio signal
    print(f"\nTesting feature extraction...")
    sr = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))

    # Create a C major chord (C4 + E4 + G4)
    freq_C = 261.63
    freq_E = 329.63
    freq_G = 392.00

    audio = (
        np.sin(2 * np.pi * freq_C * t) +
        np.sin(2 * np.pi * freq_E * t) +
        np.sin(2 * np.pi * freq_G * t)
    ) / 3.0

    result = analyzer.extract_features(audio, sr, timestamp=0.0, detected_f0=freq_C)

    if result.wav2vec_features is not None and len(result.wav2vec_features) > 0:
        print(f"SUCCESS: Feature extraction successful!")
        print(f"   Feature dimension: {result.wav2vec_features.shape}")
        print(f"   Feature range: [{result.wav2vec_features.min():.3f}, {result.wav2vec_features.max():.3f}]")
        print(f"   Chord detected: {result.chord_label}")
        print("\n" + "=" * 70)
        print("MERT integration test PASSED!")
        print("=" * 70)
        return True
    else:
        print(f"FAILED: Feature extraction failed!")
        print("\n" + "=" * 70)
        print("MERT integration test FAILED!")
        print("=" * 70)
        return False


if __name__ == "__main__":
    success = test_mert_loading()
    sys.exit(0 if success else 1)
