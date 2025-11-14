#!/usr/bin/env python3
"""
Test CLAP Style Detection
=========================

Simple test script to verify CLAP is installed and working correctly.

Usage:
    python test_clap_detection.py
    python test_clap_detection.py --audio input_audio/your_file.wav
"""

import sys
import argparse
import numpy as np

# Fix Unicode encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

def test_clap_availability():
    """Test if CLAP is available"""
    print("=" * 70)
    print("Testing CLAP Availability")
    print("=" * 70)

    try:
        from listener.clap_style_detector import CLAPStyleDetector, is_clap_available

        if is_clap_available():
            print("✅ CLAP is available!")
            return True
        else:
            print("❌ CLAP not available (laion-clap not installed)")
            print("   Install: pip install laion-clap")
            return False
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def test_clap_detector_initialization():
    """Test CLAP detector initialization"""
    print("\n" + "=" * 70)
    print("Testing CLAP Detector Initialization")
    print("=" * 70)

    try:
        from listener.clap_style_detector import CLAPStyleDetector

        print("\n🔄 Initializing CLAP detector...")
        detector = CLAPStyleDetector(
            model_name="laion/clap-htsat-unfused",
            use_gpu=True,
            confidence_threshold=0.3
        )

        print("✅ CLAP detector initialized successfully!")
        return detector
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return None


def test_synthetic_audio(detector):
    """Test CLAP with synthetic audio"""
    print("\n" + "=" * 70)
    print("Testing CLAP with Synthetic Audio")
    print("=" * 70)

    sr = 44100
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration))

    # Test 1: Soft ballad-like audio (slow, quiet sine waves)
    print("\n🎵 Test 1: Soft ballad-like audio (quiet, slow)...")
    audio_ballad = (
        np.sin(2 * np.pi * 261.63 * t) * 0.3 +  # Soft C4
        np.sin(2 * np.pi * 329.63 * t) * 0.2    # Soft E4
    ).astype(np.float32)

    result = detector.detect_style(audio_ballad, sr)

    if result:
        print(f"✅ Detected: {result.style_label} (confidence: {result.confidence:.2f})")
        print(f"   Recommended mode: {result.recommended_mode.value}")
        print(f"   Secondary styles: {result.secondary_styles}")
    else:
        print("❌ No style detected (confidence too low)")

    # Test 2: Energetic rock-like audio (loud, distorted)
    print("\n🎵 Test 2: Energetic rock-like audio (loud, distorted)...")
    audio_rock = (
        np.sin(2 * np.pi * 130.81 * t) * 0.8 +  # Loud C3
        np.sin(2 * np.pi * 196.00 * t) * 0.7 +  # Loud G3
        np.random.randn(len(t)) * 0.1           # Noise (distortion)
    )
    audio_rock = np.clip(audio_rock, -1, 1).astype(np.float32)  # Clip (distortion)

    result = detector.detect_style(audio_rock, sr)

    if result:
        print(f"✅ Detected: {result.style_label} (confidence: {result.confidence:.2f})")
        print(f"   Recommended mode: {result.recommended_mode.value}")
        print(f"   Secondary styles: {result.secondary_styles}")
    else:
        print("❌ No style detected (confidence too low)")


def test_real_audio(detector, audio_path):
    """Test CLAP with real audio file"""
    print("\n" + "=" * 70)
    print(f"Testing CLAP with Real Audio: {audio_path}")
    print("=" * 70)

    try:
        import librosa

        # Load audio
        print(f"\n🔄 Loading audio file: {audio_path}")
        audio, sr = librosa.load(audio_path, sr=44100, mono=True, duration=5.0)

        print(f"✅ Loaded {len(audio) / sr:.2f}s of audio at {sr}Hz")

        # Detect style
        print("\n🔄 Detecting musical style...")
        result = detector.detect_style(audio, sr)

        if result:
            print(f"\n✅ Detection Results:")
            print(f"   Primary style: {result.style_label}")
            print(f"   Confidence: {result.confidence:.2f}")
            print(f"   Recommended mode: {result.recommended_mode.value}")
            print(f"\n   Secondary styles:")
            for style, conf in result.secondary_styles.items():
                print(f"      {style}: {conf:.2f}")
        else:
            print("❌ No style detected (confidence too low)")

    except FileNotFoundError:
        print(f"❌ Audio file not found: {audio_path}")
    except Exception as e:
        print(f"❌ Error testing real audio: {e}")


def main():
    parser = argparse.ArgumentParser(description="Test CLAP style detection")
    parser.add_argument("--audio", type=str, help="Path to audio file to test (optional)")
    args = parser.parse_args()

    # Step 1: Check CLAP availability
    if not test_clap_availability():
        sys.exit(1)

    # Step 2: Initialize detector
    detector = test_clap_detector_initialization()
    if not detector:
        sys.exit(1)

    # Step 3: Test with synthetic audio
    test_synthetic_audio(detector)

    # Step 4: Test with real audio (if provided)
    if args.audio:
        test_real_audio(detector, args.audio)

    print("\n" + "=" * 70)
    print("CLAP Testing Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
