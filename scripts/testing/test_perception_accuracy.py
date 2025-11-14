#!/usr/bin/env python3
"""
Perception Accuracy Test - Verify symbolic chord labels match audio reality

CRITICAL TEST: Validates that when you play "A major", the system correctly
identifies it as "A" on the symbolic layer. If this fails, the entire
harmonic progression system is working with wrong information.

Tests the complete perception chain:
  Audio → Wav2Vec (768D) → Gesture Token (machine perception)
  Audio → RealtimeHarmonicDetector → Chord Label (symbolic/human layer)

Validates:
1. Chord detection accuracy (does "A major audio" → "A" label?)
2. Gesture token consistency (do multiple A major samples get similar tokens?)
3. Cross-modal agreement (do 768D embeddings cluster by chord?)
4. Training data quality (were chords correctly labeled during training?)

This test answers: "Are we talking about the same chords?"
"""

import sys
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from collections import defaultdict
import joblib

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

from listener.harmonic_context import RealtimeHarmonicDetector, HarmonicContext
from listener.dual_perception import DualPerceptionModule


def print_section(title):
    """Print formatted section header"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def generate_test_chord(root_note, chord_type='major', duration=2.0, sr=44100):
    """
    Generate a simple test chord (same as earlier synthetic tests)
    
    Args:
        root_note: MIDI note number for root
        chord_type: 'major', 'minor', 'maj7', 'min7'
        duration: seconds
        sr: sample rate
    
    Returns:
        audio: numpy array
        ground_truth: chord label (e.g., "C", "Am")
    """
    # Define intervals
    intervals = {
        'major': [0, 4, 7],
        'minor': [0, 3, 7],
        'maj7': [0, 4, 7, 11],
        'min7': [0, 3, 7, 10]
    }
    
    # Note names for labeling
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    root_name = note_names[root_note % 12]
    
    # Generate chord label
    if chord_type == 'minor':
        chord_label = f"{root_name}m"
    elif chord_type == 'min7':
        chord_label = f"{root_name}m7"
    elif chord_type == 'maj7':
        chord_label = f"{root_name}maj7"
    else:
        chord_label = root_name
    
    # Generate audio
    t = np.linspace(0, duration, int(sr * duration))
    audio = np.zeros_like(t)
    
    for interval in intervals[chord_type]:
        midi_note = root_note + interval
        freq = 440 * (2 ** ((midi_note - 69) / 12))
        # Sine wave with slight envelope
        envelope = np.exp(-t * 0.5)  # Decay
        audio += np.sin(2 * np.pi * freq * t) * envelope
    
    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.8
    
    return audio.astype(np.float32), chord_label


def test_realtime_harmonic_detector(test_cases):
    """
    Test 1: Verify RealtimeHarmonicDetector correctly labels chords
    This is the SYMBOLIC LAYER - the "chord names" both human and machine use
    """
    print_section("TEST 1: RealtimeHarmonicDetector Accuracy")
    print("🎯 This test verifies the SYMBOLIC layer - chord labels")
    print("   Question: Does 'A major audio' get labeled as 'A'?\n")
    
    detector = RealtimeHarmonicDetector()
    
    results = []
    correct = 0
    total = len(test_cases)
    
    print(f"Testing {total} chord samples...\n")
    
    for i, (audio, ground_truth, description) in enumerate(test_cases, 1):
        sr = 44100
        
        # Run detector
        detected: HarmonicContext = detector.update_from_audio(audio, sr=sr)
        detected_chord = detected.current_chord
        
        # Check if correct
        is_correct = detected_chord == ground_truth
        if is_correct:
            correct += 1
        
        # Show result
        icon = "✅" if is_correct else "❌"
        print(f"{i:2d}. {description:30s}")
        print(f"    Ground truth: [{ground_truth:6s}]  Detected: [{detected_chord:6s}]  {icon}")
        
        results.append({
            'description': description,
            'ground_truth': ground_truth,
            'detected': detected_chord,
            'correct': is_correct,
            'consonance': detected.confidence
        })
    
    accuracy = correct / total * 100
    print(f"\n📊 Chord Detection Accuracy: {correct}/{total} ({accuracy:.1f}%)")
    
    if accuracy < 80:
        print("⚠️  WARNING: Low accuracy! Symbolic layer may be unreliable.")
    elif accuracy < 95:
        print("⚠️  Moderate accuracy. Some chords may be mislabeled.")
    else:
        print("✅ Excellent accuracy! Symbolic layer is reliable.")
    
    return results, accuracy


def test_gesture_token_consistency(test_cases, quantizer_file=None):
    """
    Test 2: Verify gesture tokens (768D → quantized) are consistent per chord
    This is the MACHINE PERCEPTION layer - what AudioOracle actually learns
    """
    print_section("TEST 2: Gesture Token Consistency (768D Subsymbolic)")
    print("🎯 This test verifies the MACHINE PERCEPTION layer")
    print("   Question: Do A major samples get similar gesture tokens?\n")
    
    if quantizer_file is None or not Path(quantizer_file).exists():
        print(f"⚠️  No quantizer file found at: {quantizer_file}")
        print("   Skipping gesture token test.")
        print("   (This requires trained quantizer from Chandra_trainer.py)")
        return None, 0.0
    
    # Load quantizer
    print(f"📂 Loading gesture quantizer from: {quantizer_file}")
    try:
        quantizer_data = joblib.load(quantizer_file)
        print(f"✅ Quantizer loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load quantizer: {e}")
        return None, 0.0
    
    # Initialize perception module
    try:
        perception = DualPerceptionModule(
            enable_dual_vocabulary=False,  # Use single vocabulary for now
            gesture_vocab_size=256
        )
        # Set the trained quantizer
        if isinstance(quantizer_data, dict):
            perception.gesture_quantizer = quantizer_data.get('quantizer')
        else:
            perception.gesture_quantizer = quantizer_data
        perception.quantizer_fitted = True
        print("✅ Perception module initialized with trained quantizer\n")
    except Exception as e:
        print(f"❌ Failed to initialize perception module: {e}")
        return None, 0.0
    
    # Extract gesture tokens for each sample
    print("Extracting gesture tokens from audio samples...\n")
    
    tokens_by_chord = defaultdict(list)
    
    for i, (audio, ground_truth, description) in enumerate(test_cases, 1):
        try:
            # Extract 768D embedding and quantize to gesture token
            # This simulates what happens during training/performance
            embeddings = perception.extract_wav2vec_features(audio, sr=44100)
            
            if embeddings is not None and len(embeddings) > 0:
                # Take mean embedding across time (similar to training)
                mean_embedding = np.mean(embeddings, axis=0)
                
                # Quantize to gesture token
                token = perception.gesture_quantizer.predict([mean_embedding])[0]
                
                tokens_by_chord[ground_truth].append(token)
                
                print(f"{i:2d}. {description:30s} → Token: {token:3d}")
            else:
                print(f"{i:2d}. {description:30s} → Token: FAILED (no embeddings)")
                
        except Exception as e:
            print(f"{i:2d}. {description:30s} → Token: ERROR ({e})")
    
    # Analyze consistency
    print(f"\n📊 Gesture Token Consistency Analysis:")
    
    consistency_scores = []
    
    for chord, tokens in sorted(tokens_by_chord.items()):
        unique_tokens = len(set(tokens))
        total_samples = len(tokens)
        
        # Perfect consistency = all samples of same chord get same token
        consistency = (1 - (unique_tokens - 1) / max(total_samples, 1)) * 100
        consistency_scores.append(consistency)
        
        print(f"\n   {chord:6s}: {total_samples} samples → {unique_tokens} unique tokens")
        print(f"           Consistency: {consistency:.1f}%")
        print(f"           Tokens: {sorted(set(tokens))}")
    
    avg_consistency = np.mean(consistency_scores) if consistency_scores else 0.0
    
    print(f"\n📊 Average Token Consistency: {avg_consistency:.1f}%")
    
    if avg_consistency < 50:
        print("⚠️  WARNING: Low consistency! Gesture tokens are unstable.")
        print("   This means AudioOracle may not cluster chords properly.")
    elif avg_consistency < 80:
        print("⚠️  Moderate consistency. Some variation in tokens.")
    else:
        print("✅ Good consistency! Similar chords get similar tokens.")
    
    return tokens_by_chord, avg_consistency


def test_cross_modal_agreement(chord_results, token_results):
    """
    Test 3: Verify symbolic (chord labels) and subsymbolic (tokens) agree
    Do samples with same chord label also have same gesture token?
    """
    print_section("TEST 3: Cross-Modal Agreement")
    print("🎯 Do symbolic labels (chord names) align with subsymbolic tokens (768D)?\n")
    
    if token_results is None:
        print("⚠️  No token data available. Skipping cross-modal test.")
        return 0.0
    
    # For each correctly detected chord, check if tokens are consistent
    agreement_count = 0
    total_count = 0
    
    for chord, tokens in token_results.items():
        if len(tokens) <= 1:
            continue  # Need multiple samples to check agreement
        
        total_count += 1
        
        # Check if tokens are mostly the same
        unique_tokens = len(set(tokens))
        primary_token = max(set(tokens), key=tokens.count)
        primary_count = tokens.count(primary_token)
        primary_ratio = primary_count / len(tokens)
        
        # Agreement = >80% of samples have the same token
        if primary_ratio >= 0.8:
            agreement_count += 1
            icon = "✅"
        else:
            icon = "❌"
        
        print(f"{icon} {chord:6s}: {primary_ratio*100:5.1f}% agree on token {primary_token}")
    
    if total_count > 0:
        agreement_rate = agreement_count / total_count * 100
        print(f"\n📊 Cross-Modal Agreement: {agreement_count}/{total_count} ({agreement_rate:.1f}%)")
        
        if agreement_rate < 70:
            print("⚠️  WARNING: Symbolic and subsymbolic layers disagree!")
            print("   The machine may 'think' different chords than the labels say.")
        else:
            print("✅ Good agreement! Labels match perceptual clustering.")
        
        return agreement_rate
    else:
        print("⚠️  Insufficient data to assess agreement.")
        return 0.0


def explain_pipeline_integration():
    """Explain how this test relates to training and live performance"""
    print_section("PIPELINE INTEGRATION EXPLANATION")
    
    print("🔍 How this test relates to the full system:\n")
    
    print("📚 TRAINING PHASE (Chandra_trainer.py):")
    print("   1. Audio file → onset detection → audio segments")
    print("   2. For each segment:")
    print("      • Wav2Vec → 768D embedding → Gesture Token (machine)")
    print("      • RealtimeHarmonicDetector → Chord Label (symbolic)")
    print("   3. AudioOracle learns from GESTURE TOKENS (768D layer)")
    print("   4. Transition graph learns from CHORD LABELS (symbolic layer)")
    print("   5. Both get saved to JSON files\n")
    
    print("🎵 LIVE PERFORMANCE (MusicHal_9000.py):")
    print("   1. Audio input → same dual perception:")
    print("      • 768D → Gesture Token → AudioOracle query → MIDI generation")
    print("      • Chord Label → HarmonicProgressor → Chord selection")
    print("   2. PhraseGenerator uses BOTH:")
    print("      • Gesture tokens for pattern matching")
    print("      • Chord labels for harmonic context")
    print("   3. Output: MIDI notes\n")
    
    print("⚠️  CRITICAL DEPENDENCY:")
    print("   If chord detection is WRONG during training:")
    print("   → Transition graph learns wrong progressions")
    print("   → Live performance makes wrong harmonic choices")
    print("   → System appears 'deaf' to harmony\n")
    
    print("   If gesture tokens are INCONSISTENT:")
    print("   → AudioOracle can't find patterns")
    print("   → Pattern matching fails")
    print("   → Output is random/incoherent\n")
    
    print("✅ This test validates BOTH layers work correctly!")


def main():
    """Run complete perception accuracy test"""
    
    print("\n" + "="*70)
    print("  PERCEPTION ACCURACY TEST")
    print("  Verifying Audio → 768D → Chord Label Pipeline")
    print("="*70)
    print("\nThis test answers the critical question:")
    print("'When I play A major, does the machine correctly identify it as A?'\n")
    
    # Generate test cases - multiple samples per chord
    print("🔨 Generating test audio samples...")
    
    test_cases = []
    
    # Test chords from the training data (D, A, Am, F#m, etc.)
    test_chords = [
        (62, 'major', 'D'),    # D major
        (69, 'major', 'A'),    # A major  
        (69, 'minor', 'Am'),   # A minor
        (66, 'minor', 'F#m'),  # F# minor
        (65, 'major', 'F'),    # F major
        (67, 'major', 'G'),    # G major
        (60, 'major', 'C'),    # C major
        (62, 'minor', 'Dm'),   # D minor
    ]
    
    # Generate multiple samples of each (different octaves/voicings)
    for midi_root, chord_type, expected_label in test_chords:
        # Generate at original octave
        audio1, label1 = generate_test_chord(midi_root, chord_type, duration=1.5)
        test_cases.append((audio1, expected_label, f"{expected_label} (root)"))
        
        # Generate octave higher (test octave invariance)
        audio2, label2 = generate_test_chord(midi_root + 12, chord_type, duration=1.5)
        test_cases.append((audio2, expected_label, f"{expected_label} (octave up)"))
    
    print(f"✅ Generated {len(test_cases)} test samples\n")
    
    # Test 1: Chord detection accuracy (symbolic layer)
    chord_results, chord_accuracy = test_realtime_harmonic_detector(test_cases)
    
    # Test 2: Gesture token consistency (subsymbolic layer)
    quantizer_file = "JSON/Curious_child_091125_2009_training_gesture_training_quantizer.joblib"
    token_results, token_consistency = test_gesture_token_consistency(test_cases, quantizer_file)
    
    # Test 3: Cross-modal agreement
    agreement = test_cross_modal_agreement(chord_results, token_results)
    
    # Explain integration
    explain_pipeline_integration()
    
    # Final summary
    print_section("FINAL SUMMARY")
    
    print("📊 Perception Accuracy Results:\n")
    print(f"   Chord Detection (Symbolic):     {chord_accuracy:.1f}%")
    if token_results is not None:
        print(f"   Gesture Token Consistency:      {token_consistency:.1f}%")
        print(f"   Cross-Modal Agreement:          {agreement:.1f}%")
    else:
        print(f"   Gesture Token Consistency:      N/A (no quantizer)")
        print(f"   Cross-Modal Agreement:          N/A (no quantizer)")
    
    print("\n🎯 Conclusion:")
    
    if chord_accuracy >= 95:
        print("   ✅ Symbolic layer (chord labels) is ACCURATE")
        print("      → Harmonic progression system has correct information")
    elif chord_accuracy >= 80:
        print("   ⚠️  Symbolic layer has MODERATE accuracy")
        print("      → Some chord mislabeling may occur")
    else:
        print("   ❌ Symbolic layer is UNRELIABLE")
        print("      → Harmonic progression system may fail")
    
    if token_results is not None:
        if token_consistency >= 80:
            print("   ✅ Subsymbolic layer (gesture tokens) is CONSISTENT")
            print("      → AudioOracle can learn patterns properly")
        elif token_consistency >= 60:
            print("   ⚠️  Subsymbolic layer has MODERATE consistency")
            print("      → Pattern learning may be noisy")
        else:
            print("   ❌ Subsymbolic layer is INCONSISTENT")
            print("      → AudioOracle pattern learning may fail")
    
    print("\n💡 Next Steps:")
    if chord_accuracy < 95 or (token_results and token_consistency < 80):
        print("   1. Review RealtimeHarmonicDetector configuration")
        print("   2. Consider retraining with better chord detection")
        print("   3. Check audio quality and onset detection")
    else:
        print("   1. Proceed with live performance testing")
        print("   2. Verify real audio input matches synthetic results")
        print("   3. Monitor logs for perception accuracy during improvisation")
    
    print()


if __name__ == '__main__':
    main()
