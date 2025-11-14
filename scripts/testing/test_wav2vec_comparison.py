#!/usr/bin/env python3
"""
Test and Compare Wav2Vec vs Ratio+Chroma Performance
======================================================

This script compares the performance of:
1. Traditional ratio-based + chroma features
2. Wav2Vec 2.0 neural encoding

Metrics:
- Feature extraction speed
- Feature dimension
- Chord detection accuracy (if ground truth available)
- Memory usage
- Real-time capability

Usage:
    # Test with audio file
    python test_wav2vec_comparison.py --file input_audio/test.wav
    
    # Test with GPU acceleration
    python test_wav2vec_comparison.py --file input_audio/test.wav --gpu
    
    # Test with different Wav2Vec model
    python test_wav2vec_comparison.py --file input_audio/test.wav --model facebook/wav2vec2-large
"""

import time
import numpy as np
import argparse
import librosa
from typing import Dict, List, Tuple
import os
import json
from datetime import datetime

from listener.hybrid_perception import HybridPerceptionModule
from listener.wav2vec_perception import is_wav2vec_available


def load_test_audio(audio_path: str, duration: float = 30.0) -> Tuple[np.ndarray, int]:
    """
    Load test audio file
    
    Args:
        audio_path: Path to audio file
        duration: Duration to load (seconds)
        
    Returns:
        (audio, sample_rate)
    """
    print(f"📂 Loading audio: {audio_path}")
    audio, sr = librosa.load(audio_path, sr=44100, duration=duration)
    print(f"   Duration: {len(audio)/sr:.2f}s, Sample rate: {sr}Hz")
    return audio, sr


def test_feature_extraction_speed(perception_module: HybridPerceptionModule,
                                  audio: np.ndarray, sr: int,
                                  num_segments: int = 100) -> Dict:
    """
    Test feature extraction speed on multiple segments
    
    Args:
        perception_module: Perception module to test
        audio: Audio signal
        sr: Sample rate
        num_segments: Number of segments to test
        
    Returns:
        Performance metrics
    """
    segment_duration = 0.5  # 500ms segments
    segment_samples = int(segment_duration * sr)
    
    extraction_times = []
    features_list = []
    
    print(f"⏱️  Testing extraction speed on {num_segments} segments...")
    
    for i in range(num_segments):
        # Get random segment
        if len(audio) > segment_samples:
            start = np.random.randint(0, len(audio) - segment_samples)
            segment = audio[start:start + segment_samples]
        else:
            segment = audio
        
        # Time feature extraction
        start_time = time.time()
        result = perception_module.extract_features(segment, sr, timestamp=float(i))
        extraction_time = time.time() - start_time
        
        if result is not None:
            extraction_times.append(extraction_time)
            features_list.append(result)
        
        # Progress
        if (i + 1) % 20 == 0:
            print(f"   Processed {i + 1}/{num_segments} segments")
    
    # Calculate statistics
    avg_time = np.mean(extraction_times)
    std_time = np.std(extraction_times)
    min_time = np.min(extraction_times)
    max_time = np.max(extraction_times)
    
    # Real-time factor (how many times real-time)
    rt_factor = segment_duration / avg_time
    
    return {
        'avg_extraction_time': avg_time,
        'std_extraction_time': std_time,
        'min_extraction_time': min_time,
        'max_extraction_time': max_time,
        'realtime_factor': rt_factor,
        'features_extracted': len(features_list),
        'feature_dimension': features_list[0].features.shape[0] if features_list else 0
    }


def test_feature_quality(perception_module: HybridPerceptionModule,
                        audio: np.ndarray, sr: int,
                        module_name: str) -> Dict:
    """
    Test feature quality and characteristics
    
    Args:
        perception_module: Perception module to test
        audio: Audio signal
        sr: Sample rate
        module_name: Name for logging
        
    Returns:
        Quality metrics
    """
    segment_duration = 1.0  # 1 second segments
    segment_samples = int(segment_duration * sr)
    
    print(f"🔍 Analyzing feature quality for {module_name}...")
    
    # Extract features from multiple segments
    num_segments = min(30, len(audio) // segment_samples)
    features_list = []
    consonance_scores = []
    
    for i in range(num_segments):
        start = i * segment_samples
        end = start + segment_samples
        if end > len(audio):
            break
        
        segment = audio[start:end]
        result = perception_module.extract_features(segment, sr, timestamp=float(i))
        
        if result is not None:
            features_list.append(result.features)
            consonance_scores.append(result.consonance)
    
    # Calculate feature statistics
    features_array = np.array(features_list)
    
    # Feature variance (how much features vary across segments)
    feature_variance = np.var(features_array, axis=0).mean()
    
    # Feature range
    feature_range = (features_array.max() - features_array.min())
    
    # Feature sparsity (% of near-zero values)
    sparsity = (np.abs(features_array) < 0.01).sum() / features_array.size
    
    # Consonance statistics
    avg_consonance = np.mean(consonance_scores)
    std_consonance = np.std(consonance_scores)
    
    return {
        'num_segments_analyzed': num_segments,
        'feature_variance': float(feature_variance),
        'feature_range': float(feature_range),
        'feature_sparsity': float(sparsity),
        'avg_consonance': float(avg_consonance),
        'std_consonance': float(std_consonance),
        'consonance_range': (float(np.min(consonance_scores)), float(np.max(consonance_scores)))
    }


def compare_features(ratio_module: HybridPerceptionModule,
                    wav2vec_module: HybridPerceptionModule,
                    audio: np.ndarray, sr: int) -> Dict:
    """
    Compare features from both modules on same audio
    
    Args:
        ratio_module: Ratio-based perception module
        wav2vec_module: Wav2Vec perception module
        audio: Audio signal
        sr: Sample rate
        
    Returns:
        Comparison metrics
    """
    print(f"📊 Comparing features on same audio segment...")
    
    # Use a 2-second segment
    segment_samples = int(2.0 * sr)
    segment = audio[:segment_samples] if len(audio) >= segment_samples else audio
    
    # Extract features from both
    ratio_result = ratio_module.extract_features(segment, sr, timestamp=0.0)
    wav2vec_result = wav2vec_module.extract_features(segment, sr, timestamp=0.0)
    
    if ratio_result is None or wav2vec_result is None:
        return {'error': 'Feature extraction failed'}
    
    # Compare consonance predictions
    consonance_diff = abs(ratio_result.consonance - wav2vec_result.consonance)
    
    # Compare active pitch classes (if available)
    ratio_pcs = set(ratio_result.active_pitch_classes.tolist())
    wav2vec_pcs = set(wav2vec_result.active_pitch_classes.tolist())
    
    pitch_class_overlap = len(ratio_pcs & wav2vec_pcs) / max(len(ratio_pcs | wav2vec_pcs), 1)
    
    return {
        'consonance_difference': float(consonance_diff),
        'ratio_consonance': float(ratio_result.consonance),
        'wav2vec_consonance': float(wav2vec_result.consonance),
        'pitch_class_overlap': float(pitch_class_overlap),
        'ratio_pitch_classes': list(ratio_pcs),
        'wav2vec_pitch_classes': list(wav2vec_pcs),
        'ratio_feature_dim': ratio_result.features.shape[0],
        'wav2vec_feature_dim': wav2vec_result.features.shape[0]
    }


def print_results(results: Dict):
    """Pretty print comparison results"""
    print("\n" + "="*70)
    print("📊 WAV2VEC vs RATIO+CHROMA COMPARISON RESULTS")
    print("="*70)
    
    # Speed comparison
    print("\n⏱️  SPEED COMPARISON:")
    print("-" * 70)
    
    ratio_speed = results['ratio_speed']
    wav2vec_speed = results['wav2vec_speed']
    
    print(f"Ratio+Chroma:")
    print(f"  Avg extraction time: {ratio_speed['avg_extraction_time']*1000:.2f}ms")
    print(f"  Real-time factor: {ratio_speed['realtime_factor']:.1f}x")
    print(f"  Feature dimension: {ratio_speed['feature_dimension']}D")
    
    print(f"\nWav2Vec:")
    print(f"  Avg extraction time: {wav2vec_speed['avg_extraction_time']*1000:.2f}ms")
    print(f"  Real-time factor: {wav2vec_speed['realtime_factor']:.1f}x")
    print(f"  Feature dimension: {wav2vec_speed['feature_dimension']}D")
    
    speedup = ratio_speed['avg_extraction_time'] / wav2vec_speed['avg_extraction_time']
    if speedup > 1:
        print(f"\n✅ Ratio+Chroma is {speedup:.1f}x FASTER")
    else:
        print(f"\n✅ Wav2Vec is {1/speedup:.1f}x FASTER")
    
    # Quality comparison
    print("\n🔍 FEATURE QUALITY:")
    print("-" * 70)
    
    ratio_quality = results['ratio_quality']
    wav2vec_quality = results['wav2vec_quality']
    
    print(f"Ratio+Chroma:")
    print(f"  Feature variance: {ratio_quality['feature_variance']:.4f}")
    print(f"  Feature sparsity: {ratio_quality['feature_sparsity']:.1%}")
    print(f"  Avg consonance: {ratio_quality['avg_consonance']:.3f} ± {ratio_quality['std_consonance']:.3f}")
    
    print(f"\nWav2Vec:")
    print(f"  Feature variance: {wav2vec_quality['feature_variance']:.4f}")
    print(f"  Feature sparsity: {wav2vec_quality['feature_sparsity']:.1%}")
    print(f"  Avg consonance: {wav2vec_quality['avg_consonance']:.3f} ± {wav2vec_quality['std_consonance']:.3f}")
    
    # Direct comparison
    if 'comparison' in results:
        comp = results['comparison']
        print("\n📊 DIRECT COMPARISON (same audio):")
        print("-" * 70)
        print(f"  Consonance difference: {comp['consonance_difference']:.3f}")
        print(f"  Pitch class overlap: {comp['pitch_class_overlap']:.1%}")
        print(f"  Ratio PCs: {comp['ratio_pitch_classes']}")
        print(f"  Wav2Vec PCs: {comp['wav2vec_pitch_classes']}")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    print("-" * 70)
    
    if ratio_speed['realtime_factor'] > 50:
        print("✅ Ratio+Chroma: Excellent for real-time applications")
    
    if wav2vec_speed['realtime_factor'] > 10:
        print("✅ Wav2Vec: Suitable for real-time with GPU acceleration")
    else:
        print("⚠️  Wav2Vec: May struggle in real-time without GPU")
    
    if wav2vec_quality['feature_variance'] > ratio_quality['feature_variance'] * 2:
        print("✅ Wav2Vec: Richer feature representation (higher variance)")
    
    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(description='Compare Wav2Vec vs Ratio+Chroma Performance')
    parser.add_argument('--file', required=True, help='Audio file to test')
    parser.add_argument('--duration', type=float, default=30.0, help='Duration to analyze (seconds)')
    parser.add_argument('--segments', type=int, default=100, help='Number of segments to test')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for Wav2Vec')
    parser.add_argument('--model', type=str, default='facebook/wav2vec2-base',
                       help='Wav2Vec model name')
    parser.add_argument('--output', type=str, help='Output JSON file for results')
    
    args = parser.parse_args()
    
    # Check Wav2Vec availability
    if not is_wav2vec_available():
        print("❌ Wav2Vec not available!")
        print("   Install: pip install transformers torch")
        return 1
    
    # Load audio
    audio, sr = load_test_audio(args.file, args.duration)
    
    print("\n" + "="*70)
    print("🧪 INITIALIZING PERCEPTION MODULES")
    print("="*70)
    
    # Initialize ratio-based module
    print("\n1️⃣  Initializing Ratio+Chroma module...")
    ratio_module = HybridPerceptionModule(
        vocabulary_size=64,
        enable_ratio_analysis=True,
        enable_symbolic=False,
        enable_wav2vec=False
    )
    
    # Initialize Wav2Vec module
    print("\n2️⃣  Initializing Wav2Vec module...")
    wav2vec_module = HybridPerceptionModule(
        vocabulary_size=64,
        enable_ratio_analysis=False,
        enable_symbolic=False,
        enable_wav2vec=True,
        wav2vec_model=args.model,
        use_gpu=args.gpu
    )
    
    # Run tests
    results = {}
    
    print("\n" + "="*70)
    print("🧪 RUNNING PERFORMANCE TESTS")
    print("="*70)
    
    # Test 1: Speed comparison
    print("\n📊 Test 1: Extraction Speed")
    results['ratio_speed'] = test_feature_extraction_speed(
        ratio_module, audio, sr, args.segments
    )
    
    print("\n📊 Test 2: Wav2Vec Extraction Speed")
    results['wav2vec_speed'] = test_feature_extraction_speed(
        wav2vec_module, audio, sr, args.segments
    )
    
    # Test 2: Feature quality
    print("\n📊 Test 3: Feature Quality Analysis")
    results['ratio_quality'] = test_feature_quality(
        ratio_module, audio, sr, "Ratio+Chroma"
    )
    
    print("\n📊 Test 4: Wav2Vec Quality Analysis")
    results['wav2vec_quality'] = test_feature_quality(
        wav2vec_module, audio, sr, "Wav2Vec"
    )
    
    # Test 3: Direct comparison
    print("\n📊 Test 5: Direct Feature Comparison")
    results['comparison'] = compare_features(
        ratio_module, wav2vec_module, audio, sr
    )
    
    # Print results
    print_results(results)
    
    # Save results
    if args.output:
        results['metadata'] = {
            'audio_file': args.file,
            'duration': args.duration,
            'num_segments': args.segments,
            'wav2vec_model': args.model,
            'gpu_enabled': args.gpu,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {args.output}")
    
    return 0


if __name__ == "__main__":
    exit(main())





























