#!/usr/bin/env python3
"""
Quick Gesture Token Analysis for 24-bit 48kHz audio files
=========================================================

Analyzes why Nineteen.wav and other complex audio files might produce
limited gesture token diversity despite having rich musical content.

Focus areas:
1. Sample rate conversion effects (48kHz → 16kHz for Wav2Vec)
2. Bit depth normalization issues
3. Segment length and overlap effects
4. K-means clustering quality
"""

import numpy as np
import librosa
import sys
import os
from collections import Counter

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from listener.wav2vec_perception import Wav2VecMusicEncoder
from listener.symbolic_quantizer import SymbolicQuantizer


def quick_gesture_analysis(audio_file: str):
    """Quick analysis of gesture token diversity"""
    
    print(f"🎵 Analyzing: {audio_file}")
    print("=" * 50)
    
    # Load audio with original specs
    print("📼 Loading audio...")
    audio_orig, sr_orig = librosa.load(audio_file, sr=None)
    print(f"   Original: {sr_orig} Hz, {audio_orig.dtype}, {len(audio_orig)/sr_orig:.1f}s")
    print(f"   Dynamic range: {audio_orig.min():.3f} to {audio_orig.max():.3f}")
    print(f"   RMS level: {np.sqrt(np.mean(audio_orig**2)):.3f}")
    
    # Also load as 44.1kHz (standard)
    audio_44k, _ = librosa.load(audio_file, sr=44100)
    
    # Check for clipping or silence
    if np.abs(audio_orig).max() < 0.01:
        print("⚠️  Audio seems very quiet - might affect feature extraction")
    if np.abs(audio_orig).max() > 0.99:
        print("⚠️  Audio might be clipped")
    
    # Initialize Wav2Vec encoder
    print("\n🧠 Initializing Wav2Vec encoder...")
    wav2vec = Wav2VecMusicEncoder("facebook/wav2vec2-base", use_gpu=False)
    
    # Extract features from different segment lengths
    segment_lengths = [1.0, 2.0, 4.0]  # Test different segment sizes
    
    for seg_len in segment_lengths:
        print(f"\n🎯 Testing {seg_len}s segments...")
        
        # Extract segments
        hop_size = seg_len / 2  # 50% overlap
        segments = []
        
        for i in range(0, int((len(audio_orig)/sr_orig - seg_len) / hop_size)):
            start_time = i * hop_size
            start_sample = int(start_time * sr_orig)
            end_sample = int((start_time + seg_len) * sr_orig)
            
            segment = audio_orig[start_sample:end_sample]
            segments.append(segment)
            
            if len(segments) >= 100:  # Limit for quick test
                break
        
        print(f"   Extracted {len(segments)} segments")
        
        # Extract Wav2Vec features
        features = []
        for i, segment in enumerate(segments):
            if i % 20 == 0:
                print(f"   Processing {i+1}/{len(segments)}...")
            
            result = wav2vec.encode(segment, sr=sr_orig, timestamp=i*hop_size)
            if result is not None:
                features.append(result.features)
        
        if len(features) == 0:
            print("❌ No features extracted!")
            continue
            
        features = np.array(features)
        print(f"   Features shape: {features.shape}")
        
        # Analyze feature diversity
        print(f"   Feature stats:")
        print(f"      Mean: {features.mean():.6f}")
        print(f"      Std: {features.std():.6f}")
        print(f"      Range: {features.min():.6f} to {features.max():.6f}")
        
        # Check for very similar features
        similarities = []
        for i in range(len(features)-1):
            cos_sim = np.dot(features[i], features[i+1]) / (np.linalg.norm(features[i]) * np.linalg.norm(features[i+1]))
            similarities.append(cos_sim)
        
        similarities = np.array(similarities)
        print(f"      Consecutive similarity: {similarities.mean():.3f} ± {similarities.std():.3f}")
        print(f"      Very similar pairs (>0.99): {(similarities > 0.99).sum()}/{len(similarities)}")
        
        # Train quantizer
        print(f"   Training quantizer...")
        quantizer = SymbolicQuantizer(vocabulary_size=64, use_l2_norm=True)
        quantizer.fit(features)
        
        # Generate tokens
        tokens = quantizer.transform(features)
        token_counts = Counter(tokens)
        
        print(f"   Token diversity:")
        print(f"      Unique tokens: {len(token_counts)}/64")
        print(f"      Most frequent: {max(token_counts.values())} occurrences")
        print(f"      Token distribution: {list(token_counts.most_common(5))}")
        
        # Check clustering quality
        stats = quantizer.get_codebook_statistics()
        print(f"      Entropy: {stats['entropy']:.2f} bits")
        print(f"      Active/Total tokens: {stats['active_tokens']}/{stats['vocabulary_size']}")
    
    # Test different normalization methods
    print(f"\n🔧 Testing normalization methods...")
    
    # Extract 50 segments of 2s each
    segments = []
    for i in range(50):
        start_sample = int(i * 2 * sr_orig)
        end_sample = int((i + 1) * 2 * sr_orig)
        if end_sample > len(audio_orig):
            break
        segments.append(audio_orig[start_sample:end_sample])
    
    # Extract features
    features = []
    for segment in segments:
        result = wav2vec.encode(segment, sr=sr_orig)
        if result is not None:
            features.append(result.features)
    
    if len(features) > 64:  # Need at least 64 samples for 64-class quantizer
        features = np.array(features)
        
        # Test L2 normalization vs StandardScaler
        norm_methods = [
            ("L2 normalization", True),
            ("StandardScaler", False)
        ]
        
        for norm_name, use_l2 in norm_methods:
            print(f"\n   {norm_name}:")
            quantizer = SymbolicQuantizer(vocabulary_size=64, use_l2_norm=use_l2)
            quantizer.fit(features)
            tokens = quantizer.transform(features)
            token_counts = Counter(tokens)
            
            print(f"      Unique tokens: {len(token_counts)}/64")
            stats = quantizer.get_codebook_statistics()
            print(f"      Entropy: {stats['entropy']:.2f} bits")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    print("=" * 50)
    
    if sr_orig != 44100:
        print(f"🔍 Your audio is {sr_orig} Hz, Wav2Vec internally uses 16kHz")
        print("   Multiple resampling steps may reduce feature quality")
        print("   Try: Convert to 44.1kHz before training")
    
    if np.sqrt(np.mean(audio_orig**2)) < 0.1:
        print("🔍 Audio level seems low")
        print("   Try: Normalize audio to -12dB to -6dB range")
    
    print("🔍 For complex music like Nineteen/Daybreak:")
    print("   - Use longer segments (4-8s) to capture musical phrases")
    print("   - Ensure training covers different musical sections")
    print("   - Check that onset detection captures rhythmic variety")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python quick_gesture_check.py <audio_file>")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    if not os.path.exists(audio_file):
        print(f"❌ File not found: {audio_file}")
        sys.exit(1)
    
    quick_gesture_analysis(audio_file)