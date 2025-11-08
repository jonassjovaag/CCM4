#!/usr/bin/env python3
"""
Test: Consonance vs Gesture Token as MIDI Predictor
====================================================

Critical Question:
Which audio_data parameter better predicts MIDI output?
- gesture_token (768D Wav2Vec → quantized)
- consonance (Brandtsegg ratio analysis)

If consonance has lower MIDI variance than gesture_token,
it means the harmonic analysis pathway is the actual bridge to MIDI space.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict
from collections import defaultdict
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.utils.test_helpers import load_trained_model, save_test_results


def compare_predictors(model_path: Path) -> Dict:
    """Compare consonance vs gesture_token as MIDI predictors"""
    
    print("=" * 80)
    print("CONSONANCE VS GESTURE TOKEN: MIDI PREDICTION COMPARISON")
    print("=" * 80)
    
    model_data = load_trained_model(model_path)
    audio_frames = model_data.get('audio_frames', {})
    
    print(f"\n📂 Model: {model_path.name}")
    print(f"📊 States: {len(audio_frames)}")
    
    # Group frames by gesture_token
    token_to_midi = defaultdict(list)
    token_to_consonance = defaultdict(list)
    
    # Group frames by consonance bins
    consonance_bins = {
        '0.9-1.0': [],  # Very consonant
        '0.8-0.9': [],
        '0.7-0.8': [],
        '0.6-0.7': [],
        '0.5-0.6': [],
        '0.4-0.5': [],
        '0.0-0.4': [],  # Dissonant
    }
    
    for frame_id, frame in audio_frames.items():
        if hasattr(frame, 'audio_data'):
            audio_data = frame.audio_data
        else:
            audio_data = frame
        
        if 'gesture_token' in audio_data and 'midi' in audio_data and 'consonance' in audio_data:
            token = int(audio_data['gesture_token'])
            midi = int(audio_data['midi'])
            cons = float(audio_data['consonance'])
            
            token_to_midi[token].append(midi)
            token_to_consonance[token].append(cons)
            
            # Bin by consonance
            if cons >= 0.9:
                consonance_bins['0.9-1.0'].append(midi)
            elif cons >= 0.8:
                consonance_bins['0.8-0.9'].append(midi)
            elif cons >= 0.7:
                consonance_bins['0.7-0.8'].append(midi)
            elif cons >= 0.6:
                consonance_bins['0.6-0.7'].append(midi)
            elif cons >= 0.5:
                consonance_bins['0.5-0.6'].append(midi)
            elif cons >= 0.4:
                consonance_bins['0.4-0.5'].append(midi)
            else:
                consonance_bins['0.0-0.4'].append(midi)
    
    # Analyze gesture token variance
    print("\n" + "=" * 80)
    print("GESTURE TOKEN → MIDI VARIANCE")
    print("=" * 80)
    
    token_variances = []
    for token, midis in token_to_midi.items():
        if len(midis) >= 5:  # Only tokens with sufficient data
            variance = np.var(midis)
            token_variances.append(variance)
    
    mean_token_variance = np.mean(token_variances)
    median_token_variance = np.median(token_variances)
    
    print(f"\nGesture Token as MIDI predictor:")
    print(f"   Tokens analyzed: {len(token_variances)}")
    print(f"   Mean MIDI variance: {mean_token_variance:.2f}")
    print(f"   Median MIDI variance: {median_token_variance:.2f}")
    print(f"   Min variance: {min(token_variances):.2f}")
    print(f"   Max variance: {max(token_variances):.2f}")
    
    # Analyze consonance bin variance
    print("\n" + "=" * 80)
    print("CONSONANCE BIN → MIDI VARIANCE")
    print("=" * 80)
    
    consonance_variances = []
    
    print(f"\nConsonance as MIDI predictor:")
    for bin_name, midis in sorted(consonance_bins.items()):
        if len(midis) >= 5:
            variance = np.var(midis)
            midi_range = (min(midis), max(midis))
            consonance_variances.append(variance)
            print(f"   {bin_name}: {len(midis)} frames, variance={variance:.2f}, range={midi_range}")
    
    mean_cons_variance = np.mean(consonance_variances) if consonance_variances else 0
    median_cons_variance = np.median(consonance_variances) if consonance_variances else 0
    
    print(f"\n   Mean MIDI variance: {mean_cons_variance:.2f}")
    print(f"   Median MIDI variance: {median_cons_variance:.2f}")
    
    # COMPARISON
    print("\n" + "=" * 80)
    print("CRITICAL COMPARISON")
    print("=" * 80)
    
    variance_reduction = ((mean_token_variance - mean_cons_variance) / mean_token_variance * 100) if mean_token_variance > 0 else 0
    
    print(f"\n📊 MIDI Prediction Accuracy:")
    print(f"   Gesture Token mean variance: {mean_token_variance:.2f}")
    print(f"   Consonance mean variance: {mean_cons_variance:.2f}")
    print(f"   Variance reduction: {variance_reduction:.1f}%")
    
    if variance_reduction > 20:
        print(f"\n✅ CONSONANCE IS BETTER PREDICTOR")
        print(f"   Consonance binning reduces MIDI variance by {variance_reduction:.1f}%")
        print(f"   → Harmonic analysis pathway should be used for MIDI translation")
    elif variance_reduction < -20:
        print(f"\n❌ GESTURE TOKEN IS BETTER PREDICTOR")
        print(f"   This is unexpected! Investigate why tokens predict pitch better than consonance.")
    else:
        print(f"\n⚠️  NEITHER IS GOOD PREDICTOR")
        print(f"   Both have similar variance ({abs(variance_reduction):.1f}% difference)")
        print(f"   → Need different approach to bridge 768D → MIDI")
    
    # Additional insight: Correlation between consonance and MIDI
    print("\n" + "=" * 80)
    print("CONSONANCE ↔ MIDI CORRELATION")
    print("=" * 80)
    
    all_consonances = []
    all_midis = []
    
    for frame_id, frame in audio_frames.items():
        if hasattr(frame, 'audio_data'):
            audio_data = frame.audio_data
        else:
            audio_data = frame
        
        if 'consonance' in audio_data and 'midi' in audio_data:
            all_consonances.append(float(audio_data['consonance']))
            all_midis.append(int(audio_data['midi']))
    
    if all_consonances and all_midis:
        correlation = np.corrcoef(all_consonances, all_midis)[0, 1]
        print(f"\nPearson correlation (consonance ↔ MIDI): {correlation:.3f}")
        
        if abs(correlation) < 0.1:
            print(f"   → No linear relationship between consonance and MIDI pitch")
            print(f"   → This is EXPECTED (consonance is independent of pitch)")
        else:
            print(f"   → Unexpected correlation detected")
    else:
        correlation = 0.0
    
    # Save results
    results = {
        'model_path': str(model_path),
        'gesture_token_analysis': {
            'mean_variance': float(mean_token_variance),
            'median_variance': float(median_token_variance),
            'tokens_analyzed': len(token_variances),
        },
        'consonance_analysis': {
            'mean_variance': float(mean_cons_variance),
            'median_variance': float(median_cons_variance),
            'bins_analyzed': len(consonance_variances),
        },
        'comparison': {
            'variance_reduction_percent': float(variance_reduction),
            'better_predictor': 'consonance' if variance_reduction > 20 else ('gesture_token' if variance_reduction < -20 else 'neither'),
        },
        'correlation': {
            'consonance_midi': float(correlation) if all_consonances else 0.0,
        }
    }
    
    save_test_results('predictor_comparison', results)
    
    print("\n" + "=" * 80)
    print("✅ Analysis complete!")
    print(f"📄 Results saved: tests/test_output/predictor_comparison_results.json")
    print("=" * 80)
    
    return results


def main():
    model_path = Path("JSON/Itzama_071125_2130_training_model.pkl.gz")
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return False
    
    results = compare_predictors(model_path)
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
