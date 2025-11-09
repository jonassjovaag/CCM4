#!/usr/bin/env python3
"""
Fast Translation Mechanism Test - Using Existing Trained Model
===============================================================

Instead of training from scratch (slow), analyze the existing Itzama model
to understand gesture token → MIDI mapping behavior.

Questions:
Q1. In the trained model, how many unique gesture tokens exist?
Q2. For each token, what MIDI notes does it map to? (consistent or scattered?)
Q3. Do tokens with similar consonance cluster together?
Q4. Can we predict MIDI output from gesture token alone?

This reveals whether the translation mechanism is working as designed.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List
from collections import defaultdict
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.utils.test_helpers import load_trained_model, save_test_results


def analyze_gesture_to_midi_mapping(model_path: Path) -> Dict:
    """
    Analyze gesture token → MIDI mapping in trained model
    
    Returns scientific findings about the translation mechanism
    """
    
    print("=" * 80)
    print("FAST TRANSLATION MECHANISM ANALYSIS")
    print("=" * 80)
    print(f"\n📂 Model: {model_path.name}")
    
    # Load model
    print("\n⏳ Loading model...")
    model_data = load_trained_model(model_path)
    audio_frames = model_data.get('audio_frames', {})
    
    print(f"✅ Loaded {len(audio_frames)} states")
    
    # Extract gesture token → MIDI mappings
    print("\n🔍 Analyzing gesture token → MIDI mappings...")
    
    token_to_midi = defaultdict(list)
    token_to_consonance = defaultdict(list)
    token_to_f0 = defaultdict(list)
    
    frames_with_tokens = 0
    frames_with_midi = 0
    frames_with_both = 0
    
    for frame_id, frame in audio_frames.items():
        # Handle both dict and AudioFrame object formats
        if hasattr(frame, 'audio_data'):
            audio_data = frame.audio_data
        elif isinstance(frame, dict):
            audio_data = frame.get('audio_data', frame)
        else:
            audio_data = frame
        
        has_token = 'gesture_token' in audio_data
        has_midi = 'midi' in audio_data
        
        if has_token:
            frames_with_tokens += 1
        if has_midi:
            frames_with_midi += 1
        if has_token and has_midi:
            frames_with_both += 1
            
            token = int(audio_data['gesture_token'])
            midi = int(audio_data['midi'])
            consonance = float(audio_data.get('consonance', 0.0))
            f0 = audio_data.get('f0', None)
            
            token_to_midi[token].append(midi)
            token_to_consonance[token].append(consonance)
            if f0:
                token_to_f0[token].append(float(f0))
    
    print(f"\n📊 Data availability:")
    print(f"   Frames with gesture_token: {frames_with_tokens}/{len(audio_frames)} ({frames_with_tokens/len(audio_frames)*100:.1f}%)")
    print(f"   Frames with midi: {frames_with_midi}/{len(audio_frames)} ({frames_with_midi/len(audio_frames)*100:.1f}%)")
    print(f"   Frames with BOTH: {frames_with_both}/{len(audio_frames)} ({frames_with_both/len(audio_frames)*100:.1f}%)")
    
    if frames_with_both == 0:
        print("\n❌ ERROR: No frames have both gesture_token and midi fields!")
        print("   The translation mechanism cannot be tested.")
        return {'error': 'No data with both fields'}
    
    # Analyze token→MIDI mapping quality
    print("\n" + "=" * 80)
    print("GESTURE TOKEN → MIDI MAPPING QUALITY")
    print("=" * 80)
    
    unique_tokens = len(token_to_midi)
    print(f"\n📊 Found {unique_tokens} unique gesture tokens")
    
    # Token statistics
    token_stats = {}
    
    for token in sorted(token_to_midi.keys()):
        midi_notes = token_to_midi[token]
        consonances = token_to_consonance[token]
        f0s = token_to_f0.get(token, [])
        
        unique_midi = set(midi_notes)
        midi_variance = np.var(midi_notes)
        midi_mean = np.mean(midi_notes)
        consonance_mean = np.mean(consonances)
        consonance_std = np.std(consonances)
        
        token_stats[token] = {
            'frame_count': len(midi_notes),
            'unique_midi_notes': len(unique_midi),
            'midi_range': (int(min(midi_notes)), int(max(midi_notes))),
            'midi_mean': float(midi_mean),
            'midi_variance': float(midi_variance),
            'consonance_mean': float(consonance_mean),
            'consonance_std': float(consonance_std),
            'f0_count': len(f0s),
        }
    
    # Print top tokens (most frequent)
    sorted_by_frequency = sorted(token_stats.items(), key=lambda x: x[1]['frame_count'], reverse=True)
    
    print(f"\n🎯 Top 10 most frequent gesture tokens:")
    print(f"{'Token':<8} {'Frames':<8} {'Unique MIDI':<12} {'MIDI Range':<15} {'Variance':<10} {'Consonance':<12}")
    print("-" * 80)
    
    for token, stats in sorted_by_frequency[:10]:
        print(f"{token:<8} {stats['frame_count']:<8} {stats['unique_midi_notes']:<12} {str(stats['midi_range']):<15} "
              f"{stats['midi_variance']:<10.2f} {stats['consonance_mean']:<12.3f}")
    
    # Analysis: Token consistency
    print("\n" + "=" * 80)
    print("TOKEN CONSISTENCY ANALYSIS")
    print("=" * 80)
    
    # How many tokens map to single MIDI note?
    single_note_tokens = sum(1 for stats in token_stats.values() if stats['unique_midi_notes'] == 1)
    # How many have low variance (< 5 semitones²)?
    low_variance_tokens = sum(1 for stats in token_stats.values() if stats['midi_variance'] < 5.0)
    # How many are used frequently (>= 10 frames)?
    frequent_tokens = sum(1 for stats in token_stats.values() if stats['frame_count'] >= 10)
    
    print(f"\n📊 Token mapping consistency:")
    print(f"   Tokens mapping to single MIDI note: {single_note_tokens}/{unique_tokens} ({single_note_tokens/unique_tokens*100:.1f}%)")
    print(f"   Tokens with low MIDI variance (<5): {low_variance_tokens}/{unique_tokens} ({low_variance_tokens/unique_tokens*100:.1f}%)")
    print(f"   Frequently used tokens (≥10 frames): {frequent_tokens}/{unique_tokens} ({frequent_tokens/unique_tokens*100:.1f}%)")
    
    # Analysis: Consonance clustering
    print("\n" + "=" * 80)
    print("CONSONANCE CLUSTERING ANALYSIS")
    print("=" * 80)
    
    # Do tokens with similar consonance have similar MIDI?
    # Group tokens by consonance ranges
    consonance_groups = {
        'high (>0.8)': [],
        'medium (0.5-0.8)': [],
        'low (<0.5)': [],
    }
    
    for token, stats in token_stats.items():
        cons = stats['consonance_mean']
        if cons > 0.8:
            consonance_groups['high (>0.8)'].append((token, stats))
        elif cons > 0.5:
            consonance_groups['medium (0.5-0.8)'].append((token, stats))
        else:
            consonance_groups['low (<0.5)'].append((token, stats))
    
    print(f"\nToken distribution by consonance:")
    for group_name, tokens in consonance_groups.items():
        if tokens:
            midi_ranges = [stats['midi_range'] for _, stats in tokens]
            print(f"\n   {group_name}: {len(tokens)} tokens")
            print(f"      Example MIDI ranges: {midi_ranges[:5]}")
    
    # Key findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    
    findings = []
    
    # Finding 1: Token vocabulary usage
    vocab_usage = unique_tokens / 64 * 100  # Assuming 64-token vocabulary
    findings.append(f"Vocabulary usage: {unique_tokens}/64 tokens ({vocab_usage:.1f}%)")
    
    # Finding 2: Mapping consistency
    if low_variance_tokens / unique_tokens > 0.7:
        findings.append(f"✅ Good mapping consistency: {low_variance_tokens/unique_tokens*100:.1f}% of tokens have low MIDI variance")
    else:
        findings.append(f"⚠️  Poor mapping consistency: Only {low_variance_tokens/unique_tokens*100:.1f}% of tokens have low MIDI variance")
    
    # Finding 3: Data coverage
    if frames_with_both / len(audio_frames) > 0.9:
        findings.append(f"✅ Excellent data coverage: {frames_with_both/len(audio_frames)*100:.1f}% of frames have both token and MIDI")
    else:
        findings.append(f"⚠️  Incomplete data coverage: Only {frames_with_both/len(audio_frames)*100:.1f}% of frames have both token and MIDI")
    
    # Finding 4: Single-note dominance
    if single_note_tokens / unique_tokens > 0.5:
        findings.append(f"⚠️  High single-note mapping: {single_note_tokens/unique_tokens*100:.1f}% of tokens map to only one MIDI note (may indicate monophonic behavior)")
    
    print()
    for i, finding in enumerate(findings, 1):
        print(f"{i}. {finding}")
    
    # Save results
    results = {
        'model_path': str(model_path),
        'total_frames': len(audio_frames),
        'frames_with_tokens': frames_with_tokens,
        'frames_with_midi': frames_with_midi,
        'frames_with_both': frames_with_both,
        'unique_tokens': unique_tokens,
        'token_statistics': {int(k): v for k, v in token_stats.items()},
        'consistency_metrics': {
            'single_note_tokens': single_note_tokens,
            'low_variance_tokens': low_variance_tokens,
            'frequent_tokens': frequent_tokens,
            'vocabulary_usage_percent': float(vocab_usage),
        },
        'consonance_groups': {k: len(v) for k, v in consonance_groups.items()},
        'findings': findings,
    }
    
    save_test_results('fast_translation_analysis', results)
    
    print("\n" + "=" * 80)
    print("✅ Analysis complete!")
    print(f"📄 Results saved: tests/test_output/fast_translation_analysis_results.json")
    print("=" * 80)
    
    return results


def main():
    """Run fast translation analysis on existing model"""
    
    model_path = Path("JSON/Itzama_071125_2130_training_model.pkl.gz")
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print("   Available models:")
        for p in Path("JSON").glob("*.pkl.gz"):
            print(f"      {p}")
        return False
    
    results = analyze_gesture_to_midi_mapping(model_path)
    
    return 'error' not in results


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
