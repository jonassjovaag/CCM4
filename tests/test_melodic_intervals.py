#!/usr/bin/env python3
"""
Analyze NOTE-TO-NOTE intervals (removing sustains)
to see if gesture tokens have useful melodic patterns.
"""

import pickle
import gzip
import sys
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_model(model_path: str):
    """Load serialized AudioOracle model"""
    with gzip.open(model_path, 'rb') as f:
        return pickle.load(f)


def analyze_melodic_intervals(model):
    """
    Extract NOTE-TO-NOTE intervals (removing sustains/repeats)
    to see actual melodic motion patterns.
    """
    audio_frames = model.get('audio_frames', {})
    
    # Group frames by gesture token
    token_to_frames = defaultdict(list)
    for frame_id, frame_data in audio_frames.items():
        audio_data = frame_data.audio_data if hasattr(frame_data, 'audio_data') else frame_data
        
        gesture_token = audio_data.get('gesture_token')
        midi_note = audio_data.get('midi')
        
        if gesture_token is not None and midi_note is not None:
            token_to_frames[gesture_token].append({
                'frame_id': int(frame_id),
                'midi': int(midi_note)
            })
    
    # Sort by frame_id
    for token in token_to_frames:
        token_to_frames[token].sort(key=lambda x: x['frame_id'])
    
    # Analyze NOTE-TO-NOTE intervals (skip sustains)
    results = {}
    for token, frames in token_to_frames.items():
        if len(frames) < 3:
            continue
        
        midi_sequence = [f['midi'] for f in frames]
        
        # Calculate ALL intervals (including sustains)
        all_intervals = []
        for i in range(1, len(midi_sequence)):
            all_intervals.append(midi_sequence[i] - midi_sequence[i-1])
        
        # Calculate NOTE-TO-NOTE intervals (skip sustains)
        melodic_intervals = []
        prev_note = midi_sequence[0]
        for i in range(1, len(midi_sequence)):
            current_note = midi_sequence[i]
            if current_note != prev_note:  # Skip sustains
                interval = current_note - prev_note
                melodic_intervals.append(interval)
                prev_note = current_note
        
        if len(melodic_intervals) < 2:
            continue
        
        # Statistics
        all_counter = Counter(all_intervals)
        melodic_counter = Counter(melodic_intervals)
        
        sustain_percentage = all_counter[0] / len(all_intervals) * 100 if all_intervals else 0
        
        results[token] = {
            'num_frames': len(frames),
            'total_intervals': len(all_intervals),
            'sustain_count': all_counter[0],
            'sustain_percentage': sustain_percentage,
            'melodic_intervals': len(melodic_intervals),
            'melodic_mean': float(np.mean(melodic_intervals)),
            'melodic_std': float(np.std(melodic_intervals)),
            'melodic_most_common': melodic_counter.most_common(5),
            'all_most_common': all_counter.most_common(5)
        }
    
    return results


def print_results(results):
    """Print melodic interval analysis"""
    print("\n" + "="*80)
    print("NOTE-TO-NOTE INTERVAL ANALYSIS (Sustains Removed)")
    print("="*80)
    print("\nQuestion: When notes actually CHANGE, what intervals do gesture tokens use?")
    print("(This ignores sustained notes to see real melodic motion)\n")
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['num_frames'], reverse=True)
    
    print(f"Analyzed {len(results)} tokens\n")
    print("-"*80)
    print("TOP 10 TOKENS - MELODIC MOTION PATTERNS")
    print("-"*80)
    
    for i, (token, data) in enumerate(sorted_results[:10], 1):
        print(f"\n{i}. Token {token} ({data['num_frames']} frames)")
        print(f"   Sustain: {data['sustain_count']}/{data['total_intervals']} ({data['sustain_percentage']:.1f}%)")
        print(f"   Melodic moves: {data['melodic_intervals']} actual note changes")
        print(f"   Mean melodic interval: {data['melodic_mean']:+.1f} semitones")
        print(f"   Std dev: {data['melodic_std']:.1f} semitones")
        
        print(f"   Top melodic intervals (when note changes):")
        for interval, count in data['melodic_most_common'][:5]:
            percentage = count / data['melodic_intervals'] * 100
            direction = "↑" if interval > 0 else ("↓" if interval < 0 else "=")
            print(f"      {direction} {interval:+2d} semitones: {count:3d} times ({percentage:.1f}%)")
        
        # Interpretation
        if data['melodic_intervals'] > 10:
            top_interval = data['melodic_most_common'][0][0]
            if abs(top_interval) <= 2:
                print(f"   → Pattern: Stepwise motion (most common: {top_interval:+d})")
            elif abs(top_interval) <= 5:
                print(f"   → Pattern: Small leaps (most common: {top_interval:+d})")
            else:
                print(f"   → Pattern: Large leaps (most common: {top_interval:+d})")
    
    # Overall statistics
    print("\n" + "="*80)
    print("OVERALL MELODIC BEHAVIOR")
    print("="*80)
    
    all_melodic_intervals = []
    for token, data in results.items():
        for interval, count in data['melodic_most_common']:
            all_melodic_intervals.extend([interval] * count)
    
    if all_melodic_intervals:
        melodic_counter = Counter(all_melodic_intervals)
        print(f"\nTotal melodic moves across all tokens: {len(all_melodic_intervals)}")
        print("\nMost common melodic intervals (when notes change):")
        for interval, count in melodic_counter.most_common(10):
            percentage = count / len(all_melodic_intervals) * 100
            direction = "↑" if interval > 0 else ("↓" if interval < 0 else "=")
            print(f"   {direction} {interval:+3d} semitones: {count:5d} times ({percentage:.1f}%)")
        
        # Categorize
        stepwise = sum(count for i, count in melodic_counter.items() if abs(i) <= 2)
        small_leaps = sum(count for i, count in melodic_counter.items() if 3 <= abs(i) <= 5)
        large_leaps = sum(count for i, count in melodic_counter.items() if abs(i) > 5)
        
        total = len(all_melodic_intervals)
        print(f"\nMelodic motion categories:")
        print(f"   Stepwise (±1-2):  {stepwise:5d} ({stepwise/total*100:.1f}%)")
        print(f"   Small leaps (3-5): {small_leaps:5d} ({small_leaps/total*100:.1f}%)")
        print(f"   Large leaps (>5):  {large_leaps:5d} ({large_leaps/total*100:.1f}%)")


def main():
    model_path = project_root / "JSON" / "Itzama_071125_2130_training_model.pkl.gz"
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    print(f"Loading model: {model_path}")
    model = load_model(str(model_path))
    
    results = analyze_melodic_intervals(model)
    print_results(results)


if __name__ == "__main__":
    main()
