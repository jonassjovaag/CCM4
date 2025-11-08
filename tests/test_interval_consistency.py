#!/usr/bin/env python3
"""
Test if gesture tokens have consistent INTERVAL patterns
even when absolute MIDI notes are scattered.

Quick validation test before implementing harmonic translation layer.
"""

import pickle
import gzip
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_model(model_path: str) -> Dict:
    """Load serialized AudioOracle model (pickle or JSON)"""
    path = Path(model_path)
    
    if path.suffix == '.gz':
        with gzip.open(model_path, 'rb') as f:
            return pickle.load(f)
    else:
        with open(model_path, 'r') as f:
            return json.load(f)


def analyze_interval_consistency(model: Dict) -> Dict:
    """
    Analyze if gesture tokens have consistent INTERVAL patterns
    even when absolute MIDI notes vary widely.
    
    Key insight: If Token 27 always goes [+2, -1, +4] (stepwise up, back, leap up),
    then we can use it for melodic motion even if absolute pitch varies.
    
    Returns:
        Dict with interval variance analysis per token
    """
    audio_frames = model.get('audio_frames', {})
    
    # Group frames by gesture token
    token_to_frames = defaultdict(list)
    for frame_id, frame_data in audio_frames.items():
        # Handle both dict and AudioFrame object
        if isinstance(frame_data, dict):
            audio_data = frame_data.get('audio_data', frame_data)
        else:
            audio_data = frame_data.audio_data if hasattr(frame_data, 'audio_data') else {}
        
        gesture_token = audio_data.get('gesture_token')
        midi_note = audio_data.get('midi')
        
        if gesture_token is not None and midi_note is not None:
            token_to_frames[gesture_token].append({
                'frame_id': int(frame_id),
                'midi': int(midi_note)
            })
    
    # Sort frames by frame_id to get temporal order
    for token in token_to_frames:
        token_to_frames[token].sort(key=lambda x: x['frame_id'])
    
    # Calculate intervals for each token
    results = {}
    for token, frames in token_to_frames.items():
        if len(frames) < 3:  # Need at least 3 notes to calculate 2 intervals
            continue
        
        # Extract MIDI sequence
        midi_sequence = [f['midi'] for f in frames]
        
        # Calculate intervals (melodic motion between consecutive notes)
        intervals = []
        for i in range(1, len(midi_sequence)):
            interval = midi_sequence[i] - midi_sequence[i-1]
            intervals.append(interval)
        
        # Calculate interval statistics
        intervals_array = np.array(intervals)
        interval_mean = float(np.mean(intervals_array))
        interval_std = float(np.std(intervals_array))
        interval_variance = float(np.var(intervals_array))
        
        # Find most common interval
        from collections import Counter
        interval_counts = Counter(intervals)
        most_common_interval, count = interval_counts.most_common(1)[0]
        
        # Count interval direction patterns
        up_moves = sum(1 for i in intervals if i > 0)
        down_moves = sum(1 for i in intervals if i < 0)
        repeats = sum(1 for i in intervals if i == 0)
        
        # Calculate MIDI note variance for comparison
        midi_variance = float(np.var(np.array(midi_sequence)))
        
        results[token] = {
            'num_frames': len(frames),
            'num_intervals': len(intervals),
            'midi_variance': midi_variance,
            'interval_mean': interval_mean,
            'interval_std': interval_std,
            'interval_variance': interval_variance,
            'most_common_interval': most_common_interval,
            'most_common_count': count,
            'interval_distribution': {
                'up_moves': up_moves,
                'down_moves': down_moves,
                'repeats': repeats
            },
            # Improvement ratio: if intervals are more consistent than absolute MIDI,
            # this will be > 1.0
            'improvement_ratio': midi_variance / interval_variance if interval_variance > 0 else 0
        }
    
    return results


def print_analysis_results(results: Dict):
    """Print interval consistency analysis"""
    print("\n" + "="*80)
    print("INTERVAL CONSISTENCY ANALYSIS")
    print("="*80)
    print("\nHypothesis: Gesture tokens might have consistent INTERVAL patterns")
    print("even when absolute MIDI notes are scattered.\n")
    
    if not results:
        print("❌ No gesture tokens found in model")
        print("   This may be an older model without gesture_token field")
        return
    
    # Sort by number of frames (most frequent first)
    sorted_results = sorted(results.items(), key=lambda x: x[1]['num_frames'], reverse=True)
    
    print(f"Analyzed {len(results)} gesture tokens\n")
    
    # Overall statistics
    improvement_ratios = [r['improvement_ratio'] for _, r in sorted_results if r['improvement_ratio'] > 0]
    avg_improvement = np.mean(improvement_ratios) if improvement_ratios else 0
    
    print(f"Average improvement ratio: {avg_improvement:.2f}x")
    print(f"  (>1.0 means intervals are more consistent than absolute MIDI)\n")
    
    # Count how many tokens have good interval consistency
    low_interval_variance = sum(1 for _, r in sorted_results if r['interval_variance'] < 10)
    med_interval_variance = sum(1 for _, r in sorted_results if 10 <= r['interval_variance'] < 30)
    high_interval_variance = sum(1 for _, r in sorted_results if r['interval_variance'] >= 30)
    
    print("Interval Variance Distribution:")
    print(f"  Low (<10):     {low_interval_variance}/{len(results)} ({low_interval_variance/len(results)*100:.1f}%)")
    print(f"  Medium (10-30): {med_interval_variance}/{len(results)} ({med_interval_variance/len(results)*100:.1f}%)")
    print(f"  High (≥30):    {high_interval_variance}/{len(results)} ({high_interval_variance/len(results)*100:.1f}%)")
    
    # Detailed examples
    print("\n" + "-"*80)
    print("TOP 10 MOST FREQUENT TOKENS (Detailed Analysis)")
    print("-"*80)
    
    for i, (token, data) in enumerate(sorted_results[:10], 1):
        print(f"\nToken {token} ({data['num_frames']} frames, {data['num_intervals']} intervals):")
        print(f"  MIDI variance:     {data['midi_variance']:.2f}")
        print(f"  Interval variance: {data['interval_variance']:.2f}")
        print(f"  Improvement ratio: {data['improvement_ratio']:.2f}x")
        print(f"  Mean interval:     {data['interval_mean']:.2f} semitones")
        print(f"  Std dev interval:  {data['interval_std']:.2f} semitones")
        print(f"  Most common:       {data['most_common_interval']:+d} semitones ({data['most_common_count']} times)")
        
        dist = data['interval_distribution']
        total = sum(dist.values())
        print(f"  Direction: ↑{dist['up_moves']/total*100:.0f}% ↓{dist['down_moves']/total*100:.0f}% ={dist['repeats']/total*100:.0f}%")
        
        # Interpretation
        if data['improvement_ratio'] > 2.0:
            print(f"  ✅ STRONG pattern: Intervals {data['improvement_ratio']:.1f}x more consistent than MIDI")
        elif data['improvement_ratio'] > 1.5:
            print(f"  ✅ GOOD pattern: Intervals {data['improvement_ratio']:.1f}x more consistent")
        elif data['improvement_ratio'] > 1.0:
            print(f"  ⚠️  WEAK pattern: Intervals {data['improvement_ratio']:.1f}x more consistent")
        else:
            print(f"  ❌ NO pattern: Intervals as scattered as MIDI")
    
    # Conclusion
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    
    if avg_improvement > 2.0:
        print(f"✅ STRONG RESULT: Average {avg_improvement:.2f}x improvement")
        print("   Gesture tokens DO have consistent interval patterns!")
        print("   → Interval-based retrieval is HIGHLY PROMISING")
        print("   → Recommend: Build HarmonicTranslator using interval extraction")
    elif avg_improvement > 1.5:
        print(f"✅ GOOD RESULT: Average {avg_improvement:.2f}x improvement")
        print("   Gesture tokens have MODERATE interval consistency")
        print("   → Interval-based retrieval will help but not solve everything")
        print("   → Recommend: Hybrid approach (intervals + scale constraints)")
    elif avg_improvement > 1.0:
        print(f"⚠️  WEAK RESULT: Average {avg_improvement:.2f}x improvement")
        print("   Gesture tokens have SLIGHT interval consistency")
        print("   → Interval-based retrieval alone won't be enough")
        print("   → Recommend: Scale constraint filtering (Option 1)")
    else:
        print(f"❌ NO IMPROVEMENT: Average {avg_improvement:.2f}x")
        print("   Gesture tokens do NOT have interval patterns")
        print("   → Interval-based retrieval won't work")
        print("   → Recommend: Pure harmonic generation (ignore AudioOracle)")


def main():
    # Load Itzama model (use pickle version from earlier test)
    model_path = project_root / "JSON" / "Itzama_071125_2130_training_model.pkl.gz"
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print("Please ensure Itzama.json exists in JSON/ directory")
        return
    
    print(f"Loading model: {model_path}")
    model = load_model(str(model_path))
    
    # Analyze interval consistency
    results = analyze_interval_consistency(model)
    
    # Print results
    print_analysis_results(results)
    
    print("\n" + "="*80)
    print(f"Analysis complete. Results based on {len(results)} gesture tokens.")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
