#!/usr/bin/env python3
"""
Verify MERT Melodic Clustering
==============================

This script analyzes whether MERT tokens cluster by melodic content or just timbre.

Key question: When two audio frames have the same token, do they have similar MIDI notes?
- If YES (low MIDI variance within token) → MERT captures MELODIC content
- If NO (high MIDI variance within token) → MERT only captures TIMBRE

Run:
    python scripts/analysis/verify_mert_melodic_clustering.py JSON/Moon_stars.json

Or with vocab files to also check dual vocabulary:
    python scripts/analysis/verify_mert_melodic_clustering.py JSON/Moon_stars.json \
        --harmonic-vocab input_audio/Moon_stars_harmonic_vocab.joblib \
        --percussive-vocab input_audio/Moon_stars_percussive_vocab.joblib
"""

import json
import argparse
import numpy as np
from collections import defaultdict
from pathlib import Path


def load_training_data(filepath: str, harmonic_vocab_path: str = None, percussive_vocab_path: str = None) -> dict:
    """Load training data from JSON file, optionally assigning dual vocab tokens"""
    print(f"📂 Loading {filepath}...")

    with open(filepath, 'r') as f:
        data = json.load(f)

    # Handle different formats
    if 'data' in data and 'audio_oracle' in data['data']:
        oracle_data = data['data']['audio_oracle']
        print("   Format: Unified (data.audio_oracle)")
    else:
        oracle_data = data
        print("   Format: Legacy")

    frames = oracle_data.get('audio_frames', {})
    print(f"   ✅ Loaded {len(frames)} frames")

    # Load vocab files and assign tokens if provided
    harmonic_quantizer = None
    percussive_quantizer = None

    if harmonic_vocab_path:
        try:
            import joblib
            harmonic_quantizer = joblib.load(harmonic_vocab_path)
            print(f"   🎸 Loaded harmonic vocab: {harmonic_vocab_path}")
        except Exception as e:
            print(f"   ⚠️  Could not load harmonic vocab: {e}")

    if percussive_vocab_path:
        try:
            import joblib
            percussive_quantizer = joblib.load(percussive_vocab_path)
            print(f"   🥁 Loaded percussive vocab: {percussive_vocab_path}")
        except Exception as e:
            print(f"   ⚠️  Could not load percussive vocab: {e}")

    # Assign tokens from vocab files if they're not already in the data
    if harmonic_quantizer or percussive_quantizer:
        tokens_assigned = 0
        for frame_id, frame in frames.items():
            audio_data = frame.get('audio_data', {})

            # Features can be at frame level OR inside audio_data
            features = frame.get('features')
            if features is None:
                features = audio_data.get('features')
            if features is None:
                features = audio_data.get('wav2vec_features')

            if features is None:
                continue

            features = np.array(features)

            # Assign harmonic token
            if harmonic_quantizer and 'harmonic_token' not in audio_data:
                try:
                    features_64 = features.astype(np.float64)
                    token = int(harmonic_quantizer.transform(features_64.reshape(1, -1))[0])
                    audio_data['harmonic_token'] = token
                    tokens_assigned += 1
                except Exception:
                    pass

            # Assign percussive token
            if percussive_quantizer and 'percussive_token' not in audio_data:
                try:
                    features_64 = features.astype(np.float64)
                    token = int(percussive_quantizer.transform(features_64.reshape(1, -1))[0])
                    audio_data['percussive_token'] = token
                except Exception:
                    pass

        if tokens_assigned > 0:
            print(f"   🎯 Assigned dual vocab tokens to {tokens_assigned} frames")

    return frames


def analyze_token_clustering(frames: dict, token_field: str = 'gesture_token') -> dict:
    """
    Analyze whether tokens cluster by melodic content

    Returns dict with analysis results
    """
    # Group frames by token
    token_groups = defaultdict(list)

    for frame_id, frame in frames.items():
        audio_data = frame.get('audio_data', {})
        token = audio_data.get(token_field)
        midi = audio_data.get('midi')
        f0 = audio_data.get('f0')
        consonance = audio_data.get('consonance') or audio_data.get('hybrid_consonance')

        if token is not None and midi is not None and midi > 0:
            token_groups[token].append({
                'midi': float(midi),
                'f0': float(f0) if f0 else None,
                'consonance': float(consonance) if consonance else None,
                'frame_id': frame_id
            })

    if not token_groups:
        return {'error': f'No {token_field} found in frames'}

    # Analyze each token
    results = {
        'token_field': token_field,
        'unique_tokens': len(token_groups),
        'total_frames': sum(len(g) for g in token_groups.values()),
        'melodic_tokens': [],  # std < 3 semitones
        'mixed_tokens': [],    # std 3-8 semitones
        'timbral_tokens': [],  # std > 8 semitones
        'token_details': {}
    }

    for token in sorted(token_groups.keys()):
        frames_list = token_groups[token]
        midis = [f['midi'] for f in frames_list]

        if len(midis) < 2:
            continue

        midi_std = np.std(midis)
        midi_mean = np.mean(midis)
        midi_min = min(midis)
        midi_max = max(midis)

        # Classify token type
        if midi_std < 3:
            token_type = 'melodic'
            results['melodic_tokens'].append(token)
        elif midi_std < 8:
            token_type = 'mixed'
            results['mixed_tokens'].append(token)
        else:
            token_type = 'timbral'
            results['timbral_tokens'].append(token)

        results['token_details'][token] = {
            'count': len(midis),
            'midi_mean': midi_mean,
            'midi_std': midi_std,
            'midi_range': (midi_min, midi_max),
            'type': token_type
        }

    return results


def analyze_interval_consistency(frames: dict, token_field: str = 'gesture_token') -> dict:
    """
    Analyze whether consecutive frames within a token follow consistent interval patterns
    """
    # Group frames by token, preserving order
    token_sequences = defaultdict(list)

    # Sort by frame_id to get temporal order
    sorted_frames = sorted(frames.items(), key=lambda x: int(x[0]))

    for frame_id, frame in sorted_frames:
        audio_data = frame.get('audio_data', {})
        token = audio_data.get(token_field)
        midi = audio_data.get('midi')

        if token is not None and midi is not None and midi > 0:
            token_sequences[token].append(float(midi))

    # Analyze interval patterns within each token
    results = {}

    for token, midis in token_sequences.items():
        if len(midis) < 3:
            continue

        # Calculate intervals between consecutive notes
        intervals = [midis[i+1] - midis[i] for i in range(len(midis)-1)]

        # Count interval frequencies
        interval_counts = defaultdict(int)
        for interval in intervals:
            # Round to nearest semitone
            interval_rounded = round(interval)
            interval_counts[interval_rounded] += 1

        # Get top intervals
        top_intervals = sorted(interval_counts.items(), key=lambda x: -x[1])[:5]

        # Calculate interval consistency (how often the most common interval appears)
        if intervals:
            most_common_count = top_intervals[0][1] if top_intervals else 0
            consistency = most_common_count / len(intervals)
        else:
            consistency = 0

        results[token] = {
            'num_notes': len(midis),
            'num_intervals': len(intervals),
            'top_intervals': top_intervals,
            'consistency': consistency
        }

    return results


def print_results(token_results: dict, interval_results: dict):
    """Print formatted results"""

    print(f"\n{'='*70}")
    print(f"MERT MELODIC CLUSTERING ANALYSIS")
    print(f"{'='*70}")

    print(f"\n📊 Token Distribution ({token_results['token_field']}):")
    print(f"   Unique tokens: {token_results['unique_tokens']}")
    print(f"   Total frames with tokens: {token_results['total_frames']}")

    print(f"\n🎵 Token Classification:")
    print(f"   MELODIC tokens (MIDI std < 3):  {len(token_results['melodic_tokens'])}")
    print(f"   MIXED tokens (MIDI std 3-8):    {len(token_results['mixed_tokens'])}")
    print(f"   TIMBRAL tokens (MIDI std > 8):  {len(token_results['timbral_tokens'])}")

    # Calculate percentages
    total = len(token_results['melodic_tokens']) + len(token_results['mixed_tokens']) + len(token_results['timbral_tokens'])
    if total > 0:
        melodic_pct = len(token_results['melodic_tokens']) / total * 100
        mixed_pct = len(token_results['mixed_tokens']) / total * 100
        timbral_pct = len(token_results['timbral_tokens']) / total * 100

        print(f"\n   Percentages:")
        print(f"   MELODIC: {melodic_pct:.1f}%  |  MIXED: {mixed_pct:.1f}%  |  TIMBRAL: {timbral_pct:.1f}%")

    # Interpretation
    print(f"\n📈 INTERPRETATION:")
    if melodic_pct > 50:
        print(f"   ✅ MERT captures MELODIC content!")
        print(f"   Most tokens cluster similar MIDI notes together.")
        print(f"   Token matching = melodic similarity matching.")
    elif melodic_pct + mixed_pct > 60:
        print(f"   ⚠️  MERT captures SOME melodic content")
        print(f"   Tokens have moderate pitch consistency.")
    else:
        print(f"   ❌ MERT primarily captures TIMBRE, not melody")
        print(f"   Tokens group by sound texture, not pitch content.")

    # Show details for sample tokens
    print(f"\n{'─'*70}")
    print(f"Token Details (first 15):")
    print(f"{'─'*70}")
    print(f"{'Token':>6} | {'Count':>5} | {'MIDI Range':>12} | {'Std':>6} | {'Type':>8}")
    print(f"{'─'*70}")

    for token in sorted(token_results['token_details'].keys())[:15]:
        details = token_results['token_details'][token]
        midi_range = f"{details['midi_range'][0]:.0f}-{details['midi_range'][1]:.0f}"
        print(f"{token:>6} | {details['count']:>5} | {midi_range:>12} | {details['midi_std']:>6.2f} | {details['type']:>8}")

    # Interval analysis
    print(f"\n{'─'*70}")
    print(f"Interval Consistency (sample tokens):")
    print(f"{'─'*70}")

    sample_tokens = list(interval_results.keys())[:8]
    for token in sample_tokens:
        ir = interval_results[token]
        top_intervals_str = ", ".join([f"{i}:{c}" for i, c in ir['top_intervals'][:3]])
        print(f"   Token {token}: {ir['num_notes']} notes, consistency={ir['consistency']:.2f}")
        print(f"      Top intervals: {top_intervals_str}")


def main():
    parser = argparse.ArgumentParser(description='Verify MERT melodic clustering')
    parser.add_argument('training_file', help='Path to training JSON file (e.g., JSON/Moon_stars.json)')
    parser.add_argument('--harmonic-vocab', help='Path to harmonic vocab joblib (optional)')
    parser.add_argument('--percussive-vocab', help='Path to percussive vocab joblib (optional)')
    parser.add_argument('--output', help='Save results to JSON file')

    args = parser.parse_args()

    # Load training data (with optional vocab files for dual token assignment)
    frames = load_training_data(
        args.training_file,
        harmonic_vocab_path=args.harmonic_vocab,
        percussive_vocab_path=args.percussive_vocab
    )

    # Check what token fields exist
    sample_frame = list(frames.values())[0] if frames else {}
    audio_data = sample_frame.get('audio_data', {})
    available_fields = list(audio_data.keys())

    print(f"\n🔍 Available fields in audio_data:")
    print(f"   {available_fields[:20]}...")

    # Determine which token field to use
    token_fields_to_check = []
    if 'gesture_token' in audio_data:
        token_fields_to_check.append('gesture_token')
    if 'harmonic_token' in audio_data:
        token_fields_to_check.append('harmonic_token')
    if 'percussive_token' in audio_data:
        token_fields_to_check.append('percussive_token')

    if not token_fields_to_check:
        print(f"\n⚠️  No token fields found in training data!")
        print(f"   This may be because tokens were not embedded during training.")
        print(f"   Run MusicHal with vocab files to assign tokens at load time.")
        return

    # Analyze each token field
    all_results = {}

    for token_field in token_fields_to_check:
        print(f"\n{'='*70}")
        print(f"Analyzing: {token_field}")
        print(f"{'='*70}")

        token_results = analyze_token_clustering(frames, token_field)
        interval_results = analyze_interval_consistency(frames, token_field)

        if 'error' in token_results:
            print(f"   ⚠️  {token_results['error']}")
            continue

        print_results(token_results, interval_results)
        all_results[token_field] = {
            'token_analysis': token_results,
            'interval_analysis': interval_results
        }

    # Save results if requested
    if args.output:
        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(i) for i in obj]
            return obj

        with open(args.output, 'w') as f:
            json.dump(convert_numpy(all_results), f, indent=2)
        print(f"\n💾 Results saved to {args.output}")

    print(f"\n{'='*70}")
    print("Analysis complete!")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
