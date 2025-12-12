#!/usr/bin/env python3
"""
Test Dual Vocab Cross-Modal Response Logic

Tests the core dual vocabulary behavior:
- Harmonic input → percussive response (cross-modal)
- Percussive input → harmonic response (cross-modal)

This simulates what happens during live performance when the system
detects harmonic vs percussive content and generates appropriate responses.

Usage:
    python scripts/test/test_dual_vocab_response.py JSON/your_model.json
"""

import sys
import json
from pathlib import Path
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def load_trained_data(json_path: Path) -> dict:
    """Load trained model data."""
    with open(json_path, 'r') as f:
        return json.load(f)


def build_token_frame_maps(audio_data: list) -> tuple:
    """
    Build mappings from tokens to frames.

    Returns:
        harmonic_to_frames: {harmonic_token: [frame_indices]}
        percussive_to_frames: {percussive_token: [frame_indices]}
    """
    harmonic_to_frames = defaultdict(list)
    percussive_to_frames = defaultdict(list)

    for i, frame in enumerate(audio_data):
        h_token = frame.get('harmonic_token')
        p_token = frame.get('percussive_token')

        if h_token is not None:
            harmonic_to_frames[h_token].append(i)
        if p_token is not None:
            percussive_to_frames[p_token].append(i)

    return dict(harmonic_to_frames), dict(percussive_to_frames)


def simulate_cross_modal_response(
    input_token: int,
    input_type: str,  # 'harmonic' or 'percussive'
    harmonic_to_frames: dict,
    percussive_to_frames: dict,
    audio_data: list
) -> dict:
    """
    Simulate cross-modal response.

    If input is harmonic: find frames with that harmonic_token, return their percussive info
    If input is percussive: find frames with that percussive_token, return their harmonic info
    """
    if input_type == 'harmonic':
        # User played harmonic content with token X
        # Find all frames with harmonic_token == X
        matching_frames = harmonic_to_frames.get(input_token, [])

        if not matching_frames:
            return {'matched': False, 'reason': f'No frames with harmonic_token={input_token}'}

        # Get percussive tokens from those frames (cross-modal response)
        response_tokens = []
        response_frames = []
        for frame_idx in matching_frames:
            frame = audio_data[frame_idx]
            p_token = frame.get('percussive_token')
            if p_token is not None:
                response_tokens.append(p_token)
                response_frames.append({
                    'frame_idx': frame_idx,
                    'timestamp': frame.get('t', 0),
                    'midi': frame.get('midi'),
                    'percussive_token': p_token,
                    'harmonic_token': frame.get('harmonic_token')
                })

        return {
            'matched': True,
            'input_type': 'harmonic',
            'input_token': input_token,
            'matching_frame_count': len(matching_frames),
            'response_type': 'percussive',
            'response_tokens': list(set(response_tokens)),
            'response_token_counts': {t: response_tokens.count(t) for t in set(response_tokens)},
            'sample_frames': response_frames[:5]  # First 5 for brevity
        }

    else:  # percussive input
        # User played percussive content with token Y
        # Find all frames with percussive_token == Y
        matching_frames = percussive_to_frames.get(input_token, [])

        if not matching_frames:
            return {'matched': False, 'reason': f'No frames with percussive_token={input_token}'}

        # Get harmonic tokens from those frames (cross-modal response)
        response_tokens = []
        response_frames = []
        for frame_idx in matching_frames:
            frame = audio_data[frame_idx]
            h_token = frame.get('harmonic_token')
            if h_token is not None:
                response_tokens.append(h_token)
                response_frames.append({
                    'frame_idx': frame_idx,
                    'timestamp': frame.get('t', 0),
                    'midi': frame.get('midi'),
                    'harmonic_token': h_token,
                    'percussive_token': frame.get('percussive_token')
                })

        return {
            'matched': True,
            'input_type': 'percussive',
            'input_token': input_token,
            'matching_frame_count': len(matching_frames),
            'response_type': 'harmonic',
            'response_tokens': list(set(response_tokens)),
            'response_token_counts': {t: response_tokens.count(t) for t in set(response_tokens)},
            'sample_frames': response_frames[:5]
        }


def test_dual_vocab_responses(json_path: Path):
    """Main test function."""
    print(f"Loading trained data from: {json_path}")
    data = load_trained_data(json_path)

    audio_data = data.get('audio_data', [])
    if not audio_data:
        print("ERROR: No audio_data found in JSON")
        return

    print(f"Loaded {len(audio_data)} frames")

    # Check for dual vocab tokens
    has_harmonic = any(f.get('harmonic_token') is not None for f in audio_data)
    has_percussive = any(f.get('percussive_token') is not None for f in audio_data)

    if not has_harmonic or not has_percussive:
        print("\nERROR: Dual vocab tokens not found!")
        print(f"  Has harmonic_token: {has_harmonic}")
        print(f"  Has percussive_token: {has_percussive}")
        print("\nMake sure the model was trained with enable_dual_vocabulary=true")
        return

    # Build token-to-frame mappings
    harmonic_to_frames, percussive_to_frames = build_token_frame_maps(audio_data)

    print(f"\nDual Vocabulary Statistics:")
    print(f"  Unique harmonic tokens: {len(harmonic_to_frames)}")
    print(f"  Unique percussive tokens: {len(percussive_to_frames)}")

    # Get most common tokens for testing
    h_tokens_by_freq = sorted(harmonic_to_frames.keys(),
                               key=lambda t: len(harmonic_to_frames[t]),
                               reverse=True)
    p_tokens_by_freq = sorted(percussive_to_frames.keys(),
                               key=lambda t: len(percussive_to_frames[t]),
                               reverse=True)

    print(f"\nTop 5 harmonic tokens (by frequency):")
    for t in h_tokens_by_freq[:5]:
        print(f"  Token {t}: {len(harmonic_to_frames[t])} frames")

    print(f"\nTop 5 percussive tokens (by frequency):")
    for t in p_tokens_by_freq[:5]:
        print(f"  Token {t}: {len(percussive_to_frames[t])} frames")

    # ========================================
    # TEST 1: Harmonic Input → Percussive Response
    # ========================================
    print("\n" + "="*60)
    print("TEST 1: HARMONIC INPUT -> PERCUSSIVE RESPONSE")
    print("="*60)
    print("Scenario: You play melodic/harmonic content")
    print("Expected: System responds with rhythmic/percussive frames")
    print()

    # Test with top 3 harmonic tokens
    for h_token in h_tokens_by_freq[:3]:
        result = simulate_cross_modal_response(
            h_token, 'harmonic',
            harmonic_to_frames, percussive_to_frames,
            audio_data
        )

        print(f"Input: harmonic_token={h_token}")
        if result['matched']:
            print(f"  Matched {result['matching_frame_count']} frames")
            print(f"  Response percussive tokens: {result['response_tokens']}")
            print(f"  Token distribution: {result['response_token_counts']}")
            if result['sample_frames']:
                print(f"  Sample response frame:")
                sf = result['sample_frames'][0]
                print(f"    t={sf['timestamp']:.3f}s, MIDI={sf['midi']}, perc_token={sf['percussive_token']}")
        else:
            print(f"  No match: {result['reason']}")
        print()

    # ========================================
    # TEST 2: Percussive Input → Harmonic Response
    # ========================================
    print("="*60)
    print("TEST 2: PERCUSSIVE INPUT -> HARMONIC RESPONSE")
    print("="*60)
    print("Scenario: You play rhythmic/percussive content")
    print("Expected: System responds with melodic/harmonic frames")
    print()

    # Test with top 3 percussive tokens
    for p_token in p_tokens_by_freq[:3]:
        result = simulate_cross_modal_response(
            p_token, 'percussive',
            harmonic_to_frames, percussive_to_frames,
            audio_data
        )

        print(f"Input: percussive_token={p_token}")
        if result['matched']:
            print(f"  Matched {result['matching_frame_count']} frames")
            print(f"  Response harmonic tokens: {result['response_tokens']}")
            print(f"  Token distribution: {result['response_token_counts']}")
            if result['sample_frames']:
                print(f"  Sample response frame:")
                sf = result['sample_frames'][0]
                print(f"    t={sf['timestamp']:.3f}s, MIDI={sf['midi']}, harm_token={sf['harmonic_token']}")
        else:
            print(f"  No match: {result['reason']}")
        print()

    # ========================================
    # TEST 3: Cross-Modal Diversity Analysis
    # ========================================
    print("="*60)
    print("TEST 3: CROSS-MODAL DIVERSITY ANALYSIS")
    print("="*60)
    print("Checks if different harmonic tokens lead to different percussive responses")
    print()

    # For each harmonic token, what percussive tokens does it map to?
    h_to_p_mapping = {}
    for h_token in harmonic_to_frames:
        p_tokens = set()
        for frame_idx in harmonic_to_frames[h_token]:
            p_token = audio_data[frame_idx].get('percussive_token')
            if p_token is not None:
                p_tokens.add(p_token)
        h_to_p_mapping[h_token] = p_tokens

    # Check diversity
    total_h_tokens = len(h_to_p_mapping)
    h_tokens_with_multiple_p = sum(1 for pts in h_to_p_mapping.values() if len(pts) > 1)
    avg_p_per_h = sum(len(pts) for pts in h_to_p_mapping.values()) / total_h_tokens if total_h_tokens > 0 else 0

    print(f"Harmonic -> Percussive mapping diversity:")
    print(f"  Total harmonic tokens: {total_h_tokens}")
    print(f"  Harmonic tokens with multiple percussive responses: {h_tokens_with_multiple_p}")
    print(f"  Average percussive tokens per harmonic token: {avg_p_per_h:.2f}")

    # Same for percussive -> harmonic
    p_to_h_mapping = {}
    for p_token in percussive_to_frames:
        h_tokens = set()
        for frame_idx in percussive_to_frames[p_token]:
            h_token = audio_data[frame_idx].get('harmonic_token')
            if h_token is not None:
                h_tokens.add(h_token)
        p_to_h_mapping[p_token] = h_tokens

    total_p_tokens = len(p_to_h_mapping)
    p_tokens_with_multiple_h = sum(1 for hts in p_to_h_mapping.values() if len(hts) > 1)
    avg_h_per_p = sum(len(hts) for hts in p_to_h_mapping.values()) / total_p_tokens if total_p_tokens > 0 else 0

    print(f"\nPercussive -> Harmonic mapping diversity:")
    print(f"  Total percussive tokens: {total_p_tokens}")
    print(f"  Percussive tokens with multiple harmonic responses: {p_tokens_with_multiple_h}")
    print(f"  Average harmonic tokens per percussive token: {avg_h_per_p:.2f}")

    # ========================================
    # SUMMARY
    # ========================================
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    # Check if cross-modal is actually working
    cross_modal_works = (
        len(harmonic_to_frames) > 1 and
        len(percussive_to_frames) > 1 and
        avg_p_per_h > 1.0 and
        avg_h_per_p > 1.0
    )

    if cross_modal_works:
        print("PASS: Cross-modal response logic is working")
        print("  - Harmonic input can trigger varied percussive responses")
        print("  - Percussive input can trigger varied harmonic responses")
        print("  - This enables style-dependent counterpoint generation")
    else:
        print("WARNING: Cross-modal diversity may be limited")
        if avg_p_per_h <= 1.0:
            print("  - Low harmonic->percussive diversity")
        if avg_h_per_p <= 1.0:
            print("  - Low percussive->harmonic diversity")
        print("  - This may lead to predictable responses")


def main():
    if len(sys.argv) < 2:
        # Try to find a recent JSON file
        json_dir = project_root / "JSON"
        if json_dir.exists():
            json_files = list(json_dir.glob("*.json"))
            if json_files:
                # Use most recent
                json_path = max(json_files, key=lambda p: p.stat().st_mtime)
                print(f"Using most recent JSON: {json_path}")
            else:
                print("Usage: python test_dual_vocab_response.py <model.json>")
                return 1
        else:
            print("Usage: python test_dual_vocab_response.py <model.json>")
            return 1
    else:
        json_path = Path(sys.argv[1])

    if not json_path.exists():
        print(f"ERROR: File not found: {json_path}")
        return 1

    test_dual_vocab_responses(json_path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
