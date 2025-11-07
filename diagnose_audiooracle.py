#!/usr/bin/env python3
"""
AudioOracle Health Diagnostics

Investigates potential silent failures in AudioOracle:
1. Suffix link connectivity
2. Transition distribution
3. Feature dimensions
4. Distance threshold effectiveness
5. Graph connectivity issues
"""

import json
import numpy as np
from collections import Counter

def analyze_audiooracle(model_path):
    print(f"Analyzing: {model_path}\n")
    print("=" * 80)
    
    with open(model_path, 'r') as f:
        data = json.load(f)
    
    # Initialize tracking variables
    state_0_trans = 0
    low_variance_dims = 0
    isolated = 0
    reachability_pct = 0.0
    
    # Basic stats
    print("\n1. BASIC STRUCTURE")
    print(f"   Total states: {data['size']}")
    print(f"   Distance threshold: {data['distance_threshold']}")
    print(f"   Feature dimensions: {data['feature_dimensions']}")
    print(f"   Is trained: {data['is_trained']}")
    
    # Transition analysis
    print("\n2. TRANSITION DISTRIBUTION")
    trans = data['transitions']
    print(f"   Transitions type: {type(trans)}, length: {len(trans)}")
    
    # Check structure - transitions might be list or dict
    if isinstance(trans, dict):
        trans_keys = list(trans.keys())[:5]
        print(f"   First transition keys: {trans_keys}")
        
        # Count transitions per state if structured as state->transitions
        states_data = data['states']
        trans_counts = []
        for i in range(min(100, len(states_data))):
            state = states_data.get(str(i), {})
            if 'next' in state:
                trans_counts.append(len(state['next']))
            else:
                trans_counts.append(0)
        
        print(f"   States with transitions: {sum(1 for c in trans_counts if c > 0)}/{len(trans_counts)}")
        print(f"   Avg transitions per state (first 100): {np.mean(trans_counts):.1f}")
        print(f"   Max transitions in a state: {max(trans_counts)}")
        print(f"   Min transitions: {min(trans_counts)}")
        
        # Check if state 0 has abnormal connectivity
        state_0 = states_data.get('0', {})
        state_0_trans = len(state_0.get('next', {}))
        print(f"\n   ⚠️  State 0 has {state_0_trans} transitions!")
        if state_0_trans > 100:
            print("       This is ABNORMAL - state 0 should not connect to all states")
            print("       Suggests root state initialization issue")
    else:
        print(f"   Transitions are stored as: {type(trans)}")
    
    # Suffix link analysis
    print("\n3. SUFFIX LINK COVERAGE")
    suffix = data['suffix_links']
    has_suffix = sum(1 for s in suffix.values() if s is not None and s != -1)
    suffix_pct = (has_suffix / len(suffix)) * 100
    
    print(f"   States with suffix links: {has_suffix}/{len(suffix)} ({suffix_pct:.1f}%)")
    
    # Sample some suffix links
    suffix_sample = [(int(k), v) for k, v in list(suffix.items())[:20]]
    print(f"   First 20 suffix links: {suffix_sample[:5]}...")
    
    # Audio frames analysis  
    print("\n4. AUDIO FRAMES (Features)")
    frames = data['audio_frames']
    print(f"   Total frames: {len(frames)}")
    
    # Check frame structure
    first_frame_key = list(frames.keys())[0] if frames else None
    if first_frame_key:
        first_frame = frames[first_frame_key]
        print(f"   Frame keys: {list(first_frame.keys())}")
        if 'features' in first_frame:
            feats = first_frame['features']
            print(f"   Feature dimension: {len(feats)}D")
            print(f"   Feature range: [{min(feats):.3f}, {max(feats):.3f}]")
            print(f"   Feature mean: {np.mean(feats):.3f}")
            print(f"   Feature std: {np.std(feats):.3f}")
    
    # Check for feature diversity
    print("\n5. FEATURE DIVERSITY CHECK")
    sample_size = min(100, len(frames))
    sample_features = []
    for i in range(sample_size):
        frame_key = str(i)
        if frame_key in frames and 'features' in frames[frame_key]:
            sample_features.append(frames[frame_key]['features'])
    
    if sample_features:
        feature_array = np.array(sample_features)
        per_dim_variance = np.var(feature_array, axis=0)
        low_variance_dims = sum(1 for v in per_dim_variance if v < 0.01)
        
        print(f"   Sampled {sample_size} frames")
        print(f"   Per-dimension variance range: [{np.min(per_dim_variance):.4f}, {np.max(per_dim_variance):.4f}]")
        print(f"   Low-variance dimensions (<0.01): {low_variance_dims}/{len(per_dim_variance)}")
        
        if low_variance_dims > 5:
            print(f"   ⚠️  Many low-variance dimensions - features may lack diversity")
    
    # Graph connectivity check
    print("\n6. GRAPH CONNECTIVITY")
    
    states_data = data['states']
    
    # Build reachability graph
    reachable = set([0])  # Start from state 0
    to_explore = [0]
    explored = set()
    
    while to_explore and len(explored) < 1000:  # Limit exploration
        current = to_explore.pop(0)
        if current in explored:
            continue
        explored.add(current)
        
        # Get transitions from current state via states['next']
        current_state = states_data.get(str(current), {})
        if 'next' in current_state:
            for next_state in current_state['next'].values():
                if isinstance(next_state, int) and next_state not in reachable:
                    reachable.add(next_state)
                    to_explore.append(next_state)
    
    reachability_pct = (len(reachable) / data['size']) * 100
    print(f"   Reachable states from state 0: {len(reachable)}/{data['size']} ({reachability_pct:.1f}%)")
    
    if reachability_pct < 50:
        print("   ⚠️  Low reachability - graph may have isolated components")
    
    # Check for isolated states
    isolated = sum(1 for i in range(min(1000, data['size'])) 
                  if not states_data.get(str(i), {}).get('next', {}))
    print(f"   Isolated states (no outgoing transitions): {isolated}/1000 sampled")
    
    print("\n" + "=" * 80)
    print("\nDIAGNOSIS SUMMARY:")
    print("=" * 80)
    
    issues = []
    if state_0_trans > 100:
        issues.append("❌ State 0 over-connected (initialization bug)")
    if suffix_pct < 50:
        issues.append("❌ Low suffix link coverage (pattern matching weak)")
    if low_variance_dims > 5:
        issues.append("⚠️  Feature diversity low (may cause distance threshold issues)")
    if reachability_pct < 80:
        issues.append("⚠️  Poor graph connectivity (many unreachable states)")
    if isolated > 50:
        issues.append("❌ Many isolated states (graph fragmentation)")
    
    if issues:
        print("\nISSUES FOUND:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\n✅ No major structural issues detected")
    
    print("\n")
    
    return {
        'state_0_transitions': state_0_trans,
        'suffix_coverage': suffix_pct,
        'reachability': reachability_pct,
        'isolated_states': isolated,
        'low_variance_dims': low_variance_dims if sample_features else 0
    }


if __name__ == "__main__":
    model_path = "JSON/Itzama_061125_2255_training_model.json"
    results = analyze_audiooracle(model_path)
    
    print("\nKEY METRICS:")
    for key, value in results.items():
        print(f"  {key}: {value}")
