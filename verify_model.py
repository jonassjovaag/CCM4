#!/usr/bin/env python3
"""Verify trained model has diverse suffix links and correct 768D MERT features."""
import json
import numpy as np

with open('JSON/curious_gpu_new.json', 'r') as f:
    model = json.load(f)

data = model['data']
oracle = data.get('audio_oracle', {})

print('🎵 AudioOracle Structure:')
print(f'   Oracle keys: {list(oracle.keys())}')
print()

if 'states' in oracle:
    states = oracle['states']
    suffix_links = oracle.get('suffix_links', {})
    transitions = oracle.get('transitions', {})
    
    unique_targets = len(set(str(v) for v in suffix_links.values()))
    total_links = len(suffix_links)
    
    print(f'📊 Suffix Link Analysis:')
    print(f'   Total suffix links: {total_links}')
    print(f'   Unique targets: {unique_targets}')
    print(f'   States: {len(states)}')
    print(f'   Transitions: {len(transitions)}')
    print()
    
    if unique_targets == 1:
        print('❌ BROKEN: All suffix links point to state 0 (feature collapse!)')
    elif unique_targets > 100:
        print('✅ WORKING: Diverse suffix links (768D features working)')
    elif unique_targets > 10:
        print(f'⚠️  PARTIAL: {unique_targets} unique targets (expected >100)')
    else:
        print(f'❌ POOR: Only {unique_targets} unique targets')
    print()
    
    # Sample suffix link targets
    sample_targets = sorted(list(set(str(v) for v in suffix_links.values())))[:10]
    print(f'   Sample targets: {sample_targets}')
    print()
    
    # Check first state features
    first_state = states.get('0') or states.get(0) or next(iter(states.values()))
    if first_state and 'feature_vector' in first_state:
        features = np.array(first_state['feature_vector'])
        non_zero = np.count_nonzero(features)
        feature_dim = len(features)
        
        print(f'📐 Feature Vector Analysis (first state):')
        print(f'   Dimensions: {feature_dim}')
        print(f'   Non-zero: {non_zero}')
        print(f'   Zero padding: {feature_dim - non_zero}')
        print(f'   Sample values: {features[:5]}')
        print()
        
        if feature_dim == 768 and non_zero > 700:
            print('✅ CORRECT: Full 768D MERT features!')
        elif feature_dim == 768 and non_zero < 50:
            print('❌ BROKEN: Feature collapse (753/768 zeros)')
        elif feature_dim == 15:
            print('❌ ERROR: Wrong dimensions (15D polyphonic, not 768D MERT)')
        else:
            print(f'✅ GOOD: {non_zero}/{feature_dim} active')
    else:
        print('⚠️  No feature_vector found in first state')

print()
print('Training stats:', data.get('audio_oracle_stats', {}))
