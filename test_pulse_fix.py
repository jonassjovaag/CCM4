#!/usr/bin/env python3
"""Test pulse detection fix for straight 8-beat patterns"""

import numpy as np
from rhythmic_engine.ratio_analyzer import RatioAnalyzer

# Test case: Straight 8-beat pattern (should detect pulse 4, not pulse 3!)
# Simulate 8 evenly-spaced onsets (like 4 bars of 4/4 with kick on each beat)
tempo = 120  # BPM
interval = 60.0 / tempo  # 0.5 seconds per beat
onsets = np.array([i * interval for i in range(8)])  # 0.0, 0.5, 1.0, 1.5, ... 3.5

print('Testing pulse detection on straight 8-beat pattern (4/4 time):')
print(f'Onsets: {onsets}')
print('Expected: pulse=4 (quadruple meter) or pulse=2 (duple)')
print('WRONG: pulse=3 (triple meter - was the bug!)')
print()

analyzer = RatioAnalyzer(complexity_weight=0.5, deviation_weight=0.5)

# PATCH: Add debug output to indispensability_subdiv
import rhythmic_engine.ratio_analyzer as ra_module
original_method = ra_module.RatioAnalyzer.indispensability_subdiv

def debug_indispensability(self, trigger_seq):
    print(f'🔍 DEBUG trigger_seq length: {len(trigger_seq)}')
    print(f'🔍 DEBUG trigger_seq: {trigger_seq}')
    result = original_method(self, trigger_seq)
    print(f'🔍 DEBUG result: pulse={result[0]}, position={result[1]}')
    return result

ra_module.RatioAnalyzer.indispensability_subdiv = debug_indispensability

result = analyzer.analyze(onsets)

print(f'✅ Result:')
print(f'   Duration pattern: {result["duration_pattern"]}')
print(f'   Tempo: {result["tempo"]:.1f} BPM')
print(f'   Pulse: {result["pulse"]} {"✓ CORRECT!" if result["pulse"] in [2, 4] else "✗ STILL WRONG (pulse 3 for straight beats)"}')
print(f'   Complexity: {result["complexity"]:.2f}')
print(f'   Confidence: {result["confidence"]:.2f}')
print()

# Test case 2: Triplet pattern (should detect pulse 3)
print('Testing pulse detection on triplet pattern (should be pulse 3):')
triplet_interval = interval / 3  # 3 notes per beat
triplet_onsets = np.array([i * triplet_interval for i in range(12)])  # 12 triplet subdivisions

print(f'Triplet onsets: {triplet_onsets}')
print(f'Length: {len(triplet_onsets)} onsets → {len(triplet_onsets)-1} intervals')

analyzer2 = RatioAnalyzer(complexity_weight=0.5, deviation_weight=0.5)
result2 = analyzer2.analyze(triplet_onsets)

print(f'   Duration pattern: {result2["duration_pattern"]} (length={len(result2["duration_pattern"])})')
print(f'   Pulse: {result2["pulse"]} {"✓ CORRECT!" if result2["pulse"] == 3 else "✗ Should be 3"}')

