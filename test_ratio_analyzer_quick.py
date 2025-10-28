#!/usr/bin/env python3
"""
Quick test for ratio analyzer fix
"""

import numpy as np
from rhythmic_engine.ratio_analyzer import RatioAnalyzer

print("Testing RatioAnalyzer...")

# Create analyzer
analyzer = RatioAnalyzer(complexity_weight=0.5, deviation_weight=0.5)

# Test with simple pattern
timeseries = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
print(f"Input: {timeseries}")

# This should now work
result = analyzer.analyze(timeseries)

print(f"\n✅ SUCCESS!")
print(f"Duration pattern: {result['duration_pattern']}")
print(f"Tempo: {result['tempo']:.1f} BPM")
print(f"Pulse: {result['pulse']}")
print(f"Complexity: {result['complexity']:.2f}")
print(f"Confidence: {result['confidence']:.2f}")
print(f"\nTest PASSED - ratio_to_each() now accepts div_limit parameter")

