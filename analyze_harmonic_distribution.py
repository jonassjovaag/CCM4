#!/usr/bin/env python3
"""Deep analysis of harmonic data distribution"""
import gzip
import pickle
from collections import Counter

print('📂 Loading model...')
with gzip.open('JSON/Itzama_071125_2130_training_model.pkl.gz', 'rb') as f:
    data = pickle.load(f)

fundamentals = data.get('fundamentals', {})
consonances = data.get('consonances', {})

print(f"\n📊 Total Data:")
print(f"   Fundamentals: {len(fundamentals)} entries")
print(f"   Consonances: {len(consonances)} entries")

# Analyze fundamental frequency distribution
fund_values = [round(f, 2) for f in fundamentals.values() if f > 0]
fund_counter = Counter(fund_values)

print(f"\n🔍 Fundamental Frequency Distribution:")
print(f"   Unique frequencies: {len(fund_counter)}")
print(f"   Most common frequencies (top 20):")

for freq, count in fund_counter.most_common(20):
    percentage = (count / len(fund_values)) * 100
    print(f"      {freq:8.2f} Hz: {count:4d} occurrences ({percentage:5.1f}%)")

# Analyze consonance distribution
cons_values = [round(c, 3) for c in consonances.values()]
cons_counter = Counter(cons_values)

print(f"\n🔍 Consonance Distribution:")
print(f"   Unique consonance values: {len(cons_counter)}")
print(f"   Most common consonances (top 10):")

for cons, count in cons_counter.most_common(10):
    percentage = (count / len(cons_values)) * 100
    print(f"      {cons:.3f}: {count:4d} occurrences ({percentage:5.1f}%)")

# Check for stuck values (same value repeated many times)
print(f"\n⚠️  Repetition Analysis:")

# Check if first 100 fundamentals are identical
first_100_funds = [fundamentals.get(i, 0) for i in range(1, 101)]
unique_in_first_100 = len(set(first_100_funds))
print(f"   First 100 states: {unique_in_first_100} unique fundamentals")

# Sample across the full range
print(f"\n📍 Sample across full state range:")
sample_points = [1, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
for state_id in sample_points:
    fund = fundamentals.get(state_id, 0)
    cons = consonances.get(state_id, 0)
    print(f"      State {state_id:4d}: {fund:8.2f} Hz, consonance {cons:.3f}")

# Check for diversity
print(f"\n📈 Diversity Metrics:")
if len(fund_counter) > 0:
    # Calculate entropy (rough measure of diversity)
    import math
    total = len(fund_values)
    entropy = -sum((count/total) * math.log2(count/total) 
                   for count in fund_counter.values())
    max_entropy = math.log2(len(fund_counter))
    
    print(f"   Frequency entropy: {entropy:.2f} bits (max {max_entropy:.2f})")
    print(f"   Diversity: {(entropy/max_entropy)*100:.1f}%")
    
    # Check if top frequency dominates
    top_freq, top_count = fund_counter.most_common(1)[0]
    dominance = (top_count / total) * 100
    if dominance > 50:
        print(f"   ⚠️  WARNING: {top_freq} Hz appears in {dominance:.1f}% of states!")
        print(f"              This suggests stuck/repeated values")
    else:
        print(f"   ✅ Good spread: top frequency only {dominance:.1f}% of total")
