#!/usr/bin/env python3
"""Check if newly trained model has harmonic data"""
import json
import gzip
import pickle

# Try loading the pickle (faster)
print('📂 Loading model from pickle...')
with gzip.open('JSON/Itzama_071125_2130_training_model.pkl.gz', 'rb') as f:
    data = pickle.load(f)

# Check for harmonic data
print(f"\n📊 Model Statistics:")
print(f"   States: {data.get('size', 'N/A')}")
print(f"   Audio frames: {len(data.get('audio_frames', {}))}")

# Check fundamentals
fundamentals = data.get('fundamentals', {})
consonances = data.get('consonances', {})

print(f"\n🎵 Harmonic Data:")
print(f"   Fundamentals dict: {len(fundamentals)} entries")
print(f"   Consonances dict: {len(consonances)} entries")

if fundamentals:
    print(f"\n✅ SUCCESS: Model has harmonic data!")
    # Show sample data
    sample_ids = list(fundamentals.keys())[:10]
    print(f"\n   Sample fundamentals (Hz):")
    for state_id in sample_ids:
        fund = fundamentals.get(state_id, 0)
        cons = consonances.get(state_id, 0)
        print(f"      State {state_id}: {fund:.2f} Hz, consonance {cons:.3f}")
    
    # Statistics
    fund_values = [f for f in fundamentals.values() if f > 0]
    cons_values = list(consonances.values())
    
    if fund_values:
        print(f"\n📈 Statistics:")
        print(f"   Valid fundamentals: {len(fund_values)}/{len(fundamentals)}")
        print(f"   Frequency range: {min(fund_values):.1f} - {max(fund_values):.1f} Hz")
        print(f"   Average fundamental: {sum(fund_values)/len(fund_values):.1f} Hz")
        print(f"   Average consonance: {sum(cons_values)/len(cons_values):.3f}")
else:
    print(f"\n❌ WARNING: No harmonic data found!")
    print(f"   This means Phase 8 won't work - need to debug trainer capture")
