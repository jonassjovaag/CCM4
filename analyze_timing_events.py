#!/usr/bin/env python3
"""
Quick timing events analyzer - shows what's blocking generation
"""

import csv
import sys
from collections import Counter
from pathlib import Path

def analyze_timing_log(csv_path: str):
    """Analyze timing events log"""
    
    print(f"\n{'='*80}")
    print(f"TIMING EVENTS ANALYSIS: {Path(csv_path).name}")
    print(f"{'='*80}\n")
    
    # Read CSV
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    if not rows:
        print("❌ No data in timing events log")
        return
    
    print(f"📊 Total events: {len(rows)}")
    
    # Analyze by voice
    melodic_events = [r for r in rows if r['voice'] == 'melodic']
    bass_events = [r for r in rows if r['voice'] == 'bass']
    
    print(f"   Melodic: {len(melodic_events)} events")
    print(f"   Bass: {len(bass_events)} events")
    
    # Analyze decisions
    allowed = [r for r in rows if r['decision'] == 'allowed']
    blocked = [r for r in rows if r['decision'] == 'blocked']
    
    print(f"\n📈 Decisions:")
    print(f"   ✅ Allowed: {len(allowed)} ({len(allowed)/len(rows)*100:.1f}%)")
    print(f"   ❌ Blocked: {len(blocked)} ({len(blocked)/len(rows)*100:.1f}%)")
    
    # Analyze blocking reasons
    reasons = Counter(r['blocking_reason'] for r in rows)
    
    print(f"\n🔍 Blocking reasons:")
    for reason, count in reasons.most_common():
        pct = count / len(rows) * 100
        emoji = "✅" if reason == "ready" else "❌"
        print(f"   {emoji} {reason}: {count} ({pct:.1f}%)")
    
    # Calculate generation rate
    if rows:
        start_time = float(rows[0]['timestamp'])
        end_time = float(rows[-1]['timestamp'])
        duration_sec = end_time - start_time
        duration_min = duration_sec / 60
        
        allowed_per_min = len(allowed) / duration_min if duration_min > 0 else 0
        
        print(f"\n⏱️  Generation rate:")
        print(f"   Duration: {duration_min:.2f} minutes")
        print(f"   Allowed attempts: {allowed_per_min:.1f} per minute")
    
    # Analyze gaps
    gaps = [float(r['gap_since_last_attempt']) for r in rows if r['gap_since_last_attempt']]
    
    if gaps:
        print(f"\n⏰ Gap distribution:")
        print(f"   Min gap: {min(gaps):.2f}s")
        print(f"   Max gap: {max(gaps):.2f}s")
        print(f"   Avg gap: {sum(gaps)/len(gaps):.2f}s")
        
        # Histogram
        gap_bins = {
            '0-1s': len([g for g in gaps if 0 <= g < 1]),
            '1-2s': len([g for g in gaps if 1 <= g < 2]),
            '2-5s': len([g for g in gaps if 2 <= g < 5]),
            '5-10s': len([g for g in gaps if 5 <= g < 10]),
            '>10s': len([g for g in gaps if g >= 10])
        }
        
        print(f"\n   Gap histogram:")
        for bin_name, count in gap_bins.items():
            pct = count / len(gaps) * 100
            bar = '█' * int(pct / 2)  # Scale to 50 chars max
            print(f"   {bin_name:>6}: {count:3d} ({pct:5.1f}%) {bar}")
    
    # Verdict
    print(f"\n{'='*80}")
    if len(blocked) == 0:
        print("✅ VERDICT: NO BLOCKING - All generation attempts allowed!")
    elif len(blocked) / len(rows) < 0.2:
        print("✅ VERDICT: MINIMAL BLOCKING - System is generating well")
    elif len(blocked) / len(rows) < 0.5:
        print("⚠️  VERDICT: MODERATE BLOCKING - Some generation issues")
    else:
        print("❌ VERDICT: HEAVY BLOCKING - Generation severely limited")
        
    # Show primary blocker
    if blocked:
        primary_blocker = reasons.most_common(1)[0]
        if primary_blocker[0] != 'ready':
            print(f"\n🎯 PRIMARY BLOCKER: {primary_blocker[0]}")
            print(f"   Affects {primary_blocker[1]} events ({primary_blocker[1]/len(rows)*100:.1f}%)")
    
    print(f"{'='*80}\n")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python analyze_timing_events.py <timing_events.csv>")
        sys.exit(1)
    
    analyze_timing_log(sys.argv[1])
