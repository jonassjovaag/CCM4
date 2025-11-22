#!/usr/bin/env python3
"""
Quick analysis of existing MIDI log from Somax test
Compare to baseline and provide conclusions
"""

import csv
from collections import Counter
from pathlib import Path

# Baseline from Somax (Nov 22, 21:50)
BASELINE = {
    'total_bass': 68,
    'c2_count': 37,
    'c2_pct': 54.4,
    'unique_bass': 16,
    'max_consecutive': 16,
    'variety_pct': 23.5
}

def analyze_midi_log(filepath):
    """Analyze MIDI log file"""
    bass_notes = []
    melody_notes = []
    
    with open(filepath, 'r') as f:
        # Skip comment lines
        lines = [line for line in f if not line.startswith('#')]
        reader = csv.DictReader(lines)
        for row in reader:
            # Skip non-note messages
            if row['message_type'] != 'note_on':
                continue
            
            note = int(row['note'])
            # Determine voice by MIDI port
            port = row['port']
            
            if 'Bass' in port:
                bass_notes.append(note)
            else:
                melody_notes.append(note)
    
    if not bass_notes:
        print("No bass notes found")
        return None
    
    # Calculate stats
    bass_counter = Counter(bass_notes)
    total_bass = len(bass_notes)
    unique_bass = len(bass_counter)
    
    # Most common
    most_common_note, most_common_count = bass_counter.most_common(1)[0]
    most_common_pct = (most_common_count / total_bass) * 100
    
    # Consecutive
    max_consecutive = 1
    current_consecutive = 1
    for i in range(1, len(bass_notes)):
        if bass_notes[i] == bass_notes[i-1]:
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 1
    
    variety_pct = (unique_bass / total_bass) * 100
    
    print(f"📊 ANALYSIS: {filepath}")
    print(f"   Total bass notes: {total_bass}")
    print(f"   Unique bass notes: {unique_bass} ({variety_pct:.1f}%)")
    print(f"   Most common: MIDI {most_common_note} = {most_common_pct:.1f}%")
    print(f"   Max consecutive: {max_consecutive}")
    
    print(f"\n📈 Top 5 bass notes:")
    for i, (note, count) in enumerate(bass_counter.most_common(5), 1):
        pct = (count / total_bass) * 100
        print(f"   {i}. MIDI {note}: {count}x ({pct:.1f}%)")
    
    print(f"\n🔍 COMPARISON TO BASELINE (Somax):")
    print(f"   Most common: {BASELINE['c2_pct']:.1f}% → {most_common_pct:.1f}% ({most_common_pct - BASELINE['c2_pct']:+.1f}%)")
    print(f"   Unique notes: {BASELINE['unique_bass']} → {unique_bass} ({unique_bass - BASELINE['unique_bass']:+d})")
    print(f"   Max consecutive: {BASELINE['max_consecutive']} → {max_consecutive} ({max_consecutive - BASELINE['max_consecutive']:+d})")
    print(f"   Variety: {BASELINE['variety_pct']:.1f}% → {variety_pct:.1f}% ({variety_pct - BASELINE['variety_pct']:+.1f}%)")
    
    # Verdict
    improved = most_common_pct < BASELINE['c2_pct']
    
    print(f"\n{'='*60}")
    if improved:
        improvement = BASELINE['c2_pct'] - most_common_pct
        print(f"✅ DIVERSITY IMPROVED by {improvement:.1f}%")
        print(f"   SomaxBridge Factor Oracle navigation was likely causing")
        print(f"   the bass C2 repetition problem.")
    else:
        print(f"❌ DIVERSITY NOT IMPROVED")
        print(f"   Problem likely lies in harmonic translation or oracle training,")
        print(f"   not SomaxBridge navigation.")
    print(f"{'='*60}")
    
    return {
        'total_bass': total_bass,
        'unique_bass': unique_bass,
        'most_common_pct': most_common_pct,
        'max_consecutive': max_consecutive,
        'improved': improved
    }

if __name__ == "__main__":
    # Find most recent MIDI log
    import glob
    files = sorted(glob.glob("logs/midi_output_*.csv"), 
                   key=lambda f: Path(f).stat().st_mtime, reverse=True)
    
    if not files:
        print("No MIDI logs found")
        exit(1)
    
    print(f"Analyzing most recent log: {files[0]}\n")
    analyze_midi_log(files[0])
