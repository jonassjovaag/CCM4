#!/usr/bin/env python3
import re
from collections import Counter

log_text = """
🎼 AudioOracle generated 4 notes from learned patterns (context frame=0)
   Generated notes: [72, 72, 72, 72]
🎼 AudioOracle generated 1 notes from learned patterns (context frame=0)
   Generated notes: [60]
🎼 AudioOracle generated 2 notes from learned patterns (context frame=0)
   Generated notes: [54, 54]
🎼 AudioOracle generated 1 notes from learned patterns (context frame=0)
   Generated notes: [56]
🎼 AudioOracle generated 4 notes from learned patterns (context frame=0)
   Generated notes: [72, 72, 72, 72]
🎼 AudioOracle generated 1 notes from learned patterns (context frame=0)
   Generated notes: [64]
🎼 AudioOracle generated 22 notes from learned patterns (context frame=0)
   Generated notes: [64, 65, 54, 55, 54, 54, 57, 60, 59, 59]...
🎼 AudioOracle generated 23 notes from learned patterns (context frame=0)
   Generated notes: [76, 77, 78, 79, 78, 78, 72, 72, 72, 72]...
🎼 AudioOracle generated 20 notes from learned patterns (context frame=0)
   Generated notes: [64, 65, 54, 55, 54, 54, 57, 60, 59, 59]...
🎼 AudioOracle generated 12 notes from learned patterns (context frame=0)
   Generated notes: [76, 77, 78, 79, 78, 78, 72, 72, 72, 72]...
🎼 AudioOracle generated 23 notes from learned patterns (context frame=0)
   Generated notes: [64, 65, 54, 55, 54, 54, 57, 60, 59, 59]...
🎼 AudioOracle generated 9 notes from learned patterns (context frame=0)
   Generated notes: [76, 77, 78, 79, 78, 78, 72, 72, 72]
"""

print("=" * 80)
print("🔍 ITZAMA RUN ANALYSIS")
print("=" * 80)

# Extract all generated notes
all_notes = []
pattern = r'Generated notes: \[([\d, ]+)\]'
for match in re.finditer(pattern, log_text):
    notes_str = match.group(1)
    notes = [int(n.strip()) for n in notes_str.split(',')]
    all_notes.extend(notes)

# Also parse the "..." lines
pattern2 = r'Generated notes: \[([\d, ]+)\.\.\.'
for match in re.finditer(pattern2, log_text):
    notes_str = match.group(1)
    notes = [int(n.strip()) for n in notes_str.split(',')]
    all_notes.extend(notes)

print(f"\n📊 ALL GENERATED NOTES:")
print(f"  Total: {len(all_notes)}")
print(f"  Unique MIDI: {len(set(all_notes))}")
print(f"  Range: {min(all_notes)} to {max(all_notes)}")

# Count occurrences
counter = Counter(all_notes)
print(f"\n🎵 MIDI Note Distribution:")
for midi, count in counter.most_common():
    pct = count / len(all_notes) * 100
    bar = '█' * int(pct / 2)
    pc_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    pc = midi % 12
    octave = (midi // 12) - 1
    print(f"  {midi:>3} ({pc_names[pc]}{octave}): {count:>3} ({pct:>5.1f}%) {bar}")

# Check repetition patterns
print(f"\n⚠️  REPETITION ISSUES:")
repetitions = []
prev = None
rep_count = 0
for note in all_notes:
    if note == prev:
        rep_count += 1
    else:
        if rep_count > 1:
            repetitions.append((prev, rep_count))
        rep_count = 1
        prev = note

if rep_count > 1:
    repetitions.append((prev, rep_count))

if repetitions:
    print(f"  Found {len(repetitions)} repetition sequences:")
    for note, count in repetitions[:10]:
        pc_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        pc = note % 12
        print(f"    MIDI {note} ({pc_names[pc]}) repeated {count} times")

# Analyze patterns
print(f"\n🔍 PATTERN ANALYSIS:")
print(f"  Notice: ALL queries use 'context frame=0'!")
print(f"  AudioOracle is STUCK on frame 0")
print(f"  Even with phrase context, it returns to frame 0")
print(f"\n  This suggests:")
print(f"    1. generate_next() always returns frame 0")
print(f"    2. Frame 0 might be the only reachable frame")
print(f"    3. Transitions from frame 0 might loop back to 0")
