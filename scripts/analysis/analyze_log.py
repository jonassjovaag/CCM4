#!/usr/bin/env python3
"""Analyze the latest conversation log"""
import csv

log_file = "logs/conversation_20251003_181158.csv"

with open(log_file, 'r') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

inputs = [r for r in rows if r['event_type'] == 'INPUT']
outputs = [r for r in rows if r['event_type'] == 'OUTPUT']
melodic = [r for r in outputs if r['voice'] == 'melodic']
bass = [r for r in outputs if r['voice'] == 'bass']

print("📊 REAL PERFORMANCE ANALYSIS")
print("=" * 60)
print(f"Duration: {float(rows[-1]['elapsed_time']):.1f}s")
print(f"Total events: {len(rows)}")
print(f"  INPUT (human): {len(inputs)}")
print(f"  OUTPUT (AI): {len(outputs)}")
print(f"    - Melodic: {len(melodic)}")
print(f"    - Bass: {len(bass)}")
print()

# Calculate timing
if len(melodic) >= 2:
    times = [float(r['elapsed_time']) for r in melodic]
    diffs = [times[i+1] - times[i] for i in range(len(times)-1)]
    print(f"Melody timing:")
    print(f"  Average gap: {sum(diffs)/len(diffs):.2f}s")
    print(f"  Min gap: {min(diffs):.2f}s")
    print(f"  Max gap: {max(diffs):.2f}s")
    print()

if len(bass) >= 2:
    times = [float(r['elapsed_time']) for r in bass]
    diffs = [times[i+1] - times[i] for i in range(len(times)-1)]
    print(f"Bass timing:")
    print(f"  Average gap: {sum(diffs)/len(diffs):.2f}s")
    print(f"  Min gap: {min(diffs):.2f}s")
    print(f"  Max gap: {max(diffs):.2f}s")
    print()

# Check note range
melodic_notes = [int(r['midi_note']) for r in melodic]
bass_notes = [int(r['midi_note']) for r in bass]

print(f"Melodic range: {min(melodic_notes)}-{max(melodic_notes)} (expected 72-79)")
print(f"Bass range: {min(bass_notes)}-{max(bass_notes)} (expected 36-84)")
print()

# Mode distribution
modes = {}
for r in outputs:
    mode = r['mode']
    modes[mode] = modes.get(mode, 0) + 1

print("Mode distribution:")
for mode, count in sorted(modes.items(), key=lambda x: -x[1]):
    print(f"  {mode}: {count} ({count/len(outputs)*100:.1f}%)")
print()

# Show first 10 AI outputs
print("First 10 AI outputs:")
for i, r in enumerate(outputs[:10]):
    elapsed = float(r['elapsed_time'])
    voice = r['voice']
    midi = r['midi_note']
    mode = r['mode']
    print(f"  {i+1}. {elapsed:5.1f}s  {voice:8}  MIDI {midi:3}  {mode}")
