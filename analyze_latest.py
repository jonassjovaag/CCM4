#!/usr/bin/env python3
import csv

log_file = "logs/conversation_20251003_182426.csv"

with open(log_file, 'r') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

inputs = [r for r in rows if r['event_type'] == 'INPUT']
outputs = [r for r in rows if r['event_type'] == 'OUTPUT']
melodic = [r for r in outputs if r['voice'] == 'melodic']
bass = [r for r in outputs if r['voice'] == 'bass']

print("📊 LATEST RUN ANALYSIS (18:27)")
print("=" * 60)
print(f"Duration: {float(rows[-1]['elapsed_time']):.1f}s")
print(f"Total events: {len(rows)}")
print(f"  INPUT (human): {len(inputs)}")
print(f"  OUTPUT (AI): {len(outputs)}")
print(f"    - Melodic: {len(melodic)}")
print(f"    - Bass: {len(bass)}")
print()

# Melody/Bass ratio
if len(outputs) > 0:
    mel_pct = len(melodic) / len(outputs) * 100
    bass_pct = len(bass) / len(outputs) * 100
    print(f"Voice distribution:")
    print(f"  Melodic: {mel_pct:.1f}%")
    print(f"  Bass: {bass_pct:.1f}%")
    print()

# Timing analysis
if len(melodic) >= 2:
    times = [float(r['elapsed_time']) for r in melodic]
    diffs = [times[i+1] - times[i] for i in range(len(times)-1)]
    print(f"Melody timing:")
    print(f"  Average gap: {sum(diffs)/len(diffs):.2f}s")
    print(f"  Min gap: {min(diffs):.2f}s")
    print(f"  Max gap: {max(diffs):.2f}s")
    
    # Count rapid-fire notes (< 0.5s gap)
    rapid = sum(1 for d in diffs if d < 0.5)
    print(f"  Rapid-fire notes (< 0.5s): {rapid}/{len(diffs)} ({rapid/len(diffs)*100:.1f}%)")
    print()

if len(bass) >= 2:
    times = [float(r['elapsed_time']) for r in bass]
    diffs = [times[i+1] - times[i] for i in range(len(times)-1)]
    print(f"Bass timing:")
    print(f"  Average gap: {sum(diffs)/len(diffs):.2f}s")
    print(f"  Min gap: {min(diffs):.2f}s")
    print(f"  Max gap: {max(diffs):.2f}s")
    
    rapid = sum(1 for d in diffs if d < 0.5)
    print(f"  Rapid-fire notes (< 0.5s): {rapid}/{len(diffs)} ({rapid/len(diffs)*100:.1f}%)")
    print()

# Note range
melodic_notes = [int(r['midi_note']) for r in melodic if r['midi_note']]
bass_notes = [int(r['midi_note']) for r in bass if r['midi_note']]

if melodic_notes:
    print(f"Melodic range: {min(melodic_notes)}-{max(melodic_notes)} (expected 72-79)")
if bass_notes:
    print(f"Bass range: {min(bass_notes)}-{max(bass_notes)} (expected 36-84)")
print()

# Mode distribution
modes = {}
for r in outputs:
    mode = r.get('mode', 'unknown')
    modes[mode] = modes.get(mode, 0) + 1

print("Mode distribution:")
for mode, count in sorted(modes.items(), key=lambda x: -x[1]):
    print(f"  {mode}: {count} ({count/len(outputs)*100:.1f}%)")
print()

# Sample conversation flow
print("Conversation sample (first 20 events):")
for i, r in enumerate(rows[:20]):
    elapsed = float(r['elapsed_time'])
    evt_type = r['event_type']
    if evt_type == 'INPUT':
        midi = r.get('midi_note', '?')
        print(f"  {elapsed:5.1f}s  👤 Human: MIDI {midi}")
    else:
        voice = r['voice']
        midi = r.get('midi_note', '?')
        mode = r.get('mode', '?')
        icon = "🎵" if voice == "melodic" else "🎸"
        print(f"  {elapsed:5.1f}s  {icon} AI {voice:8}: MIDI {midi:3}  [{mode}]")
