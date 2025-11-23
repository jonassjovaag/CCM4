#!/usr/bin/env python3
"""
Quick analysis of conversation log - shows interaction patterns
"""

import csv
import sys
from collections import Counter

def analyze_conversation_log(csv_path: str):
    """Analyze conversation log from real performance"""
    
    print(f"\n{'='*80}")
    print(f"CONVERSATION LOG ANALYSIS: {csv_path.split('/')[-1]}")
    print(f"{'='*80}\n")
    
    # Read CSV
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    if not rows:
        print("❌ No data in conversation log")
        return
    
    # Calculate duration
    start_time = float(rows[0]['timestamp'])
    end_time = float(rows[-1]['timestamp'])
    duration_sec = end_time - start_time
    duration_min = duration_sec / 60
    
    print(f"📊 Session duration: {duration_min:.2f} minutes ({duration_sec:.1f}s)")
    print(f"📊 Total events: {len(rows)}")
    
    # Analyze by event type
    inputs = [r for r in rows if r['event_type'] == 'INPUT']
    outputs = [r for r in rows if r['event_type'] == 'OUTPUT']
    
    print(f"\n🎤 Human input events: {len(inputs)}")
    print(f"🤖 AI output events: {len(outputs)}")
    
    if len(inputs) > 0:
        input_rate = len(inputs) / duration_min
        print(f"   Human rate: {input_rate:.1f} events/minute")
    
    if len(outputs) > 0:
        output_rate = len(outputs) / duration_min
        print(f"   AI rate: {output_rate:.1f} events/minute")
        print(f"   AI/Human ratio: {len(outputs)/len(inputs):.2f}:1")
    
    # Analyze voices
    melodic_outputs = [r for r in outputs if r['voice'] == 'melodic']
    bass_outputs = [r for r in outputs if r['voice'] == 'bass']
    
    print(f"\n🎵 Voice breakdown:")
    print(f"   Melodic: {len(melodic_outputs)} notes ({len(melodic_outputs)/len(outputs)*100:.1f}%)")
    print(f"   Bass: {len(bass_outputs)} notes ({len(bass_outputs)/len(outputs)*100:.1f}%)")
    
    # Analyze modes
    modes = Counter(r['mode'] for r in outputs)
    
    print(f"\n🎭 Behavioral modes:")
    for mode, count in modes.most_common():
        pct = count / len(outputs) * 100
        print(f"   {mode}: {count} events ({pct:.1f}%)")
    
    # Analyze MIDI notes
    melodic_notes = [int(r['midi_note']) for r in melodic_outputs if r['midi_note']]
    bass_notes = [int(r['midi_note']) for r in bass_outputs if r['midi_note']]
    
    if melodic_notes:
        print(f"\n🎹 Melodic range:")
        print(f"   Min: MIDI {min(melodic_notes)} ({midi_to_note(min(melodic_notes))})")
        print(f"   Max: MIDI {max(melodic_notes)} ({midi_to_note(max(melodic_notes))})")
        print(f"   Unique notes: {len(set(melodic_notes))}")
        print(f"   Diversity: {len(set(melodic_notes))/len(melodic_notes)*100:.1f}%")
        
        # Most common notes
        note_counts = Counter(melodic_notes)
        print(f"\n   Most common melodic notes:")
        for note, count in note_counts.most_common(5):
            pct = count / len(melodic_notes) * 100
            print(f"      {midi_to_note(note)} (MIDI {note}): {count}x ({pct:.1f}%)")
    
    if bass_notes:
        print(f"\n🎸 Bass range:")
        print(f"   Min: MIDI {min(bass_notes)} ({midi_to_note(min(bass_notes))})")
        print(f"   Max: MIDI {max(bass_notes)} ({midi_to_note(max(bass_notes))})")
        print(f"   Unique notes: {len(set(bass_notes))}")
        print(f"   Diversity: {len(set(bass_notes))/len(bass_notes)*100:.1f}%")
        
        # Most common notes
        note_counts = Counter(bass_notes)
        print(f"\n   Most common bass notes:")
        for note, count in note_counts.most_common(5):
            pct = count / len(bass_notes) * 100
            print(f"      {midi_to_note(note)} (MIDI {note}): {count}x ({pct:.1f}%)")
    
    # Analyze response timing
    response_times = []
    for i, output_event in enumerate(outputs):
        output_time = float(output_event['timestamp'])
        
        # Find most recent input before this output
        relevant_inputs = [inp for inp in inputs if float(inp['timestamp']) < output_time]
        if relevant_inputs:
            latest_input = relevant_inputs[-1]
            response_time = output_time - float(latest_input['timestamp'])
            response_times.append(response_time)
    
    if response_times:
        print(f"\n⏱️  Response timing:")
        print(f"   Min response: {min(response_times):.2f}s")
        print(f"   Max response: {max(response_times):.2f}s")
        print(f"   Avg response: {sum(response_times)/len(response_times):.2f}s")
    
    # Verdict
    print(f"\n{'='*80}")
    if len(outputs) / duration_min > 40:
        print(f"✅ VERDICT: HIGH ACTIVITY - {output_rate:.1f} notes/min")
    elif len(outputs) / duration_min > 20:
        print(f"✅ VERDICT: GOOD ACTIVITY - {output_rate:.1f} notes/min")
    else:
        print(f"⚠️  VERDICT: LOW ACTIVITY - {output_rate:.1f} notes/min")
    
    print(f"{'='*80}\n")

def midi_to_note(midi_num):
    """Convert MIDI number to note name"""
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (midi_num // 12) - 1
    note = notes[midi_num % 12]
    return f"{note}{octave}"

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python analyze_conversation.py <conversation.csv>")
        sys.exit(1)
    
    analyze_conversation_log(sys.argv[1])
