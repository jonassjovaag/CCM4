#!/usr/bin/env python3
"""
Regenerate harmonic transition graph from existing training session.
Use this after fixing the json variable shadowing bug.
"""

import json
import sys
from pathlib import Path

def regenerate_transition_graph(training_json_path):
    """
    Load training session data and rebuild harmonic transition graph.
    
    Args:
        training_json_path: Path to the *_training.json file
    """
    print(f"📂 Loading training session from {training_json_path}...")
    
    # Load training session data
    with open(training_json_path, 'r') as f:
        training_data = json.load(f)
    
    # Extract events with chord labels
    events = training_data.get('events', [])
    print(f"📊 Found {len(events)} events in training session")
    
    # Count events with chord data
    events_with_chords = [e for e in events if 'chord_label' in e and e['chord_label']]
    print(f"🎵 {len(events_with_chords)} events have chord labels")
    
    if len(events_with_chords) < 2:
        print("❌ Insufficient chord data to build transition graph")
        return None
    
    # Build transition graph
    transitions = {}
    chord_durations = {}
    chord_counts = {}
    unique_chords = set()
    
    print("🔨 Building transition graph...")
    
    for i in range(len(events_with_chords) - 1):
        current = events_with_chords[i]
        next_event = events_with_chords[i + 1]
        
        from_chord = current['chord_label']
        to_chord = next_event['chord_label']
        duration = next_event['time'] - current['time']
        
        unique_chords.add(from_chord)
        unique_chords.add(to_chord)
        
        # Track transition
        key = f"{from_chord} -> {to_chord}"
        if key not in transitions:
            transitions[key] = {
                'from': from_chord,
                'to': to_chord,
                'count': 0,
                'total_duration': 0.0
            }
        
        transitions[key]['count'] += 1
        transitions[key]['total_duration'] += duration
        
        # Track chord durations
        if from_chord not in chord_durations:
            chord_durations[from_chord] = []
            chord_counts[from_chord] = 0
        chord_durations[from_chord].append(duration)
        chord_counts[from_chord] += 1
    
    # Calculate probabilities
    for key, data in transitions.items():
        from_chord = data['from']
        total_from = chord_counts[from_chord]
        data['probability'] = data['count'] / total_from if total_from > 0 else 0.0
        data['avg_duration'] = data['total_duration'] / data['count']
    
    # Build final graph structure
    graph = {
        'unique_chords': sorted(list(unique_chords)),
        'total_transitions': len(transitions),
        'transitions': transitions,
        'chord_statistics': {
            chord: {
                'count': chord_counts[chord],
                'avg_duration': sum(chord_durations[chord]) / len(chord_durations[chord]),
                'min_duration': min(chord_durations[chord]),
                'max_duration': max(chord_durations[chord])
            }
            for chord in unique_chords
        }
    }
    
    print(f"✅ Transition graph built:")
    print(f"   • {len(unique_chords)} unique chords")
    print(f"   • {len(transitions)} transition types")
    print(f"   • {len(events_with_chords)} total chord events")
    
    # Show top transitions
    sorted_transitions = sorted(
        transitions.items(),
        key=lambda x: x[1]['count'],
        reverse=True
    )
    
    if sorted_transitions:
        print(f"   Top 5 transitions:")
        for key, data in sorted_transitions[:5]:
            print(f"      {key}: {data['probability']:.1%} ({data['count']} times)")
    
    return graph


def main():
    if len(sys.argv) != 2:
        print("Usage: python regenerate_harmonic_transitions.py <training_json_file>")
        print("Example: python regenerate_harmonic_transitions.py JSON/Curious_child_091125_1912_training.json")
        sys.exit(1)
    
    training_json = sys.argv[1]
    
    if not Path(training_json).exists():
        print(f"❌ File not found: {training_json}")
        sys.exit(1)
    
    # Generate transition graph
    graph = regenerate_transition_graph(training_json)
    
    if graph is None:
        print("❌ Failed to build transition graph")
        sys.exit(1)
    
    # Determine output filename
    output_file = training_json.replace('_training.json', '_training_harmonic_transitions.json')
    
    print(f"\n💾 Saving harmonic transition graph to {output_file}...")
    
    try:
        with open(output_file, 'w') as f:
            json.dump(graph, f, indent=2)
        print(f"✅ Harmonic transition graph saved successfully!")
        print(f"   File size: {Path(output_file).stat().st_size / 1024:.1f} KB")
        print(f"\n📌 Use this with HarmonicProgressor for intelligent chord selection")
    except Exception as e:
        print(f"❌ Failed to save transition graph: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
