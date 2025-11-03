#!/usr/bin/env python3
"""
Test arc structure learning from training audio

This tests the Brandtsegg-based section analysis integrated with
the performance timeline manager's arc learning.

Usage:
    python test_arc_learning.py --file input_audio/Georgia.wav
    python test_arc_learning.py --file input_audio/Daybreak.wav --section-duration 120
"""

import argparse
from performance_timeline_manager import PerformanceTimelineManager, PerformanceConfig
import json


def main():
    parser = argparse.ArgumentParser(description='Test arc structure learning from audio')
    parser.add_argument('--file', type=str, required=True, help='Training audio file')
    parser.add_argument('--section-duration', type=float, default=60.0,
                       help='Section duration in seconds (default: 60)')
    parser.add_argument('--output', type=str, default=None,
                       help='Optional: Save learned structure to JSON')
    
    args = parser.parse_args()
    
    print("🎭 ARC STRUCTURE LEARNING TEST")
    print("=" * 70)
    
    # Create minimal config (no existing arc file)
    config = PerformanceConfig(
        duration_minutes=5,  # Arbitrary - won't use this for training analysis
        arc_file_path=None,  # No existing arc
        engagement_profile="balanced",
        silence_tolerance=5.0,
        adaptation_rate=0.1
    )
    
    # Create timeline manager
    timeline_manager = PerformanceTimelineManager(config)
    
    # Learn arc structure from training audio
    print(f"\n📂 Learning from: {args.file}")
    timeline_manager.initialize_from_training_audio(
        args.file,
        section_duration=args.section_duration
    )
    
    # Verify learned structure
    if not hasattr(timeline_manager, 'learned_sections') or not timeline_manager.learned_sections:
        print("❌ No sections learned!")
        return
    
    sections = timeline_manager.learned_sections
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("📊 LEARNED ARC STRUCTURE SUMMARY")
    print("=" * 70)
    
    tempos = [s['local_tempo'] for s in sections]
    complexities = [s['rhythmic_complexity'] for s in sections]
    densities = [s['onset_density'] for s in sections]
    engagements = [s['engagement_level'] for s in sections]
    
    print(f"\nTotal sections: {len(sections)}")
    print(f"Duration: {sections[-1]['end_time']:.0f}s ({sections[-1]['end_time']/60:.1f} min)")
    
    print(f"\nTempo: {min(tempos):.1f} - {max(tempos):.1f} BPM (avg: {sum(tempos)/len(tempos):.1f})")
    print(f"Complexity: {min(complexities):.2f} - {max(complexities):.2f} (avg: {sum(complexities)/len(complexities):.2f})")
    print(f"Density: {min(densities):.2f} - {max(densities):.2f} onsets/s")
    print(f"Engagement: {min(engagements):.2f} - {max(engagements):.2f}")
    
    # Phase distribution
    phases = {}
    for s in sections:
        phase = s['arc_phase']
        phases[phase] = phases.get(phase, 0) + 1
    
    print(f"\nPhase distribution:")
    for phase in ['opening', 'development', 'peak', 'resolution', 'closing']:
        count = phases.get(phase, 0)
        pct = (count / len(sections)) * 100
        print(f"   {phase:12s}: {count:2d} sections ({pct:4.1f}%)")
    
    # Test section context retrieval at different time points
    print("\n" + "=" * 70)
    print("🎯 SECTION CONTEXT AT KEY MOMENTS")
    print("=" * 70)
    
    test_times = [
        0,  # Start
        sections[-1]['end_time'] * 0.25,  # 25%
        sections[-1]['end_time'] * 0.50,  # 50% (likely peak)
        sections[-1]['end_time'] * 0.75,  # 75%
        sections[-1]['end_time'] * 0.95   # Near end
    ]
    
    for elapsed_time in test_times:
        context = timeline_manager.get_section_context(elapsed_time)
        
        if context:
            progress_pct = (elapsed_time / sections[-1]['end_time']) * 100
            print(f"\n⏱️  At {elapsed_time:.0f}s ({progress_pct:.0f}% through recording):")
            print(f"   Phase: {context['arc_phase']}")
            print(f"   Local tempo: {context['local_tempo']:.1f} BPM")
            print(f"   Engagement: {context['engagement_level']:.2f}")
            print(f"   Complexity: {context['rhythmic_complexity']:.2f}")
            print(f"   Density: {context['onset_density']:.2f} onsets/s")
            print(f"   Section progress: {context['section_progress']*100:.0f}%")
            
            if context['tempo_change']:
                change_str = f"{'+' if context['tempo_change'] > 1 else ''}{(context['tempo_change']-1)*100:.0f}%"
                print(f"   Tempo change: {context['tempo_change']:.2f}x ({change_str})")
    
    # Show detailed section breakdown
    print("\n" + "=" * 70)
    print("📋 DETAILED SECTION BREAKDOWN")
    print("=" * 70)
    
    for i, s in enumerate(sections):
        tempo_change_str = f" (→{s['tempo_change']:.2f}x)" if s['tempo_change'] else ""
        print(f"\n{i+1:2d}. Section {s['start_time']:5.0f}s - {s['end_time']:5.0f}s")
        print(f"    Phase: {s['arc_phase']:12s} | Engagement: {s['engagement_level']:.2f}")
        print(f"    Tempo: {s['local_tempo']:5.1f} BPM{tempo_change_str}")
        print(f"    Complexity: {s['rhythmic_complexity']:.2f} | Density: {s['onset_density']:.2f}")
        
        # Show first 2 rhythm patterns
        if s['dominant_ratios']:
            pattern_strs = [str(p[:10] if len(p) > 10 else p) for p in s['dominant_ratios'][:2]]
            print(f"    Rhythm patterns: {', '.join(pattern_strs)}")
    
    # Save to JSON if requested
    if args.output:
        output_data = {
            'source_audio': args.file,
            'section_duration': args.section_duration,
            'total_duration': sections[-1]['end_time'],
            'num_sections': len(sections),
            'tempo_range': [min(tempos), max(tempos)],
            'complexity_range': [min(complexities), max(complexities)],
            'phase_distribution': phases,
            'sections': sections
        }
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n💾 Learned structure saved to: {args.output}")
    
    print("\n✅ Arc learning test complete!")


if __name__ == "__main__":
    main()
