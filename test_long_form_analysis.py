#!/usr/bin/env python3
"""
Test script for long-form improvisation analysis with Brandtsegg ratios

Usage:
    python test_long_form_analysis.py --file input_audio/Daybreak.wav
    python test_long_form_analysis.py --file input_audio/Georgia.wav --section-duration 30
"""

import argparse
from rhythmic_engine.audio_file_learning.heavy_rhythmic_analyzer import HeavyRhythmicAnalyzer
import json


def main():
    parser = argparse.ArgumentParser(description='Analyze long-form improvisation structure')
    parser.add_argument('--file', type=str, required=True, help='Audio file to analyze')
    parser.add_argument('--section-duration', type=float, default=60.0, 
                       help='Section duration in seconds (default: 60)')
    parser.add_argument('--output', type=str, default=None,
                       help='Optional: Save results to JSON file')
    
    args = parser.parse_args()
    
    print("🎭 Long-Form Improvisation Analysis")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = HeavyRhythmicAnalyzer()
    
    # Analyze
    sections = analyzer.analyze_long_form_improvisation(
        args.file,
        section_duration=args.section_duration
    )
    
    # Summary
    print("\n" + "=" * 60)
    print(f"📊 ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Total sections: {len(sections)}")
    
    if len(sections) > 0:
        tempos = [s['local_tempo'] for s in sections]
        complexities = [s['rhythmic_complexity'] for s in sections]
        densities = [s['onset_density'] for s in sections]
        
        print(f"\nTempo range: {min(tempos):.1f} - {max(tempos):.1f} BPM")
        print(f"Average tempo: {sum(tempos)/len(tempos):.1f} BPM")
        print(f"\nComplexity range: {min(complexities):.2f} - {max(complexities):.2f}")
        print(f"Average complexity: {sum(complexities)/len(complexities):.2f}")
        print(f"\nDensity range: {min(densities):.2f} - {max(densities):.2f} onsets/s")
        print(f"Average density: {sum(densities)/len(densities):.2f} onsets/s")
        
        # Detect tempo changes
        tempo_changes = [s for s in sections if s['tempo_change'] and abs(s['tempo_change'] - 1.0) > 0.15]
        if tempo_changes:
            print(f"\n🎵 Significant tempo changes detected: {len(tempo_changes)}")
            for s in tempo_changes[:5]:  # Show first 5
                print(f"   {s['start_time']:.0f}s: {s['local_tempo']:.1f} BPM "
                      f"({s['tempo_change']:.2f}x = {'+' if s['tempo_change'] > 1 else ''}{(s['tempo_change']-1)*100:.0f}%)")
        
        # Show section details
        print(f"\n📋 SECTION DETAILS:")
        print("-" * 60)
        for i, s in enumerate(sections):
            tempo_change_str = f" (→{s['tempo_change']:.2f}x)" if s['tempo_change'] else ""
            print(f"{i+1:2d}. {s['start_time']:5.0f}s-{s['end_time']:5.0f}s: "
                  f"{s['local_tempo']:5.1f} BPM{tempo_change_str:12s} "
                  f"| complexity {s['rhythmic_complexity']:.2f} "
                  f"| density {s['onset_density']:.2f}")
            
            # Show dominant ratios
            if s['dominant_ratios']:
                ratio_str = ", ".join([str(r) for r in s['dominant_ratios'][:2]])
                print(f"      Rhythm patterns: {ratio_str}")
    
    # Save to JSON if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump({
                'audio_file': args.file,
                'section_duration': args.section_duration,
                'sections': sections
            }, f, indent=2)
        print(f"\n💾 Results saved to: {args.output}")
    
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
