#!/usr/bin/env python3
"""
Rhythmic Engine Demo
Demonstrates the complete rhythmic analysis system for MusicHal 9000

This script shows:
- Heavy rhythmic analysis for audio files
- Lightweight rhythmic analysis for live performance
- Rhythmic pattern memory and retrieval
- Rhythmic decision making
- Integration with existing system
"""

import numpy as np
import os
import sys
import time

# Add parent directory to path for imports
sys.path.append('..')

from audio_file_learning.heavy_rhythmic_analyzer import HeavyRhythmicAnalyzer
from audio_file_learning.lightweight_rhythmic_analyzer import LightweightRhythmicAnalyzer
from memory.rhythm_oracle import RhythmOracle
from agent.rhythmic_behavior_engine import RhythmicBehaviorEngine

def demo_heavy_analysis():
    """Demonstrate heavy rhythmic analysis"""
    print("🥁 === HEAVY RHYTHMIC ANALYSIS DEMO ===")
    
    analyzer = HeavyRhythmicAnalyzer()
    
    # Test with available audio file
    audio_file = "../input_audio/Grab-a-hold.mp3"
    
    if os.path.exists(audio_file):
        print(f"Analyzing: {audio_file}")
        
        try:
            analysis = analyzer.analyze_rhythmic_structure(audio_file)
            
            print(f"\n📊 Analysis Results:")
            print(f"   Tempo: {analysis.tempo:.1f} BPM")
            print(f"   Meter: {analysis.meter}")
            print(f"   Syncopation: {analysis.syncopation_score:.3f}")
            print(f"   Complexity: {analysis.rhythmic_complexity:.3f}")
            print(f"   Patterns detected: {len(analysis.patterns)}")
            
            # Show first few patterns
            for i, pattern in enumerate(analysis.patterns[:3]):
                print(f"   Pattern {i+1}: {pattern.pattern_type} "
                      f"({pattern.start_time:.1f}-{pattern.end_time:.1f}s, "
                      f"conf: {pattern.confidence:.3f})")
            
            return analysis
            
        except Exception as e:
            print(f"Error in analysis: {e}")
            return None
    else:
        print(f"Audio file not found: {audio_file}")
        return None

def demo_lightweight_analysis():
    """Demonstrate lightweight rhythmic analysis"""
    print("\n🥁 === LIGHTWEIGHT RHYTHMIC ANALYSIS DEMO ===")
    
    analyzer = LightweightRhythmicAnalyzer()
    
    print("Simulating live audio frames...")
    
    # Simulate different types of audio frames
    frame_types = [
        ("Silent", np.random.randn(512) * 0.01),
        ("Quiet", np.random.randn(512) * 0.1),
        ("Moderate", np.random.randn(512) * 0.3),
        ("Loud", np.random.randn(512) * 0.8),
        ("Very Loud", np.random.randn(512) * 1.2)
    ]
    
    for frame_type, frame in frame_types:
        context = analyzer.analyze_live_rhythm(frame)
        
        print(f"   {frame_type:10}: "
              f"Onset: {context.onset_detected:5}, "
              f"Tempo: {context.tempo:5.1f}, "
              f"Density: {context.rhythmic_density:.2f}, "
              f"Beat: {context.beat_position:.2f}")
        
        time.sleep(0.1)  # Simulate real-time

def demo_rhythm_oracle():
    """Demonstrate rhythmic pattern memory"""
    print("\n🥁 === RHYTHM ORACLE DEMO ===")
    
    oracle = RhythmOracle()
    
    # Add some test patterns
    test_patterns = [
        {
            'tempo': 120.0,
            'density': 0.8,
            'syncopation': 0.2,
            'pattern_type': 'dense',
            'confidence': 0.9
        },
        {
            'tempo': 140.0,
            'density': 0.4,
            'syncopation': 0.6,
            'pattern_type': 'syncopated',
            'confidence': 0.8
        },
        {
            'tempo': 100.0,
            'density': 0.2,
            'syncopation': 0.1,
            'pattern_type': 'sparse',
            'confidence': 0.7
        },
        {
            'tempo': 160.0,
            'density': 0.9,
            'syncopation': 0.3,
            'pattern_type': 'complex',
            'confidence': 0.85
        }
    ]
    
    print("Adding patterns to memory...")
    for pattern_data in test_patterns:
        pattern_id = oracle.add_rhythmic_pattern(pattern_data)
        print(f"   Added: {pattern_id} ({pattern_data['pattern_type']})")
    
    # Test similarity search
    print("\nTesting similarity search...")
    query = {
        'tempo': 125.0,
        'density': 0.7,
        'syncopation': 0.3,
        'pattern_type': 'dense'
    }
    
    similar = oracle.find_similar_patterns(query, threshold=0.5)
    print(f"   Found {len(similar)} similar patterns:")
    for pattern, similarity in similar:
        print(f"     {pattern.pattern_id}: {similarity:.3f} ({pattern.pattern_type})")
    
    # Test prediction
    print("\nTesting pattern prediction...")
    current_context = {
        'tempo': 120.0,
        'density': 0.8,
        'syncopation': 0.2
    }
    
    predicted = oracle.predict_next_pattern(current_context)
    if predicted:
        print(f"   Predicted next: {predicted.pattern_id} ({predicted.pattern_type})")
    else:
        print("   No prediction available")
    
    # Show statistics
    stats = oracle.get_rhythmic_statistics()
    print(f"\n📊 Oracle Statistics:")
    print(f"   Total patterns: {stats['total_patterns']}")
    print(f"   Average tempo: {stats['avg_tempo']:.1f} BPM")
    print(f"   Average density: {stats['avg_density']:.2f}")
    print(f"   Pattern types: {stats['pattern_types']}")

def demo_rhythmic_behavior():
    """Demonstrate rhythmic decision making"""
    print("\n🥁 === RHYTHMIC BEHAVIOR DEMO ===")
    
    engine = RhythmicBehaviorEngine()
    
    # Test different rhythmic contexts
    test_contexts = [
        {
            'tempo': 120.0,
            'rhythmic_density': 0.8,
            'beat_position': 0.0,
            'confidence': 0.9
        },
        {
            'tempo': 140.0,
            'rhythmic_density': 0.3,
            'beat_position': 0.5,
            'confidence': 0.7
        },
        {
            'tempo': 100.0,
            'rhythmic_density': 0.6,
            'beat_position': 0.25,
            'confidence': 0.8
        }
    ]
    
    # Test patterns
    test_patterns = [
        {
            'tempo': 120.0,
            'density': 0.7,
            'pattern_type': 'dense'
        },
        {
            'tempo': 140.0,
            'density': 0.4,
            'pattern_type': 'syncopated'
        }
    ]
    
    print("Testing rhythmic decisions...")
    for i, context in enumerate(test_contexts):
        print(f"\n   Context {i+1}: Tempo={context['tempo']}, Density={context['rhythmic_density']:.1f}")
        
        decision = engine.decide_rhythmic_response(context, test_patterns)
        
        print(f"   Decision: Play={decision.should_play}, "
              f"Timing={decision.timing:.2f}, "
              f"Density={decision.density:.2f}, "
              f"Type={decision.pattern_type}")
        print(f"   Reasoning: {decision.reasoning}")
    
    # Show behavior state
    state = engine.get_behavior_state()
    print(f"\n📊 Behavior State:")
    print(f"   Current mode: {state['current_mode']}")
    print(f"   Mode history: {state['mode_history_length']} decisions")

def demo_integration():
    """Demonstrate integration with existing system"""
    print("\n🥁 === INTEGRATION DEMO ===")
    
    # Simulate integration with existing system
    print("Simulating integration with existing MusicHal 9000 system...")
    
    # Create components
    heavy_analyzer = HeavyRhythmicAnalyzer()
    lightweight_analyzer = LightweightRhythmicAnalyzer()
    oracle = RhythmOracle()
    behavior_engine = RhythmicBehaviorEngine()
    
    # Simulate training from audio file
    print("\n1. Training from audio file:")
    print("   - Heavy analysis extracts rhythmic patterns")
    print("   - Patterns stored in RhythmOracle")
    print("   - System learns rhythmic transitions")
    
    # Simulate live performance
    print("\n2. Live performance:")
    print("   - Lightweight analysis provides real-time context")
    print("   - Behavior engine makes rhythmic decisions")
    print("   - Decisions integrated with harmonic analysis")
    
    # Show integration benefits
    print("\n3. Integration benefits:")
    print("   ✅ Parallel processing (harmonic + rhythmic)")
    print("   ✅ Dual training (audio files + live input)")
    print("   ✅ Sophisticated silence/activity decisions")
    print("   ✅ Rhythmic pattern matching and generation")
    print("   ✅ Real-time tempo and density adaptation")

def main():
    """Run the complete rhythmic engine demo"""
    print("🎵 MusicHal 9000 - Rhythmic Engine Demo")
    print("=" * 50)
    
    # Run all demos
    demo_heavy_analysis()
    demo_lightweight_analysis()
    demo_rhythm_oracle()
    demo_rhythmic_behavior()
    demo_integration()
    
    print("\n🎉 Rhythmic Engine Demo Complete!")
    print("\nThe rhythmic engine is now ready for integration with MusicHal 9000!")
    print("Key features implemented:")
    print("  • Heavy rhythmic analysis for training")
    print("  • Lightweight rhythmic analysis for live performance")
    print("  • Rhythmic pattern memory and retrieval")
    print("  • Sophisticated rhythmic decision making")
    print("  • Seamless integration with existing harmonic system")

if __name__ == "__main__":
    main()
