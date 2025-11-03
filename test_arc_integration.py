#!/usr/bin/env python3
"""
Quick test of arc structure flag integration

Tests that --analyze-arc-structure flag works in Chandra_trainer.py
"""

import sys
sys.path.insert(0, '.')

from Chandra_trainer import EnhancedHybridTrainingPipeline

# Create minimal pipeline
pipeline = EnhancedHybridTrainingPipeline(
    max_events=100,
    enable_hierarchical=False,
    enable_rhythmic=True,
    enable_gpt_oss=False,
    enable_hybrid_perception=False,
    enable_wav2vec=False
)

print("🧪 Testing arc structure analysis integration...")
print("=" * 60)

try:
    # Test arc structure analysis step
    result = pipeline.train_from_audio_file(
        audio_file="input_audio/Georgia.wav",
        output_file="test_arc_integration.json",
        max_events=100,
        training_events=20,
        use_transformer=False,
        use_hierarchical=False,
        use_rhythmic=True,
        analyze_arc=True,
        section_duration=30.0
    )
    
    # Check if arc structure was saved
    if 'arc_structure' in result:
        print("\n✅ Arc structure analysis integrated successfully!")
        print(f"   Sections detected: {result['arc_structure']['num_sections']}")
        print(f"   Duration: {result['arc_structure']['total_duration']:.0f}s")
        print(f"   Tempo range: {result['arc_structure']['tempo_range']}")
        print(f"   Phase distribution: {result['arc_structure']['phase_distribution']}")
    else:
        print("\n❌ Arc structure not found in results!")
        
except Exception as e:
    print(f"\n❌ Test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("✅ Integration test complete!")
