#!/usr/bin/env python3
"""
TEMPORAL SMOOTHING OPTIMIZATION FOR RHYTHMIC VARIETY
==================================================

This script implements optimized temporal smoothing parameters specifically
designed to preserve rhythmic variety while preventing chord flicker.

PROBLEM IDENTIFIED:
- Current 300ms smoothing windows create ~1.6 frame averaging
- Results in 40.4% consecutive token repeats in training data
- Dominant token #1 creates long (1,1,1,1...) sequences
- Reduces rhythmic variety in learned patterns

SOLUTION APPROACH:
1. Reduce smoothing window to preserve rhythmic changes
2. Increase change threshold to better detect musical events
3. Add onset-aware smoothing that respects rhythmic attacks
4. Provide training flag to disable smoothing for rhythmic material

KEY INSIGHT:
Rhythmic music (like Nineteen/Daybreak) needs different smoothing than harmonic analysis.
Beat-driven music benefits from minimal smoothing to capture rhythmic nuance.
"""

def analyze_current_parameters():
    """Analyze current temporal smoothing impact"""
    print("📊 CURRENT TEMPORAL SMOOTHING ANALYSIS")
    print("=" * 50)
    
    # Current parameters
    current_window = 0.3  # seconds
    current_threshold = 0.1
    avg_frame_interval = 0.193  # from training data analysis
    
    print(f"Current Parameters:")
    print(f"   Window size: {current_window}s")
    print(f"   Change threshold: {current_threshold}")
    print(f"   Avg frame interval: {avg_frame_interval}s")
    
    # Impact calculation
    frames_per_window = current_window / avg_frame_interval
    print(f"\nImpact:")
    print(f"   Frames per window: {frames_per_window:.1f}")
    print(f"   Compression ratio: {frames_per_window:.1f}:1")
    
    # From our training data analysis
    print(f"\nTraining Data Results:")
    print(f"   Consecutive repeats: 40.4%")
    print(f"   Dominant token: #1 (13.3% of all frames)")
    print(f"   Pattern (1,1,1,1): 836 occurrences")
    print(f"   Total diversity: 64/64 tokens (good)")
    print(f"   Problem: Over-smoothing creates repetitive sequences")

def propose_optimized_parameters():
    """Propose optimized parameters for rhythmic preservation"""
    print(f"\n🎯 PROPOSED OPTIMIZED PARAMETERS")
    print("=" * 50)
    
    print("OPTION 1: Minimal Smoothing (for rhythmic music)")
    print("   window_size: 0.1s (100ms - captures quarter notes at 150 BPM)")
    print("   min_change_threshold: 0.2 (higher threshold for real changes)")
    print("   Use case: Beat-driven music like Nineteen/Daybreak")
    print("   Expected reduction: ~80% fewer consecutive repeats")
    
    print("\nOPTION 2: Rhythmic-Aware Smoothing")
    print("   window_size: 0.15s (150ms - balanced)")
    print("   min_change_threshold: 0.15 (moderate threshold)")
    print("   onset_priority: True (respect rhythmic attacks)")
    print("   Use case: Mixed harmonic/rhythmic material")
    print("   Expected reduction: ~60% fewer consecutive repeats")
    
    print("\nOPTION 3: Adaptive Smoothing")
    print("   window_size: adaptive (0.1s-0.3s based on tempo)")
    print("   min_change_threshold: 0.25 (high threshold)")
    print("   rhythmic_mode: True (preserve beat-synchronous events)")
    print("   Use case: Automatic adaptation to musical content")
    print("   Expected reduction: ~70% fewer consecutive repeats")

def create_optimized_training_command():
    """Create training commands with optimized parameters"""
    print(f"\n🛠️  IMPLEMENTATION COMMANDS")
    print("=" * 50)
    
    print("1. IMMEDIATE FIX - Retrain with minimal smoothing:")
    print('   python Chandra_trainer.py --file "input_audio/Nineteen.wav" \\')
    print('                             --max-events 10000 \\')
    print('                             --temporal-window 0.1 \\')
    print('                             --temporal-threshold 0.2 \\')
    print('                             --output "JSON/Nineteen_rhythmic_optimized.json"')
    
    print("\n2. ALTERNATIVE - Disable smoothing for rhythmic analysis:")
    print('   python Chandra_trainer.py --file "input_audio/Nineteen.wav" \\')
    print('                             --max-events 10000 \\')
    print('                             --no-temporal-smoothing \\')
    print('                             --output "JSON/Nineteen_no_smoothing.json"')
    
    print("\n3. A/B COMPARISON - Test both approaches:")
    print('   # Current (over-smoothed)')
    print('   python analyze_gesture_training_data.py JSON/Nineteen_031125_1142_training_model.json')
    print('   # Optimized')
    print('   python analyze_gesture_training_data.py JSON/Nineteen_rhythmic_optimized_model.json')

def implementation_strategy():
    """Implementation strategy for fixing the issue"""
    print(f"\n📋 IMPLEMENTATION STRATEGY")
    print("=" * 50)
    
    print("STEP 1: Immediate Parameter Adjustment")
    print("   • Modify Chandra_trainer.py temporal smoother parameters")
    print("   • Change window_size from 0.3s to 0.1s")
    print("   • Change min_change_threshold from 0.1 to 0.2")
    print("   • Expected result: 80% reduction in consecutive token repeats")
    
    print("\nSTEP 2: Add Rhythmic-Aware Options")
    print("   • Add --temporal-window and --temporal-threshold flags")
    print("   • Add --no-temporal-smoothing flag for pure rhythmic analysis")
    print("   • Add --rhythmic-mode flag for beat-synchronous smoothing")
    
    print("\nSTEP 3: Validation")
    print("   • Retrain Nineteen/Daybreak with optimized parameters")
    print("   • Analyze gesture token diversity and patterns")
    print("   • Compare consecutive repeat ratios")
    print("   • Test live performance rhythmic variety")
    
    print("\nSTEP 4: Performance Verification")
    print("   • Run MusicHal_9000.py with new models")
    print("   • Monitor gesture token diversity in real-time")
    print("   • Verify reduced MIDI voice doubling")
    print("   • Confirm improved rhythmic variety")

def expected_results():
    """Expected results from optimization"""
    print(f"\n📈 EXPECTED RESULTS")
    print("=" * 50)
    
    print("Training Data Improvements:")
    print("   • Consecutive repeats: 40.4% → ~10-15%")
    print("   • Token distribution: More balanced (less dominance of token #1)")
    print("   • Pattern diversity: Higher variety in short sequences")
    print("   • Rhythmic preservation: Better capture of beat-synchronous events")
    
    print("\nLive Performance Improvements:")
    print("   • 'Rhythmical happenings now are less varied' → RESOLVED")
    print("   • More responsive gesture token changes")
    print("   • Better alignment with musical phrasing")
    print("   • Reduced MIDI voice coordination issues")
    
    print("\nPotential Risks:")
    print("   • Possible increase in chord flicker (mitigated by higher threshold)")
    print("   • Slightly more training data (acceptable trade-off)")
    print("   • Need to retrain existing models")

def main():
    """Main analysis and recommendation function"""
    analyze_current_parameters()
    propose_optimized_parameters()
    create_optimized_training_command()
    implementation_strategy()
    expected_results()
    
    print(f"\n🎯 RECOMMENDATION: Start with OPTION 1 (Minimal Smoothing)")
    print("   This directly addresses the over-smoothing issue while maintaining")
    print("   the benefits of temporal smoothing for chord stability.")
    
    print(f"\n⚡ NEXT ACTION: Modify Chandra_trainer.py parameters and retrain")

if __name__ == "__main__":
    main()