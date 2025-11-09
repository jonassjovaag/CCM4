"""
Test 1: Model Structure Inspection (Scientific)
================================================
Comprehensive statistical analysis of trained model structure.

Purpose:
- Validate model architecture and serialization format
- Compute statistical distributions (gesture tokens, consonance, pitch)
- Answer Research Questions Q1, Q2, Q3 from SCIENTIFIC_TEST_SUITE_PLAN.md
- Document actual field names and data structures

This test is READ-ONLY: never modifies production code or models.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import test utilities
from tests.utils.test_helpers import (
    load_trained_model,
    compute_model_statistics,
    print_statistics_summary,
    save_test_results,
    assert_model_structure,
)
from tests.config import (
    REFERENCE_MODEL_PATH,
    STATISTICAL_THRESHOLDS,
    RESEARCH_QUESTIONS,
)


def test_model_inspection():
    """
    Comprehensive model structure inspection with statistical analysis.
    
    Tests:
    1. Model loads successfully (dict serialization format)
    2. Required structure present (audio_frames, states, transitions, suffix_links)
    3. Statistical distributions meet expectations
    4. Answers Research Questions Q1, Q2, Q3
    
    Returns:
        bool: True if all checks pass, False otherwise
    """
    
    print("=" * 80)
    print("TEST 1: SCIENTIFIC MODEL INSPECTION")
    print("=" * 80)
    
    try:
        # === STEP 1: Load Model ===
        print("\n📂 Step 1: Loading trained model...")
        model_data = load_trained_model(REFERENCE_MODEL_PATH)
        
        model_size_mb = REFERENCE_MODEL_PATH.stat().st_size / (1024 * 1024)
        print(f"   ✅ Model loaded: {REFERENCE_MODEL_PATH.name}")
        print(f"   File size: {model_size_mb:.2f} MB")
        print(f"   Format: {type(model_data).__name__}")
        
        # === STEP 2: Validate Structure ===
        print("\n🔍 Step 2: Validating model structure...")
        assert_model_structure(model_data)
        print("   ✅ All required keys present")
        print(f"   ✅ audio_frames is dict: {isinstance(model_data['audio_frames'], dict)}")
        print(f"   ✅ feature_dimensions: {model_data['feature_dimensions']}")
        print(f"   ✅ distance_threshold: {model_data['distance_threshold']}")
        
        # === STEP 3: Compute Statistics ===
        print("\n📊 Step 3: Computing comprehensive statistics...")
        stats = compute_model_statistics(model_data)
        print_statistics_summary(stats)
        
        # === STEP 4: Answer Research Questions ===
        print("\n� Step 4: Answering Research Questions...")
        
        # Q1: How many unique gesture tokens?
        unique_tokens = stats['unique_gesture_tokens']
        print(f"\n   Q1: {RESEARCH_QUESTIONS['Q1']}")
        print(f"       Expected: {RESEARCH_QUESTIONS['Q1_expected']}")
        print(f"       Actual: {unique_tokens} unique tokens")
        print(f"       Status: {'✅ PASS' if unique_tokens >= STATISTICAL_THRESHOLDS['min_unique_tokens'] else '❌ FAIL'}")
        
        # Q2: What is consonance distribution?
        if stats['consonance_stats']:
            cons_mean = stats['consonance_stats']['mean']
            cons_std = stats['consonance_stats']['std']
            cons_range = (stats['consonance_stats']['min'], stats['consonance_stats']['max'])
            
            print(f"\n   Q2: {RESEARCH_QUESTIONS['Q2']}")
            print(f"       Expected: {RESEARCH_QUESTIONS['Q2_expected']}")
            print(f"       Actual: Mean={cons_mean:.3f} ± {cons_std:.3f}, Range={cons_range}")
            
            cons_variation_ok = cons_std >= STATISTICAL_THRESHOLDS['consonance_std_min']
            cons_mean_ok = (STATISTICAL_THRESHOLDS['consonance_mean_min'] <= cons_mean <= 
                           STATISTICAL_THRESHOLDS['consonance_mean_max'])
            print(f"       Status: {'✅ PASS' if (cons_variation_ok and cons_mean_ok) else '❌ FAIL'}")
        
        # Q3: How dense is the AudioOracle graph?
        avg_transitions = stats['transition_count'] / max(stats['state_count'], 1)
        suffix_coverage = stats['suffix_link_count'] / max(stats['state_count'], 1)
        
        print(f"\n   Q3: {RESEARCH_QUESTIONS['Q3']}")
        print(f"       Expected: {RESEARCH_QUESTIONS['Q3_expected']}")
        print(f"       Actual: Avg {avg_transitions:.2f} transitions/state, {suffix_coverage:.1%} suffix link coverage")
        
        graph_density_ok = avg_transitions >= STATISTICAL_THRESHOLDS['min_avg_transitions_per_state']
        suffix_ok = suffix_coverage >= STATISTICAL_THRESHOLDS['min_suffix_link_coverage']
        print(f"       Status: {'✅ PASS' if (graph_density_ok and suffix_ok) else '❌ FAIL'}")
        
        # === STEP 5: Inspect MIDI Field Names ===
        print("\n🎹 Step 5: Identifying MIDI field names...")
        audio_frames = model_data['audio_frames']
        frame_ids = sorted(list(audio_frames.keys()))[:20]  # Check first 20 frames
        
        field_counts = {'midi': 0, 'midi_note': 0, 'pitch_hz': 0, 'f0': 0}
        
        for frame_id in frame_ids:
            frame = audio_frames[frame_id]
            if isinstance(frame, dict):
                audio_data = frame.get('audio_data', frame)
            else:
                audio_data = frame.audio_data
            
            for field in field_counts.keys():
                if field in audio_data and audio_data[field] is not None:
                    field_counts[field] += 1
        
        print(f"   Field occurrence in first 20 frames:")
        for field, count in field_counts.items():
            percentage = (count / 20) * 100
            indicator = "✅" if count > 15 else "⚠️" if count > 0 else "❌"
            print(f"      {indicator} '{field}': {count}/20 ({percentage:.0f}%)")
        
        # Identify primary MIDI field
        primary_field = max(field_counts.items(), key=lambda x: x[1])[0]
        print(f"   🎯 Primary MIDI field: '{primary_field}' ({field_counts[primary_field]}/20 frames)")
        
        # === STEP 6: Save Results ===
        print("\n💾 Step 6: Saving test results...")
        
        results = {
            'model_path': str(REFERENCE_MODEL_PATH),
            'model_size_mb': model_size_mb,
            'statistics': stats,
            'research_answers': {
                'Q1_unique_tokens': unique_tokens,
                'Q2_consonance': stats['consonance_stats'] if stats['consonance_stats'] else {},
                'Q3_graph_density': {
                    'avg_transitions_per_state': avg_transitions,
                    'suffix_link_coverage': suffix_coverage,
                },
            },
            'midi_field_analysis': {
                'field_counts': field_counts,
                'primary_field': primary_field,
            },
            'validation': {
                'structure_valid': True,
                'statistics_valid': unique_tokens >= STATISTICAL_THRESHOLDS['min_unique_tokens'],
                'graph_density_valid': graph_density_ok and suffix_ok,
            }
        }
        
        save_test_results('model_inspection', results)
        
        # === FINAL STATUS ===
        all_valid = all(results['validation'].values())
        
        print("\n" + "=" * 80)
        if all_valid:
            print("✅ TEST 1 PASSED: Model structure and statistics validated")
        else:
            print("❌ TEST 1 FAILED: Some validation checks did not pass")
        print("=" * 80)
        
        return all_valid
        
    except Exception as e:
        print(f"\n❌ FAIL: Exception during inspection: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_model_inspection()
    sys.exit(0 if success else 1)
