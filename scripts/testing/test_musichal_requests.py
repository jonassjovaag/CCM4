#!/usr/bin/env python3
"""
Test MusicHal Request-Based Generation

Validates that behavior modes correctly use request-based generation
with the retrained Georgia model.

Usage:
    python test_musichal_requests.py [--model-path PATH]
"""

import argparse
import sys
import json
import numpy as np


def test_phrase_generator_context():
    """Test PhraseGenerator context analysis methods"""
    print("=" * 60)
    print("TEST 1: PhraseGenerator Context Analysis")
    print("=" * 60)
    
    from agent.phrase_generator import PhraseGenerator
    
    # Create phrase generator (no oracles needed for context tests)
    pg = PhraseGenerator(rhythm_oracle=None, audio_oracle=None)
    
    # Add test events
    print("\n📍 Adding test events...")
    test_events = [
        {'gesture_token': 5, 'consonance': 0.8, 'midi': 60, 'midi_relative': 0},
        {'gesture_token': 7, 'consonance': 0.7, 'midi': 62, 'midi_relative': 2},
        {'gesture_token': 10, 'consonance': 0.6, 'midi': 65, 'midi_relative': 3},
        {'gesture_token': 12, 'consonance': 0.9, 'midi': 67, 'midi_relative': 2},
        {'gesture_token': 15, 'consonance': 0.5, 'midi': 64, 'midi_relative': -3},
    ]
    
    for event in test_events:
        pg.track_event(event, source='human')
    
    # Test token extraction
    print("\n📊 Testing _get_recent_human_tokens()...")
    tokens = pg._get_recent_human_tokens(n=3)
    print(f"   Recent tokens (last 3): {tokens}")
    assert len(tokens) == 3, "Should have 3 tokens"
    assert tokens == [10, 12, 15], f"Expected [10, 12, 15], got {tokens}"
    print("   ✅ Token extraction works")
    
    # Test consonance calculation
    print("\n📊 Testing _calculate_avg_consonance()...")
    avg_cons = pg._calculate_avg_consonance(n=5)
    expected_avg = np.mean([0.8, 0.7, 0.6, 0.9, 0.5])
    print(f"   Average consonance: {avg_cons:.3f} (expected: {expected_avg:.3f})")
    assert abs(avg_cons - expected_avg) < 0.01, "Consonance calculation incorrect"
    print("   ✅ Consonance calculation works")
    
    # Test melodic tendency
    print("\n📊 Testing _get_melodic_tendency()...")
    tendency = pg._get_melodic_tendency(n=5)
    expected_tendency = np.mean([0, 2, 3, 2, -3])  # Average interval
    print(f"   Melodic tendency: {tendency:.3f} (expected: {expected_tendency:.3f})")
    assert abs(tendency - expected_tendency) < 0.01, "Melodic tendency incorrect"
    print("   ✅ Melodic tendency works")
    
    print("\n✅ Test 1 PASSED")
    return True


def test_request_builders():
    """Test request builder methods"""
    print("\n" + "=" * 60)
    print("TEST 2: Request Builder Methods")
    print("=" * 60)
    
    from agent.phrase_generator import PhraseGenerator
    
    pg = PhraseGenerator(rhythm_oracle=None, audio_oracle=None)
    
    # Add events with tokens for shadowing
    for i, token in enumerate([5, 7, 10]):
        pg.track_event({'gesture_token': token, 'midi': 60 + i}, source='human')
    
    # Test shadowing request
    print("\n📍 Testing _build_shadowing_request()...")
    shadow_req = pg._build_shadowing_request()
    print(f"   Request: {shadow_req}")
    assert shadow_req is not None, "Should return a request"
    assert shadow_req['parameter'] == 'gesture_token', "Should request gesture_token"
    assert shadow_req['value'] == 10, "Should match last token (10)"
    assert shadow_req['weight'] == 0.8, "Weight should be 0.8"
    print("   ✅ Shadowing request builds correctly")
    
    # Test mirroring request
    print("\n📍 Testing _build_mirroring_request()...")
    # Add events with ascending tendency
    for i in range(5):
        pg.track_event({'midi_relative': 2, 'midi': 60 + i*2}, source='human')
    
    mirror_req = pg._build_mirroring_request()
    print(f"   Request: {mirror_req}")
    assert mirror_req is not None, "Should return a request"
    assert mirror_req['parameter'] == 'midi_relative', "Should request midi_relative"
    assert mirror_req['value'] == -2.0, "Should prefer descending (negative gradient)"
    assert mirror_req['weight'] == 0.7, "Weight should be 0.7"
    print("   ✅ Mirroring request builds correctly")
    
    # Test coupling request
    print("\n📍 Testing _build_coupling_request()...")
    couple_req = pg._build_coupling_request()
    print(f"   Request: {couple_req}")
    assert couple_req is not None, "Should return a request"
    assert couple_req['parameter'] == 'consonance', "Should request consonance"
    assert couple_req['type'] == '>', "Should use > threshold"
    assert couple_req['value'] == 0.7, "Threshold should be 0.7"
    assert couple_req['weight'] == 0.9, "Weight should be 0.9"
    print("   ✅ Coupling request builds correctly")
    
    print("\n✅ Test 2 PASSED")
    return True


def test_model_has_required_fields(model_path: str):
    """Verify that Georgia model has required fields for request-based generation"""
    print("\n" + "=" * 60)
    print("TEST 3: Georgia Model Field Validation")
    print("=" * 60)
    
    print(f"\n📂 Loading model: {model_path}")
    
    try:
        with open(model_path, 'r') as f:
            model_data = json.load(f)
    except Exception as e:
        print(f"   ⚠️  Failed to load model: {e}")
        return False
    
    # Check for audio_frames
    if 'audio_frames' not in model_data:
        print("   ❌ Model missing 'audio_frames'")
        return False
    
    audio_frames = model_data['audio_frames']
    print(f"   ✅ Model has {len(audio_frames)} audio frames")
    
    if len(audio_frames) == 0:
        print("   ⚠️  No audio frames in model")
        return False
    
    # Check first frame for required fields
    first_frame = audio_frames[0]
    audio_data = first_frame.get('audio_data', {})
    
    required_fields = [
        'gesture_token',
        'consonance', 
        'midi_relative',
        'velocity_relative',
        'ioi_relative'
    ]
    
    optional_fields = [
        'rhythm_ratio',
        'deviation',
        'deviation_polarity',
        'tempo_factor'
    ]
    
    print("\n📊 Checking required fields...")
    missing_required = []
    for field in required_fields:
        if field in audio_data:
            value = audio_data[field]
            print(f"   ✅ {field}: {value}")
        else:
            missing_required.append(field)
            print(f"   ❌ {field}: MISSING")
    
    print("\n📊 Checking optional fields (ratio analysis)...")
    missing_optional = []
    for field in optional_fields:
        if field in audio_data:
            value = audio_data[field]
            print(f"   ✅ {field}: {value}")
        else:
            missing_optional.append(field)
            print(f"   ⚠️  {field}: not found")
    
    if missing_required:
        print(f"\n❌ Test 3 FAILED: Missing required fields: {missing_required}")
        print("   Model may need retraining with --hybrid-perception --wav2vec")
        return False
    
    if missing_optional:
        print(f"\n⚠️  Optional fields missing: {missing_optional}")
        print("   Ratio analysis may not have been enabled during training")
    
    print("\n✅ Test 3 PASSED (required fields present)")
    return True


def test_oracle_has_generate_with_request():
    """Test that PolyphonicAudioOracle has generate_with_request method"""
    print("\n" + "=" * 60)
    print("TEST 4: Oracle Method Availability")
    print("=" * 60)
    
    from memory.polyphonic_audio_oracle import PolyphonicAudioOracle
    
    oracle = PolyphonicAudioOracle(feature_dimensions=10)
    
    print("\n📍 Checking for generate_with_request()...")
    if hasattr(oracle, 'generate_with_request'):
        print("   ✅ generate_with_request() method found")
    else:
        print("   ❌ generate_with_request() method missing")
        return False
    
    print("\n📍 Checking for generate_next() (fallback)...")
    if hasattr(oracle, 'generate_next'):
        print("   ✅ generate_next() method found")
    else:
        print("   ⚠️  generate_next() method missing (not critical)")
    
    print("\n✅ Test 4 PASSED")
    return True


def run_all_tests(model_path: str = None):
    """Run all MusicHal integration tests"""
    print("\n" + "=" * 60)
    print("MUSICHAL REQUEST INTEGRATION TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("PhraseGenerator Context", test_phrase_generator_context),
        ("Request Builders", test_request_builders),
        ("Oracle Methods", test_oracle_has_generate_with_request),
    ]
    
    # Add model validation if path provided
    if model_path:
        tests.append(("Georgia Model Fields", lambda: test_model_has_required_fields(model_path)))
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            failed += 1
            print(f"\n❌ {name} FAILED with exception:")
            print(f"   {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED!")
        print("\n📋 Next steps:")
        print("   1. Load retrained Georgia model in MusicHal")
        print("   2. Test with: python MusicHal_9000.py --input-device 5 --hybrid-perception --wav2vec --gpu")
        print("   3. Switch behavior modes with MIDI CC 16 (values 0-127 → shadow/mirror/couple)")
        print("   4. Monitor terminal for '🎯 Using request-based generation' messages")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed")
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test MusicHal request integration')
    parser.add_argument('--model-path', type=str, 
                       help='Path to Georgia model JSON for validation')
    
    args = parser.parse_args()
    
    sys.exit(run_all_tests(args.model_path))

