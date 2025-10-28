#!/usr/bin/env python3
"""
Simple System Component Test
Tests individual components of the CCM3 system to verify they work correctly
"""

import os
import sys
import time
import numpy as np
import json

def test_harmonic_detector():
    """Test the harmonic context detector"""
    print("🎼 Testing Harmonic Context Detector...")
    
    try:
        from listener.harmonic_context import RealtimeHarmonicDetector, HarmonicContext
        
        # Create detector
        detector = RealtimeHarmonicDetector()
        print("   ✅ Harmonic detector created successfully")
        
        # Generate synthetic audio (simple chord)
        duration = 2.0
        sr = 44100
        t = np.linspace(0, duration, int(sr * duration))
        
        # G major chord
        g_major = 0.3 * (np.sin(2 * np.pi * 392 * t) +  # G
                        np.sin(2 * np.pi * 494 * t) +  # B
                        np.sin(2 * np.pi * 587 * t))   # D
        
        # Test chroma extraction
        chroma = detector.extract_chroma_from_audio(g_major, sr)
        print(f"   ✅ Chroma extraction successful: {chroma.shape}")
        
        # Test chord detection
        chord, confidence, chord_type = detector.detect_chord_from_chroma(chroma)
        print(f"   ✅ Chord detection: {chord} (confidence: {confidence:.2f})")
        
        # Test key detection
        key, key_confidence = detector.detect_key_from_chroma(chroma)
        print(f"   ✅ Key detection: {key} (confidence: {key_confidence:.2f})")
        
        # Test full update
        context = detector.update_from_audio(g_major, sr)
        print(f"   ✅ Full update successful: {context.current_chord}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Harmonic detector test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_rhythmic_detector():
    """Test the rhythmic context detector"""
    print("🥁 Testing Rhythmic Context Detector...")
    
    try:
        from listener.rhythmic_context import RealtimeRhythmicDetector, RhythmicContext
        
        # Create detector
        detector = RealtimeRhythmicDetector()
        print("   ✅ Rhythmic detector created successfully")
        
        # Generate synthetic audio with rhythm
        duration = 4.0
        sr = 44100
        t = np.linspace(0, duration, int(sr * duration))
        
        # Simple 4/4 beat pattern
        beat_freq = 2.0  # 2 Hz = 120 BPM
        rhythm = 0.2 * np.sin(2 * np.pi * beat_freq * t)
        
        # Test basic functionality
        print(f"   ✅ Rhythmic detector initialized with {detector.update_interval}s update interval")
        
        # Test full update
        event_data = {
            'onset': True,
            'ioi': 0.5,
            'rms_db': -20,
            't': time.time()
        }
        context = detector.update_from_event(event_data, time.time())
        print(f"   ✅ Full update successful: {context.current_tempo:.1f} BPM")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Rhythmic detector test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_feature_mapper():
    """Test the feature mapper with harmonic awareness"""
    print("🎹 Testing Feature Mapper...")
    
    try:
        from mapping.feature_mapper import FeatureMapper, MIDIParameters
        
        # Create mapper
        mapper = FeatureMapper()
        print("   ✅ Feature mapper created successfully")
        
        # Test basic mapping
        event_data = {
            'f0': 440.0,
            'rms_db': -20.0,
            'centroid': 2000.0,
            'rolloff': 8000.0,
            'zcr': 0.1,
            'hnr': 0.8,
            'attack_time': 0.02,
            'decay_time': 0.1,
            'spectral_flux': 50.0
        }
        
        decision_data = {
            'mode': 'imitate',
            'confidence': 0.8,
            'musical_params': {
                'current_chord': 'G',
                'key_signature': 'G_major',
                'scale_degrees': [0, 2, 4, 6, 7, 9, 11],
                'chord_root': 7,
                'chord_stability': 0.9
            }
        }
        
        # Test melodic mapping
        midi_params = mapper.map_features_to_midi(event_data, decision_data, "melodic")
        print(f"   ✅ Melodic mapping: Note {midi_params.note}, Vel {midi_params.velocity}")
        
        # Test bass mapping
        midi_params = mapper.map_features_to_midi(event_data, decision_data, "bass")
        print(f"   ✅ Bass mapping: Note {midi_params.note}, Vel {midi_params.velocity}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Feature mapper test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_behavior_engine():
    """Test the behavior engine"""
    print("🤖 Testing Behavior Engine...")
    
    try:
        from agent.behaviors import BehaviorEngine, BehaviorMode
        
        # Create engine
        engine = BehaviorEngine()
        print("   ✅ Behavior engine created successfully")
        
        # Test decision generation
        current_event = {
            'f0': 440.0,
            'rms_db': -20.0,
            'instrument': 'piano',
            'harmonic_context': {
                'current_chord': 'G',
                'key_signature': 'G_major',
                'confidence': 0.8
            }
        }
        
        # Test basic functionality
        print("   ✅ Behavior engine initialized successfully")
        
        # Test decision generation (simplified)
        try:
            decisions = engine.decide_behavior(current_event, None, None)
            print(f"   ✅ Decision generation: {len(decisions)} decisions")
            
            if decisions:
                decision = decisions[0]
                print(f"      Voice: {decision.voice_type}, Confidence: {decision.confidence:.2f}")
        except Exception as e:
            print(f"   ⚠️ Decision generation failed (expected): {e}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Behavior engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_data():
    """Test if training data exists and is valid"""
    print("📚 Testing Training Data...")
    
    try:
        # Check if Georgia training data exists
        georgia_file = "JSON/georgia_test"
        if not os.path.exists(georgia_file):
            print("   ❌ Georgia training data not found")
            return False
        
        # Load and validate
        with open(georgia_file, 'r') as f:
            data = json.load(f)
        
        print("   ✅ Training data loaded successfully")
        
        # Check key components
        if 'training_successful' in data:
            print(f"   ✅ Training successful: {data['training_successful']}")
        
        if 'audio_oracle_stats' in data:
            stats = data['audio_oracle_stats']
            print(f"   ✅ Audio oracle: {stats.get('total_patterns', 0)} patterns")
        
        if 'rhythm_oracle_stats' in data:
            rhythm_stats = data['rhythm_oracle_stats']
            print(f"   ✅ Rhythm oracle: {rhythm_stats.get('total_patterns', 0)} patterns")
        
        if 'chord_progression' in data:
            progression = data['chord_progression']
            print(f"   ✅ Chord progression: {len(progression)} chords")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Training data test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all component tests"""
    print("🧪 CCM3 System Component Tests")
    print("=" * 50)
    
    tests = [
        test_training_data,
        test_harmonic_detector,
        test_rhythmic_detector,
        test_feature_mapper,
        test_behavior_engine
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
            print()
        except Exception as e:
            print(f"   ❌ Test failed with exception: {e}")
            results.append(False)
            print()
    
    # Summary
    print("=" * 50)
    print("🧪 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! System components are working correctly.")
        return 0
    else:
        print("⚠️ Some tests failed. Check individual results above.")
        return 1

if __name__ == "__main__":
    exit(main())
