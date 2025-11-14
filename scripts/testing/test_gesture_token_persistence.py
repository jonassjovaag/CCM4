#!/usr/bin/env python3
"""
Test that gesture_token field persists through the training pipeline
Isolated test without full audio processing
"""

import sys
import numpy as np
from audio_file_learning.hybrid_batch_trainer import HybridBatchTrainer

print("=" * 60)
print("GESTURE TOKEN PERSISTENCE TEST")
print("=" * 60)

# Create a minimal test event with gesture_token
test_event = {
    't': 1.0,
    'f0': 440.0,
    'midi': 69,
    'cents': 0.0,
    'rms_db': -20.0,
    'centroid': 2000.0,
    'rolloff': 5000.0,
    'bandwidth': 3000.0,
    'onset': True,
    'ioi': 0.5,
    'features': list(np.random.randn(768)),  # 768D Wav2Vec features
    'gesture_token': 42,  # THE KEY FIELD WE'RE TESTING
    'consonance': 0.75,
    'chord': 'Cmaj7',
    'significance_score': 0.8
}

print(f"\n1. Original test event:")
print(f"   Has gesture_token: {'gesture_token' in test_event}")
print(f"   gesture_token value: {test_event.get('gesture_token')}")

# Initialize trainer
trainer = HybridBatchTrainer(cpu_threshold=0)  # Force CPU

# Test the _extract_musical_features method
print(f"\n2. Testing _extract_musical_features()...")
musical_features = trainer._extract_musical_features(test_event)

print(f"   Result has gesture_token: {'gesture_token' in musical_features}")
if 'gesture_token' in musical_features:
    print(f"   ✅ gesture_token value: {musical_features['gesture_token']}")
    print(f"   ✅ PASSED: gesture_token preserved!")
else:
    print(f"   ❌ FAILED: gesture_token was lost!")
    print(f"   Available keys: {list(musical_features.keys())[:20]}")
    sys.exit(1)

# Test AudioOracle storage
print(f"\n3. Testing AudioOracle.add_audio_frame()...")
features = np.array(musical_features['features'])

# Import AudioOracle directly
from memory.audio_oracle import AudioOracle

audio_oracle = AudioOracle(
    distance_threshold=0.15,
    distance_function='euclidean',
    max_pattern_length=50
)

audio_oracle.add_audio_frame(features, musical_features)

# Check if stored correctly
if 0 in audio_oracle.audio_frames:
    stored_frame = audio_oracle.audio_frames[0]
    stored_data = stored_frame.audio_data
    
    print(f"   Stored audio_data has gesture_token: {'gesture_token' in stored_data}")
    if 'gesture_token' in stored_data:
        print(f"   ✅ Stored gesture_token value: {stored_data['gesture_token']}")
        print(f"   ✅ PASSED: gesture_token persisted to AudioOracle!")
    else:
        print(f"   ❌ FAILED: gesture_token lost during storage!")
        print(f"   Available keys in stored_data: {list(stored_data.keys())[:20]}")
        sys.exit(1)
else:
    print(f"   ❌ FAILED: No frame stored!")
    sys.exit(1)

# Test serialization
print(f"\n4. Testing JSON serialization...")
import tempfile
import json

with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
    temp_file = f.name

success = audio_oracle.save_to_file(temp_file)

if success:
    with open(temp_file, 'r') as f:
        saved_model = json.load(f)
    
    audio_frames = saved_model.get('audio_frames', {})
    if '0' in audio_frames or 0 in audio_frames:
        key = '0' if '0' in audio_frames else 0
        saved_frame = audio_frames[key]
        saved_audio_data = saved_frame.get('audio_data', {})
        
        print(f"   Saved audio_data has gesture_token: {'gesture_token' in saved_audio_data}")
        if 'gesture_token' in saved_audio_data:
            print(f"   ✅ Saved gesture_token value: {saved_audio_data['gesture_token']}")
            print(f"\n   ✅✅✅ ALL TESTS PASSED!")
            print(f"   gesture_token persists through entire pipeline!")
        else:
            print(f"   ❌ FAILED: gesture_token lost during serialization!")
            print(f"   Available keys in saved_audio_data: {list(saved_audio_data.keys())[:20]}")
            sys.exit(1)
    else:
        print(f"   ❌ FAILED: No frame in saved model!")
        sys.exit(1)
else:
    print(f"   ❌ FAILED: Could not save model!")
    sys.exit(1)

# Cleanup
import os
os.unlink(temp_file)

print("=" * 60)
print("✅ Test completed successfully!")
print("=" * 60)
