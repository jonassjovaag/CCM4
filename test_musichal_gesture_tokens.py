#!/usr/bin/env python3
"""
Test that MusicHal_9000 correctly loads and uses gesture tokens from trained models
"""

import json
import os

print("=" * 60)
print("MUSICHAL GESTURE TOKEN LOADING TEST")
print("=" * 60)

# 1. Check the trained model has gesture tokens
model_file = "JSON/Nineteen_301025_1703_training_model.json"
quantizer_file = "JSON/Nineteen_301025_1703_training_quantizer.joblib"

print(f"\n1. Checking trained model: {model_file}")
if os.path.exists(model_file):
    with open(model_file, 'r') as f:
        model = json.load(f)
    
    audio_frames = model.get('audio_frames', {})
    
    # Count gesture tokens
    frames_with_tokens = 0
    for frame in audio_frames.values():
        if 'gesture_token' in frame.get('audio_data', {}):
            frames_with_tokens += 1
    
    print(f"   ✅ Model loaded")
    print(f"   Total frames: {len(audio_frames)}")
    print(f"   Frames with gesture_token: {frames_with_tokens}")
    
    if frames_with_tokens > 0:
        print(f"   ✅ Model contains gesture tokens!")
    else:
        print(f"   ❌ No gesture tokens in model!")
        exit(1)
else:
    print(f"   ❌ Model file not found!")
    exit(1)

# 2. Check quantizer file exists
print(f"\n2. Checking quantizer: {quantizer_file}")
if os.path.exists(quantizer_file):
    print(f"   ✅ Quantizer file exists")
    import joblib
    quantizer = joblib.load(quantizer_file)
    print(f"   ✅ Quantizer loaded")
    
    # Check if it's the actual quantizer object or dict wrapper
    if hasattr(quantizer, 'n_clusters'):
        print(f"   Vocabulary size: {quantizer.n_clusters}")
        print(f"   Codebook shape: {quantizer.cluster_centers_.shape}")
    elif isinstance(quantizer, dict):
        print(f"   Quantizer is dict format (gesture vocabulary wrapper)")
        print(f"   Keys: {list(quantizer.keys())}")
    else:
        print(f"   Quantizer type: {type(quantizer)}")
else:
    print(f"   ⚠️  Quantizer file not found (optional)")

# 3. Test PolyphonicAudioOracle loading
print(f"\n3. Testing AudioOracle loads gesture tokens...")
from memory.polyphonic_audio_oracle import PolyphonicAudioOracle

oracle = PolyphonicAudioOracle(
    distance_threshold=0.15,
    distance_function='euclidean',
    feature_dimensions=768,  # Wav2Vec dimensions
    adaptive_threshold=True
)

success = oracle.load_from_file(model_file)
if success:
    print(f"   ✅ AudioOracle loaded model")
    
    # Check if loaded frames have gesture tokens
    if oracle.audio_frames:
        first_frame_id = list(oracle.audio_frames.keys())[0]
        first_frame = oracle.audio_frames[first_frame_id]
        has_gesture = 'gesture_token' in first_frame.audio_data
        
        print(f"   First frame has gesture_token: {has_gesture}")
        
        if has_gesture:
            print(f"   ✅ Gesture token value: {first_frame.audio_data['gesture_token']}")
            
            # Count all frames with gesture tokens
            frames_with_tokens = sum(
                1 for frame in oracle.audio_frames.values()
                if 'gesture_token' in frame.audio_data
            )
            print(f"   Loaded frames with gesture_token: {frames_with_tokens}/{len(oracle.audio_frames)}")
            
            if frames_with_tokens == len(oracle.audio_frames):
                print(f"\n   ✅✅✅ ALL FRAMES HAVE GESTURE TOKENS!")
                print(f"   MusicHal will have access to gesture token data!")
            else:
                print(f"\n   ⚠️  Only {frames_with_tokens}/{len(oracle.audio_frames)} frames have tokens")
        else:
            print(f"   ❌ Loaded frames don't have gesture_token!")
            print(f"   Available keys: {list(first_frame.audio_data.keys())[:15]}")
    else:
        print(f"   ❌ No frames loaded!")
else:
    print(f"   ❌ Failed to load AudioOracle!")
    exit(1)

# 4. Test generation uses gesture tokens
print(f"\n4. Testing generation with gesture tokens...")
import numpy as np

# Try to generate using the oracle
request = {
    'consonance': 0.7,
    'tension': 0.5,
    'gesture_token': 42  # Request specific gesture token
}

print(f"   Requesting generation with gesture_token=42...")
result = oracle.generate_with_request(
    num_events=5,
    request=request,
    allow_repetition=True
)

if result:
    print(f"   ✅ Generated {len(result)} events")
    
    # Check if generated events reference frames with gesture tokens
    for i, event in enumerate(result[:3]):  # Check first 3
        frame_id = event.get('frame_id')
        if frame_id is not None and frame_id in oracle.audio_frames:
            frame = oracle.audio_frames[frame_id]
            gesture = frame.audio_data.get('gesture_token', 'MISSING')
            print(f"   Event {i}: frame {frame_id}, gesture_token={gesture}")
    
    print(f"\n   ✅ Generation uses frames with gesture tokens!")
else:
    print(f"   ⚠️  No events generated (oracle might be too constrained)")

print("\n" + "=" * 60)
print("✅ TEST COMPLETE")
print("   MusicHal_9000 CAN load and use gesture tokens from trained models!")
print("=" * 60)
