#!/usr/bin/env python3
"""
Gesture Token Failure Diagnostic and Fix
========================================

Based on the training output, the gesture token generation is completely broken:
- All 30,000 events get assigned the same token (10)
- Only 1 unique token is generated instead of diverse patterns
- This explains the reduced rhythmic variety

This script will:
1. Identify why gesture token generation fails
2. Propose immediate fixes to restore token diversity
"""

import json
import numpy as np
from typing import Dict, List

def diagnose_gesture_token_failure():
    """Diagnose why gesture tokens aren't being generated properly"""
    print("🔍 DIAGNOSING GESTURE TOKEN FAILURE")
    print("=" * 50)
    
    # Check training file structure
    training_file = 'JSON/Daybreak_031125_2323_training.json'
    model_file = 'JSON/Daybreak_031125_2323_training_model.json'
    
    print(f"1. Checking training summary: {training_file}")
    try:
        with open(training_file, 'r') as f:
            training_data = json.load(f)
        
        print(f"   ✅ Training file loaded")
        print(f"   Events processed: {training_data.get('events_processed', 'N/A')}")
        print(f"   Training successful: {training_data.get('training_successful', 'N/A')}")
        
        # Check for dual perception info
        if 'dual_perception_analysis' in training_data:
            dual_info = training_data['dual_perception_analysis']
            print(f"   Dual vocabulary enabled: {dual_info.get('dual_vocabulary_mode', 'N/A')}")
            print(f"   Unique gesture patterns: {dual_info.get('unique_gesture_patterns', 'N/A')}")
            print(f"   Events with tokens: {dual_info.get('events_with_tokens', 'N/A')}")
            
    except FileNotFoundError:
        print(f"   ❌ Training file not found: {training_file}")
        return
    
    print(f"\n2. Checking model file structure: {model_file}")
    try:
        with open(model_file, 'r') as f:
            model_data = json.load(f)
        
        frames = model_data.get('audio_frames', {})
        print(f"   ✅ Model file loaded")
        print(f"   Audio frames: {len(frames)}")
        
        # Check first few frames for token fields
        print(f"\n3. Analyzing frame structure:")
        sample_frame = None
        for i in range(min(10, len(frames))):
            frame_key = str(i)
            if frame_key in frames:
                sample_frame = frames[frame_key]
                break
        
        if sample_frame:
            print(f"   Frame keys: {list(sample_frame.keys())}")
            
            # Check for token fields
            token_fields = ['gesture_token', 'harmonic_token', 'percussive_token']
            for field in token_fields:
                if field in sample_frame:
                    print(f"   ✅ {field}: {sample_frame[field]}")
                else:
                    print(f"   ❌ {field}: MISSING")
            
            # Check features
            if 'features' in sample_frame:
                features = sample_frame['features']
                print(f"   ✅ Features: {len(features)}D array")
                if len(features) == 768:
                    print(f"      → Wav2Vec features present")
                else:
                    print(f"      → Unexpected feature dimension")
        else:
            print(f"   ❌ No sample frame found")
            
    except FileNotFoundError:
        print(f"   ❌ Model file not found: {model_file}")
        return
    except Exception as e:
        print(f"   ❌ Error reading model file: {e}")
        return
    
    print(f"\n4. ROOT CAUSE ANALYSIS:")
    
    # From the training log, we know:
    # - "Unique tokens assigned: 1"
    # - "Token range: 10 to 10" 
    # - All events get the same token
    
    print(f"   Based on training output:")
    print(f"   • Gesture token quantizer IS working (tokens are generated)")
    print(f"   • But ALL events get mapped to the same token (10)")
    print(f"   • This suggests either:")
    print(f"     A) All audio segments are too similar (over-smoothing)")
    print(f"     B) Quantizer training failed (poor clustering)")
    print(f"     C) Token assignment logic has a bug")
    
    return model_data

def propose_fixes():
    """Propose immediate fixes for gesture token failure"""
    print(f"\n🛠️  PROPOSED FIXES")
    print("=" * 50)
    
    print(f"FIX 1: Disable temporal smoothing completely")
    print(f"   Command: python Chandra_trainer.py --file input_audio/Daybreak.wav --no-temporal-smoothing")
    print(f"   Reason: Temporal smoothing may be making all segments too similar")
    
    print(f"\nFIX 2: Reduce quantizer vocabulary size")
    print(f"   Modify: --symbolic-vocabulary-size 16  # Instead of 64")
    print(f"   Reason: Fewer clusters might capture actual audio diversity better")
    
    print(f"\nFIX 3: Disable dual vocabulary mode")
    print(f"   Add flag: --no-dual-vocabulary")
    print(f"   Reason: Dual mode might have bugs, single vocab is simpler")
    
    print(f"\nFIX 4: Check quantizer training data")
    print(f"   The training shows 3291 segments for quantizer training")
    print(f"   But may all be too similar due to audio preprocessing")
    
    print(f"\nFIX 5: IMMEDIATE TEST - Skip Wav2Vec entirely")
    print(f"   Command: python Chandra_trainer.py --file input_audio/Daybreak.wav --no-wav2vec --no-temporal-smoothing")
    print(f"   This will test if the issue is in Wav2Vec processing or elsewhere")

def main():
    """Main diagnostic function"""
    print("🚨 GESTURE TOKEN FAILURE DIAGNOSTIC")
    print("Investigating why all events get the same gesture token")
    print()
    
    # Diagnose the failure
    model_data = diagnose_gesture_token_failure()
    
    # Propose fixes
    propose_fixes()
    
    print(f"\n🎯 RECOMMENDED NEXT STEP:")
    print(f"Run this command to test the simplest fix:")
    print(f"")
    print(f"python Chandra_trainer.py \\")
    print(f"  --file 'input_audio/Daybreak.wav' \\")
    print(f"  --max-events 5000 \\")
    print(f"  --no-temporal-smoothing \\")
    print(f"  --no-hierarchical \\")
    print(f"  --symbolic-vocabulary-size 16 \\")
    print(f"  --output 'JSON/Daybreak_debug_training.json'")
    print(f"")
    print(f"This will:")
    print(f"• Skip slow hierarchical analysis")
    print(f"• Disable temporal smoothing (root cause suspect)")
    print(f"• Use smaller vocabulary (16 vs 64 tokens)")
    print(f"• Finish quickly for testing")

if __name__ == "__main__":
    main()