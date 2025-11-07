#!/usr/bin/env python3
"""
Test Harmonic Data Storage in AudioOracle Training

This script verifies that fundamental frequencies and consonance scores
are being captured and stored during training.
"""

import json
import sys

def test_harmonic_data_storage():
    """Test that trained model contains harmonic data"""
    
    print("="*70)
    print("  TESTING HARMONIC DATA STORAGE")
    print("="*70)
    
    # Step 1: Check if we need to retrain
    print("\n📋 Step 1: Checking existing model...")
    
    try:
        with open('JSON/short_Itzama_071125_1807_training_model.json', 'r') as f:
            model = json.load(f)
        
        has_fundamentals = 'fundamentals' in model and len(model['fundamentals']) > 0
        has_consonances = 'consonances' in model and len(model['consonances']) > 0
        
        if has_fundamentals and has_consonances:
            print(f"   ✅ Model already has harmonic data:")
            print(f"      • {len(model['fundamentals'])} fundamental frequencies")
            print(f"      • {len(model['consonances'])} consonance scores")
            print("\n   No need to retrain!")
            return True
        else:
            print(f"   ⚠️  Model missing harmonic data:")
            print(f"      • fundamentals: {'YES' if has_fundamentals else 'NO'}")
            print(f"      • consonances: {'YES' if has_consonances else 'NO'}")
            print("\n   ⚠️  RETRAINING REQUIRED")
            return False
            
    except FileNotFoundError:
        print("   ❌ Model file not found")
        print("\n   ⚠️  TRAINING REQUIRED")
        return False

if __name__ == "__main__":
    print("\n🎵 MusicHal 9000 - Harmonic Data Storage Test\n")
    
    has_data = test_harmonic_data_storage()
    
    if not has_data:
        print("\n" + "="*70)
        print("  📝 TO CAPTURE HARMONIC DATA, RETRAIN THE MODEL:")
        print("="*70)
        print("\n   Run this command:")
        print("   python Chandra_trainer.py --file \"input_audio/short_Itzama.wav\" --max-events 5000\n")
        print("   This will:")
        print("   1. Extract fundamental frequencies (Groven method)")
        print("   2. Calculate consonance scores (0.0-1.0)")
        print("   3. Store in AudioOracle.fundamentals and AudioOracle.consonances")
        print("   4. Serialize to JSON for autonomous root progression")
        print("\n" + "="*70)
        sys.exit(1)
    else:
        print("\n" + "="*70)
        print("  ✅ READY FOR AUTONOMOUS ROOT PROGRESSION!")
        print("="*70)
        print("\n   Next steps:")
        print("   1. Create performance arc with waypoints (see performance_arcs/ROOT_HINTS_GUIDE.md)")
        print("   2. Implement AutonomousRootExplorer")
        print("   3. Test in live performance")
        sys.exit(0)
