#!/usr/bin/env python3
"""
Convert AudioOracle JSON model to compressed pickle format

This converts the slow-loading JSON format to a much faster pickle format.
Typical speedup: 10-50x faster loading for large models.

Usage:
    python convert_model_to_pickle.py JSON/yourmodel_model.json
    
This will create JSON/yourmodel_model.pkl.gz alongside the original file.
"""

import sys
import os
import time
from memory.polyphonic_audio_oracle import PolyphonicAudioOracle

def convert_json_to_pickle(json_path: str):
    """Convert a JSON model to pickle format"""
    if not os.path.exists(json_path):
        print(f"❌ File not found: {json_path}")
        return False
    
    if not json_path.endswith('_model.json'):
        print(f"⚠️  Warning: File doesn't end with '_model.json': {json_path}")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return False
    
    # Create output path
    pickle_path = json_path.replace('_model.json', '_model.pkl.gz')
    
    if os.path.exists(pickle_path):
        file_size_mb = os.path.getsize(pickle_path) / (1024 * 1024)
        print(f"⚠️  Pickle file already exists: {pickle_path} ({file_size_mb:.1f} MB)")
        response = input("Overwrite? (y/n): ")
        if response.lower() != 'y':
            return False
    
    print("\n" + "="*60)
    print(f"Converting: {json_path}")
    print(f"Output:     {pickle_path}")
    print("="*60 + "\n")
    
    # Load from JSON
    print("📂 Step 1: Loading from JSON...")
    oracle = PolyphonicAudioOracle()
    
    start_time = time.time()
    success = oracle.load_from_file(json_path)
    json_load_time = time.time() - start_time
    
    if not success:
        print("❌ Failed to load JSON file")
        return False
    
    print(f"✅ JSON loaded in {json_load_time:.2f}s")
    
    # Save to pickle
    print("\n💾 Step 2: Saving to pickle...")
    start_time = time.time()
    success = oracle.save_to_pickle(pickle_path)
    pickle_save_time = time.time() - start_time
    
    if not success:
        print("❌ Failed to save pickle file")
        return False
    
    print(f"✅ Pickle saved in {pickle_save_time:.2f}s")
    
    # Test load speed
    print("\n🧪 Step 3: Testing load speed...")
    test_oracle = PolyphonicAudioOracle()
    
    start_time = time.time()
    test_oracle.load_from_pickle(pickle_path)
    pickle_load_time = time.time() - start_time
    
    # Report results
    json_size_mb = os.path.getsize(json_path) / (1024 * 1024)
    pickle_size_mb = os.path.getsize(pickle_path) / (1024 * 1024)
    speedup = json_load_time / pickle_load_time if pickle_load_time > 0 else 0
    compression = (1 - pickle_size_mb / json_size_mb) * 100 if json_size_mb > 0 else 0
    
    print("\n" + "="*60)
    print("✅ CONVERSION COMPLETE")
    print("="*60)
    print(f"Original JSON:  {json_size_mb:.1f} MB")
    print(f"Pickle output:  {pickle_size_mb:.1f} MB ({compression:.1f}% size reduction)")
    print(f"\nLoad time comparison:")
    print(f"  JSON:         {json_load_time:.2f}s")
    print(f"  Pickle:       {pickle_load_time:.2f}s")
    print(f"  Speedup:      {speedup:.1f}x faster!")
    print("="*60)
    
    print(f"\n💡 Next steps:")
    print(f"   1. Keep both files (JSON as backup, pickle for performance)")
    print(f"   2. MusicHal_9000.py will automatically use pickle when available")
    print(f"   3. Original JSON: {json_path}")
    print(f"   4. Fast pickle:   {pickle_path}")
    
    return True

def main():
    if len(sys.argv) < 2:
        print("Usage: python convert_model_to_pickle.py <path_to_model.json>")
        print("\nExample:")
        print("  python convert_model_to_pickle.py JSON/Subtle_southern_051125_0038_training_model.json")
        print("\nThis will create a .pkl.gz file for 10-50x faster loading!")
        sys.exit(1)
    
    json_path = sys.argv[1]
    success = convert_json_to_pickle(json_path)
    
    if success:
        print("\n✅ Conversion successful!")
        sys.exit(0)
    else:
        print("\n❌ Conversion failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
