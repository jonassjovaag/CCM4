"""
Test 5: Viewport Chord Name Translation
========================================
Diagnostic test to check if chord names exist in audio_data:
- Load trained model
- Check if 'chord_name' or 'chord_label' fields exist in audio_data
- Identify where chord name translation happens (training vs runtime)

READ-ONLY diagnostic to help locate potential viewport issue.
"""

import pickle
import gzip
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_viewport_translation():
    """Check for chord name fields in trained model."""
    
    print("=" * 80)
    print("TEST 5: VIEWPORT CHORD NAME TRANSLATION DIAGNOSTIC")
    print("=" * 80)
    
    model_path = Path("JSON/Itzama_071125_2130_training_model.pkl.gz")
    
    if not model_path.exists():
        print(f"❌ FAIL: Model file not found at {model_path}")
        return False
    
    try:
        # Load trained model (serialized as dict)
        print(f"\n📂 Loading model from: {model_path}")
        with gzip.open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Extract audio_frames from dict format
        if isinstance(model_data, dict):
            audio_frames = model_data.get('audio_frames', {})
        else:
            audio_frames = model_data.audio_frames
        
        print(f"✅ Model loaded: {len(audio_frames)} frames")
        
        # Check for chord-related fields in audio_data
        print(f"\n🔍 Scanning frames for chord name fields...")
        
        chord_field_counts = {
            'chord_name': 0,
            'chord_label': 0,
            'chord': 0,
            'chord_symbol': 0,
            'chord_quality': 0,
            'all_fields': set()
        }
        
        # Sample 100 frames (or all if fewer)
        frame_ids = sorted(list(audio_frames.keys()))[:100]
        
        for frame_id in frame_ids:
            frame = audio_frames[frame_id]
            # Handle both dict and AudioFrame object formats
            if isinstance(frame, dict):
                audio_data = frame.get('audio_data', frame)
            else:
                audio_data = frame.audio_data
            
            # Track all field names
            for field in audio_data.keys():
                chord_field_counts['all_fields'].add(field)
            
            # Check specific chord-related fields
            if 'chord_name' in audio_data:
                chord_field_counts['chord_name'] += 1
            if 'chord_label' in audio_data:
                chord_field_counts['chord_label'] += 1
            if 'chord' in audio_data:
                chord_field_counts['chord'] += 1
            if 'chord_symbol' in audio_data:
                chord_field_counts['chord_symbol'] += 1
            if 'chord_quality' in audio_data:
                chord_field_counts['chord_quality'] += 1
        
        # Report findings
        print(f"\n📊 Chord Field Scan Results (first {len(frame_ids)} frames):")
        print(f"   'chord_name': {chord_field_counts['chord_name']} frames")
        print(f"   'chord_label': {chord_field_counts['chord_label']} frames")
        print(f"   'chord': {chord_field_counts['chord']} frames")
        print(f"   'chord_symbol': {chord_field_counts['chord_symbol']} frames")
        print(f"   'chord_quality': {chord_field_counts['chord_quality']} frames")
        
        print(f"\n📋 All Fields Found in audio_data:")
        for field in sorted(chord_field_counts['all_fields']):
            print(f"   - {field}")
        
        # Check if ANY chord field exists
        has_chord_field = any([
            chord_field_counts['chord_name'] > 0,
            chord_field_counts['chord_label'] > 0,
            chord_field_counts['chord'] > 0,
            chord_field_counts['chord_symbol'] > 0,
            chord_field_counts['chord_quality'] > 0
        ])
        
        if has_chord_field:
            print(f"\n✅ Chord name fields FOUND in training data")
            print(f"   Viewport should be able to display chord names")
        else:
            print(f"\n⚠️  NO chord name fields in training data")
            print(f"   Chord translation likely happens at runtime (not in trained model)")
            print(f"   Check viewport code or real-time analysis modules")
        
        # Show example frame data
        if len(frame_ids) > 0:
            example_id = frame_ids[0]
            example_frame = audio_frames[example_id]
            
            if isinstance(example_frame, dict):
                example_data = example_frame.get('audio_data', example_frame)
            else:
                example_data = example_frame.audio_data
            
            print(f"\n🔍 Example Frame Data (frame {example_id}):")
            for key, value in example_data.items():
                print(f"   {key}: {value}")
        
        print(f"\n" + "=" * 80)
        print("✅ TEST 5 PASSED: Viewport diagnostic complete")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"\n❌ FAIL: Exception during test: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_viewport_translation()
    sys.exit(0 if success else 1)
