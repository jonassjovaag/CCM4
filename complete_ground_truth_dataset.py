#!/usr/bin/env python3
"""
Complete Ground Truth Dataset - All Octaves
============================================

Automatically runs validation for 3 octaves (low, mid, high) then trains.
Total: ~1,116 samples across 108 chord types = ~10 per type

Usage:
    python complete_ground_truth_dataset.py --input-device 5 --gpu

This will take ~90 minutes. You can walk away!
"""

import json
import time
from datetime import datetime
from chord_ground_truth_trainer_wav2vec import ChordGroundTruthTrainer
from generate_full_chord_dataset import generate_full_chord_list


def run_octave_validation(trainer, octave, chord_list, octave_label):
    """Run validation for one octave"""
    print("\n" + "=" * 70)
    print(f"🎹 OCTAVE {octave} ({octave_label})")
    print("=" * 70)
    
    # Generate chord list for this octave
    octave_chords = generate_full_chord_list(base_octave=octave)
    
    print(f"   Chords to validate: {len(octave_chords)}")
    print(f"   Estimated time: {len(octave_chords) * 3 / 60:.1f} minutes")
    print(f"\n   Starting in 3 seconds...")
    time.sleep(3)
    
    # Run validation
    success = trainer.run_training_session(
        chord_list=octave_chords,
        save_to_json=True
    )
    
    if success:
        print(f"\n✅ Octave {octave} complete!")
        print(f"   Validated: {len(trainer.validated_chords)} chords")
        return len(trainer.validated_chords)
    else:
        print(f"\n❌ Octave {octave} failed")
        return 0


def merge_validation_results(json_files):
    """Merge multiple validation JSON files"""
    print("\n" + "=" * 70)
    print("🔄 MERGING DATASETS")
    print("=" * 70)
    
    all_chords = []
    
    for filepath in json_files:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                all_chords.extend(data)
                print(f"   ✅ {filepath}: {len(data)} chords")
            elif isinstance(data, dict):
                chords = data.get('validated_chords', [])
                all_chords.extend(chords)
                print(f"   ✅ {filepath}: {len(chords)} chords")
        except Exception as e:
            print(f"   ❌ {filepath}: Error - {e}")
    
    print(f"\n   Total merged: {len(all_chords)} chords")
    
    # Save merged dataset
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    merged_file = f"validation_results_merged_{timestamp}.json"
    
    with open(merged_file, 'w') as f:
        json.dump(all_chords, f)
    
    print(f"   💾 Saved: {merged_file}")
    return merged_file


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Complete ground truth dataset - all octaves automatically',
    )
    
    parser.add_argument('--input-device', type=int, required=True,
                       help='Audio input device number')
    parser.add_argument('--gpu', action='store_true',
                       help='Use GPU acceleration for Wav2Vec')
    parser.add_argument('--octaves', type=str, default='3,4,5',
                       help='Comma-separated octaves to validate (default: 3,4,5)')
    
    args = parser.parse_args()
    
    # Parse octaves
    octaves = [int(o.strip()) for o in args.octaves.split(',')]
    octave_labels = {3: 'Low', 4: 'Middle C', 5: 'High'}
    
    print("🎯 COMPLETE GROUND TRUTH DATASET GENERATION")
    print("=" * 70)
    print(f"\n📊 Plan:")
    print(f"   Octaves: {octaves}")
    print(f"   Chords per octave: ~432")
    print(f"   Total chords: ~{len(octaves) * 432}")
    print(f"   Estimated total time: {len(octaves) * 432 * 3 / 60:.0f} minutes")
    
    print(f"\n⚠️  Make sure:")
    print(f"   1. MIDI: 'IAC Driver Chord Trainer Output' → Ableton/synth")
    print(f"   2. Audio: Device {args.input_device} is receiving sound")
    print(f"   3. Volume is audible but not clipping")
    print(f"   4. You can walk away! This will run automatically.")
    
    print("\nPress Enter to start all octaves, Ctrl+C to abort...")
    input()
    
    start_time = time.time()
    validation_files = []
    total_validated = 0
    
    try:
        for i, octave in enumerate(octaves, 1):
            print(f"\n{'='*70}")
            print(f"OCTAVE {i}/{len(octaves)}: {octave} ({octave_labels.get(octave, 'Unknown')})")
            print(f"{'='*70}")
            
            # Create fresh trainer for each octave
            trainer = ChordGroundTruthTrainer(
                use_gpu=args.gpu,
                audio_input_device=args.input_device
            )
            
            if not trainer.start():
                print(f"\n❌ Failed to start trainer for octave {octave}")
                continue
            
            # Generate chord list for this octave
            octave_chords = generate_full_chord_list(base_octave=octave)
            
            print(f"\n   Chords: {len(octave_chords)}")
            print(f"   Time estimate: {len(octave_chords) * 3 / 60:.1f} minutes")
            print(f"   Progress: Octave {i}/{len(octaves)}")
            
            # Small pause before starting
            print(f"\n   Starting in 3 seconds...")
            for countdown in range(3, 0, -1):
                print(f"   {countdown}...")
                time.sleep(1)
            
            # Run validation
            success = trainer.run_training_session(
                chord_list=octave_chords,
                save_to_json=True
            )
            
            if success and len(trainer.validated_chords) > 0:
                # Find the most recently created validation file
                import glob
                recent_files = sorted(glob.glob('validation_results_*.json'), 
                                    key=lambda x: os.path.getmtime(x), 
                                    reverse=True)
                if recent_files:
                    validation_files.append(recent_files[0])
                    total_validated += len(trainer.validated_chords)
                    print(f"\n✅ Octave {octave} complete: {len(trainer.validated_chords)} chords")
            else:
                print(f"\n⚠️  Octave {octave}: No chords validated")
            
            # Short break between octaves
            if i < len(octaves):
                print(f"\n   ⏸️  5 second break before next octave...")
                time.sleep(5)
        
        elapsed = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"✅ ALL OCTAVES COMPLETE!")
        print(f"{'='*70}")
        print(f"   Total validated: {total_validated} chords")
        print(f"   Time taken: {elapsed / 60:.1f} minutes")
        print(f"   Validation files: {len(validation_files)}")
        
        # Merge datasets
        if len(validation_files) > 1:
            merged_file = merge_validation_results(validation_files)
            print(f"\n✅ Datasets merged: {merged_file}")
        elif len(validation_files) == 1:
            merged_file = validation_files[0]
            print(f"\n   Using single file: {merged_file}")
        else:
            print(f"\n❌ No validation files to train on")
            return
        
        # Train the model
        if total_validated >= 10:
            print(f"\n{'='*70}")
            print(f"🤖 TRAINING CHORD CLASSIFIER")
            print(f"{'='*70}")
            
            trainer = ChordGroundTruthTrainer(
                use_gpu=args.gpu,
                audio_input_device=args.input_device
            )
            
            if trainer.start():
                if trainer.load_from_validation_results(merged_file):
                    print(f"\n   Loaded: {len(trainer.validated_chords)} samples")
                    
                    if trainer.train_ml_model():
                        print(f"\n{'='*70}")
                        print(f"✅ COMPLETE SUCCESS!")
                        print(f"{'='*70}")
                        print(f"   Model: models/chord_model_wav2vec.pkl")
                        print(f"   Samples: {total_validated}")
                        print(f"   Total time: {elapsed / 60:.1f} minutes")
                        print(f"\n🎯 Next steps:")
                        print(f"   1. Connect to Chandra_trainer.py")
                        print(f"   2. Retrain Georgia with ground truth")
                        print(f"   3. Test MusicHal_9000!")
                    else:
                        print(f"\n❌ Training failed")
                else:
                    print(f"\n❌ Failed to load merged data")
        else:
            print(f"\n⚠️  Not enough samples ({total_validated}) for training (need 10+)")
        
    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"⚠️  INTERRUPTED BY USER")
        print(f"{'='*70}")
        print(f"   Validated so far: {total_validated} chords")
        print(f"   Time elapsed: {elapsed / 60:.1f} minutes")
        print(f"   Files created: {len(validation_files)}")
        if validation_files:
            print(f"\n   You can still merge and train on partial data:")
            print(f"   python chord_ground_truth_trainer_wav2vec.py \\")
            print(f"       --from-validation {validation_files[0]} --gpu")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    import os
    main()





