#!/usr/bin/env python3
"""
Full 600-Chord Ground Truth Dataset Generator
==============================================

Generates MIDI for all 600 chord variations and validates them.

Chord types: Major, Minor, 7, maj7, 9, aug, dim, sus2, sus4
Roots: 12 (C, C#, D, D#, E, F, F#, G, G#, A, A#, B)
Inversions: 3-4 per chord type
Total: ~600 chords

Usage:
    python generate_full_chord_dataset.py --input-device 5 --gpu
"""

from chord_ground_truth_trainer_wav2vec import ChordGroundTruthTrainer

# 12 chromatic roots
ROOTS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Chord types with MIDI intervals
CHORD_TYPES = {
    '': [0, 4, 7],           # Major (root, maj3, P5)
    'm': [0, 3, 7],          # Minor (root, min3, P5)
    '7': [0, 4, 7, 10],      # Dominant 7 (root, maj3, P5, min7)
    'maj7': [0, 4, 7, 11],   # Major 7 (root, maj3, P5, maj7)
    '9': [0, 4, 7, 10, 14],  # Dominant 9 (root, maj3, P5, min7, maj9)
    'aug': [0, 4, 8],        # Augmented (root, maj3, aug5)
    'dim': [0, 3, 6],        # Diminished (root, min3, dim5)
    'sus2': [0, 2, 7],       # Sus2 (root, maj2, P5)
    'sus4': [0, 5, 7],       # Sus4 (root, P4, P5)
}


def generate_full_chord_list(base_octave=4):
    """
    Generate all 600 chord variations
    
    Returns list of (root, quality, midi_notes, inversion, octave_offset) tuples
    """
    chord_list = []
    
    for root_idx, root in enumerate(ROOTS):
        # Base MIDI note for this root (C4 = 60)
        root_midi = 60 + root_idx + (base_octave - 4) * 12
        
        for quality, intervals in CHORD_TYPES.items():
            # Generate MIDI notes for this chord
            num_notes = len(intervals)
            
            # Root position
            midi_notes = [root_midi + interval for interval in intervals]
            chord_list.append((root, quality, midi_notes, 0, 0))
            
            # Inversions (rotate the notes up an octave)
            for inv in range(1, num_notes):
                inverted_notes = midi_notes.copy()
                # Move bottom notes up an octave
                for i in range(inv):
                    inverted_notes[i] += 12
                inverted_notes.sort()
                chord_list.append((root, quality, inverted_notes, inv, 0))
    
    return chord_list


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate full 600-chord ground truth dataset',
    )
    
    parser.add_argument('--input-device', type=int, required=True,
                       help='Audio input device number')
    parser.add_argument('--gpu', action='store_true',
                       help='Use GPU acceleration for Wav2Vec')
    parser.add_argument('--base-octave', type=int, default=4,
                       help='Base octave for chords (default: 4, C4=middle C)')
    parser.add_argument('--test-run', action='store_true',
                       help='Test with first 20 chords only')
    
    args = parser.parse_args()
    
    # Generate full chord list
    print("🎼 Generating chord list...")
    full_chord_list = generate_full_chord_list(base_octave=args.base_octave)
    
    if args.test_run:
        chord_list = full_chord_list[:20]
        print("   🧪 TEST MODE: Using first 20 chords only")
    else:
        chord_list = full_chord_list
        print("   📊 FULL DATASET: {} chords".format(len(chord_list)))
    
    print("\n   Breakdown:")
    print("      Roots: {}".format(len(ROOTS)))
    print("      Chord types: {}".format(len(CHORD_TYPES)))
    print("      Avg inversions: ~3.5 per type")
    print("      Total: {} chords".format(len(chord_list)))
    
    # Create trainer
    trainer = ChordGroundTruthTrainer(
        use_gpu=args.gpu,
        audio_input_device=args.input_device
    )
    
    if not trainer.start():
        print("\n❌ Failed to start trainer (check MIDI setup)")
        return
    
    print("\n⚠️  Make sure:")
    print("   1. MIDI: 'IAC Driver Chord Trainer Output' → Ableton/synth")
    print("   2. Audio: Device {} is receiving sound".format(args.input_device))
    print("   3. Volume is audible but not clipping")
    print("   4. You have time! ({} chords × 3s = {:.1f} minutes)".format(
        len(chord_list), len(chord_list) * 3 / 60))
    
    print("\nPress Enter to start, Ctrl+C to abort...")
    input()
    
    try:
        # Run validation session
        success = trainer.run_training_session(
            chord_list=chord_list,
            save_to_json=True
        )
        
        if success:
            print("\n✅ Validation complete!")
            print("   Validated: {}/{} chords".format(
                len(trainer.validated_chords), len(chord_list)))
            
            # Automatically train
            if len(trainer.validated_chords) >= 10:
                print("\n🤖 Training ML model on {} samples...".format(
                    len(trainer.validated_chords)))
                if trainer.train_ml_model():
                    print("\n✅ Training complete!")
                    print("   Model saved: models/chord_model_wav2vec.pkl")
                    print("\n🎯 Next steps:")
                    print("   1. Connect to Chandra_trainer.py")
                    print("   2. Retrain Georgia with ground truth guidance")
                    print("   3. Test with MusicHal_9000")
                else:
                    print("\n❌ Training failed")
            else:
                print("\n⚠️  Not enough samples to train (need at least 10)")
        else:
            print("\n❌ Validation failed")
            
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        print(f"   Validated so far: {len(trainer.validated_chords)} chords")
        if len(trainer.validated_chords) > 0:
            print("   Data saved to JSON - you can resume or train on partial dataset")
    except Exception as e:
        print("\n❌ Error: {}".format(e))
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

