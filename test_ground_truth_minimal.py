#!/usr/bin/env python3
"""
Minimal Ground Truth Test - Just 2 Chords
==========================================

Quick test with C major and C minor (root + inversions) = 6 total chords

Usage:
    python test_ground_truth_minimal.py --input-device 0 --gpu
"""

from chord_ground_truth_trainer_wav2vec import ChordGroundTruthTrainer

# Minimal chord list: C and Cm with inversions
MINIMAL_CHORD_LIST = [
    # C major
    ("C", "", [60, 64, 67], 0, 0),      # Root position (C-E-G)
    ("C", "", [64, 67, 72], 1, 0),      # 1st inversion (E-G-C)
    ("C", "", [67, 72, 76], 2, 0),      # 2nd inversion (G-C-E)
    
    # C minor
    ("Cm", "m", [60, 63, 67], 0, 0),    # Root position (C-Eb-G)
    ("Cm", "m", [63, 67, 72], 1, 0),    # 1st inversion (Eb-G-C)
    ("Cm", "m", [67, 72, 75], 2, 0),    # 2nd inversion (G-C-Eb)
]


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Minimal Ground Truth Test - 2 chords (6 variations)',
    )
    
    parser.add_argument('--input-device', type=int, required=True,
                       help='Audio input device number')
    parser.add_argument('--gpu', action='store_true',
                       help='Use GPU acceleration for Wav2Vec')
    parser.add_argument('--train-after', action='store_true',
                       help='Train ML model after validation')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = ChordGroundTruthTrainer(
        use_gpu=args.gpu,
        audio_input_device=args.input_device
    )
    
    if not trainer.start():
        print("\n❌ Failed to start trainer (check MIDI setup)")
        return
    
    print("\n🎯 MINIMAL TEST: 2 chords (C, Cm) × 3 inversions = 6 validations")
    print("=" * 60)
    print("\n⚠️  Make sure:")
    print("   1. MIDI: 'IAC Driver Chord Trainer Output' → Ableton/synth")
    print("   2. Audio: Device {} is receiving sound".format(args.input_device))
    print("   3. Volume is audible but not clipping")
    print("\n📊 Chord list:")
    for i, (root, quality, midi, inv, oct) in enumerate(MINIMAL_CHORD_LIST, 1):
        chord_name = "{}{}".format(root, quality)
        print("   {}. {} - Inversion {} - MIDI: {}".format(i, chord_name, inv, midi))
    
    print("\nPress Enter to start, Ctrl+C to abort...")
    input()
    
    try:
        # Run minimal validation session
        success = trainer.run_training_session(
            chord_list=MINIMAL_CHORD_LIST,
            save_to_json=True
        )
        
        if success:
            print("\n✅ Minimal test complete!")
            print("   Validated: {}/6 chords".format(len(trainer.validated_chords)))
            
            # Train if requested
            if args.train_after and len(trainer.validated_chords) >= 4:
                print("\n🤖 Training ML model on {} samples...".format(len(trainer.validated_chords)))
                if trainer.train_ml_model():
                    print("\n✅ Training complete!")
                    print("   Model saved: models/chord_model_wav2vec.pkl")
                else:
                    print("\n❌ Training failed (need at least 10 samples)")
            elif args.train_after:
                print("\n⚠️  Not enough samples to train (need at least 10)")
                print("   Run full chord list or expand test")
        else:
            print("\n❌ Minimal test failed")
            
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except Exception as e:
        print("\n❌ Error: {}".format(e))
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

