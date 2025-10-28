# audio_file_learning/learn_from_files_mps.py
# MPS-Accelerated Command-line interface for learning from audio files
# Standalone script for training MPS AudioOracle from pre-recorded audio

import os
import sys
import argparse
import glob
from typing import List

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_file_learning.file_processor import AudioFileProcessor
from audio_file_learning.batch_trainer_mps import MPSBatchTrainer

# Import PyTorch to check MPS availability
import torch


def find_audio_files(directory: str) -> List[str]:
    """
    Find all audio files in a directory
    
    Args:
        directory: Directory to search
        
    Returns:
        List of audio file paths
    """
    audio_extensions = ['*.wav', '*.mp3', '*.flac', '*.m4a', '*.aiff', '*.ogg']
    audio_files = []
    
    for ext in audio_extensions:
        pattern = os.path.join(directory, '**', ext)
        audio_files.extend(glob.glob(pattern, recursive=True))
    
    return sorted(audio_files)


def main():
    """Main command-line interface for MPS-accelerated learning"""
    parser = argparse.ArgumentParser(
        description='MPS-Accelerated Learning from audio files for Drift Engine AI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Learn from a single file with MPS acceleration
  python learn_from_files_mps.py --file "jazz_sample.wav"
  
  # Learn from all files in a directory
  python learn_from_files_mps.py --dir "training_samples/"
  
  # Learn from multiple specific files
  python learn_from_files_mps.py --files "song1.wav" "song2.mp3" "song3.flac"
  
  # Save trained model
  python learn_from_files_mps.py --file "sample.wav" --output "jazz_model_mps.json"
  
  # Load existing model and continue training
  python learn_from_files_mps.py --file "new_sample.wav" --load "existing_model.json"
  
  # Force CPU-only mode (disable MPS)
  python learn_from_files_mps.py --file "sample.wav" --no-mps
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--file', '-f', 
                           help='Single audio file to process')
    input_group.add_argument('--dir', '-d', 
                           help='Directory containing audio files')
    input_group.add_argument('--files', nargs='+', 
                           help='Multiple audio files to process')
    
    # Model options
    parser.add_argument('--output', '-o', 
                       default='trained_audio_oracle_mps.json',
                       help='Output model file (default: trained_audio_oracle_mps.json)')
    parser.add_argument('--load', '-l', 
                       help='Load existing model and continue training')
    
    # Processing options
    parser.add_argument('--sample-rate', type=int, default=44100,
                       help='Sample rate for processing (default: 44100)')
    parser.add_argument('--hop-length', type=int, default=256,
                       help='Hop length for frame processing (default: 256)')
    parser.add_argument('--frame-length', type=int, default=2048,
                       help='Frame length for analysis (default: 2048)')
    parser.add_argument('--max-events', type=int, default=None,
                       help='Maximum number of events to process (for testing)')
    
    # AudioOracle options
    parser.add_argument('--distance-threshold', type=float, default=0.15,
                       help='Distance threshold for AudioOracle (default: 0.15)')
    parser.add_argument('--distance-function', 
                       choices=['euclidean', 'cosine', 'manhattan', 'weighted_euclidean', 'chebyshev', 'minkowski'],
                       default='euclidean',
                       help='Distance function for AudioOracle (default: euclidean)')
    parser.add_argument('--feature-dimensions', type=int, default=6,
                       help='Number of feature dimensions (default: 6)')
    parser.add_argument('--adaptive-threshold', action='store_true',
                       help='Use adaptive thresholding')
    
    # MPS options
    parser.add_argument('--no-mps', action='store_true',
                       help='Disable MPS acceleration (force CPU-only)')
    parser.add_argument('--mps-info', action='store_true',
                       help='Show MPS availability information')
    
    # Output options
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--stats', action='store_true',
                       help='Show detailed statistics')
    
    args = parser.parse_args()
    
    # Check MPS availability
    mps_available = torch.backends.mps.is_available()
    use_mps = not args.no_mps and mps_available
    
    if args.mps_info:
        print("🎯 MPS Information:")
        print(f"   MPS Available: {mps_available}")
        print(f"   PyTorch Version: {torch.__version__}")
        print(f"   Device: {torch.device('mps') if mps_available else 'CPU only'}")
        if mps_available:
            print(f"   MPS Backend: {torch.backends.mps.is_built()}")
        return 0
    
    # Determine input files
    input_files = []
    
    if args.file:
        if not os.path.exists(args.file):
            print(f"❌ File not found: {args.file}")
            return 1
        input_files = [args.file]
        
    elif args.dir:
        if not os.path.exists(args.dir):
            print(f"❌ Directory not found: {args.dir}")
            return 1
        input_files = find_audio_files(args.dir)
        if not input_files:
            print(f"❌ No audio files found in: {args.dir}")
            return 1
            
    elif args.files:
        for file in args.files:
            if not os.path.exists(file):
                print(f"❌ File not found: {file}")
                return 1
        input_files = args.files
    
    print(f"🎯 Drift Engine AI - MPS-Accelerated Audio File Learning")
    print(f"========================================================")
    print(f"📁 Input files: {len(input_files)}")
    for file in input_files:
        print(f"  • {os.path.basename(file)}")
    
    print(f"\n🎯 MPS Configuration:")
    print(f"   MPS Available: {mps_available}")
    print(f"   Using GPU: {use_mps}")
    print(f"   Device: {torch.device('mps') if use_mps else torch.device('cpu')}")
    
    # Initialize components
    print(f"\n🔧 Initializing MPS components...")
    
    processor = AudioFileProcessor(
        sample_rate=args.sample_rate,
        hop_length=args.hop_length,
        frame_length=args.frame_length,
        max_events=args.max_events
    )
    
    trainer = MPSBatchTrainer(
        distance_threshold=args.distance_threshold,
        distance_function=args.distance_function,
        feature_dimensions=args.feature_dimensions,
        adaptive_threshold=args.adaptive_threshold,
        use_mps=use_mps
    )
    
    # Load existing model if specified
    if args.load:
        if not os.path.exists(args.load):
            print(f"❌ Model file not found: {args.load}")
            return 1
        print(f"📥 Loading existing MPS model: {args.load}")
        if not trainer.load_model(args.load):
            print(f"❌ Failed to load MPS model")
            return 1
    
    # Process audio files
    print(f"\n🎵 Processing audio files...")
    file_results = processor.process_multiple_files(input_files)
    
    if not file_results:
        print(f"❌ No files processed successfully")
        return 1
    
    # Train MPS AudioOracle
    print(f"\n🎓 Training MPS AudioOracle...")
    success = trainer.train_from_multiple_files(file_results)
    
    if not success:
        print(f"❌ MPS Training failed")
        return 1
    
    # Save model
    print(f"\n💾 Saving MPS model...")
    if not trainer.save_model(args.output):
        print(f"❌ Failed to save MPS model")
        return 1
    
    # Show statistics
    if args.stats:
        trainer.print_training_summary()
    
    print(f"\n✅ MPS-Accelerated audio file learning complete!")
    print(f"📁 Model saved to: {args.output}")
    
    # Performance summary
    stats = trainer.get_model_statistics()
    training_time = stats['training_stats']['training_time']
    total_events = stats['training_stats']['total_events']
    
    if total_events > 0:
        events_per_second = total_events / training_time
        print(f"⚡ Performance: {events_per_second:.1f} events/second")
        print(f"🎯 Acceleration: {'MPS GPU' if use_mps else 'CPU'}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
