# audio_file_learning/learn_from_files.py
# Command-line interface for learning from audio files
# Standalone script for training AudioOracle from pre-recorded audio

import os
import sys
import argparse
import glob
from typing import List

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_file_learning.file_processor import AudioFileProcessor
from audio_file_learning.batch_trainer import BatchTrainer

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
    """Main command-line interface"""
    parser = argparse.ArgumentParser(
        description='Learn from audio files for Drift Engine AI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Learn from a single file
  python learn_from_files.py --file "jazz_sample.wav"
  
  # Learn from all files in a directory
  python learn_from_files.py --dir "training_samples/"
  
  # Learn from multiple specific files
  python learn_from_files.py --files "song1.wav" "song2.mp3" "song3.flac"
  
  # Save trained model
  python learn_from_files.py --file "sample.wav" --output "jazz_model.json"
  
  # Load existing model and continue training
  python learn_from_files.py --file "new_sample.wav" --load "existing_model.json"
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
                       default='trained_audio_oracle.json',
                       help='Output model file (default: trained_audio_oracle.json)')
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
    
    # Output options
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--stats', action='store_true',
                       help='Show detailed statistics')
    
    args = parser.parse_args()
    
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
    
    print(f"🎵 Drift Engine AI - Audio File Learning")
    print(f"========================================")
    print(f"📁 Input files: {len(input_files)}")
    for file in input_files:
        print(f"  • {os.path.basename(file)}")
    
    # Initialize components
    print(f"\\n🔧 Initializing components...")
    
    processor = AudioFileProcessor(
        sample_rate=args.sample_rate,
        hop_length=args.hop_length,
        frame_length=args.frame_length,
        max_events=args.max_events
    )
    
    trainer = BatchTrainer(
        distance_threshold=args.distance_threshold,
        distance_function=args.distance_function,
        feature_dimensions=args.feature_dimensions,
        adaptive_threshold=args.adaptive_threshold
    )
    
    # Load existing model if specified
    if args.load:
        if not os.path.exists(args.load):
            print(f"❌ Model file not found: {args.load}")
            return 1
        print(f"📥 Loading existing model: {args.load}")
        if not trainer.load_model(args.load):
            print(f"❌ Failed to load model")
            return 1
    
    # Process audio files
    print(f"\\n🎵 Processing audio files...")
    file_results = processor.process_multiple_files(input_files)
    
    if not file_results:
        print(f"❌ No files processed successfully")
        return 1
    
    # Train AudioOracle
    print(f"\\n🎓 Training AudioOracle...")
    success = trainer.train_from_multiple_files(file_results)
    
    if not success:
        print(f"❌ Training failed")
        return 1
    
    # Save model
    print(f"\\n💾 Saving model...")
    if not trainer.save_model(args.output):
        print(f"❌ Failed to save model")
        return 1
    
    # Show statistics
    if args.stats:
        trainer.print_training_summary()
    
    print(f"\\n✅ Audio file learning complete!")
    print(f"📁 Model saved to: {args.output}")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
