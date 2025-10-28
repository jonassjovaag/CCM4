# audio_file_learning/learn_from_files_mps_complete.py
# Complete MPS-Accelerated Learning Pipeline
# Uses MPS for both feature extraction AND AudioOracle training

import os
import sys
import argparse
import glob
from typing import List

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_file_learning.file_processor_mps import MPSAudioFileProcessor
from audio_file_learning.batch_trainer_mps import MPSBatchTrainer

# Import PyTorch to check MPS availability
import torch


def main():
    """Complete MPS-accelerated learning pipeline"""
    parser = argparse.ArgumentParser(
        description='Complete MPS-Accelerated Learning Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Complete MPS pipeline
  python learn_from_files_mps_complete.py --file "audio.mp3" --max-events 5000
  
  # Test with small dataset
  python learn_from_files_mps_complete.py --file "audio.mp3" --max-events 1000
        """
    )
    
    # Input options
    parser.add_argument('--file', '-f', required=True, help='Audio file to process')
    parser.add_argument('--output', '-o', default='mps_complete_model.json', help='Output model file')
    
    # Processing options
    parser.add_argument('--max-events', type=int, default=None, help='Maximum events to process')
    parser.add_argument('--sample-rate', type=int, default=44100, help='Sample rate')
    parser.add_argument('--hop-length', type=int, default=256, help='Hop length')
    parser.add_argument('--frame-length', type=int, default=2048, help='Frame length')
    
    # AudioOracle options
    parser.add_argument('--distance-threshold', type=float, default=0.15, help='Distance threshold')
    parser.add_argument('--distance-function', choices=['euclidean', 'cosine', 'manhattan'], default='euclidean', help='Distance function')
    parser.add_argument('--feature-dimensions', type=int, default=6, help='Feature dimensions')
    parser.add_argument('--adaptive-threshold', action='store_true', help='Use adaptive thresholding')
    
    # MPS options
    parser.add_argument('--no-mps', action='store_true', help='Disable MPS acceleration')
    parser.add_argument('--mps-info', action='store_true', help='Show MPS information')
    
    # Output options
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--stats', action='store_true', help='Show detailed statistics')
    
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
    
    # Check if file exists
    if not os.path.exists(args.file):
        print(f"❌ File not found: {args.file}")
        return 1
    
    print(f"🎯 Complete MPS-Accelerated Learning Pipeline")
    print(f"=============================================")
    print(f"📁 Input file: {os.path.basename(args.file)}")
    print(f"💾 Output model: {args.output}")
    print(f"🎯 MPS Available: {mps_available}")
    print(f"🎯 Using GPU: {use_mps}")
    print(f"🎯 Device: {torch.device('mps') if use_mps else torch.device('cpu')}")
    
    # Initialize MPS components
    print(f"\n🔧 Initializing MPS components...")
    
    processor = MPSAudioFileProcessor(
        sample_rate=args.sample_rate,
        hop_length=args.hop_length,
        frame_length=args.frame_length,
        max_events=args.max_events,
        use_mps=use_mps
    )
    
    trainer = MPSBatchTrainer(
        distance_threshold=args.distance_threshold,
        distance_function=args.distance_function,
        feature_dimensions=args.feature_dimensions,
        adaptive_threshold=args.adaptive_threshold,
        use_mps=use_mps
    )
    
    # Process audio file
    print(f"\n🎵 Processing audio file...")
    events, file_info = processor.process_audio_file(args.file)
    
    if not events:
        print(f"❌ No events extracted")
        return 1
    
    # Train MPS AudioOracle
    print(f"\n🎓 Training MPS AudioOracle...")
    success = trainer.train_from_events(events, file_info)
    
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
    
    print(f"\n✅ Complete MPS pipeline finished!")
    print(f"📁 Model saved to: {args.output}")
    
    # Performance summary
    stats = trainer.get_model_statistics()
    training_time = stats['training_stats']['training_time']
    total_events = stats['training_stats']['total_events']
    
    if total_events > 0 and training_time > 0:
        events_per_second = total_events / training_time
        print(f"⚡ Performance: {events_per_second:.1f} events/second")
        print(f"🎯 Acceleration: {'MPS GPU' if use_mps else 'CPU'}")
    else:
        print(f"⚡ Performance: Training completed very quickly!")
        print(f"🎯 Acceleration: {'MPS GPU' if use_mps else 'CPU'}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
