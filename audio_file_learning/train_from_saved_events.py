# audio_file_learning/train_from_saved_events.py
# Train MPS AudioOracle from pre-extracted events (no re-extraction needed)

import os
import sys
import pickle
import argparse
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_file_learning.batch_trainer_mps import MPSBatchTrainer
from audio_file_learning.save_extracted_events import load_extracted_events


def train_from_saved_events(events_file: str, output_model: str, **kwargs) -> bool:
    """
    Train MPS AudioOracle from pre-extracted events
    
    Args:
        events_file: Path to saved events pickle file
        output_model: Path to save trained model
        **kwargs: Additional training parameters
        
    Returns:
        True if training successful
    """
    print(f"🎯 Training MPS AudioOracle from saved events")
    print(f"📁 Events file: {events_file}")
    print(f"💾 Output model: {output_model}")
    
    # Load extracted events
    events, file_info = load_extracted_events(events_file)
    if not events:
        print(f"❌ Failed to load events from {events_file}")
        return False
    
    # Initialize MPS trainer
    trainer = MPSBatchTrainer(**kwargs)
    
    # Train from loaded events
    print(f"\n🎓 Training MPS AudioOracle...")
    success = trainer.train_from_events(events, file_info)
    
    if success:
        # Save trained model
        print(f"\n💾 Saving trained model...")
        model_saved = trainer.save_model(output_model)
        
        if model_saved:
            print(f"✅ Training complete!")
            trainer.print_training_summary()
            return True
        else:
            print(f"❌ Failed to save model")
            return False
    else:
        print(f"❌ Training failed")
        return False


def main():
    """Main command-line interface"""
    parser = argparse.ArgumentParser(
        description='Train MPS AudioOracle from pre-extracted events',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train from saved events
  python train_from_saved_events.py --events "grab_a_hold_events.pkl" --output "grab_a_hold_mps_model.json"
  
  # Train with custom parameters
  python train_from_saved_events.py --events "grab_a_hold_events.pkl" --output "model.json" --distance-threshold 0.2 --max-events 10000
        """
    )
    
    # Required arguments
    parser.add_argument('--events', '-e', required=True,
                       help='Path to saved events pickle file')
    parser.add_argument('--output', '-o', required=True,
                       help='Output model file path')
    
    # Optional training parameters
    parser.add_argument('--distance-threshold', type=float, default=0.15,
                       help='Distance threshold for AudioOracle')
    parser.add_argument('--distance-function', 
                       choices=['euclidean', 'cosine', 'manhattan'],
                       default='euclidean',
                       help='Distance function for AudioOracle')
    parser.add_argument('--feature-dimensions', type=int, default=6,
                       help='Number of feature dimensions')
    parser.add_argument('--adaptive-threshold', action='store_true',
                       help='Use adaptive thresholding')
    parser.add_argument('--no-mps', action='store_true',
                       help='Disable MPS acceleration')
    
    # Output options
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--stats', action='store_true',
                       help='Show detailed statistics')
    
    args = parser.parse_args()
    
    # Check if events file exists
    if not os.path.exists(args.events):
        print(f"❌ Events file not found: {args.events}")
        return 1
    
    # Prepare training parameters
    training_params = {
        'distance_threshold': args.distance_threshold,
        'distance_function': args.distance_function,
        'feature_dimensions': args.feature_dimensions,
        'adaptive_threshold': args.adaptive_threshold,
        'use_mps': not args.no_mps
    }
    
    print(f"🎯 MPS AudioOracle Training from Saved Events")
    print(f"=============================================")
    print(f"📁 Events file: {args.events}")
    print(f"💾 Output model: {args.output}")
    print(f"🎯 MPS acceleration: {not args.no_mps}")
    
    # Train from saved events
    success = train_from_saved_events(args.events, args.output, **training_params)
    
    if success:
        print(f"\n✅ Training completed successfully!")
        print(f"📁 Model saved to: {args.output}")
        return 0
    else:
        print(f"\n❌ Training failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
