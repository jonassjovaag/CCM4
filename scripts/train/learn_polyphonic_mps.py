# audio_file_learning/learn_polyphonic_mps.py
# Complete Polyphonic MPS Learning Pipeline
# Combines polyphonic processing with MPS acceleration

import os
import sys
import argparse
import glob
import time
import json
from typing import List, Dict, Any, Tuple

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_file_learning.polyphonic_processor import PolyphonicAudioProcessor
from memory.polyphonic_audio_oracle_mps import PolyphonicAudioOracleMPS

# Import PyTorch to check MPS availability
import torch


class PolyphonicMPSBatchTrainer:
    """
    MPS-Accelerated trainer for Polyphonic AudioOracle models
    """
    
    def __init__(self, 
                 distance_threshold: float = 0.15,
                 distance_function: str = 'euclidean',
                 feature_dimensions: int = 15,  # Enhanced for polyphonic features
                 adaptive_threshold: bool = True,
                 chord_similarity_weight: float = 0.3,
                 use_mps: bool = True):
        """
        Initialize Polyphonic MPS batch trainer
        
        Args:
            distance_threshold: Threshold for AudioOracle similarity
            distance_function: Distance function for AudioOracle
            feature_dimensions: Number of feature dimensions
            adaptive_threshold: Whether to use adaptive thresholding
            chord_similarity_weight: Weight for chord similarity
            use_mps: Whether to use MPS acceleration
        """
        self.audio_oracle = PolyphonicAudioOracleMPS(
            distance_threshold=distance_threshold,
            distance_function=distance_function,
            feature_dimensions=feature_dimensions,
            adaptive_threshold=adaptive_threshold,
            chord_similarity_weight=chord_similarity_weight,
            use_mps=use_mps
        )
        
        self.training_stats = {
            'files_processed': 0,
            'total_events': 0,
            'total_duration': 0.0,
            'training_time': 0.0,
            'patterns_found': 0,
            'chord_patterns_found': 0,
            'polyphonic_frames': 0,
            'mps_acceleration': use_mps and torch.backends.mps.is_available(),
            'device': str(torch.device("mps") if use_mps and torch.backends.mps.is_available() else torch.device("cpu"))
        }
        
        print(f"🎵 Polyphonic MPS Batch Trainer initialized:")
        print(f"   Device: {self.training_stats['device']}")
        print(f"   MPS Available: {torch.backends.mps.is_available()}")
        print(f"   Using GPU: {self.training_stats['mps_acceleration']}")
        print(f"   Chord Similarity Weight: {chord_similarity_weight}")
    
    def train_from_polyphonic_events(self, events: List, file_info: Dict) -> bool:
        """
        Train Polyphonic AudioOracle from extracted events
        
        Args:
            events: List of Event objects from polyphonic processing
            file_info: Information about the source file
            
        Returns:
            True if training successful
        """
        try:
            print(f"🎓 Training Polyphonic AudioOracle from {len(events)} events...")
            
            # Convert events to polyphonic sequence format
            musical_sequence = []
            for event in events:
                # Create enhanced event data dictionary
                event_data = {
                    't': float(event.t),
                    'midi': int(event.midi),
                    'cents': float(event.cents),
                    'f0': float(event.f0),
                    'rms_db': float(event.rms_db),
                    'centroid': float(event.centroid),
                    'ioi': float(event.ioi),
                    'onset': bool(event.onset),
                    'rolloff': float(getattr(event, 'rolloff', 0.0)),
                    'bandwidth': float(getattr(event, 'bandwidth', 0.0)),
                    'contrast': float(getattr(event, 'contrast', 0.0)),
                    'flatness': float(getattr(event, 'flatness', 0.0)),
                    'mfcc_1': float(getattr(event, 'mfcc_1', 0.0)),
                    'mfcc_2': float(getattr(event, 'mfcc_2', 0.0)),
                    'mfcc_3': float(getattr(event, 'mfcc_3', 0.0)),
                    'zcr': float(getattr(event, 'zcr', 0.0)),
                    'attack_time': float(getattr(event, 'attack_time', 0.1)),
                    'release_time': float(getattr(event, 'release_time', 0.3)),
                    'tempo': float(getattr(event, 'tempo', 120.0)),
                    'beat_position': float(getattr(event, 'beat_position', 0.0)),
                    # Polyphonic information
                    'polyphonic_pitches': getattr(event, 'polyphonic_pitches', [event.f0]),
                    'polyphonic_midi': getattr(event, 'polyphonic_midi', [event.midi]),
                    'polyphonic_cents': getattr(event, 'polyphonic_cents', [event.cents]),
                    'chord_quality': getattr(event, 'chord_quality', 'single'),
                    'root_note': getattr(event, 'root_note', event.midi),
                    'num_pitches': getattr(event, 'num_pitches', 1)
                }
                musical_sequence.append(event_data)
            
            # Train Polyphonic AudioOracle
            print(f"Training Polyphonic AudioOracle with {len(musical_sequence)} events...")
            success = self.audio_oracle.add_polyphonic_sequence(musical_sequence)
            
            if success:
                # Update training stats
                self.training_stats['files_processed'] += 1
                self.training_stats['total_events'] += len(events)
                self.training_stats['total_duration'] += getattr(file_info, 'duration', 0.0)
                
                # Get pattern counts
                patterns = self.audio_oracle.find_patterns(musical_sequence[:10])
                chord_patterns = self.audio_oracle.find_chord_patterns(min_freq=1, min_len=1)
                
                self.training_stats['patterns_found'] = len(patterns)
                self.training_stats['chord_patterns_found'] = len(chord_patterns)
                self.training_stats['polyphonic_frames'] = self.audio_oracle.stats.get('polyphonic_frames', 0)
                
                print(f"✅ Polyphonic Training successful: {len(events)} events processed")
                print(f"   Patterns found: {len(patterns)}")
                print(f"   Chord patterns found: {len(chord_patterns)}")
                print(f"   Polyphonic frames: {self.training_stats['polyphonic_frames']}")
                return True
            else:
                print(f"❌ Polyphonic Training failed")
                return False
                
        except Exception as e:
            print(f"❌ Error during polyphonic training: {e}")
            return False
    
    def train_from_multiple_files(self, file_results: Dict[str, Tuple[List, Dict]]) -> bool:
        """
        Train Polyphonic AudioOracle from multiple processed files
        
        Args:
            file_results: Dictionary from PolyphonicAudioProcessor.process_multiple_files()
            
        Returns:
            True if all training successful
        """
        print(f"🎓 Polyphonic MPS Batch training from {len(file_results)} files...")
        
        start_time = time.time()
        success_count = 0
        
        for filepath, (events, file_info) in file_results.items():
            print(f"\n📁 Processing: {os.path.basename(filepath)}")
            
            if self.train_from_polyphonic_events(events, file_info):
                success_count += 1
            else:
                print(f"⚠️  Failed to train from {filepath}")
        
        # Update training time
        self.training_stats['training_time'] = time.time() - start_time
        
        print(f"\n🎵 Polyphonic MPS Batch training complete:")
        print(f"  • Files processed: {success_count}/{len(file_results)}")
        print(f"  • Total events: {self.training_stats['total_events']}")
        print(f"  • Total duration: {self.training_stats['total_duration']:.2f}s")
        print(f"  • Training time: {self.training_stats['training_time']:.2f}s")
        print(f"  • Patterns found: {self.training_stats['patterns_found']}")
        print(f"  • Chord patterns found: {self.training_stats['chord_patterns_found']}")
        print(f"  • Polyphonic frames: {self.training_stats['polyphonic_frames']}")
        print(f"  • MPS Acceleration: {self.training_stats['mps_acceleration']}")
        print(f"  • Device: {self.training_stats['device']}")
        
        return success_count == len(file_results)
    
    def save_model(self, filepath: str) -> bool:
        """
        Save trained Polyphonic AudioOracle model
        
        Args:
            filepath: Path to save model
            
        Returns:
            True if save successful
        """
        try:
            # Save Polyphonic AudioOracle model
            success = self.audio_oracle.save_to_file(filepath)
            
            if success:
                # Save training stats
                stats_file = filepath.replace('.json', '_stats.json')
                with open(stats_file, 'w') as f:
                    json.dump(self.training_stats, f, indent=2)
                
                print(f"✅ Polyphonic Model saved to: {filepath}")
                print(f"✅ Stats saved to: {stats_file}")
                return True
            else:
                print(f"❌ Failed to save polyphonic model")
                return False
                
        except Exception as e:
            print(f"❌ Error saving polyphonic model: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """
        Load existing Polyphonic AudioOracle model
        
        Args:
            filepath: Path to model file
            
        Returns:
            True if load successful
        """
        try:
            success = self.audio_oracle.load_from_file(filepath)
            
            if success:
                print(f"✅ Polyphonic Model loaded from: {filepath}")
                
                # Try to load training stats
                stats_file = filepath.replace('.json', '_stats.json')
                if os.path.exists(stats_file):
                    with open(stats_file, 'r') as f:
                        self.training_stats = json.load(f)
                    print(f"✅ Training stats loaded")
                
                return True
            else:
                print(f"❌ Failed to load polyphonic model")
                return False
                
        except Exception as e:
            print(f"❌ Error loading polyphonic model: {e}")
            return False
    
    def get_model_statistics(self) -> Dict:
        """
        Get comprehensive statistics about the trained polyphonic model
        
        Returns:
            Dictionary with model and training statistics
        """
        oracle_stats = self.audio_oracle.get_polyphonic_statistics()
        
        combined_stats = {
            'training_stats': self.training_stats,
            'oracle_stats': oracle_stats,
            'model_info': {
                'distance_threshold': self.audio_oracle.distance_threshold,
                'distance_function': self.audio_oracle.distance_function,
                'feature_dimensions': self.audio_oracle.feature_dimensions,
                'adaptive_threshold': self.audio_oracle.adaptive_threshold,
                'chord_similarity_weight': self.audio_oracle.chord_similarity_weight,
                'mps_acceleration': self.training_stats['mps_acceleration'],
                'device': self.training_stats['device']
            }
        }
        
        return combined_stats
    
    def print_training_summary(self):
        """Print a summary of the polyphonic training results"""
        stats = self.get_model_statistics()
        
        print(f"\n📊 Polyphonic Training Summary:")
        print(f"  • Files processed: {stats['training_stats']['files_processed']}")
        print(f"  • Total events: {stats['training_stats']['total_events']}")
        print(f"  • Total duration: {stats['training_stats']['total_duration']:.2f}s")
        print(f"  • Training time: {stats['training_stats']['training_time']:.2f}s")
        print(f"  • Patterns found: {stats['training_stats']['patterns_found']}")
        print(f"  • Chord patterns found: {stats['training_stats']['chord_patterns_found']}")
        print(f"  • Polyphonic frames: {stats['training_stats']['polyphonic_frames']}")
        print(f"  • MPS Acceleration: {stats['training_stats']['mps_acceleration']}")
        print(f"  • Device: {stats['training_stats']['device']}")
        
        print(f"\n🧠 Polyphonic AudioOracle Statistics:")
        print(f"  • Total states: {stats['oracle_stats'].get('total_states', 0)}")
        print(f"  • Sequence length: {stats['oracle_stats'].get('sequence_length', 0)}")
        print(f"  • Distance function: {stats['model_info']['distance_function']}")
        print(f"  • Distance threshold: {stats['model_info']['distance_threshold']}")
        print(f"  • Chord similarity weight: {stats['model_info']['chord_similarity_weight']}")
        print(f"  • Is trained: {stats['oracle_stats'].get('is_trained', False)}")
        print(f"  • Using GPU: {stats['model_info']['mps_acceleration']}")
        
        # Show chord type distribution
        chord_types = stats['oracle_stats'].get('chord_types', {})
        if chord_types:
            print(f"\n🎵 Chord Type Distribution:")
            for chord_type, count in sorted(chord_types.items(), key=lambda x: x[1], reverse=True):
                print(f"  • {chord_type}: {count}")


def main():
    """Main command-line interface for polyphonic MPS learning"""
    parser = argparse.ArgumentParser(
        description='Polyphonic MPS-Accelerated Learning from audio files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Learn from a single polyphonic file
  python learn_polyphonic_mps.py --file "jazz_chord_sample.wav"
  
  # Learn from all files in a directory
  python learn_polyphonic_mps.py --dir "polyphonic_samples/"
  
  # Learn with custom chord similarity weight
  python learn_polyphonic_mps.py --file "sample.wav" --chord-weight 0.5
  
  # Test with limited events
  python learn_polyphonic_mps.py --file "sample.wav" --max-events 1000
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--file', '-f', help='Single audio file to process')
    input_group.add_argument('--dir', '-d', help='Directory containing audio files')
    input_group.add_argument('--files', nargs='+', help='Multiple audio files to process')
    
    # Model options
    parser.add_argument('--output', '-o', default='polyphonic_audio_oracle.json',
                       help='Output model file (default: polyphonic_audio_oracle.json)')
    parser.add_argument('--load', '-l', help='Load existing model and continue training')
    
    # Processing options
    parser.add_argument('--sample-rate', type=int, default=44100,
                       help='Sample rate for processing (default: 44100)')
    parser.add_argument('--hop-length', type=int, default=256,
                       help='Hop length for frame processing (default: 256)')
    parser.add_argument('--frame-length', type=int, default=2048,
                       help='Frame length for analysis (default: 2048)')
    parser.add_argument('--max-events', type=int, default=None,
                       help='Maximum number of events to process (for testing)')
    parser.add_argument('--max-pitches', type=int, default=4,
                       help='Maximum pitches to detect per frame (default: 4)')
    
    # AudioOracle options
    parser.add_argument('--distance-threshold', type=float, default=0.15,
                       help='Distance threshold for AudioOracle (default: 0.15)')
    parser.add_argument('--distance-function', 
                       choices=['euclidean', 'cosine', 'manhattan'],
                       default='euclidean',
                       help='Distance function for AudioOracle (default: euclidean)')
    parser.add_argument('--feature-dimensions', type=int, default=15,
                       help='Number of feature dimensions (default: 15)')
    parser.add_argument('--adaptive-threshold', action='store_true',
                       help='Use adaptive thresholding')
    parser.add_argument('--chord-weight', type=float, default=0.3,
                       help='Weight for chord similarity in distance calculation (default: 0.3)')
    
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
        print("🎵 MPS Information:")
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
    
    print(f"🎵 Polyphonic MPS-Accelerated Audio Learning")
    print(f"=============================================")
    print(f"📁 Input files: {len(input_files)}")
    for file in input_files:
        print(f"  • {os.path.basename(file)}")
    
    print(f"\n🎵 MPS Configuration:")
    print(f"   MPS Available: {mps_available}")
    print(f"   Using GPU: {use_mps}")
    print(f"   Device: {torch.device('mps') if use_mps else torch.device('cpu')}")
    print(f"   Max Pitches: {args.max_pitches}")
    print(f"   Chord Weight: {args.chord_weight}")
    
    # Initialize components
    print(f"\n🔧 Initializing polyphonic components...")
    
    processor = PolyphonicAudioProcessor(
        sample_rate=args.sample_rate,
        hop_length=args.hop_length,
        frame_length=args.frame_length,
        max_events=args.max_events,
        max_pitches=args.max_pitches
    )
    
    trainer = PolyphonicMPSBatchTrainer(
        distance_threshold=args.distance_threshold,
        distance_function=args.distance_function,
        feature_dimensions=args.feature_dimensions,
        adaptive_threshold=args.adaptive_threshold,
        chord_similarity_weight=args.chord_weight,
        use_mps=use_mps
    )
    
    # Load existing model if specified
    if args.load:
        if not os.path.exists(args.load):
            print(f"❌ Model file not found: {args.load}")
            return 1
        print(f"📥 Loading existing polyphonic model: {args.load}")
        if not trainer.load_model(args.load):
            print(f"❌ Failed to load polyphonic model")
            return 1
    
    # Process audio files
    print(f"\n🎵 Processing polyphonic audio files...")
    file_results = {}
    
    for filepath in input_files:
        try:
            events, file_info = processor.process_audio_file(filepath)
            file_results[filepath] = (events, file_info)
        except Exception as e:
            print(f"❌ Error processing {filepath}: {e}")
            continue
    
    if not file_results:
        print(f"❌ No files processed successfully")
        return 1
    
    # Train Polyphonic AudioOracle
    print(f"\n🎓 Training Polyphonic AudioOracle...")
    success = trainer.train_from_multiple_files(file_results)
    
    if not success:
        print(f"❌ Polyphonic Training failed")
        return 1
    
    # Save model
    print(f"\n💾 Saving polyphonic model...")
    if not trainer.save_model(args.output):
        print(f"❌ Failed to save polyphonic model")
        return 1
    
    # Show statistics
    if args.stats:
        trainer.print_training_summary()
    
    print(f"\n✅ Polyphonic MPS learning complete!")
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


def find_audio_files(directory: str) -> List[str]:
    """Find all audio files in a directory"""
    audio_extensions = ['*.wav', '*.mp3', '*.flac', '*.m4a', '*.aiff', '*.ogg']
    audio_files = []
    
    for ext in audio_extensions:
        pattern = os.path.join(directory, '**', ext)
        audio_files.extend(glob.glob(pattern, recursive=True))
    
    return sorted(audio_files)


if __name__ == '__main__':
    import time
    import json
    sys.exit(main())
