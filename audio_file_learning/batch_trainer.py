# audio_file_learning/batch_trainer.py
# Batch Trainer for AudioOracle using audio files
# Trains AudioOracle models from pre-recorded audio data

import os
import json
import time
from typing import List, Dict, Optional, Tuple
import numpy as np

# Import AudioOracle from the main system
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from memory.audio_oracle import AudioOracle
from memory.memory_buffer import MemoryBuffer, MusicalMoment

class BatchTrainer:
    """
    Trains AudioOracle models from audio files
    Compatible with existing Drift Engine AI system
    """
    
    def __init__(self, 
                 distance_threshold: float = 0.15,
                 distance_function: str = 'euclidean',
                 feature_dimensions: int = 6,
                 adaptive_threshold: bool = True):
        """
        Initialize batch trainer
        
        Args:
            distance_threshold: Threshold for AudioOracle similarity
            distance_function: Distance function for AudioOracle
            feature_dimensions: Number of feature dimensions
            adaptive_threshold: Whether to use adaptive thresholding
        """
        self.audio_oracle = AudioOracle(
            distance_threshold=distance_threshold,
            distance_function=distance_function,
            feature_dimensions=feature_dimensions,
            adaptive_threshold=adaptive_threshold
        )
        
        self.memory_buffer = MemoryBuffer()
        self.training_stats = {
            'files_processed': 0,
            'total_events': 0,
            'total_duration': 0.0,
            'training_time': 0.0,
            'patterns_found': 0
        }
    
    def train_from_events(self, events: List, file_info: Dict) -> bool:
        """
        Train AudioOracle from extracted events
        
        Args:
            events: List of Event objects from file processing
            file_info: Information about the source file
            
        Returns:
            True if training successful
        """
        try:
            print(f"🎓 Training AudioOracle from {len(events)} events...")
            
            # Convert events directly to AudioOracle format (skip memory buffer for now)
            musical_sequence = []
            for event in events:
                # Create simple event data dictionary (same format as live system)
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
                    'duration': float(getattr(event, 'duration', 0.5)),
                    'attack_time': float(getattr(event, 'attack_time', 0.1)),
                    'release_time': float(getattr(event, 'release_time', 0.3)),
                    'tempo': float(getattr(event, 'tempo', 120.0)),
                    'beat_position': float(getattr(event, 'beat_position', 0.0))
                }
                musical_sequence.append(event_data)
            
            # Train AudioOracle
            print(f"Training AudioOracle with {len(musical_sequence)} events...")
            print(f"First event data: {musical_sequence[0] if musical_sequence else 'None'}")
            print(f"First event data types: {[(k, type(v)) for k, v in musical_sequence[0].items()] if musical_sequence else 'None'}")
            
            # Add progress indicator for AudioOracle training
            success = self._train_audio_oracle_with_progress(musical_sequence)
            
            if success:
                # Update training stats
                self.training_stats['files_processed'] += 1
                self.training_stats['total_events'] += len(events)
                self.training_stats['total_duration'] += getattr(file_info, 'duration', 0.0)
                
                # Get pattern count
                patterns = self.audio_oracle.find_patterns(musical_sequence[:10])  # Sample for pattern count
                self.training_stats['patterns_found'] = len(patterns)
                
                print(f"✅ Training successful: {len(events)} events processed")
                return True
            else:
                print(f"❌ Training failed")
                return False
                
        except Exception as e:
            print(f"❌ Error during training: {e}")
            return False
    
    def _train_audio_oracle_with_progress(self, musical_sequence: List[Dict]) -> bool:
        """
        Train AudioOracle with fine-grained progress indicator (1% updates)
        
        Args:
            musical_sequence: List of musical event data
            
        Returns:
            True if training successful
        """
        try:
            total_events = len(musical_sequence)
            print(f"🎓 Training AudioOracle with {total_events} events...")
            
            # Process events one by one for fine-grained progress
            processed = 0
            last_progress = 0
            
            for event_data in musical_sequence:
                # Add single event to AudioOracle
                self.audio_oracle.add_sequence([event_data])
                processed += 1
                
                # Calculate current progress
                current_progress = int((processed / total_events) * 100)
                
                # Update progress bar every 1% or every 100 events (whichever comes first)
                if current_progress > last_progress or processed % 100 == 0 or processed == total_events:
                    progress = (processed / total_events) * 100
                    bar_length = 25  # Even shorter bar to prevent line wrapping
                    filled_length = int(bar_length * processed // total_events)
                    bar = '█' * filled_length + '░' * (bar_length - filled_length)
                    
                    # Add ETA estimation
                    if not hasattr(self, '_training_start_time'):
                        self._training_start_time = time.time()
                    
                    if processed > 0:
                        elapsed_time = time.time() - self._training_start_time
                        if elapsed_time > 0:
                            rate = processed / elapsed_time
                            remaining_events = total_events - processed
                            eta_seconds = remaining_events / rate if rate > 0 else 0
                            eta_minutes = eta_seconds / 60
                            eta_str = f"ETA:{eta_minutes:.1f}m" if eta_minutes > 1 else f"ETA:{eta_seconds:.0f}s"
                        else:
                            eta_str = "ETA:calc"
                    else:
                        eta_str = "ETA:calc"
                    
                    # Ensure we use carriage return for in-place updates
                    print(f"\r🎓 [{bar}] {progress:.1f}% ({processed}/{total_events}) {eta_str}", end='', flush=True)
                    last_progress = current_progress
            
            print()  # New line after progress bar
            return True
            
        except Exception as e:
            print(f"\n❌ Error in AudioOracle training: {e}")
            return False
    
    def train_from_multiple_files(self, file_results: Dict[str, Tuple[List, Dict]]) -> bool:
        """
        Train AudioOracle from multiple processed files
        
        Args:
            file_results: Dictionary from AudioFileProcessor.process_multiple_files()
            
        Returns:
            True if all training successful
        """
        print(f"🎓 Batch training from {len(file_results)} files...")
        
        start_time = time.time()
        success_count = 0
        
        for filepath, (events, file_info) in file_results.items():
            print(f"\\n📁 Processing: {os.path.basename(filepath)}")
            
            if self.train_from_events(events, file_info):
                success_count += 1
            else:
                print(f"⚠️  Failed to train from {filepath}")
        
        # Update training time
        self.training_stats['training_time'] = time.time() - start_time
        
        print(f"\\n🎯 Batch training complete:")
        print(f"  • Files processed: {success_count}/{len(file_results)}")
        print(f"  • Total events: {self.training_stats['total_events']}")
        print(f"  • Total duration: {self.training_stats['total_duration']:.2f}s")
        print(f"  • Training time: {self.training_stats['training_time']:.2f}s")
        
        return success_count == len(file_results)
    
    def save_model(self, filepath: str) -> bool:
        """
        Save trained AudioOracle model
        
        Args:
            filepath: Path to save model
            
        Returns:
            True if save successful
        """
        try:
            # Save AudioOracle model
            success = self.audio_oracle.save_to_file(filepath)
            
            if success:
                # Save training stats
                stats_file = filepath.replace('.json', '_stats.json')
                with open(stats_file, 'w') as f:
                    json.dump(self.training_stats, f, indent=2)
                
                print(f"✅ Model saved to: {filepath}")
                print(f"✅ Stats saved to: {stats_file}")
                return True
            else:
                print(f"❌ Failed to save model")
                return False
                
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """
        Load existing AudioOracle model
        
        Args:
            filepath: Path to model file
            
        Returns:
            True if load successful
        """
        try:
            success = self.audio_oracle.load_from_file(filepath)
            
            if success:
                print(f"✅ Model loaded from: {filepath}")
                
                # Try to load training stats
                stats_file = filepath.replace('.json', '_stats.json')
                if os.path.exists(stats_file):
                    with open(stats_file, 'r') as f:
                        self.training_stats = json.load(f)
                    print(f"✅ Training stats loaded")
                
                return True
            else:
                print(f"❌ Failed to load model")
                return False
                
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def get_model_statistics(self) -> Dict:
        """
        Get comprehensive statistics about the trained model
        
        Returns:
            Dictionary with model and training statistics
        """
        oracle_stats = self.audio_oracle.get_statistics()
        
        combined_stats = {
            'training_stats': self.training_stats,
            'oracle_stats': oracle_stats,
            'model_info': {
                'distance_threshold': self.audio_oracle.distance_threshold,
                'distance_function': self.audio_oracle.distance_function,
                'feature_dimensions': self.audio_oracle.feature_dimensions,
                'adaptive_threshold': self.audio_oracle.adaptive_threshold
            }
        }
        
        return combined_stats
    
    def print_training_summary(self):
        """Print a summary of the training results"""
        stats = self.get_model_statistics()
        
        print(f"\\n📊 Training Summary:")
        print(f"  • Files processed: {stats['training_stats']['files_processed']}")
        print(f"  • Total events: {stats['training_stats']['total_events']}")
        print(f"  • Total duration: {stats['training_stats']['total_duration']:.2f}s")
        print(f"  • Training time: {stats['training_stats']['training_time']:.2f}s")
        print(f"  • Patterns found: {stats['training_stats']['patterns_found']}")
        
        print(f"\\n🧠 AudioOracle Statistics:")
        print(f"  • Total states: {stats['oracle_stats'].get('total_states', 0)}")
        print(f"  • Sequence length: {stats['oracle_stats'].get('sequence_length', 0)}")
        print(f"  • Distance function: {stats['model_info']['distance_function']}")
        print(f"  • Distance threshold: {stats['model_info']['distance_threshold']}")
        print(f"  • Is trained: {stats['oracle_stats'].get('is_trained', False)}")
