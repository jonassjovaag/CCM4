"""
Hybrid Batch Trainer - Automatically chooses CPU or MPS based on dataset size
"""

import time
import torch
from typing import List, Dict, Any, Optional
from memory.memory_buffer import MemoryBuffer
from memory.polyphonic_audio_oracle import PolyphonicAudioOracle
from memory.polyphonic_audio_oracle_mps import PolyphonicAudioOracleMPS


class HybridBatchTrainer:
    """
    Hybrid batch trainer that automatically chooses CPU or MPS based on dataset size
    
    Performance Analysis:
    - CPU-only: ~5-10 events/sec (consistent, good for large datasets)
    - MPS GPU: ~15-20 events/sec (fast for small datasets, overhead for large)
    - Threshold: ~5000 events (where MPS overhead becomes significant)
    """
    
    def __init__(self, 
                 distance_threshold: float = 0.35,  # Increased for 768D Wav2Vec features (cosine distance)
                 distance_function: str = 'cosine',  # FIXED: Use cosine for high-D features (768D Wav2Vec)
                 chord_similarity_weight: float = 0.3,
                 cpu_threshold: int = 5000):
        """
        Initialize hybrid batch trainer
        
        Args:
            distance_threshold: Distance threshold for similarity
            distance_function: Distance function to use
            chord_similarity_weight: Weight for chord similarity
            cpu_threshold: Switch to CPU when dataset exceeds this size
        """
        self.distance_threshold = distance_threshold
        self.distance_function = distance_function
        self.chord_similarity_weight = chord_similarity_weight
        self.cpu_threshold = cpu_threshold
        
        # Initialize components
        self.memory_buffer = MemoryBuffer()
        
        # Will be initialized based on dataset size
        self.audio_oracle: Optional[PolyphonicAudioOracle] = None
        self.audio_oracle_mps: Optional[PolyphonicAudioOracleMPS] = None
        self.use_mps = False
        self.device = "cpu"
        
        # Performance tracking
        self.training_stats = {
            'total_events': 0,
            'training_time': 0.0,
            'events_per_second': 0.0,
            'device_used': 'unknown',
            'switch_reason': 'none'
        }
    
    def _choose_device(self, dataset_size: int) -> str:
        """
        Choose optimal device based on dataset size
        
        Args:
            dataset_size: Number of events to process
            
        Returns:
            'cpu' or 'mps'
        """
        if dataset_size <= self.cpu_threshold:
            return 'mps'
        else:
            return 'cpu'
    
    def _initialize_oracle(self, device: str, feature_dimensions: int = 15):
        """
        Initialize the appropriate AudioOracle based on device choice
        
        Args:
            device: 'cpu' or 'mps'
            feature_dimensions: Number of feature dimensions (default 15, or 22 with hybrid perception)
        """
        if device == 'mps' and torch.backends.mps.is_available():
            print(f"🚀 Initializing MPS GPU AudioOracle (optimized for small datasets)")
            self.audio_oracle_mps = PolyphonicAudioOracleMPS(
                distance_threshold=self.distance_threshold,
                distance_function=self.distance_function,
                feature_dimensions=feature_dimensions,
                adaptive_threshold=False,  # Disabled for 768D features - use fixed threshold
                chord_similarity_weight=self.chord_similarity_weight,
                use_mps=True
            )
            self.use_mps = True
            self.device = "mps"
        else:
            print(f"💻 Initializing CPU AudioOracle (optimized for large datasets)")
            self.audio_oracle = PolyphonicAudioOracle(
                distance_threshold=self.distance_threshold,
                distance_function=self.distance_function,
                feature_dimensions=feature_dimensions,
                adaptive_threshold=False,  # Disabled for 768D features - use fixed threshold
                chord_similarity_weight=self.chord_similarity_weight
            )
            self.use_mps = False
            self.device = "cpu"
    
    def train_from_events(self, events: List[Any], file_info: Any = None) -> bool:
        """
        Train AudioOracle from events with automatic device selection
        
        Args:
            events: List of Event objects
            file_info: File information object
            
        Returns:
            True if training successful
        """
        try:
            dataset_size = len(events)
            print(f"🎓 Hybrid Training: {dataset_size} events detected")
            
            # Detect feature dimensions from first event
            feature_dims = 15  # Default
            if events and len(events) > 0:
                first_event = events[0]
                if isinstance(first_event, dict) and 'features' in first_event:
                    features = first_event['features']
                    if isinstance(features, list):
                        feature_dims = len(features)
                    elif hasattr(features, '__len__'):
                        feature_dims = len(features)
            
            # Choose optimal device
            chosen_device = self._choose_device(dataset_size)
            print(f"🎯 Device Selection: {chosen_device.upper()} "
                  f"({'GPU' if chosen_device == 'mps' else 'CPU'})")
            
            if dataset_size <= self.cpu_threshold:
                print(f"⚡ Reason: Small dataset ({dataset_size} ≤ {self.cpu_threshold}) - MPS GPU optimal")
            else:
                print(f"⚡ Reason: Large dataset ({dataset_size} > {self.cpu_threshold}) - CPU optimal")
            
            # Initialize appropriate oracle with detected feature dimensions
            self._initialize_oracle(chosen_device, feature_dimensions=feature_dims)
            
            # Process events
            print(f"🔄 Processing {dataset_size} events...")
            for event in events:
                # Convert Event to dictionary format
                if hasattr(event, 'to_dict'):
                    event_data = event.to_dict()
                    # Add additional attributes
                    event_data.update({
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
                    })
                else:
                    event_data = event
                
                # Add to memory buffer
                self.memory_buffer.add_moment(event_data)
            
            # Convert events to simple format for AudioOracle
            musical_sequence = []
            for event in events:
                # Handle both dictionary and object events
                if isinstance(event, dict):
                    event_data = event.copy()
                else:
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

                # CRITICAL: Include 768D Wav2Vec/MERT features from FeatureAnalysisStage
                # Keep as NumPy arrays for performance (pickle handles them efficiently)
                if hasattr(event, 'features') and event.features is not None:
                    import numpy as np
                    features_array = event.features
                    # Keep NumPy arrays (avoids expensive list→array conversion in oracle)
                    if isinstance(features_array, np.ndarray):
                        event_data['features'] = features_array
                    else:
                        # Convert non-arrays to NumPy for consistency
                        event_data['features'] = np.array(features_array, dtype=np.float32)

                # Include other dual perception fields
                if hasattr(event, 'wav2vec_features') and event.wav2vec_features is not None:
                    import numpy as np
                    wav2vec_array = event.wav2vec_features
                    # Keep NumPy arrays (avoids expensive list→array conversion in oracle)
                    if isinstance(wav2vec_array, np.ndarray):
                        event_data['wav2vec_features'] = wav2vec_array
                    else:
                        # Convert non-arrays to NumPy for consistency
                        event_data['wav2vec_features'] = np.array(wav2vec_array, dtype=np.float32)

                if hasattr(event, 'gesture_token'):
                    event_data['gesture_token'] = event.gesture_token

                if hasattr(event, 'chord'):
                    event_data['chord'] = event.chord

                if hasattr(event, 'consonance'):
                    event_data['consonance'] = float(event.consonance)

                musical_sequence.append(event_data)
            
            # Train with progress indicator
            start_time = time.time()
            success = self._train_with_progress(musical_sequence)
            training_time = time.time() - start_time
            
            # Update stats
            self.training_stats.update({
                'total_events': dataset_size,
                'training_time': training_time,
                'events_per_second': dataset_size / training_time if training_time > 0 else 0,
                'device_used': chosen_device,
                'switch_reason': f"{dataset_size} {'≤' if dataset_size <= self.cpu_threshold else '>'} {self.cpu_threshold}"
            })
            
            return success
            
        except Exception as e:
            print(f"❌ Error during hybrid training: {e}")
            return False
    
    def _train_with_progress(self, musical_sequence: List[Dict]) -> bool:
        """
        Train AudioOracle with progress indicator
        
        Args:
            musical_sequence: List of musical event data
            
        Returns:
            True if training successful
        """
        try:
            total_events = len(musical_sequence)
            print(f"🎓 Training Polyphonic AudioOracle with {total_events} events...")
            
            # Process events one by one for progress tracking
            processed = 0
            last_progress = 0
            
            for event_data in musical_sequence:
                # Extract musical features from transformer insights
                musical_features = self._extract_musical_features(event_data)
                
                # Add single event to AudioOracle with musical features
                if self.use_mps:
                    self.audio_oracle_mps.add_sequence([musical_features])
                    # Capture harmonic data for autonomous root progression (Groven method)
                    if 'fundamental_freq' in event_data and event_data['fundamental_freq'] > 0:
                        state_id = self.audio_oracle_mps.last  # Current state ID
                        self.audio_oracle_mps.fundamentals[state_id] = float(event_data['fundamental_freq'])
                    if 'consonance' in event_data:
                        state_id = self.audio_oracle_mps.last
                        self.audio_oracle_mps.consonances[state_id] = float(event_data['consonance'])
                else:
                    self.audio_oracle.add_sequence([musical_features])
                    # Capture harmonic data for autonomous root progression (Groven method)
                    if 'fundamental_freq' in event_data and event_data['fundamental_freq'] > 0:
                        state_id = self.audio_oracle.last  # Current state ID
                        self.audio_oracle.fundamentals[state_id] = float(event_data['fundamental_freq'])
                    if 'consonance' in event_data:
                        state_id = self.audio_oracle.last
                        self.audio_oracle.consonances[state_id] = float(event_data['consonance'])
                
                processed += 1
                
                # Update progress bar every 1% or every 100 events
                current_progress = int((processed / total_events) * 100)
                if current_progress > last_progress or processed % 100 == 0 or processed == total_events:
                    progress = (processed / total_events) * 100
                    bar_length = 25
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
                    
                    device_str = f"[{self.device.upper()}]"
                    print(f"\r🎓 {device_str} [{bar}] {progress:.1f}% ({processed}/{total_events}) {eta_str}", end='', flush=True)
                    last_progress = current_progress
            
            print()  # New line after progress bar
            
            # Get training results
            if self.use_mps:
                patterns = self.audio_oracle_mps.find_patterns()
                harmonic_patterns = self.audio_oracle_mps.find_harmonic_patterns()
                polyphonic_patterns = self.audio_oracle_mps.find_polyphonic_patterns()
            else:
                patterns = self.audio_oracle.find_patterns()
                harmonic_patterns = self.audio_oracle.find_harmonic_patterns()
                polyphonic_patterns = self.audio_oracle.find_polyphonic_patterns()
            
            print(f"✅ Training successful: {len(patterns)} patterns, {len(harmonic_patterns)} harmonic patterns, {len(polyphonic_patterns)} polyphonic patterns")
            return True
            
        except Exception as e:
            print(f"\n❌ Error in training: {e}")
            return False
    
    def save_model(self, filepath: str) -> bool:
        """
        Save the trained model
        
        Args:
            filepath: Path to save the model
            
        Returns:
            True if successful
        """
        try:
            if self.use_mps and self.audio_oracle_mps:
                return self.audio_oracle_mps.save_to_file(filepath)
            elif self.audio_oracle:
                return self.audio_oracle.save_to_file(filepath)
            else:
                print("❌ No trained model to save")
                return False
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            return False
    
    def get_training_stats(self) -> Dict[str, Any]:
        """
        Get training statistics
        
        Returns:
            Dictionary with training statistics
        """
        stats = self.training_stats.copy()
        
        # Add device-specific stats
        if self.use_mps and self.audio_oracle_mps:
            oracle_stats = self.audio_oracle_mps.get_statistics()
        elif self.audio_oracle:
            oracle_stats = self.audio_oracle.get_statistics()
        else:
            oracle_stats = {}
        
        stats.update({
            'patterns_found': oracle_stats.get('total_patterns', 0),
            'chord_patterns': oracle_stats.get('chord_patterns', 0),
            'polyphonic_frames': oracle_stats.get('polyphonic_frames', 0),
            'total_states': oracle_stats.get('total_states', 0),
            'sequence_length': oracle_stats.get('sequence_length', 0),
            'cpu_threshold': self.cpu_threshold,
            'mps_available': torch.backends.mps.is_available()
        })
        
        return stats
    
    def _extract_musical_features(self, event_data: Dict) -> Dict:
        """
        Extract musical features from transformer insights for AudioOracle training
        
        Args:
            event_data: Enhanced event data with transformer insights
            
        Returns:
            Dict with musical features for AudioOracle
        """
        # Handle case where event_data might not be a dict
        if not isinstance(event_data, dict):
            print(f"⚠️ Warning: event_data is not a dict, got {type(event_data)}: {event_data}")
            # Return basic features if not a dict
            return {
                't': 0,
                'features': [],
                'significance_score': 0.5,
                'sampling_reason': 'unknown',
                'chord_tension': 0.5,
                'key_stability': 0.5,
                'tempo': 120,
                'tempo_stability': 0.5,
                'hierarchical_level': 'phrase',
                'structural_importance': 0.5,
                'rhythmic_density': 0.5,
                'rhythmic_syncopation': 0.0,
                'stream_id': 0,
                'stream_confidence': 0.5
            }
        
        # CRITICAL FIX: Start with ALL audio features from event_data
        # This preserves f0, midi, rms_db, centroid, gesture_token, etc.
        musical_features = event_data.copy()
        
        # CRITICAL: Explicitly preserve gesture_token if present
        if 'gesture_token' in event_data:
            musical_features['gesture_token'] = event_data['gesture_token']
        
        # Ensure minimum required fields exist
        musical_features.setdefault('t', 0)
        musical_features.setdefault('features', [])
        musical_features.setdefault('significance_score', 0.5)
        musical_features.setdefault('sampling_reason', 'unknown')
        
        # Extract transformer insights
        transformer_insights = event_data.get('transformer_insights', {})
        
        # PRIORITY: Use correlation insights (real chord data) if available
        correlation_insights = event_data.get('correlation_insights', {})
        if correlation_insights:
            # Use real chord data from correlation analysis
            musical_features['chord'] = correlation_insights.get('chord', 'C')
            musical_features['chord_tension'] = correlation_insights.get('chord_tension', 0.5)
            musical_features['key_stability'] = correlation_insights.get('key_stability', 0.5)
            musical_features['chord_change_rate'] = correlation_insights.get('chord_change_rate', 0.0)
            musical_features['harmonic_diversity'] = correlation_insights.get('harmonic_diversity', 0.0)
            
            # Extract key signature from transformer insights as fallback
            key_signature = transformer_insights.get('key_signature', 'C major')
            musical_features['key_signature'] = key_signature
        elif transformer_insights:
            # Fallback to transformer insights if no correlation data
            chord_progression = transformer_insights.get('chord_progression', [])
            if chord_progression:
                # Use the most recent chord in the progression
                current_chord = chord_progression[-1] if chord_progression else 'C'
                musical_features['chord'] = current_chord
                
                # Calculate chord tension based on chord type
                musical_features['chord_tension'] = self._calculate_chord_tension(current_chord)
                
                # Extract key signature
                key_signature = transformer_insights.get('key_signature', 'C major')
                musical_features['key_signature'] = key_signature
                musical_features['key_stability'] = self._calculate_key_stability(key_signature, current_chord)
            
            # Extract tempo information
            tempo_analysis = transformer_insights.get('tempo_analysis', {})
            if tempo_analysis:
                musical_features['tempo'] = tempo_analysis.get('tempo', 120)
                musical_features['tempo_stability'] = tempo_analysis.get('stability', 0.5)
        
        # Extract hierarchical insights
        hierarchical_level = event_data.get('hierarchical_level', 'phrase')
        if isinstance(hierarchical_level, str):
            musical_features['hierarchical_level'] = hierarchical_level
            musical_features['structural_importance'] = 0.5  # Default importance
        elif isinstance(hierarchical_level, dict):
            musical_features['hierarchical_level'] = hierarchical_level.get('level', 'phrase')
            musical_features['structural_importance'] = hierarchical_level.get('importance', 0.5)
        
        # Extract rhythmic insights
        rhythmic_insights = event_data.get('rhythmic_insights', {})
        if rhythmic_insights:
            # Handle numpy arrays in rhythmic insights
            tempo_value = rhythmic_insights.get('tempo', 120)
            if hasattr(tempo_value, 'item'):  # numpy array
                tempo_value = tempo_value.item()
            
            syncopation_value = rhythmic_insights.get('syncopation', 0.0)
            if hasattr(syncopation_value, 'item'):  # numpy array
                syncopation_value = syncopation_value.item()
            
            musical_features['rhythmic_density'] = rhythmic_insights.get('density', 0.5)
            musical_features['rhythmic_syncopation'] = syncopation_value
            musical_features['rhythmic_tempo'] = tempo_value
        
        # Extract stream information
        stream_info = event_data.get('stream_info', {})
        if stream_info:
            musical_features['stream_id'] = stream_info.get('stream_id', 0)
            musical_features['stream_confidence'] = stream_info.get('confidence', 0.5)
        else:
            # Create polyphonic streams based on musical features
            # Use chord tension and rhythmic density to create different streams
            chord_tension = musical_features.get('chord_tension', 0.5)
            rhythmic_density = musical_features.get('rhythmic_density', 0.5)
            
            # Create 3-5 streams based on musical characteristics
            if chord_tension < 0.3:
                stream_id = 0  # Low tension stream
            elif chord_tension < 0.7:
                stream_id = 1  # Medium tension stream
            else:
                stream_id = 2  # High tension stream
            
            # Add rhythmic variation
            if rhythmic_density > 0.7:
                stream_id = (stream_id + 3) % 5  # High density gets different stream
            
            musical_features['stream_id'] = stream_id
            musical_features['stream_confidence'] = 0.8  # High confidence for synthetic streams
        
        return musical_features
    
    def _calculate_chord_tension(self, chord: str) -> float:
        """Calculate chord tension based on chord type"""
        chord_tensions = {
            'C': 0.0, 'C#': 0.1, 'D': 0.2, 'D#': 0.3, 'E': 0.4, 'F': 0.5,
            'F#': 0.6, 'G': 0.7, 'G#': 0.8, 'A': 0.9, 'A#': 1.0, 'B': 0.1,
            # Extended chords
            'Cm': 0.3, 'Cm7': 0.4, 'C7': 0.6, 'Cmaj7': 0.2, 'Cdim': 0.8,
            'Caug': 0.7, 'Csus2': 0.3, 'Csus4': 0.4
        }
        return chord_tensions.get(chord, 0.5)
    
    def _calculate_key_stability(self, key_signature: str, current_chord: str) -> float:
        """Calculate key stability based on chord-key relationship"""
        # Simple key stability calculation
        if key_signature.lower() in current_chord.lower():
            return 0.9  # Chord is in the key
        else:
            return 0.3  # Chord is outside the key
    
    def get_performance_summary(self) -> str:
        """
        Get performance summary string
        
        Returns:
            Formatted performance summary
        """
        stats = self.get_training_stats()
        
        summary = f"""
🎯 Hybrid Training Performance Summary:
   • Device Used: {stats['device_used'].upper()} ({'GPU' if stats['device_used'] == 'mps' else 'CPU'})
   • Switch Reason: {stats['switch_reason']}
   • Events Processed: {stats['total_events']:,}
   • Training Time: {stats['training_time']:.2f}s
   • Performance: {stats['events_per_second']:.1f} events/sec
   • Patterns Found: {stats['patterns_found']}
   • Chord Patterns: {stats['chord_patterns']}
   • Polyphonic Frames: {stats['polyphonic_frames']}
   • CPU Threshold: {stats['cpu_threshold']:,} events
   • MPS Available: {stats['mps_available']}
        """
        
        return summary.strip()
