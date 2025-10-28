# memory/polyphonic_audio_oracle_mps.py
# MPS-Accelerated Polyphonic AudioOracle Implementation
# Combines polyphonic features with Apple Silicon GPU acceleration

import numpy as np
import json
import os
import time
from collections import defaultdict, deque
from typing import List, Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass
import math

# Import PyTorch for MPS acceleration
import torch

# Import base classes
from .polyphonic_audio_oracle import PolyphonicAudioOracle, PolyphonicAudioFrame


class PolyphonicAudioOracleMPS(PolyphonicAudioOracle):
    """
    MPS-Accelerated Polyphonic AudioOracle
    
    Combines polyphonic features (multi-pitch, chord recognition) with GPU acceleration
    using PyTorch MPS for Apple Silicon Macs.
    """
    
    def __init__(self, 
                 distance_threshold: float = 0.15, 
                 distance_function: str = 'euclidean',
                 max_pattern_length: int = 50,
                 feature_dimensions: int = 15,
                 adaptive_threshold: bool = True,
                 chord_similarity_weight: float = 0.3,
                 use_mps: bool = True):
        """
        Initialize MPS-Accelerated Polyphonic AudioOracle
        
        Args:
            distance_threshold: Threshold θ for similarity matching
            distance_function: Distance function ('euclidean', 'cosine', 'manhattan')
            max_pattern_length: Maximum length of patterns to recognize
            feature_dimensions: Number of dimensions in feature vectors
            adaptive_threshold: Whether to adapt threshold based on data distribution
            chord_similarity_weight: Weight for chord similarity in distance calculation
            use_mps: Whether to use MPS acceleration
        """
        super().__init__(
            distance_threshold=distance_threshold,
            distance_function=distance_function,
            max_pattern_length=max_pattern_length,
            feature_dimensions=feature_dimensions,
            adaptive_threshold=adaptive_threshold,
            chord_similarity_weight=chord_similarity_weight
        )
        
        # MPS configuration
        self.use_mps = use_mps and torch.backends.mps.is_available()
        self.device = torch.device("mps") if self.use_mps else torch.device("cpu")
        
        # MPS-specific attributes
        self.mps_enabled = self.use_mps
        self.gpu_memory_usage = 0
        
        print(f"🎵 Polyphonic AudioOracle MPS initialized:")
        print(f"   Device: {self.device}")
        print(f"   MPS Available: {torch.backends.mps.is_available()}")
        print(f"   Using GPU: {self.use_mps}")
        print(f"   Feature dimensions: {feature_dimensions}")
        print(f"   Chord similarity weight: {chord_similarity_weight}")
    
    def _calculate_distance(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """
        MPS-accelerated distance calculation with chord similarity
        
        Args:
            features1: First feature vector
            features2: Second feature vector
            
        Returns:
            Enhanced distance score
        """
        try:
            if self.use_mps:
                # Convert to PyTorch tensors and move to MPS
                tensor1 = torch.tensor(features1, dtype=torch.float32).to(self.device)
                tensor2 = torch.tensor(features2, dtype=torch.float32).to(self.device)
                
                # Calculate basic distance on GPU
                if self.distance_function == 'euclidean':
                    basic_distance = torch.norm(tensor1 - tensor2).item()
                elif self.distance_function == 'cosine':
                    # Cosine distance = 1 - cosine similarity
                    cosine_sim = torch.nn.functional.cosine_similarity(
                        tensor1.unsqueeze(0), tensor2.unsqueeze(0)
                    ).item()
                    basic_distance = 1.0 - cosine_sim
                elif self.distance_function == 'manhattan':
                    basic_distance = torch.sum(torch.abs(tensor1 - tensor2)).item()
                else:
                    basic_distance = torch.norm(tensor1 - tensor2).item()
                
                # If we have polyphonic information, enhance with chord similarity
                if len(features1) >= 15 and len(features2) >= 15:
                    # Extract chord features
                    chord_features1 = tensor1[-3:]  # Last 3 features are chord-related
                    chord_features2 = tensor2[-3:]
                    
                    # Calculate chord distance on GPU
                    chord_distance = torch.norm(chord_features1 - chord_features2).item()
                    
                    # Combine basic and chord distances
                    enhanced_distance = (
                        (1.0 - self.chord_similarity_weight) * basic_distance +
                        self.chord_similarity_weight * chord_distance
                    )
                    
                    return enhanced_distance
                else:
                    return basic_distance
            else:
                # Fallback to CPU calculation
                return super()._calculate_distance(features1, features2)
                
        except Exception as e:
            print(f"Error in MPS distance calculation: {e}")
            # Fallback to CPU calculation
            return super()._calculate_distance(features1, features2)
    
    def _find_similar_frames(self, features: np.ndarray, max_results: int = 10) -> List[Tuple[int, float]]:
        """
        MPS-accelerated similar frame finding
        
        Args:
            features: Query feature vector
            max_results: Maximum number of results to return
            
        Returns:
            List of (frame_id, distance) tuples
        """
        try:
            if not self.use_mps or len(self.sequence) == 0:
                return super()._find_similar_frames(features, max_results)
            
            # Convert query features to tensor
            query_tensor = torch.tensor(features, dtype=torch.float32).to(self.device)
            
            # Get all stored features as tensors
            stored_features = []
            frame_ids = []
            
            for frame_id in self.sequence:
                if hasattr(self, 'frames') and frame_id in self.frames:
                    frame_features = self.frames[frame_id].features
                    stored_features.append(frame_features)
                    frame_ids.append(frame_id)
                elif hasattr(self, 'audio_frames') and frame_id in self.audio_frames:
                    frame_features = self.audio_frames[frame_id].features
                    stored_features.append(frame_features)
                    frame_ids.append(frame_id)
            
            if not stored_features:
                return []
            
            # Convert to batch tensor
            stored_tensor = torch.tensor(np.array(stored_features), dtype=torch.float32).to(self.device)
            
            # Calculate distances in batch on GPU
            if self.distance_function == 'euclidean':
                distances = torch.norm(stored_tensor - query_tensor.unsqueeze(0), dim=1)
            elif self.distance_function == 'cosine':
                cosine_sims = torch.nn.functional.cosine_similarity(
                    stored_tensor, query_tensor.unsqueeze(0), dim=1
                )
                distances = 1.0 - cosine_sims
            elif self.distance_function == 'manhattan':
                distances = torch.sum(torch.abs(stored_tensor - query_tensor.unsqueeze(0)), dim=1)
            else:
                distances = torch.norm(stored_tensor - query_tensor.unsqueeze(0), dim=1)
            
            # Filter by threshold and sort
            valid_indices = distances < self.distance_threshold
            valid_distances = distances[valid_indices]
            valid_frame_ids = [frame_ids[i] for i in range(len(frame_ids)) if valid_indices[i]]
            
            # Sort by distance
            sorted_indices = torch.argsort(valid_distances)
            results = []
            
            for i in sorted_indices[:max_results]:
                frame_id = valid_frame_ids[i.item()]
                distance = valid_distances[i].item()
                results.append((frame_id, distance))
            
            return results
            
        except Exception as e:
            print(f"Error in MPS similar frame finding: {e}")
            # Fallback to CPU calculation with correct signature
            try:
                return super()._find_similar_frames(features)
            except:
                return []
    
    def add_polyphonic_sequence(self, musical_sequence: List[Dict]) -> bool:
        """
        MPS-accelerated polyphonic sequence addition
        
        Args:
            musical_sequence: List of event data dictionaries with polyphonic information
            
        Returns:
            True if successful
        """
        try:
            print(f"🎵 Adding polyphonic sequence with {len(musical_sequence)} events (MPS GPU)...")
            
            polyphonic_frames_added = 0
            start_time = time.time()
            
            for i, event_data in enumerate(musical_sequence):
                # Extract enhanced features
                features = self.extract_polyphonic_features(event_data)
                
                # Create polyphonic audio frame
                polyphonic_frame = PolyphonicAudioFrame(
                    timestamp=event_data.get('t', time.time()),
                    features=features,
                    audio_data=event_data,
                    frame_id=self.frame_counter,
                    polyphonic_pitches=event_data.get('polyphonic_pitches', [event_data.get('f0', 440.0)]),
                    polyphonic_midi=event_data.get('polyphonic_midi', [event_data.get('midi', 69)]),
                    polyphonic_cents=event_data.get('polyphonic_cents', [0.0]),
                    chord_quality=event_data.get('chord_quality', 'single'),
                    root_note=event_data.get('root_note', event_data.get('midi', 69)),
                    num_pitches=event_data.get('num_pitches', 1),
                    mfcc_2=event_data.get('mfcc_2', 0.0),
                    mfcc_3=event_data.get('mfcc_3', 0.0),
                    zcr=event_data.get('zcr', 0.0),
                    rolloff_85=event_data.get('rolloff_85', 0.0),
                    rolloff_95=event_data.get('rolloff_95', 0.0)
                )
                
                # Add frame to AudioOracle (uses MPS-accelerated distance calculation)
                success = self.add_audio_frame(features, event_data)
                
                if success:
                    polyphonic_frames_added += 1
                    
                    # Track chord progressions
                    if i > 0:
                        prev_chord = musical_sequence[i-1].get('chord_quality', 'single')
                        curr_chord = event_data.get('chord_quality', 'single')
                        progression = f"{prev_chord}->{curr_chord}"
                        self.chord_progressions[progression] += 1
                    
                    # Track harmonic patterns
                    if len(polyphonic_frame.polyphonic_pitches) > 1:
                        harmonic_pattern = tuple(sorted(polyphonic_frame.polyphonic_midi))
                        self.harmonic_patterns[harmonic_pattern] += 1
                
                # Update progress
                if (i + 1) % 100 == 0:
                    progress = ((i + 1) / len(musical_sequence)) * 100
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed if elapsed > 0 else 0
                    print(f"\r🎵 MPS Polyphonic Learning: {progress:.1f}% ({i + 1}/{len(musical_sequence)}) - {rate:.1f} events/sec", end='', flush=True)
            
            training_time = time.time() - start_time
            print(f"\n✅ MPS Polyphonic sequence added: {polyphonic_frames_added} frames in {training_time:.2f}s")
            
            # Update statistics
            self.stats['polyphonic_frames'] = polyphonic_frames_added
            self.stats['chord_progressions'] = len(self.chord_progressions)
            self.stats['harmonic_patterns'] = len(self.harmonic_patterns)
            self.stats['mps_acceleration'] = self.use_mps
            self.stats['training_time'] = training_time
            
            # Update chord type statistics
            for event_data in musical_sequence:
                chord_type = event_data.get('chord_quality', 'single')
                self.stats['chord_types'][chord_type] += 1
            
            # Calculate average pitches per frame
            total_pitches = sum(event_data.get('num_pitches', 1) for event_data in musical_sequence)
            self.stats['average_pitches_per_frame'] = total_pitches / len(musical_sequence) if musical_sequence else 0.0
            
            return True
            
        except Exception as e:
            print(f"❌ Error adding MPS polyphonic sequence: {e}")
            return False
    
    def get_mps_statistics(self) -> Dict[str, Any]:
        """
        Get MPS-specific statistics
        
        Returns:
            Dictionary with MPS statistics
        """
        base_stats = self.get_polyphonic_statistics()
        
        mps_stats = {
            'mps_enabled': self.use_mps,
            'device': str(self.device),
            'gpu_memory_usage': self.gpu_memory_usage,
            'mps_available': torch.backends.mps.is_available(),
            'training_time': self.stats.get('training_time', 0.0)
        }
        
        # Combine with base statistics
        combined_stats = {**base_stats, **mps_stats}
        
        return combined_stats
    
    def save_to_file(self, filepath: str) -> bool:
        """
        Save MPS Polyphonic AudioOracle to file
        
        Args:
            filepath: Path to save file
            
        Returns:
            True if successful
        """
        try:
            # Prepare data for serialization
            data = {
                'distance_threshold': self.distance_threshold,
                'distance_function': self.distance_function,
                'max_pattern_length': self.max_pattern_length,
                'feature_dimensions': self.feature_dimensions,
                'adaptive_threshold': self.adaptive_threshold,
                'chord_similarity_weight': self.chord_similarity_weight,
                'mps_enabled': self.use_mps,
                'device': str(self.device),
                'size': self.size,
                'last': self.last,
                'frame_counter': self.frame_counter,
                'sequence_length': self.sequence_length,
                'is_trained': self.is_trained,
                'last_update': self.last_update,
                'stats': dict(self.stats),
                'chord_progressions': dict(self.chord_progressions),
                'harmonic_patterns': {str(k): v for k, v in self.harmonic_patterns.items()},
                'distance_history': list(self.distance_history),
                'threshold_adjustments': getattr(self, 'threshold_adjustments', 0)
            }
            
            # Convert transitions to serializable format
            transitions_serializable = {}
            for (state, symbol), target_state in self.transitions.items():
                key = f"{state}_{symbol}"
                transitions_serializable[key] = target_state
            data['transitions'] = transitions_serializable
            
            # Convert suffix links to serializable format
            suffix_links_serializable = {}
            for state, link_state in self.suffix_links.items():
                suffix_links_serializable[str(state)] = link_state
            data['suffix_links'] = suffix_links_serializable
            
            # Convert states to serializable format
            states_serializable = {}
            for state_id, state_data in self.states.items():
                states_serializable[str(state_id)] = {
                    'len': state_data['len'],
                    'link': state_data['link'],
                    'next': {str(k): v for k, v in state_data['next'].items()}
                }
            data['states'] = states_serializable
            
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"✅ MPS Polyphonic AudioOracle saved to: {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving MPS polyphonic AudioOracle: {e}")
            return False
    
    def load_from_file(self, filepath: str) -> bool:
        """
        Load MPS Polyphonic AudioOracle from file
        
        Args:
            filepath: Path to load file
            
        Returns:
            True if successful
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Load basic parameters
            self.distance_threshold = data.get('distance_threshold', 0.15)
            self.distance_function = data.get('distance_function', 'euclidean')
            self.max_pattern_length = data.get('max_pattern_length', 50)
            self.feature_dimensions = data.get('feature_dimensions', 15)
            self.adaptive_threshold = data.get('adaptive_threshold', True)
            self.chord_similarity_weight = data.get('chord_similarity_weight', 0.3)
            
            # Load MPS parameters
            self.use_mps = data.get('mps_enabled', True) and torch.backends.mps.is_available()
            self.device = torch.device(data.get('device', 'mps')) if self.use_mps else torch.device('cpu')
            
            # Load state information
            self.size = data.get('size', 0)
            self.last = data.get('last', 0)
            self.frame_counter = data.get('frame_counter', 0)
            self.sequence_length = data.get('sequence_length', 0)
            self.is_trained = data.get('is_trained', False)
            self.last_update = data.get('last_update', time.time())
            
            # Load statistics
            self.stats = data.get('stats', {})
            self.chord_progressions = defaultdict(int, data.get('chord_progressions', {}))
            self.harmonic_patterns = defaultdict(int, data.get('harmonic_patterns', {}))
            self.distance_history = deque(data.get('distance_history', []), maxlen=1000)
            self.threshold_adjustments = data.get('threshold_adjustments', 0)
            
            # Load transitions
            self.transitions = {}
            for key, target_state in data.get('transitions', {}).items():
                state, symbol = key.split('_', 1)
                self.transitions[(int(state), symbol)] = target_state
            
            # Load suffix links
            self.suffix_links = {}
            for state, link_state in data.get('suffix_links', {}).items():
                self.suffix_links[int(state)] = link_state
            
            # Load states
            self.states = {}
            for state_id, state_data in data.get('states', {}).items():
                self.states[int(state_id)] = {
                    'len': state_data['len'],
                    'link': state_data['link'],
                    'next': {int(k): v for k, v in state_data['next'].items()}
                }
            
            print(f"✅ MPS Polyphonic AudioOracle loaded from: {filepath}")
            print(f"   Device: {self.device}")
            print(f"   MPS Enabled: {self.use_mps}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading MPS polyphonic AudioOracle: {e}")
            return False


def main():
    """Test MPS Polyphonic AudioOracle"""
    print("🎵 Testing MPS Polyphonic AudioOracle")
    print("=" * 50)
    
    # Create MPS polyphonic AudioOracle
    ao = PolyphonicAudioOracleMPS(distance_threshold=0.2, distance_function='euclidean')
    
    # Test with mock polyphonic data
    mock_polyphonic_data = [
        {
            't': time.time(),
            'midi': 60, 'rms_db': -20, 'f0': 440, 'centroid': 2000, 'ioi': 0.5, 'onset': 1,
            'rolloff': 3000, 'bandwidth': 1000, 'contrast': 0.6, 'flatness': 0.1, 'mfcc_1': 10.0,
            'mfcc_2': 5.0, 'mfcc_3': 2.0, 'zcr': 0.1, 'attack_time': 0.1,
            'polyphonic_pitches': [440.0, 523.25, 659.25],  # C-E-G chord
            'polyphonic_midi': [60, 64, 67],
            'polyphonic_cents': [0.0, 0.0, 0.0],
            'chord_quality': 'major',
            'root_note': 60,
            'num_pitches': 3
        },
        {
            't': time.time() + 0.5,
            'midi': 64, 'rms_db': -18, 'f0': 523, 'centroid': 2200, 'ioi': 0.3, 'onset': 1,
            'rolloff': 3200, 'bandwidth': 1100, 'contrast': 0.65, 'flatness': 0.12, 'mfcc_1': 12.0,
            'mfcc_2': 6.0, 'mfcc_3': 3.0, 'zcr': 0.15, 'attack_time': 0.08,
            'polyphonic_pitches': [523.25, 659.25, 783.99],  # E-G-B chord
            'polyphonic_midi': [64, 67, 71],
            'polyphonic_cents': [0.0, 0.0, 0.0],
            'chord_quality': 'major',
            'root_note': 64,
            'num_pitches': 3
        }
    ]
    
    # Add polyphonic sequence
    print("📝 Adding MPS polyphonic sequence...")
    success = ao.add_polyphonic_sequence(mock_polyphonic_data)
    
    if success:
        print("✅ MPS Polyphonic sequence added successfully!")
        
        # Get MPS statistics
        stats = ao.get_mps_statistics()
        print(f"\n📊 MPS Statistics:")
        print(f"  • MPS Enabled: {stats.get('mps_enabled', False)}")
        print(f"  • Device: {stats.get('device', 'unknown')}")
        print(f"  • Training Time: {stats.get('training_time', 0):.3f}s")
        print(f"  • Polyphonic Frames: {stats.get('polyphonic_frames', 0)}")
        print(f"  • Chord Progressions: {stats.get('chord_progressions', 0)}")
        print(f"  • Harmonic Patterns: {stats.get('harmonic_patterns', 0)}")
    
    print("\n✅ MPS Polyphonic AudioOracle test complete!")


if __name__ == "__main__":
    main()
