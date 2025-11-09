#!/usr/bin/env python3
"""
Enhanced AudioOracle Implementation

This implements the AudioOracle innovations from the paper:
- Continuous feature vectors instead of discrete symbols
- Distance functions with thresholding (Euclidean, Cosine, Manhattan)
- Approximate matching with threshold θ
- Audio-specific feature processing
- Concatenative synthesis capabilities
- Multi-channel support for different musical attributes

Based on:
- Dubnov, S., Assayag, G., & Cont, A. (2007). "Audio Oracle: A New Algorithm for Fast Learning of Audio Structures"
- Allauzen, C., Crochemore, M., & Raffinot, M. (1999). "Factor Oracle: A New Structure for Pattern Matching"
"""

import numpy as np
import json
import os
import time
from collections import defaultdict, deque
from typing import List, Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass
import math


@dataclass
class AudioFrame:
    """Represents an audio frame with feature vector"""
    timestamp: float
    features: np.ndarray
    audio_data: Dict  # Original audio event data
    frame_id: int
    feature_type: str = "audio"  # Type of features (audio, spectral, cepstral, etc.)


@dataclass
class MusicalPattern:
    """Represents a learned musical pattern (compatibility with FactorOracle)"""
    pattern_id: int
    sequence: List[Any]
    occurrences: List[int]  # Positions where pattern appears
    frequency: int
    length: int
    first_occurrence: int
    last_occurrence: int


class AudioOracle:
    """
    Enhanced AudioOracle implementation based on the paper
    Extends Factor Oracle for continuous audio data with advanced features:
    
    Key Innovations:
    - Continuous feature vectors instead of discrete symbols
    - Multiple distance functions (Euclidean, Cosine, Manhattan)
    - Threshold-based approximate matching
    - Audio-specific feature extraction
    - Concatenative synthesis capabilities
    - Multi-channel support for different musical attributes
    """
    
    def __init__(self, 
                 distance_threshold: float = 0.1, 
                 distance_function: str = 'euclidean',
                 max_pattern_length: int = 50,
                 feature_dimensions: int = 6,
                 adaptive_threshold: bool = True):
        """
        Initialize AudioOracle
        
        Args:
            distance_threshold: Threshold θ for similarity matching
            distance_function: Distance function ('euclidean', 'cosine', 'manhattan', 'weighted_euclidean')
            max_pattern_length: Maximum length of patterns to recognize
            feature_dimensions: Number of dimensions in feature vectors
            adaptive_threshold: Whether to adapt threshold based on data distribution
        """
        # Core Factor Oracle structure (compatible with original)
        self.states = {}  # state_id -> state_data
        self.transitions = {}  # (state_id, frame_id) -> next_state_id
        self.suffix_links = {}  # state_id -> suffix_state_id
        
        # Musical pattern storage
        self.patterns = {}  # pattern_id -> MusicalPattern
        self.pattern_counter = 0
        
        # Sequence tracking
        self.sequence = []  # Current musical sequence (frame_ids)
        self.sequence_length = 0
        
        # AudioOracle specific enhancements
        self.distance_threshold = distance_threshold
        self.distance_function = distance_function
        self.max_pattern_length = max_pattern_length
        self.feature_dimensions = feature_dimensions
        self.adaptive_threshold = adaptive_threshold
        
        # Audio frame storage
        self.audio_frames = {}  # frame_id -> AudioFrame
        self.frame_counter = 0
        
        # Factor Oracle compatibility attributes
        self.size = 1  # Number of states
        self.last = 0  # Last state
        
        # Initialize first state
        self.states[0] = {'len': 0, 'link': -1, 'next': {}}
        
        # Distance function weights (for weighted distance)
        self.feature_weights = np.ones(feature_dimensions)
        
        # Adaptive threshold management
        self.distance_history = deque(maxlen=1000)  # Store recent distances
        self.threshold_history = deque(maxlen=100)  # Store threshold changes
        
        # Training status
        self.is_trained = False
        
        # Statistics
        self.stats = {
            'total_patterns': 0,
            'total_states': 0,
            'sequence_length': 0,
            'max_pattern_length': max_pattern_length,
            'is_trained': False,
            'last_update': None,
            'memory_usage': 0,
            'distance_threshold': distance_threshold,
            'distance_function': distance_function,
            'feature_dimensions': feature_dimensions,
            'adaptive_threshold': adaptive_threshold,
            'total_distances_calculated': 0,
            'average_distance': 0.0,
            'threshold_adjustments': 0,
            # Musical pattern statistics
            'harmonic_patterns': 0,
            'polyphonic_patterns': 0,
            'chord_progressions': 0,
            'rhythmic_patterns': 0,
            'musical_insights_used': 0
        }
        
        # Multi-channel support
        self.channels = {}  # channel_name -> AudioOracle instance
        self.active_channels = ['audio']  # Default channel
        
        # Concatenative synthesis
        self.synthesis_enabled = False
        self.synthesis_buffer = deque(maxlen=1000)
    
    def _calculate_distance(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """
        Calculate distance between two feature vectors using various distance functions
        
        Args:
            features1: First feature vector
            features2: Second feature vector
            
        Returns:
            float: Distance between the vectors
        """
        try:
            # Ensure vectors are numpy arrays and not None
            if features1 is None or features2 is None:
                return float('inf')
                
            f1 = np.array(features1, dtype=np.float64)
            f2 = np.array(features2, dtype=np.float64)
            
            # Check for invalid arrays
            if len(f1) == 0 or len(f2) == 0:
                return float('inf')
            
            # Ensure same dimensions
            if len(f1) != len(f2):
                min_len = min(len(f1), len(f2))
                f1 = f1[:min_len]
                f2 = f2[:min_len]
            
            if self.distance_function == 'euclidean':
                distance = np.linalg.norm(f1 - f2)
            elif self.distance_function == 'cosine':
                dot_product = np.dot(f1, f2)
                norm1 = np.linalg.norm(f1)
                norm2 = np.linalg.norm(f2)
                if norm1 == 0 or norm2 == 0:
                    distance = 1.0
                else:
                    cosine_sim = dot_product / (norm1 * norm2)
                    distance = 1 - cosine_sim
            elif self.distance_function == 'manhattan':
                distance = np.sum(np.abs(f1 - f2))
            elif self.distance_function == 'weighted_euclidean':
                weighted_diff = self.feature_weights[:len(f1)] * (f1 - f2)
                distance = np.sqrt(np.sum(weighted_diff ** 2))
            elif self.distance_function == 'chebyshev':
                distance = np.max(np.abs(f1 - f2))
            elif self.distance_function == 'minkowski':
                p = 3  # Minkowski parameter
                distance = np.sum(np.abs(f1 - f2) ** p) ** (1/p)
            else:
                raise ValueError(f"Unknown distance function: {self.distance_function}")
            
            # Update distance statistics
            self.distance_history.append(distance)
            self.stats['total_distances_calculated'] += 1
            
            # Update average distance
            if len(self.distance_history) > 0:
                self.stats['average_distance'] = np.mean(list(self.distance_history))
            
            # Adaptive threshold adjustment
            if self.adaptive_threshold and len(self.distance_history) > 100:
                self._adjust_threshold()
            
            return distance
            
        except Exception as e:
            print(f"Error calculating distance: {e}")
            return float('inf')
    
    def _adjust_threshold(self):
        """
        Adjust threshold based on recent distance distribution
        
        FIXED: Use percentile-based threshold for better stability with high-D features
        """
        if len(self.distance_history) < 50:
            return
        
        recent_distances = list(self.distance_history)[-100:]
        
        # For cosine distance (0-2 range), use mean + 0.5*std approach
        if self.distance_function == 'cosine':
            mean_dist = np.mean(recent_distances)
            std_dist = np.std(recent_distances)
            new_threshold = mean_dist + 0.5 * std_dist
            # Clamp to reasonable range for cosine [0.05, 0.5]
            new_threshold = np.clip(new_threshold, 0.05, 0.5)
        else:
            # For Euclidean/Manhattan on high-D features, use percentile
            # 25th percentile = accept closest 25% of distances
            new_threshold = np.percentile(recent_distances, 25)
            # Never let threshold grow beyond 2x the initial value
            max_threshold = self.stats['distance_threshold'] * 2.0
            new_threshold = min(new_threshold, max_threshold)
        
        # Only adjust if change is significant (>10%) and threshold is not zero
        if self.distance_threshold > 0 and abs(new_threshold - self.distance_threshold) / self.distance_threshold > 0.1:
            old_threshold = self.distance_threshold
            self.distance_threshold = new_threshold
            self.threshold_history.append((time.time(), old_threshold, new_threshold))
            self.stats['threshold_adjustments'] += 1
            self.stats['distance_threshold'] = new_threshold
    
    def _find_similar_frames(self, target_features: np.ndarray) -> List[int]:
        """Find frames with similar features within threshold"""
        similar_frames = []
        
        try:
            for frame_id, frame in self.audio_frames.items():
                if hasattr(frame, 'features') and frame.features is not None:
                    distance = self._calculate_distance(target_features, frame.features)
                    if distance < self.distance_threshold:
                        similar_frames.append(frame_id)
        except Exception as e:
            # If there's an error, return empty list to prevent crashes
            print(f"Warning: Error in _find_similar_frames: {e}")
            return []
        
        return similar_frames
    
    def extract_audio_features(self, event_data: Dict) -> np.ndarray:
        """
        Extract comprehensive audio features from event data, including musical insights
        
        Args:
            event_data: Audio event data dictionary with transformer insights
            
        Returns:
            np.ndarray: Feature vector with multiple audio and musical dimensions
        """
        try:
            # Handle case where event_data might not be a dict
            if not isinstance(event_data, dict):
                print(f"⚠️ Warning: event_data is not a dict in extract_audio_features, got {type(event_data)}: {event_data}")
                # Return default features if not a dict
                return np.zeros(self.feature_dimensions, dtype=np.float32)
            
            # Basic audio features
            midi_note = event_data.get('midi', 60)
            rms_db = event_data.get('rms_db', -20)
            f0 = event_data.get('f0', 440)
            centroid = event_data.get('centroid', 2000)
            ioi = event_data.get('ioi', 0.5)
            onset = event_data.get('onset', 0)
            
            # Musical features from transformer insights
            chord_tension = event_data.get('chord_tension', 0.5)
            key_stability = event_data.get('key_stability', 0.5)
            tempo = event_data.get('tempo', 120)
            tempo_stability = event_data.get('tempo_stability', 0.5)
            hierarchical_level = self._encode_hierarchical_level(event_data.get('hierarchical_level', 'phrase'))
            structural_importance = event_data.get('structural_importance', 0.5)
            rhythmic_density = event_data.get('rhythmic_density', 0.5)
            rhythmic_syncopation = event_data.get('rhythmic_syncopation', 0.0)
            stream_id = event_data.get('stream_id', 0)
            stream_confidence = event_data.get('stream_confidence', 0.5)
            
            # Normalize features to [0, 1] range
            features = np.array([
                midi_note / 127.0,  # Normalized MIDI note [0, 1]
                (rms_db + 60) / 60.0,  # Normalized RMS [-60, 0] -> [0, 1]
                f0 / 1000.0,  # Normalized frequency [0, 1000] -> [0, 1]
                centroid / 5000.0,  # Normalized spectral centroid [0, 5000] -> [0, 1]
                min(ioi, 2.0) / 2.0,  # Normalized IOI [0, 2] -> [0, 1]
                float(onset),  # Onset flag [0, 1]
                # Musical features
                chord_tension,  # Chord tension [0, 1]
                key_stability,  # Key stability [0, 1]
                tempo / 200.0,  # Normalized tempo [0, 200] -> [0, 1]
                tempo_stability,  # Tempo stability [0, 1]
                hierarchical_level,  # Hierarchical level encoding [0, 1]
                structural_importance,  # Structural importance [0, 1]
                rhythmic_density,  # Rhythmic density [0, 1]
                rhythmic_syncopation,  # Rhythmic syncopation [0, 1]
                stream_id / 10.0,  # Normalized stream ID [0, 10] -> [0, 1]
                stream_confidence  # Stream confidence [0, 1]
            ])
            
            # Ensure we have the right number of dimensions
            if len(features) < self.feature_dimensions:
                # Pad with zeros
                padded_features = np.zeros(self.feature_dimensions)
                padded_features[:len(features)] = features
                features = padded_features
            elif len(features) > self.feature_dimensions:
                # Truncate to required dimensions
                features = features[:self.feature_dimensions]
            
            return features
            
        except Exception as e:
            print(f"Error extracting audio features: {e}")
            # Return default feature vector
            return np.zeros(self.feature_dimensions)
    
    def _encode_hierarchical_level(self, level: str) -> float:
        """Encode hierarchical level as a float value"""
        level_encoding = {
            'measure': 0.2,
            'phrase': 0.4,
            'section': 0.6,
            'movement': 0.8,
            'piece': 1.0
        }
        return level_encoding.get(level, 0.4)
    
    def find_harmonic_patterns(self, min_length: int = 2, min_frequency: int = 2) -> List[tuple]:
        """
        Find harmonic patterns based on chord relationships and key stability
        
        Args:
            min_length: Minimum pattern length
            min_frequency: Minimum frequency of occurrence
            
        Returns:
            List of (pattern, frequency) tuples
        """
        try:
            harmonic_patterns = []
            
            # Extract chord sequences from musical features
            chord_sequence = []
            for frame_id, frame in self.audio_frames.items():
                if hasattr(frame, 'metadata') and 'chord' in frame.metadata:
                    chord_sequence.append(frame.metadata['chord'])
                elif hasattr(frame, 'chord_tension'):
                    # Infer chord from tension if available
                    chord_sequence.append(f"chord_{frame.chord_tension:.2f}")
            
            if len(chord_sequence) < min_length:
                return []
            
            # Find repeated chord progressions using efficient sliding window
            pattern_counts = {}
            
            # Limit pattern length for efficiency with large datasets
            max_pattern_length = min(6, len(chord_sequence) // 10) if len(chord_sequence) > 1000 else 8
            
            for length in range(min_length, max_pattern_length + 1):
                # Use sliding window with hash-based counting for efficiency
                for start in range(len(chord_sequence) - length + 1):
                    pattern = tuple(chord_sequence[start:start + length])
                    
                    if pattern in pattern_counts:
                        pattern_counts[pattern] += 1
                    else:
                        pattern_counts[pattern] = 1
            
            # Filter patterns by frequency
            for pattern, count in pattern_counts.items():
                if count >= min_frequency:
                    harmonic_patterns.append((list(pattern), count))
            
            # Remove duplicates and sort by frequency
            unique_patterns = {}
            for pattern, freq in harmonic_patterns:
                pattern_key = tuple(pattern)
                if pattern_key not in unique_patterns or unique_patterns[pattern_key] < freq:
                    unique_patterns[pattern_key] = freq
            
            result = [(list(pattern), freq) for pattern, freq in unique_patterns.items()]
            result.sort(key=lambda x: x[1], reverse=True)
            
            self.stats['harmonic_patterns'] = len(result)
            return result
            
        except Exception as e:
            print(f"Error finding harmonic patterns: {e}")
            return []
    
    def find_polyphonic_patterns(self, min_length: int = 2, min_frequency: int = 2) -> List[tuple]:
        """
        Find polyphonic patterns based on multiple simultaneous voices/streams
        
        Args:
            min_length: Minimum pattern length
            min_frequency: Minimum frequency of occurrence
            
        Returns:
            List of (pattern, frequency) tuples
        """
        try:
            polyphonic_patterns = []
            
            # Extract stream information from musical features
            stream_sequences = {}
            
            for frame_id, frame in self.audio_frames.items():
                if hasattr(frame, 'metadata'):
                    stream_id = frame.metadata.get('stream_id', 0)
                    if stream_id not in stream_sequences:
                        stream_sequences[stream_id] = []
                    
                    # Create a polyphonic event descriptor
                    event_desc = {
                        'stream_id': stream_id,
                        'chord_tension': frame.metadata.get('chord_tension', 0.5),
                        'rhythmic_density': frame.metadata.get('rhythmic_density', 0.5),
                        'hierarchical_level': frame.metadata.get('hierarchical_level', 'phrase')
                    }
                    stream_sequences[stream_id].append(event_desc)
            
            
            # Find patterns across multiple streams
            if len(stream_sequences) < 2:
                return []
            
            # Create synchronized polyphonic events
            max_length = min(len(seq) for seq in stream_sequences.values())
            
            polyphonic_events = []
            
            for i in range(max_length):
                event = {}
                for stream_id, sequence in stream_sequences.items():
                    if i < len(sequence):
                        event[f'stream_{stream_id}'] = sequence[i]
                polyphonic_events.append(event)
            
            
            # Find repeated polyphonic patterns using efficient hash-based counting
            pattern_counts = {}
            
            # Limit pattern length for efficiency with large datasets
            max_pattern_length = min(4, len(polyphonic_events) // 20) if len(polyphonic_events) > 1000 else 6
            
            for length in range(min_length, max_pattern_length + 1):
                # Use sliding window with hash-based counting for efficiency
                for start in range(len(polyphonic_events) - length + 1):
                    pattern = tuple(str(event) for event in polyphonic_events[start:start + length])
                    
                    if pattern in pattern_counts:
                        pattern_counts[pattern] += 1
                    else:
                        pattern_counts[pattern] = 1
            
            # Filter patterns by frequency with flexible requirements
            for pattern_tuple, count in pattern_counts.items():
                pattern_length = len(pattern_tuple)
                min_freq_for_length = max(1, min_frequency - pattern_length + 1)
                if count >= min_freq_for_length:
                    # Convert tuple pattern back to original format
                    pattern_list = []
                    for i in range(pattern_length):
                        if i < len(polyphonic_events):
                            pattern_list.append(polyphonic_events[i])
                    polyphonic_patterns.append((pattern_list, count))
            
            # Remove duplicates and sort by frequency
            unique_patterns = {}
            for pattern, freq in polyphonic_patterns:
                pattern_key = str(pattern)  # Convert to string for hashing
                if pattern_key not in unique_patterns or unique_patterns[pattern_key] < freq:
                    unique_patterns[pattern_key] = freq
            
            result = [(pattern, freq) for pattern, freq in unique_patterns.items()]
            result.sort(key=lambda x: x[1], reverse=True)
            
            self.stats['polyphonic_patterns'] = len(result)
            return result
            
        except Exception as e:
            print(f"Error finding polyphonic patterns: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def extract_spectral_features(self, event_data: Dict) -> np.ndarray:
        """
        Extract spectral features for advanced audio analysis
        
        Args:
            event_data: Audio event data dictionary
            
        Returns:
            np.ndarray: Spectral feature vector
        """
        try:
            # Spectral features
            centroid = event_data.get('centroid', 2000)
            rolloff = event_data.get('rolloff', 4000)
            bandwidth = event_data.get('bandwidth', 1000)
            contrast = event_data.get('contrast', 0.5)
            flatness = event_data.get('flatness', 0.1)
            mfcc_1 = event_data.get('mfcc_1', 0)
            
            # Normalize spectral features
            features = np.array([
                centroid / 8000.0,  # Spectral centroid [0, 8000] -> [0, 1]
                rolloff / 8000.0,  # Spectral rolloff [0, 8000] -> [0, 1]
                bandwidth / 4000.0,  # Spectral bandwidth [0, 4000] -> [0, 1]
                contrast,  # Spectral contrast [0, 1]
                flatness,  # Spectral flatness [0, 1]
                (mfcc_1 + 50) / 100.0  # First MFCC coefficient [-50, 50] -> [0, 1]
            ])
            
            return features
            
        except Exception as e:
            print(f"Error extracting spectral features: {e}")
            return np.zeros(6)
    
    def extract_temporal_features(self, event_data: Dict) -> np.ndarray:
        """
        Extract temporal features for rhythm analysis
        
        Args:
            event_data: Audio event data dictionary
            
        Returns:
            np.ndarray: Temporal feature vector
        """
        try:
            # Temporal features
            ioi = event_data.get('ioi', 0.5)
            duration = event_data.get('duration', 1.0)
            attack_time = event_data.get('attack_time', 0.1)
            release_time = event_data.get('release_time', 0.3)
            tempo = event_data.get('tempo', 120)
            beat_position = event_data.get('beat_position', 0.0)
            
            # Normalize temporal features
            features = np.array([
                min(ioi, 2.0) / 2.0,  # IOI [0, 2] -> [0, 1]
                min(duration, 4.0) / 4.0,  # Duration [0, 4] -> [0, 1]
                min(attack_time, 1.0),  # Attack time [0, 1] -> [0, 1]
                min(release_time, 1.0),  # Release time [0, 1] -> [0, 1]
                (tempo - 60) / 120.0,  # Tempo [60, 180] -> [0, 1]
                beat_position  # Beat position [0, 1] -> [0, 1]
            ])
            
            return features
            
        except Exception as e:
            print(f"Error extracting temporal features: {e}")
            return np.zeros(6)
    
    def add_audio_frame(self, features: np.ndarray, audio_data: Dict) -> bool:
        """
        Add an audio frame to the AudioOracle
        
        Args:
            features: Feature vector representing the audio frame
            audio_data: Original audio event data
            
        Returns:
            bool: Success status
        """
        try:
            # Create new frame
            frame_id = self.frame_counter
            audio_frame = AudioFrame(
                timestamp=time.time(),
                features=features,
                audio_data=audio_data,
                frame_id=frame_id
            )
            
            # Store musical metadata for harmonic/polyphonic analysis
            audio_frame.metadata = {
                'chord_tension': audio_data.get('chord_tension', 0.5),
                'key_stability': audio_data.get('key_stability', 0.5),
                'tempo': audio_data.get('tempo', 120),
                'tempo_stability': audio_data.get('tempo_stability', 0.5),
                'hierarchical_level': audio_data.get('hierarchical_level', 'phrase'),
                'structural_importance': audio_data.get('structural_importance', 0.5),
                'rhythmic_density': audio_data.get('rhythmic_density', 0.5),
                'rhythmic_syncopation': audio_data.get('rhythmic_syncopation', 0.0),
                'stream_id': audio_data.get('stream_id', 0),
                'stream_confidence': audio_data.get('stream_confidence', 0.5),
                'chord': audio_data.get('chord', 'C'),  # Extract chord from transformer insights
                'key_signature': audio_data.get('key_signature', 'C major'),
                # DUAL VOCABULARY: Store both harmonic and percussive tokens
                'harmonic_token': audio_data.get('harmonic_token'),
                'percussive_token': audio_data.get('percussive_token'),
                'gesture_token': audio_data.get('gesture_token')  # Legacy compatibility
            }
            
            # Store frame
            self.audio_frames[frame_id] = audio_frame
            
            # Add to sequence (using frame_id as symbol)
            self.sequence.append(frame_id)
            self.sequence_length += 1
            
            # Create new state
            new_state = self.size
            self.states[new_state] = {
                'len': self.states[self.last]['len'] + 1,
                'link': -1,
                'next': {}
            }
            
            # Add forward transition
            self.states[self.last]['next'][frame_id] = new_state
            self.transitions[(self.last, frame_id)] = new_state
            
            # Update suffix links with approximate matching
            k = self.states[self.last]['link']
            while k != -1 and k in self.states:
                # Find similar frames in the sequence
                similar_frames = self._find_similar_frames(features)
                
                for similar_frame_id in similar_frames:
                    if (k, similar_frame_id) not in self.transitions:
                        self.transitions[(k, similar_frame_id)] = new_state
                        self.states[k]['next'][similar_frame_id] = new_state
                
                # Safely get next link
                if k in self.states and 'link' in self.states[k]:
                    k = self.states[k]['link']
                else:
                    k = -1
            
            # Set suffix link
            if k == -1:
                self.states[new_state]['link'] = 0
                self.suffix_links[new_state] = 0
            else:
                # Find best transition (minimum distance)
                best_state = None
                min_distance = float('inf')
                
                for similar_frame_id in self._find_similar_frames(features):
                    if (k, similar_frame_id) in self.transitions:
                        distance = self._calculate_distance(features, self.audio_frames[similar_frame_id].features)
                        if distance < min_distance:
                            min_distance = distance
                            best_state = self.transitions[(k, similar_frame_id)]
                
                if best_state is not None:
                    self.states[new_state]['link'] = best_state
                    self.suffix_links[new_state] = best_state
                else:
                    self.states[new_state]['link'] = 0
                    self.suffix_links[new_state] = 0
            
            # Update state
            self.last = new_state
            self.size += 1
            self.frame_counter += 1
            
            # Update statistics
            self.is_trained = True
            self.last_update = time.time()
            self.stats['total_states'] = self.size
            self.stats['sequence_length'] = self.sequence_length
            self.stats['is_trained'] = True
            self.stats['last_update'] = self.last_update
            self.stats['memory_usage'] = len(self.audio_frames)
            
            return True
            
        except Exception as e:
            print(f"Error adding audio frame: {e}")
            return False
    
    def _symbol_to_features(self, symbol: str) -> np.ndarray:
        """Convert a musical symbol to feature vector (fallback method)"""
        try:
            # Parse symbol like "N57_R-20_F220"
            parts = symbol.split('_')
            features = np.zeros(self.feature_dimensions)
            
            for part in parts:
                if part.startswith('N'):
                    # MIDI note
                    note = int(part[1:])
                    features[0] = note / 127.0
                elif part.startswith('R'):
                    # RMS
                    rms = int(part[1:])
                    features[1] = (rms + 60) / 60.0
                elif part.startswith('F'):
                    # Frequency
                    freq = int(part[1:])
                    features[2] = freq / 1000.0
            
            return features
            
        except Exception as e:
            print(f"Error converting symbol to features: {e}")
            return np.zeros(self.feature_dimensions)
    
    def add_sequence(self, musical_sequence: List[Any]) -> bool:
        """
        Add a sequence of musical events (compatibility with FactorOracle interface)
        
        Args:
            musical_sequence: List of musical symbols or event data
            
        Returns:
            bool: Success status
        """
        try:
            if not musical_sequence:
                return False
            
            for item in musical_sequence:
                if isinstance(item, dict):
                    # Extract features from event data
                    features = self.extract_audio_features(item)
                    self.add_audio_frame(features, item)
                else:
                    # Convert symbol to features (fallback)
                    features = self._symbol_to_features(str(item))
                    audio_data = {'symbol': str(item), 'timestamp': time.time()}
                    self.add_audio_frame(features, audio_data)
            
            return True
            
        except Exception as e:
            print(f"Error adding sequence to AudioOracle: {e}")
            return False
    
    def find_patterns(self, query_sequence: List[int] = None, min_freq: int = 2, min_len: int = 2) -> List[Tuple[List[int], int]]:
        """
        Find patterns in the audio sequence using distance-based similarity
        
        This method groups similar audio frames by feature similarity rather than
        exact frame ID matching, which is more appropriate for AudioOracle.
        """
        patterns = []
        
        try:
            # Group similar frames by feature similarity
            frame_groups = self._group_similar_frames()
            
            # Find patterns by looking for repeated sequences of similar frame groups
            for length in range(min_len, min(self.sequence_length, 10)):
                for start in range(self.sequence_length - length + 1):
                    pattern = self.sequence[start:start + length]
                    if len(pattern) == length:
                        # Count approximate occurrences using frame groups
                        count = self._count_approximate_pattern(pattern, frame_groups)
                        
                        if count >= min_freq:
                            patterns.append((pattern, count))
            
            # Remove duplicates and sort by frequency
            unique_patterns = {}
            for pattern, freq in patterns:
                pattern_key = tuple(pattern)
                if pattern_key not in unique_patterns or unique_patterns[pattern_key] < freq:
                    unique_patterns[pattern_key] = freq
            
            result = [(list(pattern), freq) for pattern, freq in unique_patterns.items()]
            result.sort(key=lambda x: x[1], reverse=True)
            
            self.stats['total_patterns'] = len(result)
            return result
            
        except Exception as e:
            print(f"Error in find_patterns: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _group_similar_frames(self) -> Dict[int, List[int]]:
        """Group frames by feature similarity"""
        frame_groups = {}
        group_id = 0
        
        for frame_id in self.sequence:
            if frame_id not in frame_groups:
                # Find all frames similar to this one
                similar_frames = [frame_id]
                current_features = self.audio_frames[frame_id].features
                
                for other_frame_id in self.sequence:
                    if other_frame_id != frame_id and other_frame_id not in frame_groups:
                        other_features = self.audio_frames[other_frame_id].features
                        distance = self._calculate_distance(current_features, other_features)
                        
                        if distance < self.distance_threshold:
                            similar_frames.append(other_frame_id)
                
                # Assign group ID to all similar frames
                for similar_frame in similar_frames:
                    frame_groups[similar_frame] = group_id
                
                group_id += 1
        
        return frame_groups
    
    def _count_approximate_pattern(self, pattern: List[int], frame_groups: Dict[int, int]) -> int:
        """Count approximate occurrences of a pattern using frame groups"""
        if not pattern:
            return 0
        
        count = 0
        pattern_groups = [frame_groups.get(frame_id, -1) for frame_id in pattern]
        
        for start in range(self.sequence_length - len(pattern) + 1):
            sequence_groups = []
            for i in range(len(pattern)):
                frame_id = self.sequence[start + i]
                group_id = frame_groups.get(frame_id, -1)
                sequence_groups.append(group_id)
            
            # Debug: Check types before comparison
            if len(sequence_groups) != len(pattern_groups):
                continue
                
            if sequence_groups == pattern_groups:
                count += 1
        
        return count
    
    def generate_next(self, current_context: List[int], max_length: int = 10) -> List[int]:
        """Generate next audio frames based on context"""
        if not current_context:
            return []
        
        # Find similar patterns and generate continuations
        generated = []
        current_state = 0
        
        # Navigate to the best matching state
        for frame_id in current_context:
            if (current_state, frame_id) in self.transitions:
                current_state = self.transitions[(current_state, frame_id)]
            else:
                # Find similar frame
                similar_frames = self._find_similar_frames(self.audio_frames[frame_id].features)
                for similar_frame_id in similar_frames:
                    if (current_state, similar_frame_id) in self.transitions:
                        current_state = self.transitions[(current_state, similar_frame_id)]
                        break
        
        # Generate continuation
        for _ in range(max_length):
            if current_state in self.states and self.states[current_state]['next']:
                # Choose next frame (simplified - could be probabilistic)
                next_frames = list(self.states[current_state]['next'].keys())
                if next_frames:
                    next_frame = next_frames[0]  # Simple choice
                    generated.append(next_frame)
                    current_state = self.states[current_state]['next'][next_frame]
                else:
                    break
            else:
                break
        
        return generated
    
    def find_similar_moments(self, query_features: np.ndarray, n_results: int = 5) -> List[Dict[str, Any]]:
        """Find similar audio moments (compatibility interface)"""
        similar_frames = self._find_similar_frames(query_features)
        
        results = []
        for frame_id in similar_frames[:n_results]:
            frame = self.audio_frames[frame_id]
            distance = self._calculate_distance(query_features, frame.features)
            
            results.append({
                'frame_id': frame_id,
                'similarity': 1.0 - distance,  # Convert distance to similarity
                'timestamp': frame.timestamp,
                'audio_data': frame.audio_data,
                'features': frame.features.tolist()
            })
        
        return results
    
    def retrain_if_needed(self, memory_buffer, retrain_threshold: int = 100) -> bool:
        """
        AudioOracle doesn't need retraining since it learns incrementally.
        This method exists for compatibility with the main system.
        """
        # AudioOracle learns incrementally, so no retraining needed
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get AudioOracle statistics"""
        return self.stats.copy()
    
    def save_to_file(self, filepath: str) -> bool:
        """Save AudioOracle to file"""
        try:
            print(f"💾 Saving AudioOracle: {len(self.audio_frames)} audio_frames, {len(self.states)} states")
            
            save_data = {
                'states': self.states,
                'transitions': {f"{k[0]}_{k[1]}": v for k, v in self.transitions.items()},
                'suffix_links': self.suffix_links,
                'sequence': self.sequence,
                'sequence_length': self.sequence_length,
                'audio_frames': {
                    str(frame_id): {
                        'timestamp': frame.timestamp,
                        'features': frame.features.tolist(),
                        'audio_data': frame.audio_data,
                        'frame_id': frame.frame_id
                    }
                    for frame_id, frame in self.audio_frames.items()
                },
                'frame_counter': self.frame_counter,
                'distance_threshold': self.distance_threshold,
                'distance_function': self.distance_function,
                'pattern_counter': self.pattern_counter,
                'max_pattern_length': self.max_pattern_length,
                'is_trained': self.is_trained,
                'stats': self.stats
            }
            
            with open(filepath, 'w') as f:
                json.dump(save_data, f, indent=2, default=str)
            
            return True
            
        except Exception as e:
            print(f"Error saving AudioOracle: {e}")
            return False
    
    def load_from_file(self, filepath: str) -> bool:
        """Load AudioOracle from file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.states = data['states']
            self.transitions = {tuple(k.split('_')): v for k, v in data['transitions'].items()}
            self.suffix_links = data['suffix_links']
            self.sequence = data['sequence']
            self.sequence_length = data['sequence_length']
            
            # Reconstruct audio frames
            self.audio_frames = {}
            for frame_id_str, frame_data in data['audio_frames'].items():
                frame_id = int(frame_id_str)
                self.audio_frames[frame_id] = AudioFrame(
                    timestamp=frame_data['timestamp'],
                    features=np.array(frame_data['features']),
                    audio_data=frame_data['audio_data'],
                    frame_id=frame_data['frame_id']
                )
            
            self.frame_counter = data['frame_counter']
            self.distance_threshold = data['distance_threshold']
            self.distance_function = data['distance_function']
            self.pattern_counter = data.get('pattern_counter', 0)
            self.max_pattern_length = data['max_pattern_length']
            self.is_trained = data['is_trained']
            self.stats = data['stats']
            
            # Reconstruct Factor Oracle compatibility attributes
            self.size = data.get('size', len(self.states))
            self.last = data.get('last', max(self.states.keys()) if self.states else 0)
            
            return True
            
        except Exception as e:
            print(f"Error loading AudioOracle: {e}")
            return False


def extract_audio_features(event_data: Dict) -> np.ndarray:
    """
    Extract audio features from event data for AudioOracle
    
    Args:
        event_data: Audio event data dictionary
        
    Returns:
        np.ndarray: Feature vector
    """
    # Extract basic features
    features = [
        event_data.get('midi', 60) / 127.0,  # Normalized MIDI note
        event_data.get('rms_db', -20) / 20.0,  # Normalized RMS
        event_data.get('f0', 440) / 1000.0,  # Normalized frequency
        event_data.get('centroid', 2000) / 5000.0,  # Normalized spectral centroid
        event_data.get('ioi', 0.5),  # Inter-onset interval
        event_data.get('onset', 0)  # Onset flag
    ]
    
    return np.array(features)


def main():
    """Test AudioOracle implementation"""
    print("🎵 Testing Enhanced AudioOracle")
    print("=" * 50)
    
    # Create AudioOracle
    ao = AudioOracle(distance_threshold=0.2, distance_function='euclidean')
    
    # Test with mock audio data
    mock_audio_data = [
        ({'midi': 60, 'rms_db': -20, 'f0': 440, 'centroid': 2000, 'ioi': 0.5, 'onset': 1}, {'note': 'C'}),
        ({'midi': 64, 'rms_db': -18, 'f0': 523, 'centroid': 2200, 'ioi': 0.3, 'onset': 1}, {'note': 'E'}),
        ({'midi': 67, 'rms_db': -16, 'f0': 659, 'centroid': 2400, 'ioi': 0.4, 'onset': 1}, {'note': 'G'}),
        ({'midi': 60, 'rms_db': -20, 'f0': 440, 'centroid': 2000, 'ioi': 0.5, 'onset': 1}, {'note': 'C'}),
        ({'midi': 64, 'rms_db': -18, 'f0': 523, 'centroid': 2200, 'ioi': 0.3, 'onset': 1}, {'note': 'E'}),
    ]
    
    # Add frames
    print("📝 Adding audio frames...")
    for event_data, audio_data in mock_audio_data:
        features = extract_audio_features(event_data)
        success = ao.add_audio_frame(features, audio_data)
        print(f"  Frame {ao.frame_counter}: {audio_data['note']} - {'✅' if success else '❌'}")
    
    # Test pattern finding
    print("\n🔍 Finding patterns...")
    patterns = ao.find_patterns(min_freq=2, min_len=2)
    print(f"Found {len(patterns)} patterns:")
    for i, (pattern, freq) in enumerate(patterns[:5]):
        pattern_notes = [ao.audio_frames[frame_id].audio_data['note'] for frame_id in pattern]
        print(f"  {i+1}. {' '.join(pattern_notes)} (freq: {freq})")
    
    # Test similarity search
    print("\n🎯 Testing similarity search...")
    query_features = extract_audio_features({'midi': 60, 'rms_db': -20, 'f0': 440, 'centroid': 2000, 'ioi': 0.5, 'onset': 1})
    similar_moments = ao.find_similar_moments(query_features, n_results=3)
    print(f"Found {len(similar_moments)} similar moments:")
    for moment in similar_moments:
        print(f"  Similarity: {moment['similarity']:.3f}, Note: {moment['audio_data']['note']}")
    
    # Test generation
    print("\n🎼 Testing generation...")
    context = [0, 1]  # First two frames
    generated = ao.generate_next(context, max_length=3)
    print(f"Generated continuation: {generated}")
    
    # Statistics
    print("\n📊 AudioOracle Statistics:")
    stats = ao.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n✅ AudioOracle test complete!")


if __name__ == "__main__":
    main()
