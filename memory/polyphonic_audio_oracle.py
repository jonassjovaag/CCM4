# memory/polyphonic_audio_oracle.py
# Enhanced AudioOracle with Polyphonic Support
# Handles multi-pitch features, chord recognition, and enhanced musical analysis

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

# Import base AudioOracle
from .audio_oracle import AudioOracle, AudioFrame


@dataclass
class PolyphonicAudioFrame(AudioFrame):
    """
    Enhanced AudioFrame with polyphonic information
    """
    # Multi-pitch information
    polyphonic_pitches: List[float] = None
    polyphonic_midi: List[int] = None
    polyphonic_cents: List[float] = None
    chord_quality: str = "single"
    root_note: int = 0
    num_pitches: int = 1
    
    # Enhanced features
    mfcc_2: float = 0.0
    mfcc_3: float = 0.0
    zcr: float = 0.0
    rolloff_85: float = 0.0
    rolloff_95: float = 0.0
    
    def __post_init__(self):
        if self.polyphonic_pitches is None:
            self.polyphonic_pitches = [self.features[1] if len(self.features) > 1 else 440.0]  # f0
        if self.polyphonic_midi is None:
            self.polyphonic_midi = [int(self.features[1]) if len(self.features) > 1 else 69]  # midi
        if self.polyphonic_cents is None:
            self.polyphonic_cents = [0.0]


class PolyphonicAudioOracle(AudioOracle):
    """
    Enhanced AudioOracle with polyphonic support
    
    Features:
    - Multi-pitch pattern recognition
    - Chord progression analysis
    - Enhanced feature vectors with polyphonic information
    - Chord similarity matching
    - Harmonic relationship preservation
    """
    
    def __init__(self, 
                 distance_threshold: float = 0.15, 
                 distance_function: str = 'euclidean',
                 max_pattern_length: int = 50,
                 feature_dimensions: int = 12,  # Increased for polyphonic features
                 adaptive_threshold: bool = True,
                 chord_similarity_weight: float = 0.3):
        """
        Initialize Polyphonic AudioOracle
        
        Args:
            distance_threshold: Threshold θ for similarity matching
            distance_function: Distance function ('euclidean', 'cosine', 'manhattan')
            max_pattern_length: Maximum length of patterns to recognize
            feature_dimensions: Number of dimensions in feature vectors
            adaptive_threshold: Whether to adapt threshold based on data distribution
            chord_similarity_weight: Weight for chord similarity in distance calculation
        """
        super().__init__(
            distance_threshold=distance_threshold,
            distance_function=distance_function,
            max_pattern_length=max_pattern_length,
            feature_dimensions=feature_dimensions,
            adaptive_threshold=adaptive_threshold
        )
        
        # Polyphonic-specific attributes
        self.chord_similarity_weight = chord_similarity_weight
        self.chord_progressions = defaultdict(int)  # Track chord progressions
        self.harmonic_patterns = defaultdict(int)  # Track harmonic patterns
        
        # Enhanced statistics
        self.stats.update({
            'chord_progressions': 0,
            'harmonic_patterns': 0,
            'polyphonic_frames': 0,
            'chord_types': defaultdict(int),
            'average_pitches_per_frame': 0.0
        })
        
        print(f"🎵 Polyphonic AudioOracle initialized:")
        print(f"   Feature dimensions: {feature_dimensions}")
        print(f"   Chord similarity weight: {chord_similarity_weight}")
        print(f"   Distance function: {distance_function}")
    
    def _sanitize_audio_data(self, audio_data: Dict) -> Dict:
        """
        Sanitize audio data for JSON serialization
        Preserves ALL fields including gesture_token and other dual perception data
        
        Args:
            audio_data: Original audio event data dict
            
        Returns:
            Sanitized dict ready for JSON serialization
        """
        sanitized = {}
        for key, value in audio_data.items():
            try:
                # Convert numpy types to Python types
                if isinstance(value, (np.integer, np.int64, np.int32)):
                    sanitized[key] = int(value)
                elif isinstance(value, (np.floating, np.float64, np.float32)):
                    sanitized[key] = float(value)
                elif isinstance(value, np.ndarray):
                    sanitized[key] = value.tolist()
                elif isinstance(value, (list, tuple)):
                    # Recursively sanitize lists
                    sanitized[key] = [
                        item.tolist() if isinstance(item, np.ndarray) else
                        float(item) if isinstance(item, (np.floating, np.float64, np.float32)) else
                        int(item) if isinstance(item, (np.integer, np.int64, np.int32)) else
                        item
                        for item in value
                    ]
                elif isinstance(value, dict):
                    # Recursively sanitize dicts
                    sanitized[key] = self._sanitize_audio_data(value)
                else:
                    # Keep as is (strings, ints, floats, bools, None)
                    sanitized[key] = value
            except Exception as e:
                # If conversion fails, skip this field
                print(f"⚠️ Warning: Could not sanitize field '{key}': {e}")
                continue
        
        return sanitized
    
    def extract_polyphonic_features(self, event_data: Dict) -> np.ndarray:
        """
        Extract enhanced feature vector with polyphonic information
        
        Args:
            event_data: Event data dictionary
            
        Returns:
            Enhanced feature vector
        """
        try:
            # Basic features
            features = [
                float(event_data.get('rms_db', -20.0)),
                float(event_data.get('f0', 440.0)),
                float(event_data.get('centroid', 2000.0)),
                float(event_data.get('rolloff', 3000.0)),
                float(event_data.get('bandwidth', 1000.0)),
                float(event_data.get('contrast', 0.5)),
                float(event_data.get('flatness', 0.1)),
                float(event_data.get('mfcc_1', 0.0)),
                float(event_data.get('mfcc_2', 0.0)),
                float(event_data.get('mfcc_3', 0.0)),
                float(event_data.get('zcr', 0.0)),
                float(event_data.get('attack_time', 0.1))
            ]
            
            # Add polyphonic information
            polyphonic_pitches = event_data.get('polyphonic_pitches', [event_data.get('f0', 440.0)])
            polyphonic_midi = event_data.get('polyphonic_midi', [event_data.get('midi', 69)])
            chord_quality = event_data.get('chord_quality', 'single')
            root_note = event_data.get('root_note', event_data.get('midi', 69))
            
            # Add chord information to features
            chord_features = self._encode_chord_features(chord_quality, root_note, len(polyphonic_pitches))
            features.extend(chord_features)
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            print(f"Error extracting polyphonic features: {e}")
            # Return default features
            return np.array([-20.0, 440.0, 2000.0, 3000.0, 1000.0, 0.5, 0.1, 0.0, 0.0, 0.0, 0.0, 0.1], dtype=np.float32)
    
    def _encode_chord_features(self, chord_quality: str, root_note: int, num_pitches: int) -> List[float]:
        """
        Encode chord information as numerical features
        
        Args:
            chord_quality: Chord quality string
            root_note: Root note MIDI number
            num_pitches: Number of pitches in chord
            
        Returns:
            List of chord feature values
        """
        # Chord quality encoding
        chord_encoding = {
            'single': 0.0,
            'major': 1.0,
            'minor': 2.0,
            'diminished': 3.0,
            'augmented': 4.0,
            'complex': 5.0,
            'unknown': 6.0
        }
        
        chord_value = chord_encoding.get(chord_quality, 6.0)
        
        # Root note (normalized to 0-1)
        root_normalized = (root_note % 12) / 12.0
        
        # Number of pitches (normalized)
        num_pitches_normalized = min(num_pitches / 4.0, 1.0)  # Cap at 4 pitches
        
        return [chord_value, root_normalized, num_pitches_normalized]
    
    def calculate_chord_similarity(self, frame1: PolyphonicAudioFrame, frame2: PolyphonicAudioFrame) -> float:
        """
        Calculate chord similarity between two frames
        
        Args:
            frame1: First polyphonic frame
            frame2: Second polyphonic frame
            
        Returns:
            Chord similarity score (0-1, where 1 is identical)
        """
        try:
            # Same chord quality
            if frame1.chord_quality == frame2.chord_quality:
                quality_similarity = 1.0
            else:
                # Different chord qualities
                chord_types = ['single', 'major', 'minor', 'diminished', 'augmented', 'complex', 'unknown']
                idx1 = chord_types.index(frame1.chord_quality) if frame1.chord_quality in chord_types else 6
                idx2 = chord_types.index(frame2.chord_quality) if frame2.chord_quality in chord_types else 6
                quality_similarity = 1.0 - abs(idx1 - idx2) / 6.0
            
            # Root note similarity (considering enharmonic equivalents)
            root_diff = abs(frame1.root_note - frame2.root_note) % 12
            root_similarity = 1.0 - (root_diff / 12.0)
            
            # Number of pitches similarity
            num_pitches_similarity = 1.0 - abs(frame1.num_pitches - frame2.num_pitches) / 4.0
            
            # Weighted combination
            chord_similarity = (
                0.5 * quality_similarity +
                0.3 * root_similarity +
                0.2 * num_pitches_similarity
            )
            
            return chord_similarity
            
        except Exception as e:
            print(f"Error calculating chord similarity: {e}")
            return 0.0
    
    def _calculate_distance(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """
        Enhanced distance calculation with chord similarity
        
        Args:
            features1: First feature vector
            features2: Second feature vector
            
        Returns:
            Enhanced distance score
        """
        try:
            # Calculate basic distance
            basic_distance = super()._calculate_distance(features1, features2)
            
            # If we have polyphonic information, enhance with chord similarity
            if len(features1) >= 15 and len(features2) >= 15:  # Check if we have chord features
                # Extract chord features
                chord_features1 = features1[-3:]  # Last 3 features are chord-related
                chord_features2 = features2[-3:]
                
                # Calculate chord distance
                chord_distance = np.linalg.norm(chord_features1 - chord_features2)
                
                # Combine basic and chord distances
                enhanced_distance = (
                    (1.0 - self.chord_similarity_weight) * basic_distance +
                    self.chord_similarity_weight * chord_distance
                )
                
                return enhanced_distance
            else:
                return basic_distance
                
        except Exception as e:
            print(f"Error in enhanced distance calculation: {e}")
            return super()._calculate_distance(features1, features2)
    
    def add_polyphonic_sequence(self, musical_sequence: List[Dict]) -> bool:
        """
        Add polyphonic musical sequence to AudioOracle
        
        Args:
            musical_sequence: List of event data dictionaries with polyphonic information
            
        Returns:
            True if successful
        """
        try:
            print(f"🎵 Adding polyphonic sequence with {len(musical_sequence)} events...")
            
            polyphonic_frames_added = 0
            
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
                
                # Add frame to AudioOracle
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
                    print(f"\r🎵 Polyphonic Learning: {progress:.1f}% ({i + 1}/{len(musical_sequence)})", end='', flush=True)
            
            print(f"\n✅ Polyphonic sequence added: {polyphonic_frames_added} frames")
            
            # Update statistics
            self.stats['polyphonic_frames'] = polyphonic_frames_added
            self.stats['chord_progressions'] = len(self.chord_progressions)
            self.stats['harmonic_patterns'] = len(self.harmonic_patterns)
            
            # Update chord type statistics
            for event_data in musical_sequence:
                chord_type = event_data.get('chord_quality', 'single')
                self.stats['chord_types'][chord_type] += 1
            
            # Calculate average pitches per frame
            total_pitches = sum(event_data.get('num_pitches', 1) for event_data in musical_sequence)
            self.stats['average_pitches_per_frame'] = total_pitches / len(musical_sequence) if musical_sequence else 0.0
            
            return True
            
        except Exception as e:
            print(f"❌ Error adding polyphonic sequence: {e}")
            return False
    
    def find_chord_patterns(self, min_freq: int = 2, min_len: int = 2) -> List[Tuple[List[str], int]]:
        """
        Find chord progression patterns
        
        Args:
            min_freq: Minimum frequency for pattern recognition
            min_len: Minimum length for chord patterns
            
        Returns:
            List of chord patterns with frequencies
        """
        try:
            chord_patterns = []
            
            # Convert chord progressions to sequences
            progression_sequences = []
            for progression, freq in self.chord_progressions.items():
                if freq >= min_freq:
                    chords = progression.split('->')
                    if len(chords) >= min_len:
                        progression_sequences.append((chords, freq))
            
            # Find longer patterns by combining progressions
            for length in range(min_len, min(10, len(progression_sequences))):
                for start in range(len(progression_sequences) - length + 1):
                    pattern = []
                    for i in range(length):
                        pattern.extend(progression_sequences[start + i][0])
                    
                    # Count occurrences of this pattern
                    count = 0
                    for seq, freq in progression_sequences:
                        if len(seq) >= len(pattern):
                            for j in range(len(seq) - len(pattern) + 1):
                                if seq[j:j + len(pattern)] == pattern:
                                    count += freq
                    
                    if count >= min_freq:
                        chord_patterns.append((pattern, count))
            
            # Remove duplicates and sort by frequency
            unique_patterns = {}
            for pattern, freq in chord_patterns:
                pattern_key = tuple(pattern)
                if pattern_key not in unique_patterns or unique_patterns[pattern_key] < freq:
                    unique_patterns[pattern_key] = freq
            
            result = [(list(pattern), freq) for pattern, freq in unique_patterns.items()]
            result.sort(key=lambda x: x[1], reverse=True)
            
            return result
            
        except Exception as e:
            print(f"Error finding chord patterns: {e}")
            return []
    
    def get_polyphonic_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive polyphonic statistics
        
        Returns:
            Dictionary with polyphonic statistics
        """
        base_stats = self.get_statistics()
        
        polyphonic_stats = {
            'polyphonic_frames': self.stats.get('polyphonic_frames', 0),
            'chord_progressions': self.stats.get('chord_progressions', 0),
            'harmonic_patterns': self.stats.get('harmonic_patterns', 0),
            'average_pitches_per_frame': self.stats.get('average_pitches_per_frame', 0.0),
            'chord_types': dict(self.stats.get('chord_types', {})),
            'top_chord_progressions': dict(list(self.chord_progressions.items())[:10]),
            'top_harmonic_patterns': dict(list(self.harmonic_patterns.items())[:10])
        }
        
        # Combine with base statistics
        combined_stats = {**base_stats, **polyphonic_stats}
        
        return combined_stats
    
    def _sanitize_audio_data(self, audio_data: Dict) -> Dict:
        """Sanitize audio_data for JSON serialization (convert numpy bools, etc.)"""
        sanitized = {}
        for key, value in audio_data.items():
            if isinstance(value, (np.bool_, bool)):
                sanitized[key] = bool(value)
            elif isinstance(value, (np.integer, np.int64, np.int32)):
                sanitized[key] = int(value)
            elif isinstance(value, (np.floating, np.float64, np.float32)):
                sanitized[key] = float(value)
            elif isinstance(value, np.ndarray):
                sanitized[key] = value.tolist()
            elif isinstance(value, (list, tuple)):
                # Recursively sanitize lists
                sanitized[key] = [float(x) if isinstance(x, (np.floating, np.float64, np.float32)) else x for x in value]
            else:
                sanitized[key] = value
        return sanitized
    
    def _json_serialize_helper(self, obj):
        """
        JSON serialization helper for numpy types
        
        Args:
            obj: Object to serialize
            
        Returns:
            Serialized object or raises TypeError
        """
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        else:
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    def generate_with_request(self, 
                             current_context: List[int],
                             request: Optional[Dict] = None,
                             temperature: float = 1.0,
                             max_length: int = 10) -> List[int]:
        """
        Generate sequence with conditional bias based on request
        
        This enables goal-directed generation where you can ask for specific
        qualities (e.g., high consonance, specific gesture tokens, ascending motion).
        
        DUAL VOCABULARY SUPPORT:
        When request includes 'response_mode', filters states based on dual tokens:
        - 'harmonic': User plays drums → AI finds harmonic responses that co-occurred
        - 'percussive': User plays guitar → AI finds rhythmic responses that co-occurred
        - 'hybrid': Contextual filling based on both tokens
        
        Args:
            current_context: List of frame IDs for context
            request: Request specification dict with:
                - parameter: Parameter name (e.g., 'consonance', 'gesture_token', 'midi_relative')
                - type: Request type ('==', '>', '<', 'gradient', etc.)
                - value: Target value or gradient shape
                - weight: Blend weight (0.0-1.0), 1.0 = hard constraint
                - harmonic_token: (DUAL VOCAB) Harmonic token from current input
                - percussive_token: (DUAL VOCAB) Percussive token from current input
                - response_mode: (DUAL VOCAB) 'harmonic' | 'percussive' | 'hybrid'
            temperature: Sampling temperature (lower = more deterministic)
            max_length: Maximum sequence length to generate
            
        Returns:
            List of generated frame IDs
        """
        if not current_context:
            return []
        
        if not self.is_trained or len(self.audio_frames) == 0:
            print("⚠️  AudioOracle not trained yet")
            return []
        
        # Import request mask
        try:
            from memory.request_mask import RequestMask
            request_mask_generator = RequestMask()
        except ImportError:
            print("⚠️  Could not import RequestMask, falling back to standard generation")
            request = None
        
        generated = []
        current_state = 0
        
        # Navigate to matching state from context
        for frame_id in current_context:
            if (current_state, frame_id) in self.transitions:
                current_state = self.transitions[(current_state, frame_id)]
            else:
                # Find similar frame
                if frame_id < len(self.audio_frames):
                    similar_frames = self._find_similar_frames(self.audio_frames[frame_id].features)
                    for similar_frame_id in similar_frames:
                        if (current_state, similar_frame_id) in self.transitions:
                            current_state = self.transitions[(current_state, similar_frame_id)]
                            break
        
        # Generate continuation
        for _ in range(max_length):
            if current_state not in self.states or not self.states[current_state]['next']:
                break
            
            # Get possible next frames
            next_frames = list(self.states[current_state]['next'].keys())
            if not next_frames:
                break
            
            # DUAL VOCABULARY FILTERING: Apply response_mode filter if specified
            if request and 'response_mode' in request:
                response_mode = request['response_mode']
                input_harmonic_token = request.get('harmonic_token')
                input_percussive_token = request.get('percussive_token')
                
                filtered_frames = []
                
                for frame_id in next_frames:
                    if frame_id >= len(self.audio_frames):
                        continue
                    
                    # Get frame metadata (contains dual tokens)
                    frame_metadata = self.audio_frames[frame_id].metadata
                    frame_harmonic = frame_metadata.get('harmonic_token')
                    frame_percussive = frame_metadata.get('percussive_token')
                    
                    # Filter based on response_mode
                    if response_mode == 'harmonic':
                        # User plays drums (percussive input) → respond with harmony
                        # Match: input percussive token matches frame percussive token
                        # AND frame has harmonic content
                        if (input_percussive_token is not None and 
                            frame_percussive == input_percussive_token and 
                            frame_harmonic is not None):
                            filtered_frames.append(frame_id)
                    
                    elif response_mode == 'percussive':
                        # User plays guitar (harmonic input) → respond with rhythm
                        # Match: input harmonic token matches frame harmonic token
                        # AND frame has percussive content
                        if (input_harmonic_token is not None and 
                            frame_harmonic == input_harmonic_token and 
                            frame_percussive is not None):
                            filtered_frames.append(frame_id)
                    
                    elif response_mode == 'hybrid':
                        # Hybrid input → contextual filling
                        # Match either token
                        if ((input_harmonic_token is not None and frame_harmonic == input_harmonic_token) or
                            (input_percussive_token is not None and frame_percussive == input_percussive_token)):
                            filtered_frames.append(frame_id)
                
                # Use filtered frames if any found, otherwise fall back to all frames
                if filtered_frames:
                    next_frames = filtered_frames
                    # Debug logging (can be removed in production)
                    if len(filtered_frames) < len(list(self.states[current_state]['next'].keys())):
                        pass  # Filtering is working
                else:
                    # No matches found - fall back to unfiltered
                    pass
            
            # If no request, choose uniformly
            if request is None or 'response_mode' not in request:
                # Temperature-based sampling
                probabilities = np.ones(len(next_frames))
                probabilities = probabilities / np.sum(probabilities)
                
                if temperature != 1.0:
                    probabilities = np.power(probabilities, 1.0 / temperature)
                    probabilities = probabilities / np.sum(probabilities)
                
                next_idx = np.random.choice(len(next_frames), p=probabilities)
                next_frame = next_frames[next_idx]
            
            else:
                # Apply request mask
                parameter_name = request.get('parameter')
                
                # Extract parameter values for all candidate frames
                parameter_values = np.zeros(len(next_frames))
                for i, frame_id in enumerate(next_frames):
                    if frame_id < len(self.audio_frames):
                        audio_data = self.audio_frames[frame_id].audio_data
                        parameter_values[i] = audio_data.get(parameter_name, 0.0)
                
                # Create request mask
                try:
                    mask = request_mask_generator.create_mask(
                        None,  # corpus not needed for this use case
                        parameter_values,
                        request,
                        len(next_frames)
                    )
                    
                    # Safety check: ensure mask size matches next_frames
                    if len(mask) != len(next_frames):
                        print(f"⚠️  Mask size mismatch: {len(mask)} vs {len(next_frames)}, using uniform")
                        mask = np.ones(len(next_frames))
                    
                    # Blend with uniform probability
                    base_prob = np.ones(len(next_frames)) / len(next_frames)
                    probabilities = request_mask_generator.blend_with_probability(
                        base_prob,
                        mask,
                        request.get('weight', 1.0)
                    )
                    
                    # Safety check: ensure probabilities size matches
                    if len(probabilities) != len(next_frames):
                        print(f"⚠️  Probability size mismatch: {len(probabilities)} vs {len(next_frames)}, using uniform")
                        probabilities = np.ones(len(next_frames)) / len(next_frames)
                    
                    # Apply temperature
                    if temperature != 1.0:
                        probabilities = np.power(probabilities, 1.0 / temperature)
                        prob_sum = np.sum(probabilities)
                        if prob_sum > 0:
                            probabilities = probabilities / prob_sum
                        else:
                            probabilities = np.ones(len(next_frames)) / len(next_frames)
                    
                    # Final safety check before sampling
                    if len(probabilities) != len(next_frames) or not np.isclose(np.sum(probabilities), 1.0, atol=0.01):
                        print(f"⚠️  Invalid probabilities for sampling, using uniform")
                        probabilities = np.ones(len(next_frames)) / len(next_frames)
                    
                    # Sample
                    next_idx = np.random.choice(len(next_frames), p=probabilities)
                    next_frame = next_frames[next_idx]
                    
                except Exception as e:
                    print(f"⚠️  Error applying request mask: {e}")
                    # Fallback to uniform sampling
                    next_frame = next_frames[np.random.randint(len(next_frames))]
            
            # Add to generated sequence
            generated.append(next_frame)
            
            # Update state
            current_state = self.states[current_state]['next'][next_frame]
        
        return generated
    
    def save_to_file(self, filepath: str) -> bool:
        """
        Save polyphonic AudioOracle to file
        
        Args:
            filepath: Path to save file
            
        Returns:
            True if successful
        """
        try:
            # Prepare data for serialization
            print(f"💾 Saving PolyphonicAudioOracle: {len(self.audio_frames)} audio_frames, {len(self.states)} states")
            
            data = {
                'format_version': '2.0',  # Version for backward compatibility tracking
                'distance_threshold': self.distance_threshold,
                'distance_function': self.distance_function,
                'max_pattern_length': self.max_pattern_length,
                'feature_dimensions': self.feature_dimensions,
                'adaptive_threshold': self.adaptive_threshold,
                'chord_similarity_weight': self.chord_similarity_weight,
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
                'threshold_adjustments': getattr(self, 'threshold_adjustments', 0),
                # CRITICAL: Save audio_frames for note generation!
                'audio_frames': {
                    str(frame_id): {
                        'timestamp': frame.timestamp,
                        'features': frame.features.tolist() if hasattr(frame.features, 'tolist') else list(frame.features),
                        'audio_data': self._sanitize_audio_data(frame.audio_data),
                        'frame_id': frame.frame_id
                    }
                    for frame_id, frame in self.audio_frames.items()
                }
            }
            
            # Convert transitions to serializable format
            transitions_serializable = {}
            for (state, symbol), target_state in self.transitions.items():
                key = f"{state}_{symbol}"
                # Ensure target_state is native Python int
                transitions_serializable[key] = int(target_state) if isinstance(target_state, (np.integer, np.int64, np.int32)) else target_state
            data['transitions'] = transitions_serializable
            
            # Convert suffix links to serializable format
            suffix_links_serializable = {}
            for state, link_state in self.suffix_links.items():
                # Ensure both keys and values are native Python ints
                suffix_links_serializable[str(state)] = int(link_state) if isinstance(link_state, (np.integer, np.int64, np.int32)) else link_state
            data['suffix_links'] = suffix_links_serializable
            
            # Convert states to serializable format
            states_serializable = {}
            for state_id, state_data in self.states.items():
                states_serializable[str(state_id)] = {
                    'len': int(state_data['len']) if isinstance(state_data['len'], (np.integer, np.int64, np.int32)) else state_data['len'],
                    'link': int(state_data['link']) if isinstance(state_data['link'], (np.integer, np.int64, np.int32)) else state_data['link'],
                    'next': {
                        str(k): int(v) if isinstance(v, (np.integer, np.int64, np.int32)) else v 
                        for k, v in state_data['next'].items()
                    }
                }
            data['states'] = states_serializable
            
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=self._json_serialize_helper)
            
            print(f"✅ Polyphonic AudioOracle saved to: {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving polyphonic AudioOracle: {e}")
            return False
    
    def load_from_file(self, filepath: str) -> bool:
        """
        Load polyphonic AudioOracle from file
        
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
            self.feature_dimensions = data.get('feature_dimensions', 12)
            self.adaptive_threshold = data.get('adaptive_threshold', True)
            self.chord_similarity_weight = data.get('chord_similarity_weight', 0.3)
            
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
            
            # Load audio_frames (CRITICAL for note generation!)
            self.audio_frames = {}
            if 'audio_frames' in data:
                from .audio_oracle import AudioFrame
                for frame_id_str, frame_data in data['audio_frames'].items():
                    frame_id = int(frame_id_str)
                    self.audio_frames[frame_id] = AudioFrame(
                        timestamp=frame_data['timestamp'],
                        features=np.array(frame_data['features']),
                        audio_data=frame_data['audio_data'],
                        frame_id=frame_data['frame_id']
                    )
                print(f"✅ Loaded {len(self.audio_frames)} audio_frames")
            else:
                print(f"⚠️  No audio_frames in saved file (old model format)")
            
            print(f"✅ Polyphonic AudioOracle loaded from: {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading polyphonic AudioOracle: {e}")
            return False


def main():
    """Test Polyphonic AudioOracle"""
    print("🎵 Testing Polyphonic AudioOracle")
    print("=" * 50)
    
    # Create polyphonic AudioOracle
    ao = PolyphonicAudioOracle(distance_threshold=0.2, distance_function='euclidean')
    
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
    print("📝 Adding polyphonic sequence...")
    success = ao.add_polyphonic_sequence(mock_polyphonic_data)
    
    if success:
        print("✅ Polyphonic sequence added successfully!")
        
        # Test chord pattern finding
        print("\n🔍 Finding chord patterns...")
        chord_patterns = ao.find_chord_patterns(min_freq=1, min_len=1)
        print(f"Found {len(chord_patterns)} chord patterns:")
        for i, (pattern, freq) in enumerate(chord_patterns[:5]):
            print(f"  {i+1}. {' -> '.join(pattern)} (freq: {freq})")
        
        # Get polyphonic statistics
        print("\n📊 Polyphonic Statistics:")
        stats = ao.get_polyphonic_statistics()
        for key, value in stats.items():
            if isinstance(value, dict) and len(value) > 0:
                print(f"  {key}: {len(value)} entries")
            else:
                print(f"  {key}: {value}")
    
    print("\n✅ Polyphonic AudioOracle test complete!")


if __name__ == "__main__":
    main()
