#!/usr/bin/env python3
"""
Musical Gesture Processor - Perceptually-Grounded Feature Consolidation
=======================================================================

MUSICAL REASONING:
Humans perceive music in gestures and phrases, not millisecond-level samples.
A sustained C major chord is ONE musical gesture, not 100 separate events.
A rapid ornament or chord change is DISTINCT gestures, not smoothed together.

This processor bridges the gap between computer sampling (events every ~10ms)
and human musical perception (gestures lasting ~0.5-2 seconds).

KEY DIFFERENCES FROM TEMPORAL SMOOTHING:
- Temporal smoothing: Fixed time windows, arbitrary thresholds
- Musical gesture processing: Adaptive boundaries based on feature change magnitude
- Temporal smoothing: Can collapse genuine musical variation
- Musical gesture processing: Preserves transitions, consolidates sustains

PIPELINE PLACEMENT:
This happens AFTER raw feature extraction, BEFORE AudioOracle training.
Same processor used in both training and performance for consistency.

Musical time scales (reference):
- Note onset: ~10-500ms
- Musical gesture: ~300-2000ms (typical)
- Phrase: ~2-8 seconds
- Section: ~30-120 seconds

This processor works at the GESTURE level.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from collections import deque


@dataclass
class MusicalGesture:
    """
    Represents a consolidated musical gesture.
    
    A gesture is a perceptually unified musical event, such as:
    - A sustained chord (multiple onsets, similar features)
    - A rapid ornament sequence (brief cluster of related events)
    - A single articulated note or chord
    """
    feature: np.ndarray          # Consolidated feature vector
    timestamp: float              # Representative timestamp
    duration: float               # Gesture duration in seconds
    event_count: int              # Number of raw events consolidated
    feature_variance: float       # Internal feature variance (stability indicator)
    boundary_type: str           # 'transition', 'sustain', 'single', 'timeout'
    
    def __repr__(self):
        return (f"MusicalGesture(t={self.timestamp:.2f}s, dur={self.duration:.3f}s, "
                f"events={self.event_count}, var={self.feature_variance:.3f}, type={self.boundary_type})")


class MusicalGestureProcessor:
    """
    Process raw audio features into musical gestures based on perceptual boundaries.
    
    MUSICAL LOGIC:
    1. Accumulate events into gesture buffer
    2. Detect gesture boundaries when:
       - Feature change > transition_threshold (new musical gesture)
       - Duration exceeds max_gesture_duration (force boundary for responsiveness)
       - Feature stability indicates sustained gesture completion
    3. Consolidate gesture events into single representative feature
    4. Preserve transitions, consolidate sustains
    
    PARAMETERS (Musical Interpretation):
    - transition_threshold: How different features must be to indicate new gesture
      (0.2 = subtle change, 0.3 = moderate, 0.5 = dramatic)
    - sustain_threshold: How similar features can be to consider sustained
      (0.1 = very stable, 0.2 = moderately stable)
    - min_gesture_duration: Minimum gesture length (prevents micro-gestures)
    - max_gesture_duration: Maximum gesture length (ensures responsiveness)
    """
    
    def __init__(self,
                 transition_threshold: float = 0.3,    # Moderate feature change = new gesture
                 sustain_threshold: float = 0.15,      # Stable features = sustained gesture
                 min_gesture_duration: float = 0.2,    # Minimum ~200ms gesture
                 max_gesture_duration: float = 2.0,    # Maximum 2s gesture (force boundary)
                 consolidation_method: str = 'weighted_median'):  # or 'mean', 'first', 'stable'
        """
        Initialize musical gesture processor.
        
        Args:
            transition_threshold: Feature distance indicating new gesture (higher = fewer boundaries)
            sustain_threshold: Feature distance indicating sustained gesture (lower = more sustain)
            min_gesture_duration: Minimum gesture duration in seconds
            max_gesture_duration: Maximum gesture duration before forced boundary
            consolidation_method: How to consolidate gesture events into representative feature
        """
        self.transition_threshold = transition_threshold
        self.sustain_threshold = sustain_threshold
        self.min_gesture_duration = min_gesture_duration
        self.max_gesture_duration = max_gesture_duration
        self.consolidation_method = consolidation_method
        
        # Statistics for transparency
        self.stats = {
            'total_raw_events': 0,
            'total_gestures': 0,
            'transition_boundaries': 0,
            'sustain_boundaries': 0,
            'timeout_boundaries': 0,
            'single_event_gestures': 0,
            'avg_consolidation_ratio': 0.0
        }
    
    def process_features(self, 
                        features: np.ndarray, 
                        timestamps: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[MusicalGesture]]:
        """
        Convert raw features into musical gestures.
        
        Args:
            features: Array of feature vectors (N x feature_dim)
            timestamps: Array of timestamps (N,)
        
        Returns:
            Tuple of:
            - consolidated_features: Array of gesture features (M x feature_dim, M <= N)
            - consolidated_timestamps: Array of gesture timestamps (M,)
            - gestures: List of MusicalGesture objects with metadata
        """
        if len(features) == 0:
            return np.array([]), np.array([]), []
        
        if len(features) != len(timestamps):
            raise ValueError(f"Features and timestamps length mismatch: {len(features)} vs {len(timestamps)}")
        
        # L2 normalize features for consistent distance calculation
        # This ensures thresholds (0.3-0.6) work in normalized space
        feature_norms = np.linalg.norm(features, axis=1, keepdims=True)
        feature_norms = np.where(feature_norms == 0, 1, feature_norms)  # Avoid division by zero
        features = features / feature_norms
        
        self.stats['total_raw_events'] = len(features)
        
        gestures = []
        current_gesture_events = []
        
        for i, (feature, timestamp) in enumerate(zip(features, timestamps)):
            if not current_gesture_events:
                # Start new gesture
                current_gesture_events.append((feature, timestamp))
                continue
            
            # Check if this event continues current gesture or starts new one
            gesture_start_time = current_gesture_events[0][1]
            gesture_duration = timestamp - gesture_start_time
            
            # Calculate feature similarity to current gesture
            gesture_features = np.array([e[0] for e in current_gesture_events])
            avg_gesture_feature = np.mean(gesture_features, axis=0)
            feature_distance = np.linalg.norm(feature - avg_gesture_feature)
            
            # Determine if we need a gesture boundary
            boundary_detected = False
            boundary_type = None
            
            # Condition 1: Dramatic feature change (musical transition)
            if feature_distance > self.transition_threshold:
                boundary_detected = True
                boundary_type = 'transition'
                self.stats['transition_boundaries'] += 1
            
            # Condition 2: Maximum duration exceeded (force boundary for responsiveness)
            elif gesture_duration > self.max_gesture_duration:
                boundary_detected = True
                boundary_type = 'timeout'
                self.stats['timeout_boundaries'] += 1
            
            # Condition 3: Stable sustained gesture with sufficient duration
            elif (gesture_duration > self.min_gesture_duration and 
                  feature_distance < self.sustain_threshold and
                  len(current_gesture_events) >= 3):
                # Check if adding this event would maintain stability
                # If so, continue gesture; if stability breaks, create boundary
                gesture_variance = np.std([np.linalg.norm(e[0] - avg_gesture_feature) 
                                          for e in current_gesture_events])
                
                if gesture_variance < self.sustain_threshold * 0.5:
                    # Very stable gesture, check if we should finalize it
                    # Only finalize if next event seems different
                    if i < len(features) - 1:
                        next_feature = features[i + 1]
                        next_distance = np.linalg.norm(next_feature - avg_gesture_feature)
                        if next_distance > self.transition_threshold * 0.7:
                            boundary_detected = True
                            boundary_type = 'sustain'
                            self.stats['sustain_boundaries'] += 1
            
            if boundary_detected:
                # Consolidate current gesture
                gesture = self._consolidate_gesture(current_gesture_events, boundary_type)
                gestures.append(gesture)
                
                # Start new gesture with current event
                current_gesture_events = [(feature, timestamp)]
            else:
                # Continue current gesture
                current_gesture_events.append((feature, timestamp))
        
        # Consolidate final gesture
        if current_gesture_events:
            gesture = self._consolidate_gesture(current_gesture_events, 'end')
            gestures.append(gesture)
        
        # Update statistics
        self.stats['total_gestures'] = len(gestures)
        self.stats['single_event_gestures'] = sum(1 for g in gestures if g.event_count == 1)
        if len(gestures) > 0:
            self.stats['avg_consolidation_ratio'] = self.stats['total_raw_events'] / len(gestures)
        
        # Extract arrays for return
        consolidated_features = np.array([g.feature for g in gestures])
        consolidated_timestamps = np.array([g.timestamp for g in gestures])
        
        return consolidated_features, consolidated_timestamps, gestures
    
    def _consolidate_gesture(self, 
                            gesture_events: List[Tuple[np.ndarray, float]], 
                            boundary_type: str) -> MusicalGesture:
        """
        Consolidate multiple events into a single musical gesture.
        
        Args:
            gesture_events: List of (feature, timestamp) tuples
            boundary_type: Type of boundary that ended this gesture
        
        Returns:
            MusicalGesture object with consolidated representation
        """
        if not gesture_events:
            raise ValueError("Cannot consolidate empty gesture")
        
        features = np.array([e[0] for e in gesture_events])
        timestamps = np.array([e[1] for e in gesture_events])
        
        # Calculate gesture metadata
        duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0.0
        event_count = len(gesture_events)
        
        # Calculate feature variance (stability indicator)
        if len(features) > 1:
            mean_feature = np.mean(features, axis=0)
            feature_variance = np.mean([np.linalg.norm(f - mean_feature) for f in features])
        else:
            feature_variance = 0.0
        
        # Consolidate feature based on method
        if self.consolidation_method == 'mean':
            consolidated_feature = np.mean(features, axis=0)
        
        elif self.consolidation_method == 'weighted_median':
            # Weight recent events more heavily (they're more representative of gesture's final state)
            weights = np.linspace(0.5, 1.0, len(features))
            weights = weights / np.sum(weights)
            consolidated_feature = np.average(features, axis=0, weights=weights)
        
        elif self.consolidation_method == 'first':
            # Use first event (gesture onset)
            consolidated_feature = features[0]
        
        elif self.consolidation_method == 'peak':
            # Use the event with highest feature magnitude (most salient moment)
            # This preserves the most perceptually distinct moment within the gesture
            # WITHOUT averaging (which destroys diversity)
            magnitudes = [np.linalg.norm(f) for f in features]
            peak_idx = np.argmax(magnitudes)
            consolidated_feature = features[peak_idx]
        
        elif self.consolidation_method == 'stable':
            # Use the most stable feature (closest to mean)
            mean_feature = np.mean(features, axis=0)
            distances = [np.linalg.norm(f - mean_feature) for f in features]
            stable_idx = np.argmin(distances)
            consolidated_feature = features[stable_idx]
        
        else:
            consolidated_feature = np.mean(features, axis=0)
        
        # Representative timestamp (weighted toward gesture center)
        representative_time = np.mean(timestamps)
        
        return MusicalGesture(
            feature=consolidated_feature,
            timestamp=representative_time,
            duration=duration,
            event_count=event_count,
            feature_variance=feature_variance,
            boundary_type=boundary_type
        )
    
    def get_statistics(self) -> Dict:
        """Get processing statistics for transparency and debugging."""
        stats = self.stats.copy()
        stats['parameters'] = {
            'transition_threshold': self.transition_threshold,
            'sustain_threshold': self.sustain_threshold,
            'min_gesture_duration': self.min_gesture_duration,
            'max_gesture_duration': self.max_gesture_duration,
            'consolidation_method': self.consolidation_method
        }
        return stats
    
    def print_statistics(self):
        """Print human-readable processing statistics."""
        print(f"\n🎵 Musical Gesture Processing Statistics:")
        print(f"   Raw events: {self.stats['total_raw_events']}")
        print(f"   Gestures created: {self.stats['total_gestures']}")
        print(f"   Consolidation ratio: {self.stats['avg_consolidation_ratio']:.2f}x")
        print(f"   Boundary types:")
        print(f"     - Transition (feature change): {self.stats['transition_boundaries']}")
        print(f"     - Sustain (stable completion): {self.stats['sustain_boundaries']}")
        print(f"     - Timeout (max duration): {self.stats['timeout_boundaries']}")
        print(f"     - Single event gestures: {self.stats['single_event_gestures']}")
        print(f"   Parameters:")
        print(f"     - Transition threshold: {self.transition_threshold}")
        print(f"     - Sustain threshold: {self.sustain_threshold}")
        print(f"     - Gesture duration: {self.min_gesture_duration:.1f}s - {self.max_gesture_duration:.1f}s")


def demo():
    """Demonstrate musical gesture processing on synthetic data."""
    print("🎼 Musical Gesture Processor Demo")
    print("=" * 60)
    
    # Create synthetic musical scenario
    # Sustained chord (10 similar events) → transition → new chord (10 similar events)
    
    np.random.seed(42)
    
    # Gesture 1: Sustained C major (events 0-9)
    base_feature_1 = np.random.randn(768)
    base_feature_1 = base_feature_1 / np.linalg.norm(base_feature_1)  # Normalize
    features_gesture_1 = [base_feature_1 + np.random.randn(768) * 0.01 for _ in range(10)]
    timestamps_1 = np.linspace(0.0, 1.0, 10)
    
    # Gesture 2: Transition to F major (events 10-19)
    direction = np.random.randn(768)
    direction = direction / np.linalg.norm(direction)
    base_feature_2 = base_feature_1 + direction * 0.5  # Large change in normalized space
    features_gesture_2 = [base_feature_2 + np.random.randn(768) * 0.01 for _ in range(10)]
    timestamps_2 = np.linspace(1.1, 2.1, 10)
    
    # Combine
    features = np.array(features_gesture_1 + features_gesture_2)
    timestamps = np.concatenate([timestamps_1, timestamps_2])
    
    print(f"\n📊 Input: {len(features)} raw events over {timestamps[-1]:.1f}s")
    print(f"   Expected: 2 musical gestures (sustained chord → transition → new chord)")
    
    # Process with musical gesture processor
    processor = MusicalGestureProcessor(
        transition_threshold=0.3,
        sustain_threshold=0.15,
        min_gesture_duration=0.3,
        max_gesture_duration=2.0
    )
    
    consolidated_features, consolidated_timestamps, gestures = processor.process_features(features, timestamps)
    
    print(f"\n✅ Output: {len(gestures)} musical gestures")
    for i, gesture in enumerate(gestures):
        print(f"   Gesture {i+1}: {gesture}")
    
    processor.print_statistics()


if __name__ == "__main__":
    demo()
