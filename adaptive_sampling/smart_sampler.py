#!/usr/bin/env python3
"""
Smart Sampler
Based on current research in music information retrieval and adaptive sampling

This module implements adaptive sampling strategies that:
- Sample events based on structural analysis
- Focus on perceptually significant moments
- Maintain temporal distribution
- Limit to 10K events for efficiency
"""

import numpy as np
import librosa
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import random

@dataclass
class SamplingStrategy:
    """Represents a sampling strategy configuration"""
    name: str
    max_events: int
    structural_weight: float
    perceptual_weight: float
    temporal_weight: float
    onset_weight: float

@dataclass
class SampledEvent:
    """Represents a sampled musical event"""
    event_id: str
    time: float
    features: np.ndarray
    significance_score: float
    sampling_reason: str
    original_index: int

class SmartSampler:
    """
    Implements smart sampling strategies for musical events
    
    Based on research showing that humans focus on:
    - Structural boundaries
    - Perceptually significant moments
    - Onset events
    - Temporally distributed patterns
    """
    
    def __init__(self, 
                 max_events: int = 10000,
                 structural_weight: float = 0.3,
                 perceptual_weight: float = 0.3,
                 temporal_weight: float = 0.2,
                 onset_weight: float = 0.2):
        
        self.max_events = max_events
        self.structural_weight = structural_weight
        self.perceptual_weight = perceptual_weight
        self.temporal_weight = temporal_weight
        self.onset_weight = onset_weight
        
        # Sampling strategies
        self.strategies = {
            'structural': SamplingStrategy(
                name='structural',
                max_events=max_events,
                structural_weight=0.5,
                perceptual_weight=0.2,
                temporal_weight=0.2,
                onset_weight=0.1
            ),
            'perceptual': SamplingStrategy(
                name='perceptual',
                max_events=max_events,
                structural_weight=0.2,
                perceptual_weight=0.5,
                temporal_weight=0.2,
                onset_weight=0.1
            ),
            'balanced': SamplingStrategy(
                name='balanced',
                max_events=max_events,
                structural_weight=0.25,
                perceptual_weight=0.25,
                temporal_weight=0.25,
                onset_weight=0.25
            )
        }
    
    def sample_events(self, audio_file: str, strategy: str = 'balanced') -> List[SampledEvent]:
        """
        Sample events from audio file using specified strategy
        
        Args:
            audio_file: Path to audio file
            strategy: Sampling strategy ('structural', 'perceptual', 'balanced')
            
        Returns:
            List of sampled events
        """
        print(f"🎯 Sampling events from: {audio_file}")
        print(f"📊 Strategy: {strategy}, Max events: {self.max_events}")
        
        # Load audio and extract features
        y, sr = librosa.load(audio_file, sr=22050)
        duration = len(y) / sr
        
        print(f"✅ Audio loaded: {duration:.2f}s duration")
        
        # Extract comprehensive features
        features = self._extract_comprehensive_features(y, sr)
        print(f"✅ Features extracted: {features.shape}")
        
        # Calculate sampling scores
        sampling_scores = self._calculate_sampling_scores(features, y, sr, strategy)
        print(f"✅ Sampling scores calculated: {len(sampling_scores)} events")
        
        # Select events based on scores
        sampled_events = self._select_events(features, sampling_scores, sr, strategy)
        print(f"✅ Selected {len(sampled_events)} events")
        
        return sampled_events
    
    def _extract_comprehensive_features(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Extract comprehensive features for sampling analysis"""
        
        hop_length = 512
        frame_length = 2048
        
        # Chroma features for harmonic analysis
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
        
        # Spectral features for timbral analysis
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
        spectral_rolloffs = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)[0]
        spectral_bandwidths = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)[0]
        
        # Temporal features
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        
        # Onset features for rhythmic analysis
        onset_strength = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        
        # MFCC features for timbral analysis
        mfccs = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)
        
        # Combine features - ensure all arrays have the same length
        min_length = min(len(chroma[0]), len(spectral_centroids), len(spectral_rolloffs), 
                       len(spectral_bandwidths), len(rms), len(onset_strength), len(mfccs[0]))
        
        features = np.vstack([
            chroma[:, :min_length],
            spectral_centroids[:min_length],
            spectral_rolloffs[:min_length],
            spectral_bandwidths[:min_length],
            rms[:min_length],
            onset_strength[:min_length],
            mfccs[:, :min_length]
        ])
        
        return features.T  # Shape: (time_frames, features)
    
    def _calculate_sampling_scores(self, features: np.ndarray, y: np.ndarray, 
                                 sr: int, strategy: str) -> np.ndarray:
        """Calculate sampling scores for each frame"""
        
        strategy_config = self.strategies[strategy]
        n_frames = len(features)
        
        # Initialize scores
        scores = np.zeros(n_frames)
        
        # 1. Structural significance (based on feature changes)
        structural_scores = self._calculate_structural_scores(features)
        scores += structural_scores * strategy_config.structural_weight
        
        # 2. Perceptual significance (based on feature magnitude and variation)
        perceptual_scores = self._calculate_perceptual_scores(features)
        scores += perceptual_scores * strategy_config.perceptual_weight
        
        # 3. Temporal distribution (ensure even distribution)
        temporal_scores = self._calculate_temporal_scores(features, sr)
        scores += temporal_scores * strategy_config.temporal_weight
        
        # 4. Onset significance (based on onset strength)
        onset_scores = self._calculate_onset_scores(features)
        scores += onset_scores * strategy_config.onset_weight
        
        return scores
    
    def _calculate_structural_scores(self, features: np.ndarray) -> np.ndarray:
        """Calculate structural significance scores"""
        
        n_frames = len(features)
        scores = np.zeros(n_frames)
        
        # Calculate feature differences between consecutive frames
        feature_diffs = np.diff(features, axis=0)
        
        # Structural significance based on feature changes
        for i in range(n_frames - 1):
            # Magnitude of change
            change_magnitude = np.linalg.norm(feature_diffs[i])
            scores[i + 1] = change_magnitude
        
        # Normalize scores
        if np.max(scores) > 0:
            scores = scores / np.max(scores)
        
        return scores
    
    def _calculate_perceptual_scores(self, features: np.ndarray) -> np.ndarray:
        """Calculate perceptual significance scores"""
        
        n_frames = len(features)
        scores = np.zeros(n_frames)
        
        # Perceptual significance based on feature magnitude and variation
        for i in range(n_frames):
            # Feature magnitude
            magnitude = np.linalg.norm(features[i])
            
            # Feature variation (deviation from mean)
            if i > 0 and i < n_frames - 1:
                local_mean = np.mean([features[i-1], features[i+1]], axis=0)
                variation = np.linalg.norm(features[i] - local_mean)
            else:
                variation = 0.0
            
            # Combine magnitude and variation
            scores[i] = magnitude + variation
        
        # Normalize scores
        if np.max(scores) > 0:
            scores = scores / np.max(scores)
        
        return scores
    
    def _calculate_temporal_scores(self, features: np.ndarray, sr: int) -> np.ndarray:
        """Calculate temporal distribution scores"""
        
        n_frames = len(features)
        scores = np.zeros(n_frames)
        
        # Temporal distribution based on position in the piece
        for i in range(n_frames):
            # Position-based scoring (higher scores for beginning, middle, end)
            position = i / n_frames
            
            if position < 0.1 or position > 0.9:  # Beginning or end
                scores[i] = 1.0
            elif 0.4 < position < 0.6:  # Middle
                scores[i] = 0.8
            else:  # Other positions
                scores[i] = 0.5
        
        return scores
    
    def _calculate_onset_scores(self, features: np.ndarray) -> np.ndarray:
        """Calculate onset significance scores"""
        
        n_frames = len(features)
        scores = np.zeros(n_frames)
        
        # Onset significance based on onset strength (last feature)
        if features.shape[1] > 0:
            onset_strength = features[:, -1]  # Last feature is onset strength
            scores = onset_strength.copy()
        
        # Normalize scores
        if np.max(scores) > 0:
            scores = scores / np.max(scores)
        
        return scores
    
    def _select_events(self, features: np.ndarray, scores: np.ndarray, 
                      sr: int, strategy: str) -> List[SampledEvent]:
        """Select events based on sampling scores"""
        
        n_frames = len(features)
        hop_length = 512
        
        # Sort frames by score
        sorted_indices = np.argsort(scores)[::-1]  # Descending order
        
        # Select top events
        selected_indices = sorted_indices[:self.max_events]
        
        # Create sampled events
        sampled_events = []
        
        for i, frame_idx in enumerate(selected_indices):
            time = frame_idx * hop_length / sr
            
            # Determine sampling reason
            reason = self._determine_sampling_reason(frame_idx, scores, strategy)
            
            event = SampledEvent(
                event_id=f"event_{i:06d}",
                time=time,
                features=features[frame_idx].copy(),
                significance_score=scores[frame_idx],
                sampling_reason=reason,
                original_index=frame_idx
            )
            
            sampled_events.append(event)
        
        return sampled_events
    
    def _determine_sampling_reason(self, frame_idx: int, scores: np.ndarray, 
                                 strategy: str) -> str:
        """Determine why this frame was sampled"""
        
        score = scores[frame_idx]
        
        # Determine reason based on score and strategy
        if strategy == 'structural' and score > 0.8:
            return 'structural_boundary'
        elif strategy == 'perceptual' and score > 0.8:
            return 'perceptual_significance'
        elif strategy == 'balanced':
            if score > 0.9:
                return 'high_significance'
            elif score > 0.7:
                return 'medium_significance'
            else:
                return 'temporal_distribution'
        else:
            return 'sampled'

class StructuralSampler:
    """
    Specialized sampler for structural analysis
    
    Focuses on detecting structural boundaries and important musical moments.
    """
    
    def __init__(self, max_events: int = 10000):
        self.max_events = max_events
    
    def sample_structural_events(self, audio_file: str) -> List[SampledEvent]:
        """Sample events focusing on structural boundaries"""
        
        print(f"🏗️  Sampling structural events from: {audio_file}")
        
        # Load audio and extract features
        y, sr = librosa.load(audio_file, sr=22050)
        
        # Extract structural features
        features = self._extract_structural_features(y, sr)
        
        # Detect structural boundaries
        boundaries = self._detect_structural_boundaries(features, sr)
        
        # Sample events around boundaries
        sampled_events = self._sample_around_boundaries(features, boundaries, sr)
        
        print(f"✅ Sampled {len(sampled_events)} structural events")
        
        return sampled_events
    
    def _extract_structural_features(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Extract features optimized for structural analysis"""
        
        hop_length = 1024  # Larger hop for structural analysis
        
        # Chroma features for harmonic analysis
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
        
        # Spectral features for timbral analysis
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
        
        # Temporal features
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        
        # Combine features
        min_length = min(len(chroma[0]), len(spectral_centroids), len(rms))
        
        features = np.vstack([
            chroma[:, :min_length],
            spectral_centroids[:min_length],
            rms[:min_length]
        ])
        
        return features.T
    
    def _detect_structural_boundaries(self, features: np.ndarray, sr: int) -> List[int]:
        """Detect structural boundaries in features"""
        
        # Calculate feature differences
        feature_diffs = np.diff(features, axis=0)
        
        # Calculate boundary strength
        boundary_strength = np.linalg.norm(feature_diffs, axis=1)
        
        # Find peaks in boundary strength
        from scipy.signal import find_peaks
        
        peaks, _ = find_peaks(boundary_strength, height=np.mean(boundary_strength))
        
        return peaks.tolist()
    
    def _sample_around_boundaries(self, features: np.ndarray, boundaries: List[int], 
                                sr: int) -> List[SampledEvent]:
        """Sample events around structural boundaries"""
        
        hop_length = 1024
        sampled_events = []
        
        # Sample events around each boundary
        for i, boundary in enumerate(boundaries):
            # Sample events in a window around the boundary
            window_size = 5  # frames
            
            start_idx = max(0, boundary - window_size)
            end_idx = min(len(features), boundary + window_size)
            
            for j in range(start_idx, end_idx):
                time = j * hop_length / sr
                
                event = SampledEvent(
                    event_id=f"structural_{i:04d}_{j:04d}",
                    time=time,
                    features=features[j].copy(),
                    significance_score=1.0 if j == boundary else 0.5,
                    sampling_reason='structural_boundary',
                    original_index=j
                )
                
                sampled_events.append(event)
        
        # Limit to max_events
        if len(sampled_events) > self.max_events:
            # Sort by significance and take top events
            sampled_events.sort(key=lambda x: x.significance_score, reverse=True)
            sampled_events = sampled_events[:self.max_events]
        
        return sampled_events

def main():
    """Test the smart sampler"""
    
    print("🚀 Starting smart sampler test...")
    
    sampler = SmartSampler(max_events=1000)  # Smaller for testing
    
    # Test with sample audio file
    audio_file = "input_audio/Grab-a-hold.mp3"
    
    try:
        # Test different strategies
        strategies = ['balanced', 'structural', 'perceptual']
        
        for strategy in strategies:
            print(f"\n🎯 Testing {strategy} strategy...")
            
            events = sampler.sample_events(audio_file, strategy)
            
            print(f"✅ Sampled {len(events)} events")
            
            # Show some examples
            print(f"   Top 5 events:")
            for i, event in enumerate(events[:5]):
                print(f"     {i+1}. {event.event_id}: {event.time:.2f}s, "
                      f"score: {event.significance_score:.3f}, "
                      f"reason: {event.sampling_reason}")
            
            # Show sampling reason distribution
            reasons = [event.sampling_reason for event in events]
            reason_counts = {}
            for reason in reasons:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
            
            print(f"   Sampling reason distribution:")
            for reason, count in reason_counts.items():
                print(f"     {reason}: {count} events")
        
        # Test structural sampler
        print(f"\n🏗️  Testing structural sampler...")
        
        structural_sampler = StructuralSampler(max_events=500)
        structural_events = structural_sampler.sample_structural_events(audio_file)
        
        print(f"✅ Sampled {len(structural_events)} structural events")
        
    except FileNotFoundError:
        print(f"❌ Audio file not found: {audio_file}")
        print("Please provide a valid audio file path")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
