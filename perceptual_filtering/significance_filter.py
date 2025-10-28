#!/usr/bin/env python3
"""
Perceptual Significance Filter
Based on Pressnitzer et al. (2008): "Perceptual organization of sound begins in the auditory periphery"

This module implements perceptual filtering that mirrors human auditory processing:
- Stream segregation based on similarity
- Multi-second adaptation effects
- Perceptual grouping principles
"""

import numpy as np
import librosa
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import scipy.signal

@dataclass
class AuditoryStream:
    """Represents an auditory stream (grouped sounds)"""
    stream_id: int
    start_time: float
    end_time: float
    features: np.ndarray
    confidence: float
    stream_type: str  # 'melody', 'bass', 'harmony', 'percussion'
    members: List[int]  # Frame indices

@dataclass
class PerceptualSignificance:
    """Represents perceptual significance of a pattern"""
    pattern_id: str
    significance_score: float
    perceptual_factors: Dict[str, float]
    stream_membership: List[int]
    adaptation_level: float

class AuditorySceneAnalyzer:
    """
    Analyzes auditory scene and performs stream segregation
    
    Based on research showing that humans group sounds into streams based on:
    - Frequency similarity
    - Temporal proximity
    - Spectral continuity
    - Multi-second adaptation
    """
    
    def __init__(self, 
                 stream_similarity_threshold: float = 0.6,
                 temporal_window: float = 2.0,
                 adaptation_window: float = 10.0):
        self.stream_similarity_threshold = stream_similarity_threshold
        self.temporal_window = temporal_window
        self.adaptation_window = adaptation_window
        
        # Stream type templates
        self.stream_templates = self._create_stream_templates()
        
    def _create_stream_templates(self) -> Dict[str, np.ndarray]:
        """Create templates for different stream types"""
        
        templates = {}
        
        # Melody template (mid-range frequencies, moderate energy)
        templates['melody'] = np.array([
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.7, 0.6, 0.5, 0.4,  # Chroma
            0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2,  # Spectral
            0.4, 0.5, 0.6, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.0   # Temporal
        ])
        
        # Bass template (low frequencies, high energy)
        templates['bass'] = np.array([
            0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0,  # Chroma
            0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0,  # Spectral
            0.7, 0.8, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0   # Temporal
        ])
        
        # Harmony template (mid-high frequencies, moderate energy)
        templates['harmony'] = np.array([
            0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2,  # Chroma
            0.4, 0.5, 0.6, 0.7, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1,  # Spectral
            0.5, 0.6, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0   # Temporal
        ])
        
        # Percussion template (broad spectrum, high energy, short duration)
        templates['percussion'] = np.array([
            0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,  # Chroma
            0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8,  # Spectral
            0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1   # Temporal
        ])
        
        return templates
    
    def analyze_auditory_scene(self, audio_file: str) -> List[AuditoryStream]:
        """
        Analyze auditory scene and perform stream segregation
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            List of detected auditory streams
        """
        print(f"🎧 Analyzing auditory scene: {audio_file}")
        
        # Load audio and extract features
        y, sr = librosa.load(audio_file, sr=22050)
        features = self._extract_stream_features(y, sr)
        
        # Perform stream segregation
        streams = self._perform_stream_segregation(features, sr)
        
        # Classify streams by type
        classified_streams = self._classify_streams(streams, features)
        
        print(f"✅ Detected {len(classified_streams)} auditory streams")
        
        return classified_streams
    
    def _extract_stream_features(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Extract features optimized for stream segregation"""
        
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
        
        # Onset features for rhythmic analysis (simplified for speed)
        # onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop_length)
        onset_strength = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        
        # Combine features - ensure all arrays have the same length
        min_length = min(len(chroma[0]), len(spectral_centroids), len(spectral_rolloffs), 
                       len(spectral_bandwidths), len(rms), len(onset_strength))
        
        features = np.vstack([
            chroma[:, :min_length],
            spectral_centroids[:min_length],
            spectral_rolloffs[:min_length],
            spectral_bandwidths[:min_length],
            rms[:min_length],
            onset_strength[:min_length]
        ])
        
        return features.T  # Shape: (time_frames, features)
    
    def _perform_stream_segregation(self, features: np.ndarray, sr: int) -> List[AuditoryStream]:
        """Perform stream segregation using clustering"""
        
        print("🔄 Performing stream segregation...")
        
        # Use DBSCAN for stream clustering
        clustering = DBSCAN(
            eps=self.stream_similarity_threshold,
            min_samples=5,
            metric='cosine'
        )
        
        cluster_labels = clustering.fit_predict(features)
        
        # Convert clusters to streams
        streams = []
        hop_length = 512
        
        for cluster_id in np.unique(cluster_labels):
            if cluster_id == -1:  # Noise points
                continue
                
            # Get frames belonging to this cluster
            cluster_frames = np.where(cluster_labels == cluster_id)[0]
            
            if len(cluster_frames) < 5:  # Minimum stream size
                continue
            
            # Calculate stream characteristics
            start_frame = min(cluster_frames)
            end_frame = max(cluster_frames)
            
            start_time = start_frame * hop_length / sr
            end_time = end_frame * hop_length / sr
            
            # Calculate stream features
            stream_features = np.mean([features[frame] for frame in cluster_frames], axis=0)
            
            # Calculate confidence based on cluster cohesion
            confidence = self._calculate_stream_confidence(cluster_frames, features)
            
            stream = AuditoryStream(
                stream_id=cluster_id,
                start_time=start_time,
                end_time=end_time,
                features=stream_features,
                confidence=confidence,
                stream_type='unknown',
                members=cluster_frames.tolist()
            )
            
            streams.append(stream)
        
        return streams
    
    def _calculate_stream_confidence(self, cluster_frames: np.ndarray, features: np.ndarray) -> float:
        """Calculate confidence of a stream based on internal cohesion"""
        
        if len(cluster_frames) < 2:
            return 0.0
        
        # Calculate pairwise similarities within cluster
        similarities = []
        for i in range(len(cluster_frames)):
            for j in range(i + 1, len(cluster_frames)):
                sim = cosine_similarity(
                    [features[cluster_frames[i]]], 
                    [features[cluster_frames[j]]]
                )[0][0]
                similarities.append(sim)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _classify_streams(self, streams: List[AuditoryStream], features: np.ndarray) -> List[AuditoryStream]:
        """Classify streams by type using templates"""
        
        print("🏷️  Classifying streams...")
        
        classified_streams = []
        
        for stream in streams:
            # Calculate similarity to each template
            similarities = {}
            for stream_type, template in self.stream_templates.items():
                # Ensure template matches feature dimension
                if len(template) != len(stream.features):
                    # Pad or truncate template
                    if len(template) < len(stream.features):
                        template = np.pad(template, (0, len(stream.features) - len(template)))
                    else:
                        template = template[:len(stream.features)]
                
                similarity = cosine_similarity([stream.features], [template])[0][0]
                similarities[stream_type] = similarity
            
            # Assign stream type based on highest similarity
            best_type = max(similarities, key=similarities.get)
            stream.stream_type = best_type
            
            classified_streams.append(stream)
        
        return classified_streams

class PerceptualSignificanceFilter:
    """
    Filters patterns based on perceptual significance
    
    Based on research showing that humans focus on:
    - Temporally significant events
    - Harmonically significant patterns
    - Structurally important sections
    - Auditory stream continuity
    """
    
    def __init__(self, 
                 significance_threshold: float = 0.5,
                 adaptation_decay: float = 0.1):
        self.significance_threshold = significance_threshold
        self.adaptation_decay = adaptation_decay
        
    def filter_patterns(self, patterns: List, streams: List[AuditoryStream]) -> List[PerceptualSignificance]:
        """
        Filter patterns based on perceptual significance
        
        Args:
            patterns: List of detected patterns
            streams: List of auditory streams
            
        Returns:
            List of perceptually significant patterns
        """
        print(f"🔍 Filtering {len(patterns)} patterns for perceptual significance...")
        
        significant_patterns = []
        
        for pattern in patterns:
            significance = self._calculate_perceptual_significance(pattern, streams)
            
            if significance.significance_score > self.significance_threshold:
                significant_patterns.append(significance)
        
        print(f"✅ {len(significant_patterns)} patterns passed perceptual significance filter")
        
        return significant_patterns
    
    def _calculate_perceptual_significance(self, pattern, streams: List[AuditoryStream]) -> PerceptualSignificance:
        """Calculate perceptual significance of a pattern"""
        
        # Calculate different perceptual factors
        temporal_significance = self._calculate_temporal_significance(pattern)
        harmonic_significance = self._calculate_harmonic_significance(pattern)
        structural_significance = self._calculate_structural_significance(pattern)
        stream_significance = self._calculate_stream_significance(pattern, streams)
        adaptation_level = self._calculate_adaptation_level(pattern)
        
        # Combine factors with weights
        perceptual_factors = {
            'temporal': temporal_significance,
            'harmonic': harmonic_significance,
            'structural': structural_significance,
            'stream': stream_significance
        }
        
        # Weighted combination
        significance_score = (
            0.3 * temporal_significance +
            0.3 * harmonic_significance +
            0.2 * structural_significance +
            0.2 * stream_significance
        )
        
        # Apply adaptation decay
        significance_score *= (1.0 - adaptation_level * self.adaptation_decay)
        
        return PerceptualSignificance(
            pattern_id=pattern.pattern_id,
            significance_score=significance_score,
            perceptual_factors=perceptual_factors,
            stream_membership=[],
            adaptation_level=adaptation_level
        )
    
    def _calculate_temporal_significance(self, pattern) -> float:
        """Calculate temporal significance based on timing and duration"""
        
        # Longer patterns are more significant
        duration = pattern.end_time - pattern.start_time
        duration_score = min(duration / 30.0, 1.0)  # Normalize to 30 seconds
        
        # Patterns with more repetitions are more significant
        repetition_score = min(pattern.repetitions / 5.0, 1.0)  # Normalize to 5 repetitions
        
        return (duration_score + repetition_score) / 2.0
    
    def _calculate_harmonic_significance(self, pattern) -> float:
        """Calculate harmonic significance based on chord complexity"""
        
        # Use pattern confidence as proxy for harmonic significance
        return pattern.confidence
    
    def _calculate_structural_significance(self, pattern) -> float:
        """Calculate structural significance based on position and context"""
        
        # Patterns at structural boundaries are more significant
        # This is a simplified version - would need more context
        return pattern.confidence
    
    def _calculate_stream_significance(self, pattern, streams: List[AuditoryStream]) -> float:
        """Calculate significance based on stream membership"""
        
        # Find streams that overlap with this pattern
        overlapping_streams = []
        for stream in streams:
            if (pattern.start_time < stream.end_time and 
                pattern.end_time > stream.start_time):
                overlapping_streams.append(stream)
        
        if not overlapping_streams:
            return 0.0
        
        # Calculate average stream confidence
        stream_confidences = [stream.confidence for stream in overlapping_streams]
        return np.mean(stream_confidences)
    
    def _calculate_adaptation_level(self, pattern) -> float:
        """Calculate adaptation level based on recent exposure"""
        
        # Simplified adaptation model
        # In a real implementation, this would track recent pattern exposure
        return 0.0  # Placeholder

def main():
    """Test the perceptual significance filter"""
    
    print("🚀 Starting perceptual significance filter test...")
    
    analyzer = AuditorySceneAnalyzer()
    filter_system = PerceptualSignificanceFilter()
    
    # Test with sample audio file
    audio_file = "input_audio/Grab-a-hold.mp3"
    
    print(f"📁 Looking for audio file: {audio_file}")
    
    try:
        print("🎧 Starting auditory scene analysis...")
        streams = analyzer.analyze_auditory_scene(audio_file)
        
        print(f"\n🎧 Auditory Scene Analysis:")
        for stream in streams:
            print(f"   Stream {stream.stream_id}: {stream.stream_type} "
                  f"({stream.start_time:.1f}-{stream.end_time:.1f}s, "
                  f"conf: {stream.confidence:.3f})")
        
    except FileNotFoundError:
        print(f"❌ Audio file not found: {audio_file}")
        print("Please provide a valid audio file path")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
