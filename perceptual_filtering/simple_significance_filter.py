#!/usr/bin/env python3
"""
Simplified Perceptual Significance Filter
Based on Pressnitzer et al. (2008): "Perceptual organization of sound begins in the auditory periphery"

This is a simplified version that actually works!
"""

import numpy as np
import librosa
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class AuditoryStream:
    """Represents an auditory stream (grouped sounds)"""
    stream_id: int
    start_time: float
    end_time: float
    features: np.ndarray
    confidence: float
    stream_type: str  # 'melody', 'bass', 'harmony', 'percussion'

class SimpleAuditorySceneAnalyzer:
    """
    Simplified auditory scene analyzer that actually works
    """
    
    def __init__(self):
        self.stream_similarity_threshold = 0.6
        
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
        print(f"✅ Audio loaded: {len(y)} samples, {sr} Hz")
        
        # Extract simplified features
        features = self._extract_simple_features(y, sr)
        print(f"✅ Features extracted: {features.shape}")
        
        # Perform simple stream segregation
        streams = self._simple_stream_segregation(features, sr)
        print(f"✅ Detected {len(streams)} auditory streams")
        
        return streams
    
    def _extract_simple_features(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Extract simplified features for speed"""
        
        hop_length = 1024  # Larger hop for speed
        
        # Chroma features for harmonic analysis
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
        
        # Spectral features for timbral analysis
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
        
        # Temporal features
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        
        # Combine features - ensure all arrays have the same length
        min_length = min(len(chroma[0]), len(spectral_centroids), len(rms))
        
        features = np.vstack([
            chroma[:, :min_length],
            spectral_centroids[:min_length],
            rms[:min_length]
        ])
        
        return features.T  # Shape: (time_frames, features)
    
    def _simple_stream_segregation(self, features: np.ndarray, sr: int) -> List[AuditoryStream]:
        """Perform simple stream segregation using basic clustering"""
        
        print("🔄 Performing simple stream segregation...")
        
        # Use simple k-means clustering instead of DBSCAN
        from sklearn.cluster import KMeans
        
        # Limit features for speed
        max_features = 1000
        if len(features) > max_features:
            # Sample features evenly
            indices = np.linspace(0, len(features)-1, max_features, dtype=int)
            features = features[indices]
        
        print(f"📊 Using {len(features)} features for clustering")
        
        # Simple k-means clustering
        n_clusters = min(5, len(features) // 10)  # Adaptive number of clusters
        if n_clusters < 2:
            n_clusters = 2
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features)
        
        print(f"✅ K-means completed: {n_clusters} clusters")
        
        # Convert clusters to streams
        streams = []
        hop_length = 1024
        
        for cluster_id in range(n_clusters):
            # Get frames belonging to this cluster
            cluster_frames = np.where(cluster_labels == cluster_id)[0]
            
            if len(cluster_frames) < 3:  # Minimum stream size
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
            
            # Determine stream type
            stream_type = self._determine_stream_type(stream_features)
            
            stream = AuditoryStream(
                stream_id=cluster_id,
                start_time=start_time,
                end_time=end_time,
                features=stream_features,
                confidence=confidence,
                stream_type=stream_type
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
                sim = 1 - np.linalg.norm(features[cluster_frames[i]] - features[cluster_frames[j]])
                similarities.append(sim)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _determine_stream_type(self, stream_features: np.ndarray) -> str:
        """Determine the type of a stream based on features"""
        
        # Simple classification based on feature characteristics
        if len(stream_features) > 12:
            # Check spectral centroid (feature index 12)
            spectral_centroid = stream_features[12]
            
            if spectral_centroid < 0.3:
                return 'bass'
            elif spectral_centroid > 0.7:
                return 'melody'
            else:
                return 'harmony'
        
        return 'unknown'

def main():
    """Test the simplified perceptual significance filter"""
    
    print("🚀 Starting simplified perceptual significance filter test...")
    
    analyzer = SimpleAuditorySceneAnalyzer()
    
    # Test with sample audio file
    audio_file = "input_audio/Grab-a-hold.mp3"
    
    print(f"📁 Looking for audio file: {audio_file}")
    
    try:
        print("🎧 Starting auditory scene analysis...")
        streams = analyzer.analyze_auditory_scene(audio_file)
        
        print(f"\n🎧 Auditory Scene Analysis Results:")
        for stream in streams:
            print(f"   Stream {stream.stream_id}: {stream.stream_type} "
                  f"({stream.start_time:.1f}-{stream.end_time:.1f}s, "
                  f"conf: {stream.confidence:.3f})")
        
        print(f"\n✅ Analysis complete! Found {len(streams)} streams.")
        
    except FileNotFoundError:
        print(f"❌ Audio file not found: {audio_file}")
        print("Please provide a valid audio file path")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
