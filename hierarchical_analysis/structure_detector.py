#!/usr/bin/env python3
"""
Structure Detector
Based on de Berardinis et al. (2020): "Unveiling the Hierarchical Structure of Music by Multi-Resolution Community Detection"

This module implements the MSCOM algorithm for detecting musical structure using graph theory
and community detection algorithms.
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity
import librosa

@dataclass
class StructuralBoundary:
    """Represents a structural boundary in music"""
    time: float
    confidence: float
    boundary_type: str  # 'section', 'phrase', 'measure'
    features: np.ndarray

@dataclass
class StructuralCommunity:
    """Represents a structural community (section/phrase)"""
    community_id: int
    start_time: float
    end_time: float
    members: List[int]  # Frame indices
    features: np.ndarray
    stability: float

class MSCOMStructureDetector:
    """
    Multi-Resolution Community Detection for Musical Structure Analysis
    
    Based on the MSCOM algorithm that uses graph theory to detect hierarchical
    musical structures through community detection at multiple resolutions.
    """
    
    def __init__(self, 
                 min_community_size: int = 10,
                 resolution_range: Tuple[float, float] = (0.1, 2.0),
                 similarity_threshold: float = 0.6):
        self.min_community_size = min_community_size
        self.resolution_range = resolution_range
        self.similarity_threshold = similarity_threshold
        
    def detect_structure(self, audio_file: str) -> Dict[str, List[StructuralCommunity]]:
        """
        Detect hierarchical musical structure using MSCOM algorithm
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            Dictionary containing communities at different resolution levels
        """
        print(f"🔍 Detecting musical structure using MSCOM algorithm: {audio_file}")
        
        # Load audio and extract features
        y, sr = librosa.load(audio_file, sr=22050)
        features = self._extract_structural_features(y, sr)
        
        # Build similarity graph
        graph = self._build_similarity_graph(features)
        
        # Detect communities at multiple resolutions
        communities = self._detect_multi_resolution_communities(graph, features)
        
        # Convert to structural communities
        structural_communities = self._convert_to_structural_communities(
            communities, features, sr
        )
        
        print(f"✅ Detected {len(structural_communities)} structural communities")
        
        return structural_communities
    
    def _extract_structural_features(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Extract features optimized for structural analysis"""
        
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
        
        # MFCC features for timbral analysis
        mfccs = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)
        
        # Combine features
        features = np.vstack([
            chroma,
            spectral_centroids,
            spectral_rolloffs,
            spectral_bandwidths,
            rms,
            mfccs
        ])
        
        return features.T  # Shape: (time_frames, features)
    
    def _build_similarity_graph(self, features: np.ndarray) -> nx.Graph:
        """Build similarity graph based on feature similarity"""
        
        print("🔗 Building similarity graph...")
        
        graph = nx.Graph()
        n_frames = len(features)
        
        # Add nodes
        for i in range(n_frames):
            graph.add_node(i, features=features[i])
        
        # Add edges based on similarity
        for i in range(n_frames):
            for j in range(i + 1, min(i + 50, n_frames)):  # Local neighborhood
                similarity = cosine_similarity([features[i]], [features[j]])[0][0]
                
                if similarity > self.similarity_threshold:
                    graph.add_edge(i, j, weight=similarity)
        
        print(f"📊 Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        
        return graph
    
    def _detect_multi_resolution_communities(self, graph: nx.Graph, 
                                           features: np.ndarray) -> Dict[str, List[int]]:
        """Detect communities at multiple resolution levels"""
        
        print("🎯 Detecting multi-resolution communities...")
        
        communities = {}
        
        # Resolution levels for different structural scales
        resolution_levels = {
            'section': 0.1,    # Large communities (sections)
            'phrase': 0.5,     # Medium communities (phrases)
            'measure': 1.0     # Small communities (measures)
        }
        
        for level_name, resolution in resolution_levels.items():
            print(f"   Analyzing {level_name} level (resolution: {resolution})")
            
            # Use Leiden algorithm for community detection
            try:
                import leidenalg
                partition = leidenalg.find_partition(
                    graph, 
                    leidenalg.ModularityVertexPartition,
                    resolution_parameter=resolution
                )
                
                level_communities = []
                for community in partition.communities:
                    if len(community) >= self.min_community_size:
                        level_communities.append(list(community))
                
                communities[level_name] = level_communities
                
            except ImportError:
                # Fallback to NetworkX community detection
                communities[level_name] = self._fallback_community_detection(
                    graph, resolution
                )
        
        return communities
    
    def _fallback_community_detection(self, graph: nx.Graph, resolution: float) -> List[List[int]]:
        """Fallback community detection using NetworkX"""
        
        try:
            # Use Louvain algorithm
            communities = nx.community.louvain_communities(graph, resolution=resolution)
            
            # Filter by minimum size
            filtered_communities = []
            for community in communities:
                if len(community) >= self.min_community_size:
                    filtered_communities.append(list(community))
            
            return filtered_communities
            
        except Exception as e:
            print(f"⚠️  Community detection failed: {e}")
            return []
    
    def _convert_to_structural_communities(self, communities: Dict[str, List[int]], 
                                         features: np.ndarray, sr: int) -> Dict[str, List[StructuralCommunity]]:
        """Convert detected communities to structural communities"""
        
        structural_communities = {}
        hop_length = 512
        
        for level_name, level_communities in communities.items():
            structural_level = []
            
            for i, community in enumerate(level_communities):
                if not community:
                    continue
                
                # Calculate temporal boundaries
                start_frame = min(community)
                end_frame = max(community)
                
                start_time = start_frame * hop_length / sr
                end_time = end_frame * hop_length / sr
                
                # Calculate community features
                community_features = np.mean([features[frame] for frame in community], axis=0)
                
                # Calculate stability (internal similarity)
                stability = self._calculate_community_stability(community, features)
                
                structural_community = StructuralCommunity(
                    community_id=i,
                    start_time=start_time,
                    end_time=end_time,
                    members=community,
                    features=community_features,
                    stability=stability
                )
                
                structural_level.append(structural_community)
            
            structural_communities[level_name] = structural_level
        
        return structural_communities
    
    def _calculate_community_stability(self, community: List[int], 
                                      features: np.ndarray) -> float:
        """Calculate stability of a community based on internal similarity"""
        
        if len(community) < 2:
            return 0.0
        
        # Calculate pairwise similarities within community
        similarities = []
        for i in range(len(community)):
            for j in range(i + 1, len(community)):
                sim = cosine_similarity(
                    [features[community[i]]], 
                    [features[community[j]]]
                )[0][0]
                similarities.append(sim)
        
        return np.mean(similarities) if similarities else 0.0
    
    def detect_boundaries(self, structural_communities: Dict[str, List[StructuralCommunity]]) -> List[StructuralBoundary]:
        """Detect structural boundaries between communities"""
        
        boundaries = []
        
        for level_name, communities in structural_communities.items():
            for i in range(len(communities) - 1):
                current_community = communities[i]
                next_community = communities[i + 1]
                
                # Boundary is at the transition between communities
                boundary_time = current_community.end_time
                
                # Calculate boundary confidence based on feature difference
                feature_diff = np.linalg.norm(
                    current_community.features - next_community.features
                )
                confidence = min(feature_diff / 10.0, 1.0)  # Normalize
                
                boundary = StructuralBoundary(
                    time=boundary_time,
                    confidence=confidence,
                    boundary_type=level_name,
                    features=next_community.features - current_community.features
                )
                
                boundaries.append(boundary)
        
        # Sort by time
        boundaries.sort(key=lambda x: x.time)
        
        return boundaries
    
    def analyze_structure_coherence(self, structural_communities: Dict[str, List[StructuralCommunity]]) -> Dict[str, float]:
        """Analyze coherence of detected structure"""
        
        coherence_scores = {}
        
        for level_name, communities in structural_communities.items():
            if len(communities) < 2:
                coherence_scores[level_name] = 0.0
                continue
            
            # Calculate average stability
            stabilities = [c.stability for c in communities]
            coherence_scores[level_name] = np.mean(stabilities)
        
        return coherence_scores

def main():
    """Test the structure detector"""
    
    detector = MSCOMStructureDetector()
    
    # Test with sample audio file
    audio_file = "input_audio/Grab-a-hold.mp3"
    
    try:
        communities = detector.detect_structure(audio_file)
        boundaries = detector.detect_boundaries(communities)
        coherence = detector.analyze_structure_coherence(communities)
        
        print(f"\n🎵 Structure Detection Results:")
        for level, comms in communities.items():
            print(f"📊 {level.capitalize()}: {len(comms)} communities")
        
        print(f"\n🎯 Detected {len(boundaries)} structural boundaries")
        
        print(f"\n📈 Structure Coherence:")
        for level, score in coherence.items():
            print(f"   {level.capitalize()}: {score:.3f}")
        
    except FileNotFoundError:
        print(f"❌ Audio file not found: {audio_file}")
        print("Please provide a valid audio file path")

if __name__ == "__main__":
    main()
