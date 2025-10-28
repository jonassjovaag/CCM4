#!/usr/bin/env python3
"""
Auditory Scene Analyzer
Based on Bregman (1990): "Auditory Scene Analysis" and Pressnitzer et al. (2008)

This module implements auditory scene analysis principles:
- Stream segregation
- Perceptual grouping
- Multi-second adaptation
- Gestalt principles
"""

import numpy as np
import librosa
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import scipy.signal
from scipy.spatial.distance import pdist, squareform

@dataclass
class PerceptualGroup:
    """Represents a perceptually grouped set of sounds"""
    group_id: int
    start_time: float
    end_time: float
    members: List[int]  # Frame indices
    group_type: str  # 'melody', 'bass', 'harmony', 'percussion'
    cohesion_score: float
    gestalt_principles: Dict[str, float]

class GestaltAnalyzer:
    """
    Analyzes music using Gestalt principles of perceptual organization
    
    Based on research showing that humans group sounds based on:
    - Proximity (temporal and spectral)
    - Similarity (timbre and pitch)
    - Continuity (smooth transitions)
    - Closure (completion of patterns)
    """
    
    def __init__(self, 
                 proximity_threshold: float = 0.5,
                 similarity_threshold: float = 0.6,
                 continuity_threshold: float = 0.7):
        self.proximity_threshold = proximity_threshold
        self.similarity_threshold = similarity_threshold
        self.continuity_threshold = continuity_threshold
        
    def analyze_gestalt_principles(self, audio_file: str) -> List[PerceptualGroup]:
        """
        Analyze audio using Gestalt principles
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            List of perceptually grouped sounds
        """
        print(f"🎨 Analyzing Gestalt principles: {audio_file}")
        
        # Load audio and extract features
        y, sr = librosa.load(audio_file, sr=22050)
        features = self._extract_gestalt_features(y, sr)
        
        # Apply Gestalt principles
        groups = self._apply_gestalt_principles(features, sr)
        
        print(f"✅ Detected {len(groups)} perceptual groups")
        
        return groups
    
    def _extract_gestalt_features(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Extract features optimized for Gestalt analysis"""
        
        hop_length = 512
        frame_length = 2048
        
        # Chroma features for pitch analysis
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
        
        # Spectral features for timbre analysis
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
        spectral_rolloffs = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)[0]
        spectral_bandwidths = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)[0]
        
        # Temporal features
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        
        # Onset features for rhythmic analysis
        onset_strength = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        
        # Combine features
        features = np.vstack([
            chroma,
            spectral_centroids,
            spectral_rolloffs,
            spectral_bandwidths,
            rms,
            onset_strength
        ])
        
        return features.T  # Shape: (time_frames, features)
    
    def _apply_gestalt_principles(self, features: np.ndarray, sr: int) -> List[PerceptualGroup]:
        """Apply Gestalt principles to group sounds"""
        
        print("🔄 Applying Gestalt principles...")
        
        groups = []
        n_frames = len(features)
        hop_length = 512
        
        # Principle 1: Proximity - group temporally close sounds
        proximity_groups = self._apply_proximity_principle(features, sr)
        
        # Principle 2: Similarity - group spectrally similar sounds
        similarity_groups = self._apply_similarity_principle(features)
        
        # Principle 3: Continuity - group smoothly transitioning sounds
        continuity_groups = self._apply_continuity_principle(features)
        
        # Combine groups using intersection
        combined_groups = self._combine_gestalt_groups(
            proximity_groups, similarity_groups, continuity_groups
        )
        
        # Convert to perceptual groups
        for i, group_frames in enumerate(combined_groups):
            if len(group_frames) < 3:  # Minimum group size
                continue
            
            start_frame = min(group_frames)
            end_frame = max(group_frames)
            
            start_time = start_frame * hop_length / sr
            end_time = end_frame * hop_length / sr
            
            # Calculate group characteristics
            group_features = np.mean([features[frame] for frame in group_frames], axis=0)
            cohesion_score = self._calculate_group_cohesion(group_frames, features)
            
            # Determine group type
            group_type = self._determine_group_type(group_features)
            
            # Calculate Gestalt principle scores
            gestalt_principles = {
                'proximity': self._calculate_proximity_score(group_frames, sr),
                'similarity': self._calculate_similarity_score(group_frames, features),
                'continuity': self._calculate_continuity_score(group_frames, features)
            }
            
            group = PerceptualGroup(
                group_id=i,
                start_time=start_time,
                end_time=end_time,
                members=group_frames,
                group_type=group_type,
                cohesion_score=cohesion_score,
                gestalt_principles=gestalt_principles
            )
            
            groups.append(group)
        
        return groups
    
    def _apply_proximity_principle(self, features: np.ndarray, sr: int) -> List[List[int]]:
        """Apply proximity principle - group temporally close sounds"""
        
        groups = []
        n_frames = len(features)
        hop_length = 512
        
        # Convert proximity threshold to frames
        proximity_frames = int(self.proximity_threshold * sr / hop_length)
        
        current_group = []
        for i in range(n_frames):
            if not current_group:
                current_group.append(i)
            else:
                # Check if current frame is close to the last frame in current group
                if i - current_group[-1] <= proximity_frames:
                    current_group.append(i)
                else:
                    # Start new group
                    if len(current_group) >= 3:
                        groups.append(current_group)
                    current_group = [i]
        
        # Add final group
        if len(current_group) >= 3:
            groups.append(current_group)
        
        return groups
    
    def _apply_similarity_principle(self, features: np.ndarray) -> List[List[int]]:
        """Apply similarity principle - group spectrally similar sounds"""
        
        groups = []
        n_frames = len(features)
        
        # Calculate pairwise similarities
        similarities = 1 - pdist(features, metric='cosine')
        similarity_matrix = squareform(similarities)
        
        # Group frames with high similarity
        visited = set()
        for i in range(n_frames):
            if i in visited:
                continue
            
            current_group = [i]
            visited.add(i)
            
            # Find similar frames
            for j in range(i + 1, n_frames):
                if j in visited:
                    continue
                
                if similarity_matrix[i, j] > self.similarity_threshold:
                    current_group.append(j)
                    visited.add(j)
            
            if len(current_group) >= 3:
                groups.append(current_group)
        
        return groups
    
    def _apply_continuity_principle(self, features: np.ndarray) -> List[List[int]]:
        """Apply continuity principle - group smoothly transitioning sounds"""
        
        groups = []
        n_frames = len(features)
        
        # Calculate feature differences between consecutive frames
        feature_diffs = np.diff(features, axis=0)
        
        current_group = [0]
        for i in range(1, n_frames):
            # Calculate continuity score
            continuity_score = self._calculate_continuity_score([i-1, i], features)
            
            if continuity_score > self.continuity_threshold:
                current_group.append(i)
            else:
                # Start new group
                if len(current_group) >= 3:
                    groups.append(current_group)
                current_group = [i]
        
        # Add final group
        if len(current_group) >= 3:
            groups.append(current_group)
        
        return groups
    
    def _combine_gestalt_groups(self, proximity_groups: List[List[int]], 
                               similarity_groups: List[List[int]], 
                               continuity_groups: List[List[int]]) -> List[List[int]]:
        """Combine groups from different Gestalt principles"""
        
        # Find intersections between different grouping principles
        combined_groups = []
        
        for prox_group in proximity_groups:
            for sim_group in similarity_groups:
                # Find intersection
                intersection = list(set(prox_group) & set(sim_group))
                
                if len(intersection) >= 3:
                    # Check if this intersection also has continuity
                    for cont_group in continuity_groups:
                        if len(set(intersection) & set(cont_group)) >= 3:
                            combined_groups.append(intersection)
                            break
        
        return combined_groups
    
    def _calculate_group_cohesion(self, group_frames: List[int], features: np.ndarray) -> float:
        """Calculate cohesion score for a group"""
        
        if len(group_frames) < 2:
            return 0.0
        
        # Calculate pairwise similarities within group
        similarities = []
        for i in range(len(group_frames)):
            for j in range(i + 1, len(group_frames)):
                sim = 1 - np.linalg.norm(features[group_frames[i]] - features[group_frames[j]])
                similarities.append(sim)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _determine_group_type(self, group_features: np.ndarray) -> str:
        """Determine the type of a perceptual group"""
        
        # Simple classification based on feature characteristics
        # This is a simplified version - would need more sophisticated analysis
        
        # Check if it's percussion (high onset strength, broad spectrum)
        if len(group_features) > 12 and group_features[-1] > 0.7:  # High onset strength
            return 'percussion'
        
        # Check if it's bass (low spectral centroid)
        if len(group_features) > 12 and group_features[12] < 0.3:  # Low spectral centroid
            return 'bass'
        
        # Check if it's melody (moderate spectral centroid, high chroma variation)
        if len(group_features) > 12 and 0.3 < group_features[12] < 0.7:
            return 'melody'
        
        # Default to harmony
        return 'harmony'
    
    def _calculate_proximity_score(self, group_frames: List[int], sr: int) -> float:
        """Calculate proximity score for a group"""
        
        if len(group_frames) < 2:
            return 0.0
        
        hop_length = 512
        time_gaps = []
        
        for i in range(len(group_frames) - 1):
            gap = (group_frames[i + 1] - group_frames[i]) * hop_length / sr
            time_gaps.append(gap)
        
        # Lower gaps = higher proximity score
        avg_gap = np.mean(time_gaps)
        return max(0.0, 1.0 - avg_gap / 2.0)  # Normalize to 2 seconds
    
    def _calculate_similarity_score(self, group_frames: List[int], features: np.ndarray) -> float:
        """Calculate similarity score for a group"""
        
        if len(group_frames) < 2:
            return 0.0
        
        # Calculate pairwise similarities within group
        similarities = []
        for i in range(len(group_frames)):
            for j in range(i + 1, len(group_frames)):
                sim = 1 - np.linalg.norm(features[group_frames[i]] - features[group_frames[j]])
                similarities.append(sim)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _calculate_continuity_score(self, group_frames: List[int], features: np.ndarray) -> float:
        """Calculate continuity score for a group"""
        
        if len(group_frames) < 2:
            return 0.0
        
        # Calculate feature differences between consecutive frames
        continuity_scores = []
        for i in range(len(group_frames) - 1):
            diff = np.linalg.norm(features[group_frames[i + 1]] - features[group_frames[i]])
            continuity_scores.append(1.0 - diff)  # Lower diff = higher continuity
        
        return np.mean(continuity_scores) if continuity_scores else 0.0

def main():
    """Test the Gestalt analyzer"""
    
    analyzer = GestaltAnalyzer()
    
    # Test with sample audio file
    audio_file = "input_audio/Grab-a-hold.mp3"
    
    try:
        groups = analyzer.analyze_gestalt_principles(audio_file)
        
        print(f"\n🎨 Gestalt Analysis Results:")
        for group in groups:
            print(f"   Group {group.group_id}: {group.group_type} "
                  f"({group.start_time:.1f}-{group.end_time:.1f}s, "
                  f"cohesion: {group.cohesion_score:.3f})")
            
            print(f"     Gestalt principles:")
            for principle, score in group.gestalt_principles.items():
                print(f"       {principle}: {score:.3f}")
        
    except FileNotFoundError:
        print(f"❌ Audio file not found: {audio_file}")
        print("Please provide a valid audio file path")

if __name__ == "__main__":
    main()
