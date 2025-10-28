#!/usr/bin/env python3
"""
Fixed Multi-Timescale Analysis Engine
Properly detects musical patterns at different timescales

This fixed version addresses the issues with pattern detection:
- Lower similarity thresholds
- Better musical feature extraction
- More appropriate time windows
- Improved pattern detection logic
"""

import numpy as np
import librosa
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

@dataclass
class MusicalPattern:
    """Represents a musical pattern at a specific timescale"""
    pattern_id: str
    timescale: str  # 'measure', 'phrase', 'section'
    start_time: float
    end_time: float
    features: np.ndarray
    confidence: float
    repetitions: int
    variations: List[str]

@dataclass
class HierarchicalStructure:
    """Represents the hierarchical structure of a musical piece"""
    sections: List[MusicalPattern]
    phrases: List[MusicalPattern]
    measures: List[MusicalPattern]
    relationships: Dict[str, List[str]]  # parent -> children

class FixedMultiTimescaleAnalyzer:
    """
    Fixed multi-timescale analyzer that properly detects musical patterns
    
    Key fixes:
    - Lower similarity thresholds (0.3 instead of 0.7)
    - Better musical feature extraction
    - More appropriate time windows
    - Improved pattern detection logic
    """
    
    def __init__(self, 
                 measure_window: float = 1.5,      # Even shorter for more measures
                 phrase_window: float = 6.0,       # Shorter for more phrases
                 section_window: float = 30.0,     # Keep sections the same
                 sample_rate: int = 22050):
        self.measure_window = measure_window
        self.phrase_window = phrase_window
        self.section_window = section_window
        self.sample_rate = sample_rate
        
        # Much lower thresholds for better detection
        self.similarity_threshold = 0.2  # Even lower for more patterns
        self.min_repetitions = 1         # Was 2
        self.min_pattern_length = 0.3   # seconds - shorter minimum
        
    def analyze_hierarchical_structure(self, audio_file: str) -> HierarchicalStructure:
        """
        Analyze audio file at multiple timescales with fixed detection
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            HierarchicalStructure containing patterns at all levels
        """
        print(f"🎵 Analyzing hierarchical structure of: {audio_file}")
        
        # Load audio
        y, sr = librosa.load(audio_file, sr=self.sample_rate)
        duration = len(y) / sr
        
        print(f"📊 Audio duration: {duration:.2f} seconds")
        
        # Extract features optimized for pattern detection
        features = self._extract_musical_features(y, sr)
        
        # Analyze at each timescale with fixed logic
        measures = self._detect_measure_patterns_fixed(features, duration, sr)
        phrases = self._detect_phrase_patterns_fixed(features, duration, sr)
        sections = self._detect_section_patterns_fixed(features, duration, sr)
        
        # Build hierarchical relationships
        relationships = self._build_hierarchical_relationships(sections, phrases, measures)
        
        structure = HierarchicalStructure(
            sections=sections,
            phrases=phrases,
            measures=measures,
            relationships=relationships
        )
        
        print(f"✅ Found {len(sections)} sections, {len(phrases)} phrases, {len(measures)} measures")
        
        return structure
    
    def _extract_musical_features(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Extract features optimized for musical pattern detection"""
        
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
    
    def _detect_measure_patterns_fixed(self, features: np.ndarray, duration: float, sr: int) -> List[MusicalPattern]:
        """Detect measure-level patterns with fixed logic"""
        
        patterns = []
        hop_length = 512
        
        # Use clustering to find measure patterns
        window_size = max(1, int(self.measure_window * sr / hop_length))
        
        # Create windows
        windows = []
        window_times = []
        
        for i in range(0, len(features) - window_size, window_size // 2):
            window_features = features[i:i + window_size]
            window_mean = np.mean(window_features, axis=0)
            windows.append(window_mean)
            window_times.append((i * hop_length / sr, (i + window_size) * hop_length / sr))
        
        if len(windows) < 2:
            return patterns
        
        # Use K-means clustering to find similar measures
        # Scale with song length - no artificial caps!
        n_clusters = max(3, len(windows) // 4)  # More aggressive scaling
        # Remove artificial upper limit to allow proper scaling
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(windows)
        
        # Convert clusters to patterns
        for cluster_id in range(n_clusters):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            
            if len(cluster_indices) >= self.min_repetitions:
                # Calculate pattern characteristics
                cluster_windows = [windows[i] for i in cluster_indices]
                cluster_times = [window_times[i] for i in cluster_indices]
                
                # Pattern time span
                start_time = min([t[0] for t in cluster_times])
                end_time = max([t[1] for t in cluster_times])
                
                # Pattern features (mean of cluster)
                pattern_features = np.mean(cluster_windows, axis=0)
                
                # Confidence based on cluster cohesion
                confidence = self._calculate_cluster_cohesion(cluster_windows)
                
                pattern = MusicalPattern(
                    pattern_id=f"measure_cluster_{cluster_id}",
                    timescale='measure',
                    start_time=start_time,
                    end_time=end_time,
                    features=pattern_features,
                    confidence=confidence,
                    repetitions=len(cluster_indices),
                    variations=[]
                )
                
                patterns.append(pattern)
        
        return patterns
    
    def _detect_phrase_patterns_fixed(self, features: np.ndarray, duration: float, sr: int) -> List[MusicalPattern]:
        """Detect phrase-level patterns with fixed logic"""
        
        patterns = []
        hop_length = 512
        
        # Use larger windows for phrases
        window_size = max(1, int(self.phrase_window * sr / hop_length))
        
        # Create windows
        windows = []
        window_times = []
        
        for i in range(0, len(features) - window_size, window_size // 2):
            window_features = features[i:i + window_size]
            window_mean = np.mean(window_features, axis=0)
            windows.append(window_mean)
            window_times.append((i * hop_length / sr, (i + window_size) * hop_length / sr))
        
        if len(windows) < 2:
            return patterns
        
        # Use K-means clustering for phrases
        # Scale with song length - no artificial caps!
        n_clusters = max(3, len(windows) // 3)  # More aggressive scaling
        # Remove artificial upper limit to allow proper scaling
        
        # Ensure we don't have more clusters than samples
        n_clusters = min(n_clusters, len(windows))
        
        # If we have too few samples, use a simpler approach
        if len(windows) < 3:
            # For very short audio, create simple patterns
            patterns = []
            for i, (window, window_time) in enumerate(zip(windows, window_times)):
                pattern = {
                    'start_time': window_time[0],
                    'end_time': window_time[1],
                    'duration': window_time[1] - window_time[0],
                    'pattern_type': 'phrase',
                    'confidence': 0.5,
                    'repetitions': 1,
                    'features': window.tolist()
                }
                patterns.append(pattern)
            return patterns
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(windows)
        
        # Convert clusters to patterns
        for cluster_id in range(n_clusters):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            
            if len(cluster_indices) >= self.min_repetitions:
                cluster_windows = [windows[i] for i in cluster_indices]
                cluster_times = [window_times[i] for i in cluster_indices]
                
                start_time = min([t[0] for t in cluster_times])
                end_time = max([t[1] for t in cluster_times])
                
                pattern_features = np.mean(cluster_windows, axis=0)
                confidence = self._calculate_cluster_cohesion(cluster_windows)
                
                pattern = MusicalPattern(
                    pattern_id=f"phrase_cluster_{cluster_id}",
                    timescale='phrase',
                    start_time=start_time,
                    end_time=end_time,
                    features=pattern_features,
                    confidence=confidence,
                    repetitions=len(cluster_indices),
                    variations=[]
                )
                
                patterns.append(pattern)
        
        return patterns
    
    def _detect_section_patterns_fixed(self, features: np.ndarray, duration: float, sr: int) -> List[MusicalPattern]:
        """Detect section-level patterns with fixed logic"""
        
        patterns = []
        hop_length = 512
        
        # Use even larger windows for sections
        window_size = max(1, int(self.section_window * sr / hop_length))
        
        # Create windows
        windows = []
        window_times = []
        
        for i in range(0, len(features) - window_size, window_size // 2):
            window_features = features[i:i + window_size]
            window_mean = np.mean(window_features, axis=0)
            windows.append(window_mean)
            window_times.append((i * hop_length / sr, (i + window_size) * hop_length / sr))
        
        if len(windows) < 2:
            return patterns
        
        # Use K-means clustering for sections
        # Scale with song length - no artificial caps!
        n_clusters = max(2, len(windows) // 5)  # More aggressive scaling
        # Remove artificial upper limit to allow proper scaling
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(windows)
        
        # Convert clusters to patterns
        for cluster_id in range(n_clusters):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            
            if len(cluster_indices) >= self.min_repetitions:
                cluster_windows = [windows[i] for i in cluster_indices]
                cluster_times = [window_times[i] for i in cluster_indices]
                
                start_time = min([t[0] for t in cluster_times])
                end_time = max([t[1] for t in cluster_times])
                
                pattern_features = np.mean(cluster_windows, axis=0)
                confidence = self._calculate_cluster_cohesion(cluster_windows)
                
                pattern = MusicalPattern(
                    pattern_id=f"section_cluster_{cluster_id}",
                    timescale='section',
                    start_time=start_time,
                    end_time=end_time,
                    features=pattern_features,
                    confidence=confidence,
                    repetitions=len(cluster_indices),
                    variations=[]
                )
                
                patterns.append(pattern)
        
        return patterns
    
    def _calculate_cluster_cohesion(self, cluster_windows: List[np.ndarray]) -> float:
        """Calculate cohesion score for a cluster"""
        
        if len(cluster_windows) < 2:
            return 1.0
        
        # Calculate pairwise similarities within cluster
        similarities = []
        for i in range(len(cluster_windows)):
            for j in range(i + 1, len(cluster_windows)):
                sim = cosine_similarity([cluster_windows[i]], [cluster_windows[j]])[0][0]
                similarities.append(sim)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _build_hierarchical_relationships(self, sections: List[MusicalPattern], 
                                       phrases: List[MusicalPattern], 
                                       measures: List[MusicalPattern]) -> Dict[str, List[str]]:
        """Build hierarchical relationships between patterns"""
        
        relationships = defaultdict(list)
        
        # Map measures to phrases
        for phrase in phrases:
            for measure in measures:
                if (measure.start_time >= phrase.start_time and 
                    measure.end_time <= phrase.end_time):
                    relationships[phrase.pattern_id].append(measure.pattern_id)
        
        # Map phrases to sections
        for section in sections:
            for phrase in phrases:
                if (phrase.start_time >= section.start_time and 
                    phrase.end_time <= section.end_time):
                    relationships[section.pattern_id].append(phrase.pattern_id)
        
        return dict(relationships)

def main():
    """Test the fixed multi-timescale analyzer"""
    
    analyzer = FixedMultiTimescaleAnalyzer()
    
    # Test with short audio file
    audio_file = "input_audio/short_grab.mp3"
    
    try:
        structure = analyzer.analyze_hierarchical_structure(audio_file)
        
        print(f"\n🎵 Fixed Hierarchical Structure Analysis:")
        print(f"📊 Sections: {len(structure.sections)}")
        print(f"📊 Phrases: {len(structure.phrases)}")
        print(f"📊 Measures: {len(structure.measures)}")
        
        # Print some examples
        if structure.measures:
            print(f"\n🎼 Measure Examples:")
            for i, measure in enumerate(structure.measures[:3]):
                print(f"   {i+1}. {measure.pattern_id}: "
                      f"{measure.start_time:.2f}-{measure.end_time:.2f}s "
                      f"(conf: {measure.confidence:.3f}, reps: {measure.repetitions})")
        
        if structure.phrases:
            print(f"\n🎵 Phrase Examples:")
            for i, phrase in enumerate(structure.phrases[:3]):
                print(f"   {i+1}. {phrase.pattern_id}: "
                      f"{phrase.start_time:.2f}-{phrase.end_time:.2f}s "
                      f"(conf: {phrase.confidence:.3f}, reps: {phrase.repetitions})")
        
        if structure.sections:
            print(f"\n🎶 Section Examples:")
            for i, section in enumerate(structure.sections[:3]):
                print(f"   {i+1}. {section.pattern_id}: "
                      f"{section.start_time:.2f}-{section.end_time:.2f}s "
                      f"(conf: {section.confidence:.3f}, reps: {section.repetitions})")
        
    except FileNotFoundError:
        print(f"❌ Audio file not found: {audio_file}")
        print("Please provide a valid audio file path")

if __name__ == "__main__":
    main()
