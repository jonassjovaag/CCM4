#!/usr/bin/env python3
"""
Multi-Timescale Analysis Engine
Based on Farbood et al. (2015): "The neural processing of hierarchical structure in music and speech at different timescales"

This module implements hierarchical music analysis that mirrors human neural processing:
- Level 1: Measure-level patterns (6 seconds)
- Level 2: Phrase-level patterns (30-60 seconds)  
- Level 3: Section-level patterns (minutes)
"""

import numpy as np
import librosa
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

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

class MultiTimescaleAnalyzer:
    """
    Analyzes music at multiple timescales based on human neural processing
    
    Based on research showing that different brain regions process music at:
    - Measure level: ~6 seconds
    - Phrase level: 30-60 seconds
    - Section level: minutes
    """
    
    def __init__(self, 
                 measure_window: float = 6.0,
                 phrase_window: float = 30.0,
                 section_window: float = 120.0,
                 sample_rate: int = 22050):
        self.measure_window = measure_window
        self.phrase_window = phrase_window
        self.section_window = section_window
        self.sample_rate = sample_rate
        
        # Pattern detection thresholds
        self.similarity_threshold = 0.7
        self.min_repetitions = 2
        self.min_pattern_length = 0.5  # seconds
        
    def analyze_hierarchical_structure(self, audio_file: str) -> HierarchicalStructure:
        """
        Analyze audio file at multiple timescales
        
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
        
        # Extract features
        features = self._extract_multi_scale_features(y, sr)
        
        # Analyze at each timescale
        measures = self._detect_measure_patterns(features, duration)
        phrases = self._detect_phrase_patterns(measures, duration)
        sections = self._detect_section_patterns(phrases, duration)
        
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
    
    def _extract_multi_scale_features(self, y: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """Extract features optimized for different timescales"""
        
        features = {}
        
        # Measure-level features (high resolution)
        features['measure'] = self._extract_measure_features(y, sr)
        
        # Phrase-level features (medium resolution)
        features['phrase'] = self._extract_phrase_features(y, sr)
        
        # Section-level features (low resolution)
        features['section'] = self._extract_section_features(y, sr)
        
        return features
    
    def _extract_measure_features(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Extract features for measure-level analysis (6-second windows)"""
        
        # High-resolution features for detailed analysis
        hop_length = 512
        frame_length = 2048
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
        spectral_rolloffs = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)[0]
        spectral_bandwidths = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)[0]
        
        # Harmonic features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        
        # Temporal features
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
        
        # Combine features
        features = np.vstack([
            spectral_centroids,
            spectral_rolloffs,
            spectral_bandwidths,
            chroma.mean(axis=0),
            tonnetz.mean(axis=0),
            rms
        ])
        
        return features.T  # Shape: (time_frames, features)
    
    def _extract_phrase_features(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Extract features for phrase-level analysis (30-second windows)"""
        
        # Medium-resolution features for phrase analysis
        hop_length = 1024
        frame_length = 4096
        
        # Harmonic progression features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
        chroma_mean = chroma.mean(axis=1)  # Average chroma over time
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
        spectral_rolloffs = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)[0]
        
        # Temporal features
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        
        # Combine features - ensure all arrays have the same length
        min_length = min(len(chroma_mean), len(spectral_centroids), len(spectral_rolloffs), len(rms))
        
        features = np.vstack([
            chroma_mean[:min_length],
            spectral_centroids[:min_length],
            spectral_rolloffs[:min_length],
            rms[:min_length]
        ])
        
        return features.T
    
    def _extract_section_features(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Extract features for section-level analysis (2-minute windows)"""
        
        # Low-resolution features for section analysis
        hop_length = 2048
        frame_length = 8192
        
        # Overall spectral characteristics
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
        spectral_rolloffs = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)[0]
        
        # Harmonic features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
        chroma_mean = chroma.mean(axis=1)
        
        # Energy features
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        
        # Combine features - ensure all arrays have the same length
        min_length = min(len(chroma_mean), len(spectral_centroids), len(spectral_rolloffs), len(rms))
        
        features = np.vstack([
            chroma_mean[:min_length],
            spectral_centroids[:min_length],
            spectral_rolloffs[:min_length],
            rms[:min_length]
        ])
        
        return features.T
    
    def _detect_measure_patterns(self, features: Dict[str, np.ndarray], duration: float) -> List[MusicalPattern]:
        """Detect measure-level patterns (6-second windows)"""
        
        measure_features = features['measure']
        patterns = []
        
        # Sliding window analysis
        window_size = int(self.measure_window * self.sample_rate / 512)  # Convert to frames
        
        for i in range(0, len(measure_features) - window_size, window_size // 2):
            window_features = measure_features[i:i + window_size]
            
            # Calculate pattern characteristics
            pattern_id = f"measure_{i}"
            start_time = i * 512 / self.sample_rate
            end_time = start_time + self.measure_window
            
            # Feature vector (mean of window)
            feature_vector = np.mean(window_features, axis=0)
            
            # Find similar patterns
            repetitions = self._count_pattern_repetitions(
                measure_features, feature_vector, i, window_size
            )
            
            if repetitions >= self.min_repetitions:
                pattern = MusicalPattern(
                    pattern_id=pattern_id,
                    timescale='measure',
                    start_time=start_time,
                    end_time=end_time,
                    features=feature_vector,
                    confidence=min(repetitions / 5.0, 1.0),
                    repetitions=repetitions,
                    variations=[]
                )
                patterns.append(pattern)
        
        return patterns
    
    def _detect_phrase_patterns(self, measures: List[MusicalPattern], duration: float) -> List[MusicalPattern]:
        """Detect phrase-level patterns (30-second windows)"""
        
        patterns = []
        
        # Group measures into phrases
        phrase_groups = self._group_measures_into_phrases(measures)
        
        for i, phrase_group in enumerate(phrase_groups):
            if len(phrase_group) < 3:  # Minimum phrase length
                continue
                
            # Calculate phrase characteristics
            start_time = phrase_group[0].start_time
            end_time = phrase_group[-1].end_time
            
            # Feature vector (combination of measure features)
            phrase_features = np.mean([m.features for m in phrase_group], axis=0)
            
            # Find similar phrases
            repetitions = self._count_phrase_repetitions(phrase_groups, phrase_features, i)
            
            if repetitions >= self.min_repetitions:
                pattern = MusicalPattern(
                    pattern_id=f"phrase_{i}",
                    timescale='phrase',
                    start_time=start_time,
                    end_time=end_time,
                    features=phrase_features,
                    confidence=min(repetitions / 3.0, 1.0),
                    repetitions=repetitions,
                    variations=[]
                )
                patterns.append(pattern)
        
        return patterns
    
    def _detect_section_patterns(self, phrases: List[MusicalPattern], duration: float) -> List[MusicalPattern]:
        """Detect section-level patterns (2-minute windows)"""
        
        patterns = []
        
        # Group phrases into sections
        section_groups = self._group_phrases_into_sections(phrases)
        
        for i, section_group in enumerate(section_groups):
            if len(section_group) < 2:  # Minimum section length
                continue
                
            # Calculate section characteristics
            start_time = section_group[0].start_time
            end_time = section_group[-1].end_time
            
            # Feature vector (combination of phrase features)
            section_features = np.mean([p.features for p in section_group], axis=0)
            
            # Find similar sections
            repetitions = self._count_section_repetitions(section_groups, section_features, i)
            
            if repetitions >= self.min_repetitions:
                pattern = MusicalPattern(
                    pattern_id=f"section_{i}",
                    timescale='section',
                    start_time=start_time,
                    end_time=end_time,
                    features=section_features,
                    confidence=min(repetitions / 2.0, 1.0),
                    repetitions=repetitions,
                    variations=[]
                )
                patterns.append(pattern)
        
        return patterns
    
    def _group_measures_into_phrases(self, measures: List[MusicalPattern]) -> List[List[MusicalPattern]]:
        """Group measures into phrases based on musical structure"""
        
        phrases = []
        current_phrase = []
        
        for measure in measures:
            if not current_phrase:
                current_phrase.append(measure)
            else:
                # Check if measure belongs to current phrase
                time_gap = measure.start_time - current_phrase[-1].end_time
                
                if time_gap < 2.0:  # Less than 2 seconds gap
                    current_phrase.append(measure)
                else:
                    # Start new phrase
                    if len(current_phrase) >= 3:
                        phrases.append(current_phrase)
                    current_phrase = [measure]
        
        # Add final phrase
        if len(current_phrase) >= 3:
            phrases.append(current_phrase)
        
        return phrases
    
    def _group_phrases_into_sections(self, phrases: List[MusicalPattern]) -> List[List[MusicalPattern]]:
        """Group phrases into sections based on musical structure"""
        
        sections = []
        current_section = []
        
        for phrase in phrases:
            if not current_section:
                current_section.append(phrase)
            else:
                # Check if phrase belongs to current section
                time_gap = phrase.start_time - current_section[-1].end_time
                
                if time_gap < 10.0:  # Less than 10 seconds gap
                    current_section.append(phrase)
                else:
                    # Start new section
                    if len(current_section) >= 2:
                        sections.append(current_section)
                    current_section = [phrase]
        
        # Add final section
        if len(current_section) >= 2:
            sections.append(current_section)
        
        return sections
    
    def _count_pattern_repetitions(self, features: np.ndarray, pattern_features: np.ndarray, 
                                 start_idx: int, window_size: int) -> int:
        """Count repetitions of a pattern"""
        
        repetitions = 0
        similarity_threshold = self.similarity_threshold
        
        for i in range(0, len(features) - window_size, window_size // 2):
            if i == start_idx:
                continue
                
            window_features = features[i:i + window_size]
            window_mean = np.mean(window_features, axis=0)
            
            # Calculate similarity
            similarity = cosine_similarity([pattern_features], [window_mean])[0][0]
            
            if similarity > similarity_threshold:
                repetitions += 1
        
        return repetitions
    
    def _count_phrase_repetitions(self, phrase_groups: List[List[MusicalPattern]], 
                                phrase_features: np.ndarray, start_idx: int) -> int:
        """Count repetitions of a phrase"""
        
        repetitions = 0
        similarity_threshold = self.similarity_threshold
        
        for i, phrase_group in enumerate(phrase_groups):
            if i == start_idx:
                continue
                
            group_features = np.mean([p.features for p in phrase_group], axis=0)
            similarity = cosine_similarity([phrase_features], [group_features])[0][0]
            
            if similarity > similarity_threshold:
                repetitions += 1
        
        return repetitions
    
    def _count_section_repetitions(self, section_groups: List[List[MusicalPattern]], 
                                 section_features: np.ndarray, start_idx: int) -> int:
        """Count repetitions of a section"""
        
        repetitions = 0
        similarity_threshold = self.similarity_threshold
        
        for i, section_group in enumerate(section_groups):
            if i == start_idx:
                continue
                
            group_features = np.mean([p.features for p in section_group], axis=0)
            similarity = cosine_similarity([section_features], [group_features])[0][0]
            
            if similarity > similarity_threshold:
                repetitions += 1
        
        return repetitions
    
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
    """Test the multi-timescale analyzer"""
    
    analyzer = MultiTimescaleAnalyzer()
    
    # Test with sample audio file
    audio_file = "input_audio/Grab-a-hold.mp3"
    
    try:
        structure = analyzer.analyze_hierarchical_structure(audio_file)
        
        print(f"\n🎵 Hierarchical Structure Analysis:")
        print(f"📊 Sections: {len(structure.sections)}")
        print(f"📊 Phrases: {len(structure.phrases)}")
        print(f"📊 Measures: {len(structure.measures)}")
        
        # Print some examples
        if structure.sections:
            print(f"\n🎼 Section Example:")
            section = structure.sections[0]
            print(f"   Time: {section.start_time:.2f} - {section.end_time:.2f}")
            print(f"   Confidence: {section.confidence:.2f}")
            print(f"   Repetitions: {section.repetitions}")
        
    except FileNotFoundError:
        print(f"❌ Audio file not found: {audio_file}")
        print("Please provide a valid audio file path")

if __name__ == "__main__":
    main()
