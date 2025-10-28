#!/usr/bin/env python3
"""
Timescale Manager
Coordinates multi-timescale analysis and manages temporal relationships

This module manages the coordination between different timescales and ensures
proper temporal alignment and hierarchical relationships.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import librosa

@dataclass
class TimescaleConfig:
    """Configuration for a specific timescale"""
    name: str
    window_size: float
    hop_size: float
    min_pattern_length: float
    similarity_threshold: float

class TimescaleManager:
    """
    Manages multiple timescales and their relationships
    
    Based on research showing that different brain regions process music at
    different timescales, this manager coordinates analysis across scales.
    """
    
    def __init__(self):
        # Define timescale configurations based on research
        self.timescales = {
            'measure': TimescaleConfig(
                name='measure',
                window_size=6.0,      # 6 seconds (Farbood et al., 2015)
                hop_size=3.0,          # 50% overlap
                min_pattern_length=2.0,
                similarity_threshold=0.7
            ),
            'phrase': TimescaleConfig(
                name='phrase',
                window_size=30.0,     # 30 seconds
                hop_size=15.0,         # 50% overlap
                min_pattern_length=10.0,
                similarity_threshold=0.6
            ),
            'section': TimescaleConfig(
                name='section',
                window_size=120.0,    # 2 minutes
                hop_size=60.0,         # 50% overlap
                min_pattern_length=30.0,
                similarity_threshold=0.5
            )
        }
        
        # Temporal alignment parameters
        self.alignment_tolerance = 2.0  # seconds
        self.hierarchical_weight = 0.3  # Weight for hierarchical consistency
    
    def create_temporal_windows(self, duration: float, timescale: str) -> List[Tuple[float, float]]:
        """Create temporal windows for a specific timescale"""
        
        config = self.timescales[timescale]
        windows = []
        
        start_time = 0.0
        while start_time < duration:
            end_time = min(start_time + config.window_size, duration)
            windows.append((start_time, end_time))
            start_time += config.hop_size
        
        return windows
    
    def align_patterns_across_timescales(self, patterns_by_scale: Dict[str, List]) -> Dict[str, List]:
        """Align patterns across different timescales"""
        
        print("🔄 Aligning patterns across timescales...")
        
        aligned_patterns = {}
        
        # Start with the coarsest scale (sections)
        if 'section' in patterns_by_scale:
            aligned_patterns['section'] = patterns_by_scale['section']
            
            # Align phrases to sections
            if 'phrase' in patterns_by_scale:
                aligned_patterns['phrase'] = self._align_to_parent_scale(
                    patterns_by_scale['phrase'], 
                    patterns_by_scale['section'],
                    'phrase'
                )
                
                # Align measures to phrases
                if 'measure' in patterns_by_scale:
                    aligned_patterns['measure'] = self._align_to_parent_scale(
                        patterns_by_scale['measure'],
                        aligned_patterns['phrase'],
                        'measure'
                    )
        
        return aligned_patterns
    
    def _align_to_parent_scale(self, child_patterns: List, parent_patterns: List, 
                              child_scale: str) -> List:
        """Align child patterns to parent patterns"""
        
        aligned_patterns = []
        
        for child_pattern in child_patterns:
            # Find the best matching parent pattern
            best_parent = None
            best_overlap = 0.0
            
            for parent_pattern in parent_patterns:
                overlap = self._calculate_temporal_overlap(child_pattern, parent_pattern)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_parent = parent_pattern
            
            # Only include child pattern if it has significant overlap with a parent
            if best_overlap > 0.3:  # 30% overlap threshold
                # Update child pattern with parent information
                child_pattern.parent_id = best_parent.pattern_id
                child_pattern.hierarchical_confidence = best_overlap
                aligned_patterns.append(child_pattern)
        
        return aligned_patterns
    
    def _calculate_temporal_overlap(self, pattern1, pattern2) -> float:
        """Calculate temporal overlap between two patterns"""
        
        start1, end1 = pattern1.start_time, pattern1.end_time
        start2, end2 = pattern2.start_time, pattern2.end_time
        
        # Calculate overlap
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)
        
        if overlap_start >= overlap_end:
            return 0.0
        
        overlap_duration = overlap_end - overlap_start
        total_duration = max(end1, end2) - min(start1, start2)
        
        return overlap_duration / total_duration if total_duration > 0 else 0.0
    
    def validate_hierarchical_consistency(self, patterns_by_scale: Dict[str, List]) -> Dict[str, float]:
        """Validate hierarchical consistency across timescales"""
        
        consistency_scores = {}
        
        # Check section-phrase consistency
        if 'section' in patterns_by_scale and 'phrase' in patterns_by_scale:
            consistency_scores['section_phrase'] = self._calculate_scale_consistency(
                patterns_by_scale['section'], patterns_by_scale['phrase']
            )
        
        # Check phrase-measure consistency
        if 'phrase' in patterns_by_scale and 'measure' in patterns_by_scale:
            consistency_scores['phrase_measure'] = self._calculate_scale_consistency(
                patterns_by_scale['phrase'], patterns_by_scale['measure']
            )
        
        return consistency_scores
    
    def _calculate_scale_consistency(self, parent_patterns: List, child_patterns: List) -> float:
        """Calculate consistency between parent and child scales"""
        
        if not parent_patterns or not child_patterns:
            return 0.0
        
        total_consistency = 0.0
        valid_pairs = 0
        
        for parent in parent_patterns:
            parent_children = []
            
            # Find all child patterns within this parent
            for child in child_patterns:
                if (child.start_time >= parent.start_time and 
                    child.end_time <= parent.end_time):
                    parent_children.append(child)
            
            if parent_children:
                # Calculate consistency within this parent-child group
                consistency = self._calculate_group_consistency(parent, parent_children)
                total_consistency += consistency
                valid_pairs += 1
        
        return total_consistency / valid_pairs if valid_pairs > 0 else 0.0
    
    def _calculate_group_consistency(self, parent_pattern, child_patterns: List) -> float:
        """Calculate consistency within a parent-child group"""
        
        if len(child_patterns) < 2:
            return 1.0
        
        # Calculate feature consistency among children
        child_features = [child.features for child in child_patterns]
        feature_variance = np.var(child_features, axis=0).mean()
        
        # Calculate temporal consistency
        child_times = [(child.start_time, child.end_time) for child in child_patterns]
        temporal_consistency = self._calculate_temporal_consistency(child_times)
        
        # Combine feature and temporal consistency
        consistency = (1.0 - feature_variance) * temporal_consistency
        
        return max(0.0, min(1.0, consistency))
    
    def _calculate_temporal_consistency(self, time_intervals: List[Tuple[float, float]]) -> float:
        """Calculate temporal consistency of intervals"""
        
        if len(time_intervals) < 2:
            return 1.0
        
        # Calculate average interval duration
        durations = [end - start for start, end in time_intervals]
        avg_duration = np.mean(durations)
        
        # Calculate duration variance (lower is more consistent)
        duration_variance = np.var(durations)
        
        # Calculate temporal spacing consistency
        starts = [start for start, end in time_intervals]
        starts.sort()
        
        if len(starts) > 1:
            intervals = [starts[i+1] - starts[i] for i in range(len(starts)-1)]
            interval_variance = np.var(intervals)
        else:
            interval_variance = 0.0
        
        # Combine duration and interval consistency
        duration_consistency = 1.0 / (1.0 + duration_variance / (avg_duration ** 2))
        interval_consistency = 1.0 / (1.0 + interval_variance / (avg_duration ** 2))
        
        return (duration_consistency + interval_consistency) / 2.0
    
    def optimize_timescale_parameters(self, patterns_by_scale: Dict[str, List], 
                                   audio_duration: float) -> Dict[str, TimescaleConfig]:
        """Optimize timescale parameters based on detected patterns"""
        
        print("⚙️  Optimizing timescale parameters...")
        
        optimized_configs = {}
        
        for scale_name, patterns in patterns_by_scale.items():
            if not patterns:
                optimized_configs[scale_name] = self.timescales[scale_name]
                continue
            
            config = self.timescales[scale_name]
            
            # Analyze pattern characteristics
            pattern_durations = [p.end_time - p.start_time for p in patterns]
            avg_duration = np.mean(pattern_durations)
            
            # Adjust window size based on actual pattern durations
            optimal_window_size = avg_duration * 1.5  # 50% padding
            
            # Adjust hop size to maintain overlap
            optimal_hop_size = optimal_window_size * 0.5
            
            # Adjust similarity threshold based on pattern quality
            pattern_confidences = [p.confidence for p in patterns]
            avg_confidence = np.mean(pattern_confidences)
            optimal_threshold = max(0.3, min(0.9, avg_confidence))
            
            optimized_config = TimescaleConfig(
                name=scale_name,
                window_size=optimal_window_size,
                hop_size=optimal_hop_size,
                min_pattern_length=config.min_pattern_length,
                similarity_threshold=optimal_threshold
            )
            
            optimized_configs[scale_name] = optimized_config
            
            print(f"   {scale_name}: window={optimal_window_size:.1f}s, "
                  f"hop={optimal_hop_size:.1f}s, threshold={optimal_threshold:.2f}")
        
        return optimized_configs
    
    def generate_timescale_report(self, patterns_by_scale: Dict[str, List]) -> str:
        """Generate a comprehensive report of timescale analysis"""
        
        report = []
        report.append("🎵 Multi-Timescale Analysis Report")
        report.append("=" * 50)
        
        for scale_name, patterns in patterns_by_scale.items():
            report.append(f"\n📊 {scale_name.upper()} LEVEL:")
            report.append(f"   Patterns detected: {len(patterns)}")
            
            if patterns:
                durations = [p.end_time - p.start_time for p in patterns]
                confidences = [p.confidence for p in patterns]
                repetitions = [p.repetitions for p in patterns]
                
                report.append(f"   Average duration: {np.mean(durations):.2f}s")
                report.append(f"   Duration range: {np.min(durations):.2f}s - {np.max(durations):.2f}s")
                report.append(f"   Average confidence: {np.mean(confidences):.3f}")
                report.append(f"   Average repetitions: {np.mean(repetitions):.1f}")
                
                # Show top patterns
                top_patterns = sorted(patterns, key=lambda x: x.confidence, reverse=True)[:3]
                report.append(f"   Top patterns:")
                for i, pattern in enumerate(top_patterns, 1):
                    report.append(f"     {i}. {pattern.pattern_id}: "
                                f"{pattern.start_time:.1f}-{pattern.end_time:.1f}s "
                                f"(conf: {pattern.confidence:.3f})")
        
        return "\n".join(report)

def main():
    """Test the timescale manager"""
    
    manager = TimescaleManager()
    
    # Test temporal window creation
    duration = 300.0  # 5 minutes
    
    for scale_name in ['measure', 'phrase', 'section']:
        windows = manager.create_temporal_windows(duration, scale_name)
        print(f"📊 {scale_name}: {len(windows)} windows")
        if windows:
            print(f"   First window: {windows[0][0]:.1f}-{windows[0][1]:.1f}s")
            print(f"   Last window: {windows[-1][0]:.1f}-{windows[-1][1]:.1f}s")

if __name__ == "__main__":
    main()
