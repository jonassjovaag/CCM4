"""
Harmonic-Rhythmic Correlation Analysis Engine

This module implements cross-modal analysis between harmonic and rhythmic patterns,
learning correlations and joint patterns for unified musical understanding.
"""

import numpy as np
import librosa
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
import json
from datetime import datetime


@dataclass
class HarmonicRhythmicEvent:
    """Represents a musical event with both harmonic and rhythmic information"""
    timestamp: float
    harmonic_features: Dict[str, Any]  # Chord, key, harmonic tension, etc.
    rhythmic_features: Dict[str, Any]  # Tempo, beat position, syncopation, etc.
    joint_features: Dict[str, Any]    # Cross-modal features
    confidence: float


@dataclass
class CorrelationPattern:
    """Represents a learned correlation between harmonic and rhythmic patterns"""
    pattern_id: str
    harmonic_signature: Dict[str, Any]
    rhythmic_signature: Dict[str, Any]
    correlation_strength: float
    frequency: int
    contexts: List[str]  # Musical contexts where this pattern appears
    examples: List[HarmonicRhythmicEvent]


@dataclass
class TemporalAlignment:
    """Represents temporal alignment between harmonic and rhythmic events"""
    harmonic_event_time: float
    rhythmic_event_time: float
    alignment_strength: float
    phase_relationship: float  # How harmonic and rhythmic events align


class HarmonicRhythmicCorrelator:
    """
    Main correlation analysis engine that learns relationships between
    harmonic and rhythmic patterns
    """
    
    def __init__(self, window_size: float = 2.0, hop_length: int = 512):
        self.window_size = window_size
        self.hop_length = hop_length
        self.sample_rate = 22050
        
        # Correlation storage
        self.correlation_patterns: Dict[str, CorrelationPattern] = {}
        self.temporal_alignments: List[TemporalAlignment] = []
        self.joint_patterns: Dict[str, List[HarmonicRhythmicEvent]] = defaultdict(list)
        
        # Analysis parameters - Enhanced sensitivity
        self.correlation_threshold = 0.15  # Lowered from 0.3 to detect weaker patterns
        self.min_pattern_frequency = 2     # Lowered from 3 to catch patterns appearing fewer times
        self.max_patterns = 1000
        
    def analyze_correlations(self, 
                           harmonic_events: List[Dict],
                           rhythmic_events: List[Dict],
                           audio_file: str) -> Dict[str, Any]:
        """
        Analyze correlations between harmonic and rhythmic events
        
        Args:
            harmonic_events: List of harmonic analysis results
            rhythmic_events: List of rhythmic analysis results  
            audio_file: Path to audio file for temporal analysis
            
        Returns:
            Dictionary containing correlation analysis results
        """
        print("🔄 Analyzing harmonic-rhythmic correlations...")
        
        # Step 1: Extract joint events
        joint_events = self._extract_joint_events(harmonic_events, rhythmic_events, audio_file)
        
        # Step 2: Analyze temporal alignments
        temporal_analysis = self._analyze_temporal_alignments(joint_events)
        
        # Step 3: Discover correlation patterns
        correlation_patterns = self._discover_correlation_patterns(joint_events)
        
        # Step 4: Learn cross-modal relationships
        cross_modal_insights = self._learn_cross_modal_relationships(joint_events)
        
        return {
            'joint_events': joint_events,
            'temporal_analysis': temporal_analysis,
            'correlation_patterns': correlation_patterns,
            'cross_modal_insights': cross_modal_insights,
            'analysis_stats': {
                'total_joint_events': len(joint_events),
                'patterns_discovered': len(correlation_patterns),
                'temporal_alignments': len(temporal_analysis),
                'correlation_strength_avg': np.mean([p.correlation_strength for p in correlation_patterns]) if correlation_patterns else 0.0
            }
        }
    
    def _extract_joint_events(self, 
                            harmonic_events: List[Dict],
                            rhythmic_events: List[Dict],
                            audio_file: str) -> List[HarmonicRhythmicEvent]:
        """Extract harmonic events for ALL events - processes every harmonic event individually"""
        
        joint_events = []
        
        # Process EVERY harmonic event individually, not just time windows
        for i, harmonic_event in enumerate(harmonic_events):
            # Extract harmonic features for this single event
            harmonic_features = self._extract_harmonic_features([harmonic_event])
            
            # Find corresponding rhythmic event (if any)
            event_time = harmonic_event.get('timestamp', 0)
            rhythmic_event = self._find_closest_rhythmic_event(event_time, rhythmic_events)
            
            # Extract rhythmic features
            rhythmic_features = {}
            if rhythmic_event:
                rhythmic_features = self._extract_rhythmic_features([rhythmic_event])
            else:
                # Create default rhythmic features when no rhythmic event found
                rhythmic_features = {
                    'tempo': 120.0,
                    'tempo_stability': 0.5,
                    'syncopation': 0.0,
                    'rhythmic_density': 0.5,
                    'rhythmic_complexity': 0.0,
                    'event_count': 0
                }
            
            # Extract joint features
            joint_features = self._extract_joint_features(harmonic_features, rhythmic_features)
            
            # Calculate confidence based on feature strength
            confidence = self._calculate_event_confidence(harmonic_features, rhythmic_features)
            
            joint_event = HarmonicRhythmicEvent(
                timestamp=event_time,
                harmonic_features=harmonic_features,
                rhythmic_features=rhythmic_features,
                joint_features=joint_features,
                confidence=confidence
            )
            
            joint_events.append(joint_event)
        
        return joint_events
    
    def _find_closest_rhythmic_event(self, target_time: float, rhythmic_events: List[Dict]) -> Optional[Dict]:
        """Find the closest rhythmic event to a given time"""
        if not rhythmic_events:
            return None
        
        closest_event = None
        min_time_diff = float('inf')
        
        for event in rhythmic_events:
            event_time = event.get('timestamp', 0)
            time_diff = abs(event_time - target_time)
            if time_diff < min_time_diff:
                min_time_diff = time_diff
                closest_event = event
        
        return closest_event
    
    def _extract_harmonic_features(self, harmonic_window: List[Dict]) -> Dict[str, Any]:
        """Extract harmonic features from a time window"""
        if not harmonic_window:
            return {}
            
        # Aggregate harmonic features
        chords = [e.get('chord', '') for e in harmonic_window if e.get('chord')]
        keys = [e.get('key', '') for e in harmonic_window if e.get('key')]
        tensions = [e.get('tension', 0) for e in harmonic_window if e.get('tension')]
        
        return {
            'primary_chord': max(set(chords), key=chords.count) if chords else '',
            'chord_diversity': len(set(chords)) / len(chords) if chords else 0,
            'key_stability': len(set(keys)) / len(keys) if keys else 0,
            'harmonic_tension': np.mean(tensions) if tensions else 0,
            'chord_change_rate': len(set(chords)) / len(harmonic_window) if harmonic_window else 0,
            'event_count': len(harmonic_window)
        }
    
    def _extract_rhythmic_features(self, rhythmic_window: List[Dict]) -> Dict[str, Any]:
        """Extract rhythmic features from a time window"""
        if not rhythmic_window:
            return {}
            
        # Aggregate rhythmic features
        tempos = [e.get('tempo', 0) for e in rhythmic_window if e.get('tempo')]
        syncopations = [e.get('syncopation', 0) for e in rhythmic_window if e.get('syncopation')]
        densities = [e.get('density', 0) for e in rhythmic_window if e.get('density')]
        
        return {
            'tempo': np.mean(tempos) if tempos else 0,
            'tempo_stability': 1 - (np.std(tempos) / np.mean(tempos)) if tempos and np.mean(tempos) > 0 else 0,
            'syncopation': np.mean(syncopations) if syncopations else 0,
            'rhythmic_density': np.mean(densities) if densities else 0,
            'rhythmic_complexity': np.std(syncopations) if syncopations else 0,
            'event_count': len(rhythmic_window)
        }
    
    def _extract_joint_features(self, 
                              harmonic_features: Dict[str, Any],
                              rhythmic_features: Dict[str, Any]) -> Dict[str, Any]:
        """Extract cross-modal features that capture harmonic-rhythmic relationships"""
        
        joint_features = {}
        
        # Harmonic-rhythmic coupling
        if harmonic_features and rhythmic_features:
            # Tempo-harmonic coupling
            joint_features['tempo_harmonic_coupling'] = (
                rhythmic_features.get('tempo', 0) * harmonic_features.get('chord_change_rate', 0)
            )
            
            # Syncopation-harmonic tension coupling
            joint_features['syncopation_tension_coupling'] = (
                rhythmic_features.get('syncopation', 0) * harmonic_features.get('harmonic_tension', 0)
            )
            
            # Density-harmonic complexity coupling
            joint_features['density_complexity_coupling'] = (
                rhythmic_features.get('rhythmic_density', 0) * harmonic_features.get('chord_diversity', 0)
            )
            
            # Stability correlation
            joint_features['stability_correlation'] = (
                harmonic_features.get('key_stability', 0) * rhythmic_features.get('tempo_stability', 0)
            )
            
            # Event density correlation
            harmonic_density = harmonic_features.get('event_count', 0) / self.window_size
            rhythmic_density = rhythmic_features.get('event_count', 0) / self.window_size
            joint_features['event_density_correlation'] = harmonic_density * rhythmic_density
        
        return joint_features
    
    def _calculate_event_confidence(self, 
                                  harmonic_features: Dict[str, Any],
                                  rhythmic_features: Dict[str, Any]) -> float:
        """Calculate confidence score for a joint event"""
        
        confidence_factors = []
        
        # Harmonic confidence factors
        if harmonic_features:
            chord_strength = 1.0 if harmonic_features.get('primary_chord') else 0.0
            tension_strength = min(harmonic_features.get('harmonic_tension', 0), 1.0)
            confidence_factors.append((chord_strength + tension_strength) / 2)
        
        # Rhythmic confidence factors
        if rhythmic_features:
            tempo_strength = min(rhythmic_features.get('tempo', 0) / 200.0, 1.0)  # Normalize tempo
            syncopation_strength = min(rhythmic_features.get('syncopation', 0) / 10.0, 1.0)  # Normalize syncopation
            confidence_factors.append((tempo_strength + syncopation_strength) / 2)
        
        return np.mean(confidence_factors) if confidence_factors else 0.0
    
    def _analyze_temporal_alignments(self, joint_events: List[HarmonicRhythmicEvent]) -> List[TemporalAlignment]:
        """Analyze temporal alignment between harmonic and rhythmic events"""
        
        alignments = []
        
        for event in joint_events:
            # Calculate phase relationship
            harmonic_phase = event.harmonic_features.get('chord_change_rate', 0)
            rhythmic_phase = event.rhythmic_features.get('rhythmic_density', 0)
            
            # Normalize phases
            harmonic_phase_norm = min(harmonic_phase, 1.0)
            rhythmic_phase_norm = min(rhythmic_phase, 1.0)
            
            # Calculate alignment strength
            alignment_strength = 1 - abs(harmonic_phase_norm - rhythmic_phase_norm)
            
            # Calculate phase relationship (0 = in phase, 0.5 = out of phase)
            phase_relationship = abs(harmonic_phase_norm - rhythmic_phase_norm)
            
            alignment = TemporalAlignment(
                harmonic_event_time=event.timestamp,
                rhythmic_event_time=event.timestamp,
                alignment_strength=alignment_strength,
                phase_relationship=phase_relationship
            )
            
            alignments.append(alignment)
        
        return alignments
    
    def _discover_correlation_patterns(self, joint_events: List[HarmonicRhythmicEvent]) -> List[CorrelationPattern]:
        """Discover recurring correlation patterns between harmonic and rhythmic features"""
        
        patterns = []
        
        # Group events by similar feature combinations
        feature_groups = defaultdict(list)
        
        for event in joint_events:
            # Create a signature for grouping
            harmonic_sig = self._create_harmonic_signature(event.harmonic_features)
            rhythmic_sig = self._create_rhythmic_signature(event.rhythmic_features)
            
            group_key = f"{harmonic_sig}_{rhythmic_sig}"
            feature_groups[group_key].append(event)
        
        # Convert groups to correlation patterns
        for group_key, events in feature_groups.items():
            if len(events) >= self.min_pattern_frequency:
                # Calculate correlation strength
                correlation_strength = self._calculate_pattern_correlation_strength(events)
                
                if correlation_strength >= self.correlation_threshold:
                    pattern = CorrelationPattern(
                        pattern_id=f"pattern_{len(patterns)}",
                        harmonic_signature=self._create_harmonic_signature(events[0].harmonic_features),
                        rhythmic_signature=self._create_rhythmic_signature(events[0].rhythmic_features),
                        correlation_strength=correlation_strength,
                        frequency=len(events),
                        contexts=[],  # Could be enhanced with musical context detection
                        examples=events[:5]  # Keep first 5 examples
                    )
                    
                    patterns.append(pattern)
        
        # Sort by correlation strength and frequency
        patterns.sort(key=lambda p: (p.correlation_strength, p.frequency), reverse=True)
        
        return patterns[:self.max_patterns]
    
    def _create_harmonic_signature(self, harmonic_features: Dict[str, Any]) -> str:
        """Create a signature string for harmonic features with less granularity"""
        if not harmonic_features:
            return "empty"
            
        chord = harmonic_features.get('primary_chord', '')
        
        # Group tension levels into broader categories (0-2, 3-5, 6-8, 9-10)
        tension_raw = harmonic_features.get('harmonic_tension', 0) * 10
        if tension_raw <= 2:
            tension_level = "low"
        elif tension_raw <= 5:
            tension_level = "med"
        elif tension_raw <= 8:
            tension_level = "high"
        else:
            tension_level = "very_high"
        
        # Group change rates into broader categories (0-2, 3-5, 6-8, 9-10)
        change_rate_raw = harmonic_features.get('chord_change_rate', 0) * 10
        if change_rate_raw <= 2:
            change_level = "low"
        elif change_rate_raw <= 5:
            change_level = "med"
        elif change_rate_raw <= 8:
            change_level = "high"
        else:
            change_level = "very_high"
        
        return f"{chord}_t{tension_level}_c{change_level}"
    
    def _create_rhythmic_signature(self, rhythmic_features: Dict[str, Any]) -> str:
        """Create a signature string for rhythmic features with less granularity"""
        if not rhythmic_features:
            return "empty"
            
        # Group tempos into broader ranges (0-80, 81-120, 121-160, 161+)
        tempo_raw = rhythmic_features.get('tempo', 0)
        if tempo_raw <= 80:
            tempo_level = "slow"
        elif tempo_raw <= 120:
            tempo_level = "moderate"
        elif tempo_raw <= 160:
            tempo_level = "fast"
        else:
            tempo_level = "very_fast"
        
        # Group syncopation levels into broader categories (0-2, 3-5, 6-8, 9-10)
        syncopation_raw = rhythmic_features.get('syncopation', 0) * 10
        if syncopation_raw <= 2:
            syncopation_level = "low"
        elif syncopation_raw <= 5:
            syncopation_level = "med"
        elif syncopation_raw <= 8:
            syncopation_level = "high"
        else:
            syncopation_level = "very_high"
        
        # Group density levels into broader categories (0-2, 3-5, 6-8, 9-10)
        density_raw = rhythmic_features.get('rhythmic_density', 0) * 10
        if density_raw <= 2:
            density_level = "low"
        elif density_raw <= 5:
            density_level = "med"
        elif density_raw <= 8:
            density_level = "high"
        else:
            density_level = "very_high"
        
        return f"t{tempo_level}_s{syncopation_level}_d{density_level}"
    
    def _calculate_pattern_correlation_strength(self, events: List[HarmonicRhythmicEvent]) -> float:
        """Calculate the correlation strength of a pattern"""
        
        if len(events) < 2:
            return 0.0
        
        # Calculate consistency of joint features
        joint_features_list = [e.joint_features for e in events if e.joint_features]
        
        if not joint_features_list:
            return 0.0
        
        # Calculate variance in joint features (lower variance = higher correlation)
        feature_names = list(joint_features_list[0].keys())
        variances = []
        
        for feature_name in feature_names:
            values = [f.get(feature_name, 0) for f in joint_features_list]
            if values:
                variance = np.var(values)
                variances.append(variance)
        
        # Convert variance to correlation strength (inverse relationship)
        avg_variance = np.mean(variances) if variances else 1.0
        correlation_strength = 1.0 / (1.0 + avg_variance)
        
        return correlation_strength
    
    def _learn_cross_modal_relationships(self, joint_events: List[HarmonicRhythmicEvent]) -> Dict[str, Any]:
        """Learn cross-modal relationships and insights"""
        
        insights = {
            'tempo_harmonic_correlations': [],
            'syncopation_tension_correlations': [],
            'density_complexity_correlations': [],
            'stability_patterns': [],
            'cross_modal_insights': []
        }
        
        # Analyze tempo-harmonic correlations
        tempo_harmonic_data = []
        for event in joint_events:
            if event.harmonic_features and event.rhythmic_features:
                tempo = event.rhythmic_features.get('tempo', 0)
                chord_change_rate = event.harmonic_features.get('chord_change_rate', 0)
                if tempo > 0 and chord_change_rate > 0:
                    tempo_harmonic_data.append((tempo, chord_change_rate))
        
        if tempo_harmonic_data:
            tempos, change_rates = zip(*tempo_harmonic_data)
            tempos = np.array(tempos)
            change_rates = np.array(change_rates)
            
            # Check for sufficient variance before correlation
            if np.std(tempos) > 1e-10 and np.std(change_rates) > 1e-10:
                try:
                    correlation = np.corrcoef(tempos, change_rates)[0, 1]
                    if not np.isnan(correlation):
                        insights['tempo_harmonic_correlations'].append({
                            'correlation': correlation,
                            'strength': abs(correlation),
                            'interpretation': 'positive' if correlation > 0 else 'negative'
                        })
                except (ValueError, RuntimeWarning):
                    # Skip correlation if calculation fails
                    pass
        
        # Analyze syncopation-tension correlations
        syncopation_tension_data = []
        for event in joint_events:
            if event.harmonic_features and event.rhythmic_features:
                syncopation = event.rhythmic_features.get('syncopation', 0)
                tension = event.harmonic_features.get('harmonic_tension', 0)
                if syncopation > 0 and tension > 0:
                    syncopation_tension_data.append((syncopation, tension))
        
        if syncopation_tension_data:
            syncopations, tensions = zip(*syncopation_tension_data)
            syncopations = np.array(syncopations)
            tensions = np.array(tensions)
            
            # Check for sufficient variance before correlation
            if np.std(syncopations) > 1e-10 and np.std(tensions) > 1e-10:
                try:
                    correlation = np.corrcoef(syncopations, tensions)[0, 1]
                    if not np.isnan(correlation):
                        insights['syncopation_tension_correlations'].append({
                            'correlation': correlation,
                            'strength': abs(correlation),
                            'interpretation': 'positive' if correlation > 0 else 'negative'
                        })
                except (ValueError, RuntimeWarning):
                    # Skip correlation if calculation fails
                    pass
        
        # Generate cross-modal insights
        if insights['tempo_harmonic_correlations']:
            tempo_corr = insights['tempo_harmonic_correlations'][0]
            if tempo_corr['strength'] > 0.3:
                direction = "increases" if tempo_corr['interpretation'] == 'positive' else "decreases"
                insights['cross_modal_insights'].append(
                    f"Tempo {direction} with harmonic change rate (strength: {tempo_corr['strength']:.2f})"
                )
        
        if insights['syncopation_tension_correlations']:
            sync_corr = insights['syncopation_tension_correlations'][0]
            if sync_corr['strength'] > 0.3:
                direction = "increases" if sync_corr['interpretation'] == 'positive' else "decreases"
                insights['cross_modal_insights'].append(
                    f"Syncopation {direction} with harmonic tension (strength: {sync_corr['strength']:.2f})"
                )
        
        return insights
    
    def save_correlations(self, filepath: str):
        """Save correlation patterns to file"""
        data = {
            'correlation_patterns': [
                {
                    'pattern_id': p.pattern_id,
                    'harmonic_signature': p.harmonic_signature,
                    'rhythmic_signature': p.rhythmic_signature,
                    'correlation_strength': p.correlation_strength,
                    'frequency': p.frequency,
                    'contexts': p.contexts
                }
                for p in self.correlation_patterns.values()
            ],
            'temporal_alignments': [
                {
                    'harmonic_event_time': ta.harmonic_event_time,
                    'rhythmic_event_time': ta.rhythmic_event_time,
                    'alignment_strength': ta.alignment_strength,
                    'phase_relationship': ta.phase_relationship
                }
                for ta in self.temporal_alignments
            ],
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'total_patterns': len(self.correlation_patterns),
                'total_alignments': len(self.temporal_alignments)
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_correlations(self, filepath: str):
        """Load correlation patterns from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Reconstruct correlation patterns
        self.correlation_patterns = {}
        for p_data in data.get('correlation_patterns', []):
            pattern = CorrelationPattern(
                pattern_id=p_data['pattern_id'],
                harmonic_signature=p_data['harmonic_signature'],
                rhythmic_signature=p_data['rhythmic_signature'],
                correlation_strength=p_data['correlation_strength'],
                frequency=p_data['frequency'],
                contexts=p_data['contexts'],
                examples=[]  # Examples not saved for space
            )
            self.correlation_patterns[pattern.pattern_id] = pattern
        
        # Reconstruct temporal alignments
        self.temporal_alignments = []
        for ta_data in data.get('temporal_alignments', []):
            alignment = TemporalAlignment(
                harmonic_event_time=ta_data['harmonic_event_time'],
                rhythmic_event_time=ta_data['rhythmic_event_time'],
                alignment_strength=ta_data['alignment_strength'],
                phase_relationship=ta_data['phase_relationship']
            )
            self.temporal_alignments.append(alignment)


def main():
    """Test the correlation engine"""
    print("🔄 Testing Harmonic-Rhythmic Correlation Engine...")
    
    # Create mock data for testing
    harmonic_events = [
        {'timestamp': 0.0, 'chord': 'C', 'key': 'C', 'tension': 0.3},
        {'timestamp': 2.0, 'chord': 'F', 'key': 'C', 'tension': 0.5},
        {'timestamp': 4.0, 'chord': 'G', 'key': 'C', 'tension': 0.7},
        {'timestamp': 6.0, 'chord': 'C', 'key': 'C', 'tension': 0.2},
    ]
    
    rhythmic_events = [
        {'timestamp': 0.0, 'tempo': 120, 'syncopation': 0.2, 'density': 0.5},
        {'timestamp': 2.0, 'tempo': 125, 'syncopation': 0.4, 'density': 0.7},
        {'timestamp': 4.0, 'tempo': 130, 'syncopation': 0.6, 'density': 0.8},
        {'timestamp': 6.0, 'tempo': 120, 'syncopation': 0.3, 'density': 0.6},
    ]
    
    correlator = HarmonicRhythmicCorrelator()
    
    # Test with mock audio file (we'll create a dummy one)
    import tempfile
    import os
    
    # Create a temporary audio file for testing
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        # Create a simple sine wave for testing
        duration = 8.0
        sample_rate = 22050
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        audio = (audio * 32767).astype(np.int16)
        
        import soundfile as sf
        sf.write(tmp_file.name, audio, sample_rate)
        
        try:
            # Run correlation analysis
            results = correlator.analyze_correlations(harmonic_events, rhythmic_events, tmp_file.name)
            
            print(f"✅ Correlation analysis complete!")
            print(f"   Joint events: {results['analysis_stats']['total_joint_events']}")
            print(f"   Patterns discovered: {results['analysis_stats']['patterns_discovered']}")
            print(f"   Average correlation strength: {results['analysis_stats']['correlation_strength_avg']:.3f}")
            
            # Print cross-modal insights
            insights = results['cross_modal_insights']
            if insights['cross_modal_insights']:
                print(f"\n🎵 Cross-Modal Insights:")
                for insight in insights['cross_modal_insights']:
                    print(f"   • {insight}")
            
        finally:
            # Clean up temporary file
            os.unlink(tmp_file.name)


if __name__ == "__main__":
    main()
