#!/usr/bin/env python3
"""
Heavy Rhythmic Analyzer
Comprehensive rhythmic analysis for audio file training

This module provides detailed rhythmic analysis including:
- Beat tracking and tempo estimation
- Rhythmic pattern recognition
- Syncopation analysis
- Meter detection
- Rhythmic density analysis
"""

import numpy as np
import librosa
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import os

@dataclass
class RhythmicPattern:
    """Represents a detected rhythmic pattern"""
    pattern_id: str
    start_time: float
    end_time: float
    tempo: float
    density: float
    syncopation: float
    meter: str
    pattern_type: str
    confidence: float

@dataclass
class RhythmicAnalysis:
    """Complete rhythmic analysis result"""
    tempo: float
    beats: np.ndarray
    onsets: np.ndarray
    patterns: List[RhythmicPattern]
    syncopation_score: float
    meter: str
    rhythmic_complexity: float
    density_profile: np.ndarray
    beat_strength: np.ndarray

class HeavyRhythmicAnalyzer:
    """
    Heavy rhythmic analysis for training data generation
    
    Performs comprehensive rhythmic analysis including:
    - Beat tracking and tempo estimation
    - Rhythmic pattern recognition
    - Syncopation and meter detection
    - Rhythmic density analysis
    """
    
    def __init__(self, sample_rate: int = 22050, hop_length: int = 512):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        
        # Analysis parameters
        self.min_tempo = 60.0
        self.max_tempo = 200.0
        self.pattern_min_duration = 0.5  # seconds
        self.pattern_max_duration = 8.0  # seconds
        
    def analyze_rhythmic_structure(self, audio_file: str) -> RhythmicAnalysis:
        """
        Perform comprehensive rhythmic analysis
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            RhythmicAnalysis: Complete rhythmic analysis
        """
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
        
        print(f"🥁 Analyzing rhythmic structure: {audio_file}")
        
        # Load audio
        audio, sr = librosa.load(audio_file, sr=self.sample_rate, mono=True)
        
        # Beat tracking and tempo estimation using harmonic analysis approach
        try:
            # Method 1: Multi-band onset strength analysis (inspired by Deep-Rhythm paper)
            tempo_candidates = []
            
            # Compute onset strength function
            onset_strength = librosa.onset.onset_strength(
                y=audio, sr=sr, hop_length=self.hop_length
            )
            
            # Method 1a: Tempo estimation from onset strength
            try:
                tempo1 = librosa.beat.tempo(
                    onset_envelope=onset_strength,
                    sr=sr,
                    hop_length=self.hop_length
                )
                if tempo1 is not None and len(tempo1) > 0:
                    tempo_val = float(tempo1[0]) if hasattr(tempo1, '__getitem__') else float(tempo1)
                    if 60 <= tempo_val <= 200:
                        tempo_candidates.append(tempo_val)
            except:
                pass
            
            # Method 1b: Beat tracking with onset strength
            try:
                tempo2, _ = librosa.beat.beat_track(
                    onset_envelope=onset_strength,
                    sr=sr,
                    hop_length=self.hop_length,
                    start_bpm=None,
                    tightness=100.0,
                    trim=False,
                    units='time'
                )
                if tempo2 is not None and 60 <= tempo2 <= 200:
                    tempo_candidates.append(float(tempo2))
            except:
                pass
            
            # Method 2: Multi-band analysis (inspired by Deep-Rhythm)
            try:
                # Compute onset strength in different frequency bands
                for n_bands in [4, 8]:
                    # Create frequency bands
                    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
                    band_boundaries = np.logspace(np.log10(freqs[1]), np.log10(freqs[-1]), n_bands+1)
                    
                    band_tempos = []
                    for i in range(n_bands):
                        # Extract frequency band
                        low_freq = band_boundaries[i]
                        high_freq = band_boundaries[i+1]
                        
                        # Apply bandpass filter
                        filtered_audio = librosa.effects.preemphasis(audio)
                        
                        # Compute onset strength for this band
                        band_onset = librosa.onset.onset_strength(
                            y=filtered_audio, sr=sr, hop_length=self.hop_length
                        )
                        
                        # Estimate tempo for this band
                        try:
                            band_tempo = librosa.beat.tempo(
                                onset_envelope=band_onset,
                                sr=sr,
                                hop_length=self.hop_length
                            )
                            if band_tempo is not None and len(band_tempo) > 0:
                                tempo_val = float(band_tempo[0]) if hasattr(band_tempo, '__getitem__') else float(band_tempo)
                                if 60 <= tempo_val <= 200:
                                    band_tempos.append(tempo_val)
                        except:
                            continue
                    
                    # Use median tempo from all bands
                    if band_tempos:
                        tempo_candidates.append(np.median(band_tempos))
            except:
                pass
            
            # Method 3: SMART PULSE detection (detect both slow and fast tempos)
            # This is the PRIMARY method - tag it so we can prioritize it later
            smart_tempo_detected = None
            try:
                # Compute onset strength envelope with higher threshold to reduce false positives
                # The default onset_detect was finding spurious ~99.4 BPM patterns
                onset_env = librosa.onset.onset_strength(
                    y=audio, sr=sr, hop_length=self.hop_length
                )
                
                # Use a moderate threshold to filter weak onsets (spurious detections)
                # The 99.4 BPM false positive was caused by weak periodic noise
                # Don't make it too aggressive or we lose all onsets
                onset_threshold = 0.05  # Fixed threshold - librosa's default is too sensitive
                
                # Detect onsets with threshold
                onset_frames = librosa.onset.onset_detect(
                    onset_envelope=onset_env,
                    sr=sr,
                    hop_length=self.hop_length,
                    units='frames',
                    backtrack=True,  # Backtrack to find true onset position
                    delta=onset_threshold  # Only detect onsets above this threshold
                )
                
                # Convert to time
                onsets = librosa.frames_to_time(onset_frames, sr=sr, hop_length=self.hop_length)
                
                # Debug: Print number of onsets detected
                print(f"🎵 Detected {len(onsets)} onsets with threshold {onset_threshold}")
                
                # Extract onset strengths for weighting (strong onsets = structural pulse)
                onset_strengths = onset_env[onset_frames]
                
                if len(onsets) >= 4:
                    # Calculate intervals between onsets
                    intervals = np.diff(onsets)
                    
                    # CRITICAL FIX: Weight intervals by onset strength
                    # Strong onset → strong onset = structural pulse (kick drums)
                    # Weak onset → weak onset = surface rhythm (hi-hats)
                    interval_weights = []
                    for i in range(len(intervals)):
                        # Geometric mean of consecutive onset strengths
                        # Both onsets must be strong for the interval to be strong
                        weight = np.sqrt(onset_strengths[i] * onset_strengths[i + 1])
                        interval_weights.append(weight)
                    
                    interval_weights = np.array(interval_weights)
                    
                    # Check SLOW TEMPO range (45-75 BPM = 0.8-1.35 second intervals)
                    slow_mask = (intervals >= 0.8) & (intervals <= 1.35)
                    slow_intervals = intervals[slow_mask]
                    slow_weights = interval_weights[slow_mask]
                    slow_tempo = None
                    slow_confidence = 0
                    
                    if len(slow_intervals) >= 3:
                        slow_bpms = 60.0 / slow_intervals
                        valid_mask = (slow_bpms >= 45) & (slow_bpms <= 75)
                        valid_slow = slow_bpms[valid_mask]
                        valid_slow_weights = slow_weights[valid_mask]
                        
                        if len(valid_slow) >= 3:
                            # Weighted confidence = sum of onset strengths
                            slow_confidence = float(np.sum(valid_slow_weights))
                            
                            # Weighted median tempo - intervals with strong onsets count more
                            sorted_idx = np.argsort(slow_intervals[valid_mask])
                            cumsum = np.cumsum(valid_slow_weights[sorted_idx])
                            median_idx = np.searchsorted(cumsum, cumsum[-1] / 2) if cumsum[-1] > 0 else 0
                            median_interval = slow_intervals[valid_mask][sorted_idx[median_idx]]
                            slow_tempo = float(60.0 / median_interval)
                            
                            print(f"🎵 SLOW tempo candidate: {slow_tempo:.1f} BPM (weighted confidence: {slow_confidence:.1f}, {len(valid_slow)} intervals)")
                    
                    # Check FAST TEMPO range (85-120 BPM = 0.5-0.7 second intervals)
                    fast_mask = (intervals >= 0.5) & (intervals <= 0.7)
                    fast_intervals = intervals[fast_mask]
                    fast_weights = interval_weights[fast_mask]
                    fast_tempo = None
                    fast_confidence = 0
                    
                    if len(fast_intervals) >= 3:
                        fast_bpms = 60.0 / fast_intervals
                        valid_mask = (fast_bpms >= 85) & (fast_bpms <= 120)
                        valid_fast = fast_bpms[valid_mask]
                        valid_fast_weights = fast_weights[valid_mask]
                        
                        if len(valid_fast) >= 3:
                            # Weighted confidence = sum of onset strengths
                            fast_confidence = float(np.sum(valid_fast_weights))
                            
                            # Weighted median tempo
                            sorted_idx = np.argsort(fast_intervals[valid_mask])
                            cumsum = np.cumsum(valid_fast_weights[sorted_idx])
                            median_idx = np.searchsorted(cumsum, cumsum[-1] / 2) if cumsum[-1] > 0 else 0
                            median_interval = fast_intervals[valid_mask][sorted_idx[median_idx]]
                            fast_tempo = float(60.0 / median_interval)
                            
                            print(f"🎵 FAST tempo candidate: {fast_tempo:.1f} BPM (weighted confidence: {fast_confidence:.1f}, {len(valid_fast)} intervals)")
                    
                    # Choose based on WEIGHTED confidence (not just count!)
                    # Bias toward slow tempo (structural pulse) by requiring fast to be 1.5x stronger
                    if slow_confidence > 0 and fast_confidence > 0:
                        # Both detected - choose based on weighted confidence with slow bias
                        if slow_confidence > fast_confidence / 1.5:
                            # Slow tempo wins (structural pulse usually stronger per-onset)
                            smart_tempo_detected = slow_tempo
                            tempo_candidates.append(slow_tempo)
                            print(f"🎵 ✅ SMART method chose SLOW tempo ({slow_tempo:.1f} BPM) - weighted confidence: {slow_confidence:.1f} vs {fast_confidence:.1f}")
                        else:
                            # Fast tempo has significantly more weighted evidence
                            smart_tempo_detected = fast_tempo
                            tempo_candidates.append(fast_tempo)
                            print(f"🎵 ✅ SMART method chose FAST tempo ({fast_tempo:.1f} BPM) - weighted confidence: {fast_confidence:.1f} vs {slow_confidence:.1f}")
                    elif slow_tempo:
                        smart_tempo_detected = slow_tempo
                        tempo_candidates.append(slow_tempo)
                        print(f"🎵 Only SLOW tempo detected")
                    elif fast_tempo:
                        smart_tempo_detected = fast_tempo
                        tempo_candidates.append(fast_tempo)
                        print(f"🎵 Only FAST tempo detected")
                        
            except Exception as e:
                print(f"⚠️ Smart pulse detection failed: {e}")
                pass
            
            # Method 4: Traditional beat tracking with multiple parameters
            for tightness in [50.0, 100.0, 200.0]:
                for start_bpm in [None, 110.0, 100.0, 120.0]:
                    try:
                        tempo4, _ = librosa.beat.beat_track(
                            y=audio, sr=sr,
                            start_bpm=start_bpm,
                            tightness=tightness,
                            trim=False,
                            units='time'
                        )
                        if tempo4 is not None and 60 <= tempo4 <= 200:
                            tempo_candidates.append(float(tempo4))
                    except:
                        continue
            
            # Choose the best tempo estimate - SMART selection for both slow and fast
            if tempo_candidates:
                # Remove duplicates and outliers
                unique_candidates = list(set(tempo_candidates))
                
                # Smart weighting for both slow and fast tempos
                weighted_candidates = []
                
                for candidate in unique_candidates:
                    weight = 1.0
                    
                    # MASSIVE priority for the smart pulse detection result
                    if smart_tempo_detected is not None and abs(candidate - smart_tempo_detected) < 0.5:
                        weight = 100.0  # Overwhelming priority for smart method
                        print(f"🎵 Candidate {candidate:.1f} is from SMART method - weight: {weight}")
                    # Lower priority for other methods
                    elif 45 <= candidate <= 75:
                        weight = 3.0   # Medium priority for slow main pulse
                    elif 85 <= candidate <= 120:
                        weight = 3.0   # Medium priority for fast main pulse
                    elif 75 <= candidate <= 85:
                        weight = 2.0   # Low priority for in-between
                    else:
                        weight = 1.0   # Very low priority for unusual tempos
                    
                    weighted_candidates.append((candidate, weight))
                
                # Sort by weight and take the best candidate
                weighted_candidates.sort(key=lambda x: x[1], reverse=True)
                tempo = weighted_candidates[0][0]
                
                print(f"🎵 All tempo candidates: {unique_candidates}")
                print(f"🎵 FINAL selection: {tempo:.1f} BPM (weight: {weighted_candidates[0][1]:.1f})")
                
                # Show why this tempo was chosen
                if 45 <= tempo <= 75:
                    print(f"🎵 ✅ Selected SLOW main pulse (53-60 BPM range)")
                elif 85 <= tempo <= 120:
                    print(f"🎵 ✅ Selected FAST main pulse (100 BPM range)")
                else:
                    print(f"🎵 ? Selected unusual tempo")
            else:
                tempo = 100.0  # Default to moderate tempo
                print(f"⚠️ No valid tempo candidates, using default: {tempo} BPM")
            
            # Clamp tempo to reasonable range (allow slower main pulse)
            tempo = max(30.0, min(200.0, tempo))
            
            # Now get beats with the estimated tempo
            try:
                beats = librosa.beat.beat_track(
                    y=audio, sr=sr,
                    start_bpm=tempo,
                    tightness=100.0,
                    trim=False,
                    units='time'
                )[1]  # Get beats array
            except:
                beats = np.array([])
                    
        except Exception as e:
            print(f"⚠️ Beat tracking failed: {e}")
            tempo = 120.0
            beats = np.array([])
        
        # Onset detection
        onsets = librosa.onset.onset_detect(
            y=audio, sr=sr, 
            hop_length=self.hop_length,
            units='time'
        )
        
        # Beat strength analysis (simplified - librosa doesn't have beat_strength)
        # Use onset strength as a proxy for beat strength
        beat_strength = librosa.onset.onset_strength(
            y=audio, sr=sr, 
            hop_length=self.hop_length
        )
        
        # Rhythmic pattern recognition
        patterns = self._extract_rhythmic_patterns(onsets, beats, tempo)
        
        # Syncopation analysis
        syncopation_score = self._analyze_syncopation(onsets, beats)
        
        # Meter detection
        meter = self._detect_meter(beats, tempo)
        
        # Rhythmic complexity
        complexity = self._calculate_rhythmic_complexity(onsets, beats, tempo)
        
        # Density profile
        density_profile = self._calculate_density_profile(onsets, beats)
        
        return RhythmicAnalysis(
            tempo=tempo,
            beats=beats,
            onsets=onsets,
            patterns=patterns,
            syncopation_score=syncopation_score,
            meter=meter,
            rhythmic_complexity=complexity,
            density_profile=density_profile,
            beat_strength=beat_strength
        )
    
    def _extract_rhythmic_patterns(self, onsets: np.ndarray, beats: np.ndarray, tempo: float) -> List[RhythmicPattern]:
        """Extract rhythmic patterns from onsets and beats"""
        patterns = []
        
        if len(onsets) < 2:
            return patterns
        
        # Group onsets into patterns based on beat structure
        beat_interval = 60.0 / tempo  # Beat duration in seconds
        
        pattern_id = 0
        current_pattern_start = onsets[0]
        current_pattern_onsets = [onsets[0]]
        
        for i in range(1, len(onsets)):
            onset = onsets[i]
            
            # Check if onset is within reasonable distance from previous
            if onset - current_pattern_onsets[-1] <= beat_interval * 2:
                current_pattern_onsets.append(onset)
            else:
                # End current pattern and start new one
                if len(current_pattern_onsets) >= 2:
                    pattern = self._create_pattern(
                        pattern_id, current_pattern_start, 
                        current_pattern_onsets[-1], current_pattern_onsets,
                        tempo
                    )
                    patterns.append(pattern)
                    pattern_id += 1
                
                current_pattern_start = onset
                current_pattern_onsets = [onset]
        
        # Add final pattern
        if len(current_pattern_onsets) >= 2:
            pattern = self._create_pattern(
                pattern_id, current_pattern_start,
                current_pattern_onsets[-1], current_pattern_onsets,
                tempo
            )
            patterns.append(pattern)
        
        return patterns
    
    def _create_pattern(self, pattern_id: int, start_time: float, end_time: float, 
                       onsets: List[float], tempo: float) -> RhythmicPattern:
        """Create a rhythmic pattern from onset data"""
        
        duration = end_time - start_time
        density = len(onsets) / duration if duration > 0 else 0
        
        # Calculate syncopation
        syncopation = self._calculate_pattern_syncopation(onsets, tempo)
        
        # Determine pattern type
        pattern_type = self._classify_pattern_type(onsets, tempo)
        
        # Calculate confidence based on pattern consistency
        confidence = self._calculate_pattern_confidence(onsets, tempo)
        
        return RhythmicPattern(
            pattern_id=f"pattern_{pattern_id}",
            start_time=start_time,
            end_time=end_time,
            tempo=tempo,
            density=density,
            syncopation=syncopation,
            meter="4/4",  # Default, could be improved
            pattern_type=pattern_type,
            confidence=confidence
        )
    
    def _estimate_tempo_from_onsets(self, audio: np.ndarray, sr: int) -> float:
        """Fallback tempo estimation from onset intervals"""
        try:
            # Detect onsets
            onsets = librosa.onset.onset_detect(
                y=audio, sr=sr,
                hop_length=self.hop_length,
                units='time'
            )
            
            if len(onsets) < 4:
                return 120.0  # Default fallback
            
            # Calculate intervals between onsets
            intervals = np.diff(onsets)
            
            # Filter out very short intervals (likely artifacts)
            intervals = intervals[intervals > 0.1]  # Minimum 100ms
            
            if len(intervals) < 2:
                return 120.0
            
            # Find common interval patterns
            # Look for intervals that correspond to beat subdivisions
            common_intervals = []
            
            for interval in intervals:
                # Check if interval corresponds to common beat subdivisions
                for subdivision in [1, 2, 4, 8]:  # Whole, half, quarter, eighth notes
                    tempo = 60.0 / (interval * subdivision)
                    if 60 <= tempo <= 200:
                        common_intervals.append(tempo)
            
            if common_intervals:
                # Use median tempo from common intervals
                return float(np.median(common_intervals))
            else:
                # Last resort: use median interval as quarter note
                median_interval = np.median(intervals)
                tempo = 60.0 / median_interval
                return max(60.0, min(200.0, tempo))
                
        except Exception as e:
            print(f"⚠️ Tempo estimation failed: {e}")
            return 120.0
    
    def _analyze_syncopation(self, onsets: np.ndarray, beats: np.ndarray) -> float:
        """Analyze syncopation in the rhythm"""
        if len(onsets) < 2 or len(beats) < 2:
            return 0.0
        
        syncopation_score = 0.0
        
        for onset in onsets:
            # Find nearest beat
            beat_distances = np.abs(beats - onset)
            nearest_beat_idx = np.argmin(beat_distances)
            nearest_beat = beats[nearest_beat_idx]
            
            # Calculate offset from beat
            offset = abs(onset - nearest_beat)
            
            # Syncopation increases with offset from beat
            if offset > 0.1:  # Significant offset
                syncopation_score += offset
        
        # Normalize by number of onsets
        return syncopation_score / len(onsets) if len(onsets) > 0 else 0.0
    
    def _detect_meter(self, beats: np.ndarray, tempo: float) -> str:
        """Detect musical meter (time signature)"""
        if len(beats) < 4:
            return "4/4"  # Default
        
        # Analyze beat intervals
        beat_intervals = np.diff(beats)
        avg_interval = np.mean(beat_intervals)
        
        # Look for patterns in beat strength (simplified)
        # In a real implementation, this would analyze beat strength patterns
        
        return "4/4"  # Default for now
    
    def _calculate_rhythmic_complexity(self, onsets: np.ndarray, beats: np.ndarray, tempo: float) -> float:
        """Calculate rhythmic complexity score"""
        if len(onsets) < 2:
            return 0.0
        
        # Factors contributing to complexity:
        # 1. Onset density
        # 2. Syncopation
        # 3. Tempo variation
        # 4. Pattern irregularity
        
        duration = onsets[-1] - onsets[0] if len(onsets) > 1 else 1.0
        density = len(onsets) / duration
        
        syncopation = self._analyze_syncopation(onsets, beats)
        
        # Calculate onset interval irregularity
        intervals = np.diff(onsets)
        if len(intervals) > 1:
            irregularity = np.std(intervals) / np.mean(intervals)
        else:
            irregularity = 0.0
        
        # Combine factors
        complexity = (density * 0.3 + syncopation * 0.4 + irregularity * 0.3)
        
        return min(complexity, 1.0)  # Normalize to 0-1
    
    def _calculate_density_profile(self, onsets: np.ndarray, beats: np.ndarray) -> np.ndarray:
        """Calculate rhythmic density over time"""
        if len(onsets) < 2:
            return np.array([0.5])
        
        # Create time windows
        duration = onsets[-1] - onsets[0]
        window_size = 2.0  # seconds
        num_windows = int(duration / window_size) + 1
        
        density_profile = np.zeros(num_windows)
        
        for i in range(num_windows):
            window_start = onsets[0] + i * window_size
            window_end = window_start + window_size
            
            # Count onsets in window
            onsets_in_window = np.sum((onsets >= window_start) & (onsets < window_end))
            density_profile[i] = onsets_in_window / window_size
        
        return density_profile
    
    def _calculate_pattern_syncopation(self, onsets: List[float], tempo: float) -> float:
        """Calculate syncopation for a specific pattern"""
        if len(onsets) < 2:
            return 0.0
        
        beat_interval = 60.0 / tempo
        syncopation = 0.0
        
        for onset in onsets:
            # Find nearest beat position
            beat_position = onset % beat_interval
            beat_offset = min(beat_position, beat_interval - beat_position)
            
            # Syncopation increases with offset from beat
            syncopation += beat_offset / beat_interval
        
        return syncopation / len(onsets)
    
    def _classify_pattern_type(self, onsets: List[float], tempo: float) -> str:
        """Classify the type of rhythmic pattern"""
        if len(onsets) < 2:
            return "simple"
        
        beat_interval = 60.0 / tempo
        
        # Analyze onset intervals
        intervals = np.diff(onsets)
        avg_interval = np.mean(intervals)
        
        # Classify based on interval patterns
        if avg_interval < beat_interval * 0.5:
            return "dense"
        elif avg_interval < beat_interval:
            return "moderate"
        elif avg_interval < beat_interval * 2:
            return "sparse"
        else:
            return "very_sparse"
    
    def _calculate_pattern_confidence(self, onsets: List[float], tempo: float) -> float:
        """Calculate confidence in pattern detection"""
        if len(onsets) < 2:
            return 0.0
        
        # Confidence based on:
        # 1. Number of onsets (more = more confident)
        # 2. Regularity of intervals
        # 3. Alignment with tempo
        
        beat_interval = 60.0 / tempo
        
        # Regularity score
        intervals = np.diff(onsets)
        if len(intervals) > 1:
            regularity = 1.0 - (np.std(intervals) / np.mean(intervals))
        else:
            regularity = 0.5
        
        # Tempo alignment score
        tempo_alignment = 0.0
        for interval in intervals:
            # Check how well interval aligns with beat subdivisions
            beat_ratio = interval / beat_interval
            if abs(beat_ratio - np.round(beat_ratio)) < 0.1:
                tempo_alignment += 1.0
        
        tempo_alignment /= len(intervals)
        
        # Combine factors
        confidence = (regularity * 0.5 + tempo_alignment * 0.5)
        
        return min(confidence, 1.0)
    
    def analyze_rational_structure(self, onset_times: np.ndarray) -> Optional[Dict]:
        """
        Analyze rhythmic sequence using rational ratio method
        
        This provides structural rhythm understanding beyond statistical features,
        finding the simplest rational relationships between delta times.
        
        Args:
            onset_times: Array of onset timestamps (in seconds)
            
        Returns:
            Dictionary with:
                - duration_pattern: Integer duration pattern (e.g., [2,1,1,2])
                - tempo: Subdiv tempo in BPM
                - pulse: Pulse subdivision
                - pulse_position: Phase offset
                - complexity: Barlow indigestability score
                - deviations: Per-event timing deviations
                - deviation_polarity: Quantized deviations (-1, 0, 1)
                - confidence: Analysis confidence
                - all_suggestions: All competing theories with scores
            
            Returns None if analysis fails or insufficient onsets
        """
        if len(onset_times) < 3:
            print("⚠️  Not enough onsets for rational structure analysis (need >= 3)")
            return None
        
        try:
            # Import ratio analyzer
            from rhythmic_engine.ratio_analyzer import RatioAnalyzer
            
            # Initialize analyzer with balanced complexity/deviation weights
            analyzer = RatioAnalyzer(
                complexity_weight=0.5,
                deviation_weight=0.5,
                simplify=True,
                div_limit=4
            )
            
            # Perform rational analysis
            result = analyzer.analyze(onset_times)
            
            return result
            
        except Exception as e:
            print(f"⚠️  Error in rational structure analysis: {e}")
            import traceback
            traceback.print_exc()
            return None


    def analyze_long_form_improvisation(self, 
                                       audio_file: str,
                                       section_duration: float = 60.0) -> List[Dict]:
        """
        Analyze long-form improvisation (10-20 minutes) with varied BPM across sections.
        
        Uses Brandtsegg rhythm ratios to detect tempo changes and section boundaries.
        Perfect for recordings like Daybreak.wav where tempo varies throughout.
        
        Philosophy: Instead of global tempo (meaningless average), analyze local tempo
        in each section and detect tempo relationships via rhythm ratios.
        
        Args:
            audio_file: Path to audio file (long recording 10-20 min)
            section_duration: Analysis window size in seconds (default 60s = 1-minute sections)
            
        Returns:
            List of section dictionaries with:
            - start_time, end_time: Section boundaries (seconds)
            - local_tempo: BPM in this section
            - dominant_ratios: Top 3 rhythm ratio patterns [[num, denom], ...]
            - tempo_change: Ratio relative to previous section (e.g., 1.5 = tempo increased 50%)
            - onset_density: Onsets per second
            - rhythmic_complexity: Barlow indigestability score
            - beat_strength: Average onset strength (structural pulse indicator)
        """
        from rhythmic_engine.ratio_analyzer import RatioAnalyzer
        
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
        
        print(f"🎭 Analyzing long-form structure: {audio_file}")
        print(f"   Section duration: {section_duration}s")
        
        # Load audio
        audio, sr = librosa.load(audio_file, sr=self.sample_rate)
        total_duration = len(audio) / sr
        
        print(f"   Total duration: {total_duration:.1f}s ({total_duration/60:.1f} minutes)")
        
        # Detect onsets across entire track
        onset_env = librosa.onset.onset_strength(
            y=audio, sr=sr, hop_length=self.hop_length
        )
        
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=sr,
            hop_length=self.hop_length,
            units='frames',
            backtrack=True,
            delta=0.05  # Use same threshold as SMART tempo detection
        )
        
        onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=self.hop_length)
        onset_strengths = onset_env[onset_frames]
        
        print(f"   Detected {len(onset_times)} onsets across full recording")
        
        # Segment into analysis windows
        sections = []
        ratio_analyzer = RatioAnalyzer(complexity_weight=0.5, deviation_weight=0.5, div_limit=4)
        
        for start_time in np.arange(0, total_duration, section_duration):
            end_time = min(start_time + section_duration, total_duration)
            
            # Get onsets in this window
            window_mask = (onset_times >= start_time) & (onset_times < end_time)
            window_onsets = onset_times[window_mask]
            window_strengths = onset_strengths[window_mask]
            
            if len(window_onsets) < 5:
                print(f"   Section {start_time:.0f}s-{end_time:.0f}s: Too sparse, skipping")
                continue
            
            # Compute local tempo using SMART method (slow/fast clustering with structural pulse bias)
            intervals = np.diff(window_onsets)
            
            # Weight intervals by onset strength
            interval_weights = []
            for i in range(len(intervals)):
                weight = np.sqrt(window_strengths[i] * window_strengths[i + 1])
                interval_weights.append(weight)
            interval_weights = np.array(interval_weights)
            
            # SMART method: Check slow and fast ranges with structural pulse bias
            # Slow tempo (45-75 BPM = 0.8-1.35s intervals)
            slow_mask = (intervals >= 0.8) & (intervals <= 1.35)
            slow_intervals = intervals[slow_mask]
            slow_weights = interval_weights[slow_mask]
            
            # Fast tempo (85-120 BPM = 0.5-0.7s intervals)
            fast_mask = (intervals >= 0.5) & (intervals <= 0.7)
            fast_intervals = intervals[fast_mask]
            fast_weights = interval_weights[fast_mask]
            
            local_tempo = 90.0  # Fallback
            
            if len(slow_intervals) >= 3:
                slow_confidence = float(np.sum(slow_weights))
                sorted_idx = np.argsort(slow_intervals)
                cumsum = np.cumsum(slow_weights[sorted_idx])
                median_idx = np.searchsorted(cumsum, cumsum[-1] / 2) if cumsum[-1] > 0 else 0
                slow_tempo = 60.0 / slow_intervals[sorted_idx[median_idx]]
            else:
                slow_confidence = 0.0
                slow_tempo = None
            
            if len(fast_intervals) >= 3:
                fast_confidence = float(np.sum(fast_weights))
                sorted_idx = np.argsort(fast_intervals)
                cumsum = np.cumsum(fast_weights[sorted_idx])
                median_idx = np.searchsorted(cumsum, cumsum[-1] / 2) if cumsum[-1] > 0 else 0
                fast_tempo = 60.0 / fast_intervals[sorted_idx[median_idx]]
            else:
                fast_confidence = 0.0
                fast_tempo = None
            
            # Apply structural pulse bias (favor slow tempo)
            if slow_confidence > 0 and fast_confidence > 0:
                if slow_confidence > fast_confidence / 1.5:
                    local_tempo = slow_tempo
                else:
                    local_tempo = fast_tempo
            elif slow_tempo:
                local_tempo = slow_tempo
            elif fast_tempo:
                local_tempo = fast_tempo
            
            # Analyze rhythm ratios using Brandtsegg
            dominant_ratios = []
            rhythmic_complexity = 0.5  # Default
            
            try:
                # Brandtsegg analysis requires at least 3 onsets with non-zero intervals
                if len(window_onsets) >= 3:
                    # Check for valid intervals
                    test_intervals = np.diff(window_onsets)
                    if len(test_intervals) > 0 and np.all(test_intervals > 0):
                        ratio_result = ratio_analyzer.analyze(window_onsets)
                        
                        # Extract top 3 ratio suggestions
                        for sug in ratio_result['all_suggestions'][:3]:
                            # Duration pattern tells us the rhythmic structure
                            # E.g., [2, 2, 3] = two equal durations then a longer one
                            dominant_ratios.append(sug['duration_pattern'])
                        
                        rhythmic_complexity = ratio_result['complexity']
                    else:
                        dominant_ratios = [[1, 1]]  # Fallback: steady pulse
            except (ZeroDivisionError, ValueError, RuntimeError) as e:
                # Brandtsegg can fail on edge cases (sparse onsets, div-by-zero)
                dominant_ratios = [[1, 1]]  # Fallback: steady pulse
            
            # Compute onset density
            onset_density = len(window_onsets) / (end_time - start_time)
            
            # Compute average onset strength (structural pulse indicator)
            beat_strength = float(np.mean(window_strengths)) if len(window_strengths) > 0 else 0.0
            
            # Compare to previous section (tempo change detection)
            tempo_change = None
            if sections:
                prev_tempo = sections[-1]['local_tempo']
                tempo_change = local_tempo / prev_tempo
            
            section_data = {
                'start_time': float(start_time),
                'end_time': float(end_time),
                'local_tempo': float(local_tempo),
                'dominant_ratios': dominant_ratios,
                'tempo_change': float(tempo_change) if tempo_change else None,
                'onset_density': float(onset_density),
                'rhythmic_complexity': float(rhythmic_complexity),
                'beat_strength': float(beat_strength),
                'num_onsets': int(len(window_onsets))
            }
            
            sections.append(section_data)
            
            # Progress indicator
            tempo_change_str = f" (→{tempo_change:.2f}x)" if tempo_change else ""
            print(f"   Section {len(sections)}: {start_time:.0f}s-{end_time:.0f}s: "
                  f"{local_tempo:.1f} BPM{tempo_change_str}, "
                  f"complexity {rhythmic_complexity:.2f}, "
                  f"density {onset_density:.2f} onsets/s")
        
        print(f"✅ Long-form analysis complete: {len(sections)} sections detected")
        
        return sections


def main():
    """Test the heavy rhythmic analyzer"""
    analyzer = HeavyRhythmicAnalyzer()
    
    # Test with a sample audio file
    audio_file = "input_audio/Grab-a-hold.mp3"
    
    if os.path.exists(audio_file):
        print(f"Testing rhythmic analysis with: {audio_file}")
        analysis = analyzer.analyze_rhythmic_structure(audio_file)
        
        print(f"\n🎵 Rhythmic Analysis Results:")
        tempo_val = analysis.tempo.item() if hasattr(analysis.tempo, 'item') else analysis.tempo
        syncopation_val = analysis.syncopation_score.item() if hasattr(analysis.syncopation_score, 'item') else analysis.syncopation_score
        complexity_val = analysis.rhythmic_complexity.item() if hasattr(analysis.rhythmic_complexity, 'item') else analysis.rhythmic_complexity
        print(f"   Tempo: {float(tempo_val):.1f} BPM")
        print(f"   Meter: {analysis.meter}")
        print(f"   Syncopation: {float(syncopation_val):.3f}")
        print(f"   Complexity: {float(complexity_val):.3f}")
        print(f"   Patterns detected: {len(analysis.patterns)}")
        
        for i, pattern in enumerate(analysis.patterns[:5]):  # Show first 5
            start_val = pattern.start_time.item() if hasattr(pattern.start_time, 'item') else pattern.start_time
            end_val = pattern.end_time.item() if hasattr(pattern.end_time, 'item') else pattern.end_time
            conf_val = pattern.confidence.item() if hasattr(pattern.confidence, 'item') else pattern.confidence
            print(f"   Pattern {i+1}: {pattern.pattern_type} "
                  f"({float(start_val):.1f}-{float(end_val):.1f}s, "
                  f"conf: {float(conf_val):.3f})")
    else:
        print(f"Audio file not found: {audio_file}")


if __name__ == "__main__":
    main()
