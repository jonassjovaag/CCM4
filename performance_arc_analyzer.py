#!/usr/bin/env python3
"""
Performance Arc Analyzer
Analyzes musical structure and evolution from training audio to create performance arcs
"""

import os
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import librosa
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import pearsonr

@dataclass
class MusicalPhase:
    """Represents a phase in the musical arc"""
    start_time: float
    end_time: float
    phase_type: str  # 'intro', 'development', 'climax', 'resolution'
    engagement_level: float  # 0.0 to 1.0
    instrument_roles: Dict[str, str]  # 'lead', 'accompaniment', 'silent'
    musical_density: float  # 0.0 to 1.0
    dynamic_level: float  # 0.0 to 1.0
    silence_ratio: float  # 0.0 to 1.0

@dataclass
class PerformanceArc:
    """Complete performance arc extracted from training audio"""
    total_duration: float
    phases: List[MusicalPhase]
    overall_engagement_curve: List[float]
    instrument_evolution: Dict[str, List[float]]
    silence_patterns: List[Tuple[float, float]]  # (start_time, duration)
    theme_development: List[Dict[str, Any]]
    dynamic_evolution: List[float]

class PerformanceArcAnalyzer:
    """Analyzes training audio to extract performance arcs"""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.phase_duration_threshold = 30.0  # Minimum phase duration in seconds
        
    def analyze_audio_file(self, audio_path: str) -> PerformanceArc:
        """Analyze audio file and extract performance arc"""
        print(f"🎵 Analyzing performance arc from: {audio_path}")
        
        # Load audio
        y, sr = librosa.load(audio_path, sr=self.sample_rate)
        duration = len(y) / sr
        
        print(f"📊 Audio duration: {duration:.2f} seconds")
        
        # Extract features
        features = self._extract_audio_features(y, sr)
        
        # Analyze musical structure
        phases = self._identify_musical_phases(features, duration)
        
        # Analyze instrument evolution
        instrument_evolution = self._analyze_instrument_evolution(features, duration)
        
        # Analyze silence patterns
        silence_patterns = self._identify_silence_patterns(features, duration)
        
        # Analyze theme development
        theme_development = self._analyze_theme_development(features, duration)
        
        # Create overall engagement curve
        engagement_curve = self._create_engagement_curve(features, duration)
        
        # Create dynamic evolution
        dynamic_evolution = self._create_dynamic_evolution(features, duration)
        
        return PerformanceArc(
            total_duration=duration,
            phases=phases,
            overall_engagement_curve=engagement_curve,
            instrument_evolution=instrument_evolution,
            silence_patterns=silence_patterns,
            theme_development=theme_development,
            dynamic_evolution=dynamic_evolution
        )
    
    def _extract_audio_features(self, y: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """Extract comprehensive audio features"""
        print("🔍 Extracting audio features...")
        
        # Basic features
        rms = librosa.feature.rms(y=y)[0]
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
        
        # Harmonic and percussive components
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        
        # Tempo and beat tracking
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beats, sr=sr)
        
        # Onset detection
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        
        # MFCC features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        
        # Spectral contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        
        return {
            'rms': rms,
            'spectral_centroid': spectral_centroid,
            'spectral_rolloff': spectral_rolloff,
            'zero_crossing_rate': zero_crossing_rate,
            'harmonic': y_harmonic,
            'percussive': y_percussive,
            'tempo': tempo,
            'beat_times': beat_times,
            'onset_times': onset_times,
            'mfcc': mfcc,
            'chroma': chroma,
            'spectral_contrast': spectral_contrast,
            'duration': len(y) / sr
        }
    
    def _identify_musical_phases(self, features: Dict[str, np.ndarray], duration: float) -> List[MusicalPhase]:
        """Identify musical phases based on feature analysis"""
        print("🎼 Identifying musical phases...")
        
        phases = []
        
        # Analyze RMS energy for dynamic evolution
        rms = features['rms']
        rms_smooth = signal.savgol_filter(rms, 51, 3)  # Smooth the curve
        
        # Find peaks and valleys in the energy curve
        peaks, _ = signal.find_peaks(rms_smooth, height=np.mean(rms_smooth))
        valleys, _ = signal.find_peaks(-rms_smooth, height=-np.mean(rms_smooth))
        
        # Convert frame indices to time
        frame_times = np.linspace(0, duration, len(rms_smooth))
        peak_times = frame_times[peaks]
        valley_times = frame_times[valleys]
        
        # Define phase boundaries
        phase_boundaries = [0.0]
        
        # Add significant peaks and valleys as phase boundaries
        for peak_time in peak_times:
            if peak_time > phase_boundaries[-1] + self.phase_duration_threshold:
                phase_boundaries.append(peak_time)
        
        for valley_time in valley_times:
            if valley_time > phase_boundaries[-1] + self.phase_duration_threshold:
                phase_boundaries.append(valley_time)
        
        # Ensure we have the end
        if phase_boundaries[-1] < duration - self.phase_duration_threshold:
            phase_boundaries.append(duration)
        
        # Create phases
        phase_types = ['intro', 'development', 'climax', 'resolution']
        for i in range(len(phase_boundaries) - 1):
            start_time = phase_boundaries[i]
            end_time = phase_boundaries[i + 1]
            phase_type = phase_types[min(i, len(phase_types) - 1)]
            
            # Calculate phase characteristics
            phase_start_frame = int(start_time * len(rms_smooth) / duration)
            phase_end_frame = int(end_time * len(rms_smooth) / duration)
            phase_rms = rms_smooth[phase_start_frame:phase_end_frame]
            
            engagement_level = np.mean(phase_rms) / np.max(rms_smooth)
            musical_density = len(features['onset_times'][
                (features['onset_times'] >= start_time) & 
                (features['onset_times'] <= end_time)
            ]) / (end_time - start_time)
            dynamic_level = np.std(phase_rms) / np.std(rms_smooth)
            silence_ratio = self._calculate_silence_ratio(phase_rms)
            
            # Estimate instrument roles (simplified)
            instrument_roles = self._estimate_instrument_roles(features, start_time, end_time)
            
            phases.append(MusicalPhase(
                start_time=start_time,
                end_time=end_time,
                phase_type=phase_type,
                engagement_level=engagement_level,
                instrument_roles=instrument_roles,
                musical_density=musical_density,
                dynamic_level=dynamic_level,
                silence_ratio=silence_ratio
            ))
        
        return phases
    
    def _analyze_instrument_evolution(self, features: Dict[str, np.ndarray], duration: float) -> Dict[str, List[float]]:
        """Analyze how different instrument characteristics evolve over time"""
        print("🎹 Analyzing instrument evolution...")
        
        # Create time windows
        window_size = 10.0  # 10-second windows
        num_windows = int(duration / window_size)
        
        instrument_evolution = {
            'drums': [],
            'piano': [],
            'bass': [],
            'overall': []
        }
        
        for i in range(num_windows):
            start_time = i * window_size
            end_time = min((i + 1) * window_size, duration)
            
            # Calculate instrument presence in this window
            window_start_frame = int(start_time * len(features['rms']) / duration)
            window_end_frame = int(end_time * len(features['rms']) / duration)
            
            # Analyze percussive vs harmonic content
            percussive_energy = np.mean(features['percussive'][window_start_frame:window_end_frame] ** 2)
            harmonic_energy = np.mean(features['harmonic'][window_start_frame:window_end_frame] ** 2)
            
            # Estimate instrument presence
            drums_presence = percussive_energy / (percussive_energy + harmonic_energy + 1e-6)
            piano_presence = harmonic_energy / (percussive_energy + harmonic_energy + 1e-6)
            bass_presence = self._estimate_bass_presence(features, start_time, end_time)
            overall_presence = (drums_presence + piano_presence + bass_presence) / 3
            
            instrument_evolution['drums'].append(drums_presence)
            instrument_evolution['piano'].append(piano_presence)
            instrument_evolution['bass'].append(bass_presence)
            instrument_evolution['overall'].append(overall_presence)
        
        return instrument_evolution
    
    def _identify_silence_patterns(self, features: Dict[str, np.ndarray], duration: float) -> List[Tuple[float, float]]:
        """Identify strategic silence patterns"""
        print("🔇 Identifying silence patterns...")
        
        # Find low-energy regions
        rms = features['rms']
        silence_threshold = np.percentile(rms, 20)  # Bottom 20% of energy
        
        silence_regions = []
        in_silence = False
        silence_start = 0
        
        for i, energy in enumerate(rms):
            time = i * duration / len(rms)
            
            if energy < silence_threshold and not in_silence:
                in_silence = True
                silence_start = time
            elif energy >= silence_threshold and in_silence:
                in_silence = False
                silence_duration = time - silence_start
                if silence_duration > 2.0:  # Only significant silences
                    silence_regions.append((silence_start, silence_duration))
        
        return silence_regions
    
    def _analyze_theme_development(self, features: Dict[str, np.ndarray], duration: float) -> List[Dict[str, Any]]:
        """Analyze theme development and variation"""
        print("🎵 Analyzing theme development...")
        
        # Analyze chroma features for harmonic development
        chroma = features['chroma']
        
        # Find recurring harmonic patterns
        theme_development = []
        
        # Simple theme analysis based on chroma similarity
        window_size = 8.0  # 8-second windows
        num_windows = int(duration / window_size)
        
        for i in range(num_windows):
            start_time = i * window_size
            end_time = min((i + 1) * window_size, duration)
            
            # Calculate chroma profile for this window
            window_start_frame = int(start_time * chroma.shape[1] / duration)
            window_end_frame = int(end_time * chroma.shape[1] / duration)
            window_chroma = chroma[:, window_start_frame:window_end_frame]
            
            # Calculate harmonic complexity
            chroma_mean = np.mean(window_chroma, axis=1)
            harmonic_complexity = np.std(chroma_mean)
            
            # Calculate tonal stability
            tonal_stability = np.max(chroma_mean) / (np.sum(chroma_mean) + 1e-6)
            
            theme_development.append({
                'start_time': start_time,
                'end_time': end_time,
                'harmonic_complexity': harmonic_complexity,
                'tonal_stability': tonal_stability,
                'chroma_profile': chroma_mean.tolist()
            })
        
        return theme_development
    
    def _create_engagement_curve(self, features: Dict[str, np.ndarray], duration: float) -> List[float]:
        """Create overall engagement curve"""
        print("📈 Creating engagement curve...")
        
        # Combine multiple features for engagement
        rms = features['rms']
        spectral_centroid = features['spectral_centroid']
        onset_density = self._calculate_onset_density(features, duration)
        
        # Normalize features
        rms_norm = (rms - np.min(rms)) / (np.max(rms) - np.min(rms) + 1e-6)
        centroid_norm = (spectral_centroid - np.min(spectral_centroid)) / (np.max(spectral_centroid) - np.min(spectral_centroid) + 1e-6)
        
        # Combine into engagement curve
        engagement = (rms_norm + centroid_norm + onset_density) / 3
        
        return engagement.tolist()
    
    def _create_dynamic_evolution(self, features: Dict[str, np.ndarray], duration: float) -> List[float]:
        """Create dynamic evolution curve"""
        print("🎚️ Creating dynamic evolution...")
        
        rms = features['rms']
        
        # Smooth the RMS curve
        rms_smooth = signal.savgol_filter(rms, 51, 3)
        
        # Normalize
        dynamic_evolution = (rms_smooth - np.min(rms_smooth)) / (np.max(rms_smooth) - np.min(rms_smooth) + 1e-6)
        
        return dynamic_evolution.tolist()
    
    def _calculate_silence_ratio(self, rms_segment: np.ndarray) -> float:
        """Calculate silence ratio in a segment"""
        silence_threshold = np.percentile(rms_segment, 30)
        silence_frames = np.sum(rms_segment < silence_threshold)
        return silence_frames / len(rms_segment)
    
    def _estimate_instrument_roles(self, features: Dict[str, np.ndarray], start_time: float, end_time: float) -> Dict[str, str]:
        """Estimate instrument roles in a time segment"""
        # Simplified role estimation
        return {
            'drums': 'accompaniment',
            'piano': 'lead',
            'bass': 'accompaniment'
        }
    
    def _estimate_bass_presence(self, features: Dict[str, np.ndarray], start_time: float, end_time: float) -> float:
        """Estimate bass presence in a time segment"""
        # Simplified bass estimation based on low-frequency content
        return 0.5  # Placeholder
    
    def _calculate_onset_density(self, features: Dict[str, np.ndarray], duration: float) -> np.ndarray:
        """Calculate onset density over time"""
        onset_times = features['onset_times']
        
        # Create time bins
        num_bins = len(features['rms'])
        time_bins = np.linspace(0, duration, num_bins)
        
        # Count onsets in each bin
        onset_density = np.zeros(num_bins)
        for onset_time in onset_times:
            bin_idx = np.argmin(np.abs(time_bins - onset_time))
            onset_density[bin_idx] += 1
        
        # Normalize
        if np.max(onset_density) > 0:
            onset_density = onset_density / np.max(onset_density)
        
        return onset_density
    
    def save_arc(self, arc: PerformanceArc, output_path: str):
        """Save performance arc to JSON file"""
        print(f"💾 Saving performance arc to: {output_path}")
        
        def convert_numpy(obj):
            """Convert numpy types to Python native types"""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        # Convert phases to dict and handle numpy types
        phases_dict = []
        for phase in arc.phases:
            phase_dict = asdict(phase)
            phases_dict.append(convert_numpy(phase_dict))
        
        arc_dict = {
            'total_duration': float(arc.total_duration),
            'phases': phases_dict,
            'overall_engagement_curve': convert_numpy(arc.overall_engagement_curve),
            'instrument_evolution': convert_numpy(arc.instrument_evolution),
            'silence_patterns': convert_numpy(arc.silence_patterns),
            'theme_development': convert_numpy(arc.theme_development),
            'dynamic_evolution': convert_numpy(arc.dynamic_evolution)
        }
        
        with open(output_path, 'w') as f:
            json.dump(arc_dict, f, indent=2)
        
        print(f"✅ Performance arc saved successfully")
    
    def load_arc(self, input_path: str) -> PerformanceArc:
        """Load performance arc from JSON file"""
        print(f"📂 Loading performance arc from: {input_path}")
        
        with open(input_path, 'r') as f:
            arc_dict = json.load(f)
        
        # Reconstruct phases
        phases = []
        for phase_dict in arc_dict['phases']:
            phases.append(MusicalPhase(**phase_dict))
        
        return PerformanceArc(
            total_duration=arc_dict['total_duration'],
            phases=phases,
            overall_engagement_curve=arc_dict['overall_engagement_curve'],
            instrument_evolution=arc_dict['instrument_evolution'],
            silence_patterns=arc_dict['silence_patterns'],
            theme_development=arc_dict['theme_development'],
            dynamic_evolution=arc_dict['dynamic_evolution']
        )
    
    def visualize_arc(self, arc: PerformanceArc, output_path: str = None):
        """Visualize the performance arc"""
        print("📊 Creating performance arc visualization...")
        
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
        
        # Time axis
        time_axis = np.linspace(0, arc.total_duration, len(arc.overall_engagement_curve))
        
        # 1. Overall engagement curve
        axes[0].plot(time_axis, arc.overall_engagement_curve, 'b-', linewidth=2)
        axes[0].set_title('Overall Engagement Curve')
        axes[0].set_ylabel('Engagement Level')
        axes[0].grid(True, alpha=0.3)
        
        # 2. Instrument evolution
        for instrument, evolution in arc.instrument_evolution.items():
            if instrument != 'overall':
                # Create time axis for instrument evolution (shorter)
                evolution_time_axis = np.linspace(0, arc.total_duration, len(evolution))
                axes[1].plot(evolution_time_axis, evolution, label=instrument, linewidth=2)
        axes[1].set_title('Instrument Evolution')
        axes[1].set_ylabel('Presence Level')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 3. Dynamic evolution
        axes[2].plot(time_axis, arc.dynamic_evolution, 'g-', linewidth=2)
        axes[2].set_title('Dynamic Evolution')
        axes[2].set_ylabel('Dynamic Level')
        axes[2].grid(True, alpha=0.3)
        
        # 4. Phases
        phase_colors = {'intro': 'blue', 'development': 'green', 'climax': 'red', 'resolution': 'orange'}
        for phase in arc.phases:
            axes[3].axvspan(phase.start_time, phase.end_time, 
                          alpha=0.3, color=phase_colors.get(phase.phase_type, 'gray'),
                          label=phase.phase_type)
        axes[3].set_title('Musical Phases')
        axes[3].set_xlabel('Time (seconds)')
        axes[3].set_ylabel('Phase')
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"📊 Visualization saved to: {output_path}")
        else:
            plt.show()

def main():
    """Test the performance arc analyzer"""
    analyzer = PerformanceArcAnalyzer()
    
    # Analyze Itzama.wav
    audio_path = "input_audio/Itzama.wav"
    if os.path.exists(audio_path):
        arc = analyzer.analyze_audio_file(audio_path)
        
        # Save the arc
        output_path = "ai_learning_data/itzama_performance_arc.json"
        analyzer.save_arc(arc, output_path)
        
        # Create visualization
        viz_path = "ai_learning_data/itzama_performance_arc.png"
        analyzer.visualize_arc(arc, viz_path)
        
        print(f"\n🎵 Performance Arc Analysis Complete!")
        print(f"📊 Total duration: {arc.total_duration:.2f} seconds")
        print(f"🎼 Number of phases: {len(arc.phases)}")
        print(f"🔇 Silence patterns: {len(arc.silence_patterns)}")
        print(f"🎹 Theme development segments: {len(arc.theme_development)}")
        
    else:
        print(f"❌ Audio file not found: {audio_path}")

if __name__ == "__main__":
    main()
