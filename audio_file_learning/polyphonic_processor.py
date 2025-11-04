# audio_file_learning/polyphonic_processor.py
# Enhanced Audio File Processor with Polyphonic Support
# Implements multi-pitch detection, chord recognition, and complete feature extraction

import os
import sys
import time
import numpy as np
import librosa
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from scipy import signal
from scipy.fft import fft, fftfreq

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from listener.jhs_listener_core import Event


@dataclass
class AudioFileInfo:
    """Information about processed audio file"""
    duration: float
    sample_rate: int
    format: str
    channels: int


@dataclass
class MultiPitchFrame:
    """Represents a frame with multiple detected pitches"""
    timestamp: float
    pitches: List[float]  # Multiple fundamental frequencies
    amplitudes: List[float]  # Amplitudes for each pitch
    midi_notes: List[int]  # MIDI notes for each pitch
    cents_deviations: List[float]  # Cents deviation for each pitch
    chord_quality: str  # e.g., "major", "minor", "diminished", "augmented"
    root_note: int  # Root note of the chord (MIDI)
    spectral_features: Dict[str, float]  # Spectral characteristics
    temporal_features: Dict[str, float]  # Temporal characteristics
    melodic_voice: Optional[int] = None  # Index of the melodic voice (if detected)
    voice_roles: Optional[List[str]] = None  # Role tags: "melody", "harmony", "bass"


class MelodicSalienceDetector:
    """
    Detects melodic voice from polyphonic audio using multiple salience heuristics
    
    Heuristics:
    1. Highest pitch (top-note melody - common in piano)
    2. Loudest pitch (amplitude-based salience)
    3. Pitch continuity (smooth melodic contours)
    4. Spectral brightness (melody often brighter)
    5. Duration (sustained notes = melody)
    """
    
    def __init__(self, history_length: int = 10):
        """
        Initialize melodic salience detector
        
        Args:
            history_length: Number of frames to keep for continuity analysis
        """
        self.history_length = history_length
        self.pitch_history = []  # Recent melodic pitches for continuity
        self.last_melodic_midi = None
        
    def detect_melodic_voice(self, 
                            pitches: List[float],
                            amplitudes: List[float],
                            midi_notes: List[int],
                            spectral_centroid: float) -> Tuple[int, List[str]]:
        """
        Detect which voice is melodic using multiple heuristics
        
        Args:
            pitches: List of detected pitches (Hz)
            amplitudes: List of amplitudes for each pitch
            midi_notes: List of MIDI notes
            spectral_centroid: Spectral centroid of the frame
            
        Returns:
            (melodic_voice_index, voice_roles)
            melodic_voice_index: Index of the most melodic voice
            voice_roles: List of role tags for each voice
        """
        if not pitches or len(pitches) == 0:
            return (0, ["unknown"])
        
        if len(pitches) == 1:
            # Single voice - it's the melody
            self.last_melodic_midi = midi_notes[0]
            self.pitch_history.append(midi_notes[0])
            if len(self.pitch_history) > self.history_length:
                self.pitch_history.pop(0)
            return (0, ["melody"])
        
        # Multiple voices - apply heuristics
        scores = [0.0] * len(pitches)
        
        # Heuristic 1: Highest pitch (weight: 3.0)
        # Top note is often melody in piano music
        highest_idx = np.argmax(midi_notes)
        scores[highest_idx] += 3.0
        
        # Heuristic 2: Loudest pitch (weight: 2.0)
        # Melody is often emphasized dynamically
        loudest_idx = np.argmax(amplitudes)
        scores[loudest_idx] += 2.0
        
        # Heuristic 3: Pitch continuity (weight: 4.0)
        # Melody moves smoothly, harmony jumps
        if self.last_melodic_midi is not None:
            for i, midi in enumerate(midi_notes):
                interval = abs(midi - self.last_melodic_midi)
                if interval <= 2:  # Step motion (very melodic)
                    scores[i] += 4.0
                elif interval <= 5:  # Small leap (melodic)
                    scores[i] += 2.0
                elif interval <= 12:  # Octave or less (possible)
                    scores[i] += 1.0
                # Large leaps get no bonus (more likely harmony)
        
        # Heuristic 4: Spectral brightness (weight: 1.5)
        # Melody often has brighter timbre than bass/harmony
        if spectral_centroid > 3000:  # Bright sound
            # Give bonus to higher pitches
            for i, midi in enumerate(midi_notes):
                if midi >= 60:  # C4 and above
                    scores[i] += 1.5
        
        # Heuristic 5: Avoid bass register (weight: -2.0)
        # Lowest note is likely bass, not melody
        lowest_idx = np.argmin(midi_notes)
        if len(pitches) > 2:  # Only penalize if there are other options
            scores[lowest_idx] -= 2.0
        
        # Select melodic voice
        melodic_idx = int(np.argmax(scores))
        
        # Assign voice roles
        voice_roles = []
        for i, midi in enumerate(midi_notes):
            if i == melodic_idx:
                voice_roles.append("melody")
            elif i == lowest_idx and len(pitches) > 2:
                voice_roles.append("bass")
            else:
                voice_roles.append("harmony")
        
        # Update history
        self.last_melodic_midi = midi_notes[melodic_idx]
        self.pitch_history.append(midi_notes[melodic_idx])
        if len(self.pitch_history) > self.history_length:
            self.pitch_history.pop(0)
        
        return (melodic_idx, voice_roles)


class PolyphonicAudioProcessor:
    """
    Enhanced Audio File Processor with Polyphonic Support
    
    Features:
    - Multi-pitch detection using HPS (Harmonic Product Spectrum)
    - Chord recognition and analysis
    - Complete feature extraction including cents, IOI, attack/release
    - Spectral and temporal feature analysis
    """
    
    def __init__(self, 
                 sample_rate: int = 44100,
                 hop_length: int = 256,
                 frame_length: int = 2048,
                 max_events: Optional[int] = None,
                 max_pitches: int = 4,
                 min_pitch_amplitude: float = 0.1):
        """
        Initialize Polyphonic Audio Processor
        
        Args:
            sample_rate: Sample rate for processing
            hop_length: Hop length for frame processing
            frame_length: Frame length for analysis
            max_events: Maximum number of events to process
            max_pitches: Maximum number of pitches to detect per frame
            min_pitch_amplitude: Minimum amplitude threshold for pitch detection
        """
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.frame_length = frame_length
        self.max_events = max_events
        self.max_pitches = max_pitches
        self.min_pitch_amplitude = min_pitch_amplitude
        
        # Audio processing parameters
        self.fmin = 40.0
        self.fmax = 8000.0
        self.level_db_threshold = -40.0
        
        # Feature extraction parameters
        self.attack_threshold = 0.1  # 10% of peak for attack detection
        self.release_threshold = 0.1  # 10% of peak for release detection
        
        # IOI calculation
        self.last_onset_time = 0.0
        self.onset_history = []
        
        # Melodic salience detection
        self.melodic_detector = MelodicSalienceDetector(history_length=10)
        
        print(f"🎵 Polyphonic Audio Processor initialized:")
        print(f"   Sample Rate: {sample_rate}Hz")
        print(f"   Frame Length: {frame_length}")
        print(f"   Hop Length: {hop_length}")
        print(f"   Max Pitches: {max_pitches}")
        print(f"   Min Pitch Amplitude: {min_pitch_amplitude}")
        print(f"   Melodic Salience Detection: Enabled")
    
    def extract_multiple_pitches(self, frame: np.ndarray) -> Tuple[List[float], List[float]]:
        """
        Extract multiple fundamental frequencies using HPS (Harmonic Product Spectrum)
        
        Args:
            frame: Audio frame
            
        Returns:
            Tuple of (pitches, amplitudes)
        """
        try:
            # Apply window function
            windowed_frame = frame * np.hanning(len(frame))
            
            # Compute FFT
            fft_frame = fft(windowed_frame)
            magnitude_spectrum = np.abs(fft_frame[:len(fft_frame)//2])
            frequencies = fftfreq(len(frame), 1/self.sample_rate)[:len(frame)//2]
            
            # Harmonic Product Spectrum for multi-pitch detection
            pitches = []
            amplitudes = []
            
            # Use HPS with different harmonic numbers
            harmonic_numbers = [1, 2, 3, 4]
            hps_spectrum = np.ones_like(magnitude_spectrum)
            
            for h in harmonic_numbers:
                # Downsample spectrum by harmonic number
                downsampled = magnitude_spectrum[::h]
                # Pad or truncate to match original length
                if len(downsampled) < len(hps_spectrum):
                    padded = np.zeros_like(hps_spectrum)
                    padded[:len(downsampled)] = downsampled
                    downsampled = padded
                else:
                    downsampled = downsampled[:len(hps_spectrum)]
                
                hps_spectrum *= downsampled
            
            # Find peaks in HPS spectrum
            peaks, properties = signal.find_peaks(hps_spectrum, 
                                                height=np.max(hps_spectrum) * self.min_pitch_amplitude,
                                                distance=int(self.sample_rate / self.fmax))  # Minimum distance between peaks
            
            # Convert peak frequencies to pitches
            for peak_idx in peaks:
                if len(pitches) >= self.max_pitches:
                    break
                    
                freq = frequencies[peak_idx]
                if self.fmin <= freq <= self.fmax:
                    amplitude = hps_spectrum[peak_idx]
                    pitches.append(freq)
                    amplitudes.append(amplitude)
            
            # Sort by amplitude (strongest first)
            if pitches:
                sorted_indices = np.argsort(amplitudes)[::-1]
                pitches = [pitches[i] for i in sorted_indices]
                amplitudes = [amplitudes[i] for i in sorted_indices]
            
            return pitches[:self.max_pitches], amplitudes[:self.max_pitches]
            
        except Exception as e:
            print(f"Error in multi-pitch detection: {e}")
            return [], []
    
    def analyze_chord(self, pitches: List[float], amplitudes: List[float]) -> Tuple[str, int]:
        """
        Analyze chord quality and root note from detected pitches
        
        Args:
            pitches: List of fundamental frequencies
            amplitudes: List of amplitudes for each pitch
            
        Returns:
            Tuple of (chord_quality, root_note_midi)
        """
        if len(pitches) < 2:
            return "single", int(self._freq_to_midi(pitches[0])) if pitches else 0
        
        # Convert pitches to MIDI notes
        midi_notes = [int(round(self._freq_to_midi(freq))) for freq in pitches]
        
        # Find root note (lowest pitch)
        root_midi = min(midi_notes)
        
        # Calculate intervals from root
        intervals = [(note - root_midi) % 12 for note in midi_notes]
        intervals = sorted(list(set(intervals)))  # Remove duplicates and sort
        
        # Determine chord quality based on intervals
        if len(intervals) == 1:
            chord_quality = "single"
        elif len(intervals) == 2:
            if 4 in intervals and 7 in intervals:
                chord_quality = "major"
            elif 3 in intervals and 7 in intervals:
                chord_quality = "minor"
            elif 3 in intervals and 6 in intervals:
                chord_quality = "diminished"
            elif 4 in intervals and 8 in intervals:
                chord_quality = "augmented"
            else:
                chord_quality = "unknown"
        elif len(intervals) >= 3:
            if 4 in intervals and 7 in intervals:
                chord_quality = "major"
            elif 3 in intervals and 7 in intervals:
                chord_quality = "minor"
            elif 3 in intervals and 6 in intervals:
                chord_quality = "diminished"
            elif 4 in intervals and 8 in intervals:
                chord_quality = "augmented"
            else:
                chord_quality = "complex"
        else:
            chord_quality = "unknown"
        
        return chord_quality, root_midi
    
    def calculate_cents_deviation(self, freq: float, midi_note: float) -> float:
        """
        Calculate cents deviation from equal temperament
        
        Args:
            freq: Actual frequency
            midi_note: MIDI note number (can be fractional)
            
        Returns:
            Cents deviation from equal temperament
        """
        # Calculate expected frequency for equal temperament
        expected_freq = 440.0 * (2 ** ((midi_note - 69) / 12))
        
        # Calculate cents deviation
        if expected_freq > 0 and freq > 0:
            cents = 1200 * np.log2(freq / expected_freq)
            return cents
        return 0.0
    
    def calculate_ioi(self, current_time: float) -> float:
        """
        Calculate Inter-Onset Interval (time between musical events)
        
        Args:
            current_time: Current timestamp
            
        Returns:
            IOI in seconds
        """
        if self.last_onset_time > 0:
            ioi = current_time - self.last_onset_time
            self.last_onset_time = current_time
            return ioi
        else:
            self.last_onset_time = current_time
            return 0.0
    
    def detect_attack_release_times(self, frame: np.ndarray) -> Tuple[float, float]:
        """
        Detect attack and release times from audio frame
        
        Args:
            frame: Audio frame
            
        Returns:
            Tuple of (attack_time, release_time) in seconds
        """
        try:
            # Calculate envelope
            envelope = np.abs(frame)
            
            # Find peak
            peak_idx = np.argmax(envelope)
            peak_value = envelope[peak_idx]
            
            # Calculate attack time (time to reach attack_threshold of peak)
            attack_threshold_value = peak_value * self.attack_threshold
            attack_indices = np.where(envelope[:peak_idx] >= attack_threshold_value)[0]
            
            if len(attack_indices) > 0:
                attack_idx = attack_indices[0]
                attack_time = attack_idx / self.sample_rate
            else:
                attack_time = 0.0
            
            # Calculate release time (time from peak to release_threshold)
            release_threshold_value = peak_value * self.release_threshold
            release_indices = np.where(envelope[peak_idx:] <= release_threshold_value)[0]
            
            if len(release_indices) > 0:
                release_idx = release_indices[0] + peak_idx
                release_time = (release_idx - peak_idx) / self.sample_rate
            else:
                release_time = (len(frame) - peak_idx) / self.sample_rate
            
            return attack_time, release_time
            
        except Exception as e:
            print(f"Error in attack/release detection: {e}")
            return 0.1, 0.3  # Default values
    
    def extract_spectral_features(self, frame: np.ndarray) -> Dict[str, float]:
        """
        Extract comprehensive spectral features
        
        Args:
            frame: Audio frame
            
        Returns:
            Dictionary of spectral features
        """
        try:
            # Basic spectral features
            centroid = librosa.feature.spectral_centroid(y=frame, sr=self.sample_rate)[0, 0]
            rolloff = librosa.feature.spectral_rolloff(y=frame, sr=self.sample_rate)[0, 0]
            bandwidth = librosa.feature.spectral_bandwidth(y=frame, sr=self.sample_rate)[0, 0]
            contrast = librosa.feature.spectral_contrast(y=frame, sr=self.sample_rate)[0, 0]
            flatness = librosa.feature.spectral_flatness(y=frame)[0, 0]
            
            # MFCC features
            mfcc = librosa.feature.mfcc(y=frame, sr=self.sample_rate, n_mfcc=13)
            mfcc_1 = mfcc[1, 0] if mfcc.shape[1] > 0 else 0.0
            mfcc_2 = mfcc[2, 0] if mfcc.shape[1] > 0 else 0.0
            mfcc_3 = mfcc[3, 0] if mfcc.shape[1] > 0 else 0.0
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(frame)[0, 0]
            
            # Spectral rolloff at different percentiles
            rolloff_85 = librosa.feature.spectral_rolloff(y=frame, sr=self.sample_rate, roll_percent=0.85)[0, 0]
            rolloff_95 = librosa.feature.spectral_rolloff(y=frame, sr=self.sample_rate, roll_percent=0.95)[0, 0]
            
            return {
                'centroid': float(centroid),
                'rolloff': float(rolloff),
                'rolloff_85': float(rolloff_85),
                'rolloff_95': float(rolloff_95),
                'bandwidth': float(bandwidth),
                'contrast': float(contrast),
                'flatness': float(flatness),
                'mfcc_1': float(mfcc_1),
                'mfcc_2': float(mfcc_2),
                'mfcc_3': float(mfcc_3),
                'zcr': float(zcr)
            }
            
        except Exception as e:
            print(f"Error in spectral feature extraction: {e}")
            return {
                'centroid': 2000.0, 'rolloff': 3000.0, 'rolloff_85': 2500.0, 'rolloff_95': 3500.0,
                'bandwidth': 1000.0, 'contrast': 0.5, 'flatness': 0.1,
                'mfcc_1': 0.0, 'mfcc_2': 0.0, 'mfcc_3': 0.0, 'zcr': 0.0
            }
    
    def extract_temporal_features(self, frame: np.ndarray) -> Dict[str, float]:
        """
        Extract temporal features including RMS, onset detection, etc.
        
        Args:
            frame: Audio frame
            
        Returns:
            Dictionary of temporal features
        """
        try:
            # RMS energy
            rms_db = self._rms_db(frame)
            
            # Onset detection
            onset_detected = self._detect_onset(frame)
            
            # Attack and release times
            attack_time, release_time = self.detect_attack_release_times(frame)
            
            # Tempo estimation (simplified)
            tempo = 120.0  # Default tempo, could be estimated from IOI patterns
            
            # Beat position (simplified)
            beat_position = 0.0  # Could be calculated from onset timing
            
            return {
                'rms_db': float(rms_db),
                'onset': bool(onset_detected),
                'attack_time': float(attack_time),
                'release_time': float(release_time),
                'tempo': float(tempo),
                'beat_position': float(beat_position)
            }
            
        except Exception as e:
            print(f"Error in temporal feature extraction: {e}")
            return {
                'rms_db': -20.0, 'onset': False, 'attack_time': 0.1, 'release_time': 0.3,
                'tempo': 120.0, 'beat_position': 0.0
            }
    
    def process_audio_frame(self, frame: np.ndarray, timestamp: float) -> Optional[MultiPitchFrame]:
        """
        Process a single audio frame with polyphonic analysis
        
        Args:
            frame: Audio frame
            timestamp: Frame timestamp
            
        Returns:
            MultiPitchFrame object or None if frame is silent
        """
        try:
            # Calculate RMS level
            rms_db = self._rms_db(frame)
            
            # Skip silent frames
            if rms_db < self.level_db_threshold:
                return None
            
            # Extract multiple pitches
            pitches, amplitudes = self.extract_multiple_pitches(frame)
            
            if not pitches:
                return None
            
            # Analyze chord
            chord_quality, root_note = self.analyze_chord(pitches, amplitudes)
            
            # Convert pitches to MIDI notes and calculate cents
            midi_notes = []
            cents_deviations = []
            
            for pitch, amplitude in zip(pitches, amplitudes):
                midi_note = self._freq_to_midi(pitch)
                cents = self.calculate_cents_deviation(pitch, midi_note)
                midi_notes.append(int(round(midi_note)))
                cents_deviations.append(cents)
            
            # Calculate IOI
            ioi = self.calculate_ioi(timestamp)
            
            # Extract spectral and temporal features
            spectral_features = self.extract_spectral_features(frame)
            temporal_features = self.extract_temporal_features(frame)
            
            # Create MultiPitchFrame
            multi_pitch_frame = MultiPitchFrame(
                timestamp=timestamp,
                pitches=pitches,
                amplitudes=amplitudes,
                midi_notes=midi_notes,
                cents_deviations=cents_deviations,
                chord_quality=chord_quality,
                root_note=root_note,
                spectral_features=spectral_features,
                temporal_features=temporal_features
            )
            
            return multi_pitch_frame
            
        except Exception as e:
            print(f"Error processing audio frame: {e}")
            return None
    
    def convert_to_event(self, multi_pitch_frame: MultiPitchFrame) -> Event:
        """
        Convert MultiPitchFrame to Event object for compatibility
        
        Args:
            multi_pitch_frame: MultiPitchFrame object
            
        Returns:
            Event object
        """
        # Use the strongest pitch for the main Event object
        strongest_pitch_idx = 0  # Already sorted by amplitude
        strongest_pitch = multi_pitch_frame.pitches[strongest_pitch_idx]
        strongest_midi = multi_pitch_frame.midi_notes[strongest_pitch_idx]
        strongest_cents = multi_pitch_frame.cents_deviations[strongest_pitch_idx]
        
        # Create Event object
        event = Event(
            t=multi_pitch_frame.timestamp,
            rms_db=multi_pitch_frame.temporal_features['rms_db'],
            f0=strongest_pitch,
            midi=strongest_midi,
            cents=strongest_cents,
            centroid=multi_pitch_frame.spectral_features['centroid'],
            ioi=multi_pitch_frame.temporal_features.get('ioi', 0.5),
            onset=multi_pitch_frame.temporal_features['onset']
        )
        
        # Add additional features as attributes
        event.rolloff = multi_pitch_frame.spectral_features['rolloff']
        event.bandwidth = multi_pitch_frame.spectral_features['bandwidth']
        event.contrast = multi_pitch_frame.spectral_features['contrast']
        event.flatness = multi_pitch_frame.spectral_features['flatness']
        event.mfcc_1 = multi_pitch_frame.spectral_features['mfcc_1']
        event.mfcc_2 = multi_pitch_frame.spectral_features['mfcc_2']
        event.mfcc_3 = multi_pitch_frame.spectral_features['mfcc_3']
        event.zcr = multi_pitch_frame.spectral_features['zcr']
        event.attack_time = multi_pitch_frame.temporal_features['attack_time']
        event.release_time = multi_pitch_frame.temporal_features['release_time']
        event.tempo = multi_pitch_frame.temporal_features['tempo']
        event.beat_position = multi_pitch_frame.temporal_features['beat_position']
        
        # Add polyphonic information
        event.polyphonic_pitches = multi_pitch_frame.pitches
        event.polyphonic_midi = multi_pitch_frame.midi_notes
        event.polyphonic_cents = multi_pitch_frame.cents_deviations
        event.chord_quality = multi_pitch_frame.chord_quality
        event.root_note = multi_pitch_frame.root_note
        event.num_pitches = len(multi_pitch_frame.pitches)
        
        # Detect melodic voice using salience detector
        melodic_idx, voice_roles = self.melodic_detector.detect_melodic_voice(
            multi_pitch_frame.pitches,
            multi_pitch_frame.amplitudes,
            multi_pitch_frame.midi_notes,
            multi_pitch_frame.spectral_features['centroid']
        )
        
        # Add melodic salience information
        event.melodic_voice_idx = melodic_idx
        event.voice_roles = voice_roles
        event.melodic_pitch = multi_pitch_frame.pitches[melodic_idx]
        event.melodic_midi = multi_pitch_frame.midi_notes[melodic_idx]
        event.is_melody = (voice_roles[melodic_idx] == "melody")
        
        # Override main f0/midi with melodic voice (for melody-focused training)
        event.f0 = multi_pitch_frame.pitches[melodic_idx]
        event.midi = multi_pitch_frame.midi_notes[melodic_idx]
        event.cents = multi_pitch_frame.cents_deviations[melodic_idx]
        
        return event
    
    def process_audio_file(self, filepath: str) -> Tuple[List[Event], AudioFileInfo]:
        """
        Process entire audio file with polyphonic analysis
        
        Args:
            filepath: Path to audio file
            
        Returns:
            Tuple of (events, file_info)
        """
        print(f"🎵 Processing polyphonic audio file: {os.path.basename(filepath)}")
        
        # Load audio file
        audio_data, file_info = self.load_audio_file(filepath)
        
        if audio_data is None:
            print(f"❌ Failed to load audio file")
            return [], None
        
        print(f"📊 File info: {file_info.duration:.2f}s, {file_info.sample_rate}Hz, {file_info.format}")
        
        # Process audio frames
        events = []
        total_frames = (len(audio_data) - self.frame_length) // self.hop_length
        processed_frames = 0
        
        print(f"🔄 Processing {total_frames} audio frames with polyphonic analysis...")
        
        for i in range(0, len(audio_data) - self.frame_length, self.hop_length):
            frame = audio_data[i:i + self.frame_length]
            timestamp = i / self.sample_rate  # FIXED: Use relative time from audio start, not absolute Unix time
            
            # Process frame
            multi_pitch_frame = self.process_audio_frame(frame, timestamp)
            
            if multi_pitch_frame:
                # Convert to Event object
                event = self.convert_to_event(multi_pitch_frame)
                events.append(event)
            
            processed_frames += 1
            
            # Update progress
            if processed_frames % 500 == 0 or processed_frames == total_frames:
                progress = (processed_frames / total_frames) * 100
                print(f"\r🔄 Polyphonic Processing: {progress:.1f}% ({processed_frames}/{total_frames}) - {len(events)} events", end='', flush=True)
            
            # Stop if we've reached the maximum number of events
            if self.max_events and len(events) >= self.max_events:
                print(f"\n🛑 Reached maximum events limit: {self.max_events}")
                break
        
        print(f"\n✅ Polyphonic processing complete: {len(events)} events extracted")
        
        return events, file_info
    
    def load_audio_file(self, filepath: str) -> Tuple[np.ndarray, AudioFileInfo]:
        """Load audio file using librosa"""
        try:
            audio_data, sample_rate = librosa.load(filepath, sr=self.sample_rate)
            duration = len(audio_data) / sample_rate
            
            file_info = AudioFileInfo(
                duration=duration,
                sample_rate=sample_rate,
                format=os.path.splitext(filepath)[1],
                channels=1 if len(audio_data.shape) == 1 else audio_data.shape[0]
            )
            
            return audio_data, file_info
            
        except Exception as e:
            print(f"❌ Error loading audio file: {e}")
            return None, None
    
    # Helper methods
    @staticmethod
    def _rms_db(x: np.ndarray) -> float:
        """Calculate RMS in dB"""
        rms = float(np.sqrt(np.mean(x*x) + 1e-12))
        return 20.0 * np.log10(rms + 1e-12)
    
    @staticmethod
    def _freq_to_midi(f: float, A4: float = 440.0) -> float:
        """Convert frequency to MIDI note"""
        f = max(1e-9, float(f))
        return 69.0 + 12.0 * np.log2(f / A4)
    
    def _detect_onset(self, x: np.ndarray) -> bool:
        """Simple onset detection"""
        try:
            onset_frames = librosa.onset.onset_detect(y=x, sr=self.sample_rate, 
                                                    hop_length=self.hop_length, 
                                                    units='frames')
            return len(onset_frames) > 0
        except:
            return False


def main():
    """Test polyphonic audio processor"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Polyphonic Audio Processor')
    parser.add_argument('--file', '-f', required=True, help='Audio file to process')
    parser.add_argument('--max-events', type=int, default=100, help='Maximum events to process')
    parser.add_argument('--max-pitches', type=int, default=4, help='Maximum pitches per frame')
    
    args = parser.parse_args()
    
    print(f"🎵 Testing Polyphonic Audio Processor")
    print(f"📁 File: {args.file}")
    print(f"🎯 Max Pitches: {args.max_pitches}")
    
    # Initialize processor
    processor = PolyphonicAudioProcessor(
        max_events=args.max_events,
        max_pitches=args.max_pitches
    )
    
    # Process file
    events, file_info = processor.process_audio_file(args.file)
    
    if events:
        print(f"\n✅ Successfully processed {len(events)} events")
        
        # Show first few events with polyphonic information
        for i, event in enumerate(events[:5]):
            print(f"\nEvent {i+1}:")
            print(f"  Main Pitch: {event.f0:.1f} Hz (MIDI {event.midi})")
            print(f"  Cents: {event.cents:.1f}")
            print(f"  Chord: {event.chord_quality} (root: {event.root_note})")
            print(f"  Pitches: {len(event.polyphonic_pitches)} detected")
            if hasattr(event, 'polyphonic_pitches'):
                for j, pitch in enumerate(event.polyphonic_pitches):
                    print(f"    Pitch {j+1}: {pitch:.1f} Hz (MIDI {event.polyphonic_midi[j]})")
            print(f"  Spectral: centroid={event.centroid:.1f}, rolloff={event.rolloff:.1f}")
            print(f"  Temporal: attack={event.attack_time:.3f}s, release={event.release_time:.3f}s")
    else:
        print(f"❌ No events processed")


if __name__ == "__main__":
    main()
