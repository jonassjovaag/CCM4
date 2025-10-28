# audio_file_learning/file_processor.py
# Audio File Processor for Drift Engine AI
# Extracts features from audio files compatible with live system

import os
import time
import numpy as np
import librosa
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

# Import the Event class from the main system
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from listener.jhs_listener_core import Event

@dataclass
class AudioFileInfo:
    """Information about an audio file"""
    filepath: str
    duration: float
    sample_rate: int
    channels: int
    format: str

class AudioFileProcessor:
    """
    Processes audio files and extracts features compatible with Drift Engine AI
    Uses the same feature extraction as the live DriftListener
    """
    
    def __init__(self, 
                 sample_rate: int = 44100,
                 hop_length: int = 256,
                 frame_length: int = 2048,
                 fmin: float = 40.0,
                 fmax: float = 2000.0,
                 max_events: Optional[int] = None):
        """
        Initialize audio file processor
        
        Args:
            sample_rate: Target sample rate for processing
            hop_length: Hop length for frame processing (same as DriftListener)
            frame_length: Frame length for analysis (same as DriftListener)
            fmin: Minimum frequency for pitch tracking
            fmax: Maximum frequency for pitch tracking
        """
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.frame_length = frame_length
        self.fmin = fmin
        self.fmax = fmax
        self.max_events = max_events
        
        # Parameters matching DriftListener
        self.level_db_threshold = -45.0
        self.onset_threshold = 0.3
        self.centroid_alpha = 0.1
        
    def load_audio_file(self, filepath: str) -> Tuple[np.ndarray, AudioFileInfo]:
        """
        Load audio file and return audio data and metadata
        
        Args:
            filepath: Path to audio file
            
        Returns:
            Tuple of (audio_data, file_info)
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Audio file not found: {filepath}")
        
        try:
            # Load audio file
            audio_data, sr = librosa.load(filepath, sr=self.sample_rate, mono=True)
            
            # Get file info
            file_info = AudioFileInfo(
                filepath=filepath,
                duration=len(audio_data) / sr,
                sample_rate=sr,
                channels=1,  # librosa loads as mono
                format=os.path.splitext(filepath)[1].lower()
            )
            
            return audio_data, file_info
            
        except Exception as e:
            raise RuntimeError(f"Error loading audio file {filepath}: {e}")
    
    def extract_features_from_audio(self, audio_data: np.ndarray) -> List[Event]:
        """
        Extract features from audio data using same methods as DriftListener
        
        Args:
            audio_data: Audio samples
            
        Returns:
            List of Event objects compatible with live system
        """
        events = []
        
        # Calculate total frames for progress tracking
        total_frames = (len(audio_data) - self.frame_length) // self.hop_length
        processed_frames = 0
        
        print(f"🔄 Processing {total_frames} audio frames...")
        
        # Process audio in frames (same as DriftListener)
        for i in range(0, len(audio_data) - self.frame_length, self.hop_length):
            frame = audio_data[i:i + self.frame_length]
            
            # Calculate RMS level
            rms_db = self._rms_db(frame)
            
            # Skip silent frames
            if rms_db < self.level_db_threshold:
                continue
            
            # Extract pitch using YIN algorithm (same as DriftListener)
            f0 = self._yin_pitch(frame)
            if f0 <= 0.0:
                continue
            
            # Convert to MIDI note
            midi_note = self._freq_to_midi(f0)
            cents = (midi_note - round(midi_note)) * 100
            
            # Detect onset
            onset_detected = self._detect_onset(frame)
            
            # Calculate spectral centroid
            centroid = self._calculate_spectral_centroid(frame)
            
            # Calculate additional features
            rolloff = self._calculate_spectral_rolloff(frame)
            bandwidth = self._calculate_spectral_bandwidth(frame)
            contrast = self._calculate_spectral_contrast(frame)
            flatness = self._calculate_spectral_flatness(frame)
            
            # Calculate MFCC (first coefficient)
            mfcc = librosa.feature.mfcc(y=frame, sr=self.sample_rate, n_mfcc=13)
            mfcc_1 = mfcc[0, 0] if mfcc.size > 0 else 0.0
            
            # Calculate IOI (Inter-Onset Interval) - simplified
            ioi = self.hop_length / self.sample_rate  # Frame duration
            
            # Create Event object (same format as live system)
            event = Event(
                t=time.time() + (i / self.sample_rate),  # Approximate timestamp
                rms_db=rms_db,
                f0=f0,
                midi=int(round(midi_note)),
                cents=cents,
                centroid=centroid,
                ioi=ioi,
                onset=onset_detected
            )
            
            # Add additional features as attributes (for compatibility with AudioOracle)
            event.rolloff = rolloff
            event.bandwidth = bandwidth
            event.contrast = contrast
            event.flatness = flatness
            event.mfcc_1 = mfcc_1
            event.duration = ioi
            event.attack_time = 0.1
            event.release_time = 0.3
            event.tempo = 120.0
            event.beat_position = 0.0
            
            events.append(event)
            
            # Update progress indicator
            processed_frames += 1
            
            # Update every 1% or every 500 frames (whichever comes first)
            current_progress = int((processed_frames / total_frames) * 100)
            if current_progress > getattr(self, '_last_progress', 0) or processed_frames % 500 == 0 or processed_frames == total_frames:
                progress = (processed_frames / total_frames) * 100
                bar_length = 25  # Even shorter bar to prevent line wrapping
                filled_length = int(bar_length * processed_frames // total_frames)
                bar = '█' * filled_length + '░' * (bar_length - filled_length)
                
                # Add ETA estimation
                if not hasattr(self, '_processing_start_time'):
                    self._processing_start_time = time.time()
                
                if processed_frames > 0:
                    elapsed_time = time.time() - self._processing_start_time
                    if elapsed_time > 0:
                        rate = processed_frames / elapsed_time
                        remaining_frames = total_frames - processed_frames
                        eta_seconds = remaining_frames / rate if rate > 0 else 0
                        eta_minutes = eta_seconds / 60
                        eta_str = f"ETA:{eta_minutes:.1f}m" if eta_minutes > 1 else f"ETA:{eta_seconds:.0f}s"
                    else:
                        eta_str = "ETA:calc"
                else:
                    eta_str = "ETA:calc"
                
                # Ensure we use carriage return for in-place updates
                print(f"\r🔄 [{bar}] {progress:.1f}% ({processed_frames}/{total_frames}) {eta_str}", end='', flush=True)
                self._last_progress = current_progress
            
            # Stop if we've reached the maximum number of events
            if self.max_events and len(events) >= self.max_events:
                print(f"\n🛑 Reached maximum events limit: {self.max_events}")
                break
        
        print()  # New line after progress bar
        return events
    
    def process_audio_file(self, filepath: str) -> Tuple[List[Event], AudioFileInfo]:
        """
        Complete processing of an audio file
        
        Args:
            filepath: Path to audio file
            
        Returns:
            Tuple of (events, file_info)
        """
        print(f"🎵 Processing audio file: {os.path.basename(filepath)}")
        
        # Load audio file
        audio_data, file_info = self.load_audio_file(filepath)
        
        print(f"📊 File info: {file_info.duration:.2f}s, {file_info.sample_rate}Hz, {file_info.format}")
        
        # Extract features
        events = self.extract_features_from_audio(audio_data)
        
        print(f"✅ Extracted {len(events)} musical events")
        
        return events, file_info
    
    def process_multiple_files(self, filepaths: List[str]) -> Dict[str, Tuple[List[Event], AudioFileInfo]]:
        """
        Process multiple audio files
        
        Args:
            filepaths: List of audio file paths
            
        Returns:
            Dictionary mapping filepath to (events, file_info)
        """
        results = {}
        
        for filepath in filepaths:
            try:
                events, file_info = self.process_audio_file(filepath)
                results[filepath] = (events, file_info)
            except Exception as e:
                print(f"❌ Error processing {filepath}: {e}")
                continue
        
        return results
    
    # Feature extraction methods (matching DriftListener)
    
    @staticmethod
    def _rms_db(x: np.ndarray) -> float:
        """Calculate RMS in dB (same as DriftListener)"""
        rms = float(np.sqrt(np.mean(x*x) + 1e-12))
        return 20.0 * np.log10(rms + 1e-12)
    
    @staticmethod
    def _freq_to_midi(f: float, A4: float = 440.0) -> float:
        """Convert frequency to MIDI note (same as DriftListener)"""
        f = max(1e-9, float(f))
        return 69.0 + 12.0 * np.log2(f / A4)
    
    def _yin_pitch(self, x: np.ndarray) -> float:
        """YIN pitch detection (simplified version of DriftListener)"""
        try:
            # Use librosa's YIN implementation for consistency
            f0 = librosa.yin(x, fmin=self.fmin, fmax=self.fmax, sr=self.sample_rate)
            return float(f0[0]) if len(f0) > 0 and f0[0] > 0 else 0.0
        except:
            return 0.0
    
    def _detect_onset(self, x: np.ndarray) -> bool:
        """Simple onset detection (matching DriftListener logic)"""
        try:
            # Use librosa's onset detection
            onset_frames = librosa.onset.onset_detect(y=x, sr=self.sample_rate, 
                                                    hop_length=self.hop_length, 
                                                    units='frames')
            return len(onset_frames) > 0
        except:
            return False
    
    def _calculate_spectral_centroid(self, x: np.ndarray) -> float:
        """Calculate spectral centroid (matching DriftListener)"""
        try:
            centroid = librosa.feature.spectral_centroid(y=x, sr=self.sample_rate)[0]
            return float(centroid[0]) if len(centroid) > 0 else 0.0
        except:
            return 0.0
    
    def _calculate_spectral_rolloff(self, x: np.ndarray) -> float:
        """Calculate spectral rolloff"""
        try:
            rolloff = librosa.feature.spectral_rolloff(y=x, sr=self.sample_rate)[0]
            return float(rolloff[0]) if len(rolloff) > 0 else 0.0
        except:
            return 0.0
    
    def _calculate_spectral_bandwidth(self, x: np.ndarray) -> float:
        """Calculate spectral bandwidth"""
        try:
            bandwidth = librosa.feature.spectral_bandwidth(y=x, sr=self.sample_rate)[0]
            return float(bandwidth[0]) if len(bandwidth) > 0 else 0.0
        except:
            return 0.0
    
    def _calculate_spectral_contrast(self, x: np.ndarray) -> float:
        """Calculate spectral contrast"""
        try:
            contrast = librosa.feature.spectral_contrast(y=x, sr=self.sample_rate)[0]
            return float(np.mean(contrast)) if len(contrast) > 0 else 0.0
        except:
            return 0.0
    
    def _calculate_spectral_flatness(self, x: np.ndarray) -> float:
        """Calculate spectral flatness"""
        try:
            flatness = librosa.feature.spectral_flatness(y=x)[0]
            return float(flatness[0]) if len(flatness) > 0 else 0.0
        except:
            return 0.0
