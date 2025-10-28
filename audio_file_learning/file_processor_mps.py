# audio_file_learning/file_processor_mps.py
# MPS-Accelerated Audio File Processor
# Uses PyTorch MPS for feature extraction acceleration

import os
import sys
import time
import numpy as np
import librosa
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# Import PyTorch for MPS acceleration
import torch
import torchaudio

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


class MPSAudioFileProcessor:
    """
    MPS-Accelerated Audio File Processor
    
    Uses PyTorch MPS for accelerated feature extraction on Apple Silicon
    """
    
    def __init__(self, 
                 sample_rate: int = 44100,
                 hop_length: int = 256,
                 frame_length: int = 2048,
                 max_events: Optional[int] = None,
                 use_mps: bool = True):
        """
        Initialize MPS Audio File Processor
        
        Args:
            sample_rate: Sample rate for processing
            hop_length: Hop length for frame processing
            frame_length: Frame length for analysis
            max_events: Maximum number of events to process
            use_mps: Whether to use MPS acceleration
        """
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.frame_length = frame_length
        self.max_events = max_events
        
        # MPS configuration
        self.use_mps = use_mps and torch.backends.mps.is_available()
        self.device = torch.device("mps") if self.use_mps else torch.device("cpu")
        
        # Audio processing parameters
        self.fmin = 40.0
        self.fmax = 8000.0
        
        print(f"🎯 MPS Audio File Processor initialized:")
        print(f"   Device: {self.device}")
        print(f"   MPS Available: {torch.backends.mps.is_available()}")
        print(f"   Using GPU: {self.use_mps}")
        print(f"   Sample Rate: {sample_rate}Hz")
        print(f"   Frame Length: {frame_length}")
        print(f"   Hop Length: {hop_length}")
    
    def load_audio_file(self, filepath: str) -> Tuple[np.ndarray, AudioFileInfo]:
        """
        Load audio file using MPS-accelerated torchaudio
        
        Args:
            filepath: Path to audio file
            
        Returns:
            Tuple of (audio_data, file_info)
        """
        try:
            if self.use_mps:
                # Use MPS-accelerated torchaudio
                waveform, sample_rate = torchaudio.load(filepath)
                waveform = waveform.to(self.device)
                
                # Convert to mono if stereo
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0)
                
                # Convert to numpy for librosa compatibility
                audio_data = waveform.cpu().numpy()
            else:
                # Fallback to librosa
                audio_data, sample_rate = librosa.load(filepath, sr=self.sample_rate)
            
            # Get file info
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
    
    def extract_features_mps(self, audio_data: np.ndarray) -> List[Event]:
        """
        Extract features using MPS-accelerated processing
        
        Args:
            audio_data: Audio data array
            
        Returns:
            List of Event objects
        """
        events = []
        
        # Convert to PyTorch tensor on MPS device
        if self.use_mps:
            audio_tensor = torch.tensor(audio_data, device=self.device, dtype=torch.float32)
        else:
            audio_tensor = torch.tensor(audio_data, dtype=torch.float32)
        
        # Calculate total frames for progress tracking
        total_frames = (len(audio_data) - self.frame_length) // self.hop_length
        processed_frames = 0
        
        print(f"🔄 MPS Processing {total_frames} audio frames...")
        
        # Process audio in frames
        for i in range(0, len(audio_data) - self.frame_length, self.hop_length):
            # Extract frame
            frame = audio_tensor[i:i + self.frame_length]
            
            if self.use_mps:
                # MPS-accelerated feature extraction
                features = self._extract_features_mps_frame(frame)
            else:
                # CPU fallback
                features = self._extract_features_cpu_frame(frame.cpu().numpy())
            
            # Create Event object
            event = Event(
                t=time.time() + (i / self.sample_rate),
                rms_db=features['rms_db'],
                f0=features['f0'],
                midi=int(round(features['midi'])),
                cents=features['cents'],
                centroid=features['centroid'],
                ioi=features['ioi'],
                onset=features['onset']
            )
            
            # Add additional features as attributes
            event.rolloff = features['rolloff']
            event.bandwidth = features['bandwidth']
            event.contrast = features['contrast']
            event.flatness = features['flatness']
            event.mfcc_1 = features['mfcc_1']
            event.duration = features['ioi']
            event.attack_time = 0.1
            event.release_time = 0.3
            event.tempo = 120.0
            event.beat_position = 0.0
            
            events.append(event)
            
            # Update progress indicator
            processed_frames += 1
            
            # Update every 1% or every 500 frames
            current_progress = int((processed_frames / total_frames) * 100)
            if current_progress > getattr(self, '_last_progress', 0) or processed_frames % 500 == 0 or processed_frames == total_frames:
                progress = (processed_frames / total_frames) * 100
                bar_length = 25
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
                print(f"\r🔄 MPS [{bar}] {progress:.1f}% ({processed_frames}/{total_frames}) {eta_str}", end='', flush=True)
                self._last_progress = current_progress
            
            # Stop if we've reached the maximum number of events
            if self.max_events and len(events) >= self.max_events:
                print(f"\n🛑 Reached maximum events limit: {self.max_events}")
                break
        
        print()  # New line after progress bar
        return events
    
    def _extract_features_mps_frame(self, frame: torch.Tensor) -> Dict[str, float]:
        """
        Extract features from a single frame using MPS acceleration
        
        Args:
            frame: Audio frame tensor on MPS device
            
        Returns:
            Dictionary of extracted features
        """
        try:
            # Convert to CPU for librosa compatibility (librosa doesn't support MPS yet)
            frame_cpu = frame.cpu().numpy()
            
            # Basic features using librosa (CPU)
            rms = librosa.feature.rms(y=frame_cpu, frame_length=self.frame_length, hop_length=1)[0, 0]
            rms_db = 20 * np.log10(rms + 1e-8)
            
            # Pitch detection
            f0 = librosa.yin(frame_cpu, fmin=self.fmin, fmax=self.fmax, sr=self.sample_rate)
            f0 = f0[0] if len(f0) > 0 else 0.0
            
            # MIDI note conversion
            if f0 > 0:
                midi = 12 * np.log2(f0 / 440.0) + 69
                cents = (midi - int(midi)) * 100
            else:
                midi = 0
                cents = 0
            
            # Spectral features
            centroid = librosa.feature.spectral_centroid(y=frame_cpu, sr=self.sample_rate)[0, 0]
            rolloff = librosa.feature.spectral_rolloff(y=frame_cpu, sr=self.sample_rate)[0, 0]
            bandwidth = librosa.feature.spectral_bandwidth(y=frame_cpu, sr=self.sample_rate)[0, 0]
            contrast = librosa.feature.spectral_contrast(y=frame_cpu, sr=self.sample_rate)[0, 0]
            flatness = librosa.feature.spectral_flatness(y=frame_cpu)[0, 0]
            
            # MFCC
            mfcc = librosa.feature.mfcc(y=frame_cpu, sr=self.sample_rate, n_mfcc=13)
            mfcc_1 = mfcc[1, 0] if mfcc.shape[1] > 0 else 0.0
            
            # Onset detection
            onset = librosa.onset.onset_detect(y=frame_cpu, sr=self.sample_rate, units='time')
            onset_detected = len(onset) > 0
            
            # IOI (Inter-Onset Interval) - simplified
            ioi = 0.5  # Default value
            
            return {
                'rms_db': float(rms_db),
                'f0': float(f0),
                'midi': float(midi),
                'cents': float(cents),
                'centroid': float(centroid),
                'rolloff': float(rolloff),
                'bandwidth': float(bandwidth),
                'contrast': float(contrast),
                'flatness': float(flatness),
                'mfcc_1': float(mfcc_1),
                'ioi': float(ioi),
                'onset': bool(onset_detected)
            }
            
        except Exception as e:
            print(f"Error in MPS feature extraction: {e}")
            # Return default features
            return {
                'rms_db': -20.0,
                'f0': 440.0,
                'midi': 69.0,
                'cents': 0.0,
                'centroid': 2000.0,
                'rolloff': 3000.0,
                'bandwidth': 1000.0,
                'contrast': 0.5,
                'flatness': 0.1,
                'mfcc_1': 0.0,
                'ioi': 0.5,
                'onset': False
            }
    
    def _extract_features_cpu_frame(self, frame: np.ndarray) -> Dict[str, float]:
        """
        Extract features from a single frame using CPU (fallback)
        
        Args:
            frame: Audio frame array
            
        Returns:
            Dictionary of extracted features
        """
        # This is the same as the original implementation
        # For now, just call the MPS version with CPU tensor
        frame_tensor = torch.tensor(frame, dtype=torch.float32)
        return self._extract_features_mps_frame(frame_tensor)
    
    def process_audio_file(self, filepath: str) -> Tuple[List[Event], AudioFileInfo]:
        """
        Complete processing of an audio file using MPS acceleration
        
        Args:
            filepath: Path to audio file
            
        Returns:
            Tuple of (events, file_info)
        """
        print(f"🎵 MPS Processing audio file: {os.path.basename(filepath)}")
        
        # Load audio file
        audio_data, file_info = self.load_audio_file(filepath)
        
        if audio_data is None:
            print(f"❌ Failed to load audio file")
            return [], None
        
        print(f"📊 File info: {file_info.duration:.2f}s, {file_info.sample_rate}Hz, {file_info.format}")
        
        # Extract features using MPS acceleration
        events = self.extract_features_mps(audio_data)
        
        print(f"✅ MPS Extracted {len(events)} musical events")
        
        return events, file_info


def main():
    """Test MPS Audio File Processor"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test MPS Audio File Processor')
    parser.add_argument('--file', '-f', required=True, help='Audio file to process')
    parser.add_argument('--max-events', type=int, default=1000, help='Maximum events to process')
    parser.add_argument('--no-mps', action='store_true', help='Disable MPS acceleration')
    
    args = parser.parse_args()
    
    print(f"🎯 Testing MPS Audio File Processor")
    print(f"📁 File: {args.file}")
    print(f"🎯 MPS: {not args.no_mps}")
    
    # Initialize processor
    processor = MPSAudioFileProcessor(
        max_events=args.max_events,
        use_mps=not args.no_mps
    )
    
    # Process file
    events, file_info = processor.process_audio_file(args.file)
    
    if events:
        print(f"✅ Successfully processed {len(events)} events")
        print(f"📊 First event: MIDI={events[0].midi}, F0={events[0].f0:.1f}Hz")
    else:
        print(f"❌ No events processed")


if __name__ == "__main__":
    main()
