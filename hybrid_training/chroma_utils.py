"""
Shared chroma extraction utilities
Works around NumPy 2.2 / numba incompatibility with librosa
"""

import numpy as np


def extract_chroma_from_audio(audio_buffer: np.ndarray, sr: int = 44100, 
                              window_size: int = 8192, hop_length: int = 2048) -> np.ndarray:
    """
    Extract chroma features from audio buffer
    Simplified implementation that doesn't require librosa's chroma features
    
    Args:
        audio_buffer: Audio samples (mono)
        sr: Sample rate
        window_size: FFT window size
        hop_length: Hop length for frame analysis
        
    Returns:
        Chroma matrix (12 x num_frames)
    """
    num_frames = 1 + (len(audio_buffer) - window_size) // hop_length
    chroma_matrix = np.zeros((12, num_frames))
    
    for frame_idx in range(num_frames):
        start = frame_idx * hop_length
        end = start + window_size
        
        if end > len(audio_buffer):
            # Pad last frame if necessary
            frame = np.pad(audio_buffer[start:], (0, end - len(audio_buffer)))
        else:
            frame = audio_buffer[start:end]
        
        # Extract chroma for this frame
        chroma_matrix[:, frame_idx] = _extract_chroma_frame(frame, sr, window_size)
    
    return chroma_matrix


def _extract_chroma_frame(frame: np.ndarray, sr: int, window_size: int) -> np.ndarray:
    """
    Extract chroma from a single frame using FFT
    
    Args:
        frame: Audio frame
        sr: Sample rate
        window_size: Window size
        
    Returns:
        12-dimensional chroma vector
    """
    chroma = np.zeros(12)
    
    # Apply window
    windowed = frame * np.hanning(window_size)
    
    # Compute FFT
    spectrum = np.fft.rfft(windowed)
    magnitude = np.abs(spectrum)
    freqs = np.fft.rfftfreq(window_size, 1/sr)
    
    # Map frequencies to pitch classes (0-11 for C-B)
    # A4 = 440 Hz = MIDI 69 = pitch class 9 (A)
    for i, freq in enumerate(freqs):
        if freq < 50 or freq > 4000:  # Skip very low and very high frequencies
            continue
        
        # Convert frequency to MIDI note number
        if freq > 0:
            midi_note = 69 + 12 * np.log2(freq / 440.0)
            pitch_class = int(round(midi_note)) % 12
            
            # Accumulate energy for this pitch class
            chroma[pitch_class] += magnitude[i]
    
    # Normalize
    if np.sum(chroma) > 0:
        chroma = chroma / np.sum(chroma)
    
    return chroma


def chroma_stft_fallback(y: np.ndarray, sr: int = 44100, hop_length: int = 512, **kwargs) -> np.ndarray:
    """
    Drop-in replacement for librosa.feature.chroma_stft()
    
    Args:
        y: Audio time series
        sr: Sample rate
        hop_length: Hop length
        **kwargs: Ignored (for compatibility with librosa API)
        
    Returns:
        Chroma matrix (12 x num_frames)
    """
    window_size = kwargs.get('n_fft', 2048)
    return extract_chroma_from_audio(y, sr, window_size, hop_length)

