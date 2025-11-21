#!/usr/bin/env python3
"""
Harmonic-Aware Chroma Extraction
=================================

Based on research:
- Kronvall et al. (2015): Sparse Chroma Estimation for Harmonic Audio
- Juhlin et al. (2015): Sparse Chroma for Non-Stationary Audio
- Rao et al. (2016): Temporal Correlation for Chord Recognition

Key improvements over standard chroma:
1. Uses CQT (Constant-Q Transform) for logarithmic frequency spacing
2. Harmonic weighting: distinguishes fundamentals from overtones
3. Temporal correlation: exploits stability across frames
4. Reduces tone ambiguity significantly
"""

import numpy as np
import librosa
from typing import Tuple, Optional, List


class HarmonicAwareChromaExtractor:
    """
    Extract chroma with harmonic structure awareness
    
    This addresses the main problem with standard FFT-based chroma:
    it can't distinguish between a fundamental frequency and its harmonics.
    
    For example, playing C4 (261 Hz) also produces harmonics at:
    - 522 Hz (C5, octave)
    - 783 Hz (G5, perfect fifth)
    - 1044 Hz (C6, 2nd octave)
    
    Standard chroma sees all of these and gets confused.
    Our approach weights fundamentals much higher than harmonics.
    """
    
    def __init__(self, n_octaves: int = 6, bins_per_octave: int = 12,
                 fmin: float = 130.81, fmax: float = 4186.0):
        """
        Initialize harmonic-aware chroma extractor
        
        Args:
            n_octaves: Number of octaves to analyze (default 7)
            bins_per_octave: Bins per octave for CQT (default 12 = semitones)
            fmin: Minimum frequency (default 130.81 Hz = C3)
            fmax: Maximum frequency (default 4186.0 Hz = C8)
        """
        self.n_octaves = n_octaves
        self.bins_per_octave = bins_per_octave
        self.n_bins = n_octaves * bins_per_octave
        self.fmin = fmin
        self.fmax = fmax
        
        # Harmonic weights: fundamental >> harmonics
        # Based on psychoacoustic research (Helmholtz, Shapira Lots & Stone)
        # Aggressive suppression of harmonics
        self.harmonic_weights = {
            1: 1.0,      # Fundamental (100% weight)
            2: 0.10,     # Octave (suppress heavily)
            3: 0.05,     # Perfect fifth (2:3 ratio)
            4: 0.03,     # 2nd octave
            5: 0.02,     # Major third region
            6: 0.01,     # Minor third + octave
            7: 0.005,    # Everything else extremely low
        }
        
        # Note names for debugging
        self.note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 
                          'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    def extract(self, audio: np.ndarray, sr: int = 44100, 
                use_temporal: bool = True, live_mode: bool = False) -> np.ndarray:
        """
        Extract harmonic-aware chroma from audio
        
        Args:
            audio: Audio signal (mono)
            sr: Sample rate
            use_temporal: Whether to use temporal correlation (recommended)
            live_mode: If True, apply aggressive preprocessing for live mic input
            
        Returns:
            12-dimensional chroma vector (normalized)
        """
        if len(audio) < 2048:
            return np.zeros(12)
        
        # Preprocess for live microphone input
        if live_mode:
            # 1. Noise gate: Check BEFORE preemphasis (preemphasis can reduce RMS)
            rms = np.sqrt(np.mean(audio**2))
            if rms < 0.001:  # Lowered threshold to -60dB for sensitivity
                return np.zeros(12)
            
            # Suppress librosa warnings about buffer sizes
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # 2. High-pass filter to remove low-frequency rumble
                audio = librosa.effects.preemphasis(audio, coef=0.97)
            
            # 3. Normalize to consistent level
            audio = audio / (np.max(np.abs(audio)) + 1e-9)
        
        # Use CQT instead of STFT
        # CQT has logarithmic frequency spacing (like musical notes!)
        fmin_use = self.fmin if not live_mode else max(self.fmin, 196.0)  # C3 for live
        cqt = librosa.cqt(
            y=audio,
            sr=sr,
            hop_length=512,
            n_bins=self.n_bins,
            bins_per_octave=self.bins_per_octave,
            fmin=fmin_use
        )
        cqt_mag = np.abs(cqt)
        
        if use_temporal:
            # Temporal correlation approach (Rao et al., 2016)
            chroma = self._extract_with_temporal_correlation(cqt_mag)
        else:
            # Basic harmonic weighting
            chroma = self._extract_with_harmonic_weighting(cqt_mag)
        
        # Normalize
        if np.sum(chroma) > 0:
            chroma = chroma / np.sum(chroma)
        
        return chroma
    
    def _extract_with_harmonic_weighting(self, cqt_mag: np.ndarray) -> np.ndarray:
        """
        Extract chroma using harmonic weighting
        
        Strategy:
        - For each pitch class (C, C#, D, ..., B)
        - Look at all occurrences across octaves
        - Weight lower octaves (fundamentals) higher than upper octaves (harmonics)
        """
        chroma = np.zeros(12)
        
        # Average CQT magnitude over time
        cqt_mean = np.mean(cqt_mag, axis=1)
        
        for pitch_class in range(12):
            total_energy = 0.0
            
            # Collect energy from this pitch class across all octaves
            for octave in range(self.n_octaves):
                bin_idx = pitch_class + octave * self.bins_per_octave
                
                if bin_idx < len(cqt_mean):
                    energy = cqt_mean[bin_idx]
                    
                    # Apply harmonic weighting
                    # Lower octaves = more likely to be fundamentals
                    # Higher octaves = more likely to be harmonics
                    
                    if octave <= 3:  # Lower octaves - likely fundamentals
                        weight = self.harmonic_weights[1]  # Full weight
                    else:  # Upper octaves - likely harmonics
                        harmonic_order = octave - 2
                        weight = self.harmonic_weights.get(harmonic_order, 0.01)
                    
                    total_energy += energy * weight
            
            chroma[pitch_class] = total_energy
        
        return chroma
    
    def _extract_with_temporal_correlation(self, cqt_mag: np.ndarray) -> np.ndarray:
        """
        Extract chroma using temporal correlation
        
        Key insight (Rao et al., 2016):
        Stable chord tones have:
        1. High mean energy
        2. Low variance (consistent across time)
        3. High presence ratio (active in most frames)
        
        Harmonics and transients have:
        - High variance
        - Sporadic presence
        """
        chroma = np.zeros(12)
        n_frames = cqt_mag.shape[1]
        
        for pitch_class in range(12):
            # Collect time series for this pitch class across octaves
            time_series_weighted = np.zeros(n_frames)
            
            for octave in range(self.n_octaves):
                bin_idx = pitch_class + octave * self.bins_per_octave
                
                if bin_idx < cqt_mag.shape[0]:
                    # Get time series for this bin
                    time_series = cqt_mag[bin_idx, :]
                    
                    # Apply harmonic weighting to time series
                    if octave <= 3:
                        weight = self.harmonic_weights[1]
                    else:
                        harmonic_order = octave - 2
                        weight = self.harmonic_weights.get(harmonic_order, 0.01)
                    
                    time_series_weighted += time_series * weight
            
            # Calculate stability metrics
            mean_energy = np.mean(time_series_weighted)
            variance = np.var(time_series_weighted)
            
            # Presence ratio: fraction of frames above threshold
            threshold = 0.1 * np.max(time_series_weighted)
            presence = np.sum(time_series_weighted > threshold) / n_frames
            
            # Stability score: high mean, low variance, high presence
            # This emphasizes sustained tones (chord tones) over transients
            stability = mean_energy * presence / (variance + 0.01)
            
            chroma[pitch_class] = stability
        
        return chroma
    
    def extract_with_threshold(self, audio: np.ndarray, sr: int = 44100,
                               threshold: float = 0.30) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract chroma and return both full vector and active pitch classes
        
        Args:
            audio: Audio signal
            sr: Sample rate
            threshold: Relative threshold for active pitch classes (0-1)
            
        Returns:
            (chroma_vector, active_pitch_classes)
        """
        chroma = self.extract(audio, sr, use_temporal=True)
        
        # Find active pitch classes (above threshold)
        max_val = np.max(chroma)
        active_mask = chroma > (threshold * max_val)
        active_pitch_classes = np.where(active_mask)[0]
        
        return chroma, active_pitch_classes
    
    def extract_top_k(self, audio: np.ndarray, sr: int = 44100,
                     k: int = 3, min_separation: int = 1, min_threshold: float = 0.25) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract chroma and return top-k strongest pitch classes
        
        This is often more reliable than thresholding for chord detection.
        
        Args:
            audio: Audio signal
            sr: Sample rate
            k: Number of top pitch classes to select
            min_separation: Minimum semitone separation (prevents detecting harmonics)
            min_threshold: Minimum relative strength (0-1) compared to max
            
        Returns:
            (chroma_vector, top_k_pitch_classes)
        """
        chroma = self.extract(audio, sr, use_temporal=True, live_mode=True)
        
        # Get indices sorted by energy (highest first)
        sorted_indices = np.argsort(chroma)[::-1]
        
        # Apply threshold first
        max_val = np.max(chroma)
        if max_val == 0:
            return chroma, np.array([])
        
        # Select top-k with minimum separation and threshold
        selected = []
        for idx in sorted_indices:
            if len(selected) >= k:
                break
            
            # Must be above threshold
            if chroma[idx] < min_threshold * max_val:
                continue
            
            # Check separation from already selected
            too_close = False
            for sel_idx in selected:
                # Calculate circular distance on chromatic circle
                dist = min(abs(idx - sel_idx), 12 - abs(idx - sel_idx))
                if dist < min_separation:
                    too_close = True
                    break
            
            if not too_close:
                selected.append(idx)
        
        return chroma, np.array(sorted(selected))
    
    def chroma_to_frequencies(self, chroma: np.ndarray, 
                             threshold: float = 0.30,
                             reference_octave: int = 4) -> np.ndarray:
        """
        Convert chroma vector to frequencies
        
        Args:
            chroma: 12-dimensional chroma vector
            threshold: Relative threshold for active notes
            reference_octave: Octave for frequency conversion (4 = C4, middle C)
            
        Returns:
            Array of frequencies (Hz)
        """
        # Find active pitch classes
        max_val = np.max(chroma)
        if max_val == 0:
            return np.array([])
        
        active_pcs = np.where(chroma > threshold * max_val)[0]
        
        # Convert to frequencies
        frequencies = []
        for pc in active_pcs:
            # MIDI note = 60 (C4) + pitch class
            midi_note = 60 + pc  # C4 = middle C
            freq = 440.0 * (2 ** ((midi_note - 69) / 12.0))
            frequencies.append(freq)
        
        return np.array(sorted(frequencies))
    
    def analyze_with_guided_search(self, audio: np.ndarray, 
                                    expected_freqs: List[float],
                                    sr: int = 44100,
                                    tolerance_cents: float = 50.0) -> dict:
        """
        Guided frequency search using ground truth (for training/validation)
        
        This is supervised detection: we KNOW what we're looking for,
        so we search for peaks near expected frequencies.
        
        Args:
            audio: Audio signal
            expected_freqs: Expected frequencies (ground truth)
            sr: Sample rate
            tolerance_cents: Search tolerance in cents (default ±50 cents = ±semitone)
            
        Returns:
            Dictionary with detected frequencies, confidence, etc.
        """
        if len(audio) < 2048:
            return {
                'detected_frequencies': [],
                'peak_confidences': [],
                'note_names': [],
                'success': False,
                'num_detected': 0
            }
        
        # Preprocess
        audio = librosa.effects.preemphasis(audio, coef=0.97)
        rms = np.sqrt(np.mean(audio**2))
        
        # Debug output removed for real-time performance compatibility
        # print(f"      Audio RMS: {rms:.6f}, Max: {np.max(np.abs(audio)):.6f}")
        
        if rms < 0.0001:  # Very low threshold (only reject complete silence)
            # Debug output removed for real-time performance compatibility
            # print(f"      ⚠ Audio too quiet (RMS {rms:.6f} < 0.0001)")
            return {'detected_frequencies': [], 'peak_confidences': [], 
                    'note_names': [], 'success': False, 'num_detected': 0}
        
        audio = audio / (np.max(np.abs(audio)) + 1e-9)
        
        # Get spectrum
        D = librosa.stft(audio, n_fft=4096, hop_length=512)
        magnitude = np.abs(D)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)
        
        # Average over time
        mag_mean = np.mean(magnitude, axis=1)
        
        detected_freqs = []
        confidences = []
        
        # Debug output removed for real-time performance compatibility
        # print(f"      Spectrum: {len(freqs)} bins, max magnitude: {np.max(mag_mean):.6f}")
        
        # For each expected frequency, search for peak nearby
        for expected_f in expected_freqs:
            # Calculate frequency search window (±tolerance_cents)
            ratio = 2 ** (tolerance_cents / 1200.0)
            f_low = expected_f / ratio
            f_high = expected_f * ratio
            
            # Find frequency bins in this range
            mask = (freqs >= f_low) & (freqs <= f_high)
            if not np.any(mask):
                continue
            
            # Find peak within this range
            windowed_mag = mag_mean.copy()
            windowed_mag[~mask] = 0
            
            if np.max(windowed_mag) == 0:
                continue
            
            peak_idx = np.argmax(windowed_mag)
            peak_freq = freqs[peak_idx]
            peak_magnitude = mag_mean[peak_idx]
            
            # Confidence based on:
            # 1. How close to expected frequency
            # 2. Peak magnitude
            freq_error_cents = 1200 * np.log2(peak_freq / expected_f) if peak_freq > 0 else 1000
            frequency_confidence = 1.0 - (abs(freq_error_cents) / tolerance_cents)
            magnitude_confidence = min(1.0, peak_magnitude / (np.max(mag_mean) + 1e-9))
            
            confidence = frequency_confidence * magnitude_confidence
            
            detected_freqs.append(peak_freq)
            confidences.append(confidence)
        
        # Convert to note names
        note_names = []
        for f in detected_freqs:
            midi = int(round(69 + 12 * np.log2(f / 440.0)))
            note_names.append(self.note_names[midi % 12])
        
        return {
            'detected_frequencies': np.array(detected_freqs),
            'peak_confidences': np.array(confidences),
            'note_names': note_names,
            'expected_frequencies': expected_freqs,
            'frequency_errors_cents': [
                1200 * np.log2(d/e) if d > 0 and e > 0 else 0
                for d, e in zip(detected_freqs, expected_freqs)
            ],
            'success': len(detected_freqs) == len(expected_freqs),
            'num_detected': len(detected_freqs)
        }
    
    def analyze_with_debug(self, audio: np.ndarray, sr: int = 44100, 
                           use_top_k: bool = True, k: int = 3, live_mode: bool = False) -> dict:
        """
        Extract chroma with detailed debugging information
        
        Returns dictionary with:
        - chroma: Full chroma vector
        - active_pitch_classes: Indices of active notes
        - note_names: Names of detected notes
        - energies: Energy values for each pitch class
        - frequencies: Corresponding frequencies
        """
        # Extract chroma with live mode preprocessing if requested
        if use_top_k:
            chroma = self.extract(audio, sr, use_temporal=True, live_mode=live_mode)
            # Get top-k manually
            sorted_indices = np.argsort(chroma)[::-1]
            selected = []
            for idx in sorted_indices:
                if len(selected) >= k:
                    break
                too_close = False
                for sel_idx in selected:
                    dist = min(abs(idx - sel_idx), 12 - abs(idx - sel_idx))
                    if dist < 1:
                        too_close = True
                        break
                if not too_close:
                    selected.append(idx)
            active_pcs = np.array(sorted(selected))
        else:
            chroma = self.extract(audio, sr, use_temporal=True, live_mode=live_mode)
            max_val = np.max(chroma)
            active_mask = chroma > (0.30 * max_val)
            active_pcs = np.where(active_mask)[0]
        
        # Get note names
        note_names = [self.note_names[pc] for pc in active_pcs]
        
        # Get frequencies ONLY for the selected pitch classes (not threshold-based!)
        frequencies = []
        for pc in active_pcs:
            midi_note = 60 + pc  # C4 + semitones
            freq = 440.0 * (2 ** ((midi_note - 69) / 12.0))
            frequencies.append(freq)
        frequencies = np.array(sorted(frequencies))
        
        # Energy dict for all pitch classes
        energy_dict = {
            self.note_names[i]: float(chroma[i]) 
            for i in range(12)
        }
        
        return {
            'chroma': chroma,
            'active_pitch_classes': active_pcs,
            'note_names': note_names,
            'energies': energy_dict,
            'frequencies': frequencies,
            'num_detected': len(active_pcs)
        }


def demo():
    """Demo of harmonic-aware chroma extraction"""
    print("=" * 70)
    print("Harmonic-Aware Chroma Extraction - Demo")
    print("=" * 70)
    
    # Generate test signal: C major chord (C4 + E4 + G4)
    sr = 44100
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # Frequencies for C major
    f_c = 261.63  # C4
    f_e = 329.63  # E4
    f_g = 392.00  # G4
    
    # Generate chord with harmonics (realistic)
    audio = np.zeros_like(t)
    
    for f0 in [f_c, f_e, f_g]:
        # Fundamental
        audio += 0.5 * np.sin(2 * np.pi * f0 * t)
        # 2nd harmonic (octave) - much quieter
        audio += 0.1 * np.sin(2 * np.pi * 2 * f0 * t)
        # 3rd harmonic (fifth) - even quieter
        audio += 0.05 * np.sin(2 * np.pi * 3 * f0 * t)
    
    # Normalize
    audio = audio / np.max(np.abs(audio))
    
    print("\nTest Signal: C major chord (C4 + E4 + G4)")
    print("With harmonics added to simulate real instrument")
    print()
    
    # Extract with standard librosa chroma
    print("1. Standard librosa chroma (FFT-based):")
    chroma_std = librosa.feature.chroma_stft(y=audio, sr=sr)
    chroma_std_mean = np.mean(chroma_std, axis=1)
    
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    threshold_std = 0.2 * np.max(chroma_std_mean)
    active_std = np.where(chroma_std_mean > threshold_std)[0]
    
    print(f"   Detected pitch classes: {active_std}")
    print(f"   Notes: {[note_names[i] for i in active_std]}")
    for pc in active_std:
        print(f"      {note_names[pc]}: {chroma_std_mean[pc]:.3f}")
    
    # Extract with harmonic-aware chroma
    print("\n2. Harmonic-Aware Chroma (our method):")
    extractor = HarmonicAwareChromaExtractor()
    result = extractor.analyze_with_debug(audio, sr)
    
    print(f"   Detected pitch classes: {result['active_pitch_classes']}")
    print(f"   Notes: {result['note_names']}")
    for note in result['note_names']:
        print(f"      {note}: {result['energies'][note]:.3f}")
    print(f"   Frequencies: {result['frequencies']}")
    
    print("\n" + "=" * 70)
    print("Expected: C, E, G (3 notes)")
    print(f"Standard chroma detected: {len(active_std)} notes")
    print(f"Harmonic-aware detected: {result['num_detected']} notes")
    
    if result['num_detected'] == 3 and set(result['note_names']) == {'C', 'E', 'G'}:
        print("✅ SUCCESS: Correctly detected C major triad!")
    else:
        print(f"⚠️  Detected: {result['note_names']}")
    print("=" * 70)


if __name__ == "__main__":
    demo()

