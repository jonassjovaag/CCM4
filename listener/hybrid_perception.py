#!/usr/bin/env python3
"""
Hybrid Perception Module
========================

Combines the best of:
- Our ratio-based harmonic analysis (interpretable, psychoacoustic)
- IRCAM's symbolic quantization approach (efficient, learnable)
- Existing spectral features (proven, reliable)

Architecture:
    Audio → [Ratio Analyzer] → Ratio features (interpretable)
          → [Harmonic Chroma] → Chroma features (harmonic-aware)
          → [Spectral Analysis] → Spectral features (timbral)
          → [CONCATENATE] → Full feature vector
          → [Vector Quantization] → Symbolic token (for memory efficiency)

Output:
    - Continuous features: For ML classification
    - Symbolic token: For AudioOracle/pattern memory
    - Ratio analysis: For musical interpretation
    - Consonance score: For decision making
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from listener.ratio_analyzer import FrequencyRatioAnalyzer, ChordAnalysis
from listener.harmonic_chroma import HarmonicAwareChromaExtractor
from listener.symbolic_quantizer import SymbolicQuantizer


@dataclass
class HybridPerceptionResult:
    """Complete perception analysis result"""
    # Continuous features (for ML)
    features: np.ndarray  # Full feature vector
    
    # Symbolic representation (for memory)
    symbolic_token: Optional[int]  # Quantized class ID
    
    # Ratio analysis (interpretable)
    ratio_analysis: Optional[ChordAnalysis]
    consonance: float
    
    # Chroma (harmonic)
    chroma: np.ndarray  # 12-dimensional
    active_pitch_classes: np.ndarray
    
    # Metadata
    timestamp: float
    feature_breakdown: Dict  # Which features contributed what


class HybridPerceptionModule:
    """
    Combined perception module integrating:
    - Ratio-based harmonic analysis (ours)
    - Harmonic-aware chroma (ours, based on Kronvall et al.)
    - Symbolic quantization (IRCAM approach)
    - Wav2Vec 2.0 neural encoding (optional, Ragano et al. 2023)
    """
    
    def __init__(self, 
                 vocabulary_size: int = 64,
                 enable_ratio_analysis: bool = True,
                 enable_symbolic: bool = True,
                 enable_wav2vec: bool = False,
                 wav2vec_model: str = "facebook/wav2vec2-base",
                 use_gpu: bool = False):
        """
        Initialize hybrid perception
        
        Args:
            vocabulary_size: Size of symbolic alphabet (16, 64, or 256)
            enable_ratio_analysis: Include ratio-based features
            enable_symbolic: Use vector quantization
            enable_wav2vec: Use Wav2Vec 2.0 neural encoding (replaces chroma+ratio)
            wav2vec_model: HuggingFace model name for Wav2Vec
            use_gpu: Use GPU for Wav2Vec (MPS/CUDA)
        """
        self.vocabulary_size = vocabulary_size
        self.enable_ratio = enable_ratio_analysis
        self.enable_symbolic = enable_symbolic
        self.enable_wav2vec = enable_wav2vec
        
        # Sub-modules
        if enable_wav2vec:
            # Use Wav2Vec encoder for perceptual features
            from listener.wav2vec_perception import Wav2VecMusicEncoder
            self.wav2vec_encoder = Wav2VecMusicEncoder(wav2vec_model, use_gpu)
            # ALSO run ratio analyzer for mathematical harmonic analysis (parallel)
            self.ratio_analyzer = FrequencyRatioAnalyzer() if enable_ratio_analysis else None
            self.chroma_extractor = None  # Don't need chroma if we have Wav2Vec
            print(f"🎵 Using Wav2Vec encoder: {wav2vec_model}")
            if self.ratio_analyzer:
                print(f"   + Ratio analyzer: parallel harmonic analysis")
        else:
            # Traditional ratio + chroma approach
            self.wav2vec_encoder = None
            self.ratio_analyzer = FrequencyRatioAnalyzer() if enable_ratio_analysis else None
            self.chroma_extractor = HarmonicAwareChromaExtractor()
        
        self.quantizer = SymbolicQuantizer(vocabulary_size) if enable_symbolic else None
        
        # Feature dimensions
        if enable_wav2vec:
            # Wav2Vec outputs 768D features by default
            self.total_dim = 768  # Will be set after model loads
        else:
            self.chroma_dim = 12
            self.ratio_dim = 10  # ratios + consonance + interval features
            self.total_dim = self.chroma_dim + (self.ratio_dim if enable_ratio_analysis else 0)
    
    def extract_features(self, audio: np.ndarray, sr: int = 44100,
                        timestamp: float = 0.0, detected_f0: Optional[float] = None) -> HybridPerceptionResult:
        """
        Extract comprehensive features from audio
        
        Args:
            audio: Audio signal
            sr: Sample rate
            timestamp: Timestamp for this analysis
            detected_f0: Detected fundamental frequency (optional, for ratio analysis)
            
        Returns:
            HybridPerceptionResult with all features
        """
        # Debug: check buffer size (first time only)
        if not hasattr(self, '_buffer_size_checked'):
            # Silently note buffer size (don't print for clean terminal)
            self._buffer_size_checked = True
        
        # NEW: Wav2Vec encoding path
        if self.enable_wav2vec:
            return self._extract_wav2vec_features(audio, sr, timestamp, detected_f0)
        
        # Traditional ratio + chroma path
        # 1. Harmonic-aware chroma
        chroma = self.chroma_extractor.extract(audio, sr, use_temporal=True, live_mode=True)
        # Use top-k with threshold for clean chord detection
        # k=4 means max 4 notes, min_threshold=0.15 means must be 15% of strongest note (lowered from 0.4)
        chroma_result = self.chroma_extractor.extract_top_k(
            audio, sr, k=4, min_separation=2, min_threshold=0.15
        )
        active_pcs = chroma_result[1]
        
        # 2. Ratio analysis (if enabled)
        ratio_analysis = None
        consonance = 0.5  # Default
        ratio_features = np.zeros(self.ratio_dim)
        
        if self.enable_ratio and len(active_pcs) >= 2:
            # Convert pitch classes to frequencies
            # Use detected_f0 to determine correct octave if available
            frequencies = []
            
            if detected_f0 and detected_f0 > 0:
                # Find which pitch class corresponds to the bass note
                midi_bass = round(12 * np.log2(detected_f0 / 440.0) + 69)
                bass_pc = midi_bass % 12
                base_octave = midi_bass // 12
                
                # Build frequency list from active pitch classes
                # including the detected F0 for the matching pitch class
                for pc in active_pcs:
                    if pc == bass_pc:
                        # Use actual detected frequency for bass note
                        frequencies.append(detected_f0)
                    else:
                        # Place other notes in same or next octave
                        midi_note = base_octave * 12 + pc
                        # If note would be below bass, move up an octave
                        if midi_note < midi_bass:
                            midi_note += 12
                        freq = 440.0 * (2 ** ((midi_note - 69) / 12.0))
                        if 30 < freq < 8000:
                            frequencies.append(freq)
            else:
                # Fallback: use arbitrary octave (C4 = 261Hz)
                for pc in active_pcs:
                    midi = 60 + pc
                    freq = 440.0 * (2 ** ((midi - 69) / 12.0))
                    frequencies.append(freq)
            
            ratio_analysis = self.ratio_analyzer.analyze_frequencies(frequencies)
            
            # Debug: show what we're analyzing (first few times)
            if not hasattr(self, '_ratio_debug_count'):
                self._ratio_debug_count = 0
            if self._ratio_debug_count < 3 and len(frequencies) > 1:
                freq_str = ", ".join([f"{f:.1f}Hz" for f in frequencies[:5]])
                print(f"\n🔍 Ratio analysis: {len(frequencies)} freqs: {freq_str}")
                if ratio_analysis:
                    print(f"   → Chord: {ratio_analysis.chord_match['type']} ({ratio_analysis.chord_match['confidence']:.0%})")
                self._ratio_debug_count += 1
            
            if ratio_analysis:
                consonance = ratio_analysis.consonance_score
                
                # Create ratio feature vector
                ratio_features = np.array([
                    ratio_analysis.fundamental / 1000.0,  # Normalized fundamental
                    *ratio_analysis.ratios[:4],  # Up to 4 ratios
                    *([0.0] * (4 - len(ratio_analysis.ratios[:4]))),  # Pad if needed
                    ratio_analysis.consonance_score,
                    len(ratio_analysis.intervals),  # Number of intervals
                    float(ratio_analysis.chord_match['confidence']),
                    # Encode chord type as number (for now)
                    hash(ratio_analysis.chord_match['type']) % 1000 / 1000.0,
                ])[:self.ratio_dim]  # Ensure exact size
        
        # 3. Combine features
        if self.enable_ratio:
            full_features = np.concatenate([chroma, ratio_features])
        else:
            full_features = chroma
        
        # 4. Symbolic quantization (if enabled and fitted)
        symbolic_token = None
        if self.enable_symbolic and self.quantizer and self.quantizer.is_fitted:
            symbolic_token = int(self.quantizer.transform(full_features.reshape(1, -1))[0])
        
        # 5. Build result
        feature_breakdown = {
            'chroma': chroma.tolist(),
            'ratio_features': ratio_features.tolist() if self.enable_ratio else [],
            'active_pitch_classes': active_pcs.tolist(),
            'consonance': consonance
        }
        
        return HybridPerceptionResult(
            features=full_features,
            symbolic_token=symbolic_token,
            ratio_analysis=ratio_analysis,
            consonance=consonance,
            chroma=chroma,
            active_pitch_classes=active_pcs,
            timestamp=timestamp,
            feature_breakdown=feature_breakdown
        )
    
    def train_vocabulary(self, feature_samples: List[np.ndarray]):
        """
        Train symbolic vocabulary from feature samples
        
        Args:
            feature_samples: List of feature vectors from training data
        """
        if not self.enable_symbolic:
            print("⚠️ Symbolic quantization not enabled")
            return
        
        features_array = np.array(feature_samples)
        self.quantizer.fit(features_array)
        
        print("✅ Symbolic vocabulary trained!")
        stats = self.quantizer.get_codebook_statistics()
        print(f"   Vocabulary utilization: {stats['active_tokens']}/{stats['vocabulary_size']}")
        print(f"   Entropy: {stats['entropy']:.2f} bits")
    
    def save_quantizer(self, filepath: str):
        """Save trained quantizer"""
        if self.quantizer and self.quantizer.is_fitted:
            self.quantizer.save(filepath)
    
    def load_quantizer(self, filepath: str):
        """Load trained quantizer"""
        if self.quantizer:
            self.quantizer.load(filepath)
    
    def _extract_wav2vec_features(self, audio: np.ndarray, sr: int, 
                                  timestamp: float, detected_f0: Optional[float] = None) -> HybridPerceptionResult:
        """
        Extract features using Wav2Vec 2.0 encoder
        
        Also runs ratio analysis in parallel for mathematical harmonic analysis.
        """
        # Encode audio with Wav2Vec
        wav2vec_result = self.wav2vec_encoder.encode(audio, sr, timestamp)
        
        if wav2vec_result is None:
            # Fallback to zeros if encoding fails
            features = np.zeros(self.total_dim)
            consonance = 0.5
            chroma = np.zeros(12)
            active_pcs = np.array([])
        else:
            features = wav2vec_result.features
            
            # Extract pseudo-chroma from Wav2Vec features for compatibility
            # (Use first 12 dimensions as proxy)
            chroma = features[:12] if len(features) >= 12 else np.zeros(12)
            # Normalize to [0, 1] range
            chroma = (chroma - chroma.min()) / (chroma.max() - chroma.min() + 1e-8)
            
            # Extract active pitch classes (top 3)
            active_pcs = np.argsort(chroma)[-3:][::-1]
            
            # Estimate consonance from feature variance
            # (Lower variance = more consonant)
            consonance = 1.0 - min(1.0, np.std(features) * 2.0)
        
        # PARALLEL: Run ratio analysis for mathematical harmonic analysis
        ratio_analysis = None
        if self.ratio_analyzer and len(active_pcs) >= 2 and detected_f0 and detected_f0 > 0:
            # Build frequency list from active pitch classes and detected F0
            frequencies = []
            midi_bass = round(12 * np.log2(detected_f0 / 440.0) + 69)
            bass_pc = midi_bass % 12
            base_octave = midi_bass // 12
            
            for pc in active_pcs:
                if pc == bass_pc:
                    frequencies.append(detected_f0)
                else:
                    midi_note = base_octave * 12 + pc
                    if midi_note < midi_bass:
                        midi_note += 12
                    freq = 440.0 * (2 ** ((midi_note - 69) / 12.0))
                    if 30 < freq < 8000:
                        frequencies.append(freq)
            
            if len(frequencies) >= 2:
                ratio_analysis = self.ratio_analyzer.analyze_frequencies(frequencies)
                if ratio_analysis:
                    # Use ratio analyzer's consonance (more accurate than variance estimate)
                    consonance = ratio_analysis.consonance_score
                    # DEBUG: Log chord detection (first 5 times)
                    if not hasattr(self, '_chord_debug_count'):
                        self._chord_debug_count = 0
                    if self._chord_debug_count < 5:
                        print(f"🎸 Ratio chord detected: {ratio_analysis.chord_match['type']} ({ratio_analysis.chord_match['confidence']:.1%})")
                        self._chord_debug_count += 1
                else:
                    # DEBUG: Log why ratio analysis failed
                    if not hasattr(self, '_ratio_fail_count'):
                        self._ratio_fail_count = 0
                    if self._ratio_fail_count < 3:
                        print(f"🔍 Ratio analysis returned None for freqs: {frequencies}")
                        self._ratio_fail_count += 1
            else:
                # DEBUG: Not enough frequencies
                if not hasattr(self, '_freq_count_debug'):
                    self._freq_count_debug = 0
                if self._freq_count_debug < 3:
                    print(f"🔍 Only {len(frequencies)} frequencies, need >= 2")
                    self._freq_count_debug += 1
        else:
            # DEBUG: Log why ratio analysis was skipped
            if not hasattr(self, '_ratio_skip_count'):
                self._ratio_skip_count = 0
            if self._ratio_skip_count < 3:
                reason = []
                if not self.ratio_analyzer:
                    reason.append("no analyzer")
                if len(active_pcs) < 2:
                    reason.append(f"only {len(active_pcs)} active_pcs")
                if not detected_f0 or detected_f0 <= 0:
                    reason.append(f"invalid f0={detected_f0}")
                print(f"🔍 Ratio analysis skipped: {', '.join(reason)}")
                self._ratio_skip_count += 1
        
        # Symbolic quantization (if enabled and fitted)
        symbolic_token = None
        if self.enable_symbolic and self.quantizer and self.quantizer.is_fitted:
            symbolic_token = int(self.quantizer.transform(features.reshape(1, -1))[0])
        
        # Build result
        feature_breakdown = {
            'wav2vec_features': features.tolist(),
            'pseudo_chroma': chroma.tolist(),
            'active_pitch_classes': active_pcs.tolist(),
            'consonance': consonance
        }
        
        return HybridPerceptionResult(
            features=features,
            symbolic_token=symbolic_token,
            ratio_analysis=ratio_analysis,  # Now includes ratio analysis!
            consonance=consonance,
            chroma=chroma,
            active_pitch_classes=active_pcs,
            timestamp=timestamp,
            feature_breakdown=feature_breakdown
        )


def demo():
    """Demo of hybrid perception"""
    import time
    
    print("=" * 70)
    print("Hybrid Perception Module - Demo")
    print("=" * 70)
    
    # Generate synthetic audio (C major chord)
    sr = 44100
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # C major: C4 (261.63) + E4 (329.63) + G4 (392.00)
    audio = (
        0.5 * np.sin(2 * np.pi * 261.63 * t) +
        0.5 * np.sin(2 * np.pi * 329.63 * t) +
        0.5 * np.sin(2 * np.pi * 392.00 * t)
    )
    audio = audio / np.max(np.abs(audio))
    
    # Create hybrid perception module
    perception = HybridPerceptionModule(
        vocabulary_size=64,
        enable_ratio_analysis=True,
        enable_symbolic=False  # Will train later
    )
    
    print("\n🎵 Analyzing C major chord...")
    result = perception.extract_features(audio, sr, timestamp=time.time())
    
    print(f"\n📊 Results:")
    print(f"   Feature vector shape: {result.features.shape}")
    print(f"   Chroma peaks: {result.active_pitch_classes}")
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    detected_notes = [note_names[pc] for pc in result.active_pitch_classes]
    print(f"   Note names: {detected_notes}")
    
    if result.ratio_analysis:
        print(f"\n   Ratio Analysis:")
        print(f"      Chord type: {result.ratio_analysis.chord_match['type']}")
        print(f"      Confidence: {result.ratio_analysis.chord_match['confidence']:.1%}")
        print(f"      Consonance: {result.consonance:.3f}")
        print(f"      Ratios: {[f'{r:.3f}' for r in result.ratio_analysis.ratios]}")
    
    print("\n" + "=" * 70)
    print("Feature vector ready for:")
    print("  • ML classification (RandomForest, SVM, etc.)")
    print("  • AudioOracle pattern learning")
    print("  • Symbolic quantization (when vocabulary is trained)")
    print("=" * 70)


if __name__ == "__main__":
    demo()


