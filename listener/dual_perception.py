#!/usr/bin/env python3
"""
Dual Perception Module - Machine Logic vs Human Interface
==========================================================

KEY ARCHITECTURAL INSIGHT:
The machine should NOT work with chord names like "Cmaj" or "E7"!
Instead, it works in pure pattern space:

MACHINE LOGIC (Internal Processing):
    - Gesture tokens (0-63): Learned musical patterns in pure token space
    - Mathematical ratios: Psychoacoustic truth ([1.0, 1.25, 1.5])
    - Consonance scores: Perceptual reality (0.0-1.0)
    
    Machine thinks: "Token 42 usually followed by Token 87,
                    especially when consonance > 0.8 and ratios like [1.0, 1.2, 1.5]"

HUMAN INTERFACE (Translation Layer):
    - Chord names: "Cmaj", "E7", "Am7" (ONLY for display!)
    - Note names: "C", "E", "G" (ONLY for humans to understand!)
    
    Humans see: "Cmaj → Fmaj"

Philosophy:
    - Tokens ARE the meaningful patterns (not descriptions like "Cmaj")
    - Ratios ARE the mathematical relationships
    - Chord names are POST-HOC labels for human consumption
    - The machine discovers gestures that may not even have names!

Architecture:
    Audio → [Wav2Vec Encoder] → 768D features → Quantizer → Token 42
          → [Ratio Analyzer]   → [1.0, 1.26, 1.5] → Consonance 0.73
          ↓
          AudioOracle learns: Token patterns + ratio context
          ↓
          [TRANSLATION LAYER] ← Only when displaying to humans
          ↓
          Display: "Cmaj chord"

Based on:
- Bujard et al. (2025) - IRCAM Musical Agents work in symbolic token space
- Ragano et al. (2023) - Wav2Vec for music representation
- Psychological truth: Humans think in chord names, machines don't need to
"""

import numpy as np
import librosa  # For HPSS content detection
from typing import Optional, Dict, List, Tuple, Union
from dataclasses import dataclass

from listener.wav2vec_perception import Wav2VecMusicEncoder, Wav2VecFeatures
from listener.ratio_analyzer import FrequencyRatioAnalyzer, ChordAnalysis
from listener.harmonic_chroma import HarmonicAwareChromaExtractor
from listener.symbolic_quantizer import SymbolicQuantizer
from listener.gesture_smoothing import GestureTokenSmoother
from listener.chroma_validator import ChromaValidator, ChromaValidation


@dataclass
class DualPerceptionResult:
    """Result from dual perception analysis"""
    
    # Neural gesture representation (for machine learning)
    wav2vec_features: np.ndarray  # 768D neural encoding
    gesture_token: Optional[int]  # Quantized token (0-63) - LEGACY: use harmonic_token or percussive_token instead
    
    # Dual vocabulary tokens (for percussion-aware listening)
    harmonic_token: Optional[int] = None  # Token from harmonic vocabulary (0-63)
    percussive_token: Optional[int] = None  # Token from percussive vocabulary (0-63)
    
    # Ratio-based harmonic analysis (mathematical truth)
    ratio_analysis: Optional[ChordAnalysis] = None
    consonance: float = 0.5
    detected_frequencies: Optional[List[float]] = None
    
    # Chroma (for compatibility and display)
    chroma: Optional[np.ndarray] = None  # 12D
    active_pitch_classes: Optional[np.ndarray] = None
    
    # Human-friendly labels (derived from ratios, for display only)
    chord_label: str = "---"  # "Cmaj", "E7", "Am7", etc.
    chord_confidence: float = 0.0
    
    # Metadata
    timestamp: float = 0.0
    
    # Content type classification (for dual vocabulary mode)
    content_type: str = "hybrid"  # "harmonic", "percussive", or "hybrid"
    harmonic_ratio: float = 0.5  # Energy ratio of harmonic component (0-1)
    percussive_ratio: float = 0.5  # Energy ratio of percussive component (0-1)

    # Chroma validation (noise gate)
    chroma_validation: Optional[ChromaValidation] = None  # Validation result from ChromaValidator
    is_musical: bool = True  # Quick access to chroma_validation.is_musical
    musical_confidence: float = 1.0  # Quick access to chroma_validation.musical_confidence
    
    def __post_init__(self):
        """Ensure default values for mutable fields"""
        if self.detected_frequencies is None:
            self.detected_frequencies = []
        if self.chroma is None:
            self.chroma = np.zeros(12)
        if self.active_pitch_classes is None:
            self.active_pitch_classes = np.array([])
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage"""
        return {
            'wav2vec_features': self.wav2vec_features.tolist(),
            'gesture_token': self.gesture_token,  # LEGACY
            'harmonic_token': self.harmonic_token,  # Dual vocabulary
            'percussive_token': self.percussive_token,  # Dual vocabulary
            'ratio_analysis': {
                'fundamental': float(self.ratio_analysis.fundamental) if self.ratio_analysis else 0.0,
                'ratios': self.ratio_analysis.ratios if self.ratio_analysis else [],
                'consonance': self.consonance,
                'intervals': len(self.ratio_analysis.intervals) if self.ratio_analysis else 0
            } if self.ratio_analysis else None,
            'consonance': self.consonance,
            'detected_frequencies': self.detected_frequencies,
            'chroma': self.chroma.tolist() if self.chroma is not None else [],
            'active_pitch_classes': self.active_pitch_classes.tolist() if self.active_pitch_classes is not None else [],
            'chord_label': self.chord_label,
            'chord_confidence': self.chord_confidence,
            'timestamp': self.timestamp,
            'content_type': self.content_type,  # NEW: Content classification
            'harmonic_ratio': self.harmonic_ratio,  # NEW: Energy ratios
            'percussive_ratio': self.percussive_ratio,  # NEW: Energy ratios
            # Chroma validation (noise gate)
            'is_musical': self.is_musical,
            'musical_confidence': self.musical_confidence,
            'chroma_validation': {
                'is_musical': self.chroma_validation.is_musical,
                'musical_confidence': self.chroma_validation.musical_confidence,
                'should_process_mert': self.chroma_validation.should_process_mert,
                'reason': self.chroma_validation.reason
            } if self.chroma_validation else None
        }


class DualPerceptionModule:
    """
    Dual perception system combining:
    - Wav2Vec neural encoding (for machine learning in token space)
    - Ratio-based analysis (for mathematical context and human labels)
    
    The two systems run in parallel and complement each other:
    - Wav2Vec captures learned musical gestures
    - Ratios provide interpretable harmonic context
    """
    
    def __init__(self,
                 vocabulary_size: int = 64,
                 wav2vec_model: str = "facebook/wav2vec2-base",
                 use_gpu: bool = False,
                 enable_symbolic: bool = True,
                 gesture_window: float = 1.5,      # Reduced from 3.0s - more responsive to rapid changes
                 gesture_min_tokens: int = 2,      # Reduced from 3 - faster consensus
                 enable_dual_vocabulary: bool = False,  # NEW: Enable dual vocab mode
                 enable_wav2vec: bool = True,      # Control neural encoding
                 extract_all_frames: bool = True): # NEW: Extract all MERT frames (matches training)
        """
        Initialize dual perception module
        
        Args:
            vocabulary_size: Size of symbolic alphabet (16-64 recommended)
            wav2vec_model: HuggingFace model name (speech or music-pretrained)
            use_gpu: Use GPU for Wav2Vec (MPS/CUDA)
            enable_symbolic: Enable gesture token quantization
            gesture_window: Temporal smoothing window for gesture tokens (seconds)
                          1.5s balances phrase coherence with rhythmic responsiveness
            gesture_min_tokens: Minimum tokens needed for consensus (2 = faster response)
            enable_dual_vocabulary: Enable dual harmonic/percussive vocabularies (for drums)
            enable_wav2vec: Enable/disable neural audio encoding (MERT/Wav2Vec)
            extract_all_frames: If True, extract all MERT time-step frames (matches training, ~74 frames/sec)
                              If False, average frames to single vector (legacy, causes diversity collapse)
        """
        print("🔬 Initializing Dual Perception Module...")
        
        # Neural gesture pathway
        self.enable_wav2vec = enable_wav2vec
        self.extract_all_frames = extract_all_frames
        if self.enable_wav2vec:
            self.wav2vec_encoder = Wav2VecMusicEncoder(wav2vec_model, use_gpu)
            if extract_all_frames:
                print("   ⚡ All-frames mode: Extracting all MERT time steps (matches training)")
        else:
            self.wav2vec_encoder = None
            print("   Neural encoding (Wav2Vec/MERT) disabled")

        self.enable_dual_vocabulary = enable_dual_vocabulary
        
        if enable_dual_vocabulary:
            # Dual vocabulary mode: separate harmonic and percussive quantizers
            self.harmonic_quantizer = SymbolicQuantizer(vocabulary_size) if enable_symbolic else None
            self.percussive_quantizer = SymbolicQuantizer(vocabulary_size) if enable_symbolic else None
            self.quantizer = None  # LEGACY - not used in dual mode
            print(f"   Dual vocabulary mode: {vocabulary_size} harmonic + {vocabulary_size} percussive tokens")
        else:
            # Traditional mode: single quantizer
            self.quantizer = SymbolicQuantizer(vocabulary_size) if enable_symbolic else None
            self.harmonic_quantizer = None
            self.percussive_quantizer = None
        
        self.vocabulary_size = vocabulary_size
        self.enable_symbolic = enable_symbolic
        
        # Gesture token temporal smoothing (phrase-level coherence)
        self.gesture_smoother = GestureTokenSmoother(
            window_duration=gesture_window,
            min_tokens=gesture_min_tokens,
            decay_time=0.5  # Faster decay (0.5s) - more responsive to recent tokens
        )
        
        # Ratio-based harmonic pathway
        self.ratio_analyzer = FrequencyRatioAnalyzer()
        self.chroma_extractor = HarmonicAwareChromaExtractor()

        # Chroma-based validation (noise gate)
        self.chroma_validator = ChromaValidator(
            min_chroma_energy=0.5,      # Minimum total chroma energy
            min_chroma_peak=0.15,       # Minimum peak chroma value
            min_active_pitches=1,       # At least one clear pitch
            coherence_threshold=0.3,    # Pitch coherence (vs spread noise)
            gate_threshold=0.3          # Below this → skip MERT matching
        )

        print("✅ Dual perception initialized:")
        print(f"   Wav2Vec model: {wav2vec_model}")
        print(f"   Vocabulary size: {vocabulary_size} gesture tokens")
        print(f"   Gesture smoothing: {gesture_window}s window, {gesture_min_tokens} min tokens, 0.5s decay")
        print(f"   GPU: {'Yes' if use_gpu else 'No'}")
        print(f"   Dual vocabulary: {'Yes' if enable_dual_vocabulary else 'No'}")
    
    def extract_features(self,
                        audio: np.ndarray,
                        sr: int = 44100,
                        timestamp: float = 0.0,
                        detected_f0: Optional[float] = None) -> Union[DualPerceptionResult, List[DualPerceptionResult]]:
        """
        Extract features from both pathways
        
        Args:
            audio: Audio signal
            sr: Sample rate
            timestamp: Timestamp
            detected_f0: Detected fundamental frequency (for ratio analysis)
            
        Returns:
            If extract_all_frames=False: Single DualPerceptionResult
            If extract_all_frames=True: List of DualPerceptionResult (one per MERT frame)
        """
        
        # === DUAL VOCABULARY MODE: Separate harmonic and percussive sources ===
        if self.enable_dual_vocabulary and self.enable_wav2vec and self.wav2vec_encoder:
            # Perform HPSS separation
            audio_harmonic, audio_percussive = librosa.effects.hpss(
                audio,
                kernel_size=31,  # Standard separation quality
                power=2.0,
                mask=True
            )
            
            # Detect content type based on energy ratios
            content_type, harmonic_ratio, percussive_ratio = self.detect_content_type(audio, sr)
            
            # Extract Wav2Vec features from both sources
            harmonic_wav2vec_result = self.wav2vec_encoder.encode(
                audio_harmonic, sr, timestamp,
                return_all_frames=self.extract_all_frames
            )
            
            percussive_wav2vec_result = self.wav2vec_encoder.encode(
                audio_percussive, sr, timestamp,
                return_all_frames=self.extract_all_frames
            )
            
            # Handle all-frames mode with dual sources
            if self.extract_all_frames and isinstance(harmonic_wav2vec_result, list) and isinstance(percussive_wav2vec_result, list):
                # Process each frame pair and return list of results
                frame_results = []
                for harm_frame, perc_frame in zip(harmonic_wav2vec_result, percussive_wav2vec_result):
                    frame_result = self._process_dual_frame(
                        harmonic_features=harm_frame.features,
                        percussive_features=perc_frame.features,
                        timestamp=harm_frame.timestamp,
                        audio=audio,
                        sr=sr,
                        detected_f0=detected_f0,
                        content_type=content_type,
                        harmonic_ratio=harmonic_ratio,
                        percussive_ratio=percussive_ratio
                    )
                    frame_results.append(frame_result)
                return frame_results
            
            # Single-frame dual vocabulary mode
            harmonic_features = harmonic_wav2vec_result.features if not isinstance(harmonic_wav2vec_result, list) else harmonic_wav2vec_result[0].features
            percussive_features = percussive_wav2vec_result.features if not isinstance(percussive_wav2vec_result, list) else percussive_wav2vec_result[0].features
            
            return self._process_dual_frame(
                harmonic_features=harmonic_features,
                percussive_features=percussive_features,
                timestamp=timestamp,
                audio=audio,
                sr=sr,
                detected_f0=detected_f0,
                content_type=content_type,
                harmonic_ratio=harmonic_ratio,
                percussive_ratio=percussive_ratio
            )
        
        # === SINGLE VOCABULARY MODE (Legacy) ===
        # === PATHWAY 1: Neural Gesture Encoding (for machine) ===
        if self.enable_wav2vec and self.wav2vec_encoder:
            wav2vec_result = self.wav2vec_encoder.encode(
                audio, sr, timestamp, 
                return_all_frames=self.extract_all_frames
            )
        else:
            wav2vec_result = None
        
        # Handle all-frames mode: process each frame separately
        if self.extract_all_frames and wav2vec_result is not None and isinstance(wav2vec_result, list):
            # Process each frame and return list of results
            frame_results = []
            for frame_features_obj in wav2vec_result:
                frame_result = self._process_single_frame(
                    frame_features_obj.features,
                    frame_features_obj.timestamp,
                    audio, sr, detected_f0
                )
                frame_results.append(frame_result)
            return frame_results
        
        # Legacy single-frame mode
        if wav2vec_result is None:
            wav2vec_features = np.zeros(768)
        else:
            wav2vec_features = wav2vec_result.features if not isinstance(wav2vec_result, list) else wav2vec_result[0].features
        
        return self._process_single_frame(wav2vec_features, timestamp, audio, sr, detected_f0)
    
    def _process_dual_frame(self,
                           harmonic_features: np.ndarray,
                           percussive_features: np.ndarray,
                           timestamp: float,
                           audio: np.ndarray,
                           sr: int,
                           detected_f0: Optional[float],
                           content_type: str,
                           harmonic_ratio: float,
                           percussive_ratio: float) -> DualPerceptionResult:
        """
        Process dual MERT frames (harmonic + percussive) into a DualPerceptionResult
        
        This quantizes harmonic and percussive features separately using their respective vocabularies.
        """
        
        # Quantize harmonic features with harmonic vocabulary
        harmonic_token = None
        if self.harmonic_quantizer and self.harmonic_quantizer.is_fitted:
            harmonic_features_64 = harmonic_features.astype(np.float64)
            harmonic_token = int(self.harmonic_quantizer.transform(harmonic_features_64.reshape(1, -1))[0])
        
        # Quantize percussive features with percussive vocabulary
        percussive_token = None
        if self.percussive_quantizer and self.percussive_quantizer.is_fitted:
            percussive_features_64 = percussive_features.astype(np.float64)
            percussive_token = int(self.percussive_quantizer.transform(percussive_features_64.reshape(1, -1))[0])
        
        # For gesture_token (legacy), use the dominant source token
        if content_type == "harmonic":
            gesture_token = harmonic_token
            primary_features = harmonic_features
        elif content_type == "percussive":
            gesture_token = percussive_token
            primary_features = percussive_features
        else:  # hybrid
            # Use harmonic by default for hybrid content
            gesture_token = harmonic_token
            primary_features = harmonic_features
        
        # Apply temporal smoothing (on the primary gesture_token)
        smoothed_gesture_token = None
        if gesture_token is not None:
            smoothed_gesture_token = self.gesture_smoother.add_token(gesture_token, timestamp)
        
        # Use smoothed token for legacy compatibility
        final_gesture_token = smoothed_gesture_token if smoothed_gesture_token is not None else gesture_token
        
        # === PATHWAY 2: Ratio-Based Harmonic Analysis (for context + display) ===
        # Extract chroma and active pitch classes from original (combined) audio
        
        # DEBUG: Check audio buffer
        audio_rms = np.sqrt(np.mean(audio**2)) if len(audio) > 0 else 0.0
        audio_length = len(audio)
        
        chroma = self.chroma_extractor.extract(audio, sr, use_temporal=True, live_mode=True)
        chroma_result = self.chroma_extractor.extract_top_k(
            audio, sr, k=4, min_separation=2, min_threshold=0.15
        )
        active_pcs = chroma_result[1]
        
        # DEBUG: Log chroma extraction results
        if len(active_pcs) == 0:
            if not hasattr(self, '_chroma_debug_count'):
                self._chroma_debug_count = 0
            self._chroma_debug_count += 1
            if self._chroma_debug_count % 20 == 1:
                print(f"🔍 DEBUG: No pitch classes detected")
                print(f"  - chroma max: {chroma.max():.4f}, sum: {chroma.sum():.4f}")
                print(f"  - detected_f0: {detected_f0}")
                print(f"  - audio_rms: {audio_rms:.6f}, length: {audio_length} samples ({audio_length/sr:.3f}s)")
                print(f"  - sr: {sr}")
        
        # Analyze frequency ratios
        ratio_analysis = None
        consonance = 0.5
        detected_frequencies = []
        chord_label = "---"
        chord_confidence = 0.0
        
        if len(active_pcs) >= 2:
            # Convert pitch classes to frequencies
            if detected_f0 and detected_f0 > 0:
                # Use detected F0 to determine octave
                midi_bass = round(12 * np.log2(detected_f0 / 440.0) + 69)
                bass_pc = midi_bass % 12
                base_octave = midi_bass // 12
                
                for pc in active_pcs:
                    if pc == bass_pc:
                        detected_frequencies.append(detected_f0)
                    else:
                        midi_note = base_octave * 12 + pc
                        if midi_note < midi_bass:
                            midi_note += 12
                        freq = 440.0 * (2 ** ((midi_note - 69) / 12.0))
                        if 30 < freq < 8000:
                            detected_frequencies.append(freq)
            else:
                # Fallback: use middle octave
                for pc in active_pcs:
                    midi = 60 + pc
                    freq = 440.0 * (2 ** ((midi - 69) / 12.0))
                    detected_frequencies.append(freq)
            
            # Analyze ratios
            ratio_analysis = self.ratio_analyzer.analyze_frequencies(detected_frequencies)
            
            if ratio_analysis:
                consonance = ratio_analysis.consonance_score
                chord_label = ratio_analysis.chord_match['type']
                chord_confidence = ratio_analysis.chord_match['confidence']
        
        # === VALIDATE: Chroma-based noise gate ===
        chroma_validation = self.chroma_validator.validate(
            chroma=chroma,
            active_pitch_classes=active_pcs
        )

        # === COMBINE: Both representations in one result ===
        return DualPerceptionResult(
            wav2vec_features=primary_features,  # Use dominant source features
            gesture_token=final_gesture_token,  # Legacy field (smoothed)
            harmonic_token=harmonic_token,  # Dual vocabulary - separate token!
            percussive_token=percussive_token,  # Dual vocabulary - separate token!
            ratio_analysis=ratio_analysis,
            consonance=consonance,
            detected_frequencies=detected_frequencies,
            chroma=chroma,
            active_pitch_classes=active_pcs,
            chord_label=chord_label,
            chord_confidence=chord_confidence,
            timestamp=timestamp,
            content_type=content_type,  # NEW: Content type classification
            harmonic_ratio=harmonic_ratio,  # NEW: Energy ratios
            percussive_ratio=percussive_ratio,  # NEW: Energy ratios
            # Chroma validation (noise gate)
            chroma_validation=chroma_validation,
            is_musical=chroma_validation.is_musical,
            musical_confidence=chroma_validation.musical_confidence
        )

    def _process_single_frame(self,
                             wav2vec_features: np.ndarray,
                             timestamp: float,
                             audio: np.ndarray,
                             sr: int,
                             detected_f0: Optional[float]) -> DualPerceptionResult:
        """
        Process a single MERT frame into a DualPerceptionResult
        
        This is extracted as a helper to support both single-frame and all-frames modes.
        """
        
        # Quantize to gesture token (if enabled and fitted)
        raw_gesture_token = None
        smoothed_gesture_token = None
        
        # CRITICAL FIX: In dual vocabulary mode, use harmonic_quantizer for gesture tokens
        # (harmonic/percussive vocabularies are identical 768D Wav2Vec quantizers)
        active_quantizer = None
        if self.enable_dual_vocabulary and self.harmonic_quantizer and self.harmonic_quantizer.is_fitted:
            active_quantizer = self.harmonic_quantizer
        elif self.enable_symbolic and self.quantizer and self.quantizer.is_fitted:
            active_quantizer = self.quantizer
        
        if active_quantizer:
            # Ensure float64 for sklearn
            features_64 = wav2vec_features.astype(np.float64)
            
            # DEBUG: Check feature statistics before quantization
            if not hasattr(self, '_debug_feature_count'):
                self._debug_feature_count = 0
            self._debug_feature_count += 1
            
            if self._debug_feature_count % 20 == 1:
                print(f"\n🔍 DEBUG Wav2Vec Features (event {self._debug_feature_count}):")
                print(f"   Mean: {features_64.mean():.4f}, Std: {features_64.std():.4f}")
                print(f"   Min: {features_64.min():.4f}, Max: {features_64.max():.4f}")
                print(f"   First 5 values: {features_64[:5]}")
            
            raw_gesture_token = int(active_quantizer.transform(features_64.reshape(1, -1))[0])
            
            # DEBUG: Log token with feature stats
            if self._debug_feature_count % 20 == 1:
                print(f"   → Quantized to token: {raw_gesture_token}")
            
            # Apply temporal smoothing to get phrase-level token
            smoothed_gesture_token = self.gesture_smoother.add_token(raw_gesture_token, timestamp)
        
        # Use smoothed token as the primary gesture_token for AI
        gesture_token = smoothed_gesture_token if smoothed_gesture_token is not None else raw_gesture_token
        
        # DEBUG: Log gesture token diversity (every 10th call to avoid spam)
        if hasattr(self, '_gesture_debug_counter'):
            self._gesture_debug_counter += 1
        else:
            self._gesture_debug_counter = 1
            
        # Suppress debug output during training (tokens not assigned yet)
        # Only print during live performance when quantizer is fitted
        if self._gesture_debug_counter % 10 == 0 and gesture_token is not None:
            smoother_stats = self.gesture_smoother.get_statistics()
            print(f"🎯 Gesture tokens - Raw: {raw_gesture_token}, Smoothed: {gesture_token}, "
                  f"Window: {smoother_stats['tokens_in_window']}, Changes: {smoother_stats['consensus_changes']}")
        
        # === PATHWAY 2: Ratio-Based Harmonic Analysis (for context + display) ===
        
        # Extract chroma and active pitch classes
        chroma = self.chroma_extractor.extract(audio, sr, use_temporal=True, live_mode=True)
        chroma_result = self.chroma_extractor.extract_top_k(
            audio, sr, k=4, min_separation=2, min_threshold=0.15
        )
        active_pcs = chroma_result[1]
        
        # Analyze frequency ratios
        ratio_analysis = None
        consonance = 0.5
        detected_frequencies = []
        chord_label = "---"
        chord_confidence = 0.0
        
        if len(active_pcs) >= 2:
            # Convert pitch classes to frequencies
            if detected_f0 and detected_f0 > 0:
                # Use detected F0 to determine octave
                midi_bass = round(12 * np.log2(detected_f0 / 440.0) + 69)
                bass_pc = midi_bass % 12
                base_octave = midi_bass // 12
                
                for pc in active_pcs:
                    if pc == bass_pc:
                        detected_frequencies.append(detected_f0)
                    else:
                        midi_note = base_octave * 12 + pc
                        if midi_note < midi_bass:
                            midi_note += 12
                        freq = 440.0 * (2 ** ((midi_note - 69) / 12.0))
                        if 30 < freq < 8000:
                            detected_frequencies.append(freq)
            else:
                # Fallback: use middle octave
                for pc in active_pcs:
                    midi = 60 + pc
                    freq = 440.0 * (2 ** ((midi - 69) / 12.0))
                    detected_frequencies.append(freq)
            
            # Analyze ratios
            ratio_analysis = self.ratio_analyzer.analyze_frequencies(detected_frequencies)
            
            if ratio_analysis:
                consonance = ratio_analysis.consonance_score
                chord_label = ratio_analysis.chord_match['type']
                chord_confidence = ratio_analysis.chord_match['confidence']
        
        # === VALIDATE: Chroma-based noise gate ===
        chroma_validation = self.chroma_validator.validate(
            chroma=chroma,
            active_pitch_classes=active_pcs
        )

        # === COMBINE: Both representations in one result ===
        return DualPerceptionResult(
            wav2vec_features=wav2vec_features,
            gesture_token=gesture_token,  # Legacy field
            harmonic_token=gesture_token,  # Dual vocabulary - use same token for now
            percussive_token=gesture_token,  # Dual vocabulary - use same token for now
            ratio_analysis=ratio_analysis,
            consonance=consonance,
            detected_frequencies=detected_frequencies,
            chroma=chroma,
            active_pitch_classes=active_pcs,
            chord_label=chord_label,
            chord_confidence=chord_confidence,
            timestamp=timestamp,
            # Chroma validation (noise gate)
            chroma_validation=chroma_validation,
            is_musical=chroma_validation.is_musical,
            musical_confidence=chroma_validation.musical_confidence
        )
    
    def detect_content_type(self, audio: np.ndarray, sr: int = 44100) -> Tuple[str, float, float]:
        """
        Detect whether audio is primarily harmonic, percussive, or hybrid
        using HPSS (Harmonic-Percussive Source Separation).
        
        This enables the system to respond appropriately:
        - Percussive input (drums) → AI generates harmonic response
        - Harmonic input (guitar/keys) → AI generates rhythmic response
        - Hybrid input → AI fills in contextually
        
        Args:
            audio: Audio signal
            sr: Sample rate
            
        Returns:
            Tuple of (content_type, harmonic_ratio, percussive_ratio)
            where content_type is "harmonic", "percussive", or "hybrid"
        """
        # Apply HPSS to separate harmonic and percussive components
        y_harmonic, y_percussive = librosa.effects.hpss(
            audio,
            kernel_size=31,  # Larger = better separation but more latency
            power=2.0,       # Standard power spectrogram
            mask=True        # Use masking for cleaner separation
        )
        
        # Calculate energy in each component
        harmonic_energy = np.sum(y_harmonic ** 2)
        percussive_energy = np.sum(y_percussive ** 2)
        total_energy = harmonic_energy + percussive_energy
        
        if total_energy < 1e-6:
            # Silence or noise
            return "hybrid", 0.5, 0.5
        
        # Normalize to ratios
        harmonic_ratio = harmonic_energy / total_energy
        percussive_ratio = percussive_energy / total_energy
        
        # Classify based on dominant component
        # Threshold: 0.7 means 70% of energy in one component
        if percussive_ratio > 0.7:
            return "percussive", harmonic_ratio, percussive_ratio
        elif harmonic_ratio > 0.7:
            return "harmonic", harmonic_ratio, percussive_ratio
        else:
            return "hybrid", harmonic_ratio, percussive_ratio
    
    def train_gesture_vocabulary(self, wav2vec_features_list: List[np.ndarray],
                                 vocabulary_type: str = "single"):
        """
        Train gesture token vocabulary from Wav2Vec features
        
        This learns the k-means codebook that maps 768D features → 64 tokens
        
        Args:
            wav2vec_features_list: List of 768D feature vectors
            vocabulary_type: "single" (traditional), "harmonic", or "percussive"
        """
        if not self.enable_symbolic:
            print("⚠️ Symbolic quantization not enabled")
            return
        
        print(f"🎯 Training {vocabulary_type} gesture vocabulary ({self.vocabulary_size} tokens)...")
        print(f"   Using {len(wav2vec_features_list)} feature vectors")
        
        # Convert to float64 array for sklearn
        features_array = np.array([f.astype(np.float64) for f in wav2vec_features_list])
        
        # Select appropriate quantizer
        if vocabulary_type == "harmonic":
            if self.harmonic_quantizer is None:
                print("⚠️ Harmonic quantizer not initialized (enable_dual_vocabulary=False)")
                return
            quantizer = self.harmonic_quantizer
        elif vocabulary_type == "percussive":
            if self.percussive_quantizer is None:
                print("⚠️ Percussive quantizer not initialized (enable_dual_vocabulary=False)")
                return
            quantizer = self.percussive_quantizer
        else:  # "single" or default
            if self.quantizer is None:
                print("⚠️ Single quantizer not initialized (enable_dual_vocabulary=True)")
                return
            quantizer = self.quantizer
        
        # Train quantizer
        quantizer.fit(features_array)
        
        # Print stats
        stats = quantizer.get_codebook_statistics()
        print(f"✅ {vocabulary_type.capitalize()} gesture vocabulary trained!")
        print(f"   Active tokens: {stats['active_tokens']}/{stats['vocabulary_size']}")
        print(f"   Entropy: {stats['entropy']:.2f} bits")
    
    def save_vocabulary(self, filepath: str, vocabulary_type: str = "single"):
        """
        Save trained gesture vocabulary
        
        Args:
            filepath: Path to save vocabulary
            vocabulary_type: "single", "harmonic", or "percussive"
        """
        if vocabulary_type == "harmonic" and self.harmonic_quantizer and self.harmonic_quantizer.is_fitted:
            self.harmonic_quantizer.save(filepath)
        elif vocabulary_type == "percussive" and self.percussive_quantizer and self.percussive_quantizer.is_fitted:
            self.percussive_quantizer.save(filepath)
        elif vocabulary_type == "single" and self.quantizer and self.quantizer.is_fitted:
            self.quantizer.save(filepath)
        else:
            print(f"⚠️ Cannot save {vocabulary_type} vocabulary (not fitted or not initialized)")
    
    def load_vocabulary(self, filepath: str, vocabulary_type: str = "single"):
        """
        Load trained gesture vocabulary
        
        Args:
            filepath: Path to load vocabulary from
            vocabulary_type: "single", "harmonic", or "percussive"
        """
        if vocabulary_type == "harmonic" and self.harmonic_quantizer:
            self.harmonic_quantizer.load(filepath)
            print(f"📂 Harmonic gesture vocabulary loaded: {filepath}")
        elif vocabulary_type == "percussive" and self.percussive_quantizer:
            self.percussive_quantizer.load(filepath)
            print(f"📂 Percussive gesture vocabulary loaded: {filepath}")
        elif vocabulary_type == "single" and self.quantizer:
            self.quantizer.load(filepath)
            print(f"📂 Gesture vocabulary loaded: {filepath}")
        else:
            print(f"⚠️ Cannot load {vocabulary_type} vocabulary (quantizer not initialized)")
    
    # Aliases for consistency with HybridPerceptionModule
    def save_quantizer(self, filepath: str):
        """Alias for save_vocabulary()"""
        self.save_vocabulary(filepath)
    
    def load_quantizer(self, filepath: str):
        """Alias for load_vocabulary()"""
        self.load_vocabulary(filepath)
    
    def get_gesture_smoothing_stats(self) -> dict:
        """Get gesture token smoothing statistics for transparency/debugging."""
        return self.gesture_smoother.get_statistics()
    
    def reset_gesture_smoothing(self):
        """Reset gesture token smoothing window (e.g., on behavioral mode change)."""
        self.gesture_smoother.reset()


def ratio_to_chord_name(ratio_analysis: Optional[ChordAnalysis], 
                        root_freq: float = 0.0) -> str:
    """
    Convert ratio analysis to human-friendly chord name
    
    This is ONLY for display purposes - the machine doesn't use this!
    
    Args:
        ratio_analysis: Ratio analysis result
        root_freq: Root frequency (for determining note name)
        
    Returns:
        Chord name like "Cmaj", "E7", "Am7"
    """
    if not ratio_analysis or ratio_analysis.chord_match['confidence'] < 0.3:
        return "---"
    
    # Get chord type
    chord_type = ratio_analysis.chord_match['type']
    
    # Get root note from fundamental frequency
    if root_freq > 0:
        midi = round(12 * np.log2(root_freq / 440.0) + 69)
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        root_note = note_names[midi % 12]
    else:
        root_note = "?"
    
    # Combine
    if chord_type == "major":
        return root_note
    elif chord_type == "minor":
        return f"{root_note}m"
    elif chord_type == "dominant_7th":
        return f"{root_note}7"
    elif chord_type == "minor_7th":
        return f"{root_note}m7"
    elif chord_type == "major_7th":
        return f"{root_note}maj7"
    elif chord_type == "diminished":
        return f"{root_note}dim"
    elif chord_type == "diminished_7th":
        return f"{root_note}dim7"
    elif chord_type == "augmented":
        return f"{root_note}aug"
    else:
        return f"{root_note}{chord_type}"


def demo():
    """Demo dual perception on synthetic chords"""
    print("=" * 70)
    print("🎵 Dual Perception Module - Demo")
    print("=" * 70)
    
    # Create module
    perception = DualPerceptionModule(
        vocabulary_size=64,
        wav2vec_model="facebook/wav2vec2-base",
        use_gpu=True,
        enable_symbolic=False  # Will train vocabulary later
    )
    
    # Test chords
    test_chords = [
        ("C major", [261.63, 329.63, 392.00]),
        ("E7", [329.63, 415.30, 493.88, 587.33]),
        ("Am7", [220.00, 261.63, 329.63, 392.00])
    ]
    
    sr = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    
    wav2vec_features_collection = []
    
    for chord_name, frequencies in test_chords:
        print(f"\n🎵 Testing: {chord_name}")
        print(f"   Frequencies: {frequencies}")
        
        # Generate audio
        audio = np.zeros_like(t)
        for freq in frequencies:
            audio += np.sin(2 * np.pi * freq * t)
        audio = (audio / len(frequencies)).astype(np.float32)
        
        # Extract features
        result = perception.extract_features(
            audio, sr, timestamp=time.time(), 
            detected_f0=frequencies[0]
        )
        
        wav2vec_features_collection.append(result.wav2vec_features)
        
        print(f"   ✅ Wav2Vec features: {result.wav2vec_features.shape}")
        print(f"   ✅ Ratio analysis: {result.chord_label} (conf: {result.chord_confidence:.1%})")
        print(f"   ✅ Consonance: {result.consonance:.3f}")
        print(f"   ✅ Detected ratios: {[f'{r:.3f}' for r in result.ratio_analysis.ratios] if result.ratio_analysis else 'N/A'}")
    
    # Train gesture vocabulary
    print("\n🎯 Training gesture vocabulary...")
    perception.quantizer = SymbolicQuantizer(vocabulary_size=64)
    perception.enable_symbolic = True
    perception.train_gesture_vocabulary(wav2vec_features_collection)
    
    # Now extract with tokens
    print("\n🔄 Extracting features with gesture tokens...")
    for chord_name, frequencies in test_chords:
        audio = np.zeros_like(t)
        for freq in frequencies:
            audio += np.sin(2 * np.pi * freq * t)
        audio = (audio / len(frequencies)).astype(np.float32)
        
        result = perception.extract_features(
            audio, sr, timestamp=time.time(),
            detected_f0=frequencies[0]
        )
        
        print(f"\n{chord_name}:")
        print(f"   Machine view: Gesture Token {result.gesture_token}")
        print(f"   Human view: {result.chord_label} (ratios: {[f'{r:.2f}' for r in result.ratio_analysis.ratios[:3]] if result.ratio_analysis else 'N/A'})")
        print(f"   Context: consonance={result.consonance:.3f}, confidence={result.chord_confidence:.1%}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    import time
    demo()

