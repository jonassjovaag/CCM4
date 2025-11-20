#!/usr/bin/env python3
"""
Wav2Vec 2.0 Music Perception Module
====================================

Implements neural audio encoding using Wav2Vec 2.0 pre-trained on music.

Based on:
- Ragano et al. (2023) - Wav2Vec 2.0 for music
- Bujard et al. (2025) - IRCAM Musical Agents paper
- Baevski et al. (2020) - Original Wav2Vec 2.0

Architecture:
    Audio → Wav2Vec Encoder → Condenser (temporal avg) → Features
    
Features:
    - 768D contextualized audio features (from Wav2Vec)
    - Rich musical information automatically learned
    - Better than hand-crafted features for complex music
"""

import numpy as np
import torch
import torchaudio
import contextlib
import os
from typing import Optional, Dict
from dataclasses import dataclass


@dataclass
class Wav2VecFeatures:
    """Wav2Vec audio encoding result"""
    features: np.ndarray  # 768D encoded features
    timestamp: float
    duration: float  # Duration of audio analyzed
    sample_rate: int


class Wav2VecMusicEncoder:
    """
    Wav2Vec 2.0 encoder for music
    
    Uses pre-trained Wav2Vec model to extract rich musical features
    """
    
    def __init__(self, 
                 model_name: str = "facebook/wav2vec2-base",
                 use_gpu: bool = False):
        """
        Initialize Wav2Vec encoder
        
        Args:
            model_name: HuggingFace model name or path
                       Options:
                       - "facebook/wav2vec2-base" (standard)
                       - "facebook/wav2vec2-large" (better but slower)
                       - Custom music-pretrained model if available
            use_gpu: Use GPU if available (Mac: MPS, NVIDIA: CUDA)
        """
        self.model_name = model_name
        
        # Determine device
        if use_gpu:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
                print(f"🎮 Using MPS (Apple Silicon GPU) for Wav2Vec")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
                print(f"🎮 Using CUDA GPU for Wav2Vec")
            else:
                self.device = torch.device("cpu")
                print(f"💻 GPU not available, using CPU for Wav2Vec")
        else:
            self.device = torch.device("cpu")
            print(f"💻 Using CPU for Wav2Vec")
        
        # Load model (lazy loading - only load when first used)
        self.model = None
        self.processor = None
        self._initialized = False
        
        # Cache for feature dimensions
        self.feature_dim = 768  # Wav2Vec base outputs 768D
    
    def _initialize_model(self):
        """Lazy load the model (first call only)"""
        if self._initialized:
            return
        
        try:
            print(f"🔄 Loading Neural Audio Model: {self.model_name}...")
            
            # Import transformers (only when needed)
            # Support both Wav2Vec2 and MERT models
            try:
                from transformers import Wav2Vec2FeatureExtractor, AutoModel
                import logging
                import warnings
            except ImportError:
                print("❌ transformers library not installed!")
                print("   Install: pip install transformers")
                return
            
            # Load processor and model
            # Use AutoModel to support both Wav2Vec2 and MERT
            # trust_remote_code=True required for MERT models
            
            # Suppress nnAudio warning from MERT
            # The warning "feature_extractor_cqt requires the libray 'nnAudio'" 
            # comes from remote code and is hard to suppress via logging/warnings.
            # We redirect stdout/stderr to devnull during loading to silence it.
            logging.getLogger("transformers").setLevel(logging.ERROR)
            
            with open(os.devnull, 'w') as fnull:
                with contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
                    self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
                        self.model_name,
                        trust_remote_code=True
                    )
                    self.model = AutoModel.from_pretrained(
                        self.model_name,
                        trust_remote_code=True
                    )
            
            # Move to device
            self.model = self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            # Update feature dimension from model config
            self.feature_dim = self.model.config.hidden_size
            
            self._initialized = True
            print(f"✅ Neural Audio model loaded!")
            print(f"   Feature dimension: {self.feature_dim}D")
            print(f"   Device: {self.device}")
            
        except Exception as e:
            print(f"❌ Failed to load Wav2Vec model: {e}")
            print("   Falling back to standard features")
            self._initialized = False
    
    def encode(self, 
               audio: np.ndarray,
               sr: int = 44100,
               timestamp: float = 0.0) -> Optional[Wav2VecFeatures]:
        """
        Encode audio using Wav2Vec
        
        Args:
            audio: Audio signal (mono)
            sr: Sample rate
            timestamp: Timestamp for this analysis
            
        Returns:
            Wav2VecFeatures with 768D encoding
        """
        # Initialize model on first call
        if not self._initialized:
            self._initialize_model()
        
        if not self._initialized:
            return None
        
        try:
            # Ensure audio is float32 and 1D
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            if len(audio.shape) > 1:
                audio = audio.flatten()
            
            # Resample if needed (MERT expects 24kHz, Wav2Vec expects 16kHz)
            # Using 24kHz for MERT music-optimized features
            if sr != 24000:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=24000)
                sr = 24000
            
            # Normalize audio
            if np.abs(audio).max() > 0:
                audio = audio / np.abs(audio).max()
            
            # Process audio
            inputs = self.processor(
                audio, 
                sampling_rate=sr, 
                return_tensors="pt",
                padding=True
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Extract features
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Get hidden states (last layer)
            # Shape: [batch=1, time_steps, hidden_dim=768]
            hidden_states = outputs.last_hidden_state
            
            # Condense over time (temporal average)
            # This gives us a single 768D vector representing the audio segment
            features = hidden_states.mean(dim=1).squeeze().cpu().numpy()
            
            return Wav2VecFeatures(
                features=features,
                timestamp=timestamp,
                duration=len(audio) / sr,
                sample_rate=sr
            )
            
        except Exception as e:
            print(f"⚠️ Wav2Vec encoding error: {e}")
            return None
    
    def encode_batch(self,
                    audio_segments: list,
                    sr: int = 44100) -> list:
        """
        Encode multiple audio segments (more efficient)
        
        Args:
            audio_segments: List of audio arrays
            sr: Sample rate
            
        Returns:
            List of feature vectors (768D each)
        """
        # Initialize model on first call
        if not self._initialized:
            self._initialize_model()
        
        if not self._initialized:
            return []
        
        try:
            features_list = []
            
            for audio in audio_segments:
                result = self.encode(audio, sr)
                if result:
                    features_list.append(result.features)
            
            return features_list
            
        except Exception as e:
            print(f"⚠️ Batch encoding error: {e}")
            return []


class Wav2VecPerceptionModule:
    """
    High-level perception module using Wav2Vec encoding
    
    Replaces ratio + chroma with learned Wav2Vec features
    """
    
    def __init__(self,
                 model_name: str = "facebook/wav2vec2-base",
                 use_gpu: bool = False,
                 enable_dimensionality_reduction: bool = True,
                 reduced_dim: int = 64):
        """
        Initialize perception module
        
        Args:
            model_name: Wav2Vec model to use
            use_gpu: Use GPU acceleration
            enable_dimensionality_reduction: Reduce 768D → reduced_dim via PCA
            reduced_dim: Target dimension (default 64D)
        """
        self.encoder = Wav2VecMusicEncoder(model_name, use_gpu)
        
        # Dimensionality reduction (optional)
        self.enable_reduction = enable_dimensionality_reduction
        self.reduced_dim = reduced_dim
        self.pca = None  # Will be fitted during training
        
        if enable_dimensionality_reduction:
            print(f"🔬 Will reduce Wav2Vec features: 768D → {reduced_dim}D")
    
    def extract_features(self,
                        audio: np.ndarray,
                        sr: int = 44100,
                        timestamp: float = 0.0) -> Optional[np.ndarray]:
        """
        Extract features from audio using Wav2Vec
        
        Args:
            audio: Audio signal
            sr: Sample rate
            timestamp: Timestamp
            
        Returns:
            Feature vector (768D or reduced_dim if PCA fitted)
        """
        result = self.encoder.encode(audio, sr, timestamp)
        
        if result is None:
            return None
        
        features = result.features
        
        # Apply dimensionality reduction if fitted
        if self.enable_reduction and self.pca is not None:
            features = self.pca.transform(features.reshape(1, -1))[0]
        
        return features
    
    def fit_dimensionality_reduction(self, feature_collection: list):
        """
        Fit PCA for dimensionality reduction
        
        Args:
            feature_collection: List of 768D feature vectors
        """
        if not self.enable_reduction:
            return
        
        try:
            from sklearn.decomposition import PCA
            
            print(f"\n🔬 Fitting PCA: {len(feature_collection)} samples")
            print(f"   Reducing 768D → {self.reduced_dim}D")
            
            X = np.array(feature_collection)
            
            self.pca = PCA(n_components=self.reduced_dim)
            self.pca.fit(X)
            
            explained_var = self.pca.explained_variance_ratio_.sum()
            print(f"   ✅ PCA fitted: {explained_var:.1%} variance explained")
            
        except Exception as e:
            print(f"⚠️ PCA fitting error: {e}")
            self.pca = None


# Helper function to check if Wav2Vec is available
def is_wav2vec_available() -> bool:
    """Check if Wav2Vec dependencies are installed"""
    try:
        import transformers
        import torch
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    # Test Wav2Vec encoding
    print("🧪 Testing Wav2Vec Music Encoder")
    print("=" * 60)
    
    # Check availability
    if not is_wav2vec_available():
        print("❌ Wav2Vec not available")
        print("   Install: pip install transformers")
        exit(1)
    
    # Create encoder
    encoder = Wav2VecMusicEncoder(use_gpu=True)
    
    # Test with synthetic audio
    sr = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # Create a C major chord (C4 + E4 + G4)
    freq_C = 261.63
    freq_E = 329.63
    freq_G = 392.00
    
    audio = (
        np.sin(2 * np.pi * freq_C * t) +
        np.sin(2 * np.pi * freq_E * t) +
        np.sin(2 * np.pi * freq_G * t)
    ) / 3.0
    
    print("\n🎵 Encoding C major chord (1 second)...")
    result = encoder.encode(audio, sr, timestamp=0.0)
    
    if result:
        print(f"✅ Encoded successfully!")
        print(f"   Feature dimension: {result.features.shape}")
        print(f"   Feature range: [{result.features.min():.3f}, {result.features.max():.3f}]")
        print(f"   Feature mean: {result.features.mean():.3f}")
        print(f"   Feature std: {result.features.std():.3f}")
    else:
        print("❌ Encoding failed")





























