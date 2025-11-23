#!/usr/bin/env python3
"""
CLAP Style Detection Module
============================

Uses CLAP (Contrastive Language-Audio Pretraining) to detect musical style/mood
and automatically select appropriate behavioral modes.

Based on:
- LAION-AI CLAP: https://github.com/LAION-AI/CLAP
- CLAP paper: https://arxiv.org/abs/2211.06687

Architecture:
    Audio → CLAP Encoder → Style Embedding → Behavior Mode Selection

Style Categories:
    - Intimate (ballad, jazz, contemplative) → SHADOW mode
    - Energetic (rock, funk, aggressive) → COUPLE mode
    - Balanced (ambient, classical, meditative) → MIRROR mode
"""

import numpy as np
import torch
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
from enum import Enum


# Import BehaviorMode from behaviors module
try:
    from agent.behaviors import BehaviorMode
except ImportError:
    # Fallback if agent module not available
    class BehaviorMode(Enum):
        SHADOW = "shadow"
        MIRROR = "mirror"
        COUPLE = "couple"
        IMITATE = "imitate"
        CONTRAST = "contrast"
        LEAD = "lead"


@dataclass
class StyleResult:
    """Result from CLAP style detection"""
    style_label: str  # Primary style (e.g., "ballad", "rock", "ambient")
    confidence: float  # Confidence score 0-1
    recommended_mode: BehaviorMode  # Recommended behavioral mode
    style_embedding: np.ndarray  # Full CLAP embedding (512D)
    secondary_styles: Dict[str, float]  # Other detected styles with scores


class CLAPStyleDetector:
    """
    CLAP-based style detection for automatic behavioral mode selection

    Uses audio-text alignment to detect musical style and map to
    appropriate interaction modes.
    """

    # Style text prompts (CLAP learns audio-text alignment)
    STYLE_PROMPTS = {
        # Intimate styles → SHADOW mode (quiet, following, supportive)
        'ballad': "intimate ballad, soft and contemplative music",
        'jazz': "smooth jazz, intimate and conversational",
        'blues': "slow blues, emotional and expressive",

        # Energetic styles → COUPLE mode (independent, loud, sparse)
        'rock': "rock music, energetic and powerful",
        'funk': "funky groove, rhythmic and danceable",
        'metal': "heavy metal, aggressive and intense",
        'punk': "punk rock, fast and energetic",

        # Balanced styles → MIRROR mode (phrase-aware, complementary)
        'ambient': "ambient soundscape, atmospheric and meditative",
        'classical': "classical music, structured and elegant",
        'electronic': "electronic music, textural and evolving",
        'world': "world music, diverse and cultural",
    }
    
    # Role detection prompts for complementary behavior
    # AI uses these to detect what the human is playing and respond complementarily
    ROLE_PROMPTS = {
        'bass_present': "bass instrument playing, low frequency bass line",
        'melody_dense': "melodic lead instrument, prominent melody line",
        'drums_heavy': "heavy drum percussion, rhythmic drums and beats",
    }

    # Style → BehaviorMode mappings
    STYLE_MODE_MAP = {
        'ballad': BehaviorMode.SHADOW,
        'jazz': BehaviorMode.SHADOW,
        'blues': BehaviorMode.SHADOW,
        'rock': BehaviorMode.COUPLE,
        'funk': BehaviorMode.COUPLE,
        'metal': BehaviorMode.COUPLE,
        'punk': BehaviorMode.COUPLE,
        'ambient': BehaviorMode.MIRROR,
        'classical': BehaviorMode.MIRROR,
        'electronic': BehaviorMode.MIRROR,
        'world': BehaviorMode.MIRROR,
    }

    def __init__(self,
                 model_name: str = "laion/clap-htsat-unfused",
                 use_gpu: bool = True,
                 confidence_threshold: float = 0.3):
        """
        Initialize CLAP style detector

        Args:
            model_name: HuggingFace CLAP model name
                       - "laion/clap-htsat-unfused" (~300MB, faster)
                       - "laion/larger_clap_music" (~1GB, more accurate)
            use_gpu: Use GPU if available (MPS/CUDA)
            confidence_threshold: Minimum confidence to return a style
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold

        # Determine device
        if use_gpu:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
                print(f"🎮 Using MPS (Apple Silicon GPU) for CLAP")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
                print(f"🎮 Using CUDA GPU for CLAP")
            else:
                self.device = torch.device("cpu")
                print(f"💻 GPU not available, using CPU for CLAP")
        else:
            self.device = torch.device("cpu")
            print(f"💻 Using CPU for CLAP")

        # Lazy loading - model loaded on first use
        self.model = None
        self.processor = None
        self._initialized = False

        # Cache text embeddings for style prompts (compute once)
        self.style_text_embeddings = None
        self.role_text_embeddings = None  # For role detection
        
        # Role smoothing (60s moving average to prevent jitter)
        self.role_history = []  # List of role_analysis dicts
        self.role_history_max_len = 12  # 60s @ 5s intervals

        print(f"🎨 CLAP Style Detector initialized:")
        print(f"   Model: {model_name}")
        print(f"   Device: {self.device}")
        print(f"   Styles: {len(self.STYLE_PROMPTS)}")
        print(f"   Roles: {len(self.ROLE_PROMPTS)} (bass/melody/drums detection)")

    def _initialize_model(self):
        """Lazy load CLAP model (first call only)"""
        if self._initialized:
            return

        try:
            print(f"🔄 Loading CLAP model: {self.model_name}...")

            # Import CLAP
            try:
                import laion_clap
            except ImportError:
                print("❌ laion-clap not installed!")
                print("   Install: pip install laion-clap")
                return

            # Load model
            # For laion-clap, we need to specify the model variant
            # The model_name format should be a checkpoint path, not HuggingFace ID
            # Use the default checkpoint loading
            self.model = laion_clap.CLAP_Module(enable_fusion=False)

            # Load checkpoint - use default 'music_speech_audioset_epoch_15_esc_89.98.pt'
            # Or download from HuggingFace if model_name is specified
            try:
                self.model.load_ckpt()  # Load default checkpoint
            except:
                # If default fails, try with model_name as path
                self.model.load_ckpt(ckpt=self.model_name)

            self.model.eval()

            # Move to device (CLAP handles this internally)
            # Note: CLAP uses its own device handling

            # Pre-compute text embeddings for all style prompts
            print(f"🔄 Pre-computing style text embeddings...")
            style_texts = list(self.STYLE_PROMPTS.values())

            with torch.no_grad():
                text_embeddings = self.model.get_text_embedding(style_texts)

            # Store as numpy for faster comparison
            # CLAP get_text_embedding() returns numpy by default
            if isinstance(text_embeddings, torch.Tensor):
                self.style_text_embeddings = text_embeddings.cpu().numpy()
            else:
                self.style_text_embeddings = text_embeddings
            
            # Pre-compute role text embeddings for complementary behavior
            print(f"🔄 Pre-computing role text embeddings...")
            role_texts = list(self.ROLE_PROMPTS.values())
            
            with torch.no_grad():
                role_embeddings = self.model.get_text_embedding(role_texts)
            
            if isinstance(role_embeddings, torch.Tensor):
                self.role_text_embeddings = role_embeddings.cpu().numpy()
            else:
                self.role_text_embeddings = role_embeddings

            self._initialized = True
            print(f"✅ CLAP model loaded!")
            print(f"   Text embeddings cached for {len(style_texts)} styles + {len(role_texts)} roles")

        except Exception as e:
            print(f"❌ Failed to load CLAP model: {e}")
            print("   Falling back to manual mode selection")
            self._initialized = False

    def detect_style(self,
                     audio: np.ndarray,
                     sr: int = 44100) -> Optional[StyleResult]:
        """
        Detect musical style from audio using CLAP

        Args:
            audio: Audio signal (mono, float32)
            sr: Sample rate

        Returns:
            StyleResult with detected style and recommended mode
        """
        # Initialize model on first call
        if not self._initialized:
            self._initialize_model()

        if not self._initialized:
            return None

        try:
            # Ensure audio is float32 and 1D
            if not isinstance(audio, np.ndarray):
                print(f"⚠️ CLAP: audio is not ndarray, got {type(audio)}")
                return None
                
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            if len(audio.shape) > 1:
                audio = audio.flatten()
            
            # Ensure we have actual audio data
            if audio.size == 0:
                print(f"⚠️ CLAP: empty audio buffer")
                return None

            # CLAP expects 48kHz audio
            if sr != 48000:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=48000)
                sr = 48000

            # Normalize audio
            if np.abs(audio).max() > 0:
                audio = audio / np.abs(audio).max()
            
            # Ensure still ndarray after operations
            if not isinstance(audio, np.ndarray):
                print(f"⚠️ CLAP: audio became {type(audio)} after preprocessing")
                return None

            # Extract audio embedding (CLAP expects list of arrays)
            with torch.no_grad():
                audio_embedding = self.model.get_audio_embedding_from_data(
                    x=[audio],  # Wrap in list for batch processing
                    use_tensor=False  # Return numpy
                )

            # Ensure it's a numpy array
            if isinstance(audio_embedding, torch.Tensor):
                audio_embedding = audio_embedding.cpu().numpy()

            # Flatten if needed
            if len(audio_embedding.shape) > 1:
                audio_embedding = audio_embedding.flatten()

            # Compute similarities to all style prompts
            # Using cosine similarity (embeddings are normalized)
            similarities = np.dot(
                self.style_text_embeddings,
                audio_embedding
            )

            # Find best match
            best_idx = np.argmax(similarities)
            best_score = similarities[best_idx]

            # Get style labels
            style_labels = list(self.STYLE_PROMPTS.keys())
            best_style = style_labels[best_idx]

            # Check confidence threshold
            if best_score < self.confidence_threshold:
                return None

            # Get recommended behavior mode
            recommended_mode = self.STYLE_MODE_MAP.get(
                best_style,
                BehaviorMode.MIRROR  # Default fallback
            )

            # Get secondary styles (top 3)
            top_k = 3
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            secondary_styles = {
                style_labels[idx]: float(similarities[idx])
                for idx in top_indices[1:]  # Skip best (already reported)
            }

            return StyleResult(
                style_label=best_style,
                confidence=float(best_score),
                recommended_mode=recommended_mode,
                style_embedding=audio_embedding,
                secondary_styles=secondary_styles
            )

        except Exception as e:
            print(f"⚠️ CLAP style detection error: {e}")
            return None
    
    def detect_roles(self,
                     audio: np.ndarray,
                     sr: int = 44100) -> Optional[Dict[str, float]]:
        """
        Detect musical roles (bass/melody/drums presence) for complementary behavior
        
        Args:
            audio: Audio signal (mono, float32)
            sr: Sample rate
        
        Returns:
            Dict with role scores: {'bass_present': float, 'melody_dense': float, 'drums_heavy': float}
            Scores are 0.0-1.0, smoothed over 60s window
        """
        # Initialize model on first call
        if not self._initialized:
            self._initialize_model()
        
        if not self._initialized:
            return None
        
        try:
            # Ensure audio is float32 and 1D
            if not isinstance(audio, np.ndarray):
                print(f"⚠️ CLAP roles: audio is not ndarray, got {type(audio)}")
                return None
                
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            if len(audio.shape) > 1:
                audio = audio.flatten()
            
            # Ensure we have actual audio data
            if audio.size == 0:
                print(f"⚠️ CLAP roles: empty audio buffer")
                return None
            
            # CLAP expects 48kHz audio
            if sr != 48000:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=48000)
                sr = 48000
            
            # Normalize audio
            if np.abs(audio).max() > 0:
                audio = audio / np.abs(audio).max()
            
            # Ensure still ndarray after operations
            if not isinstance(audio, np.ndarray):
                print(f"⚠️ CLAP roles: audio became {type(audio)} after preprocessing")
                return None
            
            # Extract audio embedding (CLAP expects list of arrays)
            with torch.no_grad():
                audio_embedding = self.model.get_audio_embedding_from_data(
                    x=[audio],  # Wrap in list for batch processing
                    use_tensor=False
                )
            
            if isinstance(audio_embedding, torch.Tensor):
                audio_embedding = audio_embedding.cpu().numpy()
            
            if len(audio_embedding.shape) > 1:
                audio_embedding = audio_embedding.flatten()
            
            # Compute similarities to role prompts
            similarities = np.dot(self.role_text_embeddings, audio_embedding)
            
            # Map to role dict
            role_labels = list(self.ROLE_PROMPTS.keys())
            current_roles = {
                role_labels[i]: float(similarities[i])
                for i in range(len(role_labels))
            }
            
            # Add to history for smoothing
            self.role_history.append(current_roles)
            if len(self.role_history) > self.role_history_max_len:
                self.role_history.pop(0)
            
            # Compute 60s moving average (smoothed roles)
            if len(self.role_history) > 0:
                smoothed_roles = {}
                for role in role_labels:
                    values = [r[role] for r in self.role_history]
                    smoothed_roles[role] = sum(values) / len(values)
            else:
                smoothed_roles = current_roles
            
            return smoothed_roles
        
        except Exception as e:
            print(f"⚠️ CLAP role detection error: {e}")
            return None

    def is_available(self) -> bool:
        """Check if CLAP is available and initialized"""
        return self._initialized


# Helper function to check if CLAP is available
def is_clap_available() -> bool:
    """Check if CLAP dependencies are installed"""
    try:
        import laion_clap
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    # Test CLAP style detection
    print("🧪 Testing CLAP Style Detector")
    print("=" * 70)

    # Check availability
    if not is_clap_available():
        print("❌ CLAP not available")
        print("   Install: pip install laion-clap")
        exit(1)

    # Create detector
    detector = CLAPStyleDetector(use_gpu=True)

    # Test with synthetic audio
    sr = 44100
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration))

    # Test 1: Soft ballad-like audio (slow, quiet sine waves)
    print("\n🎵 Test 1: Soft ballad-like audio...")
    audio_ballad = (
        np.sin(2 * np.pi * 261.63 * t) * 0.3 +  # Soft C4
        np.sin(2 * np.pi * 329.63 * t) * 0.2    # Soft E4
    )

    result = detector.detect_style(audio_ballad.astype(np.float32), sr)

    if result:
        print(f"✅ Detected: {result.style_label} (confidence: {result.confidence:.2f})")
        print(f"   Recommended mode: {result.recommended_mode.value}")
        print(f"   Secondary styles: {result.secondary_styles}")
    else:
        print("❌ No style detected")

    # Test 2: Energetic rock-like audio (loud, distorted)
    print("\n🎵 Test 2: Energetic rock-like audio...")
    audio_rock = (
        np.sin(2 * np.pi * 130.81 * t) * 0.8 +  # Loud C3
        np.sin(2 * np.pi * 196.00 * t) * 0.7 +  # Loud G3
        np.random.randn(len(t)) * 0.1           # Noise (distortion)
    )
    audio_rock = np.clip(audio_rock, -1, 1)  # Clip (distortion)

    result = detector.detect_style(audio_rock.astype(np.float32), sr)

    if result:
        print(f"✅ Detected: {result.style_label} (confidence: {result.confidence:.2f})")
        print(f"   Recommended mode: {result.recommended_mode.value}")
        print(f"   Secondary styles: {result.secondary_styles}")
    else:
        print("❌ No style detected")

    print("\n" + "=" * 70)
    print("CLAP Style Detection test complete!")
