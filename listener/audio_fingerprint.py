#!/usr/bin/env python3
"""
Audio Fingerprint System
Creates and matches audio fingerprints for target learning
"""

import numpy as np
import librosa
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque
import time
import json
import os

@dataclass
class AudioFingerprint:
    """Audio fingerprint data structure"""
    spectral_profile: np.ndarray
    harmonic_profile: np.ndarray
    temporal_profile: np.ndarray
    mfcc_profile: np.ndarray
    created_at: float
    duration: float
    sample_rate: int
    description: str = ""

class AudioFingerprintSystem:
    """Audio fingerprinting system for target learning and matching"""
    
    def __init__(self, sample_rate: int = 44100, fingerprint_duration: float = 15.0):
        self.sample_rate = sample_rate
        self.fingerprint_duration = fingerprint_duration
        self.target_fingerprint: Optional[AudioFingerprint] = None
        self.learning_buffer = deque(maxlen=int(sample_rate * fingerprint_duration))
        self.is_learning = False
        self.learning_start_time = 0.0
        
        # Matching parameters
        self.match_threshold = 0.7
        self.spectral_weight = 0.3
        self.harmonic_weight = 0.25
        self.temporal_weight = 0.2
        self.mfcc_weight = 0.25
        
        print(f"🎯 AudioFingerprintSystem initialized (duration: {fingerprint_duration}s)")
    
    def start_learning(self, description: str = "target") -> bool:
        """Start learning a new target fingerprint"""
        if self.is_learning:
            print("⚠️ Already learning a target")
            return False
        
        self.learning_buffer.clear()
        self.is_learning = True
        self.learning_start_time = time.time()
        self.learning_description = description
        
        print(f"🎯 Started learning target: {description}")
        print(f"   Duration: {self.fingerprint_duration}s")
        print(f"   Speak/play now...")
        
        return True
    
    def add_audio_sample(self, audio_buffer: np.ndarray) -> bool:
        """Add audio sample to learning buffer"""
        if not self.is_learning:
            return False
        
        # Add to buffer
        self.learning_buffer.extend(audio_buffer)
        
        # Check if we have enough data
        elapsed = time.time() - self.learning_start_time
        if elapsed >= self.fingerprint_duration:
            return self._complete_learning()
        
        return True
    
    def _complete_learning(self) -> bool:
        """Complete the learning process and create fingerprint"""
        if not self.is_learning or len(self.learning_buffer) == 0:
            print("❌ No audio data to learn from")
            return False
        
        try:
            # Convert buffer to numpy array
            audio_data = np.array(self.learning_buffer, dtype=np.float32)
            
            # Create fingerprint
            fingerprint = self._create_fingerprint(audio_data, self.learning_description)
            
            if fingerprint:
                self.target_fingerprint = fingerprint
                self.is_learning = False
                
                print(f"✅ Target learned successfully!")
                print(f"   Description: {self.learning_description}")
                print(f"   Duration: {fingerprint.duration:.1f}s")
                print(f"   Spectral profile: {fingerprint.spectral_profile.shape}")
                print(f"   Harmonic profile: {fingerprint.harmonic_profile.shape}")
                print(f"   Temporal profile: {fingerprint.temporal_profile.shape}")
                print(f"   MFCC profile: {fingerprint.mfcc_profile.shape}")
                
                return True
            else:
                print("❌ Failed to create fingerprint")
                return False
                
        except Exception as e:
            print(f"❌ Error completing learning: {e}")
            self.is_learning = False
            return False
    
    def _create_fingerprint(self, audio_data: np.ndarray, description: str) -> Optional[AudioFingerprint]:
        """Create audio fingerprint from audio data"""
        try:
            # Extract features
            spectral_profile = self._extract_spectral_profile(audio_data)
            harmonic_profile = self._extract_harmonic_profile(audio_data)
            temporal_profile = self._extract_temporal_profile(audio_data)
            mfcc_profile = self._extract_mfcc_profile(audio_data)
            
            if all(profile is not None for profile in [spectral_profile, harmonic_profile, temporal_profile, mfcc_profile]):
                return AudioFingerprint(
                    spectral_profile=spectral_profile,
                    harmonic_profile=harmonic_profile,
                    temporal_profile=temporal_profile,
                    mfcc_profile=mfcc_profile,
                    created_at=time.time(),
                    duration=len(audio_data) / self.sample_rate,
                    sample_rate=self.sample_rate,
                    description=description
                )
            else:
                return None
                
        except Exception as e:
            print(f"❌ Error creating fingerprint: {e}")
            return None
    
    def _extract_spectral_profile(self, audio_data: np.ndarray) -> Optional[np.ndarray]:
        """Extract spectral profile (spectral centroid, rolloff, bandwidth)"""
        try:
            # Spectral centroid
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=self.sample_rate)[0]
            
            # Spectral rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=self.sample_rate)[0]
            
            # Spectral bandwidth
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=self.sample_rate)[0]
            
            # Combine features
            profile = np.array([
                np.mean(spectral_centroids),
                np.std(spectral_centroids),
                np.mean(spectral_rolloff),
                np.std(spectral_rolloff),
                np.mean(spectral_bandwidth),
                np.std(spectral_bandwidth)
            ])
            
            return profile
            
        except Exception as e:
            print(f"❌ Error extracting spectral profile: {e}")
            return None
    
    def _extract_harmonic_profile(self, audio_data: np.ndarray) -> Optional[np.ndarray]:
        """Extract harmonic profile (harmonic ratio, fundamental frequency)"""
        try:
            # Harmonic-percussive separation
            y_harmonic, y_percussive = librosa.effects.hpss(audio_data)
            
            # Harmonic ratio
            harmonic_ratio = np.sum(y_harmonic**2) / (np.sum(y_harmonic**2) + np.sum(y_percussive**2) + 1e-10)
            
            # Fundamental frequency
            f0 = librosa.yin(y_harmonic, fmin=50, fmax=2000)
            f0_mean = np.mean(f0[f0 > 0]) if np.any(f0 > 0) else 0
            
            # Zero-crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
            zcr_mean = np.mean(zcr)
            
            profile = np.array([harmonic_ratio, f0_mean, zcr_mean])
            return profile
            
        except Exception as e:
            print(f"❌ Error extracting harmonic profile: {e}")
            return None
    
    def _extract_temporal_profile(self, audio_data: np.ndarray) -> Optional[np.ndarray]:
        """Extract temporal profile (attack, decay, rhythm)"""
        try:
            # Onset detection
            onsets = librosa.onset.onset_detect(y=audio_data, sr=self.sample_rate)
            
            # Rhythm features
            if len(onsets) > 1:
                onset_intervals = np.diff(onsets)
                rhythm_regularity = 1.0 / (np.std(onset_intervals) + 1e-10)
                rhythm_density = len(onsets) / (len(audio_data) / self.sample_rate)
            else:
                rhythm_regularity = 0.0
                rhythm_density = 0.0
            
            # RMS energy
            rms = librosa.feature.rms(y=audio_data)[0]
            rms_mean = np.mean(rms)
            rms_std = np.std(rms)
            
            profile = np.array([rhythm_regularity, rhythm_density, rms_mean, rms_std])
            return profile
            
        except Exception as e:
            print(f"❌ Error extracting temporal profile: {e}")
            return None
    
    def _extract_mfcc_profile(self, audio_data: np.ndarray) -> Optional[np.ndarray]:
        """Extract MFCC profile"""
        try:
            # MFCC features
            mfccs = librosa.feature.mfcc(y=audio_data, sr=self.sample_rate, n_mfcc=13)
            
            # Statistical features of MFCCs
            mfcc_mean = np.mean(mfccs, axis=1)
            mfcc_std = np.std(mfccs, axis=1)
            
            # Combine mean and std
            profile = np.concatenate([mfcc_mean, mfcc_std])
            return profile
            
        except Exception as e:
            print(f"❌ Error extracting MFCC profile: {e}")
            return None
    
    def match_target(self, audio_buffer: np.ndarray) -> Tuple[bool, float]:
        """Match audio buffer against target fingerprint"""
        if self.target_fingerprint is None:
            return False, 0.0
        
        try:
            # Create fingerprint for current audio
            current_fingerprint = self._create_fingerprint(audio_buffer, "current")
            if current_fingerprint is None:
                return False, 0.0
            
            # Calculate similarity scores
            spectral_similarity = self._calculate_similarity(
                current_fingerprint.spectral_profile,
                self.target_fingerprint.spectral_profile
            )
            
            harmonic_similarity = self._calculate_similarity(
                current_fingerprint.harmonic_profile,
                self.target_fingerprint.harmonic_profile
            )
            
            temporal_similarity = self._calculate_similarity(
                current_fingerprint.temporal_profile,
                self.target_fingerprint.temporal_profile
            )
            
            mfcc_similarity = self._calculate_similarity(
                current_fingerprint.mfcc_profile,
                self.target_fingerprint.mfcc_profile
            )
            
            # Weighted combination
            total_similarity = (
                self.spectral_weight * spectral_similarity +
                self.harmonic_weight * harmonic_similarity +
                self.temporal_weight * temporal_similarity +
                self.mfcc_weight * mfcc_similarity
            )
            
            # Check if above threshold
            is_match = total_similarity >= self.match_threshold
            
            return is_match, total_similarity
            
        except Exception as e:
            print(f"❌ Error matching target: {e}")
            return False, 0.0
    
    def _calculate_similarity(self, profile1: np.ndarray, profile2: np.ndarray) -> float:
        """Calculate cosine similarity between two profiles"""
        try:
            # Normalize profiles
            norm1 = np.linalg.norm(profile1)
            norm2 = np.linalg.norm(profile2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Cosine similarity
            similarity = np.dot(profile1, profile2) / (norm1 * norm2)
            
            # Clamp to [0, 1]
            return max(0.0, min(1.0, similarity))
            
        except Exception as e:
            print(f"❌ Error calculating similarity: {e}")
            return 0.0
    
    def save_fingerprint(self, filepath: str) -> bool:
        """Save target fingerprint to file"""
        if self.target_fingerprint is None:
            print("❌ No fingerprint to save")
            return False
        
        try:
            fingerprint_data = {
                'spectral_profile': self.target_fingerprint.spectral_profile.tolist(),
                'harmonic_profile': self.target_fingerprint.harmonic_profile.tolist(),
                'temporal_profile': self.target_fingerprint.temporal_profile.tolist(),
                'mfcc_profile': self.target_fingerprint.mfcc_profile.tolist(),
                'created_at': self.target_fingerprint.created_at,
                'duration': self.target_fingerprint.duration,
                'sample_rate': self.target_fingerprint.sample_rate,
                'description': self.target_fingerprint.description
            }
            
            with open(filepath, 'w') as f:
                json.dump(fingerprint_data, f, indent=2)
            
            print(f"✅ Fingerprint saved to: {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving fingerprint: {e}")
            return False
    
    def load_fingerprint(self, filepath: str) -> bool:
        """Load target fingerprint from file"""
        try:
            if not os.path.exists(filepath):
                print(f"❌ Fingerprint file not found: {filepath}")
                return False
            
            with open(filepath, 'r') as f:
                fingerprint_data = json.load(f)
            
            self.target_fingerprint = AudioFingerprint(
                spectral_profile=np.array(fingerprint_data['spectral_profile']),
                harmonic_profile=np.array(fingerprint_data['harmonic_profile']),
                temporal_profile=np.array(fingerprint_data['temporal_profile']),
                mfcc_profile=np.array(fingerprint_data['mfcc_profile']),
                created_at=fingerprint_data['created_at'],
                duration=fingerprint_data['duration'],
                sample_rate=fingerprint_data['sample_rate'],
                description=fingerprint_data['description']
            )
            
            print(f"✅ Fingerprint loaded from: {filepath}")
            print(f"   Description: {self.target_fingerprint.description}")
            print(f"   Duration: {self.target_fingerprint.duration:.1f}s")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading fingerprint: {e}")
            return False
    
    def get_learning_progress(self) -> float:
        """Get learning progress (0.0 to 1.0)"""
        if not self.is_learning:
            return 1.0
        
        elapsed = time.time() - self.learning_start_time
        return min(1.0, elapsed / self.fingerprint_duration)
    
    def is_target_learned(self) -> bool:
        """Check if target is learned"""
        return self.target_fingerprint is not None
    
    def get_target_description(self) -> str:
        """Get target description"""
        if self.target_fingerprint:
            return self.target_fingerprint.description
        return "No target learned"
