#!/usr/bin/env python3
"""
Hybrid Audio Detector
Combines instrument classification and target fingerprinting
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import time

from .audio_fingerprint import AudioFingerprintSystem
from .jhs_listener_core import DriftListener

@dataclass
class DetectionResult:
    """Result of hybrid detection"""
    detection_type: str  # "target", "instrument", or "none"
    confidence: float
    instrument: Optional[str] = None
    target_description: Optional[str] = None
    is_target_match: bool = False

class HybridDetector:
    """Hybrid detector combining instrument classification and target fingerprinting"""
    
    def __init__(self, sample_rate: int = 44100, fingerprint_duration: float = 15.0):
        self.sample_rate = sample_rate
        self.fingerprint_duration = fingerprint_duration
        
        # Initialize components
        self.fingerprint_system = AudioFingerprintSystem(sample_rate, fingerprint_duration)
        
        # Create a mock listener for instrument classification
        # We'll use the classification method directly
        self.instrument_classifier = None
        
        # Detection parameters
        self.target_priority = True  # Target detection takes priority
        self.target_threshold = 0.7
        self.instrument_threshold = 0.3
        
        # Statistics
        self.detection_stats = {
            'target_detections': 0,
            'instrument_detections': 0,
            'no_detections': 0,
            'total_detections': 0
        }
        
        print(f"🔀 HybridDetector initialized")
        print(f"   Target priority: {self.target_priority}")
        print(f"   Target threshold: {self.target_threshold}")
        print(f"   Instrument threshold: {self.instrument_threshold}")
    
    def set_instrument_classifier(self, classifier):
        """Set the instrument classifier (DriftListener instance)"""
        self.instrument_classifier = classifier
        print("🎯 Instrument classifier connected")
    
    def start_target_learning(self, description: str = "target") -> bool:
        """Start learning a new target"""
        return self.fingerprint_system.start_learning(description)
    
    def add_learning_sample(self, audio_buffer: np.ndarray) -> bool:
        """Add audio sample for target learning"""
        return self.fingerprint_system.add_audio_sample(audio_buffer)
    
    def get_learning_progress(self) -> float:
        """Get target learning progress"""
        return self.fingerprint_system.get_learning_progress()
    
    def is_learning_target(self) -> bool:
        """Check if currently learning a target"""
        return self.fingerprint_system.is_learning
    
    def is_target_learned(self) -> bool:
        """Check if target is learned"""
        return self.fingerprint_system.is_target_learned()
    
    def get_target_description(self) -> str:
        """Get target description"""
        return self.fingerprint_system.get_target_description()
    
    def detect(self, audio_buffer: np.ndarray, event_data: Dict) -> DetectionResult:
        """Perform hybrid detection on audio buffer"""
        self.detection_stats['total_detections'] += 1
        
        # If learning target, add to learning buffer
        if self.is_learning_target():
            self.add_learning_sample(audio_buffer)
            return DetectionResult(
                detection_type="learning",
                confidence=self.get_learning_progress(),
                target_description=f"Learning: {self.get_target_description()}"
            )
        
        # 1. Check target fingerprint first (if learned and priority enabled)
        if self.target_priority and self.is_target_learned():
            is_target_match, target_confidence = self.fingerprint_system.match_target(audio_buffer)
            
            if is_target_match and target_confidence >= self.target_threshold:
                self.detection_stats['target_detections'] += 1
                return DetectionResult(
                    detection_type="target",
                    confidence=target_confidence,
                    target_description=self.get_target_description(),
                    is_target_match=True
                )
        
        # 2. Fall back to instrument classification
        if self.instrument_classifier:
            try:
                # Use the instrument classifier
                instrument = self.instrument_classifier._classify_instrument(event_data)
                
                if instrument and instrument != "unknown":
                    self.detection_stats['instrument_detections'] += 1
                    return DetectionResult(
                        detection_type="instrument",
                        confidence=0.8,  # High confidence for instrument detection
                        instrument=instrument
                    )
            except Exception as e:
                print(f"⚠️ Instrument classification failed: {e}")
        
        # 3. No detection
        self.detection_stats['no_detections'] += 1
        return DetectionResult(
            detection_type="none",
            confidence=0.0
        )
    
    def get_detection_stats(self) -> Dict:
        """Get detection statistics"""
        total = self.detection_stats['total_detections']
        if total == 0:
            return self.detection_stats.copy()
        
        stats = self.detection_stats.copy()
        stats['target_percentage'] = (stats['target_detections'] / total) * 100
        stats['instrument_percentage'] = (stats['instrument_detections'] / total) * 100
        stats['no_detection_percentage'] = (stats['no_detections'] / total) * 100
        
        return stats
    
    def print_detection_stats(self):
        """Print detection statistics"""
        stats = self.get_detection_stats()
        print(f"🔀 Hybrid Detection Stats:")
        print(f"   Total detections: {stats['total_detections']}")
        print(f"   Target detections: {stats['target_detections']} ({stats.get('target_percentage', 0):.1f}%)")
        print(f"   Instrument detections: {stats['instrument_detections']} ({stats.get('instrument_percentage', 0):.1f}%)")
        print(f"   No detections: {stats['no_detections']} ({stats.get('no_detection_percentage', 0):.1f}%)")
    
    def save_target_fingerprint(self, filepath: str) -> bool:
        """Save target fingerprint to file"""
        return self.fingerprint_system.save_fingerprint(filepath)
    
    def load_target_fingerprint(self, filepath: str) -> bool:
        """Load target fingerprint from file"""
        return self.fingerprint_system.load_fingerprint(filepath)
    
    def set_target_priority(self, priority: bool):
        """Set whether target detection takes priority"""
        self.target_priority = priority
        print(f"🔀 Target priority: {'enabled' if priority else 'disabled'}")
    
    def set_target_threshold(self, threshold: float):
        """Set target detection threshold"""
        self.target_threshold = max(0.0, min(1.0, threshold))
        self.fingerprint_system.match_threshold = self.target_threshold
        print(f"🔀 Target threshold: {self.target_threshold}")
    
    def set_instrument_threshold(self, threshold: float):
        """Set instrument detection threshold"""
        self.instrument_threshold = max(0.0, min(1.0, threshold))
        print(f"🔀 Instrument threshold: {self.instrument_threshold}")
    
    def reset_stats(self):
        """Reset detection statistics"""
        self.detection_stats = {
            'target_detections': 0,
            'instrument_detections': 0,
            'no_detections': 0,
            'total_detections': 0
        }
        print("🔀 Detection stats reset")
    
    def get_status(self) -> Dict:
        """Get current detector status"""
        return {
            'target_learned': self.is_target_learned(),
            'target_description': self.get_target_description(),
            'is_learning': self.is_learning_target(),
            'learning_progress': self.get_learning_progress(),
            'target_priority': self.target_priority,
            'target_threshold': self.target_threshold,
            'instrument_threshold': self.instrument_threshold,
            'stats': self.get_detection_stats()
        }
