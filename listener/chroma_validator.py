#!/usr/bin/env python3
"""
Chroma-Based Musical Content Validator

Uses chroma analysis as a validation/noise gate layer for MERT-based matching.
This addresses the user's insight that MERT alone might match on room noise.

Purpose:
    1. Filter out non-musical content (room noise, background sounds)
    2. Validate that detected audio has clear pitch content
    3. Provide a "musical confidence" score for MERT match weighting

Philosophy:
    - MERT (768D) is the PRIMARY feature for pattern matching
    - Chroma (12D) is the VALIDATION layer to ensure we're matching music, not noise
    - Low chroma coherence → probably noise → skip MERT matching
    - High chroma coherence → probably music → trust MERT matching

Based on user insight:
    "Using MERT only... what if it picks up on noise in the room as music?
     That's not really what I'm after."
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class ChromaValidation:
    """Result of chroma-based validation"""

    # Core validation result
    is_musical: bool  # True if content appears musical (not noise)
    musical_confidence: float  # 0-1 confidence that this is music

    # Component scores
    chroma_energy: float  # Sum of chroma vector (0-12 theoretical max)
    chroma_peak: float  # Max value in chroma (0-1)
    pitch_coherence: float  # How focused the pitch content is (0-1)
    num_active_pitches: int  # Number of clear pitch classes

    # Gating recommendation
    should_process_mert: bool  # Should we use MERT result for matching?
    confidence_multiplier: float  # Multiply MERT confidence by this (0-1)

    # Human-readable reason
    reason: str


class ChromaValidator:
    """
    Validates whether audio content is musical (vs noise) using chroma analysis.

    This acts as a gate for MERT matching:
    - High confidence → full MERT matching
    - Low confidence → reduce/skip MERT matching (likely noise)
    """

    def __init__(self,
                 min_chroma_energy: float = 0.5,      # Minimum chroma sum to consider musical
                 min_chroma_peak: float = 0.15,       # Minimum peak chroma value
                 min_active_pitches: int = 1,         # Minimum clear pitch classes
                 coherence_threshold: float = 0.3,    # Pitch coherence threshold
                 gate_threshold: float = 0.3):        # Below this → skip MERT
        """
        Initialize chroma validator.

        Args:
            min_chroma_energy: Minimum total chroma energy to consider musical
            min_chroma_peak: Minimum peak chroma value (suppresses uniform noise)
            min_active_pitches: Minimum number of clear pitch classes
            coherence_threshold: Minimum pitch coherence score
            gate_threshold: Musical confidence below this → don't process MERT
        """
        self.min_chroma_energy = min_chroma_energy
        self.min_chroma_peak = min_chroma_peak
        self.min_active_pitches = min_active_pitches
        self.coherence_threshold = coherence_threshold
        self.gate_threshold = gate_threshold

        # Running statistics for adaptive thresholds
        self._recent_energies = []
        self._max_history = 100

    def validate(self,
                 chroma: np.ndarray,
                 active_pitch_classes: Optional[np.ndarray] = None,
                 audio_rms: float = 0.0) -> ChromaValidation:
        """
        Validate whether chroma indicates musical content.

        Args:
            chroma: 12D chroma vector
            active_pitch_classes: Array of detected pitch class indices (optional)
            audio_rms: RMS level of audio (for silence detection)

        Returns:
            ChromaValidation with validation result and recommendations
        """
        # Handle None/empty inputs
        if chroma is None or len(chroma) != 12:
            return ChromaValidation(
                is_musical=False,
                musical_confidence=0.0,
                chroma_energy=0.0,
                chroma_peak=0.0,
                pitch_coherence=0.0,
                num_active_pitches=0,
                should_process_mert=False,
                confidence_multiplier=0.0,
                reason="Invalid chroma input"
            )

        # Calculate chroma metrics
        chroma_energy = float(np.sum(chroma))
        chroma_peak = float(np.max(chroma))

        # Pitch coherence: how focused is the energy?
        # High coherence = energy concentrated in few pitch classes (musical)
        # Low coherence = energy spread evenly (noise)
        if chroma_energy > 0:
            chroma_norm = chroma / chroma_energy
            # Entropy-based coherence: low entropy = high coherence
            entropy = -np.sum(chroma_norm * np.log2(chroma_norm + 1e-10))
            max_entropy = np.log2(12)  # Uniform distribution
            pitch_coherence = 1.0 - (entropy / max_entropy)
        else:
            pitch_coherence = 0.0

        # Count active pitch classes
        if active_pitch_classes is not None:
            num_active_pitches = len(active_pitch_classes)
        else:
            # Estimate from chroma
            threshold = max(0.15, chroma_peak * 0.3)
            num_active_pitches = int(np.sum(chroma > threshold))

        # Calculate component scores (0-1)
        energy_score = min(1.0, chroma_energy / (self.min_chroma_energy * 3))
        peak_score = min(1.0, chroma_peak / (self.min_chroma_peak * 3))
        coherence_score = pitch_coherence

        # Pitch count score: 1-4 pitches is ideal (single note to chord)
        if num_active_pitches == 0:
            pitch_score = 0.0
        elif num_active_pitches <= 4:
            pitch_score = 1.0
        else:
            # Penalize too many pitches (might be noise or complex clusters)
            pitch_score = max(0.3, 1.0 - (num_active_pitches - 4) * 0.1)

        # Combined musical confidence
        # Weight: energy=25%, peak=25%, coherence=30%, pitch_count=20%
        musical_confidence = (
            0.25 * energy_score +
            0.25 * peak_score +
            0.30 * coherence_score +
            0.20 * pitch_score
        )

        # Update running statistics
        self._recent_energies.append(chroma_energy)
        if len(self._recent_energies) > self._max_history:
            self._recent_energies.pop(0)

        # Determine if musical
        is_musical = (
            chroma_energy >= self.min_chroma_energy and
            chroma_peak >= self.min_chroma_peak and
            num_active_pitches >= self.min_active_pitches and
            pitch_coherence >= self.coherence_threshold * 0.5  # Softer threshold
        )

        # Gating decision
        should_process_mert = musical_confidence >= self.gate_threshold

        # Confidence multiplier for MERT (smooth transition)
        if musical_confidence < self.gate_threshold:
            confidence_multiplier = 0.0
        elif musical_confidence < self.gate_threshold + 0.2:
            # Ramp up from 0 to 1 over 0.2 range
            confidence_multiplier = (musical_confidence - self.gate_threshold) / 0.2
        else:
            confidence_multiplier = 1.0

        # Generate reason
        reasons = []
        if chroma_energy < self.min_chroma_energy:
            reasons.append(f"low energy ({chroma_energy:.2f}<{self.min_chroma_energy})")
        if chroma_peak < self.min_chroma_peak:
            reasons.append(f"low peak ({chroma_peak:.2f}<{self.min_chroma_peak})")
        if num_active_pitches < self.min_active_pitches:
            reasons.append(f"few pitches ({num_active_pitches}<{self.min_active_pitches})")
        if pitch_coherence < self.coherence_threshold:
            reasons.append(f"low coherence ({pitch_coherence:.2f}<{self.coherence_threshold})")

        if is_musical:
            reason = f"Musical content detected (conf={musical_confidence:.2f})"
        elif reasons:
            reason = "Not musical: " + ", ".join(reasons)
        else:
            reason = f"Below confidence threshold ({musical_confidence:.2f}<{self.gate_threshold})"

        return ChromaValidation(
            is_musical=is_musical,
            musical_confidence=musical_confidence,
            chroma_energy=chroma_energy,
            chroma_peak=chroma_peak,
            pitch_coherence=pitch_coherence,
            num_active_pitches=num_active_pitches,
            should_process_mert=should_process_mert,
            confidence_multiplier=confidence_multiplier,
            reason=reason
        )

    def get_statistics(self) -> dict:
        """Get running statistics for debugging/tuning."""
        if not self._recent_energies:
            return {
                'avg_energy': 0.0,
                'min_energy': 0.0,
                'max_energy': 0.0,
                'samples': 0
            }

        return {
            'avg_energy': np.mean(self._recent_energies),
            'min_energy': np.min(self._recent_energies),
            'max_energy': np.max(self._recent_energies),
            'samples': len(self._recent_energies)
        }

    def reset_statistics(self):
        """Reset running statistics."""
        self._recent_energies = []


def demo():
    """Demo the chroma validator with synthetic examples."""
    print("=" * 60)
    print("Chroma Validator Demo")
    print("=" * 60)

    validator = ChromaValidator()

    # Test cases
    test_cases = [
        ("Clear C major chord (C, E, G)",
         np.array([0.9, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 0.0])),

        ("Single note (A)",
         np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.95, 0.0, 0.0])),

        ("Uniform noise (all pitch classes equal)",
         np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])),

        ("Low energy (quiet)",
         np.array([0.05, 0.0, 0.0, 0.0, 0.03, 0.0, 0.0, 0.04, 0.0, 0.0, 0.0, 0.0])),

        ("Silence",
         np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])),

        ("Complex cluster (all pitches high)",
         np.array([0.8, 0.7, 0.9, 0.6, 0.8, 0.7, 0.9, 0.8, 0.7, 0.8, 0.6, 0.7])),
    ]

    for name, chroma in test_cases:
        result = validator.validate(chroma)
        print(f"\n{name}:")
        print(f"  Chroma: [{', '.join(f'{v:.2f}' for v in chroma)}]")
        print(f"  Musical: {result.is_musical}")
        print(f"  Confidence: {result.musical_confidence:.2f}")
        print(f"  Should process MERT: {result.should_process_mert}")
        print(f"  MERT confidence multiplier: {result.confidence_multiplier:.2f}")
        print(f"  Reason: {result.reason}")
        print(f"  Details: energy={result.chroma_energy:.2f}, peak={result.chroma_peak:.2f}, "
              f"coherence={result.pitch_coherence:.2f}, pitches={result.num_active_pitches}")


if __name__ == "__main__":
    demo()
