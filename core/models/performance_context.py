"""
Performance Context Models
Represents live performance state and context.
Part of Phase 2.3: Standardized Data Models
"""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class BehaviorMode(str, Enum):
    """AI agent behavior modes."""
    IMITATE = "imitate"
    CONTRAST = "contrast"
    LEAD = "lead"
    AUTONOMOUS = "autonomous"


class PerformanceContext(BaseModel):
    """
    Context for live performance decision-making.

    Represents the current state needed by the AI agent
    to make musically intelligent decisions.

    Attributes:
        current_time: Current timestamp
        behavior_mode: Current behavior mode
        is_musician_playing: Whether musician is currently playing
        time_since_last_event: Seconds since last musical event
        current_density: Current note density (notes/second)
        performance_progress: Progress through performance (0-1)
        arc_intensity: Current arc intensity (0-1)
        recent_pitches: Recent pitches played
        recent_chords: Recent chord progression
    """
    current_time: float = Field(..., description="Current timestamp", ge=0)

    behavior_mode: BehaviorMode = Field(
        ...,
        description="Current AI behavior mode"
    )

    is_musician_playing: bool = Field(
        ...,
        description="Whether musician is currently playing"
    )

    time_since_last_event: float = Field(
        ...,
        description="Seconds since last musical event",
        ge=0
    )

    current_density: float = Field(
        ...,
        description="Current note density (notes/second)",
        ge=0,
        le=10
    )

    # Performance arc
    performance_progress: Optional[float] = Field(
        None,
        description="Progress through performance (0-1)",
        ge=0,
        le=1
    )

    arc_intensity: Optional[float] = Field(
        None,
        description="Current arc intensity (0-1)",
        ge=0,
        le=1
    )

    # Musical context
    recent_pitches: Optional[list[float]] = Field(
        None,
        description="Recent pitches (MIDI note numbers)"
    )

    recent_chords: Optional[list[str]] = Field(
        None,
        description="Recent chord progression"
    )

    current_key: Optional[str] = Field(
        None,
        description="Current detected key"
    )

    current_tempo: Optional[float] = Field(
        None,
        description="Current tempo (BPM)",
        ge=0,
        le=300
    )

    model_config = {
        'json_schema_extra': {
            'examples': [{
                'current_time': 45.5,
                'behavior_mode': 'imitate',
                'is_musician_playing': True,
                'time_since_last_event': 0.5,
                'current_density': 2.5,
                'performance_progress': 0.35,
                'arc_intensity': 0.7,
                'recent_chords': ['Cmaj7', 'Dm7', 'G7'],
                'current_key': 'C major',
                'current_tempo': 120.0
            }]
        }
    }

    def should_play_autonomous(self, silence_threshold: float = 2.0) -> bool:
        """
        Determine if AI should play autonomously.

        Args:
            silence_threshold: Seconds of silence before autonomous mode

        Returns:
            True if should play autonomously
        """
        return (
            not self.is_musician_playing
            and self.time_since_last_event > silence_threshold
        )

    def get_intensity_adjusted_density(self) -> float:
        """
        Get density adjusted for performance arc intensity.

        Returns:
            Adjusted density value
        """
        if self.arc_intensity is not None:
            return self.current_density * self.arc_intensity
        return self.current_density


class DecisionContext(BaseModel):
    """
    Context for a single AI decision.

    Attributes:
        performance_context: Current performance state
        matched_pattern: Pattern matched from oracle (if any)
        similarity_score: Similarity to matched pattern (0-1)
        oracle_suggestions: Suggested continuations from oracle
    """
    performance_context: PerformanceContext = Field(
        ...,
        description="Current performance state"
    )

    matched_pattern: Optional[dict] = Field(
        None,
        description="Matched pattern from oracle"
    )

    similarity_score: Optional[float] = Field(
        None,
        description="Similarity to matched pattern",
        ge=0,
        le=1
    )

    oracle_suggestions: Optional[list] = Field(
        None,
        description="Suggested continuations from oracle"
    )

    model_config = {
        'json_schema_extra': {
            'examples': [{
                'performance_context': {
                    'current_time': 45.5,
                    'behavior_mode': 'imitate',
                    'is_musician_playing': True,
                    'time_since_last_event': 0.5,
                    'current_density': 2.5
                },
                'similarity_score': 0.85
            }]
        }
    }
