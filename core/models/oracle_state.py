"""
Oracle State Models
Represents AudioOracle and RhythmOracle state.
Part of Phase 2.3: Standardized Data Models
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from enum import Enum


class DistanceFunction(str, Enum):
    """Distance function for pattern matching."""
    EUCLIDEAN = "euclidean"
    COSINE = "cosine"
    MANHATTAN = "manhattan"


class AudioOracleStats(BaseModel):
    """
    Statistics from AudioOracle training.

    Attributes:
        total_states: Number of states in oracle
        total_patterns: Number of patterns learned
        sequence_length: Length of training sequence
        max_pattern_length: Maximum pattern length
        is_trained: Whether oracle is trained
        distance_threshold: Threshold for pattern matching
        distance_function: Distance metric used
        feature_dimensions: Dimensionality of feature vectors
    """
    total_states: int = Field(..., description="Number of oracle states", ge=0)
    total_patterns: int = Field(..., description="Number of patterns learned", ge=0)
    sequence_length: int = Field(..., description="Training sequence length", ge=0)
    max_pattern_length: int = Field(default=50, description="Max pattern length", ge=1)
    is_trained: bool = Field(..., description="Whether oracle is trained")

    # Configuration
    distance_threshold: float = Field(..., description="Pattern matching threshold", ge=0)
    distance_function: DistanceFunction = Field(..., description="Distance metric")
    feature_dimensions: int = Field(..., description="Feature vector dimensions", ge=1)
    adaptive_threshold: bool = Field(default=True, description="Adaptive threshold enabled")

    # Performance metrics
    total_distances_calculated: Optional[int] = Field(
        None,
        description="Total distance calculations",
        ge=0
    )
    average_distance: Optional[float] = Field(
        None,
        description="Average distance between states",
        ge=0
    )
    threshold_adjustments: Optional[int] = Field(
        None,
        description="Number of threshold adjustments",
        ge=0
    )

    # Musical insights
    harmonic_patterns: Optional[int] = Field(None, description="Harmonic patterns found", ge=0)
    polyphonic_patterns: Optional[int] = Field(None, description="Polyphonic patterns", ge=0)
    chord_progressions: Optional[int] = Field(None, description="Chord progressions", ge=0)
    rhythmic_patterns: Optional[int] = Field(None, description="Rhythmic patterns", ge=0)

    model_config = {
        'json_schema_extra': {
            'examples': [{
                'total_states': 3001,
                'total_patterns': 23868,
                'sequence_length': 3000,
                'max_pattern_length': 50,
                'is_trained': True,
                'distance_threshold': 1.185,
                'distance_function': 'euclidean',
                'feature_dimensions': 15,
                'adaptive_threshold': True
            }]
        }
    }

    def validate_state_count(self) -> tuple[bool, List[str]]:
        """
        Validate state count is consistent.

        Returns:
            Tuple of (is_valid, warnings)
        """
        warnings = []

        if self.total_states == 0:
            warnings.append("Oracle has no states")
        elif self.total_states < 100:
            warnings.append(f"Very few states ({self.total_states})")

        if self.sequence_length > 0:
            expected_states = self.sequence_length + 1
            if self.total_states != expected_states:
                warnings.append(
                    f"State count mismatch: {self.total_states} states "
                    f"for {self.sequence_length} events (expected {expected_states})"
                )

        # Valid if we have states and no critical errors
        is_valid = self.total_states > 0
        return is_valid, warnings


class OracleState(BaseModel):
    """
    Represents a complete AudioOracle state.

    This is a simplified representation for serialization.
    The full oracle contains complex graph structures.

    Attributes:
        stats: Oracle statistics
        config: Configuration used
        states: State graph (simplified)
        transitions: State transitions
        suffix_links: Suffix links for pattern matching
    """
    stats: AudioOracleStats = Field(..., description="Oracle statistics")

    # Configuration snapshot
    config: Dict = Field(
        default_factory=dict,
        description="Configuration used for training"
    )

    # Simplified representations (full oracle is too large)
    states: Optional[Dict] = Field(
        None,
        description="State graph (simplified or omitted for size)"
    )
    transitions: Optional[Dict] = Field(
        None,
        description="State transitions"
    )
    suffix_links: Optional[Dict] = Field(
        None,
        description="Suffix links"
    )

    model_config = {
        'json_schema_extra': {
            'examples': [{
                'stats': {
                    'total_states': 3001,
                    'total_patterns': 23868,
                    'sequence_length': 3000,
                    'is_trained': True,
                    'distance_threshold': 1.185,
                    'distance_function': 'euclidean',
                    'feature_dimensions': 15
                },
                'config': {
                    'adaptive_threshold': True,
                    'max_pattern_length': 50
                }
            }]
        }
    }

    def validate(self) -> tuple[bool, List[str]]:
        """
        Validate oracle state.

        Returns:
            Tuple of (is_valid, warnings)
        """
        warnings = self.stats.validate_state_count()
        is_valid = len([w for w in warnings if 'no states' in w.lower()]) == 0

        return is_valid, warnings


class RhythmOracleStats(BaseModel):
    """
    Statistics from RhythmOracle training.

    Attributes:
        total_events: Number of rhythmic events
        tempo: Average tempo in BPM
        beat_grid: Whether beat grid was detected
        syncopation: Average syncopation measure
    """
    total_events: int = Field(..., description="Number of rhythmic events", ge=0)
    tempo: Optional[float] = Field(None, description="Average tempo (BPM)", ge=0, le=300)
    beat_grid: bool = Field(default=False, description="Beat grid detected")
    syncopation: Optional[float] = Field(
        None,
        description="Average syncopation",
        ge=0,
        le=1
    )

    total_states: Optional[int] = Field(None, description="Rhythm oracle states", ge=0)
    sequence_length: Optional[int] = Field(None, description="Sequence length", ge=0)

    model_config = {
        'json_schema_extra': {
            'examples': [{
                'total_events': 500,
                'tempo': 120.0,
                'beat_grid': True,
                'syncopation': 0.35
            }]
        }
    }
