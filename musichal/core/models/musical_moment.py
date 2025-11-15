"""
Musical Moment Model
Represents a moment in the memory buffer.
Part of Phase 2.3: Standardized Data Models
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from .audio_event import AudioEvent


class MusicalMoment(BaseModel):
    """
    Represents a musical moment in the memory buffer.

    A musical moment combines:
    - Audio event data
    - Normalized features for clustering
    - Cluster assignment
    - Contextual information

    Attributes:
        timestamp: Moment timestamp in seconds
        event: The audio event
        normalized_features: Normalized feature vector [centroid_z, rms_norm, cents_norm, ioi_norm]
        cluster_id: Assigned cluster ID (optional)
        significance: Perceptual significance score (0-1)
    """
    timestamp: float = Field(..., description="Moment timestamp", ge=0)
    event: AudioEvent = Field(..., description="Audio event data")

    normalized_features: List[float] = Field(
        ...,
        description="Normalized features for clustering",
        min_length=4,
        max_length=4
    )

    cluster_id: Optional[int] = Field(
        None,
        description="Assigned cluster ID",
        ge=0
    )

    significance: Optional[float] = Field(
        None,
        description="Perceptual significance score",
        ge=0,
        le=1
    )

    model_config = {
        'json_schema_extra': {
            'examples': [{
                'timestamp': 1.5,
                'event': {
                    'timestamp': 1.5,
                    'features': {
                        'f0': 440.0,
                        'rms_db': -30.0
                    }
                },
                'normalized_features': [0.5, -0.2, 0.1, 0.3],
                'cluster_id': 12,
                'significance': 0.85
            }]
        }
    }

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return self.model_dump(mode='json', exclude_none=True)

    @classmethod
    def from_legacy_moment(cls, legacy_moment: any) -> 'MusicalMoment':
        """
        Convert legacy MusicalMoment to Pydantic model.

        Args:
            legacy_moment: Legacy moment (dataclass or dict)

        Returns:
            MusicalMoment instance
        """
        if hasattr(legacy_moment, '__dict__'):
            data = vars(legacy_moment)
        else:
            data = legacy_moment

        # Convert event
        event = AudioEvent.from_legacy_event(data.get('event_data', {}))

        # Get features (might be numpy array)
        features = data.get('features', [0.0, 0.0, 0.0, 0.0])
        if hasattr(features, 'tolist'):
            features = features.tolist()

        return cls(
            timestamp=data.get('timestamp', 0.0),
            event=event,
            normalized_features=features,
            cluster_id=data.get('cluster_id'),
            significance=data.get('significance')
        )
