"""
Audio Event Models
Represents audio events with features.
Part of Phase 2.3: Standardized Data Models
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
import numpy as np


class AudioEventFeatures(BaseModel):
    """
    Features extracted from an audio event.

    Attributes:
        f0: Fundamental frequency in Hz
        rms_db: RMS energy in dB
        centroid: Spectral centroid
        cents: Pitch in cents (relative to A440)
        ioi: Inter-onset interval in seconds
        consonance: Consonance measure (0-1)
        wav2vec_features: 768D neural encoding (optional)
        gesture_token: Discrete symbolic token (0-63)
        chroma: 12D chroma features
    """
    f0: float = Field(..., description="Fundamental frequency in Hz", ge=0, le=10000)
    rms_db: float = Field(..., description="RMS energy in dB", ge=-120, le=0)
    centroid: Optional[float] = Field(None, description="Spectral centroid", ge=0)
    cents: Optional[float] = Field(None, description="Pitch in cents")
    ioi: Optional[float] = Field(None, description="Inter-onset interval", ge=0)
    consonance: Optional[float] = Field(None, description="Consonance measure", ge=0, le=1)

    # Advanced features
    wav2vec_features: Optional[List[float]] = Field(
        None,
        description="768D Wav2Vec neural encoding",
        min_length=768,
        max_length=768
    )
    gesture_token: Optional[int] = Field(
        None,
        description="Discrete symbolic token",
        ge=0,
        le=63
    )
    chroma: Optional[List[float]] = Field(
        None,
        description="12D chroma features",
        min_length=12,
        max_length=12
    )

    model_config = {
        'arbitrary_types_allowed': True,  # Allow numpy arrays
        'json_schema_extra': {
            'examples': [{
                'f0': 440.0,
                'rms_db': -30.0,
                'centroid': 2000.0,
                'cents': 0.0,
                'ioi': 0.5,
                'consonance': 0.75,
                'gesture_token': 42
            }]
        }
    }

    @field_validator('f0')
    @classmethod
    def validate_f0(cls, v: float) -> float:
        """Validate f0 is in reasonable range."""
        if v == 440.0:
            # This might be a default value, which could indicate missing data
            pass  # Just a warning, don't fail validation
        return v

    @field_validator('rms_db')
    @classmethod
    def validate_rms(cls, v: float) -> float:
        """Validate RMS is in reasonable range."""
        if v == -20.0:
            # This might be a default value
            pass
        return v


class AudioEvent(BaseModel):
    """
    Represents a single audio event with timestamp and features.

    Attributes:
        timestamp: Event timestamp in seconds
        features: Extracted audio features
        detected_frequencies: List of detected frequencies (polyphonic)
        chord: Detected chord name (if any)
        key: Detected key (if any)
        beat_position: Position in beat grid (if rhythmic analysis enabled)
    """
    timestamp: float = Field(..., description="Event timestamp in seconds", ge=0)
    features: AudioEventFeatures = Field(..., description="Audio features")

    # Optional polyphonic/harmonic info
    detected_frequencies: Optional[List[float]] = Field(
        None,
        description="Detected frequencies for polyphonic events"
    )
    chord: Optional[str] = Field(None, description="Detected chord name")
    key: Optional[str] = Field(None, description="Detected key")

    # Optional rhythmic info
    beat_position: Optional[float] = Field(
        None,
        description="Position in beat grid",
        ge=0
    )
    tempo: Optional[float] = Field(None, description="Local tempo in BPM", ge=0, le=300)

    model_config = {
        'json_schema_extra': {
            'examples': [{
                'timestamp': 1.5,
                'features': {
                    'f0': 440.0,
                    'rms_db': -30.0,
                    'centroid': 2000.0,
                    'gesture_token': 42
                },
                'chord': 'Cmaj7',
                'key': 'C major'
            }]
        }
    }

    @field_validator('timestamp')
    @classmethod
    def validate_timestamp(cls, v: float) -> float:
        """Ensure timestamp is reasonable."""
        if v > 3600:  # More than 1 hour
            raise ValueError("Timestamp seems unreasonably large (>1 hour)")
        return v

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return self.model_dump(mode='json', exclude_none=True)

    @classmethod
    def from_legacy_event(cls, legacy_event: any) -> 'AudioEvent':
        """
        Convert legacy event format to Pydantic model.

        Args:
            legacy_event: Legacy event object (dataclass or dict)

        Returns:
            AudioEvent instance
        """
        if hasattr(legacy_event, '__dict__'):
            data = vars(legacy_event)
        else:
            data = legacy_event

        # Extract features
        features = AudioEventFeatures(
            f0=data.get('f0', 440.0),
            rms_db=data.get('rms_db', -20.0),
            centroid=data.get('centroid'),
            cents=data.get('cents'),
            ioi=data.get('ioi'),
            consonance=data.get('consonance'),
            gesture_token=data.get('gesture_token')
        )

        return cls(
            timestamp=data.get('t', data.get('timestamp', 0.0)),
            features=features,
            detected_frequencies=data.get('detected_frequencies'),
            chord=data.get('chord'),
            key=data.get('key'),
            beat_position=data.get('beat_position'),
            tempo=data.get('tempo')
        )
