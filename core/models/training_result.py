"""
Training Result Models
Represents complete training output with metadata.
Part of Phase 2.3: Standardized Data Models
"""

from pydantic import BaseModel, Field
from typing import Dict, Optional, List
from datetime import datetime
from .oracle_state import AudioOracleStats, RhythmOracleStats


class TrainingMetadata(BaseModel):
    """
    Metadata about the training run.

    Attributes:
        version: Data format version
        created_at: Training timestamp
        audio_file: Source audio file path
        duration_seconds: Training duration
        parameters: Training parameters used
        git_commit: Git commit hash
        python_version: Python version used
    """
    version: str = Field(default="2.0", description="Data format version")
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Training timestamp"
    )

    audio_file: str = Field(..., description="Source audio file")
    duration_seconds: float = Field(..., description="Training duration", ge=0)

    parameters: Dict = Field(
        default_factory=dict,
        description="Training parameters"
    )

    # System info
    git_commit: Optional[str] = Field(None, description="Git commit hash")
    git_branch: Optional[str] = Field(None, description="Git branch")
    python_version: Optional[str] = Field(None, description="Python version")
    platform: Optional[str] = Field(None, description="System platform")

    model_config = {
        'json_schema_extra': {
            'examples': [{
                'version': '2.0',
                'created_at': '2025-11-13T12:00:00',
                'audio_file': 'curious_child.wav',
                'duration_seconds': 245.3,
                'parameters': {'max_events': 15000},
                'git_commit': 'f81aa4a',
                'python_version': '3.13.0'
            }]
        }
    }


class TrainingResult(BaseModel):
    """
    Complete training result with all data.

    Attributes:
        metadata: Training metadata
        training_successful: Whether training completed successfully
        events_processed: Number of events processed
        audio_oracle_stats: AudioOracle statistics
        rhythm_oracle_stats: RhythmOracle statistics (optional)
        harmonic_patterns: Number of harmonic patterns found
        polyphonic_patterns: Number of polyphonic patterns
        pipeline_metrics: Per-stage metrics from pipeline
        errors: List of errors encountered
        warnings: List of warnings
    """
    metadata: TrainingMetadata = Field(..., description="Training metadata")

    training_successful: bool = Field(..., description="Training success status")
    events_processed: int = Field(..., description="Events processed", ge=0)

    # Oracle statistics
    audio_oracle_stats: AudioOracleStats = Field(
        ...,
        description="AudioOracle training statistics"
    )

    rhythm_oracle_stats: Optional[RhythmOracleStats] = Field(
        None,
        description="RhythmOracle statistics"
    )

    # Pattern analysis
    harmonic_patterns: int = Field(default=0, description="Harmonic patterns", ge=0)
    polyphonic_patterns: int = Field(default=0, description="Polyphonic patterns", ge=0)

    # Pipeline metrics
    pipeline_metrics: Optional[Dict] = Field(
        None,
        description="Per-stage pipeline metrics"
    )

    # Issues
    errors: List[str] = Field(default_factory=list, description="Errors encountered")
    warnings: List[str] = Field(default_factory=list, description="Warnings")

    model_config = {
        'json_schema_extra': {
            'examples': [{
                'metadata': {
                    'version': '2.0',
                    'created_at': '2025-11-13T12:00:00',
                    'audio_file': 'curious_child.wav',
                    'duration_seconds': 245.3
                },
                'training_successful': True,
                'events_processed': 3000,
                'audio_oracle_stats': {
                    'total_states': 3001,
                    'total_patterns': 23868,
                    'sequence_length': 3000,
                    'is_trained': True,
                    'distance_threshold': 1.185,
                    'distance_function': 'euclidean',
                    'feature_dimensions': 15
                },
                'harmonic_patterns': 150,
                'polyphonic_patterns': 45
            }]
        }
    }

    def to_json_dict(self) -> dict:
        """
        Convert to JSON-serializable dictionary.

        Returns:
            Dictionary suitable for JSON serialization
        """
        return self.model_dump(mode='json', exclude_none=True)

    def validate_training(self) -> tuple[bool, List[str]]:
        """
        Validate training result.

        Returns:
            Tuple of (is_valid, issues)
        """
        issues = []

        if not self.training_successful:
            issues.append("Training was not successful")

        if self.events_processed == 0:
            issues.append("No events were processed")

        # Validate oracle
        oracle_valid, oracle_warnings = self.audio_oracle_stats.validate_state_count()
        issues.extend(oracle_warnings)

        if self.errors:
            issues.extend([f"Error: {e}" for e in self.errors])

        is_valid = self.training_successful and oracle_valid

        return is_valid, issues

    @classmethod
    def from_legacy_result(cls, legacy_result: Dict) -> 'TrainingResult':
        """
        Convert legacy training result to Pydantic model.

        Args:
            legacy_result: Legacy result dictionary

        Returns:
            TrainingResult instance
        """
        # Extract metadata
        metadata = TrainingMetadata(
            version=legacy_result.get('version', '2.0'),
            created_at=datetime.fromisoformat(
                legacy_result.get('created_at', datetime.now().isoformat())
            ) if isinstance(legacy_result.get('created_at'), str) else datetime.now(),
            audio_file=legacy_result.get('audio_file', 'unknown'),
            duration_seconds=legacy_result.get('duration_seconds', 0.0),
            parameters=legacy_result.get('parameters', {}),
            git_commit=legacy_result.get('git_commit'),
            python_version=legacy_result.get('python_version')
        )

        # Extract oracle stats
        oracle_stats_dict = legacy_result.get('audio_oracle_stats', {})
        oracle_stats = AudioOracleStats(
            total_states=oracle_stats_dict.get('total_states', 0),
            total_patterns=oracle_stats_dict.get('total_patterns', 0),
            sequence_length=oracle_stats_dict.get('sequence_length', 0),
            is_trained=oracle_stats_dict.get('is_trained', False),
            distance_threshold=oracle_stats_dict.get('distance_threshold', 0.15),
            distance_function=oracle_stats_dict.get('distance_function', 'euclidean'),
            feature_dimensions=oracle_stats_dict.get('feature_dimensions', 15)
        )

        return cls(
            metadata=metadata,
            training_successful=legacy_result.get('training_successful', False),
            events_processed=legacy_result.get('events_processed', 0),
            audio_oracle_stats=oracle_stats,
            rhythm_oracle_stats=None,  # TODO: Parse if present
            harmonic_patterns=legacy_result.get('harmonic_patterns', 0),
            polyphonic_patterns=legacy_result.get('polyphonic_patterns', 0),
            pipeline_metrics=legacy_result.get('pipeline_metrics'),
            errors=legacy_result.get('errors', []),
            warnings=legacy_result.get('warnings', [])
        )
