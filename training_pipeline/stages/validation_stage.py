"""
Validation Stage
Validates trained models and generates final output.
Part of Phase 2.2: Modular Training Pipeline
Updated in Phase 2.3: Uses Pydantic models
"""

from typing import Dict, Any, List
import logging
from datetime import datetime

from .base_stage import PipelineStage, StageResult
from core.models.oracle_state import AudioOracleStats, DistanceFunction
from core.models.training_result import TrainingResult, TrainingMetadata

logger = logging.getLogger(__name__)


class ValidationStage(PipelineStage):
    """
    Stage 5: Validation

    Responsibilities:
    - Validate trained models
    - Check data quality
    - Generate training summary
    - Prepare output data structure

    Inputs:
    - audio_oracle: Trained AudioOracle from Stage 4
    - oracle_stats: Statistics from training
    - All previous stage results

    Outputs:
    - validation_report: Validation results
    - training_results: Complete training output
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__("Validation", config)

    def get_required_inputs(self) -> List[str]:
        return ['audio_oracle', 'oracle_stats']

    def get_output_keys(self) -> List[str]:
        return ['validation_report', 'training_results']

    def validate_inputs(self, context: Dict[str, Any]) -> List[str]:
        errors = []

        if 'audio_oracle' not in context:
            errors.append("Missing required input: audio_oracle")

        if 'oracle_stats' not in context:
            errors.append("Missing required input: oracle_stats")

        return errors

    def execute(self, context: Dict[str, Any]) -> StageResult:
        audio_oracle = context['audio_oracle']
        oracle_stats = context['oracle_stats']

        self.logger.info("Validating trained models")

        # Validate oracle
        validation_report = self._validate_oracle(audio_oracle, oracle_stats)

        # Generate training results
        training_results = self._generate_training_results(context, validation_report)

        self.logger.info("Validation complete")

        return StageResult(
            stage_name=self.name,
            success=validation_report['is_valid'],
            duration_seconds=0,
            data={
                'validation_report': validation_report,
                'training_results': training_results
            },
            metrics={
                'is_valid': validation_report['is_valid'],
                'warnings_count': len(validation_report.get('warnings', []))
            },
            warnings=validation_report.get('warnings', [])
        )

    def _validate_oracle(self, oracle: Any, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Validate AudioOracle model using Pydantic validation."""
        try:
            # Convert dict to Pydantic model for validation
            oracle_stats = AudioOracleStats(
                total_states=stats.get('total_states', 0),
                total_patterns=stats.get('total_patterns', 0),
                sequence_length=stats.get('sequence_length', 0),
                is_trained=stats.get('is_trained', False),
                distance_threshold=stats.get('distance_threshold', 0.15),
                distance_function=DistanceFunction(stats.get('distance_function', 'euclidean')),
                feature_dimensions=stats.get('feature_dimensions', 15),
                adaptive_threshold=stats.get('adaptive_threshold', True)
            )

            # Use Pydantic validation
            is_valid, warnings = oracle_stats.validate_state_count()

            # Additional checks
            if oracle_stats.total_patterns == 0:
                warnings.append("No patterns learned")

            return {
                'is_valid': is_valid,
                'warnings': warnings,
                'stats': oracle_stats.model_dump(mode='json'),
                'oracle_stats_model': oracle_stats  # Keep model for TrainingResult
            }

        except Exception as e:
            self.logger.error(f"Failed to create AudioOracleStats model: {e}")
            return {
                'is_valid': False,
                'warnings': [f"Validation error: {str(e)}"],
                'stats': stats
            }

    def _generate_training_results(
        self,
        context: Dict[str, Any],
        validation_report: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate complete training results structure using Pydantic models."""
        try:
            # Create metadata
            metadata = TrainingMetadata(
                audio_file=context.get('audio_file', 'unknown'),
                duration_seconds=context.get('audio_duration', 0.0),
                parameters=context.get('training_parameters', {}),
                git_commit=self._get_git_commit(),
                python_version=self._get_python_version()
            )

            # Get AudioOracleStats from validation
            oracle_stats = validation_report.get('oracle_stats_model')
            if not oracle_stats:
                # Fallback: create from dict
                stats_dict = validation_report.get('stats', {})
                oracle_stats = AudioOracleStats(
                    total_states=stats_dict.get('total_states', 0),
                    total_patterns=stats_dict.get('total_patterns', 0),
                    sequence_length=stats_dict.get('sequence_length', 0),
                    is_trained=stats_dict.get('is_trained', False),
                    distance_threshold=stats_dict.get('distance_threshold', 0.15),
                    distance_function=DistanceFunction(stats_dict.get('distance_function', 'euclidean')),
                    feature_dimensions=stats_dict.get('feature_dimensions', 15)
                )

            # Create TrainingResult
            training_result = TrainingResult(
                metadata=metadata,
                training_successful=validation_report['is_valid'],
                events_processed=len(context.get('sampled_events', [])),
                audio_oracle_stats=oracle_stats,
                harmonic_patterns=len([c for c in context.get('chord_detections', []) if c]),
                warnings=validation_report.get('warnings', [])
            )

            # Add stage metrics
            if 'previous_results' in context:
                pipeline_metrics = {}
                for stage_result in context['previous_results']:
                    pipeline_metrics[stage_result.stage_name] = {
                        'duration_seconds': stage_result.duration_seconds,
                        'success': stage_result.success,
                        'metrics': stage_result.metrics
                    }
                training_result.pipeline_metrics = pipeline_metrics

            # Return as dict (validated by Pydantic)
            return training_result.to_json_dict()

        except Exception as e:
            self.logger.error(f"Failed to create TrainingResult model: {e}")
            # Fallback to dict-based result
            return {
                'training_successful': validation_report['is_valid'],
                'audio_oracle_stats': context.get('oracle_stats', {}),
                'events_processed': len(context.get('sampled_events', [])),
                'errors': [f"Failed to create structured result: {str(e)}"]
            }

    def _get_git_commit(self) -> str:
        """Get current git commit hash."""
        try:
            import subprocess
            result = subprocess.run(
                ['git', 'rev-parse', '--short', 'HEAD'],
                capture_output=True,
                text=True,
                timeout=2
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except:
            return None

    def _get_python_version(self) -> str:
        """Get Python version."""
        import sys
        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
