"""
Training Orchestrator
Coordinates execution of all training pipeline stages.
Part of Phase 2.2: Modular Training Pipeline
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import json
import pickle

from ..stages.base_stage import PipelineStage, StageResult
from ..stages.audio_extraction_stage import AudioExtractionStage
from ..stages.feature_analysis_stage import FeatureAnalysisStage
from ..stages.hierarchical_sampling_stage import HierarchicalSamplingStage
from ..stages.oracle_training_stage import OracleTrainingStage
from ..stages.validation_stage import ValidationStage

logger = logging.getLogger(__name__)


class TrainingOrchestrator:
    """
    Orchestrates the complete training pipeline.

    Features:
    - Stage-by-stage execution
    - Progress tracking
    - Error handling and recovery
    - Checkpointing (save/resume)
    - Performance metrics collection
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize orchestrator.

        Args:
            config: Configuration dictionary (from ConfigManager)
        """
        self.config = config
        self.stages: List[PipelineStage] = []
        self.results: List[StageResult] = []
        self.context: Dict[str, Any] = {}

        # Initialize stages
        self._setup_stages()

    def _setup_stages(self) -> None:
        """Setup pipeline stages based on configuration."""
        # Stage 1: Audio Extraction
        self.stages.append(
            AudioExtractionStage(self.config.get('audio', {}))
        )

        # Stage 2: Feature Analysis
        self.stages.append(
            FeatureAnalysisStage(self.config.get('feature_extraction', {}))
        )

        # Stage 3: Hierarchical Sampling
        if self.config.get('hierarchical_analysis', {}).get('enabled', True):
            self.stages.append(
                HierarchicalSamplingStage(self.config.get('hierarchical_analysis', {}))
            )

        # Stage 4: Oracle Training
        self.stages.append(
            OracleTrainingStage(self.config.get('audio_oracle', {}))
        )

        # Stage 5: Validation
        self.stages.append(
            ValidationStage(self.config.get('validation', {}))
        )

        logger.info(f"Initialized pipeline with {len(self.stages)} stages")

    def run(
        self,
        audio_file: str | Path,
        output_file: Optional[str | Path] = None,
        checkpoint_dir: Optional[str | Path] = None,
        cached_events_file: Optional[str | Path] = None
    ) -> Dict[str, Any]:
        """
        Run complete training pipeline.

        Args:
            audio_file: Path to input audio file
            output_file: Path to save training results (optional)
            checkpoint_dir: Directory to save checkpoints (optional)
            cached_events_file: Path to cached events pickle (skips extraction) (optional)

        Returns:
            Training results dictionary

        Raises:
            RuntimeError: If pipeline execution fails
        """
        audio_file = Path(audio_file)

        logger.info("=" * 60)
        logger.info(f"Starting training pipeline: {audio_file.name}")
        logger.info("=" * 60)

        # Load cached events if provided
        cached_events = None
        if cached_events_file:
            cached_events_file = Path(cached_events_file)
            if not cached_events_file.exists():
                raise FileNotFoundError(f"Cached events file not found: {cached_events_file}")
            
            logger.info(f"Loading cached events from: {cached_events_file.name}")
            with open(cached_events_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            cached_events = cache_data['events']
            logger.info(f"Loaded {len(cached_events)} cached events")
            logger.info(f"Original audio: {cache_data.get('audio_path', 'unknown')}")
            logger.info("Skipping Stage 1 (AudioExtraction)")

        # Initialize context
        self.context = {
            'audio_file': str(audio_file),
            'config': self.config,
            'previous_results': []
        }

        # Execute each stage
        for i, stage in enumerate(self.stages, 1):
            # Skip AudioExtraction if we have cached events
            if cached_events and stage.name == "AudioExtraction":
                logger.info(f"\nStage {i}/{len(self.stages)}: {stage.name} [SKIPPED - using cache]")
                logger.info("-" * 60)
                
                # Create a fake result with cached data
                from ..stages.base_stage import StageResult
                result = StageResult(
                    stage_name="AudioExtraction",
                    success=True,
                    duration_seconds=0.0,
                    data={
                        'audio_events': cached_events,
                        'total_events': len(cached_events),
                        'sample_rate': 44100,  # Default, adjust if needed
                        'duration': cached_events[-1].t if cached_events else 0.0
                    },
                    metrics={'events_extracted': len(cached_events), 'from_cache': True}
                )
                self.results.append(result)
                self.context.update(result.data)
                self.context['previous_results'] = self.results
                continue
            
            logger.info(f"\nStage {i}/{len(self.stages)}: {stage.name}")
            logger.info("-" * 60)

            # Run stage
            result = stage.run(self.context)
            self.results.append(result)

            # Update context with stage outputs
            self.context.update(result.data)
            self.context['previous_results'] = self.results

            # Check if stage failed
            if not result.success:
                logger.error(f"Stage {stage.name} failed!")
                for error in result.errors:
                    logger.error(f"  - {error}")

                raise RuntimeError(f"Pipeline failed at stage: {stage.name}")

            # Log warnings
            for warning in result.warnings:
                logger.warning(f"  - {warning}")

            # Log metrics
            if result.metrics:
                logger.info("Metrics:")
                for key, value in result.metrics.items():
                    logger.info(f"  {key}: {value}")

            # Save checkpoint if requested
            if checkpoint_dir:
                self._save_checkpoint(checkpoint_dir, i, result)

        # Generate final results
        training_results = self.context.get('training_results', {})

        # Save to file if requested
        if output_file:
            self._save_results(output_file, training_results)

        logger.info("\n" + "=" * 60)
        logger.info("Pipeline execution complete!")
        logger.info("=" * 60)

        return training_results

    def _save_checkpoint(
        self,
        checkpoint_dir: Path,
        stage_num: int,
        result: StageResult
    ) -> None:
        """Save checkpoint after each stage."""
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_file = checkpoint_dir / f"stage_{stage_num}_{result.stage_name}.json"

        checkpoint_data = {
            'stage_num': stage_num,
            'stage_name': result.stage_name,
            'success': result.success,
            'duration_seconds': result.duration_seconds,
            'metrics': result.metrics,
            'errors': result.errors,
            'warnings': result.warnings
        }

        try:
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2)
            logger.debug(f"Saved checkpoint: {checkpoint_file}")
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")

    def _save_results(self, output_file: Path, results: Dict[str, Any]) -> None:
        """Save training results to file."""
        output_file = Path(output_file)

        # Use enhanced save if available
        try:
            from core.data_safety import enhanced_save_json
            from core.metadata_manager import wrap_with_metadata

            # Add metadata
            wrapped_results = wrap_with_metadata(
                results,
                training_source=self.context.get('audio_file'),
                parameters=self.config,
                description=f"Training results for {Path(self.context['audio_file']).name}"
            )

            success = enhanced_save_json(
                wrapped_results,
                output_file,
                description="Training pipeline output"
            )

            if success:
                logger.info(f"Saved results to: {output_file}")
            else:
                logger.error(f"Failed to save results to: {output_file}")

        except ImportError:
            # Fallback to basic save
            logger.warning("Enhanced save not available, using basic save")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Saved results to: {output_file}")

    def get_summary(self) -> Dict[str, Any]:
        """
        Get pipeline execution summary.

        Returns:
            Summary dictionary with timing and metrics
        """
        total_duration = sum(r.duration_seconds for r in self.results)
        successful_stages = sum(1 for r in self.results if r.success)

        summary = {
            'total_stages': len(self.stages),
            'successful_stages': successful_stages,
            'failed_stages': len(self.stages) - successful_stages,
            'total_duration_seconds': total_duration,
            'stages': []
        }

        for result in self.results:
            stage_summary = {
                'name': result.stage_name,
                'success': result.success,
                'duration_seconds': result.duration_seconds,
                'metrics': result.metrics,
                'error_count': len(result.errors),
                'warning_count': len(result.warnings)
            }
            summary['stages'].append(stage_summary)

        return summary

    def print_summary(self) -> None:
        """Print execution summary to console."""
        summary = self.get_summary()

        print("\n" + "=" * 60)
        print("PIPELINE EXECUTION SUMMARY")
        print("=" * 60)
        print(f"Total stages:    {summary['total_stages']}")
        print(f"Successful:      {summary['successful_stages']}")
        print(f"Failed:          {summary['failed_stages']}")
        print(f"Total duration:  {summary['total_duration_seconds']:.2f}s")
        print()
        print("Stage breakdown:")
        for stage in summary['stages']:
            status = "✅" if stage['success'] else "❌"
            print(f"  {status} {stage['name']:30s} {stage['duration_seconds']:6.2f}s")

        print("=" * 60)
