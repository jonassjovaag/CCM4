"""
Oracle Training Stage
Trains AudioOracle and RhythmOracle models.
Part of Phase 2.2: Modular Training Pipeline
"""

from typing import Dict, Any, List
import logging

from .base_stage import PipelineStage, StageResult

logger = logging.getLogger(__name__)


class OracleTrainingStage(PipelineStage):
    """
    Stage 4: Oracle Training

    Responsibilities:
    - Train AudioOracle with sampled events
    - Train RhythmOracle with rhythmic patterns
    - Learn harmonic-rhythmic correlations
    - Calculate statistics

    Inputs:
    - sampled_events: Events from Stage 3
    - audio_file: Original audio file

    Outputs:
    - audio_oracle: Trained AudioOracle model
    - rhythm_oracle: Trained RhythmOracle model (if enabled)
    - oracle_stats: Training statistics
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__("OracleTraining", config)

    def get_required_inputs(self) -> List[str]:
        # Accept either sampled_events (from hierarchical sampling) or enriched_events
        return []  # Flexible - validated in validate_inputs

    def get_output_keys(self) -> List[str]:
        return ['audio_oracle', 'rhythm_oracle', 'oracle_stats', 'rhythmic_analysis']

    def validate_inputs(self, context: Dict[str, Any]) -> List[str]:
        errors = []

        # Accept either sampled_events or enriched_events
        if 'sampled_events' not in context and 'enriched_events' not in context:
            errors.append("Missing required input: sampled_events or enriched_events")

        # Check we have events to train on
        events = context.get('sampled_events') or context.get('enriched_events')
        if events is not None and not events:
            errors.append("No events to train on")

        return errors

    def execute(self, context: Dict[str, Any]) -> StageResult:
        # Use sampled_events if available (from hierarchical sampling)
        # Otherwise use enriched_events (when hierarchical sampling is disabled)
        sampled_events = context.get('sampled_events') or context.get('enriched_events')
        audio_file = context.get('audio_file')

        self.logger.info(f"Training oracles on {len(sampled_events)} events")

        # Train AudioOracle
        audio_oracle, oracle_stats = self._train_audio_oracle(sampled_events)

        # Train RhythmOracle if enabled
        rhythm_oracle = None
        rhythmic_analysis = None

        if self.config.get('enable_rhythmic', True) and audio_file:
            rhythm_oracle, rhythmic_analysis = self._train_rhythm_oracle(
                sampled_events,
                audio_file
            )

        self.logger.info("Oracle training complete")

        return StageResult(
            stage_name=self.name,
            success=True,
            duration_seconds=0,
            data={
                'audio_oracle': audio_oracle,
                'rhythm_oracle': rhythm_oracle,
                'oracle_stats': oracle_stats,
                'rhythmic_analysis': rhythmic_analysis
            },
            metrics={
                'total_states': oracle_stats.get('total_states', 0),
                'total_patterns': oracle_stats.get('total_patterns', 0),
                'sequence_length': oracle_stats.get('sequence_length', 0)
            }
        )

    def _train_audio_oracle(self, events: List[Any]) -> tuple:
        """Train AudioOracle model."""
        from audio_file_learning.hybrid_batch_trainer import HybridBatchTrainer

        trainer = HybridBatchTrainer(cpu_threshold=0)
        success = trainer.train_from_events(events)

        if not success:
            raise RuntimeError("AudioOracle training failed")

        oracle = trainer.audio_oracle

        # Get statistics
        stats = oracle.get_statistics() if hasattr(oracle, 'get_statistics') else {}

        return oracle, stats

    def _train_rhythm_oracle(self, events: List[Any], audio_file: str) -> tuple:
        """Train RhythmOracle model."""
        try:
            from rhythmic_engine.audio_file_learning.heavy_rhythmic_analyzer import HeavyRhythmicAnalyzer
            from rhythmic_engine.memory.rhythm_oracle import RhythmOracle

            # Analyze rhythmic features
            analyzer = HeavyRhythmicAnalyzer()
            rhythmic_analysis = analyzer.analyze_audio_file(audio_file)

            # Train rhythm oracle
            rhythm_oracle = RhythmOracle()
            rhythm_oracle.train(rhythmic_analysis)

            return rhythm_oracle, rhythmic_analysis

        except Exception as e:
            self.logger.warning(f"Rhythmic training failed: {e}")
            return None, None
