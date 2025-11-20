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

        # Apply training_events limit if specified
        training_events_limit = self.config.get('training', {}).get('training_events')
        
        # Also check if it's directly in the config root (passed from CLI override)
        if not training_events_limit:
            training_events_limit = self.config.get('training_events')

        if training_events_limit and len(sampled_events) > training_events_limit:
            # Distribute training events evenly across the full track
            # Instead of taking first N events (which only covers beginning),
            # sample every Kth event to get full structural coverage
            step = len(sampled_events) // training_events_limit
            if step < 1: step = 1
            
            original_count = len(sampled_events)
            sampled_events = sampled_events[::step][:training_events_limit]
            
            self.logger.info(f"📉 Downsampling: {original_count} -> {len(sampled_events)} events")
            self.logger.info(f"   Strategy: Taking every {step}th event to preserve structure")
        
        self.logger.info(f"Training oracles on {len(sampled_events)} events")

        # Train AudioOracle
        audio_oracle, oracle_stats = self._train_audio_oracle(sampled_events)

        # Train RhythmOracle if enabled
        rhythm_oracle = None
        rhythmic_analysis = None

        enable_rhythmic_flag = self.config.get('enable_rhythmic', True)
        self.logger.info(f"🔍 DEBUG Oracle Training: enable_rhythmic={enable_rhythmic_flag}, audio_file={audio_file is not None}")
        
        if enable_rhythmic_flag and audio_file:
            self.logger.info("🎵 Starting RhythmOracle training...")
            rhythm_oracle, rhythmic_analysis = self._train_rhythm_oracle(
                sampled_events,
                audio_file
            )
            self.logger.info(f"🎵 RhythmOracle training complete: rhythm_oracle is {'None' if rhythm_oracle is None else 'dict with ' + str(len(rhythm_oracle.get('patterns', []))) + ' patterns'}")
        else:
            self.logger.info(f"⚠️  RhythmOracle training SKIPPED: enable_rhythmic={enable_rhythmic_flag}, has_audio_file={audio_file is not None}")

        self.logger.info("Oracle training complete")

        # Debug logging for what's being returned
        self.logger.info(f"🔍 DEBUG Oracle Training Return: rhythm_oracle type={type(rhythm_oracle)}, is_None={rhythm_oracle is None}")
        if rhythm_oracle and isinstance(rhythm_oracle, dict):
            self.logger.info(f"   rhythm_oracle has {len(rhythm_oracle.get('patterns', []))} patterns")

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
            from dataclasses import asdict

            # Analyze rhythmic features
            analyzer = HeavyRhythmicAnalyzer()
            rhythmic_analysis = analyzer.analyze_rhythmic_structure(audio_file)

            # Train rhythm oracle by adding patterns from analysis
            rhythm_oracle = RhythmOracle()
            
            if hasattr(rhythmic_analysis, 'patterns') and rhythmic_analysis.patterns:
                self.logger.info(f"Adding {len(rhythmic_analysis.patterns)} rhythmic patterns to RhythmOracle...")
                for pattern in rhythmic_analysis.patterns:
                    # Convert dataclass to dict
                    pattern_dict = asdict(pattern) if hasattr(pattern, '__dataclass_fields__') else pattern
                    rhythm_oracle.add_rhythmic_pattern(pattern_dict)
                self.logger.info(f"✅ RhythmOracle trained with {len(rhythmic_analysis.patterns)} patterns")
            else:
                self.logger.warning("No patterns found in rhythmic analysis")

            # Serialize for JSON output
            rhythm_oracle_dict = rhythm_oracle.to_dict()

            return rhythm_oracle_dict, rhythmic_analysis

        except Exception as e:
            self.logger.warning(f"Rhythmic training failed: {e}")
            import traceback
            self.logger.warning(f"Traceback: {traceback.format_exc()}")
            return None, None
