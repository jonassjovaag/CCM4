"""
Hierarchical Sampling Stage
Performs multi-timescale hierarchical analysis and adaptive sampling.
Part of Phase 2.2: Modular Training Pipeline
"""

from typing import Dict, Any, List
import logging

from .base_stage import PipelineStage, StageResult

logger = logging.getLogger(__name__)


class HierarchicalSamplingStage(PipelineStage):
    """
    Stage 3: Hierarchical Sampling

    Responsibilities:
    - Multi-timescale analysis (0.1s, 0.5s, 2s, 8s)
    - Perceptual significance filtering
    - Adaptive sampling (select musically important events)
    - Temporal smoothing (prevent chord flicker)

    Inputs:
    - enriched_events: Events with features from Stage 2

    Outputs:
    - sampled_events: Subset of most significant events
    - significance_scores: Significance score per event
    - timescale_analysis: Analysis at each timescale
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__("HierarchicalSampling", config)

    def get_required_inputs(self) -> List[str]:
        return ['enriched_events']

    def get_output_keys(self) -> List[str]:
        return ['sampled_events', 'significance_scores', 'timescale_analysis']

    def validate_inputs(self, context: Dict[str, Any]) -> List[str]:
        errors = []

        if 'enriched_events' not in context:
            errors.append("Missing required input: enriched_events")
        elif not context['enriched_events']:
            errors.append("No events to sample")

        return errors

    def execute(self, context: Dict[str, Any]) -> StageResult:
        enriched_events = context['enriched_events']
        max_events = self.config.get('max_events', 15000)

        self.logger.info(f"Hierarchical sampling from {len(enriched_events)} events")

        # Skip if hierarchical disabled
        if not self.config.get('enabled', True):
            self.logger.info("Hierarchical analysis disabled, using all events")
            return StageResult(
                stage_name=self.name,
                success=True,
                duration_seconds=0,
                data={
                    'sampled_events': enriched_events[:max_events],
                    'significance_scores': [1.0] * min(len(enriched_events), max_events),
                    'timescale_analysis': {}
                },
                metrics={'events_sampled': min(len(enriched_events), max_events)}
            )

        # TEMPORARY FIX: SimpleHierarchicalAnalyzer expects audio files, not events
        # For now, just apply simple sampling without hierarchical analysis
        # TODO: Refactor SimpleHierarchicalAnalyzer to work with event lists
        self.logger.warning("Hierarchical analysis not yet integrated with modular pipeline")
        self.logger.warning("Applying simple sampling strategy instead")

        # Simple sampling: just take first N events
        sampled_events = enriched_events[:max_events]

        # Create mock result
        from collections import namedtuple
        MockResult = namedtuple('MockResult', ['sampled_events', 'significance_scores', 'timescale_stats'])
        result = MockResult(
            sampled_events=sampled_events,
            significance_scores=[1.0] * len(sampled_events),
            timescale_stats={}
        )

        # Apply temporal smoothing if enabled
        sampled_events = result.sampled_events
        if self.config.get('temporal_smoothing', True):
            from core.temporal_smoothing import TemporalSmoother
            
            # Convert Event objects to dicts for temporal smoother
            events_as_dicts = []
            for event in sampled_events:
                if hasattr(event, 'to_dict'):
                    events_as_dicts.append(event.to_dict())
                elif isinstance(event, dict):
                    events_as_dicts.append(event)
                else:
                    raise TypeError(f"Unexpected event type: {type(event)}")
            
            smoother = TemporalSmoother(
                window_seconds=self.config.get('smoothing_window', 0.5)
            )
            sampled_events = smoother.smooth_events(events_as_dicts)

        self.logger.info(f"Sampled {len(sampled_events)} significant events")

        return StageResult(
            stage_name=self.name,
            success=True,
            duration_seconds=0,
            data={
                'sampled_events': sampled_events,
                'significance_scores': result.significance_scores if hasattr(result, 'significance_scores') else [],
                'timescale_analysis': result.timescale_stats if hasattr(result, 'timescale_stats') else {}
            },
            metrics={
                'events_sampled': len(sampled_events),
                'sampling_rate': len(sampled_events) / len(enriched_events) if enriched_events else 0
            }
        )
