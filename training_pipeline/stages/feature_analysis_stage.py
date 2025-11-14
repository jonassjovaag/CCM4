"""
Feature Analysis Stage
Performs dual perception analysis (Wav2Vec + frequency ratios).
Part of Phase 2.2: Modular Training Pipeline
"""

from typing import Dict, Any, List
import logging

from .base_stage import PipelineStage, StageResult

logger = logging.getLogger(__name__)


class FeatureAnalysisStage(PipelineStage):
    """
    Stage 2: Feature Analysis

    Responsibilities:
    - Apply Wav2Vec neural encoding (768D features)
    - Extract frequency ratios and consonance
    - Apply K-means clustering to create gesture tokens
    - Extract chroma features
    - Detect chords and harmonic context

    Inputs:
    - audio_events: List of audio events from Stage 1
    - audio_file: Path to audio file

    Outputs:
    - enriched_events: Audio events with features added
    - wav2vec_features: 768D neural features per event
    - gesture_tokens: Discrete tokens (0-63) per event
    - chord_detections: Detected chords
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__("FeatureAnalysis", config)

    def get_required_inputs(self) -> List[str]:
        return ['audio_events', 'audio_file']

    def get_output_keys(self) -> List[str]:
        return ['enriched_events', 'wav2vec_features', 'gesture_tokens', 'chord_detections']

    def validate_inputs(self, context: Dict[str, Any]) -> List[str]:
        errors = []

        if 'audio_events' not in context:
            errors.append("Missing required input: audio_events")
        elif not context['audio_events']:
            errors.append("No audio events to analyze")

        if 'audio_file' not in context:
            errors.append("Missing required input: audio_file")

        return errors

    def execute(self, context: Dict[str, Any]) -> StageResult:
        audio_events = context['audio_events']
        audio_file = context['audio_file']

        self.logger.info(f"Analyzing features for {len(audio_events)} events")

        # Import here to avoid circular dependencies
        from listener.dual_perception import DualPerceptionAnalyzer

        # Create analyzer with config
        analyzer = DualPerceptionAnalyzer(
            enable_wav2vec=self.config.get('enable_wav2vec', True),
            symbolic_vocabulary_size=self.config.get('symbolic_vocabulary_size', 64),
            use_gpu=self.config.get('use_gpu', True)
        )

        # Process events
        enriched_events = []
        wav2vec_features = []
        gesture_tokens = []
        chord_detections = []

        for event in audio_events:
            # Analyze event (adds gesture_token, wav2vec_features, etc.)
            analyzed_event = analyzer.analyze_event(event, audio_file)
            enriched_events.append(analyzed_event)

            # Collect features
            if hasattr(analyzed_event, 'wav2vec_features'):
                wav2vec_features.append(analyzed_event.wav2vec_features)

            if hasattr(analyzed_event, 'gesture_token'):
                gesture_tokens.append(analyzed_event.gesture_token)

            if hasattr(analyzed_event, 'chord'):
                chord_detections.append(analyzed_event.chord)

        self.logger.info(f"Feature analysis complete")

        return StageResult(
            stage_name=self.name,
            success=True,
            duration_seconds=0,
            data={
                'enriched_events': enriched_events,
                'wav2vec_features': wav2vec_features,
                'gesture_tokens': gesture_tokens,
                'chord_detections': chord_detections
            },
            metrics={
                'events_analyzed': len(enriched_events),
                'unique_tokens': len(set(gesture_tokens)) if gesture_tokens else 0,
                'chords_detected': len([c for c in chord_detections if c])
            }
        )
