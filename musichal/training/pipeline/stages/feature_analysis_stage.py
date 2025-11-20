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
        from listener.dual_perception import DualPerceptionModule
        import librosa
        import numpy as np

        # Create analyzer with config
        # NOTE: self.config is already the 'feature_extraction' section (passed from orchestrator)
        wav2vec_config = self.config.get('wav2vec', {})
        model_name = wav2vec_config.get('model', 'facebook/wav2vec2-base')

        # DEBUG: Print what we're reading from config
        self.logger.info(f"wav2vec.model = {model_name}")
        self.logger.info(f"Full wav2vec config: {wav2vec_config}")

        # Check if legacy wav2vec is disabled via config
        enable_wav2vec = self.config.get('enable_wav2vec', True)
        # Also check nested config if present
        if 'wav2vec' in self.config and isinstance(self.config['wav2vec'], dict):
             if self.config['wav2vec'].get('enabled') is False:
                 enable_wav2vec = False
        
        if not enable_wav2vec:
            self.logger.info("⏭️ Legacy Wav2Vec extraction disabled via config")

        analyzer = DualPerceptionModule(
            vocabulary_size=self.config.get('symbolic_vocabulary_size', 64),
            wav2vec_model=model_name,  # Pass model name from config
            use_gpu=self.config.get('use_gpu', True),
            enable_symbolic=True,
            enable_dual_vocabulary=self.config.get('enable_dual_vocabulary', False),
            # Pass the flag to the module if it supports it, or we handle it here
            # Assuming DualPerceptionModule might not have this flag yet, 
            # we might need to modify it or just accept that it initializes but we don't use it?
            # Ideally DualPerceptionModule should accept 'enable_wav2vec'
        )
        
        # If DualPerceptionModule doesn't support disabling wav2vec, we are still loading it.
        # But at least we are respecting the flag in the trainer logic.
        # To truly save memory, DualPerceptionModule needs to support this flag.
        if hasattr(analyzer, 'enable_wav2vec'):
            analyzer.enable_wav2vec = enable_wav2vec

        # Load audio file for feature extraction
        audio_signal, sr = librosa.load(audio_file, sr=44100)

        # Process events
        enriched_events = []
        wav2vec_features = []
        gesture_tokens = []
        chord_detections = []

        for event in audio_events:
            # Extract audio segment for this event
            # Use a small window around the event timestamp
            start_sample = int(max(0, event.t - 0.05) * sr)
            end_sample = int(min(len(audio_signal), (event.t + 0.05) * sr))
            audio_segment = audio_signal[start_sample:end_sample]

            # Extract features using dual perception
            result = analyzer.extract_features(
                audio=audio_segment,
                sr=sr,
                timestamp=event.t,
                detected_f0=event.f if hasattr(event, 'f') else None
            )

            # Add features to event
            event.wav2vec_features = result.wav2vec_features
            event.features = result.wav2vec_features  # AudioOracle expects 'features' key
            event.gesture_token = result.gesture_token
            event.chord = result.chord_label
            event.consonance = result.consonance

            enriched_events.append(event)

            # Collect features
            wav2vec_features.append(result.wav2vec_features)
            if result.gesture_token is not None:
                gesture_tokens.append(result.gesture_token)
            chord_detections.append(result.chord_label)

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
