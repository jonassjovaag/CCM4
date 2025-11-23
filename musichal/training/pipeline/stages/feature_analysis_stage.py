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
        self.logger.info(f"Neural Audio Model (Wav2Vec/MERT): {model_name}")
        self.logger.info(f"Neural Encoder Config: {wav2vec_config}")

        # Check if legacy wav2vec is disabled via config
        enable_wav2vec = self.config.get('enable_wav2vec', True)
        # Also check nested config if present
        if 'wav2vec' in self.config and isinstance(self.config['wav2vec'], dict):
             if self.config['wav2vec'].get('enabled') is False:
                 enable_wav2vec = False
        
        if not enable_wav2vec:
            self.logger.info("⏭️ Neural Audio Encoding (MERT/Wav2Vec) disabled via config")

        analyzer = DualPerceptionModule(
            vocabulary_size=self.config.get('symbolic_vocabulary_size', 64),
            wav2vec_model=model_name,  # Pass model name from config
            use_gpu=self.config.get('use_gpu', True),
            enable_symbolic=True,
            enable_dual_vocabulary=self.config.get('enable_dual_vocabulary', False),
            enable_wav2vec=enable_wav2vec
        )
        
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
            # Dual vocabulary support (only if available)
            event.harmonic_token = getattr(result, 'harmonic_token', None)
            event.percussive_token = getattr(result, 'percussive_token', None)
            event.chord = result.chord_label
            event.consonance = result.consonance

            enriched_events.append(event)

            # Collect features
            wav2vec_features.append(result.wav2vec_features)
            # Note: gesture_token will be None here because quantizer isn't trained yet
            chord_detections.append(result.chord_label)

        self.logger.info("Feature extraction complete. Training gesture quantizer...")

        # Train quantizer(s)
        if wav2vec_features:
            # Determine output base path for saving vocabularies
            output_file = context.get('output_file')
            if output_file:
                from pathlib import Path
                base_path = str(Path(output_file).with_suffix(''))
            else:
                from pathlib import Path
                base_path = str(Path(audio_file).with_suffix(''))

            if analyzer.enable_dual_vocabulary:
                # Train both vocabularies
                analyzer.train_gesture_vocabulary(wav2vec_features, vocabulary_type="harmonic")
                analyzer.train_gesture_vocabulary(wav2vec_features, vocabulary_type="percussive")
                
                # Save vocabularies
                analyzer.save_vocabulary(f"{base_path}_harmonic_vocab.joblib", "harmonic")
                analyzer.save_vocabulary(f"{base_path}_percussive_vocab.joblib", "percussive")
                self.logger.info(f"Saved dual vocabularies to {base_path}_*.joblib")
            else:
                # Train single vocabulary
                analyzer.train_gesture_vocabulary(wav2vec_features, vocabulary_type="single")
                
                # Save vocabulary (using new naming convention)
                analyzer.save_vocabulary(f"{base_path}_gesture_training_quantizer.joblib", "single")
                self.logger.info(f"Saved gesture vocabulary to {base_path}_gesture_training_quantizer.joblib")

            # Re-process events to assign tokens now that quantizer is trained
            self.logger.info("Assigning gesture tokens to events...")
            gesture_tokens = []
            for i, event in enumerate(enriched_events):
                # Get the features we extracted earlier
                features = wav2vec_features[i]
                
                # Get token from analyzer (which now has trained quantizer)
                # We need to manually call the internal method or use a helper
                # Since extract_features does a lot of work, let's just use the quantizer directly
                
                token = None
                if analyzer.enable_dual_vocabulary:
                    # Use harmonic quantizer for primary token
                    if analyzer.harmonic_quantizer:
                        # Ensure float64 and 2D for sklearn
                        f_64 = features.astype(np.float64).reshape(1, -1)
                        token = int(analyzer.harmonic_quantizer.transform(f_64)[0])
                else:
                    if analyzer.quantizer:
                        f_64 = features.astype(np.float64).reshape(1, -1)
                        token = int(analyzer.quantizer.transform(f_64)[0])
                
                if token is not None:
                    event.gesture_token = token
                    gesture_tokens.append(token)
                    
            self.logger.info(f"Assigned {len(gesture_tokens)} tokens")

        self.logger.info("Feature analysis complete")

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
