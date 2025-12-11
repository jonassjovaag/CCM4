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
            enable_wav2vec=enable_wav2vec,
            extract_all_frames=False  # Training uses single frame per event, not all MERT frames
        )
        
        # Load audio file for feature extraction
        audio_signal, sr = librosa.load(audio_file, sr=44100)

        # Process events
        enriched_events = []
        wav2vec_features = []  # For single vocab mode (mixed audio)
        harmonic_features_list = []  # For dual vocab mode (HPSS harmonic)
        percussive_features_list = []  # For dual vocab mode (HPSS percussive)
        gesture_tokens = []
        chord_detections = []

        enable_dual_vocab = self.config.get('enable_dual_vocabulary', False)

        for event in audio_events:
            # Extract audio segment for this event
            # Use a small window around the event timestamp
            start_sample = int(max(0, event.t - 0.05) * sr)
            end_sample = int(min(len(audio_signal), (event.t + 0.05) * sr))
            audio_segment = audio_signal[start_sample:end_sample]

            if enable_dual_vocab and enable_wav2vec and analyzer.wav2vec_encoder:
                # DUAL VOCABULARY MODE: HPSS-separate and extract features from both sources
                # This ensures harmonic vocab trains on harmonic features only
                audio_harmonic, audio_percussive = librosa.effects.hpss(
                    audio_segment,
                    kernel_size=31,
                    power=2.0,
                    mask=True
                )

                # Extract MERT features from harmonic component
                harmonic_result = analyzer.wav2vec_encoder.encode(
                    audio_harmonic, sr, event.t,
                    return_all_frames=False
                )
                harmonic_features = harmonic_result.features

                # Extract MERT features from percussive component
                percussive_result = analyzer.wav2vec_encoder.encode(
                    audio_percussive, sr, event.t,
                    return_all_frames=False
                )
                percussive_features = percussive_result.features

                # Store both feature sets on event
                event.harmonic_features = harmonic_features
                event.percussive_features = percussive_features
                event.features = harmonic_features  # Primary features = harmonic (for melodic matching)

                # Collect for vocab training
                harmonic_features_list.append(harmonic_features)
                percussive_features_list.append(percussive_features)
                wav2vec_features.append(harmonic_features)  # For backwards compat

                # Also extract chord/consonance using full dual perception
                result = analyzer.extract_features(
                    audio=audio_segment,
                    sr=sr,
                    timestamp=event.t,
                    detected_f0=event.f if hasattr(event, 'f') else None
                )
                event.chord = result.chord_label
                event.consonance = result.consonance
                chord_detections.append(result.chord_label)
            else:
                # SINGLE VOCABULARY MODE: Extract from mixed audio
                result = analyzer.extract_features(
                    audio=audio_segment,
                    sr=sr,
                    timestamp=event.t,
                    detected_f0=event.f if hasattr(event, 'f') else None
                )

                # Add features to event
                event.features = result.wav2vec_features
                event.gesture_token = result.gesture_token
                event.chord = result.chord_label
                event.consonance = result.consonance

                # Collect features
                wav2vec_features.append(result.wav2vec_features)
                chord_detections.append(result.chord_label)

            enriched_events.append(event)

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

            if analyzer.enable_dual_vocabulary and harmonic_features_list and percussive_features_list:
                # Train harmonic vocabulary on HPSS-separated HARMONIC features
                self.logger.info(f"Training harmonic vocab on {len(harmonic_features_list)} HPSS-harmonic features")
                analyzer.train_gesture_vocabulary(harmonic_features_list, vocabulary_type="harmonic")

                # Train percussive vocabulary on HPSS-separated PERCUSSIVE features
                self.logger.info(f"Training percussive vocab on {len(percussive_features_list)} HPSS-percussive features")
                analyzer.train_gesture_vocabulary(percussive_features_list, vocabulary_type="percussive")

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
            harmonic_tokens = []
            percussive_tokens = []

            for i, event in enumerate(enriched_events):
                if analyzer.enable_dual_vocabulary and harmonic_features_list and percussive_features_list:
                    # DUAL VOCAB: Assign BOTH harmonic and percussive tokens
                    # using their respective HPSS-separated features
                    harm_feat = harmonic_features_list[i]
                    perc_feat = percussive_features_list[i]

                    # Harmonic token from harmonic features
                    if analyzer.harmonic_quantizer and analyzer.harmonic_quantizer.is_fitted:
                        f_64 = harm_feat.astype(np.float64).reshape(1, -1)
                        harm_token = int(analyzer.harmonic_quantizer.transform(f_64)[0])
                        event.harmonic_token = harm_token
                        harmonic_tokens.append(harm_token)

                    # Percussive token from percussive features
                    if analyzer.percussive_quantizer and analyzer.percussive_quantizer.is_fitted:
                        f_64 = perc_feat.astype(np.float64).reshape(1, -1)
                        perc_token = int(analyzer.percussive_quantizer.transform(f_64)[0])
                        event.percussive_token = perc_token
                        percussive_tokens.append(perc_token)

                    # gesture_token = harmonic_token for backwards compat
                    event.gesture_token = event.harmonic_token if hasattr(event, 'harmonic_token') else None
                    if event.gesture_token is not None:
                        gesture_tokens.append(event.gesture_token)
                else:
                    # SINGLE VOCAB: Use mixed audio features
                    features = wav2vec_features[i]
                    if analyzer.quantizer and analyzer.quantizer.is_fitted:
                        f_64 = features.astype(np.float64).reshape(1, -1)
                        token = int(analyzer.quantizer.transform(f_64)[0])
                        event.gesture_token = token
                        gesture_tokens.append(token)

            self.logger.info(f"Assigned {len(gesture_tokens)} gesture tokens")
            if harmonic_tokens:
                self.logger.info(f"Assigned {len(harmonic_tokens)} harmonic tokens, {len(percussive_tokens)} percussive tokens")

        self.logger.info("Feature analysis complete")

        # Build output data
        output_data = {
            'enriched_events': enriched_events,
            'features': wav2vec_features,  # Primary features (harmonic in dual mode, mixed in single)
            'gesture_tokens': gesture_tokens,
            'chord_detections': chord_detections
        }

        # Add dual vocab data if available
        if harmonic_features_list:
            output_data['harmonic_features'] = harmonic_features_list
        if percussive_features_list:
            output_data['percussive_features'] = percussive_features_list
        if harmonic_tokens:
            output_data['harmonic_tokens'] = harmonic_tokens
        if percussive_tokens:
            output_data['percussive_tokens'] = percussive_tokens

        # Calculate metrics
        metrics = {
            'events_analyzed': len(enriched_events),
            'unique_gesture_tokens': len(set(gesture_tokens)) if gesture_tokens else 0,
            'chords_detected': len([c for c in chord_detections if c])
        }
        if harmonic_tokens:
            metrics['unique_harmonic_tokens'] = len(set(harmonic_tokens))
            metrics['unique_percussive_tokens'] = len(set(percussive_tokens)) if percussive_tokens else 0

        return StageResult(
            stage_name=self.name,
            success=True,
            duration_seconds=0,
            data=output_data,
            metrics=metrics
        )
