"""
Music Theory Stage
Applies deep music theory analysis using the MusicTheoryTransformer.
Part of Phase 2.2: Modular Training Pipeline
"""

from typing import Dict, Any, List
import logging
import torch
import numpy as np

from .base_stage import PipelineStage, StageResult

logger = logging.getLogger(__name__)


class MusicTheoryStage(PipelineStage):
    """
    Stage 6: Music Theory Analysis

    Responsibilities:
    - Analyze harmonic progressions using MusicTheoryTransformer
    - Extract scale and key information
    - Analyze musical form
    - Generate high-level musical insights

    Inputs:
    - audio_oracle: Trained AudioOracle from Stage 5
    - enriched_events: Audio events with features

    Outputs:
    - music_theory_insights: MusicalInsights object
    - transformer_stats: Analysis statistics
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__("MusicTheoryAnalysis", config)

    def get_required_inputs(self) -> List[str]:
        return ['audio_oracle']

    def get_output_keys(self) -> List[str]:
        return ['music_theory_insights', 'transformer_stats']

    def validate_inputs(self, context: Dict[str, Any]) -> List[str]:
        errors = []
        if 'audio_oracle' not in context:
            errors.append("Missing required input: audio_oracle")
        return errors

    def execute(self, context: Dict[str, Any]) -> StageResult:
        audio_oracle = context['audio_oracle']
        
        # Extract features from AudioOracle
        # AudioOracle stores frames in self.audio_frames (index 1 to N)
        # We need to collect features from all frames
        
        # Check if we have frames
        if not hasattr(audio_oracle, 'audio_frames') or not audio_oracle.audio_frames:
            return StageResult(
                stage_name=self.name,
                success=False,
                duration_seconds=0.0,
                errors=["AudioOracle has no frames to analyze"]
            )
            
        # Collect features sorted by frame_id to ensure correct sequence
        # audio_frames is a dict: frame_id -> AudioFrame
        try:
            sorted_frames = sorted(audio_oracle.audio_frames.values(), key=lambda f: f.frame_id)
            features_list = [f.features for f in sorted_frames]
        except Exception as e:
             return StageResult(
                stage_name=self.name,
                success=False,
                duration_seconds=0.0,
                errors=[f"Failed to extract features from AudioOracle: {e}"]
            )
        
        if not features_list:
            return StageResult(
                stage_name=self.name,
                success=False,
                duration_seconds=0.0,
                errors=["No features found in AudioOracle frames"]
            )
            
        # Convert to tensor
        try:
            features_array = np.array(features_list, dtype=np.float32)
            features_tensor = torch.from_numpy(features_array)
            
            # Check dimensions
            if len(features_tensor.shape) != 2:
                 return StageResult(
                    stage_name=self.name,
                    success=False,
                    duration_seconds=0.0,
                    errors=[f"Invalid feature shape: {features_tensor.shape}. Expected (seq_len, feature_dim)"]
                )

            feature_dim = features_tensor.shape[1]
            self.logger.info(f"Analyzing {len(features_list)} events with {feature_dim} dimensions")
            
            # Initialize Transformer
            # Import here to avoid circular dependencies if not needed elsewhere
            from hybrid_training.music_theory_transformer import MusicTheoryTransformer
            
            transformer = MusicTheoryTransformer(
                feature_dim=feature_dim,
                hidden_dim=self.config.get('hidden_dim', 128),
                num_heads=self.config.get('num_heads', 4),
                num_layers=self.config.get('num_layers', 3)
            )
            
            # Run analysis
            self.logger.info("Running Music Theory Transformer analysis...")
            
            # Check if MPS is available and configured
            use_gpu = self.config.get('use_gpu', True)
            if use_gpu and torch.backends.mps.is_available():
                device = torch.device("mps")
                self.logger.info("Using MPS acceleration for Music Theory Transformer")
            elif use_gpu and torch.cuda.is_available():
                device = torch.device("cuda")
                self.logger.info("Using CUDA acceleration for Music Theory Transformer")
            else:
                device = torch.device("cpu")
                self.logger.info("Using CPU for Music Theory Transformer")
                
            transformer = transformer.to(device)
            features_tensor = features_tensor.to(device)
            
            insights = transformer.analyze_musical_features(features_tensor)
            
            # Log some results
            self.logger.info(f"Detected Key: {insights.key_signature}")
            self.logger.info(f"Confidence: {insights.confidence_scores.get('scale', 0.0):.2f}")
            
            return StageResult(
                stage_name=self.name,
                success=True,
                duration_seconds=0.0, # Will be filled by wrapper
                data={
                    'music_theory_insights': insights,
                    'transformer_stats': insights.confidence_scores
                }
            )
            
        except Exception as e:
            self.logger.error(f"Music Theory Analysis failed: {e}", exc_info=True)
            return StageResult(
                stage_name=self.name,
                success=False,
                duration_seconds=0.0,
                errors=[str(e)]
            )
