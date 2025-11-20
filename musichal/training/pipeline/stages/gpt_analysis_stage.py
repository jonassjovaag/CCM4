"""
GPT Analysis Stage
Uses GPT-OSS to generate high-level musical insights.
Part of Phase 2.2: Modular Training Pipeline
"""

from typing import Dict, Any, List
import logging
import sys
from pathlib import Path

# Add scripts/utils to path to import gpt_oss_client
# This is a bit hacky but necessary given the project structure
try:
    project_root = Path(__file__).resolve().parents[4] # musichal/training/pipeline/stages -> root
    utils_path = project_root / 'scripts' / 'utils'
    if str(utils_path) not in sys.path:
        sys.path.append(str(utils_path))
except Exception:
    pass

from .base_stage import PipelineStage, StageResult

logger = logging.getLogger(__name__)


class GPTAnalysisStage(PipelineStage):
    """
    Stage 7: GPT Analysis

    Responsibilities:
    - Generate high-level musical description using GPT-OSS
    - Analyze harmonic and rhythmic patterns
    - Provide semantic understanding of the track

    Inputs:
    - enriched_events: Audio events with features
    - music_theory_insights: Insights from Stage 6 (optional)
    - rhythmic_analysis: Analysis from Stage 4 (optional)

    Outputs:
    - gpt_analysis: GPTOSSAnalysis object
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__("GPTAnalysis", config)

    def get_required_inputs(self) -> List[str]:
        return ['enriched_events']

    def get_output_keys(self) -> List[str]:
        return ['gpt_analysis']

    def validate_inputs(self, context: Dict[str, Any]) -> List[str]:
        errors = []
        if 'enriched_events' not in context:
            errors.append("Missing required input: enriched_events")
        return errors

    def execute(self, context: Dict[str, Any]) -> StageResult:
        enriched_events = context['enriched_events']
        music_theory_insights = context.get('music_theory_insights')
        rhythmic_analysis = context.get('rhythmic_analysis')
        
        # Check if GPT analysis is enabled in config
        if not self.config.get('enabled', True):
            self.logger.info("GPT Analysis disabled in config")
            return StageResult(
                stage_name=self.name,
                success=True,
                duration_seconds=0.0,
                data={'gpt_analysis': None}
            )

        try:
            # Import client here
            try:
                from gpt_oss_client import GPTOSSClient
            except ImportError:
                self.logger.warning("Could not import GPTOSSClient. Skipping GPT analysis.")
                return StageResult(
                    stage_name=self.name,
                    success=True, # Treat as success but with warning
                    duration_seconds=0.0,
                    warnings=["GPTOSSClient not found"],
                    data={'gpt_analysis': None}
                )

            # Initialize client
            client = GPTOSSClient(
                model=self.config.get('model', 'gpt-oss:20b'),
                timeout=self.config.get('timeout', 120),
                auto_start=self.config.get('auto_start', True)
            )
            
            if not client.is_available:
                self.logger.warning("GPT-OSS not available. Skipping analysis.")
                return StageResult(
                    stage_name=self.name,
                    success=True,
                    duration_seconds=0.0,
                    warnings=["GPT-OSS service not available"],
                    data={'gpt_analysis': None}
                )
                
            self.logger.info("Running GPT-OSS analysis...")
            
            # Prepare data for analysis
            # We can pass music theory insights if available
            harmonic_patterns = None
            if music_theory_insights:
                harmonic_patterns = music_theory_insights.chord_progression
                
            # Run analysis
            analysis = client.analyze_musical_events(
                events=enriched_events,
                harmonic_patterns=harmonic_patterns,
                rhythmic_patterns=rhythmic_analysis.get('patterns') if rhythmic_analysis else None
            )
            
            if analysis:
                self.logger.info("GPT Analysis complete")
                self.logger.info(f"Style: {analysis.style_analysis}")
            else:
                self.logger.warning("GPT Analysis returned no result")
            
            return StageResult(
                stage_name=self.name,
                success=True,
                duration_seconds=0.0,
                data={'gpt_analysis': analysis}
            )
            
        except Exception as e:
            self.logger.error(f"GPT Analysis failed: {e}", exc_info=True)
            # Don't fail the whole pipeline for this optional stage
            return StageResult(
                stage_name=self.name,
                success=True, 
                duration_seconds=0.0,
                errors=[str(e)],
                warnings=["GPT Analysis failed but pipeline continues"],
                data={'gpt_analysis': None}
            )
