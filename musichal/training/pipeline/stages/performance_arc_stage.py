"""
Performance Arc Analysis Stage
Analyzes musical structure from training audio to create performance arcs
"""

import logging
from pathlib import Path
from typing import Dict, Any

from .base_stage import PipelineStage, StageResult
from scripts.analysis.performance_arc_analyzer import PerformanceArcAnalyzer

logger = logging.getLogger(__name__)


class PerformanceArcStage(PipelineStage):
    """
    Pipeline stage for analyzing performance arc from training audio.
    
    Extracts:
    - Musical phases (intro, development, climax, resolution)
    - Engagement curve over time
    - Instrument evolution
    - Silence patterns
    - Theme development
    - Dynamic evolution
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Performance Arc stage.
        
        Args:
            config: Configuration dictionary with optional keys:
                - analysis.enabled: bool (default True)
                - analysis.sample_rate: int (default 44100)
                - analysis.phase_duration_threshold: float (default 30.0)
                - analysis.save_to_ai_learning_data: bool (default True)
        """
        super().__init__('PerformanceArcAnalysis', config)
        
        # Get analysis config (nested under 'analysis' key)
        analysis_config = config.get('analysis', {})
        
        self.enabled = analysis_config.get('enabled', True)
        self.sample_rate = analysis_config.get('sample_rate', 44100)
        self.phase_duration_threshold = analysis_config.get('phase_duration_threshold', 30.0)
        self.save_to_ai_learning_data = analysis_config.get('save_to_ai_learning_data', True)
        
    def validate(self, context: Dict[str, Any]) -> bool:
        """Validate that audio file exists."""
        if not self.enabled:
            logger.info("Performance Arc analysis disabled")
            return False
            
        if 'audio_file' not in context:
            logger.error("No audio_file in context")
            return False
            
        audio_path = Path(context['audio_file'])
        if not audio_path.exists():
            logger.error(f"Audio file not found: {audio_path}")
            return False
            
        return True
        
    def execute(self, context: Dict[str, Any]) -> StageResult:
        """
        Analyze performance arc from audio file.
        
        Args:
            context: Must contain 'audio_file' path
            
        Returns:
            StageResult with performance_arc data
        """
        try:
            audio_path = Path(context['audio_file'])
            logger.info(f"Analyzing performance arc: {audio_path.name}")
            
            # Create analyzer
            analyzer = PerformanceArcAnalyzer(sample_rate=self.sample_rate)
            analyzer.phase_duration_threshold = self.phase_duration_threshold
            
            # Analyze audio
            arc = analyzer.analyze_audio_file(str(audio_path))
            
            # Convert to dict for serialization
            arc_dict = arc.to_dict()
            
            # Optionally save to ai_learning_data/
            if self.save_to_ai_learning_data:
                self._save_to_ai_learning_data(arc_dict, audio_path)
            
            logger.info(f"Performance arc analysis complete: {len(arc.phases)} phases, {arc.total_duration:.1f}s duration")
            
            return StageResult(
                stage_name=self.name,
                success=True,
                duration_seconds=0,  # Will be set by run()
                data={'performance_arc': arc_dict},
                metrics={
                    'total_duration': arc.total_duration,
                    'num_phases': len(arc.phases),
                    'num_silence_patterns': len(arc.silence_patterns),
                    'engagement_curve_points': len(arc.overall_engagement_curve)
                }
            )
            
        except Exception as e:
            logger.error(f"Performance arc analysis failed: {e}", exc_info=True)
            return StageResult(
                stage_name=self.name,
                success=False,
                duration_seconds=0,
                errors=[str(e)]
            )
        
    def _save_to_ai_learning_data(self, arc_dict: Dict[str, Any], audio_path: Path):
        """
        Save performance arc to ai_learning_data/ directory.
        
        Args:
            arc_dict: Performance arc as dictionary
            audio_path: Path to source audio file
        """
        import json
        import tempfile
        import shutil
        
        try:
            # Create ai_learning_data directory if it doesn't exist
            ai_data_dir = Path('ai_learning_data')
            ai_data_dir.mkdir(exist_ok=True)
            
            # Create filename based on audio file
            output_file = ai_data_dir / f"{audio_path.stem}_performance_arc.json"
            
            # Write to temporary file first (atomic write to prevent corruption)
            with tempfile.NamedTemporaryFile(
                mode='w', 
                encoding='utf-8', 
                delete=False, 
                suffix='.json',
                dir=ai_data_dir
            ) as tmp_file:
                json.dump(arc_dict, tmp_file, indent=2, ensure_ascii=False)
                tmp_file.flush()  # Ensure all data is written
                import os
                os.fsync(tmp_file.fileno())  # Force write to disk
                temp_path = tmp_file.name
            
            # Move temp file to final location (atomic on most systems)
            shutil.move(temp_path, output_file)
                
            logger.info(f"Saved performance arc to: {output_file}")
            
        except Exception as e:
            logger.warning(f"Failed to save performance arc to ai_learning_data: {e}")
            # Clean up temp file if it exists
            if 'temp_path' in locals() and Path(temp_path).exists():
                try:
                    Path(temp_path).unlink()
                except:
                    pass
