"""
Audio Extraction Stage
Extracts raw audio features from audio file.
Part of Phase 2.2: Modular Training Pipeline
"""

from pathlib import Path
from typing import Dict, Any, List
import logging

from .base_stage import PipelineStage, StageResult

logger = logging.getLogger(__name__)


class AudioExtractionStage(PipelineStage):
    """
    Stage 1: Audio Extraction

    Responsibilities:
    - Load audio file
    - Extract basic audio features (f0, RMS, spectral centroid, etc.)
    - Perform onset detection
    - Create audio event timeline

    Inputs:
    - audio_file: Path to audio file

    Outputs:
    - audio_events: List of audio events with timestamps and features
    - sample_rate: Audio sample rate
    - duration: Audio duration in seconds
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__("AudioExtraction", config)

    def get_required_inputs(self) -> List[str]:
        return ['audio_file']

    def get_output_keys(self) -> List[str]:
        return ['audio_events', 'sample_rate', 'duration', 'total_events']

    def validate_inputs(self, context: Dict[str, Any]) -> List[str]:
        errors = []

        if 'audio_file' not in context:
            errors.append("Missing required input: audio_file")
        else:
            audio_file = Path(context['audio_file'])
            if not audio_file.exists():
                errors.append(f"Audio file not found: {audio_file}")
            if audio_file.suffix.lower() not in ['.wav', '.mp3', '.aiff', '.flac']:
                errors.append(f"Unsupported audio format: {audio_file.suffix}")

        return errors

    def execute(self, context: Dict[str, Any]) -> StageResult:
        audio_file = Path(context['audio_file'])

        self.logger.info(f"Processing audio file: {audio_file.name}")

        # Import here to avoid circular dependencies
        from audio_file_learning.polyphonic_processor import PolyphonicAudioProcessor

        # Create processor
        processor = PolyphonicAudioProcessor()

        # Process audio file (returns tuple of events and file_info)
        audio_events, file_info = processor.process_audio_file(str(audio_file))

        self.logger.info(f"Extracted {len(audio_events)} audio events")

        # Calculate duration
        if audio_events:
            duration = audio_events[-1].t - audio_events[0].t
        else:
            duration = 0.0

        return StageResult(
            stage_name=self.name,
            success=True,
            duration_seconds=0,  # Will be set by run()
            data={
                'audio_events': audio_events,
                'sample_rate': file_info.sample_rate if file_info else 44100,
                'duration': duration,
                'total_events': len(audio_events),
                'audio_file': str(audio_file)
            },
            metrics={
                'events_extracted': len(audio_events),
                'duration_seconds': duration,
                'events_per_second': len(audio_events) / duration if duration > 0 else 0
            }
        )
