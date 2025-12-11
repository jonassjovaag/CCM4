"""
Core Infrastructure Module
Contains data safety, models, configuration, and metadata management.
"""

# Legacy core components (sound generation)
from .voice import Voice, ShortTone, DroneTone
from .engine import FrequencyEngine
from .manager import HarmonicManager

# Phase 1: Data Safety (expose key components)
from .data_safety.atomic_file_writer import AtomicFileWriter
from .data_safety.backup_manager import BackupManager
from .data_safety.data_validator import DataValidator

# Phase 2.1: Configuration
from .config_manager import ConfigManager

# Phase 2.3: Pydantic Models (expose commonly used models)
from .models.audio_event import AudioEvent, AudioEventFeatures
from .models.oracle_state import AudioOracleStats, OracleState
from .models.training_result import TrainingResult, TrainingMetadata
from .models.performance_context import PerformanceContext, BehaviorMode

__all__ = [
    # Legacy sound generation
    'Voice',
    'ShortTone',
    'DroneTone',
    'FrequencyEngine',
    'HarmonicManager',

    # Data safety (Phase 1)
    'AtomicFileWriter',
    'BackupManager',
    'DataValidator',

    # Configuration (Phase 2.1)
    'ConfigManager',

    # Pydantic Models (Phase 2.3)
    'AudioEvent',
    'AudioEventFeatures',
    'AudioOracleStats',
    'OracleState',
    'TrainingResult',
    'TrainingMetadata',
    'PerformanceContext',
    'BehaviorMode',
]
