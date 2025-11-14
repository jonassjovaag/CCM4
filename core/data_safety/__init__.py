"""
Data Safety Module
Provides robust data handling with backup, validation, and atomic writes.
Part of Phase 1: Data Safety Foundation
"""

from .atomic_file_writer import AtomicFileWriter, atomic_write_json
from .backup_manager import BackupManager, create_backup, restore_backup
from .data_validator import DataValidator, validate_json_file
from .enhanced_save_load import (
    EnhancedSaveLoad,
    enhanced_save_json,
    enhanced_load_json,
    wrap_save_method,
    wrap_load_method
)

__all__ = [
    'AtomicFileWriter',
    'BackupManager',
    'DataValidator',
    'EnhancedSaveLoad',
    'atomic_write_json',
    'create_backup',
    'restore_backup',
    'validate_json_file',
    'enhanced_save_json',
    'enhanced_load_json',
    'wrap_save_method',
    'wrap_load_method'
]
