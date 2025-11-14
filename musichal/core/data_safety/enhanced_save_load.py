"""
Enhanced Save/Load Wrappers
Provides robust save/load operations with proper error handling, retry logic, and logging.
Part of Phase 1.3: Data Safety Foundation

This module provides drop-in replacements for existing save_to_file/load_from_file methods
that use the AtomicFileWriter and proper error handling.
"""

import logging
import json
from pathlib import Path
from typing import Any, Optional, Dict, Callable
from .atomic_file_writer import AtomicFileWriter
from .data_validator import DataValidator
from .backup_manager import BackupManager

# Import metadata manager from parent directory
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from metadata_manager import MetadataManager

logger = logging.getLogger(__name__)


class SaveLoadError(Exception):
    """Raised when save/load operations fail after all retries."""
    pass


class EnhancedSaveLoad:
    """
    Provides enhanced save/load operations with:
    - Atomic writes
    - Retry logic
    - Validation
    - Automatic backups
    - Detailed error logging
    """

    def __init__(
        self,
        validate: bool = True,
        create_backups: bool = True,
        add_metadata: bool = True,
        max_retries: int = 3
    ):
        """
        Initialize enhanced save/load.

        Args:
            validate: Perform data validation before saving
            create_backups: Create backups before overwriting
            add_metadata: Automatically add metadata to saved files
            max_retries: Number of retry attempts
        """
        self.validate = validate
        self.create_backups = create_backups
        self.add_metadata = add_metadata
        self.max_retries = max_retries

        self.writer = AtomicFileWriter(max_retries=max_retries)
        self.validator = DataValidator() if validate else None
        self.backup_manager = BackupManager() if create_backups else None
        self.metadata_manager = MetadataManager() if add_metadata else None

    def save_json(
        self,
        data: Any,
        filepath: str | Path,
        encoder_cls: Optional[type] = None,
        schema_name: Optional[str] = None,
        description: str = "",
        training_source: Optional[str] = None,
        parameters: Optional[Dict] = None
    ) -> bool:
        """
        Save data to JSON file with full safety features.

        Args:
            data: Data to save
            filepath: Target file path
            encoder_cls: Custom JSON encoder (e.g., NumpyEncoder)
            schema_name: Optional schema for validation
            description: Description for backup and metadata
            training_source: Source audio/data file for metadata
            parameters: Training parameters for metadata

        Returns:
            True if successful, False otherwise

        Raises:
            SaveLoadError: If save fails after all retries and strict mode enabled
        """
        filepath = Path(filepath)

        # Step 0: Add metadata if requested
        if self.add_metadata and self.metadata_manager:
            # Only wrap if data doesn't already have metadata
            if not ("metadata" in data and "data" in data):
                data = self.metadata_manager.wrap_data_with_metadata(
                    data,
                    training_source=training_source,
                    parameters=parameters,
                    description=description or f"Saved {filepath.name}"
                )

        # Step 1: Validate data if requested
        if self.validate and schema_name:
            try:
                # Extract data for validation if wrapped
                validate_data = data.get("data", data) if isinstance(data, dict) else data
                errors = self.validator.validate(validate_data, schema_name)
                if errors:
                    logger.error(f"Validation failed for {filepath}: {errors[:3]}")
                    return False
            except Exception as e:
                logger.warning(f"Validation check failed: {e}")
                # Continue anyway - validation is not critical

        # Step 2: Create backup if file exists
        if self.create_backups and filepath.exists():
            try:
                backup_path = self.backup_manager.create_backup(
                    filepath,
                    description=description or f"Before saving {filepath.name}"
                )
                if backup_path:
                    logger.info(f"Created backup: {backup_path}")
            except Exception as e:
                logger.warning(f"Backup creation failed: {e}")
                # Continue anyway - backup failure shouldn't prevent save

        # Step 3: Perform atomic write with retries
        success = self.writer.write_json(
            data,
            filepath,
            encoder_cls=encoder_cls,
            create_backup=False  # We already created backup above
        )

        if success:
            logger.info(f"Successfully saved {filepath}")
        else:
            logger.error(f"Failed to save {filepath} after {self.max_retries} attempts")

        return success

    def load_json(
        self,
        filepath: str | Path,
        schema_name: Optional[str] = None,
        fallback_to_backup: bool = True
    ) -> Optional[Dict]:
        """
        Load data from JSON file with validation and fallback.

        Args:
            filepath: Path to JSON file
            schema_name: Optional schema for validation
            fallback_to_backup: If load fails, try to load from backup

        Returns:
            Loaded data, or None if load failed

        Raises:
            SaveLoadError: If load fails and strict mode enabled
        """
        filepath = Path(filepath)

        if not filepath.exists():
            logger.error(f"File not found: {filepath}")
            return None

        # Step 1: Try to load file
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            logger.info(f"Successfully loaded {filepath}")

            # Step 2: Validate if requested
            if self.validate and schema_name:
                try:
                    errors = self.validator.validate(data, schema_name)
                    if errors:
                        logger.warning(f"Loaded data has validation errors: {errors[:3]}")
                        # Return anyway - data is structurally loaded
                except Exception as e:
                    logger.warning(f"Validation check failed: {e}")

            return data

        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error in {filepath}: {e}")

            # Try to recover from backup
            if fallback_to_backup and self.backup_manager:
                return self._try_load_from_backup(filepath)

            return None

        except Exception as e:
            logger.error(f"Failed to load {filepath}: {e}")

            # Try to recover from backup
            if fallback_to_backup and self.backup_manager:
                return self._try_load_from_backup(filepath)

            return None

    def _try_load_from_backup(self, filepath: Path) -> Optional[Dict]:
        """Try to load data from most recent backup."""
        logger.info(f"Attempting to recover from backup...")

        try:
            backups = self.backup_manager.list_backups()
            if not backups:
                logger.error("No backups available")
                return None

            # Find backups for this file
            relevant_backups = [
                b for b in backups
                if filepath.name in b.get('source', '')
            ]

            if not relevant_backups:
                logger.error(f"No backups found for {filepath.name}")
                return None

            # Try most recent backup
            recent_backup = relevant_backups[0]
            backup_name = recent_backup['backup_name']

            logger.info(f"Found backup: {backup_name}")

            # Load from backup
            backup_file = self.backup_manager.backup_root / backup_name / filepath.name

            if not backup_file.exists():
                logger.error(f"Backup file not found: {backup_file}")
                return None

            with open(backup_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            logger.info(f"Successfully recovered from backup: {backup_name}")
            return data

        except Exception as e:
            logger.error(f"Backup recovery failed: {e}")
            return None


# Convenience functions for drop-in replacement
_default_handler = EnhancedSaveLoad()


def enhanced_save_json(
    data: Any,
    filepath: str | Path,
    encoder_cls: Optional[type] = None,
    **kwargs
) -> bool:
    """
    Drop-in replacement for save_to_file() methods.

    Usage:
        from core.data_safety.enhanced_save_load import enhanced_save_json
        success = enhanced_save_json(data, "model.json", encoder_cls=NumpyEncoder)
    """
    return _default_handler.save_json(data, filepath, encoder_cls=encoder_cls, **kwargs)


def enhanced_load_json(
    filepath: str | Path,
    **kwargs
) -> Optional[Dict]:
    """
    Drop-in replacement for load_from_file() methods.

    Usage:
        from core.data_safety.enhanced_save_load import enhanced_load_json
        data = enhanced_load_json("model.json")
    """
    return _default_handler.load_json(filepath, **kwargs)


def wrap_save_method(
    original_save_method: Callable,
    encoder_cls: Optional[type] = None
) -> Callable:
    """
    Decorator to wrap existing save_to_file methods.

    Usage:
        class MyClass:
            @wrap_save_method
            def save_to_file(self, filepath: str) -> bool:
                # Original implementation (will be replaced)
                pass
    """
    def wrapped(self, filepath: str) -> bool:
        # Extract data from self
        # This is a generic wrapper - specific classes may need custom logic
        return enhanced_save_json(
            self.__dict__,
            filepath,
            encoder_cls=encoder_cls
        )
    return wrapped


def wrap_load_method(original_load_method: Callable) -> Callable:
    """
    Decorator to wrap existing load_from_file methods.

    Usage:
        class MyClass:
            @wrap_load_method
            def load_from_file(self, filepath: str) -> bool:
                # Original implementation (will be replaced)
                pass
    """
    def wrapped(self, filepath: str) -> bool:
        data = enhanced_load_json(filepath)
        if data is None:
            return False

        # Update self with loaded data
        for key, value in data.items():
            setattr(self, key, value)

        return True
    return wrapped
