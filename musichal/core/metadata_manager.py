"""
Metadata Manager
Adds comprehensive metadata to training data for reproducibility.
Part of Phase 1.4: Data Safety Foundation
"""

import sys
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import json
import logging

logger = logging.getLogger(__name__)


class MetadataManager:
    """
    Manages metadata for training data files.

    Provides:
    - Version tracking
    - Timestamp tracking
    - Training parameter recording
    - Git commit tracking
    - System information
    - Reproducibility data
    """

    METADATA_VERSION = "2.0"

    def __init__(self):
        """Initialize metadata manager."""
        self._git_commit = None
        self._system_info = None

    def create_metadata(
        self,
        training_source: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        description: str = "",
        additional_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create comprehensive metadata dictionary.

        Args:
            training_source: Path to source audio/data file
            parameters: Training parameters used
            description: Human-readable description
            additional_info: Any additional metadata fields

        Returns:
            Metadata dictionary
        """
        metadata = {
            "version": self.METADATA_VERSION,
            "created_at": datetime.now().isoformat(),
            "description": description
        }

        # Add training source
        if training_source:
            metadata["training_source"] = str(training_source)
            # Add source file info if it exists
            source_path = Path(training_source)
            if source_path.exists():
                metadata["source_file_size"] = source_path.stat().st_size
                metadata["source_file_modified"] = datetime.fromtimestamp(
                    source_path.stat().st_mtime
                ).isoformat()

        # Add parameters
        if parameters:
            metadata["parameters"] = self._sanitize_parameters(parameters)

        # Add git information
        git_info = self.get_git_info()
        if git_info:
            metadata["git_commit"] = git_info["commit"]
            metadata["git_branch"] = git_info["branch"]
            metadata["git_dirty"] = git_info["dirty"]

        # Add system information
        metadata["system_info"] = self.get_system_info()

        # Add any additional info
        if additional_info:
            metadata["additional"] = additional_info

        return metadata

    def wrap_data_with_metadata(
        self,
        data: Dict[str, Any],
        training_source: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        description: str = ""
    ) -> Dict[str, Any]:
        """
        Wrap existing data with metadata.

        Args:
            data: Existing data dictionary
            training_source: Path to source audio/data file
            parameters: Training parameters used
            description: Human-readable description

        Returns:
            Dictionary with metadata and data sections
        """
        metadata = self.create_metadata(
            training_source=training_source,
            parameters=parameters,
            description=description
        )

        return {
            "metadata": metadata,
            "data": data
        }

    def extract_data(self, wrapped_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract data from metadata-wrapped structure.

        Args:
            wrapped_data: Data dictionary potentially containing metadata

        Returns:
            Just the data portion (or entire dict if no metadata)
        """
        if "metadata" in wrapped_data and "data" in wrapped_data:
            return wrapped_data["data"]
        return wrapped_data

    def get_metadata(self, wrapped_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Extract metadata from wrapped structure.

        Args:
            wrapped_data: Data dictionary potentially containing metadata

        Returns:
            Metadata dictionary, or None if not found
        """
        return wrapped_data.get("metadata")

    def get_git_info(self) -> Optional[Dict[str, str]]:
        """
        Get current git commit information.

        Returns:
            Dictionary with commit, branch, and dirty status
        """
        if self._git_commit is not None:
            return self._git_commit

        try:
            # Get commit hash
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True
            ).strip()

            # Get branch name
            branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True
            ).strip()

            # Check if working directory is dirty
            status = subprocess.check_output(
                ["git", "status", "--porcelain"],
                stderr=subprocess.DEVNULL,
                text=True
            ).strip()

            dirty = len(status) > 0

            self._git_commit = {
                "commit": commit[:8],  # Short hash
                "commit_full": commit,
                "branch": branch,
                "dirty": dirty
            }

            return self._git_commit

        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.debug("Git information not available")
            return None

    def get_system_info(self) -> Dict[str, str]:
        """
        Get system information.

        Returns:
            Dictionary with Python version, platform, etc.
        """
        if self._system_info is not None:
            return self._system_info

        self._system_info = {
            "python_version": sys.version.split()[0],
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "machine": platform.machine()
        }

        # Add key package versions
        try:
            import numpy
            self._system_info["numpy_version"] = numpy.__version__
        except ImportError:
            pass

        try:
            import torch
            self._system_info["torch_version"] = torch.__version__
            if torch.cuda.is_available():
                self._system_info["cuda_available"] = True
                self._system_info["cuda_version"] = torch.version.cuda
        except ImportError:
            pass

        try:
            import librosa
            self._system_info["librosa_version"] = librosa.__version__
        except ImportError:
            pass

        return self._system_info

    def _sanitize_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize parameters for JSON serialization.

        Args:
            parameters: Raw parameters dictionary

        Returns:
            Sanitized parameters dictionary
        """
        sanitized = {}

        for key, value in parameters.items():
            # Convert non-serializable types
            if isinstance(value, Path):
                sanitized[key] = str(value)
            elif isinstance(value, type):
                sanitized[key] = value.__name__
            elif hasattr(value, '__dict__') and not isinstance(value, (str, int, float, bool, list, dict)):
                # Complex object - convert to string representation
                sanitized[key] = str(value)
            else:
                sanitized[key] = value

        return sanitized

    def add_metadata_to_existing_file(
        self,
        filepath: str | Path,
        training_source: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        description: str = "",
        backup: bool = True
    ) -> bool:
        """
        Add metadata to an existing JSON file.

        Args:
            filepath: Path to existing JSON file
            training_source: Source audio file
            parameters: Training parameters
            description: Description
            backup: Create backup before modifying

        Returns:
            True if successful, False otherwise
        """
        filepath = Path(filepath)

        if not filepath.exists():
            logger.error(f"File not found: {filepath}")
            return False

        try:
            # Load existing data
            with open(filepath, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)

            # Check if already has metadata
            if "metadata" in existing_data:
                logger.info(f"File already has metadata: {filepath}")
                return True

            # Create backup if requested
            if backup:
                backup_path = filepath.with_suffix(filepath.suffix + '.pre_metadata_backup')
                import shutil
                shutil.copy2(str(filepath), str(backup_path))
                logger.info(f"Created backup: {backup_path}")

            # Wrap with metadata
            wrapped = self.wrap_data_with_metadata(
                existing_data,
                training_source=training_source,
                parameters=parameters,
                description=description or f"Migrated from {filepath.name}"
            )

            # Save back
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(wrapped, f, indent=2)

            logger.info(f"Added metadata to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to add metadata to {filepath}: {e}")
            return False

    def validate_metadata(self, metadata: Dict[str, Any]) -> bool:
        """
        Validate metadata structure.

        Args:
            metadata: Metadata dictionary

        Returns:
            True if valid, False otherwise
        """
        required_fields = ["version", "created_at"]

        for field in required_fields:
            if field not in metadata:
                logger.error(f"Missing required metadata field: {field}")
                return False

        # Check version format
        if not isinstance(metadata["version"], str):
            logger.error("Metadata version must be a string")
            return False

        # Check timestamp format
        try:
            datetime.fromisoformat(metadata["created_at"])
        except ValueError:
            logger.error("Invalid timestamp format in metadata")
            return False

        return True


# Singleton instance for convenience
_default_manager = MetadataManager()


def create_metadata(**kwargs) -> Dict[str, Any]:
    """
    Convenience function to create metadata.

    Usage:
        metadata = create_metadata(
            training_source="audio.wav",
            parameters={"max_events": 15000}
        )
    """
    return _default_manager.create_metadata(**kwargs)


def wrap_with_metadata(data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """
    Convenience function to wrap data with metadata.

    Usage:
        wrapped = wrap_with_metadata(
            training_data,
            training_source="audio.wav",
            parameters=training_params
        )
    """
    return _default_manager.wrap_data_with_metadata(data, **kwargs)


def extract_data(wrapped_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to extract data from wrapped structure.

    Usage:
        data = extract_data(loaded_json)
    """
    return _default_manager.extract_data(wrapped_data)


def get_metadata(wrapped_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Convenience function to get metadata from wrapped structure.

    Usage:
        metadata = get_metadata(loaded_json)
        if metadata:
            print(f"Created: {metadata['created_at']}")
    """
    return _default_manager.get_metadata(wrapped_data)
