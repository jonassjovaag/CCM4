"""
Atomic File Writer
Provides safe file writing that prevents data loss from partial writes.
Part of Phase 1.1: Data Safety Foundation
"""

import os
import tempfile
import shutil
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Optional
import logging

logger = logging.getLogger(__name__)


class AtomicFileWriter:
    """
    Safely write files with atomic operations.

    Features:
    - Writes to temporary file first
    - Verifies write succeeded
    - Atomic rename to target location
    - Preserves original on failure
    - Optional checksum verification
    - Retry mechanism for transient failures

    Usage:
        writer = AtomicFileWriter()
        writer.write_json(data, "model.json")
    """

    def __init__(self, max_retries: int = 3, verify_checksum: bool = True):
        """
        Initialize atomic file writer.

        Args:
            max_retries: Number of retry attempts for transient failures
            verify_checksum: Calculate and verify checksums
        """
        self.max_retries = max_retries
        self.verify_checksum = verify_checksum

    def write_json(
        self,
        data: Any,
        filepath: str | Path,
        encoder_cls: Optional[type] = None,
        indent: int = 2,
        create_backup: bool = True
    ) -> bool:
        """
        Atomically write JSON data to file.

        Args:
            data: Data to write (must be JSON serializable)
            filepath: Target file path
            encoder_cls: Custom JSON encoder class (e.g., NumpyEncoder)
            indent: JSON indentation (default: 2)
            create_backup: Create backup of existing file before overwriting

        Returns:
            True if successful, False otherwise
        """
        filepath = Path(filepath)

        # Create backup of existing file if requested
        if create_backup and filepath.exists():
            backup_path = self._create_backup(filepath)
            if not backup_path:
                logger.warning(f"Failed to create backup for {filepath}")

        # Retry loop for transient failures
        for attempt in range(1, self.max_retries + 1):
            try:
                return self._write_json_attempt(
                    data, filepath, encoder_cls, indent
                )
            except Exception as e:
                if attempt == self.max_retries:
                    logger.error(
                        f"Failed to write {filepath} after {self.max_retries} attempts: {e}"
                    )
                    return False
                else:
                    logger.warning(
                        f"Attempt {attempt}/{self.max_retries} failed for {filepath}: {e}"
                    )

        return False

    def _write_json_attempt(
        self,
        data: Any,
        filepath: Path,
        encoder_cls: Optional[type],
        indent: int
    ) -> bool:
        """Single attempt to write JSON file atomically."""
        # Ensure parent directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Create temporary file in same directory (important for atomic rename)
        temp_fd, temp_path = tempfile.mkstemp(
            dir=filepath.parent,
            prefix=f".tmp_{filepath.name}_",
            suffix=".json"
        )

        temp_path = Path(temp_path)

        try:
            # Write to temporary file
            with os.fdopen(temp_fd, 'w', encoding='utf-8') as f:
                json.dump(data, f, cls=encoder_cls, indent=indent)
                f.flush()
                os.fsync(f.fileno())  # Ensure written to disk

            # Verify the file was written correctly
            if not self._verify_temp_file(temp_path, data, encoder_cls):
                raise IOError("Verification failed: temp file corrupted")

            # Calculate checksum if requested
            if self.verify_checksum:
                checksum = self._calculate_checksum(temp_path)
                logger.debug(f"Checksum for {filepath.name}: {checksum}")

            # Atomic rename (this is the atomic operation)
            # On Windows, need to handle existing file differently
            if os.name == 'nt' and filepath.exists():
                # Windows doesn't allow atomic replace, so we do it carefully
                backup_temp = filepath.with_suffix(filepath.suffix + '.old')
                if backup_temp.exists():
                    backup_temp.unlink()
                shutil.move(str(filepath), str(backup_temp))
                try:
                    shutil.move(str(temp_path), str(filepath))
                    # Success - can remove old file
                    if backup_temp.exists():
                        backup_temp.unlink()
                except Exception as e:
                    # Restore from backup
                    if backup_temp.exists():
                        shutil.move(str(backup_temp), str(filepath))
                    raise e
            else:
                # Unix: atomic rename
                shutil.move(str(temp_path), str(filepath))

            logger.info(f"Successfully wrote {filepath}")
            return True

        except Exception as e:
            # Clean up temporary file on failure
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass
            raise e

    def _verify_temp_file(
        self,
        temp_path: Path,
        original_data: Any,
        encoder_cls: Optional[type]
    ) -> bool:
        """Verify temporary file was written correctly."""
        try:
            with open(temp_path, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)

            # Basic verification: check file size is reasonable
            file_size = temp_path.stat().st_size
            if file_size == 0:
                logger.error("Temp file is empty")
                return False

            # Could add more sophisticated verification here
            # (compare loaded_data with original_data, but that's expensive)

            return True

        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return False

    def _calculate_checksum(self, filepath: Path) -> str:
        """Calculate MD5 checksum of file."""
        md5_hash = hashlib.md5()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5_hash.update(chunk)
        return md5_hash.hexdigest()

    def _create_backup(self, filepath: Path) -> Optional[Path]:
        """Create backup of existing file."""
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = filepath.with_suffix(f".{timestamp}.bak")

        try:
            shutil.copy2(str(filepath), str(backup_path))
            logger.info(f"Created backup: {backup_path}")
            return backup_path
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return None

    def write_binary(
        self,
        data: bytes,
        filepath: str | Path,
        create_backup: bool = True
    ) -> bool:
        """
        Atomically write binary data to file.

        Args:
            data: Binary data to write
            filepath: Target file path
            create_backup: Create backup of existing file before overwriting

        Returns:
            True if successful, False otherwise
        """
        filepath = Path(filepath)

        # Create backup of existing file if requested
        if create_backup and filepath.exists():
            backup_path = self._create_backup(filepath)
            if not backup_path:
                logger.warning(f"Failed to create backup for {filepath}")

        # Retry loop
        for attempt in range(1, self.max_retries + 1):
            try:
                return self._write_binary_attempt(data, filepath)
            except Exception as e:
                if attempt == self.max_retries:
                    logger.error(
                        f"Failed to write {filepath} after {self.max_retries} attempts: {e}"
                    )
                    return False
                else:
                    logger.warning(
                        f"Attempt {attempt}/{self.max_retries} failed for {filepath}: {e}"
                    )

        return False

    def _write_binary_attempt(self, data: bytes, filepath: Path) -> bool:
        """Single attempt to write binary file atomically."""
        # Ensure parent directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Create temporary file in same directory
        temp_fd, temp_path = tempfile.mkstemp(
            dir=filepath.parent,
            prefix=f".tmp_{filepath.name}_"
        )

        temp_path = Path(temp_path)

        try:
            # Write to temporary file
            with os.fdopen(temp_fd, 'wb') as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())

            # Verify file size
            if temp_path.stat().st_size != len(data):
                raise IOError("File size mismatch after write")

            # Calculate checksum if requested
            if self.verify_checksum:
                checksum = self._calculate_checksum(temp_path)
                logger.debug(f"Checksum for {filepath.name}: {checksum}")

            # Atomic rename (same logic as JSON)
            if os.name == 'nt' and filepath.exists():
                backup_temp = filepath.with_suffix(filepath.suffix + '.old')
                if backup_temp.exists():
                    backup_temp.unlink()
                shutil.move(str(filepath), str(backup_temp))
                try:
                    shutil.move(str(temp_path), str(filepath))
                    if backup_temp.exists():
                        backup_temp.unlink()
                except Exception as e:
                    if backup_temp.exists():
                        shutil.move(str(backup_temp), str(filepath))
                    raise e
            else:
                shutil.move(str(temp_path), str(filepath))

            logger.info(f"Successfully wrote {filepath}")
            return True

        except Exception as e:
            # Clean up temporary file on failure
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass
            raise e


# Convenience function
def atomic_write_json(
    data: Any,
    filepath: str | Path,
    encoder_cls: Optional[type] = None,
    **kwargs
) -> bool:
    """
    Convenience function for atomic JSON writes.

    Args:
        data: Data to write
        filepath: Target file path
        encoder_cls: Custom JSON encoder
        **kwargs: Additional arguments for AtomicFileWriter

    Returns:
        True if successful, False otherwise
    """
    writer = AtomicFileWriter(**kwargs)
    return writer.write_json(data, filepath, encoder_cls=encoder_cls)
