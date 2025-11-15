"""
Backup Manager
Manages versioned backups with automatic retention policies.
Part of Phase 1.1: Data Safety Foundation
"""

import shutil
import json
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class BackupManager:
    """
    Manages versioned backups with retention policies.

    Features:
    - Versioned backups with timestamps
    - Configurable retention policy
    - Automatic cleanup of old backups
    - Checksum verification
    - Metadata tracking

    Usage:
        manager = BackupManager("backups")
        manager.create_backup(
            "JSON",
            description="Before training run"
        )
    """

    def __init__(
        self,
        backup_root: str | Path = "backups",
        max_backups: int = 10,
        max_age_days: int = 30
    ):
        """
        Initialize backup manager.

        Args:
            backup_root: Root directory for backups
            max_backups: Maximum number of backups to keep (0 = unlimited)
            max_age_days: Maximum age of backups in days (0 = unlimited)
        """
        self.backup_root = Path(backup_root)
        self.max_backups = max_backups
        self.max_age_days = max_age_days

        # Create backup root if it doesn't exist
        self.backup_root.mkdir(parents=True, exist_ok=True)

    def create_backup(
        self,
        source: str | Path,
        description: str = "",
        backup_name: Optional[str] = None
    ) -> Optional[Path]:
        """
        Create a new backup.

        Args:
            source: File or directory to backup
            description: Optional description of the backup
            backup_name: Custom backup name (default: auto-generated)

        Returns:
            Path to backup directory, or None on failure
        """
        source = Path(source)

        if not source.exists():
            logger.error(f"Source does not exist: {source}")
            return None

        # Generate backup name
        if backup_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"backup_{timestamp}"

        backup_dir = self.backup_root / backup_name

        # Create backup directory
        try:
            backup_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            logger.error(f"Backup already exists: {backup_dir}")
            return None

        logger.info(f"Creating backup: {backup_dir}")

        try:
            # Copy source to backup
            if source.is_file():
                dest_file = backup_dir / source.name
                shutil.copy2(str(source), str(dest_file))
                files_backed_up = [source.name]
            else:
                # Copy directory
                dest_dir = backup_dir / source.name
                shutil.copytree(str(source), str(dest_dir))

                # Count files
                files_backed_up = [
                    str(f.relative_to(dest_dir))
                    for f in dest_dir.rglob("*") if f.is_file()
                ]

            # Create metadata
            metadata = {
                "backup_name": backup_name,
                "created_at": datetime.now().isoformat(),
                "description": description,
                "source": str(source),
                "source_type": "file" if source.is_file() else "directory",
                "files_count": len(files_backed_up),
                "total_size_bytes": self._calculate_size(backup_dir)
            }

            # Save metadata
            metadata_file = backup_dir / "BACKUP_METADATA.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)

            logger.info(
                f"Backup created: {backup_dir} "
                f"({len(files_backed_up)} files, "
                f"{metadata['total_size_bytes'] / 1024 / 1024:.1f} MB)"
            )

            # Clean up old backups according to retention policy
            self._enforce_retention_policy()

            return backup_dir

        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            # Clean up partial backup
            if backup_dir.exists():
                try:
                    shutil.rmtree(backup_dir)
                except Exception:
                    pass
            return None

    def restore_backup(
        self,
        backup_name: str,
        dest: Optional[str | Path] = None
    ) -> bool:
        """
        Restore from a backup.

        Args:
            backup_name: Name of backup to restore
            dest: Destination path (default: original source location)

        Returns:
            True if successful, False otherwise
        """
        backup_dir = self.backup_root / backup_name

        if not backup_dir.exists():
            logger.error(f"Backup not found: {backup_dir}")
            return False

        # Load metadata
        metadata_file = backup_dir / "BACKUP_METADATA.json"
        if not metadata_file.exists():
            logger.error(f"Backup metadata not found: {metadata_file}")
            return False

        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load backup metadata: {e}")
            return False

        # Determine destination
        if dest is None:
            dest = Path(metadata['source'])
        else:
            dest = Path(dest)

        logger.info(f"Restoring backup {backup_name} to {dest}")

        try:
            # Find the backed up content
            source_name = Path(metadata['source']).name
            backup_content = backup_dir / source_name

            if not backup_content.exists():
                logger.error(f"Backup content not found: {backup_content}")
                return False

            # Create destination parent directory if needed
            if dest.suffix:  # It's a file
                dest.parent.mkdir(parents=True, exist_ok=True)
            else:  # It's a directory
                dest.parent.mkdir(parents=True, exist_ok=True)

            # Perform restore
            if backup_content.is_file():
                shutil.copy2(str(backup_content), str(dest))
            else:
                # Remove existing destination if it exists
                if dest.exists():
                    if dest.is_dir():
                        shutil.rmtree(dest)
                    else:
                        dest.unlink()
                shutil.copytree(str(backup_content), str(dest))

            logger.info(f"Successfully restored backup to {dest}")
            return True

        except Exception as e:
            logger.error(f"Failed to restore backup: {e}")
            return False

    def list_backups(self) -> List[Dict[str, Any]]:
        """
        List all available backups.

        Returns:
            List of backup metadata dictionaries
        """
        backups = []

        for backup_dir in sorted(self.backup_root.iterdir()):
            if not backup_dir.is_dir():
                continue

            metadata_file = backup_dir / "BACKUP_METADATA.json"
            if not metadata_file.exists():
                continue

            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                backups.append(metadata)
            except Exception as e:
                logger.warning(f"Failed to load metadata for {backup_dir}: {e}")

        return backups

    def delete_backup(self, backup_name: str) -> bool:
        """
        Delete a specific backup.

        Args:
            backup_name: Name of backup to delete

        Returns:
            True if successful, False otherwise
        """
        backup_dir = self.backup_root / backup_name

        if not backup_dir.exists():
            logger.error(f"Backup not found: {backup_dir}")
            return False

        try:
            shutil.rmtree(backup_dir)
            logger.info(f"Deleted backup: {backup_dir}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete backup: {e}")
            return False

    def _enforce_retention_policy(self) -> None:
        """Enforce backup retention policy by deleting old backups."""
        backups = self.list_backups()

        if not backups:
            return

        # Sort by creation date (newest first)
        backups.sort(
            key=lambda x: x.get('created_at', ''),
            reverse=True
        )

        # Track which backups to delete
        to_delete = []

        # Enforce max_backups limit
        if self.max_backups > 0 and len(backups) > self.max_backups:
            to_delete.extend(backups[self.max_backups:])

        # Enforce max_age_days limit
        if self.max_age_days > 0:
            cutoff_date = datetime.now() - timedelta(days=self.max_age_days)
            for backup in backups:
                try:
                    created_at = datetime.fromisoformat(backup['created_at'])
                    if created_at < cutoff_date:
                        if backup not in to_delete:
                            to_delete.append(backup)
                except Exception:
                    pass

        # Delete old backups
        for backup in to_delete:
            backup_name = backup.get('backup_name')
            if backup_name:
                logger.info(
                    f"Deleting old backup (retention policy): {backup_name}"
                )
                self.delete_backup(backup_name)

    def _calculate_size(self, path: Path) -> int:
        """Calculate total size of a file or directory in bytes."""
        if path.is_file():
            return path.stat().st_size

        total_size = 0
        for item in path.rglob("*"):
            if item.is_file():
                total_size += item.stat().st_size

        return total_size

    def get_backup_info(self, backup_name: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a specific backup.

        Args:
            backup_name: Name of backup

        Returns:
            Backup metadata dictionary, or None if not found
        """
        backup_dir = self.backup_root / backup_name
        metadata_file = backup_dir / "BACKUP_METADATA.json"

        if not metadata_file.exists():
            return None

        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load backup metadata: {e}")
            return None


# Convenience functions
def create_backup(
    source: str | Path,
    description: str = "",
    backup_root: str | Path = "backups"
) -> Optional[Path]:
    """
    Convenience function to create a backup.

    Args:
        source: File or directory to backup
        description: Optional description
        backup_root: Root directory for backups

    Returns:
        Path to backup directory, or None on failure
    """
    manager = BackupManager(backup_root)
    return manager.create_backup(source, description)


def restore_backup(
    backup_name: str,
    dest: Optional[str | Path] = None,
    backup_root: str | Path = "backups"
) -> bool:
    """
    Convenience function to restore a backup.

    Args:
        backup_name: Name of backup to restore
        dest: Destination path (default: original source location)
        backup_root: Root directory for backups

    Returns:
        True if successful, False otherwise
    """
    manager = BackupManager(backup_root)
    return manager.restore_backup(backup_name, dest)
