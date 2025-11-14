#!/usr/bin/env python3
"""
Add Metadata to Existing JSON Files
Migrates old JSON files to new format with metadata.
Part of Phase 1.4: Data Safety Foundation
"""

import sys
import json
from pathlib import Path
from typing import List, Dict
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.metadata_manager import MetadataManager


class MetadataMigrator:
    """Migrates JSON files to include metadata."""

    def __init__(self, dry_run: bool = False, backup: bool = True):
        """
        Initialize migrator.

        Args:
            dry_run: If True, show what would be done without making changes
            backup: Create backups before modifying files
        """
        self.dry_run = dry_run
        self.backup = backup
        self.manager = MetadataManager()

        self.processed = 0
        self.already_migrated = 0
        self.failed = 0
        self.skipped = 0

    def migrate_file(
        self,
        filepath: Path,
        parameters: Dict = None
    ) -> bool:
        """
        Migrate a single file to include metadata.

        Args:
            filepath: Path to JSON file
            parameters: Optional parameters to include

        Returns:
            True if successful, False otherwise
        """
        if not filepath.exists():
            print(f"  [SKIP] File not found: {filepath}")
            self.skipped += 1
            return False

        # Check file size
        file_size = filepath.stat().st_size
        if file_size == 0:
            print(f"  [SKIP] Empty file: {filepath}")
            self.skipped += 1
            return False

        # Try to load file
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"  [FAIL] Invalid JSON: {filepath} - {e}")
            self.failed += 1
            return False
        except Exception as e:
            print(f"  [FAIL] Could not read: {filepath} - {e}")
            self.failed += 1
            return False

        # Check if already has metadata
        if "metadata" in data and "data" in data:
            print(f"  [SKIP] Already has metadata: {filepath.name}")
            self.already_migrated += 1
            return True

        # In dry-run mode, just report what would be done
        if self.dry_run:
            print(f"  [DRY-RUN] Would add metadata to: {filepath.name}")
            self.processed += 1
            return True

        # Infer description from filename
        description = self._infer_description(filepath, data)

        # Infer training source if possible
        training_source = self._infer_training_source(filepath, data)

        # Add metadata
        success = self.manager.add_metadata_to_existing_file(
            filepath,
            training_source=training_source,
            parameters=parameters,
            description=description,
            backup=self.backup
        )

        if success:
            print(f"  [OK] Added metadata: {filepath.name}")
            self.processed += 1
        else:
            print(f"  [FAIL] Could not add metadata: {filepath.name}")
            self.failed += 1

        return success

    def migrate_directory(
        self,
        directory: Path,
        pattern: str = "*.json",
        parameters: Dict = None
    ) -> None:
        """
        Migrate all JSON files in a directory.

        Args:
            directory: Directory containing JSON files
            pattern: Glob pattern for files to migrate
            parameters: Optional parameters to include
        """
        if not directory.exists():
            print(f"Error: Directory not found: {directory}")
            return

        json_files = sorted(directory.glob(pattern))
        total = len(json_files)

        print(f"\nMigrating {total} files in {directory}...")
        print()

        for i, filepath in enumerate(json_files, 1):
            print(f"[{i}/{total}] {filepath.name}")
            self.migrate_file(filepath, parameters)

    def _infer_description(self, filepath: Path, data: Dict) -> str:
        """Infer description from filename and data."""
        name = filepath.stem

        # Check if it's a training results file
        if data.get("training_successful") is not None:
            if data.get("training_successful"):
                return f"Training results for {name} (successful)"
            else:
                return f"Training results for {name} (failed)"

        # Check if it's a rhythm oracle
        if "rhythm_oracle" in name.lower():
            return f"Rhythm oracle data: {name}"

        # Check if it's correlation data
        if "correlation" in name.lower():
            return f"Correlation analysis: {name}"

        # Check if it's harmonic transitions
        if "harmonic" in name.lower():
            return f"Harmonic transitions: {name}"

        # Default
        return f"Musical data: {name}"

    def _infer_training_source(self, filepath: Path, data: Dict) -> str:
        """Try to infer the training source audio file."""
        name = filepath.stem.lower()

        # Common patterns
        source_names = [
            "curious_child",
            "curious child",
            "georgia",
            "grab-a-hold",
            "grab_a_hold"
        ]

        for source in source_names:
            if source in name:
                # Try to find the audio file
                possible_extensions = [".wav", ".mp3", ".aiff", ".flac"]
                for ext in possible_extensions:
                    audio_file = filepath.parent.parent / "input_audio" / f"{source}{ext}"
                    if audio_file.exists():
                        return str(audio_file)

                # Return source name even if file doesn't exist
                return f"{source}.wav"

        return "unknown"

    def print_summary(self) -> None:
        """Print migration summary."""
        print()
        print("=" * 60)
        print("MIGRATION SUMMARY")
        print("=" * 60)
        print(f"Files processed:       {self.processed}")
        print(f"Already migrated:      {self.already_migrated}")
        print(f"Failed:                {self.failed}")
        print(f"Skipped:               {self.skipped}")
        print(f"Total:                 {self.processed + self.already_migrated + self.failed + self.skipped}")
        print("=" * 60)

        if self.dry_run:
            print("\n[DRY-RUN MODE] No files were actually modified.")
            print("Run without --dry-run to perform migration.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Add metadata to existing JSON files"
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="Directory containing JSON files"
    )
    parser.add_argument(
        "--pattern",
        default="*.json",
        help="Glob pattern for files to migrate (default: *.json)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Don't create backup files"
    )

    args = parser.parse_args()

    # Initialize migrator
    migrator = MetadataMigrator(
        dry_run=args.dry_run,
        backup=not args.no_backup
    )

    # Run migration
    migrator.migrate_directory(
        args.directory,
        pattern=args.pattern
    )

    # Print summary
    migrator.print_summary()

    # Exit with error code if any failures
    if migrator.failed > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
