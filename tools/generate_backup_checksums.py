#!/usr/bin/env python3
"""
Generate checksums for backup verification
Part of Phase 0: Pre-Flight Safety
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List
import sys


def calculate_md5(filepath: Path) -> str:
    """Calculate MD5 checksum of a file."""
    md5_hash = hashlib.md5()
    try:
        with open(filepath, 'rb') as f:
            # Read file in chunks to handle large files
            for chunk in iter(lambda: f.read(4096), b""):
                md5_hash.update(chunk)
        return md5_hash.hexdigest()
    except Exception as e:
        return f"ERROR: {str(e)}"


def generate_checksums(directory: Path) -> Dict[str, str]:
    """Generate checksums for all files in directory."""
    checksums = {}

    if not directory.exists():
        print(f"Error: Directory {directory} does not exist")
        return checksums

    # Get all JSON files recursively
    json_files = sorted(directory.rglob("*.json"))

    total_files = len(json_files)
    print(f"Calculating checksums for {total_files} JSON files...")

    for i, filepath in enumerate(json_files, 1):
        relative_path = filepath.relative_to(directory)
        checksum = calculate_md5(filepath)
        checksums[str(relative_path)] = checksum

        if i % 10 == 0 or i == total_files:
            print(f"Progress: {i}/{total_files} files processed")

    return checksums


def save_checksums(checksums: Dict[str, str], output_file: Path) -> None:
    """Save checksums to JSON file."""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(checksums, f, indent=2, sort_keys=True)
        print(f"\nChecksums saved to: {output_file}")
    except Exception as e:
        print(f"Error saving checksums: {e}")


def verify_checksums(directory: Path, checksums_file: Path) -> bool:
    """Verify files against saved checksums."""
    try:
        with open(checksums_file, 'r', encoding='utf-8') as f:
            expected_checksums = json.load(f)
    except Exception as e:
        print(f"Error loading checksums file: {e}")
        return False

    print(f"\nVerifying {len(expected_checksums)} files...")

    all_valid = True
    errors = []

    for relative_path, expected_checksum in expected_checksums.items():
        filepath = directory / relative_path

        if not filepath.exists():
            errors.append(f"MISSING: {relative_path}")
            all_valid = False
            continue

        actual_checksum = calculate_md5(filepath)

        if actual_checksum != expected_checksum:
            errors.append(f"MISMATCH: {relative_path}")
            errors.append(f"  Expected: {expected_checksum}")
            errors.append(f"  Actual:   {actual_checksum}")
            all_valid = False

    if all_valid:
        print("[SUCCESS] All files verified successfully!")
    else:
        print(f"\n[ERROR] Verification failed with {len(errors)} errors:")
        for error in errors:
            print(f"  {error}")

    return all_valid


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Generate: python generate_backup_checksums.py generate <backup_dir>")
        print("  Verify:   python generate_backup_checksums.py verify <backup_dir>")
        sys.exit(1)

    command = sys.argv[1]

    if command == "generate":
        if len(sys.argv) < 3:
            print("Error: Please provide backup directory")
            sys.exit(1)

        backup_dir = Path(sys.argv[2])
        json_dir = backup_dir / "JSON"

        if not json_dir.exists():
            print(f"Error: JSON directory not found in {backup_dir}")
            sys.exit(1)

        checksums = generate_checksums(json_dir)

        if checksums:
            output_file = backup_dir / "checksums.json"
            save_checksums(checksums, output_file)

            # Also create a human-readable version
            manifest_file = backup_dir / "BACKUP_MANIFEST.md"
            if manifest_file.exists():
                with open(manifest_file, 'a', encoding='utf-8') as f:
                    f.write(f"\n\n## Checksum Verification\n")
                    f.write(f"- **Total files:** {len(checksums)}\n")
                    f.write(f"- **Checksum file:** `checksums.json`\n")
                    f.write(f"- **Algorithm:** MD5\n")
                    f.write(f"\n### Sample Checksums (first 5 files)\n")
                    for i, (path, checksum) in enumerate(sorted(checksums.items())[:5]):
                        f.write(f"- `{path}`: `{checksum}`\n")

            print(f"\n[SUCCESS] Backup checksums generated successfully!")
            print(f"Total files: {len(checksums)}")
        else:
            print("[ERROR] No checksums generated")
            sys.exit(1)

    elif command == "verify":
        if len(sys.argv) < 3:
            print("Error: Please provide backup directory")
            sys.exit(1)

        backup_dir = Path(sys.argv[2])
        json_dir = backup_dir / "JSON"
        checksums_file = backup_dir / "checksums.json"

        if not checksums_file.exists():
            print(f"Error: Checksums file not found: {checksums_file}")
            sys.exit(1)

        if verify_checksums(json_dir, checksums_file):
            sys.exit(0)
        else:
            sys.exit(1)

    else:
        print(f"Unknown command: {command}")
        print("Use 'generate' or 'verify'")
        sys.exit(1)


if __name__ == "__main__":
    main()
