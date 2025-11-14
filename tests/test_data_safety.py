"""
Test Data Safety Components
Tests for AtomicFileWriter and BackupManager
"""

import sys
import json
import tempfile
import shutil
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from musichal.core.data_safety import AtomicFileWriter, BackupManager


def test_atomic_file_writer():
    """Test atomic file writing."""
    print("Testing AtomicFileWriter...")

    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        # Test 1: Write JSON file
        writer = AtomicFileWriter()
        test_data = {
            "name": "test_model",
            "version": "1.0",
            "data": [1, 2, 3, 4, 5]
        }

        test_file = temp_dir / "test.json"
        result = writer.write_json(test_data, test_file)

        assert result, "Failed to write JSON file"
        assert test_file.exists(), "JSON file was not created"

        # Verify contents
        with open(test_file, 'r') as f:
            loaded_data = json.load(f)

        assert loaded_data == test_data, "Data mismatch"
        print("  [PASS] JSON write test")

        # Test 2: Overwrite with backup
        test_data['version'] = "2.0"
        result = writer.write_json(test_data, test_file, create_backup=True)

        assert result, "Failed to overwrite JSON file"

        # Check backup was created
        backups = list(temp_dir.glob("*.bak"))
        assert len(backups) == 1, "Backup was not created"
        print("  [PASS] Backup creation test")

        # Test 3: Write binary file
        binary_data = b"Hello, World! This is binary data."
        binary_file = temp_dir / "test.bin"

        result = writer.write_binary(binary_data, binary_file)

        assert result, "Failed to write binary file"
        assert binary_file.exists(), "Binary file was not created"

        with open(binary_file, 'rb') as f:
            loaded_binary = f.read()

        assert loaded_binary == binary_data, "Binary data mismatch"
        print("  [PASS] Binary write test")

    print("[SUCCESS] AtomicFileWriter tests passed!")
    print()


def test_backup_manager():
    """Test backup management."""
    print("Testing BackupManager...")

    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        # Create test data to backup
        source_dir = temp_dir / "source_data"
        source_dir.mkdir()

        (source_dir / "file1.json").write_text('{"test": 1}')
        (source_dir / "file2.json").write_text('{"test": 2}')

        backup_root = temp_dir / "backups"

        # Test 1: Create backup
        manager = BackupManager(backup_root, max_backups=3)
        backup_path = manager.create_backup(
            source_dir,
            description="Test backup"
        )

        assert backup_path is not None, "Failed to create backup"
        assert backup_path.exists(), "Backup directory was not created"

        # Check metadata
        metadata_file = backup_path / "BACKUP_METADATA.json"
        assert metadata_file.exists(), "Metadata file was not created"

        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        assert metadata['files_count'] == 2, "File count mismatch"
        assert metadata['description'] == "Test backup", "Description mismatch"
        print("  [PASS] Backup creation test")

        # Test 2: List backups
        backups = manager.list_backups()
        assert len(backups) == 1, "Backup list count mismatch"
        print("  [PASS] List backups test")

        # Test 3: Restore backup
        restore_dir = temp_dir / "restored"
        result = manager.restore_backup(
            metadata['backup_name'],
            dest=restore_dir
        )

        assert result, "Failed to restore backup"
        assert restore_dir.exists(), "Restored directory was not created"
        assert (restore_dir / "file1.json").exists(), "file1 not restored"
        assert (restore_dir / "file2.json").exists(), "file2 not restored"
        print("  [PASS] Restore backup test")

        # Test 4: Retention policy
        # Create additional backups to test retention
        for i in range(5):
            manager.create_backup(
                source_dir,
                description=f"Backup {i}"
            )

        backups = manager.list_backups()
        assert len(backups) <= 3, f"Retention policy failed: {len(backups)} backups exist"
        print("  [PASS] Retention policy test")

    print("[SUCCESS] BackupManager tests passed!")
    print()


def main():
    """Run all tests."""
    print("=" * 60)
    print("DATA SAFETY COMPONENT TESTS")
    print("=" * 60)
    print()

    try:
        test_atomic_file_writer()
        test_backup_manager()

        print("=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        return 0

    except AssertionError as e:
        print(f"\n[FAIL] Test failed: {e}")
        return 1

    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
