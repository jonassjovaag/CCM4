"""
Test Metadata Manager
Tests for metadata creation and management
"""

import sys
import json
import tempfile
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from musichal.core.metadata_manager import MetadataManager, create_metadata, wrap_with_metadata


def test_metadata_creation():
    """Test basic metadata creation."""
    print("Testing metadata creation...")

    manager = MetadataManager()

    # Test 1: Basic metadata
    metadata = manager.create_metadata(
        description="Test metadata"
    )

    assert "version" in metadata, "Missing version"
    assert "created_at" in metadata, "Missing timestamp"
    assert "description" in metadata, "Missing description"
    assert metadata["version"] == "2.0", "Wrong version"
    print("  [PASS] Basic metadata creation")

    # Test 2: Metadata with parameters
    params = {
        "max_events": 15000,
        "distance_threshold": 1.5,
        "feature_dimensions": 768
    }

    metadata = manager.create_metadata(
        training_source="test_audio.wav",
        parameters=params,
        description="Training run"
    )

    assert "training_source" in metadata, "Missing training source"
    assert "parameters" in metadata, "Missing parameters"
    assert metadata["parameters"]["max_events"] == 15000, "Parameter not preserved"
    print("  [PASS] Metadata with parameters")

    # Test 3: System info
    assert "system_info" in metadata, "Missing system info"
    assert "python_version" in metadata["system_info"], "Missing Python version"
    assert "platform" in metadata["system_info"], "Missing platform"
    print("  [PASS] System info included")

    # Test 4: Git info (if available)
    git_info = manager.get_git_info()
    if git_info:
        assert "commit" in git_info, "Missing commit"
        assert "branch" in git_info, "Missing branch"
        print("  [PASS] Git info available")
    else:
        print("  [SKIP] Git not available")

    print("[SUCCESS] Metadata creation tests passed!")
    print()


def test_data_wrapping():
    """Test wrapping data with metadata."""
    print("Testing data wrapping...")

    manager = MetadataManager()

    # Test 1: Wrap data
    original_data = {
        "training_successful": True,
        "events_processed": 3000,
        "audio_oracle_stats": {
            "total_states": 3001,
            "sequence_length": 3000
        }
    }

    wrapped = manager.wrap_data_with_metadata(
        original_data,
        training_source="test.wav",
        description="Test training"
    )

    assert "metadata" in wrapped, "Missing metadata section"
    assert "data" in wrapped, "Missing data section"
    assert wrapped["data"] == original_data, "Data was modified"
    print("  [PASS] Data wrapping")

    # Test 2: Extract data
    extracted = manager.extract_data(wrapped)
    assert extracted == original_data, "Extracted data doesn't match"
    print("  [PASS] Data extraction")

    # Test 3: Get metadata
    metadata = manager.get_metadata(wrapped)
    assert metadata is not None, "Couldn't get metadata"
    assert "version" in metadata, "Metadata missing version"
    print("  [PASS] Metadata extraction")

    # Test 4: Handle unwrapped data
    extracted_unwrapped = manager.extract_data(original_data)
    assert extracted_unwrapped == original_data, "Should return data as-is if not wrapped"
    print("  [PASS] Handle unwrapped data")

    print("[SUCCESS] Data wrapping tests passed!")
    print()


def test_file_migration():
    """Test adding metadata to existing files."""
    print("Testing file migration...")

    manager = MetadataManager()

    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        # Test 1: Create test file without metadata
        test_file = temp_dir / "test.json"
        original_data = {
            "training_successful": True,
            "events_processed": 1000
        }

        with open(test_file, 'w') as f:
            json.dump(original_data, f)

        # Add metadata
        success = manager.add_metadata_to_existing_file(
            test_file,
            training_source="test.wav",
            description="Migration test",
            backup=True
        )

        assert success, "Failed to add metadata"
        print("  [PASS] Add metadata to file")

        # Test 2: Verify backup was created
        backup_files = list(temp_dir.glob("*.pre_metadata_backup"))
        assert len(backup_files) == 1, "Backup not created"
        print("  [PASS] Backup created")

        # Test 3: Verify file has metadata
        with open(test_file, 'r') as f:
            migrated_data = json.load(f)

        assert "metadata" in migrated_data, "Metadata not added"
        assert "data" in migrated_data, "Data section missing"
        assert migrated_data["data"] == original_data, "Data was changed"
        print("  [PASS] File structure correct")

        # Test 4: Try to add metadata again (should detect already has it)
        success = manager.add_metadata_to_existing_file(
            test_file,
            description="Second attempt"
        )

        assert success, "Should succeed on already-migrated file"
        print("  [PASS] Handle already-migrated file")

    print("[SUCCESS] File migration tests passed!")
    print()


def test_metadata_validation():
    """Test metadata validation."""
    print("Testing metadata validation...")

    manager = MetadataManager()

    # Test 1: Valid metadata
    valid_metadata = {
        "version": "2.0",
        "created_at": datetime.now().isoformat(),
        "description": "Test"
    }

    assert manager.validate_metadata(valid_metadata), "Valid metadata rejected"
    print("  [PASS] Valid metadata accepted")

    # Test 2: Missing version
    invalid_metadata = {
        "created_at": datetime.now().isoformat()
    }

    assert not manager.validate_metadata(invalid_metadata), "Should reject missing version"
    print("  [PASS] Reject missing version")

    # Test 3: Missing timestamp
    invalid_metadata = {
        "version": "2.0"
    }

    assert not manager.validate_metadata(invalid_metadata), "Should reject missing timestamp"
    print("  [PASS] Reject missing timestamp")

    # Test 4: Invalid timestamp format
    invalid_metadata = {
        "version": "2.0",
        "created_at": "not-a-timestamp"
    }

    assert not manager.validate_metadata(invalid_metadata), "Should reject invalid timestamp"
    print("  [PASS] Reject invalid timestamp")

    print("[SUCCESS] Metadata validation tests passed!")
    print()


def test_convenience_functions():
    """Test convenience functions."""
    print("Testing convenience functions...")

    # Test create_metadata
    metadata = create_metadata(
        training_source="test.wav",
        description="Convenience test"
    )

    assert "version" in metadata, "Convenience function failed"
    print("  [PASS] create_metadata()")

    # Test wrap_with_metadata
    data = {"test": "data"}
    wrapped = wrap_with_metadata(
        data,
        training_source="test.wav"
    )

    assert "metadata" in wrapped, "wrap_with_metadata failed"
    print("  [PASS] wrap_with_metadata()")

    print("[SUCCESS] Convenience function tests passed!")
    print()


def main():
    """Run all tests."""
    print("=" * 60)
    print("METADATA MANAGER TESTS")
    print("=" * 60)
    print()

    try:
        test_metadata_creation()
        test_data_wrapping()
        test_file_migration()
        test_metadata_validation()
        test_convenience_functions()

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
