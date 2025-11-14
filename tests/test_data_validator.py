"""
Test Data Validator
Tests for schema validation and data quality checks
"""

import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from musichal.core.data_safety import DataValidator


def test_schema_validation():
    """Test JSON schema validation."""
    print("Testing DataValidator schema validation...")

    validator = DataValidator()

    # Test 1: Valid training_results data
    valid_data = {
        "training_successful": True,
        "audio_oracle_stats": {
            "total_states": 3001,
            "sequence_length": 3000,
            "is_trained": True,
            "feature_dimensions": 15,
            "distance_function": "euclidean"
        },
        "events_processed": 3000
    }

    errors = validator.validate(valid_data, "training_results_schema")
    assert len(errors) == 0, f"Valid data should have no errors: {errors}"
    print("  [PASS] Valid training_results validation")

    # Test 2: Invalid data (missing required field)
    invalid_data = {
        "training_successful": True
        # Missing audio_oracle_stats
    }

    errors = validator.validate(invalid_data, "training_results_schema")
    assert len(errors) > 0, "Invalid data should have errors"
    assert any("audio_oracle_stats" in e for e in errors), "Should detect missing audio_oracle_stats"
    print("  [PASS] Missing required field detection")

    # Test 3: Type mismatch
    type_mismatch = {
        "training_successful": "yes",  # Should be boolean
        "audio_oracle_stats": {
            "total_states": 3001,
            "sequence_length": 3000,
            "is_trained": True
        }
    }

    errors = validator.validate(type_mismatch, "training_results_schema")
    assert len(errors) > 0, "Type mismatch should be detected"
    assert any("boolean" in e.lower() for e in errors), "Should detect type error"
    print("  [PASS] Type mismatch detection")

    # Test 4: Nested validation
    nested_invalid = {
        "training_successful": True,
        "audio_oracle_stats": {
            "total_states": -100,  # Should be >= 0
            "sequence_length": 3000,
            "is_trained": True
        }
    }

    errors = validator.validate(nested_invalid, "training_results_schema")
    assert len(errors) > 0, "Negative values should be caught"
    assert any("minimum" in e.lower() for e in errors), "Should detect minimum constraint violation"
    print("  [PASS] Nested constraint validation")

    # Test 5: Audio oracle schema
    audio_oracle_data = {
        "distance_threshold": 1.5,
        "distance_function": "euclidean",
        "feature_dimensions": 15,
        "size": 3001,
        "sequence_length": 3000
    }

    errors = validator.validate(audio_oracle_data, "audio_oracle_schema")
    assert len(errors) == 0, f"Valid audio oracle should have no errors: {errors}"
    print("  [PASS] Audio oracle schema validation")

    print("[SUCCESS] Schema validation tests passed!")
    print()


def test_data_quality_checks():
    """Test data quality checking."""
    print("Testing data quality checks...")

    validator = DataValidator()

    # Test 1: Training results with warnings
    data_with_issues = {
        "training_successful": False,  # Failed training
        "audio_oracle_stats": {
            "total_states": 50,  # Very few states
            "sequence_length": 49,
            "is_trained": True,
            "feature_dimensions": 0  # No features!
        }
    }

    warnings = validator.check_data_quality(data_with_issues, "training_results")
    assert len(warnings) > 0, "Should detect quality issues"
    assert any("not successful" in w.lower() for w in warnings), "Should detect failed training"
    assert any("few" in w.lower() for w in warnings), "Should detect insufficient states"
    assert any("feature" in w.lower() for w in warnings), "Should detect missing features"
    print(f"  [PASS] Detected {len(warnings)} quality issues")

    # Test 2: Good quality data
    good_data = {
        "training_successful": True,
        "audio_oracle_stats": {
            "total_states": 3001,
            "sequence_length": 3000,
            "is_trained": True,
            "feature_dimensions": 15
        }
    }

    warnings = validator.check_data_quality(good_data, "training_results")
    assert len(warnings) == 0, f"Good data should have no warnings: {warnings}"
    print("  [PASS] Good quality data has no warnings")

    # Test 3: Rhythm oracle quality
    rhythm_data = {
        "events": [],  # No events
        "tempo": 300  # Too fast
    }

    warnings = validator.check_data_quality(rhythm_data, "rhythm_oracle")
    assert len(warnings) > 0, "Should detect quality issues"
    assert any("events" in w.lower() for w in warnings), "Should detect no events"
    assert any("tempo" in w.lower() for w in warnings), "Should detect unusual tempo"
    print(f"  [PASS] Rhythm oracle quality checks")

    print("[SUCCESS] Data quality checks passed!")
    print()


def test_real_json_file():
    """Test with a real JSON file from the system."""
    print("Testing with real JSON file...")

    validator = DataValidator()

    # Find a training results file
    json_dir = Path("JSON")
    if not json_dir.exists():
        print("  [SKIP] JSON directory not found")
        return

    test_file = json_dir / "Curious_child_041025_2233.json"
    if not test_file.exists():
        print("  [SKIP] Test file not found")
        return

    # Load and validate
    with open(test_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    errors = validator.validate(data, "training_results_schema")
    print(f"  Validation errors: {len(errors)}")
    if errors:
        print(f"    First error: {errors[0]}")

    warnings = validator.check_data_quality(data, "training_results")
    print(f"  Quality warnings: {len(warnings)}")
    if warnings:
        print(f"    First warning: {warnings[0]}")

    print("  [PASS] Real file validation completed")
    print()


def main():
    """Run all tests."""
    print("=" * 60)
    print("DATA VALIDATOR TESTS")
    print("=" * 60)
    print()

    try:
        test_schema_validation()
        test_data_quality_checks()
        test_real_json_file()

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
