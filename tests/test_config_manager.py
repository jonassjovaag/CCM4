"""
Test Configuration Manager
Tests for centralized configuration system
"""

import sys
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from musichal.core.config_manager import ConfigManager, get_config, load_config


def test_load_default_config():
    """Test loading default configuration."""
    print("Testing default config load...")

    config = ConfigManager()
    config.load()

    # Check required sections exist
    assert config.get('system.name') == "MusicHal 9000", "System name missing"
    assert config.get('audio_oracle.distance_threshold') is not None, "Missing distance_threshold"
    assert config.get('memory_buffer.max_duration_seconds') is not None, "Missing max_duration"

    print("  [PASS] Default config loads successfully")
    print(f"  [INFO] System: {config.get('system.name')} v{config.get('system.version')}")
    print(f"  [INFO] Distance threshold: {config.get('audio_oracle.distance_threshold')}")
    print()


def test_load_profile():
    """Test loading configuration profiles."""
    print("Testing profile loading...")

    # Load quick_test profile
    config = ConfigManager()
    config.load(profile='quick_test')

    # Check profile overrides
    max_events = config.get('audio_oracle.training.max_events')
    assert max_events == 1000, f"Profile override failed: expected 1000, got {max_events}"

    # Check default values still present
    assert config.get('system.name') == "MusicHal 9000", "Default values lost"

    print("  [PASS] Profile loading works")
    print(f"  [INFO] Loaded profile: {config.profile}")
    print(f"  [INFO] Max events (overridden): {max_events}")
    print()


def test_dot_notation():
    """Test dot notation access."""
    print("Testing dot notation access...")

    config = ConfigManager()
    config.load()

    # Test nested access
    threshold = config.get('audio_oracle.distance_threshold')
    assert threshold is not None, "Dot notation failed"

    # Test deep nesting
    max_events = config.get('audio_oracle.training.max_events')
    assert max_events == 15000, "Deep nesting failed"

    # Test default value
    missing = config.get('nonexistent.key', 'default')
    assert missing == 'default', "Default value failed"

    print("  [PASS] Dot notation access works")
    print()


def test_set_config():
    """Test setting configuration values."""
    print("Testing config value setting...")

    config = ConfigManager()
    config.load()

    # Set a value
    config.set('audio_oracle.distance_threshold', 0.25)
    assert config.get('audio_oracle.distance_threshold') == 0.25, "Set failed"

    # Set nested value
    config.set('custom.nested.value', 42)
    assert config.get('custom.nested.value') == 42, "Nested set failed"

    print("  [PASS] Setting values works")
    print()


def test_get_section():
    """Test getting configuration sections."""
    print("Testing section retrieval...")

    config = ConfigManager()
    config.load()

    # Get entire section
    oracle_config = config.get_section('audio_oracle')
    assert isinstance(oracle_config, dict), "Section not a dict"
    assert 'distance_threshold' in oracle_config, "Section incomplete"

    print("  [PASS] Section retrieval works")
    print(f"  [INFO] Oracle config keys: {list(oracle_config.keys())[:5]}...")
    print()


def test_overrides():
    """Test configuration overrides."""
    print("Testing overrides...")

    config = ConfigManager()
    overrides = {
        'audio_oracle.distance_threshold': 0.3,
        'memory_buffer.max_duration_seconds': 60.0
    }
    config.load(overrides=overrides)

    assert config.get('audio_oracle.distance_threshold') == 0.3, "Override failed"
    assert config.get('memory_buffer.max_duration_seconds') == 60.0, "Override failed"

    print("  [PASS] Overrides work")
    print()


def test_save_and_load():
    """Test saving and loading configuration."""
    print("Testing save/load...")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        # Create and modify config
        config = ConfigManager()
        config.load()
        config.set('custom.test', 'value')

        # Save
        save_path = temp_dir / "test_config.yaml"
        config.save(save_path)

        assert save_path.exists(), "Save failed"

        # Load saved config
        config2 = ConfigManager()
        config2.load(config_file=save_path)

        assert config2.get('custom.test') == 'value', "Load failed"

    print("  [PASS] Save/load works")
    print()


def test_validation():
    """Test configuration validation."""
    print("Testing validation...")

    config = ConfigManager()
    config.load()

    # Should be valid
    assert config.validate(), "Valid config rejected"

    # Break config
    config.set('audio_oracle.distance_threshold', -1)
    assert not config.validate(), "Invalid config accepted"

    print("  [PASS] Validation works")
    print()


def test_list_profiles():
    """Test listing available profiles."""
    print("Testing profile listing...")

    config = ConfigManager()
    profiles = config.list_profiles()

    assert isinstance(profiles, list), "Profiles not a list"
    assert len(profiles) > 0, "No profiles found"
    assert 'quick_test' in profiles, "quick_test profile missing"
    assert 'live_performance' in profiles, "live_performance profile missing"

    print("  [PASS] Profile listing works")
    print(f"  [INFO] Available profiles: {', '.join(profiles)}")
    print()


def test_singleton():
    """Test singleton instance."""
    print("Testing singleton...")

    # Get config twice
    config1 = get_config()
    config2 = get_config()

    # Should be same instance
    assert config1 is config2, "Singleton broken"

    print("  [PASS] Singleton works")
    print()


def test_profile_inheritance():
    """Test that profiles properly inherit from default."""
    print("Testing profile inheritance...")

    config = ConfigManager()
    config.load(profile='quick_test')

    # Check overridden value
    assert config.get('audio_oracle.training.max_events') == 1000, "Override failed"

    # Check inherited value
    assert config.get('audio_oracle.distance_function') == 'euclidean', "Inheritance failed"
    assert config.get('system.name') == 'MusicHal 9000', "Base values lost"

    print("  [PASS] Profile inheritance works")
    print()


def main():
    """Run all tests."""
    print("=" * 60)
    print("CONFIGURATION MANAGER TESTS")
    print("=" * 60)
    print()

    try:
        test_load_default_config()
        test_load_profile()
        test_dot_notation()
        test_set_config()
        test_get_section()
        test_overrides()
        test_save_and_load()
        test_validation()
        test_list_profiles()
        test_singleton()
        test_profile_inheritance()

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
