"""
Configuration Manager
Centralized configuration management with profile support.
Part of Phase 2.1: Code Organization
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from copy import deepcopy

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing."""
    pass


class ConfigManager:
    """
    Manages application configuration with profile support.

    Features:
    - Load from YAML files
    - Profile inheritance (e.g., quick_test inherits from default)
    - CLI override support
    - Dot notation access (e.g., config.get('audio_oracle.distance_threshold'))
    - Validation
    - Environment variable substitution

    Usage:
        config = ConfigManager()
        config.load()  # Loads default config

        # Or load a specific profile
        config.load(profile='quick_test')

        # Access values
        threshold = config.get('audio_oracle.distance_threshold')

        # Override at runtime
        config.set('audio_oracle.distance_threshold', 0.2)
    """

    def __init__(self, config_dir: Optional[str | Path] = None):
        """
        Initialize configuration manager.

        Args:
            config_dir: Directory containing config files (default: ./config)
        """
        if config_dir is None:
            # Assume config/ directory in project root
            # __file__ is in musichal/core/, so go up 2 levels to project root
            config_dir = Path(__file__).parent.parent.parent / "config"

        self.config_dir = Path(config_dir)
        self.profiles_dir = self.config_dir / "profiles"

        self._config: Dict[str, Any] = {}
        self._profile: Optional[str] = None

    def load(
        self,
        profile: Optional[str] = None,
        config_file: Optional[str | Path] = None,
        overrides: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Load configuration from file(s).

        Args:
            profile: Profile name (e.g., 'quick_test', 'live_performance')
            config_file: Custom config file path (overrides profile)
            overrides: Dictionary of overrides to apply

        Raises:
            ConfigurationError: If config file not found or invalid
        """
        # Step 1: Load default config
        default_file = self.config_dir / "default_config.yaml"
        if not default_file.exists():
            raise ConfigurationError(f"Default config not found: {default_file}")

        self._config = self._load_yaml(default_file)
        logger.info(f"Loaded default configuration from {default_file}")

        # Step 2: Load profile if specified
        if profile:
            profile_file = self.profiles_dir / f"{profile}.yaml"
            if not profile_file.exists():
                raise ConfigurationError(f"Profile not found: {profile_file}")

            profile_config = self._load_yaml(profile_file)
            self._merge_configs(self._config, profile_config)
            self._profile = profile
            logger.info(f"Applied profile: {profile}")

        # Step 3: Load custom config file if specified
        if config_file:
            config_file = Path(config_file)
            if not config_file.exists():
                raise ConfigurationError(f"Config file not found: {config_file}")

            custom_config = self._load_yaml(config_file)
            self._merge_configs(self._config, custom_config)
            logger.info(f"Loaded custom config from {config_file}")

        # Step 4: Apply overrides
        if overrides:
            self._apply_overrides(overrides)
            logger.info(f"Applied {len(overrides)} overrides")

    def _load_yaml(self, filepath: Path) -> Dict[str, Any]:
        """Load and parse YAML file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            return data or {}
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Failed to parse YAML: {e}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load config: {e}")

    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> None:
        """
        Recursively merge override config into base config.

        Args:
            base: Base configuration (modified in place)
            override: Override configuration
        """
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                # Recursively merge nested dicts
                self._merge_configs(base[key], value)
            else:
                # Override value
                base[key] = deepcopy(value)

    def _apply_overrides(self, overrides: Dict[str, Any]) -> None:
        """
        Apply override values using dot notation.

        Args:
            overrides: Dictionary of overrides (e.g., {'audio_oracle.distance_threshold': 0.2})
        """
        for key, value in overrides.items():
            self.set(key, value)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            key: Configuration key (e.g., 'audio_oracle.distance_threshold')
            default: Default value if key not found

        Returns:
            Configuration value

        Examples:
            >>> config.get('audio_oracle.distance_threshold')
            0.15
            >>> config.get('audio.buffer_size', 512)
            512
        """
        parts = key.split('.')
        value = self._config

        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value using dot notation.

        Args:
            key: Configuration key (e.g., 'audio_oracle.distance_threshold')
            value: Value to set

        Examples:
            >>> config.set('audio_oracle.distance_threshold', 0.2)
        """
        parts = key.split('.')
        current = self._config

        # Navigate to the nested dict
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        # Set the value
        current[parts[-1]] = value
        logger.debug(f"Set {key} = {value}")

    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get entire configuration section.

        Args:
            section: Section name (e.g., 'audio_oracle')

        Returns:
            Section configuration dictionary

        Examples:
            >>> oracle_config = config.get_section('audio_oracle')
            >>> print(oracle_config['distance_threshold'])
            0.15
        """
        return self.get(section, {})

    def to_dict(self) -> Dict[str, Any]:
        """
        Get complete configuration as dictionary.

        Returns:
            Full configuration dictionary
        """
        return deepcopy(self._config)

    def save(self, filepath: str | Path) -> None:
        """
        Save current configuration to YAML file.

        Args:
            filepath: Output file path
        """
        filepath = Path(filepath)

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)
            logger.info(f"Saved configuration to {filepath}")
        except Exception as e:
            raise ConfigurationError(f"Failed to save config: {e}")

    def validate(self) -> bool:
        """
        Validate configuration.

        Returns:
            True if valid, False otherwise
        """
        # Basic validation
        required_sections = [
            'system',
            'audio_oracle',
            'memory_buffer',
            'ai_agent',
            'midi',
            'persistence'
        ]

        for section in required_sections:
            if section not in self._config:
                logger.error(f"Missing required section: {section}")
                return False

        # Validate specific values
        distance_threshold = self.get('audio_oracle.distance_threshold')
        if distance_threshold is not None and (distance_threshold <= 0 or distance_threshold > 10):
            logger.error(f"Invalid distance_threshold: {distance_threshold}")
            return False

        return True

    def list_profiles(self) -> list[str]:
        """
        List available configuration profiles.

        Returns:
            List of profile names
        """
        if not self.profiles_dir.exists():
            return []

        profiles = []
        for file in self.profiles_dir.glob("*.yaml"):
            profiles.append(file.stem)

        return sorted(profiles)

    @property
    def profile(self) -> Optional[str]:
        """Get current profile name."""
        return self._profile


# Singleton instance for convenience
_default_config = None


def get_config() -> ConfigManager:
    """
    Get global configuration instance (singleton).

    Returns:
        ConfigManager instance

    Usage:
        from core.config_manager import get_config
        config = get_config()
        threshold = config.get('audio_oracle.distance_threshold')
    """
    global _default_config
    if _default_config is None:
        _default_config = ConfigManager()
        _default_config.load()  # Load default config
    return _default_config


def load_config(profile: Optional[str] = None, **kwargs) -> ConfigManager:
    """
    Load configuration with optional profile.

    Args:
        profile: Profile name
        **kwargs: Additional arguments for load()

    Returns:
        ConfigManager instance

    Usage:
        from core.config_manager import load_config
        config = load_config(profile='quick_test')
    """
    global _default_config
    _default_config = ConfigManager()
    _default_config.load(profile=profile, **kwargs)
    return _default_config
