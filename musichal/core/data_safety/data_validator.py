"""
Data Validator
Validates data structures against schemas with detailed error reporting.
Part of Phase 1.2: Data Safety Foundation
"""

from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when validation fails."""
    pass


def _validate_against_schema(data: Any, schema: Dict, path: str = "root") -> List[str]:
    """
    Simple JSON schema validator (subset of JSON Schema spec).

    Args:
        data: Data to validate
        schema: JSON schema
        path: Current path in data structure (for error messages)

    Returns:
        List of validation errors
    """
    errors = []

    # Check type
    if "type" in schema:
        expected_type = schema["type"]
        actual_type = type(data).__name__

        type_map = {
            "object": "dict",
            "array": "list",
            "string": "str",
            "number": ["int", "float"],
            "integer": "int",
            "boolean": "bool",
            "null": "NoneType"
        }

        expected = type_map.get(expected_type, expected_type)

        if isinstance(expected, list):
            if actual_type not in expected:
                errors.append(f"{path}: expected {expected_type}, got {actual_type}")
        else:
            if actual_type != expected:
                errors.append(f"{path}: expected {expected_type}, got {actual_type}")
                return errors  # Can't continue validation if type is wrong

    # Validate based on type
    if isinstance(data, dict) and schema.get("type") == "object":
        # Check required properties
        if "required" in schema:
            for required_prop in schema["required"]:
                if required_prop not in data:
                    errors.append(f"{path}: missing required property '{required_prop}'")

        # Validate properties
        if "properties" in schema:
            for prop_name, prop_schema in schema["properties"].items():
                if prop_name in data:
                    prop_errors = _validate_against_schema(
                        data[prop_name],
                        prop_schema,
                        f"{path}.{prop_name}"
                    )
                    errors.extend(prop_errors)

    elif isinstance(data, list) and schema.get("type") == "array":
        # Validate array items
        if "items" in schema:
            item_schema = schema["items"]
            for i, item in enumerate(data):
                item_errors = _validate_against_schema(
                    item,
                    item_schema,
                    f"{path}[{i}]"
                )
                errors.extend(item_errors)

    elif isinstance(data, (int, float)) and schema.get("type") in ["number", "integer"]:
        # Check numeric constraints
        if "minimum" in schema and data < schema["minimum"]:
            errors.append(f"{path}: value {data} is less than minimum {schema['minimum']}")
        if "maximum" in schema and data > schema["maximum"]:
            errors.append(f"{path}: value {data} is greater than maximum {schema['maximum']}")

    elif isinstance(data, str) and schema.get("type") == "string":
        # Check enum
        if "enum" in schema and data not in schema["enum"]:
            errors.append(f"{path}: value '{data}' not in allowed values {schema['enum']}")

    # Handle oneOf (for union types)
    if "oneOf" in schema:
        # Try each schema, if at least one passes, it's valid
        all_errors = []
        for i, sub_schema in enumerate(schema["oneOf"]):
            sub_errors = _validate_against_schema(data, sub_schema, path)
            if not sub_errors:
                # Found valid schema
                return []
            all_errors.extend(sub_errors)
        # None of the schemas matched
        errors.append(f"{path}: data doesn't match any of the oneOf schemas")

    return errors


class DataValidator:
    """
    Validates data structures against JSON schemas.

    Features:
    - Schema-based validation
    - Default value detection
    - Data quality checks
    - Detailed error reporting

    Usage:
        validator = DataValidator()
        validator.validate_audio_oracle(data)
    """

    def __init__(self, schemas_dir: Optional[str | Path] = None):
        """
        Initialize data validator.

        Args:
            schemas_dir: Directory containing JSON schema files
        """
        if schemas_dir is None:
            # __file__ is in musichal/core/data_safety/, go up 3 levels to project root
            schemas_dir = Path(__file__).parent.parent.parent.parent / "schemas"

        self.schemas_dir = Path(schemas_dir)
        self.schemas: Dict[str, Dict] = {}

    def load_schema(self, schema_name: str) -> Dict:
        """
        Load a JSON schema file.

        Args:
            schema_name: Name of schema file (without .json extension)

        Returns:
            Schema dictionary

        Raises:
            FileNotFoundError: If schema file not found
        """
        if schema_name in self.schemas:
            return self.schemas[schema_name]

        schema_path = self.schemas_dir / f"{schema_name}.json"

        if not schema_path.exists():
            raise FileNotFoundError(f"Schema not found: {schema_path}")

        with open(schema_path, 'r', encoding='utf-8') as f:
            schema = json.load(f)

        self.schemas[schema_name] = schema
        return schema

    def validate(
        self,
        data: Any,
        schema_name: str,
        strict: bool = False
    ) -> List[str]:
        """
        Validate data against a schema.

        Args:
            data: Data to validate
            schema_name: Name of schema to use
            strict: If True, raise exception on validation failure

        Returns:
            List of validation errors (empty if valid)

        Raises:
            ValidationError: If strict=True and validation fails
        """
        try:
            schema = self.load_schema(schema_name)
        except FileNotFoundError as e:
            error_msg = f"Schema not found: {schema_name}"
            logger.error(error_msg)
            if strict:
                raise ValidationError(error_msg) from e
            return [error_msg]

        # Validate using schema
        errors = _validate_against_schema(data, schema)

        if strict and errors:
            raise ValidationError(f"Validation failed with {len(errors)} errors: {errors[:3]}")

        return errors

    def validate_audio_oracle(
        self,
        data: Dict,
        strict: bool = False
    ) -> List[str]:
        """
        Validate AudioOracle data structure.

        Args:
            data: AudioOracle data
            strict: If True, raise exception on validation failure

        Returns:
            List of validation errors

        Raises:
            ValidationError: If strict=True and validation fails
        """
        return self.validate(data, "audio_oracle_schema", strict=strict)

    def check_data_quality(
        self,
        data: Dict,
        data_type: str = "audio_oracle"
    ) -> List[str]:
        """
        Check for data quality issues (defaults, missing values, etc.).

        Args:
            data: Data to check
            data_type: Type of data (audio_oracle, rhythm_oracle, etc.)

        Returns:
            List of warnings about data quality
        """
        warnings = []

        if data_type == "audio_oracle":
            warnings.extend(self._check_audio_oracle_quality(data))
        elif data_type == "training_results":
            warnings.extend(self._check_training_results_quality(data))
        elif data_type == "rhythm_oracle":
            warnings.extend(self._check_rhythm_oracle_quality(data))

        return warnings

    def _check_audio_oracle_quality(self, data: Dict) -> List[str]:
        """Check AudioOracle data quality."""
        warnings = []

        # Check if audio_frames exists and has data
        if 'audio_frames' in data:
            frames = data['audio_frames']

            # Handle both list and dict formats
            if isinstance(frames, dict):
                frames = list(frames.values())

            if not frames:
                warnings.append("No audio frames found (empty oracle)")
            elif len(frames) < 10:
                warnings.append(f"Very few audio frames ({len(frames)}), may not be trained properly")

            # Sample frames to check for default values
            if frames and len(frames) > 0:
                sample_size = min(100, len(frames))
                default_count = 0

                for frame in frames[:sample_size]:
                    if isinstance(frame, dict):
                        # Check for suspicious default values
                        if 'audio_data' in frame:
                            audio_data = frame['audio_data']
                            # f0 = 440.0 is often a default
                            if audio_data.get('f0') == 440.0 and audio_data.get('rms_db') == -20.0:
                                default_count += 1

                if default_count > sample_size * 0.5:
                    warnings.append(
                        f"High percentage of default values detected ({default_count}/{sample_size}), "
                        "may indicate feature extraction failure"
                    )

        # Check distance threshold
        if 'distance_threshold' in data:
            threshold = data['distance_threshold']
            if threshold <= 0:
                warnings.append(f"Invalid distance threshold: {threshold} (should be positive)")
            elif threshold > 10:
                warnings.append(f"Very high distance threshold: {threshold} (may match too loosely)")

        return warnings

    def _check_training_results_quality(self, data: Dict) -> List[str]:
        """Check training results data quality."""
        warnings = []

        # Check training success
        if not data.get('training_successful', False):
            warnings.append("Training was not successful")

        # Check oracle stats
        if 'audio_oracle_stats' in data:
            stats = data['audio_oracle_stats']

            total_states = stats.get('total_states', 0)
            sequence_length = stats.get('sequence_length', 0)

            if total_states == 0:
                warnings.append("No oracle states created (oracle is empty)")
            elif total_states < 100:
                warnings.append(f"Very few oracle states ({total_states}), training may be insufficient")

            if sequence_length == 0:
                warnings.append("Sequence length is zero (no data learned)")
            elif total_states != sequence_length + 1:
                warnings.append(
                    f"State count mismatch: {total_states} states for {sequence_length} events "
                    "(expected {sequence_length + 1})"
                )

            # Check if features were extracted
            feature_dims = stats.get('feature_dimensions', 0)
            if feature_dims == 0:
                warnings.append("Feature dimensions is zero (no features extracted)")

        return warnings

    def _check_rhythm_oracle_quality(self, data: Dict) -> List[str]:
        """Check rhythm oracle data quality."""
        warnings = []

        if 'events' in data:
            events = data['events']
            if not events:
                warnings.append("No rhythm events found")
            elif len(events) < 10:
                warnings.append(f"Very few rhythm events ({len(events)})")

        if 'tempo' in data:
            tempo = data['tempo']
            if tempo <= 0:
                warnings.append(f"Invalid tempo: {tempo} (should be positive)")
            elif tempo < 40 or tempo > 240:
                warnings.append(f"Unusual tempo: {tempo} BPM (outside typical range 40-240)")

        return warnings


# Convenience function
def validate_json_file(
    filepath: str | Path,
    schema_name: str,
    strict: bool = False
) -> List[str]:
    """
    Validate a JSON file against a schema.

    Args:
        filepath: Path to JSON file
        schema_name: Name of schema to use
        strict: If True, raise exception on validation failure

    Returns:
        List of validation errors

    Raises:
        ValidationError: If strict=True and validation fails
    """
    filepath = Path(filepath)

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    validator = DataValidator()
    return validator.validate(data, schema_name, strict=strict)
