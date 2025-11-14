#!/usr/bin/env python3
"""
Data Integrity Audit Tool
Part of Phase 0.2: Pre-Flight Safety

Validates all JSON files in the system for:
- Parse-ability
- Required field presence
- Data quality issues
- Default value detection (indicators of missing data)
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class ValidationResult:
    """Result of validating a single file."""
    filepath: str
    is_valid: bool
    file_size: int
    errors: List[str]
    warnings: List[str]
    stats: Dict[str, Any]


class DataIntegrityAuditor:
    """Audits JSON training data for integrity issues."""

    def __init__(self):
        self.results: List[ValidationResult] = []
        self.total_files = 0
        self.valid_files = 0
        self.invalid_files = 0

    def audit_file(self, filepath: Path) -> ValidationResult:
        """Audit a single JSON file."""
        errors = []
        warnings = []
        stats = {}
        is_valid = True

        file_size = filepath.stat().st_size

        # Check 1: Can we parse the JSON?
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            errors.append(f"JSON parse error: {e}")
            return ValidationResult(
                filepath=str(filepath),
                is_valid=False,
                file_size=file_size,
                errors=errors,
                warnings=warnings,
                stats=stats
            )
        except Exception as e:
            errors.append(f"Failed to read file: {e}")
            return ValidationResult(
                filepath=str(filepath),
                is_valid=False,
                file_size=file_size,
                errors=errors,
                warnings=warnings,
                stats=stats
            )

        # Check 2: Identify file type and validate structure
        file_type = self._identify_file_type(data, filepath.name)
        stats['file_type'] = file_type

        if file_type == 'audio_oracle':
            errors.extend(self._validate_audio_oracle(data))
            warnings.extend(self._check_audio_oracle_quality(data))
            stats.update(self._get_audio_oracle_stats(data))

        elif file_type == 'training_results':
            errors.extend(self._validate_training_results(data))
            stats.update(self._get_training_results_stats(data))

        elif file_type == 'rhythm_oracle':
            errors.extend(self._validate_rhythm_oracle(data))
            warnings.extend(self._check_rhythm_oracle_quality(data))
            stats.update(self._get_rhythm_oracle_stats(data))

        elif file_type == 'correlation':
            errors.extend(self._validate_correlation(data))
            stats.update(self._get_correlation_stats(data))

        elif file_type == 'harmonic_transitions':
            errors.extend(self._validate_harmonic_transitions(data))
            stats.update(self._get_harmonic_transitions_stats(data))

        else:
            warnings.append(f"Unknown file type: {file_type}")

        # Determine if valid
        is_valid = len(errors) == 0

        return ValidationResult(
            filepath=str(filepath),
            is_valid=is_valid,
            file_size=file_size,
            errors=errors,
            warnings=warnings,
            stats=stats
        )

    def _identify_file_type(self, data: Dict, filename: str) -> str:
        """Identify the type of training data file."""
        if 'rhythm_oracle' in filename.lower():
            return 'rhythm_oracle'
        elif 'harmonic_transitions' in filename.lower():
            return 'harmonic_transitions'
        elif 'correlation' in filename.lower():
            return 'correlation'
        elif 'audio_frames' in data:
            return 'audio_oracle'
        elif 'training_successful' in data and 'audio_oracle_stats' in data:
            # New format: training results without audio_frames
            return 'training_results'
        elif 'audio_oracle_stats' in data:
            return 'audio_oracle'
        else:
            return 'unknown'

    def _validate_audio_oracle(self, data: Dict) -> List[str]:
        """Validate AudioOracle structure."""
        errors = []

        # audio_frames is optional (some formats don't include it)
        # but if audio_oracle_stats exists, validate it
        if 'audio_oracle_stats' in data:
            stats = data['audio_oracle_stats']
            if 'total_states' not in stats:
                errors.append("Missing 'total_states' in audio_oracle_stats")
            if 'sequence_length' not in stats:
                errors.append("Missing 'sequence_length' in audio_oracle_stats")

        return errors

    def _validate_training_results(self, data: Dict) -> List[str]:
        """Validate training results structure."""
        errors = []

        # Check for training_successful field
        if 'training_successful' not in data:
            errors.append("Missing 'training_successful' field")
        elif not isinstance(data['training_successful'], bool):
            errors.append("'training_successful' must be a boolean")

        # Check for audio_oracle_stats
        if 'audio_oracle_stats' not in data:
            errors.append("Missing 'audio_oracle_stats' field")

        return errors

    def _get_training_results_stats(self, data: Dict) -> Dict:
        """Get statistics from training results."""
        stats = {}

        stats['training_successful'] = data.get('training_successful', False)

        if 'audio_oracle_stats' in data:
            oracle_stats = data['audio_oracle_stats']
            stats['total_states'] = oracle_stats.get('total_states', 0)
            stats['total_patterns'] = oracle_stats.get('total_patterns', 0)
            stats['sequence_length'] = oracle_stats.get('sequence_length', 0)

        if 'events_processed' in data:
            stats['events_processed'] = data['events_processed']

        return stats

    def _check_audio_oracle_quality(self, data: Dict) -> List[str]:
        """Check AudioOracle data quality."""
        warnings = []

        if 'audio_frames' not in data:
            return warnings

        frames = data['audio_frames']

        # Handle both list and dict formats
        if isinstance(frames, dict):
            frames = list(frames.values())

        if not frames:
            warnings.append("No audio frames found (empty training data)")
            return warnings

        # Sample first 100 frames to check for quality issues
        sample_size = min(100, len(frames))
        default_f0_count = 0
        default_rms_count = 0
        missing_features_count = 0
        missing_gesture_token_count = 0

        for i, frame in enumerate(frames[:sample_size]):
            if 'audio_data' not in frame:
                missing_features_count += 1
                continue

            audio_data = frame['audio_data']

            # Check for default f0 value (440.0 Hz)
            if audio_data.get('f0') == 440.0:
                default_f0_count += 1

            # Check for default RMS value (-20.0 dB)
            if audio_data.get('rms_db') == -20.0:
                default_rms_count += 1

            # Check for missing gesture_token
            if 'gesture_token' not in audio_data:
                missing_gesture_token_count += 1

        # Calculate percentages
        default_f0_pct = (default_f0_count / sample_size) * 100
        default_rms_pct = (default_rms_count / sample_size) * 100
        missing_features_pct = (missing_features_count / sample_size) * 100
        missing_token_pct = (missing_gesture_token_count / sample_size) * 100

        if default_f0_pct > 50:
            warnings.append(f"High default f0 values: {default_f0_pct:.1f}% (may indicate feature extraction failure)")

        if default_rms_pct > 50:
            warnings.append(f"High default RMS values: {default_rms_pct:.1f}% (may indicate feature extraction failure)")

        if missing_features_pct > 5:
            warnings.append(f"Missing audio_data in {missing_features_pct:.1f}% of frames")

        if missing_token_pct > 5:
            warnings.append(f"Missing gesture_token in {missing_token_pct:.1f}% of frames")

        return warnings

    def _get_audio_oracle_stats(self, data: Dict) -> Dict:
        """Get statistics from AudioOracle data."""
        stats = {}

        if 'audio_oracle_stats' in data:
            oracle_stats = data['audio_oracle_stats']
            stats['total_states'] = oracle_stats.get('total_states', 0)
            stats['total_patterns'] = oracle_stats.get('total_patterns', 0)
            stats['sequence_length'] = oracle_stats.get('sequence_length', 0)

        if 'audio_frames' in data:
            frames = data['audio_frames']
            # Handle both list and dict formats
            if isinstance(frames, dict):
                frames = list(frames.values())
            stats['total_frames'] = len(frames)

            # Calculate timestamp range
            if frames:
                timestamps = [f.get('timestamp', 0) for f in frames if 'timestamp' in f]
                if timestamps:
                    stats['min_timestamp'] = min(timestamps)
                    stats['max_timestamp'] = max(timestamps)
                    stats['duration_seconds'] = max(timestamps) - min(timestamps)

        return stats

    def _validate_rhythm_oracle(self, data: Dict) -> List[str]:
        """Validate RhythmOracle structure."""
        errors = []

        # Basic structure check
        if not isinstance(data, dict):
            errors.append("RhythmOracle data is not a dictionary")

        return errors

    def _check_rhythm_oracle_quality(self, data: Dict) -> List[str]:
        """Check RhythmOracle data quality."""
        warnings = []

        if 'events' in data:
            if not data['events']:
                warnings.append("No rhythm events found")

        return warnings

    def _get_rhythm_oracle_stats(self, data: Dict) -> Dict:
        """Get statistics from RhythmOracle data."""
        stats = {}

        if 'events' in data:
            stats['total_events'] = len(data['events'])

        if 'tempo' in data:
            stats['tempo'] = data['tempo']

        return stats

    def _validate_correlation(self, data: Dict) -> List[str]:
        """Validate correlation data structure."""
        errors = []

        if not isinstance(data, dict):
            errors.append("Correlation data is not a dictionary")

        return errors

    def _get_correlation_stats(self, data: Dict) -> Dict:
        """Get statistics from correlation data."""
        stats = {}
        stats['keys_count'] = len(data.keys()) if isinstance(data, dict) else 0
        return stats

    def _validate_harmonic_transitions(self, data: Dict) -> List[str]:
        """Validate harmonic transitions structure."""
        errors = []

        if not isinstance(data, dict):
            errors.append("Harmonic transitions data is not a dictionary")

        return errors

    def _get_harmonic_transitions_stats(self, data: Dict) -> Dict:
        """Get statistics from harmonic transitions data."""
        stats = {}
        stats['transitions_count'] = len(data.keys()) if isinstance(data, dict) else 0
        return stats

    def audit_directory(self, directory: Path) -> None:
        """Audit all JSON files in a directory."""
        json_files = sorted(directory.glob("*.json"))
        self.total_files = len(json_files)

        print(f"Auditing {self.total_files} JSON files in {directory}...")
        print()

        for i, filepath in enumerate(json_files, 1):
            result = self.audit_file(filepath)
            self.results.append(result)

            if result.is_valid:
                self.valid_files += 1
            else:
                self.invalid_files += 1

            # Progress indicator
            if i % 20 == 0 or i == self.total_files:
                print(f"Progress: {i}/{self.total_files} files audited")

    def generate_report(self) -> str:
        """Generate a comprehensive audit report."""
        report_lines = []

        report_lines.append("=" * 80)
        report_lines.append("DATA INTEGRITY AUDIT REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")

        # Summary
        report_lines.append("SUMMARY")
        report_lines.append("-" * 80)
        report_lines.append(f"Total files audited:  {self.total_files}")
        report_lines.append(f"Valid files:          {self.valid_files} ({self.valid_files/self.total_files*100:.1f}%)")
        report_lines.append(f"Invalid files:        {self.invalid_files} ({self.invalid_files/self.total_files*100:.1f}%)")
        report_lines.append("")

        # File type breakdown
        file_types = defaultdict(int)
        for result in self.results:
            file_type = result.stats.get('file_type', 'unknown')
            file_types[file_type] += 1

        report_lines.append("FILE TYPES")
        report_lines.append("-" * 80)
        for file_type, count in sorted(file_types.items()):
            report_lines.append(f"  {file_type:25s} {count:5d} files")
        report_lines.append("")

        # Invalid files (if any)
        if self.invalid_files > 0:
            report_lines.append("INVALID FILES (ERRORS)")
            report_lines.append("-" * 80)
            for result in self.results:
                if not result.is_valid:
                    report_lines.append(f"\nFile: {result.filepath}")
                    report_lines.append(f"  Size: {result.file_size:,} bytes")
                    for error in result.errors:
                        report_lines.append(f"  ERROR: {error}")
            report_lines.append("")

        # Warnings
        files_with_warnings = [r for r in self.results if r.warnings]
        if files_with_warnings:
            report_lines.append("FILES WITH WARNINGS (DATA QUALITY ISSUES)")
            report_lines.append("-" * 80)
            report_lines.append(f"Total: {len(files_with_warnings)} files")
            report_lines.append("")

            for result in files_with_warnings[:10]:  # Show first 10
                report_lines.append(f"\nFile: {Path(result.filepath).name}")
                for warning in result.warnings:
                    report_lines.append(f"  WARNING: {warning}")

            if len(files_with_warnings) > 10:
                report_lines.append(f"\n... and {len(files_with_warnings) - 10} more files with warnings")
            report_lines.append("")

        # Statistics summary
        report_lines.append("STATISTICS SUMMARY")
        report_lines.append("-" * 80)

        total_frames = sum(r.stats.get('total_frames', 0) for r in self.results)
        total_states = sum(r.stats.get('total_states', 0) for r in self.results)
        total_patterns = sum(r.stats.get('total_patterns', 0) for r in self.results)

        report_lines.append(f"Total audio frames:   {total_frames:,}")
        report_lines.append(f"Total oracle states:  {total_states:,}")
        report_lines.append(f"Total patterns:       {total_patterns:,}")
        report_lines.append("")

        report_lines.append("=" * 80)

        return "\n".join(report_lines)


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python data_integrity_audit.py <json_directory>")
        print("\nExample:")
        print("  python data_integrity_audit.py JSON")
        print("  python data_integrity_audit.py backups/pre_refactor_20251113_124720/JSON")
        sys.exit(1)

    directory = Path(sys.argv[1])

    if not directory.exists():
        print(f"Error: Directory not found: {directory}")
        sys.exit(1)

    if not directory.is_dir():
        print(f"Error: Not a directory: {directory}")
        sys.exit(1)

    # Run audit
    auditor = DataIntegrityAuditor()
    auditor.audit_directory(directory)

    # Generate and print report
    print("\n")
    report = auditor.generate_report()
    print(report)

    # Save report to file
    report_file = Path("DATA_AUDIT_REPORT.md")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"# Data Integrity Audit Report\n\n")
        f.write(f"**Directory:** `{directory}`\n\n")
        f.write(f"**Date:** {Path('backups').glob('pre_refactor_*')}\n\n")
        f.write("```\n")
        f.write(report)
        f.write("\n```\n")

    print(f"\nReport saved to: {report_file}")

    # Exit with error code if any files are invalid
    if auditor.invalid_files > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
