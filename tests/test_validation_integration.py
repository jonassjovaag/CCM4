"""
Integration test for Pydantic models in validation stage.
Part of Phase 2.3b: Integration Testing
"""

import pytest
from musichal.training.pipeline.stages.validation_stage import ValidationStage


class TestValidationStageIntegration:
    """Test Pydantic model integration in validation stage."""

    def test_validate_oracle_with_good_stats(self):
        """Test validation with valid oracle stats."""
        config = {}
        stage = ValidationStage(config)

        # Valid oracle stats
        stats = {
            'total_states': 1001,
            'total_patterns': 5000,
            'sequence_length': 1000,
            'is_trained': True,
            'distance_threshold': 1.185,
            'distance_function': 'euclidean',
            'feature_dimensions': 15
        }

        result = stage._validate_oracle(oracle=None, stats=stats)

        assert result['is_valid']
        assert len(result['warnings']) == 0
        assert 'oracle_stats_model' in result
        assert result['stats']['total_states'] == 1001

    def test_validate_oracle_with_state_mismatch(self):
        """Test validation with state count mismatch."""
        config = {}
        stage = ValidationStage(config)

        # State count doesn't match sequence
        stats = {
            'total_states': 999,
            'total_patterns': 5000,
            'sequence_length': 1000,
            'is_trained': True,
            'distance_threshold': 1.0,
            'distance_function': 'euclidean',
            'feature_dimensions': 15
        }

        result = stage._validate_oracle(oracle=None, stats=stats)

        # Should still be valid (has states) but has warnings
        assert result['is_valid']
        assert len(result['warnings']) > 0
        assert any('mismatch' in w.lower() for w in result['warnings'])

    def test_validate_oracle_with_no_states(self):
        """Test validation with empty oracle."""
        config = {}
        stage = ValidationStage(config)

        stats = {
            'total_states': 0,
            'total_patterns': 0,
            'sequence_length': 0,
            'is_trained': False,
            'distance_threshold': 1.0,
            'distance_function': 'euclidean',
            'feature_dimensions': 15
        }

        result = stage._validate_oracle(oracle=None, stats=stats)

        assert not result['is_valid']
        assert any('no states' in w.lower() for w in result['warnings'])

    def test_generate_training_results(self):
        """Test generating TrainingResult from context."""
        config = {}
        stage = ValidationStage(config)

        # Create validation report
        validation_report = {
            'is_valid': True,
            'warnings': [],
            'stats': {
                'total_states': 1001,
                'total_patterns': 5000,
                'sequence_length': 1000,
                'is_trained': True,
                'distance_threshold': 1.185,
                'distance_function': 'euclidean',
                'feature_dimensions': 15
            }
        }

        # Create context
        context = {
            'audio_file': 'test.wav',
            'audio_duration': 100.0,
            'training_parameters': {'max_events': 1000},
            'sampled_events': [{'t': i} for i in range(1000)],
            'chord_detections': ['Cmaj7', 'Dm7', None, 'G7']
        }

        result = stage._generate_training_results(context, validation_report)

        # Verify Pydantic-generated structure
        assert result['training_successful']
        assert result['events_processed'] == 1000
        assert result['harmonic_patterns'] == 3  # 3 non-None chords
        assert 'metadata' in result
        assert result['metadata']['audio_file'] == 'test.wav'
        assert result['metadata']['duration_seconds'] == 100.0
        assert result['metadata']['version'] == '2.0'
        assert 'created_at' in result['metadata']
        assert 'python_version' in result['metadata']

    def test_full_validation_stage_execution(self):
        """Test complete validation stage with Pydantic models."""
        config = {}
        stage = ValidationStage(config)

        # Create complete context
        context = {
            'audio_oracle': None,  # Mock oracle
            'oracle_stats': {
                'total_states': 501,
                'total_patterns': 2500,
                'sequence_length': 500,
                'is_trained': True,
                'distance_threshold': 1.0,
                'distance_function': 'euclidean',
                'feature_dimensions': 15
            },
            'audio_file': 'test.wav',
            'audio_duration': 50.0,
            'training_parameters': {'max_events': 500},
            'sampled_events': [{'t': i} for i in range(500)],
            'chord_detections': ['Cmaj7', 'Dm7', 'G7']
        }

        result = stage.execute(context)

        assert result.success
        assert result.stage_name == 'Validation'
        assert 'validation_report' in result.data
        assert 'training_results' in result.data

        training_results = result.data['training_results']
        assert training_results['training_successful']
        assert training_results['events_processed'] == 500
        assert training_results['harmonic_patterns'] == 3
        assert 'metadata' in training_results


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
