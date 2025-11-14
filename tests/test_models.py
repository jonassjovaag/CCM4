"""
Tests for Pydantic Data Models
Part of Phase 2.3: Standardized Data Models
"""

import pytest
from datetime import datetime
from musichal.core.models.audio_event import AudioEvent, AudioEventFeatures
from musichal.core.models.musical_moment import MusicalMoment
from musichal.core.models.oracle_state import (
    AudioOracleStats, OracleState, RhythmOracleStats, DistanceFunction
)
from musichal.core.models.training_result import TrainingResult, TrainingMetadata
from musichal.core.models.performance_context import (
    PerformanceContext, DecisionContext, BehaviorMode
)


class TestAudioEventFeatures:
    """Test AudioEventFeatures validation."""

    def test_valid_features(self):
        """Test creating valid features."""
        features = AudioEventFeatures(
            f0=440.0,
            rms_db=-30.0,
            centroid=2000.0,
            cents=0.0,
            ioi=0.5,
            consonance=0.75,
            gesture_token=42
        )
        assert features.f0 == 440.0
        assert features.rms_db == -30.0
        assert features.gesture_token == 42

    def test_f0_validation(self):
        """Test f0 range validation."""
        # Valid range
        AudioEventFeatures(f0=100.0, rms_db=-20.0)
        AudioEventFeatures(f0=5000.0, rms_db=-20.0)

        # Invalid: negative
        with pytest.raises(ValueError):
            AudioEventFeatures(f0=-10.0, rms_db=-20.0)

        # Invalid: too high
        with pytest.raises(ValueError):
            AudioEventFeatures(f0=20000.0, rms_db=-20.0)

    def test_rms_validation(self):
        """Test RMS dB range validation."""
        # Valid range
        AudioEventFeatures(f0=440.0, rms_db=-60.0)
        AudioEventFeatures(f0=440.0, rms_db=0.0)

        # Invalid: positive
        with pytest.raises(ValueError):
            AudioEventFeatures(f0=440.0, rms_db=10.0)

        # Invalid: too low
        with pytest.raises(ValueError):
            AudioEventFeatures(f0=440.0, rms_db=-150.0)

    def test_wav2vec_features_length(self):
        """Test Wav2Vec features must be 768D."""
        # Valid: 768 dimensions
        features = AudioEventFeatures(
            f0=440.0,
            rms_db=-20.0,
            wav2vec_features=[0.1] * 768
        )
        assert len(features.wav2vec_features) == 768

        # Invalid: wrong length
        with pytest.raises(ValueError):
            AudioEventFeatures(
                f0=440.0,
                rms_db=-20.0,
                wav2vec_features=[0.1] * 100
            )

    def test_gesture_token_range(self):
        """Test gesture token must be 0-63."""
        # Valid range
        AudioEventFeatures(f0=440.0, rms_db=-20.0, gesture_token=0)
        AudioEventFeatures(f0=440.0, rms_db=-20.0, gesture_token=63)

        # Invalid: negative
        with pytest.raises(ValueError):
            AudioEventFeatures(f0=440.0, rms_db=-20.0, gesture_token=-1)

        # Invalid: too high
        with pytest.raises(ValueError):
            AudioEventFeatures(f0=440.0, rms_db=-20.0, gesture_token=64)


class TestAudioEvent:
    """Test AudioEvent validation."""

    def test_valid_event(self):
        """Test creating valid audio event."""
        features = AudioEventFeatures(f0=440.0, rms_db=-30.0)
        event = AudioEvent(
            timestamp=1.5,
            features=features,
            chord='Cmaj7',
            key='C major'
        )
        assert event.timestamp == 1.5
        assert event.chord == 'Cmaj7'

    def test_timestamp_validation(self):
        """Test timestamp must be reasonable."""
        features = AudioEventFeatures(f0=440.0, rms_db=-30.0)

        # Valid timestamps
        AudioEvent(timestamp=0.0, features=features)
        AudioEvent(timestamp=300.0, features=features)

        # Invalid: negative
        with pytest.raises(ValueError):
            AudioEvent(timestamp=-1.0, features=features)

        # Invalid: too large (>1 hour)
        with pytest.raises(ValueError):
            AudioEvent(timestamp=4000.0, features=features)

    def test_to_dict(self):
        """Test JSON serialization."""
        features = AudioEventFeatures(f0=440.0, rms_db=-30.0)
        event = AudioEvent(timestamp=1.5, features=features, chord='Cmaj7')

        data = event.to_dict()
        assert data['timestamp'] == 1.5
        assert data['features']['f0'] == 440.0
        assert data['chord'] == 'Cmaj7'
        assert 'key' not in data  # exclude_none=True


class TestMusicalMoment:
    """Test MusicalMoment validation."""

    def test_valid_moment(self):
        """Test creating valid musical moment."""
        features = AudioEventFeatures(f0=440.0, rms_db=-30.0)
        event = AudioEvent(timestamp=1.5, features=features)
        moment = MusicalMoment(
            timestamp=1.5,
            event=event,
            normalized_features=[0.5, -0.2, 0.1, 0.3],
            cluster_id=12,
            significance=0.85
        )
        assert moment.cluster_id == 12
        assert moment.significance == 0.85

    def test_normalized_features_length(self):
        """Test normalized features must be 4D."""
        features = AudioEventFeatures(f0=440.0, rms_db=-30.0)
        event = AudioEvent(timestamp=1.5, features=features)

        # Valid: 4 dimensions
        MusicalMoment(
            timestamp=1.5,
            event=event,
            normalized_features=[0.1, 0.2, 0.3, 0.4]
        )

        # Invalid: wrong length
        with pytest.raises(ValueError):
            MusicalMoment(
                timestamp=1.5,
                event=event,
                normalized_features=[0.1, 0.2]
            )


class TestAudioOracleStats:
    """Test AudioOracleStats validation."""

    def test_valid_stats(self):
        """Test creating valid oracle stats."""
        stats = AudioOracleStats(
            total_states=3001,
            total_patterns=23868,
            sequence_length=3000,
            is_trained=True,
            distance_threshold=1.185,
            distance_function=DistanceFunction.EUCLIDEAN,
            feature_dimensions=15
        )
        assert stats.total_states == 3001
        assert stats.distance_function == DistanceFunction.EUCLIDEAN

    def test_validate_state_count(self):
        """Test state count validation."""
        # Valid: states = sequence + 1
        stats = AudioOracleStats(
            total_states=1001,
            total_patterns=5000,
            sequence_length=1000,
            is_trained=True,
            distance_threshold=1.0,
            distance_function=DistanceFunction.EUCLIDEAN,
            feature_dimensions=15
        )
        is_valid, warnings = stats.validate_state_count()
        assert is_valid
        assert len(warnings) == 0

        # Invalid: state count mismatch
        stats.total_states = 999
        is_valid, warnings = stats.validate_state_count()
        assert is_valid  # Still valid (has states), just has warnings
        assert len(warnings) == 1
        assert 'mismatch' in warnings[0].lower()

    def test_no_states_warning(self):
        """Test warning for empty oracle."""
        stats = AudioOracleStats(
            total_states=0,
            total_patterns=0,
            sequence_length=0,
            is_trained=False,
            distance_threshold=1.0,
            distance_function=DistanceFunction.EUCLIDEAN,
            feature_dimensions=15
        )
        is_valid, warnings = stats.validate_state_count()
        assert not is_valid  # Invalid because no states
        assert any('no states' in w.lower() for w in warnings)


class TestTrainingResult:
    """Test TrainingResult validation."""

    def test_valid_result(self):
        """Test creating valid training result."""
        metadata = TrainingMetadata(
            audio_file='test.wav',
            duration_seconds=245.3,
            parameters={'max_events': 15000}
        )

        oracle_stats = AudioOracleStats(
            total_states=3001,
            total_patterns=23868,
            sequence_length=3000,
            is_trained=True,
            distance_threshold=1.185,
            distance_function=DistanceFunction.EUCLIDEAN,
            feature_dimensions=15
        )

        result = TrainingResult(
            metadata=metadata,
            training_successful=True,
            events_processed=3000,
            audio_oracle_stats=oracle_stats
        )

        assert result.training_successful
        assert result.events_processed == 3000

    def test_validate_training(self):
        """Test training validation."""
        metadata = TrainingMetadata(
            audio_file='test.wav',
            duration_seconds=100.0,
            parameters={}
        )

        oracle_stats = AudioOracleStats(
            total_states=1001,
            total_patterns=5000,
            sequence_length=1000,
            is_trained=True,
            distance_threshold=1.0,
            distance_function=DistanceFunction.EUCLIDEAN,
            feature_dimensions=15
        )

        # Valid training
        result = TrainingResult(
            metadata=metadata,
            training_successful=True,
            events_processed=1000,
            audio_oracle_stats=oracle_stats
        )
        is_valid, issues = result.validate_training()
        assert is_valid
        assert len(issues) == 0

        # Failed training
        result.training_successful = False
        is_valid, issues = result.validate_training()
        assert not is_valid
        assert any('not successful' in i.lower() for i in issues)


class TestPerformanceContext:
    """Test PerformanceContext validation."""

    def test_valid_context(self):
        """Test creating valid performance context."""
        context = PerformanceContext(
            current_time=45.5,
            behavior_mode=BehaviorMode.IMITATE,
            is_musician_playing=True,
            time_since_last_event=0.5,
            current_density=2.5,
            performance_progress=0.35,
            arc_intensity=0.7
        )
        assert context.behavior_mode == BehaviorMode.IMITATE
        assert context.current_density == 2.5

    def test_should_play_autonomous(self):
        """Test autonomous play decision."""
        context = PerformanceContext(
            current_time=10.0,
            behavior_mode=BehaviorMode.AUTONOMOUS,
            is_musician_playing=False,
            time_since_last_event=3.0,
            current_density=0.0
        )

        # Should play (silence > threshold)
        assert context.should_play_autonomous(silence_threshold=2.0)

        # Should not play (musician playing)
        context.is_musician_playing = True
        assert not context.should_play_autonomous()

        # Should not play (not enough silence)
        context.is_musician_playing = False
        context.time_since_last_event = 1.0
        assert not context.should_play_autonomous(silence_threshold=2.0)

    def test_density_validation(self):
        """Test density range validation."""
        # Valid range
        PerformanceContext(
            current_time=10.0,
            behavior_mode=BehaviorMode.IMITATE,
            is_musician_playing=True,
            time_since_last_event=0.5,
            current_density=5.0
        )

        # Invalid: negative
        with pytest.raises(ValueError):
            PerformanceContext(
                current_time=10.0,
                behavior_mode=BehaviorMode.IMITATE,
                is_musician_playing=True,
                time_since_last_event=0.5,
                current_density=-1.0
            )

        # Invalid: too high
        with pytest.raises(ValueError):
            PerformanceContext(
                current_time=10.0,
                behavior_mode=BehaviorMode.IMITATE,
                is_musician_playing=True,
                time_since_last_event=0.5,
                current_density=15.0
            )


class TestDecisionContext:
    """Test DecisionContext validation."""

    def test_valid_decision_context(self):
        """Test creating valid decision context."""
        perf_context = PerformanceContext(
            current_time=10.0,
            behavior_mode=BehaviorMode.IMITATE,
            is_musician_playing=True,
            time_since_last_event=0.5,
            current_density=2.5
        )

        decision_context = DecisionContext(
            performance_context=perf_context,
            similarity_score=0.85
        )

        assert decision_context.similarity_score == 0.85
        assert decision_context.performance_context.behavior_mode == BehaviorMode.IMITATE


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
