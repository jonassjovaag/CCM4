#!/usr/bin/env python3
"""
Predictive Processing Engine
Based on Schaefer (2014): "Mental Representations in Musical Processing and their Role in Action-Perception Loops"

This module implements predictive processing that mirrors human musical cognition:
- Multi-level predictions (measure, phrase, section)
- Prediction error calculation and learning
- Mental model updates
- Action-perception loops
"""

import numpy as np
import librosa
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import torch
import torch.nn as nn

@dataclass
class MusicalPrediction:
    """Represents a musical prediction at a specific level"""
    prediction_id: str
    level: str  # 'measure', 'phrase', 'section'
    time_horizon: float  # How far ahead we're predicting
    predicted_features: np.ndarray
    confidence: float
    prediction_error: float
    context_features: np.ndarray

@dataclass
class MentalModel:
    """Represents a mental model of musical structure"""
    model_id: str
    level: str
    learned_patterns: List[np.ndarray]
    transition_probabilities: Dict[str, float]
    adaptation_rate: float
    last_update: float

class PredictiveProcessingEngine:
    """
    Implements predictive processing for musical understanding
    
    Based on research showing that humans constantly predict incoming musical events
    and learn from prediction errors to update their mental models.
    """
    
    def __init__(self, 
                 prediction_horizons: Dict[str, float] = None,
                 learning_rate: float = 0.1,
                 error_threshold: float = 0.3):
        
        if prediction_horizons is None:
            self.prediction_horizons = {
                'measure': 6.0,    # 6 seconds ahead
                'phrase': 30.0,    # 30 seconds ahead
                'section': 120.0   # 2 minutes ahead
            }
        else:
            self.prediction_horizons = prediction_horizons
            
        self.learning_rate = learning_rate
        self.error_threshold = error_threshold
        
        # Mental models for different levels
        self.mental_models = {
            'measure': MentalModel(
                model_id='measure_model',
                level='measure',
                learned_patterns=[],
                transition_probabilities={},
                adaptation_rate=0.1,
                last_update=0.0
            ),
            'phrase': MentalModel(
                model_id='phrase_model',
                level='phrase',
                learned_patterns=[],
                transition_probabilities={},
                adaptation_rate=0.05,
                last_update=0.0
            ),
            'section': MentalModel(
                model_id='section_model',
                level='section',
                learned_patterns=[],
                transition_probabilities={},
                adaptation_rate=0.02,
                last_update=0.0
            )
        }
        
        # Prediction history for learning
        self.prediction_history = defaultdict(list)
        
    def generate_predictions(self, current_features: np.ndarray, 
                           current_time: float, level: str) -> MusicalPrediction:
        """
        Generate predictions for a specific level
        
        Args:
            current_features: Current musical features
            current_time: Current time in the piece
            level: Prediction level ('measure', 'phrase', 'section')
            
        Returns:
            Musical prediction
        """
        
        if level not in self.mental_models:
            raise ValueError(f"Unknown prediction level: {level}")
        
        model = self.mental_models[level]
        horizon = self.prediction_horizons[level]
        
        # Generate prediction based on learned patterns
        predicted_features = self._generate_prediction_from_model(
            current_features, model, horizon
        )
        
        # Calculate prediction confidence
        confidence = self._calculate_prediction_confidence(
            current_features, model, predicted_features
        )
        
        # Create prediction object
        prediction = MusicalPrediction(
            prediction_id=f"{level}_{current_time:.1f}",
            level=level,
            time_horizon=horizon,
            predicted_features=predicted_features,
            confidence=confidence,
            prediction_error=0.0,  # Will be calculated when actual features arrive
            context_features=current_features
        )
        
        return prediction
    
    def update_with_actual_features(self, prediction: MusicalPrediction, 
                                  actual_features: np.ndarray) -> float:
        """
        Update mental model based on prediction error
        
        Args:
            prediction: The prediction that was made
            actual_features: The actual features that occurred
            
        Returns:
            Prediction error magnitude
        """
        
        # Calculate prediction error
        prediction_error = np.linalg.norm(
            prediction.predicted_features - actual_features
        )
        
        # Update prediction with error
        prediction.prediction_error = prediction_error
        
        # Update mental model if error is significant
        if prediction_error > self.error_threshold:
            self._update_mental_model(prediction, actual_features)
        
        # Store in history for learning
        self.prediction_history[prediction.level].append({
            'prediction': prediction,
            'actual': actual_features,
            'error': prediction_error,
            'time': prediction.prediction_id.split('_')[1]
        })
        
        return prediction_error
    
    def _generate_prediction_from_model(self, current_features: np.ndarray, 
                                      model: MentalModel, horizon: float) -> np.ndarray:
        """Generate prediction from mental model"""
        
        if not model.learned_patterns:
            # No learned patterns yet, return current features as prediction
            return current_features.copy()
        
        # Find most similar learned pattern
        similarities = []
        for pattern in model.learned_patterns:
            if len(pattern) == len(current_features):
                sim = 1 - np.linalg.norm(current_features - pattern)
                similarities.append(sim)
            else:
                similarities.append(0.0)
        
        if similarities:
            best_pattern_idx = np.argmax(similarities)
            best_pattern = model.learned_patterns[best_pattern_idx]
            
            # Generate prediction by extrapolating from best pattern
            prediction = self._extrapolate_pattern(best_pattern, current_features, horizon)
            return prediction
        
        return current_features.copy()
    
    def _extrapolate_pattern(self, pattern: np.ndarray, current_features: np.ndarray, 
                           horizon: float) -> np.ndarray:
        """Extrapolate pattern to generate prediction"""
        
        # Simple linear extrapolation
        if len(pattern) == len(current_features):
            # Calculate trend
            trend = current_features - pattern
            
            # Apply trend with horizon scaling
            horizon_scale = horizon / 6.0  # Normalize to measure length
            prediction = current_features + trend * horizon_scale
            
            return prediction
        
        return current_features.copy()
    
    def _calculate_prediction_confidence(self, current_features: np.ndarray, 
                                       model: MentalModel, 
                                       predicted_features: np.ndarray) -> float:
        """Calculate confidence in prediction"""
        
        if not model.learned_patterns:
            return 0.1  # Low confidence for new models
        
        # Calculate similarity to learned patterns
        similarities = []
        for pattern in model.learned_patterns:
            if len(pattern) == len(current_features):
                sim = 1 - np.linalg.norm(current_features - pattern)
                similarities.append(sim)
            else:
                similarities.append(0.0)
        
        if similarities:
            max_similarity = max(similarities)
            # Confidence based on similarity and model maturity
            confidence = max_similarity * (1.0 + len(model.learned_patterns) * 0.1)
            return min(confidence, 1.0)
        
        return 0.1
    
    def _update_mental_model(self, prediction: MusicalPrediction, 
                           actual_features: np.ndarray):
        """Update mental model based on prediction error"""
        
        model = self.mental_models[prediction.level]
        
        # Add actual features as new pattern if significantly different
        if not model.learned_patterns:
            model.learned_patterns.append(actual_features.copy())
        else:
            # Check if this is a new pattern
            is_new_pattern = True
            for pattern in model.learned_patterns:
                if len(pattern) == len(actual_features):
                    similarity = 1 - np.linalg.norm(actual_features - pattern)
                    if similarity > 0.8:  # Similar to existing pattern
                        is_new_pattern = False
                        break
            
            if is_new_pattern:
                model.learned_patterns.append(actual_features.copy())
        
        # Update transition probabilities
        self._update_transition_probabilities(model, prediction, actual_features)
        
        # Update adaptation rate based on prediction error
        model.adaptation_rate = min(0.2, model.adaptation_rate + prediction.prediction_error * 0.01)
        
        # Update last update time
        model.last_update = float(prediction.prediction_id.split('_')[1])
    
    def _update_transition_probabilities(self, model: MentalModel, 
                                       prediction: MusicalPrediction, 
                                       actual_features: np.ndarray):
        """Update transition probabilities between patterns"""
        
        # This is a simplified version - in practice, would track actual transitions
        # For now, just update based on prediction accuracy
        if prediction.prediction_error < self.error_threshold:
            # Good prediction, strengthen current transition
            transition_key = f"{prediction.prediction_id}_success"
            model.transition_probabilities[transition_key] = \
                model.transition_probabilities.get(transition_key, 0.0) + 0.1
        else:
            # Poor prediction, weaken current transition
            transition_key = f"{prediction.prediction_id}_failure"
            model.transition_probabilities[transition_key] = \
                model.transition_probabilities.get(transition_key, 0.0) + 0.05
    
    def get_model_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics about mental models"""
        
        stats = {}
        
        for level, model in self.mental_models.items():
            stats[level] = {
                'num_patterns': len(model.learned_patterns),
                'adaptation_rate': model.adaptation_rate,
                'last_update': model.last_update,
                'num_transitions': len(model.transition_probabilities)
            }
        
        return stats
    
    def generate_multi_level_predictions(self, current_features: np.ndarray, 
                                      current_time: float) -> Dict[str, MusicalPrediction]:
        """Generate predictions at all levels simultaneously"""
        
        predictions = {}
        
        for level in self.mental_models.keys():
            predictions[level] = self.generate_predictions(
                current_features, current_time, level
            )
        
        return predictions

class MentalModelManager:
    """
    Manages multiple mental models and their interactions
    
    Based on research showing that humans maintain multiple mental models
    at different timescales and integrate them for musical understanding.
    """
    
    def __init__(self):
        self.models = {}
        self.integration_weights = {
            'measure': 0.4,
            'phrase': 0.4,
            'section': 0.2
        }
    
    def integrate_predictions(self, predictions: Dict[str, MusicalPrediction]) -> np.ndarray:
        """Integrate predictions from multiple levels"""
        
        if not predictions:
            return np.array([])
        
        # Weighted combination of predictions
        integrated_prediction = None
        
        for level, prediction in predictions.items():
            weight = self.integration_weights.get(level, 0.0)
            
            if integrated_prediction is None:
                integrated_prediction = prediction.predicted_features * weight
            else:
                integrated_prediction += prediction.predicted_features * weight
        
        return integrated_prediction
    
    def calculate_prediction_consistency(self, predictions: Dict[str, MusicalPrediction]) -> float:
        """Calculate consistency between predictions at different levels"""
        
        if len(predictions) < 2:
            return 1.0
        
        # Calculate pairwise similarities between predictions
        similarities = []
        prediction_list = list(predictions.values())
        
        for i in range(len(prediction_list)):
            for j in range(i + 1, len(prediction_list)):
                pred1 = prediction_list[i]
                pred2 = prediction_list[j]
                
                if len(pred1.predicted_features) == len(pred2.predicted_features):
                    sim = 1 - np.linalg.norm(
                        pred1.predicted_features - pred2.predicted_features
                    )
                    similarities.append(sim)
        
        return np.mean(similarities) if similarities else 0.0

def main():
    """Test the predictive processing engine"""
    
    print("🚀 Starting predictive processing engine test...")
    
    engine = PredictiveProcessingEngine()
    manager = MentalModelManager()
    
    # Simulate some musical features
    print("🎵 Simulating musical features...")
    
    # Generate some test features
    np.random.seed(42)
    test_features = np.random.rand(14)  # 14-dimensional feature vector
    
    # Generate predictions at all levels
    print("🔮 Generating multi-level predictions...")
    predictions = engine.generate_multi_level_predictions(test_features, 0.0)
    
    print(f"\n🎯 Prediction Results:")
    for level, prediction in predictions.items():
        print(f"   {level.capitalize()}: confidence={prediction.confidence:.3f}, "
              f"horizon={prediction.time_horizon:.1f}s")
    
    # Test prediction integration
    print("\n🔄 Testing prediction integration...")
    integrated_prediction = manager.integrate_predictions(predictions)
    consistency = manager.calculate_prediction_consistency(predictions)
    
    print(f"   Integrated prediction shape: {integrated_prediction.shape}")
    print(f"   Prediction consistency: {consistency:.3f}")
    
    # Simulate learning with actual features
    print("\n📚 Simulating learning...")
    actual_features = test_features + np.random.rand(14) * 0.1  # Slightly different
    
    for level, prediction in predictions.items():
        error = engine.update_with_actual_features(prediction, actual_features)
        print(f"   {level.capitalize()} prediction error: {error:.3f}")
    
    # Show model statistics
    print("\n📊 Model Statistics:")
    stats = engine.get_model_statistics()
    for level, level_stats in stats.items():
        print(f"   {level.capitalize()}: {level_stats['num_patterns']} patterns, "
              f"adaptation rate: {level_stats['adaptation_rate']:.3f}")
    
    print("\n✅ Predictive processing test complete!")

if __name__ == "__main__":
    main()
