#!/usr/bin/env python3
"""
Mental Model Manager
Based on Schaefer (2014): "Mental Representations in Musical Processing and their Role in Action-Perception Loops"

This module manages mental models and their interactions for musical understanding.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import json
import pickle

@dataclass
class ModelState:
    """Represents the state of a mental model"""
    model_id: str
    level: str
    patterns: List[np.ndarray]
    transitions: Dict[str, float]
    confidence: float
    last_used: float
    usage_count: int

class MentalModelManager:
    """
    Manages multiple mental models and their interactions
    
    Based on research showing that humans maintain multiple mental models
    at different timescales and integrate them for musical understanding.
    """
    
    def __init__(self, 
                 model_capacity: int = 100,
                 decay_rate: float = 0.01,
                 integration_weights: Dict[str, float] = None):
        
        self.model_capacity = model_capacity
        self.decay_rate = decay_rate
        
        if integration_weights is None:
            self.integration_weights = {
                'measure': 0.4,
                'phrase': 0.4,
                'section': 0.2
            }
        else:
            self.integration_weights = integration_weights
        
        # Model storage
        self.models: Dict[str, ModelState] = {}
        
        # Model interaction tracking
        self.interaction_history = defaultdict(list)
        
        # Model performance tracking
        self.performance_metrics = defaultdict(lambda: {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0
        })
    
    def create_model(self, model_id: str, level: str) -> ModelState:
        """Create a new mental model"""
        
        model = ModelState(
            model_id=model_id,
            level=level,
            patterns=[],
            transitions={},
            confidence=0.0,
            last_used=0.0,
            usage_count=0
        )
        
        self.models[model_id] = model
        return model
    
    def update_model(self, model_id: str, new_pattern: np.ndarray, 
                    transition_from: Optional[str] = None) -> bool:
        """Update a mental model with new pattern"""
        
        if model_id not in self.models:
            return False
        
        model = self.models[model_id]
        
        # Add new pattern
        model.patterns.append(new_pattern.copy())
        
        # Update transitions
        if transition_from:
            transition_key = f"{transition_from}->{model_id}"
            model.transitions[transition_key] = model.transitions.get(transition_key, 0.0) + 1.0
        
        # Update usage statistics
        model.usage_count += 1
        model.last_used = float(model_id.split('_')[-1]) if '_' in model_id else 0.0
        
        # Maintain model capacity
        if len(model.patterns) > self.model_capacity:
            # Remove oldest patterns
            model.patterns = model.patterns[-self.model_capacity:]
        
        return True
    
    def get_model_prediction(self, model_id: str, context: np.ndarray) -> Tuple[np.ndarray, float]:
        """Get prediction from a specific model"""
        
        if model_id not in self.models:
            return context.copy(), 0.0
        
        model = self.models[model_id]
        
        if not model.patterns:
            return context.copy(), 0.0
        
        # Find most similar pattern
        similarities = []
        for pattern in model.patterns:
            if len(pattern) == len(context):
                sim = 1 - np.linalg.norm(context - pattern)
                similarities.append(sim)
            else:
                similarities.append(0.0)
        
        if similarities:
            best_idx = np.argmax(similarities)
            best_pattern = model.patterns[best_idx]
            confidence = similarities[best_idx]
            
            # Generate prediction by extrapolating from best pattern
            prediction = self._extrapolate_from_pattern(best_pattern, context)
            
            return prediction, confidence
        
        return context.copy(), 0.0
    
    def _extrapolate_from_pattern(self, pattern: np.ndarray, context: np.ndarray) -> np.ndarray:
        """Extrapolate prediction from pattern"""
        
        if len(pattern) != len(context):
            return context.copy()
        
        # Simple linear extrapolation
        trend = context - pattern
        prediction = context + trend * 0.5  # Moderate extrapolation
        
        return prediction
    
    def integrate_predictions(self, predictions: Dict[str, Tuple[np.ndarray, float]]) -> np.ndarray:
        """Integrate predictions from multiple models"""
        
        if not predictions:
            return np.array([])
        
        # Weighted combination of predictions
        integrated_prediction = None
        total_weight = 0.0
        
        for model_id, (prediction, confidence) in predictions.items():
            if model_id in self.models:
                level = self.models[model_id].level
                weight = self.integration_weights.get(level, 0.0) * confidence
                
                if integrated_prediction is None:
                    integrated_prediction = prediction * weight
                else:
                    integrated_prediction += prediction * weight
                
                total_weight += weight
        
        if total_weight > 0:
            integrated_prediction /= total_weight
        
        return integrated_prediction if integrated_prediction is not None else np.array([])
    
    def calculate_model_consistency(self, predictions: Dict[str, Tuple[np.ndarray, float]]) -> float:
        """Calculate consistency between model predictions"""
        
        if len(predictions) < 2:
            return 1.0
        
        # Calculate pairwise similarities between predictions
        similarities = []
        prediction_list = list(predictions.values())
        
        for i in range(len(prediction_list)):
            for j in range(i + 1, len(prediction_list)):
                pred1, conf1 = prediction_list[i]
                pred2, conf2 = prediction_list[j]
                
                if len(pred1) == len(pred2):
                    sim = 1 - np.linalg.norm(pred1 - pred2)
                    similarities.append(sim)
        
        return np.mean(similarities) if similarities else 0.0
    
    def update_performance_metrics(self, model_id: str, 
                                 actual_features: np.ndarray, 
                                 predicted_features: np.ndarray):
        """Update performance metrics for a model"""
        
        if model_id not in self.models:
            return
        
        # Calculate prediction error
        error = np.linalg.norm(actual_features - predicted_features)
        
        # Update accuracy (inverse of error)
        accuracy = max(0.0, 1.0 - error)
        
        # Update performance metrics
        self.performance_metrics[model_id]['accuracy'] = \
            self.performance_metrics[model_id]['accuracy'] * 0.9 + accuracy * 0.1
        
        # Update model confidence based on performance
        self.models[model_id].confidence = self.performance_metrics[model_id]['accuracy']
    
    def get_model_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get comprehensive statistics about all models"""
        
        stats = {}
        
        for model_id, model in self.models.items():
            stats[model_id] = {
                'level': model.level,
                'num_patterns': len(model.patterns),
                'confidence': model.confidence,
                'usage_count': model.usage_count,
                'last_used': model.last_used,
                'num_transitions': len(model.transitions),
                'performance': self.performance_metrics[model_id].copy()
            }
        
        return stats
    
    def save_models(self, filepath: str) -> bool:
        """Save mental models to file"""
        
        try:
            # Convert numpy arrays to lists for JSON serialization
            serializable_models = {}
            for model_id, model in self.models.items():
                serializable_models[model_id] = {
                    'model_id': model.model_id,
                    'level': model.level,
                    'patterns': [pattern.tolist() for pattern in model.patterns],
                    'transitions': model.transitions,
                    'confidence': model.confidence,
                    'last_used': model.last_used,
                    'usage_count': model.usage_count
                }
            
            with open(filepath, 'w') as f:
                json.dump({
                    'models': serializable_models,
                    'integration_weights': self.integration_weights,
                    'performance_metrics': dict(self.performance_metrics)
                }, f, indent=2)
            
            return True
            
        except Exception as e:
            print(f"Error saving models: {e}")
            return False
    
    def load_models(self, filepath: str) -> bool:
        """Load mental models from file"""
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Restore models
            self.models = {}
            for model_id, model_data in data['models'].items():
                model = ModelState(
                    model_id=model_data['model_id'],
                    level=model_data['level'],
                    patterns=[np.array(pattern) for pattern in model_data['patterns']],
                    transitions=model_data['transitions'],
                    confidence=model_data['confidence'],
                    last_used=model_data['last_used'],
                    usage_count=model_data['usage_count']
                )
                self.models[model_id] = model
            
            # Restore other data
            self.integration_weights = data.get('integration_weights', self.integration_weights)
            self.performance_metrics = defaultdict(lambda: {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0
            }, data.get('performance_metrics', {}))
            
            return True
            
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
    
    def prune_models(self, min_usage_count: int = 5) -> int:
        """Remove models with low usage"""
        
        pruned_count = 0
        models_to_remove = []
        
        for model_id, model in self.models.items():
            if model.usage_count < min_usage_count:
                models_to_remove.append(model_id)
        
        for model_id in models_to_remove:
            del self.models[model_id]
            pruned_count += 1
        
        return pruned_count

def main():
    """Test the mental model manager"""
    
    print("🚀 Starting mental model manager test...")
    
    manager = MentalModelManager()
    
    # Create some test models
    print("🏗️  Creating test models...")
    
    measure_model = manager.create_model("measure_model_1", "measure")
    phrase_model = manager.create_model("phrase_model_1", "phrase")
    section_model = manager.create_model("section_model_1", "section")
    
    print(f"✅ Created {len(manager.models)} models")
    
    # Add some patterns to models
    print("📚 Adding patterns to models...")
    
    np.random.seed(42)
    for i in range(5):
        pattern = np.random.rand(14)
        manager.update_model("measure_model_1", pattern, f"pattern_{i-1}" if i > 0 else None)
        
        pattern = np.random.rand(14)
        manager.update_model("phrase_model_1", pattern, f"pattern_{i-1}" if i > 0 else None)
    
    print("✅ Patterns added")
    
    # Test predictions
    print("🔮 Testing predictions...")
    
    context = np.random.rand(14)
    predictions = {}
    
    for model_id in manager.models.keys():
        prediction, confidence = manager.get_model_prediction(model_id, context)
        predictions[model_id] = (prediction, confidence)
        print(f"   {model_id}: confidence={confidence:.3f}")
    
    # Test integration
    print("🔄 Testing prediction integration...")
    
    integrated_prediction = manager.integrate_predictions(predictions)
    consistency = manager.calculate_model_consistency(predictions)
    
    print(f"   Integrated prediction shape: {integrated_prediction.shape}")
    print(f"   Model consistency: {consistency:.3f}")
    
    # Test performance updates
    print("📊 Testing performance updates...")
    
    actual_features = context + np.random.rand(14) * 0.1
    for model_id, (prediction, _) in predictions.items():
        manager.update_performance_metrics(model_id, actual_features, prediction)
    
    # Show statistics
    print("\n📈 Model Statistics:")
    stats = manager.get_model_statistics()
    for model_id, model_stats in stats.items():
        print(f"   {model_id}: {model_stats['num_patterns']} patterns, "
              f"confidence: {model_stats['confidence']:.3f}, "
              f"usage: {model_stats['usage_count']}")
    
    # Test saving/loading
    print("\n💾 Testing save/load...")
    
    save_success = manager.save_models("test_models.json")
    print(f"   Save successful: {save_success}")
    
    if save_success:
        # Create new manager and load
        new_manager = MentalModelManager()
        load_success = new_manager.load_models("test_models.json")
        print(f"   Load successful: {load_success}")
        
        if load_success:
            print(f"   Loaded {len(new_manager.models)} models")
    
    print("\n✅ Mental model manager test complete!")

if __name__ == "__main__":
    main()
