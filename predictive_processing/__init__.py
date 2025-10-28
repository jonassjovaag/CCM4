"""
Predictive Processing Module
Based on current research in music cognition and predictive processing

This module implements predictive processing that mirrors human musical cognition:
- Multi-level predictions (Schaefer, 2014)
- Mental model management
- Prediction error learning

Key Components:
- PredictiveProcessingEngine: Generates predictions and learns from errors
- MentalModelManager: Manages multiple mental models and their interactions
"""

from .prediction_engine import (
    MusicalPrediction,
    MentalModel,
    PredictiveProcessingEngine,
    MentalModelManager as PredictionMentalModelManager
)
from .mental_model_manager import (
    ModelState,
    MentalModelManager
)

__all__ = [
    'MusicalPrediction',
    'MentalModel',
    'PredictiveProcessingEngine',
    'PredictionMentalModelManager',
    'ModelState',
    'MentalModelManager'
]

__version__ = "1.0.0"
__author__ = "MusicHal 9000 Development Team"
