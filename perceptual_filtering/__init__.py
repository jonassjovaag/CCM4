"""
Perceptual Filtering Module
Based on current research in auditory perception and music cognition

This module implements perceptual filtering that mirrors human auditory processing:
- Stream segregation (Pressnitzer et al., 2008)
- Gestalt principles (Bregman, 1990)
- Perceptual significance filtering

Key Components:
- AuditorySceneAnalyzer: Performs stream segregation and classification
- PerceptualSignificanceFilter: Filters patterns based on perceptual importance
- GestaltAnalyzer: Applies Gestalt principles for perceptual grouping
"""

from .significance_filter import (
    AuditoryStream, 
    PerceptualSignificance, 
    AuditorySceneAnalyzer, 
    PerceptualSignificanceFilter
)
from .auditory_scene_analyzer import (
    PerceptualGroup, 
    GestaltAnalyzer
)

__all__ = [
    'AuditoryStream',
    'PerceptualSignificance', 
    'AuditorySceneAnalyzer',
    'PerceptualSignificanceFilter',
    'PerceptualGroup',
    'GestaltAnalyzer'
]

__version__ = "1.0.0"
__author__ = "MusicHal 9000 Development Team"
