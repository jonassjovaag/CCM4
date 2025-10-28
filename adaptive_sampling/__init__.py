"""
Adaptive Sampling Module
Based on current research in music information retrieval and adaptive sampling

This module implements adaptive sampling strategies that:
- Sample events based on structural analysis
- Focus on perceptually significant moments
- Maintain temporal distribution
- Limit to 10K events for efficiency

Key Components:
- SmartSampler: Implements multiple sampling strategies
- StructuralSampler: Specialized sampler for structural analysis
"""

from .smart_sampler import (
    SamplingStrategy,
    SampledEvent,
    SmartSampler,
    StructuralSampler
)

__all__ = [
    'SamplingStrategy',
    'SampledEvent',
    'SmartSampler',
    'StructuralSampler'
]

__version__ = "1.0.0"
__author__ = "MusicHal 9000 Development Team"
