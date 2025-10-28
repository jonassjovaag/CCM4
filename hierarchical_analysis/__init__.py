"""
Hierarchical Analysis Module
Based on current research in music perception and neural processing

This module implements hierarchical music analysis that mirrors human neural processing:
- Multi-timescale analysis (Farbood et al., 2015)
- Structural community detection (de Berardinis et al., 2020)
- Temporal relationship management

Key Components:
- MultiTimescaleAnalyzer: Analyzes music at measure, phrase, and section levels
- MSCOMStructureDetector: Detects structural communities using graph theory
- TimescaleManager: Coordinates analysis across different timescales
"""

from .multi_timescale_analyzer import MultiTimescaleAnalyzer, MusicalPattern, HierarchicalStructure
from .structure_detector import MSCOMStructureDetector, StructuralBoundary, StructuralCommunity
from .timescale_manager import TimescaleManager, TimescaleConfig

__all__ = [
    'MultiTimescaleAnalyzer',
    'MusicalPattern', 
    'HierarchicalStructure',
    'MSCOMStructureDetector',
    'StructuralBoundary',
    'StructuralCommunity',
    'TimescaleManager',
    'TimescaleConfig'
]

__version__ = "1.0.0"
__author__ = "MusicHal 9000 Development Team"
