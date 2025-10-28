"""
Multi-Viewport Visualization System for MusicHal_9000

Provides real-time visualization of internal AI processes during performance.
"""

from .visualization_manager import VisualizationManager
from .event_bus import VisualizationEventBus, EventType
from .layout_manager import LayoutManager

__all__ = [
    'VisualizationManager',
    'VisualizationEventBus',
    'EventType',
    'LayoutManager'
]
