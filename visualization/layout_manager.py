#!/usr/bin/env python3
"""
Automatic Layout Manager for Multi-Viewport Visualization
Calculates optimal grid arrangements for N viewports using available screen space
"""

import math
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class ViewportPosition:
    """Position and size for a viewport"""
    x: int
    y: int
    width: int
    height: int
    viewport_id: str
    priority: int = 0


class LayoutManager:
    """
    Automatically arranges viewports in optimal grid layout
    
    Handles:
    - Screen size detection
    - Automatic grid calculation (optimal rows x cols)
    - Centering incomplete rows
    - Padding between viewports
    - Viewport priority positioning
    """
    
    def __init__(self, padding: int = 10, margin: int = 20):
        """
        Initialize layout manager
        
        Args:
            padding: Space between viewports in pixels
            margin: Space around edges of screen in pixels
        """
        self.padding = padding
        self.margin = margin
    
    def calculate_layout(self,
                        num_viewports: int,
                        screen_width: int,
                        screen_height: int,
                        viewport_ids: Optional[List[str]] = None,
                        viewport_priorities: Optional[Dict[str, int]] = None) -> List[ViewportPosition]:
        """
        Calculate optimal grid layout for N viewports
        
        Args:
            num_viewports: Number of viewports to arrange
            screen_width: Available screen width in pixels
            screen_height: Available screen height in pixels
            viewport_ids: Optional list of viewport identifiers
            viewport_priorities: Optional dict mapping viewport_id to priority (higher = better position)
        
        Returns:
            List of ViewportPosition objects with calculated positions
        
        Examples:
            1 viewport: 1x1 (fullscreen minus margins)
            2 viewports: 2x1 (side by side)
            3 viewports: 3x1 or 2x2 with one empty
            4 viewports: 2x2 perfect grid
            5 viewports: 3x2 (top row: 3, bottom row: 2 centered)
            6 viewports: 3x2 or 2x3 depending on screen aspect
            9 viewports: 3x3 perfect grid
        """
        if num_viewports == 0:
            return []
        
        # Create default viewport IDs if not provided
        if viewport_ids is None:
            viewport_ids = [f"viewport_{i}" for i in range(num_viewports)]
        
        # Sort viewports by priority if provided (higher priority = top-left)
        if viewport_priorities:
            sorted_ids = sorted(viewport_ids, 
                              key=lambda vid: viewport_priorities.get(vid, 0), 
                              reverse=True)
        else:
            sorted_ids = viewport_ids
        
        # Calculate usable area (screen minus margins)
        usable_width = screen_width - (2 * self.margin)
        usable_height = screen_height - (2 * self.margin)
        
        # Calculate optimal grid dimensions
        cols, rows = self._calculate_grid_dimensions(num_viewports, usable_width, usable_height)
        
        # Calculate viewport sizes
        viewport_width = (usable_width - (cols - 1) * self.padding) // cols
        viewport_height = (usable_height - (rows - 1) * self.padding) // rows
        
        # Calculate positions for each viewport
        positions = []
        for i in range(num_viewports):
            row = i // cols
            col = i % cols
            
            # For incomplete last row, center the viewports
            if row == rows - 1 and num_viewports % cols != 0:
                remaining = num_viewports % cols
                col_offset = (cols - remaining) / 2
                x = self.margin + int((col + col_offset) * (viewport_width + self.padding))
            else:
                x = self.margin + col * (viewport_width + self.padding)
            
            y = self.margin + row * (viewport_height + self.padding)
            
            priority = viewport_priorities.get(sorted_ids[i], 0) if viewport_priorities else 0
            
            positions.append(ViewportPosition(
                x=int(x),
                y=int(y),
                width=viewport_width,
                height=viewport_height,
                viewport_id=sorted_ids[i],
                priority=priority
            ))
        
        return positions
    
    def _calculate_grid_dimensions(self, 
                                   num_viewports: int, 
                                   width: int, 
                                   height: int) -> Tuple[int, int]:
        """
        Calculate optimal columns and rows for grid layout
        
        Strategy:
        - Start with cols = ceil(sqrt(num_viewports))
        - Calculate rows = ceil(num_viewports / cols)
        - Adjust for very wide screens (prefer horizontal layout)
        
        Returns:
            (columns, rows) tuple
        """
        if num_viewports == 1:
            return (1, 1)
        
        if num_viewports == 2:
            return (2, 1)  # Side by side
        
        # Calculate aspect ratio
        aspect_ratio = width / height
        
        # For very wide screens (ultrawide), prefer horizontal layouts
        if aspect_ratio > 2.0 and num_viewports <= 4:
            return (num_viewports, 1)
        
        # Standard grid calculation
        cols = math.ceil(math.sqrt(num_viewports))
        rows = math.ceil(num_viewports / cols)
        
        # Optimize: if we can fit in fewer rows with one more column, do it
        # Example: 5 viewports could be 3x2 or 2x3, prefer 3x2 (wider)
        if aspect_ratio > 1.3:  # Landscape orientation
            alt_cols = cols + 1
            alt_rows = math.ceil(num_viewports / alt_cols)
            if alt_rows < rows:
                cols = alt_cols
                rows = alt_rows
        
        return (cols, rows)
    
    def get_screen_dimensions(self) -> Tuple[int, int]:
        """
        Get available screen dimensions
        
        Uses Qt to detect primary screen size
        Falls back to reasonable defaults if Qt not available
        
        Returns:
            (width, height) tuple in pixels
        """
        try:
            from PyQt5.QtWidgets import QApplication, QDesktopWidget
            app = QApplication.instance()
            if app is None:
                app = QApplication([])
            
            desktop = QDesktopWidget()
            screen = desktop.availableGeometry(desktop.primaryScreen())
            
            # Use 90% of available screen (leave room for dock/menubar)
            width = int(screen.width() * 0.9)
            height = int(screen.height() * 0.9)
            
            return (width, height)
        
        except ImportError:
            # PyQt5 not available, use reasonable defaults
            print("⚠️  PyQt5 not available, using default screen size 1920x1080")
            return (1728, 972)  # 90% of 1920x1080
    
    def format_layout_info(self, positions: List[ViewportPosition]) -> str:
        """
        Format layout information for debugging/logging
        
        Args:
            positions: List of viewport positions
        
        Returns:
            Formatted string describing the layout
        """
        if not positions:
            return "No viewports"
        
        cols = max(p.x for p in positions) // (positions[0].width + self.padding) + 1
        rows = max(p.y for p in positions) // (positions[0].height + self.padding) + 1
        
        info = f"Layout: {cols}x{rows} grid, {len(positions)} viewports\n"
        info += f"Viewport size: {positions[0].width}x{positions[0].height}px\n"
        info += f"Padding: {self.padding}px, Margin: {self.margin}px"
        
        return info


if __name__ == "__main__":
    # Test the layout manager
    print("🧪 Testing LayoutManager\n")
    
    manager = LayoutManager(padding=10, margin=20)
    
    # Test different viewport counts
    test_cases = [1, 2, 3, 4, 5, 6, 8, 9]
    screen_width, screen_height = 1920, 1080
    
    for num_viewports in test_cases:
        print(f"\n{'='*60}")
        print(f"Test: {num_viewports} viewports on {screen_width}x{screen_height} screen")
        print('='*60)
        
        viewport_ids = [f"viewport_{i}" for i in range(num_viewports)]
        positions = manager.calculate_layout(num_viewports, screen_width, screen_height, viewport_ids)
        
        print(manager.format_layout_info(positions))
        print("\nViewport positions:")
        for pos in positions:
            print(f"  {pos.viewport_id}: ({pos.x}, {pos.y}) {pos.width}x{pos.height}px")
    
    print("\n✅ LayoutManager tests complete!")

