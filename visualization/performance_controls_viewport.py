#!/usr/bin/env python3
"""
Performance Controls Viewport
Provides live-adjustable parameters for real-time performance control
"""

from typing import Optional
from PyQt5.QtWidgets import (QVBoxLayout, QLabel, QSlider, 
                              QComboBox, QGroupBox, QGridLayout)
from PyQt5.QtCore import Qt, pyqtSignal
from visualization.base_viewport import BaseViewport


class PerformanceControlsViewport(BaseViewport):
    """Viewport for live performance parameter controls"""
    
    # Signals emitted when controls change
    engagement_profile_changed = pyqtSignal(str)
    engagement_level_changed = pyqtSignal(float)
    behavior_mode_changed = pyqtSignal(str)
    confidence_changed = pyqtSignal(float)
    density_changed = pyqtSignal(float)
    give_space_changed = pyqtSignal(float)
    initiative_changed = pyqtSignal(float)
    silence_tolerance_changed = pyqtSignal(float)
    adaptation_rate_changed = pyqtSignal(float)
    momentum_changed = pyqtSignal(float)
    
    def __init__(self):
        """Initialize the performance controls viewport"""
        super().__init__(viewport_id="performance_controls", title="Performance Controls")
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the user interface"""
        # Create main layout
        main_layout = QVBoxLayout()
        main_layout.setSpacing(10)
        
        # Section 1: Core Behavior
        core_group = self._create_core_behavior_section()
        main_layout.addWidget(core_group)
        
        # Section 2: Musical Interaction
        interaction_group = self._create_musical_interaction_section()
        main_layout.addWidget(interaction_group)
        
        # Section 3: Timing & Dynamics
        timing_group = self._create_timing_dynamics_section()
        main_layout.addWidget(timing_group)
        
        # Section 4: Performance Arc (Display Only)
        arc_group = self._create_performance_arc_section()
        main_layout.addWidget(arc_group)
        
        # Add stretch at bottom to push everything up
        main_layout.addStretch()
        
        # Apply layout to content widget
        self.content_widget.setLayout(main_layout)
    
    def _create_core_behavior_section(self) -> QGroupBox:
        """Create core behavior controls section"""
        group = QGroupBox("Core Behavior")
        layout = QGridLayout()
        layout.setSpacing(8)
        
        row = 0
        
        # Engagement Profile (Dropdown)
        layout.addWidget(QLabel("Engagement Profile:"), row, 0)
        self.engagement_profile_combo = QComboBox()
        self.engagement_profile_combo.addItems(['conservative', 'balanced', 'experimental'])
        self.engagement_profile_combo.setCurrentText('balanced')
        self.engagement_profile_combo.currentTextChanged.connect(
            lambda val: self.engagement_profile_changed.emit(val)
        )
        layout.addWidget(self.engagement_profile_combo, row, 1, 1, 2)
        row += 1
        
        # Engagement Level (Slider)
        layout.addWidget(QLabel("Engagement Level:"), row, 0)
        self.engagement_level_slider, self.engagement_level_label = self._create_slider(
            0.0, 1.0, 0.75, 100, self.engagement_level_changed.emit
        )
        layout.addWidget(self.engagement_level_slider, row, 1)
        layout.addWidget(self.engagement_level_label, row, 2)
        row += 1
        
        # Behavior Mode (Dropdown)
        layout.addWidget(QLabel("Behavior Mode:"), row, 0)
        self.behavior_mode_combo = QComboBox()
        self.behavior_mode_combo.addItems(['imitate', 'contrast', 'lead', 'wait'])
        self.behavior_mode_combo.setCurrentText('contrast')
        self.behavior_mode_combo.currentTextChanged.connect(
            lambda val: self.behavior_mode_changed.emit(val)
        )
        layout.addWidget(self.behavior_mode_combo, row, 1, 1, 2)
        row += 1
        
        # Confidence Threshold (Slider)
        layout.addWidget(QLabel("Confidence:"), row, 0)
        self.confidence_slider, self.confidence_label = self._create_slider(
            0.6, 0.9, 0.7, 30, self.confidence_changed.emit
        )
        layout.addWidget(self.confidence_slider, row, 1)
        layout.addWidget(self.confidence_label, row, 2)
        row += 1
        
        group.setLayout(layout)
        return group
    
    def _create_musical_interaction_section(self) -> QGroupBox:
        """Create musical interaction controls section"""
        group = QGroupBox("Musical Interaction")
        layout = QGridLayout()
        layout.setSpacing(8)
        
        row = 0
        
        # Density Level (Slider)
        layout.addWidget(QLabel("Density Level:"), row, 0)
        self.density_slider, self.density_label = self._create_slider(
            0.0, 1.0, 0.5, 100, self.density_changed.emit
        )
        layout.addWidget(self.density_slider, row, 1)
        layout.addWidget(self.density_label, row, 2)
        row += 1
        
        # Give Space Factor (Slider)
        layout.addWidget(QLabel("Give Space:"), row, 0)
        self.give_space_slider, self.give_space_label = self._create_slider(
            0.0, 1.0, 0.3, 100, self.give_space_changed.emit
        )
        layout.addWidget(self.give_space_slider, row, 1)
        layout.addWidget(self.give_space_label, row, 2)
        row += 1
        
        # Initiative Budget (Slider)
        layout.addWidget(QLabel("Initiative:"), row, 0)
        self.initiative_slider, self.initiative_label = self._create_slider(
            0.0, 1.0, 0.7, 100, self.initiative_changed.emit
        )
        layout.addWidget(self.initiative_slider, row, 1)
        layout.addWidget(self.initiative_label, row, 2)
        row += 1
        
        group.setLayout(layout)
        return group
    
    def _create_timing_dynamics_section(self) -> QGroupBox:
        """Create timing and dynamics controls section"""
        group = QGroupBox("Timing & Dynamics")
        layout = QGridLayout()
        layout.setSpacing(8)
        
        row = 0
        
        # Silence Tolerance (Slider)
        layout.addWidget(QLabel("Silence Tolerance:"), row, 0)
        self.silence_tolerance_slider, self.silence_tolerance_label = self._create_slider(
            0.0, 20.0, 5.0, 200, self.silence_tolerance_changed.emit, suffix="s"
        )
        layout.addWidget(self.silence_tolerance_slider, row, 1)
        layout.addWidget(self.silence_tolerance_label, row, 2)
        row += 1
        
        # Adaptation Rate (Slider)
        layout.addWidget(QLabel("Adaptation Rate:"), row, 0)
        self.adaptation_rate_slider, self.adaptation_rate_label = self._create_slider(
            0.0, 1.0, 0.1, 100, self.adaptation_rate_changed.emit
        )
        layout.addWidget(self.adaptation_rate_slider, row, 1)
        layout.addWidget(self.adaptation_rate_label, row, 2)
        row += 1
        
        # Musical Momentum (Slider - Override)
        layout.addWidget(QLabel("Momentum:"), row, 0)
        self.momentum_slider, self.momentum_label = self._create_slider(
            0.0, 1.0, 0.65, 100, self.momentum_changed.emit
        )
        layout.addWidget(self.momentum_slider, row, 1)
        layout.addWidget(self.momentum_label, row, 2)
        row += 1
        
        group.setLayout(layout)
        return group
    
    def _create_performance_arc_section(self) -> QGroupBox:
        """Create performance arc display section"""
        group = QGroupBox("Performance Arc")
        layout = QGridLayout()
        layout.setSpacing(8)
        
        row = 0
        
        # Performance Phase (Display)
        layout.addWidget(QLabel("Phase:"), row, 0)
        self.phase_display = QLabel("--")
        self.phase_display.setStyleSheet("color: #00FF00; font-weight: bold;")
        layout.addWidget(self.phase_display, row, 1, 1, 2)
        row += 1
        
        # Activity Multiplier (Display)
        layout.addWidget(QLabel("Activity:"), row, 0)
        self.activity_display = QLabel("1.00")
        self.activity_display.setStyleSheet("color: #00FF00;")
        layout.addWidget(self.activity_display, row, 1, 1, 2)
        row += 1
        
        # Time Remaining (Display)
        layout.addWidget(QLabel("Time Left:"), row, 0)
        self.time_remaining_display = QLabel("--:--")
        self.time_remaining_display.setStyleSheet("color: #00FF00;")
        layout.addWidget(self.time_remaining_display, row, 1, 1, 2)
        row += 1
        
        # Detected Instrument (Display)
        layout.addWidget(QLabel("Instrument:"), row, 0)
        self.instrument_display = QLabel("unknown")
        self.instrument_display.setStyleSheet("color: #00FF00;")
        layout.addWidget(self.instrument_display, row, 1, 1, 2)
        row += 1
        
        group.setLayout(layout)
        return group
    
    def _create_slider(self, min_val: float, max_val: float, default_val: float, 
                       steps: int, signal_callback, suffix: str = "") -> tuple:
        """Create a slider with label"""
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(steps)
        slider.setValue(int((default_val - min_val) / (max_val - min_val) * steps))
        
        label = QLabel(f"{default_val:.2f}{suffix}")
        label.setMinimumWidth(60)
        label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        
        def on_slider_change(value):
            actual_value = min_val + (value / steps) * (max_val - min_val)
            label.setText(f"{actual_value:.2f}{suffix}")
            signal_callback(actual_value)
        
        slider.valueChanged.connect(on_slider_change)
        
        return slider, label
    
    def update_display_fields(self, phase: Optional[str] = None, activity: Optional[float] = None, 
                              time_remaining: Optional[str] = None, instrument: Optional[str] = None):
        """Update the display-only fields"""
        if phase is not None:
            self.phase_display.setText(phase)
        
        if activity is not None:
            self.activity_display.setText(f"{activity:.2f}")
        
        if time_remaining is not None:
            self.time_remaining_display.setText(time_remaining)
        
        if instrument is not None:
            self.instrument_display.setText(instrument)
    
    # Programmatic setters (for initialization or external updates)
    def set_engagement_profile(self, value: str):
        """Set engagement profile programmatically"""
        self.engagement_profile_combo.setCurrentText(value)
    
    def set_engagement_level(self, value: float):
        """Set engagement level programmatically"""
        steps = 100
        slider_value = int(value * steps)
        self.engagement_level_slider.setValue(slider_value)
    
    def set_behavior_mode(self, value: str):
        """Set behavior mode programmatically"""
        self.behavior_mode_combo.setCurrentText(value)
    
    def set_confidence(self, value: float):
        """Set confidence programmatically"""
        steps = 30
        min_val, max_val = 0.6, 0.9
        slider_value = int((value - min_val) / (max_val - min_val) * steps)
        self.confidence_slider.setValue(slider_value)
    
    def set_density(self, value: float):
        """Set density programmatically"""
        steps = 100
        slider_value = int(value * steps)
        self.density_slider.setValue(slider_value)
    
    def set_give_space(self, value: float):
        """Set give space programmatically"""
        steps = 100
        slider_value = int(value * steps)
        self.give_space_slider.setValue(slider_value)
    
    def set_initiative(self, value: float):
        """Set initiative programmatically"""
        steps = 100
        slider_value = int(value * steps)
        self.initiative_slider.setValue(slider_value)
    
    def set_silence_tolerance(self, value: float):
        """Set silence tolerance programmatically"""
        steps = 200
        max_val = 20.0
        slider_value = int((value / max_val) * steps)
        self.silence_tolerance_slider.setValue(slider_value)
    
    def set_adaptation_rate(self, value: float):
        """Set adaptation rate programmatically"""
        steps = 100
        slider_value = int(value * steps)
        self.adaptation_rate_slider.setValue(slider_value)
    
    def set_momentum(self, value: float):
        """Set momentum programmatically"""
        steps = 100
        slider_value = int(value * steps)
        self.momentum_slider.setValue(slider_value)
