#!/usr/bin/env python3
"""
Enhanced Main.py with Rhythmic Engine Integration
Adds rhythmic analysis alongside existing harmonic system

This enhanced version:
- Keeps all existing harmonic functionality intact
- Adds rhythmic analysis as a parallel system
- Combines harmonic and rhythmic decisions
- Maintains backward compatibility
"""

# FIRST: Ensure CCM3 virtual environment is active
try:
    from ccm3_venv_manager import ensure_ccm3_venv_active
    ensure_ccm3_venv_active()
    print("✅ CCM3 virtual environment activated")
except ImportError:
    print("⚠️  CCM3 virtual environment manager not found, continuing with current environment")
except Exception as e:
    print(f"⚠️  CCM3 virtual environment activation failed: {e}")
    print("Continuing with current environment...")

import time
import threading
import argparse
import os
import json
import random
import numpy as np
from typing import Dict, Optional, List
import joblib
import signal
import sys

# Existing harmonic components (unchanged)
from listener.jhs_listener_core import DriftListener, Event
from memory.memory_buffer import MemoryBuffer
from memory.clustering import MusicalClustering
from memory.polyphonic_audio_oracle import PolyphonicAudioOracle
from agent.ai_agent import AIAgent
from agent.phrase_generator import PhraseGenerator
from agent.decision_explainer import MusicalDecisionExplainer
from mapping.feature_mapper import FeatureMapper
from midi_io.midi_output import MIDIOutput
from midi_io.mpe_midi_output import MPEMIDIOutput
from core.logger import PerformanceLogger
from performance_timeline_manager import PerformanceTimelineManager, PerformanceConfig

# New rhythmic components
from rhythmic_engine.audio_file_learning.lightweight_rhythmic_analyzer import LightweightRhythmicAnalyzer
from rhythmic_engine.memory.rhythm_oracle import RhythmOracle
from rhythmic_engine.agent.rhythmic_behavior_engine import RhythmicBehaviorEngine

# New correlation components
from correlation_engine.unified_decision_engine import UnifiedDecisionEngine, CrossModalContext, MusicalContext

# Hybrid detection components
from listener.hybrid_detector import HybridDetector, DetectionResult

# Visualization system
try:
    from visualization import VisualizationManager
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False


# IRCAM Phase 2.1: Musical Pause Detection
class MusicalPauseDetector:
    """
    Detects musical pauses based on onset gaps and RMS level
    Based on NIME2023 research: "Give space to the machine" requires pause awareness
    """
    
    def __init__(self):
        self.onset_times = []
        self.silence_threshold_db = -50  # dB threshold for silence
        self.pause_duration_threshold = 0.5  # seconds - minimum gap to be considered a pause
        self.max_history = 10.0  # Keep last 10 seconds of onset history
    
    def add_onset(self, timestamp: float):
        """Track an onset event"""
        self.onset_times.append(timestamp)
        # Clean old onsets (keep only recent history)
        self.onset_times = [t for t in self.onset_times 
                           if timestamp - t < self.max_history]
    
    def detect_pause(self, current_time: float, rms_db: float) -> bool:
        """
        Returns True if currently in a musical pause
        
        Args:
            current_time: Current timestamp
            rms_db: Current RMS level in dB
            
        Returns:
            True if in a pause (gap + quietness)
        """
        if not self.onset_times:
            return True  # No recent onsets = pause
        
        time_since_last_onset = current_time - self.onset_times[-1]
        is_quiet = rms_db < self.silence_threshold_db
        
        return (time_since_last_onset > self.pause_duration_threshold 
                and is_quiet)
    
    def time_since_last_onset(self, current_time: float) -> float:
        """How long since last onset"""
        if not self.onset_times:
            return 999.0
        return current_time - self.onset_times[-1]


# IRCAM Phase 2.2: Phrase Boundary Detection
class PhraseBoundaryDetector:
    """
    Detects phrase boundaries based on onset density
    Based on NIME2023: Better to respond at phrase boundaries, not mid-phrase
    """
    
    def __init__(self, window_size: float = 2.0):
        self.window_size = window_size  # Analysis window in seconds
        self.onset_times = []
        self.density_threshold_high = 2.0  # onsets/second for "actively in phrase"
        self.density_threshold_low = 0.5   # onsets/second for "at boundary"
        self.max_history = 10.0  # Keep last 10 seconds of onset history
    
    def add_onset(self, timestamp: float):
        """Track an onset event"""
        self.onset_times.append(timestamp)
        # Clean old onsets
        self.onset_times = [t for t in self.onset_times 
                           if timestamp - t < self.max_history]
    
    def get_onset_density(self, current_time: float) -> float:
        """Calculate onsets per second in recent window"""
        recent = [t for t in self.onset_times 
                 if current_time - t < self.window_size]
        return len(recent) / self.window_size
    
    def is_phrase_boundary(self, current_time: float) -> bool:
        """Returns True if at a phrase boundary"""
        density = self.get_onset_density(current_time)
        return density < self.density_threshold_low
    
    def is_in_phrase(self, current_time: float) -> bool:
        """Returns True if actively in a phrase"""
        density = self.get_onset_density(current_time)
        return density > self.density_threshold_high
    
    def get_state(self, current_time: float) -> str:
        """
        Returns phrase state: 'in_phrase', 'transition', or 'boundary'
        
        Args:
            current_time: Current timestamp
            
        Returns:
            State string indicating phrase position
        """
        density = self.get_onset_density(current_time)
        if density > self.density_threshold_high:
            return 'in_phrase'
        elif density > self.density_threshold_low:
            return 'transition'
        else:
            return 'boundary'


class EnhancedDriftEngineAI:
    """
    Enhanced Drift Engine AI with rhythmic analysis
    Combines harmonic and rhythmic intelligence
    """
    
    def __init__(self, midi_port: Optional[str] = None, input_device: Optional[int] = None, 
                 enable_rhythmic: bool = True, enable_mpe: bool = True, performance_duration: int = 0,
                 enable_hybrid_perception: bool = False, enable_wav2vec: bool = False,
                 enable_live_training: bool = False,
                 wav2vec_model: str = "facebook/wav2vec2-base", use_gpu: bool = False,
                 debug_decisions: bool = False, enable_visualization: bool = False):
        # Core harmonic components (unchanged)
        self.listener: Optional[DriftListener] = None
        self.memory_buffer = MemoryBuffer()
        
        # Initialize clustering with default parameters (will be updated from saved model)
        # These are just defaults - the actual values come from the trained model
        self.clustering = None  # Will be initialized after loading model config
        self.ai_agent = None  # Will be initialized after visualization_manager is set up
        self.feature_mapper = FeatureMapper()
        
        # Musical decision explainer (for transparency and trust)
        self.decision_explainer = MusicalDecisionExplainer(enable_console_output=debug_decisions)
        self.debug_decisions = debug_decisions
        
        # Hybrid detection system
        self.hybrid_detector = HybridDetector(sample_rate=44100, fingerprint_duration=15.0)
        
        # NEW: Hybrid perception (ratio + symbolic + chroma OR Wav2Vec)
        self.enable_hybrid_perception = enable_hybrid_perception
        self.enable_wav2vec = enable_wav2vec
        self.wav2vec_chord_classifier = None  # Optional chord classifier
        
        # Live training control
        self.enable_live_training = enable_live_training
        
        if enable_hybrid_perception:
            from listener.hybrid_perception import HybridPerceptionModule
            self.hybrid_perception = HybridPerceptionModule(
                vocabulary_size=64,
                enable_ratio_analysis=True,  # ALWAYS enable ratio analysis for chord detection
                enable_symbolic=True,  # Enable symbolic quantization for gesture tokens
                enable_wav2vec=enable_wav2vec,
                wav2vec_model=wav2vec_model,
                use_gpu=use_gpu
            )
            if enable_wav2vec:
                print(f"🎵 Wav2Vec perception enabled: {wav2vec_model}")
                print(f"   GPU: {'Yes (MPS/CUDA)' if use_gpu else 'CPU'}")
                print(f"   Features: 768D neural encoding")
                
                # Try to load Wav2Vec chord classifier (for human-readable labels)
                # DISABLED: Not critical for operation, ratio analyzer provides chord detection
                self.wav2vec_chord_classifier = None
                # try:
                #     from hybrid_training.wav2vec_chord_classifier import Wav2VecChordClassifier
                #     classifier_path = "models/wav2vec_chord_classifier.pt"
                #     self.wav2vec_chord_classifier = Wav2VecChordClassifier.load(classifier_path)
                #     print(f"   🎹 Chord classifier loaded: {classifier_path}")
                # except Exception as e:
                #     print(f"   ⚠️  Chord classifier not available: {e}")
                #     import traceback
                #     traceback.print_exc()  # Show full error for debugging
            else:
                print("🔬 Hybrid perception enabled - ratio + chroma features")
        else:
            self.hybrid_perception = None
        
        # MIDI output system - Multiple outputs for voice separation
        self.enable_mpe = enable_mpe
        self.midi_outputs = {
            'melodic': None,
            'bass': None,
            'rhythm': None
        }
        self.mpe_midi_outputs = {
            'melodic': None,
            'bass': None,
            'rhythm': None
        }
        
        # NEW: Enable rhythmic components for full dual-modal perception
        self.enable_rhythmic = True  # RE-ENABLED
        
        # Onset buffer for rational rhythm analysis
        self.onset_buffer = []
        self.max_onset_buffer = 12  # Medium buffer: ~3-5 seconds
        self.onset_counter = 0
        self.re_analysis_interval = 2  # Re-analyze every 2-3 onsets
        
        # Latest rhythm analysis result
        self.latest_rhythm_ratio = None
        self.latest_barlow_complexity = None
        self.latest_deviation_polarity = None
        
        # Rhythm analyzer will be initialized in start() after listener is created
        self.rhythmic_analyzer = None
        self.ratio_analyzer = None
        
        # Rhythm oracle and agent (not used for now, but initialized for compatibility)
        self.rhythm_oracle = None
        self.rhythmic_agent = None
        
        # Unified decision engine for cross-modal intelligence
        self.unified_decision_engine = UnifiedDecisionEngine()
        print("🔗 Unified decision engine enabled")
        
        # Performance logging
        self.performance_logger = PerformanceLogger(log_dir="logs")
        
        # Visualization system (optional)
        self.visualization_manager = None
        if enable_visualization:
            if VISUALIZATION_AVAILABLE:
                try:
                    self.visualization_manager = VisualizationManager()
                    self.visualization_manager.start()
                    print("🎨 Visualization system enabled (5 viewports)")
                except Exception as e:
                    print(f"⚠️  Visualization initialization failed: {e}")
                    self.visualization_manager = None
            else:
                print("⚠️  PyQt5 not available, visualization disabled")
        
        # Initialize AI Agent now that visualization_manager is available
        self.ai_agent = AIAgent(visualization_manager=self.visualization_manager)
        
        # Performance arc system
        self.timeline_manager: Optional[PerformanceTimelineManager] = None
        self.performance_duration = performance_duration
        
        # Status bar state
        self._last_status_update = 0.0
        self._status_update_interval = 0.1  # Update 10 times per second
        
        # Conversation logging
        self._last_conversation_log = 0.0
        self._conversation_log_interval = 1.0  # Log every second
        self._conversation_log_file = None
        
        # Terminal output logging
        self._terminal_log_file = None
        
        # Autonomous generation state
        self.autonomous_generation_enabled = True
        self.last_autonomous_generation = 0.0
        self.autonomous_interval_base = 3.0  # Base interval (seconds) for autonomous generation - more listening
        self.human_activity_level = 0.0  # 0.0 = silence, 1.0 = very active
        self.last_human_event_time = 0.0
        self.silence_timeout = 1.5  # Seconds of silence before switching to autonomous mode
        self.was_in_autonomous_mode = False  # Track mode transitions
        
        # Voice-specific autonomous behavior
        self.bass_accompaniment_probability = 0.2  # Probability bass plays during human activity (reduced for sparsity)
        self.melody_silence_when_active = True  # Melody stays quiet when human is active
        
        # Track phrases that started in autonomous mode (let them complete)
        self.autonomous_phrase_ids = set()
        
        # Load GPT-OSS behavioral insights (if available)
        self.gpt_oss_silence_strategy = None
        self.gpt_oss_role_development = None
        
        # IRCAM Phase 1.1: Self-output filtering to prevent learning from own MIDI
        self.own_output_tracker = {
            'recent_notes': [],  # List of (note, timestamp, velocity)
            'max_age_seconds': 2.0,
            'enabled': True
        }
        
        # IRCAM Phase 2: Musical intelligence detectors
        self.pause_detector = MusicalPauseDetector()
        self.phrase_detector = PhraseBoundaryDetector()
        self.in_pause = False
        self.phrase_state = 'boundary'
        self.last_autonomous_time = 0
        
        # IRCAM Phase 3: Behavior controller for interaction modes
        from agent.behaviors import BehaviorController, BehaviorMode
        self.behavior_controller = BehaviorController()
        
        # ML Chord Detection
        self.ml_chord_model = None
        self.ml_chord_scaler = None
        self.ml_chord_enabled = False
        self.last_ml_chord_time = 0
        self.ml_chord_interval = 2.0  # Reduced response frequency - 2 seconds between ML predictions
        
        # Configuration - Multiple IAC drivers for voice separation
        self.midi_ports = {
            'melodic': midi_port or "IAC Driver Melody Channel",
            'bass': "IAC Driver Bass",
            'rhythm': "IAC Driver Rhythm Channel"
        }
        self.input_device = input_device
        
        # Persistence configuration
        self.data_dir = "ai_learning_data"
        self.memory_file = os.path.join(self.data_dir, "musical_memory.json")
        self.clustering_file = os.path.join(self.data_dir, "polyphonic_audio_oracle_model.json")
        
        # Rhythmic persistence
        if self.enable_rhythmic:
            self.rhythmic_file = os.path.join(self.data_dir, "rhythmic_patterns.json")
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize timeline manager if performance duration is specified
        if self.performance_duration > 0:
            self._initialize_timeline_manager()
        
        # Initialize conversation logging
        self._initialize_conversation_logging()
        
        # Initialize MIDI output system
        if self.enable_mpe:
            print("🎹 MPE MIDI mode enabled")
        else:
            print("🎵 Standard MIDI mode enabled")
        
        # State
        self.running = False
        self.main_thread: Optional[threading.Thread] = None
        
        # Statistics
        self.stats = {
            'events_processed': 0,
            'decisions_made': 0,
            'notes_sent': 0,
            'uptime': 0.0,
            'last_event_time': 0.0,
            'rhythmic_decisions': 0,
            'combined_decisions': 0,
            'harmonic_patterns_detected': 0,
            'polyphonic_patterns_detected': 0
        }
        
        self.start_time = time.time()
    
    def _initialize_timeline_manager(self):
        """Initialize performance timeline manager"""
        try:
            config = PerformanceConfig(
                duration_minutes=self.performance_duration,
                arc_file_path="ai_learning_data/itzama_performance_arc.json",  # Default arc
                engagement_profile="balanced",
                silence_tolerance=5.0,
                adaptation_rate=0.1
            )
            self.timeline_manager = PerformanceTimelineManager(config)
            print(f"🎭 Performance timeline initialized: {self.performance_duration} minutes")
        except Exception as e:
            print(f"⚠️ Failed to initialize timeline manager: {e}")
            self.timeline_manager = None
    
    def _initialize_conversation_logging(self):
        """Initialize conversation logging for analysis"""
        import datetime
        import sys
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"logs/conversation_{timestamp}.csv"
        terminal_log_filename = f"logs/terminal_{timestamp}.log"
        
        try:
            os.makedirs("logs", exist_ok=True)
            self._conversation_log_file = open(log_filename, 'w')
            # Write header
            self._conversation_log_file.write("timestamp,elapsed_time,event_type,voice,pitch_hz,midi_note,rms_db,activity_level,mode,mpe_pitchbend,mpe_pressure,mpe_timbre,note_velocity\n")
            self._conversation_log_file.flush()
            print(f"📊 Conversation logging: {log_filename}")
            
            # Initialize terminal output logging
            self._terminal_log_file = open(terminal_log_filename, 'w', buffering=1)  # Line buffered
            
            # Note: We DON'T replace builtins.print because it breaks numba/librosa imports
            # Instead, important events are already logged via conversation_log_file
            # Terminal output can be captured with: python script.py 2>&1 | tee output.log
            
            print(f"📝 Terminal logging: {terminal_log_filename} (passive)")
            
        except Exception as e:
            print(f"⚠️ Could not initialize conversation logging: {e}")
            self._conversation_log_file = None
            self._terminal_log_file = None
    
    def start(self) -> bool:
        """Start the Enhanced Drift Engine AI system"""
        print("🎵 Starting Enhanced Drift Engine AI...")
        
        try:
            # Load previous learning data
            self._load_learning_data()
            
            # Start multiple MIDI outputs for voice separation
            if self.enable_mpe:
                # Initialize MPE outputs for each voice
                for voice_type, port_name in self.midi_ports.items():
                    if voice_type in ['melodic', 'bass']:  # Only melodic and bass for now
                        self.mpe_midi_outputs[voice_type] = MPEMIDIOutput(port_name, enable_mpe=True)
                        if not self.mpe_midi_outputs[voice_type].start():
                            print(f"❌ Failed to start MPE MIDI output for {voice_type}")
                            return False
                        print(f"✅ MPE MIDI output started for {voice_type}: {port_name}")
            else:
                # Initialize standard MIDI outputs for each voice
                for voice_type, port_name in self.midi_ports.items():
                    if voice_type in ['melodic', 'bass']:  # Only melodic and bass for now
                        self.midi_outputs[voice_type] = MIDIOutput(port_name)
                        if not self.midi_outputs[voice_type].start():
                            print(f"❌ Failed to start MIDI output for {voice_type}")
                            return False
                        print(f"✅ MIDI output started for {voice_type}: {port_name}")
            
            # Start listener
            self.listener = DriftListener(
                ref_fn=self._get_reference_frequency,
                a4_fn=lambda: 440.0,
                device=self.input_device
            )
            
            # Initialize rhythmic analyzers now that listener is available
            if self.enable_rhythmic:
                from rhythmic_engine.ratio_analyzer import RatioAnalyzer
                
                # Initialize lightweight onset detector
                self.rhythmic_analyzer = LightweightRhythmicAnalyzer(
                    sample_rate=self.listener.sr,  # Note: DriftListener uses .sr not .sample_rate
                    hop_length=1024
                )
                
                # Initialize Brandtsegg ratio analyzer
                self.ratio_analyzer = RatioAnalyzer(
                    complexity_weight=0.5,
                    deviation_weight=0.5,
                    simplify=True,
                    div_limit=4
                )
                
                print("🥁 Rhythmic analysis ENABLED (Brandtsegg RatioAnalyzer)")
            
            # Set up event callback
            self.listener.start(self._on_audio_event)
            
            # Connect hybrid detector to instrument classifier
            self.hybrid_detector.set_instrument_classifier(self.listener)
            
            # Start timeline manager if available
            if self.timeline_manager:
                self.timeline_manager.start_performance()
                print(f"🎭 Performance arc active: {self.performance_duration} minutes")
            
            # Start main loop
            self.running = True
            self.main_thread = threading.Thread(target=self._main_loop, daemon=True)
            self.main_thread.start()
            
            print("✅ Enhanced Drift Engine AI started successfully!")
            if self.enable_mpe:
                for voice_type, output in self.mpe_midi_outputs.items():
                    if output:
                        print(f"🎹 MPE {voice_type.capitalize()} Output: {output.port.name if output.port else 'None'}")
            else:
                for voice_type, output in self.midi_outputs.items():
                    if output:
                        print(f"📡 {voice_type.capitalize()} Output: {output.port.name if output.port else 'None'}")
            print(f"🎤 Audio Input: Device {self.input_device if self.input_device else 'Default'}")
            if self.enable_rhythmic:
                print("🥁 Rhythmic Analysis: Enabled")
            else:
                print("🎵 Rhythmic Analysis: Disabled")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to start Enhanced Drift Engine AI: {e}")
            return False
    
    def _is_own_output(self, event_data: Dict, current_time: float) -> bool:
        """
        Check if detected audio matches recently sent MIDI output
        IRCAM Phase 1.1: Prevent feedback loop degrading learning quality
        
        Args:
            event_data: Event data dictionary
            current_time: Current timestamp
            
        Returns:
            True if this event matches our own output
        """
        if not self.own_output_tracker['enabled']:
            return False
        
        # Clean old entries (older than max_age_seconds)
        self.own_output_tracker['recent_notes'] = [
            (note, t, vel) for note, t, vel in self.own_output_tracker['recent_notes']
            if current_time - t < self.own_output_tracker['max_age_seconds']
        ]
        
        # Check if detected MIDI matches recent output
        detected_midi = event_data.get('midi', 0)
        for note, sent_time, vel in self.own_output_tracker['recent_notes']:
            # Match if same note and within 1 second of sending
            if note == detected_midi and (current_time - sent_time) < 1.0:
                return True
        
        return False
    
    def _calculate_adaptive_interval(self) -> float:
        """
        Calculate autonomous generation interval based on musical context
        IRCAM Phase 2.3: Adaptive timing based on pause and phrase state
        
        Returns:
            Interval in seconds
        """
        # Fast response in pauses (fill silence appropriately)
        if self.in_pause:
            return 1.0
        
        # Slow during active phrases (give space to human)
        if self.phrase_state == 'in_phrase':
            return 5.0
        
        # Medium at boundaries (good time to respond)
        if self.phrase_state == 'boundary':
            return 2.0
        
        # Transition state - use base interval
        return self.autonomous_interval_base
    
    def _handle_mode_change_cc(self, cc_value: int):
        """
        Switch behavior mode via MIDI CC
        IRCAM Phase 3: Allow runtime mode switching via MIDI controller
        
        Args:
            cc_value: MIDI CC value (0-127)
        """
        from agent.behaviors import BehaviorMode
        
        if cc_value < 42:
            self.behavior_controller.set_mode(BehaviorMode.SHADOW)
        elif cc_value < 85:
            self.behavior_controller.set_mode(BehaviorMode.MIRROR)
        else:
            self.behavior_controller.set_mode(BehaviorMode.COUPLE)
    
    def stop(self):
        """Stop the Enhanced Drift Engine AI system"""
        print("🛑 Stopping Enhanced Drift Engine AI...")
        
        self.running = False
        
        # Close conversation log
        if self._conversation_log_file:
            self._conversation_log_file.close()
            print("📊 Conversation log saved")
        
        if self._terminal_log_file:
            self._terminal_log_file.close()
            print("📝 Terminal log saved")
        
        # Save learning data before stopping
        self._save_learning_data()
        
        # Stop listener
        if self.listener:
            self.listener.stop()
        
        # Stop all MIDI outputs
        for voice_type, output in self.midi_outputs.items():
            if output:
                output.stop()
        for voice_type, output in self.mpe_midi_outputs.items():
            if output:
                output.stop()
        
        # Close performance logger
        if hasattr(self.performance_logger, 'close'):
            self.performance_logger.close()
        
        # Close visualization system
        if self.visualization_manager:
            self.visualization_manager.close()
            print("🎨 Visualization system closed")
        
        # Wait for main thread (only if we're NOT in it)
        if self.main_thread and self.main_thread.is_alive():
            if threading.current_thread() != self.main_thread:
                self.main_thread.join(timeout=2.0)
            # If we ARE in the main thread, it will exit naturally after this returns
        
        print("✅ Enhanced Drift Engine AI stopped")
    
    def _on_audio_event(self, *args):
        """Enhanced audio event handler with rhythmic analysis"""
        if not self.running:
            return
        
        # Handle different callback patterns
        if len(args) == 1 and isinstance(args[0], Event):
            event = args[0]
        elif len(args) == 9:
            # Handle the old callback pattern: (None, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False)
            # This indicates silence, so we can skip processing
            return
        else:
            return
        
        current_time = time.time()
        
        # Convert Event to dict
        event_data = event.to_dict()
        
        # Track human activity for autonomous generation adjustment
        self._track_human_activity(event_data, current_time)
        
        # Log human input (every second)
        self._log_conversation_input(event_data, current_time)
        
        # Update live status bar (updates in place at bottom of terminal)
        # Note: We'll update status bar AFTER hybrid extraction so it can show hybrid info
        # Moved below to line after hybrid extraction
        
        # Extract musical metadata for enhanced pattern detection
        musical_metadata = self._extract_musical_metadata(event_data)
        event_data.update(musical_metadata)
        
        # NEW: Extract hybrid perception features (if enabled)
        if self.hybrid_perception:
            try:
                # Get raw audio buffer from event (if available)
                audio_buffer = event.raw_audio if hasattr(event, 'raw_audio') and event.raw_audio is not None else None
                
                if audio_buffer is not None and len(audio_buffer) > 0:
                    # Extract hybrid features (pass detected F0 for accurate chord root)
                    hybrid_result = self.hybrid_perception.extract_features(
                        audio_buffer, 
                        self.listener.sr, 
                        current_time,
                        detected_f0=event.f0  # Pass actual detected frequency
                    )
                    
                    # Store hybrid analysis in event data
                    event_data['hybrid_consonance'] = hybrid_result.consonance
                    event_data['hybrid_chroma'] = hybrid_result.chroma.tolist()
                    event_data['hybrid_active_pcs'] = hybrid_result.active_pitch_classes.tolist()
                    
                    # Add ratio analysis if available
                    if hybrid_result.ratio_analysis:
                        event_data['hybrid_ratio_chord'] = hybrid_result.ratio_analysis.chord_match['type']
                        event_data['hybrid_ratio_confidence'] = hybrid_result.ratio_analysis.chord_match['confidence']
                        event_data['hybrid_ratio_fundamental'] = float(hybrid_result.ratio_analysis.fundamental)
                        # Store actual ratios for visualization
                        if hasattr(hybrid_result.ratio_analysis, 'ratios') and len(hybrid_result.ratio_analysis.ratios) > 0:
                            # Store first ratio as a list [num, den] for display
                            # Use simplified ratios if available, otherwise approximate from decimal
                            if hasattr(hybrid_result.ratio_analysis, 'simplified_ratios') and len(hybrid_result.ratio_analysis.simplified_ratios) > 0:
                                # Use first simplified ratio (tuple of ints)
                                ratio_tuple = hybrid_result.ratio_analysis.simplified_ratios[0]
                                event_data['hybrid_rhythm_ratio'] = list(ratio_tuple)  # [numerator, denominator]
                            else:
                                # Approximate from decimal ratio
                                ratio_val = hybrid_result.ratio_analysis.ratios[0]
                                event_data['hybrid_rhythm_ratio'] = [int(ratio_val * 2), 2]  # Simple approximation
                        # Store Barlow complexity (approximate from ratio complexity)
                        # Barlow = sum of prime factors of numerator and denominator
                        if 'hybrid_rhythm_ratio' in event_data:
                            num, den = event_data['hybrid_rhythm_ratio']
                            # Simple proxy: larger numbers = more complex
                            barlow_approx = (num + den) / 2.0  # Simplified approximation
                            event_data['hybrid_barlow_complexity'] = barlow_approx
                    
                    # Replace standard features with hybrid features for AudioOracle
                    event_data['features'] = hybrid_result.features.tolist()
                    
                    # Store Wav2Vec features for chord classification (if available)
                    # When using Wav2Vec mode, the main features ARE the Wav2Vec features (768D)
                    if self.enable_wav2vec and hybrid_result.features is not None and len(hybrid_result.features) == 768:
                        event_data['hybrid_wav2vec_features'] = hybrid_result.features  # Use numpy array directly for classifier
                    
                    # IRCAM Phase 1.2: Extract symbolic token for AudioOracle pattern matching
                    if hasattr(hybrid_result, 'symbolic_token') and hybrid_result.symbolic_token is not None:
                        event_data['gesture_token'] = hybrid_result.symbolic_token
                        
                        # Emit pattern matching visualization (simplified - full scoring happens in agent)
                        if self.visualization_manager:
                            # Estimate a pattern match score based on consonance (placeholder logic)
                            match_score = hybrid_result.consonance * 100 if hybrid_result.consonance else 50.0
                            self.visualization_manager.emit_pattern_match(
                                score=match_score,
                                state_id=0,  # Would need actual AudioOracle state ID
                                gesture_token=hybrid_result.symbolic_token,
                                context={'consonance': hybrid_result.consonance}
                            )
                        
                        # Debug log (first 10 times)
                        if not hasattr(self, '_gesture_token_count'):
                            self._gesture_token_count = 0
                        if self._gesture_token_count < 10:
                            print(f"🎵 Gesture token: {hybrid_result.symbolic_token}")
                            self._gesture_token_count += 1
                    
            except Exception as e:
                # Debug: print first few errors to help diagnose
                if not hasattr(self, '_hybrid_error_count'):
                    self._hybrid_error_count = 0
                if self._hybrid_error_count < 3:
                    import traceback
                    print(f"\n⚠️ Hybrid perception error: {e}")
                    print("Traceback:")
                    traceback.print_exc()
                    self._hybrid_error_count += 1
        
        # Update status bar NOW (after hybrid extraction)
        self._update_status_bar(event, event_data)
        
        # Emit audio analysis visualization event (ALWAYS emit, even without full data)
        if self.visualization_manager:
            # Get raw audio buffer if available
            audio_buffer = event.raw_audio if hasattr(event, 'raw_audio') and event.raw_audio is not None else None
            
            # Create simple waveform from RMS if no buffer available
            if audio_buffer is None:
                import numpy as np
                rms = event_data.get('rms', 0.0)
                # Create fake waveform for visualization (sine wave with amplitude = RMS)
                audio_buffer = np.sin(np.linspace(0, 10*np.pi, 1024)) * rms
            
            # Use actual rhythm ratio from Brandtsegg (not placeholder)
            ratio = None
            complexity = None
            if 'rhythm_ratio' in event_data:
                # Convert single duration value to ratio format [numerator, denominator]
                ratio_val = event_data['rhythm_ratio']
                ratio = [int(ratio_val), 1]  # Simplified representation
            
            if 'barlow_complexity' in event_data:
                complexity = event_data['barlow_complexity']
            
            # Get consonance (try multiple sources)
            consonance = event_data.get('hybrid_consonance', 
                                       event_data.get('consonance', 0.5))
            
            self.visualization_manager.emit_audio_analysis(
                waveform=audio_buffer,
                onset=event_data.get('onset', False),
                ratio=ratio,
                consonance=consonance,
                timestamp=current_time,
                complexity=complexity  # Now includes REAL Barlow complexity from Brandtsegg
            )
            
            # Emit timeline event for human input
            print(f"🎨 TIMELINE: Emitting human_input event at {current_time}")  # DEBUG
            self.visualization_manager.emit_timeline_update('human_input', timestamp=current_time)
            
            # Emit pattern matching even without gesture token
            # Use pitch as a simple "token" for visualization
            simple_token = int(event_data.get('f0', 0) % 256)  # Convert frequency to 0-255
            
            # Debug: Print first few emissions to verify they're happening
            if not hasattr(self, '_viz_emit_count'):
                self._viz_emit_count = 0
            if self._viz_emit_count < 5:
                print(f"🎨 VIZ: Emitting events (token={simple_token}, consonance={consonance:.2f})")
                self._viz_emit_count += 1
            
            self.visualization_manager.emit_pattern_match(
                score=consonance * 100 if consonance else 50.0,
                state_id=self.stats.get('notes_sent', 0),  # Use notes sent as state approximation
                gesture_token=simple_token,
                context={'consonance': consonance, 'pitch': event_data.get('f0', 0)}
            )
        
        # Log audio analysis
        self.performance_logger.log_audio_analysis(
            instant_pitch=event_data.get('f0', 0.0),
            avg_pitch=event_data.get('f0', 0.0),  # Using same value for simplicity
            onset_detected=event_data.get('onset', False),
            onset_rate=1.0 if event_data.get('onset', False) else 0.0,
            activity_level=min(1.0, max(0.0, (event_data.get('rms_db', -80) + 80) / 80)),
            energy_level=event_data.get('rms_db', -80)
        )
        
        # IRCAM Phase 1.1: Check if this is our own output (prevent feedback loop)
        if self._is_own_output(event_data, current_time):
            self._update_status_bar(event, event_data)  # Still show it in status
            return  # Don't learn from it
        
        # Only human input reaches here
        # Add to memory buffer
        self.memory_buffer.add_moment(event_data)
        
        # Learn musical patterns with PolyphonicAudioOracle (if enabled)
        # Pass event_data directly for polyphonic feature extraction
        if self.enable_live_training:
            self.clustering.add_sequence([event_data])
        
        # Periodic pattern detection (every 10 events for efficiency)
        if self.stats['events_processed'] % 10 == 0:
            try:
                # Detect harmonic patterns
                harmonic_patterns = self.clustering.find_harmonic_patterns()
                if harmonic_patterns:
                    self.stats['harmonic_patterns_detected'] = len(harmonic_patterns)
                
                # Detect polyphonic patterns
                polyphonic_patterns = self.clustering.find_polyphonic_patterns()
                if polyphonic_patterns:
                    self.stats['polyphonic_patterns_detected'] = len(polyphonic_patterns)
                    
            except Exception as e:
                print(f"⚠️ Pattern detection error: {e}")
        
        # NEW: Rhythmic analysis (if enabled)
        rhythmic_context = None
        if self.enable_rhythmic and self.rhythmic_analyzer:
            try:
                # Convert event data to audio frame for rhythmic analysis
                audio_frame = self._event_to_audio_frame(event_data)
                rhythmic_context = self.rhythmic_analyzer.analyze_live_rhythm(audio_frame)
            except Exception as e:
                print(f"⚠️ Rhythmic analysis error: {e}")
                rhythmic_context = None
        
        # IRCAM Phase 2: Track onsets for pause and phrase detection
        if event_data.get('onset', False):
            self.pause_detector.add_onset(current_time)
            self.phrase_detector.add_onset(current_time)
            
            # NEW: Real-time rhythm analysis using Brandtsegg
            if self.enable_rhythmic and hasattr(self, 'ratio_analyzer'):
                # Add to onset buffer
                self.onset_buffer.append(current_time)
                
                # Trim buffer to max size
                if len(self.onset_buffer) > self.max_onset_buffer:
                    self.onset_buffer.pop(0)
                
                # Increment counter for re-analysis trigger
                self.onset_counter += 1
                
                # Perform Brandtsegg rational analysis when buffer is sufficient and counter triggers
                if len(self.onset_buffer) >= 4 and self.onset_counter >= self.re_analysis_interval:
                    try:
                        # Analyze IOI ratios using Brandtsegg
                        onset_times = np.array(self.onset_buffer)
                        rhythm_result = self.ratio_analyzer.analyze(onset_times)
                        
                        if rhythm_result:
                            # Extract current position in duration pattern
                            # (Use most recent onset's characteristics)
                            pattern_idx = -1  # Last position in pattern
                            
                            self.latest_rhythm_ratio = rhythm_result['duration_pattern'][pattern_idx] if len(rhythm_result['duration_pattern']) > 0 else 1
                            self.latest_barlow_complexity = rhythm_result['complexity']
                            self.latest_deviation_polarity = rhythm_result['deviation_polarity'][pattern_idx] if len(rhythm_result['deviation_polarity']) > 0 else 0
                            
                            # Store in event_data for downstream use
                            event_data['rhythm_ratio'] = self.latest_rhythm_ratio
                            event_data['barlow_complexity'] = self.latest_barlow_complexity
                            event_data['deviation_polarity'] = self.latest_deviation_polarity
                            event_data['rhythm_subdiv_tempo'] = rhythm_result['tempo']
                            event_data['rhythm_pulse'] = rhythm_result['pulse']
                            
                    except Exception as e:
                        # Silently handle rhythm analysis errors (don't clutter terminal)
                        pass
                    
                    # Reset counter
                    self.onset_counter = 0
                
                # Apply latest rhythm features even if not re-analyzing this onset
                elif self.latest_rhythm_ratio is not None:
                    event_data['rhythm_ratio'] = self.latest_rhythm_ratio
                    event_data['barlow_complexity'] = self.latest_barlow_complexity
                    event_data['deviation_polarity'] = self.latest_deviation_polarity
        
        # Update pause and phrase state
        self.in_pause = self.pause_detector.detect_pause(
            current_time, 
            event_data.get('rms_db', -80)
        )
        self.phrase_state = self.phrase_detector.get_state(current_time)
        
        # Update statistics
        self.stats['events_processed'] += 1
        self.stats['last_event_time'] = current_time
        
        # Update performance timeline if available
        if self.timeline_manager:
            # Detect human activity based on event
            human_activity = event_data.get('rms_db', -80) > -60  # Simple activity detection
            instrument_detected = event_data.get('instrument', 'unknown')
            
            # Update timeline state
            self.timeline_manager.update_performance_state(
                human_activity=human_activity,
                instrument_detected=instrument_detected
            )
        
        # Hybrid detection: Check for target or instrument
        audio_buffer = self._event_to_audio_buffer(event_data)
        detection_result = self.hybrid_detector.detect(audio_buffer, event_data)
        
        # Update event data with detection result
        if detection_result.detection_type == "target":
            event_data['instrument'] = 'target'
            event_data['target_confidence'] = detection_result.confidence
            print(f"🎯 Target detected: {detection_result.target_description} (conf: {detection_result.confidence:.2f})")
        elif detection_result.detection_type == "instrument":
            event_data['instrument'] = detection_result.instrument
            # Instrument detection (silent - no terminal output)
        elif detection_result.detection_type == "learning":
            print(f"🎓 Learning progress: {detection_result.confidence:.1%} - {detection_result.target_description}")
            return  # Skip processing during learning
        
        # Process with AI agent - nuanced response based on activity
        # Melody: quiet when human is active (gives space)
        # Bass: sparse accompaniment when human is active
        time_since_last_human = current_time - self.last_human_event_time
        
        # IRCAM Phase 3+: Track human event for request-based generation
        if not hasattr(self, '_track_debug_logged'):
            has_behavior_engine = hasattr(self.ai_agent, 'behavior_engine') if self.ai_agent else False
            has_phrase_gen = hasattr(self.ai_agent.behavior_engine, 'phrase_generator') if has_behavior_engine and self.ai_agent.behavior_engine else False
            phrase_gen_exists = self.ai_agent.behavior_engine.phrase_generator is not None if has_phrase_gen else False
            print(f"🔍 MAIN: About to track event - ai_agent={self.ai_agent is not None}, behavior_engine={has_behavior_engine}, has_phrase_gen={has_phrase_gen}, phrase_gen={phrase_gen_exists}")
            print(f"🔍 MAIN: event_data has gesture_token: {'gesture_token' in event_data}, value: {event_data.get('gesture_token', 'MISSING')}")
            self._track_debug_logged = True
        
        if self.ai_agent and hasattr(self.ai_agent, 'behavior_engine') and self.ai_agent.behavior_engine and hasattr(self.ai_agent.behavior_engine, 'phrase_generator') and self.ai_agent.behavior_engine.phrase_generator:
            self.ai_agent.behavior_engine.phrase_generator.track_event(event_data, source='human')
        
        # Get activity multiplier from timeline manager (for 3-phase arc)
        activity_multiplier = 1.0
        if self.timeline_manager:
            guidance = self.timeline_manager.get_performance_guidance()
            activity_multiplier = guidance.get('activity_multiplier', 1.0)
        
        # Get all potential decisions (with activity multiplier for phrase length scaling)
        all_decisions = self.ai_agent.process_event(
            event_data, self.memory_buffer, self.clustering, activity_multiplier
        )
        
        # Filter decisions based on human activity and voice type
        is_autonomous = time_since_last_human >= self.silence_timeout and self.human_activity_level <= 0.3
        
        if is_autonomous:
            # Autonomous mode - both voices can play freely
            decisions = all_decisions
        else:
            # Human is actively playing - apply voice-specific filtering
            decisions = []
            import random
            for decision in all_decisions:
                if decision.voice_type == 'bass':
                    # Bass can play sparsely as accompaniment
                    if random.random() < self.bass_accompaniment_probability:
                        decisions.append(decision)
                elif decision.voice_type == 'melodic':
                    # Melody behavior depends on configuration
                    if not self.melody_silence_when_active:
                        # Allow sparse melody (20% probability)
                        if random.random() < 0.2:
                            decisions.append(decision)
                    # Otherwise: melody stays quiet to give space
        
        # Apply performance arc guidance (including fade-out)
        if self.timeline_manager:
            guidance = self.timeline_manager.get_performance_guidance()
            if not guidance['should_respond']:
                return  # Skip processing if timeline says we should be silent
            
            # Apply graceful fade-out in final minute
            fade_factor = self.timeline_manager.get_fade_out_factor(fade_duration=60.0)
            if random.random() > fade_factor:
                return  # Randomly skip responses during fade-out
            
            # Override behavior mode based on performance arc
            if guidance.get('behavior_mode'):
                from agent.behaviors import BehaviorMode
                for decision in decisions:
                    decision.mode = BehaviorMode(guidance['behavior_mode'])
        
        # Focus on melody/bass partnership - no rhythmic decisions for now
        final_decisions = decisions
        
        if final_decisions:
            self.stats['decisions_made'] += len(final_decisions)
            
            # Process each decision (melodic and bass)
            for decision in final_decisions:
                # Map to MIDI parameters
                midi_params = self.feature_mapper.map_features_to_midi(
                    event_data, {
                        'mode': decision.mode.value,
                        'confidence': decision.confidence,
                        'musical_params': decision.musical_params
                    },
                    decision.voice_type
                )
            
                # Log performance event
                self.performance_logger.log_system_event(
                    event_type="ai_decision",
                    voice=decision.voice_type,
                    trigger_reason=f"{decision.mode.value}_mode",
                    activity_level=decision.confidence,
                    phase="generation",
                    additional_data=f"note={midi_params.note},vel={midi_params.velocity}"
                )
                
                # Send MIDI note to appropriate voice-specific output
                voice_type = decision.voice_type
                
                if self.enable_mpe and self.mpe_midi_outputs.get(voice_type):
                    # MPE mode: route to specific voice output
                    # Calculate expressive parameters based on musical context
                    expression_level = self._calculate_expressive_intensity(decision, event_data)
                    timbre_variation = self._calculate_timbre_variation(decision, event_data)
                    pressure_sensitivity = self._calculate_pressure_sensitivity(decision, event_data)
                    
                    if self.mpe_midi_outputs[voice_type].send_note(midi_params, voice_type, 0.0, expression_level, 
                                                                  timbre_variation, pressure_sensitivity):
                        self.stats['notes_sent'] += 1
                        # IRCAM Phase 1.1: Track sent note to prevent learning from own output
                        self.own_output_tracker['recent_notes'].append(
                            (midi_params.note, current_time, midi_params.velocity)
                        )
                        # IRCAM Phase 3+: Track AI-generated event for request context
                        if self.ai_agent and hasattr(self.ai_agent, 'phrase_generator') and self.ai_agent.phrase_generator:
                            ai_event = {
                                'midi': midi_params.note,
                                'velocity': midi_params.velocity,
                                'voice_type': voice_type
                            }
                            self.ai_agent.phrase_generator.track_event(ai_event, source='ai')
                        # Log AI response
                        self._log_conversation_response(midi_params, voice_type, decision.mode.value, current_time,
                                                       expression_level, pressure_sensitivity, timbre_variation)
                    
                elif self.midi_outputs.get(voice_type):
                    # Standard MIDI mode: route to specific voice output
                    if self.midi_outputs[voice_type].send_note(midi_params, voice_type):
                        self.stats['notes_sent'] += 1
                        # IRCAM Phase 1.1: Track sent note to prevent learning from own output
                        self.own_output_tracker['recent_notes'].append(
                            (midi_params.note, current_time, midi_params.velocity)
                        )
                        # IRCAM Phase 3+: Track AI-generated event for request context
                        if self.ai_agent and hasattr(self.ai_agent, 'phrase_generator') and self.ai_agent.phrase_generator:
                            ai_event = {
                                'midi': midi_params.note,
                                'velocity': midi_params.velocity,
                                'voice_type': voice_type
                            }
                            self.ai_agent.phrase_generator.track_event(ai_event, source='ai')
                        # Log AI response
                        self._log_conversation_response(midi_params, voice_type, decision.mode.value, current_time,
                                                       0.0, 0.0, 0.0)
    
    def _calculate_expressive_intensity(self, decision, event_data: Dict) -> float:
        """Calculate expressive intensity based on musical context and AI decision"""
        base_expression = decision.confidence
        
        # Boost expression based on musical tension
        musical_tension = self._analyze_musical_tension(event_data)
        tension_boost = musical_tension * 0.3  # Up to 30% boost
        
        # Boost expression based on AI mode
        mode_boost = 0.0
        if decision.mode.value == 'lead':
            mode_boost = 0.2  # Lead mode gets more expression
        elif decision.mode.value == 'contrast':
            mode_boost = 0.15  # Contrast mode gets moderate boost
        elif decision.mode.value == 'imitate':
            mode_boost = 0.1   # Imitate mode gets subtle boost
        
        # Boost expression based on activity level
        activity_level = min(1.0, max(0.0, (event_data.get('rms_db', -80) + 80) / 80))
        activity_boost = activity_level * 0.2  # Up to 20% boost from activity
        
        # Combine all boosts
        total_expression = min(1.0, base_expression + tension_boost + mode_boost + activity_boost)
        
        return total_expression
    
    def _calculate_timbre_variation(self, decision, event_data: Dict) -> float:
        """Calculate timbre variation based on musical context"""
        # Base timbre from decision confidence
        base_timbre = decision.confidence
        
        # Add variation based on spectral content
        spectral_centroid = event_data.get('centroid', 0.5)
        timbre_variation = abs(spectral_centroid - 0.5) * 0.4  # Up to 20% variation
        
        # Add variation based on AI mode
        if decision.mode.value == 'contrast':
            timbre_variation += 0.15  # Contrast mode gets more timbre variation
        elif decision.mode.value == 'lead':
            timbre_variation += 0.1   # Lead mode gets moderate variation
        
        return min(1.0, base_timbre + timbre_variation)
    
    def _update_status_bar(self, event, event_data: Optional[Dict] = None):
        """Update live status bar with current pitch and system state - Clean minimal version"""
        current_time = time.time()
        
        # Throttle updates to avoid terminal spam
        if current_time - self._last_status_update < self._status_update_interval:
            return
        
        self._last_status_update = current_time
        
        # Build status line
        if event.f0 > 0:
            note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            note_name = note_names[event.midi % 12]
            octave = (event.midi // 12) - 1
            
            # Ensemble approach: Combine ratio-based + ML with weighted confidence
            chord_display = "---"
            chord_conf = 0.0
            consonance = 0.5  # Default consonance
            
            # Collect candidates with source weights
            candidates = []
            
            # 0. Wav2Vec-based chord classification (weight: 1.0x - neural perception)
            if self.wav2vec_chord_classifier and event_data and 'hybrid_wav2vec_features' in event_data:
                try:
                    wav2vec_features = event_data['hybrid_wav2vec_features']
                    chord_name, w2v_conf = self.wav2vec_chord_classifier.predict_with_confidence(wav2vec_features)
                    # Only add if confidence is reasonable (>15%)
                    if chord_name and w2v_conf > 0.15:
                        weighted_w2v = w2v_conf * 1.0  # Equal weight
                        candidates.append((chord_name, w2v_conf, weighted_w2v, 'W2V'))
                except Exception as ex:
                    # Silently skip if classification fails
                    pass
            
            # 1. Ratio-based detection (weight: 1.2x - good for clear triads/7ths)
            if event_data and self.enable_hybrid_perception:
                ratio_chord = event_data.get('hybrid_ratio_chord')
                ratio_conf = event_data.get('hybrid_ratio_confidence')
                consonance = event_data.get('hybrid_consonance', 0.5)
                
                if ratio_chord and ratio_conf and ratio_chord != 'unknown':
                    # Boost ratio confidence by 20% (it's very accurate when it works)
                    weighted_conf = min(1.0, ratio_conf * 1.2)
                    candidates.append((ratio_chord, ratio_conf, weighted_conf, 'ratio'))
            
            # 2. ML model (weight: 1.0x - trained on patterns)
            if self.ml_chord_enabled:
                ml_chord = self._predict_ml_chord(event, event_data)
                if ml_chord and '(' in ml_chord:
                    chord_name, conf_str = ml_chord.rsplit('(', 1)
                    ml_conf = float(conf_str.rstrip(')'))
                    # Only use ML if reasonably confident (>40%)
                    if ml_conf > 0.4:
                        candidates.append((chord_name, ml_conf, ml_conf, 'ML'))
            
            # 3. Harmonic context (weight: 0.8x - basic fallback)
            if event.harmonic_context and event.harmonic_context.current_chord:
                hc_conf = event.harmonic_context.confidence if hasattr(event.harmonic_context, 'confidence') else 0.5
                # Only use harmonic context if reasonably confident (>25% - lowered threshold)
                if hc_conf > 0.25:
                    weighted_hc = hc_conf * 0.8
                    candidates.append((event.harmonic_context.current_chord, hc_conf, weighted_hc, 'HC'))
            
            # Pick candidate with highest WEIGHTED confidence
            if candidates:
                best = max(candidates, key=lambda x: x[2])  # x[2] is weighted_conf
                chord_display = best[0]
                chord_conf = best[1]  # Display actual confidence, not weighted
            else:
                # No confident detection available
                chord_display = "---"
                chord_conf = 0.0
            
            # Always get consonance from hybrid if available
            if event_data and self.enable_hybrid_perception:
                consonance = event_data.get('hybrid_consonance', consonance)
            
            # Consonance score
            consonance_display = f"{consonance:.2f}" if event_data and self.enable_hybrid_perception else "---"
            
            # Recent MIDI notes (keep track of last few)
            if not hasattr(self, '_recent_midi_notes'):
                self._recent_midi_notes = []
            
            # Build minimal status line - one clean line
            status = (
                f"\r🎹 {note_name}{octave} | "
                f"CHORD: {chord_display:12s} ({chord_conf:>4.0%}) | "
                f"Consonance: {consonance_display} | "
                f"MIDI: {self.stats['notes_sent']:3d} notes | "
                f"Events: {self.stats['events_processed']:4d}"
            )
            
            # Print with ANSI codes to stay on same line
            print(status.ljust(120), end='', flush=True)
    
    def _calculate_pressure_sensitivity(self, decision, event_data: Dict) -> float:
        """Calculate pressure sensitivity based on musical context"""
        # Base pressure from decision confidence
        base_pressure = decision.confidence
        
        # Add sensitivity based on onset strength
        onset_strength = 1.0 if event_data.get('onset', False) else 0.0
        pressure_boost = onset_strength * 0.25  # Up to 25% boost for strong onsets
        
        # Add sensitivity based on AI mode
        if decision.mode.value == 'lead':
            pressure_boost += 0.2  # Lead mode gets more pressure sensitivity
        elif decision.mode.value == 'contrast':
            pressure_boost += 0.15  # Contrast mode gets moderate sensitivity
        
        # Add sensitivity based on note velocity
        velocity = decision.musical_params.get('velocity', 64)
        velocity_boost = (velocity - 64) / 127.0 * 0.2  # Up to 20% boost from velocity
        
        return min(1.0, base_pressure + pressure_boost + velocity_boost)
    
    def _analyze_musical_tension(self, event_data: Dict) -> float:
        """Analyze musical tension from event data"""
        # Calculate tension from harmonic content
        harmonic_tension = 0.0
        
        # Tension from pitch height (higher notes = more tension)
        f0 = event_data.get('f0', 440.0)
        if f0 > 0:
            # Normalize frequency to tension (440Hz = 0.5, 880Hz = 1.0)
            pitch_tension = min(1.0, (f0 - 220) / 660)  # 220Hz to 880Hz range
            harmonic_tension += pitch_tension * 0.3
        
        # Tension from spectral centroid (brighter = more tension)
        centroid = event_data.get('centroid', 0.5)
        spectral_tension = abs(centroid - 0.5) * 0.4
        harmonic_tension += spectral_tension
        
        # Tension from activity level
        activity_level = min(1.0, max(0.0, (event_data.get('rms_db', -80) + 80) / 80))
        harmonic_tension += activity_level * 0.3
        
        return min(1.0, harmonic_tension)
    
    def _calculate_chord_tension(self, event_data: Dict) -> float:
        """Calculate chord tension based on pitch content"""
        try:
            f0 = event_data.get('f0', 0)
            if f0 <= 0:
                return 0.5  # Neutral tension
            
            # Simple chord tension based on pitch height and stability
            # Higher pitches tend to create more tension
            pitch_tension = min(1.0, max(0.0, (f0 - 220) / 440))  # Normalize around A3 (220Hz)
            
            # Add some randomness for musical interest
            import random
            tension_variation = random.uniform(-0.1, 0.1)
            
            return min(1.0, max(0.0, pitch_tension + tension_variation))
        except Exception:
            return 0.5  # Default neutral tension
    
    def _extract_musical_metadata(self, event_data: Dict) -> Dict:
        """Extract musical metadata for enhanced pattern detection"""
        try:
            # Extract basic musical features
            f0 = event_data.get('f0', 0.0)
            rms_db = event_data.get('rms_db', -80)
            onset = event_data.get('onset', False)
            
            # Calculate chord tension based on pitch content
            chord_tension = self._calculate_chord_tension(event_data)
            
            # Calculate key stability (simplified - based on pitch stability)
            key_stability = 0.5  # Default neutral
            if f0 > 0:
                # Simple key stability based on pitch consistency
                pitch_stability = min(1.0, max(0.0, 1.0 - abs(f0 - 440.0) / 440.0))
                key_stability = pitch_stability
            
            # Calculate tempo (simplified - based on onset rate)
            tempo = 120.0  # Default tempo
            if onset:
                tempo = min(200.0, max(60.0, 120.0 + (rms_db + 40) * 2))  # Tempo varies with energy
            
            # Calculate rhythmic density
            rhythmic_density = min(1.0, max(0.0, (rms_db + 80) / 80))  # Based on energy level
            
            # Calculate rhythmic syncopation (simplified)
            rhythmic_syncopation = 0.0  # Default no syncopation
            if onset and rms_db > -60:  # Strong onset
                rhythmic_syncopation = min(1.0, (rms_db + 60) / 40)
            
            # Create synthetic stream ID based on musical characteristics
            stream_id = 0  # Default stream
            if chord_tension > 0.7:
                stream_id = 1  # High tension stream
            elif chord_tension < 0.3:
                stream_id = 2  # Low tension stream
            
            # Add rhythmic variation to stream assignment
            if rhythmic_density > 0.7:
                stream_id = (stream_id + 3) % 5  # High density gets different stream
            
            # Extract chord (simplified - based on fundamental frequency)
            chord = "C"  # Default chord
            if f0 > 0:
                # Simple chord mapping based on frequency
                note_index = int(12 * np.log2(f0 / 440.0)) % 12
                chord_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                chord = chord_names[note_index]
            
            # Key signature (simplified)
            key_signature = f"{chord} major"
            
            return {
                'chord_tension': chord_tension,
                'key_stability': key_stability,
                'tempo': tempo,
                'tempo_stability': 0.5,  # Default neutral
                'hierarchical_level': 'phrase',  # Default level
                'structural_importance': min(1.0, max(0.0, (rms_db + 60) / 40)),  # Based on energy
                'rhythmic_density': rhythmic_density,
                'rhythmic_syncopation': rhythmic_syncopation,
                'stream_id': stream_id,
                'stream_confidence': 0.8,  # High confidence for synthetic streams
                'chord': chord,
                'key_signature': key_signature
            }
            
        except Exception as e:
            print(f"⚠️ Error extracting musical metadata: {e}")
            # Return default values
            return {
                'chord_tension': 0.5,
                'key_stability': 0.5,
                'tempo': 120.0,
                'tempo_stability': 0.5,
                'hierarchical_level': 'phrase',
                'structural_importance': 0.5,
                'rhythmic_density': 0.5,
                'rhythmic_syncopation': 0.0,
                'stream_id': 0,
                'stream_confidence': 0.5,
                'chord': 'C',
                'key_signature': 'C major'
            }
    
    def _event_to_audio_frame(self, event_data: Dict) -> np.ndarray:
        """Convert event data to audio frame for rhythmic analysis"""
        # Create a synthetic audio frame from event data
        # This is a simplified conversion - in a real implementation,
        # you'd want to use the actual audio frame from the listener
        
        frame_size = 512
        audio_frame = np.zeros(frame_size)
        
        # Use RMS level to create synthetic audio
        rms_db = event_data.get('rms_db', -80.0)
        rms_linear = 10 ** (rms_db / 20.0)
        
        # Create synthetic audio frame
        if rms_linear > 0.001:  # If there's significant audio
            audio_frame = np.random.randn(frame_size) * rms_linear
        
        return audio_frame
    
    def _combine_decisions(self, harmonic_decisions: List, rhythmic_decisions: List, event_data: Dict) -> List:
        """Combine harmonic and rhythmic decisions using unified decision engine"""
        if not harmonic_decisions:
            return []
        
        if not self.enable_rhythmic or not rhythmic_decisions:
            # No rhythmic analysis - return harmonic decisions as-is
            return harmonic_decisions
        
        # Prepare harmonic context for unified decision engine
        harmonic_context = {
            'primary_chord': event_data.get('chord', ''),
            'key': event_data.get('key', ''),
            'harmonic_tension': event_data.get('harmonic_tension', 0.5),
            'chord_diversity': event_data.get('chord_diversity', 0.5),
            'key_stability': event_data.get('key_stability', 0.8),
            'chord_change_rate': event_data.get('chord_change_rate', 0.3),
            'event_count': 1
        }
        
        # Prepare rhythmic context for unified decision engine
        rhythmic_context = {
            'tempo': rhythmic_decisions[0].tempo if hasattr(rhythmic_decisions[0], 'tempo') else 120,
            'syncopation': rhythmic_decisions[0].syncopation if hasattr(rhythmic_decisions[0], 'syncopation') else 0.5,
            'rhythmic_density': rhythmic_decisions[0].rhythmic_density if hasattr(rhythmic_decisions[0], 'rhythmic_density') else 0.5,
            'tempo_stability': 0.8,  # Default stability
            'rhythmic_complexity': 0.5,  # Default complexity
            'event_count': 1
        }
        
        # Create cross-modal context
        # Pass loaded correlation patterns from training to enable correlation-based decisions
        correlation_patterns_list = list(self.unified_decision_engine.correlation_patterns.values()) if self.unified_decision_engine.correlation_patterns else []
        
        cross_modal_context = CrossModalContext(
            harmonic_context=harmonic_context,
            rhythmic_context=rhythmic_context,
            correlation_patterns=correlation_patterns_list,  # Now uses learned patterns from training
            temporal_alignment={'alignment_strength': 0.8},
            musical_context=MusicalContext.UNKNOWN
        )
        
        # Make unified decision
        try:
            unified_decision = self.unified_decision_engine.make_unified_decision(
                harmonic_context, rhythmic_context, cross_modal_context
            )
            
            # Apply unified decision to harmonic decisions
            combined_decisions = []
            rhythmic_decision = rhythmic_decisions[0]
            
            for harmonic_decision in harmonic_decisions:
                # Apply unified decision logic
                if unified_decision.response_mode.value == 'support' and rhythmic_decision.should_play:
                    # Support mode + rhythmic says play - use harmonic decision
                    combined_decisions.append(harmonic_decision)
                elif unified_decision.response_mode.value == 'contrast' and not rhythmic_decision.should_play:
                    # Contrast mode + rhythmic says don't play - create contrast
                    contrast_decision = harmonic_decision
                    contrast_decision.confidence *= 0.8  # Slight confidence reduction
                    combined_decisions.append(contrast_decision)
                elif unified_decision.response_mode.value == 'imitate' and rhythmic_decision.should_play:
                    # Imitate mode + rhythmic says play - use harmonic decision
                    combined_decisions.append(harmonic_decision)
                elif unified_decision.response_mode.value == 'lead':
                    # Lead mode - always play regardless of rhythmic decision
                    combined_decisions.append(harmonic_decision)
                else:
                    # Default behavior - respect rhythmic decision
                    if rhythmic_decision.should_play:
                        combined_decisions.append(harmonic_decision)
                    else:
                        silent_decision = harmonic_decision
                        silent_decision.confidence *= 0.1
                        combined_decisions.append(silent_decision)
            
            # Log unified decision reasoning
            self.performance_logger.log_system_event(
                event_type="unified_decision",
                voice="ai_agent",
                trigger_reason=f"{unified_decision.response_mode.value}_mode",
                activity_level=unified_decision.confidence,
                phase="decision",
                additional_data=f"context={unified_decision.context.value},reasoning={unified_decision.reasoning}"
            )
            
            return combined_decisions
            
        except Exception as e:
            print(f"⚠️ Unified decision error: {e}")
            # Fallback to simple combination
            return self._simple_combine_decisions(harmonic_decisions, rhythmic_decisions)
    
    def _simple_combine_decisions(self, harmonic_decisions: List, rhythmic_decisions: List) -> List:
        """Simple fallback decision combination"""
        combined_decisions = []
        rhythmic_decision = rhythmic_decisions[0] if rhythmic_decisions else None
        
        for harmonic_decision in harmonic_decisions:
            if rhythmic_decision and rhythmic_decision.should_play:
                combined_decisions.append(harmonic_decision)
            elif rhythmic_decision and not rhythmic_decision.should_play:
                silent_decision = harmonic_decision
                silent_decision.confidence *= 0.1
                combined_decisions.append(silent_decision)
            else:
                combined_decisions.append(harmonic_decision)
        
        return combined_decisions
    
    # Context-aware system removed - using original AI agent
    
    # Context-aware decision generation removed - using original AI agent
    
    # Rhythmic context analysis removed - using original AI agent
    
    def _get_learned_rhythmic_patterns(self) -> List[Dict]:
        """Get learned patterns from rhythm oracle"""
        if not self.enable_rhythmic or not self.rhythm_oracle:
            return []
        
        patterns = []
        for pattern in self.rhythm_oracle.rhythmic_patterns:
            pattern_dict = {
                'tempo': pattern.tempo,
                'density': pattern.density,
                'syncopation': pattern.syncopation,
                'pattern_type': pattern.pattern_type,
                'confidence': pattern.confidence
            }
            patterns.append(pattern_dict)
        
        return patterns
    
    def _main_loop(self):
        """Main processing loop"""
        last_clustering_time = time.time()
        clustering_interval = 30.0  # Retrain clustering every 30 seconds
        
        while self.running:
            try:
                current_time = time.time()
                
                # Update statistics
                self.stats['uptime'] = current_time - self.start_time
                
                # Check for graceful performance ending
                if self.timeline_manager:
                    # Update timeline clock (regardless of audio input)
                    self.timeline_manager.update_performance_state(
                        human_activity=False,  # Will be overridden by actual audio events
                        instrument_detected='unknown'
                    )
                    
                    # DEBUG: Show timeline status every 30 seconds
                    remaining = self.timeline_manager.get_time_remaining()
                    if int(remaining) % 30 == 0 and int(remaining) > 0 and not hasattr(self, f'_timeline_msg_{int(remaining)}'):
                        elapsed = (self.timeline_manager.performance_state.total_duration - remaining) / 60
                        total_min = self.timeline_manager.performance_state.total_duration / 60
                        print(f"⏱️  Timeline: {elapsed:.1f}/{total_min:.1f} min | Remaining: {remaining/60:.1f} min | Complete: {self.timeline_manager.is_complete()}")
                        setattr(self, f'_timeline_msg_{int(remaining)}', True)
                    
                    if self.timeline_manager.is_complete():
                        print("\n🎭 Performance complete - gracefully ending...")
                        # Set running flag to false to stop all background processes
                        self.running = False
                        # Call stop() to properly clean up all resources
                        self.stop()
                        break  # Exit main loop immediately
                    
                    # Get fade-out factor (1.0 → 0.0 over final 60 seconds)
                    fade_factor = self.timeline_manager.get_fade_out_factor(fade_duration=60.0)
                    
                    # Show fade status in final minute
                    remaining = self.timeline_manager.get_time_remaining()
                    if remaining < 60 and int(remaining) % 10 == 0 and not hasattr(self, f'_fade_msg_{int(remaining)}'):
                        setattr(self, f'_fade_msg_{int(remaining)}', True)
                        print(f"🌅 Fading out... {int(remaining)}s remaining (activity: {fade_factor*100:.0f}%)")
                    
                    # DEBUG: Check if timeline should have stopped
                    if remaining <= 0:
                        print(f"⚠️  Timeline remaining={remaining:.2f}s but is_complete()={self.timeline_manager.is_complete()}")  # DEBUG
                else:
                    fade_factor = 1.0  # Full activity when no performance arc
                
                # IRCAM Phase 2.3: Autonomous generation with adaptive interval (with fade-out)
                if self.autonomous_generation_enabled and self.running:  # Check self.running
                    adaptive_interval = self._calculate_adaptive_interval()
                    
                    if current_time - self.last_autonomous_time >= adaptive_interval:
                        # Apply fade-out: randomly skip generation as fade_factor decreases
                        if random.random() < fade_factor:
                            self._autonomous_generation_tick(current_time)
                        self.last_autonomous_time = current_time
                
                # Continue active phrases (check every 100ms for 250ms note intervals)
                # Apply same activity-based filtering as reactive events
                continue_decisions = []
                if self.running:  # Check self.running
                    continue_decisions = self.ai_agent.behavior_engine.continue_phrase_iteration()
                
                # Process phrase continuations - let phrases complete once started
                # (Phrase continuation represents notes in an already-started phrase)
                if continue_decisions:
                    for decision in continue_decisions:
                        # Process phrase continuation decision like a normal decision
                        event_data = {
                            'instrument': decision.voice_type,
                            'harmonic_context': None,
                            'rhythmic_context': None
                        }
                        
                        midi_params = self.feature_mapper.map_features_to_midi(
                            event_data, {
                                'mode': decision.mode.value,
                                'confidence': decision.confidence,
                                'musical_params': decision.musical_params
                            },
                            decision.voice_type
                        )
                        
                        # Send MIDI note
                        voice_type = decision.voice_type
                        if self.enable_mpe and self.mpe_midi_outputs.get(voice_type):
                            expression_level = self._calculate_expressive_intensity(decision, event_data)
                            timbre_variation = self._calculate_timbre_variation(decision, event_data)
                            pressure_sensitivity = self._calculate_pressure_sensitivity(decision, event_data)
                            
                            if self.mpe_midi_outputs[voice_type].send_note(midi_params, voice_type, 0.0, expression_level, 
                                                                             timbre_variation, pressure_sensitivity):
                                self.stats['notes_sent'] += 1
                                # IRCAM Phase 1.1: Track sent note to prevent learning from own output
                                self.own_output_tracker['recent_notes'].append(
                                    (midi_params.note, current_time, midi_params.velocity)
                                )
                                # IRCAM Phase 3+: Track AI-generated event
                                if self.ai_agent and hasattr(self.ai_agent, 'phrase_generator') and self.ai_agent.phrase_generator:
                                    ai_event = {
                                        'midi': midi_params.note,
                                        'velocity': midi_params.velocity,
                                        'voice_type': voice_type
                                    }
                                    self.ai_agent.phrase_generator.track_event(ai_event, source='ai')
                                
                                # Emit visualization event for AI response
                                if self.visualization_manager:
                                    print(f"🎨 TIMELINE: Emitting response event at {current_time}")  # DEBUG
                                    self.visualization_manager.emit_timeline_update('response', timestamp=current_time)
                                
                                # Log phrase continuation
                                self._log_conversation_response(midi_params, voice_type, decision.mode.value, current_time,
                                                               expression_level, pressure_sensitivity, timbre_variation)
                        
                        elif self.midi_outputs.get(voice_type):
                            if self.midi_outputs[voice_type].send_note(midi_params, voice_type):
                                self.stats['notes_sent'] += 1
                                # IRCAM Phase 1.1: Track sent note to prevent learning from own output
                                self.own_output_tracker['recent_notes'].append(
                                    (midi_params.note, current_time, midi_params.velocity)
                                )
                                # IRCAM Phase 3+: Track AI-generated event
                                if self.ai_agent and hasattr(self.ai_agent, 'phrase_generator') and self.ai_agent.phrase_generator:
                                    ai_event = {
                                        'midi': midi_params.note,
                                        'velocity': midi_params.velocity,
                                        'voice_type': voice_type
                                    }
                                    self.ai_agent.phrase_generator.track_event(ai_event, source='ai')
                                
                                # Emit visualization event for AI response
                                if self.visualization_manager:
                                    self.visualization_manager.emit_timeline_update('response', timestamp=current_time)
                                
                                # Log phrase continuation
                                self._log_conversation_response(midi_params, voice_type, decision.mode.value, current_time,
                                                               0.0, 0.0, 0.0)
                        
                        self.stats['notes_played'] = self.stats.get('notes_played', 0) + 1
                
                # Retrain clustering periodically
                if current_time - last_clustering_time > clustering_interval:
                    if self.clustering.retrain_if_needed(self.memory_buffer):
                        print(f"🧠 Retrained clustering with {len(self.memory_buffer.get_all_moments())} moments")
                    last_clustering_time = current_time
                
                # Print status periodically (disabled - using live status bar)
                # if int(current_time) % 60 == 0:  # Every minute
                #     self._print_status()
                
                time.sleep(0.1)
                
            except Exception as e:
                print(f"❌ Main loop error: {e}")
                time.sleep(1.0)
    
    def _print_status(self):
        """Print system status"""
        buffer_stats = self.memory_buffer.get_buffer_stats()
        agent_state = self.ai_agent.get_agent_state()
        
        print("\n📊 Status Update:")
        print(f"   Events: {self.stats['events_processed']}, "
              f"Decisions: {self.stats['decisions_made']}, "
              f"Notes: {self.stats['notes_sent']}")
        print(f"   Memory: {buffer_stats['count']} moments, "
              f"{buffer_stats['duration_seconds']:.1f}s")
        print(f"   Agent: {agent_state.current_mode} mode, "
              f"conf={agent_state.confidence:.2f}")
        print(f"   Patterns: {self.stats['harmonic_patterns_detected']} harmonic, "
              f"{self.stats['polyphonic_patterns_detected']} polyphonic")
        if self.enable_rhythmic:
            print(f"   Rhythmic: {self.stats['rhythmic_decisions']} decisions, "
                  f"{self.stats['combined_decisions']} combined")
        print(f"   Uptime: {self.stats['uptime']:.1f}s")
    
    def _get_reference_frequency(self, midi_note: int) -> float:
        """Get reference frequency for MIDI note (12TET)"""
        return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))
    
    def set_density_level(self, level: float):
        """Set musical density level"""
        self.ai_agent.set_density_level(level)
        print(f"🎚️ Density level set to {level:.2f}")
    
    def set_give_space_factor(self, factor: float):
        """Set give space factor"""
        self.ai_agent.set_give_space_factor(factor)
        print(f"🎚️ Give space factor set to {factor:.2f}")
    
    def set_initiative_budget(self, budget: float):
        """Set initiative budget"""
        self.ai_agent.set_initiative_budget(budget)
        print(f"🎚️ Initiative budget set to {budget:.2f}")
    
    def get_status(self) -> Dict:
        """Get complete system status"""
        status = {
            'running': self.running,
            'stats': self.stats.copy(),
            'memory_buffer': self.memory_buffer.get_buffer_stats(),
            'agent_state': self.ai_agent.get_agent_state(),
            'midi_outputs': {voice: output.get_status() if output else None 
                            for voice, output in self.midi_outputs.items()},
            'mpe_midi_outputs': {voice: output.get_status() if output else None 
                                for voice, output in self.mpe_midi_outputs.items()},
            'clustering': {
                'is_trained': self.clustering.is_trained,
                'total_states': self.clustering.get_statistics().get('total_states', 0),
                'sequence_length': self.clustering.get_statistics().get('sequence_length', 0),
                'distance_function': self.clustering.get_statistics().get('distance_function', 'unknown'),
                'distance_threshold': self.clustering.get_statistics().get('distance_threshold', 0),
                'total_distances_calculated': self.clustering.get_statistics().get('total_distances_calculated', 0),
                'average_distance': self.clustering.get_statistics().get('average_distance', 0),
                'threshold_adjustments': self.clustering.get_statistics().get('threshold_adjustments', 0),
                'chord_patterns': self.clustering.get_statistics().get('chord_patterns', 0),
                'polyphonic_frames': self.clustering.get_statistics().get('polyphonic_frames', 0),
                'harmonic_patterns': self.clustering.get_statistics().get('harmonic_patterns', 0),
                'polyphonic_patterns': self.clustering.get_statistics().get('polyphonic_patterns', 0),
                'mps_enabled': False
            }
        }
        
        # Add rhythmic status if enabled
        if self.enable_rhythmic and self.rhythm_oracle:
            status['rhythmic'] = {
                'enabled': True,
                'patterns_learned': len(self.rhythm_oracle.rhythmic_patterns),
                'rhythmic_decisions': self.stats['rhythmic_decisions'],
                'combined_decisions': self.stats['combined_decisions']
            }
        else:
            status['rhythmic'] = {'enabled': False}
        
        return status
    
    def _load_learning_data(self):
        """Load previous learning data from files with dynamic configuration"""
        print("🧠 Loading previous learning data...")
        print("🔍 Debug: _load_learning_data() called")
        
        # Initialize model_loaded flag
        model_loaded = False
        
        # Load memory buffer
        memory_loaded = self.memory_buffer.load_from_file(self.memory_file)
        
        # Automatically find the most recent JSON file
        json_dir = "JSON"
        if os.path.exists(json_dir):
            # Look for model files first (AudioOracle models)
            model_files = [f for f in os.listdir(json_dir) if f.endswith('_model.json')]
            if model_files:
                # Sort by modification time (most recent first)
                model_files.sort(key=lambda x: os.path.getmtime(os.path.join(json_dir, x)), reverse=True)
                most_recent_file = os.path.join(json_dir, model_files[0])
                print(f"🎓 Loading most recent AudioOracle model: {most_recent_file}")
                
                # Load model configuration first to initialize clustering with correct parameters
                model_config = self._load_model_config(most_recent_file)
                if model_config:
                    # Initialize clustering with parameters from the saved model
                    self.clustering = PolyphonicAudioOracle(
                        distance_threshold=model_config.get('distance_threshold', 0.15),
                        distance_function=model_config.get('distance_function', 'euclidean'),
                        feature_dimensions=model_config.get('feature_dimensions', 15),
                        adaptive_threshold=model_config.get('adaptive_threshold', True),
                        chord_similarity_weight=model_config.get('chord_similarity_weight', 0.3),
                        max_pattern_length=model_config.get('max_pattern_length', 50)
                    )
                    print(f"✅ Initialized PolyphonicAudioOracle with model config:")
                    print(f"   📐 Feature dimensions: {model_config.get('feature_dimensions', 15)}")
                    print(f"   📏 Distance threshold: {model_config.get('distance_threshold', 0.15)}")
                    print(f"   📊 Distance function: {model_config.get('distance_function', 'euclidean')}")
                    
                    # Check for format version compatibility
                    format_version = model_config.get('format_version', '1.0')
                    print(f"   🏷️  Model format version: {format_version}")
                    
                    # DEBUG: Check loaded data size after initialization
                    if self.clustering:
                        print(f"🧠 AudioOracle after init: {len(self.clustering.states)} states, {len(self.clustering.audio_frames)} frames")  # DEBUG
                else:
                    # Fallback to defaults if config loading fails
                    print("⚠️  Could not load model config, using defaults")
                    self.clustering = PolyphonicAudioOracle(
                        distance_threshold=0.15,
                        distance_function='euclidean',
                        feature_dimensions=15,
                        adaptive_threshold=True,
                        chord_similarity_weight=0.3
                    )
                
                # Now load the actual model data
                polyphonic_oracle_loaded = self.clustering.load_from_file(most_recent_file)
                if polyphonic_oracle_loaded:
                    print("✅ Successfully loaded AudioOracle model!")
                    # Get model statistics
                    stats = self.clustering.get_statistics()
                    print(f"📊 Model stats: {stats.get('total_states', 0)} states, "
                          f"{stats.get('sequence_length', 0)} sequence length, "
                          f"{stats.get('total_patterns', 0)} patterns learned, "
                          f"{stats.get('harmonic_patterns', 0)} harmonic patterns, "
                          f"{stats.get('polyphonic_patterns', 0)} polyphonic patterns")
                    
                    # Check if this is a music theory enhanced model
                    if "enhanced" in most_recent_file or "music_theory" in most_recent_file:
                        print("🧠 Music theory enhanced model detected!")
                    
                    # Load gesture vocabulary (quantizer) if available
                    if self.hybrid_perception and self.enable_hybrid_perception:
                        quantizer_file = most_recent_file.replace('_model.json', '_quantizer.joblib')
                        if os.path.exists(quantizer_file):
                            try:
                                self.hybrid_perception.load_quantizer(quantizer_file)
                                print(f"✅ Gesture vocabulary (quantizer) loaded!")
                                
                                # CRITICAL FIX: Monkey-patch transform to convert ALL intermediate dtypes to float32
                                # The pickled quantizer uses sklearn normalize() which returns float64
                                import numpy as np
                                from sklearn.preprocessing import normalize as sklearn_normalize
                                quantizer = self.hybrid_perception.quantizer
                                if quantizer is not None:
                                    # Patch transform to ensure float32 throughout
                                    original_transform = quantizer.transform
                                    def fixed_transform(features):
                                        try:
                                            # Ensure input features are float32
                                            if len(features) == 0:
                                                return np.array([])
                                            if features.ndim == 1:
                                                features = features.reshape(1, -1)
                                            if features.dtype != np.float32:
                                                features = features.astype(np.float32)
                                            
                                            # Manually do what transform does, but with float32
                                            if quantizer.use_l2_norm:
                                                # sklearn normalize returns float64, so we need to convert
                                                features_scaled = sklearn_normalize(features, norm='l2', axis=1)
                                                if features_scaled.dtype != np.float32:
                                                    features_scaled = features_scaled.astype(np.float32)
                                            else:
                                                features_scaled = quantizer.scaler.transform(features)
                                                if features_scaled.dtype != np.float32:
                                                    features_scaled = features_scaled.astype(np.float32)
                                            
                                            # Ensure kmeans centroids are also float32
                                            if hasattr(quantizer.kmeans, 'cluster_centers_'):
                                                if quantizer.kmeans.cluster_centers_.dtype != np.float32:
                                                    quantizer.kmeans.cluster_centers_ = quantizer.kmeans.cluster_centers_.astype(np.float32)
                                            
                                            # Now call kmeans.predict with float32
                                            tokens = quantizer.kmeans.predict(features_scaled)
                                            return tokens
                                        except Exception as e:
                                            print(f"🔧 Monkey-patch error: {e}")
                                            raise
                                    
                                    quantizer.transform = fixed_transform
                                    print("✅ Quantizer dtype fix monkey-patched (full transform override + kmeans centers)!")
                            except Exception as e:
                                print(f"⚠️  Could not load quantizer: {e}")
                        else:
                            print(f"⚠️  No quantizer file found ({quantizer_file})")
                            print(f"   Gesture tokens will not be available - retrain model to generate quantizer")
                    
                    model_loaded = True
                else:
                    print(f"❌ Failed to load {most_recent_file}")
            
            # Fallback: look for regular JSON files (training results) - but skip if we have model files
            if not model_files:  # Only try training results if no model files exist
                json_files = [f for f in os.listdir(json_dir) if f.endswith('.json') and not f.endswith('_model.json')]
                if json_files:
                    # Sort by modification time (most recent first)
                    json_files.sort(key=lambda x: os.path.getmtime(os.path.join(json_dir, x)), reverse=True)
                    most_recent_file = os.path.join(json_dir, json_files[0])
                    print(f"🎓 Loading most recent training results: {most_recent_file}")
                    
                    # Load config and initialize clustering if not already done
                    if self.clustering is None:
                        model_config = self._load_model_config(most_recent_file)
                        if model_config:
                            self.clustering = PolyphonicAudioOracle(
                                distance_threshold=model_config.get('distance_threshold', 0.15),
                                distance_function=model_config.get('distance_function', 'euclidean'),
                                feature_dimensions=model_config.get('feature_dimensions', 15),
                                adaptive_threshold=model_config.get('adaptive_threshold', True),
                                chord_similarity_weight=model_config.get('chord_similarity_weight', 0.3),
                                max_pattern_length=model_config.get('max_pattern_length', 50)
                            )
                            print(f"✅ Initialized from training results config")
                        else:
                            # Final fallback to defaults
                            self.clustering = PolyphonicAudioOracle(
                                distance_threshold=0.15,
                                distance_function='euclidean',
                                feature_dimensions=15,
                                adaptive_threshold=True,
                                chord_similarity_weight=0.3
                            )
                    
                    polyphonic_oracle_loaded = self.clustering.load_from_file(most_recent_file)
                    if polyphonic_oracle_loaded:
                        print("✅ Successfully loaded training results!")
                        # Get model statistics
                        stats = self.clustering.get_statistics()
                        print(f"📊 Model stats: {stats.get('total_states', 0)} states, "
                              f"{stats.get('sequence_length', 0)} sequence length, "
                              f"{stats.get('total_patterns', 0)} patterns learned, "
                              f"{stats.get('harmonic_patterns', 0)} harmonic patterns, "
                              f"{stats.get('polyphonic_patterns', 0)} polyphonic patterns")
                        
                        # Check if this is a music theory enhanced model
                        if "enhanced" in most_recent_file or "music_theory" in most_recent_file:
                            print("🧠 Music theory enhanced model detected!")
                        
                        model_loaded = True
                    else:
                        print(f"❌ Failed to load {most_recent_file}")
        
        # Ensure clustering is initialized even if no model found yet
        if self.clustering is None:
            print("⚙️  No model found yet, initializing with defaults...")
            self.clustering = PolyphonicAudioOracle(
                distance_threshold=0.15,
                distance_function='euclidean',
                feature_dimensions=15,
                adaptive_threshold=True,
                chord_similarity_weight=0.3
            )
        
        # Only load fallback models if we haven't already loaded a model
        if not model_loaded:
            # Try to load the best available trained model (prioritize enhanced models)
            enhanced_model_paths = [
                "JSON/grab-a-hold_enhanced.json",  # Music theory transformer enhanced model
                "JSON/test_music_theory.json",     # Music theory transformer test model
                "JSON/test_polyphonic_model.json", # Polyphonic model
                "JSON/test_correlation_fix.json",  # Correlation fix test model
                "JSON/test_final_correlation_fix.json"  # Final correlation fix model
            ]
            
            for model_path in enhanced_model_paths:
                if os.path.exists(model_path):
                    print(f"🎓 Loading trained model from {model_path}...")
                    # Load config to ensure compatibility
                    model_config = self._load_model_config(model_path)
                    if model_config and model_config.get('feature_dimensions') != self.clustering.feature_dimensions:
                        print(f"⚠️  Model has different feature dimensions ({model_config.get('feature_dimensions')}) than current ({self.clustering.feature_dimensions})")
                        print(f"   Reinitializing with model's configuration...")
                        self.clustering = PolyphonicAudioOracle(
                            distance_threshold=model_config.get('distance_threshold', 0.15),
                            distance_function=model_config.get('distance_function', 'euclidean'),
                            feature_dimensions=model_config.get('feature_dimensions', 15),
                            adaptive_threshold=model_config.get('adaptive_threshold', True),
                            chord_similarity_weight=model_config.get('chord_similarity_weight', 0.3),
                            max_pattern_length=model_config.get('max_pattern_length', 50)
                        )
                    
                    polyphonic_oracle_loaded = self.clustering.load_from_file(model_path)
                    if polyphonic_oracle_loaded:
                        print("✅ Successfully loaded trained model!")
                        # Get model statistics
                        stats = self.clustering.get_statistics()
                        print(f"📊 Model stats: {stats.get('total_states', 0)} states, "
                              f"{stats.get('sequence_length', 0)} sequence length, "
                              f"{stats.get('total_patterns', 0)} patterns learned, "
                              f"{stats.get('chord_patterns', 0)} chord patterns")
                        
                        # Check if this is a music theory enhanced model
                        if "enhanced" in model_path or "music_theory" in model_path:
                            print("🧠 Music theory enhanced model detected!")
                        
                        model_loaded = True
                        break
                    else:
                        print(f"❌ Failed to load {model_path}, trying next...")
        
        if not model_loaded:
            # Load PolyphonicAudioOracle model from default location
            print("📝 No trained models found, loading default model...")
            polyphonic_oracle_loaded = self.clustering.load_from_file(self.clustering_file)
        
        # Load rhythmic data if enabled
        print(f"🔍 Debug: enable_rhythmic={self.enable_rhythmic}, rhythm_oracle={self.rhythm_oracle is not None}")
        rhythmic_loaded = False
        if self.enable_rhythmic and self.rhythm_oracle:
            try:
                self.rhythm_oracle.load_patterns(self.rhythmic_file)
                rhythmic_loaded = True
                print("🥁 Loaded rhythmic patterns")
            except Exception as e:
                print(f"⚠️ Could not load rhythmic patterns: {e}")
        
        # Update AI agent with rhythm oracle for phrase generation
        print(f"🔍 Debug: About to initialize phrase generator...")
        if self.rhythm_oracle:
            self.ai_agent.behavior_engine.phrase_generator = PhraseGenerator(
                self.rhythm_oracle, 
                visualization_manager=self.visualization_manager
            )
            print("🎼 AI Agent updated with rhythmic phrase generation")
        else:
            # Initialize phrase generator without rhythm oracle
            self.ai_agent.behavior_engine.phrase_generator = PhraseGenerator(
                None,
                visualization_manager=self.visualization_manager
            )
            print("🎼 AI Agent initialized with phrase generation (no rhythm oracle)")
        
        # Verify phrase generator is set
        if self.ai_agent.behavior_engine.phrase_generator:
            print("✅ Phrase generator verified - ready for musical phrases!")
        else:
            print("❌ ERROR: Phrase generator not properly initialized!")
        
        # Load correlation patterns from training
        if model_loaded and most_recent_file:
            correlation_file = os.path.join(os.path.dirname(most_recent_file), 
                                           'correlation_patterns.json')
            if os.path.exists(correlation_file):
                try:
                    import json
                    with open(correlation_file, 'r') as f:
                        correlation_data = json.load(f)
                        self.unified_decision_engine.load_correlation_patterns(correlation_data)
                        print(f"🔗 Loaded {len(correlation_data.get('patterns', []))} correlation patterns")
                except Exception as e:
                    print(f"⚠️  Could not load correlation patterns: {e}")
        
        # Load GPT-OSS behavioral insights if available
        self._load_gpt_oss_insights(most_recent_file if model_loaded else None)
        
        # Load ML chord detection model if available
        self._load_ml_chord_model()
        
        if memory_loaded and polyphonic_oracle_loaded:
            print("✅ Successfully loaded previous learning data")
        elif memory_loaded:
            print("✅ Loaded musical memory, PolyphonicAudioOracle will learn from it")
        elif polyphonic_oracle_loaded:
            print("✅ Loaded PolyphonicAudioOracle model, building new memory")
        else:
            print("📝 Starting with fresh learning data")
        
        # Display live training status
        print()  # Add blank line for clarity
        if self.enable_live_training:
            print("⚠️  LIVE TRAINING ENABLED - AudioOracle will learn from your playing")
            print("   (trained patterns may be overwritten as new patterns accumulate)")
        else:
            print("🔒 LIVE TRAINING DISABLED - generating purely from pre-trained model")
            print("   (use --enable-live-training to enable continuous learning)")
        print()  # Add blank line for clarity
    
    def _load_model_config(self, filepath: str) -> Optional[Dict]:
        """
        Load model configuration from file without loading full model
        This allows us to initialize AudioOracle with correct parameters before loading
        
        Returns:
            Dict with model configuration or None if loading fails
        """
        try:
            import json
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Extract configuration parameters
            config = {
                'format_version': data.get('format_version', '1.0'),
                'distance_threshold': data.get('distance_threshold', 0.15),
                'distance_function': data.get('distance_function', 'euclidean'),
                'max_pattern_length': data.get('max_pattern_length', 50),
                'feature_dimensions': data.get('feature_dimensions', 15),
                'adaptive_threshold': data.get('adaptive_threshold', True),
                'chord_similarity_weight': data.get('chord_similarity_weight', 0.3),
                # Additional metadata for version checking
                'size': data.get('size', 0),
                'sequence_length': data.get('sequence_length', 0),
                'is_trained': data.get('is_trained', False),
            }
            
            return config
            
        except Exception as e:
            print(f"⚠️  Error loading model config from {filepath}: {e}")
            return None
    
    def _save_learning_data(self):
        """Save current learning data to files"""
        print("💾 Saving learning data...")
        
        # Save memory buffer (always save - contains recent 30s context)
        memory_saved = self.memory_buffer.save_to_file(self.memory_file)
        
        # Save PolyphonicAudioOracle model (only if live training was enabled)
        polyphonic_oracle_saved = False
        if self.enable_live_training:
            polyphonic_oracle_saved = self.clustering.save_to_file(self.clustering_file)
        else:
            print("🔒 Skipping AudioOracle save (live training was disabled)")
        
        # Save rhythmic data if enabled
        rhythmic_saved = False
        if self.enable_rhythmic and self.rhythm_oracle:
            try:
                self.rhythm_oracle.save_patterns(self.rhythmic_file)
                rhythmic_saved = True
            except Exception as e:
                print(f"⚠️ Could not save rhythmic patterns: {e}")
        
        if memory_saved and polyphonic_oracle_saved:
            print("✅ Successfully saved all learning data")
        elif memory_saved:
            print("✅ Saved musical memory")
        elif polyphonic_oracle_saved:
            print("✅ Saved PolyphonicAudioOracle model")
        else:
            print("⚠️ No learning data to save")
        
        if rhythmic_saved:
            print("🥁 Saved rhythmic patterns")
    
    def _log_conversation_input(self, event_data: Dict, current_time: float):
        """Log human input for conversation analysis"""
        if not self._conversation_log_file:
            return
        
        # Only log once per second to avoid spam
        if current_time - self._last_conversation_log < self._conversation_log_interval:
            return
        
        self._last_conversation_log = current_time
        
        try:
            elapsed = current_time - self.start_time
            f0 = event_data.get('f0', 0.0)
            midi = event_data.get('midi', 0)
            rms_db = event_data.get('rms_db', -80.0)
            
            # Write input row
            self._conversation_log_file.write(
                f"{current_time:.3f},{elapsed:.3f},INPUT,human,{f0:.1f},{midi},{rms_db:.1f},"
                f"{self.human_activity_level:.2f},{'AUTO' if self.was_in_autonomous_mode else 'LISTEN'},"
                f",,,,\n"
            )
            self._conversation_log_file.flush()
        except Exception as e:
            print(f"⚠️ Conversation log error: {e}")
    
    def _log_conversation_response(self, midi_params, voice_type: str, mode: str, 
                                   current_time: float, mpe_expression: float, 
                                   mpe_pressure: float, mpe_timbre: float):
        """Log AI response for conversation analysis"""
        if not self._conversation_log_file:
            return
        
        try:
            elapsed = current_time - self.start_time
            # Convert MIDI note to frequency
            pitch_hz = 440.0 * (2.0 ** ((midi_params.note - 69) / 12.0))
            
            # Convert MPE values to pitch bend (semitones), pressure (0-127), timbre (0-127)
            pitchbend_semitones = mpe_expression * 2.0  # Assuming 2 semitone range
            pressure_midi = int(mpe_pressure * 127)
            timbre_midi = int(mpe_timbre * 127)
            
            # Write response row
            self._conversation_log_file.write(
                f"{current_time:.3f},{elapsed:.3f},OUTPUT,{voice_type},{pitch_hz:.1f},"
                f"{midi_params.note},{-40.0:.1f},{self.human_activity_level:.2f},{mode},"
                f"{pitchbend_semitones:.2f},{pressure_midi},{timbre_midi},{midi_params.velocity}\n"
            )
            self._conversation_log_file.flush()
        except Exception as e:
            print(f"⚠️ Conversation log error: {e}")
    
    def _event_to_audio_buffer(self, event_data: Dict) -> np.ndarray:
        """Convert event data to audio buffer for hybrid detection"""
        try:
            # Create a simple audio buffer from event features
            # This is a simplified approach - in a real implementation,
            # you'd want to use the actual audio buffer from the listener
            
            # Extract basic features
            f0 = event_data.get('f0', 440.0)
            rms_db = event_data.get('rms_db', -60.0)
            
            # Convert RMS dB to linear amplitude
            rms_linear = 10 ** (rms_db / 20.0)
            
            # Create a short audio buffer (1024 samples)
            buffer_length = 1024
            t = np.linspace(0, buffer_length / 44100, buffer_length)
            
            # Generate a simple sine wave at the detected frequency
            audio_buffer = rms_linear * np.sin(2 * np.pi * f0 * t)
            
            # Add some noise to make it more realistic
            noise_level = rms_linear * 0.1
            audio_buffer += noise_level * np.random.randn(buffer_length)
            
            return audio_buffer.astype(np.float32)
            
        except Exception as e:
            print(f"⚠️ Error creating audio buffer: {e}")
            # Return silence if error
            return np.zeros(1024, dtype=np.float32)
    
    def _load_gpt_oss_insights(self, model_file: Optional[str]):
        """Load GPT-OSS behavioral insights from training JSON"""
        if not model_file or not os.path.exists(model_file):
            return
        
        try:
            # Try to load the companion JSON file (not _model.json)
            companion_file = model_file.replace('_model.json', '.json')
            if not os.path.exists(companion_file):
                return
            
            print(f"🧠 Loading GPT-OSS behavioral insights from {companion_file}...")
            
            import json
            with open(companion_file, 'r') as f:
                data = json.load(f)
            
            # Extract GPT-OSS arc analysis if available
            gpt_oss_arc = data.get('gpt_oss_arc_analysis', {})
            if gpt_oss_arc:
                self.gpt_oss_silence_strategy = gpt_oss_arc.get('silence_strategy', '')
                self.gpt_oss_role_development = gpt_oss_arc.get('role_development', '')
                
                if self.gpt_oss_silence_strategy:
                    print(f"✅ Loaded silence strategy insights ({len(self.gpt_oss_silence_strategy)} chars)")
                if self.gpt_oss_role_development:
                    print(f"✅ Loaded role development insights ({len(self.gpt_oss_role_development)} chars)")
                
                # Apply pause ranges to behavior engine if available
                if hasattr(self, 'ai_agent') and hasattr(self.ai_agent, 'behavior_engine') and self.ai_agent.behavior_engine:
                    self.ai_agent.behavior_engine.set_pause_ranges_from_gpt_oss(gpt_oss_arc)
                    print(f"✅ Applied GPT-OSS pause ranges to behavior engine")
                
        except Exception as e:
            print(f"⚠️ Could not load GPT-OSS insights: {e}")
    
    def _load_ml_chord_model(self):
        """Load ML chord detection model if available"""
        try:
            # Try hybrid model first
            hybrid_model_path = 'models/chord_model_hybrid.pkl'
            hybrid_scaler_path = 'models/chord_scaler_hybrid.pkl'
            
            if os.path.exists(hybrid_model_path) and os.path.exists(hybrid_scaler_path):
                self.ml_chord_model = joblib.load(hybrid_model_path)
                self.ml_chord_scaler = joblib.load(hybrid_scaler_path)
                self.ml_chord_enabled = True
                print(f"✅ Loaded ML chord detection model (hybrid): {hybrid_model_path}")
                
                # Load metadata to show training info
                metadata_path = 'models/chord_model_hybrid_metadata.json'
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    print(f"   📊 Test accuracy: {metadata.get('test_accuracy', 0):.1%}")
                    print(f"   📊 CV mean: {metadata.get('cv_mean', 0):.1%}")
                    print(f"   📚 {metadata.get('num_classes', 0)} chord types, {metadata.get('num_samples', 0)} samples")
                return
            
            # Fallback to old model
            model_path = 'models/chord_model.pkl'
            scaler_path = 'models/chord_scaler.pkl'
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                self.ml_chord_model = joblib.load(model_path)
                self.ml_chord_scaler = joblib.load(scaler_path)
                self.ml_chord_enabled = True
                print(f"✅ Loaded ML chord detection model (old): {model_path}")
            else:
                print("📝 No ML chord model found - using harmonic detection only")
                
        except Exception as e:
            print(f"⚠️ Could not load ML chord model: {e}")
    
    def _predict_ml_chord(self, event: Event, event_data: Optional[Dict] = None) -> Optional[str]:
        """Predict chord using ML model - using hybrid features (22D)"""
        try:
            if not self.ml_chord_model or not self.ml_chord_scaler:
                return None
            
            # Extract hybrid chroma (12D)
            if event_data and 'hybrid_chroma' in event_data:
                chroma = np.array(event_data['hybrid_chroma'])
            elif event.harmonic_context and hasattr(event.harmonic_context, 'chroma'):
                chroma = event.harmonic_context.chroma
            else:
                return None
            
            # Extract ratio features (9D) from hybrid perception
            # Only use ML if we have actual ratio analysis, otherwise return None
            if not (event_data and 'hybrid_ratio_chord' in event_data):
                return None  # Don't predict with placeholder features
            
            # We have ratio analysis
            fundamental = event.f0 if event.f0 > 0 else 440.0
            consonance = event_data.get('hybrid_consonance', 0.5)
            confidence = event_data.get('hybrid_ratio_confidence', 0.5)
            
            # Build ratio features (matching training format)
            # [fundamental/1000, ratio1-4 (4 elements), consonance, num_intervals, confidence, avg_consonance]
            ratio_features = np.array([
                fundamental / 1000.0,
                1.0, 1.0, 1.0, 1.0,  # 4 ratio values (placeholder - would need to extract from ratio_analysis)
                consonance,
                2.0 / 10.0,  # Estimate number of intervals
                confidence,
                consonance  # Use overall consonance as avg
            ])  # 1 + 4 + 1 + 1 + 1 + 1 = 9D
            
            # Combine: 12D chroma + 9D ratio = 21D (matching hybrid training)
            features = np.concatenate([chroma, ratio_features])
            
            # Verify feature dimensions
            if len(features) != 21:
                print(f"⚠️ Feature dimension mismatch: expected 21, got {len(features)}")
                return None
            
            # Scale features
            features_scaled = self.ml_chord_scaler.transform(features.reshape(1, -1))
            
            # Predict chord TYPE (not full chord name)
            prediction = self.ml_chord_model.predict(features_scaled)[0]
            confidence_score = self.ml_chord_model.predict_proba(features_scaled).max()
            
            # Only return predictions with reasonable confidence (>60%)
            # This filters out low-quality predictions from placeholder features
            if confidence_score < 0.6:
                return None
            
            # Return chord type with confidence
            # Model predicts types like: major, minor, dom7, min7, maj7, dim7, etc.
            return f"{prediction}({confidence_score:.2f})"
            
        except Exception as e:
            # Silent errors after first few
            if not hasattr(self, '_ml_error_count'):
                self._ml_error_count = 0
            if self._ml_error_count < 3:
                print(f"⚠️ ML chord prediction error: {e}")
                self._ml_error_count += 1
            return None
    
    def _track_human_activity(self, event_data: Dict, current_time: float):
        """Track human activity level for autonomous generation adjustment"""
        # Update last human event time
        rms_db = event_data.get('rms_db', -80)
        if rms_db > -60:  # Significant audio detected
            self.last_human_event_time = current_time
            
            # Update activity level with exponential smoothing
            # Higher RMS = higher activity
            instant_activity = min(1.0, (rms_db + 60) / 40)  # -60dB to 0dB mapped to 0-1
            alpha = 0.3  # Smoothing factor
            self.human_activity_level = alpha * instant_activity + (1 - alpha) * self.human_activity_level
        else:
            # Decay activity level during silence
            time_since_last = current_time - self.last_human_event_time
            if time_since_last > 0.5:  # Start decaying after 0.5s
                decay_rate = 0.95  # Decay slowly
                self.human_activity_level *= decay_rate
    
    def _autonomous_generation_tick(self, current_time: float):
        """Generate music autonomously, adjusting to human activity"""
        # IRCAM Phase 3: Check if we should respond based on behavior mode
        delay = self.behavior_controller.get_response_delay(self.phrase_detector)
        
        if delay is None:
            return  # Wait for phrase boundary (MIRROR mode phrase-aware behavior)
        
        # Calculate time since last human activity
        time_in_silence = current_time - self.last_human_event_time
        
        # Determine if we're in "autonomous mode" (human is silent)
        in_autonomous_mode = time_in_silence > self.silence_timeout
        
        # Detect mode transition - respond immediately when entering autonomous mode
        just_entered_autonomous = in_autonomous_mode and not self.was_in_autonomous_mode
        self.was_in_autonomous_mode = in_autonomous_mode
        
        # Calculate generation interval based on human activity
        # High activity → much longer intervals (give LOTS of space) 
        # Low activity/silence → faster intervals (more active response)
        if in_autonomous_mode:
            # Autonomous mode: generate actively for melodic conversation
            generation_interval = self.autonomous_interval_base * 0.5  # 2x faster when alone (1.5s base)
        else:
            # Responsive mode: adjust to human activity
            # More human activity = MUCH less AI density
            activity_factor = 1.0 + (self.human_activity_level * 8.0)  # 1.0 to 9.0x slower - give MORE space
            generation_interval = self.autonomous_interval_base * activity_factor
        
        # If just entered autonomous mode, generate immediately as a response
        if just_entered_autonomous:
            # Force immediate generation when silence is detected
            pass  # Will generate below
        # Otherwise check if it's time to generate
        elif current_time - self.last_autonomous_generation < generation_interval:
            return
        
        # Generate a decision using the AI agent
        try:
            # Create synthetic event data for autonomous generation
            synthetic_event = {
                'f0': 0.0,  # No specific pitch
                'rms_db': -80,  # Silence
                'onset': False,
                'centroid': 0.5,
                'instrument': 'autonomous',
                'harmonic_context': None,
                'rhythmic_context': None,
                't': current_time
            }
            
            # Make decision - AI will use learned patterns from memory
            decisions = self.ai_agent.process_event(
                synthetic_event, self.memory_buffer, self.clustering
            )
            
            # Process decisions
            if decisions:
                for decision in decisions:
                    # Map to MIDI
                    midi_params = self.feature_mapper.map_features_to_midi(
                        synthetic_event, {
                            'mode': decision.mode.value,
                            'confidence': decision.confidence,
                            'musical_params': decision.musical_params
                        },
                        decision.voice_type
                    )
                    
                    # IRCAM Phase 3: Apply behavior mode volume factor
                    volume_factor = self.behavior_controller.get_volume_factor()
                    midi_params.velocity = int(midi_params.velocity * volume_factor)
                    
                    # Send MIDI note
                    voice_type = decision.voice_type
                    if self.enable_mpe and self.mpe_midi_outputs.get(voice_type):
                        expression_level = self._calculate_expressive_intensity(decision, synthetic_event)
                        timbre_variation = self._calculate_timbre_variation(decision, synthetic_event)
                        pressure_sensitivity = self._calculate_pressure_sensitivity(decision, synthetic_event)
                        
                        if self.mpe_midi_outputs[voice_type].send_note(midi_params, voice_type, 0.0, 
                                                                       expression_level, timbre_variation, 
                                                                       pressure_sensitivity):
                            self.stats['notes_sent'] += 1
                            # IRCAM Phase 1.1: Track sent note to prevent learning from own output
                            self.own_output_tracker['recent_notes'].append(
                                (midi_params.note, current_time, midi_params.velocity)
                            )
                            # IRCAM Phase 3+: Track AI-generated event
                            if self.ai_agent and hasattr(self.ai_agent, 'phrase_generator') and self.ai_agent.phrase_generator:
                                ai_event = {
                                    'midi': midi_params.note,
                                    'velocity': midi_params.velocity,
                                    'voice_type': voice_type
                                }
                                self.ai_agent.phrase_generator.track_event(ai_event, source='ai')
                            # Log autonomous generation
                            self._log_conversation_response(midi_params, voice_type, decision.mode.value, current_time,
                                                           expression_level, pressure_sensitivity, timbre_variation)
                    
                    elif self.midi_outputs.get(voice_type):
                        if self.midi_outputs[voice_type].send_note(midi_params, voice_type):
                            self.stats['notes_sent'] += 1
                            # IRCAM Phase 1.1: Track sent note to prevent learning from own output
                            self.own_output_tracker['recent_notes'].append(
                                (midi_params.note, current_time, midi_params.velocity)
                            )
                            # IRCAM Phase 3+: Track AI-generated event
                            if self.ai_agent and hasattr(self.ai_agent, 'phrase_generator') and self.ai_agent.phrase_generator:
                                ai_event = {
                                    'midi': midi_params.note,
                                    'velocity': midi_params.velocity,
                                    'voice_type': voice_type
                                }
                                self.ai_agent.phrase_generator.track_event(ai_event, source='ai')
                            # Log autonomous generation
                            self._log_conversation_response(midi_params, voice_type, decision.mode.value, current_time,
                                                           0.0, 0.0, 0.0)
            
            self.last_autonomous_generation = current_time
            
        except Exception as e:
            print(f"⚠️ Autonomous generation error: {e}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Enhanced Drift Engine AI - AI Musical Partner with Rhythmic Analysis')
    parser.add_argument('--midi-port', type=str, help='MIDI output port name')
    parser.add_argument('--input-device', type=int, help='Audio input device index')
    parser.add_argument('--density', type=float, default=0.5, help='Musical density level (0.0-1.0)')
    parser.add_argument('--give-space', type=float, default=0.3, help='Give space factor (0.0-1.0)')
    parser.add_argument('--initiative', type=float, default=0.7, help='Initiative budget (0.0-1.0)')
    parser.add_argument('--no-rhythmic', action='store_true', help='Disable rhythmic analysis (harmonic only)')
    parser.add_argument('--no-mpe', action='store_true', help='Disable MPE MIDI mode (use standard MIDI)')
    parser.add_argument('--no-hybrid-perception', action='store_true', 
                       help='Disable hybrid perception (ratio + consonance analysis) - DEFAULT: ENABLED')
    parser.add_argument('--no-wav2vec', action='store_true',
                       help='Disable Wav2Vec 2.0 neural encoding - DEFAULT: ENABLED')
    parser.add_argument('--wav2vec-model', type=str, default='facebook/wav2vec2-base',
                       help='Wav2Vec model name (default: facebook/wav2vec2-base)')
    parser.add_argument('--no-gpu', action='store_true',
                       help='Disable GPU for Wav2Vec (use CPU) - DEFAULT: GPU ENABLED')
    parser.add_argument('--performance-duration', type=int, default=0, help='Performance duration in minutes (0 = no timeline)')
    parser.add_argument('--learn-target', type=str, help='Learn target fingerprint (description)')
    parser.add_argument('--load-target', type=str, help='Load target fingerprint from file')
    parser.add_argument('--no-autonomous', action='store_true', help='Disable autonomous generation (reactive only)')
    parser.add_argument('--autonomous-interval', type=float, default=3.0, help='Base interval for autonomous generation (seconds)')
    parser.add_argument('--bass-accompaniment', type=float, default=0.5, help='Probability (0-1) bass plays during human activity')
    parser.add_argument('--melody-while-active', action='store_true', help='Allow melody to play (sparse) while human is active')
    parser.add_argument('--debug-decisions', action='store_true', help='Show real-time decision explanations in terminal')
    parser.add_argument('--no-visualize', action='store_true', help='Disable multi-viewport visualization system - DEFAULT: ENABLED')
    parser.add_argument('--enable-live-training', action='store_true', 
                       help='Allow AudioOracle to learn from live input (may overwrite trained patterns)')
    
    args = parser.parse_args()
    
    # Create and start Enhanced Drift Engine AI
    drift_ai = EnhancedDriftEngineAI(
        midi_port=args.midi_port,
        input_device=args.input_device,
        enable_rhythmic=not args.no_rhythmic,
        enable_mpe=not args.no_mpe,
        performance_duration=args.performance_duration,
        enable_hybrid_perception=not args.no_hybrid_perception,
        enable_wav2vec=not args.no_wav2vec,
        enable_live_training=args.enable_live_training,
        wav2vec_model=args.wav2vec_model,
        use_gpu=not args.no_gpu,
        debug_decisions=args.debug_decisions,
        enable_visualization=not args.no_visualize
    )
    
    # Set parameters
    drift_ai.set_density_level(args.density)
    drift_ai.set_give_space_factor(args.give_space)
    drift_ai.set_initiative_budget(args.initiative)
    
    # Set autonomous generation parameters
    drift_ai.autonomous_generation_enabled = not args.no_autonomous
    drift_ai.autonomous_interval_base = args.autonomous_interval
    drift_ai.bass_accompaniment_probability = args.bass_accompaniment
    drift_ai.melody_silence_when_active = not args.melody_while_active
    
    if not args.no_autonomous:
        print(f"🤖 Autonomous generation enabled (interval: {args.autonomous_interval:.1f}s)")
        print(f"🎸 Bass accompaniment: {args.bass_accompaniment:.0%} probability")
        print(f"🎤 Melody while active: {'Yes (sparse)' if args.melody_while_active else 'No (silent)'}")
    
    # Handle target learning/loading
    if args.learn_target:
        print(f"🎯 Starting target learning: {args.learn_target}")
        if not drift_ai.hybrid_detector.start_target_learning(args.learn_target):
            print("❌ Failed to start target learning")
            return 1
    elif args.load_target:
        print(f"🎯 Loading target fingerprint: {args.load_target}")
        if not drift_ai.hybrid_detector.load_target_fingerprint(args.load_target):
            print("❌ Failed to load target fingerprint")
            return 1
    
    # Setup signal handler for Ctrl+C (works with Qt)
    def signal_handler(sig, frame):
        print("\n🛑 Stopping Enhanced Drift Engine AI...")
        drift_ai.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start system
    if not drift_ai.start():
        print("❌ Failed to start Enhanced Drift Engine AI")
        return 1
    
    try:
        print("\n🎵 Enhanced Drift Engine AI is running!")
        print("Press Ctrl+C to stop...")
        
        # Keep running until system stops naturally or user interrupts
        while drift_ai.running:
            # Process Qt events if visualization is enabled (do this FIRST, before sleep)
            if drift_ai.visualization_manager:
                for _ in range(10):  # Process events 10 times per loop
                    drift_ai.visualization_manager.process_events()
                    time.sleep(0.01)  # 10ms between each process call = 100ms total
            else:
                time.sleep(0.1)  # Shorter sleep for more responsive Ctrl+C
        
        # If we exited the loop naturally (performance complete), stop cleanly
        if not drift_ai.running:
            print("🎬 Performance completed, exiting...")
            return 0
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping Enhanced Drift Engine AI...")
        drift_ai.stop()
        return 0
    except SystemExit:
        # Triggered by signal handler
        return 0
    
    except Exception as e:
        print(f"❌ Error: {e}")
        drift_ai.stop()
        return 1

if __name__ == "__main__":
    exit(main())
