# main.py
# Drift Engine AI - Main Application
# Complete AI-driven musical partner system

import time
import threading
import argparse
import os
from typing import Dict, Optional, List

from listener.jhs_listener_core import DriftListener, Event
from memory.memory_buffer import MemoryBuffer
from memory.clustering import MusicalClustering
from memory.polyphonic_audio_oracle_mps import PolyphonicAudioOracleMPS
from agent.ai_agent import AIAgent
from mapping.feature_mapper import FeatureMapper
from midi_io.midi_output import MIDIOutput
from core.logger import PerformanceLogger
from performance_timeline_manager import PerformanceTimelineManager, PerformanceConfig

class DriftEngineAI:
    """
    Main Drift Engine AI application
    Complete AI-driven musical partner system
    """
    
    def __init__(self, midi_port: Optional[str] = None, input_device: Optional[int] = None, performance_duration: int = 5):
        # Core components
        self.listener: Optional[DriftListener] = None
        self.memory_buffer = MemoryBuffer()
        self.clustering = PolyphonicAudioOracleMPS(
            distance_threshold=0.15,
            distance_function='euclidean',
            feature_dimensions=15,  # Updated for polyphonic features
            adaptive_threshold=True,
            chord_similarity_weight=0.3,
            use_mps=True
        )  # Live mode: PolyphonicAudioOracleMPS for real-time performance
        self.ai_agent = AIAgent()
        self.feature_mapper = FeatureMapper()
        self.midi_output: Optional[MIDIOutput] = None
        
        # Performance logging
        self.performance_logger = PerformanceLogger(log_dir="logs")
        
        # Performance arc system
        self.timeline_manager: Optional[PerformanceTimelineManager] = None
        self.performance_duration = performance_duration
        
        # Configuration
        self.midi_port = midi_port or "IAC Driver Melody Channel"
        self.input_device = input_device
        
        # Persistence configuration
        self.data_dir = "ai_learning_data"
        self.memory_file = os.path.join(self.data_dir, "musical_memory.json")
        self.clustering_file = os.path.join(self.data_dir, "polyphonic_audio_oracle_model.json")  # Updated for polyphonic MPS
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # State
        self.running = False
        self.main_thread: Optional[threading.Thread] = None
        
        # Statistics
        self.stats = {
            'events_processed': 0,
            'decisions_made': 0,
            'notes_sent': 0,
            'uptime': 0.0,
            'last_event_time': 0.0
        }
        
        # Self-output tracking for feedback loop testing
        self.own_output_tracker = {
            'recent_notes': [],  # List of (note, time, features) tuples
            'max_age_seconds': 5.0,  # How long to remember our own output
            'similarity_threshold': 0.8,  # Threshold for recognizing our own output
            'self_awareness_enabled': True,  # Enable self-awareness filtering
            'learning_from_self': False  # Whether to learn from own output
        }
        
        self.start_time = time.time()
        
        # Initialize timeline manager if performance duration is specified
        if self.performance_duration > 0:
            self._initialize_timeline_manager()
    
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
    
    def start(self) -> bool:
        """Start the Drift Engine AI system"""
        print("🎵 Starting Drift Engine AI...")
        
        try:
            # Load previous learning data
            self._load_learning_data()
            
            # Start MIDI output
            self.midi_output = MIDIOutput(self.midi_port)
            if not self.midi_output.start():
                print("❌ Failed to start MIDI output")
                return False
            
            # Start listener
            self.listener = DriftListener(
                ref_fn=self._get_reference_frequency,
                a4_fn=lambda: 440.0,
                device=self.input_device
            )
            
            # Set up event callback
            self.listener.start(self._on_audio_event)
            
            # Start timeline manager if available
            if self.timeline_manager:
                self.timeline_manager.start_performance()
                print(f"🎭 Performance arc active: {self.performance_duration} minutes")
            
            # Start main loop
            self.running = True
            self.main_thread = threading.Thread(target=self._main_loop, daemon=True)
            self.main_thread.start()
            
            print("✅ Drift Engine AI started successfully!")
            print(f"📡 MIDI Output: {self.midi_output.port.name if self.midi_output.port else 'None'}")
            print(f"🎤 Audio Input: Device {self.input_device if self.input_device else 'Default'}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to start Drift Engine AI: {e}")
            return False
    
    def stop(self):
        """Stop the Drift Engine AI system"""
        print("🛑 Stopping Drift Engine AI...")
        
        self.running = False
        
        # Save learning data before stopping
        self._save_learning_data()
        
        # Stop listener
        if self.listener:
            self.listener.stop()
        
        # Stop MIDI output
        if self.midi_output:
            self.midi_output.stop()
        
        # Close performance logger
        if hasattr(self.performance_logger, 'close'):
            self.performance_logger.close()
        
        # Wait for main thread
        if self.main_thread and self.main_thread.is_alive():
            self.main_thread.join(timeout=2.0)
        
        print("✅ Drift Engine AI stopped")
    
    def _on_audio_event(self, *args):
        """Handle audio events from listener"""
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
        
        # Check if this is our own output (feedback loop test)
        own_output_match = self._check_own_output_detection(event_data)
        if own_output_match and self.own_output_tracker['self_awareness_enabled']:
            print(f"🎵 FEEDBACK LOOP: Detected own output - skipping learning")
            print(f"   Original: Note {own_output_match['note']} ({own_output_match['expected_freq']:.1f}Hz)")
            print(f"   Detected: {event_data.get('f0', 0):.1f}Hz (diff={abs(event_data.get('f0', 0) - own_output_match['expected_freq']):.1f}Hz)")
            
            # Skip learning from own output to prevent feedback loops
            # Still log for analysis but don't add to memory or clustering
            self.performance_logger.log_audio_analysis(
                instant_pitch=event_data.get('f0', 0.0),
                avg_pitch=event_data.get('f0', 0.0),
                onset_detected=event_data.get('onset', False),
                onset_rate=1.0 if event_data.get('onset', False) else 0.0,
                activity_level=min(1.0, max(0.0, (event_data.get('rms_db', -80) + 80) / 80)),
                energy_level=event_data.get('rms_db', -80)
            )
            
            # Update statistics but skip learning
            self.stats['events_processed'] += 1
            self.stats['last_event_time'] = current_time
            self.stats['own_output_detected'] = self.stats.get('own_output_detected', 0) + 1
            
            # Skip the rest of the processing for own output
            return
        
        # Log audio analysis for human input
        self.performance_logger.log_audio_analysis(
            instant_pitch=event_data.get('f0', 0.0),
            avg_pitch=event_data.get('f0', 0.0),  # Using same value for simplicity
            onset_detected=event_data.get('onset', False),
            onset_rate=1.0 if event_data.get('onset', False) else 0.0,
            activity_level=min(1.0, max(0.0, (event_data.get('rms_db', -80) + 80) / 80)),
            energy_level=event_data.get('rms_db', -80)
        )
        
        # Add to memory buffer (only for human input)
        self.memory_buffer.add_moment(event_data)
        
        # Learn musical patterns with Polyphonic AudioOracle MPS (only from human input)
        # Pass event_data directly for polyphonic feature extraction
        self.clustering.add_sequence([event_data])
        
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
            
            # Get performance guidance
            guidance = self.timeline_manager.get_performance_guidance()
            
            # Check if we should respond based on timeline
            if not guidance['should_respond']:
                return  # Skip processing if timeline says we should be silent
        
        # Process with AI agent
        decisions = self.ai_agent.process_event(
            event_data, self.memory_buffer, self.clustering
        )
        
        # Apply instrument-aware guidance from timeline
        if self.timeline_manager:
            guidance = self.timeline_manager.get_performance_guidance()
            detected_instrument = guidance.get('detected_instrument', 'unknown')
            
            # Override instrument in event data for better decision making
            if detected_instrument != 'unknown':
                event_data['instrument'] = detected_instrument
        
        # Log musical features being analyzed
        if own_output_match:
            print(f"🎼 MUSICAL ANALYSIS: Processing own output")
            print(f"   Pitch: {event_data.get('f0', 0):.1f}Hz")
            print(f"   RMS: {event_data.get('rms_db', -80):.1f}dB")
            print(f"   Centroid: {event_data.get('centroid', 0):.1f}Hz")
            print(f"   Onset: {event_data.get('onset', False)}")
            print(f"   Instrument: {event_data.get('instrument', 'unknown')}")
            print(f"   AI Response: {len(decisions)} decisions generated")
        
        if decisions:
            self.stats['decisions_made'] += len(decisions)
            
            # Process each decision (melodic and bass)
            for decision in decisions:
                # Debug: print voice type
                print(f"🎹 Processing {decision.voice_type} decision: mode={decision.mode.value}, conf={decision.confidence:.2f}")
                
                # Map to MIDI parameters
                midi_params = self.feature_mapper.map_features_to_midi(
                    event_data, {
                        'mode': decision.mode.value,
                        'confidence': decision.confidence,
                        'musical_params': decision.musical_params
                    },
                    decision.voice_type
                )
                
                # Debug: print resulting note
                print(f"🎵 {decision.voice_type.capitalize()} note: {midi_params.note}")
            
                # Log performance event
                instrument = decision.instrument if hasattr(decision, 'instrument') else event_data.get('instrument', 'unknown')
                self.performance_logger.log_system_event(
                    event_type="ai_decision",
                    voice=decision.voice_type,
                    trigger_reason=f"{decision.mode.value}_mode",
                    activity_level=decision.confidence,
                    phase="generation",
                    additional_data=f"note={midi_params.note},vel={midi_params.velocity},instrument={instrument}"
                )
                
                # Send MIDI note
                if self.midi_output:
                    channel = decision.voice_type  # 'melodic' or 'bass'
                    if self.midi_output.send_note(midi_params, channel):
                        self.stats['notes_sent'] += 1
                        
                        # Track our own output for feedback loop testing
                        self._track_own_output(midi_params, channel)
                    
                    # Get instrument from decision (preferred) or event data
                    instrument = decision.instrument if hasattr(decision, 'instrument') else event_data.get('instrument', 'unknown')
                    
                    print(f"🎼 AI Decision: {decision.mode.value} "
                          f"(conf={decision.confidence:.2f}) -> "
                          f"Note {midi_params.note} (vel={midi_params.velocity}) "
                          f"[{instrument}]")
    
    def _main_loop(self):
        """Main processing loop"""
        last_clustering_time = time.time()
        clustering_interval = 30.0  # Retrain clustering every 30 seconds
        
        while self.running:
            try:
                current_time = time.time()
                
                # Update statistics
                self.stats['uptime'] = current_time - self.start_time
                
                # Retrain clustering periodically
                if current_time - last_clustering_time > clustering_interval:
                    if self.clustering.retrain_if_needed(self.memory_buffer):
                        print(f"🧠 Retrained clustering with {len(self.memory_buffer.get_all_moments())} moments")
                    last_clustering_time = current_time
                
                # Print status periodically
                if int(current_time) % 60 == 0:  # Every minute
                    self._print_status()
                
                time.sleep(0.1)
                
            except Exception as e:
                print(f"❌ Main loop error: {e}")
                time.sleep(1.0)
    
    def _print_status(self):
        """Print system status"""
        buffer_stats = self.memory_buffer.get_buffer_stats()
        agent_state = self.ai_agent.get_agent_state()
        
        # Get recent instrument classification
        recent_moments = self.memory_buffer.get_recent_moments(10.0)
        instruments = [moment.event_data.get('instrument', 'unknown') for moment in recent_moments[-5:]]
        instrument_summary = {}
        for inst in instruments:
            instrument_summary[inst] = instrument_summary.get(inst, 0) + 1
        
        # Show feedback loop activity
        active_notes = len(self.own_output_tracker['recent_notes'])
        
        print(f"\n📊 Status Update:")
        print(f"   Events: {self.stats['events_processed']}, "
              f"Decisions: {self.stats['decisions_made']}, "
              f"Notes: {self.stats['notes_sent']}")
        print(f"   Memory: {buffer_stats['count']} moments, "
              f"{buffer_stats['duration_seconds']:.1f}s")
        print(f"   Agent: {agent_state.current_mode} mode, "
              f"conf={agent_state.confidence:.2f}")
        
        # Add timeline information if available
        if self.timeline_manager:
            progress = self.timeline_manager.get_performance_progress()
            guidance = self.timeline_manager.get_performance_guidance()
            print(f"   Timeline: {progress['progress_percent']:.1f}% complete, "
                  f"Phase: {guidance['current_phase']}, "
                  f"Engagement: {guidance['engagement_level']:.2f}")
            print(f"   Performance: {guidance['behavior_mode']} mode, "
                  f"Silence: {guidance['silence_mode']}, "
                  f"Momentum: {guidance['musical_momentum']:.2f}")
        
        print(f"   Instruments: {dict(instrument_summary)}")
        print(f"   Feedback Loop: {active_notes} active notes being tracked")
        own_output_count = self.stats.get('own_output_detected', 0)
        print(f"   Self-Awareness: {own_output_count} own outputs detected and filtered")
        print(f"   Uptime: {self.stats['uptime']:.1f}s")
    
    
    def _get_reference_frequency(self, midi_note: int) -> float:
        """Get reference frequency for MIDI note (12TET)"""
        return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))
    
    def set_self_awareness(self, enabled: bool):
        """Enable or disable self-awareness filtering"""
        self.own_output_tracker['self_awareness_enabled'] = enabled
        status = "enabled" if enabled else "disabled"
        print(f"🎯 Self-awareness filtering {status}")
    
    def set_learning_from_self(self, enabled: bool):
        """Enable or disable learning from own output"""
        self.own_output_tracker['learning_from_self'] = enabled
        status = "enabled" if enabled else "disabled"
        print(f"🎓 Learning from own output {status}")
    
    def _track_own_output(self, midi_params, channel: int):
        """Track our own MIDI output for feedback loop testing"""
        current_time = time.time()
        
        # Calculate expected frequency from MIDI note
        expected_freq = self._get_reference_frequency(midi_params.note)
        
        # Store our output with timestamp and expected features
        self.own_output_tracker['recent_notes'].append({
            'note': midi_params.note,
            'velocity': midi_params.velocity,
            'channel': channel,
            'time': current_time,
            'expected_freq': expected_freq,
            'duration': midi_params.duration
        })
        
        # Clean up old entries
        self.own_output_tracker['recent_notes'] = [
            note for note in self.own_output_tracker['recent_notes']
            if current_time - note['time'] < self.own_output_tracker['max_age_seconds']
        ]
        
        print(f"🎯 Tracking own output: Note {midi_params.note} (freq={expected_freq:.1f}Hz, ch={channel})")
    
    def _check_own_output_detection(self, event_data: Dict) -> Optional[Dict]:
        """Check if the incoming audio matches our recent output"""
        current_time = time.time()
        incoming_freq = event_data.get('f0', 0.0)
        
        if incoming_freq <= 0:
            return None
        
        # Look for matches in our recent output
        for note_info in self.own_output_tracker['recent_notes']:
            time_diff = current_time - note_info['time']
            
            # Check if timing is reasonable (within 2 seconds)
            if time_diff > 2.0:
                continue
            
            # Check frequency similarity (within 10% tolerance)
            freq_diff = abs(incoming_freq - note_info['expected_freq'])
            freq_tolerance = note_info['expected_freq'] * 0.1
            
            if freq_diff <= freq_tolerance:
                print(f"🔄 SELF-DETECTION: Detected own output!")
                print(f"   Expected: Note {note_info['note']} ({note_info['expected_freq']:.1f}Hz)")
                print(f"   Detected: {incoming_freq:.1f}Hz (diff={freq_diff:.1f}Hz)")
                print(f"   Time lag: {time_diff:.2f}s")
                print(f"   Features: {event_data}")
                
                return note_info
        
        return None
    
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
        return {
            'running': self.running,
            'stats': self.stats.copy(),
            'memory_buffer': self.memory_buffer.get_buffer_stats(),
            'agent_state': self.ai_agent.get_agent_state(),
            'midi_output': self.midi_output.get_status() if self.midi_output else None,
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
                'mps_enabled': getattr(self.clustering, 'use_mps', False)
            }
        }
    
    def _load_learning_data(self):
        """Load previous learning data from files"""
        print("🧠 Loading previous learning data...")
        
        # Load memory buffer
        memory_loaded = self.memory_buffer.load_from_file(self.memory_file)
        
        # Try to load the best available trained model (prioritize enhanced models)
        enhanced_model_paths = [
            "JSON/grab-a-hold_enhanced.json",  # Music theory transformer enhanced model
            "JSON/test_music_theory.json",     # Music theory transformer test model
            "JSON/test_polyphonic_model.json", # Polyphonic MPS model
            "JSON/test_mps_polyphonic.json",   # MPS polyphonic model
            "JSON/test_mps_500.json"           # MPS test model
        ]
        
        model_loaded = False
        for model_path in enhanced_model_paths:
            if os.path.exists(model_path):
                print(f"🎓 Loading trained model from {model_path}...")
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
            # Load Polyphonic AudioOracle MPS model from default location
            print("📝 No trained models found, loading default model...")
            polyphonic_oracle_loaded = self.clustering.load_from_file(self.clustering_file)
        
        if memory_loaded and polyphonic_oracle_loaded:
            print("✅ Successfully loaded previous learning data")
        elif memory_loaded:
            print("✅ Loaded musical memory, Polyphonic AudioOracle MPS will learn from it")
        elif polyphonic_oracle_loaded:
            print("✅ Loaded Polyphonic AudioOracle MPS model, building new memory")
        else:
            print("📝 Starting with fresh learning data")
    
    def _save_learning_data(self):
        """Save current learning data to files"""
        print("💾 Saving learning data...")
        
        # Save memory buffer
        memory_saved = self.memory_buffer.save_to_file(self.memory_file)
        
        # Save Polyphonic AudioOracle MPS model
        polyphonic_oracle_saved = self.clustering.save_to_file(self.clustering_file)
        
        if memory_saved and polyphonic_oracle_saved:
            print("✅ Successfully saved all learning data")
        elif memory_saved:
            print("✅ Saved musical memory")
        elif polyphonic_oracle_saved:
            print("✅ Saved Polyphonic AudioOracle MPS model")
        else:
            print("⚠️ No learning data to save")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Drift Engine AI - AI Musical Partner')
    parser.add_argument('--midi-port', type=str, help='MIDI output port name')
    parser.add_argument('--input-device', type=int, help='Audio input device index')
    parser.add_argument('--density', type=float, default=0.5, help='Musical density level (0.0-1.0)')
    parser.add_argument('--give-space', type=float, default=0.3, help='Give space factor (0.0-1.0)')
    parser.add_argument('--initiative', type=float, default=0.7, help='Initiative budget (0.0-1.0)')
    parser.add_argument('--duration', type=int, default=5, help='Performance duration in minutes (enables performance arc)')
    
    args = parser.parse_args()
    
    # Create and start Drift Engine AI
    drift_ai = DriftEngineAI(
        midi_port=args.midi_port,
        input_device=args.input_device,
        performance_duration=args.duration
    )
    
    # Set parameters
    drift_ai.set_density_level(args.density)
    drift_ai.set_give_space_factor(args.give_space)
    drift_ai.set_initiative_budget(args.initiative)
    
    # Start system
    if not drift_ai.start():
        print("❌ Failed to start Drift Engine AI")
        return 1
    
    try:
        print("\n🎵 Drift Engine AI is running!")
        print("Press Ctrl+C to stop...")
        
        # Keep running
        while True:
            time.sleep(1.0)
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping Drift Engine AI...")
        drift_ai.stop()
        return 0
    
    except Exception as e:
        print(f"❌ Error: {e}")
        drift_ai.stop()
        return 1

if __name__ == "__main__":
    exit(main())
