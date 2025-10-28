#!/usr/bin/env python3
"""
MusicHal_bass.py - Bass-only version of MusicHal_9000
Identical functionality but only outputs to IAC Driver Bass channel

This version:
- Uses same AI learning and decision-making as MusicHal_9000
- Only outputs bass responses via MPE MIDI to IAC Driver Bass
- Maintains all training, memory, and intelligence features
- Perfect for bass-focused musical interaction
"""

import time
import threading
import argparse
import os
import numpy as np
from typing import Dict, Optional, List

# Existing harmonic components (unchanged)
from listener.jhs_listener_core import DriftListener, Event
from memory.memory_buffer import MemoryBuffer
from memory.clustering import MusicalClustering
from memory.polyphonic_audio_oracle import PolyphonicAudioOracle
from agent.ai_agent import AIAgent
from agent.phrase_generator import PhraseGenerator
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

class MusicHalBass:
    """
    MusicHal Bass - Bass-only AI musical agent
    Identical to MusicHal_9000 but only outputs bass responses
    """
    
    def __init__(self, midi_port: Optional[str] = None, input_device: Optional[int] = None, 
                 enable_mpe: bool = True, enable_rhythmic: bool = True, enable_correlation: bool = True):
        """
        Initialize MusicHal Bass system
        
        Args:
            midi_port: MIDI output port name (defaults to IAC Driver Bass)
            input_device: Audio input device index
            enable_mpe: Enable MPE MIDI output
            enable_rhythmic: Enable rhythmic analysis
            enable_correlation: Enable cross-modal correlation
        """
        
        # System state
        self.running = False
        self.listener = None
        self.audio_thread = None
        
        # Performance tracking
        self.stats = {
            'events_processed': 0,
            'notes_sent': 0,
            'chord_changes': 0,
            'rhythmic_events': 0,
            'correlation_events': 0,
            'start_time': time.time()
        }
        
        # GPT-OSS integration
        self.gpt_oss_enabled = False
        self.gpt_oss_role_development = None
        
        # Configuration - BASS ONLY
        self.midi_ports = {
            'bass': "IAC Driver Bass"  # Only bass output
        }
        self.input_device = input_device
        
        # Persistence configuration
        self.model_dir = "ai_learning_data"
        self.conversation_log = "logs/musichal_bass_conversations.csv"
        
        # Performance configuration
        self.config = PerformanceConfig(
            duration_minutes=30,  # Default 30 minutes
            arc_file_path="ai_learning_data/itzama_performance_arc.json",
            engagement_profile="balanced",
            silence_tolerance=5.0,
            adaptation_rate=0.1
        )
        
        # Initialize components
        self.memory_buffer = MemoryBuffer()
        self.clustering = MusicalClustering()
        self.audio_oracle = PolyphonicAudioOracle()
        self.ai_agent = AIAgent()
        self.feature_mapper = FeatureMapper()
        self.performance_logger = PerformanceLogger()
        self.timeline_manager = PerformanceTimelineManager(self.config)
        
        # Rhythmic components
        self.enable_rhythmic = enable_rhythmic
        if enable_rhythmic:
            self.rhythmic_analyzer = LightweightRhythmicAnalyzer()
            self.rhythm_oracle = RhythmOracle()
            self.rhythmic_behavior = RhythmicBehaviorEngine()
            self.phrase_generator = PhraseGenerator(self.rhythm_oracle, self.audio_oracle)
        else:
            # Create a dummy rhythm oracle for phrase generator
            self.rhythm_oracle = RhythmOracle()
            self.phrase_generator = PhraseGenerator(self.rhythm_oracle, self.audio_oracle)
        
        # Correlation components
        self.enable_correlation = enable_correlation
        if enable_correlation:
            self.unified_decision_engine = UnifiedDecisionEngine()
        
        # Hybrid detection
        self.hybrid_detector = HybridDetector()
        
        # MIDI outputs
        self.midi_outputs = {}
        self.mpe_midi_outputs = {}
        self.enable_mpe = enable_mpe
        
        # Activity tracking for autonomous generation
        self.last_activity_time = time.time()
        self.silence_timeout = 5.0  # 5 seconds of silence before autonomous mode
        self.autonomous_interval_base = 3.0  # Base interval for autonomous generation
        
        # Voice-specific timing (bass only) - More conservative like MusicHal_9000
        self.last_bass_time = time.time()
        self.bass_pause_min = 3.0  # Minimum pause between bass notes (increased)
        self.bass_pause_max = 8.0  # Maximum pause between bass notes (increased)
        
        print("🎸 MusicHal Bass initialized")
        print(f"   MIDI Output: {self.midi_ports['bass']}")
        print(f"   MPE Enabled: {enable_mpe}")
        print(f"   Rhythmic Analysis: {enable_rhythmic}")
        print(f"   Cross-modal Correlation: {enable_correlation}")
    
    def start(self) -> bool:
        """Start the MusicHal Bass system"""
        try:
            print("🎸 Starting MusicHal Bass...")
            
            # Load existing models
            if not self._load_models():
                print("⚠️ No existing models found, starting fresh")
            
            # Initialize MIDI outputs
            if not self._initialize_midi_outputs():
                return False
            
            # Initialize audio listener
            if not self._initialize_audio_listener():
                return False
            
            # Start audio processing
            if not self._start_audio_processing():
                return False
            
            self.running = True
            print("✅ MusicHal Bass started successfully!")
            print("🎹 Play your instrument - the bass AI will respond")
            print("   Press Ctrl+C to stop")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to start MusicHal Bass: {e}")
            return False
    
    def stop(self):
        """Stop the MusicHal Bass system"""
        print("🛑 Stopping MusicHal Bass...")
        self.running = False
        
        if self.listener:
            self.listener.stop()
        
        # Stop MIDI outputs
        for output in self.midi_outputs.values():
            if output:
                output.stop()
        
        for output in self.mpe_midi_outputs.values():
            if output:
                output.stop()
        
        print("✅ MusicHal Bass stopped")
    
    def _load_models(self) -> bool:
        """Load existing AI models"""
        try:
            # Load audio oracle
            oracle_path = os.path.join(self.model_dir, "audio_oracle_model.json")
            if os.path.exists(oracle_path):
                self.audio_oracle.load_from_file(oracle_path)
                print("✅ Loaded audio oracle model")
            
            # Load rhythm oracle
            if self.enable_rhythmic:
                rhythm_path = os.path.join(self.model_dir, "rhythm_oracle_model.json")
                if os.path.exists(rhythm_path):
                    self.rhythm_oracle.load_patterns(rhythm_path)
                    print("✅ Loaded rhythm oracle model")
            
            return True
            
        except Exception as e:
            print(f"⚠️ Error loading models: {e}")
            return False
    
    def _initialize_midi_outputs(self) -> bool:
        """Initialize MIDI outputs (bass only)"""
        try:
            if self.enable_mpe:
                # Initialize MPE outputs for bass only
                self.mpe_midi_outputs['bass'] = MPEMIDIOutput(self.midi_ports['bass'], enable_mpe=True)
                if not self.mpe_midi_outputs['bass'].start():
                    print(f"❌ Failed to start MPE MIDI output for bass")
                    return False
                print(f"✅ MPE MIDI output started for bass: {self.midi_ports['bass']}")
            else:
                # Initialize standard MIDI outputs for bass only
                self.midi_outputs['bass'] = MIDIOutput(self.midi_ports['bass'])
                if not self.midi_outputs['bass'].start():
                    print(f"❌ Failed to start MIDI output for bass")
                    return False
                print(f"✅ MIDI output started for bass: {self.midi_ports['bass']}")
            
            return True
            
        except Exception as e:
            print(f"❌ MIDI initialization error: {e}")
            return False
    
    def _initialize_audio_listener(self) -> bool:
        """Initialize audio listener"""
        try:
            def ref_fn(midi_note: int) -> float:
                """Reference frequency function"""
                return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))
            
            def a4_fn() -> float:
                """A4 reference frequency"""
                return 440.0
            
            self.listener = DriftListener(ref_fn, a4_fn)
            print("✅ Audio listener initialized")
            return True
            
        except Exception as e:
            print(f"❌ Audio listener initialization error: {e}")
            return False
    
    def _get_audio_buffer(self) -> np.ndarray:
        """Get audio buffer from listener"""
        try:
            if self.listener and hasattr(self.listener, '_ring'):
                # Get the ring buffer directly
                return self.listener._ring.copy()
            else:
                # Return silence if no audio available
                return np.zeros(1024, dtype=np.float32)
        except Exception as e:
            print(f"⚠️ Audio buffer error: {e}")
            return np.zeros(1024, dtype=np.float32)
    
    def _start_audio_processing(self) -> bool:
        """Start audio processing thread"""
        try:
            self.listener.start(self._on_audio_event)
            print("✅ Audio processing started")
            return True
            
        except Exception as e:
            print(f"❌ Audio processing error: {e}")
            return False
    
    def _on_audio_event(self, *args):
        """Handle audio events - BASS ONLY RESPONSES"""
        try:
            event = args[0] if args else None
            if event is None:
                return
            
            self.stats['events_processed'] += 1
            current_time = time.time()
            
            # Update activity tracking
            self.last_activity_time = current_time
            
            # Store event in memory
            event_data = {
                'f0': event.f0,
                'rms_db': event.rms_db,
                'centroid': event.centroid,
                'cents': event.cents,
                'ioi': event.ioi,
                'instrument': event.instrument,
                'harmonic_context': event.harmonic_context,
                'rhythmic_context': event.rhythmic_context
            }
            self.memory_buffer.add_moment(event_data)
            
            # Get hybrid detection result
            audio_buffer = self._get_audio_buffer()
            detection_result = self.hybrid_detector.detect(audio_buffer, event_data)
            
            # Process with unified decision engine if enabled
            if self.enable_correlation:
                # Create harmonic context dict
                harmonic_context = {
                    'current_chord': event.harmonic_context.current_chord if event.harmonic_context else '',
                    'key_signature': event.harmonic_context.key_signature if event.harmonic_context else '',
                    'confidence': event.harmonic_context.confidence if event.harmonic_context else 0.0,
                    'stability': event.harmonic_context.stability if event.harmonic_context else 0.0
                }
                
                # Create rhythmic context dict
                rhythmic_context = {
                    'current_tempo': event.rhythmic_context.current_tempo if event.rhythmic_context else 0.0,
                    'meter': event.rhythmic_context.meter if event.rhythmic_context else '4/4',
                    'beat_strength': 0.8  # Default value
                }
                
                cross_modal_context = CrossModalContext(
                    harmonic_context=harmonic_context,
                    rhythmic_context=rhythmic_context,
                    correlation_patterns=[],
                    temporal_alignment={'alignment_strength': 0.8},
                    musical_context=MusicalContext.UNKNOWN
                )
                
                decision = self.unified_decision_engine.make_unified_decision(
                    harmonic_context, rhythmic_context, cross_modal_context
                )
            else:
                # Fallback to AI agent decision
                decision = self.ai_agent.make_decision(event, self.memory_buffer)
            
            # Check if we should generate bass response
            if self._should_generate_bass_response(current_time):
                self._generate_bass_response(event, decision, current_time)
            
            # Update performance timeline
            self.timeline_manager.update_performance_state(human_activity=True)
            
            # Log performance
            self.performance_logger.log_system_event(
                event_type="audio_event",
                voice="bass",
                trigger_reason="audio_input",
                activity_level=event.rms_db / 100.0,  # Convert dB to 0-1 range
                phase="input",
                additional_data=f"f0={event.f0:.1f},midi={event.midi}"
            )
            
        except Exception as e:
            print(f"⚠️ Audio event error: {e}")
    
    def _should_generate_bass_response(self, current_time: float) -> bool:
        """Check if we should generate a bass response"""
        # Check timing constraints
        time_since_bass = current_time - self.last_bass_time
        
        # Check if enough time has passed
        if time_since_bass < self.bass_pause_min:
            return False
        
        # Check for silence timeout (autonomous mode)
        time_since_activity = current_time - self.last_activity_time
        if time_since_activity > self.silence_timeout:
            # Autonomous mode - generate bass response (but still respect timing)
            return time_since_bass >= self.bass_pause_min
        
        # Check if we should respond to user input (but be more conservative)
        if time_since_bass > self.bass_pause_max:
            # Add some randomness to avoid predictable patterns
            import random
            if random.random() < 0.3:  # Only 30% chance even when timing allows
                return True
        
        return False
    
    def _generate_bass_response(self, event: Event, decision, current_time: float):
        """Generate bass response"""
        try:
            # Convert event to feature data
            event_data = {
                'f0': event.f0,
                'rms_db': event.rms_db,
                'centroid': event.centroid,
                'cents': event.cents,
                'ioi': event.ioi,
                'instrument': event.instrument
            }
            
            # Convert decision to decision data
            decision_data = {
                'mode': decision.response_mode.value if hasattr(decision, 'response_mode') else 'imitate',
                'confidence': decision.confidence if hasattr(decision, 'confidence') else 0.5,
                'musical_params': decision.joint_decision if hasattr(decision, 'joint_decision') else {}
            }
            
            # Map to MIDI parameters (bass voice only)
            midi_params = self.feature_mapper.map_features_to_midi(
                event_data, decision_data, voice_type="bass"
            )
            
            # Send MIDI note
            if self.enable_mpe and self.mpe_midi_outputs.get('bass'):
                # MPE mode: send with enhanced expressiveness
                expression_level = self._calculate_expression_level(decision, event_data)
                timbre_variation = self._calculate_timbre_variation(decision, event_data)
                pressure_sensitivity = self._calculate_pressure_sensitivity(decision, event_data)
                
                if self.mpe_midi_outputs['bass'].send_note(midi_params, 'bass', 0.0, expression_level, 
                                                          timbre_variation, pressure_sensitivity):
                    self.stats['notes_sent'] += 1
                    # Log AI response
                    mode_value = decision.response_mode.value if hasattr(decision, 'response_mode') else 'imitate'
                    self._log_conversation_response(midi_params, 'bass', mode_value, current_time,
                                                   expression_level, pressure_sensitivity, timbre_variation)
                
            elif self.midi_outputs.get('bass'):
                # Standard MIDI mode: send bass note
                if self.midi_outputs['bass'].send_note(midi_params, 'bass'):
                    self.stats['notes_sent'] += 1
                    # Log AI response
                    mode_value = decision.response_mode.value if hasattr(decision, 'response_mode') else 'imitate'
                    self._log_conversation_response(midi_params, 'bass', mode_value, current_time)
            
            # Update bass timing
            self.last_bass_time = current_time
            
        except Exception as e:
            print(f"⚠️ Bass response error: {e}")
    
    def _calculate_expression_level(self, decision, event_data: Dict) -> float:
        """Calculate expression level for MPE"""
        base_expression = 0.5
        
        # Adjust based on decision confidence
        if hasattr(decision, 'confidence'):
            base_expression += (decision.confidence - 0.5) * 0.3
        
        # Adjust based on audio intensity
        rms_db = event_data.get('rms_db', -60.0)
        intensity_factor = max(0.0, min(1.0, (rms_db + 80.0) / 80.0))
        base_expression += intensity_factor * 0.2
        
        return max(0.0, min(1.0, base_expression))
    
    def _calculate_timbre_variation(self, decision, event_data: Dict) -> float:
        """Calculate timbre variation for MPE"""
        base_timbre = 0.5
        
        # Adjust based on spectral centroid
        centroid = event_data.get('centroid', 1000.0)
        brightness_factor = max(0.0, min(1.0, centroid / 4000.0))
        base_timbre += brightness_factor * 0.3
        
        return max(0.0, min(1.0, base_timbre))
    
    def _calculate_pressure_sensitivity(self, decision, event_data: Dict) -> float:
        """Calculate pressure sensitivity for MPE"""
        base_pressure = 0.5
        
        # Adjust based on RMS level
        rms_db = event_data.get('rms_db', -60.0)
        pressure_factor = max(0.0, min(1.0, (rms_db + 80.0) / 80.0))
        base_pressure += pressure_factor * 0.2
        
        return max(0.0, min(1.0, base_pressure))
    
    def _log_conversation_response(self, midi_params, voice_type: str, mode: str, 
                                 current_time: float, expression_level: float = 0.5,
                                 pressure_sensitivity: float = 0.5, timbre_variation: float = 0.5):
        """Log AI response to conversation log"""
        try:
            import csv
            import os
            
            # Ensure logs directory exists
            os.makedirs(os.path.dirname(self.conversation_log), exist_ok=True)
            
            # Check if file exists to write header
            file_exists = os.path.exists(self.conversation_log)
            
            with open(self.conversation_log, 'a', newline='') as f:
                writer = csv.writer(f)
                
                if not file_exists:
                    writer.writerow([
                        'timestamp', 'voice_type', 'mode', 'note', 'velocity', 'duration',
                        'expression_level', 'pressure_sensitivity', 'timbre_variation'
                    ])
                
                writer.writerow([
                    current_time, voice_type, mode, midi_params.note, midi_params.velocity,
                    midi_params.duration, expression_level, pressure_sensitivity, timbre_variation
                ])
                
        except Exception as e:
            print(f"⚠️ Conversation logging error: {e}")
    
    def save_models(self):
        """Save AI models"""
        try:
            os.makedirs(self.model_dir, exist_ok=True)
            
            # Save audio oracle
            oracle_path = os.path.join(self.model_dir, "audio_oracle_model.json")
            self.audio_oracle.save_to_file(oracle_path)
            print(f"✅ Saved audio oracle model: {oracle_path}")
            
            # Save rhythm oracle
            if self.enable_rhythmic:
                rhythm_path = os.path.join(self.model_dir, "rhythm_oracle_model.json")
                self.rhythm_oracle.save_patterns(rhythm_path)
                print(f"✅ Saved rhythm oracle model: {rhythm_path}")
            
        except Exception as e:
            print(f"⚠️ Error saving models: {e}")
    
    def print_stats(self):
        """Print performance statistics"""
        runtime = time.time() - self.stats['start_time']
        print(f"\n📊 MusicHal Bass Performance Stats:")
        print(f"   Runtime: {runtime:.1f}s")
        print(f"   Events Processed: {self.stats['events_processed']}")
        print(f"   Bass Notes Sent: {self.stats['notes_sent']}")
        print(f"   Events/sec: {self.stats['events_processed']/runtime:.1f}")
        print(f"   Notes/sec: {self.stats['notes_sent']/runtime:.1f}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="MusicHal Bass - Bass-only AI musical agent")
    parser.add_argument("--midi-port", type=str, help="MIDI output port name")
    parser.add_argument("--input-device", type=int, help="Audio input device index")
    parser.add_argument("--no-mpe", action="store_true", help="Disable MPE MIDI output")
    parser.add_argument("--no-rhythmic", action="store_true", help="Disable rhythmic analysis")
    parser.add_argument("--no-correlation", action="store_true", help="Disable cross-modal correlation")
    
    args = parser.parse_args()
    
    # Create and start MusicHal Bass
    musichal_bass = MusicHalBass(
        midi_port=args.midi_port,
        input_device=args.input_device,
        enable_mpe=not args.no_mpe,
        enable_rhythmic=not args.no_rhythmic,
        enable_correlation=not args.no_correlation
    )
    
    try:
        if musichal_bass.start():
            # Keep running until interrupted
            while musichal_bass.running:
                time.sleep(1)
        else:
            print("❌ Failed to start MusicHal Bass")
            
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        musichal_bass.save_models()
        musichal_bass.print_stats()
        musichal_bass.stop()

if __name__ == "__main__":
    main()
