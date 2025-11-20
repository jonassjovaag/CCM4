# meld_controller.py
# Manages MIDI CC communication with Ableton Meld synthesizer
# Integrates with HarmonicContextManager for scale-aware filter control

import time
import logging
from typing import Dict, Optional
from dataclasses import dataclass

try:
    import mido  # type: ignore
    from mido import Message  # type: ignore
except ImportError:
    mido = None
    Message = None

from mapping.meld_mapper import MeldMapper


@dataclass
class ThrottleState:
    """Track CC throttling state"""
    last_sent_time: float = 0.0
    last_sent_value: int = 0
    send_count: int = 0


class MeldController:
    """
    Controls Ableton Meld synthesizer via MIDI CC
    
    Features:
    - Dual-engine parameter mapping (Engine A melodic, Engine B bass)
    - Probabilistic macro routing for creative accidents
    - Scale-aware filter control via HarmonicContextManager
    - Intelligent CC throttling (time + change threshold)
    - Voice routing (melodic→channel 1, bass→channel 2)
    """
    
    def __init__(self, 
                 midi_port_name: str = "IAC Meld",
                 config_path: Optional[str] = None,
                 harmonic_context_manager=None,
                 logger=None):  # type: ignore
        """
        Initialize Meld controller
        
        Args:
            midi_port_name: MIDI output port name
            config_path: Path to meld_mapping.yaml (optional)
            harmonic_context_manager: Reference for scale-aware control (optional)
            logger: PerformanceLogger instance for MIDI logging (optional)
        """
        self.mapper = MeldMapper(config_path)
        self.harmonic_context_manager = harmonic_context_manager
        self.logger = logger
        
        # MIDI setup
        self.midi_port_name = midi_port_name
        self.midi_out = None
        
        if mido:
            try:
                self.midi_out = mido.open_output(midi_port_name)  # type: ignore
                logging.info(f"🎹 Meld Controller: Connected to {midi_port_name}")
            except Exception as e:
                logging.warning(f"🎹 Meld Controller: Could not open MIDI port {midi_port_name}: {e}")
        else:
            logging.warning("🎹 Meld Controller: mido not available, MIDI disabled")
        
        # Throttling state
        self.throttle_state: Dict[str, ThrottleState] = {}
        self.update_interval_ms = self.mapper.config['throttling']['update_interval_ms']
        self.update_interval_s = self.update_interval_ms / 1000.0
        
        # Scale-aware state
        self.last_chord: Optional[str] = None
        self.last_scale_update_time: float = 0.0
        self.scale_update_interval_s = self.mapper.config['throttling']['scale_update_interval_s']
        self.scale_enabled: bool = False
        
        # Voice routing config
        self.routing_config = self.mapper.config['routing']
        
        # Performance stats
        self.stats = {
            'total_cc_sent': 0,
            'throttled_cc': 0,
            'alternative_mappings': 0,
            'scale_updates': 0
        }
        
        logging.info("🎹 Meld Controller initialized")
        logging.info(f"   Update interval: {self.update_interval_ms}ms")
        logging.info(f"   Scale update interval: {self.scale_update_interval_s}s")
    
    def update_meld_parameters(self, 
                               event_data: Dict,
                               voice_type: str = "melodic",
                               timbre_variance: float = 0.5) -> None:
        """
        Update Meld synthesizer parameters from audio features
        
        Args:
            event_data: Event dictionary with extracted features
            voice_type: 'melodic' or 'bass' (determines routing channel)
            timbre_variance: 0=stable (high smoothing), 1=expressive (low smoothing)
        """
        if not self.midi_out:
            return  # MIDI not available
        
        # Map features to parameters with timbre variance control
        params = self.mapper.map_features_to_meld(event_data, voice_type, timbre_variance)
        
        # Track alternative mapping stats
        if params.used_alternative_mapping:
            self.stats['alternative_mappings'] += 1
        
        # Determine MIDI channel based on voice routing
        channel = self._get_channel_for_voice(voice_type)
        
        # Send CC messages with throttling
        self._send_cc_if_changed('engine_a_macro_1', params.engine_a_macro_1, channel)
        self._send_cc_if_changed('engine_a_macro_2', params.engine_a_macro_2, channel)
        self._send_cc_if_changed('engine_b_macro_1', params.engine_b_macro_1, channel)
        self._send_cc_if_changed('engine_b_macro_2', params.engine_b_macro_2, channel)
        self._send_cc_if_changed('ab_blend', params.ab_blend, channel)
        self._send_cc_if_changed('spread', params.spread, channel)
        self._send_cc_if_changed('filter_frequency', params.filter_frequency, channel)
        self._send_cc_if_changed('filter_resonance', params.filter_resonance, channel)
    
    def update_scale_aware_parameters(self) -> None:
        """
        Update scale-aware filter parameters from HarmonicContextManager
        
        Checks current chord and updates scale parameters if changed.
        Throttled to avoid excessive MIDI traffic (default 2s interval).
        """
        if not self.midi_out or not self.harmonic_context_manager:
            return
        
        current_time = time.time()
        
        # Throttle scale updates (slower than regular CCs)
        if current_time - self.last_scale_update_time < self.scale_update_interval_s:
            return
        
        # Get current chord from context manager
        chord_name = self.harmonic_context_manager.get_active_chord()
        
        # Check if chord changed
        if chord_name == self.last_chord:
            return  # No change
        
        self.last_chord = chord_name
        self.last_scale_update_time = current_time
        
        # Parse chord to scale parameters
        root_note, scale_type = self.mapper.parse_chord_for_scale(chord_name)
        scale_type_cc = self.mapper.get_scale_type_cc_value(scale_type)
        
        # Send scale parameters (all channels to affect global filters)
        for channel in [1, 2]:  # Both Engine A and Engine B
            # Enable scale-aware filtering
            if not self.scale_enabled:
                self._send_cc_immediate('scale_enable', 127, channel)  # On
                self.scale_enabled = True
            
            # Update root note (0-11)
            root_cc_value = int((root_note / 11.0) * 127)
            self._send_cc_immediate('root_note', root_cc_value, channel)
            
            # Update scale type (discrete mappings)
            self._send_cc_immediate('scale_type', scale_type_cc, channel)
        
        self.stats['scale_updates'] += 1
        
        logging.debug(f"🎹 Scale update: {chord_name} → Root={root_note}, Scale={scale_type}")
    
    def enable_scale_aware_filters(self, enabled: bool = True) -> None:
        """
        Enable/disable scale-aware filters
        
        Args:
            enabled: True to enable, False to disable
        """
        if not self.midi_out:
            return
        
        cc_value = 127 if enabled else 0
        
        for channel in [1, 2]:
            self._send_cc_immediate('scale_enable', cc_value, channel)
        
        self.scale_enabled = enabled
        logging.info(f"🎹 Scale-aware filters {'enabled' if enabled else 'disabled'}")
    
    def _get_channel_for_voice(self, voice_type: str) -> int:
        """
        Get MIDI channel for voice type
        
        Args:
            voice_type: 'melodic' or 'bass'
            
        Returns:
            MIDI channel number (1-16)
        """
        if voice_type == 'bass':
            return self.routing_config.get('bass_voice', {}).get('midi_channel', 2)
        else:  # melodic
            return self.routing_config.get('melodic_voice', {}).get('midi_channel', 1)
    
    def _send_cc_if_changed(self, 
                           param_name: str, 
                           value: float,
                           channel: int) -> None:
        """
        Send CC message only if value changed significantly or time elapsed
        
        Args:
            param_name: Parameter name (e.g., 'engine_a_macro_1')
            value: Normalized value (0.0-1.0)
            channel: MIDI channel (1-16)
        """
        current_time = time.time()
        cc_number = self.mapper.get_cc_mapping(param_name)
        cc_value = int(value * 127)
        
        # Create unique key for this parameter + channel
        throttle_key = f"{param_name}_ch{channel}"
        
        # Initialize throttle state if needed
        if throttle_key not in self.throttle_state:
            self.throttle_state[throttle_key] = ThrottleState()
        
        state = self.throttle_state[throttle_key]
        
        # Check time-based throttling
        time_elapsed = current_time - state.last_sent_time
        if time_elapsed < self.update_interval_s:
            self.stats['throttled_cc'] += 1
            return  # Too soon
        
        # Check value-based throttling (has it changed enough?)
        value_change = abs(cc_value - state.last_sent_value)
        change_threshold_cc = int(self.mapper.change_threshold * 127)
        
        if value_change < change_threshold_cc and state.send_count > 0:
            self.stats['throttled_cc'] += 1
            return  # Not changed enough
        
        # Send CC
        try:
            msg = Message('control_change', 
                         control=cc_number,
                         value=cc_value,
                         channel=channel - 1)  # mido uses 0-15
            self.midi_out.send(msg)
            
            # Log CC message
            if self.logger:
                self.logger.log_midi_message(
                    message_type='control_change',
                    port=self.midi_port_name,
                    channel=channel,
                    cc_number=cc_number,
                    cc_value=cc_value,
                    additional_data=f"param={param_name}"
                )
            
            # Update state
            state.last_sent_time = current_time
            state.last_sent_value = cc_value
            state.send_count += 1
            self.stats['total_cc_sent'] += 1
            
        except Exception as e:
            logging.error(f"🎹 Error sending CC {cc_number}={cc_value} on ch{channel}: {e}")
    
    def _send_cc_immediate(self,
                          param_name: str,
                          cc_value: int,
                          channel: int) -> None:
        """
        Send CC message immediately without throttling
        
        Used for scale-aware parameters that need guaranteed delivery.
        
        Args:
            param_name: Parameter name
            cc_value: MIDI CC value (0-127)
            channel: MIDI channel (1-16)
        """
        cc_number = self.mapper.get_cc_mapping(param_name)
        
        try:
            msg = Message('control_change',
                         control=cc_number,
                         value=cc_value,
                         channel=channel - 1)  # mido uses 0-15
            self.midi_out.send(msg)
            
            # Log immediate CC message
            if self.logger:
                self.logger.log_midi_message(
                    message_type='control_change',
                    port=self.midi_port_name,
                    channel=channel,
                    cc_number=cc_number,
                    cc_value=cc_value,
                    additional_data=f"param={param_name};immediate=true"
                )
            
            self.stats['total_cc_sent'] += 1
            
        except Exception as e:
            logging.error(f"🎹 Error sending immediate CC {cc_number}={cc_value} on ch{channel}: {e}")
    
    def get_stats(self) -> Dict:
        """Get performance statistics"""
        return self.stats.copy()
    
    def reset_stats(self) -> None:
        """Reset performance statistics"""
        self.stats = {
            'total_cc_sent': 0,
            'throttled_cc': 0,
            'alternative_mappings': 0,
            'scale_updates': 0
        }
    
    def close(self) -> None:
        """Close MIDI port"""
        if self.midi_out:
            self.midi_out.close()
            logging.info("🎹 Meld Controller closed")


if __name__ == "__main__":
    # Demo/test
    logging.basicConfig(level=logging.DEBUG)
    
    print("🎹 Meld Controller Demo")
    print("=" * 50)
    
    # Mock HarmonicContextManager for testing
    class MockHarmonicContext:
        def __init__(self):
            self.chord = "Cmaj7"
        
        def get_active_chord(self):
            return self.chord
    
    mock_context = MockHarmonicContext()
    
    try:
        controller = MeldController(
            midi_port_name="IAC Meld",
            harmonic_context_manager=mock_context
        )
        
        # Test event
        test_event = {
            'spectral_centroid': 0.7,
            'consonance': 0.8,
            'zcr': 0.3,
            'spectral_rolloff': 4000,
            'bandwidth': 1800,
            'flatness': 0.2,
            'mfcc_1': 15.0,
            'modulation_depth': 0.5
        }
        
        print("\nSending test event (melodic voice)...")
        controller.update_meld_parameters(test_event, voice_type='melodic')
        
        print("\nUpdating scale-aware parameters...")
        controller.update_scale_aware_parameters()
        
        # Change chord
        print("\nChanging chord to Dm...")
        mock_context.chord = "Dm"
        time.sleep(2.1)  # Wait for throttle interval
        controller.update_scale_aware_parameters()
        
        print("\nStats:")
        stats = controller.get_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        controller.close()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
