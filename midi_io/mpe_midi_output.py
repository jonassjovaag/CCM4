# mpe_midi_output.py
# MPE (Multi-Polyphonic Expression) MIDI output system for MusicHal 9000
# Enhanced expressiveness with per-note channel assignment and pitch bend

import mido
import time
import threading
import random
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass
from queue import Queue, Empty

@dataclass
class MPENote:
    """Represents an MPE MIDI note event with enhanced expressiveness"""
    note: int
    velocity: int
    channel: int  # MPE channel (2-15, channel 1 is master)
    timestamp: float
    duration: float
    
    # MPE-specific parameters
    pitch_bend: float = 0.0  # -1.0 to 1.0 (centered at 0.0)
    pressure: int = 64       # Aftertouch (0-127)
    timbre: int = 64         # CC 74 (timbre)
    brightness: int = 64     # CC 2 (brightness)
    
    # Additional expression
    attack_time: float = 0.1
    release_time: float = 0.1
    filter_cutoff: float = 0.5
    modulation_depth: float = 0.0
    pan: float = 0.0
    reverb_amount: float = 0.0

class MPEMIDIOutput:
    """
    MPE MIDI output system for MusicHal 9000
    Provides enhanced expressiveness with per-note channel assignment
    """
    
    def __init__(self, output_port_name: Optional[str] = None, enable_mpe: bool = True):
        self.output_port_name = output_port_name
        self.enable_mpe = enable_mpe
        self.port: Optional[mido.ports.BaseOutput] = None
        
        # MPE Configuration
        self.mpe_master_channel = 1  # Channel 1 is master (0-based = 0)
        self.mpe_note_channels = list(range(2, 16))  # Channels 2-15 for notes (1-based = 1-14)
        self.next_channel_index = 0
        
        # Note management
        self.active_notes: Dict[int, MPENote] = {}  # note -> MPENote
        self.channel_assignments: Dict[int, int] = {}  # note -> channel
        self.note_queue: Queue[MPENote] = Queue()
        self.note_history: List[MPENote] = []
        
        # Threading
        self.running = False
        self.worker_thread: Optional[threading.Thread] = None
        
        # Timing
        self.timing_precision = 0.01  # 10ms precision
        
        # Expression parameters
        self.global_pitch_bend_range = 2.0  # Semitones
        self.microtonal_sensitivity = 0.1   # Sensitivity for microtonal adjustments
        
    def start(self) -> bool:
        """Start MPE MIDI output system"""
        try:
            # Open MIDI port
            if self.output_port_name:
                self.port = mido.open_output(self.output_port_name)
            else:
                # Use default output
                outputs = mido.get_output_names()
                if outputs:
                    self.port = mido.open_output(outputs[0])
                else:
                    print("No MIDI output ports available")
                    return False
            
            print(f"🎹 MPE MIDI output connected to: {self.port.name}")
            
            # Initialize MPE mode
            if self.enable_mpe:
                self._initialize_mpe_mode()
            
            # Start worker thread
            self.running = True
            self.worker_thread = threading.Thread(target=self._worker, daemon=True)
            self.worker_thread.start()
            
            return True
            
        except Exception as e:
            print(f"Failed to start MPE MIDI output: {e}")
            return False
    
    def _initialize_mpe_mode(self):
        """Initialize MPE mode on the connected device"""
        try:
            # Set pitch bend range for all MPE channels
            for channel in self.mpe_note_channels:
                # Pitch bend range (RPN 0,0 = 2 semitones)
                self.port.send(mido.Message('control_change', channel=channel, control=101, value=0))  # RPN MSB
                self.port.send(mido.Message('control_change', channel=channel, control=100, value=0))  # RPN LSB
                self.port.send(mido.Message('control_change', channel=channel, control=6, value=2))    # Data MSB
                self.port.send(mido.Message('control_change', channel=channel, control=38, value=0))   # Data LSB
            
            print("🎛️ MPE mode initialized (pitch bend range: 2 semitones)")
            
        except Exception as e:
            print(f"Warning: Could not initialize MPE mode: {e}")
    
    def stop(self):
        """Stop MPE MIDI output system"""
        self.running = False
        
        # Stop all active notes
        self._stop_all_notes()
        
        # Send MIDI panic messages (All Notes Off + All Sound Off on all channels)
        if self.port:
            for channel in range(16):
                # CC 123: All Notes Off
                self.port.send(mido.Message('control_change', 
                                           channel=channel, 
                                           control=123, 
                                           value=0))
                # CC 120: All Sound Off (immediate silence)
                self.port.send(mido.Message('control_change', 
                                           channel=channel, 
                                           control=120, 
                                           value=0))
        
        # Wait for worker thread
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)
        
        # Close MIDI port
        if self.port:
            self.port.close()
            self.port = None
    
    def send_note(self, midi_params, voice_type: str = 'melodic', 
                  pitch_deviation: float = 0.0, expression_level: float = 0.5,
                  timbre_variation: float = 0.5, pressure_sensitivity: float = 0.5) -> bool:
        """Send an MPE MIDI note with enhanced expressiveness"""
        if not self.port or not self.running:
            return False
        
        try:
            # Assign MPE channel
            if self.enable_mpe:
                channel = self._assign_mpe_channel(midi_params.note)
            else:
                # Fallback to standard channels
                channel = 1 if voice_type == 'melodic' else 2
            
            # Calculate MPE parameters with enhanced expressiveness
            pitch_bend = self._calculate_pitch_bend(pitch_deviation)
            pressure = self._calculate_pressure(pressure_sensitivity)
            timbre = self._calculate_timbre(midi_params, timbre_variation)
            brightness = self._calculate_brightness(midi_params, expression_level)
            
            # Create MPE note
            note = MPENote(
                note=midi_params.note,
                velocity=midi_params.velocity,
                channel=channel,
                timestamp=time.time(),
                duration=midi_params.duration,
                pitch_bend=pitch_bend,
                pressure=pressure,
                timbre=timbre,
                brightness=brightness,
                attack_time=midi_params.attack_time,
                release_time=midi_params.release_time,
                filter_cutoff=midi_params.filter_cutoff,
                modulation_depth=midi_params.modulation_depth,
                pan=midi_params.pan,
                reverb_amount=midi_params.reverb_amount
            )
            
            # Add to queue
            self.note_queue.put(note)
            
            return True
            
        except Exception as e:
            print(f"Failed to send MPE MIDI note: {e}")
            return False
    
    def _assign_mpe_channel(self, note: int) -> int:
        """Assign an MPE channel for a note"""
        # Check if note already has a channel
        if note in self.channel_assignments:
            return self.channel_assignments[note]
        
        # Assign next available channel
        channel = self.mpe_note_channels[self.next_channel_index]
        self.channel_assignments[note] = channel
        
        # Move to next channel (round-robin)
        self.next_channel_index = (self.next_channel_index + 1) % len(self.mpe_note_channels)
        
        return channel
    
    def _calculate_pitch_bend(self, pitch_deviation: float) -> float:
        """Calculate pitch bend value from pitch deviation in cents"""
        # Convert cents to pitch bend value (-1.0 to 1.0)
        # 1 semitone = 100 cents, pitch bend range = 2 semitones = 200 cents
        pitch_bend = pitch_deviation / 200.0
        return max(-1.0, min(1.0, pitch_bend))
    
    def _calculate_pressure(self, pressure_sensitivity: float) -> int:
        """Calculate aftertouch pressure from pressure sensitivity"""
        # Map 0.0-1.0 to 0-127 with enhanced sensitivity
        base_pressure = int(pressure_sensitivity * 127)
        
        # Add some dynamic range for more expressiveness
        dynamic_boost = int((pressure_sensitivity - 0.5) * 30)  # ±15 points
        
        final_pressure = base_pressure + dynamic_boost
        return max(0, min(127, final_pressure))
    
    def _calculate_timbre(self, midi_params, timbre_variation: float) -> int:
        """Calculate timbre (CC 74) from note parameters and variation"""
        # Base timbre from filter cutoff
        base_timbre = int(midi_params.filter_cutoff * 127)
        
        # Apply timbre variation (0.0-1.0)
        variation_boost = int(timbre_variation * 40)  # Up to 40 points of variation
        
        # Add some randomness for more organic feel
        random_variation = int((timbre_variation - 0.5) * 20)  # ±10 points
        
        final_timbre = base_timbre + variation_boost + random_variation
        return max(0, min(127, final_timbre))
    
    def _calculate_brightness(self, midi_params, expression_level: float) -> int:
        """Calculate brightness (CC 2) from note parameters"""
        # Base brightness from modulation depth, enhanced by expression
        base_brightness = int(midi_params.modulation_depth * 127)
        expression_boost = int(expression_level * 15)  # Add up to 15 points
        return min(127, base_brightness + expression_boost)
    
    def _worker(self):
        """Worker thread for MPE MIDI output"""
        while self.running:
            try:
                # Process note queue
                while True:
                    note = self.note_queue.get(timeout=0.1)
                    self._process_mpe_note(note)
                    
            except Empty:
                pass
            except Exception as e:
                print(f"MPE MIDI worker error: {e}")
            
            # Update active notes
            self._update_active_notes()
            
            time.sleep(self.timing_precision)
    
    def _process_mpe_note(self, note: MPENote):
        """Process an MPE MIDI note with enhanced expressiveness"""
        current_time = time.time()
        
        # Send pitch bend first (if MPE enabled)
        if self.enable_mpe and abs(note.pitch_bend) > 0.001:
            # Convert pitch bend (-1.0 to 1.0) to MIDI pitch wheel value (-8192 to 8191)
            pitch_bend_value = int(note.pitch_bend * 8191)
            # Clamp to valid range
            pitch_bend_value = max(-8192, min(8191, pitch_bend_value))
            self.port.send(mido.Message('pitchwheel',
                                      channel=note.channel,
                                      pitch=pitch_bend_value))
        
        # Send note on (ensure note and velocity are in valid MIDI range)
        midi_note = max(0, min(127, int(note.note)))
        midi_velocity = max(1, min(127, int(note.velocity)))  # velocity 0 = note off
        
        note_on = mido.Message('note_on', 
                             channel=note.channel,
                             note=midi_note,
                             velocity=midi_velocity)
        self.port.send(note_on)
        print(f"🎵 MIDI sent: note_on channel={note.channel} note={midi_note} velocity={midi_velocity}")
        
        # Send MPE control changes with small delay to prevent MIDI congestion
        time.sleep(0.001)  # 1ms delay to prevent MIDI buffer overflow
        self._send_mpe_control_changes(note)
        
        # Store active note
        self.active_notes[note.note] = note
        
        # Schedule note off
        note_off_time = current_time + note.duration
        note.note_off_time = note_off_time
        
        # Add to history
        self.note_history.append(note)
        if len(self.note_history) > 1000:
            self.note_history.pop(0)
        
        # Silent note processing (no terminal output)
    
    def _send_mpe_control_changes(self, note: MPENote):
        """Send MPE control changes for enhanced expressiveness - simplified to reduce click noise"""
        # Only send essential control changes to reduce MIDI congestion
        # Skip most control changes that might cause audio artifacts
        
        # Only send pressure if it's significantly different from default
        pressure = max(0, min(127, int(note.pressure)))
        if abs(pressure - 64) > 10:  # Only send if significantly different
            self.port.send(mido.Message('polytouch',
                                      channel=note.channel,
                                      note=note.note,
                                      value=pressure))
            time.sleep(0.001)  # Small delay after pressure
        
        # Skip other control changes to reduce MIDI congestion and click noise
        # This should significantly reduce the number of MIDI messages per note
    
    def _update_active_notes(self):
        """Update active notes and send note offs"""
        current_time = time.time()
        notes_to_remove = []
        
        for note_num, note in self.active_notes.items():
            if hasattr(note, 'note_off_time') and current_time >= note.note_off_time:
                # Send note off
                note_off = mido.Message('note_off',
                                     channel=note.channel,
                                     note=note.note,
                                     velocity=0)
                self.port.send(note_off)
                
                # Reset pitch bend for this channel
                if self.enable_mpe:
                    self.port.send(mido.Message('pitchwheel',
                                              channel=note.channel,
                                              pitch=0))  # Center pitch bend (0 = no bend)
                
                notes_to_remove.append(note_num)
                
                # Release channel assignment
                if note_num in self.channel_assignments:
                    del self.channel_assignments[note_num]
        
        # Remove finished notes
        for note_num in notes_to_remove:
            del self.active_notes[note_num]
    
    def _stop_all_notes(self):
        """Stop all active notes"""
        for note_num, note in self.active_notes.items():
            note_off = mido.Message('note_off',
                                  channel=note.channel,
                                  note=note.note,
                                  velocity=0)
            if self.port:
                self.port.send(note_off)
                
                # Reset pitch bend
                if self.enable_mpe:
                    self.port.send(mido.Message('pitchwheel',
                                              channel=note.channel,
                                              pitch=0))
        
        self.active_notes.clear()
        self.channel_assignments.clear()
    
    def get_active_notes(self) -> Dict[int, MPENote]:
        """Get currently active notes"""
        return self.active_notes.copy()
    
    def get_note_history(self, duration_seconds: float = 30.0) -> List[MPENote]:
        """Get note history for the last duration_seconds"""
        current_time = time.time()
        cutoff_time = current_time - duration_seconds
        
        return [note for note in self.note_history 
                if note.timestamp >= cutoff_time]
    
    def get_status(self) -> Dict:
        """Get MPE MIDI output status"""
        return {
            'running': self.running,
            'port_name': self.port.name if self.port else None,
            'mpe_enabled': self.enable_mpe,
            'active_notes_count': len(self.active_notes),
            'queue_size': self.note_queue.qsize(),
            'assigned_channels': len(self.channel_assignments),
            'next_channel_index': self.next_channel_index
        }
    
    def set_mpe_enabled(self, enabled: bool):
        """Enable or disable MPE mode"""
        self.enable_mpe = enabled
        if enabled and self.port:
            self._initialize_mpe_mode()
    
    def get_available_channels(self) -> List[int]:
        """Get list of available MPE channels"""
        if self.enable_mpe:
            return [ch for ch in self.mpe_note_channels if ch not in self.channel_assignments.values()]
        else:
            return [1, 2]  # Standard channels
