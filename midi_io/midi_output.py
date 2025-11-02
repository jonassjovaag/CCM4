# midi_output.py
# MIDI output system for Drift Engine AI

import mido
import time
import threading
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from queue import Queue, Empty

@dataclass
class MIDINote:
    """Represents a MIDI note event"""
    note: int
    velocity: int
    channel: int
    timestamp: float
    duration: float
    attack_time: float
    release_time: float
    filter_cutoff: float
    modulation_depth: float
    pan: float
    reverb_amount: float

class MIDIOutput:
    """
    MIDI output system for Drift Engine AI
    Handles note scheduling, timing, and parameter control
    """
    
    def __init__(self, output_port_name: Optional[str] = None):
        self.output_port_name = output_port_name
        self.port: Optional[mido.ports.BaseOutput] = None
        
        # Note management
        self.active_notes: Dict[int, MIDINote] = {}  # note -> MIDINote
        self.note_queue: Queue[MIDINote] = Queue()
        self.note_history: List[MIDINote] = []
        
        # Threading
        self.running = False
        self.worker_thread: Optional[threading.Thread] = None
        
        # MIDI channels (mido uses 0-based channels)
        self.channels = {
            'melodic': 0,      # Track 1 (Melody) - MIDI Channel 0
            'harmonic': 1,      # Track 2 (Harmony) - MIDI Channel 1  
            'bass': 1,         # Track 2 (Bass) - MIDI Channel 1 (working)
            'percussion': 9     # Track 10 (Percussion) - MIDI Channel 9
        }
        
        # Timing
        self.timing_precision = 0.01  # 10ms precision
        
    def start(self) -> bool:
        """Start MIDI output system"""
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
            
            print(f"MIDI output connected to: {self.port.name}")
            
            # Start worker thread
            self.running = True
            self.worker_thread = threading.Thread(target=self._worker, daemon=True)
            self.worker_thread.start()
            
            return True
            
        except Exception as e:
            print(f"Failed to start MIDI output: {e}")
            return False
    
    def stop(self):
        """Stop MIDI output system"""
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
    
    def send_note(self, midi_params, channel: str = 'melodic') -> bool:
        """Send a MIDI note with parameters"""
        if not self.port or not self.running:
            return False
        
        try:
            # Create MIDI note
            midi_channel = self.channels.get(channel, 1)  # Default to channel 1 instead of 0
            note = MIDINote(
                note=midi_params.note,
                velocity=midi_params.velocity,
                channel=midi_channel,
                timestamp=time.time(),
                duration=midi_params.duration,
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
            print(f"Failed to send MIDI note: {e}")
            return False
    
    def _worker(self):
        """Worker thread for MIDI output"""
        while self.running:
            try:
                # Process note queue
                while True:
                    note = self.note_queue.get(timeout=0.1)
                    self._process_note(note)
                    
            except Empty:
                pass
            except Exception as e:
                print(f"MIDI worker error: {e}")
            
            # Update active notes
            self._update_active_notes()
            
            time.sleep(self.timing_precision)
    
    def _process_note(self, note: MIDINote):
        """Process a MIDI note"""
        current_time = time.time()
        
        # Send note on
        note_on = mido.Message('note_on', 
                             channel=note.channel,
                             note=note.note,
                             velocity=note.velocity)
        self.port.send(note_on)
        
        # Send control changes
        self._send_control_changes(note)
        
        # Store active note
        self.active_notes[note.note] = note
        
        # Schedule note off
        note_off_time = current_time + note.duration
        note.note_off_time = note_off_time
        
        # Add to history
        self.note_history.append(note)
        if len(self.note_history) > 1000:
            self.note_history.pop(0)
        
        print(f"🎵 MIDI Note: {note.note} (vel={note.velocity}, ch={note.channel}, dur={note.duration:.2f}s)")
    
    def _send_control_changes(self, note: MIDINote):
        """Send MIDI control changes for note parameters"""
        # Filter cutoff (CC 74)
        filter_cc = int(note.filter_cutoff * 127)
        self.port.send(mido.Message('control_change',
                                  channel=note.channel,
                                  control=74,
                                  value=filter_cc))
        
        # Modulation depth (CC 1)
        mod_cc = int(note.modulation_depth * 127)
        self.port.send(mido.Message('control_change',
                                  channel=note.channel,
                                  control=1,
                                  value=mod_cc))
        
        # Pan (CC 10)
        pan_cc = int((note.pan + 1.0) * 63.5)  # -1.0 to 1.0 -> 0 to 127
        self.port.send(mido.Message('control_change',
                                  channel=note.channel,
                                  control=10,
                                  value=pan_cc))
        
        # Reverb amount (CC 91)
        reverb_cc = int(note.reverb_amount * 127)
        self.port.send(mido.Message('control_change',
                                  channel=note.channel,
                                  control=91,
                                  value=reverb_cc))
    
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
                
                notes_to_remove.append(note_num)
        
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
        
        self.active_notes.clear()
    
    def get_active_notes(self) -> Dict[int, MIDINote]:
        """Get currently active notes"""
        return self.active_notes.copy()
    
    def get_note_history(self, duration_seconds: float = 30.0) -> List[MIDINote]:
        """Get note history for the last duration_seconds"""
        current_time = time.time()
        cutoff_time = current_time - duration_seconds
        
        return [note for note in self.note_history 
                if note.timestamp >= cutoff_time]
    
    def set_channel(self, voice_type: str, channel: int):
        """Set MIDI channel for a voice type"""
        self.channels[voice_type] = channel
    
    def get_status(self) -> Dict:
        """Get MIDI output status"""
        return {
            'running': self.running,
            'port_name': self.port.name if self.port else None,
            'active_notes_count': len(self.active_notes),
            'queue_size': self.note_queue.qsize(),
            'channels': self.channels.copy()
        }
