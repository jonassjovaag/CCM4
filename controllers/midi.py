import mido
import numpy as np
from typing import Set, Dict
import time

class MIDIHandler:
    def __init__(self, state_manager=None, performance_ctrl=None, thread_ctrl=None):
        self.midi_out = None
        self.previous_notes = {i: set() for i in range(4)}
        self.velocity_curve = np.linspace(0.5, 1.0, 128)
        self.state_manager = state_manager
        self.performance_ctrl = performance_ctrl
        self.thread_ctrl = thread_ctrl

    def run(self):
        last_solo_time = 0
        current_solo_index = 0
        
        while self.thread_ctrl.running.is_set():
            current_time = time.time()
            intensity = self.performance_ctrl.calculate_intensity()
            
            if intensity == 0.0:
                for channel in range(4):
                    self.process_notes(channel, set(), intensity)
                break

            # Get current notes for all voices
            current_notes = self.collect_current_notes()
            
            # Handle solo voice sequencing
            if self.state_manager.current_state.get('solo_active'):
                solo_phrase = self.state_manager.current_state.get('current_solo_phrase')
                if solo_phrase and current_solo_index < len(solo_phrase):
                    if current_time - last_solo_time >= solo_phrase[current_solo_index][1]:
                        freq, duration, velocity = solo_phrase[current_solo_index]
                        midi_note = int(69 + 12 * np.log2(freq / 440.0))
                        midi_note = max(45, min(84, midi_note))
                        current_notes[3] = {midi_note}
                        current_solo_index += 1
                        last_solo_time = current_time
                        
                        if current_solo_index >= len(solo_phrase):
                            self.state_manager.current_state['solo_active'] = False
                            current_solo_index = 0

            # Process all voices
            for channel, notes in current_notes.items():
                if channel in self.state_manager.current_state['active_voices']:
                    self.process_notes(
                        channel, 
                        notes, 
                        intensity,
                        is_drone=(channel == 2)
                    )

            time.sleep(0.05)

    def process_drone(self, freq):
        midi_note = int(69 + 12 * np.log2(freq / 440.0))
        midi_note = max(28, min(40, midi_note))  # Keep in bass range
        self.process_notes(2, {midi_note}, 1.0, is_drone=True)

    def process_notes(self, channel: int, current_notes: Set[int], intensity: float, is_drone: bool = False):
        prev_notes = self.previous_notes[channel]
        notes_on = current_notes - prev_notes
        notes_off = prev_notes - current_notes

        # Print note activity for each channel
        channel_names = {0: "Voice 1", 1: "Voice 2", 2: "Drone", 3: "Solo"}
        if notes_on:
            print(f"{channel_names[channel]} ON: {notes_on}, velocity: {int(intensity * 90)}")
        if notes_off:
            print(f"{channel_names[channel]} OFF: {notes_off}")

        # Process new notes
        for note in notes_on:
            velocity = self.calculate_velocity(note, intensity, is_drone)
            self.midi_out.send(mido.Message('note_on', note=note, velocity=velocity, channel=channel))

        # Process ended notes
        for note in notes_off:
            self.midi_out.send(mido.Message('note_off', note=note, velocity=0, channel=channel))

        self.previous_notes[channel] = current_notes

    def calculate_velocity(self, note: int, intensity: float, is_drone: bool = False) -> int:
        base_velocity = int(self.velocity_curve[note] * 100)
        if is_drone:
            return min(90, base_velocity)
        return min(90, int(base_velocity * intensity))

    def collect_current_notes(self):
        current_notes = {i: set() for i in range(4)}
        state = self.state_manager.current_state

        # Add debug print to see what frequencies we're getting
        # print(f"Voice1 frequencies: {state.get('voice1_frequencies', [])}")
        # print(f"Voice2 frequencies: {state.get('voice2_frequencies', [])}")
        # print(f"Voice3 frequencies: {state.get('voice3_frequencies', [])}")
        
        # Voice 1 (main overtones)
        if 0 in state['active_voices']:
            for freq in state.get('voice1_frequencies', []):
                midi_note = int(69 + 12 * np.log2(freq / 440.0))
                midi_note = max(21, min(108, midi_note))
                current_notes[0].add(midi_note)
        
        # Voice 2 (short tones)
        if 1 in state['active_voices']:
            for freq in state.get('voice2_frequencies', []):
                midi_note = int(69 + 12 * np.log2(freq / 440.0))
                midi_note = max(33, min(96, midi_note))
                current_notes[1].add(midi_note)
        
        # Voice 3 (drone)
        if 2 in state['active_voices']:
            for freq in state.get('voice3_frequencies', []):
                midi_note = int(69 + 12 * np.log2(freq / 440.0))
                midi_note = max(28, min(40, midi_note))
                current_notes[2].add(midi_note)
        
        print(f"Current active notes per channel: {current_notes}")
        return current_notes

    def setup_midi(self):
        ports = mido.get_output_names()
        print("\nAvailable MIDI Output Ports:")
        for i, port in enumerate(ports):
            print(f"{i}: {port}")

        while True:
            try:
                choice = int(input("\nSelect MIDI output port number: "))
                if 0 <= choice < len(ports):
                    self.midi_out = mido.open_output(ports[choice])
                    print(f"Selected: {ports[choice]}")
                    break
            except ValueError:
                print("Please enter a valid number.")

    def cleanup(self):
        if self.midi_out:
            self.cleanup_notes()
            self.midi_out.close()

    def cleanup_notes(self):
        if self.midi_out:
            for channel in range(4):
                for note in self.previous_notes[channel]:
                    self.midi_out.send(mido.Message('note_off',
                                                  note=note,
                                                  velocity=0,
                                                  channel=channel))
                self.previous_notes[channel].clear()
                
            for channel in range(4):
                self.midi_out.send(mido.Message('control_change',
                                              channel=channel,
                                              control=123,
                                              value=0))

    def configure_digitone(self):
        """Configure MIDI routing for Elektron Digitone"""
        if self.midi_out:
            self.midi_out.send(mido.Message('control_change', channel=0, control=122, value=0))
            time.sleep(0.1)
            self.midi_out.send(mido.Message('start'))
