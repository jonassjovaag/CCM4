#!/usr/bin/env python3
"""
Simple Chord Validator - Using Existing YIN Pitch Tracking
==========================================================

Simple approach:
1. Send MIDI chord (C-E-G) → Ableton
2. Use DriftListener (YIN) to track pitches over 2 seconds
3. Collect all detected MIDI notes
4. Compare to sent notes
5. Validate!

This uses the SAME pitch detection that already works in MusicHal_9000!
"""

import numpy as np
import time
import json
import os
import mido
from datetime import datetime
from typing import List, Optional, Set
from collections import Counter

from listener.jhs_listener_core import DriftListener, Event


class SimpleChordValidator:
    """Validate chords using existing YIN pitch tracking"""
    
    def __init__(self,
                 midi_output_port: str = "IAC Driver Chord Trainer Output",
                 audio_input_device: Optional[int] = None,
                 chord_duration: float = 2.5):
        
        self.midi_port_name = midi_output_port
        self.audio_device = audio_input_device
        self.chord_duration = chord_duration
        
        # MIDI
        self.midi_port = None
        
        # Audio
        self.listener = None
        
        # Pitch collection
        self.detected_pitches = []
        self.collecting = False
        
        # Results
        self.validated_chords = []
    
    def _open_midi_port(self) -> bool:
        """Open MIDI port"""
        try:
            available = mido.get_output_names()
            if self.midi_port_name in available:
                self.midi_port = mido.open_output(self.midi_port_name)
                print(f"✅ MIDI: {self.midi_port_name}")
                return True
            return False
        except Exception as e:
            print(f"❌ MIDI error: {e}")
            return False
    
    def _start_listener(self) -> bool:
        """Start YIN pitch listener"""
        try:
            self.listener = DriftListener(
                ref_fn=lambda midi_note: 440.0,
                a4_fn=lambda: 440.0,
                device=self.audio_device
            )
            self.listener.start(self._on_pitch_event)
            print(f"✅ YIN pitch tracker started (device {self.audio_device})")
            return True
        except Exception as e:
            print(f"❌ Listener error: {e}")
            return False
    
    def _on_pitch_event(self, *args):
        """Collect detected pitches during chord playback"""
        if not self.collecting:
            return
        
        # Handle callback signature
        if len(args) == 1:
            event = args[0]
        else:
            return
        
        if event is None or not isinstance(event, Event):
            return
        
        # Collect MIDI note if pitch is confident
        if event.rms_db > -50 and event.f0 > 0:
            self.detected_pitches.append(event.midi)
    
    def validate_chord(self, sent_midi: List[int], chord_name: str) -> bool:
        """
        Validate a chord by playing it and collecting pitches
        
        Returns True if validation passes
        """
        print(f"\n🎹 Validating: {chord_name}")
        print(f"   Sending: {sent_midi} ({[self._midi_to_note(n) for n in sent_midi]})")
        
        # Reset pitch collection
        self.detected_pitches = []
        self.collecting = True
        
        # Play chord
        for note in sent_midi:
            self.midi_port.send(mido.Message('note_on', channel=0, note=note, velocity=80))
        
        # Collect pitches for full duration
        time.sleep(self.chord_duration)
        
        # Stop chord
        for note in sent_midi:
            self.midi_port.send(mido.Message('note_off', channel=0, note=note, velocity=0))
        
        self.collecting = False
        
        # Analyze collected pitches
        if not self.detected_pitches:
            print(f"   ❌ No pitches detected!")
            return False
        
        # Count occurrences of each MIDI note
        pitch_counts = Counter(self.detected_pitches)
        
        # Get most common pitches (top N where N = number of sent notes)
        most_common = pitch_counts.most_common(len(sent_midi) + 2)  # Allow a couple extra
        detected_notes = [midi for midi, count in most_common if count >= 3]  # Need at least 3 detections
        
        print(f"   Detected {len(self.detected_pitches)} pitch events:")
        print(f"      Pitch counts: {dict(pitch_counts)}")
        print(f"   🎧 Final notes: {detected_notes} ({[self._midi_to_note(n) for n in detected_notes]})")
        
        # Check match
        sent_set = set(sent_midi)
        detected_set = set(detected_notes)
        
        # Allow ±1 semitone tolerance
        matches = 0
        for s in sent_set:
            if s in detected_set or (s-1) in detected_set or (s+1) in detected_set:
                matches += 1
        
        match_quality = matches / len(sent_set)
        
        print(f"   📊 Match: {matches}/{len(sent_set)} = {match_quality:.1%}")
        
        if match_quality >= 0.66:  # At least 2/3 notes correct
            print(f"   ✅ VALIDATED!")
            return True
        else:
            print(f"   ❌ Failed")
            return False
    
    def _midi_to_note(self, midi: int) -> str:
        """Convert MIDI to note name"""
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        return f"{notes[midi % 12]}{midi // 12 - 1}"
    
    def run_test(self):
        """Run simple test"""
        print("\n🎯 Simple Chord Validation Test")
        print("=" * 60)
        
        test_chords = [
            ([60, 64, 67], "C major"),
            ([48, 52, 55], "C major (low)"),
            ([60, 63, 67], "C minor"),
            ([62, 66, 69], "D major"),
            ([62, 65, 69], "D minor"),
        ]
        
        passed = 0
        for midi_notes, name in test_chords:
            if self.validate_chord(midi_notes, name):
                passed += 1
            time.sleep(0.5)
        
        print(f"\n{'=' * 60}")
        print(f"✅ Results: {passed}/{len(test_chords)} passed")
        print(f"   Success rate: {100 * passed / len(test_chords):.0f}%")
    
    def start(self) -> bool:
        """Start system"""
        if not self._open_midi_port():
            return False
        if not self._start_listener():
            return False
        print("✅ System ready!\n")
        return True
    
    def stop(self):
        """Stop system"""
        if self.listener:
            self.listener.stop()
        if self.midi_port:
            self.midi_port.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-device', type=int, default=None)
    args = parser.parse_args()
    
    validator = SimpleChordValidator(audio_input_device=args.input_device)
    
    if not validator.start():
        return
    
    try:
        validator.run_test()
    finally:
        validator.stop()


if __name__ == "__main__":
    main()

































