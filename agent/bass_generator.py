#!/usr/bin/env python3
"""
Bass Generator
==============

Specialized bass line generation with bass-specific behaviors:
- ROOT: Play chord roots for solid foundation
- WALK: Walking bass (stepwise approach to next chord)
- FIFTH: Root-fifth alternation patterns
- PEDAL: Repeated root (pedal point)
- COUNTERPOINT: Independent melodic bass

This differentiates bass from melody by using bass-specific
musical vocabulary and generation patterns.
"""

import random
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class BassMode(Enum):
    """Bass-specific behavior modes"""
    ROOT = "root"           # Play chord roots
    WALK = "walk"           # Walking bass patterns
    FIFTH = "fifth"         # Root-fifth alternation
    PEDAL = "pedal"         # Pedal point (repeated note)
    COUNTERPOINT = "counterpoint"  # Independent melodic line


@dataclass
class BassPattern:
    """A bass pattern with notes and timing"""
    notes: List[int]        # MIDI notes
    timings: List[float]    # Inter-onset intervals
    velocities: List[int]   # Note velocities
    durations: List[float]  # Note durations
    mode: BassMode          # Generation mode used
    pattern_type: str       # Description


class BassGenerator:
    """
    Generates bass lines with bass-specific musical vocabulary.

    Unlike melody generation, bass focuses on:
    - Harmonic foundation (root motion)
    - Rhythmic stability
    - Specific bass idioms (walking, pedal, fifths)
    """

    def __init__(self,
                 bass_range: Tuple[int, int] = (28, 55),  # E1 to G3
                 default_mode: BassMode = BassMode.ROOT):
        """
        Initialize bass generator.

        Args:
            bass_range: (min_midi, max_midi) for bass notes
            default_mode: Default bass behavior mode
        """
        self.bass_range = bass_range
        self.default_mode = default_mode

        # Chord to root mapping (pitch class)
        self.chord_roots = {
            'C': 0, 'Cm': 0, 'C7': 0, 'Cmaj7': 0, 'Cm7': 0,
            'D': 2, 'Dm': 2, 'D7': 2, 'Dmaj7': 2, 'Dm7': 2,
            'E': 4, 'Em': 4, 'E7': 4, 'Emaj7': 4, 'Em7': 4,
            'F': 5, 'Fm': 5, 'F7': 5, 'Fmaj7': 5, 'Fm7': 5,
            'G': 7, 'Gm': 7, 'G7': 7, 'Gmaj7': 7, 'Gm7': 7,
            'A': 9, 'Am': 9, 'A7': 9, 'Amaj7': 9, 'Am7': 9,
            'B': 11, 'Bm': 11, 'B7': 11, 'Bmaj7': 11, 'Bm7': 11,
        }

        # Walking bass approach patterns (intervals to target)
        self.approach_patterns = [
            [-1],           # Chromatic below
            [1],            # Chromatic above
            [-2, -1],       # Whole step then half step
            [2, 1],         # Whole step then half step (above)
            [-2],           # Whole step below
            [5],            # Fourth above (common in jazz)
            [-5],           # Fourth below
        ]

        # Mode weights for automatic selection
        self.mode_weights = {
            BassMode.ROOT: 0.35,
            BassMode.WALK: 0.25,
            BassMode.FIFTH: 0.20,
            BassMode.PEDAL: 0.10,
            BassMode.COUNTERPOINT: 0.10,
        }

    def generate_bass_pattern(self,
                              harmonic_context: Dict,
                              num_notes: int = 4,
                              mode: Optional[BassMode] = None,
                              tempo: float = 120.0) -> BassPattern:
        """
        Generate a bass pattern based on harmonic context.

        Args:
            harmonic_context: Dict with 'current_chord', 'next_chord', 'key'
            num_notes: Number of notes to generate
            mode: Bass mode (auto-select if None)
            tempo: Current tempo in BPM

        Returns:
            BassPattern with notes, timings, velocities, durations
        """
        # Auto-select mode if not specified
        if mode is None:
            mode = self._select_mode(harmonic_context)

        # Get root note for current chord
        current_chord = harmonic_context.get('current_chord', 'C')
        root_pc = self.chord_roots.get(current_chord, 0)

        # Find bass octave root
        root_note = self._find_bass_root(root_pc)

        # Generate pattern based on mode
        if mode == BassMode.ROOT:
            return self._generate_root_pattern(root_note, num_notes, tempo)
        elif mode == BassMode.WALK:
            next_chord = harmonic_context.get('next_chord', current_chord)
            next_pc = self.chord_roots.get(next_chord, root_pc)
            next_root = self._find_bass_root(next_pc)
            return self._generate_walking_pattern(root_note, next_root, num_notes, tempo)
        elif mode == BassMode.FIFTH:
            return self._generate_fifth_pattern(root_note, num_notes, tempo)
        elif mode == BassMode.PEDAL:
            return self._generate_pedal_pattern(root_note, num_notes, tempo)
        elif mode == BassMode.COUNTERPOINT:
            scale = harmonic_context.get('scale_degrees', [0, 2, 4, 5, 7, 9, 11])
            return self._generate_counterpoint_pattern(root_note, scale, num_notes, tempo)
        else:
            return self._generate_root_pattern(root_note, num_notes, tempo)

    def _select_mode(self, harmonic_context: Dict) -> BassMode:
        """Auto-select bass mode based on context"""
        # If chord is changing, use walking bass
        current = harmonic_context.get('current_chord', 'C')
        next_chord = harmonic_context.get('next_chord', current)

        if current != next_chord:
            # Chord change - higher chance of walking
            weights = self.mode_weights.copy()
            weights[BassMode.WALK] = 0.5
            weights[BassMode.ROOT] = 0.2
        else:
            weights = self.mode_weights

        modes = list(weights.keys())
        probs = [weights[m] for m in modes]
        return random.choices(modes, weights=probs)[0]

    def _find_bass_root(self, pitch_class: int) -> int:
        """Find the root note in bass range for given pitch class"""
        min_note, max_note = self.bass_range

        # Find lowest occurrence in range
        for octave in range(1, 5):
            note = octave * 12 + pitch_class
            if min_note <= note <= max_note:
                return note

        # Fallback to middle of range
        return (min_note + max_note) // 2

    def _generate_root_pattern(self, root: int, num_notes: int, tempo: float) -> BassPattern:
        """Generate root-focused pattern"""
        beat_duration = 60.0 / tempo
        notes = []
        timings = []
        velocities = []
        durations = []

        for i in range(num_notes):
            # Mostly root with occasional octave or fifth
            if i == 0 or random.random() < 0.7:
                note = root
            elif random.random() < 0.5:
                note = root + 12  # Octave up
                if note > self.bass_range[1]:
                    note = root
            else:
                note = root + 7   # Fifth
                if note > self.bass_range[1]:
                    note = root

            notes.append(note)

            # Steady timing for root pattern
            timing = beat_duration * random.choice([1.0, 2.0])
            timings.append(timing)

            # Strong, consistent velocity
            velocity = random.randint(85, 100)
            velocities.append(velocity)

            # Short to medium duration
            duration = beat_duration * random.uniform(0.5, 0.9)
            durations.append(duration)

        return BassPattern(
            notes=notes,
            timings=timings,
            velocities=velocities,
            durations=durations,
            mode=BassMode.ROOT,
            pattern_type="root_motion"
        )

    def _generate_walking_pattern(self, root: int, target: int,
                                   num_notes: int, tempo: float) -> BassPattern:
        """Generate walking bass pattern approaching target"""
        beat_duration = 60.0 / tempo
        notes = []
        timings = []
        velocities = []
        durations = []

        # Calculate approach
        interval = target - root

        # Normalize to single octave
        while interval > 6:
            interval -= 12
        while interval < -6:
            interval += 12

        # Build approach notes
        if num_notes <= 2:
            # Short pattern: just root and target
            notes = [root, target]
        else:
            # Start with root
            notes.append(root)

            # Choose approach pattern
            approach = random.choice(self.approach_patterns)

            # Fill middle notes with scale walk or chromatic approach
            remaining = num_notes - 2  # Reserve last for target
            current = root

            for i in range(remaining):
                if i == remaining - 1:
                    # Penultimate note: use approach pattern
                    approach_interval = approach[0] if approach else -1
                    # Approach the target
                    next_note = target + approach_interval
                else:
                    # Walk toward target
                    step = 1 if interval > 0 else -1
                    if random.random() < 0.3:
                        step *= 2  # Occasional larger step
                    next_note = current + step

                # Keep in range
                next_note = max(self.bass_range[0], min(self.bass_range[1], next_note))
                notes.append(next_note)
                current = next_note

            # End on target
            notes.append(target)

        # Walking bass has steady quarter note timing
        for i in range(len(notes)):
            timings.append(beat_duration)

            # Strong attack, slight accent on beat 1
            velocity = 90 if i % 4 == 0 else random.randint(75, 85)
            velocities.append(velocity)

            # Slightly detached for walking feel
            duration = beat_duration * 0.8
            durations.append(duration)

        return BassPattern(
            notes=notes,
            timings=timings,
            velocities=velocities,
            durations=durations,
            mode=BassMode.WALK,
            pattern_type="walking_bass"
        )

    def _generate_fifth_pattern(self, root: int, num_notes: int, tempo: float) -> BassPattern:
        """Generate root-fifth alternation pattern"""
        beat_duration = 60.0 / tempo
        notes = []
        timings = []
        velocities = []
        durations = []

        fifth = root + 7
        if fifth > self.bass_range[1]:
            fifth = root - 5  # Fifth below

        for i in range(num_notes):
            # Alternate root and fifth
            if i % 2 == 0:
                note = root
                velocity = random.randint(85, 95)  # Root stronger
            else:
                note = fifth
                velocity = random.randint(75, 85)  # Fifth softer

            notes.append(note)
            timings.append(beat_duration)
            velocities.append(velocity)
            durations.append(beat_duration * 0.7)

        return BassPattern(
            notes=notes,
            timings=timings,
            velocities=velocities,
            durations=durations,
            mode=BassMode.FIFTH,
            pattern_type="root_fifth"
        )

    def _generate_pedal_pattern(self, root: int, num_notes: int, tempo: float) -> BassPattern:
        """Generate pedal point (repeated root)"""
        beat_duration = 60.0 / tempo
        notes = []
        timings = []
        velocities = []
        durations = []

        for i in range(num_notes):
            notes.append(root)

            # Varied timing for interest
            if random.random() < 0.3:
                timing = beat_duration * 0.5  # Eighth note
            else:
                timing = beat_duration  # Quarter note
            timings.append(timing)

            # Slight velocity variation
            velocity = 80 + random.randint(-5, 10)
            velocities.append(velocity)

            # Short, punchy
            durations.append(beat_duration * 0.5)

        return BassPattern(
            notes=notes,
            timings=timings,
            velocities=velocities,
            durations=durations,
            mode=BassMode.PEDAL,
            pattern_type="pedal_point"
        )

    def _generate_counterpoint_pattern(self, root: int, scale: List[int],
                                        num_notes: int, tempo: float) -> BassPattern:
        """Generate melodic counterpoint bass line"""
        beat_duration = 60.0 / tempo
        notes = []
        timings = []
        velocities = []
        durations = []

        # Start from root
        current = root
        current_pc = root % 12

        for i in range(num_notes):
            if i == 0:
                note = root
            else:
                # Choose direction
                direction = random.choice([-1, 1, 1])  # Slight upward bias

                # Find next scale tone
                current_idx = scale.index(current_pc) if current_pc in scale else 0
                next_idx = (current_idx + direction) % len(scale)
                next_pc = scale[next_idx]

                # Calculate note in same or adjacent octave
                octave = current // 12
                note = octave * 12 + next_pc

                # Adjust octave if needed
                if note < self.bass_range[0]:
                    note += 12
                elif note > self.bass_range[1]:
                    note -= 12

                current = note
                current_pc = next_pc

            notes.append(note)

            # More varied timing for melodic interest
            timing = beat_duration * random.choice([0.5, 1.0, 1.0, 1.5])
            timings.append(timing)

            # Dynamic variation
            velocity = random.randint(70, 90)
            velocities.append(velocity)

            # Longer durations for melodic feel
            duration = beat_duration * random.uniform(0.6, 1.0)
            durations.append(duration)

        return BassPattern(
            notes=notes,
            timings=timings,
            velocities=velocities,
            durations=durations,
            mode=BassMode.COUNTERPOINT,
            pattern_type="counterpoint"
        )

    def adapt_mode_to_behavior(self, behavior_mode: str) -> BassMode:
        """
        Adapt bass mode based on overall behavior mode.

        Args:
            behavior_mode: Main behavior (SHADOW, MIRROR, COUPLE)

        Returns:
            Appropriate BassMode
        """
        mode_upper = behavior_mode.upper()

        if mode_upper == 'SHADOW':
            # Close imitation: root motion, stable
            return random.choices(
                [BassMode.ROOT, BassMode.FIFTH, BassMode.PEDAL],
                weights=[0.5, 0.3, 0.2]
            )[0]
        elif mode_upper == 'MIRROR':
            # Complementary: walking, some independence
            return random.choices(
                [BassMode.WALK, BassMode.FIFTH, BassMode.ROOT],
                weights=[0.4, 0.35, 0.25]
            )[0]
        elif mode_upper == 'COUPLE':
            # Independent: counterpoint, walking
            return random.choices(
                [BassMode.COUNTERPOINT, BassMode.WALK, BassMode.FIFTH],
                weights=[0.4, 0.35, 0.25]
            )[0]
        else:
            return self.default_mode


def demo():
    """Demonstrate bass generation"""
    print("=" * 70)
    print("Bass Generator - Demo")
    print("=" * 70)

    generator = BassGenerator()

    # Test different modes
    context = {
        'current_chord': 'C',
        'next_chord': 'G',
        'scale_degrees': [0, 2, 4, 5, 7, 9, 11]
    }

    modes = [BassMode.ROOT, BassMode.WALK, BassMode.FIFTH,
             BassMode.PEDAL, BassMode.COUNTERPOINT]

    for mode in modes:
        pattern = generator.generate_bass_pattern(
            context, num_notes=4, mode=mode, tempo=120.0
        )

        note_names = []
        for n in pattern.notes:
            names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            note_names.append(f"{names[n % 12]}{n // 12 - 1}")

        print(f"\n{mode.value.upper()}:")
        print(f"  Notes: {note_names}")
        print(f"  MIDI: {pattern.notes}")
        print(f"  Type: {pattern.pattern_type}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    demo()
