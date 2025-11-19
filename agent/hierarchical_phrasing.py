#!/usr/bin/env python3
"""
Hierarchical Phrasing System
=============================

Implements nested timing levels for human-like musical phrasing:
- Note level: Individual note timing within a phrase
- Phrase level: 4-16 beats, coherent musical statement
- Phrase Group level: 2-4 phrases forming a larger arc
- Episode level: Multiple phrase groups (2-5 minutes of music)

This creates the "give and take" feel in live performance by:
1. Tracking where we are in the hierarchy
2. Adjusting timing/dynamics based on position
3. Creating tension/release across multiple time scales

Based on music cognition research on hierarchical structure perception.
"""

import time
import random
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class PhrasePosition(Enum):
    """Position within a phrase for timing/dynamics adjustment"""
    START = "start"          # First 25% - establishing
    MIDDLE = "middle"        # 25-75% - developing
    CLIMAX = "climax"        # 60-80% - peak tension
    END = "end"              # Last 20% - resolving


class GroupPosition(Enum):
    """Position within a phrase group"""
    OPENING = "opening"      # First phrase - statement
    DEVELOPMENT = "development"  # Middle phrases - exploration
    CLOSING = "closing"      # Last phrase - resolution


@dataclass
class PhraseState:
    """Tracks state of current phrase"""
    phrase_id: str
    start_time: float
    notes_played: int = 0
    target_notes: int = 8
    position: PhrasePosition = PhrasePosition.START
    tension: float = 0.5  # 0-1 tension level

    def get_progress(self) -> float:
        """Get phrase progress as 0-1"""
        if self.target_notes == 0:
            return 1.0
        return min(1.0, self.notes_played / self.target_notes)

    def update_position(self):
        """Update position based on progress"""
        progress = self.get_progress()
        if progress < 0.25:
            self.position = PhrasePosition.START
        elif progress < 0.6:
            self.position = PhrasePosition.MIDDLE
        elif progress < 0.8:
            self.position = PhrasePosition.CLIMAX
        else:
            self.position = PhrasePosition.END


@dataclass
class PhraseGroupState:
    """Tracks state of current phrase group"""
    group_id: str
    start_time: float
    phrases_completed: int = 0
    target_phrases: int = 3
    position: GroupPosition = GroupPosition.OPENING
    arc_tension: float = 0.3  # Overall group tension

    def get_progress(self) -> float:
        """Get group progress as 0-1"""
        if self.target_phrases == 0:
            return 1.0
        return min(1.0, self.phrases_completed / self.target_phrases)

    def update_position(self):
        """Update position based on phrases completed"""
        if self.phrases_completed == 0:
            self.position = GroupPosition.OPENING
        elif self.phrases_completed < self.target_phrases - 1:
            self.position = GroupPosition.DEVELOPMENT
        else:
            self.position = GroupPosition.CLOSING


@dataclass
class EpisodeState:
    """Tracks state of current episode (longest time scale)"""
    episode_id: str
    start_time: float
    groups_completed: int = 0
    target_groups: int = 4
    overall_intensity: float = 0.5

    def get_progress(self) -> float:
        """Get episode progress as 0-1"""
        if self.target_groups == 0:
            return 1.0
        return min(1.0, self.groups_completed / self.target_groups)


class HierarchicalPhraseEngine:
    """
    Manages hierarchical phrasing for natural musical expression.

    Tracks nested time scales and provides timing/dynamics adjustments
    based on position within each level of the hierarchy.
    """

    def __init__(self,
                 phrase_length_range: Tuple[int, int] = (4, 16),
                 phrases_per_group: Tuple[int, int] = (2, 4),
                 groups_per_episode: Tuple[int, int] = (3, 6),
                 visualization_manager=None):
        """
        Initialize hierarchical phrase engine.

        Args:
            phrase_length_range: (min, max) notes per phrase
            phrases_per_group: (min, max) phrases per group
            groups_per_episode: (min, max) groups per episode
            visualization_manager: Optional visualization manager for UI updates
        """
        self.phrase_length_range = phrase_length_range
        self.phrases_per_group = phrases_per_group
        self.groups_per_episode = groups_per_episode
        self.visualization_manager = visualization_manager

        # Current state at each level
        self.current_phrase: Optional[PhraseState] = None
        self.current_group: Optional[PhraseGroupState] = None
        self.current_episode: Optional[EpisodeState] = None

        # Timing adjustment parameters
        self.timing_adjustments = {
            # Phrase position adjustments (multiply base timing)
            PhrasePosition.START: 1.1,      # Slightly slower start
            PhrasePosition.MIDDLE: 1.0,     # Normal
            PhrasePosition.CLIMAX: 0.85,    # Push forward at climax
            PhrasePosition.END: 1.15,       # Slow down for resolution
        }

        self.dynamics_adjustments = {
            # Velocity adjustments based on position
            PhrasePosition.START: 0.85,     # Softer start
            PhrasePosition.MIDDLE: 1.0,     # Normal
            PhrasePosition.CLIMAX: 1.15,    # Louder at climax
            PhrasePosition.END: 0.9,        # Softer ending
        }

        # Phrase boundary detection
        self.silence_threshold = 1.5  # seconds of silence = phrase boundary
        self.last_note_time = 0.0

        # Statistics
        self.total_phrases = 0
        self.total_groups = 0
        self.total_episodes = 0

        # Initialize first episode
        self._start_new_episode()

    def _start_new_episode(self):
        """Start a new episode"""
        self.total_episodes += 1
        target_groups = random.randint(*self.groups_per_episode)

        self.current_episode = EpisodeState(
            episode_id=f"episode_{self.total_episodes}",
            start_time=time.time(),
            target_groups=target_groups
        )

        self._start_new_group()

        print(f"🎭 New episode started: {target_groups} phrase groups planned")

    def _start_new_group(self):
        """Start a new phrase group"""
        self.total_groups += 1
        target_phrases = random.randint(*self.phrases_per_group)

        # Determine group tension based on episode progress
        episode_progress = self.current_episode.get_progress() if self.current_episode else 0.5

        # Tension arc: low → high → low across episode
        if episode_progress < 0.3:
            arc_tension = 0.3 + episode_progress
        elif episode_progress < 0.7:
            arc_tension = 0.6 + (episode_progress - 0.3) * 0.5
        else:
            arc_tension = 0.8 - (episode_progress - 0.7) * 1.5

        self.current_group = PhraseGroupState(
            group_id=f"group_{self.total_groups}",
            start_time=time.time(),
            target_phrases=target_phrases,
            arc_tension=max(0.2, min(0.9, arc_tension))
        )

        self._start_new_phrase()

    def _start_new_phrase(self):
        """Start a new phrase"""
        self.total_phrases += 1
        target_notes = random.randint(*self.phrase_length_range)

        # Adjust phrase length based on group position
        if self.current_group:
            if self.current_group.position == GroupPosition.OPENING:
                target_notes = int(target_notes * 0.8)  # Shorter opening
            elif self.current_group.position == GroupPosition.CLOSING:
                target_notes = int(target_notes * 1.2)  # Longer closing

        # Calculate initial tension based on group
        initial_tension = self.current_group.arc_tension if self.current_group else 0.5

        self.current_phrase = PhraseState(
            phrase_id=f"phrase_{self.total_phrases}",
            start_time=time.time(),
            target_notes=max(2, target_notes),  # At least 2 notes
            tension=initial_tension
        )

    def on_note_played(self, timestamp: float = None):
        """
        Called when a note is played. Updates hierarchy state.

        Args:
            timestamp: Time of note (default: now)
        """
        if timestamp is None:
            timestamp = time.time()

        self.last_note_time = timestamp

        # Ensure we have a phrase
        if self.current_phrase is None:
            self._start_new_phrase()

        # Update phrase state
        self.current_phrase.notes_played += 1
        self.current_phrase.update_position()

        # Update tension based on position
        progress = self.current_phrase.get_progress()
        if progress < 0.6:
            # Build tension
            self.current_phrase.tension = min(0.9, self.current_phrase.tension + 0.05)
        else:
            # Release tension
            self.current_phrase.tension = max(0.2, self.current_phrase.tension - 0.08)

        # Check if phrase is complete
        if self.current_phrase.notes_played >= self.current_phrase.target_notes:
            self._complete_phrase()

    def _complete_phrase(self):
        """Complete current phrase and potentially start new one"""
        if self.current_group:
            self.current_group.phrases_completed += 1
            self.current_group.update_position()

            # Check if group is complete
            if self.current_group.phrases_completed >= self.current_group.target_phrases:
                self._complete_group()
            else:
                self._start_new_phrase()
        else:
            self._start_new_phrase()

    def _complete_group(self):
        """Complete current group and potentially start new one"""
        if self.current_episode:
            self.current_episode.groups_completed += 1

            # Check if episode is complete
            if self.current_episode.groups_completed >= self.current_episode.target_groups:
                self._complete_episode()
            else:
                self._start_new_group()
        else:
            self._start_new_group()

    def _complete_episode(self):
        """Complete current episode and start new one"""
        duration = time.time() - self.current_episode.start_time
        print(f"🎭 Episode complete: {self.current_episode.groups_completed} groups in {duration:.1f}s")
        self._start_new_episode()

    def check_phrase_boundary(self, current_time: float = None) -> bool:
        """
        Check if we've crossed a phrase boundary (silence detection).

        Args:
            current_time: Current time (default: now)

        Returns:
            True if phrase boundary detected
        """
        if current_time is None:
            current_time = time.time()

        silence_duration = current_time - self.last_note_time

        if silence_duration > self.silence_threshold:
            # Silence detected - end current phrase
            if self.current_phrase and self.current_phrase.notes_played > 0:
                self._complete_phrase()
                return True

        return False

    def get_timing_adjustment(self) -> float:
        """
        Get timing multiplier based on current hierarchical position.

        Returns:
            Multiplier for note timing (< 1 = faster, > 1 = slower)
        """
        if self.current_phrase is None:
            return 1.0

        # Base adjustment from phrase position
        base_adj = self.timing_adjustments.get(self.current_phrase.position, 1.0)

        # Group-level adjustment
        group_adj = 1.0
        if self.current_group:
            if self.current_group.position == GroupPosition.OPENING:
                group_adj = 1.05  # Slightly slower opening
            elif self.current_group.position == GroupPosition.CLOSING:
                group_adj = 1.1   # Slow down for group ending

        # Episode-level adjustment (subtle)
        episode_adj = 1.0
        if self.current_episode:
            progress = self.current_episode.get_progress()
            if progress > 0.8:
                episode_adj = 1.05  # Slow down at episode end

        return base_adj * group_adj * episode_adj

    def get_dynamics_adjustment(self) -> float:
        """
        Get velocity multiplier based on current hierarchical position.

        Returns:
            Multiplier for note velocity (< 1 = softer, > 1 = louder)
        """
        if self.current_phrase is None:
            return 1.0

        # Base adjustment from phrase position
        base_adj = self.dynamics_adjustments.get(self.current_phrase.position, 1.0)

        # Tension-based adjustment
        tension_adj = 0.8 + (self.current_phrase.tension * 0.4)  # 0.8 to 1.2

        # Group arc adjustment
        group_adj = 1.0
        if self.current_group:
            group_adj = 0.9 + (self.current_group.arc_tension * 0.2)  # 0.9 to 1.1

        return base_adj * tension_adj * group_adj

    def get_phrase_length_for_mode(self, mode: str) -> int:
        """
        Get recommended phrase length based on behavior mode.

        Args:
            mode: Behavior mode (SHADOW, MIRROR, COUPLE)

        Returns:
            Number of notes for phrase
        """
        mode_upper = mode.upper()

        # Base range
        min_len, max_len = self.phrase_length_range

        # Adjust based on mode
        if mode_upper == 'SHADOW':
            # Shorter, responsive phrases
            return random.randint(min_len, min_len + (max_len - min_len) // 3)
        elif mode_upper == 'MIRROR':
            # Medium phrases
            mid = (min_len + max_len) // 2
            return random.randint(mid - 2, mid + 2)
        elif mode_upper == 'COUPLE':
            # Longer, independent phrases
            return random.randint(min_len + (max_len - min_len) // 2, max_len)
        else:
            return random.randint(min_len, max_len)

    def get_current_state(self) -> Dict:
        """
        Get current hierarchical state for visualization/debugging.

        Returns:
            Dictionary with current state at all levels
        """
        state = {
            'phrase': None,
            'group': None,
            'episode': None,
            'timing_adjustment': self.get_timing_adjustment(),
            'dynamics_adjustment': self.get_dynamics_adjustment(),
        }

        if self.current_phrase:
            state['phrase'] = {
                'id': self.current_phrase.phrase_id,
                'progress': self.current_phrase.get_progress(),
                'position': self.current_phrase.position.value,
                'tension': self.current_phrase.tension,
                'notes': f"{self.current_phrase.notes_played}/{self.current_phrase.target_notes}"
            }

        if self.current_group:
            state['group'] = {
                'id': self.current_group.group_id,
                'progress': self.current_group.get_progress(),
                'position': self.current_group.position.value,
                'arc_tension': self.current_group.arc_tension,
                'phrases': f"{self.current_group.phrases_completed}/{self.current_group.target_phrases}"
            }

        if self.current_episode:
            state['episode'] = {
                'id': self.current_episode.episode_id,
                'progress': self.current_episode.get_progress(),
                'intensity': self.current_episode.overall_intensity,
                'groups': f"{self.current_episode.groups_completed}/{self.current_episode.target_groups}"
            }

        return state

    def apply_to_phrase(self, notes: List[int], timings: List[float],
                        velocities: List[int]) -> Tuple[List[float], List[int]]:
        """
        Apply hierarchical timing and dynamics to a generated phrase.

        Args:
            notes: List of MIDI notes
            timings: List of inter-onset intervals
            velocities: List of velocities

        Returns:
            Tuple of (adjusted_timings, adjusted_velocities)
        """
        adjusted_timings = []
        adjusted_velocities = []

        for i, (timing, velocity) in enumerate(zip(timings, velocities)):
            # Track note for state updates
            self.on_note_played()

            # Get adjustments based on current position
            timing_adj = self.get_timing_adjustment()
            dynamics_adj = self.get_dynamics_adjustment()

            # Apply adjustments
            adjusted_timing = timing * timing_adj
            adjusted_velocity = int(velocity * dynamics_adj)

            # Clamp values
            adjusted_timing = max(0.1, adjusted_timing)
            adjusted_velocity = max(1, min(127, adjusted_velocity))

            adjusted_timings.append(adjusted_timing)
            adjusted_velocities.append(adjusted_velocity)

        return adjusted_timings, adjusted_velocities

    def reset(self):
        """Reset all hierarchical state"""
        self.current_phrase = None
        self.current_group = None
        self.current_episode = None
        self.total_phrases = 0
        self.total_groups = 0
        self.total_episodes = 0
        self.last_note_time = 0.0
        self._start_new_episode()


def demo():
    """Demonstrate hierarchical phrasing"""
    print("=" * 70)
    print("Hierarchical Phrasing System - Demo")
    print("=" * 70)

    engine = HierarchicalPhraseEngine(
        phrase_length_range=(4, 8),
        phrases_per_group=(2, 3),
        groups_per_episode=(2, 3)
    )

    # Simulate playing notes
    print("\n🎵 Simulating note playback...")

    for i in range(30):
        engine.on_note_played()

        state = engine.get_current_state()
        timing_adj = state['timing_adjustment']
        dynamics_adj = state['dynamics_adjustment']

        if state['phrase']:
            phrase = state['phrase']
            print(f"Note {i+1}: Phrase {phrase['notes']} ({phrase['position']}), "
                  f"timing×{timing_adj:.2f}, dynamics×{dynamics_adj:.2f}")

        # Small delay between notes
        time.sleep(0.05)

    print("\n" + "=" * 70)
    print(f"Total: {engine.total_phrases} phrases, "
          f"{engine.total_groups} groups, "
          f"{engine.total_episodes} episodes")
    print("=" * 70)


if __name__ == "__main__":
    demo()
