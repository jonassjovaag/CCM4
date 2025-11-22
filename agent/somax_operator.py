#!/usr/bin/env python3
"""
Somax Operator - AI-driven operator for Somax2
===============================================

Turns MusicHal_9000 into an intelligent operator for Somax2,
replacing the human operator with neural understanding.

Instead of generating MIDI directly, this sends OSC messages
to control Somax2's parameters based on:
- CLAP style/mood detection
- Wav2Vec musical understanding
- Episode management
- Phrase boundary detection

This gives you:
- Somax2's proven, production-ready navigation
- CCM4's neural "taste" and musical understanding
- Automated operation (no human operator needed)
"""

import time
import numpy as np
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

# OSC client
try:
    from pythonosc import udp_client
    from pythonosc.osc_message_builder import OscMessageBuilder
    OSC_AVAILABLE = True
except ImportError:
    OSC_AVAILABLE = False
    print("⚠️  python-osc not installed - install with: pip install python-osc")


class OperatorMode(Enum):
    """Operating modes for the AI operator"""
    PASSIVE = "passive"      # Only observe, don't send commands
    REACTIVE = "reactive"    # Respond to human input
    PROACTIVE = "proactive"  # Initiate musical ideas
    SHADOW = "shadow"        # Follow/imitate human closely


@dataclass
class SomaxState:
    """Current state of Somax2 parameters"""
    continuity: float = 0.5
    influence: float = 0.5
    activity: float = 0.5
    harmony_filter: bool = False
    transpose: int = 0
    memory_index: int = 0
    is_playing: bool = False
    last_trigger: float = 0.0


class SomaxOperator:
    """
    AI Operator for Somax2.

    Translates MusicHal_9000's neural analysis into OSC commands
    that control Somax2's generation parameters.

    This replaces the human operator with automated musical intelligence.
    """

    def __init__(self,
                 somax_host: str = "127.0.0.1",
                 somax_port: int = 7400,
                 player_name: str = "player1"):
        """
        Initialize Somax Operator.

        Args:
            somax_host: IP address of Somax2 (usually localhost)
            somax_port: OSC port Somax2 is listening on
            player_name: Name of the Somax2 player to control
        """
        self.somax_host = somax_host
        self.somax_port = somax_port
        self.player_name = player_name

        # OSC client
        self.osc_client = None
        if OSC_AVAILABLE:
            self.osc_client = udp_client.SimpleUDPClient(somax_host, somax_port)
            print(f"🎛️  SomaxOperator initialized: {somax_host}:{somax_port}")
            print(f"   Player: {player_name}")
        else:
            print("❌ OSC not available - SomaxOperator disabled")

        # Current state
        self.state = SomaxState()
        self.mode = OperatorMode.REACTIVE

        # Style to continuity mapping (your musical taste)
        self.style_continuity = {
            'jazz': 0.35,       # Very jumpy, spontaneous
            'blues': 0.45,      # Medium jumpy
            'rock': 0.55,       # Moderate
            'classical': 0.75,  # Smooth, connected
            'funk': 0.40,       # Rhythmic jumps
            'electronic': 0.30, # Pattern jumps
            'ambient': 0.85,    # Very smooth
            'ballad': 0.80,     # Connected phrases
            'folk': 0.60,       # Moderate flow
            'latin': 0.50,      # Rhythmic but flowing
        }

        # Timing control
        self.last_trigger_time = 0.0
        self.min_trigger_interval = 0.5  # Minimum seconds between triggers
        self.last_param_update = 0.0
        self.param_update_interval = 0.1  # Update params every 100ms

        # Influence tracking (mirrors SomaxBridge logic)
        self.current_influence = 0.5
        self.influence_attack = 0.7
        self.influence_decay = 0.95

        # Phrase tracking
        self.notes_since_trigger = 0
        self.target_phrase_length = 8

        # Activity smoothing
        self.activity_history: List[float] = []
        self.activity_window = 10  # Frames to average

        # Statistics
        self.triggers_sent = 0
        self.param_updates = 0

    def send_osc(self, address: str, *args):
        """Send OSC message to Somax2"""
        if self.osc_client:
            try:
                self.osc_client.send_message(address, args)
                return True
            except Exception as e:
                print(f"⚠️  OSC send error: {e}")
                return False
        return False

    # ==================== Core Control Methods ====================

    def set_continuity(self, value: float):
        """
        Set continuity parameter (0-1).

        Low = more jumps between patterns
        High = smoother, follow sequences
        """
        value = max(0.0, min(1.0, value))
        if abs(value - self.state.continuity) > 0.05:  # Only send if changed
            self.state.continuity = value
            self.send_osc(f"/{self.player_name}/continuity", value)
            self.param_updates += 1

    def set_influence(self, value: float):
        """
        Set influence parameter (0-1).

        How much the current input affects pattern selection.
        """
        value = max(0.0, min(1.0, value))
        if abs(value - self.state.influence) > 0.05:
            self.state.influence = value
            self.send_osc(f"/{self.player_name}/influence", value)
            self.param_updates += 1

    def set_activity(self, value: float):
        """
        Set activity/density parameter (0-1).

        How active/dense the output should be.
        """
        value = max(0.0, min(1.0, value))
        if abs(value - self.state.activity) > 0.05:
            self.state.activity = value
            self.send_osc(f"/{self.player_name}/activity", value)
            self.param_updates += 1

    def trigger(self):
        """
        Trigger Somax2 to generate output.

        This is the main "play now" command.
        """
        current_time = time.time()
        if current_time - self.last_trigger_time >= self.min_trigger_interval:
            self.send_osc(f"/{self.player_name}/trigger", 1)
            self.last_trigger_time = current_time
            self.state.last_trigger = current_time
            self.state.is_playing = True
            self.triggers_sent += 1
            self.notes_since_trigger = 0
            return True
        return False

    def silence(self):
        """Tell Somax2 to stop/be silent"""
        self.send_osc(f"/{self.player_name}/silence", 1)
        self.state.is_playing = False

    def set_transpose(self, semitones: int):
        """Set transposition in semitones"""
        if semitones != self.state.transpose:
            self.state.transpose = semitones
            self.send_osc(f"/{self.player_name}/transpose", semitones)

    def set_memory(self, index: int):
        """Select which memory bank to use"""
        if index != self.state.memory_index:
            self.state.memory_index = index
            self.send_osc(f"/{self.player_name}/memory", index)

    def set_harmony_filter(self, enabled: bool):
        """Enable/disable harmony filtering"""
        if enabled != self.state.harmony_filter:
            self.state.harmony_filter = enabled
            self.send_osc(f"/{self.player_name}/harmony_filter", 1 if enabled else 0)

    # ==================== High-Level AI Operator Methods ====================

    def on_audio_event(self,
                       event_data: Dict,
                       style: str = 'unknown',
                       episode_state: str = 'ACTIVE'):
        """
        Process audio event and send appropriate Somax2 commands.

        This is the main entry point - call this from MusicHal_9000's
        audio callback instead of generating MIDI directly.

        Args:
            event_data: Dict with audio analysis (from MusicHal_9000)
            style: CLAP-detected style
            episode_state: 'ACTIVE' or 'LISTENING'
        """
        current_time = time.time()

        # Extract relevant features
        is_onset = event_data.get('onset', False)
        consonance = event_data.get('hybrid_consonance', event_data.get('consonance', 0.5))
        rms_db = event_data.get('rms_db', -60)
        gesture_token = event_data.get('gesture_token')

        # Update influence based on input
        self._update_influence(is_onset)

        # Update activity tracking
        activity = self._calculate_activity(rms_db, is_onset)

        # Determine continuity from style
        continuity = self.style_continuity.get(style.lower(), 0.5)

        # Adjust based on episode state
        if episode_state == 'LISTENING':
            # When listening, be more conservative
            continuity = min(continuity + 0.2, 0.9)
            activity *= 0.5

        # Send parameter updates (throttled)
        if current_time - self.last_param_update >= self.param_update_interval:
            self.set_continuity(continuity)
            self.set_influence(self.current_influence)
            self.set_activity(activity)
            self.last_param_update = current_time

        # Decide whether to trigger
        should_trigger = self._should_trigger(
            is_onset=is_onset,
            consonance=consonance,
            episode_state=episode_state
        )

        if should_trigger:
            self.trigger()

        # Track notes for phrase management
        if is_onset:
            self.notes_since_trigger += 1

            # Check if phrase is complete
            if self.notes_since_trigger >= self.target_phrase_length:
                # Vary next phrase length
                import random
                self.target_phrase_length = random.randint(4, 12)

    def _update_influence(self, is_onset: bool):
        """Update influence based on input activity"""
        if is_onset:
            # Onset increases influence
            self.current_influence = min(1.0,
                self.current_influence * (1 - self.influence_attack) + self.influence_attack)
        else:
            # Decay over time
            self.current_influence *= self.influence_decay

    def _calculate_activity(self, rms_db: float, is_onset: bool) -> float:
        """Calculate activity level from audio"""
        # Normalize RMS to 0-1
        normalized_rms = max(0, min(1, (rms_db + 60) / 60))

        # Boost on onset
        if is_onset:
            normalized_rms = min(1.0, normalized_rms + 0.3)

        # Smooth over time
        self.activity_history.append(normalized_rms)
        if len(self.activity_history) > self.activity_window:
            self.activity_history = self.activity_history[-self.activity_window:]

        return np.mean(self.activity_history)

    def _should_trigger(self,
                        is_onset: bool,
                        consonance: float,
                        episode_state: str) -> bool:
        """
        Decide whether to trigger Somax2 generation.

        This is where the AI operator makes its key decision.
        """
        # Don't trigger if in listening mode with low influence
        if episode_state == 'LISTENING' and self.current_influence < 0.4:
            return False

        # Trigger on strong onsets with high influence
        if is_onset and self.current_influence > 0.6:
            return True

        # Trigger on high consonance moments (musical anchor points)
        if consonance > 0.8 and self.current_influence > 0.5:
            return True

        # Proactive mode: trigger even without onset
        if self.mode == OperatorMode.PROACTIVE:
            if self.current_influence > 0.7:
                return True

        return False

    def set_mode(self, mode: OperatorMode):
        """Set operator mode"""
        self.mode = mode
        print(f"🎛️  Operator mode: {mode.value}")

    def on_phrase_boundary(self):
        """Called when a phrase boundary is detected"""
        # Could trigger silence or new phrase
        if self.state.is_playing:
            # Let current phrase finish, don't trigger new
            pass

    def on_episode_change(self, new_state: str):
        """Called when episode state changes"""
        if new_state == 'LISTENING':
            # Reduce activity when switching to listening
            self.set_activity(self.state.activity * 0.5)
        elif new_state == 'ACTIVE':
            # Boost activity when becoming active
            self.set_activity(min(1.0, self.state.activity * 1.5))

    def get_statistics(self) -> Dict:
        """Get operator statistics"""
        return {
            'triggers_sent': self.triggers_sent,
            'param_updates': self.param_updates,
            'current_influence': self.current_influence,
            'current_continuity': self.state.continuity,
            'current_activity': self.state.activity,
            'mode': self.mode.value,
            'is_playing': self.state.is_playing,
            'notes_since_trigger': self.notes_since_trigger
        }

    def reset(self):
        """Reset operator state"""
        self.state = SomaxState()
        self.current_influence = 0.5
        self.notes_since_trigger = 0
        self.activity_history = []


# ==================== Utility Functions ====================

def create_operator_for_somax2(host: str = "127.0.0.1",
                                port: int = 7400,
                                player: str = "player1") -> Optional[SomaxOperator]:
    """
    Create a Somax operator with default settings.

    Args:
        host: Somax2 host
        port: Somax2 OSC port
        player: Player name in Somax2

    Returns:
        SomaxOperator instance or None if OSC not available
    """
    if not OSC_AVAILABLE:
        print("❌ Cannot create operator: python-osc not installed")
        return None

    return SomaxOperator(
        somax_host=host,
        somax_port=port,
        player_name=player
    )


def demo():
    """Demonstrate Somax Operator"""
    print("=" * 70)
    print("Somax Operator - Demo")
    print("=" * 70)

    # Create operator
    operator = SomaxOperator(
        somax_host="127.0.0.1",
        somax_port=7400,
        player_name="player1"
    )

    # Simulate some events
    print("\nSimulating audio events...")

    import random

    for i in range(20):
        # Fake event data
        event_data = {
            'onset': random.random() < 0.3,
            'hybrid_consonance': random.uniform(0.4, 0.9),
            'rms_db': random.uniform(-40, -10),
            'gesture_token': random.randint(0, 63)
        }

        # Process event
        operator.on_audio_event(
            event_data,
            style='jazz',
            episode_state='ACTIVE' if random.random() > 0.3 else 'LISTENING'
        )

        if i % 5 == 0:
            stats = operator.get_statistics()
            print(f"\nFrame {i}:")
            print(f"  Triggers: {stats['triggers_sent']}")
            print(f"  Influence: {stats['current_influence']:.2f}")
            print(f"  Continuity: {stats['current_continuity']:.2f}")

    print("\n" + "=" * 70)
    print("Final Statistics:")
    for k, v in operator.get_statistics().items():
        print(f"  {k}: {v}")
    print("=" * 70)


if __name__ == "__main__":
    demo()
