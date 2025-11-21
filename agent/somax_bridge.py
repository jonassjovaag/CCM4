#!/usr/bin/env python3
"""
Somax Bridge - DYCI2-style Navigation with CCM4 Understanding
==============================================================

Bridges CCM4's neural understanding (Wav2Vec/CLAP) to DYCI2-style
Factor Oracle navigation for better live response generation.

Key concepts from DYCI2/Somax2:
- Factor Oracle: Suffix automaton for pattern storage
- Influence: Continuous control parameter for similarity
- Continuity: Controls pattern jumps vs following sequences
- Navigation: Traversing oracle with influence decay

This provides the "shut up", "give and take", and phrasing behaviors
that Somax2 excels at, while keeping CCM4's neural music understanding.
"""

import time
import random
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum


class NavigationMode(Enum):
    """Navigation modes for Factor Oracle traversal"""
    FOLLOW = "follow"       # Follow suffix links (similar)
    JUMP = "jump"           # Jump to distant patterns
    CONTINUE = "continue"   # Continue current sequence
    WAIT = "wait"           # Don't generate (silence/listening)


@dataclass
class OracleState:
    """State in Factor Oracle"""
    index: int
    label: Any              # Token or event
    embedding: np.ndarray   # 768D embedding
    suffix_link: int        # Link to longest suffix
    transitions: Dict[Any, int] = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)


@dataclass
class NavigationContext:
    """Context for navigation decisions"""
    influence: float        # 0-1, how much to follow input
    continuity: float       # 0-1, tendency to continue vs jump
    temperature: float      # Randomness in selection
    phrase_position: float  # 0-1, position within phrase
    episode_state: str      # ACTIVE or LISTENING
    style: str              # From CLAP
    energy: float           # Current energy level


class FactorOracleNavigator:
    """
    Pure-Python Factor Oracle implementation compatible with DYCI2 concepts.

    This provides the navigation intelligence that Somax2 excels at:
    - Suffix-based pattern matching
    - Influence-controlled similarity
    - Continuity-based flow control
    - Natural phrase boundaries
    """

    def __init__(self,
                 embedding_dim: int = 768,
                 max_states: int = 10000):
        """
        Initialize Factor Oracle.

        Args:
            embedding_dim: Dimension of embeddings
            max_states: Maximum states before pruning
        """
        self.embedding_dim = embedding_dim
        self.max_states = max_states

        # Oracle structure
        self.states: List[OracleState] = []
        self.current_state: int = 0

        # Embedding index for similarity search
        self.embeddings: List[np.ndarray] = []

        # Label to state mapping
        self.label_index: Dict[Any, List[int]] = defaultdict(list)

        # Navigation history
        self.navigation_history: List[int] = []
        self.generation_count: int = 0

        # Initialize with empty state
        self._init_oracle()

    def _init_oracle(self):
        """Initialize oracle with start state"""
        start_state = OracleState(
            index=0,
            label=None,
            embedding=np.zeros(self.embedding_dim),
            suffix_link=-1
        )
        self.states.append(start_state)
        self.embeddings.append(start_state.embedding)

    def add_sequence(self,
                     labels: List[Any],
                     embeddings: List[np.ndarray],
                     metadata: List[Dict] = None):
        """
        Add a sequence to the oracle (incremental construction).

        This is the online Factor Oracle construction algorithm.

        Args:
            labels: Sequence of labels (tokens)
            embeddings: Corresponding embeddings
            metadata: Optional metadata for each state
        """
        if len(labels) != len(embeddings):
            return

        if metadata is None:
            metadata = [{} for _ in labels]

        for i, (label, emb, meta) in enumerate(zip(labels, embeddings, metadata)):
            self._add_state(label, emb, meta)

    def _add_state(self, label: Any, embedding: np.ndarray, metadata: Dict):
        """Add single state to oracle using incremental algorithm"""
        # Check capacity
        if len(self.states) >= self.max_states:
            self._prune_oldest()

        # Create new state
        new_idx = len(self.states)
        new_state = OracleState(
            index=new_idx,
            label=label,
            embedding=embedding,
            suffix_link=0,  # Will be updated
            metadata=metadata
        )

        # Add transition from previous state
        prev_idx = new_idx - 1
        if prev_idx >= 0:
            self.states[prev_idx].transitions[label] = new_idx

        # Compute suffix link (longest proper suffix that exists)
        k = self.states[prev_idx].suffix_link if prev_idx >= 0 else -1

        while k >= 0 and label not in self.states[k].transitions:
            # Add transition for this suffix
            self.states[k].transitions[label] = new_idx
            k = self.states[k].suffix_link

        if k == -1:
            new_state.suffix_link = 0
        else:
            # Follow existing transition
            existing = self.states[k].transitions.get(label)
            if existing == new_idx:
                new_state.suffix_link = 0
            else:
                new_state.suffix_link = existing if existing else 0

        # Store
        self.states.append(new_state)
        self.embeddings.append(embedding)
        self.label_index[label].append(new_idx)

    def _prune_oldest(self):
        """Remove oldest states to make room"""
        # Keep most recent half
        keep_from = len(self.states) // 2

        # Rebuild oracle from kept states
        kept_states = self.states[keep_from:]
        kept_embeddings = self.embeddings[keep_from:]

        self.states = []
        self.embeddings = []
        self.label_index = defaultdict(list)

        self._init_oracle()

        for state in kept_states:
            self._add_state(state.label, state.embedding, state.metadata)

    def navigate(self,
                 query_embedding: np.ndarray,
                 context: NavigationContext) -> Tuple[int, NavigationMode]:
        """
        Navigate oracle based on query and context.

        This is the core DYCI2-style navigation that determines
        what to generate next based on:
        - Query similarity (influence)
        - Current position (continuity)
        - Phrase context

        Args:
            query_embedding: Input embedding to match
            context: Navigation context with parameters

        Returns:
            (state_index, mode) - Selected state and navigation mode
        """
        if len(self.states) <= 1:
            return 0, NavigationMode.WAIT

        # Check if we should be silent (listening)
        if context.episode_state == 'LISTENING':
            if random.random() > context.influence:
                return self.current_state, NavigationMode.WAIT

        # Get candidate states based on influence
        candidates = self._get_candidates(query_embedding, context)

        if not candidates:
            return self.current_state, NavigationMode.WAIT

        # Select based on continuity
        selected, mode = self._select_with_continuity(candidates, context)

        # Update state
        self.current_state = selected
        self.navigation_history.append(selected)
        self.generation_count += 1

        return selected, mode

    def _get_candidates(self,
                        query_embedding: np.ndarray,
                        context: NavigationContext) -> List[Tuple[int, float]]:
        """Get candidate states based on similarity to query"""
        candidates = []

        # Normalize query
        query_norm = np.linalg.norm(query_embedding)
        if query_norm > 0:
            query_normalized = query_embedding / query_norm
        else:
            query_normalized = query_embedding

        # Calculate similarities
        for i, emb in enumerate(self.embeddings[1:], 1):  # Skip start state
            emb_norm = np.linalg.norm(emb)
            if emb_norm > 0:
                sim = np.dot(query_normalized, emb / emb_norm)
            else:
                sim = 0

            # Apply influence: higher influence means we need higher similarity
            threshold = 0.3 + (0.4 * context.influence)

            if sim >= threshold:
                candidates.append((i, sim))

        # Sort by similarity
        candidates.sort(key=lambda x: x[1], reverse=True)

        # Limit candidates
        max_candidates = int(10 * (1 - context.influence) + 3)
        return candidates[:max_candidates]

    def _select_with_continuity(self,
                                 candidates: List[Tuple[int, float]],
                                 context: NavigationContext) -> Tuple[int, NavigationMode]:
        """Select state based on continuity preference"""
        if not candidates:
            return self.current_state, NavigationMode.WAIT

        # Options based on continuity

        # 1. Continue from current state (high continuity)
        next_from_current = None
        if self.current_state < len(self.states) - 1:
            # Direct continuation
            next_from_current = self.current_state + 1

        # 2. Follow suffix link (medium continuity - similar context)
        suffix_option = None
        if self.states[self.current_state].suffix_link > 0:
            suffix_state = self.states[self.current_state].suffix_link
            if suffix_state < len(self.states) - 1:
                suffix_option = suffix_state + 1

        # 3. Jump to best match (low continuity)
        jump_option = candidates[0][0]

        # Decision based on continuity
        roll = random.random()

        if roll < context.continuity * 0.7:
            # High continuity: prefer sequence continuation
            if next_from_current:
                return next_from_current, NavigationMode.CONTINUE
            elif suffix_option:
                return suffix_option, NavigationMode.FOLLOW
            else:
                return jump_option, NavigationMode.JUMP

        elif roll < context.continuity:
            # Medium: prefer suffix link
            if suffix_option:
                return suffix_option, NavigationMode.FOLLOW
            elif next_from_current:
                return next_from_current, NavigationMode.CONTINUE
            else:
                return jump_option, NavigationMode.JUMP

        else:
            # Low continuity: jump to similar
            if len(candidates) > 1 and context.temperature > 0.3:
                # Add randomness
                idx = min(int(random.expovariate(2)), len(candidates) - 1)
                return candidates[idx][0], NavigationMode.JUMP
            else:
                return jump_option, NavigationMode.JUMP

    def get_state_sequence(self,
                           state_idx: int,
                           length: int = 4) -> List[OracleState]:
        """Get sequence of states starting from given state"""
        sequence = []
        current = state_idx

        for _ in range(length):
            if current >= len(self.states):
                break
            sequence.append(self.states[current])
            current += 1

        return sequence

    def get_statistics(self) -> Dict:
        """Get oracle statistics"""
        return {
            'num_states': len(self.states),
            'max_states': self.max_states,
            'generation_count': self.generation_count,
            'unique_labels': len(self.label_index),
            'avg_suffix_depth': self._calc_avg_suffix_depth()
        }

    def _calc_avg_suffix_depth(self) -> float:
        """Calculate average suffix link depth"""
        if len(self.states) <= 1:
            return 0

        depths = []
        for state in self.states[1:]:
            depth = 0
            current = state.suffix_link
            while current > 0:
                depth += 1
                current = self.states[current].suffix_link
                if depth > 100:  # Safety
                    break
            depths.append(depth)

        return np.mean(depths) if depths else 0


class SomaxBridge:
    """
    Bridge between CCM4's neural understanding and DYCI2-style generation.

    This is the key integration point that:
    - Takes CCM4's Wav2Vec/CLAP understanding
    - Feeds it into Factor Oracle navigation
    - Outputs musically coherent responses with proper phrasing

    The bridge handles:
    - Influence decay (how much to follow input over time)
    - Episode management (when to play vs listen)
    - Style-to-continuity mapping (jazz = more jumps, ballad = smooth)
    - Phrase boundary detection
    """

    def __init__(self,
                 embedding_dim: int = 768,
                 default_continuity: float = 0.6):
        """
        Initialize Somax Bridge.

        Args:
            embedding_dim: Dimension of embeddings (Wav2Vec)
            default_continuity: Default continuity (0=jumpy, 1=smooth)
        """
        self.embedding_dim = embedding_dim
        self.default_continuity = default_continuity

        # Separate oracles for melody and bass
        self.melody_oracle = FactorOracleNavigator(embedding_dim)
        self.bass_oracle = FactorOracleNavigator(embedding_dim)

        # Influence tracking (decays over time)
        self.current_influence = 0.5
        self.influence_decay = 0.95  # Per frame decay
        self.influence_attack = 0.8  # Jump on new input

        # Episode state
        self.episode_state = 'LISTENING'
        self.frames_since_input = 0
        self.silence_threshold = 30  # Frames before considering silence

        # Phrase tracking
        self.notes_in_phrase = 0
        self.target_phrase_length = 8
        self.phrase_complete = False

        # Style to continuity mapping
        self.style_continuity = {
            'jazz': 0.4,        # More jumps, spontaneous
            'blues': 0.5,       # Medium
            'rock': 0.6,        # More driving
            'classical': 0.75,  # Smooth, connected
            'funk': 0.45,       # Rhythmic jumps
            'electronic': 0.35, # Pattern jumps
            'ambient': 0.85,    # Very smooth
            'ballad': 0.8,      # Connected phrases
            'unknown': 0.6
        }

        # Statistics
        self.total_generated = 0
        self.total_silences = 0

    def load_training_data(self,
                           melody_data: List[Dict],
                           bass_data: List[Dict] = None):
        """
        Load CCM4 training data into oracles.

        Args:
            melody_data: List of {tokens, embeddings, metadata}
            bass_data: Optional separate bass data
        """
        print(f"Loading {len(melody_data)} melody sequences into oracle...")

        for seq in melody_data:
            tokens = seq.get('tokens', [])
            embeddings = seq.get('embeddings', [])
            metadata = seq.get('metadata', [{}] * len(tokens))

            if tokens and embeddings:
                self.melody_oracle.add_sequence(tokens, embeddings, metadata)

        # Use melody for bass if no separate data
        if bass_data:
            print(f"Loading {len(bass_data)} bass sequences into oracle...")
            for seq in bass_data:
                tokens = seq.get('tokens', [])
                embeddings = seq.get('embeddings', [])
                metadata = seq.get('metadata', [{}] * len(tokens))

                if tokens and embeddings:
                    self.bass_oracle.add_sequence(tokens, embeddings, metadata)
        else:
            # Share melody oracle for bass
            self.bass_oracle = self.melody_oracle

        print(f"Oracles loaded: melody={self.melody_oracle.get_statistics()['num_states']} states")

    def on_audio_input(self,
                       embedding: np.ndarray,
                       is_onset: bool,
                       consonance: float,
                       style: str = 'unknown'):
        """
        Process incoming audio from human performer.

        Updates influence and episode state based on input.

        Args:
            embedding: 768D Wav2Vec embedding
            is_onset: Whether this is a note onset
            consonance: Consonance level (0-1)
            style: CLAP-detected style
        """
        if is_onset:
            # New input: increase influence
            self.current_influence = min(1.0,
                self.current_influence * (1 - self.influence_attack) + self.influence_attack)

            # Reset silence counter
            self.frames_since_input = 0

            # Switch to active if we were listening
            if self.episode_state == 'LISTENING' and self.current_influence > 0.6:
                self.episode_state = 'ACTIVE'
        else:
            # No onset: decay influence
            self.current_influence *= self.influence_decay
            self.frames_since_input += 1

            # Consider switching to listening after silence
            if self.frames_since_input > self.silence_threshold:
                if self.phrase_complete:
                    self.episode_state = 'LISTENING'
                    self.phrase_complete = False

    def generate_melody(self,
                        query_embedding: np.ndarray,
                        style: str = 'unknown',
                        temperature: float = 0.7) -> Optional[Dict]:
        """
        Generate melody response using Factor Oracle navigation.

        Args:
            query_embedding: Input embedding to respond to
            style: CLAP-detected style
            temperature: Generation randomness

        Returns:
            Dict with {token, embedding, should_play, navigation_mode} or None
        """
        # Build navigation context
        context = NavigationContext(
            influence=self.current_influence,
            continuity=self.style_continuity.get(style.lower(), self.default_continuity),
            temperature=temperature,
            phrase_position=self.notes_in_phrase / self.target_phrase_length,
            episode_state=self.episode_state,
            style=style,
            energy=self.current_influence  # Use influence as energy proxy
        )

        # Navigate oracle
        state_idx, mode = self.melody_oracle.navigate(query_embedding, context)

        # Check if we should generate
        if mode == NavigationMode.WAIT:
            self.total_silences += 1
            return {
                'token': None,
                'embedding': None,
                'should_play': False,
                'navigation_mode': mode.value,
                'reason': 'listening' if self.episode_state == 'LISTENING' else 'phrase_complete'
            }

        # Get state
        state = self.melody_oracle.states[state_idx]

        # Update phrase tracking
        self.notes_in_phrase += 1
        if self.notes_in_phrase >= self.target_phrase_length:
            self.phrase_complete = True
            self.notes_in_phrase = 0
            # Vary next phrase length
            self.target_phrase_length = random.randint(4, 12)

        self.total_generated += 1

        return {
            'token': state.label,
            'embedding': state.embedding,
            'should_play': True,
            'navigation_mode': mode.value,
            'state_index': state_idx,
            'metadata': state.metadata
        }

    def generate_bass(self,
                      query_embedding: np.ndarray,
                      style: str = 'unknown',
                      harmonic_context: Dict = None,
                      temperature: float = 0.5) -> Optional[Dict]:
        """
        Generate bass response using Factor Oracle navigation.

        Bass has higher continuity (more stable) and lower temperature.

        Args:
            query_embedding: Input embedding
            style: CLAP-detected style
            harmonic_context: Chord information
            temperature: Generation randomness (lower for bass)

        Returns:
            Dict with generation result
        """
        # Bass has higher continuity than melody
        bass_continuity = min(1.0,
            self.style_continuity.get(style.lower(), self.default_continuity) + 0.2)

        context = NavigationContext(
            influence=self.current_influence,
            continuity=bass_continuity,
            temperature=temperature * 0.7,  # Less random for bass
            phrase_position=self.notes_in_phrase / self.target_phrase_length,
            episode_state=self.episode_state,
            style=style,
            energy=self.current_influence
        )

        # Navigate bass oracle
        state_idx, mode = self.bass_oracle.navigate(query_embedding, context)

        if mode == NavigationMode.WAIT:
            return {
                'token': None,
                'embedding': None,
                'should_play': False,
                'navigation_mode': mode.value
            }

        state = self.bass_oracle.states[state_idx]

        return {
            'token': state.label,
            'embedding': state.embedding,
            'should_play': True,
            'navigation_mode': mode.value,
            'state_index': state_idx,
            'metadata': state.metadata
        }

    def set_episode_state(self, state: str):
        """Manually set episode state"""
        if state.upper() in ['ACTIVE', 'LISTENING']:
            self.episode_state = state.upper()

    def reset_phrase(self):
        """Reset phrase tracking for new phrase"""
        self.notes_in_phrase = 0
        self.phrase_complete = False

    def get_statistics(self) -> Dict:
        """Get bridge statistics"""
        return {
            'current_influence': self.current_influence,
            'episode_state': self.episode_state,
            'notes_in_phrase': self.notes_in_phrase,
            'target_phrase_length': self.target_phrase_length,
            'total_generated': self.total_generated,
            'total_silences': self.total_silences,
            'silence_ratio': self.total_silences / max(1, self.total_generated + self.total_silences),
            'melody_oracle': self.melody_oracle.get_statistics(),
            'bass_oracle': self.bass_oracle.get_statistics() if self.bass_oracle != self.melody_oracle else 'shared'
        }

    def should_respond(self) -> bool:
        """Quick check if bridge is in responding mode"""
        return (self.episode_state == 'ACTIVE' and
                self.current_influence > 0.3 and
                not self.phrase_complete)


def demo():
    """Demonstrate Somax Bridge"""
    print("=" * 70)
    print("Somax Bridge - Demo")
    print("=" * 70)

    # Create bridge
    bridge = SomaxBridge()

    # Generate fake training data
    print("\nGenerating test training data...")
    melody_data = []
    for i in range(20):
        seq_len = random.randint(4, 12)
        tokens = [random.randint(0, 63) for _ in range(seq_len)]
        embeddings = [np.random.randn(768) for _ in range(seq_len)]
        melody_data.append({
            'tokens': tokens,
            'embeddings': embeddings,
            'metadata': [{'note': t * 2 + 36} for t in tokens]
        })

    # Load into oracle
    bridge.load_training_data(melody_data)

    # Simulate interaction
    print("\nSimulating interaction...")

    for frame in range(50):
        # Simulate input
        is_onset = random.random() < 0.3
        embedding = np.random.randn(768)

        bridge.on_audio_input(
            embedding=embedding,
            is_onset=is_onset,
            consonance=random.random(),
            style='jazz'
        )

        # Generate response
        result = bridge.generate_melody(embedding, style='jazz')

        if frame % 10 == 0:
            print(f"\nFrame {frame}:")
            print(f"  Influence: {bridge.current_influence:.2f}")
            print(f"  Episode: {bridge.episode_state}")
            print(f"  Should play: {result['should_play']}")
            if result['should_play']:
                print(f"  Mode: {result['navigation_mode']}")

    # Statistics
    stats = bridge.get_statistics()
    print(f"\n\nStatistics:")
    print(f"  Total generated: {stats['total_generated']}")
    print(f"  Total silences: {stats['total_silences']}")
    print(f"  Silence ratio: {stats['silence_ratio']:.2%}")
    print(f"  Oracle states: {stats['melody_oracle']['num_states']}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    demo()
