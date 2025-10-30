#!/usr/bin/env python3
"""
Phrase Memory System
Tracks played phrases and enables thematic development

This module remembers what MusicHal has played and develops themes through
variation techniques: transposition, inversion, retrograde, augmentation, diminution.

@author: Jonas Sjøvaag
@date: 2025-10-24
"""

import time
import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque


@dataclass
class Motif:
    """A musical motif (recurring pattern)"""
    id: str
    notes: List[int]  # MIDI notes
    durations: List[float]  # Note durations
    intervals: List[int]  # Melodic intervals (derived)
    rhythm_pattern: List[float]  # Rhythm ratios (derived)
    first_occurrence: float  # Timestamp
    occurrence_count: int = 1
    last_used: float = field(default_factory=time.time)


class PhraseMemory:
    """
    Remembers phrases played and enables thematic development
    
    Provides:
    - Phrase history tracking
    - Motif extraction from repeated patterns
    - Thematic variations (transpose, invert, retrograde, augment, diminish)
    - Decision logic for when to recall vs. generate new material
    """
    
    def __init__(self, max_history: int = 20, visualization_manager=None):
        """
        Initialize phrase memory
        
        Args:
            max_history: Maximum number of phrases to remember
            visualization_manager: Optional visualization manager for event emission
        """
        self.phrase_history = deque(maxlen=max_history)
        self.motifs = {}  # Dict[motif_id, Motif]
        self.current_theme = None  # Active thematic material
        self.last_recall_time = 0.0
        self.recall_interval_min = 30.0  # seconds - minimum time between recalls
        self.recall_interval_max = 60.0  # seconds - maximum time between recalls
        self.next_recall_time = time.time() + random.uniform(30.0, 60.0)
        self.visualization_manager = visualization_manager
        
        # Variation preferences
        self.variation_weights = {
            'transpose': 0.4,   # Most common
            'invert': 0.2,      # Interesting
            'retrograde': 0.1,  # Rare
            'augment': 0.15,    # Lengthen
            'diminish': 0.15    # Shorten
        }
    
    def add_phrase(self, notes: List[int], durations: List[float], 
                  timestamp: float = None, mode: str = None):
        """
        Store a phrase and extract potential motifs
        
        Args:
            notes: MIDI notes in the phrase
            durations: Duration for each note
            timestamp: When phrase was played (default: now)
            mode: Behavior mode that generated this phrase
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Store in history
        phrase_data = {
            'notes': notes,
            'durations': durations,
            'timestamp': timestamp,
            'mode': mode,
            'intervals': self._compute_intervals(notes),
            'rhythm_pattern': self._compute_rhythm_pattern(durations)
        }
        self.phrase_history.append(phrase_data)
        
        # Extract motifs (3-5 note patterns)
        if len(notes) >= 3:
            self._extract_motifs(phrase_data)
        
        # Emit visualization event for storing phrase
        if self.visualization_manager and len(notes) >= 3:
            self.visualization_manager.emit_phrase_memory(
                action='store',
                motif=notes[:5],  # First 5 notes as representative motif
                timestamp=timestamp
            )
    
    def _compute_intervals(self, notes: List[int]) -> List[int]:
        """Compute melodic intervals from notes"""
        if len(notes) < 2:
            return []
        return [notes[i+1] - notes[i] for i in range(len(notes)-1)]
    
    def _compute_rhythm_pattern(self, durations: List[float]) -> List[float]:
        """Compute rhythm ratios from durations"""
        if len(durations) < 2:
            return durations
        
        # Normalize to ratios relative to shortest duration
        min_dur = min(durations)
        if min_dur == 0:
            return durations
        
        return [d / min_dur for d in durations]
    
    def _extract_motifs(self, phrase_data: Dict):
        """Extract 3-5 note motifs from phrase"""
        notes = phrase_data['notes']
        durations = phrase_data['durations']
        intervals = phrase_data['intervals']
        
        # Try different motif lengths
        for motif_length in range(3, min(6, len(notes)+1)):
            for start_idx in range(len(notes) - motif_length + 1):
                motif_notes = notes[start_idx:start_idx + motif_length]
                motif_durations = durations[start_idx:start_idx + motif_length]
                motif_intervals = intervals[start_idx:start_idx + motif_length-1] if len(intervals) >= motif_length-1 else []
                
                # Create motif signature (based on intervals to be transposition-invariant)
                motif_signature = tuple(motif_intervals)
                
                if motif_signature in self.motifs:
                    # Motif exists - increment count
                    self.motifs[motif_signature].occurrence_count += 1
                    self.motifs[motif_signature].last_used = phrase_data['timestamp']
                else:
                    # New motif
                    motif_id = f"motif_{len(self.motifs)+1}"
                    self.motifs[motif_signature] = Motif(
                        id=motif_id,
                        notes=motif_notes,
                        durations=motif_durations,
                        intervals=list(motif_intervals),
                        rhythm_pattern=self._compute_rhythm_pattern(motif_durations),
                        first_occurrence=phrase_data['timestamp']
                    )
    
    def should_recall_theme(self) -> bool:
        """
        Decide if it's time to return to established material
        
        Returns:
            True if we should recall a theme, False if we should generate new
        """
        current_time = time.time()
        
        # Check if enough time has passed
        if current_time < self.next_recall_time:
            return False
        
        # Check if we have any motifs to recall
        if not self.motifs:
            return False
        
        # Update next recall time
        self.last_recall_time = current_time
        self.next_recall_time = current_time + random.uniform(
            self.recall_interval_min, 
            self.recall_interval_max
        )
        
        return True
    
    def get_current_theme(self) -> Optional[Motif]:
        """
        Get the active theme using weighted random selection
        
        Returns:
            Motif object or None
        """
        if not self.motifs:
            return None
        
        # Use weighted random selection instead of always picking most frequent
        # This adds variety while still favoring more common motifs
        motif_list = list(self.motifs.values())
        
        # Weight by occurrence count with diminishing returns (sqrt dampens dominance)
        import math
        weights = [math.sqrt(m.occurrence_count) for m in motif_list]
        
        # Also consider recency - boost motifs used recently
        current_time = time.time()
        for i, motif in enumerate(motif_list):
            time_since_use = current_time - motif.last_used
            if time_since_use < 60.0:  # Used in last minute
                weights[i] *= 1.5
            elif time_since_use > 300.0:  # Not used in 5 minutes
                weights[i] *= 0.5
        
        # Random weighted selection
        import random
        selected_motif = random.choices(motif_list, weights=weights)[0]
        self.current_theme = selected_motif
        
        # Emit visualization event for recalling theme
        if self.visualization_manager:
            self.visualization_manager.emit_phrase_memory(
                action='recall',
                motif=selected_motif.notes[:5],  # First 5 notes
                timestamp=time.time()
            )
            # Also emit timeline event
            self.visualization_manager.emit_timeline_update('thematic_recall', timestamp=time.time())
        
        return selected_motif
    
    def get_variation(self, motif: Motif, variation_type: str = None) -> Dict:
        """
        Generate variation of a motif
        
        Args:
            motif: Motif to vary
            variation_type: Type of variation (or None for random weighted selection)
                Options: 'transpose', 'invert', 'retrograde', 'augment', 'diminish'
        
        Returns:
            Dict with 'notes', 'durations', 'variation_type'
        """
        if variation_type is None:
            # Random weighted selection
            variation_type = random.choices(
                list(self.variation_weights.keys()),
                weights=list(self.variation_weights.values())
            )[0]
        
        # Apply variation
        if variation_type == 'transpose':
            varied_notes, varied_durations = self._transpose(motif)
        elif variation_type == 'invert':
            varied_notes, varied_durations = self._invert(motif)
        elif variation_type == 'retrograde':
            varied_notes, varied_durations = self._retrograde(motif)
        elif variation_type == 'augment':
            varied_notes, varied_durations = self._augment(motif)
        elif variation_type == 'diminish':
            varied_notes, varied_durations = self._diminish(motif)
        else:
            # Unknown type - return original
            varied_notes, varied_durations = motif.notes[:], motif.durations[:]
        
        # Emit visualization event for variation
        if self.visualization_manager:
            self.visualization_manager.emit_phrase_memory(
                action='variation',
                motif=varied_notes[:5],  # First 5 notes of variation
                variation_type=variation_type,
                timestamp=time.time()
            )
        
        return {
            'notes': varied_notes,
            'durations': varied_durations,
            'variation_type': variation_type,
            'source_motif_id': motif.id
        }
    
    def _transpose(self, motif: Motif) -> Tuple[List[int], List[float]]:
        """Transpose motif by random interval (-7 to +7 semitones)"""
        interval = random.randint(-7, 7)
        transposed_notes = [max(24, min(96, n + interval)) for n in motif.notes]
        return transposed_notes, motif.durations[:]
    
    def _invert(self, motif: Motif) -> Tuple[List[int], List[float]]:
        """Invert intervals around first note"""
        if len(motif.notes) < 2:
            return motif.notes[:], motif.durations[:]
        
        first_note = motif.notes[0]
        inverted_notes = [first_note]
        
        for interval in motif.intervals:
            inverted_notes.append(inverted_notes[-1] - interval)  # Invert direction
        
        # Clamp to valid MIDI range
        inverted_notes = [max(24, min(96, n)) for n in inverted_notes]
        
        return inverted_notes, motif.durations[:]
    
    def _retrograde(self, motif: Motif) -> Tuple[List[int], List[float]]:
        """Reverse the note sequence"""
        return list(reversed(motif.notes)), list(reversed(motif.durations))
    
    def _augment(self, motif: Motif) -> Tuple[List[int], List[float]]:
        """Lengthen durations by 1.5x"""
        augmented_durations = [d * 1.5 for d in motif.durations]
        return motif.notes[:], augmented_durations
    
    def _diminish(self, motif: Motif) -> Tuple[List[int], List[float]]:
        """Shorten durations by 0.67x"""
        diminished_durations = [d * 0.67 for d in motif.durations]
        return motif.notes[:], diminished_durations
    
    def get_stats(self) -> Dict:
        """Get statistics about phrase memory"""
        return {
            'phrases_remembered': len(self.phrase_history),
            'motifs_extracted': len(self.motifs),
            'current_theme': self.current_theme.id if self.current_theme else None,
            'next_recall_in': max(0, self.next_recall_time - time.time())
        }


def test_phrase_memory():
    """Test phrase memory system"""
    print("🧪 Testing PhraseMemory...")
    
    memory = PhraseMemory(max_history=10)
    
    # Test 1: Add phrases
    print("\n--- Test 1: Add Phrases ---")
    phrase1 = [60, 62, 64, 65]  # C D E F
    durations1 = [0.5, 0.5, 0.5, 1.0]
    memory.add_phrase(phrase1, durations1, mode='MIRROR')
    
    phrase2 = [67, 69, 71, 72]  # G A B C
    durations2 = [0.5, 0.5, 0.5, 1.0]
    memory.add_phrase(phrase2, durations2, mode='SHADOW')
    
    # Add similar phrase (same intervals)
    phrase3 = [65, 67, 69, 70]  # F G A Bb
    durations3 = [0.5, 0.5, 0.5, 1.0]
    memory.add_phrase(phrase3, durations3, mode='MIRROR')
    
    print(f"   Stats: {memory.get_stats()}")
    
    # Test 2: Extract and recall theme
    print("\n--- Test 2: Get Current Theme ---")
    theme = memory.get_current_theme()
    if theme:
        print(f"   Theme: {theme.id}, notes={theme.notes}, occurrences={theme.occurrence_count}")
    
    # Test 3: Generate variations
    if theme:
        print("\n--- Test 3: Generate Variations ---")
        for var_type in ['transpose', 'invert', 'retrograde', 'augment', 'diminish']:
            variation = memory.get_variation(theme, var_type)
            print(f"   {var_type}: {variation['notes']}")
    
    print("\n✅ PhraseMemory tests complete!")


if __name__ == "__main__":
    test_phrase_memory()

