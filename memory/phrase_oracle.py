#!/usr/bin/env python3
"""
Phrase-Level Oracle
===================

Stores and matches complete musical phrases instead of individual frames.
This addresses the granularity mismatch between human musical thinking
(in phrases) and frame-level AudioOracle storage.

Architecture:
- Stores phrases as sequences of tokens with averaged embeddings
- Matches by phrase similarity (not token continuation)
- Returns complete phrase responses

This complements the frame-level AudioOracle by providing
phrase-aware pattern matching for more musical responses.
"""

import time
import json
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class StoredPhrase:
    """A stored musical phrase"""
    phrase_id: str
    tokens: List[int]           # Gesture tokens in sequence
    embedding: np.ndarray       # Averaged 768D embedding
    duration: float             # Total duration in seconds
    note_count: int             # Number of notes
    avg_consonance: float       # Average consonance
    style: str                  # Detected style (from CLAP)
    timestamp: float            # When it was stored
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'phrase_id': self.phrase_id,
            'tokens': self.tokens,
            'embedding': self.embedding.tolist(),
            'duration': self.duration,
            'note_count': self.note_count,
            'avg_consonance': self.avg_consonance,
            'style': self.style,
            'timestamp': self.timestamp,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'StoredPhrase':
        """Create from dictionary"""
        return cls(
            phrase_id=data['phrase_id'],
            tokens=data['tokens'],
            embedding=np.array(data['embedding']),
            duration=data['duration'],
            note_count=data['note_count'],
            avg_consonance=data['avg_consonance'],
            style=data.get('style', 'unknown'),
            timestamp=data['timestamp'],
            metadata=data.get('metadata', {})
        )


class PhraseOracle:
    """
    Phrase-level pattern memory for musical phrase matching.

    Unlike frame-level AudioOracle which stores individual events,
    PhraseOracle stores complete phrases and matches by phrase similarity.
    This provides more musically coherent responses.
    """

    def __init__(self,
                 max_phrases: int = 1000,
                 similarity_threshold: float = 0.6,
                 embedding_dim: int = 768):
        """
        Initialize phrase oracle.

        Args:
            max_phrases: Maximum phrases to store
            similarity_threshold: Minimum similarity for matches
            embedding_dim: Dimension of phrase embeddings
        """
        self.max_phrases = max_phrases
        self.similarity_threshold = similarity_threshold
        self.embedding_dim = embedding_dim

        # Phrase storage
        self.phrases: List[StoredPhrase] = []

        # Index by style for faster lookup
        self.style_index: Dict[str, List[int]] = defaultdict(list)

        # Index by token sequence for exact/near matches
        self.token_index: Dict[tuple, List[int]] = defaultdict(list)

        # Statistics
        self.total_stored = 0
        self.total_queries = 0
        self.total_matches = 0

    def add_phrase(self,
                   tokens: List[int],
                   embeddings: List[np.ndarray],
                   duration: float,
                   consonance_values: List[float] = None,
                   style: str = 'unknown',
                   metadata: Dict = None) -> str:
        """
        Add a phrase to the oracle.

        Args:
            tokens: List of gesture tokens in the phrase
            embeddings: List of 768D embeddings for each token
            duration: Total phrase duration in seconds
            consonance_values: Consonance for each note (optional)
            style: Detected style
            metadata: Additional metadata

        Returns:
            Phrase ID
        """
        if len(tokens) == 0 or len(embeddings) == 0:
            return None

        # Generate phrase ID
        phrase_id = f"phrase_{self.total_stored}_{int(time.time() * 1000)}"

        # Average embeddings to get phrase embedding
        embeddings_array = np.array([e.astype(np.float64) for e in embeddings])
        phrase_embedding = np.mean(embeddings_array, axis=0)

        # Normalize embedding
        norm = np.linalg.norm(phrase_embedding)
        if norm > 0:
            phrase_embedding = phrase_embedding / norm

        # Calculate average consonance
        avg_consonance = np.mean(consonance_values) if consonance_values else 0.5

        # Create stored phrase
        phrase = StoredPhrase(
            phrase_id=phrase_id,
            tokens=tokens,
            embedding=phrase_embedding,
            duration=duration,
            note_count=len(tokens),
            avg_consonance=avg_consonance,
            style=style,
            timestamp=time.time(),
            metadata=metadata or {}
        )

        # Check capacity
        if len(self.phrases) >= self.max_phrases:
            self._evict_oldest()

        # Store phrase
        phrase_idx = len(self.phrases)
        self.phrases.append(phrase)

        # Update indices
        self.style_index[style].append(phrase_idx)
        token_key = tuple(tokens[:5])  # Index by first 5 tokens
        self.token_index[token_key].append(phrase_idx)

        self.total_stored += 1

        return phrase_id

    def _evict_oldest(self):
        """Remove oldest phrase to make room"""
        if not self.phrases:
            return

        # Find oldest
        oldest_idx = min(range(len(self.phrases)),
                        key=lambda i: self.phrases[i].timestamp)
        oldest = self.phrases[oldest_idx]

        # Remove from indices
        if oldest.style in self.style_index:
            self.style_index[oldest.style] = [
                i for i in self.style_index[oldest.style] if i != oldest_idx
            ]
        token_key = tuple(oldest.tokens[:5])
        if token_key in self.token_index:
            self.token_index[token_key] = [
                i for i in self.token_index[token_key] if i != oldest_idx
            ]

        # Remove phrase
        self.phrases.pop(oldest_idx)

        # Rebuild indices (indices shifted)
        self._rebuild_indices()

    def _rebuild_indices(self):
        """Rebuild indices after eviction"""
        self.style_index = defaultdict(list)
        self.token_index = defaultdict(list)

        for idx, phrase in enumerate(self.phrases):
            self.style_index[phrase.style].append(idx)
            token_key = tuple(phrase.tokens[:5])
            self.token_index[token_key].append(idx)

    def find_similar_phrases(self,
                             query_embedding: np.ndarray,
                             query_tokens: List[int] = None,
                             style: str = None,
                             top_k: int = 5) -> List[Tuple[StoredPhrase, float]]:
        """
        Find phrases similar to query.

        Args:
            query_embedding: Query phrase embedding (768D)
            query_tokens: Query tokens (for bonus matching)
            style: Preferred style (optional filter)
            top_k: Number of results to return

        Returns:
            List of (phrase, similarity) tuples, sorted by similarity
        """
        self.total_queries += 1

        if len(self.phrases) == 0:
            return []

        # Normalize query embedding
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_normalized = query_embedding / norm
        else:
            query_normalized = query_embedding

        # Calculate similarities
        similarities = []

        for idx, phrase in enumerate(self.phrases):
            # Cosine similarity of embeddings
            embedding_sim = np.dot(query_normalized, phrase.embedding)

            # Bonus for token sequence match
            token_bonus = 0.0
            if query_tokens:
                # Check overlap
                overlap = len(set(query_tokens) & set(phrase.tokens))
                token_bonus = overlap / max(len(query_tokens), len(phrase.tokens)) * 0.2

            # Style bonus
            style_bonus = 0.1 if style and phrase.style == style else 0.0

            # Total similarity
            total_sim = embedding_sim + token_bonus + style_bonus

            if total_sim >= self.similarity_threshold:
                similarities.append((phrase, total_sim))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Track matches
        if similarities:
            self.total_matches += 1

        return similarities[:top_k]

    def get_response_phrase(self,
                            query_embedding: np.ndarray,
                            query_tokens: List[int] = None,
                            style: str = None,
                            behavior_mode: str = 'MIRROR') -> Optional[StoredPhrase]:
        """
        Get a phrase to use as response to query.

        Args:
            query_embedding: Query phrase embedding
            query_tokens: Query tokens
            style: Style filter
            behavior_mode: SHADOW (similar), MIRROR (complement), COUPLE (independent)

        Returns:
            Selected response phrase or None
        """
        matches = self.find_similar_phrases(
            query_embedding, query_tokens, style, top_k=10
        )

        if not matches:
            return None

        # Select based on behavior mode
        mode_upper = behavior_mode.upper()

        if mode_upper == 'SHADOW':
            # Most similar phrase
            return matches[0][0]

        elif mode_upper == 'MIRROR':
            # Complementary - pick from middle similarity range
            if len(matches) >= 3:
                idx = len(matches) // 2
                return matches[idx][0]
            return matches[-1][0]

        elif mode_upper == 'COUPLE':
            # More independent - pick from lower similarity
            if len(matches) >= 2:
                return matches[-1][0]
            return matches[0][0]

        else:
            # Default: return best match
            return matches[0][0]

    def get_statistics(self) -> Dict:
        """Get oracle statistics"""
        return {
            'total_phrases': len(self.phrases),
            'max_phrases': self.max_phrases,
            'total_stored': self.total_stored,
            'total_queries': self.total_queries,
            'total_matches': self.total_matches,
            'match_rate': self.total_matches / max(1, self.total_queries),
            'styles': list(self.style_index.keys()),
            'avg_phrase_length': np.mean([p.note_count for p in self.phrases]) if self.phrases else 0
        }

    def save(self, filepath: str):
        """Save oracle to file"""
        data = {
            'max_phrases': self.max_phrases,
            'similarity_threshold': self.similarity_threshold,
            'embedding_dim': self.embedding_dim,
            'phrases': [p.to_dict() for p in self.phrases],
            'statistics': {
                'total_stored': self.total_stored,
                'total_queries': self.total_queries,
                'total_matches': self.total_matches
            }
        }

        with open(filepath, 'w') as f:
            json.dump(data, f)

        print(f"💾 PhraseOracle saved: {filepath} ({len(self.phrases)} phrases)")

    def load(self, filepath: str):
        """Load oracle from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        self.max_phrases = data.get('max_phrases', 1000)
        self.similarity_threshold = data.get('similarity_threshold', 0.6)
        self.embedding_dim = data.get('embedding_dim', 768)

        self.phrases = [StoredPhrase.from_dict(p) for p in data.get('phrases', [])]

        stats = data.get('statistics', {})
        self.total_stored = stats.get('total_stored', 0)
        self.total_queries = stats.get('total_queries', 0)
        self.total_matches = stats.get('total_matches', 0)

        # Rebuild indices
        self._rebuild_indices()

        print(f"📂 PhraseOracle loaded: {filepath} ({len(self.phrases)} phrases)")

    def clear(self):
        """Clear all stored phrases"""
        self.phrases = []
        self.style_index = defaultdict(list)
        self.token_index = defaultdict(list)
        self.total_stored = 0
        self.total_queries = 0
        self.total_matches = 0


def demo():
    """Demonstrate phrase oracle"""
    print("=" * 70)
    print("Phrase Oracle - Demo")
    print("=" * 70)

    oracle = PhraseOracle(max_phrases=100)

    # Add some test phrases
    print("\n📝 Adding test phrases...")

    for i in range(10):
        tokens = [i % 64, (i + 1) % 64, (i + 2) % 64, (i + 3) % 64]
        embeddings = [np.random.randn(768) for _ in tokens]
        style = ['jazz', 'blues', 'rock'][i % 3]

        phrase_id = oracle.add_phrase(
            tokens=tokens,
            embeddings=embeddings,
            duration=2.0 + i * 0.5,
            style=style
        )
        print(f"  Added: {phrase_id} ({style})")

    # Query
    print("\n🔍 Querying for similar phrases...")

    query_embedding = np.random.randn(768)
    matches = oracle.find_similar_phrases(query_embedding, top_k=3)

    print(f"  Found {len(matches)} matches:")
    for phrase, sim in matches:
        print(f"    {phrase.phrase_id}: similarity={sim:.3f}, style={phrase.style}")

    # Statistics
    stats = oracle.get_statistics()
    print(f"\n📊 Statistics:")
    print(f"  Total phrases: {stats['total_phrases']}")
    print(f"  Match rate: {stats['match_rate']:.2%}")
    print(f"  Styles: {stats['styles']}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    demo()
