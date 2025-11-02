#!/usr/bin/env python3
"""
Gesture token temporal smoothing for musically coherent token matching.

Problem: Rapid onset detection creates gesture token flicker (4+ tokens/second).
Solution: Accumulate tokens over 2-4s window, use consensus/most-common token.

Musical reasoning: Chord changes happen ~1/second, phrases unfold over 2-4s.
Individual onsets are part of unified gestural phrases, not independent events.
"""

from collections import Counter, deque
from typing import Optional, List, Tuple
import time


class GestureTokenSmoother:
    """
    Temporal smoothing for gesture tokens to match musical phrase timing.
    
    Maintains sliding window of recent tokens, returns consensus token
    representing current gestural context (not instant onset-level token).
    """
    
    def __init__(self, 
                 window_duration: float = 3.0,  # 3-second phrase window
                 min_tokens: int = 3,           # Need 3+ tokens before consensus
                 decay_time: float = 1.0):      # Tokens older than 1s have less weight
        """
        Args:
            window_duration: Time window for token accumulation (seconds)
            min_tokens: Minimum tokens needed before returning consensus
            decay_time: Older tokens weighted less (exponential decay)
        """
        self.window_duration = window_duration
        self.min_tokens = min_tokens
        self.decay_time = decay_time
        
        # Sliding window: [(timestamp, token), ...]
        self.token_history: deque = deque(maxlen=100)  # Cap at 100 tokens
        
        self.last_consensus_token: Optional[int] = None
        self.last_update_time: float = 0.0
        
        # Statistics
        self.total_tokens_processed = 0
        self.consensus_changes = 0
    
    def add_token(self, token: int, timestamp: Optional[float] = None) -> int:
        """
        Add new gesture token and return smoothed consensus token.
        
        Args:
            token: Raw gesture token from Wav2Vec quantization
            timestamp: Event timestamp (or current time if None)
        
        Returns:
            Consensus gesture token representing current gestural phrase
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Add to history
        self.token_history.append((timestamp, token))
        self.last_update_time = timestamp
        self.total_tokens_processed += 1
        
        # Clean old tokens outside window
        cutoff_time = timestamp - self.window_duration
        while self.token_history and self.token_history[0][0] < cutoff_time:
            self.token_history.popleft()
        
        # Calculate consensus token
        consensus = self._calculate_consensus(timestamp)
        
        if consensus is not None:
            # Track consensus changes for transparency
            if self.last_consensus_token is not None and consensus != self.last_consensus_token:
                self.consensus_changes += 1
            
            self.last_consensus_token = consensus
        
        return self.last_consensus_token if self.last_consensus_token is not None else token
    
    def _calculate_consensus(self, current_time: float) -> Optional[int]:
        """
        Calculate consensus token using weighted voting.
        
        Recent tokens weighted higher (exponential decay over decay_time).
        Returns most common token, weighted by recency.
        """
        if len(self.token_history) < self.min_tokens:
            return None  # Not enough data yet
        
        # Weighted voting
        weighted_votes: Counter = Counter()
        
        for timestamp, token in self.token_history:
            age = current_time - timestamp
            # Exponential decay: weight = 2^(-age / decay_time)
            weight = 2 ** (-age / self.decay_time)
            weighted_votes[token] += weight
        
        # Return token with highest weighted count
        if weighted_votes:
            consensus_token, _ = weighted_votes.most_common(1)[0]
            return consensus_token
        
        return None
    
    def get_current_consensus(self) -> Optional[int]:
        """Get current consensus token without adding new data."""
        return self.last_consensus_token
    
    def get_token_distribution(self) -> List[Tuple[int, int]]:
        """
        Get current token distribution for debugging/transparency.
        
        Returns:
            List of (token, count) tuples, sorted by count descending
        """
        tokens = [token for _, token in self.token_history]
        return Counter(tokens).most_common()
    
    def get_weighted_distribution(self, current_time: Optional[float] = None) -> List[Tuple[int, float]]:
        """
        Get weighted token distribution (accounting for decay).
        
        Returns:
            List of (token, weighted_count) tuples, sorted by weight descending
        """
        if current_time is None:
            current_time = time.time()
        
        weighted_votes: Counter = Counter()
        
        for timestamp, token in self.token_history:
            age = current_time - timestamp
            weight = 2 ** (-age / self.decay_time)
            weighted_votes[token] += weight
        
        return weighted_votes.most_common()
    
    def get_statistics(self) -> dict:
        """Get smoother statistics for debugging."""
        return {
            'window_duration': self.window_duration,
            'tokens_in_window': len(self.token_history),
            'total_processed': self.total_tokens_processed,
            'consensus_changes': self.consensus_changes,
            'current_consensus': self.last_consensus_token,
            'token_distribution': self.get_token_distribution()[:5]  # Top 5
        }
    
    def reset(self):
        """Clear token history (e.g., on behavioral mode change)."""
        self.token_history.clear()
        self.last_consensus_token = None
        # Don't reset counters - keep for session statistics
    
    def __len__(self):
        """Return number of tokens in current window."""
        return len(self.token_history)
    
    def __repr__(self):
        """String representation for debugging."""
        return (f"GestureTokenSmoother(window={self.window_duration}s, "
                f"tokens={len(self.token_history)}, "
                f"consensus={self.last_consensus_token})")
