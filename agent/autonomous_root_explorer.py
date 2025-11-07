#!/usr/bin/env python3
"""
Autonomous Root Explorer - Hybrid Intelligence for Harmonic Progression

This module implements autonomous harmonic exploration between user-defined waypoints.
Uses hybrid approach:
- 60% Training data (learned fundamentals from AudioOracle)
- 30% Live input response (what you're playing now)
- 10% Music theory (common interval bonus)

Based on Groven method for harmonic analysis (NOT Brandtsegg - that's for rhythm!)
"""

import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import deque


@dataclass
class RootWaypoint:
    """A waypoint in the harmonic progression timeline"""
    time: float  # Seconds from performance start
    root_hz: float  # Target fundamental frequency in Hz
    comment: str = ""  # Optional description


@dataclass
class ExplorationConfig:
    """Configuration for autonomous root exploration"""
    # Hybrid weights (must sum to ~1.0)
    training_weight: float = 0.6  # Prefer roots from training data
    input_response_weight: float = 0.3  # Respond to live input
    theory_bonus_weight: float = 0.1  # Slight boost for common intervals
    
    # Exploration constraints
    max_drift_semitones: float = 7.0  # Perfect fifth (can explore up to P5 from waypoint)
    return_probability: float = 0.3  # Chance to return to anchor (prevents endless drift)
    
    # Transition behavior
    transition_duration: float = 120.0  # Seconds to interpolate between waypoints (2 minutes)
    update_interval: float = 60.0  # How often to reconsider root (seconds)
    
    # Music theory intervals (semitones from root)
    # These get a small bonus if they appear in training data
    theory_intervals: List[int] = None
    
    def __post_init__(self):
        if self.theory_intervals is None:
            # Common intervals: Unison, m3, M3, P4, P5, M6, octave
            self.theory_intervals = [0, 3, 4, 5, 7, 9, 12]


@dataclass
class ExplorationDecision:
    """Record of an autonomous decision"""
    timestamp: float
    anchor_root: float  # Current waypoint root (Hz)
    chosen_root: float  # Root we chose to play (Hz)
    interval_semitones: float  # Distance from anchor
    input_fundamental: Optional[float]  # What user was playing (Hz)
    reason: str  # Why this root was chosen
    scores: Dict[str, float]  # Detailed scoring breakdown


class AutonomousRootExplorer:
    """
    Autonomous harmonic exploration engine
    
    Discovers interesting roots between user waypoints, informed by:
    1. Training data (what fundamentals appeared in learned AudioOracle states)
    2. Live input (what you're playing right now)
    3. Music theory (common intervals get small boost)
    
    Example usage:
        explorer = AutonomousRootExplorer(audio_oracle, waypoints, config)
        
        # Every 60 seconds during performance:
        new_root = explorer.update(elapsed_time=180.5, input_fundamental=220.0)
        # Returns: 264.0 Hz (C4) - discovered via training data
    """
    
    def __init__(self, 
                 audio_oracle,
                 waypoints: List[RootWaypoint],
                 config: ExplorationConfig = None):
        """
        Initialize autonomous root explorer
        
        Args:
            audio_oracle: Trained AudioOracle with fundamentals and consonances
            waypoints: List of RootWaypoint defining harmonic trajectory
            config: ExplorationConfig (uses defaults if None)
        """
        self.audio_oracle = audio_oracle
        self.waypoints = sorted(waypoints, key=lambda w: w.time)  # Ensure time-ordered
        self.config = config or ExplorationConfig()
        
        # Current state
        self.current_target_root = None  # Current root we're targeting (Hz)
        self.last_update_time = 0.0  # When we last chose a new root
        
        # Exploration history for transparency/debugging
        self.exploration_history = deque(maxlen=1000)
        
        # State frequency cache (how often each state appears in training)
        self._state_frequency_cache = {}
        self._build_state_frequency_cache()
        
        print(f"🎯 AutonomousRootExplorer initialized:")
        print(f"   Waypoints: {len(self.waypoints)}")
        print(f"   Weights: {self.config.training_weight:.0%} training, "
              f"{self.config.input_response_weight:.0%} input, "
              f"{self.config.theory_bonus_weight:.0%} theory")
        print(f"   Max drift: ±{self.config.max_drift_semitones} semitones")
        
    def _build_state_frequency_cache(self):
        """
        Build cache of how often each state appears in training
        (States that appear more often are more "important" to the learned style)
        """
        if not hasattr(self.audio_oracle, 'states'):
            return
        
        # Count transitions to each state
        state_counts = {}
        for (from_state, symbol), to_state in self.audio_oracle.transitions.items():
            state_counts[to_state] = state_counts.get(to_state, 0) + 1
        
        # Normalize to probabilities
        total = sum(state_counts.values())
        if total > 0:
            for state_id, count in state_counts.items():
                self._state_frequency_cache[state_id] = count / total
        
        print(f"   Cached {len(self._state_frequency_cache)} state frequencies")
    
    def update(self, 
               elapsed_time: float,
               input_fundamental: Optional[float] = None) -> float:
        """
        Update autonomous root exploration
        
        Called periodically during performance (e.g., every 60 seconds).
        Decides whether to explore new root or stay on current root.
        
        Args:
            elapsed_time: Seconds since performance start
            input_fundamental: Current fundamental from live input (Hz), or None
            
        Returns:
            Target root frequency in Hz
        """
        # Check if it's time to update
        if (self.current_target_root is not None and 
            elapsed_time - self.last_update_time < self.config.update_interval):
            return self.current_target_root
        
        # Get current and next waypoints
        current_waypoint = self._get_waypoint_at_time(elapsed_time)
        next_waypoint = self._get_next_waypoint(elapsed_time)
        
        if current_waypoint is None:
            # Before first waypoint - stay silent or use default
            return self.waypoints[0].root_hz if self.waypoints else 220.0
        
        # Calculate transition phase (0.0 = at current, 1.0 = at next)
        transition_phase = self._calculate_transition_phase(
            elapsed_time, current_waypoint, next_waypoint
        )
        
        # Decide: explore, transition, or stabilize?
        if transition_phase < 0.1:
            # Not transitioning yet - EXPLORE around current waypoint
            new_root = self._explore_harmonically(
                current_waypoint.root_hz,
                input_fundamental
            )
            reason = "exploration"
            
        elif transition_phase > 0.9:
            # Almost at next waypoint - STABILIZE
            new_root = next_waypoint.root_hz if next_waypoint else current_waypoint.root_hz
            reason = "stabilizing"
            
        else:
            # Mid-transition - INTERPOLATE between waypoints
            if next_waypoint:
                new_root = self._interpolate_roots(
                    current_waypoint.root_hz,
                    next_waypoint.root_hz,
                    transition_phase
                )
                reason = f"transition ({transition_phase:.0%})"
            else:
                new_root = current_waypoint.root_hz
                reason = "holding"
        
        # Update state
        self.current_target_root = new_root
        self.last_update_time = elapsed_time
        
        # Log decision
        self.exploration_history.append({
            'time': elapsed_time,
            'anchor': current_waypoint.root_hz,
            'chosen': new_root,
            'interval': 12 * np.log2(new_root / current_waypoint.root_hz),
            'input': input_fundamental,
            'reason': reason
        })
        
        return new_root
    
    def _get_waypoint_at_time(self, elapsed_time: float) -> Optional[RootWaypoint]:
        """Get the waypoint active at given time"""
        if not self.waypoints:
            return None
        
        # Find the last waypoint before or at current time
        active_waypoint = None
        for waypoint in self.waypoints:
            if waypoint.time <= elapsed_time:
                active_waypoint = waypoint
            else:
                break
        
        return active_waypoint
    
    def _get_next_waypoint(self, elapsed_time: float) -> Optional[RootWaypoint]:
        """Get the next waypoint after current time"""
        for waypoint in self.waypoints:
            if waypoint.time > elapsed_time:
                return waypoint
        return None
    
    def _calculate_transition_phase(self,
                                    elapsed_time: float,
                                    current_waypoint: RootWaypoint,
                                    next_waypoint: Optional[RootWaypoint]) -> float:
        """
        Calculate how far we are into transition (0.0 to 1.0)
        
        Returns:
            0.0 = at current waypoint (explore freely)
            0.5 = halfway to next waypoint (transitioning)
            1.0 = at next waypoint (stabilize)
        """
        if not next_waypoint:
            return 0.0  # No next waypoint, stay in exploration mode
        
        # Time until next waypoint
        time_to_next = next_waypoint.time - elapsed_time
        
        # If we're within transition_duration of next waypoint, start transitioning
        if time_to_next <= self.config.transition_duration:
            # Progress through transition (0.0 = just started, 1.0 = arrived)
            progress = 1.0 - (time_to_next / self.config.transition_duration)
            return np.clip(progress, 0.0, 1.0)
        else:
            return 0.0  # Not in transition yet
    
    def _interpolate_roots(self, 
                          root_a: float, 
                          root_b: float, 
                          phase: float) -> float:
        """
        Interpolate between two roots in log-frequency space
        (Perceptual spacing - equal intervals sound equal distances)
        
        Args:
            root_a: Starting root (Hz)
            root_b: Target root (Hz)
            phase: Transition progress 0.0-1.0
            
        Returns:
            Interpolated root frequency (Hz)
        """
        # Log-frequency interpolation (perceptually linear)
        log_a = np.log2(root_a)
        log_b = np.log2(root_b)
        log_interpolated = log_a + (log_b - log_a) * phase
        
        return 2 ** log_interpolated
    
    def _explore_harmonically(self,
                             anchor_root: float,
                             input_fundamental: Optional[float]) -> float:
        """
        CORE EXPLORATION LOGIC - Hybrid Intelligence
        
        Discover interesting roots near anchor, informed by:
        1. Training data (60%): What fundamentals appear in learned states?
        2. Live input (30%): What are you playing right now?
        3. Music theory (10%): Common intervals get small boost
        
        Args:
            anchor_root: Current waypoint root (Hz)
            input_fundamental: Live input fundamental (Hz) or None
            
        Returns:
            Chosen root frequency (Hz)
        """
        # Check if AudioOracle has harmonic data
        if not hasattr(self.audio_oracle, 'fundamentals') or not self.audio_oracle.fundamentals:
            # No training data - just return anchor
            return anchor_root
        
        # Find candidate states with fundamentals near anchor
        candidate_states = []
        for state_id, fundamental in self.audio_oracle.fundamentals.items():
            # Calculate semitone distance from anchor
            semitone_distance = 12 * np.log2(fundamental / anchor_root)
            
            # Only consider candidates within max_drift
            if abs(semitone_distance) <= self.config.max_drift_semitones:
                candidate_states.append({
                    'state_id': state_id,
                    'fundamental': fundamental,
                    'consonance': self.audio_oracle.consonances.get(state_id, 0.5),
                    'semitone_distance': semitone_distance
                })
        
        if not candidate_states:
            # No candidates found - stay at anchor
            return anchor_root
        
        # Score each candidate using HYBRID approach
        scores = []
        for candidate in candidate_states:
            score = 0.0
            score_breakdown = {}
            
            # 1. TRAINING DATA (60% weight)
            # Prefer roots that appear frequently in training
            state_freq = self._state_frequency_cache.get(candidate['state_id'], 0.001)
            training_score = state_freq
            score += training_score * self.config.training_weight
            score_breakdown['training'] = training_score * self.config.training_weight
            
            # 2. LIVE INPUT RESPONSE (30% weight)
            # If you're playing a note, bias toward that root
            if input_fundamental and input_fundamental > 0:
                input_distance = abs(
                    12 * np.log2(candidate['fundamental'] / input_fundamental)
                )
                # Exponential decay: closer = higher score
                input_match = np.exp(-input_distance / 3.0)
                score += input_match * self.config.input_response_weight
                score_breakdown['input'] = input_match * self.config.input_response_weight
            else:
                score_breakdown['input'] = 0.0
            
            # 3. CONSONANCE BONUS (helps but doesn't dominate)
            # Slightly prefer consonant states
            consonance_bonus = candidate['consonance'] * 0.1
            score += consonance_bonus
            score_breakdown['consonance'] = consonance_bonus
            
            # 4. RETURN TO ANCHOR BIAS (prevents endless drift)
            # Exponential decay toward anchor
            return_bias = np.exp(-abs(candidate['semitone_distance']) / 5.0)
            return_score = return_bias * self.config.return_probability
            score += return_score
            score_breakdown['return'] = return_score
            
            # 5. MUSIC THEORY BONUS (10% weight, ONLY if in common intervals)
            # Small boost if this fundamental is a common interval from anchor
            interval_semitones = round(candidate['semitone_distance'])
            if interval_semitones in self.config.theory_intervals:
                theory_score = self.config.theory_bonus_weight
                score += theory_score
                score_breakdown['theory'] = theory_score
            else:
                score_breakdown['theory'] = 0.0
            
            scores.append({
                'candidate': candidate,
                'total_score': score,
                'breakdown': score_breakdown
            })
        
        # Choose using weighted random (not always highest - adds variety)
        score_values = np.array([s['total_score'] for s in scores])
        
        # Softmax to convert scores to probabilities
        # Temperature controls randomness (lower = more deterministic)
        temperature = 0.5
        exp_scores = np.exp(score_values / temperature)
        probabilities = exp_scores / np.sum(exp_scores)
        
        # Weighted random choice
        chosen_idx = np.random.choice(len(scores), p=probabilities)
        chosen = scores[chosen_idx]
        
        # Log decision for transparency
        decision = ExplorationDecision(
            timestamp=time.time(),
            anchor_root=anchor_root,
            chosen_root=chosen['candidate']['fundamental'],
            interval_semitones=chosen['candidate']['semitone_distance'],
            input_fundamental=input_fundamental,
            reason=self._format_decision_reason(chosen),
            scores=chosen['breakdown']
        )
        
        return chosen['candidate']['fundamental']
    
    def _format_decision_reason(self, chosen: Dict) -> str:
        """Format human-readable decision reason"""
        breakdown = chosen['breakdown']
        reasons = []
        
        # Find dominant factor
        max_component = max(breakdown.items(), key=lambda x: x[1])
        
        if max_component[0] == 'training':
            reasons.append(f"frequent in training ({breakdown['training']:.2f})")
        if max_component[0] == 'input':
            reasons.append(f"matches input ({breakdown['input']:.2f})")
        if breakdown.get('theory', 0) > 0:
            reasons.append("common interval")
        if breakdown.get('consonance', 0) > 0.05:
            reasons.append(f"consonant ({breakdown['consonance']:.2f})")
        
        return " + ".join(reasons) if reasons else "exploration"
    
    def get_exploration_summary(self) -> Dict[str, Any]:
        """
        Get summary of exploration behavior (for debugging/transparency)
        
        Returns:
            Dictionary with exploration statistics
        """
        if not self.exploration_history:
            return {}
        
        history = list(self.exploration_history)
        
        intervals = [h['interval'] for h in history]
        
        return {
            'total_decisions': len(history),
            'mean_interval': np.mean(intervals) if intervals else 0,
            'max_interval': np.max(np.abs(intervals)) if intervals else 0,
            'input_responses': sum(1 for h in history if h['input'] is not None),
            'explorations': sum(1 for h in history if 'exploration' in h['reason']),
            'transitions': sum(1 for h in history if 'transition' in h['reason']),
            'recent_roots': [h['chosen'] for h in history[-10:]]
        }
