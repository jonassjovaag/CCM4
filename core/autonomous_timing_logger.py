#!/usr/bin/env python3
"""
Autonomous Timing Logger - Track generation attempts and blocking reasons

Logs every generation decision to identify why phrase generation is sparse.
Tracks: attempts, successes, blocking reasons, gaps, phrase progress.
"""

import time
import csv
from pathlib import Path
from collections import defaultdict, Counter
from typing import Optional, Dict, List


class AutonomousTimingLogger:
    """
    Comprehensive logging for autonomous generation timing and blocking analysis
    
    Tracks every generation attempt with:
    - Timestamp and voice (melodic/bass)
    - Decision (allowed/blocked)
    - Blocking reason (phrase_complete, interval_too_soon, ready, etc.)
    - Phrase progress (notes_in_phrase/target)
    - Gap since last attempt
    
    Exports to CSV for correlation with MIDI output logs.
    """
    
    def __init__(self, log_dir: str = "logs", verbose: bool = False):
        """
        Initialize timing logger
        
        Args:
            log_dir: Directory for log files
            verbose: If True, print every log entry to console
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.verbose = verbose
        
        # Generate timestamped filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.csv_path = self.log_dir / f"timing_events_{timestamp}.csv"
        
        # Event storage
        self.events: List[Dict] = []
        
        # Per-voice tracking
        self.last_attempt_time = {'melodic': 0.0, 'bass': 0.0}
        self.last_success_time = {'melodic': 0.0, 'bass': 0.0}
        
        # Statistics
        self.attempt_count = {'melodic': 0, 'bass': 0}
        self.success_count = {'melodic': 0, 'bass': 0}
        self.blocking_reasons = defaultdict(lambda: defaultdict(int))  # voice -> reason -> count
        
        # Initialize CSV file
        self._init_csv()
        
        if self.verbose:
            print(f"🔍 AutonomousTimingLogger initialized: {self.csv_path}")
    
    def _init_csv(self):
        """Initialize CSV file with headers"""
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'timestamp',
                'voice',
                'decision',
                'blocking_reason',
                'notes_in_phrase',
                'target_phrase_length',
                'phrase_complete',
                'gap_since_last_attempt',
                'gap_since_last_success'
            ])
            writer.writeheader()
    
    def log_attempt(self, 
                   voice: str, 
                   allowed: bool, 
                   reason: str,
                   notes_in_phrase: int = 0,
                   target_phrase_length: int = 0,
                   phrase_complete: bool = False):
        """
        Log a generation attempt
        
        Args:
            voice: 'melodic' or 'bass'
            allowed: True if generation allowed, False if blocked
            reason: Blocking reason string (phrase_complete, interval_too_soon, ready, etc.)
            notes_in_phrase: Current note count in phrase
            target_phrase_length: Target notes for phrase completion
            phrase_complete: Whether phrase is marked complete
        """
        now = time.time()
        
        # Calculate gaps
        gap_attempt = now - self.last_attempt_time[voice] if self.last_attempt_time[voice] > 0 else 0.0
        gap_success = now - self.last_success_time[voice] if self.last_success_time[voice] > 0 else 0.0
        
        event = {
            'timestamp': now,
            'voice': voice,
            'decision': 'allowed' if allowed else 'blocked',
            'blocking_reason': reason,
            'notes_in_phrase': notes_in_phrase,
            'target_phrase_length': target_phrase_length,
            'phrase_complete': phrase_complete,
            'gap_since_last_attempt': gap_attempt,
            'gap_since_last_success': gap_success
        }
        
        self.events.append(event)
        self.last_attempt_time[voice] = now
        self.attempt_count[voice] += 1
        
        if not allowed:
            self.blocking_reasons[voice][reason] += 1
        
        # Write immediately to CSV (real-time logging)
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=event.keys())
            writer.writerow(event)
        
        if self.verbose:
            status = "✅" if allowed else "❌"
            print(f"{status} [{voice}] {reason} | {notes_in_phrase}/{target_phrase_length} | gap: {gap_attempt:.2f}s")
    
    def log_success(self, 
                   voice: str, 
                   note: int,
                   notes_in_phrase: int,
                   target_phrase_length: int):
        """
        Log successful note generation
        
        Args:
            voice: 'melodic' or 'bass'
            note: MIDI note number
            notes_in_phrase: Current note count
            target_phrase_length: Target for completion
        """
        now = time.time()
        self.last_success_time[voice] = now
        self.success_count[voice] += 1
        
        if self.verbose:
            progress = f"{notes_in_phrase}/{target_phrase_length}"
            print(f"🎵 [{voice}] Generated note {note} | Progress: {progress}")
    
    def log_phrase_complete(self, voice: str, total_notes: int, duration: float):
        """
        Log phrase completion event
        
        Args:
            voice: 'melodic' or 'bass'
            total_notes: Total notes in completed phrase
            duration: Phrase duration in seconds
        """
        if self.verbose:
            rate = total_notes / duration if duration > 0 else 0
            print(f"🎼 [{voice}] Phrase complete: {total_notes} notes in {duration:.2f}s ({rate:.1f} notes/sec)")
    
    def log_auto_reset(self, voice: str):
        """
        Log auto-reset event (phrase_complete cleared after 2s pause)
        
        Args:
            voice: 'melodic' or 'bass'
        """
        if self.verbose:
            print(f"🔄 [{voice}] Auto-reset: phrase_complete cleared, ready to generate")
    
    def calculate_gaps(self, voice: Optional[str] = None) -> Dict:
        """
        Calculate gap statistics from logged events
        
        Args:
            voice: Specific voice to analyze, or None for both
        
        Returns:
            Dictionary with min/max/avg gaps and histogram
        """
        if voice:
            events = [e for e in self.events if e['voice'] == voice]
        else:
            events = self.events
        
        gaps = [e['gap_since_last_success'] for e in events if e['gap_since_last_success'] > 0]
        
        if not gaps:
            return {'min': 0, 'max': 0, 'avg': 0, 'count': 0, 'histogram': {}}
        
        # Histogram bins
        histogram = {
            '0-1s': sum(1 for g in gaps if 0 <= g < 1),
            '1-2s': sum(1 for g in gaps if 1 <= g < 2),
            '2-5s': sum(1 for g in gaps if 2 <= g < 5),
            '5-10s': sum(1 for g in gaps if 5 <= g < 10),
            '>10s': sum(1 for g in gaps if g >= 10)
        }
        
        return {
            'min': min(gaps),
            'max': max(gaps),
            'avg': sum(gaps) / len(gaps),
            'count': len(gaps),
            'histogram': histogram
        }
    
    def get_blocking_summary(self) -> Dict:
        """
        Get summary of blocking reasons per voice
        
        Returns:
            Dictionary: voice -> reason -> count
        """
        return dict(self.blocking_reasons)
    
    def print_summary(self):
        """Print comprehensive summary of logged events"""
        print("\n" + "=" * 80)
        print("AUTONOMOUS TIMING LOGGER - SUMMARY")
        print("=" * 80)
        
        for voice in ['melodic', 'bass']:
            print(f"\n{voice.upper()} VOICE:")
            print(f"  Attempts: {self.attempt_count[voice]}")
            print(f"  Successes: {self.success_count[voice]}")
            
            if self.attempt_count[voice] > 0:
                success_rate = (self.success_count[voice] / self.attempt_count[voice]) * 100
                print(f"  Success rate: {success_rate:.1f}%")
            
            # Blocking reasons
            if voice in self.blocking_reasons:
                print(f"\n  Blocking reasons:")
                for reason, count in sorted(self.blocking_reasons[voice].items(), 
                                          key=lambda x: x[1], reverse=True):
                    print(f"    {reason}: {count}x")
            
            # Gap statistics
            gaps = self.calculate_gaps(voice)
            if gaps['count'] > 0:
                print(f"\n  Gap statistics:")
                print(f"    Min: {gaps['min']:.2f}s")
                print(f"    Max: {gaps['max']:.2f}s")
                print(f"    Avg: {gaps['avg']:.2f}s")
                print(f"\n  Gap histogram:")
                for bin_name, count in gaps['histogram'].items():
                    pct = (count / gaps['count']) * 100 if gaps['count'] > 0 else 0
                    print(f"    {bin_name}: {count} ({pct:.1f}%)")
        
        print(f"\n{'=' * 80}")
        print(f"Log saved to: {self.csv_path}")
        print(f"{'=' * 80}\n")
    
    def identify_primary_blocker(self) -> str:
        """
        Identify the most common blocking reason across all voices
        
        Returns:
            String describing primary blocker
        """
        all_reasons = Counter()
        for voice_reasons in self.blocking_reasons.values():
            all_reasons.update(voice_reasons)
        
        if not all_reasons:
            return "No blocking detected"
        
        primary_reason, count = all_reasons.most_common(1)[0]
        total_blocks = sum(all_reasons.values())
        pct = (count / total_blocks) * 100
        
        return f"{primary_reason} ({count}/{total_blocks} blocks, {pct:.1f}%)"
