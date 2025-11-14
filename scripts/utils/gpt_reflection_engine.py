#!/usr/bin/env python3
"""
GPT-OSS Live Reflection Engine
Provides non-blocking, periodic reflections on musical interaction during live performance.
"""

import asyncio
import threading
import time
from typing import Dict, List, Optional, Callable
from collections import Counter
import numpy as np

from gpt_oss_client import GPTOSSClient


class AsyncGPTReflector:
    """
    Non-blocking GPT-OSS reflection engine for live performance.
    
    Architecture:
    - Runs in separate thread with its own asyncio event loop
    - Accepts reflection requests from main thread (non-blocking)
    - Calls GPT-OSS API asynchronously
    - Invokes callback with results on completion
    
    Musical Context:
    - Analyzes last 60 seconds of interaction
    - Describes harmonic direction, behavioral modes, emerging patterns
    - Provides transparency into AI's "understanding" of improvisation
    """
    
    def __init__(self, interval: float = 60.0):
        """
        Initialize async reflector.
        
        Args:
            interval: Time between reflections (seconds)
        """
        self.interval = interval
        self.loop = None
        self.thread = None
        self.is_running = False
        
        # Current reflection state
        self.current_reflection = "Listening..."
        self.last_reflection_time = 0.0
        self._reflection_lock = threading.Lock()
        
        # Callback for reflection completion
        self.callback: Optional[Callable] = None
        
        # GPT client (initialized in start())
        self.gpt_client: Optional[GPTOSSClient] = None
        
    def start(self):
        """Start the reflection engine (background thread + event loop)"""
        if self.is_running:
            return
        
        print("🤖 Starting GPT reflection engine...")
        
        # Initialize GPT client
        try:
            from gpt_oss_client import GPTOSSClient
            self.gpt_client = GPTOSSClient()
            print("✅ GPT-OSS client initialized")
        except Exception as e:
            print(f"❌ Failed to initialize GPT-OSS client: {e}")
            print("⚠️  Reflections will be disabled")
            return
        
        self.is_running = True
        
        # Start background thread with event loop (daemon = don't block shutdown)
        self.thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self.thread.start()
        
        # Wait for loop to be ready
        while self.loop is None:
            time.sleep(0.01)
        
        print("✅ GPT reflection engine started")
        
    def stop(self):
        """Stop the reflection engine"""
        if not self.is_running:
            return
        
        print("🤖 Stopping GPT reflection engine...")
        self.is_running = False
        
        # Cancel all pending tasks and stop event loop
        if self.loop and self.loop.is_running():
            # Schedule cleanup in the loop
            self.loop.call_soon_threadsafe(self._shutdown_loop)
        
        # Wait for thread to finish (with shorter timeout since it's daemon)
        if self.thread:
            self.thread.join(timeout=1.0)
            if self.thread.is_alive():
                print("⚠️  Reflection thread still running (daemon will exit with program)")
        
        print("✅ GPT reflection engine stopped")
        
    def _run_event_loop(self):
        """Run asyncio event loop in background thread"""
        # Create new event loop for this thread
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        # Run until stopped
        try:
            self.loop.run_forever()
        finally:
            self.loop.close()
    
    def _shutdown_loop(self):
        """Cancel all pending tasks and stop the event loop (call from within loop)"""
        # Get all running tasks
        pending = asyncio.all_tasks(self.loop)
        
        # Cancel them
        for task in pending:
            task.cancel()
        
        # Create a task to wait for all cancellations to complete
        async def wait_for_cancellations():
            # Gather all tasks (will raise CancelledError, but that's ok)
            await asyncio.gather(*pending, return_exceptions=True)
        
        # Schedule the wait and then stop
        if pending:
            self.loop.create_task(wait_for_cancellations()).add_done_callback(
                lambda _: self.loop.stop()
            )
        else:
            # No pending tasks, stop immediately
            self.loop.stop()
            
    def set_callback(self, callback_fn: Callable[[str, float], None]):
        """
        Set callback for reflection completion.
        
        Args:
            callback_fn: Function(reflection_text, timestamp) called when reflection ready
        """
        self.callback = callback_fn
        
    def request_reflection(self, reflection_data: Dict):
        """
        Request a reflection (non-blocking).
        
        Args:
            reflection_data: Dict containing:
                - memory_buffer: Recent musical events
                - decision_log: Recent AI decisions
                - current_mode: Current behavioral mode
                - performance_time: Elapsed performance time
        """
        if not self.is_running or not self.loop:
            return
        
        # Submit coroutine to background event loop (non-blocking)
        asyncio.run_coroutine_threadsafe(
            self._reflect_async(reflection_data),
            self.loop
        )
        
    async def _reflect_async(self, data: Dict):
        """
        Perform async reflection (runs in background loop).
        
        Args:
            data: Reflection data dict from request_reflection()
        """
        try:
            # Extract and summarize musical data
            summary = self._summarize_recent_playing(data)
            
            # Build prompt
            prompt = self._build_prompt(summary)
            
            # Call GPT-OSS API (async, with timeout)
            reflection = await asyncio.wait_for(
                self._call_gpt_async(prompt),
                timeout=10.0
            )
            
            # Update state (thread-safe)
            timestamp = time.time()
            with self._reflection_lock:
                self.current_reflection = reflection
                self.last_reflection_time = timestamp
            
            # Invoke callback if set
            if self.callback:
                self.callback(reflection, timestamp)
                
        except asyncio.TimeoutError:
            reflection = "⚠️ Reflection timed out (>10s). Will retry next interval."
            print(reflection)
            if self.callback:
                self.callback(reflection, time.time())
                
        except Exception as e:
            reflection = f"❌ Reflection error: {str(e)}"
            print(f"⚠️  GPT reflection failed: {e}")
            if self.callback:
                self.callback(reflection, time.time())
    
    async def _call_gpt_async(self, prompt: str) -> str:
        """
        Call GPT-OSS API asynchronously.
        
        Args:
            prompt: GPT prompt text
            
        Returns:
            Reflection text from GPT
        """
        # Use asyncio.to_thread to wrap synchronous GPT client
        # (Later can be replaced with aiohttp for true async)
        loop = asyncio.get_event_loop()
        reflection = await loop.run_in_executor(
            None,  # Use default executor
            self.gpt_client.analyze,
            prompt
        )
        return reflection
        
    def _summarize_recent_playing(self, data: Dict) -> Dict:
        """
        Summarize recent musical interaction for GPT prompt.
        
        Args:
            data: Reflection data containing memory_buffer, decision_log, etc.
            
        Returns:
            Summary dict with extracted features
        """
        memory_buffer = data.get('memory_buffer')
        decision_log = data.get('decision_log', [])
        current_mode = data.get('current_mode', 'unknown')
        performance_time = data.get('performance_time', 0)
        
        # Default summary if no data
        if not memory_buffer:
            return {
                'duration': 60.0,
                'gesture_tokens': [],
                'chord_progression': [],
                'consonance_mean': 0.5,
                'consonance_trend': 'stable',
                'mode_distribution': [],
                'ai_activity': 0.0,
                'error': 'No memory buffer available'
            }
        
        try:
            # Get recent events (last 60 seconds)
            # memory_buffer.get_recent_moments() returns list of MusicalMoment objects
            recent_events = memory_buffer.get_recent_moments(duration_seconds=60.0)
            
            if not recent_events:
                return {
                    'duration': 60.0,
                    'gesture_tokens': [],
                    'chord_progression': [],
                    'consonance_mean': 0.5,
                    'consonance_trend': 'stable',
                    'mode_distribution': [],
                    'ai_activity': 0.0,
                    'note': 'No recent events in memory'
                }
            
            # Extract gesture tokens (smoothed)
            gesture_tokens = [
                moment.event_data.get('gesture_token') 
                for moment in recent_events 
                if moment.event_data.get('gesture_token') is not None
            ]
            
            # Extract chord labels
            chord_progression = [
                moment.event_data.get('chord_label', 'Unknown')
                for moment in recent_events 
                if moment.event_data.get('chord_label')
            ]
            
            # Extract consonance values
            consonance_values = [
                moment.event_data.get('consonance', 0.5)
                for moment in recent_events
                if moment.event_data.get('consonance') is not None
            ]
            
            # Compute consonance statistics
            consonance_mean = np.mean(consonance_values) if consonance_values else 0.5
            
            # Compute consonance trend (rising/falling/stable)
            consonance_trend = 'stable'
            if len(consonance_values) > 3:
                # Linear regression slope
                x = np.arange(len(consonance_values))
                slope = np.polyfit(x, consonance_values, 1)[0]
                if slope > 0.01:
                    consonance_trend = 'increasing'
                elif slope < -0.01:
                    consonance_trend = 'decreasing'
            
            # Extract behavioral modes from decision log
            modes = [d.get('mode', 'unknown') for d in decision_log if 'mode' in d]
            mode_distribution = Counter(modes).most_common(3) if modes else []
            
            # Compute AI activity (how often AI made decisions)
            ai_activity = len(decision_log) / max(1, len(recent_events)) if decision_log else 0.0
            
            # Find most common gesture tokens
            token_distribution = Counter(gesture_tokens).most_common(5) if gesture_tokens else []
            
            # Find harmonic center (most common chord root)
            chord_roots = [chord.split('_')[0] if '_' in chord else chord for chord in chord_progression]
            harmonic_center = Counter(chord_roots).most_common(1)[0][0] if chord_roots else 'Unknown'
            
            return {
                'duration': 60.0,
                'num_events': len(recent_events),
                'gesture_tokens': gesture_tokens,
                'token_distribution': token_distribution,
                'chord_progression': chord_progression[:15],  # First 15 chords
                'harmonic_center': harmonic_center,
                'consonance_mean': consonance_mean,
                'consonance_trend': consonance_trend,
                'mode_distribution': mode_distribution,
                'current_mode': current_mode,
                'ai_activity': ai_activity,
                'performance_time': performance_time
            }
            
        except Exception as e:
            print(f"⚠️  Error summarizing musical data: {e}")
            return {
                'duration': 60.0,
                'error': str(e),
                'gesture_tokens': [],
                'chord_progression': [],
                'consonance_mean': 0.5,
                'consonance_trend': 'stable',
                'mode_distribution': [],
                'ai_activity': 0.0
            }
    
    def _build_prompt(self, summary: Dict) -> str:
        """
        Build GPT prompt from musical summary.
        
        Args:
            summary: Musical summary dict from _summarize_recent_playing()
            
        Returns:
            GPT prompt string
        """
        # Check for errors
        if 'error' in summary:
            return f"""
You are analyzing a live musical improvisation between human and AI.
Unfortunately, there was an error gathering recent musical data: {summary['error']}

Respond with a brief acknowledgment of the technical issue.
"""
        
        # Format mode distribution (handle empty or malformed data)
        try:
            mode_str = ', '.join([f"{mode} ({count}x)" for mode, count in summary['mode_distribution']]) \
                       if summary.get('mode_distribution') else 'No mode data'
        except (TypeError, ValueError, KeyError):
            mode_str = 'No mode data'
        
        # Format token distribution (handle empty or malformed data)
        try:
            token_str = ', '.join([f"token {token} ({count}x)" for token, count in summary['token_distribution'][:3]]) \
                        if summary.get('token_distribution') else 'No gesture data'
        except (TypeError, ValueError, KeyError):
            token_str = 'No gesture data'
        
        # Build prompt with safe access to summary fields
        num_events = summary.get('num_events', 0)
        chord_prog = summary.get('chord_progression', [])
        harmonic_center = summary.get('harmonic_center', 'Unknown')
        consonance_mean = summary.get('consonance_mean', 0.0)
        consonance_trend = summary.get('consonance_trend', 'stable')
        current_mode = summary.get('current_mode', 'unknown')
        ai_activity = summary.get('ai_activity', 0.0)
        performance_time = summary.get('performance_time', 0.0)
        
        prompt = f"""
You are analyzing a live musical improvisation between human and AI.

Last 60 seconds of interaction ({num_events} events):
- Chord progression: {', '.join(chord_prog[:10]) if chord_prog else 'No chords detected'}
- Harmonic center: {harmonic_center}
- Average consonance: {consonance_mean:.2f} ({consonance_trend} trend)
- AI behavioral modes: {mode_str}
- Current mode: {current_mode}
- Most common gesture tokens: {token_str}
- AI activity level: {ai_activity:.0%}
- Performance elapsed: {performance_time:.0f} seconds

In 2-3 sentences, describe the musical relationship currently unfolding.
Focus on: interaction dynamics (who's leading?), harmonic direction, emerging patterns.
Be specific about consonance/dissonance levels, behavioral mode characteristics, and gestural coherence.
Speak as if you're watching the performance live - describe what you're witnessing in the present tense.
"""
        
        return prompt
    
    def get_current_reflection(self) -> str:
        """
        Get the current reflection text (thread-safe).
        
        Returns:
            Most recent reflection string
        """
        with self._reflection_lock:
            return self.current_reflection


# Example usage
if __name__ == "__main__":
    print("Testing AsyncGPTReflector...")
    
    reflector = AsyncGPTReflector(interval=10.0)
    
    def on_reflection_ready(reflection: str, timestamp: float):
        print(f"\n🤖 Reflection received at {timestamp:.1f}:")
        print(f"   {reflection}\n")
    
    reflector.set_callback(on_reflection_ready)
    reflector.start()
    
    # Simulate reflection request
    print("Requesting reflection...")
    test_data = {
        'memory_buffer': None,  # Mock empty buffer
        'decision_log': [],
        'current_mode': 'shadow',
        'performance_time': 120.0
    }
    
    reflector.request_reflection(test_data)
    
    # Wait for result
    time.sleep(5)
    
    reflector.stop()
    print("Test complete!")
