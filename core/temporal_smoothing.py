#!/usr/bin/env python3
"""
Temporal Smoothing for Event Quality Enhancement

Addresses event over-sampling during sustained notes by:
1. Time-window averaging of features
2. Chord label stabilization
3. Onset-aware event grouping

This prevents AudioOracle from learning noise and decay artifacts
instead of meaningful musical patterns.

@author: Jonas Sjøvaag
@date: 2025-10-23
"""

import numpy as np
from typing import List, Dict, Optional


class TemporalSmoother:
    """
    Smooth events by averaging features within time windows
    
    Reduces noise from:
    - Overtone variations during sustained notes
    - Room noise and background artifacts
    - Decay envelope micro-variations
    
    Ensures AudioOracle learns musical events (note attacks, chord changes)
    rather than noise and decay artifacts.
    """
    
    def __init__(self, window_size: float = 0.5, min_change_threshold: float = 0.1):
        """
        Initialize temporal smoother
        
        Args:
            window_size: Window duration in seconds (0.5 = max 2 events/sec)
            min_change_threshold: Minimum normalized feature change to create new event
        """
        self.window_size = window_size
        self.min_change_threshold = min_change_threshold
    
    def smooth_events(self, events: List[Dict]) -> List[Dict]:
        """
        Group events into time windows and average features
        
        Creates new event when:
        1. Time window exceeded (window_size seconds), OR
        2. Onset detected, OR
        3. Significant feature change detected
        
        Args:
            events: List of event dicts sorted by timestamp
        
        Returns:
            Smoothed events (one per window or significant change)
        """
        if not events:
            return []
        
        smoothed = []
        current_window = []
        window_start = events[0].get('t', 0)
        last_features = None
        
        for event in events:
            event_time = event.get('t', 0)
            
            # Check if significant change occurred
            is_onset = event.get('onset', False)
            has_significant_change = False
            
            if last_features is not None:
                current_features = self._extract_key_features(event)
                feature_change = np.linalg.norm(current_features - last_features)
                has_significant_change = feature_change > self.min_change_threshold
            
            # FIXED: Create new window if:
            # 1. Window time exceeded, OR
            # 2. STRONG onset detected (onset + significant feature change), OR
            # 3. Very significant feature change (strong attack without onset flag)
            
            # Only respect onset if it's also a significant change (true attack, not just flag noise)
            is_strong_onset = is_onset and (has_significant_change or last_features is None)
            
            # Very significant change threshold (2x normal)
            is_strong_change = has_significant_change and feature_change > (self.min_change_threshold * 2.0) if last_features is not None else False
            
            if (event_time >= window_start + self.window_size or 
                is_strong_onset or 
                is_strong_change):
                
                # Average and output previous window
                if current_window:
                    smoothed_event = self._average_window(current_window)
                    smoothed.append(smoothed_event)
                    last_features = self._extract_key_features(smoothed_event)
                
                # Start new window
                current_window = [event]
                window_start = event_time
            else:
                # Add to current window
                current_window.append(event)
        
        # Handle last window
        if current_window:
            smoothed_event = self._average_window(current_window)
            smoothed.append(smoothed_event)
        
        print(f"   📊 Smoothing reduced {len(events)} events to {len(smoothed)} ({(1-len(smoothed)/len(events))*100:.1f}% reduction)")
        
        return smoothed
    
    def _extract_key_features(self, event: Dict) -> np.ndarray:
        """
        Extract key features for change detection
        
        Uses normalized features to detect meaningful musical changes,
        not just noise/decay variations.
        """
        features = [
            event.get('rms_db', -60.0) / 60.0,  # Normalize to ~[-1, 0]
            event.get('centroid', 1000.0) / 5000.0,  # Normalize to ~[0, 1]
            event.get('f0', 440.0) / 880.0,  # Normalize to ~[0, 1]
        ]
        
        # Add gesture token if available (machine-level feature)
        if 'gesture_token' in event and event['gesture_token'] is not None:
            features.append(event['gesture_token'] / 64.0)  # Normalize to [0, 1]
        
        # Add consonance if available (psychoacoustic feature)
        if 'consonance' in event:
            features.append(event['consonance'])  # Already [0, 1]
        
        return np.array(features)
    
    def _average_window(self, events: List[Dict]) -> Dict:
        """
        Average events within window
        
        Takes mean of numerical features, mode of categorical features
        """
        avg_event = events[0].copy()  # Start with first event structure
        
        # Average numerical features
        numeric_keys = [
            'rms_db', 'centroid', 'rolloff', 'bandwidth', 
            'f0', 'midi', 'cents', 'consonance', 'zcr',
            'attack_time', 'release_time', 'ioi',
            'mfcc_1', 'mfcc_2', 'mfcc_3'
        ]
        
        for key in numeric_keys:
            values = [e.get(key, 0) for e in events if key in e]
            if values:
                avg_event[key] = float(np.mean(values))
        
        # Take most recent timestamp
        avg_event['t'] = events[-1].get('t', 0)
        
        # Take most common chord label (mode, not mean)
        if 'chord' in events[0]:
            chords = [e.get('chord', 'C') for e in events]
            avg_event['chord'] = max(set(chords), key=chords.count)
        
        # Average Wav2Vec features if present
        if 'wav2vec_features' in events[0]:
            try:
                wav2vec_stack = np.array([e['wav2vec_features'] for e in events])
                avg_event['wav2vec_features'] = np.mean(wav2vec_stack, axis=0).tolist()
            except:
                # Keep first if averaging fails
                avg_event['wav2vec_features'] = events[0].get('wav2vec_features')
        
        # Take most common gesture token (mode)
        if 'gesture_token' in events[0]:
            tokens = [e.get('gesture_token') for e in events if e.get('gesture_token') is not None]
            if tokens:
                avg_event['gesture_token'] = max(set(tokens), key=tokens.count)
        
        # Keep onset flag if ANY event in window was onset
        avg_event['onset'] = any(e.get('onset', False) for e in events)
        
        # Mark as smoothed
        avg_event['smoothed'] = True
        avg_event['window_size'] = len(events)
        
        # Preserve ratio analysis fields (take most recent or average)
        if 'rhythm_ratio' in events[-1]:
            avg_event['rhythm_ratio'] = events[-1]['rhythm_ratio']
        if 'deviation_polarity' in events[-1]:
            avg_event['deviation_polarity'] = events[-1]['deviation_polarity']
        if 'tempo_factor' in events[-1]:
            avg_event['tempo_factor'] = events[-1]['tempo_factor']
        
        # Preserve relative parameters (average)
        if 'midi_relative' in events[0]:
            midi_rels = [e.get('midi_relative', 0) for e in events]
            avg_event['midi_relative'] = int(np.mean(midi_rels))
        if 'velocity_relative' in events[0]:
            vel_rels = [e.get('velocity_relative', 0) for e in events]
            avg_event['velocity_relative'] = int(np.mean(vel_rels))
        if 'ioi_relative' in events[0]:
            ioi_rels = [e.get('ioi_relative', 1.0) for e in events]
            avg_event['ioi_relative'] = float(np.mean(ioi_rels))
        
        return avg_event
    
    def smooth_chord_labels(self, events: List[Dict], 
                           min_chord_duration: float = 2.0,
                           confidence_weight: float = 0.3) -> List[Dict]:
        """
        Apply temporal smoothing specifically to chord labels
        
        Prevents rapid chord changes from overtone/noise jitter.
        A chord must be held for min_chord_duration before changing.
        
        Args:
            events: List of events with 'chord' labels
            min_chord_duration: Minimum time a chord must be held before changing (seconds)
            confidence_weight: How much to weight confidence in chord change decisions
        
        Returns:
            Events with stabilized chord labels
        """
        if not events:
            return []
        
        # Store original chords for comparison
        original_chords = [e.get('chord', 'C') for e in events]
        
        current_chord = None
        chord_start_time = 0
        chord_confidence_sum = 0.0
        chord_event_count = 0
        
        for event in events:
            detected_chord = event.get('chord', 'C')
            chord_confidence = event.get('chord_confidence', 0.5)
            event_time = event.get('t', 0)
            
            # Initialize first chord
            if current_chord is None:
                current_chord = detected_chord
                chord_start_time = event_time
                chord_confidence_sum = chord_confidence
                chord_event_count = 1
                continue
            
            # Check if detected chord differs
            if detected_chord != current_chord:
                time_since_change = event_time - chord_start_time
                
                # FIXED: More lenient change logic
                # Accept change if minimum duration has passed
                should_change = time_since_change >= min_chord_duration
                
                if should_change:
                    # Accept chord change
                    current_chord = detected_chord
                    chord_start_time = event_time
                    chord_confidence_sum = chord_confidence
                    chord_event_count = 1
                else:
                    # Reject change - keep current chord (too soon)
                    event['chord'] = current_chord
                    chord_confidence_sum += chord_confidence
                    chord_event_count += 1
            else:
                # Same chord continues
                chord_confidence_sum += chord_confidence
                chord_event_count += 1
        
        # Calculate how much cleaning was done (FIXED: compare original vs smoothed)
        smoothed_chords = [e.get('chord', 'C') for e in events]
        original_unique = len(set(original_chords))
        smoothed_unique = len(set(smoothed_chords))
        
        if original_unique != smoothed_unique:
            print(f"   📊 Chord stabilization: {original_unique} → {smoothed_unique} unique chords")
        
        return events


def test_temporal_smoothing():
    """Test temporal smoothing functionality"""
    print("🔄 Testing TemporalSmoother...")
    
    smoother = TemporalSmoother(window_size=1.0, min_change_threshold=0.1)
    
    # Test case 1: Sustained chord with noise
    print("\n📍 Test 1: Sustained chord (20 events over 5 seconds)")
    test_events = []
    for i in range(20):
        event = {
            't': i * 0.25,
            'rms_db': -25.0 + np.random.randn() * 2.0,  # Slight noise
            'centroid': 1000.0 + np.random.randn() * 50.0,
            'f0': 440.0 + np.random.randn() * 5.0,
            'midi': 69,
            'chord': 'C' if i < 15 else 'Dm',  # Chord change at 3.75s
            'onset': (i % 5 == 0),  # Onset every 1.25s
            'consonance': 0.8 + np.random.randn() * 0.05
        }
        test_events.append(event)
    
    smoothed = smoother.smooth_events(test_events)
    print(f"   Original events: {len(test_events)}")
    print(f"   Smoothed events: {len(smoothed)}")
    print(f"   Reduction: {(1 - len(smoothed)/len(test_events))*100:.1f}%")
    assert len(smoothed) < len(test_events), "Should reduce event count"
    print("   ✅ Smoothing reduces events")
    
    # Test case 2: Chord label stabilization
    print("\n📍 Test 2: Chord label jitter")
    jittery_events = []
    chord_sequence = ['C', 'C', 'Cdim', 'C', 'C', 'F#7', 'C', 'C', 'D9', 'C',
                      'Dm', 'Dm', 'Dm', 'Dm', 'Dm']
    for i, chord in enumerate(chord_sequence):
        jittery_events.append({
            't': i * 0.5,
            'chord': chord,
            'chord_confidence': 0.7 if chord in ['C', 'Dm'] else 0.3
        })
    
    stabilized = smoother.smooth_chord_labels(jittery_events, min_chord_duration=2.0)
    
    # Extract chord sequence
    original_seq = [e['chord'] for e in jittery_events]
    stabilized_seq = [e['chord'] for e in stabilized]
    
    print(f"   Original:   {original_seq}")
    print(f"   Stabilized: {stabilized_seq}")
    
    # First 10 should all be 'C' (jitter removed)
    assert all(c == 'C' for c in stabilized_seq[:10]), "First 10 should be stabilized to 'C'"
    print("   ✅ Chord jitter removed")
    
    print("\n✅ TemporalSmoother tests passed!")


if __name__ == "__main__":
    test_temporal_smoothing()

