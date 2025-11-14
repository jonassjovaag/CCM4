#!/usr/bin/env python3
"""
Performance Simulator
Simulates live performance using training data instead of actual audio I/O
"""

import os
import json
import time
import numpy as np
import random
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import threading

from performance_timeline_manager import PerformanceTimelineManager, PerformanceConfig
from memory.memory_buffer import MemoryBuffer
from memory.polyphonic_audio_oracle_mps import PolyphonicAudioOracleMPS
from agent.ai_agent import AIAgent
from mapping.feature_mapper import FeatureMapper
from midi_io.midi_output import MIDIOutput

@dataclass
class SimulatedEvent:
    """Simulated audio event based on training data"""
    timestamp: float
    f0: float
    rms_db: float
    centroid: float
    rolloff: float
    zcr: float
    hnr: float
    onset: bool
    instrument: str
    mfcc: List[float]
    attack_time: float
    decay_time: float
    spectral_flux: float

class PerformanceSimulator:
    """Simulates live performance using training data"""
    
    def __init__(self, performance_duration: int, training_data_path: str = None):
        self.performance_duration = performance_duration
        self.training_data_path = training_data_path or "ai_learning_data/musical_memory.json"
        
        # Core components
        self.memory_buffer = MemoryBuffer()
        self.clustering = PolyphonicAudioOracleMPS(
            distance_threshold=0.15,
            distance_function='euclidean',
            feature_dimensions=15,
            adaptive_threshold=True,
            chord_similarity_weight=0.3,
            use_mps=True
        )
        self.ai_agent = AIAgent()
        self.feature_mapper = FeatureMapper()
        
        # Performance timeline
        self.timeline_manager: Optional[PerformanceTimelineManager] = None
        
        # Simulation data
        self.training_events: List[Dict] = []
        self.simulation_running = False
        self.simulation_thread: Optional[threading.Thread] = None
        
        # Statistics
        self.stats = {
            'events_simulated': 0,
            'decisions_made': 0,
            'notes_sent': 0,
            'start_time': 0
        }
        
        # Load training data and initialize timeline
        self._load_training_data()
        self._initialize_timeline()
    
    def _load_training_data(self):
        """Load training data from musical memory"""
        print(f"📂 Loading training data from: {self.training_data_path}")
        
        if not os.path.exists(self.training_data_path):
            print(f"⚠️  Training data not found: {self.training_data_path}")
            return
        
        try:
            with open(self.training_data_path, 'r') as f:
                data = json.load(f)
            
            if 'moments' in data:
                self.training_events = data['moments']
            else:
                self.training_events = data
            
            print(f"✅ Loaded {len(self.training_events)} training events")
            
        except Exception as e:
            print(f"❌ Error loading training data: {e}")
            self.training_events = []
    
    def _initialize_timeline(self):
        """Initialize performance timeline manager"""
        config = PerformanceConfig(
            duration_minutes=self.performance_duration,
            arc_file_path="ai_learning_data/curious_child_performance_arc.json",
            engagement_profile="balanced",
            silence_tolerance=5.0,
            adaptation_rate=0.1
        )
        
        try:
            self.timeline_manager = PerformanceTimelineManager(config)
            print(f"🎵 Performance timeline initialized: {self.performance_duration} minutes")
        except Exception as e:
            print(f"⚠️  Warning: Could not initialize performance timeline: {e}")
            self.timeline_manager = None
    
    def _create_simulated_event(self, base_event: Dict, time_offset: float) -> SimulatedEvent:
        """Create a simulated event based on training data with some variation"""
        
        # Add some realistic variation to the base event
        f0_variation = random.uniform(0.95, 1.05)
        rms_variation = random.uniform(-2.0, 2.0)
        centroid_variation = random.uniform(0.9, 1.1)
        
        # Simulate instrument classification with some randomness
        instruments = ['piano', 'drums', 'bass']
        instrument = random.choice(instruments) if random.random() > 0.3 else 'unknown'
        
        return SimulatedEvent(
            timestamp=time.time() + time_offset,
            f0=base_event.get('f0', 440.0) * f0_variation,
            rms_db=base_event.get('rms_db', -20.0) + rms_variation,
            centroid=base_event.get('centroid', 2000.0) * centroid_variation,
            rolloff=base_event.get('rolloff', 4000.0) * random.uniform(0.9, 1.1),
            zcr=base_event.get('zcr', 0.1) * random.uniform(0.8, 1.2),
            hnr=base_event.get('hnr', 0.5) * random.uniform(0.9, 1.1),
            onset=random.random() > 0.7,  # 30% chance of onset
            instrument=instrument,
            mfcc=base_event.get('mfcc', [0.0] * 13),
            attack_time=base_event.get('attack_time', 0.02) * random.uniform(0.8, 1.2),
            decay_time=base_event.get('decay_time', 0.001) * random.uniform(0.8, 1.2),
            spectral_flux=base_event.get('spectral_flux', 100.0) * random.uniform(0.8, 1.2)
        )
    
    def _simulate_audio_events(self):
        """Simulate audio events based on training data"""
        print("🎵 Starting performance simulation...")
        
        if not self.training_events:
            print("❌ No training events available for simulation")
            return
        
        # Start timeline
        if self.timeline_manager:
            self.timeline_manager.start_performance()
        
        self.stats['start_time'] = time.time()
        event_interval = 0.1  # 100ms between events
        total_events = int(self.performance_duration * 60 / event_interval)
        
        print(f"📊 Simulating {total_events} events over {self.performance_duration} minutes")
        
        for i in range(total_events):
            if not self.simulation_running:
                break
            
            # Select a random training event as base
            base_event = random.choice(self.training_events)
            
            # Create simulated event
            simulated_event = self._create_simulated_event(base_event, i * event_interval)
            
            # Process the event
            self._process_simulated_event(simulated_event)
            
            # Update timeline
            if self.timeline_manager:
                human_activity = simulated_event.rms_db > -60
                self.timeline_manager.update_performance_state(
                    human_activity=human_activity,
                    instrument_detected=simulated_event.instrument
                )
            
            # Sleep for event interval
            time.sleep(event_interval)
            
            # Print status every 100 events
            if i % 100 == 0 and i > 0:
                self._print_simulation_status()
        
        print("✅ Performance simulation completed")
    
    def _process_simulated_event(self, event: SimulatedEvent):
        """Process a simulated audio event"""
        # Convert to event data format
        event_data = {
            't': event.timestamp,
            'f0': event.f0,
            'rms_db': event.rms_db,
            'centroid': event.centroid,
            'rolloff': event.rolloff,
            'zcr': event.zcr,
            'hnr': event.hnr,
            'onset': event.onset,
            'instrument': event.instrument,
            'mfcc': event.mfcc,
            'attack_time': event.attack_time,
            'decay_time': event.decay_time,
            'spectral_flux': event.spectral_flux
        }
        
        # Add to memory buffer
        self.memory_buffer.add_moment(event_data)
        
        # Learn patterns
        self.clustering.add_sequence([event_data])
        
        # Update statistics
        self.stats['events_simulated'] += 1
        
        # Check timeline guidance
        should_respond = True
        if self.timeline_manager:
            guidance = self.timeline_manager.get_performance_guidance()
            should_respond = guidance['should_respond']
        
        if should_respond:
            # Process with AI agent
            decisions = self.ai_agent.process_event(
                event_data, self.memory_buffer, self.clustering
            )
            
            midi_params = None
            if decisions:
                self.stats['decisions_made'] += len(decisions)
                
                # Simulate MIDI output (just log, don't actually send)
                for decision in decisions:
                    midi_params = self.feature_mapper.map_features_to_midi(
                        event_data, {
                            'mode': decision.mode.value,
                            'confidence': decision.confidence,
                            'musical_params': decision.musical_params
                        },
                        decision.voice_type
                    )
                    
        if midi_params:
            self.stats['notes_sent'] += 1
            print(f"🎵 Simulated MIDI: Note {midi_params.note} "
                  f"(vel={midi_params.velocity}, "
                  f"dur={midi_params.duration:.2f}s) "
                  f"[{event.instrument}]")
    
    def _print_simulation_status(self):
        """Print simulation status"""
        elapsed = time.time() - self.stats['start_time']
        progress = (elapsed / (self.performance_duration * 60)) * 100
        
        print(f"\n📊 Simulation Status:")
        print(f"   Progress: {progress:.1f}% ({elapsed:.1f}s / {self.performance_duration * 60:.1f}s)")
        print(f"   Events: {self.stats['events_simulated']}")
        print(f"   Decisions: {self.stats['decisions_made']}")
        print(f"   Notes: {self.stats['notes_sent']}")
        
        if self.timeline_manager:
            timeline_progress = self.timeline_manager.get_performance_progress()
            guidance = self.timeline_manager.get_performance_guidance()
            print(f"   Timeline: {timeline_progress['progress_percent']:.1f}% complete")
            print(f"   Phase: {timeline_progress['current_phase']}")
            print(f"   Engagement: {timeline_progress['engagement_level']:.2f}")
            print(f"   Behavior: {guidance['behavior_mode']}")
            print(f"   Silence: {guidance['silence_mode']}")
    
    def start_simulation(self):
        """Start the performance simulation"""
        if self.simulation_running:
            print("⚠️  Simulation already running")
            return
        
        self.simulation_running = True
        self.simulation_thread = threading.Thread(target=self._simulate_audio_events, daemon=True)
        self.simulation_thread.start()
        
        print(f"🎵 Performance simulation started: {self.performance_duration} minutes")
    
    def stop_simulation(self):
        """Stop the performance simulation"""
        if not self.simulation_running:
            print("⚠️  Simulation not running")
            return
        
        self.simulation_running = False
        
        if self.simulation_thread:
            self.simulation_thread.join(timeout=5.0)
        
        print("🛑 Performance simulation stopped")
        self._print_simulation_status()
    
    def is_simulation_complete(self) -> bool:
        """Check if simulation is complete"""
        if not self.simulation_running:
            return True
        
        elapsed = time.time() - self.stats['start_time']
        return elapsed >= (self.performance_duration * 60)

def main():
    """Test the performance simulator"""
    print("🎵 Performance Simulator Test")
    
    # Create simulator
    simulator = PerformanceSimulator(
        performance_duration=5,  # 5 minutes
        training_data_path="ai_learning_data/musical_memory.json"
    )
    
    # Start simulation
    simulator.start_simulation()
    
    try:
        # Wait for simulation to complete
        while not simulator.is_simulation_complete():
            time.sleep(1.0)
        
        print("✅ Simulation completed successfully")
        
    except KeyboardInterrupt:
        print("\n🛑 Simulation interrupted by user")
        simulator.stop_simulation()
    
    except Exception as e:
        print(f"❌ Simulation error: {e}")
        simulator.stop_simulation()

if __name__ == "__main__":
    main()
