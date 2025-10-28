#!/usr/bin/env python3
"""
Test Pydantic Logfire for monitoring MusicHal system
Helps debug click noise and performance issues
"""

import logfire
import time
import numpy as np

# Configure Logfire to send to console (no authentication needed for testing)
logfire.configure(send_to_logfire=False)  # Set to True after running: logfire auth

def simulate_midi_output():
    """Simulate MIDI output with timing monitoring"""
    with logfire.span("midi_note_sequence"):
        for i in range(5):
            note = 60 + i
            
            # Track individual note timing
            with logfire.span("send_midi_note", note=note):
                start_time = time.time()
                
                # Simulate MIDI message sending
                time.sleep(0.001)  # 1ms for note_on
                
                # Simulate control changes
                with logfire.span("send_control_changes"):
                    time.sleep(0.0005)  # Timbre
                    time.sleep(0.0005)  # Brightness
                    time.sleep(0.0005)  # Modulation
                
                end_time = time.time()
                latency_ms = (end_time - start_time) * 1000
                
                logfire.info(
                    "MIDI note sent",
                    note=note,
                    latency_ms=latency_ms,
                    total_messages=4
                )
                
            # Simulate pause between notes
            time.sleep(0.5)

def simulate_audio_processing():
    """Simulate audio event processing"""
    with logfire.span("audio_processing_loop"):
        for i in range(3):
            with logfire.span("process_audio_event"):
                # Simulate audio analysis
                f0 = 440.0 + i * 50
                rms_db = -30.0 + i * 2
                
                logfire.info(
                    "Audio event processed",
                    f0=f0,
                    rms_db=rms_db,
                    frame_number=i
                )
                
                # Simulate harmonic detection
                with logfire.span("harmonic_detection"):
                    time.sleep(0.002)
                    logfire.info("Chord detected", chord="Cmaj7", confidence=0.85)
                
                time.sleep(0.1)

def simulate_oracle_query():
    """Simulate AudioOracle query"""
    with logfire.span("oracle_query"):
        context_notes = [60, 64, 67]
        
        logfire.info(
            "Oracle query started",
            context_notes=context_notes,
            context_length=len(context_notes)
        )
        
        # Simulate query processing
        time.sleep(0.01)
        
        generated_notes = [72, 74, 76]
        
        logfire.info(
            "Oracle query completed",
            generated_notes=generated_notes,
            notes_generated=len(generated_notes)
        )

if __name__ == "__main__":
    print("🔥 Pydantic Logfire Test")
    print("=" * 50)
    print("This will monitor:")
    print("  - MIDI message timing and latency")
    print("  - Audio processing performance")
    print("  - AudioOracle query times")
    print("\nAfter running, check the Logfire dashboard!")
    print("=" * 50)
    
    with logfire.span("musichal_test_session"):
        print("\n1. Testing MIDI output...")
        simulate_midi_output()
        
        print("\n2. Testing audio processing...")
        simulate_audio_processing()
        
        print("\n3. Testing AudioOracle query...")
        simulate_oracle_query()
    
    print("\n✅ Test completed! Check Logfire dashboard for results.")
    print("   Visit: https://logfire.pydantic.dev/")

