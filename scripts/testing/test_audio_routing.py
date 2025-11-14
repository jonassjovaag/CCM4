#!/usr/bin/env python3
"""
Test Audio Routing - Verify MIDI → Ableton → Audio Input
"""

import sounddevice as sd
import mido
import numpy as np
import time

def test_routing(audio_device=5):
    """Test if audio routing is working"""
    
    print("\n🔧 Testing Audio Routing")
    print("=" * 70)
    
    # 1. Check MIDI output
    print("\n1. Checking MIDI output...")
    try:
        ports = mido.get_output_names()
        print(f"   Available MIDI ports: {ports}")
        
        if "IAC Driver Chord Trainer Output" in ports:
            port = mido.open_output("IAC Driver Chord Trainer Output")
            print(f"   ✅ MIDI port opened")
        else:
            print(f"   ❌ IAC Driver not found!")
            return
    except Exception as e:
        print(f"   ❌ MIDI error: {e}")
        return
    
    # 2. Test audio capture
    print(f"\n2. Testing audio capture (device {audio_device})...")
    print(f"   Recording 2 seconds of silence (baseline)...")
    
    silence = sd.rec(int(2 * 48000), samplerate=48000, channels=1, device=audio_device, blocking=True)
    silence_rms = np.sqrt(np.mean(silence**2))
    silence_max = np.max(np.abs(silence))
    
    print(f"   Baseline RMS: {silence_rms:.6f}")
    print(f"   Baseline Max: {silence_max:.6f}")
    
    # 3. Send MIDI and record
    print(f"\n3. Sending MIDI chord and recording...")
    print(f"   Playing C major [60, 64, 67]...")
    
    # Start recording
    audio_buffer = []
    
    def callback(indata, frames, time_info, status):
        audio_buffer.extend(indata[:, 0].copy())
    
    stream = sd.InputStream(device=audio_device, channels=1, samplerate=48000, callback=callback)
    stream.start()
    
    # Send MIDI
    for note in [60, 64, 67]:
        port.send(mido.Message('note_on', channel=0, note=note, velocity=80))
    
    time.sleep(2.5)
    
    # Stop MIDI
    for note in [60, 64, 67]:
        port.send(mido.Message('note_off', channel=0, note=note, velocity=0))
    
    stream.stop()
    stream.close()
    port.close()
    
    # Analyze
    audio_array = np.array(audio_buffer)
    audio_rms = np.sqrt(np.mean(audio_array**2))
    audio_max = np.max(np.abs(audio_array))
    
    print(f"\n   Audio captured:")
    print(f"   RMS: {audio_rms:.6f}")
    print(f"   Max: {audio_max:.6f}")
    print(f"   Ratio to baseline: {audio_rms / silence_rms if silence_rms > 0 else 0:.2f}x")
    
    # 4. Diagnosis
    print("\n" + "=" * 70)
    print("📊 DIAGNOSIS:")
    print("=" * 70)
    
    if audio_rms < 0.0001:
        print("❌ NO AUDIO DETECTED")
        print("\nPossible issues:")
        print("  1. Ableton not receiving MIDI")
        print("     → Check: Ableton MIDI preferences")
        print("     → Verify: IAC Driver is enabled in Audio MIDI Setup")
        print("  2. Ableton not outputting audio")
        print("     → Check: Ableton audio output device")
        print("     → Verify: Track is armed/monitoring")
        print("  3. Audio routing not capturing Ableton")
        print("     → Check: Merging interface routing/monitoring")
        print("     → Verify: Loopback or monitoring is enabled")
    elif audio_rms < 0.001:
        print("⚠️  VERY QUIET AUDIO")
        print(f"   Signal detected but very weak ({audio_rms:.6f})")
        print("\nActions:")
        print("  1. Increase Ableton output volume")
        print("  2. Check Merging interface gain/monitoring")
        print("  3. Move microphone closer to speakers")
    else:
        print("✅ AUDIO ROUTING WORKING!")
        print(f"   Good signal level (RMS: {audio_rms:.6f})")
        print("\nReady to run chord trainer!")
    
    print("=" * 70)


if __name__ == "__main__":
    import sys
    device = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    test_routing(device)
