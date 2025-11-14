#!/usr/bin/env python3
"""Quick 10-second test with analysis"""
import subprocess
import time
import os
import glob

print("🧪 Starting 10-second test iteration...")
print("=" * 60)

# Start MusicHal_9000
print("🎵 Starting MusicHal_9000...")
music_proc = subprocess.Popen(
    ["python", "MusicHal_9000.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

# Wait 2 seconds for it to initialize
time.sleep(2)

# Start synthetic audio for 8 seconds
print("🔊 Playing synthetic audio...")
audio_proc = subprocess.Popen(
    ["python", "test_synthetic_audio.py", "--duration", "8"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

# Wait for audio to finish
audio_proc.wait()

# Give MusicHal a moment to finish processing
time.sleep(1)

# Kill MusicHal
print("🛑 Stopping MusicHal_9000...")
music_proc.terminate()
try:
    music_proc.wait(timeout=2)
except subprocess.TimeoutExpired:
    music_proc.kill()

print("✅ Test complete!")
print("=" * 60)

# Analyze logs
print("\n📊 LOG ANALYSIS:")
print("-" * 60)

# Find most recent conversation log
conv_logs = sorted(glob.glob("logs/conversation_*.csv"), key=os.path.getmtime, reverse=True)
if conv_logs:
    latest_conv = conv_logs[0]
    print(f"Conversation log: {os.path.basename(latest_conv)}")
    
    with open(latest_conv, 'r') as f:
        lines = f.readlines()
    
    inputs = [l for l in lines if 'INPUT' in l]
    outputs = [l for l in lines if 'OUTPUT' in l]
    melodic = [l for l in outputs if 'melodic' in l]
    bass = [l for l in outputs if 'bass' in l]
    
    print(f"  Total events: {len(lines) - 1}")
    print(f"  INPUT (human): {len(inputs)}")
    print(f"  OUTPUT (AI): {len(outputs)}")
    print(f"    - Melodic: {len(melodic)}")
    print(f"    - Bass: {len(bass)}")
    
    if outputs:
        print(f"\n  First 5 AI outputs:")
        for line in outputs[:5]:
            parts = line.strip().split(',')
            if len(parts) >= 7:
                voice = parts[3]
                midi = parts[5]
                mode = parts[8] if len(parts) > 8 else 'unknown'
                print(f"    {voice:8} MIDI {midi:3} mode={mode}")
    
    # Check timing between notes
    if len(melodic) >= 2:
        try:
            times = [float(l.split(',')[0]) for l in melodic[:10]]
            if len(times) >= 2:
                diffs = [times[i+1] - times[i] for i in range(len(times)-1)]
                avg_gap = sum(diffs) / len(diffs)
                print(f"\n  Melody note timing:")
                print(f"    Average gap: {avg_gap:.2f}s")
                print(f"    Min gap: {min(diffs):.2f}s")
                print(f"    Max gap: {max(diffs):.2f}s")
        except:
            pass
    
    if len(bass) >= 2:
        try:
            times = [float(l.split(',')[0]) for l in bass[:10]]
            if len(times) >= 2:
                diffs = [times[i+1] - times[i] for i in range(len(times)-1)]
                avg_gap = sum(diffs) / len(diffs)
                print(f"\n  Bass note timing:")
                print(f"    Average gap: {avg_gap:.2f}s")
                print(f"    Min gap: {min(diffs):.2f}s")
                print(f"    Max gap: {max(diffs):.2f}s")
        except:
            pass

print("\n" + "=" * 60)
