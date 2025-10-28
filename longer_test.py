#!/usr/bin/env python3
"""15-second test"""
import subprocess
import time
import glob
import os

print("🧪 15-second test for bass...")
music_proc = subprocess.Popen(["python", "MusicHal_9000.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
time.sleep(2)
audio_proc = subprocess.Popen(["python", "test_synthetic_audio.py", "--duration", "13"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
audio_proc.wait()
time.sleep(1)
music_proc.terminate()
try:
    music_proc.wait(timeout=2)
except:
    music_proc.kill()

conv_logs = sorted(glob.glob("logs/conversation_*.csv"), key=os.path.getmtime, reverse=True)
if conv_logs:
    with open(conv_logs[0], 'r') as f:
        lines = f.readlines()
    outputs = [l for l in lines if 'OUTPUT' in l]
    melodic = [l for l in outputs if 'melodic' in l]
    bass = [l for l in outputs if 'bass' in l]
    print(f"Melodic: {len(melodic)}, Bass: {len(bass)}")
    if bass:
        print("Bass notes:")
        for line in bass[:5]:
            parts = line.strip().split(',')
            if len(parts) >= 6:
                print(f"  MIDI {parts[5]} at {parts[1]}s")
