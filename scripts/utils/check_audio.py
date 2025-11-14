#!/usr/bin/env python3
"""Quick audio device diagnostic"""

import sounddevice as sd
import numpy as np

print("🎤 Available Audio Devices:")
print("="*60)
devices = sd.query_devices()
for i, dev in enumerate(devices):
    print(f"{i}: {dev['name']}")
    print(f"   Input channels: {dev['max_input_channels']}")
    print(f"   Output channels: {dev['max_output_channels']}")
    print(f"   Default sample rate: {dev['default_samplerate']}")
    print()

print("="*60)
print(f"Default input device: {sd.default.device[0]}")
print(f"Default output device: {sd.default.device[1]}")
print()

# Try to open a stream with default settings
try:
    print("🎵 Testing default input device...")
    with sd.InputStream(channels=1, samplerate=44100, blocksize=2048):
        print("✅ Default input device works!")
except Exception as e:
    print(f"❌ Default input device failed: {e}")
    
    # Try each input device
    print("\n🔍 Testing each input device individually:")
    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            try:
                with sd.InputStream(device=i, channels=1, samplerate=44100, blocksize=2048):
                    print(f"✅ Device {i} ({dev['name']}) works!")
            except Exception as e:
                print(f"❌ Device {i} ({dev['name']}) failed: {e}")

