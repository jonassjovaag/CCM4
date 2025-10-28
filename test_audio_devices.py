#!/usr/bin/env python3
"""
Audio Device Testing and Troubleshooting for MusicHal_9000
Helps diagnose and fix audio input issues
"""

import sounddevice as sd
import numpy as np

def list_devices():
    """List all available audio devices"""
    print("🎤 Available Audio Devices:")
    print("=" * 60)
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if isinstance(device, dict):
            input_channels = device.get('max_input_channels', 0)
            output_channels = device.get('max_output_channels', 0)
            name = device.get('name', 'Unknown')
            default_sr = device.get('default_samplerate', 0)
            
            marker = ""
            if i == sd.default.device[0]:
                marker = " [DEFAULT INPUT]"
            elif i == sd.default.device[1]:
                marker = " [DEFAULT OUTPUT]"
            
            if input_channels > 0:
                print(f"\n{i}: {name}{marker}")
                print(f"   Input channels: {input_channels}")
                print(f"   Sample rate: {default_sr} Hz")
                print(f"   Hostapi: {device.get('hostapi', 'Unknown')}")
    
    print("\n" + "=" * 60)
    print(f"Default input device: {sd.default.device[0]}")
    print(f"Default output device: {sd.default.device[1]}")

def test_device(device_id=None, duration=2.0):
    """Test an audio input device"""
    try:
        device_name = sd.query_devices(device_id)['name'] if device_id is not None else "Default"
        print(f"\n🎙️  Testing device: {device_name}")
        print(f"   Recording {duration} seconds...")
        
        # Try to record audio
        recording = sd.rec(
            int(duration * 44100),
            samplerate=44100,
            channels=1,
            device=device_id,
            dtype='float32'
        )
        sd.wait()
        
        # Analyze recording
        rms = np.sqrt(np.mean(recording**2))
        max_val = np.max(np.abs(recording))
        
        print(f"   ✅ Recording successful!")
        print(f"   📊 RMS level: {20 * np.log10(rms + 1e-10):.1f} dB")
        print(f"   📊 Peak level: {20 * np.log10(max_val + 1e-10):.1f} dB")
        
        if rms < 1e-6:
            print(f"   ⚠️  Very low signal - check if device is active")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Failed to test device: {e}")
        return False

def recommend_device():
    """Recommend best audio input device"""
    print("\n🎯 Recommended Input Devices:")
    print("=" * 60)
    
    devices = sd.query_devices()
    recommendations = []
    
    for i, device in enumerate(devices):
        if isinstance(device, dict):
            input_channels = device.get('max_input_channels', 0)
            name = device.get('name', '')
            
            if input_channels > 0:
                # Score devices
                score = 0
                reason = []
                
                if 'MacBook Pro Microphone' in name:
                    score = 100
                    reason.append("Built-in microphone (always works)")
                elif 'BlackHole' in name:
                    score = 50
                    reason.append("Virtual audio device (good for routing)")
                elif 'Merging' in name or 'RAVENNA' in name:
                    score = 30
                    reason.append("Pro audio interface (may need configuration)")
                elif 'Microsoft Teams' in name or 'Zoom' in name:
                    score = 20
                    reason.append("Virtual meeting device (may conflict)")
                elif 'Background Music' in name:
                    score = 10
                    reason.append("System audio capture")
                
                if score > 0:
                    recommendations.append({
                        'id': i,
                        'name': name,
                        'score': score,
                        'reason': ', '.join(reason)
                    })
    
    # Sort by score
    recommendations.sort(key=lambda x: x['score'], reverse=True)
    
    for rec in recommendations[:3]:
        print(f"\n{rec['score']:3d}% Device {rec['id']}: {rec['name']}")
        print(f"      Reason: {rec['reason']}")
    
    if recommendations:
        best = recommendations[0]
        print(f"\n💡 Recommendation: Use device {best['id']}")
        print(f"   Run: python MusicHal_9000.py --input-device {best['id']}")

def main():
    print("🎵 MusicHal_9000 Audio Device Diagnostics")
    print("=" * 60)
    
    # List all devices
    list_devices()
    
    # Recommend best device
    recommend_device()
    
    # Test default input
    print("\n" + "=" * 60)
    print("Testing default input device...")
    default_input = sd.default.device[0]
    test_device(default_input, duration=1.0)
    
    # Test MacBook microphone if available
    print("\n" + "=" * 60)
    print("Testing MacBook Pro Microphone...")
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if isinstance(device, dict) and 'MacBook Pro Microphone' in device.get('name', ''):
            test_device(i, duration=1.0)
            break
    
    print("\n" + "=" * 60)
    print("✅ Diagnostics complete!")

if __name__ == "__main__":
    main()



































