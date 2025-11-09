#!/usr/bin/env python3
"""
Test script for TouchOSC chord override system

Sends test OSC messages to MusicHal_9000 to verify manual chord override functionality.

Usage:
    1. Start MusicHal_9000.py in one terminal
    2. Run this script in another terminal
    3. Watch console for override messages

Jonas Sjøvaag - PhD Artistic Research - University of Agder
"""

from pythonosc.udp_client import SimpleUDPClient
import time

def test_chord_override():
    """Send test OSC messages to MusicHal_9000"""
    # Create OSC client
    client = SimpleUDPClient("127.0.0.1", 5007)  # localhost, port 5007
    
    print("📱 TouchOSC Chord Override Test")
    print("=" * 50)
    print("Sending test messages to MusicHal_9000...")
    print("(Make sure MusicHal_9000.py is running!)")
    print()
    
    # Test 1: Override to C major for 10 seconds
    print("Test 1: Override to C major (10s)")
    client.send_message("/chord", ["C", 10.0])
    time.sleep(2)
    
    # Test 2: Override to D minor for 15 seconds
    print("Test 2: Override to Dm (15s)")
    client.send_message("/chord", ["Dm", 15.0])
    time.sleep(2)
    
    # Test 3: Override to A major (default 30s)
    print("Test 3: Override to A (default 30s)")
    client.send_message("/chord", ["A"])
    time.sleep(2)
    
    # Test 4: Override to F#m for 5 seconds
    print("Test 4: Override to F#m (5s)")
    client.send_message("/chord", ["F#m", 5.0])
    time.sleep(2)
    
    # Test 5: Clear override
    print("Test 5: Clear override")
    client.send_message("/chord/clear", [])
    time.sleep(2)
    
    # Test 6: Override to G major
    print("Test 6: Override to G (20s)")
    client.send_message("/chord", ["G", 20.0])
    
    print()
    print("✅ All test messages sent!")
    print("Check MusicHal_9000 console for override confirmations")
    print()
    print("Expected output:")
    print("  🎹 MANUAL OVERRIDE: C (for 10s)")
    print("     Machine detected: <chord> (conf: X.XX)")
    print("     You corrected to: C")
    print("  ...")
    print("  🎹 Override expired: C → <detected>")


if __name__ == "__main__":
    try:
        test_chord_override()
    except ConnectionRefusedError:
        print("❌ Connection refused - is MusicHal_9000.py running?")
        print("   Start it with: python MusicHal_9000.py")
    except Exception as e:
        print(f"❌ Error: {e}")
