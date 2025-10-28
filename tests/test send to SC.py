from pythonosc import udp_client
import time

client = udp_client.SimpleUDPClient("127.0.0.1", 10111)

# Test all voices
voices = {
    'voice1': 220.0,  # A3
    'voice2': 330.0,  # E4 
    'voice3': 440.0   # A4
}

for voice, freq in voices.items():
    print(f"Testing {voice} at {freq}Hz")
    client.send_message(f"/{voice}", [freq, 0.5, 1, 0.0])
    time.sleep(2)  # Wait to hear each voice clearly
