#!/usr/bin/env python3
"""
Test script to verify phrase_arc selection logic is working correctly
"""

import sys
import os
sys.path.append('CCM3/lib/python3.10/site-packages')

# Add the current directory to the path so we can import the modules
sys.path.insert(0, os.getcwd())

from agent.phrase_generator import PhraseGenerator, PhraseArc
import time

def test_phrase_arc_selection():
    """Test that phrase arc selection doesn't choose SILENCE"""
    
    print("🧪 Testing PhraseArc selection logic...")
    
    # Create a PhraseGenerator
    pg = PhraseGenerator(rhythm_oracle=None, enable_silence=True)
    
    # Test multiple arc selections
    silence_count = 0
    total_tests = 100
    
    for i in range(total_tests):
        # Simulate current_event
        current_event = {
            'instrument': 'melodic',
            'rms_db': -60.0
        }
        
        # Get phrase arc
        phrase_arc = pg._decide_phrase_arc(current_event)
        
        if phrase_arc == PhraseArc.SILENCE:
            silence_count += 1
            
        print(f"Test {i+1}: {phrase_arc}")
        
        if i == 10:  # Just test first 10 to see the pattern
            break
    
    print(f"\n🎯 Results after {i+1} tests:")
    print(f"   Silence count: {silence_count}")
    print(f"   Silence rate: {silence_count/(i+1)*100:.1f}%")
    
    if silence_count == 0:
        print("✅ SUCCESS: No silence phrases generated!")
    else:
        print("❌ PROBLEM: Silence phrases still being generated")

if __name__ == "__main__":
    test_phrase_arc_selection()