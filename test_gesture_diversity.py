#!/usr/bin/env python3
"""
Test script to check gesture token diversity during performance
"""

import sys
import os
sys.path.append('CCM3/lib/python3.10/site-packages')
sys.path.insert(0, os.getcwd())

import time
from collections import Counter
from MusicHal_9000 import MusicHal

def test_gesture_token_diversity():
    """Test gesture token generation to check for homogeneity"""
    
    print("🧪 Testing gesture token diversity...")
    
    # Initialize MusicHal system
    musichal = MusicHal()
    
    # Simulate some audio events to trigger gesture token generation
    print("🎵 Simulating audio events...")
    
    gesture_tokens = []
    
    # Get gesture tokens from recent events in the system
    try:
        # Check if the AI agent has a phrase generator
        if hasattr(musichal.ai_agent, 'behavior_engine') and hasattr(musichal.ai_agent.behavior_engine, 'phrase_generator'):
            pg = musichal.ai_agent.behavior_engine.phrase_generator
            if pg:
                # Get recent gesture tokens
                recent_tokens = pg._get_recent_human_tokens(n=20)
                gesture_tokens.extend(recent_tokens)
                print(f"🎯 Found {len(recent_tokens)} recent gesture tokens: {recent_tokens}")
        
        # Also check the listener for current gesture tokens
        if hasattr(musichal, 'listener') and hasattr(musichal.listener, 'hybrid_detector'):
            detector = musichal.listener.hybrid_detector
            # Check if there are gesture tokens available
            print(f"🎵 Gesture token detector type: {type(detector)}")
            
    except Exception as e:
        print(f"⚠️ Error accessing gesture tokens: {e}")
    
    # Analyze token diversity
    if gesture_tokens:
        token_counts = Counter(gesture_tokens)
        unique_tokens = len(token_counts)
        total_tokens = len(gesture_tokens)
        
        print(f"\n🎯 Gesture Token Analysis:")
        print(f"   Total tokens: {total_tokens}")
        print(f"   Unique tokens: {unique_tokens}")
        print(f"   Diversity ratio: {unique_tokens/total_tokens:.2f}")
        print(f"   Token distribution: {dict(token_counts)}")
        
        # Check for homogeneity
        most_common_token, most_common_count = token_counts.most_common(1)[0]
        homogeneity_ratio = most_common_count / total_tokens
        
        print(f"   Most common token: {most_common_token} ({most_common_count}/{total_tokens} = {homogeneity_ratio:.1%})")
        
        if homogeneity_ratio > 0.8:
            print("❌ HIGH HOMOGENEITY: >80% of tokens are the same!")
            print("   This could explain reduced rhythmic complexity.")
        elif homogeneity_ratio > 0.6:
            print("⚠️ MODERATE HOMOGENEITY: >60% of tokens are the same")
            print("   This might contribute to reduced variety.")
        else:
            print("✅ GOOD DIVERSITY: Tokens are reasonably varied")
    else:
        print("❌ No gesture tokens found - system may not be generating tokens properly")

if __name__ == "__main__":
    test_gesture_token_diversity()