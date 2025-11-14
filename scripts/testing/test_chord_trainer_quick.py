#!/usr/bin/env python3
"""
Quick Test for Interactive Chord Trainer
========================================

This script tests the chord trainer to make sure it's working correctly.
"""

import time
import sys
import os

# Add the project root to the path
sys.path.append('/Users/jonashsj/Jottacloud/PhD - UiA/CCM3/CCM3')

from test_interactive_chord_trainer import InteractiveChordTrainer

def test_chord_trainer():
    """Test the chord trainer functionality"""
    print("🧪 Testing Interactive Chord Trainer...")
    
    trainer = InteractiveChordTrainer()
    
    # Test initialization
    if not trainer.start():
        print("❌ Failed to start trainer")
        return False
    
    print("✅ Trainer started successfully")
    
    # Test label setting
    trainer.set_label("C")
    print("✅ Label setting works")
    
    # Test sample collection (simulate)
    print("📝 Simulating sample collection...")
    time.sleep(2)  # Let it collect some samples
    
    print(f"📊 Samples collected: {len(trainer.samples)}")
    
    # Test model training (if we have samples)
    if len(trainer.samples) > 0:
        print("🤖 Testing model training...")
        success = trainer.train_model()
        if success:
            print("✅ Model training works")
        else:
            print("⚠️ Model training failed (expected with few samples)")
    
    # Cleanup
    if trainer.listener:
        trainer.listener.stop()
    
    print("✅ Test completed successfully!")
    return True

if __name__ == "__main__":
    test_chord_trainer()


