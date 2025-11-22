#!/usr/bin/env python3
"""
Test autonomous generation on pre-Somax branch
Verifies PhraseGenerator autonomous mode (without SomaxBridge)
"""

import sys
import time
from agent.phrase_generator import PhraseGenerator

def test_autonomous_state_tracking():
    """Test per-voice phrase tracking state"""
    print("=" * 60)
    print("TEST: Autonomous State Tracking")
    print("=" * 60)
    
    # PhraseGenerator requires rhythm_oracle parameter (can be None for state testing)
    pg = PhraseGenerator(rhythm_oracle=None)
    
    # Initially autonomous mode disabled
    assert not pg.autonomous_mode, "Autonomous mode should start disabled"
    print("✓ Autonomous mode starts disabled")
    
    # Enable autonomous mode
    pg.set_autonomous_mode(True)
    assert pg.autonomous_mode, "Autonomous mode should be enabled"
    print("✓ Autonomous mode enabled successfully")
    
    # Check initial state for both voices
    for voice in ['melodic', 'bass']:
        assert not pg.phrase_complete[voice], f"{voice} should not be complete initially"
        assert pg.notes_in_phrase[voice] == 0, f"{voice} should have 0 notes initially"
        print(f"✓ {voice.capitalize()} voice state initialized correctly")
    
    # Test should_respond() - both voices ready initially
    assert pg.should_respond('melodic'), "Melodic should be ready initially"
    assert pg.should_respond('bass'), "Bass should be ready initially"
    print("✓ Both voices ready to generate initially")
    
    # Simulate note generation for melody
    for i in range(6):  # Target phrase length for melody
        pg.mark_note_generated('melodic')
        time.sleep(0.01)  # Small delay to ensure timing updates
    
    # Melody should now be complete
    assert pg.phrase_complete['melodic'], "Melody should be complete after 6 notes"
    print("✓ Melody phrase marked complete after 6 notes")
    
    # Melody should NOT respond during pause period
    assert not pg.should_respond('melodic'), "Melody should be pausing"
    print("✓ Melody pausing (phrase complete)")
    
    # Bass should still respond (independent state)
    assert pg.should_respond('bass'), "Bass should still be ready"
    print("✓ Bass still ready (independent from melody)")
    
    # Simulate bass phrase completion
    for i in range(10):  # Target phrase length for bass
        pg.mark_note_generated('bass')
        time.sleep(0.01)
    
    assert pg.phrase_complete['bass'], "Bass should be complete after 10 notes"
    assert not pg.should_respond('bass'), "Bass should be pausing"
    print("✓ Bass phrase marked complete after 10 notes")
    print("✓ Bass pausing (phrase complete)")
    
    # Wait for auto-reset (2 seconds)
    print("\nWaiting 2.1s for auto-reset...")
    time.sleep(2.1)
    
    # Both voices should auto-reset
    assert pg.should_respond('melodic'), "Melody should auto-reset after 2s"
    assert pg.should_respond('bass'), "Bass should auto-reset after 2s"
    assert not pg.phrase_complete['melodic'], "Melody phrase_complete should be cleared"
    assert not pg.phrase_complete['bass'], "Bass phrase_complete should be cleared"
    assert pg.notes_in_phrase['melodic'] == 0, "Melody note counter should reset"
    assert pg.notes_in_phrase['bass'] == 0, "Bass note counter should reset"
    print("✓ Both voices auto-reset after 2s pause")
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED: Autonomous state tracking working")
    print("=" * 60)

def test_minimum_interval():
    """Test 0.5s minimum generation interval"""
    print("\n" + "=" * 60)
    print("TEST: Minimum Generation Interval (0.5s)")
    print("=" * 60)
    
    pg = PhraseGenerator(rhythm_oracle=None)
    pg.set_autonomous_mode(True)
    
    # First call should be ready
    assert pg.should_respond('melodic'), "First call should be ready"
    print("✓ First call ready")
    
    # Mark note generated
    pg.mark_note_generated('melodic')
    
    # Immediate second call should NOT be ready (interval not passed)
    assert not pg.should_respond('melodic'), "Immediate second call should not be ready"
    print("✓ Immediate second call blocked (interval not passed)")
    
    # Wait for interval
    print("Waiting 0.51s for interval...")
    time.sleep(0.51)
    
    # Should be ready now
    assert pg.should_respond('melodic'), "Should be ready after interval"
    print("✓ Ready after 0.5s interval")
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED: Minimum interval working")
    print("=" * 60)

def test_voice_independence():
    """Test that voices operate independently"""
    print("\n" + "=" * 60)
    print("TEST: Voice Independence")
    print("=" * 60)
    
    pg = PhraseGenerator(rhythm_oracle=None)
    pg.set_autonomous_mode(True)
    
    # Generate 3 melody notes
    for i in range(3):
        pg.mark_note_generated('melodic')
        time.sleep(0.01)
    
    # Generate 8 bass notes
    for i in range(8):
        pg.mark_note_generated('bass')
        time.sleep(0.01)
    
    # Melody: 3/6 notes (not complete)
    # Bass: 8/10 notes (not complete)
    assert not pg.phrase_complete['melodic'], "Melody should not be complete (3/6)"
    assert not pg.phrase_complete['bass'], "Bass should not be complete (8/10)"
    assert pg.notes_in_phrase['melodic'] == 3, "Melody should have 3 notes"
    assert pg.notes_in_phrase['bass'] == 8, "Bass should have 8 notes"
    print("✓ Independent note counters: melody=3/6, bass=8/10")
    
    # Complete melody
    for i in range(3):  # Add 3 more to reach 6
        pg.mark_note_generated('melodic')
        time.sleep(0.01)
    
    assert pg.phrase_complete['melodic'], "Melody should be complete"
    assert not pg.phrase_complete['bass'], "Bass should still not be complete"
    print("✓ Melody complete (6 notes), bass still playing (8/10)")
    
    # Complete bass
    for i in range(2):  # Add 2 more to reach 10
        pg.mark_note_generated('bass')
        time.sleep(0.01)
    
    assert pg.phrase_complete['bass'], "Bass should be complete"
    print("✓ Bass complete (10 notes)")
    
    # Different completion times should allow different reset times
    melody_complete_time = pg.phrase_complete_time['melodic']
    bass_complete_time = pg.phrase_complete_time['bass']
    assert bass_complete_time > melody_complete_time, "Bass completed after melody"
    time_diff = bass_complete_time - melody_complete_time
    print(f"✓ Staggered completion: bass completed {time_diff:.3f}s after melody")
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED: Voice independence working")
    print("=" * 60)

if __name__ == "__main__":
    try:
        test_autonomous_state_tracking()
        test_minimum_interval()
        test_voice_independence()
        
        print("\n" + "🎉" * 30)
        print("SUCCESS: All autonomous generation tests passed!")
        print("Ready for 2-minute diagnostic test.")
        print("🎉" * 30)
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
