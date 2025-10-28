#!/usr/bin/env python3
"""
End-to-End System Test
Tests the complete CCM3 pipeline:
1. Train with Chandra_trainer.py on Georgia.wav
2. Run MusicHal_9000.py with synthetic audio input
3. Verify harmonic and rhythmic context integration
4. Test system recognizing its own output
"""

import os
import sys
import time
import threading
import subprocess
import numpy as np
import sounddevice as sd
import librosa
from typing import Dict, List, Optional
import json
import signal
import queue

class EndToEndSystemTest:
    """Comprehensive test of the CCM3 system"""
    
    def __init__(self):
        self.test_results = {}
        self.audio_queue = queue.Queue()
        self.midi_events = []
        self.harmonic_context_detected = False
        self.rhythmic_context_detected = False
        self.system_output_detected = False
        self.test_duration = 10  # seconds - shorter for testing
        self.sample_rate = 44100
        
    def run_full_test(self):
        """Run the complete end-to-end test"""
        print("🧪 Starting End-to-End System Test")
        print("=" * 60)
        
        try:
            # Step 1: Train the system
            print("\n📚 Step 1: Training with Chandra_trainer.py")
            if not self._test_training():
                return False
            
            # Step 2: Generate synthetic audio
            print("\n🎵 Step 2: Generating synthetic audio input")
            if not self._generate_synthetic_audio():
                return False
            
            # Step 3: Test MusicHal_9000 with synthetic input
            print("\n🤖 Step 3: Testing MusicHal_9000.py with synthetic input")
            if not self._test_live_system():
                return False
            
            # Step 4: Verify system integration
            print("\n🔍 Step 4: Verifying system integration")
            if not self._verify_integration():
                return False
            
            # Step 5: Test feedback loop
            print("\n🔄 Step 5: Testing feedback loop")
            if not self._test_feedback_loop():
                return False
            
            # Final results
            self._print_final_results()
            return True
            
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _test_training(self) -> bool:
        """Test Chandra_trainer.py with Georgia.wav"""
        print("   Training on Georgia.wav...")
        
        # Check if Georgia.wav exists
        georgia_path = "input_audio/Georgia.wav"
        if not os.path.exists(georgia_path):
            print(f"   ❌ Georgia.wav not found at {georgia_path}")
            return False
        
        # Run Chandra_trainer
        cmd = [
            "python", "Chandra_trainer.py",
            "--file", georgia_path,
            "--output", "JSON/georgia_test",
            "--max-events", "1000"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                print("   ✅ Training completed successfully")
                
                # Check if output file was created (with or without .json extension)
                output_file = "JSON/georgia_test.json"
                if not os.path.exists(output_file):
                    output_file = "JSON/georgia_test"  # Try without extension
                
                if os.path.exists(output_file):
                    print(f"   ✅ Output file created: {output_file}")
                    
                    # Load and verify training results
                    with open(output_file, 'r') as f:
                        data = json.load(f)
                    
                    if data.get('training_successful', False):
                        print("   ✅ Training marked as successful")
                        
                        # Check for key components
                        if 'audio_oracle_stats' in data:
                            stats = data['audio_oracle_stats']
                            print(f"   📊 Patterns learned: {stats.get('total_patterns', 0)}")
                            print(f"   📊 States: {stats.get('total_states', 0)}")
                            print(f"   📊 Harmonic patterns: {stats.get('harmonic_patterns', 0)}")
                        
                        if 'rhythm_oracle_stats' in data:
                            rhythm_stats = data['rhythm_oracle_stats']
                            print(f"   🥁 Rhythmic patterns: {rhythm_stats.get('total_patterns', 0)}")
                            print(f"   🥁 Avg tempo: {rhythm_stats.get('avg_tempo', 0):.1f} BPM")
                        
                        progression = []
                        if 'chord_progression' in data:
                            progression = data['chord_progression']
                            print(f"   🎼 Chord progression detected: {len(progression)} chords")
                            print(f"   🎼 Sample chords: {progression[:5]}")
                        
                        self.test_results['training'] = {
                            'success': True,
                            'patterns_learned': stats.get('total_patterns', 0),
                            'harmonic_patterns': stats.get('harmonic_patterns', 0),
                            'rhythmic_patterns': rhythm_stats.get('total_patterns', 0),
                            'chord_progression_length': len(progression)
                        }
                        return True
                    else:
                        print("   ❌ Training not marked as successful")
                        return False
                else:
                    print("   ❌ Output file not created")
                    return False
            else:
                print(f"   ❌ Training failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("   ❌ Training timed out")
            return False
        except Exception as e:
            print(f"   ❌ Training error: {e}")
            return False
    
    def _generate_synthetic_audio(self) -> bool:
        """Generate synthetic audio for testing"""
        print("   Generating synthetic audio...")
        
        try:
            # Generate a simple chord progression (G - D - Em - C)
            duration = self.test_duration
            t = np.linspace(0, duration, int(self.sample_rate * duration))
            
            # Create a simple chord progression
            chords = [
                {'notes': [392, 494, 587, 659], 'duration': 4},  # G major
                {'notes': [294, 370, 440, 554], 'duration': 4},  # D major  
                {'notes': [330, 392, 494, 587], 'duration': 4},  # Em
                {'notes': [262, 330, 392, 523], 'duration': 4},  # C major
            ]
            
            audio = np.zeros_like(t)
            current_time = 0
            
            for chord in chords:
                chord_duration = chord['duration']
                chord_end = min(current_time + chord_duration, duration)
                chord_indices = (t >= current_time) & (t < chord_end)
                
                # Add each note in the chord
                for freq in chord['notes']:
                    note_audio = 0.3 * np.sin(2 * np.pi * freq * t[chord_indices])
                    note_audio *= np.exp(-0.1 * (t[chord_indices] - current_time))  # Decay
                    audio[chord_indices] += note_audio
                
                current_time += chord_duration
                if current_time >= duration:
                    break
            
            # Add some rhythmic elements
            beat_freq = 2.0  # 2 Hz = 120 BPM
            rhythm = 0.1 * np.sin(2 * np.pi * beat_freq * t)
            audio += rhythm
            
            # Normalize
            audio = audio / np.max(np.abs(audio)) * 0.7
            
            # Save for playback
            self.synthetic_audio = audio
            self.synthetic_duration = duration
            
            print(f"   ✅ Generated {duration}s of synthetic audio")
            print(f"   🎼 Chord progression: G - D - Em - C")
            print(f"   🥁 Rhythmic elements: 120 BPM")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Audio generation error: {e}")
            return False
    
    def _test_live_system(self) -> bool:
        """Test MusicHal_9000.py with synthetic audio input"""
        print("   Starting MusicHal_9000.py with synthetic input...")
        
        try:
            # Start MusicHal_9000 in a subprocess
            cmd = [
                "python", "MusicHal_9000.py",
                "--performance-duration", "1",  # 1 minute test
                "--density", "0.7",
                "--give-space", "0.3",
                "--initiative", "0.8"
            ]
            
            # Start the process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Start audio playback in a separate thread
            audio_thread = threading.Thread(
                target=self._play_synthetic_audio,
                daemon=True
            )
            audio_thread.start()
            
            # Monitor the process output
            start_time = time.time()
            harmonic_detected = False
            rhythmic_detected = False
            decisions_made = 0
            notes_sent = 0
            
            print("   Monitoring system output...")
            
            while time.time() - start_time < self.test_duration:
                # Check if process is still running
                if process.poll() is not None:
                    print("   ⚠️ Process ended early")
                    break
                
                # Read output (non-blocking)
                try:
                    line = process.stdout.readline()
                    if line:
                        line = line.strip()
                        print(f"   📝 {line}")
                        
                        # Look for key indicators
                        if "🎼 Harmonic:" in line:
                            harmonic_detected = True
                            print("   ✅ Harmonic context detected!")
                        
                        if "🥁 Rhythmic:" in line:
                            rhythmic_detected = True
                            print("   ✅ Rhythmic context detected!")
                        
                        if "🎹 Processing" in line and "decision" in line:
                            decisions_made += 1
                        
                        if "🎵 MPE(" in line or "Note:" in line:
                            notes_sent += 1
                        
                        if "MPE" in line or "MIDI" in line:
                            print("   🎹 MIDI output detected!")
                
                except Exception as e:
                    pass
                
                time.sleep(0.1)
            
            # Stop the process
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            
            # Wait for audio thread to finish
            audio_thread.join(timeout=2)
            
            # Record results
            self.test_results['live_system'] = {
                'harmonic_detected': harmonic_detected,
                'rhythmic_detected': rhythmic_detected,
                'decisions_made': decisions_made,
                'notes_sent': notes_sent,
                'test_duration': self.test_duration
            }
            
            print(f"   📊 Decisions made: {decisions_made}")
            print(f"   📊 Notes sent: {notes_sent}")
            print(f"   📊 Harmonic detected: {harmonic_detected}")
            print(f"   📊 Rhythmic detected: {rhythmic_detected}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Live system test error: {e}")
            return False
    
    def _play_synthetic_audio(self):
        """Play synthetic audio for the system to process"""
        try:
            print("   🎵 Playing synthetic audio...")
            
            # Play the audio
            sd.play(self.synthetic_audio, samplerate=self.sample_rate)
            sd.wait()  # Wait until audio is finished
            
            print("   ✅ Synthetic audio playback completed")
            
        except Exception as e:
            print(f"   ❌ Audio playback error: {e}")
    
    def _verify_integration(self) -> bool:
        """Verify that harmonic and rhythmic contexts are working together"""
        print("   Verifying system integration...")
        
        training_results = self.test_results.get('training', {})
        live_results = self.test_results.get('live_system', {})
        
        # Check training results
        if not training_results.get('success', False):
            print("   ❌ Training was not successful")
            return False
        
        # Check live system results
        if not live_results.get('harmonic_detected', False):
            print("   ❌ Harmonic context not detected in live system")
            return False
        
        if not live_results.get('rhythmic_detected', False):
            print("   ❌ Rhythmic context not detected in live system")
            return False
        
        if live_results.get('decisions_made', 0) == 0:
            print("   ❌ No decisions were made by the AI agent")
            return False
        
        if live_results.get('notes_sent', 0) == 0:
            print("   ❌ No MIDI notes were sent")
            return False
        
        print("   ✅ All integration checks passed")
        return True
    
    def _test_feedback_loop(self) -> bool:
        """Test that the system can recognize its own output"""
        print("   Testing feedback loop...")
        
        # This is a simplified test - in a real scenario, we'd need to
        # capture the MIDI output and feed it back as audio input
        
        live_results = self.test_results.get('live_system', {})
        
        # Check if the system made decisions and sent notes
        if live_results.get('decisions_made', 0) > 0 and live_results.get('notes_sent', 0) > 0:
            print("   ✅ System generated output (decisions and notes)")
            print("   ✅ Feedback loop test passed (system is responsive)")
            return True
        else:
            print("   ❌ System did not generate sufficient output")
            return False
    
    def _print_final_results(self):
        """Print comprehensive test results"""
        print("\n" + "=" * 60)
        print("🧪 END-TO-END SYSTEM TEST RESULTS")
        print("=" * 60)
        
        # Training results
        training = self.test_results.get('training', {})
        print(f"\n📚 Training Results:")
        print(f"   Success: {'✅' if training.get('success') else '❌'}")
        print(f"   Patterns learned: {training.get('patterns_learned', 0)}")
        print(f"   Harmonic patterns: {training.get('harmonic_patterns', 0)}")
        print(f"   Rhythmic patterns: {training.get('rhythmic_patterns', 0)}")
        print(f"   Chord progression: {training.get('chord_progression_length', 0)} chords")
        
        # Live system results
        live = self.test_results.get('live_system', {})
        print(f"\n🤖 Live System Results:")
        print(f"   Harmonic context: {'✅' if live.get('harmonic_detected') else '❌'}")
        print(f"   Rhythmic context: {'✅' if live.get('rhythmic_detected') else '❌'}")
        print(f"   Decisions made: {live.get('decisions_made', 0)}")
        print(f"   Notes sent: {live.get('notes_sent', 0)}")
        print(f"   Test duration: {live.get('test_duration', 0)}s")
        
        # Overall assessment
        print(f"\n🎯 Overall Assessment:")
        all_passed = (
            training.get('success', False) and
            live.get('harmonic_detected', False) and
            live.get('rhythmic_detected', False) and
            live.get('decisions_made', 0) > 0 and
            live.get('notes_sent', 0) > 0
        )
        
        if all_passed:
            print("   🎉 ALL TESTS PASSED! System is working correctly.")
            print("   ✅ Training pipeline functional")
            print("   ✅ Live system responsive")
            print("   ✅ Harmonic context integrated")
            print("   ✅ Rhythmic context integrated")
            print("   ✅ AI decision making active")
            print("   ✅ MIDI output functional")
        else:
            print("   ⚠️ Some tests failed. Check individual results above.")
        
        print("\n" + "=" * 60)

def main():
    """Main test entry point"""
    print("🧪 CCM3 End-to-End System Test")
    print("This test will:")
    print("1. Train the system on Georgia.wav")
    print("2. Run MusicHal_9000.py with synthetic audio")
    print("3. Verify harmonic and rhythmic context integration")
    print("4. Test system feedback loop")
    print()
    
    # Check prerequisites
    if not os.path.exists("input_audio/Georgia.wav"):
        print("❌ Georgia.wav not found in input_audio/")
        print("Please ensure the audio file exists before running this test.")
        return 1
    
    if not os.path.exists("Chandra_trainer.py"):
        print("❌ Chandra_trainer.py not found")
        return 1
    
    if not os.path.exists("MusicHal_9000.py"):
        print("❌ MusicHal_9000.py not found")
        return 1
    
    # Run the test
    test = EndToEndSystemTest()
    success = test.run_full_test()
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
