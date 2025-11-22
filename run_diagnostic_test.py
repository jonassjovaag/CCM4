#!/usr/bin/env python3
"""
Automated diagnostic test for pre-Somax autonomous generation
Runs 2-minute test and analyzes bass note diversity
"""

import subprocess
import sys
import time
import csv
from pathlib import Path
from collections import Counter
import glob

def run_test(duration_minutes=2):
    """Run MusicHal_9000 test for specified duration"""
    print("=" * 80)
    print(f"STARTING {duration_minutes}-MINUTE DIAGNOSTIC TEST (Pre-Somax Branch)")
    print("=" * 80)
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Branch: fix/autonomous-generation-no-somax")
    print(f"Goal: Test bass diversity without SomaxBridge Factor Oracle\n")
    
    cmd = [
        sys.executable,
        "MusicHal_9000.py",
        "--enable-meld",
        f"--performance-duration={duration_minutes}"
    ]
    
    print(f"Command: {' '.join(cmd)}\n")
    print("Running test... (press Ctrl+C to stop early)\n")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent,
            timeout=(duration_minutes * 60) + 30,  # Extra 30s buffer
            capture_output=True,
            text=True
        )
        
        elapsed = time.time() - start_time
        print(f"\n✓ Test completed in {elapsed:.1f}s")
        
        # Check for errors
        if result.returncode != 0:
            print(f"\n⚠️  Warning: Process exited with code {result.returncode}")
            if result.stderr:
                print("STDERR:", result.stderr[-500:])  # Last 500 chars
        
        return True
        
    except subprocess.TimeoutExpired:
        print(f"\n⚠️  Test timed out after {duration_minutes} minutes")
        return False
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Error running test: {e}")
        return False

def find_latest_midi_log():
    """Find the most recent MIDI output log"""
    log_pattern = "logs/midi_output_*.csv"
    files = glob.glob(log_pattern)
    
    if not files:
        print(f"❌ No MIDI output files found matching {log_pattern}")
        return None
    
    # Sort by modification time, most recent first
    files.sort(key=lambda f: Path(f).stat().st_mtime, reverse=True)
    latest = files[0]
    
    print(f"📄 Latest MIDI output: {latest}")
    print(f"   Modified: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(Path(latest).stat().st_mtime))}")
    print(f"   Size: {Path(latest).stat().st_size} bytes\n")
    
    return latest

def analyze_bass_diversity(csv_file):
    """Analyze bass note diversity from MIDI output"""
    print("=" * 80)
    print("BASS NOTE DIVERSITY ANALYSIS")
    print("=" * 80)
    
    bass_notes = []
    melody_notes = []
    total_notes = 0
    
    try:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                note = int(row['note'])
                voice = row['voice']
                total_notes += 1
                
                if voice == 'bass':
                    bass_notes.append(note)
                elif voice == 'melodic':
                    melody_notes.append(note)
        
        if not bass_notes:
            print("❌ No bass notes found in log")
            return None
        
        # Calculate statistics
        bass_counter = Counter(bass_notes)
        melody_counter = Counter(melody_notes)
        
        total_bass = len(bass_notes)
        total_melody = len(melody_notes)
        unique_bass = len(bass_counter)
        unique_melody = len(melody_counter)
        
        # Find most common bass note
        most_common_bass, most_common_count = bass_counter.most_common(1)[0]
        most_common_pct = (most_common_count / total_bass) * 100
        
        # Find consecutive repetitions
        max_consecutive = 1
        current_consecutive = 1
        consecutive_sequences = []
        
        for i in range(1, len(bass_notes)):
            if bass_notes[i] == bass_notes[i-1]:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                if current_consecutive > 2:  # Track sequences > 2
                    consecutive_sequences.append((bass_notes[i-1], current_consecutive))
                current_consecutive = 1
        
        # Print results
        print(f"\n📊 OVERALL STATISTICS:")
        print(f"   Total notes: {total_notes}")
        print(f"   Bass notes: {total_bass} ({total_bass/total_notes*100:.1f}%)")
        print(f"   Melody notes: {total_melody} ({total_melody/total_notes*100:.1f}%)")
        
        print(f"\n🎸 BASS DIVERSITY:")
        print(f"   Unique notes: {unique_bass}")
        print(f"   Variety: {unique_bass/total_bass*100:.1f}%")
        print(f"   Most common: MIDI {most_common_bass} ({most_common_count}x = {most_common_pct:.1f}%)")
        
        print(f"\n🎤 MELODY DIVERSITY:")
        print(f"   Unique notes: {unique_melody}")
        print(f"   Variety: {unique_melody/total_melody*100:.1f}%")
        
        print(f"\n🔁 CONSECUTIVE REPETITIONS:")
        print(f"   Max consecutive: {max_consecutive}")
        if consecutive_sequences:
            print(f"   Sequences >2:")
            for note, count in consecutive_sequences[:5]:  # Show first 5
                print(f"      MIDI {note}: {count} times")
        else:
            print(f"   None found >2")
        
        print(f"\n📈 TOP 5 BASS NOTES:")
        for i, (note, count) in enumerate(bass_counter.most_common(5), 1):
            pct = (count / total_bass) * 100
            print(f"   {i}. MIDI {note}: {count}x ({pct:.1f}%)")
        
        # Compare to Somax baseline
        print(f"\n" + "=" * 80)
        print("COMPARISON TO SOMAX BASELINE (Nov 22, 21:50)")
        print("=" * 80)
        
        baseline_total_bass = 68
        baseline_c2_count = 37
        baseline_c2_pct = 54.4
        baseline_unique = 16
        baseline_max_consecutive = 16
        
        print(f"{'Metric':<30} {'Somax':>15} {'Pre-Somax':>15} {'Change':>15}")
        print("-" * 80)
        print(f"{'Most common note %':<30} {baseline_c2_pct:>14.1f}% {most_common_pct:>14.1f}% {most_common_pct - baseline_c2_pct:>+14.1f}%")
        print(f"{'Unique bass notes':<30} {baseline_unique:>15} {unique_bass:>15} {unique_bass - baseline_unique:>+15}")
        print(f"{'Max consecutive':<30} {baseline_max_consecutive:>15} {max_consecutive:>15} {max_consecutive - baseline_max_consecutive:>+15}")
        print(f"{'Bass variety %':<30} {baseline_unique/baseline_total_bass*100:>14.1f}% {unique_bass/total_bass*100:>14.1f}% {(unique_bass/total_bass - baseline_unique/baseline_total_bass)*100:>+14.1f}%")
        
        # Success criteria
        print(f"\n" + "=" * 80)
        print("SUCCESS CRITERIA")
        print("=" * 80)
        
        success_most_common = most_common_pct < 30.0
        success_unique = unique_bass > 30
        success_consecutive = max_consecutive <= 3
        
        print(f"✓ Most common <30%: {'PASS ✅' if success_most_common else 'FAIL ❌'} ({most_common_pct:.1f}%)")
        print(f"✓ Unique notes >30: {'PASS ✅' if success_unique else 'FAIL ❌'} ({unique_bass})")
        print(f"✓ Max consecutive ≤3: {'PASS ✅' if success_consecutive else 'FAIL ❌'} ({max_consecutive})")
        
        overall_success = success_most_common and success_unique and success_consecutive
        
        print(f"\n{'🎉 OVERALL: SUCCESS' if overall_success else '❌ OVERALL: NEEDS IMPROVEMENT'}")
        
        # Conclusion
        print(f"\n" + "=" * 80)
        print("CONCLUSION")
        print("=" * 80)
        
        if most_common_pct < baseline_c2_pct:
            improvement = baseline_c2_pct - most_common_pct
            print(f"✅ DIVERSITY IMPROVED: Most common note reduced by {improvement:.1f}%")
            print(f"   This suggests SomaxBridge Factor Oracle navigation was contributing")
            print(f"   to the bass C2 repetition problem.")
        else:
            print(f"❌ DIVERSITY NOT IMPROVED: Repetition persists")
            print(f"   This suggests the problem lies in harmonic translation or oracle")
            print(f"   training data, not SomaxBridge navigation.")
        
        return {
            'total_notes': total_notes,
            'bass_notes': total_bass,
            'melody_notes': total_melody,
            'unique_bass': unique_bass,
            'unique_melody': unique_melody,
            'most_common_note': most_common_bass,
            'most_common_pct': most_common_pct,
            'max_consecutive': max_consecutive,
            'success': overall_success
        }
        
    except Exception as e:
        print(f"❌ Error analyzing MIDI log: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("\n" + "🎵" * 40)
    print("AUTONOMOUS GENERATION DIAGNOSTIC TEST")
    print("Pre-Somax Branch (PhraseGenerator Only)")
    print("🎵" * 40 + "\n")
    
    # Run test
    success = run_test(duration_minutes=2)
    
    if not success:
        print("\n❌ Test did not complete successfully")
        sys.exit(1)
    
    # Wait a moment for file writes
    time.sleep(2)
    
    # Find latest MIDI log
    midi_log = find_latest_midi_log()
    
    if not midi_log:
        print("\n❌ Could not find MIDI output log")
        sys.exit(1)
    
    # Analyze results
    results = analyze_bass_diversity(midi_log)
    
    if results:
        print("\n" + "=" * 80)
        print("TEST COMPLETE")
        print("=" * 80)
        sys.exit(0 if results['success'] else 1)
    else:
        print("\n❌ Analysis failed")
        sys.exit(1)
