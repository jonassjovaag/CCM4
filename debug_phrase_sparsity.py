#!/usr/bin/env python3
"""
Debug Phrase Sparsity - Comprehensive Diagnostic for Autonomous Generation

Runs 3-minute test with full timing instrumentation to identify why phrase
generation is sparse. Captures ALL blocking points and generates analysis report.

Usage:
    python debug_phrase_sparsity.py [--duration MINUTES] [--verbose]
"""

import subprocess
import sys
import time
import argparse
from pathlib import Path
import csv
from collections import Counter, defaultdict


def analyze_timing_log(csv_path: Path):
    """Analyze timing events CSV and generate comprehensive report"""
    
    if not csv_path.exists():
        print(f"❌ Timing log not found: {csv_path}")
        return
    
    # Load events
    events = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            events.append(row)
    
    if not events:
        print("❌ No timing events logged")
        return
    
    print("\n" + "=" * 80)
    print("PHRASE SPARSITY DIAGNOSTIC REPORT")
    print("=" * 80)
    
    # Per-voice analysis
    for voice in ['melodic', 'bass']:
        voice_events = [e for e in events if e['voice'] == voice]
        if not voice_events:
            continue
        
        print(f"\n{voice.upper()} VOICE:")
        print("-" * 80)
        
        # Attempt counts
        attempts = len(voice_events)
        allowed = len([e for e in voice_events if e['decision'] == 'allowed'])
        blocked = len([e for e in voice_events if e['decision'] == 'blocked'])
        
        print(f"  Total attempts: {attempts}")
        print(f"  Allowed: {allowed} ({(allowed/attempts*100):.1f}%)")
        print(f"  Blocked: {blocked} ({(blocked/attempts*100):.1f}%)")
        
        # Blocking reasons
        if blocked > 0:
            blocking_reasons = Counter([e['blocking_reason'] for e in voice_events if e['decision'] == 'blocked'])
            print(f"\n  Blocking reasons:")
            for reason, count in blocking_reasons.most_common():
                pct = (count / blocked) * 100
                print(f"    {reason}: {count}x ({pct:.1f}%)")
            
            # PRIMARY BLOCKER
            primary_reason, primary_count = blocking_reasons.most_common(1)[0]
            primary_pct = (primary_count / blocked) * 100
            print(f"\n  🎯 PRIMARY BLOCKER: {primary_reason} ({primary_count}/{blocked} = {primary_pct:.1f}%)")
        
        # Gap analysis
        gaps = [float(e['gap_since_last_success']) for e in voice_events 
                if float(e['gap_since_last_success']) > 0]
        
        if gaps:
            print(f"\n  Gap statistics (time between successful generations):")
            print(f"    Min: {min(gaps):.2f}s")
            print(f"    Max: {max(gaps):.2f}s")
            print(f"    Avg: {sum(gaps)/len(gaps):.2f}s")
            
            # Gap histogram
            histogram = {
                '0-1s': sum(1 for g in gaps if 0 <= g < 1),
                '1-2s': sum(1 for g in gaps if 1 <= g < 2),
                '2-5s': sum(1 for g in gaps if 2 <= g < 5),
                '5-10s': sum(1 for g in gaps if 5 <= g < 10),
                '>10s': sum(1 for g in gaps if g >= 10)
            }
            
            print(f"\n  Gap distribution:")
            for bin_name, count in histogram.items():
                pct = (count / len(gaps)) * 100 if gaps else 0
                bar = "█" * int(pct / 5)
                print(f"    {bin_name:>6}: {count:3} ({pct:5.1f}%) {bar}")
            
            # Silence periods
            long_gaps = [g for g in gaps if g > 10]
            if long_gaps:
                print(f"\n  ⚠️  {len(long_gaps)} silence periods >10s detected")
                print(f"      Longest: {max(long_gaps):.1f}s")
    
    # Overall analysis
    print("\n" + "=" * 80)
    print("OVERALL ANALYSIS")
    print("=" * 80)
    
    # Calculate notes/minute
    duration = float(events[-1]['timestamp']) - float(events[0]['timestamp'])
    minutes = duration / 60
    
    melody_successes = len([e for e in events if e['voice'] == 'melodic' and e['decision'] == 'allowed'])
    bass_successes = len([e for e in events if e['voice'] == 'bass' and e['decision'] == 'allowed'])
    total_successes = melody_successes + bass_successes
    
    print(f"\nGeneration rate:")
    print(f"  Duration: {duration:.1f}s ({minutes:.2f} minutes)")
    print(f"  Melody: {melody_successes} notes ({melody_successes/minutes:.1f} notes/min)")
    print(f"  Bass: {bass_successes} notes ({bass_successes/minutes:.1f} notes/min)")
    print(f"  Total: {total_successes} notes ({total_successes/minutes:.1f} notes/min)")
    
    # Target comparison
    target_rate = 40  # notes/minute target
    actual_rate = total_successes / minutes
    if actual_rate < target_rate:
        deficit = target_rate - actual_rate
        print(f"\n⚠️  BELOW TARGET: {actual_rate:.1f} notes/min vs {target_rate} target ({deficit:.1f} notes/min deficit)")
    else:
        print(f"\n✅ ABOVE TARGET: {actual_rate:.1f} notes/min vs {target_rate} target")
    
    # Voice correlation
    print(f"\nVoice independence:")
    melody_events = [e for e in events if e['voice'] == 'melodic']
    bass_events = [e for e in events if e['voice'] == 'bass']
    print(f"  Melody attempts: {len(melody_events)}")
    print(f"  Bass attempts: {len(bass_events)}")
    ratio = len(melody_events) / len(bass_events) if bass_events else 0
    print(f"  Attempt ratio: {ratio:.2f}:1 (melody:bass)")
    
    if abs(ratio - 1.0) > 0.3:
        print(f"  ⚠️  Voices may not be independent (should be ~1:1)")
    else:
        print(f"  ✅ Voices appear independent")
    
    print("\n" + "=" * 80)
    print(f"Detailed log: {csv_path}")
    print("=" * 80 + "\n")


def run_diagnostic_test(duration_minutes: int = 3, verbose: bool = False):
    """Run MusicHal_9000 with timing instrumentation"""
    
    print("=" * 80)
    print(f"PHRASE SPARSITY DIAGNOSTIC TEST ({duration_minutes} minutes)")
    print("=" * 80)
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Goal: Identify blocking reasons preventing phrase generation\n")
    
    # Enable timing logger via environment variable
    import os
    os.environ['ENABLE_TIMING_LOGGER'] = '1'
    if verbose:
        os.environ['TIMING_LOGGER_VERBOSE'] = '1'
    
    cmd = [
        sys.executable,
        "MusicHal_9000.py",
        "--enable-meld",
        f"--performance-duration={duration_minutes}",
        "--no-wav2vec"  # Fast startup
    ]
    
    print(f"Command: {' '.join(cmd)}\n")
    print("Running test... (timing events will be logged)\n")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent,
            timeout=(duration_minutes * 60) + 45,  # Extra 45s for initialization
            capture_output=True,
            text=True
        )
        
        elapsed = time.time() - start_time
        print(f"\n✓ Test completed in {elapsed:.1f}s")
        
        # Find most recent timing log
        logs_dir = Path("logs")
        timing_logs = sorted(logs_dir.glob("timing_events_*.csv"), 
                           key=lambda f: f.stat().st_mtime, reverse=True)
        
        if timing_logs:
            print(f"\n📊 Analyzing timing log: {timing_logs[0]}")
            analyze_timing_log(timing_logs[0])
        else:
            print("\n❌ No timing log found - logger may not be enabled")
        
        # Check for errors
        if result.returncode != 0:
            print(f"\n⚠️  Warning: Process exited with code {result.returncode}")
            if result.stderr:
                print(f"Stderr:\n{result.stderr[:500]}")
        
    except subprocess.TimeoutExpired:
        print(f"\n⚠️  Test timed out after {(duration_minutes * 60) + 45}s")
    except KeyboardInterrupt:
        print(f"\n⚠️  Test interrupted by user")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug phrase generation sparsity")
    parser.add_argument("--duration", type=int, default=3,
                       help="Test duration in minutes (default: 3)")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose timing logger output")
    
    args = parser.parse_args()
    
    run_diagnostic_test(args.duration, args.verbose)
