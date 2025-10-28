#!/usr/bin/env python3
"""
Analyze conversation logs to understand AI responsiveness
"""

import sys
import pandas as pd
from pathlib import Path

def analyze_conversation(log_file):
    """Analyze a conversation log CSV file"""
    
    # Read CSV
    df = pd.read_csv(log_file)
    
    # Convert numeric columns
    df['elapsed_time'] = pd.to_numeric(df['elapsed_time'], errors='coerce')
    df['activity_level'] = pd.to_numeric(df['activity_level'], errors='coerce')
    
    print(f"\n{'='*80}")
    print(f"CONVERSATION ANALYSIS: {Path(log_file).name}")
    print(f"{'='*80}\n")
    
    # Basic stats
    duration = df['elapsed_time'].max()
    input_count = len(df[df['event_type'] == 'INPUT'])
    output_count = len(df[df['event_type'] == 'OUTPUT'])
    
    print(f"📊 OVERALL STATS:")
    print(f"   Duration: {duration:.1f} seconds")
    print(f"   Human inputs logged: {input_count}")
    print(f"   AI outputs: {output_count}")
    
    if output_count > 0:
        outputs = df[df['event_type'] == 'OUTPUT']
        melody_count = len(outputs[outputs['voice'] == 'melodic'])
        bass_count = len(outputs[outputs['voice'] == 'bass'])
        
        print(f"\n🎵 AI VOICE BREAKDOWN:")
        print(f"   Melodic: {melody_count} notes ({melody_count/output_count*100:.1f}%)")
        print(f"   Bass: {bass_count} notes ({bass_count/output_count*100:.1f}%)")
        
        # Mode breakdown
        print(f"\n🎭 AI MODE BREAKDOWN:")
        for mode in outputs['mode'].unique():
            count = len(outputs[outputs['mode'] == mode])
            print(f"   {mode}: {count} notes ({count/output_count*100:.1f}%)")
        
        # Response timing
        print(f"\n⏱️  RESPONSE TIMING:")
        print(f"   First AI response: {outputs.iloc[0]['elapsed_time']:.1f}s")
        print(f"   Last AI response: {outputs.iloc[-1]['elapsed_time']:.1f}s")
        
        # Calculate gaps between responses
        response_times = outputs['elapsed_time'].values
        if len(response_times) > 1:
            gaps = [response_times[i+1] - response_times[i] for i in range(len(response_times)-1)]
            avg_gap = sum(gaps) / len(gaps)
            min_gap = min(gaps)
            max_gap = max(gaps)
            print(f"   Average gap: {avg_gap:.2f}s")
            print(f"   Min gap: {min_gap:.2f}s")
            print(f"   Max gap: {max_gap:.2f}s")
    
    # Activity level analysis
    print(f"\n📈 ACTIVITY ANALYSIS:")
    inputs = df[df['event_type'] == 'INPUT']
    if len(inputs) > 0:
        avg_activity = inputs['activity_level'].mean()
        max_activity = inputs['activity_level'].max()
        print(f"   Average activity: {avg_activity:.2f}")
        print(f"   Max activity: {max_activity:.2f}")
        
        # Count mode transitions
        auto_count = len(inputs[inputs['mode'] == 'AUTO'])
        listen_count = len(inputs[inputs['mode'] == 'LISTEN'])
        print(f"   AUTO mode: {auto_count} samples ({auto_count/len(inputs)*100:.1f}%)")
        print(f"   LISTEN mode: {listen_count} samples ({listen_count/len(inputs)*100:.1f}%)")
    
    # Timeline view (every 5 seconds)
    print(f"\n📅 TIMELINE (5-second bins):")
    print(f"   Time | Human Active | AI Melody | AI Bass | Mode")
    print(f"   -----|--------------|-----------|---------|------")
    
    max_time = int(duration) + 1
    for t in range(0, max_time, 5):
        bin_df = df[(df['elapsed_time'] >= t) & (df['elapsed_time'] < t + 5)]
        
        human_active = len(bin_df[(bin_df['event_type'] == 'INPUT') & (bin_df['activity_level'] > 0.3)])
        ai_melody = len(bin_df[(bin_df['event_type'] == 'OUTPUT') & (bin_df['voice'] == 'melodic')])
        ai_bass = len(bin_df[(bin_df['event_type'] == 'OUTPUT') & (bin_df['voice'] == 'bass')])
        
        # Get mode (from last input in bin)
        mode = 'N/A'
        inputs_in_bin = bin_df[bin_df['event_type'] == 'INPUT']
        if len(inputs_in_bin) > 0:
            mode = inputs_in_bin.iloc[-1]['mode']
        
        human_indicator = '█' * human_active if human_active > 0 else '·'
        melody_indicator = '♪' * ai_melody if ai_melody > 0 else '·'
        bass_indicator = '♫' * ai_bass if ai_bass > 0 else '·'
        
        print(f"   {t:3d}s | {human_indicator:12s} | {melody_indicator:9s} | {bass_indicator:7s} | {mode}")
    
    print(f"\n{'='*80}\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
    else:
        # Find most recent log
        logs_dir = Path("logs")
        if not logs_dir.exists():
            print("❌ No logs/ directory found")
            sys.exit(1)
        
        log_files = sorted(logs_dir.glob("conversation_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not log_files:
            print("❌ No conversation logs found in logs/")
            sys.exit(1)
        
        log_file = log_files[0]
        print(f"📂 Using most recent log: {log_file.name}")
    
    analyze_conversation(log_file)

