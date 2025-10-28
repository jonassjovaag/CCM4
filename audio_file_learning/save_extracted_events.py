# audio_file_learning/save_extracted_events.py
# Script to save extracted events for reuse without re-extraction

import os
import sys
import pickle
import json
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_file_learning.file_processor import AudioFileProcessor, Event, AudioFileInfo


def save_extracted_events(events: List[Event], file_info: AudioFileInfo, output_path: str) -> bool:
    """
    Save extracted events to a pickle file for later reuse
    
    Args:
        events: List of extracted Event objects
        file_info: AudioFileInfo object
        output_path: Path to save the events
        
    Returns:
        True if save successful
    """
    try:
        # Create data structure to save
        data = {
            'events': events,
            'file_info': file_info,
            'total_events': len(events),
            'extraction_timestamp': time.time()
        }
        
        # Save as pickle (preserves object types)
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"✅ Extracted events saved to: {output_path}")
        print(f"📊 Saved {len(events)} events from {file_info.duration:.2f}s audio")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving extracted events: {e}")
        return False


def load_extracted_events(input_path: str) -> tuple:
    """
    Load extracted events from pickle file
    
    Args:
        input_path: Path to the saved events file
        
    Returns:
        Tuple of (events, file_info) or (None, None) if failed
    """
    try:
        with open(input_path, 'rb') as f:
            data = pickle.load(f)
        
        events = data['events']
        file_info = data['file_info']
        
        print(f"✅ Extracted events loaded from: {input_path}")
        print(f"📊 Loaded {len(events)} events from {file_info.duration:.2f}s audio")
        
        return events, file_info
        
    except Exception as e:
        print(f"❌ Error loading extracted events: {e}")
        return None, None


def main():
    """Main function to extract and save events"""
    import argparse
    import time
    
    parser = argparse.ArgumentParser(description='Extract and save audio events for reuse')
    parser.add_argument('--file', '-f', required=True, help='Audio file to process')
    parser.add_argument('--output', '-o', required=True, help='Output pickle file for events')
    parser.add_argument('--max-events', type=int, default=None, help='Maximum events to extract')
    parser.add_argument('--sample-rate', type=int, default=44100, help='Sample rate')
    parser.add_argument('--hop-length', type=int, default=256, help='Hop length')
    parser.add_argument('--frame-length', type=int, default=2048, help='Frame length')
    
    args = parser.parse_args()
    
    print(f"🎵 Extracting events from: {args.file}")
    print(f"💾 Will save to: {args.output}")
    
    # Initialize processor
    processor = AudioFileProcessor(
        sample_rate=args.sample_rate,
        hop_length=args.hop_length,
        frame_length=args.frame_length,
        max_events=args.max_events
    )
    
    # Process audio file
    events, file_info = processor.process_audio_file(args.file)
    
    if events:
        # Save extracted events
        success = save_extracted_events(events, file_info, args.output)
        if success:
            print(f"✅ Successfully extracted and saved {len(events)} events")
        else:
            print(f"❌ Failed to save events")
    else:
        print(f"❌ No events extracted")


if __name__ == "__main__":
    main()
