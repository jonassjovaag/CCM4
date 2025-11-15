#!/usr/bin/env python3
"""
Extract and Cache Audio Events

One-time extraction script that runs ONLY Stage 1 (AudioExtraction) and saves
the extracted events to a pickle file for reuse in subsequent training runs.

This dramatically speeds up testing by allowing you to:
1. Extract events once (spend the time)
2. Save to cache
3. Reuse for all subsequent test runs while debugging Stages 2-5

Usage:
    # Extract 1000 events for testing
    python scripts/train/extract_and_cache_events.py input_audio/Bend_like.wav \
        --max-events 1000 \
        --output cache/events/Bend_like_1000.pkl

    # Extract all events for final training
    python scripts/train/extract_and_cache_events.py input_audio/Bend_like.wav \
        --output cache/events/Bend_like_full.pkl

    # Then use with train_modular.py:
    python scripts/train/train_modular.py input_audio/Bend_like.wav \
        --cached-events cache/events/Bend_like_1000.pkl \
        --profile quick_test
"""

import sys
import os
import argparse
import pickle
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from musichal.training.pipeline.stages.audio_extraction_stage import AudioExtractionStage

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_and_cache(audio_path: str, output_path: str, max_events: int | None = None):
    """
    Extract audio events and cache to pickle file.
    
    Args:
        audio_path: Path to input audio file
        output_path: Path to output pickle file
        max_events: Optional maximum number of events to extract
    """
    # Validate input
    if not os.path.exists(audio_path):
        logger.error(f"Audio file not found: {audio_path}")
        sys.exit(1)
    
    # Create output directory if needed
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build config
    config = {
        'audio': {
            'file_path': audio_path
        }
    }
    
    if max_events is not None:
        config['training'] = {'max_events': max_events}
        logger.info(f"Limiting extraction to {max_events} events")
    
    # Create context (simple dict, not a class)
    context = {
        'audio_file': audio_path,
        'config': config
    }
    
    # Run extraction stage
    logger.info(f"Extracting events from: {audio_path}")
    stage = AudioExtractionStage(config.get('audio', {}))
    
    try:
        result = stage.run(context)
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        raise
    
    # Get extracted events from result
    events = result.data.get('audio_events')
    
    if not events:
        logger.error("No events extracted!")
        sys.exit(1)
    
    # Save to pickle
    logger.info(f"Saving {len(events)} events to: {output_path}")
    
    cache_data = {
        'audio_path': audio_path,
        'events': events,
        'max_events': max_events,
        'total_events': len(events)
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Report file size
    file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    logger.info(f"Cache file created: {file_size_mb:.2f} MB")
    logger.info(f"✅ Extraction complete! {len(events)} events cached.")
    
    return len(events)


def main():
    parser = argparse.ArgumentParser(
        description="Extract and cache audio events for rapid testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract 1000 events for testing
  python scripts/train/extract_and_cache_events.py input_audio/Bend_like.wav \\
      --max-events 1000 \\
      --output cache/events/Bend_like_1000.pkl

  # Extract all events (no limit)
  python scripts/train/extract_and_cache_events.py input_audio/Bend_like.wav \\
      --output cache/events/Bend_like_full.pkl

Then use with train_modular.py:
  python scripts/train/train_modular.py input_audio/Bend_like.wav \\
      --cached-events cache/events/Bend_like_1000.pkl \\
      --profile quick_test
        """
    )
    
    parser.add_argument(
        'audio_file',
        help='Path to input audio file'
    )
    
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Path to output pickle file (e.g., cache/events/audio_1000.pkl)'
    )
    
    parser.add_argument(
        '--max-events',
        type=int,
        default=None,
        help='Maximum number of events to extract (omit for unlimited)'
    )
    
    args = parser.parse_args()
    
    # Extract and cache
    extract_and_cache(
        audio_path=args.audio_file,
        output_path=args.output,
        max_events=args.max_events
    )


if __name__ == '__main__':
    main()
