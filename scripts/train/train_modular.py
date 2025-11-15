#!/usr/bin/env python3
"""
Modular Training Script
Uses the new modular training pipeline.
Part of Phase 2.2: Code Organization
"""

import argparse
import sys
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Fix Unicode encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from musichal.core import ConfigManager
from musichal.training import TrainingOrchestrator


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train MusicHal 9000 using modular pipeline"
    )

    parser.add_argument(
        'audio_file',
        type=Path,
        help="Path to input audio file"
    )

    parser.add_argument(
        '-o', '--output',
        type=Path,
        help="Output file path (default: JSON/<audio_name>.json)"
    )

    parser.add_argument(
        '--profile',
        type=str,
        default=None,
        help="Configuration profile (quick_test, full_training, etc.)"
    )

    parser.add_argument(
        '--checkpoint-dir',
        type=Path,
        help="Directory to save stage checkpoints"
    )

    parser.add_argument(
        '--max-events',
        type=int,
        help="Maximum number of events to extract (overrides config)"
    )

    parser.add_argument(
        '--training-events',
        type=int,
        help="Maximum events for oracle training (subset of max-events)"
    )

    parser.add_argument(
        '--no-hierarchical',
        action='store_true',
        help="Disable hierarchical analysis"
    )

    parser.add_argument(
        '--no-rhythmic',
        action='store_true',
        help="Disable rhythmic analysis"
    )

    parser.add_argument(
        '--cached-events',
        type=Path,
        help="Path to cached events pickle file (skips audio extraction)"
    )

    args = parser.parse_args()

    # Validate input
    if not args.audio_file.exists():
        print(f"Error: Audio file not found: {args.audio_file}")
        return 1

    # Load configuration
    print(f"Loading configuration...")

    config = ConfigManager()
    config.load(profile=args.profile)

    if args.profile:
        print(f"Using profile: {args.profile}")

    # Apply overrides
    if args.max_events:
        config.set('audio_oracle.training.max_events', args.max_events)

    if args.training_events:
        config.set('audio_oracle.training.training_events', args.training_events)
        # Ensure max_events is at least as large as training_events
        if args.max_events and args.training_events > args.max_events:
            print(f"Warning: --training-events ({args.training_events}) > --max-events ({args.max_events})")
            print(f"         Increasing max-events to {args.training_events}")
            config.set('audio_oracle.training.max_events', args.training_events)

    if args.no_hierarchical:
        config.set('hierarchical_analysis.enabled', False)

    if args.no_rhythmic:
        config.set('audio_oracle.enable_rhythmic', False)

    # Determine output path
    if args.output:
        output_file = args.output
    else:
        output_dir = Path("JSON")
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / f"{args.audio_file.stem}.json"

    print(f"Input:  {args.audio_file}")
    print(f"Output: {output_file}")
    print()

    # Create orchestrator
    orchestrator = TrainingOrchestrator(config.to_dict())

    # Run pipeline
    try:
        results = orchestrator.run(
            audio_file=args.audio_file,
            output_file=output_file,
            checkpoint_dir=args.checkpoint_dir,
            cached_events_file=args.cached_events
        )

        # Print summary
        orchestrator.print_summary()

        # Print key results
        print("\nKey Results:")
        print(f"  Training successful: {results.get('training_successful', False)}")
        print(f"  Events processed:    {results.get('events_processed', 0)}")

        if 'audio_oracle_stats' in results:
            stats = results['audio_oracle_stats']
            print(f"  Total states:        {stats.get('total_states', 0)}")
            print(f"  Total patterns:      {stats.get('total_patterns', 0)}")

        print(f"\nResults saved to: {output_file}")

        return 0

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
