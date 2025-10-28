#!/usr/bin/env python3
"""
Hybrid Polyphonic Learning Script
Automatically chooses CPU or MPS based on dataset size for optimal performance
"""

import argparse
import os
import time
from typing import List, Dict, Any, Tuple
import torch

from audio_file_learning.polyphonic_processor import PolyphonicAudioProcessor
from audio_file_learning.hybrid_batch_trainer import HybridBatchTrainer


def main():
    """Main entry point for hybrid polyphonic learning"""
    parser = argparse.ArgumentParser(description='Hybrid Polyphonic Audio Learning - Auto CPU/MPS Selection')
    
    # Input options
    parser.add_argument('--file', type=str, help='Single audio file to process')
    parser.add_argument('--dir', type=str, help='Directory containing audio files')
    parser.add_argument('--files', nargs='+', help='Multiple audio files to process')
    
    # Output options
    parser.add_argument('--output', type=str, default='hybrid_model.json', 
                       help='Output model file path')
    parser.add_argument('--stats', action='store_true', 
                       help='Save training statistics')
    
    # Processing options
    parser.add_argument('--max-events', type=int, default=None,
                       help='Maximum number of events to process (for testing)')
    parser.add_argument('--sample-rate', type=int, default=44100,
                       help='Audio sample rate')
    parser.add_argument('--hop-length', type=int, default=256,
                       help='Audio hop length')
    parser.add_argument('--frame-length', type=int, default=2048,
                       help='Audio frame length')
    
    # AudioOracle parameters
    parser.add_argument('--distance-threshold', type=float, default=0.15,
                       help='Distance threshold for similarity')
    parser.add_argument('--distance-function', type=str, default='euclidean',
                       choices=['euclidean', 'cosine', 'manhattan', 'chebyshev'],
                       help='Distance function to use')
    parser.add_argument('--chord-weight', type=float, default=0.3,
                       help='Chord similarity weight')
    
    # Hybrid system parameters
    parser.add_argument('--cpu-threshold', type=int, default=5000,
                       help='Switch to CPU when dataset exceeds this size')
    parser.add_argument('--force-cpu', action='store_true',
                       help='Force CPU-only processing')
    parser.add_argument('--force-mps', action='store_true',
                       help='Force MPS GPU processing')
    
    args = parser.parse_args()
    
    print("🎵 Hybrid Polyphonic Learning System")
    print("=" * 50)
    
    # Determine input files
    input_files = []
    if args.file:
        input_files.append(args.file)
    elif args.dir:
        if os.path.isdir(args.dir):
            for file in os.listdir(args.dir):
                if file.lower().endswith(('.wav', '.mp3', '.flac', '.m4a')):
                    input_files.append(os.path.join(args.dir, file))
        else:
            print(f"❌ Directory not found: {args.dir}")
            return 1
    elif args.files:
        input_files.extend(args.files)
    else:
        print("❌ No input files specified. Use --file, --dir, or --files")
        return 1
    
    print(f"📁 Input files: {len(input_files)}")
    for file in input_files:
        print(f"  • {os.path.basename(file)}")
    
    # Check MPS availability
    mps_available = torch.backends.mps.is_available()
    print(f"\n🎵 Hybrid Configuration:")
    print(f"   MPS Available: {mps_available}")
    print(f"   CPU Threshold: {args.cpu_threshold:,} events")
    
    if args.force_cpu:
        print(f"   Mode: CPU-only (forced)")
    elif args.force_mps and mps_available:
        print(f"   Mode: MPS GPU-only (forced)")
    else:
        print(f"   Mode: Hybrid (auto-select based on dataset size)")
    
    # Initialize components
    print(f"\n🔧 Initializing hybrid components...")
    
    processor = PolyphonicAudioProcessor(
        sample_rate=args.sample_rate,
        hop_length=args.hop_length,
        frame_length=args.frame_length,
        max_events=args.max_events
    )
    
    trainer = HybridBatchTrainer(
        distance_threshold=args.distance_threshold,
        distance_function=args.distance_function,
        chord_similarity_weight=args.chord_weight,
        cpu_threshold=args.cpu_threshold
    )
    
    print(f"✅ Hybrid components initialized")
    
    # Process files
    print(f"\n🎵 Processing polyphonic audio files...")
    
    all_events = []
    total_duration = 0.0
    
    for file_path in input_files:
        print(f"🎵 Processing: {os.path.basename(file_path)}")
        
        # Process audio file
        events, file_info = processor.process_audio_file(file_path)
        
        if events:
            all_events.extend(events)
            total_duration += getattr(file_info, 'duration', 0.0)
            print(f"✅ Extracted {len(events)} events")
        else:
            print(f"❌ No events extracted from {file_path}")
    
    if not all_events:
        print("❌ No events extracted from any files")
        return 1
    
    print(f"\n✅ Total: {len(all_events)} events, {total_duration:.1f}s duration")
    
    # Determine processing mode
    dataset_size = len(all_events)
    
    if args.force_cpu:
        print(f"💻 Forced CPU processing")
        trainer.cpu_threshold = 0  # Force CPU
    elif args.force_mps and mps_available:
        print(f"🚀 Forced MPS GPU processing")
        trainer.cpu_threshold = float('inf')  # Force MPS
    else:
        print(f"🎯 Auto-selecting optimal device for {dataset_size:,} events")
    
    # Train model
    print(f"\n🎓 Training hybrid model...")
    start_time = time.time()
    
    success = trainer.train_from_events(all_events)
    
    if not success:
        print("❌ Training failed")
        return 1
    
    training_time = time.time() - start_time
    
    # Save model
    print(f"\n💾 Saving hybrid model...")
    model_saved = trainer.save_model(args.output)
    
    if not model_saved:
        print("❌ Failed to save model")
        return 1
    
    # Save statistics
    if args.stats:
        stats_file = args.output.replace('.json', '_stats.json')
        stats = trainer.get_training_stats()
        
        import json
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"📊 Statistics saved to: {stats_file}")
    
    # Print performance summary
    print(trainer.get_performance_summary())
    
    print(f"\n✅ Hybrid learning complete!")
    print(f"📁 Model saved to: {args.output}")
    
    return 0


if __name__ == "__main__":
    exit(main())
