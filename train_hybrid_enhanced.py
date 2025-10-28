#!/usr/bin/env python3
"""
Enhanced Hybrid Training Pipeline
Integrates hierarchical music analysis into MusicHal 9000

This pipeline now uses:
- Hierarchical multi-timescale analysis (Farbood et al., 2015)
- Perceptual significance filtering (Pressnitzer et al., 2008)
- Predictive processing (Schaefer, 2014)
- Adaptive sampling strategies
"""

import os
import time
import json
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import asdict

from audio_file_learning.polyphonic_processor import PolyphonicAudioProcessor
from audio_file_learning.hybrid_batch_trainer import HybridBatchTrainer
from hybrid_training.music_theory_transformer import MusicTheoryAnalyzer, MusicalInsights
from simple_hierarchical_integration import SimpleHierarchicalAnalyzer, SimpleAnalysisResult

class EnhancedHybridTrainingPipeline:
    """
    Enhanced hybrid training pipeline with hierarchical music analysis
    
    This pipeline combines:
    1. Hierarchical analysis (multi-timescale, perceptual, adaptive sampling)
    2. Music theory transformer analysis
    3. AudioOracle training with enhanced insights
    """
    
    def __init__(self, 
                 transformer_model_path: Optional[str] = None,
                 cpu_threshold: int = 5000,
                 max_events: int = 10000,
                 enable_hierarchical: bool = True):
        """
        Initialize enhanced hybrid training pipeline
        
        Args:
            transformer_model_path: Path to pre-trained transformer model
            cpu_threshold: Threshold for CPU vs MPS selection
            max_events: Maximum number of events to sample
            enable_hierarchical: Enable hierarchical analysis
        """
        print("🎵 Initializing Enhanced Hybrid Training Pipeline...")
        
        # Initialize components
        self.processor = PolyphonicAudioProcessor()
        self.hybrid_trainer = HybridBatchTrainer(cpu_threshold=cpu_threshold)
        self.transformer_analyzer = MusicTheoryAnalyzer(transformer_model_path)
        
        # Initialize hierarchical analyzer
        if enable_hierarchical:
            self.hierarchical_analyzer = SimpleHierarchicalAnalyzer(max_events=max_events)
            print("✅ Hierarchical analysis enabled")
        else:
            self.hierarchical_analyzer = None
            print("⚠️  Hierarchical analysis disabled")
        
        # Training statistics
        self.training_stats = {
            'total_events': 0,
            'hierarchical_analysis_time': 0.0,
            'transformer_analysis_time': 0.0,
            'audio_oracle_training_time': 0.0,
            'total_training_time': 0.0,
            'hierarchical_insights_applied': False,
            'enhancement_applied': False
        }
        
        print("✅ Enhanced Hybrid Training Pipeline initialized")
    
    def train_from_audio_file(self, 
                             audio_file: str,
                             output_file: str,
                             max_events: Optional[int] = None,
                             training_events: Optional[int] = None,
                             sampling_strategy: str = 'balanced',
                             use_transformer: bool = True,
                             use_hierarchical: bool = True) -> Dict[str, Any]:
        """
        Train AudioOracle from audio file with enhanced analysis
        
        Args:
            audio_file: Path to audio file
            output_file: Path to output JSON file
            max_events: Maximum events to extract (overrides default)
            training_events: Number of events to use for training
            sampling_strategy: Sampling strategy ('balanced', 'structural', 'perceptual')
            use_transformer: Whether to use transformer analysis
            use_hierarchical: Whether to use hierarchical analysis
            
        Returns:
            Training statistics and results
        """
        
        print(f"🎵 Starting Enhanced Hybrid Training from: {audio_file}")
        start_time = time.time()
        
        # Step 1: Hierarchical Analysis (if enabled)
        hierarchical_result = None
        if use_hierarchical and self.hierarchical_analyzer:
            print("\n🔄 Step 1: Hierarchical Music Analysis...")
            hierarchical_start = time.time()
            
            hierarchical_result = self.hierarchical_analyzer.analyze_audio_file(
                audio_file, sampling_strategy=sampling_strategy
            )
            
            self.training_stats['hierarchical_analysis_time'] = time.time() - hierarchical_start
            self.training_stats['hierarchical_insights_applied'] = True
            
            print(f"✅ Hierarchical analysis complete: {len(hierarchical_result.sampled_events)} events sampled")
        
        # Step 2: Extract Features (using hierarchical insights if available)
        print("\n🔄 Step 2: Feature Extraction...")
        
        if hierarchical_result:
            # Use hierarchical sampling results
            events = self._convert_hierarchical_events_to_training_format(hierarchical_result.sampled_events)
            print(f"✅ Using {len(events)} hierarchically sampled events")
        else:
            # Fallback to traditional processing
            events = self.processor.extract_features(audio_file, max_events=max_events)
            print(f"✅ Extracted {len(events)} events using traditional processing")
        
        # Limit events for training if specified
        if training_events and len(events) > training_events:
            # Sample events evenly across the timeline
            indices = np.linspace(0, len(events)-1, training_events, dtype=int)
            events = [events[i] for i in indices]
            print(f"✅ Limited to {len(events)} events for training")
        
        self.training_stats['total_events'] = len(events)
        
        # Step 3: Transformer Analysis (if enabled)
        transformer_insights = None
        if use_transformer:
            print("\n🔄 Step 3: Music Theory Transformer Analysis...")
            transformer_start = time.time()
            
            transformer_insights = self.transformer_analyzer.analyze_audio_features(events)
            
            self.training_stats['transformer_analysis_time'] = time.time() - transformer_start
            self.training_stats['enhancement_applied'] = True
            
            print(f"✅ Transformer analysis complete")
        
        # Step 4: Enhanced AudioOracle Training
        print("\n🔄 Step 4: Enhanced AudioOracle Training...")
        training_start = time.time()
        
        # Create enhanced training data
        enhanced_events = self._enhance_events_with_insights(
            events, transformer_insights, hierarchical_result
        )
        
        # Train AudioOracle
        training_success = self.hybrid_trainer.train_from_events(
            enhanced_events, 
            file_info={'transformer_insights': transformer_insights}
        )
        
        # Get training result from AudioOracle
        if training_success:
            # Get statistics from the actual AudioOracle instance
            audio_oracle = self.hybrid_trainer.audio_oracle_mps if self.hybrid_trainer.use_mps else self.hybrid_trainer.audio_oracle_cpu
            training_result = {
                'training_successful': True,
                'audio_oracle_stats': audio_oracle.get_statistics() if audio_oracle else {},
                'events_processed': len(enhanced_events)
            }
        else:
            training_result = {
                'training_successful': False,
                'error': 'Training failed'
            }
        
        self.training_stats['audio_oracle_training_time'] = time.time() - training_start
        
        # Step 5: Save Enhanced Results
        print("\n🔄 Step 5: Saving Enhanced Results...")
        
        enhanced_output = self._create_enhanced_output(
            training_result, transformer_insights, hierarchical_result
        )
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(enhanced_output, f, indent=2, default=self._json_serializer)
        
        self.training_stats['total_training_time'] = time.time() - start_time
        
        # Print summary
        self._print_training_summary()
        
        return enhanced_output
    
    def _convert_hierarchical_events_to_training_format(self, sampled_events: List) -> List[Dict]:
        """Convert hierarchical sampled events to training format"""
        
        events = []
        
        for sampled_event in sampled_events:
            # Convert SampledEvent to training format
            event = {
                't': sampled_event.time,
                'features': sampled_event.features.tolist(),
                'significance_score': sampled_event.significance_score,
                'sampling_reason': sampled_event.sampling_reason,
                'event_id': sampled_event.event_id
            }
            events.append(event)
        
        return events
    
    def _enhance_events_with_insights(self, events: List[Dict], 
                                    transformer_insights: Optional[MusicalInsights],
                                    hierarchical_result: Optional[SimpleAnalysisResult]) -> List[Dict]:
        """Enhance events with transformer and hierarchical insights"""
        
        enhanced_events = []
        
        for event in events:
            enhanced_event = event.copy()
            
            # Add transformer insights if available
            if transformer_insights:
                enhanced_event['transformer_insights'] = {
                    'key_signature': transformer_insights.key_signature,
                    'tempo_analysis': transformer_insights.tempo_analysis,
                    'chord_progression': transformer_insights.chord_progression,
                    'musical_form': transformer_insights.musical_form,
                    'scale_analysis': transformer_insights.scale_analysis
                }
            
            # Add hierarchical insights if available
            if hierarchical_result:
                # Find corresponding hierarchical information
                event_time = event.get('t', 0)
                
                # Add stream information
                stream_info = self._find_stream_for_time(event_time, hierarchical_result.auditory_streams)
                if stream_info:
                    enhanced_event['stream_info'] = stream_info
                
                # Add hierarchical level information
                hierarchical_level = self._determine_hierarchical_level(event_time, hierarchical_result)
                enhanced_event['hierarchical_level'] = hierarchical_level
            
            enhanced_events.append(enhanced_event)
        
        return enhanced_events
    
    def _find_stream_for_time(self, time: float, streams: List) -> Optional[Dict]:
        """Find stream information for a given time"""
        
        for stream in streams:
            if stream.start_time <= time <= stream.end_time:
                return {
                    'stream_id': stream.stream_id,
                    'stream_type': stream.stream_type,
                    'confidence': stream.confidence
                }
        
        return None
    
    def _determine_hierarchical_level(self, time: float, hierarchical_result: SimpleAnalysisResult) -> str:
        """Determine hierarchical level for a given time"""
        
        # Simple heuristic based on time position
        duration = hierarchical_result.duration
        position = time / duration
        
        if position < 0.1 or position > 0.9:
            return 'section_boundary'
        elif 0.4 < position < 0.6:
            return 'phrase_center'
        else:
            return 'measure_level'
    
    def _create_enhanced_output(self, training_result: Dict, 
                              transformer_insights: Optional[MusicalInsights],
                              hierarchical_result: Optional[SimpleAnalysisResult]) -> Dict[str, Any]:
        """Create enhanced output with all insights"""
        
        enhanced_output = training_result.copy()
        
        # Add transformer insights
        if transformer_insights:
            enhanced_output['transformer_analysis'] = {
                'key_signature': transformer_insights.key_signature,
                'tempo_analysis': transformer_insights.tempo_analysis,
                'chord_progression': transformer_insights.chord_progression,
                'musical_form': transformer_insights.musical_form,
                'scale_analysis': transformer_insights.scale_analysis,
                'confidence_scores': transformer_insights.confidence_scores
            }
        
        # Add hierarchical analysis
        if hierarchical_result:
            enhanced_output['hierarchical_analysis'] = {
                'duration': hierarchical_result.duration,
                'hierarchical_structure': {
                    'sections': len(hierarchical_result.hierarchical_structure.sections),
                    'phrases': len(hierarchical_result.hierarchical_structure.phrases),
                    'measures': len(hierarchical_result.hierarchical_structure.measures)
                },
                'auditory_streams': [
                    {
                        'stream_id': stream.stream_id,
                        'start_time': stream.start_time,
                        'end_time': stream.end_time,
                        'stream_type': stream.stream_type,
                        'confidence': stream.confidence
                    }
                    for stream in hierarchical_result.auditory_streams
                ],
                'sampled_events_count': len(hierarchical_result.sampled_events),
                'sampling_strategy': 'hierarchical_adaptive'
            }
        
        # Add training statistics
        enhanced_output['training_statistics'] = self.training_stats
        
        return enhanced_output
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for numpy types"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    def _print_training_summary(self):
        """Print training summary"""
        
        print("\n" + "="*60)
        print("🎵 Enhanced Hybrid Training Summary")
        print("="*60)
        
        print(f"📊 Total Events: {self.training_stats['total_events']}")
        print(f"⏱️  Total Time: {self.training_stats['total_training_time']:.2f}s")
        
        if self.training_stats['hierarchical_insights_applied']:
            print(f"🏗️  Hierarchical Analysis: {self.training_stats['hierarchical_analysis_time']:.2f}s")
        
        if self.training_stats['enhancement_applied']:
            print(f"🧠 Transformer Analysis: {self.training_stats['transformer_analysis_time']:.2f}s")
        
        print(f"🎯 AudioOracle Training: {self.training_stats['audio_oracle_training_time']:.2f}s")
        
        print("="*60)

def main():
    """Test the enhanced hybrid training pipeline"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Hybrid Training Pipeline')
    parser.add_argument('--file', required=True, help='Audio file to process')
    parser.add_argument('--output', required=True, help='Output JSON file')
    parser.add_argument('--max-events', type=int, default=10000, help='Maximum events to extract')
    parser.add_argument('--training-events', type=int, help='Number of events for training')
    parser.add_argument('--sampling-strategy', choices=['balanced', 'structural', 'perceptual'], 
                       default='balanced', help='Sampling strategy')
    parser.add_argument('--no-transformer', action='store_true', help='Disable transformer analysis')
    parser.add_argument('--no-hierarchical', action='store_true', help='Disable hierarchical analysis')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = EnhancedHybridTrainingPipeline(
        max_events=args.max_events,
        enable_hierarchical=not args.no_hierarchical
    )
    
    # Train from audio file
    result = pipeline.train_from_audio_file(
        audio_file=args.file,
        output_file=args.output,
        max_events=args.max_events,
        training_events=args.training_events,
        sampling_strategy=args.sampling_strategy,
        use_transformer=not args.no_transformer,
        use_hierarchical=not args.no_hierarchical
    )
    
    print(f"\n✅ Enhanced hybrid training complete!")
    print(f"📁 Results saved to: {args.output}")

if __name__ == "__main__":
    main()
