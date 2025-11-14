#!/usr/bin/env python3
"""
Simplified Hierarchical Music Analysis Integration
A working version that combines the essential components

This module integrates:
- Multi-timescale analysis
- Perceptual significance filtering  
- Adaptive sampling strategies
- Basic predictive processing
"""

import numpy as np
import librosa
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import json
import time

# Import working components
from hierarchical_analysis.fixed_multi_timescale_analyzer import FixedMultiTimescaleAnalyzer, MusicalPattern, HierarchicalStructure
from perceptual_filtering.simple_significance_filter import SimpleAuditorySceneAnalyzer, AuditoryStream
from adaptive_sampling.smart_sampler import SmartSampler, SampledEvent

@dataclass
class SimpleAnalysisResult:
    """Simplified analysis result"""
    audio_file: str
    duration: float
    hierarchical_structure: HierarchicalStructure
    auditory_streams: List[AuditoryStream]
    sampled_events: List[SampledEvent]
    analysis_metadata: Dict[str, any]

class SimpleHierarchicalAnalyzer:
    """
    Simplified hierarchical analyzer that actually works
    """
    
    def __init__(self, max_events: int = 10000):
        self.max_events = max_events
        
        # Initialize working components
        self.multi_timescale_analyzer = FixedMultiTimescaleAnalyzer()
        self.auditory_scene_analyzer = SimpleAuditorySceneAnalyzer()
        self.smart_sampler = SmartSampler(max_events=max_events)
        
        # Analysis history
        self.analysis_history = []
        
    def analyze_audio_file(self, audio_file: str, 
                          sampling_strategy: str = 'balanced') -> SimpleAnalysisResult:
        """
        Perform simplified hierarchical analysis
        
        Args:
            audio_file: Path to audio file
            sampling_strategy: Sampling strategy ('balanced', 'structural', 'perceptual')
            
        Returns:
            Analysis result
        """
        
        print(f"🎵 Starting simplified hierarchical analysis of: {audio_file}")
        start_time = time.time()
        
        # Load audio to get duration
        y, sr = librosa.load(audio_file, sr=22050)
        duration = len(y) / sr
        
        print(f"📊 Audio duration: {duration:.2f} seconds")
        
        # Step 1: Multi-timescale analysis
        print("\n🔄 Step 1: Multi-timescale analysis...")
        hierarchical_structure = self.multi_timescale_analyzer.analyze_hierarchical_structure(audio_file)
        
        # Step 2: Auditory scene analysis
        print("\n🔄 Step 2: Auditory scene analysis...")
        auditory_streams = self.auditory_scene_analyzer.analyze_auditory_scene(audio_file)
        
        # Step 3: Adaptive sampling
        print(f"\n🔄 Step 3: Adaptive sampling ({sampling_strategy})...")
        sampled_events = self.smart_sampler.sample_events(audio_file, sampling_strategy)
        
        # Step 4: Create metadata
        print("\n🔄 Step 4: Creating analysis metadata...")
        analysis_metadata = self._create_analysis_metadata(
            hierarchical_structure, auditory_streams, sampled_events
        )
        
        # Create result
        result = SimpleAnalysisResult(
            audio_file=audio_file,
            duration=duration,
            hierarchical_structure=hierarchical_structure,
            auditory_streams=auditory_streams,
            sampled_events=sampled_events,
            analysis_metadata=analysis_metadata
        )
        
        # Store in history
        self.analysis_history.append(result)
        
        analysis_time = time.time() - start_time
        print(f"\n✅ Simplified hierarchical analysis complete in {analysis_time:.2f} seconds")
        
        return result
    
    def _create_analysis_metadata(self, hierarchical_structure: HierarchicalStructure,
                                 auditory_streams: List[AuditoryStream],
                                 sampled_events: List[SampledEvent]) -> Dict[str, any]:
        """Create analysis metadata"""
        
        metadata = {
            'analysis_timestamp': time.time(),
            'hierarchical_structure': {
                'sections': len(hierarchical_structure.sections),
                'phrases': len(hierarchical_structure.phrases),
                'measures': len(hierarchical_structure.measures)
            },
            'auditory_streams': {
                'total_streams': len(auditory_streams),
                'stream_types': {}
            },
            'sampled_events': {
                'total_events': len(sampled_events),
                'sampling_reasons': {}
            }
        }
        
        # Count stream types
        for stream in auditory_streams:
            stream_type = stream.stream_type
            metadata['auditory_streams']['stream_types'][stream_type] = \
                metadata['auditory_streams']['stream_types'].get(stream_type, 0) + 1
        
        # Count sampling reasons
        for event in sampled_events:
            reason = event.sampling_reason
            metadata['sampled_events']['sampling_reasons'][reason] = \
                metadata['sampled_events']['sampling_reasons'].get(reason, 0) + 1
        
        return metadata
    
    def get_analysis_summary(self, result: SimpleAnalysisResult) -> str:
        """Get analysis summary"""
        
        summary = []
        summary.append("🎵 Simplified Hierarchical Music Analysis Summary")
        summary.append("=" * 60)
        summary.append(f"📁 File: {result.audio_file}")
        summary.append(f"⏱️  Duration: {result.duration:.2f} seconds")
        
        # Hierarchical structure
        summary.append(f"\n🏗️  Hierarchical Structure:")
        summary.append(f"   Sections: {len(result.hierarchical_structure.sections)}")
        summary.append(f"   Phrases: {len(result.hierarchical_structure.phrases)}")
        summary.append(f"   Measures: {len(result.hierarchical_structure.measures)}")
        
        # Auditory streams
        if result.auditory_streams:
            summary.append(f"\n🎧 Auditory Streams: {len(result.auditory_streams)}")
            stream_types = {}
            for stream in result.auditory_streams:
                stream_types[stream.stream_type] = stream_types.get(stream.stream_type, 0) + 1
            
            for stream_type, count in stream_types.items():
                summary.append(f"   {stream_type}: {count}")
        
        # Sampled events
        summary.append(f"\n🎯 Sampled Events: {len(result.sampled_events)}")
        sampling_reasons = {}
        for event in result.sampled_events:
            sampling_reasons[event.sampling_reason] = sampling_reasons.get(event.sampling_reason, 0) + 1
        
        for reason, count in sampling_reasons.items():
            summary.append(f"   {reason}: {count}")
        
        return "\n".join(summary)
    
    def save_analysis_result(self, result: SimpleAnalysisResult, 
                           output_file: str) -> bool:
        """Save analysis result to file"""
        
        try:
            # Convert result to serializable format
            serializable_result = {
                'audio_file': result.audio_file,
                'duration': result.duration,
                'analysis_metadata': result.analysis_metadata,
                'hierarchical_structure': {
                    'sections': len(result.hierarchical_structure.sections),
                    'phrases': len(result.hierarchical_structure.phrases),
                    'measures': len(result.hierarchical_structure.measures)
                },
                'auditory_streams': [
                    {
                        'stream_id': stream.stream_id,
                        'start_time': stream.start_time,
                        'end_time': stream.end_time,
                        'stream_type': stream.stream_type,
                        'confidence': stream.confidence
                    }
                    for stream in result.auditory_streams
                ],
                'sampled_events': [
                    {
                        'event_id': event.event_id,
                        'time': event.time,
                        'significance_score': event.significance_score,
                        'sampling_reason': event.sampling_reason
                    }
                    for event in result.sampled_events
                ]
            }
            
            with open(output_file, 'w') as f:
                json.dump(serializable_result, f, indent=2)
            
            return True
            
        except Exception as e:
            print(f"Error saving analysis result: {e}")
            return False

def main():
    """Test the simplified hierarchical analyzer"""
    
    print("🚀 Starting simplified hierarchical analyzer test...")
    
    analyzer = SimpleHierarchicalAnalyzer(max_events=2000)  # Smaller for testing
    
    # Test with sample audio file
    audio_file = "input_audio/Grab-a-hold.mp3"
    
    try:
        # Perform analysis
        result = analyzer.analyze_audio_file(audio_file, sampling_strategy='balanced')
        
        # Show summary
        print("\n" + analyzer.get_analysis_summary(result))
        
        # Save result
        output_file = "simple_hierarchical_analysis.json"
        save_success = analyzer.save_analysis_result(result, output_file)
        print(f"\n💾 Analysis result saved: {save_success}")
        
        # Show some examples
        print(f"\n📊 Analysis Examples:")
        
        # Show top sampled events
        print(f"   Top 5 sampled events:")
        for i, event in enumerate(result.sampled_events[:5]):
            print(f"     {i+1}. {event.event_id}: {event.time:.2f}s, "
                  f"score: {event.significance_score:.3f}, "
                  f"reason: {event.sampling_reason}")
        
        # Show auditory streams
        if result.auditory_streams:
            print(f"   Auditory streams:")
            for i, stream in enumerate(result.auditory_streams[:3]):
                print(f"     {i+1}. Stream {stream.stream_id}: {stream.stream_type} "
                      f"({stream.start_time:.1f}-{stream.end_time:.1f}s, "
                      f"conf: {stream.confidence:.3f})")
        
        print(f"\n✅ Simplified hierarchical analysis test complete!")
        
    except FileNotFoundError:
        print(f"❌ Audio file not found: {audio_file}")
        print("Please provide a valid audio file path")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
