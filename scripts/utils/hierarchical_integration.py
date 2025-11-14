#!/usr/bin/env python3
"""
Hierarchical Music Analysis Integration
Integrates all hierarchical analysis components into MusicHal 9000

This module combines:
- Multi-timescale analysis (Farbood et al., 2015)
- Perceptual significance filtering (Pressnitzer et al., 2008)
- Predictive processing (Schaefer, 2014)
- Adaptive sampling strategies
"""

import numpy as np
import librosa
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import json
import time

# Import hierarchical analysis components
from hierarchical_analysis import (
    MultiTimescaleAnalyzer,
    MusicalPattern,
    HierarchicalStructure,
    MSCOMStructureDetector,
    StructuralBoundary,
    StructuralCommunity,
    TimescaleManager,
    TimescaleConfig
)

from perceptual_filtering.simple_significance_filter import (
    SimpleAuditorySceneAnalyzer,
    AuditoryStream
)

from predictive_processing import (
    PredictiveProcessingEngine,
    MusicalPrediction,
    MentalModelManager
)

from adaptive_sampling import (
    SmartSampler,
    SampledEvent,
    StructuralSampler
)

@dataclass
class HierarchicalAnalysisResult:
    """Comprehensive result of hierarchical analysis"""
    audio_file: str
    duration: float
    hierarchical_structure: HierarchicalStructure
    auditory_streams: List[AuditoryStream]
    sampled_events: List[SampledEvent]
    predictions: Dict[str, MusicalPrediction]
    analysis_metadata: Dict[str, any]

class HierarchicalMusicAnalyzer:
    """
    Integrates all hierarchical analysis components
    
    This is the main interface for hierarchical music analysis that combines
    all the research-based components into a unified system.
    """
    
    def __init__(self, 
                 max_events: int = 10000,
                 enable_predictions: bool = True,
                 enable_stream_analysis: bool = True):
        
        self.max_events = max_events
        self.enable_predictions = enable_predictions
        self.enable_stream_analysis = enable_stream_analysis
        
        # Initialize components
        self.multi_timescale_analyzer = MultiTimescaleAnalyzer()
        self.structure_detector = MSCOMStructureDetector()
        self.timescale_manager = TimescaleManager()
        
        if self.enable_stream_analysis:
            self.auditory_scene_analyzer = SimpleAuditorySceneAnalyzer()
        
        if self.enable_predictions:
            self.predictive_engine = PredictiveProcessingEngine()
            self.mental_model_manager = MentalModelManager()
        
        self.smart_sampler = SmartSampler(max_events=max_events)
        self.structural_sampler = StructuralSampler(max_events=max_events)
        
        # Analysis history
        self.analysis_history = []
        
    def analyze_audio_file(self, audio_file: str, 
                          sampling_strategy: str = 'balanced') -> HierarchicalAnalysisResult:
        """
        Perform comprehensive hierarchical analysis of an audio file
        
        Args:
            audio_file: Path to audio file
            sampling_strategy: Sampling strategy ('balanced', 'structural', 'perceptual')
            
        Returns:
            Comprehensive analysis result
        """
        
        print(f"🎵 Starting hierarchical analysis of: {audio_file}")
        start_time = time.time()
        
        # Load audio to get duration
        y, sr = librosa.load(audio_file, sr=22050)
        duration = len(y) / sr
        
        print(f"📊 Audio duration: {duration:.2f} seconds")
        
        # Step 1: Multi-timescale analysis
        print("\n🔄 Step 1: Multi-timescale analysis...")
        hierarchical_structure = self.multi_timescale_analyzer.analyze_hierarchical_structure(audio_file)
        
        # Step 2: Structural community detection
        print("\n🔄 Step 2: Structural community detection...")
        structural_communities = self.structure_detector.detect_structure(audio_file)
        
        # Step 3: Auditory scene analysis (if enabled)
        auditory_streams = []
        if self.enable_stream_analysis:
            print("\n🔄 Step 3: Auditory scene analysis...")
            auditory_streams = self.auditory_scene_analyzer.analyze_auditory_scene(audio_file)
        
        # Step 4: Adaptive sampling
        print(f"\n🔄 Step 4: Adaptive sampling ({sampling_strategy})...")
        sampled_events = self.smart_sampler.sample_events(audio_file, sampling_strategy)
        
        # Step 5: Predictive processing (if enabled)
        predictions = {}
        if self.enable_predictions:
            print("\n🔄 Step 5: Predictive processing...")
            predictions = self._generate_predictions(sampled_events)
        
        # Step 6: Integrate results
        print("\n🔄 Step 6: Integrating results...")
        analysis_metadata = self._create_analysis_metadata(
            hierarchical_structure, structural_communities, 
            auditory_streams, sampled_events, predictions
        )
        
        # Create result
        result = HierarchicalAnalysisResult(
            audio_file=audio_file,
            duration=duration,
            hierarchical_structure=hierarchical_structure,
            auditory_streams=auditory_streams,
            sampled_events=sampled_events,
            predictions=predictions,
            analysis_metadata=analysis_metadata
        )
        
        # Store in history
        self.analysis_history.append(result)
        
        analysis_time = time.time() - start_time
        print(f"\n✅ Hierarchical analysis complete in {analysis_time:.2f} seconds")
        
        return result
    
    def _generate_predictions(self, sampled_events: List[SampledEvent]) -> Dict[str, MusicalPrediction]:
        """Generate predictions for sampled events"""
        
        predictions = {}
        
        # Generate predictions for a subset of events
        prediction_events = sampled_events[::len(sampled_events)//10]  # Every 10th event
        
        for event in prediction_events:
            # Generate multi-level predictions
            multi_predictions = self.predictive_engine.generate_multi_level_predictions(
                event.features, event.time
            )
            
            # Store predictions
            for level, prediction in multi_predictions.items():
                predictions[f"{event.event_id}_{level}"] = prediction
        
        return predictions
    
    def _create_analysis_metadata(self, hierarchical_structure: HierarchicalStructure,
                                 structural_communities: Dict[str, List[StructuralCommunity]],
                                 auditory_streams: List[AuditoryStream],
                                 sampled_events: List[SampledEvent],
                                 predictions: Dict[str, MusicalPrediction]) -> Dict[str, any]:
        """Create comprehensive analysis metadata"""
        
        metadata = {
            'analysis_timestamp': time.time(),
            'hierarchical_structure': {
                'sections': len(hierarchical_structure.sections),
                'phrases': len(hierarchical_structure.phrases),
                'measures': len(hierarchical_structure.measures)
            },
            'structural_communities': {
                level: len(communities) for level, communities in structural_communities.items()
            },
            'auditory_streams': {
                'total_streams': len(auditory_streams),
                'stream_types': {}
            },
            'sampled_events': {
                'total_events': len(sampled_events),
                'sampling_reasons': {}
            },
            'predictions': {
                'total_predictions': len(predictions),
                'prediction_levels': {}
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
        
        # Count prediction levels
        for prediction_id, prediction in predictions.items():
            level = prediction.level
            metadata['predictions']['prediction_levels'][level] = \
                metadata['predictions']['prediction_levels'].get(level, 0) + 1
        
        return metadata
    
    def get_analysis_summary(self, result: HierarchicalAnalysisResult) -> str:
        """Get a summary of the analysis results"""
        
        summary = []
        summary.append("🎵 Hierarchical Music Analysis Summary")
        summary.append("=" * 50)
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
        
        # Predictions
        if result.predictions:
            summary.append(f"\n🔮 Predictions: {len(result.predictions)}")
            prediction_levels = {}
            for prediction_id, prediction in result.predictions.items():
                prediction_levels[prediction.level] = prediction_levels.get(prediction.level, 0) + 1
            
            for level, count in prediction_levels.items():
                summary.append(f"   {level}: {count}")
        
        return "\n".join(summary)
    
    def save_analysis_result(self, result: HierarchicalAnalysisResult, 
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
                ],
                'predictions': [
                    {
                        'prediction_id': prediction.prediction_id,
                        'level': prediction.level,
                        'time_horizon': prediction.time_horizon,
                        'confidence': prediction.confidence,
                        'prediction_error': prediction.prediction_error
                    }
                    for prediction in result.predictions.values()
                ]
            }
            
            with open(output_file, 'w') as f:
                json.dump(serializable_result, f, indent=2)
            
            return True
            
        except Exception as e:
            print(f"Error saving analysis result: {e}")
            return False
    
    def get_analysis_statistics(self) -> Dict[str, any]:
        """Get statistics about all analyses performed"""
        
        if not self.analysis_history:
            return {}
        
        total_analyses = len(self.analysis_history)
        total_duration = sum(result.duration for result in self.analysis_history)
        
        # Calculate averages
        avg_sections = np.mean([len(result.hierarchical_structure.sections) for result in self.analysis_history])
        avg_phrases = np.mean([len(result.hierarchical_structure.phrases) for result in self.analysis_history])
        avg_measures = np.mean([len(result.hierarchical_structure.measures) for result in self.analysis_history])
        avg_streams = np.mean([len(result.auditory_streams) for result in self.analysis_history])
        avg_events = np.mean([len(result.sampled_events) for result in self.analysis_history])
        
        return {
            'total_analyses': total_analyses,
            'total_duration': total_duration,
            'average_sections': avg_sections,
            'average_phrases': avg_phrases,
            'average_measures': avg_measures,
            'average_streams': avg_streams,
            'average_events': avg_events
        }

def main():
    """Test the hierarchical music analyzer"""
    
    print("🚀 Starting hierarchical music analyzer test...")
    
    analyzer = HierarchicalMusicAnalyzer(
        max_events=5000,  # Smaller for testing
        enable_predictions=True,
        enable_stream_analysis=True
    )
    
    # Test with sample audio file
    audio_file = "input_audio/Grab-a-hold.mp3"
    
    try:
        # Perform comprehensive analysis
        result = analyzer.analyze_audio_file(audio_file, sampling_strategy='balanced')
        
        # Show summary
        print("\n" + analyzer.get_analysis_summary(result))
        
        # Save result
        output_file = "hierarchical_analysis_result.json"
        save_success = analyzer.save_analysis_result(result, output_file)
        print(f"\n💾 Analysis result saved: {save_success}")
        
        # Show statistics
        stats = analyzer.get_analysis_statistics()
        print(f"\n📊 Analysis Statistics:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
    except FileNotFoundError:
        print(f"❌ Audio file not found: {audio_file}")
        print("Please provide a valid audio file path")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
