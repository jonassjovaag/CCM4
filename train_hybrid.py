"""
Hybrid Training Pipeline
Combines transformer analysis with AudioOracle training for enhanced musical learning
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


class HybridTrainingPipeline:
    """
    Hybrid training pipeline that uses transformer analysis to enhance AudioOracle training
    """
    
    def __init__(self, 
                 transformer_model_path: Optional[str] = None,
                 cpu_threshold: int = 5000):
        """
        Initialize hybrid training pipeline
        
        Args:
            transformer_model_path: Path to pre-trained transformer model
            cpu_threshold: Threshold for CPU vs MPS selection
        """
        print("🎵 Initializing Hybrid Training Pipeline...")
        
        # Initialize components
        self.processor = PolyphonicAudioProcessor()
        self.hybrid_trainer = HybridBatchTrainer(cpu_threshold=cpu_threshold)
        self.transformer_analyzer = MusicTheoryAnalyzer(transformer_model_path)
        
        # Training statistics
        self.training_stats = {
            'total_events': 0,
            'transformer_analysis_time': 0.0,
            'audio_oracle_training_time': 0.0,
            'total_training_time': 0.0,
            'enhancement_applied': False
        }
        
        print("✅ Hybrid Training Pipeline initialized")
    
    def train_from_audio_file(self, 
                             audio_file_path: str,
                             output_model_path: str,
                             use_transformer: bool = True,
                             max_events: Optional[int] = None,
                             training_events: Optional[int] = None) -> bool:
        """
        Train hybrid system from audio file
        
        Args:
            audio_file_path: Path to audio file
            output_model_path: Path to save trained model
            use_transformer: Whether to use transformer enhancement
            max_events: Maximum number of events to process from audio
            training_events: Maximum number of events to use for training
            
        Returns:
            True if training successful
        """
        print(f"\n🎵 Hybrid Training from Audio File")
        print(f"📁 File: {os.path.basename(audio_file_path)}")
        print(f"🎯 Output: {output_model_path}")
        print(f"🤖 Transformer Enhancement: {'Enabled' if use_transformer else 'Disabled'}")
        
        start_time = time.time()
        
        try:
            # Step 1: Extract audio features
            print(f"\n🔍 Step 1: Extracting audio features...")
            events, file_info = self.processor.process_audio_file(audio_file_path)
            
            if not events:
                print("❌ No events extracted from audio file")
                return False
            
            if max_events:
                events = events[:max_events]
                print(f"🛑 Limited processing to {max_events} events")
            
            print(f"✅ Extracted {len(events)} events")
            
            # Apply training limit if specified
            if training_events and len(events) > training_events:
                events = events[:training_events]
                print(f"🎯 Limited training to {training_events} events")
            
            # Step 2: Transformer analysis (if enabled)
            transformer_insights = None
            if use_transformer:
                print(f"\n🤖 Step 2: Transformer analysis...")
                transformer_start = time.time()
                
                # Convert events to feature format for transformer
                feature_data = self._events_to_features(events)
                
                # Analyze with transformer (includes real chord detection)
                transformer_insights = self.transformer_analyzer.analyze_audio_features(feature_data)
                
                # Also analyze the actual audio file for more accurate chord detection
                if hasattr(self.transformer_analyzer.chord_detector, 'analyze_audio_file'):
                    print(f"🎵 Analyzing actual audio file for real chord progression...")
                    try:
                        audio_chord_analysis = self.transformer_analyzer.chord_detector.analyze_audio_file(audio_file_path)
                        
                        # Replace chord progression with real audio analysis
                        transformer_insights.chord_progression = audio_chord_analysis.chord_progression
                        transformer_insights.key_signature = audio_chord_analysis.key_signature
                        transformer_insights.confidence_scores['chord'] = np.mean(audio_chord_analysis.confidence_scores) if audio_chord_analysis.confidence_scores else 0.0
                        
                        print(f"✅ Real audio chord analysis complete: {len(audio_chord_analysis.chord_progression)} chords")
                    except Exception as e:
                        print(f"⚠️ Audio file chord analysis failed: {e}, using event-based analysis")
                
                transformer_time = time.time() - transformer_start
                self.training_stats['transformer_analysis_time'] = transformer_time
                
                print(f"✅ Transformer analysis complete ({transformer_time:.2f}s)")
                self._print_transformer_insights(transformer_insights)
            
            # Step 3: Enhanced AudioOracle training
            print(f"\n🎓 Step 3: Enhanced AudioOracle training...")
            oracle_start = time.time()
            
            if transformer_insights:
                # Train with transformer enhancement
                success = self._train_with_transformer_enhancement(events, transformer_insights)
            else:
                # Train with standard AudioOracle
                success = self.hybrid_trainer.train_from_events(events, file_info)
            
            oracle_time = time.time() - oracle_start
            self.training_stats['audio_oracle_training_time'] = oracle_time
            
            if not success:
                print("❌ AudioOracle training failed")
                return False
            
            print(f"✅ AudioOracle training complete ({oracle_time:.2f}s)")
            
            # Step 4: Save enhanced model
            print(f"\n💾 Step 4: Saving enhanced model...")
            model_saved = self.hybrid_trainer.save_model(output_model_path)
            
            if not model_saved:
                print("❌ Failed to save model")
                return False
            
            # Step 5: Save training statistics
            self.training_stats.update({
                'total_events': len(events),
                'total_training_time': time.time() - start_time,
                'enhancement_applied': transformer_insights is not None
            })
            
            stats_path = output_model_path.replace('.json', '_hybrid_stats.json')
            self._save_training_stats(stats_path, transformer_insights)
            
            print(f"✅ Enhanced model saved to: {output_model_path}")
            print(f"📊 Training statistics saved to: {stats_path}")
            
            # Print final summary
            self._print_training_summary()
            
            return True
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _events_to_features(self, events: List[Any]) -> List[Dict]:
        """Convert events to feature dictionaries for transformer"""
        feature_data = []
        
        for event in events:
            if hasattr(event, 'to_dict'):
                feature_dict = event.to_dict()
                # Add additional attributes
                feature_dict.update({
                    'rolloff': float(getattr(event, 'rolloff', 0.0)),
                    'bandwidth': float(getattr(event, 'bandwidth', 0.0)),
                    'contrast': float(getattr(event, 'contrast', 0.0)),
                    'flatness': float(getattr(event, 'flatness', 0.0)),
                    'mfcc_1': float(getattr(event, 'mfcc_1', 0.0)),
                    'duration': float(getattr(event, 'duration', 0.5)),
                    'attack_time': float(getattr(event, 'attack_time', 0.1)),
                    'release_time': float(getattr(event, 'release_time', 0.3)),
                    'tempo': float(getattr(event, 'tempo', 120.0)),
                    'beat_position': float(getattr(event, 'beat_position', 0.0))
                })
            else:
                feature_dict = event
            
            feature_data.append(feature_dict)
        
        return feature_data
    
    def _train_with_transformer_enhancement(self, 
                                          events: List[Any], 
                                          transformer_insights: MusicalInsights) -> bool:
        """Train AudioOracle with transformer insights"""
        print("🎯 Applying transformer enhancement to AudioOracle training...")
        
        # Enhance events with transformer insights
        enhanced_events = self._enhance_events_with_insights(events, transformer_insights)
        
        # Train with enhanced data
        success = self.hybrid_trainer.train_from_events(enhanced_events)
        
        return success
    
    def _enhance_events_with_insights(self, 
                                    events: List[Any], 
                                    insights: MusicalInsights) -> List[Any]:
        """Enhance events with transformer insights"""
        enhanced_events = []
        
        for i, event in enumerate(events):
            # Create enhanced event with transformer insights
            enhanced_event = self._create_enhanced_event(event, insights, i)
            enhanced_events.append(enhanced_event)
        
        return enhanced_events
    
    def _create_enhanced_event(self, 
                              event: Any, 
                              insights: MusicalInsights, 
                              index: int) -> Any:
        """Create enhanced event with transformer insights"""
        # Convert event to dictionary
        if hasattr(event, 'to_dict'):
            event_dict = event.to_dict()
        else:
            event_dict = event
        
        # Add transformer insights
        event_dict.update({
            'transformer_chord': insights.chord_progression[index % len(insights.chord_progression)],
            'transformer_scale': insights.key_signature,
            'transformer_form': max(insights.musical_form.items(), key=lambda x: x[1])[0],
            'transformer_harmony': insights.harmonic_rhythm[index % len(insights.harmonic_rhythm)],
            'transformer_melody': insights.melodic_contour[index % len(insights.melodic_contour)],
            'transformer_rhythm': insights.rhythmic_patterns[index % len(insights.rhythmic_patterns)],
            'transformer_confidence': insights.confidence_scores['chord'],
            'transformer_tempo': insights.tempo_analysis['estimated_tempo']
        })
        
        return event_dict
    
    def _print_transformer_insights(self, insights: MusicalInsights):
        """Print transformer analysis insights"""
        print(f"\n🤖 Transformer Analysis Results:")
        print(f"   🎵 Key Signature: {insights.key_signature}")
        print(f"   🎼 Chord Progression: {insights.chord_progression[:10]}...")
        print(f"   🎶 Musical Form: {max(insights.musical_form.items(), key=lambda x: x[1])[0]}")
        print(f"   ⏱️ Tempo: {insights.tempo_analysis['estimated_tempo']:.1f} BPM")
        print(f"   🎯 Confidence: {insights.confidence_scores['chord']:.2f}")
        
        # Print scale analysis (top 3)
        top_scales = sorted(insights.scale_analysis.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"   🎹 Top Scales: {', '.join([f'{scale} ({prob:.2f})' for scale, prob in top_scales])}")
    
    def _save_training_stats(self, stats_path: str, transformer_insights: Optional[MusicalInsights]):
        """Save training statistics"""
        stats = self.training_stats.copy()
        
        # Add transformer insights to stats
        if transformer_insights:
            stats['transformer_insights'] = asdict(transformer_insights)
        
        # Add hybrid trainer stats
        trainer_stats = self.hybrid_trainer.get_training_stats()
        stats.update(trainer_stats)
        
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
    
    def _print_training_summary(self):
        """Print training summary"""
        print(f"\n🎯 Hybrid Training Summary:")
        print(f"   📊 Total Events: {self.training_stats['total_events']:,}")
        print(f"   ⏱️ Total Time: {self.training_stats['total_training_time']:.2f}s")
        print(f"   🤖 Transformer Time: {self.training_stats['transformer_analysis_time']:.2f}s")
        print(f"   🎓 AudioOracle Time: {self.training_stats['audio_oracle_training_time']:.2f}s")
        print(f"   🎯 Enhancement Applied: {'Yes' if self.training_stats['enhancement_applied'] else 'No'}")
        
        # Performance metrics
        if self.training_stats['total_training_time'] > 0:
            events_per_second = self.training_stats['total_events'] / self.training_stats['total_training_time']
            print(f"   ⚡ Performance: {events_per_second:.1f} events/sec")
        
        # Get trainer stats
        trainer_stats = self.hybrid_trainer.get_training_stats()
        print(f"   🎵 Patterns Found: {trainer_stats.get('patterns_found', 0)}")
        print(f"   🎼 Chord Patterns: {trainer_stats.get('chord_patterns', 0)}")
        print(f"   🎯 Device Used: {trainer_stats.get('device_used', 'unknown').upper()}")


def main():
    """Main entry point for hybrid training"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Hybrid Training Pipeline - Transformer + AudioOracle')
    
    # Input options
    parser.add_argument('--file', type=str, required=True, help='Audio file to process')
    parser.add_argument('--output', type=str, default='hybrid_model.json', help='Output model file')
    
    # Training options
    parser.add_argument('--transformer', action='store_true', help='Enable transformer enhancement')
    parser.add_argument('--oracle-only', action='store_true', help='Use AudioOracle only')
    parser.add_argument('--max-events', type=int, default=None, help='Maximum events to process from audio')
    parser.add_argument('--training-events', type=int, default=None, help='Maximum events to use for training')
    
    # Transformer options
    parser.add_argument('--transformer-model', type=str, default=None, help='Path to transformer model')
    parser.add_argument('--cpu-threshold', type=int, default=5000, help='CPU threshold for hybrid trainer')
    
    args = parser.parse_args()
    
    # Determine training mode
    use_transformer = args.transformer and not args.oracle_only
    
    print("🎵 Hybrid Training Pipeline")
    print("=" * 50)
    print(f"📁 Input: {args.file}")
    print(f"📁 Output: {args.output}")
    print(f"🤖 Transformer: {'Enabled' if use_transformer else 'Disabled'}")
    print(f"🎯 Mode: {'Hybrid' if use_transformer else 'AudioOracle Only'}")
    
    # Initialize pipeline
    pipeline = HybridTrainingPipeline(
        transformer_model_path=args.transformer_model,
        cpu_threshold=args.cpu_threshold
    )
    
    # Train
    success = pipeline.train_from_audio_file(
        audio_file_path=args.file,
        output_model_path=args.output,
        use_transformer=use_transformer,
        max_events=args.max_events,
        training_events=args.training_events
    )
    
    if success:
        print(f"\n🎉 Hybrid training completed successfully!")
        return 0
    else:
        print(f"\n❌ Hybrid training failed!")
        return 1


if __name__ == "__main__":
    exit(main())
