#!/usr/bin/env python3
"""
Enhanced Hybrid Training Pipeline with Rhythmic Analysis
Integrates hierarchical music analysis AND rhythmic analysis into MusicHal 9000

This pipeline now uses:
- Hierarchical multi-timescale analysis (Farbood et al., 2015)
- Perceptual significance filtering (Pressnitzer et al., 2008)
- Predictive processing (Schaefer, 2014)
- Adaptive sampling strategies
- Rhythmic pattern analysis and learning
"""

import os
import time
import json
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import asdict

# Activate CCM3 virtual environment if available
try:
    from ccm3_venv_manager import ensure_ccm3_venv_active
    ensure_ccm3_venv_active()
    print("✅ CCM3 virtual environment activated")
except ImportError:
    print("Note: CCM3 environment manager not available, using current environment")

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

from audio_file_learning.polyphonic_processor import PolyphonicAudioProcessor
from audio_file_learning.hybrid_batch_trainer import HybridBatchTrainer
from hybrid_training.music_theory_transformer import MusicTheoryAnalyzer, MusicalInsights
from simple_hierarchical_integration import SimpleHierarchicalAnalyzer, SimpleAnalysisResult

# New rhythmic components
from rhythmic_engine.audio_file_learning.heavy_rhythmic_analyzer import HeavyRhythmicAnalyzer, RhythmicAnalysis
from rhythmic_engine.memory.rhythm_oracle import RhythmOracle

# New correlation components
from correlation_engine.correlation_analyzer import HarmonicRhythmicCorrelator
from correlation_engine.unified_decision_engine import UnifiedDecisionEngine, CrossModalContext

# Harmonic detection for training (same as live performance)
from listener.harmonic_context import RealtimeHarmonicDetector, HarmonicContext

# GPT-OSS integration
from gpt_oss_client import GPTOSSClient, GPTOSSAnalysis, GPTOSSArcAnalysis

# Performance arc components
from performance_arc_analyzer import PerformanceArcAnalyzer, PerformanceArc

# Temporal smoothing components (legacy)
from core.temporal_smoothing import TemporalSmoother

# Musical gesture processing (new approach)
from core.musical_gesture_processor import MusicalGestureProcessor

class EnhancedHybridTrainingPipeline:
    """
    Enhanced hybrid training pipeline with hierarchical AND rhythmic music analysis
    
    This pipeline combines:
    1. Hierarchical analysis (multi-timescale, perceptual, adaptive sampling)
    2. Rhythmic analysis (beat tracking, pattern recognition, syncopation)
    3. Music theory transformer analysis
    4. AudioOracle training with enhanced insights
    5. RhythmOracle training with rhythmic patterns
    """
    
    def __init__(self, 
                 transformer_model_path: Optional[str] = None,
                 cpu_threshold: int = 5000,
                 max_events: int = 10000,
                 enable_hierarchical: bool = True,
                 enable_rhythmic: bool = True,
                 enable_gpt_oss: bool = True,
                 enable_hybrid_perception: bool = True,
                 symbolic_vocabulary_size: int = 64,
                 enable_wav2vec: bool = True,
                 wav2vec_model: str = "facebook/wav2vec2-base",
                 use_gpu: bool = True,
                 enable_dual_vocabulary: bool = False,
                 temporal_window: float = 0.1,
                 temporal_threshold: float = 0.2,
                 enable_temporal_smoothing: bool = True,
                 use_musical_gestures: bool = True,
                 gesture_transition_threshold: float = 0.3,
                 gesture_sustain_threshold: float = 0.15,
                 gesture_min_duration: float = 0.2,
                 gesture_max_duration: float = 2.0,
                 gesture_consolidation_method: str = 'peak'):
        """
        Initialize enhanced hybrid training pipeline
        
        Args:
            transformer_model_path: Path to pre-trained transformer model
            cpu_threshold: Threshold for CPU vs MPS selection
            max_events: Maximum number of events to sample
            enable_hierarchical: Enable hierarchical analysis
            enable_rhythmic: Enable rhythmic analysis
            enable_gpt_oss: Enable GPT-OSS analysis
            enable_hybrid_perception: Enable ratio-based + symbolic features (NEW!)
            symbolic_vocabulary_size: Size of symbolic alphabet (16-64 recommended)
            enable_wav2vec: Enable Wav2Vec 2.0 neural encoding (replaces ratio+chroma)
            wav2vec_model: HuggingFace model name for Wav2Vec
            use_gpu: Use GPU for Wav2Vec (MPS/CUDA)
            enable_dual_vocabulary: Enable dual harmonic/percussive vocabularies (for drums)
        """
        print("🎵 Initializing Enhanced Hybrid Training Pipeline...")
        
        # Initialize harmonic components
        self.processor = PolyphonicAudioProcessor()
        self.hybrid_trainer = HybridBatchTrainer(cpu_threshold=0)  # Force CPU AudioOracle for compatibility
        self.transformer_analyzer = MusicTheoryAnalyzer(transformer_model_path)
        
        # Initialize hierarchical analyzer
        if enable_hierarchical:
            self.hierarchical_analyzer = SimpleHierarchicalAnalyzer(max_events=max_events)
            print("✅ Hierarchical analysis enabled")
        else:
            self.hierarchical_analyzer = None
            print("⚠️  Hierarchical analysis disabled")
        
        # Initialize rhythmic components
        if enable_rhythmic:
            self.rhythmic_analyzer = HeavyRhythmicAnalyzer()
            self.rhythm_oracle = RhythmOracle()
            
            # Initialize tempo reconciliation engine
            from rhythmic_engine.tempo_reconciliation import ReconciliationEngine
            self.reconciliation_engine = ReconciliationEngine(tolerance=0.15, max_history=3)
            
            print("🥁 Rhythmic analysis enabled")
            print("🔄 Tempo reconciliation enabled")
        else:
            self.rhythmic_analyzer = None
            self.rhythm_oracle = None
            self.reconciliation_engine = None
            print("⚠️  Rhythmic analysis disabled")
        
        # Initialize correlation engine
        self.correlation_analyzer = HarmonicRhythmicCorrelator()
        self.unified_decision_engine = UnifiedDecisionEngine()
        print("🔗 Harmonic-rhythmic correlation analysis enabled")
        
        # Initialize harmonic detector for training (same as live performance)
        self.harmonic_detector = RealtimeHarmonicDetector(window_size=8192, hop_length=2048)
        self.chord_sequence = []  # Track detected chords for transition analysis
        print("🎼 Real-time harmonic detector enabled for training")
        
        # Initialize GPT-OSS client
        if enable_gpt_oss:
            self.gpt_oss_client = GPTOSSClient(auto_start=True)
            print("🧠 GPT-OSS analysis enabled (auto-start)")
        else:
            self.gpt_oss_client = None
            print("⚠️ GPT-OSS analysis disabled")
        
        # Initialize performance arc analyzer
        self.performance_arc_analyzer = PerformanceArcAnalyzer()
        print("🎭 Performance arc analysis enabled")
        
        # Store temporal smoothing parameters
        self.temporal_window = temporal_window
        self.temporal_threshold = temporal_threshold
        self.enable_temporal_smoothing = enable_temporal_smoothing
        
        # Initialize temporal smoother (prevents over-sampling of sustained notes)
        if enable_temporal_smoothing:
            # OPTIMIZED FOR RHYTHMIC VARIETY: User-configurable parameters
            # Default: 0.1s window preserves rhythmic changes while preventing flicker
            # Higher threshold ensures only genuine musical changes create new events
            self.temporal_smoother = TemporalSmoother(
                window_size=temporal_window,
                min_change_threshold=temporal_threshold
            )
            print(f"✅ Temporal smoothing enabled: {temporal_window}s window, {temporal_threshold} threshold (optimized for rhythmic variety)")
        else:
            self.temporal_smoother = None
            print("⚠️  Temporal smoothing DISABLED - pure rhythmic analysis mode")
        
        # Initialize musical gesture processor (NEW APPROACH - replaces temporal smoothing)
        self.use_musical_gestures = use_musical_gestures and not enable_temporal_smoothing
        self.gesture_consolidation_method = gesture_consolidation_method
        if self.use_musical_gestures:
            self.musical_gesture_processor = MusicalGestureProcessor(
                transition_threshold=gesture_transition_threshold,
                sustain_threshold=gesture_sustain_threshold,
                min_gesture_duration=gesture_min_duration,
                max_gesture_duration=gesture_max_duration,
                consolidation_method=gesture_consolidation_method
            )
            print(f"🎵 Musical gesture processing enabled:")
            print(f"   Transition threshold: {gesture_transition_threshold} (feature change = new gesture)")
            print(f"   Sustain threshold: {gesture_sustain_threshold} (similarity = sustained gesture)")
            print(f"   Gesture duration: {gesture_min_duration}s - {gesture_max_duration}s")
            print(f"   Consolidation: {gesture_consolidation_method} (how to select representative moment)")
        else:
            self.musical_gesture_processor = None
            if enable_temporal_smoothing:
                print("   Using legacy temporal smoothing instead of musical gestures")

        
        # Initialize perception system
        # NEW APPROACH: Dual perception (Wav2Vec + Ratios in parallel)
        if enable_hybrid_perception:
            if enable_wav2vec:
                # Use dual perception: Wav2Vec tokens + Ratio context
                from listener.dual_perception import DualPerceptionModule
                self.dual_perception = DualPerceptionModule(
                    vocabulary_size=symbolic_vocabulary_size,
                    wav2vec_model=wav2vec_model,
                    use_gpu=use_gpu,
                    enable_symbolic=True,
                    enable_dual_vocabulary=enable_dual_vocabulary
                )
                self.hybrid_perception = None  # Not using old hybrid system
                print(f"🎵 Dual perception enabled:")
                print(f"   Machine logic: {wav2vec_model} → gesture tokens (0-{symbolic_vocabulary_size-1})")
                print(f"   Machine logic: Ratio analysis → consonance + frequency ratios")
                print(f"   Human interface: Chord names for display only")
                print(f"   ✨ Tokens ARE the patterns, not chord names!")
                if enable_dual_vocabulary:
                    print(f"   🥁 Dual vocabulary mode: {symbolic_vocabulary_size} harmonic + {symbolic_vocabulary_size} percussive tokens")
            else:
                # Traditional hybrid perception (ratio + chroma)
                from listener.hybrid_perception import HybridPerceptionModule
                self.hybrid_perception = HybridPerceptionModule(
                    vocabulary_size=symbolic_vocabulary_size,
                    enable_ratio_analysis=True,
                    enable_symbolic=True,
                    enable_wav2vec=False
                )
                self.dual_perception = None
                print(f"🔬 Hybrid perception enabled (vocab: {symbolic_vocabulary_size} classes)")
                print("   Features: Ratio-based + Harmonic chroma + Symbolic tokens")
        else:
            self.hybrid_perception = None
            self.dual_perception = None
            print("⚠️  Using standard feature extraction")
        
        # Training statistics
        self.training_stats = {
            'total_events': 0,
            'hierarchical_analysis_time': 0.0,
            'rhythmic_analysis_time': 0.0,
            'transformer_analysis_time': 0.0,
            'gpt_oss_analysis_time': 0.0,
            'performance_arc_analysis_time': 0.0,
            'audio_oracle_training_time': 0.0,
            'rhythm_oracle_training_time': 0.0,
            'total_training_time': 0.0,
            'hierarchical_insights_applied': False,
            'rhythmic_insights_applied': False,
            'gpt_oss_insights_applied': False,
            'performance_arc_insights_applied': False,
            'enhancement_applied': False,
            'hybrid_perception_enabled': enable_hybrid_perception,
            'symbolic_vocabulary_size': symbolic_vocabulary_size if enable_hybrid_perception else 0,
            'wav2vec_enabled': enable_wav2vec,
            'wav2vec_model': wav2vec_model if enable_wav2vec else None
        }
        
        print("✅ Enhanced Hybrid Training Pipeline initialized")
    
    def cleanup(self):
        """Clean up resources"""
        if self.gpt_oss_client:
            self.gpt_oss_client.cleanup()
    
    def train_from_audio_file(self, 
                             audio_file: str,
                             output_file: str,
                             max_events: Optional[int] = None,
                             training_events: Optional[int] = None,
                             sampling_strategy: str = 'balanced',
                             use_transformer: bool = True,
                             use_hierarchical: bool = True,
                             use_rhythmic: bool = True,
                             analyze_arc: bool = False,
                             section_duration: float = 60.0) -> Dict[str, Any]:
        """
        Train AudioOracle and RhythmOracle from audio file with enhanced analysis
        
        Args:
            audio_file: Path to audio file
            output_file: Path to output JSON file
            max_events: Maximum events to extract (overrides default)
            training_events: Number of events to use for training
            sampling_strategy: Sampling strategy ('balanced', 'structural', 'perceptual')
            use_transformer: Whether to use transformer analysis
            use_hierarchical: Whether to use hierarchical analysis
            use_rhythmic: Whether to use rhythmic analysis
            analyze_arc: Whether to analyze long-form arc structure (Brandtsegg sections)
            section_duration: Section duration for arc analysis in seconds (default 60s)
            
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
        
        # Step 2: Rhythmic Analysis (if enabled)
        rhythmic_result = None
        if use_rhythmic and self.rhythmic_analyzer:
            print("\n🔄 Step 2: Rhythmic Analysis...")
            rhythmic_start = time.time()
            
            try:
                rhythmic_result = self.rhythmic_analyzer.analyze_rhythmic_structure(audio_file)
                
                # Train RhythmOracle with rhythmic patterns
                rhythm_oracle_start = time.time()
                if self.rhythm_oracle:
                    self._train_rhythm_oracle(rhythmic_result)
                rhythm_oracle_end = time.time()
                self.training_stats['rhythm_oracle_training_time'] = rhythm_oracle_end - rhythm_oracle_start
                print(f"🥁 RhythmOracle training took: {self.training_stats['rhythm_oracle_training_time']:.4f}s")
                
                self.training_stats['rhythmic_analysis_time'] = time.time() - rhythmic_start
                self.training_stats['rhythmic_insights_applied'] = True
                
                print(f"✅ Rhythmic analysis complete: {len(rhythmic_result.patterns)} patterns detected")
                tempo_val = rhythmic_result.tempo.item() if hasattr(rhythmic_result.tempo, 'item') else rhythmic_result.tempo
                syncopation_val = rhythmic_result.syncopation_score.item() if hasattr(rhythmic_result.syncopation_score, 'item') else rhythmic_result.syncopation_score
                complexity_val = rhythmic_result.rhythmic_complexity.item() if hasattr(rhythmic_result.rhythmic_complexity, 'item') else rhythmic_result.rhythmic_complexity
                print(f"   Tempo: {float(tempo_val):.1f} BPM")
                print(f"   Syncopation: {float(syncopation_val):.3f}")
                print(f"   Complexity: {float(complexity_val):.3f}")
                
                # Step 2b: Rational Structure Analysis
                print("\n🔄 Step 2b: Rational Rhythm Structure Analysis...")
                rational_start = time.time()
                try:
                    # Perform rational analysis on onsets
                    rational_structure = self.rhythmic_analyzer.analyze_rational_structure(rhythmic_result.onsets)
                    
                    if rational_structure:
                        # Store rational structure in rhythmic result
                        rhythmic_result.rational_structure = rational_structure
                        
                        print(f"✅ Rational structure analysis complete")
                        print(f"   Duration pattern: {rational_structure['duration_pattern']}")
                        print(f"   Subdiv tempo: {rational_structure['tempo']:.1f} BPM")
                        print(f"   Pulse: {rational_structure['pulse']}")
                        print(f"   Complexity (Barlow): {rational_structure['complexity']:.2f}")
                        print(f"   Confidence: {rational_structure['confidence']:.2f}")
                        
                        # Step 2c: Tempo Reconciliation
                        if self.reconciliation_engine:
                            print("\n🔄 Step 2c: Tempo Reconciliation...")
                            try:
                                reconciled_structure = self.reconciliation_engine.reconcile_new_phrase(
                                    rational_structure
                                )
                                
                                # Update with reconciled structure
                                rhythmic_result.rational_structure = reconciled_structure
                                
                                if reconciled_structure.get('reconciled', False):
                                    print(f"✅ Tempo reconciled with previous phrase")
                                    print(f"   Tempo factor: {reconciled_structure['tempo_factor']:.2f}")
                                    if 'prev_tempo' in reconciled_structure:
                                        print(f"   {reconciled_structure['prev_tempo']:.1f} BPM → {reconciled_structure['tempo']:.1f} BPM")
                                else:
                                    print(f"ℹ️  No reconciliation needed (first phrase or same tempo)")
                                
                            except Exception as e:
                                print(f"⚠️  Tempo reconciliation failed: {e}")
                        
                        self.training_stats['rational_analysis_time'] = time.time() - rational_start
                    else:
                        print("⚠️  Rational structure analysis skipped (insufficient onsets)")
                        rhythmic_result.rational_structure = None
                        
                except Exception as e:
                    print(f"⚠️  Rational structure analysis failed: {e}")
                    import traceback
                    traceback.print_exc()
                    rhythmic_result.rational_structure = None
                
            except Exception as e:
                print(f"⚠️  Rhythmic analysis failed: {e}")
                rhythmic_result = None
        
        # Step 2c: Long-Form Arc Structure Analysis (Brandtsegg sections)
        if use_rhythmic and analyze_arc and self.rhythmic_analyzer:
            print("\n🔄 Step 2c: Long-Form Arc Structure Analysis...")
            arc_start = time.time()
            
            try:
                sections = self.rhythmic_analyzer.analyze_long_form_improvisation(
                    audio_file,
                    section_duration=section_duration
                )
                
                if sections:
                    print(f"✅ Long-form analysis complete: {len(sections)} sections detected")
                    
                    # Map sections to arc phases (adds 'arc_phase' key to each section)
                    self._map_sections_to_arc_phases(sections)
                    
                    self.arc_structure_sections = sections
                    
                    # Now extract phase distribution (arc_phase exists now!)
                    tempos = [s['local_tempo'] for s in self.arc_structure_sections]
                    phases = self._get_phase_distribution(self.arc_structure_sections)
                    
                    self.training_stats['arc_structure_time'] = time.time() - arc_start
                    self.training_stats['arc_sections_detected'] = len(self.arc_structure_sections)
                    
                    print(f"   Total duration: {self.arc_structure_sections[-1]['end_time']:.0f}s ({self.arc_structure_sections[-1]['end_time']/60:.1f} min)")
                    print(f"   Tempo range: {min(tempos):.1f} - {max(tempos):.1f} BPM")
                    print(f"   Phase distribution: {phases}")
                else:
                    print("⚠️  No sections detected in arc structure analysis")
                    self.arc_structure_sections = None
                    
            except Exception as e:
                print(f"⚠️  Arc structure analysis failed: {e}")
                import traceback
                traceback.print_exc()
                self.arc_structure_sections = None
        else:
            self.arc_structure_sections = None
        
        # Step 3: Harmonic-Rhythmic Correlation Analysis
        print("\n🔄 Step 3: Harmonic-Rhythmic Correlation Analysis...")
        correlation_start = time.time()
        
        correlation_result = None
        if self.rhythmic_analyzer and rhythmic_result:
            try:
                # Convert rhythmic and harmonic events to correlation format
                # We need to extract events first for harmonic analysis
                if hierarchical_result:
                    harmonic_events = self._convert_to_correlation_format(hierarchical_result.sampled_events, 'harmonic')
                else:
                    # Fallback: create basic harmonic events from rhythmic analysis
                    harmonic_events = [{'timestamp': 0.0, 'chord': 'C', 'key': 'C', 'tension': 0.5, 'chord_diversity': 0.3, 'key_stability': 0.8, 'chord_change_rate': 0.2}]
                
                # Debug: Show chord detection results
                print(f"\n🎼 Chord Detection Analysis:")
                print(f"   Total harmonic events: {len(harmonic_events)}")
                
                # Show chord progression sample (first 20 events)
                chord_progression = []
                for i, event in enumerate(harmonic_events[:20]):
                    chord = event.get('chord', '')
                    if chord:
                        chord_progression.append(chord)
                
                if chord_progression:
                    print(f"   Sample chord progression (first 20): {chord_progression}")
                    
                    # Show chord distribution
                    from collections import Counter
                    chord_counts = Counter(chord_progression)
                    print(f"   Chord distribution: {dict(chord_counts)}")
                else:
                    print(f"   ⚠️  No chords detected in sample!")
                
                rhythmic_events = self._convert_to_correlation_format(rhythmic_result, 'rhythmic')
                
                # Run correlation analysis
                correlation_result = self.correlation_analyzer.analyze_correlations(
                    harmonic_events, rhythmic_events, audio_file
                )
                
                self.training_stats['correlation_analysis_time'] = time.time() - correlation_start
                self.training_stats['correlation_insights_applied'] = True
                
                print(f"✅ Correlation analysis complete: {correlation_result['analysis_stats']['patterns_discovered']} patterns discovered")
                print(f"   Joint events: {correlation_result['analysis_stats']['total_joint_events']}")
                print(f"   Average correlation strength: {correlation_result['analysis_stats']['correlation_strength_avg']:.3f}")
                
                # Print cross-modal insights
                insights = correlation_result['cross_modal_insights']
                if insights['cross_modal_insights']:
                    print(f"   Cross-modal insights:")
                    for insight in insights['cross_modal_insights']:
                        print(f"     • {insight}")
                
            except Exception as e:
                print(f"⚠️  Correlation analysis failed: {e}")
                correlation_result = None
        
        # Step 4: Extract Features (ALWAYS use PolyphonicAudioProcessor for real audio features)
        print("\n🔄 Step 4: Feature Extraction...")
        
        # CRITICAL FIX: Extract full events with audio features FIRST
        # (hierarchical sampling only had metadata, not f0/midi/rms_db)
        event_list, file_info = self.processor.process_audio_file(audio_file)
        # Convert Event objects to dictionaries
        full_events = [self._convert_event_to_dict(event) for event in event_list]
        
        # Limit to max_events if specified
        if max_events and len(full_events) > max_events:
            indices = np.linspace(0, len(full_events)-1, max_events, dtype=int)
            full_events = [full_events[i] for i in indices]
        
        print(f"✅ Extracted {len(full_events)} events with full audio features")
        
        if hierarchical_result:
            # Merge hierarchical significance scores with full audio events
            events = self._merge_hierarchical_with_audio_events(full_events, hierarchical_result.sampled_events)
            print(f"✅ Merged with hierarchical significance scores")
        else:
            # Use full events as-is
            events = full_events
            print(f"✅ Using traditional processing (no hierarchical analysis)")
        
        # Limit events for training if specified
        if training_events and len(events) > training_events:
            # Sample events evenly across the timeline
            indices = np.linspace(0, len(events)-1, training_events, dtype=int)
            events = [events[i] for i in indices]
            print(f"✅ Limited to {len(events)} events for training")
        
        self.training_stats['total_events'] = len(events)
        
        # NEW: Real-time chord detection for accurate harmonic transitions
        print("\n🎼 Step 4a: Real-time Chord Detection...")
        chord_detection_start = time.time()
        events = self._detect_chords_for_events(events, audio_file)
        chord_detection_time = time.time() - chord_detection_start
        print(f"✅ Chord detection complete in {chord_detection_time:.2f}s")
        self.training_stats['chord_detection_time'] = chord_detection_time
        
        # NEW: Hybrid/Dual Perception Feature Augmentation (if enabled)
        if self.dual_perception:
            # Dual perception: Wav2Vec + Ratios
            print("\n🔬 Step 4b: Dual Perception Feature Extraction (Wav2Vec + Ratios)...")
            hybrid_start = time.time()
            events = self._augment_with_dual_features(audio_file, events)
            hybrid_time = time.time() - hybrid_start
            print(f"✅ Dual perception features extracted in {hybrid_time:.2f}s")
            self.training_stats['hybrid_perception_time'] = hybrid_time
        elif self.hybrid_perception:
            # Traditional hybrid perception
            print("\n🔬 Step 4b: Hybrid Perception Feature Extraction...")
            hybrid_start = time.time()
            events = self._augment_with_hybrid_features(audio_file, events)
            hybrid_time = time.time() - hybrid_start
            print(f"✅ Hybrid perception features extracted in {hybrid_time:.2f}s")
            self.training_stats['hybrid_perception_time'] = hybrid_time
        
        # Step 5: Transformer Analysis (if enabled)
        transformer_insights = None
        if use_transformer:
            print("\n🔄 Step 5: Music Theory Transformer Analysis...")
            transformer_start = time.time()
            
            transformer_insights = self.transformer_analyzer.analyze_audio_features(events)
            
            self.training_stats['transformer_analysis_time'] = time.time() - transformer_start
            self.training_stats['enhancement_applied'] = True
            
            print(f"✅ Transformer analysis complete")
        
        # Step 6: Performance Arc Analysis
        print("\n🔄 Step 6: Performance Arc Analysis...")
        performance_arc_start = time.time()
        
        performance_arc = None
        try:
            # Analyze the audio file to extract performance arc
            performance_arc = self.performance_arc_analyzer.analyze_audio_file(audio_file)
            
            performance_arc_time = time.time() - performance_arc_start
            print(f"✅ Performance arc analysis complete in {performance_arc_time:.2f}s")
            print(f"   Total duration: {performance_arc.total_duration:.2f}s")
            print(f"   Number of phases: {len(performance_arc.phases)}")
            print(f"   Engagement curve points: {len(performance_arc.overall_engagement_curve)}")
            print(f"   Silence patterns: {len(performance_arc.silence_patterns)}")
            
            # Track performance arc analysis time
            self.training_stats['performance_arc_analysis_time'] = performance_arc_time
            self.training_stats['performance_arc_insights_applied'] = True
            
        except Exception as e:
            print(f"❌ Performance arc analysis failed: {e}")
            performance_arc = None
        
        # Step 7: GPT-OSS Musical Intelligence Pre-Analysis
        print("\n🔄 Step 7: GPT-OSS Musical Intelligence Pre-Analysis...")
        gpt_oss_start = time.time()
        
        gpt_oss_insights = None
        gpt_oss_arc_insights = None
        if self.gpt_oss_client and self.gpt_oss_client.is_available:
            try:
                # Analyze events BEFORE training to enhance the training process
                gpt_oss_insights = self.gpt_oss_client.analyze_musical_events_for_training(
                    events, transformer_insights, hierarchical_result, rhythmic_result, correlation_result
                )
                
                # Analyze performance arc for musical structure and evolution
                if performance_arc:
                    print("🔄 Analyzing performance arc with GPT-OSS...")
                    gpt_oss_arc_insights = self.gpt_oss_client.analyze_performance_arc(performance_arc)
                
                if gpt_oss_insights:
                    gpt_oss_time = time.time() - gpt_oss_start
                    print(f"✅ GPT-OSS pre-analysis complete in {gpt_oss_time:.2f}s")
                    print(f"   Confidence: {gpt_oss_insights.confidence_score:.2f}")
                    print(f"   Musical insights: {len(gpt_oss_insights.harmonic_analysis)} chars")
                    
                    # Track GPT-OSS pre-analysis time
                    self.training_stats['gpt_oss_analysis_time'] = gpt_oss_time
                    self.training_stats['gpt_oss_insights_applied'] = True
                else:
                    print("⚠️ GPT-OSS pre-analysis failed")
                
                if gpt_oss_arc_insights:
                    print(f"✅ GPT-OSS arc analysis complete")
                    print(f"   Arc confidence: {gpt_oss_arc_insights.confidence_score:.2f}")
                    print(f"   Structural analysis: {len(gpt_oss_arc_insights.structural_analysis)} chars")
                else:
                    print("⚠️ GPT-OSS arc analysis failed")
                    
            except Exception as e:
                print(f"❌ GPT-OSS pre-analysis failed: {e}")
                gpt_oss_insights = None
                gpt_oss_arc_insights = None
        else:
            print("⚠️ GPT-OSS not available for pre-analysis")
        
        # Step 8: Enhanced AudioOracle Training
        print("\n🔄 Step 8: Enhanced AudioOracle Training...")
        training_start = time.time()
        
        # Create enhanced training data with GPT-OSS insights
        # DEBUG: Check what audio features are in events BEFORE enhancement
        if events and len(events) > 0:
            sample_event = events[0]
            print(f"🔍 DEBUG: Sample event BEFORE enhancement:")
            print(f"   Keys: {list(sample_event.keys())[:15]}")
            print(f"   Has f0: {'f0' in sample_event}")
            print(f"   Has midi: {'midi' in sample_event}")
            print(f"   Has rms_db: {'rms_db' in sample_event}")
            print(f"   Has gesture_token: {'gesture_token' in sample_event}")
            if 'gesture_token' in sample_event:
                print(f"   gesture_token value: {sample_event['gesture_token']}")
            
            # Check all events for gesture tokens
            tokens_before = [e.get('gesture_token') for e in events]
            unique_before = set(t for t in tokens_before if t is not None)
            print(f"   🔍 Gesture tokens before enhancement: {len(unique_before)} unique tokens")
        
        enhanced_events = self._enhance_events_with_insights(
            events, transformer_insights, hierarchical_result, rhythmic_result, correlation_result, gpt_oss_insights, performance_arc, gpt_oss_arc_insights
        )
        
        # DEBUG: Check what audio features are in events AFTER enhancement
        if enhanced_events and len(enhanced_events) > 0:
            sample_enhanced = enhanced_events[0]
            print(f"🔍 DEBUG: Sample event AFTER enhancement:")
            print(f"   Keys: {list(sample_enhanced.keys())[:15]}")
            print(f"   Has f0: {'f0' in sample_enhanced}")
            print(f"   Has midi: {'midi' in sample_enhanced}")
            print(f"   Has rms_db: {'rms_db' in sample_enhanced}")
            print(f"   Has gesture_token: {'gesture_token' in sample_enhanced}")
            if 'gesture_token' in sample_enhanced:
                print(f"   gesture_token value: {sample_enhanced['gesture_token']}")
            
            # Check all events for gesture tokens
            tokens_after = [e.get('gesture_token') for e in enhanced_events]
            unique_after = set(t for t in tokens_after if t is not None)
            print(f"   🔍 Gesture tokens after enhancement: {len(unique_after)} unique tokens")
            if unique_after:
                print(f"   🔍 Sample tokens: {sorted(unique_after)[:10]}")
        
        # Apply smoothing/gesture processing to prevent over-sampling of sustained notes
        if self.use_musical_gestures and self.musical_gesture_processor:
            # Check if gestures were already applied in Step 4b (new approach)
            if hasattr(self, '_gestures_applied_in_step4b') and self._gestures_applied_in_step4b:
                print(f"\n✅ Musical gestures already applied during vocabulary training (Step 4b)")
                print(f"   Quantizer trained on consolidated gesture features")
                print(f"   Method: {self.musical_gesture_processor.consolidation_method}")
                print(f"   Events: {len(enhanced_events)}")
                # Skip re-applying gestures - tokens are already based on consolidated features
            else:
                # Legacy path: apply gestures after token assignment (for backward compatibility)
                print(f"\n⚠️  Applying musical gestures POST token assignment (legacy mode)")
                print(f"   Note: This won't affect token diversity (tokens already assigned)")
                # NEW APPROACH: Musical gesture processing
                print(f"\n🎵 Applying musical gesture processing...")
                print(f"   Before processing: {len(enhanced_events)} raw events")
                
                # Extract features and timestamps from events (use 't' key for timestamp)
                features = np.array([e['features'] for e in enhanced_events if 'features' in e])
                timestamps = np.array([e.get('t', e.get('timestamp', 0.0)) for e in enhanced_events])
                
                # Process into musical gestures
                consolidated_features, consolidated_timestamps, gestures = self.musical_gesture_processor.process_features(
                    features, timestamps
                )
                
                # Reconstruct events from gestures
                gesture_events = []
                for i, (feature, timestamp, gesture) in enumerate(zip(consolidated_features, consolidated_timestamps, gestures)):
                    # Copy first matching event as template
                    template_event = enhanced_events[0].copy()
                    template_event['features'] = feature
                    template_event['t'] = timestamp  # Use 't' key for timestamp, not 'timestamp'
                    template_event['gesture_metadata'] = {
                        'duration': gesture.duration,
                        'event_count': gesture.event_count,
                        'variance': gesture.feature_variance,
                        'boundary_type': gesture.boundary_type
                    }
                    gesture_events.append(template_event)
                
                enhanced_events = gesture_events
                print(f"   After processing: {len(enhanced_events)} musical gestures")
                self.musical_gesture_processor.print_statistics()
            
        elif self.enable_temporal_smoothing:
            # LEGACY APPROACH: Temporal smoothing
            print(f"\n🔄 Applying temporal smoothing ({self.temporal_window}s window, {self.temporal_threshold} threshold)...")
            original_count = len(enhanced_events)
            print(f"   Before smoothing: {original_count} events")
            enhanced_events = self.temporal_smoother.smooth_events(enhanced_events)
            smoothed_count = len(enhanced_events)
            print(f"   After smoothing: {smoothed_count} events")
            if original_count > smoothed_count:
                reduction_percent = ((original_count - smoothed_count) / original_count) * 100
                print(f"   ✅ Removed {original_count - smoothed_count} duplicate events ({reduction_percent:.1f}% reduction)")
            else:
                print(f"   ℹ️  No duplicates removed (all events represent distinct musical moments)")
        else:
            print(f"\n⚠️  No smoothing/gesture processing - using all {len(enhanced_events)} raw events for maximum rhythmic detail")

        
        # Train AudioOracle
        training_success = self.hybrid_trainer.train_from_events(
            enhanced_events, 
            file_info={'transformer_insights': transformer_insights}
        )
        
        # Get training result from AudioOracle
        if training_success:
            # Get statistics from the actual AudioOracle instance
            audio_oracle = self.hybrid_trainer.audio_oracle_mps if self.hybrid_trainer.use_mps else self.hybrid_trainer.audio_oracle
            
            # Find harmonic and polyphonic patterns
            harmonic_patterns = audio_oracle.find_harmonic_patterns() if audio_oracle else []
            polyphonic_patterns = audio_oracle.find_polyphonic_patterns() if audio_oracle else []
            
            # Get updated statistics after pattern detection
            audio_oracle_stats = audio_oracle.get_statistics() if audio_oracle else {}
            audio_oracle_stats['harmonic_patterns'] = len(harmonic_patterns)
            audio_oracle_stats['polyphonic_patterns'] = len(polyphonic_patterns)
            
            training_result = {
                'training_successful': True,
                'audio_oracle_stats': audio_oracle_stats,
                'events_processed': len(enhanced_events),
                'harmonic_patterns': len(harmonic_patterns),
                'polyphonic_patterns': len(polyphonic_patterns),
                'sample_harmonic_patterns': harmonic_patterns[:5],  # First 5 patterns
                'sample_polyphonic_patterns': polyphonic_patterns[:5]  # First 5 patterns
            }
        else:
            training_result = {
                'training_successful': False,
                'error': 'Training failed'
            }
        
        self.training_stats['audio_oracle_training_time'] = time.time() - training_start
        self.training_stats['training_result'] = training_result
        
        # Step 9: Save Enhanced Results
        print("\n🔄 Step 9: Saving Enhanced Results...")
        
        # Create enhanced output with progress
        print("📊 Creating enhanced output structure...")
        enhanced_output = self._create_enhanced_output(
            training_result, transformer_insights, hierarchical_result, rhythmic_result, correlation_result, gpt_oss_insights, performance_arc, gpt_oss_arc_insights
        )
        
        # Determine base filename for all model-related files
        # Avoid double "_model" in filename
        if output_file.endswith('_model.json'):
            model_base = output_file.replace('_model.json', '')  # Remove _model.json suffix
        else:
            model_base = output_file.replace('.json', '')  # Remove .json suffix
        
        # Save AudioOracle model for live performance
        model_file = f"{model_base}_model.json"
        if training_success:
            audio_oracle = self.hybrid_trainer.audio_oracle_mps if self.hybrid_trainer.use_mps else self.hybrid_trainer.audio_oracle
            if audio_oracle:
                print(f"💾 Saving AudioOracle model to {model_file}...")
                model_saved = audio_oracle.save_to_file(model_file)
                if model_saved:
                    print(f"✅ AudioOracle model saved successfully!")
                    # Get model statistics
                    stats = audio_oracle.get_statistics()
                    print(f"📊 Model contains: {stats.get('total_patterns', 0)} patterns, "
                          f"{stats.get('harmonic_patterns', 0)} harmonic patterns, "
                          f"{stats.get('polyphonic_patterns', 0)} polyphonic patterns")
                    
                    # Also save as pickle for faster loading
                    pickle_file = f"{model_base}_model.pkl.gz"
                    print(f"💾 Saving AudioOracle model to pickle format: {pickle_file}...")
                    try:
                        pickle_size = audio_oracle.save_to_pickle(pickle_file)
                        print(f"✅ Pickle model saved successfully! ({pickle_size:.1f} MB)")
                        print(f"💡 Pickle format loads 10-50× faster than JSON")
                    except Exception as e:
                        print(f"⚠️  Failed to save pickle format: {e}")
                        print(f"   (JSON model is still available)")
                else:
                    print(f"❌ Failed to save AudioOracle model")
        
        # Save RhythmOracle model for rhythmic phrasing
        rhythm_oracle_file = f"{model_base}_rhythm_oracle.json"
        if self.rhythm_oracle:
            print(f"🥁 Saving RhythmOracle model to {rhythm_oracle_file}...")
            try:
                self.rhythm_oracle.save_patterns(rhythm_oracle_file)
                rhythm_stats = self.rhythm_oracle.get_rhythmic_statistics()
                print(f"✅ RhythmOracle model saved successfully!")
                print(f"📊 Model contains: {rhythm_stats['total_patterns']} rhythmic patterns, "
                      f"avg tempo {rhythm_stats['avg_tempo']:.1f} BPM, "
                      f"avg density {rhythm_stats['avg_density']:.2f}")
            except Exception as e:
                print(f"❌ Failed to save RhythmOracle model: {e}")
        
        # Build and save harmonic transition graph for learned chord progressions
        harmonic_transition_graph = self._build_harmonic_transition_graph()
        if harmonic_transition_graph:
            transition_graph_file = f"{model_base}_harmonic_transitions.json"
            print(f"🎼 Saving harmonic transition graph to {transition_graph_file}...")
            try:
                with open(transition_graph_file, 'w') as f:
                    json.dump(harmonic_transition_graph, f, indent=2)
                print(f"✅ Harmonic transition graph saved successfully!")
                print(f"   Use this with HarmonicProgressor for intelligent chord selection")
            except Exception as e:
                print(f"❌ Failed to save harmonic transition graph: {e}")
        
        # Save PerformanceArc for timeline/structured performances
        if performance_arc:
            # Determine arc filename (use ai_learning_data/ directory for persistent models)
            import os
            base_name = os.path.basename(audio_file).replace('.wav', '').replace('.mp3', '').replace('.flac', '')
            arc_file = f"ai_learning_data/{base_name}_performance_arc.json"
            
            print(f"🎭 Saving PerformanceArc to {arc_file}...")
            try:
                arc_dict = performance_arc.to_dict()
                
                # Convert NumPy types to Python native types for JSON serialization
                def convert_numpy_types(obj):
                    """Recursively convert NumPy types to Python native types"""
                    if isinstance(obj, dict):
                        return {key: convert_numpy_types(value) for key, value in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_numpy_types(item) for item in obj]
                    elif isinstance(obj, (np.integer, np.int32, np.int64)):
                        return int(obj)
                    elif isinstance(obj, (np.floating, np.float32, np.float64)):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    else:
                        return obj
                
                arc_dict = convert_numpy_types(arc_dict)
                
                with open(arc_file, 'w') as f:
                    json.dump(arc_dict, f, indent=2)
                print(f"✅ PerformanceArc saved successfully!")
                print(f"📊 Arc contains: {len(performance_arc.phases)} phases, "
                      f"{performance_arc.total_duration:.1f}s duration, "
                      f"{len(performance_arc.silence_patterns)} silence patterns")
            except Exception as e:
                print(f"❌ Failed to save PerformanceArc: {e}")
                import traceback
                traceback.print_exc()
        
        # Save correlation patterns for live use (use same base name)
        correlation_file = f"{model_base}_correlation_patterns.json"
        if hasattr(self, 'correlation_analyzer') and self.correlation_analyzer.correlation_patterns:
            correlation_data = {
                'patterns': [
                    {
                        'pattern_id': p.pattern_id,
                        'harmonic_signature': p.harmonic_signature,
                        'rhythmic_signature': p.rhythmic_signature,
                        'correlation_strength': p.correlation_strength,
                        'frequency': p.frequency
                    }
                    for p in self.correlation_analyzer.correlation_patterns.values()
                ],
                'temporal_alignments': [
                    {
                        'harmonic_event_time': a.harmonic_event_time,
                        'rhythmic_event_time': a.rhythmic_event_time,
                        'alignment_strength': a.alignment_strength,
                        'phase_relationship': a.phase_relationship
                    }
                    for a in self.correlation_analyzer.temporal_alignments[:100]  # Top 100
                ]
            }
            
            try:
                with open(correlation_file, 'w') as f:
                    json.dump(correlation_data, f, indent=2)
                
                print(f"✅ Saved {len(correlation_data['patterns'])} correlation patterns to {correlation_file}")
            except Exception as e:
                print(f"⚠️  Failed to save correlation patterns: {e}")
        
        # Save gesture vocabulary (Wav2Vec quantizer) for live use
        if self.dual_perception:
            if self.dual_perception.enable_dual_vocabulary:
                # Save BOTH harmonic and percussive vocabularies
                harmonic_vocab_file = f"{model_base}_harmonic_vocab.joblib"
                percussive_vocab_file = f"{model_base}_percussive_vocab.joblib"
                
                try:
                    self.dual_perception.save_vocabulary(harmonic_vocab_file, "harmonic")
                    print(f"✅ Saved harmonic vocabulary to {harmonic_vocab_file}")
                except Exception as e:
                    print(f"⚠️  Failed to save harmonic vocabulary: {e}")
                    import traceback
                    traceback.print_exc()
                
                try:
                    self.dual_perception.save_vocabulary(percussive_vocab_file, "percussive")
                    print(f"✅ Saved percussive vocabulary to {percussive_vocab_file}")
                except Exception as e:
                    print(f"⚠️  Failed to save percussive vocabulary: {e}")
                    import traceback
                    traceback.print_exc()
                
                # CRITICAL FIX: Also save gesture quantizer for pattern matching
                # In dual vocabulary mode, we use harmonic quantizer as the main gesture quantizer
                # (both harmonic and percussive are gesture quantizers, but we need one for backwards compatibility)
                gesture_quantizer_file = f"{model_base}_gesture_training_quantizer.joblib"
                try:
                    # Save harmonic quantizer as the main gesture quantizer
                    # (it's trained on Wav2Vec features just like percussive)
                    if self.dual_perception.harmonic_quantizer and self.dual_perception.harmonic_quantizer.is_fitted:
                        self.dual_perception.harmonic_quantizer.save(gesture_quantizer_file)
                        print(f"✅ Saved gesture vocabulary (Wav2Vec) to {gesture_quantizer_file}")
                        # Verify file was created
                        import os
                        if os.path.exists(gesture_quantizer_file):
                            file_size = os.path.getsize(gesture_quantizer_file) / 1024  # KB
                            print(f"   📁 File verified: {file_size:.1f} KB")
                        else:
                            print(f"   ❌ ERROR: File not found after save!")
                    else:
                        print(f"⚠️  Harmonic quantizer not fitted - cannot save gesture quantizer")
                except Exception as e:
                    print(f"⚠️  Failed to save gesture quantizer: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                # Save single gesture vocabulary (traditional mode)
                gesture_quantizer_file = f"{model_base}_gesture_training_quantizer.joblib"
                try:
                    self.dual_perception.save_quantizer(gesture_quantizer_file)
                    print(f"✅ Saved gesture vocabulary (Wav2Vec) to {gesture_quantizer_file}")
                except Exception as e:
                    print(f"⚠️  Failed to save gesture quantizer: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Save symbolic quantizer vocabulary (hybrid/traditional) if available
        if self.hybrid_perception:
            symbolic_quantizer_file = f"{model_base}_symbolic_training_quantizer.joblib"
            try:
                self.hybrid_perception.save_quantizer(symbolic_quantizer_file)
                print(f"✅ Saved symbolic vocabulary (traditional) to {symbolic_quantizer_file}")
            except Exception as e:
                print(f"⚠️  Failed to save symbolic quantizer: {e}")
                import traceback
                traceback.print_exc()
        
        if not self.dual_perception and not self.hybrid_perception:
            print(f"⚠️  No perception module available to save quantizer")
        
        # Save to file with progress
        print("💾 Serializing and writing JSON file...")
        self._save_json_with_progress(enhanced_output, output_file)
        
        self.training_stats['total_training_time'] = time.time() - start_time
        
        # Print summary
        self._print_training_summary()
        
        return enhanced_output
    
    def _train_rhythm_oracle(self, rhythmic_result: RhythmicAnalysis):
        """
        Train RhythmOracle with rhythmic analysis results
        
        Now uses rational_structure analysis for tempo-independent patterns.
        Each detected pattern gets its duration pattern extracted from rational analysis.
        """
        
        print("🥁 Training RhythmOracle with tempo-independent patterns...")
        
        # Add each detected pattern to the RhythmOracle
        patterns_with_ratios = 0
        patterns_without_ratios = 0
        
        for pattern in rhythmic_result.patterns:
            # Try to extract rational structure for this pattern's onsets
            pattern_onsets = []
            for onset_time in rhythmic_result.onsets:
                if pattern.start_time <= onset_time <= pattern.end_time:
                    pattern_onsets.append(onset_time)
            
            # Analyze rational structure for this pattern if enough onsets
            duration_pattern = None
            pulse = 4
            complexity = 0.0
            
            if len(pattern_onsets) >= 3:
                try:
                    rational = self.rhythmic_analyzer.analyze_rational_structure(np.array(pattern_onsets))
                    if rational:
                        duration_pattern = rational['duration_pattern']
                        pulse = rational['pulse']
                        complexity = rational['complexity']
                        patterns_with_ratios += 1
                except Exception as e:
                    print(f"   ⚠️  Could not analyze rational structure for pattern: {e}")
            
            # Fallback: create simple duration pattern if rational analysis failed
            if not duration_pattern:
                # Use simple interval-based pattern
                if len(pattern_onsets) >= 2:
                    intervals = np.diff(pattern_onsets)
                    # Normalize to smallest interval
                    min_interval = np.min(intervals)
                    if min_interval > 0:
                        duration_pattern = [int(round(interval / min_interval)) for interval in intervals]
                    else:
                        duration_pattern = [2, 2, 2, 2]  # Default
                else:
                    duration_pattern = [2, 2, 2, 2]  # Default for insufficient data
                patterns_without_ratios += 1
            
            pattern_data = {
                'duration_pattern': duration_pattern,
                'density': pattern.density,
                'syncopation': pattern.syncopation,
                'pulse': pulse,
                'complexity': complexity,
                'meter': pattern.meter,
                'pattern_type': pattern.pattern_type,
                'confidence': pattern.confidence,
                'context': {
                    'start_time': pattern.start_time,
                    'end_time': pattern.end_time,
                    'num_onsets': len(pattern_onsets),
                    'rhythmic_complexity': rhythmic_result.rhythmic_complexity
                }
            }
            
            self.rhythm_oracle.add_rhythmic_pattern(pattern_data)
        
        print(f"✅ RhythmOracle trained with {len(rhythmic_result.patterns)} patterns")
        print(f"   📊 {patterns_with_ratios} with rational structure, {patterns_without_ratios} with fallback")
        
        # Print statistics
        stats = self.rhythm_oracle.get_rhythmic_statistics()
        print(f"   📈 Avg density: {stats['avg_density']:.2f}, Avg syncopation: {stats['avg_syncopation']:.2f}")
        print(f"   📈 Avg complexity: {stats['avg_complexity']:.2f}")
    
    def _convert_event_to_dict(self, event) -> Dict:
        """
        Convert Event object to dictionary with all audio features
        
        This preserves f0, midi, rms_db, and all other audio features
        that were missing from hierarchical sampling.
        """
        # Extract polyphonic information if available
        polyphonic_pitches = getattr(event, 'polyphonic_pitches', [event.f0])
        polyphonic_midi = getattr(event, 'polyphonic_midi', [event.midi])
        polyphonic_cents = getattr(event, 'polyphonic_cents', [event.cents])
        
        return {
            't': event.t,
            'f0': event.f0,
            'midi': event.midi,
            'cents': event.cents,
            'rms_db': event.rms_db,
            'centroid': event.centroid,
            'rolloff': getattr(event, 'rolloff', 3000.0),
            'bandwidth': getattr(event, 'bandwidth', 1000.0),
            'onset': event.onset,
            'ioi': event.ioi,
            'attack_time': getattr(event, 'attack_time', 0.1),
            'release_time': getattr(event, 'decay_time', 0.3),
            'zcr': getattr(event, 'zcr', 0.0),
            'mfcc_1': event.mfcc[0] if event.mfcc and len(event.mfcc) > 0 else 0.0,
            'mfcc_2': event.mfcc[1] if event.mfcc and len(event.mfcc) > 1 else 0.0,
            'mfcc_3': event.mfcc[2] if event.mfcc and len(event.mfcc) > 2 else 0.0,
            'instrument': event.instrument,
            # Polyphonic features
            'polyphonic_pitches': polyphonic_pitches,
            'polyphonic_midi': polyphonic_midi,
            'polyphonic_cents': polyphonic_cents,
            'num_pitches': len(polyphonic_pitches),
            'chord_quality': getattr(event, 'chord_quality', 'single'),
            'root_note': getattr(event, 'root_note', event.midi),
            # Additional features
            'rolloff_85': getattr(event, 'rolloff_85', 2500.0),
            'rolloff_95': getattr(event, 'rolloff_95', 3500.0),
            'contrast': getattr(event, 'contrast', 0.5),
            'flatness': getattr(event, 'flatness', 0.1),
            # Melodic salience features (NEW)
            'melodic_voice_idx': getattr(event, 'melodic_voice_idx', 0),
            'voice_roles': getattr(event, 'voice_roles', []),
            'melodic_pitch': getattr(event, 'melodic_pitch', event.f0),
            'melodic_midi': getattr(event, 'melodic_midi', event.midi),
            'is_melody': getattr(event, 'is_melody', False),
        }
    
    def _merge_hierarchical_with_audio_events(self, audio_events: List[Dict], 
                                             sampled_events: List) -> List[Dict]:
        """
        Merge hierarchical sampling scores with full audio events
        
        CRITICAL FIX: This ensures we keep real audio features (f0, midi, rms_db, etc.)
        while benefiting from hierarchical significance scoring.
        
        Args:
            audio_events: Events from PolyphonicAudioProcessor with full audio features
            sampled_events: SampledEvent objects from hierarchical analysis with significance scores
            
        Returns:
            List of audio events enriched with hierarchical metadata
        """
        # Create a time-based lookup for hierarchical events
        hierarchical_lookup = {}
        for sampled_event in sampled_events:
            hierarchical_lookup[sampled_event.time] = {
                'significance_score': sampled_event.significance_score,
                'sampling_reason': sampled_event.sampling_reason,
                'event_id': sampled_event.event_id
            }
        
        # Enrich audio events with hierarchical data
        merged_events = []
        for audio_event in audio_events:
            event_time = audio_event.get('t', 0)
            
            # Find closest hierarchical event (within 0.1 second tolerance)
            closest_time = None
            min_diff = float('inf')
            for h_time in hierarchical_lookup.keys():
                diff = abs(h_time - event_time)
                if diff < min_diff and diff < 0.1:  # 100ms tolerance
                    min_diff = diff
                    closest_time = h_time
            
            # Add hierarchical metadata if found
            if closest_time:
                h_data = hierarchical_lookup[closest_time]
                audio_event['significance_score'] = h_data['significance_score']
                audio_event['sampling_reason'] = h_data['sampling_reason']
                audio_event['event_id'] = h_data['event_id']
            else:
                # Default significance if no match
                audio_event['significance_score'] = 0.5
                audio_event['sampling_reason'] = 'polyphonic_processor'
            
            merged_events.append(audio_event)
        
        # Sort by significance score (descending) and select top events
        merged_events.sort(key=lambda x: x.get('significance_score', 0), reverse=True)
        
        # Select top N events based on hierarchical sampling target
        target_count = len(sampled_events)
        if len(merged_events) > target_count:
            merged_events = merged_events[:target_count]
            print(f"   Selected top {target_count} events based on hierarchical significance")
        
        # Re-sort by time for sequential learning
        merged_events.sort(key=lambda x: x.get('t', 0))
        
        return merged_events
    
    def _augment_with_hybrid_features(self, audio_file: str, events: List[Dict]) -> List[Dict]:
        """
        Augment events with hybrid perception features using temporal segmentation
        
        Uses IRCAM-recommended 350ms temporal segments instead of frame-by-frame analysis.
        This captures complete musical gestures and improves pattern learning.
        
        Args:
            audio_file: Path to audio file
            events: List of event dictionaries from standard processing
            
        Returns:
            List of events with augmented hybrid features
        """
        import librosa
        from listener.temporal_segmenter import TemporalSegmenter
        
        # Load audio file
        audio, sr = librosa.load(audio_file, sr=44100)
        
        # Create temporal segmenter (350ms = IRCAM recommended balanced mode)
        segmenter = TemporalSegmenter(segment_duration_ms=350.0)
        segments = segmenter.segment_audio(audio, sr)
        
        print(f"   Using temporal segmentation: {len(segments)} segments ({segmenter.segment_duration_ms}ms each)")
        print(f"   Extracting hybrid features for {len(segments)} segments...")
        
        # Extract features for each segment
        segment_features = []
        for i, segment in enumerate(segments):
            # Extract hybrid features from this segment
            hybrid_result = self.hybrid_perception.extract_features(
                segment.audio, 
                segment.sample_rate, 
                segment.start_time
            )
            
            segment_features.append({
                'start_time': segment.start_time,
                'end_time': segment.end_time,
                'features': hybrid_result.features,
                'consonance': hybrid_result.consonance,
                'chroma': hybrid_result.chroma,
                'active_pcs': hybrid_result.active_pitch_classes,
                'ratio_analysis': hybrid_result.ratio_analysis
            })
            
            if (i + 1) % 50 == 0 or (i + 1) == len(segments):
                print(f"   Processed {i + 1}/{len(segments)} segments")
        
        # Map events to nearest segment
        print(f"   Mapping {len(events)} events to segments...")
        for event in events:
            event_time = event.get('t', 0)
            
            # Find closest segment
            closest_segment = min(segment_features, 
                                 key=lambda s: abs(s['start_time'] + (s['end_time'] - s['start_time'])/2 - event_time))
            
            # Augment event with segment features
            event['hybrid_features'] = closest_segment['features'].tolist()
            event['hybrid_consonance'] = closest_segment['consonance']
            event['hybrid_chroma'] = closest_segment['chroma'].tolist()
            event['hybrid_active_pcs'] = closest_segment['active_pcs'].tolist()
            
            # Add ratio analysis if available
            if closest_segment['ratio_analysis']:
                event['hybrid_ratio_chord'] = closest_segment['ratio_analysis'].chord_match['type']
                event['hybrid_ratio_confidence'] = closest_segment['ratio_analysis'].chord_match['confidence']
                event['hybrid_ratio_fundamental'] = float(closest_segment['ratio_analysis'].fundamental)
            
            # Symbolic token will be added after vocabulary training
            event['hybrid_symbolic_token'] = None
        
        # Train symbolic vocabulary using segment features
        # Convert to numpy arrays and ensure consistent shape + float64 for sklearn
        all_hybrid_features = []
        for sf in segment_features:
            feat = sf['features']
            # Ensure it's a 1D numpy array with float64 dtype
            if isinstance(feat, np.ndarray):
                if feat.ndim > 1:
                    feat = feat.flatten()
                all_hybrid_features.append(feat.astype(np.float64))
            else:
                all_hybrid_features.append(np.array(feat, dtype=np.float64).flatten())
        
        if len(all_hybrid_features) > 0:
            # Verify all features have the same dimension
            feature_dims = [f.shape[0] for f in all_hybrid_features]
            if len(set(feature_dims)) > 1:
                print(f"   ⚠️ WARNING: Inconsistent feature dimensions detected: {set(feature_dims)}")
                # Filter to most common dimension
                from collections import Counter
                most_common_dim = Counter(feature_dims).most_common(1)[0][0]
                all_hybrid_features = [f for f in all_hybrid_features if f.shape[0] == most_common_dim]
                print(f"   ✓ Filtered to {len(all_hybrid_features)} features with dimension {most_common_dim}")
            else:
                print(f"   ✓ All features have consistent dimension: {feature_dims[0]}")
            
            # Adjust vocabulary size if we don't have enough samples
            original_vocab_size = self.hybrid_perception.vocabulary_size
            effective_vocab_size = min(original_vocab_size, len(all_hybrid_features))
            
            if effective_vocab_size < original_vocab_size:
                print(f"   ⚠️  Reducing vocabulary from {original_vocab_size} to {effective_vocab_size} classes (limited by {len(all_hybrid_features)} samples)")
                # Recreate quantizer with adjusted vocabulary size
                from listener.symbolic_quantizer import SymbolicQuantizer
                self.hybrid_perception.quantizer = SymbolicQuantizer(
                    vocabulary_size=effective_vocab_size,
                    n_init=10,
                    random_state=42
                )
                self.hybrid_perception.vocabulary_size = effective_vocab_size
            
            print(f"   Training symbolic vocabulary with {len(all_hybrid_features)} feature vectors...")
            self.hybrid_perception.train_vocabulary(all_hybrid_features)
            
            # Now assign symbolic tokens to events
            print(f"   Assigning symbolic tokens...")
            # Get the trained feature dimension
            trained_dim = all_hybrid_features[0].shape[0]
            
            for i, event in enumerate(events):
                hybrid_features = np.array(event['hybrid_features'], dtype=np.float64)  # Ensure float64 for sklearn
                
                # Ensure feature dimension matches trained dimension
                if len(hybrid_features) != trained_dim:
                    # Pad or truncate to match
                    if len(hybrid_features) < trained_dim:
                        # Pad with zeros
                        hybrid_features = np.pad(hybrid_features, (0, trained_dim - len(hybrid_features)))
                    else:
                        # Truncate
                        hybrid_features = hybrid_features[:trained_dim]
                    # Update event with corrected features
                    event['hybrid_features'] = hybrid_features.tolist()
                
                token = self.hybrid_perception.quantizer.transform(hybrid_features.reshape(1, -1))[0]
                event['hybrid_symbolic_token'] = int(token)
            
            # Print vocabulary stats
            vocab_info = self.hybrid_perception.quantizer.get_codebook_statistics()
            print(f"   ✅ Vocabulary trained: {vocab_info['vocabulary_size']} classes")
            print(f"      Active tokens: {vocab_info['active_tokens']}/{vocab_info['vocabulary_size']}")
            print(f"      Entropy: {vocab_info['entropy']:.3f} bits")
            print(f"   ✅ Temporal segmentation: {len(segments)} musical gestures captured")
        
        return events
    
    def _augment_with_dual_features(self, audio_file: str, events: List[Dict]) -> List[Dict]:
        """
        Augment events with DUAL perception features (Wav2Vec + Ratios in parallel)
        
        KEY ARCHITECTURAL INSIGHT:
        - Machine works with GESTURE TOKENS + RATIOS (not chord names!)
        - Wav2Vec tokens ARE the learned patterns: "Token 42 → Token 87 when consonance > 0.8"
        - Ratio analysis provides psychoacoustic truth (mathematical relationships)
        - Chord names are ONLY for human display, not for machine reasoning
        
        This is how IRCAM AudioOracle was intended to work!
        
        Args:
            audio_file: Path to audio file
            events: List of event dictionaries
            
        Returns:
            Events with dual features (machine logic + human translation)
        """
        import librosa
        from listener.temporal_segmenter import TemporalSegmenter
        
        # Load audio
        audio, sr = librosa.load(audio_file, sr=44100)
        
        # DUAL VOCABULARY MODE: Apply HPSS to separate harmonic and percussive sources
        if self.dual_perception.enable_dual_vocabulary:
            print(f"   🎸🥁 Dual vocabulary mode: Applying HPSS separation...")
            audio_harmonic, audio_percussive = librosa.effects.hpss(
                audio,
                kernel_size=31,  # Balance between separation quality and processing time
                power=2.0,       # Standard power spectrogram
                mask=True        # Use masking for cleaner separation
            )
            
            # Calculate energy ratios for verification
            harmonic_energy = np.sum(audio_harmonic ** 2)
            percussive_energy = np.sum(audio_percussive ** 2)
            total_energy = harmonic_energy + percussive_energy
            harm_ratio = harmonic_energy / total_energy if total_energy > 0 else 0.5
            perc_ratio = percussive_energy / total_energy if total_energy > 0 else 0.5
            
            print(f"      HPSS separation complete:")
            print(f"      • Harmonic energy: {harm_ratio:.1%}")
            print(f"      • Percussive energy: {perc_ratio:.1%}")
            
            # Segment both sources separately
            segmenter_harmonic = TemporalSegmenter(segment_duration_ms=350.0)
            segmenter_percussive = TemporalSegmenter(segment_duration_ms=350.0)
            
            harmonic_segments = segmenter_harmonic.segment_audio(audio_harmonic, sr)
            percussive_segments = segmenter_percussive.segment_audio(audio_percussive, sr)
            
            print(f"      • Harmonic segments: {len(harmonic_segments)}")
            print(f"      • Percussive segments: {len(percussive_segments)}")
            
            # Use combined audio for main processing (preserves correlations)
            segments = segmenter_harmonic.segment_audio(audio, sr)
            
            print(f"   Temporal segmentation: {len(segments)} combined segments (350ms each)")
            print(f"   🤖 Machine: Extracting dual gesture tokens (harmonic + percussive)")
            print(f"   👤 Human: Translating to chord labels (for display only)")
        else:
            # Traditional mode: single vocabulary
            # Segment audio (350ms as per IRCAM paper)
            segmenter = TemporalSegmenter(segment_duration_ms=350.0)
            segments = segmenter.segment_audio(audio, sr)
            harmonic_segments = None
            percussive_segments = None
            
            print(f"   Temporal segmentation: {len(segments)} segments (350ms each)")
            print(f"   🤖 Machine: Extracting gesture tokens + ratios (NO chord names!)")
            print(f"   👤 Human: Translating to chord labels (for display only)")
        
        # Extract dual features for each segment
        segment_features = []
        harmonic_wav2vec_features = [] if self.dual_perception.enable_dual_vocabulary else None
        percussive_wav2vec_features = [] if self.dual_perception.enable_dual_vocabulary else None
        
        for i, segment in enumerate(segments):
            # Get F0 from nearby events
            segment_time = (segment.start_time + segment.end_time) / 2
            nearest_event = min(events, key=lambda e: abs(e.get('t', 0) - segment_time))
            detected_f0 = nearest_event.get('f0', 0.0) if nearest_event.get('f0', 0.0) > 0 else None
            
            # Extract dual features using existing API (from COMBINED audio)
            dual_result = self.dual_perception.extract_features(
                audio=segment.audio,
                sr=segment.sample_rate,
                timestamp=segment.start_time,
                detected_f0=detected_f0
            )
            
            segment_features.append({
                'start_time': segment.start_time,
                'end_time': segment.end_time,
                'wav2vec_features': dual_result.wav2vec_features,
                'ratio_analysis': dual_result.ratio_analysis,
                'consonance': dual_result.consonance,
                'detected_frequencies': dual_result.detected_frequencies,
                'chord_label': dual_result.chord_label,  # Human display only
                'chord_confidence': dual_result.chord_confidence,
                'chroma': dual_result.chroma,
                'active_pcs': dual_result.active_pitch_classes
            })
            
            # DUAL VOCABULARY MODE: Extract Wav2Vec features from separated sources
            if self.dual_perception.enable_dual_vocabulary and harmonic_segments and percussive_segments:
                # Extract from harmonic source
                harmonic_segment = harmonic_segments[i]
                harmonic_wav2vec_result = self.dual_perception.wav2vec_encoder.encode(
                    audio=harmonic_segment.audio,
                    sr=harmonic_segment.sample_rate,
                    timestamp=harmonic_segment.start_time
                )
                if harmonic_wav2vec_result:
                    harmonic_wav2vec_features.append(harmonic_wav2vec_result.features)
                
                # Extract from percussive source
                percussive_segment = percussive_segments[i]
                percussive_wav2vec_result = self.dual_perception.wav2vec_encoder.encode(
                    audio=percussive_segment.audio,
                    sr=percussive_segment.sample_rate,
                    timestamp=percussive_segment.start_time
                )
                if percussive_wav2vec_result:
                    percussive_wav2vec_features.append(percussive_wav2vec_result.features)
            
            if (i + 1) % 50 == 0 or (i + 1) == len(segments):
                print(f"   Processed {i + 1}/{len(segments)} segments")
        
        # === MUSICAL GESTURE CONSOLIDATION (Applied BEFORE quantizer training) ===
        # This ensures the consolidation method (peak vs weighted_median) affects token diversity
        gesture_metadata = {}  # Track gesture info for later mapping
        
        if self.use_musical_gestures and self.musical_gesture_processor:
            print(f"\n   🎵 Applying musical gesture consolidation to Wav2Vec segments...")
            print(f"      Method: {self.musical_gesture_processor.consolidation_method}")
            print(f"      Segments before consolidation: {len(segment_features)}")
            
            if self.dual_perception.enable_dual_vocabulary:
                # Apply to both harmonic and percussive streams
                if harmonic_wav2vec_features:
                    harmonic_features_array = np.array(harmonic_wav2vec_features)
                    harmonic_timestamps = np.array([seg.start_time for seg in harmonic_segments])
                    
                    harmonic_consolidated, harmonic_times, harmonic_gestures = \
                        self.musical_gesture_processor.process_features(harmonic_features_array, harmonic_timestamps)
                    
                    # Store mapping info for event assignment
                    gesture_metadata['harmonic_original_features'] = harmonic_wav2vec_features
                    gesture_metadata['harmonic_consolidated_features'] = harmonic_consolidated
                    gesture_metadata['harmonic_consolidated_timestamps'] = harmonic_times
                    
                    harmonic_wav2vec_features = list(harmonic_consolidated)
                    print(f"      Harmonic: {len(harmonic_features_array)} → {len(harmonic_consolidated)} gestures " +
                          f"({len(harmonic_features_array) / len(harmonic_consolidated):.2f}x consolidation)")
                
                if percussive_wav2vec_features:
                    percussive_features_array = np.array(percussive_wav2vec_features)
                    percussive_timestamps = np.array([seg.start_time for seg in percussive_segments])
                    
                    percussive_consolidated, percussive_times, percussive_gestures = \
                        self.musical_gesture_processor.process_features(percussive_features_array, percussive_timestamps)
                    
                    # Store mapping info for event assignment
                    gesture_metadata['percussive_original_features'] = percussive_wav2vec_features
                    gesture_metadata['percussive_consolidated_features'] = percussive_consolidated
                    gesture_metadata['percussive_consolidated_timestamps'] = percussive_times
                    
                    percussive_wav2vec_features = list(percussive_consolidated)
                    print(f"      Percussive: {len(percussive_features_array)} → {len(percussive_consolidated)} gestures " +
                          f"({len(percussive_features_array) / len(percussive_consolidated):.2f}x consolidation)")
            else:
                # Single vocabulary mode
                segment_features_array = np.array([sf['wav2vec_features'] for sf in segment_features])
                segment_timestamps = np.array([sf['start_time'] for sf in segment_features])
                
                consolidated_features, consolidated_timestamps, gestures = \
                    self.musical_gesture_processor.process_features(segment_features_array, segment_timestamps)
                
                print(f"      Consolidated: {len(segment_features_array)} → {len(consolidated_features)} gestures " +
                      f"({len(segment_features_array) / len(consolidated_features):.2f}x consolidation)")
                
                # Store mapping from consolidated gestures back to original segments for event mapping
                gesture_metadata['consolidated_features'] = consolidated_features
                gesture_metadata['consolidated_timestamps'] = consolidated_timestamps
                gesture_metadata['gestures'] = gestures
                gesture_metadata['original_segment_features'] = segment_features
                
                # Update segment_features to use consolidated features for quantizer training
                # Find nearest original segment for each gesture to preserve metadata
                consolidated_segment_features = []
                for i, (feat, ts, gest) in enumerate(zip(consolidated_features, consolidated_timestamps, gestures)):
                    # Find original segment closest to this gesture timestamp
                    closest_original = min(gesture_metadata['original_segment_features'], 
                                          key=lambda s: abs(s['start_time'] - ts))
                    
                    # Debug: Check what fields are in closest_original
                    if i == 0:
                        print(f"      🔍 DEBUG: Original segment fields: {list(closest_original.keys())}")
                    
                    # Copy metadata from original segment, but use consolidated feature
                    consolidated_seg = closest_original.copy()
                    consolidated_seg['wav2vec_features'] = feat
                    consolidated_seg['start_time'] = ts
                    consolidated_seg['end_time'] = ts + gest.duration
                    consolidated_seg['gesture_metadata'] = {
                        'duration': gest.duration,
                        'event_count': gest.event_count,
                        'boundary_type': gest.boundary_type
                    }
                    consolidated_segment_features.append(consolidated_seg)
                
                segment_features = consolidated_segment_features
            
            # Mark that gestures were applied here (so Step 8 can skip)
            self._gestures_applied_in_step4b = True
            print(f"      ✅ Gesture consolidation complete (will train quantizer on consolidated features)")
        else:
            self._gestures_applied_in_step4b = False
        
        # Train gesture vocabulary/vocabularies
        if self.dual_perception.enable_dual_vocabulary:
            print(f"   🎓 Training DUAL vocabularies from {len(segment_features)} segments...")
            
            # Train harmonic vocabulary
            if harmonic_wav2vec_features:
                print(f"      Training harmonic vocabulary ({len(harmonic_wav2vec_features)} features)...")
                self.dual_perception.train_gesture_vocabulary(harmonic_wav2vec_features, "harmonic")
                print(f"      ✅ Harmonic tokens capture guitar/bass/sustained tones")
            
            # Train percussive vocabulary
            if percussive_wav2vec_features:
                print(f"      Training percussive vocabulary ({len(percussive_wav2vec_features)} features)...")
                self.dual_perception.train_gesture_vocabulary(percussive_wav2vec_features, "percussive")
                print(f"      ✅ Percussive tokens capture drums/hi-hats/transients")
            
            print(f"      ✅ Dual vocabularies trained! System can now respond appropriately to drums OR guitar")
        else:
            # Traditional single vocabulary training
            print(f"   🎓 Training gesture vocabulary from {len(segment_features)} segments...")
            wav2vec_features_list = [sf['wav2vec_features'] for sf in segment_features]
            self.dual_perception.train_gesture_vocabulary(wav2vec_features_list, "single")
            print(f"      ✅ Gesture tokens represent LEARNED PATTERNS, not chord names!")
        
        # Map segments to events and augment with DUAL representations
        print(f"   Mapping to {len(events)} events...")
        
        # 🔍 DEBUG: Check event timestamps
        sample_times = [e.get('t', 0) for e in events[:5]]
        print(f"   🔍 DEBUG: Sample event timestamps: {sample_times}")
        print(f"   🔍 DEBUG: Segment time range: {segments[0].start_time:.2f} to {segments[-1].end_time:.2f}")
        
        # Get audio duration for validation
        audio_duration = len(audio) / sr
        
        for event in events:
            event_time = event.get('t', 0)
            
            # Validate event timestamps are in correct range
            # Audio events should be in range [0, audio_duration]
            if event_time < 0 or event_time > audio_duration:
                print(f"   ⚠️  WARNING: Event timestamp {event_time:.2f}s outside audio duration {audio_duration:.2f}s")
                # Clamp to valid range
                event_time = max(0, min(event_time, audio_duration))
                event['t'] = event_time
            
            # Find closest segment
            closest_segment = min(segment_features,
                                key=lambda s: abs((s['start_time'] + s['end_time'])/2 - event_time))
            
            # Extract gesture token(s) for this segment (after vocabulary is trained)
            wav2vec_feat = closest_segment['wav2vec_features'].astype(np.float64)
            
            if self.dual_perception.enable_dual_vocabulary:
                # DUAL VOCABULARY MODE: Assign both harmonic and percussive tokens
                # Find closest consolidated features by timestamp
                harmonic_token = None
                if self.dual_perception.harmonic_quantizer and self.dual_perception.harmonic_quantizer.is_fitted:
                    if 'harmonic_consolidated_timestamps' in gesture_metadata:
                        # Use consolidated features
                        h_timestamps = gesture_metadata['harmonic_consolidated_timestamps']
                        h_features = gesture_metadata['harmonic_consolidated_features']
                        closest_h_idx = min(range(len(h_timestamps)),
                                          key=lambda i: abs(h_timestamps[i] - event_time))
                        harmonic_feat = h_features[closest_h_idx].astype(np.float64)
                    else:
                        # Fallback to direct lookup (if no consolidation)
                        segment_idx = min(range(len(harmonic_wav2vec_features)),
                                        key=lambda i: abs(harmonic_segments[i].start_time - event_time) if i < len(harmonic_segments) else float('inf'))
                        harmonic_feat = harmonic_wav2vec_features[segment_idx].astype(np.float64)
                    
                    harmonic_token = int(self.dual_perception.harmonic_quantizer.transform(harmonic_feat.reshape(1, -1))[0])
                
                # Assign percussive token
                percussive_token = None
                if self.dual_perception.percussive_quantizer and self.dual_perception.percussive_quantizer.is_fitted:
                    if 'percussive_consolidated_timestamps' in gesture_metadata:
                        # Use consolidated features
                        p_timestamps = gesture_metadata['percussive_consolidated_timestamps']
                        p_features = gesture_metadata['percussive_consolidated_features']
                        closest_p_idx = min(range(len(p_timestamps)),
                                          key=lambda i: abs(p_timestamps[i] - event_time))
                        percussive_feat = p_features[closest_p_idx].astype(np.float64)
                    else:
                        # Fallback to direct lookup (if no consolidation)
                        segment_idx = min(range(len(percussive_wav2vec_features)),
                                        key=lambda i: abs(percussive_segments[i].start_time - event_time) if i < len(percussive_segments) else float('inf'))
                        percussive_feat = percussive_wav2vec_features[segment_idx].astype(np.float64)
                    
                    percussive_token = int(self.dual_perception.percussive_quantizer.transform(percussive_feat.reshape(1, -1))[0])
                
                # === MACHINE REPRESENTATION (What AI actually works with) ===
                event['harmonic_token'] = harmonic_token
                event['percussive_token'] = percussive_token
                event['gesture_token'] = harmonic_token  # Legacy compatibility (use harmonic as default)
            else:
                # TRADITIONAL MODE: Single gesture token
                gesture_token = None
                if self.dual_perception.quantizer and self.dual_perception.quantizer.is_fitted:
                    gesture_token = int(self.dual_perception.quantizer.transform(wav2vec_feat.reshape(1, -1))[0])
                
                # === MACHINE REPRESENTATION (What AI actually works with) ===
                # Gesture token: The learned pattern ID (0-63)
                event['gesture_token'] = gesture_token
            
            # Wav2Vec features: The 768D neural encoding (for AudioOracle)
            event['features'] = closest_segment['wav2vec_features'].tolist()
            
            # Ratio analysis: Mathematical/psychoacoustic truth
            event['consonance'] = closest_segment['consonance']
            event['detected_frequencies'] = closest_segment['detected_frequencies']
            
            if closest_segment['ratio_analysis']:
                event['fundamental_freq'] = float(closest_segment['ratio_analysis'].fundamental)
                event['frequency_ratios'] = closest_segment['ratio_analysis'].ratios
            else:
                event['fundamental_freq'] = 0.0
                event['frequency_ratios'] = []
            
            # === HUMAN TRANSLATION (For display/logging only) ===
            event['chord_name_display'] = closest_segment['chord_label']  # e.g., "major", "minor"
            event['chord_confidence'] = closest_segment['chord_confidence']
            
            # Also keep chroma for compatibility
            event['dual_chroma'] = closest_segment['chroma'].tolist()
            event['dual_active_pcs'] = closest_segment['active_pcs'].tolist()
        
        # Print timestamp normalization summary
        print(f"   🔍 DEBUG: Sample event timestamps: {sample_times}")
        print(f"   🔍 DEBUG: Segment time range: {segments[0].start_time:.2f} to {segments[-1].end_time:.2f}")
        
        # Print summary showing DUAL representation
        unique_tokens = len(set(e.get('gesture_token') for e in events if e.get('gesture_token') is not None))
        avg_consonance = np.mean([e.get('consonance', 0) for e in events])
        
        # 🔍 DEBUG: Check what tokens were actually assigned
        tokens_assigned = [e.get('gesture_token') for e in events]
        none_count = tokens_assigned.count(None)
        unique_assigned = set(t for t in tokens_assigned if t is not None)
        print(f"\n   🔍 DEBUG - Token Assignment:")
        print(f"      Total events: {len(events)}")
        print(f"      Events with tokens: {len(events) - none_count}")
        print(f"      Events with None: {none_count}")
        print(f"      Unique tokens assigned: {len(unique_assigned)}")
        if unique_assigned:
            print(f"      Token range: {min(unique_assigned)} to {max(unique_assigned)}")
            print(f"      Sample tokens: {sorted(unique_assigned)[:20]}")
            print(f"      Sample events: {[(e.get('t', 0), e.get('gesture_token')) for e in events[:5]]}")
        
        print(f"\n   ✅ Dual perception complete:")
        print(f"\n   🤖 MACHINE PERCEPTION (What AI learns):")
        print(f"      • Gesture tokens: {unique_tokens} unique patterns")
        print(f"      • Average consonance: {avg_consonance:.3f}")
        print(f"      • Machine thinks in: tokens + ratios + consonance")
        
        print(f"\n   👤 HUMAN TRANSLATION (For display):")
        sample_chords = [e.get('chord_name_display', '?') for e in events[:10]]
        print(f"      • Sample chord types: {' → '.join(sample_chords[:5])} ...")
        print(f"      • Humans see: chord names")
        
        print(f"\n   ✨ KEY: Machine learns 'Token 42 → Token 87 when consonance > 0.8'")
        print(f"          Humans see 'major → minor'")
        
        return events
    
    def _convert_hierarchical_events_to_training_format(self, sampled_events: List) -> List[Dict]:
        """
        DEPRECATED: This method created events WITHOUT audio features (f0, midi, rms_db)
        Use _merge_hierarchical_with_audio_events instead
        """
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
                                    hierarchical_result: Optional[SimpleAnalysisResult],
                                    rhythmic_result: Optional[RhythmicAnalysis],
                                    correlation_result: Optional[Dict] = None,
                                    gpt_oss_insights: Optional[GPTOSSAnalysis] = None,
                                    performance_arc: Optional[PerformanceArc] = None,
                                    gpt_oss_arc_insights: Optional[GPTOSSArcAnalysis] = None) -> List[Dict]:
        """Enhance events with transformer, hierarchical, and rhythmic insights"""
        
        enhanced_events = []
        prev_event = None
        
        for i, event in enumerate(events):
            enhanced_event = event.copy()
            
            # Add relative parameters (interval relationships)
            if prev_event is not None:
                # MIDI relative (interval in semitones)
                current_midi = enhanced_event.get('midi', 0)
                prev_midi = prev_event.get('midi', 0)
                enhanced_event['midi_relative'] = current_midi - prev_midi
                
                # Velocity relative
                current_velocity = enhanced_event.get('velocity', 64)
                prev_velocity = prev_event.get('velocity', 64)
                enhanced_event['velocity_relative'] = current_velocity - prev_velocity
                
                # IOI relative (inter-onset interval ratio)
                current_ioi = enhanced_event.get('ioi', 0.0)
                prev_ioi = prev_event.get('ioi', 0.0)
                if prev_ioi > 0:
                    enhanced_event['ioi_relative'] = current_ioi / prev_ioi
                else:
                    enhanced_event['ioi_relative'] = 1.0
            else:
                # First event - no previous for comparison
                enhanced_event['midi_relative'] = 0
                enhanced_event['velocity_relative'] = 0
                enhanced_event['ioi_relative'] = 1.0
            
            # NEW: Use hybrid features if available (replaces standard features for AudioOracle)
            if 'hybrid_features' in event and event['hybrid_features']:
                # Replace standard features with hybrid perception features
                enhanced_event['features'] = event['hybrid_features']
                # Also keep hybrid metadata for reference
                enhanced_event['hybrid_enabled'] = True
                enhanced_event['hybrid_consonance'] = event.get('hybrid_consonance', 0.5)
                enhanced_event['hybrid_symbolic_token'] = event.get('hybrid_symbolic_token', None)
            
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
            
            # Add rhythmic insights if available
            if rhythmic_result:
                event_time = event.get('t', 0)
                
                # Find corresponding rhythmic pattern
                rhythmic_pattern = self._find_rhythmic_pattern_for_time(event_time, rhythmic_result)
                if rhythmic_pattern:
                    enhanced_event['rhythmic_insights'] = {
                        'pattern_type': rhythmic_pattern.pattern_type,
                        'tempo': rhythmic_pattern.tempo,
                        'density': rhythmic_pattern.density,
                        'syncopation': rhythmic_pattern.syncopation,
                        'confidence': rhythmic_pattern.confidence
                    }
                
                # Add global rhythmic context
                enhanced_event['rhythmic_context'] = {
                    'global_tempo': rhythmic_result.tempo,
                    'global_syncopation': rhythmic_result.syncopation_score,
                    'global_complexity': rhythmic_result.rhythmic_complexity,
                    'meter': rhythmic_result.meter
                }
                
                # Add rational structure fields if available
                if hasattr(rhythmic_result, 'rational_structure') and rhythmic_result.rational_structure:
                    rational = rhythmic_result.rational_structure
                    
                    # Find which position in the duration pattern this event corresponds to
                    # by matching event time to onsets
                    event_time = event.get('t', 0)
                    onset_idx = self._find_closest_onset_index(event_time, rhythmic_result.onsets)
                    
                    # Add rational rhythm fields
                    if onset_idx is not None and onset_idx < len(rational['duration_pattern']):
                        enhanced_event['rhythm_ratio'] = rational['duration_pattern'][onset_idx]
                        enhanced_event['rhythm_subdiv_tempo'] = rational['tempo']
                        
                        # Add deviation if available
                        if onset_idx < len(rational['deviations']):
                            enhanced_event['deviation'] = rational['deviations'][onset_idx]
                        else:
                            enhanced_event['deviation'] = 0.0
                        
                        # Add deviation polarity if available
                        if onset_idx < len(rational['deviation_polarity']):
                            enhanced_event['deviation_polarity'] = rational['deviation_polarity'][onset_idx]
                        else:
                            enhanced_event['deviation_polarity'] = 0
                    else:
                        # Default values if no match found
                        enhanced_event['rhythm_ratio'] = 1
                        enhanced_event['rhythm_subdiv_tempo'] = rational['tempo']
                        enhanced_event['deviation'] = 0.0
                        enhanced_event['deviation_polarity'] = 0
                    
                    # Add tempo reconciliation data if available
                    if rational.get('reconciled', False):
                        enhanced_event['tempo_factor'] = rational.get('tempo_factor', 1.0)
                        enhanced_event['tempo_reconciled'] = True
                        if 'prev_tempo' in rational:
                            enhanced_event['prev_tempo'] = rational['prev_tempo']
                    else:
                        enhanced_event['tempo_factor'] = 1.0
                        enhanced_event['tempo_reconciled'] = False
            
            # Add correlation insights (real chord data) if available
            if correlation_result:
                event_time = event.get('t', 0)
                
                # Find corresponding harmonic event from correlation analysis
                harmonic_event = self._find_harmonic_event_for_time(event_time, correlation_result)
                if harmonic_event:
                    # Extract chord data from harmonic event
                    if isinstance(harmonic_event, dict):
                        harmonic_features = harmonic_event.get('harmonic_features', {})
                    else:
                        harmonic_features = getattr(harmonic_event, 'harmonic_features', {})
                    
                    if harmonic_features:
                        enhanced_event['correlation_insights'] = {
                            'chord': harmonic_features.get('primary_chord', 'C'),
                            'chord_tension': harmonic_features.get('harmonic_tension', 0.5),
                            'key_stability': harmonic_features.get('key_stability', 0.5),
                            'chord_change_rate': harmonic_features.get('chord_change_rate', 0.0),
                            'harmonic_diversity': harmonic_features.get('chord_diversity', 0.0)
                        }
            
            # Add GPT-OSS insights if available
            if gpt_oss_insights:
                enhanced_event['gpt_oss_insights'] = {
                    'harmonic_analysis': gpt_oss_insights.harmonic_analysis,
                    'rhythmic_analysis': gpt_oss_insights.rhythmic_analysis,
                    'phrasing_analysis': gpt_oss_insights.phrasing_analysis,
                    'feel_analysis': gpt_oss_insights.feel_analysis,
                    'style_analysis': gpt_oss_insights.style_analysis,
                    'confidence_score': gpt_oss_insights.confidence_score,
                    'processing_time': gpt_oss_insights.processing_time
                }
                
                # Extract musical intelligence for pattern weighting
                enhanced_event['musical_intelligence'] = self._extract_musical_intelligence(gpt_oss_insights)
            
            # Add performance arc insights if available
            if performance_arc:
                event_time = event.get('t', 0)
                
                # Find corresponding phase in performance arc
                phase_info = self._find_phase_for_time(event_time, performance_arc)
                if phase_info:
                    enhanced_event['performance_arc_insights'] = {
                        'phase_type': phase_info.phase_type,
                        'engagement_level': phase_info.engagement_level,
                        'musical_density': phase_info.musical_density,
                        'dynamic_level': phase_info.dynamic_level,
                        'silence_ratio': phase_info.silence_ratio,
                        'instrument_roles': phase_info.instrument_roles
                    }
                
                # Add engagement curve information
                engagement_level = self._get_engagement_at_time(event_time, performance_arc)
                enhanced_event['engagement_level'] = engagement_level
            
            # Add GPT-OSS arc insights if available
            if gpt_oss_arc_insights:
                enhanced_event['gpt_oss_arc_insights'] = {
                    'structural_analysis': gpt_oss_arc_insights.structural_analysis,
                    'dynamic_evolution': gpt_oss_arc_insights.dynamic_evolution,
                    'emotional_arc': gpt_oss_arc_insights.emotional_arc,
                    'role_development': gpt_oss_arc_insights.role_development,
                    'silence_strategy': gpt_oss_arc_insights.silence_strategy,
                    'engagement_curve': gpt_oss_arc_insights.engagement_curve,
                    'confidence_score': gpt_oss_arc_insights.confidence_score,
                    'processing_time': gpt_oss_arc_insights.processing_time
                }
            
            enhanced_events.append(enhanced_event)
            
            # Update previous event for next iteration
            prev_event = enhanced_event
        
        return enhanced_events
    
    def _extract_musical_intelligence(self, gpt_oss_insights: GPTOSSAnalysis) -> Dict:
        """Extract musical intelligence metrics from GPT-OSS analysis for pattern weighting"""
        
        # Parse GPT-OSS analysis to extract musical intelligence
        intelligence = {
            'harmonic_sophistication': 0.5,  # Default
            'rhythmic_complexity': 0.5,     # Default
            'phrasing_quality': 0.5,         # Default
            'musical_coherence': 0.5,        # Default
            'creative_potential': 0.5         # Default
        }
        
        try:
            # Analyze harmonic sophistication from GPT-OSS text
            harmonic_text = gpt_oss_insights.harmonic_analysis.lower()
            if 'sophisticated' in harmonic_text or 'complex' in harmonic_text:
                intelligence['harmonic_sophistication'] = 0.8
            elif 'simple' in harmonic_text or 'basic' in harmonic_text:
                intelligence['harmonic_sophistication'] = 0.3
            
            # Analyze rhythmic complexity
            rhythmic_text = gpt_oss_insights.rhythmic_analysis.lower()
            if 'syncopated' in rhythmic_text or 'complex' in rhythmic_text:
                intelligence['rhythmic_complexity'] = 0.8
            elif 'simple' in rhythmic_text or 'straight' in rhythmic_text:
                intelligence['rhythmic_complexity'] = 0.3
            
            # Analyze phrasing quality
            phrasing_text = gpt_oss_insights.phrasing_analysis.lower()
            if 'well-structured' in phrasing_text or 'coherent' in phrasing_text:
                intelligence['phrasing_quality'] = 0.8
            elif 'fragmented' in phrasing_text or 'disjointed' in phrasing_text:
                intelligence['phrasing_quality'] = 0.3
            
            # Overall musical coherence
            intelligence['musical_coherence'] = gpt_oss_insights.confidence_score
            
            # Creative potential (based on analysis depth)
            if gpt_oss_insights.processing_time > 10:  # Longer analysis = more creative potential
                intelligence['creative_potential'] = 0.8
            else:
                intelligence['creative_potential'] = 0.6
                
        except Exception as e:
            print(f"⚠️ Error extracting musical intelligence: {e}")
        
        return intelligence
    
    def _find_phase_for_time(self, time: float, performance_arc: PerformanceArc) -> Optional[Any]:
        """Find performance phase for a given time"""
        try:
            for phase in performance_arc.phases:
                if phase.start_time <= time <= phase.end_time:
                    return phase
            return None
        except Exception as e:
            print(f"⚠️ Error finding phase for time {time}: {e}")
            return None
    
    def _get_engagement_at_time(self, time: float, performance_arc: PerformanceArc) -> float:
        """Get engagement level at a given time from the performance arc"""
        try:
            # Find the closest engagement curve point
            engagement_curve = performance_arc.overall_engagement_curve
            if not engagement_curve:
                return 0.5  # Default engagement
            
            # Calculate time position as fraction of total duration
            time_fraction = time / performance_arc.total_duration
            
            # Find the closest engagement curve index
            curve_index = int(time_fraction * (len(engagement_curve) - 1))
            curve_index = max(0, min(curve_index, len(engagement_curve) - 1))
            
            return float(engagement_curve[curve_index])
        except Exception as e:
            print(f"⚠️ Error getting engagement at time {time}: {e}")
            return 0.5  # Default engagement
    
    def _find_harmonic_event_for_time(self, time: float, correlation_result: Dict) -> Optional[Any]:
        """Find harmonic event from correlation analysis for a given time"""
        try:
            # Get joint events from correlation result
            joint_events = correlation_result.get('joint_events', [])
            
            if not joint_events:
                return None
            
            # Find the closest joint event by time
            closest_event = None
            min_time_diff = float('inf')
            
            for event in joint_events:
                # Handle both dict and object formats
                if isinstance(event, dict):
                    event_time = event.get('timestamp', 0)
                else:
                    event_time = getattr(event, 'timestamp', 0)
                
                time_diff = abs(event_time - time)
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    closest_event = event
            
            return closest_event
        except Exception as e:
            print(f"⚠️ Error finding harmonic event for time {time}: {e}")
            return None
    
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
    
    def _find_rhythmic_pattern_for_time(self, time: float, rhythmic_result: RhythmicAnalysis) -> Optional[Any]:
        """Find rhythmic pattern for a given time"""
        
        for pattern in rhythmic_result.patterns:
            if pattern.start_time <= time <= pattern.end_time:
                return pattern
        
        return None
    
    def _find_closest_onset_index(self, event_time: float, onsets: np.ndarray) -> Optional[int]:
        """
        Find the index of the closest onset to the given event time
        
        Args:
            event_time: Event timestamp
            onsets: Array of onset times
            
        Returns:
            Index of closest onset, or None if no onsets
        """
        if len(onsets) == 0:
            return None
        
        # Find closest onset
        time_diffs = np.abs(onsets - event_time)
        closest_idx = np.argmin(time_diffs)
        
        # Only return if within reasonable threshold (e.g., 0.1 seconds)
        if time_diffs[closest_idx] < 0.1:
            return int(closest_idx)
        
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
                              hierarchical_result: Optional[SimpleAnalysisResult],
                              rhythmic_result: Optional[RhythmicAnalysis],
                              correlation_result: Optional[Dict] = None,
                              gpt_oss_insights: Optional[GPTOSSAnalysis] = None,
                              performance_arc: Optional[PerformanceArc] = None,
                              gpt_oss_arc_insights: Optional[GPTOSSArcAnalysis] = None) -> Dict[str, Any]:
        """Create enhanced output with all insights"""
        
        enhanced_output = training_result.copy()
        
        # Add transformer insights
        if transformer_insights:
            transformer_data = {
                'key_signature': transformer_insights.key_signature,
                'tempo_analysis': transformer_insights.tempo_analysis,
                'chord_progression': transformer_insights.chord_progression,
                'musical_form': transformer_insights.musical_form,
                'scale_analysis': transformer_insights.scale_analysis,
                'confidence_scores': transformer_insights.confidence_scores
            }
            
            # Add voice leading analysis if available
            if transformer_insights.voice_leading_analysis:
                vl = transformer_insights.voice_leading_analysis
                transformer_data['voice_leading'] = {
                    'smoothness_score': vl.smoothness_score,
                    'stepwise_motion_percentage': vl.stepwise_motion_percentage,
                    'leap_percentage': vl.leap_percentage,
                    'parallel_fifths_count': len(vl.parallel_fifths),
                    'parallel_octaves_count': len(vl.parallel_octaves),
                    'contrary_motion_count': vl.contrary_motion_count,
                    'voice_crossings_count': len(vl.voice_crossings),
                    'voice_independence': vl.voice_independence,
                    'average_voice_range': vl.average_voice_range,
                    'parallel_fifths': vl.parallel_fifths[:5],  # First 5 violations
                    'parallel_octaves': vl.parallel_octaves[:5]
                }
            
            # Add bass line analysis if available
            if transformer_insights.bass_line_analysis:
                bl = transformer_insights.bass_line_analysis
                transformer_data['bass_line'] = {
                    'inversions': bl['inversions'],
                    'bass_contour': bl['bass_contour'],
                    'pedal_points': bl['pedal_points'],
                    'walking_bass': bl['walking_bass'],
                    'bass_range': bl['bass_range']
                }
            
            enhanced_output['transformer_analysis'] = transformer_data
        
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
        
        # Add rhythmic analysis
        if rhythmic_result:
            enhanced_output['rhythmic_analysis'] = {
                'tempo': rhythmic_result.tempo,
                'meter': rhythmic_result.meter,
                'syncopation_score': rhythmic_result.syncopation_score,
                'rhythmic_complexity': rhythmic_result.rhythmic_complexity,
                'patterns_detected': len(rhythmic_result.patterns),
                'patterns': [
                    {
                        'pattern_id': pattern.pattern_id,
                        'start_time': pattern.start_time,
                        'end_time': pattern.end_time,
                        'tempo': pattern.tempo,
                        'density': pattern.density,
                        'syncopation': pattern.syncopation,
                        'pattern_type': pattern.pattern_type,
                        'confidence': pattern.confidence
                    }
                    for pattern in rhythmic_result.patterns
                ],
                'density_profile': rhythmic_result.density_profile.tolist(),
                'beat_strength': rhythmic_result.beat_strength.tolist()
            }
            
            # Add RhythmOracle statistics
            if self.rhythm_oracle:
                enhanced_output['rhythm_oracle_stats'] = self.rhythm_oracle.get_rhythmic_statistics()
        
        # Add correlation analysis
        if correlation_result:
            enhanced_output['correlation_analysis'] = {
                'analysis_stats': correlation_result['analysis_stats'],
                'cross_modal_insights': correlation_result['cross_modal_insights'],
                'correlation_patterns': [
                    {
                        'pattern_id': pattern.pattern_id,
                        'harmonic_signature': pattern.harmonic_signature,
                        'rhythmic_signature': pattern.rhythmic_signature,
                        'correlation_strength': pattern.correlation_strength,
                        'frequency': pattern.frequency,
                        'contexts': pattern.contexts
                    }
                    for pattern in correlation_result['correlation_patterns']
                ],
                'temporal_alignments': [
                    {
                        'harmonic_event_time': alignment.harmonic_event_time,
                        'rhythmic_event_time': alignment.rhythmic_event_time,
                        'alignment_strength': alignment.alignment_strength,
                        'phase_relationship': alignment.phase_relationship
                    }
                    for alignment in correlation_result['temporal_analysis']
                ]
            }
        
        # Add GPT-OSS analysis
        # Add GPT-OSS insights if available
        if gpt_oss_insights:
            enhanced_output['gpt_oss_analysis'] = {
                'harmonic_analysis': gpt_oss_insights.harmonic_analysis,
                'rhythmic_analysis': gpt_oss_insights.rhythmic_analysis,
                'phrasing_analysis': gpt_oss_insights.phrasing_analysis,
                'feel_analysis': gpt_oss_insights.feel_analysis,
                'style_analysis': gpt_oss_insights.style_analysis,
                'confidence_score': gpt_oss_insights.confidence_score,
                'processing_time': gpt_oss_insights.processing_time,
                'analysis_timestamp': time.time()
            }
        
        # Add performance arc analysis
        if performance_arc:
            enhanced_output['performance_arc_analysis'] = {
                'total_duration': performance_arc.total_duration,
                'phases': [
                    {
                        'start_time': phase.start_time,
                        'end_time': phase.end_time,
                        'phase_type': phase.phase_type,
                        'engagement_level': phase.engagement_level,
                        'musical_density': phase.musical_density,
                        'dynamic_level': phase.dynamic_level,
                        'silence_ratio': phase.silence_ratio,
                        'instrument_roles': phase.instrument_roles
                    }
                    for phase in performance_arc.phases
                ],
                'overall_engagement_curve': performance_arc.overall_engagement_curve,
                'instrument_evolution': performance_arc.instrument_evolution,
                'silence_patterns': performance_arc.silence_patterns,
                'theme_development': performance_arc.theme_development,
                'dynamic_evolution': performance_arc.dynamic_evolution
            }
        
        # Add GPT-OSS arc analysis
        if gpt_oss_arc_insights:
            enhanced_output['gpt_oss_arc_analysis'] = {
                'structural_analysis': gpt_oss_arc_insights.structural_analysis,
                'dynamic_evolution': gpt_oss_arc_insights.dynamic_evolution,
                'emotional_arc': gpt_oss_arc_insights.emotional_arc,
                'role_development': gpt_oss_arc_insights.role_development,
                'silence_strategy': gpt_oss_arc_insights.silence_strategy,
                'engagement_curve': gpt_oss_arc_insights.engagement_curve,
                'confidence_score': gpt_oss_arc_insights.confidence_score,
                'processing_time': gpt_oss_arc_insights.processing_time,
                'analysis_timestamp': time.time()
            }
        
        # Add long-form arc structure analysis (Brandtsegg sections)
        if hasattr(self, 'arc_structure_sections') and self.arc_structure_sections:
            tempos = [s['local_tempo'] for s in self.arc_structure_sections]
            complexities = [s['rhythmic_complexity'] for s in self.arc_structure_sections]
            
            enhanced_output['arc_structure'] = {
                'sections': self.arc_structure_sections,
                'total_duration': self.arc_structure_sections[-1]['end_time'],
                'num_sections': len(self.arc_structure_sections),
                'tempo_range': [float(min(tempos)), float(max(tempos))],
                'complexity_range': [float(min(complexities)), float(max(complexities))],
                'phase_distribution': self._get_phase_distribution(self.arc_structure_sections),
                'analysis_timestamp': time.time()
            }
            print(f"✅ Arc structure analysis included: {len(self.arc_structure_sections)} sections")
        
        # Add training statistics
        enhanced_output['training_statistics'] = self.training_stats
        
        return enhanced_output
    
    def _get_phase_distribution(self, sections: List[Dict]) -> Dict[str, int]:
        """Get distribution of arc phases across sections"""
        phases = {}
        for s in sections:
            phase = s['arc_phase']
            phases[phase] = phases.get(phase, 0) + 1
        return phases
    
    def _map_sections_to_arc_phases(self, sections: List[Dict]):
        """
        Map detected sections to 5-phase arc structure.
        
        Uses tempo trajectory + rhythmic complexity to infer arc phases:
        - Opening: Low tempo, stable, low complexity
        - Development: Rising tempo/complexity
        - Peak: Highest tempo/density
        - Resolution: Decreasing tempo
        - Closing: Return to opening feel
        
        Args:
            sections: List of section dicts from analyze_long_form_improvisation()
        """
        if len(sections) == 0:
            return
        
        # Extract tempo and complexity trajectories
        tempos = np.array([s['local_tempo'] for s in sections])
        complexities = np.array([s['rhythmic_complexity'] for s in sections])
        densities = np.array([s['onset_density'] for s in sections])
        
        # Normalize to 0-1 range
        tempo_range = tempos.max() - tempos.min()
        if tempo_range < 10:  # Minimal tempo variation
            tempo_normalized = np.ones_like(tempos) * 0.5
        else:
            tempo_normalized = (tempos - tempos.min()) / tempo_range
        
        complexity_range = complexities.max() - complexities.min()
        if complexity_range < 1.0:
            complexity_normalized = np.ones_like(complexities) * 0.5
        else:
            complexity_normalized = (complexities - complexities.min()) / complexity_range
        
        # Combined energy metric (tempo + complexity + density)
        energy = (tempo_normalized + complexity_normalized + densities / densities.max()) / 3.0
        
        # Find peak section (highest energy)
        peak_idx = int(np.argmax(energy))
        
        # Assign phases
        for i, section in enumerate(sections):
            progress = i / len(sections)  # 0.0 to 1.0
            
            if progress < 0.15:
                # Opening: First 15%
                section['arc_phase'] = 'opening'
                section['engagement_level'] = 0.3 + 0.2 * (progress / 0.15)
            elif i < peak_idx:
                # Development: Building toward peak
                development_progress = (i - len(sections) * 0.15) / (peak_idx - len(sections) * 0.15)
                section['arc_phase'] = 'development'
                section['engagement_level'] = 0.5 + 0.3 * development_progress
            elif i == peak_idx:
                # Peak: Highest energy section
                section['arc_phase'] = 'peak'
                section['engagement_level'] = 0.9
            elif progress < 0.85:
                # Resolution: After peak, before closing
                resolution_progress = (i - peak_idx) / (len(sections) * 0.85 - peak_idx)
                section['arc_phase'] = 'resolution'
                section['engagement_level'] = 0.8 - 0.3 * resolution_progress
            else:
                # Closing: Final 15%
                closing_progress = (progress - 0.85) / 0.15
                section['arc_phase'] = 'closing'
                section['engagement_level'] = 0.5 - 0.3 * closing_progress
        
        # Print phase mapping
        print("🎭 Arc phase mapping:")
        for section in sections:
            tempo_change_str = f" (→{section['tempo_change']:.2f}x)" if section['tempo_change'] else ""
            print(f"   {section['start_time']:.0f}s: {section['arc_phase']:12s} "
                  f"| {section['local_tempo']:5.1f} BPM{tempo_change_str:12s} "
                  f"| engagement {section['engagement_level']:.2f}")
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for numpy types"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    def _save_json_with_progress(self, data, filepath):
        """Save JSON data with progress bar for large datasets"""
        import time
        
        # Use global tqdm availability
        use_tqdm = TQDM_AVAILABLE
        if not use_tqdm:
            print("📊 Note: tqdm not available, using simple progress reporting")
        
        # Estimate data size
        total_items = self._count_json_items(data)
        print(f"📊 Serializing {total_items:,} data items...")
        
        if use_tqdm:
            # Use a simpler approach - just show progress for file writing
            print("💾 Serializing JSON data...")
            start_time = time.time()
            json_str = json.dumps(data, indent=2, default=self._json_serializer)
            serialize_time = time.time() - start_time
            print(f"✅ JSON serialized in {serialize_time:.1f}s")
        else:
            # Simple progress tracking
            print("💾 Serializing JSON data...")
            start_time = time.time()
            json_str = json.dumps(data, indent=2, default=self._json_serializer)
            serialize_time = time.time() - start_time
            print(f"✅ JSON serialized in {serialize_time:.1f}s")
        
        try:
            # Write to file with progress
            print("💾 Writing to file...")
            file_size = len(json_str)
            chunk_size = max(1024 * 1024, file_size // 100)  # 1MB chunks or 100 chunks
            
            with open(filepath, 'w') as f:
                written = 0
                if use_tqdm:
                    write_pbar = tqdm(total=file_size, desc="📁 File Writing", unit="B", unit_scale=True)
                
                for i in range(0, file_size, chunk_size):
                    chunk = json_str[i:i+chunk_size]
                    f.write(chunk)
                    written += len(chunk)
                    if use_tqdm:
                        write_pbar.update(len(chunk))
                    else:
                        # Simple progress for file writing
                        progress = int((written / file_size) * 100)
                        if progress % 20 == 0:
                            print(f"📁 Writing: {progress}% ({written:,}/{file_size:,} bytes)")
                
                if use_tqdm:
                    write_pbar.close()
            
            elapsed = time.time() - start_time
            print(f"✅ JSON saved successfully: {file_size:,} bytes in {elapsed:.1f}s")
            
        except Exception as e:
            print(f"❌ Error saving JSON: {e}")
            raise
    
    def _count_json_items(self, data):
        """Count total items in nested JSON structure"""
        if isinstance(data, dict):
            return sum(1 + self._count_json_items(v) for v in data.values())
        elif isinstance(data, list):
            return sum(1 + self._count_json_items(item) for item in data)
        else:
            return 1
    
    def _convert_to_correlation_format(self, data, data_type: str) -> List[Dict]:
        """Convert events or analysis results to correlation analysis format"""
        
        if data_type == 'harmonic':
            # Convert events to harmonic format
            harmonic_events = []
            for event in data:
                # Handle both dictionary and object formats
                if hasattr(event, 'timestamp'):
                    # Object format (SampledEvent)
                    harmonic_event = {
                        'timestamp': getattr(event, 'timestamp', 0.0),
                        'chord': getattr(event, 'chord', ''),
                        'key': getattr(event, 'key', ''),
                        'tension': getattr(event, 'harmonic_tension', 0.0),
                        'chord_diversity': getattr(event, 'chord_diversity', 0.0),
                        'key_stability': getattr(event, 'key_stability', 0.0),
                        'chord_change_rate': getattr(event, 'chord_change_rate', 0.0)
                    }
                elif hasattr(event, 'time'):
                    # SampledEvent format - extract harmonic data from features
                    harmonic_event = self._extract_harmonic_from_features(event)
                else:
                    # Dictionary format
                    harmonic_event = {
                        'timestamp': event.get('timestamp', 0.0),
                        'chord': event.get('chord', ''),
                        'key': event.get('key', ''),
                        'tension': event.get('harmonic_tension', 0.0),
                        'chord_diversity': event.get('chord_diversity', 0.0),
                        'key_stability': event.get('key_stability', 0.0),
                        'chord_change_rate': event.get('chord_change_rate', 0.0)
                    }
                harmonic_events.append(harmonic_event)
            return harmonic_events
            
        elif data_type == 'rhythmic':
            # Convert rhythmic analysis to rhythmic format
            if hasattr(data, 'patterns'):
                rhythmic_events = []
                for pattern in data.patterns:
                    rhythmic_event = {
                        'timestamp': pattern.start_time,
                        'tempo': data.tempo,
                        'syncopation': data.syncopation_score,
                        'density': pattern.density if hasattr(pattern, 'density') else 0.5,
                        'rhythmic_complexity': data.rhythmic_complexity,
                        'tempo_stability': 0.8,  # Default stability
                        'rhythmic_density': pattern.density if hasattr(pattern, 'density') else 0.5
                    }
                    rhythmic_events.append(rhythmic_event)
                return rhythmic_events
            else:
                # Fallback for non-pattern data
                return [{
                    'timestamp': 0.0,
                    'tempo': data.tempo if hasattr(data, 'tempo') else 120,
                    'syncopation': data.syncopation_score if hasattr(data, 'syncopation_score') else 0.5,
                    'density': 0.5,
                    'rhythmic_complexity': data.rhythmic_complexity if hasattr(data, 'rhythmic_complexity') else 0.5,
                    'tempo_stability': 0.8,
                    'rhythmic_density': 0.5
                }]
        
        return []
    
    def _extract_harmonic_from_features(self, sampled_event) -> Dict:
        """Extract harmonic information from SampledEvent features array"""
        
        features = sampled_event.features
        
        # The features array contains chroma features (first 12 dimensions)
        # and other audio features (spectral, temporal, etc.)
        
        # Extract chroma features (first 12 dimensions)
        chroma_features = features[:12] if len(features) >= 12 else features
        
        # Find the strongest chroma bin (most prominent pitch class)
        strongest_pitch_class = np.argmax(chroma_features)
        pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        primary_pitch = pitch_classes[strongest_pitch_class]
        
        # Calculate harmonic tension (variance in chroma features)
        harmonic_tension = float(np.var(chroma_features))
        
        # Calculate chord diversity (number of significant pitch classes)
        threshold = np.mean(chroma_features) + np.std(chroma_features)
        significant_pitches = np.sum(chroma_features > threshold)
        chord_diversity = float(significant_pitches / 12.0)
        
        # Estimate key stability (consistency of chroma pattern)
        key_stability = float(1.0 - np.std(chroma_features))
        
        # Estimate chord change rate (based on feature variation)
        chord_change_rate = float(min(harmonic_tension, 1.0))
        
        return {
            'timestamp': sampled_event.time,
            'chord': primary_pitch,  # Use primary pitch as chord
            'key': primary_pitch,    # Use primary pitch as key
            'tension': harmonic_tension,
            'chord_diversity': chord_diversity,
            'key_stability': key_stability,
            'chord_change_rate': chord_change_rate
        }
    
    def _detect_chords_for_events(self, events: List[Dict], audio_file: str) -> List[Dict]:
        """
        Detect accurate chords for each event using RealtimeHarmonicDetector
        This replaces chroma-based chord detection with proper harmonic analysis
        
        Args:
            events: List of event dictionaries
            audio_file: Path to audio file
            
        Returns:
            Enhanced events with accurate chord data
        """
        print("🎼 Detecting chords using RealtimeHarmonicDetector...")
        
        try:
            import librosa
            
            # Load audio file once
            y, sr = librosa.load(audio_file, sr=44100, mono=True)
            
            # Initialize chord sequence tracking
            chord_sequence = []
            
            # Process each event
            enhanced_events = []
            for i, event in enumerate(events):
                event_time = event.get('t', 0.0)
                
                # Extract audio segment around this event
                # Use 1-second window centered on onset for robust detection
                window_duration = 1.0  # seconds
                start_time = max(0, event_time - window_duration / 2)
                end_time = min(len(y) / sr, event_time + window_duration / 2)
                
                start_sample = int(start_time * sr)
                end_sample = int(end_time * sr)
                
                # Extract audio buffer
                audio_buffer = y[start_sample:end_sample]
                
                # Detect chord using RealtimeHarmonicDetector
                if len(audio_buffer) > 0:
                    harmonic_context = self.harmonic_detector.update_from_audio(audio_buffer, sr=sr)
                    
                    # Store chord data in event
                    event_enhanced = event.copy()
                    event_enhanced['realtime_chord'] = harmonic_context.current_chord
                    event_enhanced['realtime_key'] = harmonic_context.key_signature
                    event_enhanced['chord_confidence'] = harmonic_context.confidence
                    event_enhanced['chord_stability'] = harmonic_context.stability
                    event_enhanced['chord_root'] = harmonic_context.chord_root
                    event_enhanced['chord_type'] = harmonic_context.chord_type
                    event_enhanced['scale_degrees'] = harmonic_context.scale_degrees
                    
                    # Track chord sequence for transition analysis
                    chord_sequence.append({
                        'time': event_time,
                        'chord': harmonic_context.current_chord,
                        'key': harmonic_context.key_signature,
                        'confidence': harmonic_context.confidence
                    })
                    
                    enhanced_events.append(event_enhanced)
                else:
                    # Fallback if audio buffer is empty
                    enhanced_events.append(event)
                
                # Progress indicator
                if (i + 1) % 100 == 0 or i == len(events) - 1:
                    progress = (i + 1) / len(events) * 100
                    print(f"   Progress: {progress:.1f}% ({i+1}/{len(events)} events)", end='\r')
            
            print()  # New line after progress
            
            # Analyze chord distribution
            from collections import Counter
            chord_counts = Counter([cs['chord'] for cs in chord_sequence])
            unique_chords = len(chord_counts)
            
            print(f"✅ Chord detection complete:")
            print(f"   Total events: {len(enhanced_events)}")
            print(f"   Unique chords: {unique_chords}")
            if unique_chords > 0:
                print(f"   Top 5 chords: {dict(chord_counts.most_common(5))}")
            
            # Store chord sequence for transition graph building
            self.chord_sequence = chord_sequence
            
            return enhanced_events
            
        except Exception as e:
            print(f"❌ Chord detection failed: {e}")
            print(f"   Falling back to events without realtime chord data")
            return events
    
    def _build_harmonic_transition_graph(self) -> Dict[str, Any]:
        """
        Build harmonic transition graph from detected chord sequence
        Learns probabilities of chord transitions from training data
        
        Returns:
            Dictionary containing transition probabilities and chord statistics
        """
        print("\n🎼 Building harmonic transition graph...")
        
        if not hasattr(self, 'chord_sequence') or len(self.chord_sequence) < 2:
            print("   ⚠️  No chord sequence available - skipping transition graph")
            return {}
        
        from collections import defaultdict, Counter
        
        # Count transitions
        transitions = defaultdict(int)
        chord_durations = defaultdict(list)
        chord_frequencies = Counter()
        
        for i in range(len(self.chord_sequence) - 1):
            current = self.chord_sequence[i]
            next_chord = self.chord_sequence[i + 1]
            
            current_chord_name = current['chord']
            next_chord_name = next_chord['chord']
            
            # Track transition
            transition_key = (current_chord_name, next_chord_name)
            transitions[transition_key] += 1
            
            # Track chord frequency
            chord_frequencies[current_chord_name] += 1
            
            # Track duration (time between chord changes)
            duration = next_chord['time'] - current['time']
            chord_durations[current_chord_name].append(duration)
        
        # Add last chord frequency
        if self.chord_sequence:
            last_chord = self.chord_sequence[-1]['chord']
            chord_frequencies[last_chord] += 1
        
        # Normalize transitions to probabilities
        transition_probabilities = {}
        for (from_chord, to_chord), count in transitions.items():
            from_chord_count = chord_frequencies[from_chord]
            if from_chord_count > 0:
                probability = count / from_chord_count
                transition_probabilities[f"{from_chord}->{to_chord}"] = {
                    'probability': float(probability),
                    'count': int(count),
                    'from_chord_occurrences': int(from_chord_count)
                }
        
        # Calculate average durations
        avg_durations = {}
        for chord, durations in chord_durations.items():
            if durations:
                avg_durations[chord] = {
                    'average': float(np.mean(durations)),
                    'std': float(np.std(durations)),
                    'min': float(np.min(durations)),
                    'max': float(np.max(durations))
                }
        
        # Build graph structure
        transition_graph = {
            'transitions': transition_probabilities,
            'chord_frequencies': {k: int(v) for k, v in chord_frequencies.items()},
            'chord_durations': avg_durations,
            'total_chords': len(self.chord_sequence),
            'unique_chords': len(chord_frequencies),
            'total_transitions': sum(transitions.values())
        }
        
        # Print summary
        print(f"✅ Transition graph built:")
        print(f"   Total chord events: {transition_graph['total_chords']}")
        print(f"   Unique chords: {transition_graph['unique_chords']}")
        print(f"   Total transitions: {transition_graph['total_transitions']}")
        print(f"   Top 5 chords: {dict(chord_frequencies.most_common(5))}")
        
        # Print top transitions
        if transition_probabilities:
            sorted_transitions = sorted(
                transition_probabilities.items(),
                key=lambda x: x[1]['count'],
                reverse=True
            )[:5]
            print(f"   Top 5 transitions:")
            for trans, data in sorted_transitions:
                print(f"      {trans}: {data['probability']:.2%} ({data['count']} times)")
        
        return transition_graph
    
    def _print_training_summary(self):
        """Print training summary"""
        
        print("\n" + "="*60)
        print("🎵 Enhanced Hybrid Training Summary")
        print("="*60)
        
        print(f"📊 Total Events: {self.training_stats['total_events']}")
        print(f"⏱️  Total Time: {self.training_stats['total_training_time']:.2f}s")
        
        if self.training_stats['hierarchical_insights_applied']:
            print(f"🏗️  Hierarchical Analysis: {self.training_stats['hierarchical_analysis_time']:.2f}s")
        
        if self.training_stats['rhythmic_insights_applied']:
            print(f"🥁 Rhythmic Analysis: {self.training_stats['rhythmic_analysis_time']:.2f}s")
        
        if self.training_stats.get('correlation_insights_applied', False):
            print(f"🔗 Correlation Analysis: {self.training_stats['correlation_analysis_time']:.2f}s")
        
        if self.training_stats['enhancement_applied']:
            print(f"🧠 Transformer Analysis: {self.training_stats['transformer_analysis_time']:.2f}s")
        
        if self.training_stats['gpt_oss_insights_applied']:
            print(f"🧠 GPT-OSS Analysis: {self.training_stats['gpt_oss_analysis_time']:.2f}s")
        
        if self.training_stats['performance_arc_insights_applied']:
            print(f"🎭 Performance Arc Analysis: {self.training_stats['performance_arc_analysis_time']:.2f}s")
        
        print(f"🎯 AudioOracle Training: {self.training_stats['audio_oracle_training_time']:.2f}s")
        
        # Print harmonic and polyphonic pattern results
        if 'training_result' in self.training_stats and self.training_stats['training_result'].get('training_successful', False):
            harmonic_count = self.training_stats['training_result'].get('harmonic_patterns', 0)
            polyphonic_count = self.training_stats['training_result'].get('polyphonic_patterns', 0)
            print(f"🎼 Harmonic Patterns: {harmonic_count}")
            print(f"🎵 Polyphonic Patterns: {polyphonic_count}")
        
        if self.training_stats['rhythmic_insights_applied']:
            print(f"🥁 RhythmOracle Training: {self.training_stats['rhythm_oracle_training_time']:.2f}s")
        
        print("="*60)

def main():
    """Test the enhanced hybrid training pipeline"""
    
    import argparse
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description='Enhanced Hybrid Training Pipeline with Rhythmic Analysis')
    parser.add_argument('--file', required=True, help='Audio file to process')
    parser.add_argument('--output', help='Output JSON file (optional - auto-generated if not provided)')
    parser.add_argument('--max-events', type=int, default=10000, help='Maximum events to extract')
    parser.add_argument('--training-events', type=int, help='Number of events for training')
    parser.add_argument('--sampling-strategy', choices=['balanced', 'structural', 'perceptual'], 
                       default='balanced', help='Sampling strategy')
    parser.add_argument('--no-transformer', action='store_true', help='Disable transformer analysis')
    parser.add_argument('--no-hierarchical', action='store_true', help='Disable hierarchical analysis')
    parser.add_argument('--no-rhythmic', action='store_true', help='Disable rhythmic analysis')
    parser.add_argument('--no-gpt-oss', action='store_true', help='Disable GPT-OSS analysis')
    parser.add_argument('--no-hybrid-perception', action='store_true', 
                       help='Disable hybrid perception (ratio + symbolic features)')
    parser.add_argument('--vocab-size', type=int, default=64, 
                       help='Symbolic vocabulary size (16, 64, or 256)')
    parser.add_argument('--no-wav2vec', action='store_true',
                       help='Disable Wav2Vec 2.0 neural encoding')
    parser.add_argument('--wav2vec-model', type=str, default='facebook/wav2vec2-base',
                       help='Wav2Vec model name (default: facebook/wav2vec2-base)')
    parser.add_argument('--no-gpu', action='store_true',
                       help='Disable GPU for Wav2Vec (force CPU)')
    parser.add_argument('--no-dual-vocabulary', action='store_true',
                       help='Disable dual vocabulary mode (separate harmonic/percussive tokens for drums)')
    parser.add_argument('--analyze-arc-structure', action='store_true',
                       help='Analyze long-form arc structure (sections, tempo changes, phase mapping)')
    parser.add_argument('--section-duration', type=float, default=60.0,
                       help='Section duration for arc analysis in seconds (default: 60)')
    
    # Temporal smoothing parameters (LEGACY - being replaced by musical gesture processing)
    parser.add_argument('--temporal-window', type=float, default=0.1,
                       help='[LEGACY] Temporal smoothing window in seconds (use --gesture-* instead)')
    parser.add_argument('--temporal-threshold', type=float, default=0.2,
                       help='[LEGACY] Minimum feature change threshold (use --gesture-* instead)')
    parser.add_argument('--no-temporal-smoothing', action='store_true', default=True,
                       help='Disable temporal smoothing (default: True - musical gestures enabled by default)')
    
    # Musical gesture processing parameters (NEW APPROACH)
    parser.add_argument('--use-musical-gestures', action='store_true', default=True,
                       help='Use musical gesture processor instead of temporal smoothing (default: True)')
    parser.add_argument('--no-musical-gestures', dest='use_musical_gestures', action='store_false',
                       help='Disable musical gesture processing (use with --no-temporal-smoothing for raw data)')
    parser.add_argument('--gesture-transition-threshold', type=float, default=0.5,
                       help='Feature change indicating new musical gesture (0.2-0.6, default: 0.5)')
    parser.add_argument('--gesture-sustain-threshold', type=float, default=0.3,
                       help='Feature similarity indicating sustained gesture (0.1-0.4, default: 0.3)')
    parser.add_argument('--gesture-min-duration', type=float, default=0.3,
                       help='Minimum gesture duration in seconds (default: 0.3)')
    parser.add_argument('--gesture-max-duration', type=float, default=3.0,
                       help='Maximum gesture duration in seconds (default: 3.0)')
    parser.add_argument('--gesture-consolidation', type=str, default='weighted_median',
                       choices=['peak', 'first', 'mean', 'weighted_median', 'stable'],
                       help='How to consolidate events within a gesture (default: weighted_median - best diversity + smoothness)')
    
    args = parser.parse_args()
    
    # Auto-generate output filename if not provided
    if args.output:
        output_file = args.output
    else:
        # Generate filename from audio file name and date
        # Format: (audiofile-name)_(date)_training.json
        from datetime import datetime
        audio_basename = os.path.splitext(os.path.basename(args.file))[0]
        date_str = datetime.now().strftime("%d%m%y_%H%M")
        output_file = f"JSON/{audio_basename}_{date_str}_training.json"
        print(f"📁 Auto-generated output filenames:")
        print(f"   Training summary: {output_file}")
        print(f"   Oracle model: JSON/{audio_basename}_{date_str}_model.json")
        print(f"   Gesture quantizer: JSON/{audio_basename}_{date_str}_gesture_training_quantizer.joblib")
        print(f"   Symbolic quantizer: JSON/{audio_basename}_{date_str}_symbolic_training_quantizer.joblib (if enabled)")
        print(f"   Correlations: JSON/{audio_basename}_{date_str}_correlation_patterns.json")
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 Output directory: {output_dir}/")
    
    # Initialize pipeline
    pipeline = EnhancedHybridTrainingPipeline(
        max_events=args.max_events,
        enable_hierarchical=not args.no_hierarchical,
        enable_rhythmic=not args.no_rhythmic,
        enable_gpt_oss=not args.no_gpt_oss,
        enable_hybrid_perception=not args.no_hybrid_perception,
        symbolic_vocabulary_size=args.vocab_size,
        enable_wav2vec=not args.no_wav2vec,
        wav2vec_model=args.wav2vec_model,
        use_gpu=not args.no_gpu,
        enable_dual_vocabulary=not args.no_dual_vocabulary,
        temporal_window=args.temporal_window,
        temporal_threshold=args.temporal_threshold,
        enable_temporal_smoothing=not args.no_temporal_smoothing,
        use_musical_gestures=args.use_musical_gestures,
        gesture_transition_threshold=args.gesture_transition_threshold,
        gesture_sustain_threshold=args.gesture_sustain_threshold,
        gesture_min_duration=args.gesture_min_duration,
        gesture_max_duration=args.gesture_max_duration,
        gesture_consolidation_method=args.gesture_consolidation
    )
    
    # Train from audio file
    result = pipeline.train_from_audio_file(
        audio_file=args.file,
        output_file=output_file,
        max_events=args.max_events,
        training_events=args.training_events,
        sampling_strategy=args.sampling_strategy,
        use_transformer=not args.no_transformer,
        use_hierarchical=not args.no_hierarchical,
        use_rhythmic=not args.no_rhythmic,
        analyze_arc=args.analyze_arc_structure,
        section_duration=args.section_duration
    )
    
    print(f"\n✅ Enhanced hybrid training complete!")
    print(f"📁 Results saved to: {output_file}")
    
    # Clean up resources
    pipeline.cleanup()

if __name__ == "__main__":
    main()
