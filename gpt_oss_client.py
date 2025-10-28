#!/usr/bin/env python3
"""
GPT-OSS Client for Musical Analysis
Integrates with Ollama to provide intelligent musical analysis
"""

import requests
import json
import time
import subprocess
import os
import signal
import psutil
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

@dataclass
class GPTOSSAnalysis:
    """GPT-OSS analysis results"""
    harmonic_analysis: str
    rhythmic_analysis: str
    phrasing_analysis: str
    feel_analysis: str
    style_analysis: str
    confidence_score: float
    processing_time: float

@dataclass
class GPTOSSArcAnalysis:
    """GPT-OSS performance arc analysis results"""
    structural_analysis: str
    dynamic_evolution: str
    emotional_arc: str
    role_development: str
    silence_strategy: str
    engagement_curve: str
    confidence_score: float
    processing_time: float

class GPTOSSClient:
    """
    GPT-OSS client for musical analysis
    Communicates with local Ollama instance running gpt-oss:20b
    Automatically starts Ollama if not running
    """
    
    def __init__(self, base_url: str = "http://localhost:11434/api/generate", 
                 model: str = "gpt-oss:20b", timeout: int = 120, 
                 auto_start: bool = True):
        """
        Initialize GPT-OSS client
        
        Args:
            base_url: Ollama API endpoint
            model: Model name (gpt-oss:20b or gpt-oss:120b)
            timeout: Request timeout in seconds
            auto_start: Whether to automatically start Ollama if not running
        """
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.auto_start = auto_start
        self.ollama_process = None
        
        # Check availability and start Ollama if needed
        self.is_available = self._ensure_ollama_running()
        
        if self.is_available:
            print(f"✅ GPT-OSS client initialized with model: {model}")
        else:
            print(f"⚠️ GPT-OSS client initialized but Ollama not available")
    
    def _check_availability(self) -> bool:
        """Check if Ollama is running and model is available"""
        try:
            response = requests.post(self.base_url, 
                                  json={'model': self.model, 'prompt': 'test', 'stream': False},
                                  timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def _is_ollama_running(self) -> bool:
        """Check if Ollama server is running"""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def _start_ollama_server(self) -> bool:
        """Start Ollama server"""
        try:
            print("🚀 Starting Ollama server...")
            
            # Check if ollama command exists
            result = subprocess.run(['which', 'ollama'], capture_output=True, text=True)
            if result.returncode != 0:
                print("❌ Ollama not found in PATH. Please install Ollama first.")
                return False
            
            # Start Ollama server in background
            self.ollama_process = subprocess.Popen(
                ['ollama', 'serve'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                preexec_fn=os.setsid if os.name != 'nt' else None
            )
            
            # Wait for server to start
            print("⏳ Waiting for Ollama server to start...")
            for i in range(60):  # Wait up to 60 seconds
                if self._is_ollama_running():
                    print("✅ Ollama server started successfully!")
                    time.sleep(2)  # Give it a moment to fully initialize
                    return True
                time.sleep(1)
            
            print("❌ Ollama server failed to start within 60 seconds")
            return False
            
        except Exception as e:
            print(f"❌ Failed to start Ollama server: {e}")
            return False
    
    def _ensure_model_available(self) -> bool:
        """Ensure the GPT-OSS model is available"""
        try:
            # Check if model is available
            response = requests.post(self.base_url, 
                                   json={'model': self.model, 'prompt': 'test', 'stream': False},
                                   timeout=15)
            
            if response.status_code == 200:
                return True
            
            # If model not found, try to pull it
            print(f"📥 Model {self.model} not found, attempting to pull...")
            result = subprocess.run(['ollama', 'pull', self.model], 
                                  capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print(f"✅ Model {self.model} pulled successfully!")
                return True
            else:
                print(f"❌ Failed to pull model {self.model}: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Error checking model availability: {e}")
            return False
    
    def _ensure_ollama_running(self) -> bool:
        """Ensure Ollama is running and model is available"""
        if not self.auto_start:
            return self._check_availability()
        
        # Check if Ollama server is running
        if not self._is_ollama_running():
            if not self._start_ollama_server():
                return False
        
        # Check if model is available
        if not self._ensure_model_available():
            return False
        
        return True
    
    def cleanup(self):
        """Clean up resources"""
        if self.ollama_process:
            try:
                print("🛑 Stopping Ollama server...")
                if os.name != 'nt':
                    os.killpg(os.getpgid(self.ollama_process.pid), signal.SIGTERM)
                else:
                    self.ollama_process.terminate()
                self.ollama_process.wait(timeout=5)
                print("✅ Ollama server stopped")
            except Exception as e:
                print(f"⚠️ Error stopping Ollama server: {e}")
            finally:
                self.ollama_process = None
    
    def analyze_musical_events(self, events: List[Dict], 
                             harmonic_patterns: List = None,
                             rhythmic_patterns: List = None,
                             correlation_analysis: Dict = None) -> Optional[GPTOSSAnalysis]:
        """
        Analyze musical events with GPT-OSS
        
        Args:
            events: List of musical events
            harmonic_patterns: Detected harmonic patterns
            rhythmic_patterns: Detected rhythmic patterns
            correlation_analysis: Harmonic-rhythmic correlation results
            
        Returns:
            GPTOSSAnalysis object or None if analysis fails
        """
        if not self.is_available:
            print("⚠️ GPT-OSS not available, skipping analysis")
            return None
        
        start_time = time.time()
        
        try:
            # Create comprehensive prompt
            prompt = self._create_analysis_prompt(events, harmonic_patterns, 
                                               rhythmic_patterns, correlation_analysis)
            
            # Send request to Ollama
            response = requests.post(self.base_url, 
                                   json={
                                       'model': self.model,
                                       'prompt': prompt,
                                       'stream': False
                                   },
                                   timeout=self.timeout)
            
            if response.status_code == 200:
                response_text = response.json()['response']
                processing_time = time.time() - start_time
                
                # Parse response into structured analysis
                analysis = self._parse_response(response_text, processing_time)
                
                print(f"✅ GPT-OSS analysis complete in {processing_time:.2f}s")
                return analysis
            else:
                print(f"❌ GPT-OSS request failed: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ GPT-OSS analysis failed: {e}")
            return None
    
    def analyze_musical_events_for_training(self, events: List[Dict], 
                                          transformer_insights = None,
                                          hierarchical_result = None,
                                          rhythmic_result = None,
                                          correlation_result = None) -> Optional[GPTOSSAnalysis]:
        """
        Analyze musical events for training enhancement (pre-training analysis)
        
        Args:
            events: List of musical events
            transformer_insights: Music theory transformer insights
            hierarchical_result: Hierarchical analysis results
            rhythmic_result: Rhythmic analysis results
            correlation_result: Correlation analysis results
            
        Returns:
            GPTOSSAnalysis object or None if analysis fails
        """
        if not self.is_available:
            print("⚠️ GPT-OSS not available for training analysis")
            return None
        
        start_time = time.time()
        
        try:
            # Create training-focused prompt
            prompt = self._create_training_prompt(events, transformer_insights, 
                                                hierarchical_result, rhythmic_result, correlation_result)
            
            # Send request to Ollama
            response = requests.post(self.base_url, 
                                   json={
                                       'model': self.model,
                                       'prompt': prompt,
                                       'stream': False
                                   },
                                   timeout=self.timeout)
            
            if response.status_code == 200:
                response_text = response.json()['response']
                processing_time = time.time() - start_time
                
                # Parse response into structured analysis
                analysis = self._parse_response(response_text, processing_time)
                
                print(f"✅ GPT-OSS training analysis complete in {processing_time:.2f}s")
                return analysis
            else:
                print(f"❌ GPT-OSS training analysis failed: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ GPT-OSS training analysis failed: {e}")
            return None
    
    def _create_training_prompt(self, events: List[Dict], 
                               transformer_insights = None,
                               hierarchical_result = None,
                               rhythmic_result = None,
                               correlation_result = None) -> str:
        """Create training-focused analysis prompt"""
        
        total_events = len(events)
        
        # Extract key musical information
        chords_found = set()
        tempos_found = set()
        
        # Sample events for analysis
        sample_events = events[:20] if events else []
        
        for event in sample_events:
            if isinstance(event, dict):
                # Look for chord information
                chord = (event.get('correlation_insights', {}).get('chord') or
                        event.get('transformer_insights', {}).get('chord_progression') or
                        event.get('chord', ''))
                if chord:
                    chords_found.add(str(chord))
                
                # Look for tempo information
                tempo = (event.get('rhythmic_context', {}).get('global_tempo') or
                        event.get('tempo', 0))
                if tempo and tempo > 0:
                    tempos_found.add(float(tempo))
        
        # Build training-focused prompt
        prompt = f"""
Analyze this musical dataset for AI training enhancement:

Dataset Overview:
- Total events: {total_events}
- Chord types: {list(chords_found) if chords_found else 'None detected'}
- Tempo range: {list(tempos_found) if tempos_found else 'Not specified'}

Training Context:
- This data will be used to train an AI music system
- The AI needs to understand harmonic relationships, phrasing, and musical flow
- Focus on patterns that would help the AI make musically intelligent decisions

Provide analysis focusing on:
1. Harmonic sophistication and chord relationships
2. Rhythmic patterns and phrasing structure  
3. Musical coherence and flow
4. Training potential for AI musical intelligence
5. Key patterns the AI should learn

Keep analysis concise but insightful for training enhancement.
"""
        
        return prompt
    
    def _create_analysis_prompt(self, events: List[Dict], 
                              harmonic_patterns: List = None,
                              rhythmic_patterns: List = None,
                              correlation_analysis: Dict = None) -> str:
        """Create comprehensive analysis prompt"""
        
        # Extract key information from events
        total_events = len(events)
        event_sample = events[:10] if events else []
        
        # Extract chord information from events
        chords_found = set()
        tempos_found = set()
        
        for event in event_sample:
            if isinstance(event, dict):
                # Look for chord information in various fields
                chord = (event.get('correlation_insights', {}).get('chord') or
                        event.get('transformer_insights', {}).get('chord_progression') or
                        event.get('chord', ''))
                if chord:
                    chords_found.add(str(chord))
                
                tempo = (event.get('rhythmic_context', {}).get('global_tempo') or
                        event.get('tempo', 0))
                if tempo and tempo > 0:
                    tempos_found.add(float(tempo))
        
        # Extract pattern information with smart truncation for large datasets
        harmonic_info = ""
        if harmonic_patterns:
            # Smart truncation: show more patterns for larger datasets, but cap at 50
            max_patterns = min(50, len(harmonic_patterns))
            harmonic_info = f"\nHarmonic Patterns Detected: {len(harmonic_patterns)} patterns (showing top {max_patterns})\n"
            for i, pattern in enumerate(harmonic_patterns[:max_patterns]):
                if isinstance(pattern, list) and len(pattern) >= 2:
                    chords = pattern[0] if isinstance(pattern[0], list) else [pattern[0]]
                    frequency = pattern[1]
                    harmonic_info += f"  Pattern {i+1}: {chords} (frequency: {frequency})\n"
        
        rhythmic_info = ""
        if rhythmic_patterns:
            rhythmic_info = f"\nRhythmic Patterns Detected: {len(rhythmic_patterns)} patterns\n"
        
        correlation_info = ""
        if correlation_analysis:
            stats = correlation_analysis.get('analysis_stats', {})
            correlation_info = f"""
Correlation Analysis:
- Joint events: {stats.get('total_joint_events', 0)}
- Patterns discovered: {stats.get('patterns_discovered', 0)}
- Average correlation strength: {stats.get('correlation_strength_avg', 0):.3f}
"""
        
        # Create concise prompt for large datasets
        if total_events > 1000:
            prompt = f"""
Analyze this large musical dataset ({total_events} events):

Key Stats:
- Chord types: {len(chords_found)} ({list(chords_found)[:10] if chords_found else 'None'})
- Tempo: {list(tempos_found) if tempos_found else 'Not specified'}
- Harmonic patterns: {len(harmonic_patterns) if harmonic_patterns else 0}

{harmonic_info}

Focus on:
1. Overall musical character and style
2. Harmonic sophistication level
3. Phrasing and feel patterns
4. Creative potential for AI responses

Keep analysis concise but insightful.
"""
        else:
            prompt = f"""
Analyze this musical dataset focusing on phrasing and feel:

Events: {total_events}
Chords: {list(chords_found) if chords_found else 'None'}
Tempo: {list(tempos_found) if tempos_found else 'Not specified'}

{harmonic_info}

Provide brief analysis of:
1. Phrasing patterns
2. Musical feel and groove
3. Harmonic sophistication
4. Overall style

Keep response concise.
"""
        
        return prompt
    
    def _parse_response(self, response_text: str, processing_time: float) -> GPTOSSAnalysis:
        """Parse GPT-OSS response into structured analysis"""
        
        # Simple parsing - could be enhanced with more sophisticated parsing
        # For now, we'll use the full response for all analysis types
        
        return GPTOSSAnalysis(
            harmonic_analysis=response_text,
            rhythmic_analysis=response_text,
            phrasing_analysis=response_text,
            feel_analysis=response_text,
            style_analysis=response_text,
            confidence_score=0.8,  # Default confidence
            processing_time=processing_time
        )
    
    def analyze_simple_patterns(self, patterns: List, pattern_type: str = "harmonic") -> Optional[str]:
        """
        Quick analysis of specific patterns
        
        Args:
            patterns: List of patterns to analyze
            pattern_type: Type of patterns ("harmonic", "rhythmic", "polyphonic")
            
        Returns:
            Analysis text or None if failed
        """
        if not self.is_available:
            return None
        
        try:
            # Create simple pattern analysis prompt
            patterns_text = "\n".join([f"Pattern {i+1}: {pattern}" for i, pattern in enumerate(patterns[:20])])
            
            prompt = f"""
Analyze these {pattern_type} patterns from a musical AI training dataset:

{patterns_text}

Provide insights on:
1. Pattern diversity and sophistication
2. Musical characteristics and style
3. Complexity level
4. Potential for creative responses
"""
            
            response = requests.post(self.base_url, 
                                   json={
                                       'model': self.model,
                                       'prompt': prompt,
                                       'stream': False
                                   },
                                   timeout=self.timeout)
            
            if response.status_code == 200:
                return response.json()['response']
            else:
                return None
                
        except Exception as e:
            print(f"❌ GPT-OSS pattern analysis failed: {e}")
            return None
    
    def analyze_performance_arc(self, performance_arc) -> Optional[GPTOSSArcAnalysis]:
        """
        Analyze performance arc for musical structure and evolution
        
        Args:
            performance_arc: PerformanceArc object from performance_arc_analyzer
            
        Returns:
            GPTOSSArcAnalysis object or None if analysis fails
        """
        if not self.is_available:
            print("⚠️ GPT-OSS not available for performance arc analysis")
            return None
        
        start_time = time.time()
        
        try:
            # Create performance arc analysis prompt
            prompt = self._create_arc_analysis_prompt(performance_arc)
            
            # Send request to Ollama
            response = requests.post(self.base_url, 
                                   json={
                                       'model': self.model,
                                       'prompt': prompt,
                                       'stream': False
                                   },
                                   timeout=self.timeout)
            
            if response.status_code == 200:
                response_text = response.json()['response']
                processing_time = time.time() - start_time
                
                # Parse response into structured arc analysis
                analysis = self._parse_arc_response(response_text, processing_time)
                
                print(f"✅ GPT-OSS performance arc analysis complete in {processing_time:.2f}s")
                return analysis
            else:
                print(f"❌ GPT-OSS arc analysis failed: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ GPT-OSS arc analysis failed: {e}")
            return None
    
    def _create_arc_analysis_prompt(self, performance_arc) -> str:
        """Create performance arc analysis prompt"""
        
        # Extract key information from performance arc
        total_duration = performance_arc.total_duration
        num_phases = len(performance_arc.phases)
        
        # Build phase information
        phase_info = ""
        for i, phase in enumerate(performance_arc.phases):
            phase_info += f"  Phase {i+1}: {phase.phase_type} ({phase.start_time:.1f}s - {phase.end_time:.1f}s)\n"
            phase_info += f"    Engagement: {phase.engagement_level:.2f}, Density: {phase.musical_density:.2f}\n"
            phase_info += f"    Dynamic: {phase.dynamic_level:.2f}, Silence: {phase.silence_ratio:.2f}\n"
            phase_info += f"    Roles: {phase.instrument_roles}\n"
        
        # Extract engagement curve characteristics
        engagement_curve = performance_arc.overall_engagement_curve
        if engagement_curve:
            min_engagement = min(engagement_curve)
            max_engagement = max(engagement_curve)
            avg_engagement = sum(engagement_curve) / len(engagement_curve)
        else:
            min_engagement = max_engagement = avg_engagement = 0.5
        
        # Extract silence patterns
        silence_info = ""
        if performance_arc.silence_patterns:
            silence_info = f"Silence Patterns: {len(performance_arc.silence_patterns)} periods\n"
            for i, (start, duration) in enumerate(performance_arc.silence_patterns[:5]):
                silence_info += f"  Silence {i+1}: {start:.1f}s for {duration:.1f}s\n"
        
        # Build comprehensive prompt
        prompt = f"""
Analyze this musical performance arc for AI musical intelligence:

Performance Overview:
- Total Duration: {total_duration:.1f} seconds
- Number of Phases: {num_phases}
- Engagement Range: {min_engagement:.2f} - {max_engagement:.2f} (avg: {avg_engagement:.2f})

Phase Structure:
{phase_info}

{silence_info}

Performance Arc Characteristics:
- Engagement Curve: {len(engagement_curve)} data points
- Instrument Evolution: {len(performance_arc.instrument_evolution)} instruments tracked
- Theme Development: {len(performance_arc.theme_development)} thematic segments
- Dynamic Evolution: {len(performance_arc.dynamic_evolution)} dynamic changes

Provide comprehensive analysis focusing on:

1. STRUCTURAL ANALYSIS: How does the musical structure evolve? What are the key structural elements and transitions?

2. DYNAMIC EVOLUTION: How do dynamics, density, and intensity change throughout the performance? What creates the musical momentum?

3. EMOTIONAL ARC: What emotional journey does this performance take? How do engagement levels reflect emotional content?

4. ROLE DEVELOPMENT: How do instrument roles change throughout the performance? What creates the musical dialogue?

5. SILENCE STRATEGY: How are silences used strategically? What role do they play in the overall arc?

6. ENGAGEMENT CURVE: What patterns emerge in the engagement curve? How does it guide the musical narrative?

Focus on insights that would help an AI musical partner understand:
- When to be silent vs. active
- How to build musical tension and release
- How to evolve roles throughout a performance
- How to create engaging musical arcs
- How to use strategic silence effectively

Provide detailed, actionable insights for AI musical intelligence.
"""
        
        return prompt
    
    def _parse_arc_response(self, response_text: str, processing_time: float) -> GPTOSSArcAnalysis:
        """Parse GPT-OSS arc analysis response into structured analysis"""
        
        # For now, use the full response for all analysis types
        # In a more sophisticated implementation, we could parse specific sections
        
        return GPTOSSArcAnalysis(
            structural_analysis=response_text,
            dynamic_evolution=response_text,
            emotional_arc=response_text,
            role_development=response_text,
            silence_strategy=response_text,
            engagement_curve=response_text,
            confidence_score=0.8,  # Default confidence
            processing_time=processing_time
        )

def test_gpt_oss_client():
    """Test GPT-OSS client functionality"""
    print("🧠 Testing GPT-OSS Client with Auto-Start")
    print("=" * 50)
    
    client = GPTOSSClient(auto_start=True)
    
    if not client.is_available:
        print("❌ GPT-OSS not available even with auto-start")
        return False
    
    # Test with sample data
    sample_events = [
        {'t': 0.0, 'chord': 'C', 'tempo': 120},
        {'t': 1.0, 'chord': 'Am', 'tempo': 120},
        {'t': 2.0, 'chord': 'F', 'tempo': 120},
        {'t': 3.0, 'chord': 'G', 'tempo': 120}
    ]
    
    sample_patterns = [
        [['C', 'Am'], 5],
        [['F', 'G'], 4],
        [['Am', 'F'], 3]
    ]
    
    print("📊 Testing musical analysis...")
    analysis = client.analyze_musical_events(sample_events, sample_patterns)
    
    if analysis:
        print("✅ GPT-OSS analysis successful!")
        print(f"   Processing time: {analysis.processing_time:.2f}s")
        print(f"   Confidence: {analysis.confidence_score:.2f}")
        print(f"   Analysis preview: {analysis.harmonic_analysis[:200]}...")
        
        # Clean up
        client.cleanup()
        return True
    else:
        print("❌ GPT-OSS analysis failed")
        client.cleanup()
        return False

if __name__ == "__main__":
    test_gpt_oss_client()
