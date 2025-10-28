"""
Autonomous Behavior Configuration
Parses GPT-OSS insights and configures agent behavior dynamically
"""

import re
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class AutonomousBehaviorConfig:
    """Configuration for autonomous AI behavior"""
    # Silence behavior
    silence_timeout: float = 1.5  # Seconds before switching to autonomous mode
    autonomous_interval: float = 3.0  # Base interval for autonomous generation
    
    # Voice-specific behavior
    bass_accompaniment_probability: float = 0.5  # Probability bass plays during human activity
    melody_silence_when_active: bool = True  # Melody stays quiet when human is active
    
    # Activity thresholds
    activity_threshold_high: float = 0.7  # High activity = give lots of space
    activity_threshold_low: float = 0.3  # Low activity = moderate density
    
    # Density modulation
    density_when_active: float = 0.1  # AI density when human is active (very low)
    density_when_autonomous: float = 0.5  # AI density when autonomous (moderate)
    
    # Give space factor
    give_space_when_active: float = 0.9  # Give lots of space when human is active
    give_space_when_autonomous: float = 0.3  # Normal space when autonomous


class GPTOSSBehaviorParser:
    """
    Parses GPT-OSS silence_strategy and role_development insights
    Extracts behavioral rules and timing parameters
    """
    
    @staticmethod
    def parse_silence_strategy(silence_strategy_text: str) -> Dict[str, float]:
        """
        Parse GPT-OSS silence strategy text to extract behavioral parameters
        
        Example insights to look for:
        - "responds quickly after 1-2 seconds of silence"
        - "gives space when human is active"
        - "fills gaps with sparse bass"
        - "melody stays quiet during vocals"
        """
        config = {}
        
        if not silence_strategy_text:
            return config
        
        text_lower = silence_strategy_text.lower()
        
        # Extract silence timeout
        timeout_patterns = [
            r'after\s+(\d+(?:\.\d+)?)\s*(?:-\s*(\d+(?:\.\d+)?))?\s*seconds?',
            r'wait(?:s)?\s+(\d+(?:\.\d+)?)\s*seconds?',
            r'(\d+(?:\.\d+)?)\s*(?:-\s*(\d+(?:\.\d+)?))?\s*seconds?.*?(?:silence|pause|gap)'
        ]
        
        for pattern in timeout_patterns:
            match = re.search(pattern, text_lower)
            if match:
                if match.group(2):  # Range like "1-2 seconds"
                    config['silence_timeout'] = (float(match.group(1)) + float(match.group(2))) / 2
                else:
                    config['silence_timeout'] = float(match.group(1))
                break
        
        # Detect space-giving behavior
        if any(phrase in text_lower for phrase in ['gives space', 'backs off', 'listens', 'quiet when']):
            config['give_space_when_active'] = 0.9  # Give lots of space
        
        # Detect accompaniment behavior
        if any(phrase in text_lower for phrase in ['sparse bass', 'bass accompaniment', 'supportive bass']):
            config['bass_accompaniment_probability'] = 0.6
        elif 'no bass' in text_lower or 'silent bass' in text_lower:
            config['bass_accompaniment_probability'] = 0.1
        
        # Detect melody behavior during human activity
        if any(phrase in text_lower for phrase in ['melody quiet', 'no melody', 'melody silent']):
            config['melody_silence_when_active'] = True
        elif any(phrase in text_lower for phrase in ['melody responds', 'sparse melody', 'occasional melody']):
            config['melody_silence_when_active'] = False
        
        return config
    
    @staticmethod
    def parse_role_development(role_development_text: str) -> Dict[str, float]:
        """
        Parse GPT-OSS role development text to extract role-based parameters
        
        Example insights:
        - "bass provides harmonic foundation throughout"
        - "melody emerges during gaps"
        - "roles alternate between lead and support"
        """
        config = {}
        
        if not role_development_text:
            return config
        
        text_lower = role_development_text.lower()
        
        # Detect bass role
        if any(phrase in text_lower for phrase in ['bass.*foundation', 'bass.*constant', 'bass.*throughout']):
            config['bass_accompaniment_probability'] = 0.8  # High accompaniment
        elif any(phrase in text_lower for phrase in ['bass.*sparse', 'bass.*occasional']):
            config['bass_accompaniment_probability'] = 0.3  # Low accompaniment
        
        # Detect melody role
        if any(phrase in text_lower for phrase in ['melody.*gaps', 'melody.*silence', 'melody.*responds']):
            config['autonomous_interval'] = 2.0  # Faster response to gaps
        elif any(phrase in text_lower for phrase in ['melody.*continuous', 'melody.*constant']):
            config['autonomous_interval'] = 4.0  # Slower, more sustained
        
        # Detect alternating roles
        if 'alternate' in text_lower or 'trade' in text_lower:
            config['melody_silence_when_active'] = True
            config['bass_accompaniment_probability'] = 0.5
        
        return config
    
    @staticmethod
    def create_config_from_gpt_oss(
        silence_strategy: Optional[str] = None,
        role_development: Optional[str] = None,
        defaults: Optional[AutonomousBehaviorConfig] = None
    ) -> AutonomousBehaviorConfig:
        """
        Create configuration from GPT-OSS insights
        
        Args:
            silence_strategy: GPT-OSS silence strategy text
            role_development: GPT-OSS role development text
            defaults: Default configuration to start from
            
        Returns:
            Configured AutonomousBehaviorConfig
        """
        # Start with defaults or create new
        config = defaults or AutonomousBehaviorConfig()
        
        # Parse silence strategy
        if silence_strategy:
            silence_params = GPTOSSBehaviorParser.parse_silence_strategy(silence_strategy)
            for key, value in silence_params.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        
        # Parse role development
        if role_development:
            role_params = GPTOSSBehaviorParser.parse_role_development(role_development)
            for key, value in role_params.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        
        return config


class AutonomousBehaviorManager:
    """
    Manages autonomous behavior and integrates with existing agent components
    Bridges GPT-OSS insights to DensityController and BehaviorScheduler
    """
    
    def __init__(self, config: AutonomousBehaviorConfig):
        self.config = config
        self.human_activity_level = 0.0
        self.last_human_event_time = 0.0
        self.was_in_autonomous_mode = False
    
    def update_from_event(self, event_data: Dict, current_time: float):
        """Update activity tracking from audio event"""
        rms_db = event_data.get('rms_db', -80)
        
        if rms_db > -60:  # Significant audio detected
            self.last_human_event_time = current_time
            
            # Update activity level with exponential smoothing
            instant_activity = min(1.0, (rms_db + 60) / 40)
            alpha = 0.3
            self.human_activity_level = alpha * instant_activity + (1 - alpha) * self.human_activity_level
        else:
            # Decay activity during silence
            time_since_last = current_time - self.last_human_event_time
            if time_since_last > 0.5:
                self.human_activity_level *= 0.95
    
    def is_in_autonomous_mode(self, current_time: float) -> bool:
        """Check if AI should be in autonomous mode"""
        time_since_human = current_time - self.last_human_event_time
        return time_since_human > self.config.silence_timeout
    
    def get_voice_filter_decision(self, voice_type: str, current_time: float) -> bool:
        """
        Decide if a voice should play based on current activity
        
        Returns:
            True if voice should play, False if it should be filtered out
        """
        in_autonomous = self.is_in_autonomous_mode(current_time)
        
        if in_autonomous:
            # Autonomous mode - both voices can play
            return True
        
        # Human is active - apply voice-specific filtering
        if voice_type == 'bass':
            # Bass can play with configured probability
            import random
            return random.random() < self.config.bass_accompaniment_probability
        elif voice_type == 'melodic':
            # Melody behavior depends on configuration
            if self.config.melody_silence_when_active:
                return False  # Stay silent
            else:
                # Allow sparse melody (20% probability)
                import random
                return random.random() < 0.2
        
        return True
    
    def get_density_parameters(self, current_time: float) -> Tuple[float, float]:
        """
        Get density and give_space parameters based on current mode
        
        Returns:
            (density_level, give_space_factor)
        """
        in_autonomous = self.is_in_autonomous_mode(current_time)
        
        if in_autonomous:
            return (self.config.density_when_autonomous, 
                   self.config.give_space_when_autonomous)
        else:
            # Responsive mode - adjust based on human activity
            if self.human_activity_level > self.config.activity_threshold_high:
                # Very active - give lots of space
                return (self.config.density_when_active * 0.5, 
                       self.config.give_space_when_active)
            elif self.human_activity_level > self.config.activity_threshold_low:
                # Moderately active
                return (self.config.density_when_active, 
                       self.config.give_space_when_active * 0.7)
            else:
                # Low activity
                return (self.config.density_when_active * 2.0, 
                       self.config.give_space_when_active * 0.5)
    
    def should_generate_autonomous(self, current_time: float, last_generation: float) -> bool:
        """Check if autonomous generation should trigger"""
        in_autonomous = self.is_in_autonomous_mode(current_time)
        
        if not in_autonomous:
            return False
        
        # Check if just entered autonomous mode
        just_entered = in_autonomous and not self.was_in_autonomous_mode
        self.was_in_autonomous_mode = in_autonomous
        
        if just_entered:
            # Generate immediately on entering autonomous mode
            return True
        
        # Check interval
        interval = self.config.autonomous_interval * 0.5  # 2x faster when autonomous
        return (current_time - last_generation) >= interval

