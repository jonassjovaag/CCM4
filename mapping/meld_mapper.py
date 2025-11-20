# meld_mapper.py
# Maps MusicHal's extracted features to Ableton Meld synthesizer parameters
# Extends FeatureMapper with probabilistic routing and dual-engine architecture

import random
import yaml
import os
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np

@dataclass
class MeldParameters:
    """Parameters for Meld synthesizer control"""
    # Engine A (Melodic)
    engine_a_macro_1: float  # 0.0-1.0
    engine_a_macro_2: float  # 0.0-1.0
    
    # Engine B (Bass)
    engine_b_macro_1: float  # 0.0-1.0
    engine_b_macro_2: float  # 0.0-1.0
    
    # Global
    ab_blend: float  # 0.0-1.0 (0=A only, 1=B only)
    spread: float  # 0.0-1.0
    
    # Filter
    filter_frequency: float  # 0.0-1.0
    filter_resonance: float  # 0.0-1.0
    
    # Scale-aware (set separately based on harmonic context)
    scale_enable: bool = False
    root_note: Optional[int] = None  # 0-11 (C-B)
    scale_type: Optional[str] = None  # 'major', 'minor', etc.
    
    # Probabilistic routing flag (for logging/transparency)
    used_alternative_mapping: bool = False


class MeldMapper:
    """
    Maps audio features to Meld synthesizer parameters
    
    Features:
    - Probabilistic macro routing (configurable accident probability)
    - Dual-engine parameter mapping (Engine A melodic, Engine B bass)
    - Scale-aware filter control
    - CC throttling to prevent MIDI congestion
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize Meld mapper
        
        Args:
            config_path: Path to meld_mapping.yaml (default: config/meld_mapping.yaml)
        """
        if config_path is None:
            # Default to config/meld_mapping.yaml
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(base_dir, 'config', 'meld_mapping.yaml')
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Probabilistic routing settings
        self.probabilistic_enabled = self.config['probabilistic_routing']['enabled']
        self.accident_probability = self.config['probabilistic_routing']['accident_probability']
        
        # CC throttling state
        self.last_values: Dict[str, float] = {}
        self.change_threshold = self.config['throttling']['change_threshold']
        
        # EMA smoothing state for timbre_variance-based control
        self.ema_state: Dict[str, float] = {}
        
        print("🎹 Meld Mapper initialized")
        print(f"   Probabilistic routing: {self.probabilistic_enabled} ({self.accident_probability:.0%} accidents)")
        print(f"   CC throttling threshold: {self.change_threshold:.1%}")
    
    def map_features_to_meld(self, 
                            event_data: Dict,
                            voice_type: str = "melodic",
                            timbre_variance: float = 0.5) -> MeldParameters:
        """
        Map audio features to Meld parameters
        
        Args:
            event_data: Event dictionary with extracted features
            voice_type: 'melodic' (Engine A) or 'bass' (Engine B)
            timbre_variance: 0=stable (high smoothing), 1=expressive (low smoothing)
            
        Returns:
            MeldParameters with all mapped values
        """
        # Determine if we use alternative mapping (probabilistic accidents)
        use_alternative = False
        if self.probabilistic_enabled:
            use_alternative = random.random() < self.accident_probability
        
        # Extract features with safe defaults
        spectral_centroid = self._get_feature(event_data, 'spectral_centroid', 0.5)
        consonance = self._get_feature(event_data, 'consonance', 0.7)
        zcr = self._get_feature(event_data, 'zcr', 0.3)
        spectral_rolloff = self._normalize_feature(
            self._get_feature(event_data, 'spectral_rolloff', 2000), 
            min_val=500, max_val=8000
        )
        bandwidth = self._normalize_feature(
            self._get_feature(event_data, 'bandwidth', 1000),
            min_val=200, max_val=4000
        )
        spectral_flatness = self._get_feature(event_data, 'flatness', 0.1)
        mfcc_1 = self._normalize_feature(
            self._get_feature(event_data, 'mfcc_1', 0.0),
            min_val=-50, max_val=50
        )
        modulation_depth = self._get_feature(event_data, 'modulation_depth', 0.5)
        
        # Engine A macros (melodic)
        if use_alternative:
            engine_a_macro_1 = bandwidth  # Alternative: bandwidth instead of centroid
            engine_a_macro_2 = mfcc_1     # Alternative: timbral instead of consonance
        else:
            engine_a_macro_1 = spectral_centroid  # Primary: brightness
            engine_a_macro_2 = consonance         # Primary: harmonic stability
        
        # Engine B macros (bass)
        if use_alternative:
            engine_b_macro_1 = spectral_flatness   # Alternative: noisiness
            engine_b_macro_2 = spectral_rolloff    # Same (no good alternative for bass)
        else:
            engine_b_macro_1 = zcr                 # Primary: percussive character
            engine_b_macro_2 = spectral_rolloff    # Primary: bass brightness
        
        # Global parameters (no probabilistic routing)
        ab_blend = mfcc_1                 # Timbral evolution drives blend
        spread = modulation_depth         # Activity drives complexity
        
        # Apply EMA smoothing based on timbre_variance
        # Low timbre_variance (bass) = high smoothing (low alpha)
        # High timbre_variance (melodic) = low smoothing (high alpha)
        engine_a_macro_1 = self._apply_ema_smoothing(
            f"{voice_type}_a1", engine_a_macro_1, timbre_variance
        )
        engine_a_macro_2 = self._apply_ema_smoothing(
            f"{voice_type}_a2", engine_a_macro_2, timbre_variance
        )
        engine_b_macro_1 = self._apply_ema_smoothing(
            f"{voice_type}_b1", engine_b_macro_1, timbre_variance
        )
        engine_b_macro_2 = self._apply_ema_smoothing(
            f"{voice_type}_b2", engine_b_macro_2, timbre_variance
        )
        
        # Add subtle jitter to prevent stuck values (0.5-1.5% variation)
        jitter_amount = 0.01 * timbre_variance  # More jitter for expressive voices
        engine_a_macro_1 += random.uniform(-jitter_amount, jitter_amount)
        engine_a_macro_2 += random.uniform(-jitter_amount, jitter_amount)
        engine_b_macro_1 += random.uniform(-jitter_amount * 0.5, jitter_amount * 0.5)  # Less jitter for bass
        engine_b_macro_2 += random.uniform(-jitter_amount * 0.5, jitter_amount * 0.5)
        
        # Filter parameters
        filter_frequency = self._apply_scaling(spectral_rolloff, 'exponential')
        filter_resonance = 1.0 - spectral_flatness  # INVERTED: noise → low Q
        
        # Create parameter object (clamp to 0-1 range after jitter)
        params = MeldParameters(
            engine_a_macro_1=self._clamp(engine_a_macro_1),
            engine_a_macro_2=self._clamp(engine_a_macro_2),
            engine_b_macro_1=self._clamp(engine_b_macro_1),
            engine_b_macro_2=self._clamp(engine_b_macro_2),
            ab_blend=self._clamp(ab_blend),
            spread=self._clamp(spread),
            filter_frequency=self._clamp(filter_frequency),
            filter_resonance=self._clamp(filter_resonance),
            used_alternative_mapping=use_alternative
        )
        
        return params
    
    def should_send_cc(self, param_name: str, value: float) -> bool:
        """
        Check if CC should be sent based on throttling settings
        
        Args:
            param_name: Parameter identifier (e.g., 'engine_a_macro_1')
            value: New parameter value (0.0-1.0)
            
        Returns:
            True if CC should be sent (value changed significantly)
        """
        if param_name not in self.last_values:
            # First time - always send
            self.last_values[param_name] = value
            return True
        
        # Check if change exceeds threshold
        last_value = self.last_values[param_name]
        change = abs(value - last_value)
        
        if change >= self.change_threshold:
            self.last_values[param_name] = value
            return True
        
        return False
    
    def parse_chord_for_scale(self, chord_name: str) -> Tuple[int, str]:
        """
        Parse chord name to extract root note and scale type
        
        Args:
            chord_name: Chord string like "Cmaj7", "Dm", "G7", etc.
            
        Returns:
            Tuple of (root_note: 0-11, scale_type: str)
        """
        if not chord_name or chord_name == "None":
            return (0, 'chromatic')  # Default to C chromatic
        
        # Parse root note
        root_map = {
            'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
            'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
            'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11
        }
        
        # Extract root (first 1-2 characters)
        root_str = chord_name[0]
        if len(chord_name) > 1 and chord_name[1] in ['#', 'b']:
            root_str += chord_name[1]
        
        root_note = root_map.get(root_str, 0)
        
        # Parse quality → scale type
        chord_lower = chord_name.lower()
        
        if 'dim' in chord_lower:
            scale_type = 'chromatic'  # Diminished → chromatic for flexibility
        elif 'm' in chord_lower and 'maj' not in chord_lower:
            if 'harmonic' in chord_lower:
                scale_type = 'harmonic_minor'
            else:
                scale_type = 'minor'
        elif 'maj' in chord_lower or chord_name[0].isupper():
            scale_type = 'major'
        elif '7' in chord_lower:
            scale_type = 'mixolydian'  # Dominant 7th → mixolydian
        else:
            scale_type = 'major'  # Default
        
        return (root_note, scale_type)
    
    def _get_feature(self, event_data: Dict, key: str, default: float) -> float:
        """Safely extract feature from event data"""
        return float(event_data.get(key, default))
    
    def _normalize_feature(self, value: float, min_val: float, max_val: float) -> float:
        """Normalize feature to 0.0-1.0 range"""
        if max_val == min_val:
            return 0.5
        normalized = (value - min_val) / (max_val - min_val)
        return np.clip(normalized, 0.0, 1.0)
    
    def _apply_scaling(self, value: float, scaling: str) -> float:
        """Apply scaling curve to normalized value"""
        if scaling == 'exponential':
            # Exponential curve (more sensitive at high end)
            return value ** 2
        elif scaling == 'logarithmic':
            # Logarithmic curve (more sensitive at low end)
            return np.sqrt(value)
        else:  # 'linear'
            return value
    
    def _clamp(self, value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """Clamp value to range"""
        return np.clip(value, min_val, max_val)
    
    def _apply_ema_smoothing(self, key: str, value: float, timbre_variance: float) -> float:
        """
        Apply exponential moving average smoothing based on timbre_variance
        
        Args:
            key: Unique identifier for this smoothing channel
            value: New raw value (0.0-1.0)
            timbre_variance: 0=stable (high smoothing), 1=expressive (low smoothing)
            
        Returns:
            Smoothed value (0.0-1.0)
        
        Smoothing formula:
            smoothed = alpha * new + (1-alpha) * old
            alpha = timbre_variance (high variance = low smoothing = high alpha)
            
        Examples:
            timbre_variance=0.2 (bass) → alpha=0.2 → 80% old, 20% new (very smooth)
            timbre_variance=0.8 (melody) → alpha=0.8 → 20% old, 80% new (responsive)
        """
        # First call for this key - no previous state
        if key not in self.ema_state:
            self.ema_state[key] = value
            return value
        
        # Apply EMA smoothing
        alpha = timbre_variance  # Higher variance = less smoothing = higher alpha
        smoothed = alpha * value + (1.0 - alpha) * self.ema_state[key]
        
        # Update state
        self.ema_state[key] = smoothed
        
        return smoothed
    
    def get_cc_mapping(self, param_name: str) -> int:
        """
        Get CC number for parameter
        
        Args:
            param_name: Parameter name (e.g., 'engine_a_macro_1')
            
        Returns:
            MIDI CC number
        """
        # Map parameter names to config paths
        cc_map = {
            'engine_a_macro_1': self.config['engine_a']['macro_1']['cc_number'],
            'engine_a_macro_2': self.config['engine_a']['macro_2']['cc_number'],
            'engine_b_macro_1': self.config['engine_b']['macro_1']['cc_number'],
            'engine_b_macro_2': self.config['engine_b']['macro_2']['cc_number'],
            'ab_blend': self.config['global']['ab_blend']['cc_number'],
            'spread': self.config['global']['spread']['cc_number'],
            'filter_frequency': self.config['filter']['frequency']['cc_number'],
            'filter_resonance': self.config['filter']['resonance']['cc_number'],
            'scale_enable': self.config['scale_aware']['enable']['cc_number'],
            'root_note': self.config['scale_aware']['root_note']['cc_number'],
            'scale_type': self.config['scale_aware']['scale_type']['cc_number'],
        }
        
        return cc_map.get(param_name, 0)
    
    def get_scale_type_cc_value(self, scale_type: str) -> int:
        """
        Get MIDI CC value for scale type
        
        Args:
            scale_type: Scale name ('major', 'minor', etc.)
            
        Returns:
            MIDI CC value (0-127)
        """
        mappings = self.config['scale_aware']['scale_type']['mappings']
        return mappings.get(scale_type, mappings['major'])


if __name__ == "__main__":
    # Demo/test
    print("🎹 Meld Mapper Demo")
    print("=" * 50)
    
    mapper = MeldMapper()
    
    # Test event data
    test_events = [
        {
            'name': 'Bright harmonic chord',
            'spectral_centroid': 0.8,
            'consonance': 0.9,
            'zcr': 0.2,
            'spectral_rolloff': 5000,
            'bandwidth': 2000,
            'flatness': 0.1,
            'mfcc_1': 20.0,
            'modulation_depth': 0.3
        },
        {
            'name': 'Dark percussive hit',
            'spectral_centroid': 0.3,
            'consonance': 0.4,
            'zcr': 0.8,
            'spectral_rolloff': 1500,
            'bandwidth': 800,
            'flatness': 0.6,
            'mfcc_1': -15.0,
            'modulation_depth': 0.9
        }
    ]
    
    for event in test_events:
        print(f"\n{event['name']}:")
        params = mapper.map_features_to_meld(event, voice_type='melodic')
        print(f"  Engine A: Macro1={params.engine_a_macro_1:.2f}, Macro2={params.engine_a_macro_2:.2f}")
        print(f"  Engine B: Macro1={params.engine_b_macro_1:.2f}, Macro2={params.engine_b_macro_2:.2f}")
        print(f"  Global: Blend={params.ab_blend:.2f}, Spread={params.spread:.2f}")
        print(f"  Filter: Freq={params.filter_frequency:.2f}, Q={params.filter_resonance:.2f}")
        if params.used_alternative_mapping:
            print("  ⚡ Alternative mapping used (probabilistic accident)")
    
    # Test chord parsing
    print("\n" + "=" * 50)
    print("Scale-aware chord parsing:")
    test_chords = ['Cmaj7', 'Dm', 'G7', 'Am7', 'F#m', 'Bbmaj7']
    for chord in test_chords:
        root, scale = mapper.parse_chord_for_scale(chord)
        cc_val = mapper.get_scale_type_cc_value(scale)
        print(f"  {chord:8} → Root={root:2} ({['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'][root]}), Scale={scale}, CC={cc_val}")
