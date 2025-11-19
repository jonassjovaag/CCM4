# behaviors.py
# AI Agent behaviors: Imitate, Contrast, Lead

import time
import random
import numpy as np
from typing import Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
from .phrase_generator import PhraseGenerator, MusicalPhrase

class BehaviorMode(Enum):
    # Original modes
    IMITATE = "imitate"
    CONTRAST = "contrast"
    LEAD = "lead"
    
    # IRCAM Phase 3: Research-based interaction modes
    SHADOW = "shadow"    # Close imitation, quick response
    MIRROR = "mirror"    # Similar with variation, phrase-aware
    COUPLE = "couple"    # Independent but complementary

@dataclass
class MusicalDecision:
    """Represents a musical decision made by the AI agent"""
    mode: BehaviorMode
    confidence: float
    target_features: np.ndarray
    musical_params: Dict
    timestamp: float
    reasoning: str
    voice_type: str = "melodic"  # "melodic" or "bass"
    instrument: str = "unknown"  # Instrument classification
    
    # CLAP-driven timing profile parameters
    timing_precision: float = 0.5  # 0=loose/expressive, 1=tight/quantized
    rhythmic_density: float = 0.5  # 0=very sparse, 1=frequent notes
    syncopation_tendency: float = 0.3  # 0=on-beat, 1=off-beat
    voice_profile: Dict[str, float] = None  # Applied timing profile


# Voice timing profiles based on CLAP style detection
class VoiceTimingProfile:
    """
    Timing characteristics for melodic vs bass voices based on CLAP style
    
    Profiles define:
    - timing_precision: 0=loose/expressive, 1=tight/quantized
    - rhythmic_density: 0=very sparse, 1=frequent notes
    - syncopation_tendency: 0=on-beat, 1=off-beat
    - timbre_variance: 0=stable, 1=highly variable (for Meld CC)
    """
    
    STYLE_PROFILES = {
        'ballad': {
            'melodic': {
                'timing_precision': 0.3,  # Loose, expressive
                'rhythmic_density': 0.4,  # Sparse, phrase-like
                'syncopation_tendency': 0.2,  # Mostly on-beat
                'timbre_variance': 0.7,  # Expressive timbre changes
                # Episode engagement (hierarchical level 3)
                'episode_active_duration_range': (3.0, 10.0),  # Short expressive statements
                'episode_listening_duration_range': (15.0, 45.0),  # Contemplative listening
                'interference_probability': 0.4,  # Sometimes play over human
                'early_exit_probability': 0.15,  # Responsive to human
                'context_sensitivity': 0.6,  # Moderately context-aware
            },
            'bass': {
                'timing_precision': 0.9,  # Tight, grounded
                'rhythmic_density': 0.2,  # Minimal movement
                'syncopation_tendency': 0.05,  # Steady on-beat
                'timbre_variance': 0.2,  # Stable, cool
                # Episode engagement (bass is steady, not hierarchical)
                'episode_active_duration_range': (10.0, 30.0),  # Long steady presence
                'episode_listening_duration_range': (3.0, 10.0),  # Brief breaks only
                'interference_probability': 0.15,  # Rarely interferes
                'early_exit_probability': 0.05,  # Very stable
                'context_sensitivity': 0.3,  # Foundation-focused
            }
        },
        'jazz': {
            'melodic': {
                'timing_precision': 0.4,  # Swing feel
                'rhythmic_density': 0.5,  # Conversational
                'syncopation_tendency': 0.4,  # Jazz phrasing
                'timbre_variance': 0.6,  # Expressive
                # Episode engagement
                'episode_active_duration_range': (4.0, 12.0),  # Conversational phrases
                'episode_listening_duration_range': (12.0, 35.0),  # Trading solos
                'interference_probability': 0.5,  # Conversational overlap
                'early_exit_probability': 0.2,  # Responsive trading
                'context_sensitivity': 0.7,  # Highly interactive
            },
            'bass': {
                'timing_precision': 0.85,  # Walking bass feel
                'rhythmic_density': 0.5,  # Active but steady
                'syncopation_tendency': 0.1,  # Mostly on-beat
                'timbre_variance': 0.25,  # Consistent tone
                # Episode engagement (walking bass consistency)
                'episode_active_duration_range': (15.0, 45.0),  # Long walking patterns
                'episode_listening_duration_range': (3.0, 8.0),  # Very brief breaks
                'interference_probability': 0.2,  # Supportive foundation
                'early_exit_probability': 0.08,  # Steady walk
                'context_sensitivity': 0.4,  # Consistent support
            }
        },
        'blues': {
            'melodic': {
                'timing_precision': 0.25,  # Very loose, emotional
                'rhythmic_density': 0.3,  # Space for expression
                'syncopation_tendency': 0.3,  # Blues phrasing
                'timbre_variance': 0.75,  # Expressive bends
                # Episode engagement
                'episode_active_duration_range': (3.0, 8.0),  # Short emotional bursts
                'episode_listening_duration_range': (40.0, 100.0),  # Long spaces between
                'interference_probability': 0.35,  # Call and response
                'early_exit_probability': 0.2,  # Emotional responsiveness
                'context_sensitivity': 0.65,  # Emotionally reactive
            },
            'bass': {
                'timing_precision': 0.8,  # Steady shuffle
                'rhythmic_density': 0.3,  # Simple patterns
                'syncopation_tendency': 0.05,  # Groove-locked
                'timbre_variance': 0.2,  # Stable foundation
                # Episode engagement (blues groove consistency)
                'episode_active_duration_range': (12.0, 35.0),  # Long groove patterns
                'episode_listening_duration_range': (3.0, 10.0),  # Brief breaks
                'interference_probability': 0.15,  # Steady support
                'early_exit_probability': 0.05,  # Locked groove
                'context_sensitivity': 0.3,  # Foundation-focused
            }
        },
        'ambient': {
            'melodic': {
                'timing_precision': 0.1,  # Very loose, floating
                'rhythmic_density': 0.2,  # Very sparse
                'syncopation_tendency': 0.1,  # Drifting
                'timbre_variance': 0.5,  # Moderate evolution
                # Episode engagement
                'episode_active_duration_range': (2.0, 6.0),  # Brief textures
                'episode_listening_duration_range': (60.0, 180.0),  # Long silences
                'interference_probability': 0.3,  # Layered textures
                'early_exit_probability': 0.25,  # Floating, unpredictable
                'context_sensitivity': 0.5,  # Moderate awareness
            },
            'bass': {
                'timing_precision': 0.7,  # Moderately tight
                'rhythmic_density': 0.1,  # Very minimal
                'syncopation_tendency': 0.02,  # Almost pedal tones
                'timbre_variance': 0.1,  # Very stable
                # Episode engagement (ambient foundation)
                'episode_active_duration_range': (20.0, 60.0),  # Long sustained presence
                'episode_listening_duration_range': (5.0, 15.0),  # Occasional breaks
                'interference_probability': 0.1,  # Minimal interference
                'early_exit_probability': 0.1,  # Stable foundation
                'context_sensitivity': 0.25,  # Low reactivity
            }
        },
        'classical': {
            'melodic': {
                'timing_precision': 0.6,  # Measured, precise
                'rhythmic_density': 0.5,  # Phrase-aware
                'syncopation_tendency': 0.2,  # Mostly metric
                'timbre_variance': 0.5,  # Controlled expression
                # Episode engagement
                'episode_active_duration_range': (5.0, 15.0),  # Formal phrases
                'episode_listening_duration_range': (30.0, 80.0),  # Structured pauses
                'interference_probability': 0.3,  # Counterpoint-aware
                'early_exit_probability': 0.1,  # Formal structure
                'context_sensitivity': 0.55,  # Moderately responsive
            },
            'bass': {
                'timing_precision': 0.9,  # Very precise
                'rhythmic_density': 0.3,  # Harmonic support
                'syncopation_tendency': 0.05,  # Mostly on-beat
                'timbre_variance': 0.15,  # Consistent
                # Episode engagement (classical foundation)
                'episode_active_duration_range': (15.0, 40.0),  # Long harmonic foundation
                'episode_listening_duration_range': (4.0, 12.0),  # Brief rests
                'interference_probability': 0.15,  # Supportive role
                'early_exit_probability': 0.05,  # Formal stability
                'context_sensitivity': 0.35,  # Structured support
            }
        },
        'electronic': {
            'melodic': {
                'timing_precision': 0.7,  # Moderate precision
                'rhythmic_density': 0.4,  # Textural
                'syncopation_tendency': 0.3,  # Some syncopation
                'timbre_variance': 0.6,  # Evolving textures
                # Episode engagement
                'episode_active_duration_range': (4.0, 12.0),  # Textural layers
                'episode_listening_duration_range': (35.0, 95.0),  # Build-release cycles
                'interference_probability': 0.45,  # Layered production
                'early_exit_probability': 0.18,  # Dynamic arrangement
                'context_sensitivity': 0.6,  # Production-aware
            },
            'bass': {
                'timing_precision': 0.95,  # Tight electronic precision
                'rhythmic_density': 0.5,  # Active patterns
                'syncopation_tendency': 0.15,  # Some off-beats
                'timbre_variance': 0.2,  # Stable foundation
                # Episode engagement (electronic consistency)
                'episode_active_duration_range': (15.0, 50.0),  # Long patterns
                'episode_listening_duration_range': (3.0, 10.0),  # Brief drops
                'interference_probability': 0.2,  # Tight production
                'early_exit_probability': 0.08,  # Programmed consistency
                'context_sensitivity': 0.4,  # Electronic stability
            }
        },
        # Energetic styles
        'funk': {
            'melodic': {
                'timing_precision': 0.6,  # Groove feel
                'rhythmic_density': 0.3,  # Sparse stabs
                'syncopation_tendency': 0.5,  # Funky syncopation
                'timbre_variance': 0.8,  # Energetic changes
                # Episode engagement
                'episode_active_duration_range': (3.0, 10.0),  # Funky stabs
                'episode_listening_duration_range': (25.0, 70.0),  # Groove pockets
                'interference_probability': 0.55,  # Energetic interplay
                'early_exit_probability': 0.2,  # Dynamic energy
                'context_sensitivity': 0.7,  # Groove-responsive
            },
            'bass': {
                'timing_precision': 0.95,  # Very tight groove
                'rhythmic_density': 0.7,  # Active groove
                'syncopation_tendency': 0.2,  # Groove-locked
                'timbre_variance': 0.15,  # Very stable groove
                # Episode engagement (funk groove consistency)
                'episode_active_duration_range': (20.0, 60.0),  # Long grooves
                'episode_listening_duration_range': (2.0, 8.0),  # Very brief breaks
                'interference_probability': 0.2,  # Groove foundation
                'early_exit_probability': 0.05,  # Locked in the pocket
                'context_sensitivity': 0.4,  # Groove-focused
            }
        },
        'rock': {
            'melodic': {
                'timing_precision': 0.6,  # Moderate precision
                'rhythmic_density': 0.4,  # Active
                'syncopation_tendency': 0.3,  # Some off-beats
                'timbre_variance': 0.75,  # Energetic
                # Episode engagement
                'episode_active_duration_range': (4.0, 14.0),  # Rock phrases
                'episode_listening_duration_range': (28.0, 75.0),  # Verse-chorus cycles
                'interference_probability': 0.42,  # Energetic layering
                'early_exit_probability': 0.16,  # Dynamic changes
                'context_sensitivity': 0.62,  # Energy-responsive
            },
            'bass': {
                'timing_precision': 0.9,  # Tight
                'rhythmic_density': 0.6,  # Driving
                'syncopation_tendency': 0.1,  # Mostly on-beat
                'timbre_variance': 0.2,  # Stable power
                # Episode engagement (rock drive)
                'episode_active_duration_range': (15.0, 45.0),  # Long driving patterns
                'episode_listening_duration_range': (3.0, 10.0),  # Brief power breaks
                'interference_probability': 0.2,  # Driving foundation
                'early_exit_probability': 0.08,  # Committed drive
                'context_sensitivity': 0.4,  # Energy foundation
            }
        },
        'metal': {
            'melodic': {
                'timing_precision': 0.7,  # Precise attack
                'rhythmic_density': 0.5,  # Dense
                'syncopation_tendency': 0.4,  # Complex rhythms
                'timbre_variance': 0.8,  # Aggressive changes
                # Episode engagement
                'episode_active_duration_range': (5.0, 16.0),  # Aggressive riffs
                'episode_listening_duration_range': (25.0, 65.0),  # Breakdown sections
                'interference_probability': 0.48,  # Aggressive layering
                'early_exit_probability': 0.18,  # Dynamic intensity
                'context_sensitivity': 0.65,  # Intensity-responsive
            },
            'bass': {
                'timing_precision': 0.95,  # Very tight
                'rhythmic_density': 0.7,  # Dense patterns
                'syncopation_tendency': 0.15,  # Locked in
                'timbre_variance': 0.15,  # Stable aggression
                # Episode engagement (metal tightness)
                'episode_active_duration_range': (15.0, 50.0),  # Heavy sustained riffs
                'episode_listening_duration_range': (3.0, 10.0),  # Brief breakdowns
                'interference_probability': 0.25,  # Tight foundation
                'early_exit_probability': 0.08,  # Locked intensity
                'context_sensitivity': 0.45,  # Metal foundation
            }
        },
        'punk': {
            'melodic': {
                'timing_precision': 0.5,  # Raw energy
                'rhythmic_density': 0.6,  # Fast
                'syncopation_tendency': 0.3,  # Energetic
                'timbre_variance': 0.85,  # Raw changes
                # Episode engagement
                'episode_active_duration_range': (3.0, 8.0),  # Fast bursts
                'episode_listening_duration_range': (20.0, 55.0),  # Brief rests
                'interference_probability': 0.52,  # Raw energy overlaps
                'early_exit_probability': 0.22,  # Unpredictable energy
                'context_sensitivity': 0.68,  # Reactive chaos
            },
            'bass': {
                'timing_precision': 0.85,  # Driving
                'rhythmic_density': 0.7,  # Fast patterns
                'syncopation_tendency': 0.1,  # Driving beat
                'timbre_variance': 0.2,  # Consistent power
                # Episode engagement (punk energy)
                'episode_active_duration_range': (12.0, 35.0),  # Fast sustained drive
                'episode_listening_duration_range': (2.0, 8.0),  # Very quick breaks
                'interference_probability': 0.25,  # Energy foundation
                'early_exit_probability': 0.1,  # Energetic but consistent
                'context_sensitivity': 0.45,  # Energy-focused
            }
        },
        'world': {
            'melodic': {
                'timing_precision': 0.4,  # Cultural flexibility
                'rhythmic_density': 0.45,  # Varied
                'syncopation_tendency': 0.35,  # Cultural rhythms
                'timbre_variance': 0.65,  # Diverse expression
                # Episode engagement
                'episode_active_duration_range': (4.0, 13.0),  # Cultural phrases
                'episode_listening_duration_range': (32.0, 85.0),  # Cultural pacing
                'interference_probability': 0.4,  # Cultural dialogue
                'early_exit_probability': 0.17,  # Flexible responsiveness
                'context_sensitivity': 0.63,  # Culturally aware
            },
            'bass': {
                'timing_precision': 0.8,  # Cultural foundation
                'rhythmic_density': 0.4,  # Supportive
                'syncopation_tendency': 0.1,  # Grounded
                'timbre_variance': 0.25,  # Stable
                # Episode engagement (cultural foundation)
                'episode_active_duration_range': (12.0, 40.0),  # Cultural foundation
                'episode_listening_duration_range': (3.0, 12.0),  # Brief breathing
                'interference_probability': 0.18,  # Supportive foundation
                'early_exit_probability': 0.08,  # Cultural stability
                'context_sensitivity': 0.4,  # Foundation-focused
            }
        },
    }
    
    # Default fallback profile
    DEFAULT_PROFILE = {
        'melodic': {
            'timing_precision': 0.5,
            'rhythmic_density': 0.4,
            'syncopation_tendency': 0.3,
            'timbre_variance': 0.6,
            # Episode engagement
            'episode_active_duration_range': (3.0, 10.0),
            'episode_listening_duration_range': (30.0, 90.0),
            'interference_probability': 0.4,
            'early_exit_probability': 0.15,
            'context_sensitivity': 0.6,
        },
        'bass': {
            'timing_precision': 0.85,
            'rhythmic_density': 0.3,
            'syncopation_tendency': 0.1,
            'timbre_variance': 0.2,
            # Episode engagement (default bass consistency)
            'episode_active_duration_range': (12.0, 35.0),  # Long steady presence
            'episode_listening_duration_range': (3.0, 10.0),  # Brief breaks
            'interference_probability': 0.15,
            'early_exit_probability': 0.05,
            'context_sensitivity': 0.3,
        }
    }
    
    @classmethod
    def get_profile(cls, style: str, voice_type: str) -> Dict[str, float]:
        """
        Get timing profile for style and voice type
        
        Args:
            style: CLAP-detected style (e.g., 'ballad', 'funk')
            voice_type: 'melodic' or 'bass'
            
        Returns:
            Profile dict with timing parameters
        """
        if style in cls.STYLE_PROFILES:
            return cls.STYLE_PROFILES[style].get(voice_type, cls.DEFAULT_PROFILE[voice_type])
        return cls.DEFAULT_PROFILE.get(voice_type, cls.DEFAULT_PROFILE['melodic'])


class MelodicVocabulary:
    """
    Style-specific melodic vocabulary preferences.

    Expands CLAP influence beyond timing to affect:
    - Interval preferences (which intervals are favored)
    - Chromatic tendency (use of non-diatonic notes)
    - Blue notes (b3, b5, b7 for blues/jazz)
    - Phrase characteristics
    - Avoid intervals (e.g., parallel 5ths in classical)
    """

    # Interval names for reference
    INTERVALS = {
        0: 'unison', 1: 'minor_2nd', 2: 'major_2nd', 3: 'minor_3rd',
        4: 'major_3rd', 5: 'perfect_4th', 6: 'tritone', 7: 'perfect_5th',
        8: 'minor_6th', 9: 'major_6th', 10: 'minor_7th', 11: 'major_7th', 12: 'octave'
    }

    # Style-specific melodic vocabularies
    STYLE_VOCABULARIES = {
        'jazz': {
            # Jazz: chromatic, extensions, bebop lines
            'interval_weights': {
                1: 0.15, 2: 0.20, 3: 0.18, 4: 0.12, 5: 0.08,
                6: 0.05, 7: 0.08, 8: 0.04, 9: 0.05, 10: 0.03, 11: 0.02
            },
            'chromatic_tendency': 0.4,  # High chromaticism
            'blue_notes': True,  # Use b3, b5, b7
            'extensions': True,  # Use 9ths, 11ths, 13ths
            'phrase_length': 'medium',  # 4-8 notes typical
            'avoid_intervals': [],  # Nothing off-limits
            'approach_tones': True,  # Chromatic enclosure
            'scale_modes': ['dorian', 'mixolydian', 'lydian', 'altered']
        },
        'blues': {
            # Blues: pentatonic with blue notes, bends
            'interval_weights': {
                1: 0.05, 2: 0.15, 3: 0.25, 4: 0.10, 5: 0.12,
                6: 0.08, 7: 0.15, 8: 0.02, 9: 0.03, 10: 0.03, 11: 0.02
            },
            'chromatic_tendency': 0.2,  # Mostly pentatonic
            'blue_notes': True,  # Essential for blues
            'extensions': False,
            'phrase_length': 'short',  # 2-5 notes, call-response
            'avoid_intervals': [11],  # Avoid major 7th
            'approach_tones': False,
            'scale_modes': ['blues', 'minor_pentatonic', 'mixolydian']
        },
        'classical': {
            # Classical: diatonic, voice leading, avoid parallels
            'interval_weights': {
                1: 0.05, 2: 0.25, 3: 0.20, 4: 0.15, 5: 0.12,
                6: 0.02, 7: 0.10, 8: 0.05, 9: 0.04, 10: 0.01, 11: 0.01
            },
            'chromatic_tendency': 0.1,  # Mostly diatonic
            'blue_notes': False,
            'extensions': False,
            'phrase_length': 'long',  # 6-12 notes, formal phrases
            'avoid_intervals': [6],  # Avoid tritone leaps
            'approach_tones': False,
            'scale_modes': ['major', 'minor', 'harmonic_minor']
        },
        'rock': {
            # Rock: power chords, pentatonic, strong intervals
            'interval_weights': {
                1: 0.02, 2: 0.15, 3: 0.18, 4: 0.12, 5: 0.15,
                6: 0.05, 7: 0.20, 8: 0.03, 9: 0.05, 10: 0.03, 11: 0.02
            },
            'chromatic_tendency': 0.15,
            'blue_notes': True,  # Rock uses blue notes
            'extensions': False,
            'phrase_length': 'medium',
            'avoid_intervals': [],
            'approach_tones': False,
            'scale_modes': ['minor_pentatonic', 'natural_minor', 'dorian']
        },
        'funk': {
            # Funk: rhythmic, 9ths, chromatic passing
            'interval_weights': {
                1: 0.10, 2: 0.20, 3: 0.15, 4: 0.12, 5: 0.10,
                6: 0.05, 7: 0.12, 8: 0.04, 9: 0.07, 10: 0.03, 11: 0.02
            },
            'chromatic_tendency': 0.3,
            'blue_notes': True,
            'extensions': True,  # 9ths especially
            'phrase_length': 'short',  # Stabs and riffs
            'avoid_intervals': [],
            'approach_tones': True,
            'scale_modes': ['dorian', 'mixolydian', 'minor_pentatonic']
        },
        'electronic': {
            # Electronic: arpeggios, sequences, octaves
            'interval_weights': {
                1: 0.02, 2: 0.10, 3: 0.15, 4: 0.18, 5: 0.12,
                6: 0.03, 7: 0.15, 8: 0.05, 9: 0.08, 10: 0.02, 11: 0.02, 12: 0.08
            },
            'chromatic_tendency': 0.15,
            'blue_notes': False,
            'extensions': True,  # Arpeggio extensions
            'phrase_length': 'medium',
            'avoid_intervals': [],
            'approach_tones': False,
            'scale_modes': ['major', 'minor', 'phrygian', 'harmonic_minor']
        },
        'ambient': {
            # Ambient: slow, consonant, drifting
            'interval_weights': {
                1: 0.05, 2: 0.15, 3: 0.12, 4: 0.15, 5: 0.18,
                6: 0.02, 7: 0.20, 8: 0.03, 9: 0.05, 10: 0.02, 11: 0.01, 12: 0.02
            },
            'chromatic_tendency': 0.05,  # Very diatonic
            'blue_notes': False,
            'extensions': False,
            'phrase_length': 'long',  # Slow, drifting
            'avoid_intervals': [1, 6],  # Avoid dissonance
            'approach_tones': False,
            'scale_modes': ['major', 'lydian', 'mixolydian']
        },
        'ballad': {
            # Ballad: expressive, vocal-like
            'interval_weights': {
                1: 0.08, 2: 0.22, 3: 0.18, 4: 0.15, 5: 0.10,
                6: 0.02, 7: 0.12, 8: 0.04, 9: 0.05, 10: 0.02, 11: 0.02
            },
            'chromatic_tendency': 0.15,
            'blue_notes': False,
            'extensions': False,
            'phrase_length': 'medium',
            'avoid_intervals': [6],  # Avoid harsh tritone
            'approach_tones': False,
            'scale_modes': ['major', 'minor', 'dorian']
        }
    }

    # Default vocabulary for unknown styles
    DEFAULT_VOCABULARY = {
        'interval_weights': {
            1: 0.05, 2: 0.20, 3: 0.18, 4: 0.15, 5: 0.12,
            6: 0.03, 7: 0.15, 8: 0.04, 9: 0.05, 10: 0.02, 11: 0.01
        },
        'chromatic_tendency': 0.15,
        'blue_notes': False,
        'extensions': False,
        'phrase_length': 'medium',
        'avoid_intervals': [],
        'approach_tones': False,
        'scale_modes': ['major', 'minor']
    }

    @classmethod
    def get_vocabulary(cls, style: str) -> Dict:
        """
        Get melodic vocabulary for a style.

        Args:
            style: Style name (jazz, blues, classical, etc.)

        Returns:
            Dictionary with melodic vocabulary preferences
        """
        style_lower = style.lower()
        return cls.STYLE_VOCABULARIES.get(style_lower, cls.DEFAULT_VOCABULARY)

    @classmethod
    def get_interval_probabilities(cls, style: str) -> List[float]:
        """
        Get interval probability distribution for a style.

        Args:
            style: Style name

        Returns:
            List of 13 probabilities (unison through octave)
        """
        vocab = cls.get_vocabulary(style)
        weights = vocab['interval_weights']

        # Build probability list
        probs = []
        total = sum(weights.values())
        for i in range(13):
            prob = weights.get(i, 0.0) / total if total > 0 else 1/13
            probs.append(prob)

        return probs

    @classmethod
    def should_use_chromatic(cls, style: str) -> bool:
        """Check if a chromatic passing tone should be used"""
        vocab = cls.get_vocabulary(style)
        import random
        return random.random() < vocab['chromatic_tendency']

    @classmethod
    def get_blue_note_pcs(cls) -> List[int]:
        """Get pitch classes for blue notes (relative to root)"""
        return [3, 6, 10]  # b3, b5, b7

    @classmethod
    def choose_interval(cls, style: str, previous_intervals: List[int] = None) -> int:
        """
        Choose an interval based on style vocabulary.

        Args:
            style: Style name
            previous_intervals: Recent intervals (for avoiding repetition)

        Returns:
            Interval in semitones (can be negative for descending)
        """
        import random

        probs = cls.get_interval_probabilities(style)
        vocab = cls.get_vocabulary(style)

        # Zero out avoided intervals
        for avoid in vocab.get('avoid_intervals', []):
            if 0 <= avoid < len(probs):
                probs[avoid] = 0

        # Renormalize
        total = sum(probs)
        if total > 0:
            probs = [p / total for p in probs]
        else:
            probs = [1/13] * 13

        # Choose interval magnitude
        intervals = list(range(13))
        magnitude = random.choices(intervals, weights=probs)[0]

        # Choose direction (slight preference for continuing direction)
        if previous_intervals and len(previous_intervals) > 0:
            last_direction = 1 if previous_intervals[-1] >= 0 else -1
            # 60% chance to continue direction
            direction = last_direction if random.random() < 0.6 else -last_direction
        else:
            direction = random.choice([-1, 1])

        return magnitude * direction


class ProfileInterpolator:
    """
    Handles gradual profile transitions at phrase boundaries
    
    Features:
    - Pending profile changes wait for phrase boundary
    - Linear interpolation over configurable duration
    - Timeout fallback (15s) to prevent stuck states
    """
    
    def __init__(self, duration_seconds: float = 7.0, timeout_seconds: float = 15.0):
        self.current_profile = None
        self.target_profile = None
        self.pending_change = False
        self.pending_since = None
        self.interpolation_start_time = None
        self.duration = duration_seconds
        self.timeout = timeout_seconds
    
    def request_profile_change(self, new_profile: Dict[str, float]):
        """Mark profile change as pending until phrase boundary"""
        if self.current_profile is None:
            # First profile, apply immediately
            self.current_profile = new_profile.copy()
            return
        
        self.target_profile = new_profile.copy()
        self.pending_change = True
        self.pending_since = time.time()
    
    def begin_interpolation_at_phrase_boundary(self):
        """Start interpolation when phrase_detector signals end or timeout"""
        if self.pending_change:
            self.interpolation_start_time = time.time()
            self.pending_change = False
            self.pending_since = None
    
    def check_timeout(self, current_time: float) -> bool:
        """
        Check if pending change has exceeded timeout
        
        Returns:
            True if should force interpolation start
        """
        if self.pending_change and self.pending_since:
            elapsed = current_time - self.pending_since
            return elapsed > self.timeout
        return False
    
    def get_interpolated_profile(self, current_time: float) -> Optional[Dict[str, float]]:
        """
        Get current interpolated profile values
        
        Returns:
            Interpolated profile or current profile if not interpolating
        """
        if self.current_profile is None:
            return None
        
        # Not interpolating, return current
        if self.interpolation_start_time is None:
            return self.current_profile
        
        # Calculate interpolation progress
        elapsed = current_time - self.interpolation_start_time
        alpha = min(1.0, elapsed / self.duration)
        
        # Linear interpolation
        interpolated = {}
        for key in self.current_profile.keys():
            current_val = self.current_profile[key]
            target_val = self.target_profile[key]
            interpolated[key] = current_val * (1 - alpha) + target_val * alpha
        
        # Complete interpolation
        if alpha >= 1.0:
            self.current_profile = self.target_profile.copy()
            self.target_profile = None
            self.interpolation_start_time = None
        
        return interpolated
    
    def is_interpolating(self) -> bool:
        """Check if currently interpolating"""
        return self.interpolation_start_time is not None
    
    def has_pending_change(self) -> bool:
        """Check if change is pending phrase boundary"""
        return self.pending_change


class EpisodeState(Enum):
    """Episode states for hierarchical engagement"""
    ACTIVE = "active"      # AI generating phrases
    LISTENING = "listening"  # AI silent, listening to human


class EngagementEpisodeManager:
    """
    Manages hierarchical engagement episodes with intelligent interference
    
    Three temporal levels:
    1. Within-phrase timing (timing_precision, syncopation)
    2. Phrase-to-phrase gaps (existing pause logic)
    3. Episode groups (ACTIVE/LISTENING states with 30-120s durations)
    
    Features:
    - Context-aware transitions: monitors human RMS activity
    - Probabilistic interference: AI sometimes plays OVER human (not polite turn-taking)
    - Voice-specific durations: melodic 30-120s listening, bass 10-30s
    - Early exits: can exit ACTIVE early if human starts playing (context sensitivity)
    """
    
    def __init__(self, voice_type: str = "melodic", enable_episodes: bool = True):
        """
        Initialize episode manager
        
        Args:
            voice_type: 'melodic' or 'bass' (affects episode durations)
            enable_episodes: If False, bypass episode logic (timing-based only)
        """
        self.voice_type = voice_type
        self.enable_episodes = enable_episodes
        self.current_state = EpisodeState.ACTIVE  # Start actively
        self.episode_start_time = None  # Will be set on first should_generate_phrase() call
        
        # Human activity tracking
        self.human_rms_history = []  # Recent RMS values
        self.human_activity_window = 5.0  # seconds
        self.human_active_threshold = 0.01  # RMS threshold
        
        # Episode parameters (set by voice profile)
        self.active_duration_range = (3.0, 10.0)  # seconds
        self.listening_duration_range = (30.0, 120.0)  # seconds
        self.interference_probability = 0.4  # chance to play during human activity
        self.early_exit_probability = 0.15  # chance to exit ACTIVE early
        self.context_sensitivity = 0.6  # 0=ignore human, 1=highly responsive
        
        # Set initial duration from default ranges (AFTER ranges are defined)
        import random
        self.episode_duration = random.uniform(*self.active_duration_range)  # Start with ACTIVE duration
        
        # Log initialization
        print(f"🎭 Episode manager initialized: {voice_type} voice, state={self.current_state.value}, duration={self.episode_duration:.1f}s")
        
        # Logging
        self.last_transition_time = None  # Will be set on first call
        self.phrases_this_episode = 0
    
    def update_profile_parameters(self, profile: Dict[str, float]):
        """
        Update episode parameters from voice timing profile
        
        Args:
            profile: VoiceTimingProfile dict with episode parameters
        """
        if 'episode_active_duration_range' in profile:
            self.active_duration_range = profile['episode_active_duration_range']
        if 'episode_listening_duration_range' in profile:
            self.listening_duration_range = profile['episode_listening_duration_range']
        if 'interference_probability' in profile:
            self.interference_probability = profile['interference_probability']
        if 'early_exit_probability' in profile:
            self.early_exit_probability = profile['early_exit_probability']
        if 'context_sensitivity' in profile:
            self.context_sensitivity = profile['context_sensitivity']
    
    def update_human_activity(self, rms: float):
        """
        Track human RMS activity for context-aware decisions
        
        Args:
            rms: Current human RMS level
        """
        current_time = time.time()
        self.human_rms_history.append((current_time, rms))
        
        # Prune old history
        cutoff = current_time - self.human_activity_window
        self.human_rms_history = [(t, r) for t, r in self.human_rms_history if t > cutoff]
    
    def is_human_active(self) -> bool:
        """
        Check if human is currently playing
        
        Returns:
            True if recent RMS above threshold
        """
        if not self.human_rms_history:
            return False
        
        recent_rms = [rms for _, rms in self.human_rms_history[-10:]]  # Last 10 samples
        avg_rms = sum(recent_rms) / len(recent_rms)
        return avg_rms > self.human_active_threshold
    
    def should_generate_phrase(self) -> Tuple[bool, str]:
        """
        Decide whether to generate phrase based on episode state
        
        Returns:
            (should_generate, reasoning)
        """
        # If episodes disabled, always allow generation (timing-based only)
        if not self.enable_episodes:
            return (True, f"{self.voice_type} episodes disabled - timing-based only")
        
        current_time = time.time()
        
        # Initialize timer on first call
        if self.episode_start_time is None:
            self.episode_start_time = current_time
            self.last_transition_time = current_time
            print(f"⏱️  {self.voice_type.upper()} episode timer started")
        
        elapsed = current_time - self.episode_start_time
        human_active = self.is_human_active()
        
        if self.current_state == EpisodeState.LISTENING:
            # LISTENING: only generate if episode duration expired
            if elapsed >= self.episode_duration:
                # Transition to ACTIVE
                self._transition_to_active()
                return (True, f"Episode LISTENING→ACTIVE after {elapsed:.1f}s")
            else:
                return (False, f"LISTENING episode ({elapsed:.1f}/{self.episode_duration:.1f}s)")
        
        elif self.current_state == EpisodeState.ACTIVE:
            # ACTIVE: check for early exit or duration expiry
            
            # Early exit if human starts playing (context-aware)
            if human_active and random.random() < (self.early_exit_probability * self.context_sensitivity):
                self._transition_to_listening()
                return (False, f"Early exit ACTIVE→LISTENING (human started playing)")
            
            # Episode duration expired → transition to LISTENING
            if elapsed >= self.episode_duration:
                self._transition_to_listening()
                return (False, f"Episode ACTIVE→LISTENING after {elapsed:.1f}s")
            
            # ACTIVE episode behavior depends on voice type
            if self.voice_type == "bass":
                # BASS: Simple, steady foundation - rarely yields to human
                # Only check interference probability as a small courtesy to give human space
                if human_active and random.random() > self.interference_probability * 3:
                    # Very low chance to yield (bass interference is 15-25%, so 45-75% chance to yield)
                    return (False, f"ACTIVE but briefly yielding to human")
                # Bass mostly just plays steadily
                return (True, f"ACTIVE episode ({elapsed:.1f}/{self.episode_duration:.1f}s)")
            
            else:  # melodic
                # MELODY: More thoughtful engagement with interference logic
                if human_active:
                    # Roll for yielding (opposite of interference)
                    if random.random() > self.interference_probability:
                        # Yield to human this time (don't interfere)
                        return (False, f"ACTIVE but yielding to human (interference roll failed)")
                    # Otherwise, allow interference (play over human)
                
                # Default ACTIVE: generate phrase
                interference_note = " + INTERFERENCE" if human_active else ""
                return (True, f"ACTIVE episode ({elapsed:.1f}/{self.episode_duration:.1f}s){interference_note}")
        
        # Fallback
        return (False, "Unknown episode state")
    
    def _transition_to_active(self):
        """Transition from LISTENING to ACTIVE"""
        self.current_state = EpisodeState.ACTIVE
        self.episode_start_time = time.time()
        self.episode_duration = random.uniform(*self.active_duration_range)
        self.phrases_this_episode = 0
        self.last_transition_time = time.time()
        print(f"🎬 {self.voice_type.upper()} LISTENING→ACTIVE (duration={self.episode_duration:.1f}s)")
    
    def _transition_to_listening(self):
        """Transition from ACTIVE to LISTENING"""
        self.current_state = EpisodeState.LISTENING
        self.episode_start_time = time.time()
        self.episode_duration = random.uniform(*self.listening_duration_range)
        self.last_transition_time = time.time()
        print(f"🔇 {self.voice_type.upper()} ACTIVE→LISTENING (duration={self.episode_duration:.1f}s, phrases={self.phrases_this_episode})")
    
    def track_phrase_generation(self):
        """Track that a phrase was generated (for logging)"""
        self.phrases_this_episode += 1
    
    def get_episode_aware_gap(self, base_min: float, base_max: float) -> float:
        """
        Get gap duration aware of episode state
        
        During LISTENING episodes, return longer gaps
        During ACTIVE episodes, return base gaps
        
        Args:
            base_min: Base minimum gap (seconds)
            base_max: Base maximum gap (seconds)
            
        Returns:
            Gap duration (seconds)
        """
        if self.current_state == EpisodeState.LISTENING:
            # Already in LISTENING, return remaining duration
            elapsed = time.time() - self.episode_start_time
            remaining = max(0, self.episode_duration - elapsed)
            return remaining if remaining > 0 else random.uniform(base_min, base_max)
        else:
            # ACTIVE: use base gaps
            return random.uniform(base_min, base_max)
    
    def get_status(self) -> Dict:
        """Get current episode status for logging"""
        # Handle case when episodes disabled or not yet initialized
        if self.episode_start_time is None:
            return {
                'state': 'disabled' if not self.enable_episodes else 'not_started',
                'elapsed': 0,
                'duration': 0,
                'phrases_this_episode': self.phrases_this_episode,
                'human_active': self.is_human_active(),
            }
        
        elapsed = time.time() - self.episode_start_time
        return {
            'state': self.current_state.value,
            'elapsed': elapsed,
            'duration': self.episode_duration,
            'phrases_this_episode': self.phrases_this_episode,
            'human_active': self.is_human_active(),
        }


# IRCAM Phase 3: Behavior Controller
class BehaviorController:
    """
    Manages IRCAM-style interaction modes
    Based on IRCAM 2023 research best practices for mixed-initiative systems

    Features:
    - Manual mode selection
    - CLAP-based automatic style detection and mode selection
    - Smooth mode transitions
    """

    def __init__(self, enable_clap: bool = False, clap_config: Optional[Dict] = None):
        """
        Initialize behavior controller

        Args:
            enable_clap: Enable CLAP style detection for auto mode selection
            clap_config: Configuration dict for CLAP (from config.yaml)
        """
        self.mode = BehaviorMode.MIRROR  # Default to phrase-aware mode

        # CLAP style detection (optional)
        self.enable_clap = enable_clap
        self.style_detector = None
        self.last_style_detection = None
        self.last_detection_time = 0

        # Voice timing profile interpolators (one per voice type)
        interpolation_config = clap_config.get('interpolation', {}) if clap_config else {}
        duration = interpolation_config.get('duration_seconds', 7.0)
        timeout = interpolation_config.get('max_pending_seconds', 15.0)
        
        self.melodic_interpolator = ProfileInterpolator(duration, timeout)
        self.bass_interpolator = ProfileInterpolator(duration, timeout)
        self.wait_for_phrase_boundary = interpolation_config.get('wait_for_phrase_boundary', True)

        if enable_clap and clap_config:
            try:
                from listener.clap_style_detector import CLAPStyleDetector

                model_name = clap_config.get('model', 'laion/clap-htsat-unfused')
                use_gpu = clap_config.get('use_gpu', True)
                confidence_threshold = clap_config.get('confidence_threshold', 0.3)

                self.style_detector = CLAPStyleDetector(
                    model_name=model_name,
                    use_gpu=use_gpu,
                    confidence_threshold=confidence_threshold
                )

                self.clap_update_interval = clap_config.get('update_interval', 5.0)
                self.auto_behavior_mode = clap_config.get('auto_behavior_mode', True)

                print(f"🎨 CLAP style detection enabled")
                print(f"   Auto-mode switching: {self.auto_behavior_mode}")

            except Exception as e:
                print(f"⚠️ Failed to initialize CLAP: {e}")
                print("   Continuing with manual mode selection")
                self.enable_clap = False

        # Mode-specific parameters (EXAGGERATED for clarity)
        self.mode_params = {
            BehaviorMode.SHADOW: {
                'similarity_threshold': 0.95,  # EXTREME imitation
                'request_weight': 0.98,        # Almost slavish following
                'temperature': 0.2,            # ULTRA LOW = highly predictable (reduced from 0.3)
                'phrase_variation': 0.02,      # Minimal variation (reduced from 0.05)
                'response_delay': 0.05,        # ULTRA IMMEDIATE (reduced from 0.1)
                'volume_factor': 0.5,          # Much quieter (reduced from 0.6 for more dramatic shadow)
                'note_density': 1.8,           # Dense chatter (increased from 1.5)
                'pitch_offset': 0              # Same register
            },
            BehaviorMode.MIRROR: {
                'similarity_threshold': 0.7,   # Recognizable but varied
                'request_weight': 0.7,         # Balanced following
                'temperature': 1.0,            # MEDIUM = balanced
                'phrase_variation': 0.4,       # Moderate variation
                'response_delay': 'phrase_aware',  # Wait for pauses
                'volume_factor': 0.85,         # Slightly quieter
                'note_density': 1.0,           # Same density
                'pitch_offset': 0              # Same register
            },
            BehaviorMode.COUPLE: {
                'similarity_threshold': 0.05,  # EXTREMELY independent
                'request_weight': 0.1,         # Almost independent (reduced from 0.2)
                'temperature': 2.0,            # ULTRA HIGH = wild exploration (increased from 1.8)
                'phrase_variation': 0.95,      # Maximum variation (increased from 0.9)
                'response_delay': 4.0,         # VERY LONG delay (increased from 3.0)
                'volume_factor': 1.3,          # LOUDER (increased from 1.2)
                'note_density': 0.5,           # Sparser (reduced from 0.6 for more space)
                'pitch_offset': 12             # Octave higher (distinct voice)
            },
            # Keep original modes for compatibility
            BehaviorMode.IMITATE: {
                'similarity_threshold': 0.85,
                'request_weight': 0.85,
                'temperature': 0.6,
                'phrase_variation': 0.15,
                'response_delay': 0.2,
                'volume_factor': 0.7,
                'note_density': 1.0,
                'pitch_offset': 0
            },
            BehaviorMode.CONTRAST: {
                'similarity_threshold': 0.1,   # Very independent (reduced from 0.15)
                'request_weight': 0.2,         # Low following (reduced from 0.3)
                'temperature': 1.6,            # High exploration (increased from 1.5)
                'phrase_variation': 0.9,       # Maximum variation (increased from 0.85)
                'response_delay': 2.0,         # Longer delay (increased from 1.5)
                'volume_factor': 1.1,          # Louder (increased from 1.0)
                'note_density': 0.6,           # Sparser (reduced from 0.7)
                'pitch_offset': -12            # Octave lower (bass counterpoint)
            },
            BehaviorMode.LEAD: {
                'similarity_threshold': 0.5,
                'request_weight': 0.6,
                'temperature': 1.1,
                'phrase_variation': 0.5,
                'response_delay': 0.0,         # NO delay (anticipatory)
                'volume_factor': 1.1,          # Slightly louder (leading)
                'note_density': 0.8,
                'pitch_offset': 7              # Perfect 5th above (harmonic lead)
            }
        }
    
    def set_mode(self, mode: BehaviorMode):
        """Change interaction mode"""
        self.mode = mode
        print(f"🎭 Behavior mode: {mode.value}")

    def auto_select_mode(self,
                         audio: np.ndarray,
                         sr: int = 44100,
                         current_time: float = None) -> Optional[Dict]:
        """
        Automatically select behavioral mode based on detected musical style using CLAP

        Args:
            audio: Audio buffer for style detection
            sr: Sample rate
            current_time: Current timestamp (for throttling detection)

        Returns:
            Dict with style info if detection succeeded, None otherwise
        """
        if not self.enable_clap or not self.style_detector:
            return None

        # Throttle detection (don't detect too frequently)
        if current_time is None:
            current_time = time.time()

        time_since_last = current_time - self.last_detection_time

        if time_since_last < self.clap_update_interval:
            # Too soon, return cached result
            return self.last_style_detection

        # Detect style using CLAP
        style_result = self.style_detector.detect_style(audio, sr)

        if style_result is None:
            # Detection failed or confidence too low
            return None

        # Update cached result
        self.last_style_detection = {
            'style': style_result.style_label,
            'confidence': style_result.confidence,
            'recommended_mode': style_result.recommended_mode,
            'secondary_styles': style_result.secondary_styles,
            'timestamp': current_time
        }
        self.last_detection_time = current_time

        # Auto-switch mode if enabled
        if self.auto_behavior_mode:
            recommended_mode = style_result.recommended_mode

            if recommended_mode != self.mode:
                print(f"\n🎨 Style detected: {style_result.style_label} "
                      f"(confidence: {style_result.confidence:.2f})")
                print(f"🎭 Auto-switching: {self.mode.value} → {recommended_mode.value}")

                self.set_mode(recommended_mode)
            else:
                print(f"🎨 Style: {style_result.style_label} "
                      f"(confidence: {style_result.confidence:.2f}) "
                      f"- mode unchanged")

        else:
            # Just suggest, don't auto-switch
            print(f"🎨 Style detected: {style_result.style_label} "
                  f"(confidence: {style_result.confidence:.2f})")
            print(f"💡 Suggested mode: {style_result.recommended_mode.value} "
                  f"(current: {self.mode.value})")

        return self.last_style_detection

    def get_similarity_threshold(self) -> float:
        """Get pattern matching threshold for current mode"""
        return self.mode_params[self.mode]['similarity_threshold']
    
    def get_note_density(self) -> float:
        """Get note density multiplier for current mode"""
        return self.mode_params[self.mode].get('note_density', 1.0)
    
    def get_pitch_offset(self) -> int:
        """Get pitch offset in semitones for current mode"""
        return self.mode_params[self.mode].get('pitch_offset', 0)
    
    def get_response_delay(self, phrase_detector) -> Optional[float]:
        """
        Calculate response delay based on mode and phrase state
        
        Args:
            phrase_detector: PhraseBoundaryDetector instance
            
        Returns:
            Delay in seconds, or None if should not respond yet
        """
        delay = self.mode_params[self.mode]['response_delay']
        
        if delay == 'phrase_aware':
            # Wait for phrase boundary in MIRROR mode
            if phrase_detector.is_in_phrase(time.time()):
                return None  # Don't respond yet
            else:
                return 0.1  # Respond quickly at boundary
        else:
            return delay
    
    def get_volume_factor(self) -> float:
        """Get volume scaling for current mode"""
        return self.mode_params[self.mode]['volume_factor']
    
    def get_voice_timing_profile(
        self,
        voice_type: str,
        phrase_detector=None,
        current_time: float = None
    ) -> Dict[str, float]:
        """
        Get timing profile for voice type with smooth interpolation
        
        Args:
            voice_type: 'melodic' or 'bass'
            phrase_detector: Optional phrase detector for boundary awareness
            current_time: Current time for interpolation
            
        Returns:
            Timing profile dict (timing_precision, rhythmic_density, etc.)
        """
        if current_time is None:
            current_time = time.time()
        
        # Select interpolator for this voice
        interpolator = (
            self.melodic_interpolator if voice_type == 'melodic' 
            else self.bass_interpolator
        )
        
        # Get current style from CLAP detection
        style = None
        if self.last_style_detection:
            style = self.last_style_detection.get('style')
        
        if style:
            # Get target profile from CLAP style
            target_profile = VoiceTimingProfile.get_profile(style, voice_type)
            
            # Check if profile needs updating
            current = interpolator.get_interpolated_profile(current_time)
            if current is None or current != target_profile:
                # Request profile change
                interpolator.request_profile_change(target_profile)
                
                # Check if we should start interpolating
                should_interpolate = False
                
                # Check timeout first (always honor timeout)
                if interpolator.check_timeout(current_time):
                    should_interpolate = True
                # Check phrase boundary if enabled
                elif self.wait_for_phrase_boundary and phrase_detector:
                    if phrase_detector.is_phrase_ending():
                        should_interpolate = True
                # If phrase boundary waiting disabled, start immediately
                elif not self.wait_for_phrase_boundary:
                    should_interpolate = True
                
                if should_interpolate:
                    interpolator.begin_interpolation_at_phrase_boundary()
            
            # Return interpolated profile
            return interpolator.get_interpolated_profile(current_time)
        else:
            # No style detected, use default
            return VoiceTimingProfile.get_profile('ballad', voice_type)
    
    def filter_pattern_matches(self, matches: List, similarity_scores: List[float]) -> List:
        """
        Filter AudioOracle matches based on mode similarity threshold
        
        Args:
            matches: List of pattern matches
            similarity_scores: Corresponding similarity scores
            
        Returns:
            Filtered list of matches
        """
        threshold = self.get_similarity_threshold()
        filtered = []
        for match, score in zip(matches, similarity_scores):
            if score >= threshold:
                filtered.append(match)
        return filtered


class BehaviorEngine:
    """
    Core behavior engine for AI agent
    Implements imitate, contrast, and lead behaviors
    """
    
    def __init__(self, rhythm_oracle=None, visualization_manager=None, config: Optional[Dict] = None):
        # Randomly select initial mode for variety
        self.current_mode = random.choice([
            BehaviorMode.IMITATE,
            BehaviorMode.CONTRAST,
            BehaviorMode.LEAD,
            BehaviorMode.SHADOW,
            BehaviorMode.MIRROR,
            BehaviorMode.COUPLE
        ])
        self.mode_history = []
        self.decision_history = []
        self.visualization_manager = visualization_manager

        # Phrase generation
        self.phrase_generator = PhraseGenerator(rhythm_oracle, visualization_manager=visualization_manager) if rhythm_oracle else None
        self.active_phrases = {}  # Store active phrases by voice_type

        # CLAP style detection (optional, Phase 2.x)
        self.enable_clap = False
        self.style_detector = None
        self.last_style_detection = None
        self.last_detection_time = 0
        self.clap_update_interval = 5.0  # Re-detect style every N seconds
        self.auto_behavior_mode = False

        if config:
            style_config = config.get('style_detection', {})
            self.enable_clap = style_config.get('enabled', False)
            self.auto_behavior_mode = style_config.get('auto_behavior_mode', False)
            self.clap_update_interval = style_config.get('update_interval', 5.0)

            if self.enable_clap:
                try:
                    from listener.clap_style_detector import CLAPStyleDetector

                    model_name = style_config.get('model', 'laion/clap-htsat-unfused')
                    use_gpu = style_config.get('use_gpu', True)
                    confidence_threshold = style_config.get('confidence_threshold', 0.3)

                    self.style_detector = CLAPStyleDetector(
                        model_name=model_name,
                        use_gpu=use_gpu,
                        confidence_threshold=confidence_threshold
                    )

                    print(f"🎨 CLAP style detection enabled in BehaviorEngine")
                    if self.auto_behavior_mode:
                        print(f"   Auto-mode selection: ON")

                except Exception as e:
                    print(f"⚠️  Failed to initialize CLAP: {e}")
                    print(f"   Continuing without style detection")
                    self.enable_clap = False
        
        # Behavior parameters
        self.imitate_strength = 0.8
        self.contrast_strength = 0.6
        self.lead_strength = 0.7
        
        # Mode-specific parameters (EXAGGERATED for clarity)
        self.mode_params = {
            BehaviorMode.SHADOW: {
                'similarity_threshold': 0.95,
                'request_weight': 0.98,
                'temperature': 0.3,
                'phrase_variation': 0.05,
                'response_delay': 0.1,
                'volume_factor': 0.6,
                'note_density': 1.5,
                'pitch_offset': 0
            },
            BehaviorMode.MIRROR: {
                'similarity_threshold': 0.7,
                'request_weight': 0.7,
                'temperature': 1.0,
                'phrase_variation': 0.4,
                'response_delay': 'phrase_aware',
                'volume_factor': 0.85,
                'note_density': 1.0,
                'pitch_offset': 0
            },
            BehaviorMode.COUPLE: {
                'similarity_threshold': 0.05,
                'request_weight': 0.2,
                'temperature': 1.8,
                'phrase_variation': 0.9,
                'response_delay': 3.0,
                'volume_factor': 1.2,
                'note_density': 0.6,
                'pitch_offset': 12
            },
            BehaviorMode.IMITATE: {
                'similarity_threshold': 0.85,
                'request_weight': 0.85,
                'temperature': 0.6,
                'phrase_variation': 0.15,
                'response_delay': 0.2,
                'volume_factor': 0.7,
                'note_density': 1.0,
                'pitch_offset': 0
            },
            BehaviorMode.CONTRAST: {
                'similarity_threshold': 0.15,
                'request_weight': 0.3,
                'temperature': 1.5,
                'phrase_variation': 0.85,
                'response_delay': 1.5,
                'volume_factor': 1.0,
                'note_density': 0.7,
                'pitch_offset': -12
            },
            BehaviorMode.LEAD: {
                'similarity_threshold': 0.5,
                'request_weight': 0.6,
                'temperature': 1.1,
                'phrase_variation': 0.5,
                'response_delay': 0.0,
                'volume_factor': 1.1,
                'note_density': 0.8,
                'pitch_offset': 7
            }
        }
        
        # Timing controls - STICKY MODES for sustained character
        # Reduced from 30-90s to 15-45s for more dynamic variety (user feedback)
        self.min_mode_duration = 15.0  # seconds (shorter for more transitions)
        self.max_mode_duration = 45.0  # seconds (prevent single-mode dominance)
        self.mode_start_time = time.time()
        self.current_mode_duration = random.uniform(15.0, 45.0)  # Target duration for current mode
        
        # Decision thresholds
        self.confidence_threshold = 0.5
        self.adaptation_rate = 0.1
        
        # Voice timing separation
        # Initialize to past time so first phrase can generate immediately
        self.last_melody_time = time.time() - 20.0  # 20 seconds ago (exceeds max pause)
        self.last_bass_time = time.time() - 15.0    # 15 seconds ago (exceeds max pause)
        self.min_voice_separation = 0.2  # Minimum 200ms between melody and bass
        
        # Hierarchical engagement episode managers (LEVEL 3)
        # Melody: enable episodes for contemplative behavior
        self.melody_episode_manager = EngagementEpisodeManager(
            voice_type="melodic",
            enable_episodes=True  # Melody uses episodes
        )
        # Bass: disable episodes for steady foundation
        self.bass_episode_manager = EngagementEpisodeManager(
            voice_type="bass",
            enable_episodes=False  # Bass bypasses episodes - timing-based only
        )
        
        # Variable pause durations (balanced for responsiveness without click noise)
        self.melody_pause_min = 6.0   # seconds (increased from 10.0 for more responsiveness)
        self.melody_pause_max = 20.0  # seconds (reduced from 40.0 for more activity)
        self.bass_pause_min = 4     # seconds (increased from 4.0 for more responsiveness)
        self.bass_pause_max = 12.0     # seconds (reduced from 15.0 for more activity)
        self.current_melody_pause = 6.0   # Start with minimum
        self.current_bass_pause = 2.5     # Start with minimum
        
        # Emit initial mode to visualization (so viewport shows starting mode immediately)
        if self.visualization_manager:
            self.visualization_manager.emit_mode_change(
                mode=self.current_mode.value,
                duration=self.current_mode_duration,
                request_params={},
                temperature=0.8
            )
            print(f"🎭 Initial mode: {self.current_mode.value.upper()} (will persist for {self.current_mode_duration:.0f}s)")
        
    def set_pause_ranges_from_gpt_oss(self, gpt_oss_data: Dict):
        """Set pause ranges from GPT-OSS training analysis"""
        if not gpt_oss_data:
            return
        
        # Extract pause information if available
        if 'silence_strategy' in gpt_oss_data:
            silence = gpt_oss_data['silence_strategy']
            
            # Use silence metrics to inform pause ranges
            if 'avg_silence_duration' in silence:
                avg_silence = silence['avg_silence_duration']
                # Set melody pause based on average silence in training
                self.melody_pause_min = max(10.0, avg_silence * 0.5)
                self.melody_pause_max = max(20.0, avg_silence * 2.0)
                print(f"🎵 GPT-OSS: Melody pause range set to {self.melody_pause_min:.1f}-{self.melody_pause_max:.1f}s")
            
            if 'silence_percentage' in silence:
                silence_pct = silence['silence_percentage']
                # More silence in training = longer pauses
                if silence_pct > 50:
                    self.melody_pause_min = 15.0
                    self.melody_pause_max = 40.0
                    self.bass_pause_min = 6.0
                    self.bass_pause_max = 20.0
                    print(f"🎵 GPT-OSS: High silence detected, using longer pauses")
    
    def decide_behavior(self, current_event: Dict, 
                       memory_buffer, clustering, activity_multiplier: float = 1.0,
                       arc_context: Optional[Dict] = None) -> List[MusicalDecision]:
        """Make musical decisions based on current context
        
        Args:
            activity_multiplier: Performance arc activity level (0.0-1.0) from timeline manager
            arc_context: Performance arc context (phase, engagement_level, etc.)
        """
        current_time = time.time()
        
        # Store last event for drum-triggered evolution
        self._last_event = current_event
        
        # Check if we should change mode (with arc context for dynamic durations)
        if self._should_change_mode(current_time, arc_context):
            self._select_new_mode(current_event, memory_buffer, clustering, arc_context)
        
        # Generate decisions based on current mode (melodic and bass) with activity multiplier
        decisions = self._generate_decision(
            self.current_mode, current_event, memory_buffer, clustering, activity_multiplier
        )
        
        # Store decisions
        for decision in decisions:
            self.decision_history.append(decision)
        if len(self.decision_history) > 100:
            self.decision_history.pop(0)
        
        return decisions
    
    def continue_phrase_iteration(self) -> List[MusicalDecision]:
        """Continue active phrase iterations - called periodically to play next notes in phrases"""
        decisions = []
        current_time = time.time()
        
        for phrase_id, phrase_data in list(self.active_phrases.items()):
            phrase = phrase_data['phrase']
            note_index = phrase_data['note_index']
            start_time = phrase_data['start_time']
            mode = phrase_data['mode']
            voice_type = phrase_data['voice_type']
            
            # Check if it's time for the next note (use phrase timings)
            elapsed_time = current_time - start_time
            # Sum up all timings up to and including the next note
            expected_time = sum(phrase.timings[:note_index + 1])
            
            if elapsed_time >= expected_time and note_index < len(phrase.notes) - 1:
                # Move to next note
                note_index += 1
                phrase_data['note_index'] = note_index
                
                # Schedule next note with mode-specific modifications
                note = phrase.notes[note_index]
                
                # Apply note density: randomly skip notes in sparse modes (COUPLE, CONTRAST)
                note_density = 1.0  # Default
                pitch_offset = 0    # Default
                if hasattr(self, 'behavior_controller'):
                    note_density = self.behavior_controller.get_note_density()
                    pitch_offset = self.behavior_controller.get_pitch_offset()
                
                # Skip note if density check fails (makes sparse modes feel sparse)
                if random.random() > note_density:
                    continue  # Skip this note
                
                # Apply pitch offset (transpose register: COUPLE +12, CONTRAST -12, LEAD +7)
                modified_note = int(note) + pitch_offset
                modified_note = max(21, min(108, modified_note))  # Keep within MIDI range
                
                musical_params = {
                    'target_note': modified_note,
                    'velocity': phrase.velocities[note_index],
                    'duration': phrase.durations[note_index],
                    'timing': phrase.timings[note_index],
                    'phrase_id': phrase.phrase_id,
                    'phrase_type': phrase.phrase_type,
                    'phrase_scheduler': phrase_id
                }
                
                decision = MusicalDecision(
                    mode=mode,
                    confidence=phrase.confidence,
                    target_features=np.array([note]),
                    musical_params=musical_params,
                    timestamp=current_time,
                    reasoning=f"Phrase continuation: {phrase.phrase_type} note {note_index + 1}/{len(phrase.notes)}",
                    voice_type=voice_type,
                    instrument=voice_type
                )
                
                # Apply CLAP-driven timing profile
                decision = self._apply_timing_profile(decision)
                decisions.append(decision)
                
                # Phrase continuation (silent)
            
            # Remove phrase if completed
            if note_index >= len(phrase.notes) - 1:
                del self.active_phrases[phrase_id]
                # Phrase completed (silent)
        
        return decisions
    
    def _should_change_mode(self, current_time: float, arc_context: Optional[Dict] = None) -> bool:
        """Determine if we should change behavior mode
        
        Args:
            current_time: Current timestamp
            arc_context: Performance arc context with phase info
        """
        mode_duration = current_time - self.mode_start_time
        
        # Get dynamic duration thresholds based on arc phase
        min_duration = self.min_mode_duration
        max_duration = self.max_mode_duration
        
        if arc_context:
            performance_phase = arc_context.get('performance_phase', 'main')
            
            # Adjust duration based on arc phase
            if performance_phase in ['buildup', 'ending']:
                # Opening/Closing: Longer modes for stability
                min_duration = self.min_mode_duration * 1.5  # 15s → 22.5s
                max_duration = self.max_mode_duration * 1.3  # 45s → 58.5s
                
            elif performance_phase == 'main':
                # Check if we're in peak energy
                engagement = arc_context.get('engagement_level', 0.5)
                if engagement > 0.7:
                    # Peak: Shorter modes for variety
                    min_duration = self.min_mode_duration * 0.7  # 15s → 10.5s
                    max_duration = self.max_mode_duration * 0.7  # 45s → 31.5s
        
        # Always change after max duration
        if mode_duration >= max_duration:
            return True
        
        # Check for drum-triggered evolution
        if hasattr(self, '_last_event'):
            instrument = self._last_event.get('instrument', 'unknown')
            if instrument == 'drums':
                # Much faster evolution for drums (0.5s minimum)
                if mode_duration >= 0.5:
                    return random.random() < 0.6  # 60% chance with drums
                return random.random() < 0.3  # Still some chance even early
        
        # Increased transition probability after min duration (from 30% to 50%)
        # This creates more dynamic mode changes while maintaining coherence
        if mode_duration >= min_duration:
            # Progress-based probability (increases over time)
            progress = (mode_duration - min_duration) / (max_duration - min_duration)
            transition_prob = 0.5 + (0.3 * progress)  # 50% → 80% over duration range
            return random.random() < transition_prob
        
        return False
    
    def _select_new_mode(self, current_event: Dict, 
                        memory_buffer, clustering, arc_context: Optional[Dict] = None):
        """Select a new behavior mode based on context and arc phase"""
        mode_change_start = time.time()
        
        # STICKY MODES: Check if we should persist with current mode
        elapsed = time.time() - self.mode_start_time
        
        # Only switch modes if duration exceeded
        if elapsed < self.current_mode_duration:
            return  # Stay in current mode (sticky behavior)
        
        # Time to switch - weighted selection based on context
        modes = list(BehaviorMode)
        weights = self._calculate_mode_weights(current_event, memory_buffer, clustering, arc_context)
        
        # Select new mode based on weights
        new_mode = random.choices(modes, weights=weights)[0]
        self.current_mode = new_mode
        self.mode_start_time = time.time()
        
        # Dynamic mode duration based on arc phase
        min_dur = self.min_mode_duration
        max_dur = self.max_mode_duration
        
        if arc_context:
            performance_phase = arc_context.get('performance_phase', 'main')
            engagement = arc_context.get('engagement_level', 0.5)
            
            if performance_phase in ['buildup', 'ending']:
                # Opening/Closing: Longer modes
                min_dur *= 1.5
                max_dur *= 1.3
            elif performance_phase == 'main' and engagement > 0.7:
                # Peak: Shorter modes
                min_dur *= 0.7
                max_dur *= 0.7
        
        self.current_mode_duration = random.uniform(min_dur, max_dur)
        
        # Emit visualization event for mode change
        if self.visualization_manager:
            # Get actual temperature from mode parameters
            current_params = self.mode_params[self.current_mode]
            self.visualization_manager.emit_mode_change(
                mode=self.current_mode.value,
                duration=self.current_mode_duration,
                request_params=current_params,  # Send full mode params (includes temperature, weights, etc.)
                temperature=current_params['temperature']
            )
            # Also emit timeline event
            self.visualization_manager.emit_timeline_update('mode_change', mode=self.current_mode.value)
        
        # Store mode change with instrument context
        instrument = current_event.get('instrument', 'unknown')
        self.mode_history.append({
            'mode': self.current_mode,
            'timestamp': time.time(),
            'reason': 'scheduled_change',
            'instrument': instrument
        })
        
        mode_change_duration = time.time() - mode_change_start
        # Print mode change for visibility (with duration and timing)
        print(f"🎭 Mode shift: {new_mode.value.upper()} (will persist for {self.current_mode_duration:.0f}s) [switch took {mode_change_duration*1000:.1f}ms]")
    
    def _calculate_mode_weights(self, current_event: Dict, 
                              memory_buffer, clustering, arc_context: Optional[Dict] = None) -> List[float]:
        """Calculate weights for mode selection based on context and arc phase
        
        Args:
            arc_context: Performance arc context with phase and engagement level
            
        Returns:
            Weights for [IMITATE, CONTRAST, LEAD, SHADOW, MIRROR, COUPLE]
        """
        # BASE WEIGHTS (rebalanced from previous 0.4 shadow dominance)
        # Order: [IMITATE, CONTRAST, LEAD, SHADOW, MIRROR, COUPLE]
        base_weights = [0.05, 0.30, 0.30, 0.20, 0.10, 0.05]
        
        # === ACTIVITY-BASED ADJUSTMENTS (light touch when arc is active) ===
        recent_moments = memory_buffer.get_recent_moments(10.0)
        
        # Reduce influence of activity adjustments if arc context is provided
        activity_influence = 0.5 if arc_context else 1.0
        
        if len(recent_moments) > 5:
            # High activity -> more MIRROR/COUPLE (phrase-aware, independent)
            base_weights[4] += 0.10 * activity_influence  # MIRROR
            base_weights[5] += 0.10 * activity_influence  # COUPLE
            base_weights[3] -= 0.15 * activity_influence  # SHADOW
        elif len(recent_moments) < 2:
            # Low activity -> more SHADOW/IMITATE (follow closely)
            base_weights[0] += 0.10 * activity_influence  # IMITATE
            base_weights[3] += 0.15 * activity_influence  # SHADOW
            base_weights[2] -= 0.15 * activity_influence  # LEAD
        
        # === ONSET-BASED ADJUSTMENTS (reduced when arc is active) ===
        onset_count = sum(1 for moment in recent_moments 
                         if moment.event_data.get('onset', False))
        
        if onset_count > 3:
            # High onset activity -> more LEAD/COUPLE (take initiative)
            base_weights[2] += 0.15 * activity_influence  # LEAD
            base_weights[5] += 0.10 * activity_influence  # COUPLE
            base_weights[3] -= 0.20 * activity_influence  # SHADOW
        
        # === DRUM-SPECIFIC EVOLUTION ===
        instrument = current_event.get('instrument', 'unknown')
        if instrument == 'drums':
            # Drums: prefer LEAD, COUPLE, CONTRAST
            base_weights = [0.05, 0.25, 0.30, 0.10, 0.15, 0.15]
            print(f"🥁 Drum evolution: weights={base_weights}")
        
        # === ARC-AWARE ADJUSTMENTS (APPLIED LAST - highest priority) ===
        if arc_context:
            performance_phase = arc_context.get('performance_phase', 'main')
            engagement = arc_context.get('engagement_level', 0.5)
            
            if performance_phase == 'buildup':
                # Opening: Learn and listen (imitate/shadow) - STRONG adjustments
                base_weights[0] += 0.25  # IMITATE (increased from 0.15)
                base_weights[3] += 0.30  # SHADOW (increased from 0.20)
                base_weights[2] -= 0.35  # LEAD (stronger reduction)
                base_weights[5] -= 0.15  # COUPLE
                base_weights[1] -= 0.10  # CONTRAST
                
            elif performance_phase == 'ending':
                # Closing: Gentle shadowing, fade out
                base_weights[3] += 0.35  # SHADOW (increased from 0.25)
                base_weights[0] += 0.15  # IMITATE
                base_weights[2] -= 0.30  # LEAD (stronger reduction)
                base_weights[5] -= 0.20  # COUPLE
                
            elif performance_phase == 'main':
                # Main phase: Adjust based on engagement
                if engagement > 0.7:
                    # Peak energy: Maximum variety (lead/couple)
                    base_weights[2] += 0.25  # LEAD (increased from 0.20)
                    base_weights[5] += 0.20  # COUPLE (increased from 0.15)
                    base_weights[3] -= 0.30  # SHADOW
                    base_weights[0] -= 0.10  # IMITATE
                elif engagement < 0.3:
                    # Low engagement: Be supportive
                    base_weights[3] += 0.20  # SHADOW
                    base_weights[0] += 0.15  # IMITATE
                    base_weights[2] -= 0.20  # LEAD
                    base_weights[5] -= 0.15  # COUPLE
        if instrument == 'drums':
            # Drums: prefer LEAD, COUPLE, CONTRAST
            base_weights = [0.05, 0.25, 0.30, 0.10, 0.15, 0.15]
            print(f"🥁 Drum evolution: weights={base_weights}")
        
        # === AVOID IMMEDIATE REPETITION ===
        current_mode_index = list(BehaviorMode).index(self.current_mode)
        base_weights[current_mode_index] *= 0.5  # Halve probability of staying
        
        # === NORMALIZE ===
        # Ensure non-negative and normalize to sum to 1.0
        base_weights = [max(0.01, w) for w in base_weights]  # Floor at 0.01
        total = sum(base_weights)
        normalized_weights = [w / total for w in base_weights]
        
        return normalized_weights
    
    def _generate_decision(self, mode: BehaviorMode, current_event: Dict,
                          memory_buffer, clustering, activity_multiplier: float = 1.0) -> List[MusicalDecision]:
        """Generate musical phrase decisions for melody and bass partnership with rhythmic awareness
        
        Args:
            activity_multiplier: Performance arc activity level (0.0-1.0)
                - 0.3-1.0 during buildup phase (sparse → full)
                - 1.0 during main phase (full activity)
                - 1.0-0.0 during ending phase (fade to silence)
        """
        decisions = []
        instrument = current_event.get('instrument', 'unknown')
        current_time = time.time()
        
        # Update phrase generator with AudioOracle if available
        if self.phrase_generator and clustering:
            self.phrase_generator.audio_oracle = clustering
        
        # If we have phrase generation, use it for musical phrases
        if self.phrase_generator:
            # Generating phrase (silent)
            # Extract harmonic context
            harmonic_context = current_event.get('harmonic_context', {})
            
            # Update episode managers with current human activity (RMS from event)
            human_rms = current_event.get('rms', 0.0)
            self.melody_episode_manager.update_human_activity(human_rms)
            self.bass_episode_manager.update_human_activity(human_rms)
            
            # Decide which voice to play based on timing, context, AND episode states
            time_since_melody = current_time - self.last_melody_time
            time_since_bass = current_time - self.last_bass_time
            
            voice_type = None
            # Use variable pause durations (randomized within range each time)
            import random
            required_melody_gap = random.uniform(self.melody_pause_min, self.melody_pause_max)
            required_bass_gap = random.uniform(self.bass_pause_min, self.bass_pause_max)
            
            # Check episode states (hierarchical level 3)
            melody_should_play, melody_reasoning = self.melody_episode_manager.should_generate_phrase()
            bass_should_play, bass_reasoning = self.bass_episode_manager.should_generate_phrase()
            
            # Check if each voice is ready based on its OWN timing (level 2)
            melody_timing_ready = time_since_melody > required_melody_gap
            bass_timing_ready = time_since_bass > required_bass_gap
            
            # INDEPENDENT voice selection - bass doesn't wait for melody
            melody_can_play = melody_timing_ready and melody_should_play
            bass_can_play = bass_timing_ready and bass_should_play
            
            if melody_can_play and bass_can_play:
                # Both voices ready - alternate to give space
                if not hasattr(self, '_voice_alternation_counter'):
                    self._voice_alternation_counter = 0
                
                voice_type = "melodic" if self._voice_alternation_counter % 2 == 0 else "bass"
                print(f"🎵 Both ready: chose {voice_type} (counter={self._voice_alternation_counter})")
                print(f"   Melody: {time_since_melody:.1f}s/{required_melody_gap:.1f}s, {melody_reasoning}")
                print(f"   Bass: {time_since_bass:.1f}s/{required_bass_gap:.1f}s, {bass_reasoning}")
                self._voice_alternation_counter += 1
                
            elif melody_can_play:
                voice_type = "melodic"
                print(f"🎵 Voice: melody ({time_since_melody:.1f}s/{required_melody_gap:.1f}s, {melody_reasoning})")
                
            elif bass_can_play:
                voice_type = "bass"
                print(f"🎵 Voice: bass ({time_since_bass:.1f}s/{required_bass_gap:.1f}s, {bass_reasoning})")
            
            # DEBUG: Log when we're blocking generation
            elif melody_timing_ready or bass_timing_ready:
                if melody_timing_ready and not melody_should_play:
                    print(f"⏸️  Melody timing ready but episode blocking: {melody_reasoning}")
                if bass_timing_ready and not bass_should_play:
                    print(f"⏸️  Bass timing ready but episode blocking: {bass_reasoning}")
                
            if not voice_type:
                # Neither voice ready - check why
                if not melody_should_play and not bass_should_play:
                    print(f"🎧 Both LISTENING - melody: {melody_reasoning}, bass: {bass_reasoning}")
                elif not melody_timing_ready and not bass_timing_ready:
                    # Still in phrase-to-phrase gaps
                    pass  # Silent - normal operation
                return decisions
            
            # If we selected a voice, generate a phrase for it
            if voice_type:
                # Generate musical phrase with mode-specific temperature and activity multiplier
                current_params = self.mode_params[mode]
                
                # Get voice timing profile for this style and voice
                voice_profile = VoiceTimingProfile.get_profile(
                    self.last_style_detection.get('style', 'ballad') if self.last_style_detection else 'ballad',
                    voice_type
                )
                
                # Update episode manager with profile parameters
                episode_manager = self.melody_episode_manager if voice_type == "melodic" else self.bass_episode_manager
                episode_manager.update_profile_parameters(voice_profile)
                
                phrase = self.phrase_generator.generate_phrase(
                    current_event, voice_type, mode.value, harmonic_context,
                    temperature=current_params['temperature'],
                    activity_multiplier=activity_multiplier,
                    voice_profile=voice_profile
                )
                
                if phrase:
                    if phrase.phrase_type == "silence":
                        # Handle silence phrase - DON'T update timing (no notes sent)
                        # Silence phrases shouldn't block future generation
                        print(f"🤫 Silence phrase: {phrase.timings[0]:.1f}s (not blocking)")
                        # Return empty decisions for silence
                        return decisions
                    
                    else:
                        # Track phrase generation in episode manager
                        episode_manager.track_phrase_generation()
                        
                        # Store phrase for scheduling instead of creating multiple immediate decisions
                        phrase_id = f"{voice_type}_{phrase.phrase_id}"
                        self.active_phrases[phrase_id] = {
                            'phrase': phrase,
                            'note_index': 0,
                            'start_time': current_time,
                            'mode': mode,
                            'voice_type':voice_type
                        }
                        
                        # Schedule first note immediately
                        first_note = phrase.notes[0]
                        musical_params = {
                            'target_note': int(first_note),
                            'velocity': phrase.velocities[0],
                            'duration': phrase.durations[0],
                            'timing': phrase.timings[0],
                            'phrase_id': phrase.phrase_id,
                            'phrase_type': phrase.phrase_type,
                            'phrase_scheduler': phrase_id  # Flag for phrase scheduling
                        }
                        
                        decision = MusicalDecision(
                            mode=mode,
                            confidence=phrase.confidence,
                            target_features=np.array([first_note]),
                            musical_params=musical_params,
                            timestamp=current_time,
                            reasoning=f"Rhythmic phrase start: {phrase.phrase_type}",
                            voice_type=voice_type,
                            instrument=voice_type
                        )
                        
                        # Apply CLAP-driven timing profile
                        decision = self._apply_timing_profile(decision)
                        decisions.append(decision)
                        
                        # Update timing for voice immediately to prevent overlap
                        phrase_duration = (len(phrase.notes) - 1) * 0.25 + phrase.durations[-1]
                        if voice_type == "melodic":
                            self.last_melody_time = current_time + phrase_duration
                        else:
                            self.last_bass_time = current_time + phrase_duration
                        
                        # Log episode status for transparency
                        melody_status = self.melody_episode_manager.get_status()
                        bass_status = self.bass_episode_manager.get_status()
                        print(f"📊 Episode status - Melody: {melody_status['state']} ({melody_status['elapsed']:.1f}s/{melody_status['duration']:.1f}s, {melody_status['phrases_this_episode']} phrases)")
                        print(f"📊 Episode status - Bass: {bass_status['state']} ({bass_status['elapsed']:.1f}s/{bass_status['duration']:.1f}s, {bass_status['phrases_this_episode']} phrases)")
                        
                        # Generated phrase (silent)
        
        else:
            print("🎹 No phrase generator - using fallback single notes")
            # Fallback to single note generation if no phrase generator
            time_since_melody = current_time - self.last_melody_time
            time_since_bass = current_time - self.last_bass_time
            
            if time_since_melody > 0.5 and time_since_bass > 0.5:
                # Both voices can play - alternate between melody and bass
                if not hasattr(self, '_voice_alternation_counter'):
                    self._voice_alternation_counter = 0
                
                if self._voice_alternation_counter % 2 == 0:
                    # Generate melody
                    melodic_decision = self._generate_single_decision(mode, current_event, memory_buffer, clustering, "melodic")
                    decisions.append(melodic_decision)
                    self.last_melody_time = current_time
                    print(f"🎵 Voice alternation: chose melody (counter={self._voice_alternation_counter})")
                else:
                    # Generate bass
                    bass_decision = self._generate_single_decision(mode, current_event, memory_buffer, clustering, "bass")
                    decisions.append(bass_decision)
                    self.last_bass_time = current_time
                    print(f"🎵 Voice alternation: chose bass (counter={self._voice_alternation_counter})")
                
                self._voice_alternation_counter += 1
            elif time_since_melody > 0.3:
                # Melody can play
                melodic_decision = self._generate_single_decision(mode, current_event, memory_buffer, clustering, "melodic")
                decisions.append(melodic_decision)
                self.last_melody_time = current_time
            elif time_since_bass > 0.3:
                # Bass can play
                bass_decision = self._generate_single_decision(mode, current_event, memory_buffer, clustering, "bass")
                decisions.append(bass_decision)
                self.last_bass_time = current_time
        
        return decisions
    
    def _generate_single_decision(self, mode: BehaviorMode, current_event: Dict,
                                 memory_buffer, clustering, voice_type: str) -> MusicalDecision:
        """Generate a single musical decision for a specific voice type"""
        current_time = time.time()
        
        if mode == BehaviorMode.IMITATE:
            return self._imitate_decision(current_event, memory_buffer, clustering, voice_type)
        elif mode == BehaviorMode.CONTRAST:
            return self._contrast_decision(current_event, memory_buffer, clustering, voice_type)
        elif mode == BehaviorMode.LEAD:
            return self._lead_decision(current_event, memory_buffer, clustering, voice_type)
        else:
            # Fallback
            return self._imitate_decision(current_event, memory_buffer, clustering, voice_type)
    
    def _should_generate_bass(self, current_event: Dict) -> bool:
        """Determine if bass should be generated based on musical context, instrument, and timing separation"""
        current_time = time.time()
        
        # Check timing separation - only block if melody was generated very recently (within 100ms)
        time_since_melody = current_time - self.last_melody_time
        if time_since_melody < 0.1:  # Only block if melody was within last 100ms
            print(f"🔇 Bass blocked: melody too recent ({time_since_melody:.3f}s)")
            return False
        
        instrument = current_event.get('instrument', 'unknown')
        rms_db = current_event.get('rms_db', -80.0)
        
        print(f"🎸 Bass check: instrument={instrument}, rms={rms_db:.1f}dB, time_since_melody={time_since_melody:.3f}s")
        
        # Instrument-specific bass generation
        if instrument == "bass":
            # Always generate bass when bass is detected
            print(f"🎸 Bass generated: instrument=bass (always)")
            return True
        elif instrument == "drums":
            # Generate bass more often with drums
            if rms_db > -30.0:
                result = random.random() < 0.7  # 70% chance
                print(f"🎸 Bass check: drums, loud ({rms_db:.1f}dB) -> {result}")
                return result
            elif rms_db > -50.0:
                result = random.random() < 0.4  # 40% chance
                print(f"🎸 Bass check: drums, moderate ({rms_db:.1f}dB) -> {result}")
                return result
            else:
                result = random.random() < 0.1  # 10% chance
                print(f"🎸 Bass check: drums, quiet ({rms_db:.1f}dB) -> {result}")
                return result
        elif instrument == "piano":
            # Generate bass moderately with piano
            if rms_db > -30.0:
                result = random.random() < 0.5  # 50% chance
                print(f"🎸 Bass check: piano, loud ({rms_db:.1f}dB) -> {result}")
                return result
            elif rms_db > -50.0:
                result = random.random() < 0.3  # 30% chance
                print(f"🎸 Bass check: piano, moderate ({rms_db:.1f}dB) -> {result}")
                return result
        elif instrument == "speech":
            # Generate bass less frequently with speech (accompaniment)
            if rms_db > -30.0:
                result = random.random() < 0.3  # 30% chance
                print(f"🎸 Bass check: speech, loud ({rms_db:.1f}dB) -> {result}")
                return result
            elif rms_db > -50.0:
                result = random.random() < 0.2  # 20% chance
                print(f"🎸 Bass check: speech, moderate ({rms_db:.1f}dB) -> {result}")
                return result
            else:
                result = random.random() < 0.1  # 10% chance
                print(f"🎸 Bass check: speech, quiet ({rms_db:.1f}dB) -> {result}")
                return result
        else:
            # Default behavior for unknown instruments
            if rms_db > -30.0:  # Active musical input
                result = random.random() < 0.4  # 40% chance
                print(f"🎸 Bass check: {instrument}, loud ({rms_db:.1f}dB) -> {result}")
                return result
            elif rms_db > -50.0:  # Moderate activity
                result = random.random() < 0.2  # 20% chance
                print(f"🎸 Bass check: {instrument}, moderate ({rms_db:.1f}dB) -> {result}")
                return result
            else:
                result = random.random() < 0.05  # 5% chance for quiet moments
                print(f"🎸 Bass check: {instrument}, quiet ({rms_db:.1f}dB) -> {result}")
                return result

    def _should_generate_percussion(self, current_event: Dict) -> bool:
        """Determine if percussion should be generated for drums"""
        instrument = current_event.get('instrument', 'unknown')
        rms_db = current_event.get('rms_db', -80.0)
        
        if instrument == "drums":
            if rms_db > -30.0:  # Active drum input
                return random.random() < 0.8  # 80% chance
            elif rms_db > -50.0:  # Moderate activity
                return random.random() < 0.5  # 50% chance
            else:
                return random.random() < 0.2  # 20% chance for quiet moments
        elif instrument == "piano":
            # Sometimes add percussion with piano for rhythm
            if rms_db > -30.0:
                return random.random() < 0.2  # 20% chance
            else:
                return random.random() < 0.05  # 5% chance
        elif instrument == "speech":
            # Minimal percussion with speech (subtle accompaniment)
            if rms_db > -30.0:
                return random.random() < 0.1  # 10% chance
            else:
                return random.random() < 0.02  # 2% chance
        return False

    def _should_generate_harmony(self, current_event: Dict) -> bool:
        """Determine if harmony should be generated for piano"""
        instrument = current_event.get('instrument', 'unknown')
        rms_db = current_event.get('rms_db', -80.0)
        
        if instrument == "piano":
            if rms_db > -30.0:  # Active piano input
                return random.random() < 0.7  # 70% chance
            elif rms_db > -50.0:  # Moderate activity
                return random.random() < 0.4  # 40% chance
            else:
                return random.random() < 0.1  # 10% chance for quiet moments
        elif instrument == "guitar":
            # Sometimes add harmony with guitar
            if rms_db > -30.0:
                return random.random() < 0.3  # 30% chance
            else:
                return random.random() < 0.1  # 10% chance
        elif instrument == "speech":
            # Moderate harmony with speech (musical accompaniment)
            if rms_db > -30.0:
                return random.random() < 0.4  # 40% chance
            elif rms_db > -50.0:
                return random.random() < 0.2  # 20% chance
            else:
                return random.random() < 0.05  # 5% chance
        return False
    
    def _imitate_decision(self, current_event: Dict, 
                         memory_buffer, clustering, voice_type: str = "melodic") -> MusicalDecision:
        """Generate imitate behavior decision with harmonic awareness"""
        # Extract harmonic context if available
        harmonic_context = current_event.get('harmonic_context')
        
        # Find similar moments in memory
        current_features = np.array([
            current_event.get('centroid', 0.0),
            current_event.get('rms_db', -80.0),
            current_event.get('cents', 0.0),
            current_event.get('ioi', 0.0)
        ])
        
        # Normalize features (simplified)
        current_features_norm = self._normalize_features(current_features, memory_buffer)
        
        # Find similar moments
        similar_moments = memory_buffer.find_neighbors(
            current_features_norm, radius=1.0, max_results=5
        )
        
        if similar_moments:
            # Imitate the most recent similar moment
            target_moment = similar_moments[-1]
            confidence = 0.8
            reasoning = f"Imitating similar moment from {target_moment.timestamp:.1f}s ago"
        else:
            # No similar moments, use current event
            target_moment = None
            confidence = 0.4
            reasoning = "No similar moments found, using current event"
        
        # Add harmonic information to reasoning
        if harmonic_context:
            chord = harmonic_context.get('current_chord', 'N/A')
            key = harmonic_context.get('key_signature', 'N/A')
            reasoning += f" | Chord: {chord}, Key: {key}"
        
        # Generate musical parameters (with harmonic awareness)
        musical_params = self._generate_imitate_params(current_event, target_moment, voice_type, harmonic_context)
        
        # 🎨 VISUALIZATION: Emit request parameters for viewport (rate-limited)
        if self.visualization_manager and harmonic_context:
            current_time = time.time()
            if not hasattr(self, '_last_viz_emit'):
                self._last_viz_emit = 0
            
            # Rate limit: max 2 updates per second to avoid Qt event queue spam
            if current_time - self._last_viz_emit > 0.5:
                request = {
                    'parameter': 'consonance',
                    'type': '==',
                    'value': harmonic_context.get('stability', 0.5),
                    'weight': 0.85
                }
                self.visualization_manager.emit_request_params(
                    mode=BehaviorMode.IMITATE.value,
                    request=request,
                    voice_type=voice_type
                )
                self._last_viz_emit = current_time
        
        decision = MusicalDecision(
            mode=BehaviorMode.IMITATE,
            confidence=confidence,
            target_features=current_features_norm,
            musical_params=musical_params,
            timestamp=time.time(),
            reasoning=reasoning,
            voice_type=voice_type,
            instrument=current_event.get('instrument', 'unknown')
        )
        
        # Apply CLAP-driven timing profile
        return self._apply_timing_profile(decision)
    
    def _contrast_decision(self, current_event: Dict, 
                          memory_buffer, clustering, voice_type: str = "melodic") -> MusicalDecision:
        """Generate contrast behavior decision with harmonic awareness"""
        harmonic_context = current_event.get('harmonic_context')
        
        current_features = np.array([
            current_event.get('centroid', 0.0),
            current_event.get('rms_db', -80.0),
            current_event.get('cents', 0.0),
            current_event.get('ioi', 0.0)
        ])
        
        current_features_norm = self._normalize_features(current_features, memory_buffer)
        
        # Find contrasting moments
        contrasting_moments = memory_buffer.find_distant_moments(
            current_features_norm, min_distance=1.5, max_results=3
        )
        
        if contrasting_moments:
            # Use contrasting moment as inspiration
            target_moment = random.choice(contrasting_moments)
            confidence = 0.7
            reasoning = f"Contrasting with moment from {target_moment.timestamp:.1f}s ago"
        else:
            # Generate artificial contrast
            target_moment = None
            confidence = 0.5
            reasoning = "Generating artificial contrast"
        
        # Add harmonic contrast information
        if harmonic_context:
            chord = harmonic_context.get('current_chord', 'N/A')
            reasoning += f" | Using related chords to {chord}"
        
        musical_params = self._generate_contrast_params(current_event, target_moment, voice_type, harmonic_context)
        
        # 🎨 VISUALIZATION: Emit request parameters for viewport (rate-limited)
        if self.visualization_manager and harmonic_context:
            current_time = time.time()
            if not hasattr(self, '_last_viz_emit'):
                self._last_viz_emit = 0
            
            # Rate limit: max 2 updates per second
            if current_time - self._last_viz_emit > 0.5:
                request = {
                    'parameter': 'consonance',
                    'type': '<',
                    'value': 0.5,
                    'weight': 0.7
                }
                self.visualization_manager.emit_request_params(
                    mode=BehaviorMode.CONTRAST.value,
                    request=request,
                    voice_type=voice_type
                )
                self._last_viz_emit = current_time
        
        decision = MusicalDecision(
            mode=BehaviorMode.CONTRAST,
            confidence=confidence,
            target_features=current_features_norm,
            musical_params=musical_params,
            timestamp=time.time(),
            reasoning=reasoning,
            voice_type=voice_type,
            instrument=current_event.get('instrument', 'unknown')
        )
        
        # Apply CLAP-driven timing profile
        return self._apply_timing_profile(decision)
    
    def _lead_decision(self, current_event: Dict, 
                      memory_buffer, clustering, voice_type: str = "melodic") -> MusicalDecision:
        """Generate lead behavior decision with harmonic awareness"""
        harmonic_context = current_event.get('harmonic_context')
        
        # Lead behavior creates new musical directions
        current_features = np.array([
            current_event.get('centroid', 0.0),
            current_event.get('rms_db', -80.0),
            current_event.get('cents', 0.0),
            current_event.get('ioi', 0.0)
        ])
        
        current_features_norm = self._normalize_features(current_features, memory_buffer)
        
        # Generate innovative musical parameters
        musical_params = self._generate_lead_params(current_event, voice_type, harmonic_context)
        
        confidence = 0.6
        reasoning = "Leading with innovative musical direction"
        
        # Add harmonic innovation info
        if harmonic_context:
            key = harmonic_context.get('key_signature', 'N/A')
            reasoning += f" | Exploring new territory in {key}"
        
        # 🎨 VISUALIZATION: Emit request parameters for viewport (rate-limited)
        if self.visualization_manager and harmonic_context:
            current_time = time.time()
            if not hasattr(self, '_last_viz_emit'):
                self._last_viz_emit = 0
            
            # Rate limit: max 2 updates per second
            if current_time - self._last_viz_emit > 0.5:
                request = {
                    'parameter': 'innovation',
                    'type': 'exploration',
                    'value': 'high',
                    'weight': 0.3
                }
                self.visualization_manager.emit_request_params(
                    mode=BehaviorMode.LEAD.value,
                    request=request,
                    voice_type=voice_type
                )
                self._last_viz_emit = current_time
        
        decision = MusicalDecision(
            mode=BehaviorMode.LEAD,
            confidence=confidence,
            target_features=current_features_norm,
            musical_params=musical_params,
            timestamp=time.time(),
            reasoning=reasoning,
            voice_type=voice_type,
            instrument=current_event.get('instrument', 'unknown')
        )
        
        # Apply CLAP-driven timing profile
        return self._apply_timing_profile(decision)
    
    def _apply_timing_profile(self, decision: MusicalDecision, phrase_detector=None) -> MusicalDecision:
        """
        Apply CLAP-driven timing profile to a decision
        
        Args:
            decision: Musical decision to enhance with timing profile
            phrase_detector: Optional phrase detector for interpolation boundaries
            
        Returns:
            Decision with timing profile applied
        """
        current_time = time.time()
        voice_type = decision.voice_type
        
        # Get current style from CLAP
        style = None
        if self.last_style_detection:
            style = self.last_style_detection.get('style')
        
        if not style:
            # No style detected, use default (ballad)
            style = 'ballad'
        
        # Get timing profile for this voice
        profile = VoiceTimingProfile.get_profile(style, voice_type)
        
        # Apply profile parameters to decision
        decision.timing_precision = profile['timing_precision']
        decision.rhythmic_density = profile['rhythmic_density']
        decision.syncopation_tendency = profile['syncopation_tendency']
        decision.voice_profile = profile.copy()
        
        return decision
    
    def _normalize_features(self, features: np.ndarray, memory_buffer) -> np.ndarray:
        """Normalize features using memory buffer statistics"""
        stats = memory_buffer.get_buffer_stats()
        feature_stats = stats.get('feature_stats', {})
        
        if not feature_stats:
            return features
        
        normalized = np.zeros_like(features)
        feature_names = ['centroid', 'rms', 'cents', 'ioi']
        
        for i, name in enumerate(feature_names):
            if name in feature_stats:
                mean = feature_stats[name]['mean']
                std = max(feature_stats[name]['std'], 1e-6)
                normalized[i] = (features[i] - mean) / std
        
        return normalized
    
    def _extract_request_params_from_context(self, harmonic_context: Dict, mode: BehaviorMode) -> List[Dict]:
        """Extract visualization-friendly request parameters from harmonic context"""
        request_params = []
        
        # Extract chord information
        if 'current_chord' in harmonic_context:
            request_params.append({
                'parameter': 'chord',
                'type': 'harmonic',
                'value': harmonic_context['current_chord'],
                'weight': 0.8
            })
        
        # Extract key signature
        if 'key_signature' in harmonic_context:
            request_params.append({
                'parameter': 'key',
                'type': 'harmonic',
                'value': harmonic_context['key_signature'],
                'weight': 0.7
            })
        
        # Extract stability/consonance
        if 'stability' in harmonic_context:
            stability = harmonic_context['stability']
            request_params.append({
                'parameter': 'consonance',
                'type': 'harmonic',
                'value': f"{stability:.2f}",
                'weight': 0.9
            })
        
        # Mode-specific parameters
        if mode == BehaviorMode.IMITATE:
            request_params.append({
                'parameter': 'similarity',
                'type': 'behavioral',
                'value': 'high',
                'weight': 0.95
            })
        elif mode == BehaviorMode.CONTRAST:
            request_params.append({
                'parameter': 'contrast',
                'type': 'behavioral',
                'value': 'high',
                'weight': 0.85
            })
        elif mode == BehaviorMode.LEAD:
            request_params.append({
                'parameter': 'innovation',
                'type': 'behavioral',
                'value': 'high',
                'weight': 0.75
            })
        
        return request_params
    
    def _generate_imitate_params(self, current_event: Dict, 
                               target_moment, voice_type: str = "melodic", 
                               harmonic_context: Optional[Dict] = None) -> Dict:
        """Generate musical parameters for imitate behavior with harmonic awareness"""
        params = {
            'velocity': 80,
            'duration': 1.0,
            'attack': 0.1,
            'release': 0.3,
            'filter_cutoff': 0.7,
            'modulation_depth': 0.3
        }
        
        # Add harmonic information if available
        if harmonic_context:
            params['current_chord'] = harmonic_context.get('current_chord', 'N/A')
            params['key_signature'] = harmonic_context.get('key_signature', 'N/A')
            params['scale_degrees'] = harmonic_context.get('scale_degrees', [])
            params['chord_root'] = harmonic_context.get('chord_root', 0)
            params['chord_stability'] = harmonic_context.get('stability', 0.5)
        
        # Adjust parameters based on voice type
        if voice_type == "bass":
            params['velocity'] = 70  # Slightly softer for bass
            params['duration'] = 1.5  # Longer notes for bass
            params['attack'] = 0.15  # Slightly slower attack
            params['filter_cutoff'] = 0.3  # Lower filter for bass
            params['modulation_depth'] = 0.1  # Less modulation
        
        if target_moment:
            # Use target moment's characteristics
            event_data = target_moment.event_data
            params['velocity'] = min(127, max(40, int(80 + event_data.get('rms_db', -60) * 0.5)))
            params['filter_cutoff'] = min(1.0, max(0.1, event_data.get('centroid', 1000) / 2000))
        
        return params
    
    def _generate_contrast_params(self, current_event: Dict, 
                                target_moment, voice_type: str = "melodic",
                                harmonic_context: Optional[Dict] = None) -> Dict:
        """Generate musical parameters for contrast behavior with harmonic awareness"""
        params = {
            'velocity': 60,
            'duration': 0.5,
            'attack': 0.05,
            'release': 0.8,
            'filter_cutoff': 0.3,
            'modulation_depth': 0.8
        }
        
        # Add harmonic information for contrast (use related chords)
        if harmonic_context:
            params['current_chord'] = harmonic_context.get('current_chord', 'N/A')
            params['key_signature'] = harmonic_context.get('key_signature', 'N/A')
            params['scale_degrees'] = harmonic_context.get('scale_degrees', [])
            params['chord_root'] = harmonic_context.get('chord_root', 0)
            params['use_contrasting_harmony'] = True  # Flag for FeatureMapper
        
        # Adjust parameters based on voice type
        if voice_type == "bass":
            params['velocity'] = 50  # Softer for bass contrast
            params['duration'] = 0.8  # Longer bass notes
            params['attack'] = 0.1  # Slower attack
            params['filter_cutoff'] = 0.2  # Lower filter
            params['modulation_depth'] = 0.4  # Less modulation
        
        if target_moment:
            # Contrast with target moment
            event_data = target_moment.event_data
            params['velocity'] = min(127, max(20, int(100 - event_data.get('rms_db', -60) * 0.3)))
            params['filter_cutoff'] = min(1.0, max(0.1, 1.0 - (event_data.get('centroid', 1000) / 2000)))
        
        return params
    
    def _generate_lead_params(self, current_event: Dict, voice_type: str = "melodic",
                           harmonic_context: Optional[Dict] = None) -> Dict:
        """Generate musical parameters for lead behavior with harmonic awareness"""
        # Lead behavior is more experimental
        params = {
            'velocity': random.randint(60, 100),
            'duration': random.uniform(0.3, 2.0),
            'attack': random.uniform(0.02, 0.2),
            'release': random.uniform(0.2, 1.0),
            'filter_cutoff': random.uniform(0.2, 0.9),
            'modulation_depth': random.uniform(0.4, 1.0)
        }
        
        # Add harmonic information for exploration
        if harmonic_context:
            params['current_chord'] = harmonic_context.get('current_chord', 'N/A')
            params['key_signature'] = harmonic_context.get('key_signature', 'N/A')
            params['scale_degrees'] = harmonic_context.get('scale_degrees', [])
            params['chord_root'] = harmonic_context.get('chord_root', 0)
            params['explore_harmony'] = True  # Flag for FeatureMapper
        
        # Adjust parameters based on voice type
        if voice_type == "bass":
            params['velocity'] = random.randint(50, 80)  # Lower velocity range
            params['duration'] = random.uniform(0.8, 2.5)  # Longer notes
            params['attack'] = random.uniform(0.1, 0.4)  # Slower attack
            params['filter_cutoff'] = random.uniform(0.1, 0.5)  # Lower filter range
            params['modulation_depth'] = random.uniform(0.05, 0.4)  # Less modulation
        
        return params
    
    def get_current_mode(self) -> BehaviorMode:
        """Get current behavior mode"""
        return self.current_mode
    
    def get_mode_history(self) -> List[Dict]:
        """Get history of mode changes"""
        return self.mode_history
    
    def get_decision_history(self) -> List[MusicalDecision]:
        """Get history of musical decisions"""
        return self.decision_history

    def auto_select_mode(self, audio: np.ndarray, sr: int = 44100, current_time: float = None) -> Optional[Dict]:
        """
        Automatically select behavioral mode based on detected musical style using CLAP

        Args:
            audio: Audio buffer (mono, float32 or float64)
            sr: Sample rate
            current_time: Current timestamp (defaults to time.time())

        Returns:
            Dict with detection results or None if CLAP disabled/failed
        """
        if not self.enable_clap or not self.style_detector:
            return None

        # Throttle detection (don't detect too frequently)
        if current_time is None:
            current_time = time.time()

        time_since_last = current_time - self.last_detection_time

        if time_since_last < self.clap_update_interval:
            # Return cached result
            return self.last_style_detection

        # Detect style using CLAP
        style_result = self.style_detector.detect_style(audio, sr)

        if style_result is None:
            return None

        # Auto-switch mode if enabled
        if self.auto_behavior_mode:
            recommended_mode = style_result.recommended_mode
            if recommended_mode != self.current_mode:
                print(f"\n🎨 Style detected: {style_result.style_label} "
                      f"(confidence: {style_result.confidence:.2f})")
                print(f"🎭 Auto-switching: {self.current_mode.value} → {recommended_mode.value}")

                # Change mode
                self.current_mode = recommended_mode

                # Record mode change
                self.mode_history.append({
                    'timestamp': current_time,
                    'mode': recommended_mode.value,
                    'reason': f'CLAP style detection: {style_result.style_label}',
                    'confidence': style_result.confidence
                })

        # Cache result
        self.last_style_detection = {
            'style': style_result.style_label,
            'confidence': float(style_result.confidence),
            'recommended_mode': style_result.recommended_mode.value,
            'secondary_styles': style_result.secondary_styles,
            'timestamp': current_time
        }
        self.last_detection_time = current_time

        return self.last_style_detection

    def _generate_percussion_decision(self, mode: BehaviorMode, current_event: Dict,
                                    memory_buffer, clustering) -> MusicalDecision:
        """Generate percussion-specific decision for drums"""
        # Percussion decisions focus on rhythmic patterns
        current_features = np.array([
            current_event.get('centroid', 0.0),
            current_event.get('rms_db', -80.0),
            current_event.get('zcr', 0.0),
            current_event.get('ioi', 0.0)
        ])
        
        current_features_norm = self._normalize_features(current_features, memory_buffer)
        
        # Generate percussion parameters
        musical_params = {
            'velocity': random.randint(80, 127),  # High velocity for percussion
            'duration': random.uniform(0.1, 0.5),  # Short duration
            'attack': 0.01,  # Very fast attack
            'release': 0.1,  # Short release
            'filter_cutoff': 0.9,  # Bright filter
            'modulation_depth': 0.2  # Low modulation
        }
        
        confidence = 0.7
        reasoning = "Generating percussion response to drums"
        
        return MusicalDecision(
            mode=mode,
            confidence=confidence,
            target_features=current_features_norm,
            musical_params=musical_params,
            timestamp=time.time(),
            reasoning=reasoning,
            voice_type="percussion",
            instrument=current_event.get('instrument', 'unknown')
        )

    def _generate_harmony_decision(self, mode: BehaviorMode, current_event: Dict,
                                 memory_buffer, clustering) -> MusicalDecision:
        """Generate harmony-specific decision for piano"""
        # Harmony decisions focus on chord progressions
        current_features = np.array([
            current_event.get('centroid', 0.0),
            current_event.get('rms_db', -80.0),
            current_event.get('hnr', 0.0),
            current_event.get('f0', 0.0)
        ])
        
        current_features_norm = self._normalize_features(current_features, memory_buffer)
        
        # Generate harmony parameters
        musical_params = {
            'velocity': random.randint(60, 90),  # Moderate velocity
            'duration': random.uniform(1.0, 3.0),  # Longer duration
            'attack': 0.1,  # Moderate attack
            'release': 0.5,  # Longer release
            'filter_cutoff': 0.6,  # Moderate filter
            'modulation_depth': 0.4  # Moderate modulation
        }
        
        confidence = 0.6
        reasoning = "Generating harmonic response to piano"
        
        return MusicalDecision(
            mode=mode,
            confidence=confidence,
            target_features=current_features_norm,
            musical_params=musical_params,
            timestamp=time.time(),
            reasoning=reasoning,
            voice_type="harmony",
            instrument=current_event.get('instrument', 'unknown')
        )
