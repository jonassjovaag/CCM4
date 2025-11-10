"""
Voice Leading Analysis System
Analyzes melodic motion between chords and detects voice leading principles
Based on Tymoczko (2011) "A Geometry of Music"
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from itertools import combinations


@dataclass
class VoiceLeadingAnalysis:
    """Container for voice leading analysis results"""
    smoothness_score: float  # Average interval size (0-12 semitones)
    parallel_fifths: List[Dict]  # List of parallel 5th violations
    parallel_octaves: List[Dict]  # List of parallel octave violations
    contrary_motion_count: int  # Instances of contrary motion (good)
    voice_crossings: List[Dict]  # Instances of voice crossing
    average_voice_range: Dict[str, float]  # Average range for each voice
    stepwise_motion_percentage: float  # Percentage of stepwise motion (good)
    leap_percentage: float  # Percentage of leaps (>2 semitones)
    voice_independence: float  # How independent the voices are (0-1)


@dataclass
class ChordVoicing:
    """Represents a chord with specific voice assignments"""
    timestamp: float
    chord_name: str
    voices: Dict[str, Optional[int]]  # {'soprano': 67, 'alto': 64, 'tenor': 60, 'bass': 48}
    pitch_classes: List[int]  # All pitch classes in the chord


class VoiceLeadingAnalyzer:
    """
    Analyzes voice leading between chords
    Based on classical voice leading principles and Tymoczko's geometric approach
    """
    
    def __init__(self, num_voices: int = 4):
        """
        Initialize voice leading analyzer
        
        Args:
            num_voices: Number of voices to track (default 4: SATB)
        """
        self.num_voices = num_voices
        self.voice_names = ['soprano', 'alto', 'tenor', 'bass'][:num_voices]
        
        # Voice ranges (MIDI note numbers)
        self.voice_ranges = {
            'soprano': (60, 81),  # C4 to A5
            'alto': (55, 74),     # G3 to D5
            'tenor': (48, 69),    # C3 to A4
            'bass': (40, 64)      # E2 to E4
        }
        
        print(f"🎼 Voice Leading Analyzer initialized with {num_voices} voices")
    
    def analyze_chord_progression(self, chord_progression: List[str], 
                                 chroma_vectors: List[np.ndarray],
                                 timestamps: List[float]) -> VoiceLeadingAnalysis:
        """
        Analyze voice leading in a chord progression
        
        Args:
            chord_progression: List of chord names
            chroma_vectors: List of chroma vectors for each chord
            timestamps: List of timestamps for each chord
            
        Returns:
            VoiceLeadingAnalysis with complete voice leading analysis
        """
        if len(chord_progression) < 2:
            return self._empty_analysis()
        
        print(f"\n🎼 Analyzing voice leading for {len(chord_progression)} chords...")
        
        # Step 1: Assign voices to each chord
        voicings = []
        for i, (chord, chroma, timestamp) in enumerate(zip(chord_progression, chroma_vectors, timestamps)):
            voicing = self._assign_voices_from_chroma(chord, chroma, timestamp)
            voicings.append(voicing)
        
        # Step 2: Analyze voice motion between consecutive chords
        parallel_fifths = []
        parallel_octaves = []
        voice_crossings = []
        contrary_motion_count = 0
        intervals = []
        stepwise_count = 0
        leap_count = 0
        
        for i in range(len(voicings) - 1):
            prev_voicing = voicings[i]
            curr_voicing = voicings[i + 1]
            
            # Calculate voice motion
            motion = self._calculate_voice_motion(prev_voicing, curr_voicing)
            
            # Detect parallel motion violations
            parallels = self._detect_parallel_motion(prev_voicing, curr_voicing, motion)
            for violation_type, v1, v2 in parallels:
                if violation_type == 'parallel_5th':
                    parallel_fifths.append({
                        'position': i,
                        'voice1': v1,
                        'voice2': v2,
                        'chord1': prev_voicing.chord_name,
                        'chord2': curr_voicing.chord_name
                    })
                elif violation_type == 'parallel_octave':
                    parallel_octaves.append({
                        'position': i,
                        'voice1': v1,
                        'voice2': v2,
                        'chord1': prev_voicing.chord_name,
                        'chord2': curr_voicing.chord_name
                    })
            
            # Detect contrary motion (good!)
            if self._has_contrary_motion(motion):
                contrary_motion_count += 1
            
            # Detect voice crossings
            crossings = self._detect_voice_crossings(curr_voicing)
            voice_crossings.extend(crossings)
            
            # Track intervals for smoothness
            for voice_name, interval in motion.items():
                if interval is not None:
                    intervals.append(abs(interval))
                    if abs(interval) <= 2:
                        stepwise_count += 1
                    elif abs(interval) > 2:
                        leap_count += 1
        
        # Step 3: Calculate metrics
        smoothness_score = np.mean(intervals) if intervals else 0.0
        total_motion = stepwise_count + leap_count
        stepwise_percentage = (stepwise_count / total_motion * 100) if total_motion > 0 else 0.0
        leap_percentage = (leap_count / total_motion * 100) if total_motion > 0 else 0.0
        
        # Calculate voice ranges
        average_voice_range = self._calculate_voice_ranges(voicings)
        
        # Calculate voice independence
        voice_independence = self._calculate_voice_independence(voicings)
        
        print(f"✅ Voice leading analysis complete")
        print(f"   Smoothness: {smoothness_score:.2f} semitones average")
        print(f"   Stepwise motion: {stepwise_percentage:.1f}%")
        print(f"   Parallel 5ths: {len(parallel_fifths)}")
        print(f"   Parallel octaves: {len(parallel_octaves)}")
        print(f"   Contrary motion: {contrary_motion_count} instances")
        
        return VoiceLeadingAnalysis(
            smoothness_score=smoothness_score,
            parallel_fifths=parallel_fifths,
            parallel_octaves=parallel_octaves,
            contrary_motion_count=contrary_motion_count,
            voice_crossings=voice_crossings,
            average_voice_range=average_voice_range,
            stepwise_motion_percentage=stepwise_percentage,
            leap_percentage=leap_percentage,
            voice_independence=voice_independence
        )
    
    def _assign_voices_from_chroma(self, chord_name: str, chroma: np.ndarray, timestamp: float) -> ChordVoicing:
        """
        Assign voices from chroma vector
        
        Strategy:
        1. Find strongest pitch classes in chroma
        2. Assign to voices (soprano=highest, bass=lowest)
        3. Space according to voice ranges
        """
        # Find pitch classes with significant energy
        threshold = np.mean(chroma) + 0.5 * np.std(chroma)
        active_pcs = [i for i, value in enumerate(chroma) if value > threshold]
        
        if not active_pcs:
            # Fallback: use top 3 pitch classes
            active_pcs = np.argsort(chroma)[-3:].tolist()
        
        # Sort pitch classes
        active_pcs.sort()
        
        # Assign to voices (spread across registers)
        voices = {}
        
        if len(active_pcs) >= 4:
            # Full 4-part harmony
            voices['bass'] = active_pcs[0] + 48      # Low register (E2-E4)
            voices['tenor'] = active_pcs[1] + 60     # Tenor register (C3-A4)
            voices['alto'] = active_pcs[2] + 60      # Alto register (G3-D5)
            voices['soprano'] = active_pcs[3] + 72   # High register (C4-A5)
        elif len(active_pcs) == 3:
            # 3-part harmony (typical)
            voices['bass'] = active_pcs[0] + 48
            voices['tenor'] = active_pcs[1] + 60
            voices['soprano'] = active_pcs[2] + 72
            voices['alto'] = None
        elif len(active_pcs) == 2:
            # 2-part harmony
            voices['bass'] = active_pcs[0] + 48
            voices['soprano'] = active_pcs[1] + 72
            voices['tenor'] = None
            voices['alto'] = None
        else:
            # Single note
            voices['soprano'] = active_pcs[0] + 60
            voices['alto'] = None
            voices['tenor'] = None
            voices['bass'] = None
        
        return ChordVoicing(
            timestamp=timestamp,
            chord_name=chord_name,
            voices=voices,
            pitch_classes=active_pcs
        )
    
    def _calculate_voice_motion(self, prev: ChordVoicing, curr: ChordVoicing) -> Dict[str, Optional[int]]:
        """
        Calculate interval motion for each voice
        
        Returns: dict of {voice_name: interval_in_semitones}
        """
        motion = {}
        
        for voice_name in self.voice_names:
            prev_note = prev.voices.get(voice_name)
            curr_note = curr.voices.get(voice_name)
            
            if prev_note is not None and curr_note is not None:
                motion[voice_name] = curr_note - prev_note
            else:
                motion[voice_name] = None
        
        return motion
    
    def _detect_parallel_motion(self, prev: ChordVoicing, curr: ChordVoicing, 
                               motion: Dict[str, Optional[int]]) -> List[Tuple[str, str, str]]:
        """
        Detect parallel 5ths and octaves (voice leading violations)
        
        Parallel 5th: Two voices move by same interval AND form perfect 5th
        Parallel octave: Two voices move by same interval AND form octave
        
        Returns: List of (violation_type, voice1, voice2)
        """
        violations = []
        
        # Check all pairs of voices
        for voice1, voice2 in combinations(self.voice_names, 2):
            # Get motion for both voices
            motion1 = motion.get(voice1)
            motion2 = motion.get(voice2)
            
            # Both voices must have moved
            if motion1 is None or motion2 is None:
                continue
            
            # Get previous and current notes
            prev1 = prev.voices.get(voice1)
            prev2 = prev.voices.get(voice2)
            curr1 = curr.voices.get(voice1)
            curr2 = curr.voices.get(voice2)
            
            if None in [prev1, prev2, curr1, curr2]:
                continue
            
            # Calculate intervals between voices
            prev_interval = abs(prev1 - prev2) % 12
            curr_interval = abs(curr1 - curr2) % 12
            
            # Check if both voices moved in same direction by same interval (parallel motion)
            if motion1 == motion2 and motion1 != 0:
                # Parallel 5ths
                if prev_interval == 7 and curr_interval == 7:
                    violations.append(('parallel_5th', voice1, voice2))
                # Parallel octaves
                elif prev_interval == 0 and curr_interval == 0:
                    violations.append(('parallel_octave', voice1, voice2))
        
        return violations
    
    def _has_contrary_motion(self, motion: Dict[str, Optional[int]]) -> bool:
        """
        Check if there's contrary motion (voices moving in opposite directions)
        Contrary motion is good in classical voice leading
        """
        # Get valid motions
        valid_motions = [m for m in motion.values() if m is not None and m != 0]
        
        if len(valid_motions) < 2:
            return False
        
        # Check if some voices go up and others go down
        has_ascending = any(m > 0 for m in valid_motions)
        has_descending = any(m < 0 for m in valid_motions)
        
        return has_ascending and has_descending
    
    def _detect_voice_crossings(self, voicing: ChordVoicing) -> List[Dict]:
        """
        Detect voice crossings (soprano below alto, etc.)
        Voice crossing is generally avoided in classical harmony
        """
        crossings = []
        
        # Check each adjacent pair
        voice_pairs = [
            ('soprano', 'alto'),
            ('alto', 'tenor'),
            ('tenor', 'bass')
        ]
        
        for upper, lower in voice_pairs:
            upper_note = voicing.voices.get(upper)
            lower_note = voicing.voices.get(lower)
            
            if upper_note is not None and lower_note is not None:
                if upper_note < lower_note:  # Upper voice below lower voice!
                    crossings.append({
                        'timestamp': voicing.timestamp,
                        'chord': voicing.chord_name,
                        'upper_voice': upper,
                        'lower_voice': lower,
                        'upper_note': upper_note,
                        'lower_note': lower_note
                    })
        
        return crossings
    
    def _calculate_voice_ranges(self, voicings: List[ChordVoicing]) -> Dict[str, float]:
        """Calculate average range for each voice"""
        voice_notes = {name: [] for name in self.voice_names}
        
        for voicing in voicings:
            for voice_name, note in voicing.voices.items():
                if note is not None:
                    voice_notes[voice_name].append(note)
        
        ranges = {}
        for voice_name, notes in voice_notes.items():
            if notes:
                ranges[voice_name] = max(notes) - min(notes)
            else:
                ranges[voice_name] = 0.0
        
        return ranges
    
    def _calculate_voice_independence(self, voicings: List[ChordVoicing]) -> float:
        """
        Calculate how independent the voices are
        High independence = voices move differently
        Low independence = voices move together (parallel motion)
        
        Returns: Score from 0 (parallel) to 1 (independent)
        """
        if len(voicings) < 2:
            return 0.0
        
        # Track correlation between voice motions
        motion_vectors = {name: [] for name in self.voice_names}
        
        for i in range(len(voicings) - 1):
            motion = self._calculate_voice_motion(voicings[i], voicings[i+1])
            for voice_name, interval in motion.items():
                if interval is not None:
                    motion_vectors[voice_name].append(interval)
        
        # Calculate correlation between voice pairs
        correlations = []
        for v1, v2 in combinations(self.voice_names, 2):
            if motion_vectors[v1] and motion_vectors[v2]:
                # Pad to same length
                min_len = min(len(motion_vectors[v1]), len(motion_vectors[v2]))
                vec1 = np.array(motion_vectors[v1][:min_len])
                vec2 = np.array(motion_vectors[v2][:min_len])
                
                # Calculate correlation
                if np.std(vec1) > 0 and np.std(vec2) > 0:
                    corr = np.corrcoef(vec1, vec2)[0, 1]
                    correlations.append(abs(corr))
        
        # Independence = 1 - average correlation
        avg_correlation = np.mean(correlations) if correlations else 0.0
        independence = 1.0 - avg_correlation
        
        return float(independence)
    
    def _empty_analysis(self) -> VoiceLeadingAnalysis:
        """Return empty analysis for edge cases"""
        return VoiceLeadingAnalysis(
            smoothness_score=0.0,
            parallel_fifths=[],
            parallel_octaves=[],
            contrary_motion_count=0,
            voice_crossings=[],
            average_voice_range={},
            stepwise_motion_percentage=0.0,
            leap_percentage=0.0,
            voice_independence=0.0
        )
    
    def get_voice_leading_quality(self, analysis: VoiceLeadingAnalysis) -> str:
        """
        Assess overall voice leading quality
        
        Returns: 'excellent', 'good', 'fair', 'poor'
        """
        score = 0
        
        # Smoothness (stepwise motion is good)
        if analysis.smoothness_score < 2.0:
            score += 3  # Excellent
        elif analysis.smoothness_score < 3.0:
            score += 2  # Good
        elif analysis.smoothness_score < 4.0:
            score += 1  # Fair
        
        # No parallel 5ths/octaves (classical rules)
        if len(analysis.parallel_fifths) == 0:
            score += 2
        if len(analysis.parallel_octaves) == 0:
            score += 2
        
        # Contrary motion (good)
        if analysis.contrary_motion_count > 0:
            score += 1
        
        # Stepwise motion percentage
        if analysis.stepwise_motion_percentage > 60:
            score += 2
        elif analysis.stepwise_motion_percentage > 40:
            score += 1
        
        # Voice independence
        if analysis.voice_independence > 0.7:
            score += 2
        elif analysis.voice_independence > 0.5:
            score += 1
        
        # Classify based on score
        if score >= 10:
            return 'excellent'
        elif score >= 7:
            return 'good'
        elif score >= 4:
            return 'fair'
        else:
            return 'poor'


class BassLineAnalyzer:
    """
    Analyzes bass line characteristics and chord inversions
    """
    
    def __init__(self):
        self.note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        print("🎸 Bass Line Analyzer initialized")
    
    def analyze_bass_line(self, voicings: List[ChordVoicing], 
                         chord_progression: List[str]) -> Dict:
        """
        Analyze bass line characteristics
        
        Returns:
        - inversions: List of chord inversions
        - bass_contour: Overall bass line shape
        - pedal_points: Sustained bass notes
        - walking_bass: Whether bass exhibits walking pattern
        """
        if not voicings:
            return self._empty_bass_analysis()
        
        print(f"\n🎸 Analyzing bass line for {len(voicings)} chords...")
        
        # Extract bass notes
        bass_notes = []
        bass_timestamps = []
        for voicing in voicings:
            if voicing.voices.get('bass') is not None:
                bass_notes.append(voicing.voices['bass'])
                bass_timestamps.append(voicing.timestamp)
        
        if not bass_notes:
            return self._empty_bass_analysis()
        
        # Analyze inversions
        inversions = self._analyze_inversions(voicings, chord_progression)
        
        # Analyze bass contour
        bass_contour = self._analyze_bass_contour(bass_notes)
        
        # Detect pedal points
        pedal_points = self._detect_pedal_points(bass_notes, bass_timestamps)
        
        # Detect walking bass
        walking_bass = self._detect_walking_bass(bass_notes)
        
        print(f"✅ Bass line analysis complete")
        print(f"   Root position: {inversions['root_position_count']}/{len(inversions['inversions'])}")
        print(f"   Pedal points: {len(pedal_points)}")
        print(f"   Walking bass: {'Yes' if walking_bass else 'No'}")
        print(f"   Bass contour: {bass_contour['description']}")
        
        return {
            'inversions': inversions,
            'bass_contour': bass_contour,
            'pedal_points': pedal_points,
            'walking_bass': walking_bass,
            'bass_range': max(bass_notes) - min(bass_notes),
            'average_bass_note': np.mean(bass_notes)
        }
    
    def _analyze_inversions(self, voicings: List[ChordVoicing], 
                           chord_progression: List[str]) -> Dict:
        """Analyze chord inversions"""
        inversions = []
        root_position_count = 0
        first_inversion_count = 0
        second_inversion_count = 0
        slash_chord_count = 0
        
        for voicing, chord_name in zip(voicings, chord_progression):
            bass_note = voicing.voices.get('bass')
            if bass_note is None:
                inversions.append('unknown')
                continue
            
            bass_pc = bass_note % 12
            
            # Extract root from chord name (first letter, handle sharps)
            root_str = chord_name[0]
            if len(chord_name) > 1 and chord_name[1] == '#':
                root_str = chord_name[:2]
            
            try:
                root_pc = self.note_names.index(root_str)
            except ValueError:
                inversions.append('unknown')
                continue
            
            # Determine inversion
            interval = (bass_pc - root_pc) % 12
            
            if interval == 0:
                inversions.append('root')
                root_position_count += 1
            elif interval == 3 or interval == 4:  # Minor or major 3rd
                inversions.append('first')
                first_inversion_count += 1
            elif interval == 7:  # Perfect 5th
                inversions.append('second')
                second_inversion_count += 1
            else:
                inversions.append('slash')  # Non-chord tone in bass
                slash_chord_count += 1
        
        return {
            'inversions': inversions,
            'root_position_count': root_position_count,
            'first_inversion_count': first_inversion_count,
            'second_inversion_count': second_inversion_count,
            'slash_chord_count': slash_chord_count
        }
    
    def _analyze_bass_contour(self, bass_notes: List[int]) -> Dict:
        """Analyze overall bass line shape"""
        if len(bass_notes) < 2:
            return {'description': 'static', 'direction': 'none'}
        
        # Calculate overall trend
        ascending_count = sum(1 for i in range(len(bass_notes)-1) if bass_notes[i+1] > bass_notes[i])
        descending_count = sum(1 for i in range(len(bass_notes)-1) if bass_notes[i+1] < bass_notes[i])
        
        total = ascending_count + descending_count
        if total == 0:
            return {'description': 'static', 'direction': 'none'}
        
        ascending_pct = ascending_count / total
        
        if ascending_pct > 0.6:
            description = 'ascending'
        elif ascending_pct < 0.4:
            description = 'descending'
        else:
            description = 'mixed'
        
        return {
            'description': description,
            'direction': 'up' if ascending_pct > 0.5 else 'down',
            'ascending_percentage': ascending_pct * 100,
            'descending_percentage': (1 - ascending_pct) * 100
        }
    
    def _detect_pedal_points(self, bass_notes: List[int], timestamps: List[float]) -> List[Dict]:
        """
        Detect pedal points (sustained bass notes)
        A pedal point is a bass note held for 3+ chords
        """
        pedal_points = []
        
        if len(bass_notes) < 3:
            return pedal_points
        
        i = 0
        while i < len(bass_notes) - 2:
            current_note = bass_notes[i]
            # Count consecutive occurrences
            count = 1
            while i + count < len(bass_notes) and bass_notes[i + count] == current_note:
                count += 1
            
            # Pedal point = 3+ consecutive chords with same bass
            if count >= 3:
                pedal_points.append({
                    'note': current_note,
                    'note_name': self.note_names[current_note % 12],
                    'start_time': timestamps[i],
                    'end_time': timestamps[i + count - 1],
                    'duration_chords': count
                })
            
            i += count
        
        return pedal_points
    
    def _detect_walking_bass(self, bass_notes: List[int]) -> bool:
        """
        Detect walking bass pattern (characteristic stepwise motion)
        Walking bass: mostly stepwise motion (1-2 semitones)
        """
        if len(bass_notes) < 4:
            return False
        
        # Calculate intervals
        intervals = [abs(bass_notes[i+1] - bass_notes[i]) for i in range(len(bass_notes)-1)]
        
        # Walking bass: >70% stepwise motion (1-2 semitones)
        stepwise = sum(1 for i in intervals if i <= 2)
        stepwise_percentage = stepwise / len(intervals)
        
        return stepwise_percentage > 0.7
    
    def _empty_bass_analysis(self) -> Dict:
        """Return empty bass analysis"""
        return {
            'inversions': {'inversions': [], 'root_position_count': 0},
            'bass_contour': {'description': 'none', 'direction': 'none'},
            'pedal_points': [],
            'walking_bass': False,
            'bass_range': 0,
            'average_bass_note': 0
        }

