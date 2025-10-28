#!/usr/bin/env python3
"""
Ratio-Based Chord Validator
============================

Enhanced chord validator using mathematical frequency ratios
instead of note name matching.

Key improvements over simple_chord_validator.py:
1. Analyzes frequency ratios (e.g., 4:5:6 for major) instead of note names
2. Calculates quantitative consonance scores
3. More robust to tuning variations
4. Works with any tuning system (not just 12-TET)
5. Provides educational insights into WHY chords sound the way they do

Usage:
    python ratio_based_chord_validator.py [--input-device N]
"""

import numpy as np
import time
import json
import mido
from datetime import datetime
from typing import List, Optional, Dict
from collections import defaultdict
from dataclasses import asdict

from listener.jhs_listener_core import DriftListener, Event
from listener.harmonic_context import RealtimeHarmonicDetector
from listener.harmonic_chroma import HarmonicAwareChromaExtractor
from listener.ratio_analyzer import FrequencyRatioAnalyzer, ChordAnalysis
import librosa


class RatioBasedChordValidator:
    """
    Chord validator using mathematical frequency ratios
    instead of note name matching
    """
    
    def __init__(self,
                 midi_output_port: str = "IAC Driver Chord Trainer Output",
                 audio_input_device: Optional[int] = None,
                 chord_duration: float = 2.5,
                 consonance_threshold: float = 0.60,
                 snap_to_notes: bool = True):
        
        self.midi_port_name = midi_output_port
        self.audio_device = audio_input_device
        self.chord_duration = chord_duration
        self.consonance_threshold = consonance_threshold
        self.snap_to_notes = snap_to_notes
        
        # MIDI
        self.midi_port = None
        
        # Audio
        self.listener = None
        self.sample_rate = 44100
        
        # Polyphonic detection (chroma-based)
        self.harmonic_detector = RealtimeHarmonicDetector()
        
        # Harmonic-aware chroma extraction (NEW - research-based!)
        self.chroma_extractor = HarmonicAwareChromaExtractor()
        
        # Ratio analysis (more tolerant for live mic input)
        self.ratio_analyzer = FrequencyRatioAnalyzer(tolerance=0.08)
        
        # Audio buffer for chroma analysis
        self.audio_buffer = []
        self.collecting = False
        
        # Results
        self.validation_results = []
    
    def _open_midi_port(self) -> bool:
        """Open MIDI port"""
        try:
            available = mido.get_output_names()
            if self.midi_port_name in available:
                self.midi_port = mido.open_output(self.midi_port_name)
                print(f"✅ MIDI: {self.midi_port_name}")
                return True
            return False
        except Exception as e:
            print(f"❌ MIDI error: {e}")
            return False
    
    def _start_listener(self) -> bool:
        """Start audio capture for polyphonic analysis"""
        try:
            import sounddevice as sd
            
            # Just capture raw audio for chroma analysis
            self.stream = sd.InputStream(
                device=self.audio_device,
                channels=1,
                samplerate=self.sample_rate,
                callback=self._audio_callback
            )
            self.stream.start()
            print(f"✅ Audio capture started (device {self.audio_device}) for chroma-based detection")
            return True
        except Exception as e:
            print(f"❌ Listener error: {e}")
            return False
    
    def _audio_callback(self, indata, frames, time_info, status):
        """
        Collect raw audio for chroma analysis
        (Chroma works with polyphonic audio, YIN does not!)
        """
        if self.collecting:
            self.audio_buffer.extend(indata[:, 0].copy())
    
    def _extract_frequencies_with_guidance(self, audio: np.ndarray, 
                                           expected_freqs: List[float]) -> List[float]:
        """
        Extract frequencies using GUIDED SEARCH (supervised learning)
        
        Since we KNOW what we sent (ground truth), we should:
        1. Search for peaks NEAR expected frequencies
        2. Validate using expected ratios
        3. Learn what "C major" looks like in this environment
        
        This is reinforcement learning for chord detection!
        """
        if len(audio) < 2048:
            return []
        
        # Use guided search with ground truth
        result = self.chroma_extractor.analyze_with_guided_search(
            audio,
            expected_freqs=expected_freqs,
            sr=self.sample_rate,
            tolerance_cents=50.0  # ±50 cents = ~half semitone tolerance
        )
        
        print(f"   Guided frequency search:")
        print(f"      Expected: {len(expected_freqs)} frequencies")
        print(f"      Detected: {result.get('num_detected', 0)} frequencies")
        
        if result.get('success', False):
            print(f"      ✓ All frequencies found!")
            for i, (exp, det, conf, err) in enumerate(zip(
                result['expected_frequencies'],
                result['detected_frequencies'],
                result['peak_confidences'],
                result['frequency_errors_cents']
            )):
                note = result['note_names'][i]
                print(f"         {note}: expected {exp:.2f} Hz, found {det:.2f} Hz "
                      f"(error: {err:+.1f} cents, confidence: {conf:.2%})")
        else:
            print(f"      ⚠ Only found {result.get('num_detected', 0)}/{len(expected_freqs)} frequencies")
            if result.get('num_detected', 0) > 0:
                for i, (exp, det, conf) in enumerate(zip(
                    result.get('expected_frequencies', []),
                    result.get('detected_frequencies', []),
                    result.get('peak_confidences', [])
                )):
                    print(f"         Expected {exp:.2f} Hz → found {det:.2f} Hz (conf: {conf:.2%})")
        
        return list(result['detected_frequencies'])
    
    def _cluster_frequencies(self, raw_freqs: List[float]) -> List[float]:
        """
        Cluster detected frequencies to find stable fundamentals
        Accounts for vibrato, tuning variations, etc.
        
        Returns:
            List of representative frequencies (one per detected note)
        """
        if not raw_freqs:
            return []
        
        # Simple clustering: group frequencies within 5% of each other (more tolerant)
        tolerance = 0.05
        clusters = []
        
        sorted_freqs = sorted(raw_freqs)
        
        for freq in sorted_freqs:
            # Find cluster this belongs to
            found_cluster = False
            for cluster in clusters:
                cluster_mean = np.mean(cluster)
                if abs(freq - cluster_mean) / cluster_mean < tolerance:
                    cluster.append(freq)
                    found_cluster = True
                    break
            
            if not found_cluster:
                clusters.append([freq])
        
        # Return mean of each cluster (needs at least 3 detections - less strict)
        representative_freqs = []
        for i, cluster in enumerate(clusters):
            if len(cluster) >= 3:  # Require at least 3 stable detections
                representative_freqs.append(np.mean(cluster))
                print(f"      Cluster {i+1}: {len(cluster)} samples, mean = {np.mean(cluster):.2f} Hz")
            else:
                print(f"      Cluster {i+1}: Only {len(cluster)} samples (rejected)")
        
        return representative_freqs
    
    def validate_chord(self, sent_midi: List[int], chord_name: str, 
                      expected_type: str = None) -> Dict:
        """
        Validate a chord using ratio analysis
        
        Args:
            sent_midi: MIDI notes sent
            chord_name: Name of chord (for reference)
            expected_type: Expected chord type (e.g., 'major', 'minor')
            
        Returns:
            Comprehensive validation dictionary with ratio analysis
        """
        print(f"\n🎹 Validating: {chord_name}")
        print(f"   Sending MIDI: {sent_midi} ({[self._midi_to_note(n) for n in sent_midi]})")
        
        # Calculate expected frequencies
        expected_freqs = [self._midi_to_freq(m) for m in sent_midi]
        print(f"   Expected frequencies: {[f'{f:.2f} Hz' for f in expected_freqs]}")
        
        # Reset audio collection
        self.audio_buffer = []
        self.collecting = True
        
        # Play chord
        for note in sent_midi:
            self.midi_port.send(mido.Message('note_on', channel=0, note=note, velocity=80))
        
        # Collect audio for full duration
        time.sleep(self.chord_duration)
        
        # Stop chord
        for note in sent_midi:
            self.midi_port.send(mido.Message('note_off', channel=0, note=note, velocity=0))
        
        self.collecting = False
        
        # Convert audio buffer to numpy array
        if not self.audio_buffer:
            print(f"   ❌ No audio captured!")
            return {
                'chord_name': chord_name,
                'sent_midi': sent_midi,
                'validation_passed': False,
                'error': 'No audio captured'
            }
        
        audio_array = np.array(self.audio_buffer, dtype=np.float32)
        print(f"   Captured {len(audio_array)} audio samples ({len(audio_array)/self.sample_rate:.2f}s)")
        
        # Extract frequencies using GUIDED SEARCH (supervised learning!)
        # We KNOW what we sent, so use it to guide detection
        detected_freqs = self._extract_frequencies_with_guidance(audio_array, expected_freqs)
        
        print(f"   Clustered into {len(detected_freqs)} stable frequencies:")
        for f in detected_freqs:
            print(f"      {f:.2f} Hz ({self._freq_to_note_name(f)})")
        
        # Perform ratio analysis
        analysis = self.ratio_analyzer.analyze_frequencies(detected_freqs)
        
        if analysis is None:
            print(f"   ❌ Analysis failed (not enough frequencies)")
            return {
                'chord_name': chord_name,
                'sent_midi': sent_midi,
                'detected_frequencies': detected_freqs,
                'validation_passed': False,
                'error': 'Analysis failed'
            }
        
        # Display analysis results
        print(f"\n   📊 Ratio Analysis:")
        print(f"      Fundamental: {analysis.fundamental:.2f} Hz")
        print(f"      Ratios: {[f'{r:.3f}' for r in analysis.ratios]}")
        print(f"      Simplified: {analysis.simplified_ratios}")
        print(f"      Chord type: {analysis.chord_match['type']} "
              f"(confidence: {analysis.chord_match['confidence']:.1%})")
        print(f"      Consonance: {analysis.consonance_score:.3f}")
        
        # Validation criteria
        criteria = self._evaluate_validation(
            analysis, 
            expected_type, 
            expected_freqs, 
            detected_freqs,
            sent_midi
        )
        
        validation_passed = criteria['overall_pass']
        
        if validation_passed:
            print(f"   ✅ VALIDATED!")
        else:
            print(f"   ❌ Failed: {criteria['failure_reason']}")
        
        # Build comprehensive result
        result = {
            'timestamp': datetime.now().isoformat(),
            'chord_name': chord_name,
            'sent_midi': sent_midi,
            'expected_frequencies': expected_freqs,
            'detected_frequencies': detected_freqs,
            'num_audio_samples': len(audio_array),
            'audio_duration_seconds': len(audio_array) / self.sample_rate,
            'ratio_analysis': {
                'fundamental': analysis.fundamental,
                'ratios': analysis.ratios,
                'simplified_ratios': [list(r) for r in analysis.simplified_ratios],
                'chord_type': analysis.chord_match['type'],
                'confidence': analysis.chord_match['confidence'],
                'consonance_score': analysis.consonance_score,
                'description': analysis.chord_match['description'],
                'intervals': [
                    {
                        'freq1': i.freq1,
                        'freq2': i.freq2,
                        'ratio': i.ratio,
                        'simplified': list(i.simplified),
                        'interval_name': i.interval_name,
                        'consonance': i.consonance,
                        'cents': i.cents
                    }
                    for i in analysis.intervals
                ]
            },
            'validation_criteria': criteria,
            'validation_passed': validation_passed
        }
        
        self.validation_results.append(result)
        
        return result
    
    def _evaluate_validation(self, analysis: ChordAnalysis, 
                           expected_type: Optional[str],
                           expected_freqs: List[float],
                           detected_freqs: List[float],
                           sent_midi: List[int]) -> Dict:
        """
        Evaluate multiple validation criteria
        
        Returns dictionary with pass/fail for each criterion
        """
        criteria = {
            'correct_num_notes': False,
            'pitch_class_match': False,
            'chord_type_match': False,
            'consonance_adequate': False,
            'frequency_match': False,
            'overall_pass': False,
            'failure_reason': ''
        }
        
        # 1. Check number of detected notes
        criteria['correct_num_notes'] = (len(detected_freqs) == len(expected_freqs))
        
        # 2. Check pitch classes match (more robust than exact frequency for live input)
        if len(detected_freqs) == len(expected_freqs):
            expected_pcs = set([int(round(69 + 12 * np.log2(f / 440.0))) % 12 
                               for f in expected_freqs])
            detected_pcs = set([int(round(69 + 12 * np.log2(f / 440.0))) % 12 
                               for f in detected_freqs])
            criteria['pitch_class_match'] = (expected_pcs == detected_pcs)
        
        # 3. Check chord type match (if expected type provided)
        if expected_type:
            detected_type = analysis.chord_match['type']
            type_confidence = analysis.chord_match['confidence']
            criteria['chord_type_match'] = (
                detected_type == expected_type and 
                type_confidence > 0.5  # More lenient for live input
            )
        else:
            criteria['chord_type_match'] = True  # Skip if not specified
        
        # 4. Check consonance is adequate (more lenient for live)
        criteria['consonance_adequate'] = (
            analysis.consonance_score >= max(0.50, self.consonance_threshold - 0.15)
        )
        
        # 5. Check frequency matching (within 10% for live mic)
        if len(detected_freqs) == len(expected_freqs):
            freq_errors = []
            for exp, det in zip(sorted(expected_freqs), sorted(detected_freqs)):
                error = abs(det - exp) / exp
                freq_errors.append(error)
            
            avg_error = np.mean(freq_errors)
            criteria['frequency_match'] = (avg_error < 0.10)  # 10% tolerance for live
        
        # Overall pass requires key criteria
        # For live input, prioritize pitch class matching over exact chord type
        all_pass = (
            criteria['correct_num_notes'] and
            criteria['pitch_class_match'] and  # Most important for supervised learning!
            (criteria['chord_type_match'] or criteria['consonance_adequate'])  # Either/or
        )
        
        criteria['overall_pass'] = all_pass
        
        # Determine failure reason if failed
        if not all_pass:
            reasons = []
            if not criteria['correct_num_notes']:
                reasons.append(f"wrong number of notes ({len(detected_freqs)} vs {len(expected_freqs)})")
            if not criteria['pitch_class_match']:
                reasons.append(f"pitch class mismatch")
            if not criteria['chord_type_match']:
                reasons.append(f"type mismatch (detected {analysis.chord_match['type']})")
            if not criteria['consonance_adequate']:
                reasons.append(f"low consonance ({analysis.consonance_score:.2f})")
            
            criteria['failure_reason'] = ', '.join(reasons)
        
        return criteria
    
    def _midi_to_freq(self, midi: int) -> float:
        """Convert MIDI note to frequency (Hz)"""
        return 440.0 * (2 ** ((midi - 69) / 12.0))
    
    def _snap_to_note(self, freq: float) -> float:
        """
        Snap frequency to nearest musical note (12-TET)
        Helps reduce noise from imperfect pitch detection
        """
        midi = 69 + 12 * np.log2(freq / 440.0)
        midi_rounded = int(round(midi))
        freq_snapped = 440.0 * (2 ** ((midi_rounded - 69) / 12.0))
        return freq_snapped
    
    def _freq_to_note_name(self, freq: float) -> str:
        """Convert frequency to note name (approximate)"""
        midi = int(round(69 + 12 * np.log2(freq / 440.0)))
        return self._midi_to_note(midi)
    
    def _midi_to_note(self, midi: int) -> str:
        """Convert MIDI to note name"""
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        return f"{notes[midi % 12]}{midi // 12 - 1}"
    
    def run_test(self):
        """Run comprehensive test suite with inversions"""
        print("\n🎯 Ratio-Based Chord Validation Test")
        print("=" * 80)
        
        # Comprehensive chord vocabulary for training
        from autonomous_chord_trainer import ChordVocabulary
        
        # All chromatic roots
        chromatic_roots = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        # Define chord types to train for each root
        chord_types = [
            ('', 'major'),           # Major triad
            ('m', 'minor'),          # Minor triad
            ('7', 'dom7'),           # Dominant 7th
            ('maj7', 'maj7'),        # Major 7th
            ('m7', 'min7'),          # Minor 7th
            ('dim', 'diminished'),   # Diminished triad
            ('dim7', 'dim7'),        # Diminished 7th
            ('m7b5', 'm7b5'),        # Half-diminished
            ('aug', 'augmented'),    # Augmented triad
            ('sus4', 'sus4'),        # Suspended 4th
            ('sus2', 'sus2'),        # Suspended 2nd
            ('9', 'dom9'),           # Dominant 9th
            ('maj9', 'maj9'),        # Major 9th
            ('m9', 'min9'),          # Minor 9th
        ]
        
        # Build comprehensive vocabulary
        chord_vocabulary = []
        for root in chromatic_roots:
            for quality, expected_type in chord_types:
                chord_vocabulary.append((root, quality, expected_type))
        
        test_chords = []
        
        # Generate 4 inversions per chord (root + 1st + 2nd + 3rd when applicable)
        for root, quality, expected_type in chord_vocabulary:
            root_notes = ChordVocabulary.get_chord_notes(root, quality, base_octave=4)
            num_notes = len(root_notes)
            
            # Test up to 3 inversions (or fewer for 3-note chords)
            max_inversions = min(3, num_notes - 1)
            
            for inv_num in range(max_inversions + 1):  # root + inversions
                notes = root_notes.copy()
                
                # Rotate for inversion
                for _ in range(inv_num):
                    bottom_note = notes.pop(0)
                    notes.append(bottom_note + 12)
                
                inv_name = ["root", "1st inv", "2nd inv", "3rd inv"][inv_num]
                chord_name = f"{root}{quality} {inv_name}"
                test_chords.append((notes, chord_name, expected_type))
        
        print(f"\n📋 Testing {len(test_chords)} chord variations")
        print(f"   Chord types: {len(chord_vocabulary)}")
        print(f"   With inversions: ~{len(test_chords) / len(chord_vocabulary):.1f} per chord\n")
        
        passed = 0
        detailed_results = []
        
        for idx, (midi_notes, name, expected_type) in enumerate(test_chords, 1):
            print(f"\n[{idx}/{len(test_chords)}] ", end="")
            result = self.validate_chord(midi_notes, name, expected_type)
            
            if result['validation_passed']:
                passed += 1
            
            # Handle both successful and failed analyses
            if 'ratio_analysis' in result and result['validation_passed']:
                detailed_results.append({
                    'name': name,
                    'passed': True,
                    'consonance': result['ratio_analysis']['consonance_score'],
                    'detected_type': result['ratio_analysis']['chord_type'],
                    'confidence': result['ratio_analysis']['confidence']
                })
            else:
                # Analysis failed or validation failed
                detailed_results.append({
                    'name': name,
                    'passed': False,
                    'consonance': 0.0,
                    'detected_type': 'failed',
                    'confidence': 0.0
                })
            
            time.sleep(0.5)  # Brief pause between chords
        
        # Summary
        print(f"\n{'=' * 80}")
        print(f"📊 RESULTS SUMMARY")
        print(f"{'=' * 80}")
        print(f"✅ Passed: {passed}/{len(test_chords)} ({100*passed/len(test_chords):.0f}%)")
        print(f"\nDetailed Results:")
        print(f"{'Chord':<30} {'Status':<10} {'Consonance':<12} {'Type':<15} {'Confidence'}")
        print(f"{'-'*80}")
        for r in detailed_results:
            status = "✅ PASS" if r['passed'] else "❌ FAIL"
            print(f"{r['name']:<30} {status:<10} {r['consonance']:<12.3f} "
                  f"{r['detected_type']:<15} {r['confidence']:.1%}")
        
        # Save results
        self._save_results()
    
    def _save_results(self):
        """Save validation results to JSON file"""
        filename = f"validation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert numpy types to native Python for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_numpy(item) for item in obj)
            return obj
        
        with open(filename, 'w') as f:
            json.dump(convert_numpy(self.validation_results), f, indent=2)
        print(f"\n💾 Results saved to: {filename}")
    
    def start(self) -> bool:
        """Start system"""
        if not self._open_midi_port():
            return False
        if not self._start_listener():
            return False
        print("✅ System ready!\n")
        return True
    
    def stop(self):
        """Stop system"""
        if hasattr(self, 'stream') and self.stream:
            self.stream.stop()
            self.stream.close()
        if self.midi_port:
            self.midi_port.close()


def list_audio_devices():
    """List all available audio input devices"""
    import sounddevice as sd
    
    print("\n" + "=" * 70)
    print("Available Audio Input Devices:")
    print("=" * 70)
    
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:  # Has input
            default = " (DEFAULT)" if i == sd.default.device[0] else ""
            print(f"{i}: {device['name']}{default}")
            print(f"   Input channels: {device['max_input_channels']}")
            print(f"   Sample rate: {device['default_samplerate']} Hz")
    
    print("=" * 70)
    print("\nUsage: python ratio_based_chord_validator.py --input-device N")
    print("(where N is the device number for your microphone/audio interface)\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Ratio-based chord validator using mathematical frequency analysis"
    )
    parser.add_argument('--list-devices', action='store_true',
                       help='List available audio input devices and exit')
    parser.add_argument('--input-device', type=int, default=None,
                       help='Audio input device index (use --list-devices to see options)')
    parser.add_argument('--duration', type=float, default=2.5,
                       help='Chord duration in seconds')
    parser.add_argument('--consonance-threshold', type=float, default=0.60,
                       help='Minimum consonance score for validation')
    parser.add_argument('--no-snap', action='store_true',
                       help='Disable snapping to musical notes (use raw frequencies)')
    
    args = parser.parse_args()
    
    # List devices and exit
    if args.list_devices:
        list_audio_devices()
        return
    
    validator = RatioBasedChordValidator(
        audio_input_device=args.input_device,
        chord_duration=args.duration,
        consonance_threshold=args.consonance_threshold,
        snap_to_notes=not args.no_snap
    )
    
    if not validator.start():
        return
    
    try:
        validator.run_test()
    finally:
        validator.stop()


if __name__ == "__main__":
    main()

