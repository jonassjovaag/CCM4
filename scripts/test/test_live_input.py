#!/usr/bin/env python3
"""
Test Live Audio Input and Pitch Detection

Tests the complete live input chain:
1. Audio device selection
2. Real-time capture
3. YIN pitch detection
4. Note display

Run this to verify your microphone/input is working before
running the full MusicHal_9000 system.

Usage:
    python scripts/test/test_live_input.py
    python scripts/test/test_live_input.py --device 3
    python scripts/test/test_live_input.py --duration 30
"""

import sys
import argparse
import time
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    import sounddevice as sd
except ImportError:
    print("ERROR: sounddevice not installed")
    print("Run: pip install sounddevice")
    sys.exit(1)


# Note names for display
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def midi_to_note(midi: float) -> str:
    """Convert MIDI number to note name."""
    midi_int = int(round(midi))
    note = NOTE_NAMES[midi_int % 12]
    octave = (midi_int // 12) - 1
    cents = int((midi - midi_int) * 100)
    if cents >= 0:
        return f"{note}{octave} +{cents:02d}c"
    else:
        return f"{note}{octave} {cents:03d}c"


def freq_to_midi(freq: float) -> float:
    """Convert frequency to MIDI note number."""
    if freq <= 0:
        return 0
    return 69 + 12 * np.log2(freq / 440.0)


class LivePitchDetector:
    """Real-time pitch detection using YIN algorithm."""

    def __init__(self, sr: int = 44100, frame_size: int = 2048):
        self.sr = sr
        self.frame_size = frame_size
        self.fmin = 80.0
        self.fmax = 1000.0
        self.yin_threshold = 0.15

        # Smoothing
        self.f_smooth = 0.0
        self.alpha = 0.3

        # Stats
        self.note_history = []
        self.onset_count = 0
        self.last_midi = 0

    def _parabolic_min(self, y: np.ndarray, i: int) -> float:
        """Parabolic interpolation for more accurate pitch."""
        if i <= 0 or i >= len(y) - 1:
            return float(i)
        y0, y1, y2 = y[i-1], y[i], y[i+1]
        denom = (y0 - 2*y1 + y2)
        if abs(denom) < 1e-12:
            return float(i)
        return float(i + 0.5*(y0 - y2)/denom)

    def detect_pitch(self, audio: np.ndarray) -> tuple:
        """
        Detect pitch using YIN algorithm.

        Returns: (frequency, midi, confidence)
        """
        x = audio.astype(np.float32)
        N = len(x)

        # Apply window
        xw = x * np.hanning(N).astype(np.float32)
        xw = xw - np.mean(xw)

        # Autocorrelation via FFT
        L = 1 << (N*2 - 1).bit_length()
        X = np.fft.rfft(xw, n=L)
        ac = np.fft.irfft(X * np.conj(X), n=L)[:N]
        ac = ac / (float(np.max(ac)) + 1e-12)

        # Difference function
        d_full = 2.0 - 2.0*ac
        d = d_full[:N//2].astype(np.float32)

        # Cumulative mean normalized difference function
        cmndf = np.empty_like(d)
        cmndf[0] = 1.0
        csum = 0.0
        for tau in range(1, len(d)):
            csum += float(d[tau])
            cmndf[tau] = d[tau] * tau / (csum + 1e-9)

        # Find first dip below threshold
        tau = None
        low = max(2, int(self.sr / self.fmax))
        for i in range(low, len(d)):
            if cmndf[i] < self.yin_threshold:
                tau = self._parabolic_min(cmndf, i)
                break

        if tau is None:
            # Fallback to global minimum
            i = int(np.argmin(cmndf[low:])) + low
            tau = self._parabolic_min(cmndf, i)

        # Convert to frequency
        f0 = self.sr / max(1e-9, float(tau))

        # Validate range
        if not (self.fmin <= f0 <= self.fmax):
            return 0.0, 0.0, 0.0

        # Calculate confidence (inverse of CMNDF at detected lag)
        confidence = 1.0 - min(1.0, cmndf[int(tau)])

        # Smooth frequency
        if self.f_smooth == 0:
            self.f_smooth = f0
        else:
            self.f_smooth = (1 - self.alpha) * self.f_smooth + self.alpha * f0

        midi = freq_to_midi(self.f_smooth)

        return self.f_smooth, midi, confidence

    def process_buffer(self, audio: np.ndarray) -> dict:
        """Process audio buffer and return detection result."""
        # Calculate RMS level
        rms = np.sqrt(np.mean(audio**2))
        level_db = 20 * np.log10(rms + 1e-10)

        # Skip if too quiet
        if level_db < -50:
            return {
                'detected': False,
                'level_db': level_db,
                'reason': 'Too quiet'
            }

        # Detect pitch
        freq, midi, confidence = self.detect_pitch(audio)

        if freq <= 0:
            return {
                'detected': False,
                'level_db': level_db,
                'reason': 'No pitch detected'
            }

        # Track note changes (onsets)
        midi_rounded = round(midi)
        if midi_rounded != self.last_midi:
            self.onset_count += 1
            self.last_midi = midi_rounded
            self.note_history.append(midi_rounded)
            if len(self.note_history) > 100:
                self.note_history.pop(0)

        return {
            'detected': True,
            'frequency': freq,
            'midi': midi,
            'note': midi_to_note(midi),
            'confidence': confidence,
            'level_db': level_db
        }


def list_devices():
    """List available audio input devices."""
    print("\nAvailable Audio Input Devices:")
    print("=" * 50)

    devices = sd.query_devices()
    input_devices = []

    for i, device in enumerate(devices):
        if isinstance(device, dict) and device.get('max_input_channels', 0) > 0:
            name = device.get('name', 'Unknown')
            sr = device.get('default_samplerate', 0)
            marker = " [DEFAULT]" if i == sd.default.device[0] else ""
            print(f"  {i}: {name}{marker}")
            print(f"      Sample rate: {sr} Hz")
            input_devices.append(i)

    print()
    return input_devices


def run_live_test(device_id=None, duration=10.0, sr=44100):
    """Run live pitch detection test."""
    frame_size = 2048
    hop_size = 512

    detector = LivePitchDetector(sr=sr, frame_size=frame_size)

    # Get device name
    if device_id is not None:
        device_info = sd.query_devices(device_id)
        device_name = device_info['name']
    else:
        device_name = "Default"

    print(f"\nStarting live input test")
    print(f"  Device: {device_name}")
    print(f"  Duration: {duration} seconds")
    print(f"  Sample rate: {sr} Hz")
    print(f"  Frame size: {frame_size} samples")
    print()
    print("Play some notes! Press Ctrl+C to stop early.")
    print("=" * 60)
    print()

    # Collection buffers
    buffer = np.zeros(0, dtype=np.float32)
    results = []
    start_time = time.time()
    last_print = 0

    def audio_callback(indata, frames, time_info, status):
        nonlocal buffer
        if status:
            print(f"Status: {status}")
        buffer = np.concatenate([buffer, indata[:, 0]])

    try:
        with sd.InputStream(
            device=device_id,
            channels=1,
            samplerate=sr,
            blocksize=hop_size,
            dtype='float32',
            callback=audio_callback
        ):
            while time.time() - start_time < duration:
                # Process when we have enough samples
                while len(buffer) >= frame_size:
                    frame = buffer[:frame_size]
                    buffer = buffer[hop_size:]

                    result = detector.process_buffer(frame)
                    results.append(result)

                    # Print detected notes (throttled)
                    now = time.time()
                    if result['detected'] and (now - last_print) > 0.1:
                        elapsed = now - start_time
                        print(f"[{elapsed:5.1f}s] {result['note']:12s} "
                              f"{result['frequency']:7.1f} Hz  "
                              f"conf: {result['confidence']:.0%}  "
                              f"level: {result['level_db']:5.1f} dB")
                        last_print = now

                time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n\nStopped by user")

    # Print summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    detected = [r for r in results if r['detected']]
    total = len(results)

    print(f"Total frames processed: {total}")
    print(f"Frames with pitch: {len(detected)} ({len(detected)/total*100:.1f}%)")
    print(f"Note changes detected: {detector.onset_count}")

    if detector.note_history:
        unique_notes = set(detector.note_history)
        print(f"Unique notes played: {len(unique_notes)}")

        # Most common notes
        from collections import Counter
        note_counts = Counter(detector.note_history)
        print("\nMost played notes:")
        for midi, count in note_counts.most_common(5):
            note = NOTE_NAMES[midi % 12]
            octave = (midi // 12) - 1
            print(f"  {note}{octave}: {count} times")

    if detected:
        freqs = [r['frequency'] for r in detected]
        print(f"\nPitch range: {min(freqs):.1f} - {max(freqs):.1f} Hz")

        midis = [r['midi'] for r in detected]
        print(f"MIDI range: {min(midis):.0f} - {max(midis):.0f}")

        levels = [r['level_db'] for r in detected]
        print(f"Level range: {min(levels):.1f} to {max(levels):.1f} dB")

    return len(detected) > 0


def main():
    parser = argparse.ArgumentParser(
        description="Test live audio input and pitch detection"
    )
    parser.add_argument(
        '--device', '-d',
        type=int,
        default=None,
        help="Audio input device ID (use --list to see devices)"
    )
    parser.add_argument(
        '--duration', '-t',
        type=float,
        default=10.0,
        help="Test duration in seconds (default: 10)"
    )
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help="List available audio devices and exit"
    )
    parser.add_argument(
        '--sample-rate', '-sr',
        type=int,
        default=44100,
        help="Sample rate (default: 44100)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("LIVE AUDIO INPUT TEST")
    print("=" * 60)

    # Always list devices first
    input_devices = list_devices()

    if args.list:
        return 0

    if not input_devices:
        print("ERROR: No input devices found!")
        return 1

    # Validate device selection
    if args.device is not None and args.device not in input_devices:
        print(f"ERROR: Device {args.device} is not a valid input device")
        print(f"Valid devices: {input_devices}")
        return 1

    # Run test
    success = run_live_test(
        device_id=args.device,
        duration=args.duration,
        sr=args.sample_rate
    )

    print()
    if success:
        print("PASS: Live input is working!")
        print("Your audio chain is ready for MusicHal_9000")
    else:
        print("WARNING: No pitched audio detected")
        print("Check:")
        print("  - Is your microphone/input active?")
        print("  - Is the input level high enough?")
        print("  - Try a different device with --device N")

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
