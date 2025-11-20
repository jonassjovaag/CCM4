# SPDX-License-Identifier: Apache-2.0
# jhs_listener_core.py
# Adapted from JHS Listener for Drift Engine AI system
# Core real-time pitch tracking with enhanced features

from __future__ import annotations
import math, time, threading, queue
from collections import deque, defaultdict
from typing import Callable, Optional, Dict, Tuple, List
import numpy as np
import librosa
import warnings

# Suppress librosa warnings about small buffer sizes
warnings.filterwarnings('ignore', message='n_fft=.*is too large for input signal')

try:
    import sounddevice as sd
    _HAVE_SD = True
except Exception:
    _HAVE_SD = False

# Import harmonic context detector
try:
    from listener.harmonic_context import RealtimeHarmonicDetector, HarmonicContext
    _HAVE_HARMONIC = True
except ImportError:
    _HAVE_HARMONIC = False

# Import rhythmic context detector
try:
    from listener.rhythmic_context import RealtimeRhythmicDetector, RhythmicContext
    _HAVE_RHYTHMIC = True
except ImportError:
    _HAVE_RHYTHMIC = False

NOTE_NAMES = ('C','C#','D','Eb','E','F','F#','G','Ab','A','Bb','B')

def midi_to_name(m: int) -> str:
    return f"{NOTE_NAMES[m%12]}{(m//12)-1}"

def _freq_to_midi(f: float, A4: float = 440.0) -> float:
    f = max(1e-9, float(f))
    return 69.0 + 12.0 * math.log2(f / A4)

def _midi_to_equal_freq(m: int, A4: float) -> float:
    return float(A4) * (2.0 ** ((int(m) - 69) / 12.0))

def _wrap50(c: float) -> float:
    # wrap to [-50, 50)
    return ((c + 50.0) % 100.0) - 50.0

class Event:
    """Standardized event frame for AI listener system with instrument-aware, harmonic, and rhythmic features"""
    def __init__(self, t: float, rms_db: float, f0: float, midi: int, 
                 cents: float, centroid: float, ioi: float, onset: bool,
                 rolloff: float = 0.0, zcr: float = 0.0, bandwidth: float = 0.0, 
                 hnr: float = 0.0, instrument: str = "unknown", mfcc: List[float] = None,
                 attack_time: float = 0.0, decay_time: float = 0.0, spectral_flux: float = 0.0,
                 harmonic_context: Optional['HarmonicContext'] = None,
                 rhythmic_context: Optional['RhythmicContext'] = None,
                 raw_audio: Optional[np.ndarray] = None):
        self.t = t
        self.rms_db = rms_db
        self.f0 = f0
        self.midi = midi
        self.cents = cents
        self.centroid = centroid
        self.ioi = ioi
        self.onset = onset
        self.rolloff = rolloff
        self.zcr = zcr
        self.bandwidth = bandwidth
        self.hnr = hnr
        self.instrument = instrument
        self.mfcc = mfcc or [0.0] * 13  # 13 MFCC coefficients
        self.attack_time = attack_time
        self.decay_time = decay_time
        self.spectral_flux = spectral_flux
        self.harmonic_context = harmonic_context  # Real-time harmonic awareness
        self.rhythmic_context = rhythmic_context  # Real-time rhythmic awareness
        self.raw_audio = raw_audio  # Raw audio buffer for hybrid perception
    
    def to_dict(self) -> Dict:
        result = {
            't': self.t,
            'rms_db': self.rms_db,
            'f0': self.f0,
            'midi': self.midi,
            'cents': self.cents,
            'centroid': self.centroid,
            'ioi': self.ioi,
            'onset': self.onset,
            'rolloff': self.rolloff,
            'zcr': self.zcr,
            'bandwidth': self.bandwidth,
            'hnr': self.hnr,
            'instrument': self.instrument,
            'mfcc': self.mfcc,
            'attack_time': self.attack_time,
            'decay_time': self.decay_time,
            'spectral_flux': self.spectral_flux
        }
        
        # Add harmonic context if available
        if self.harmonic_context:
            result['harmonic_context'] = {
                'current_chord': self.harmonic_context.current_chord,
                'key_signature': self.harmonic_context.key_signature,
                'scale_degrees': self.harmonic_context.scale_degrees,
                'chord_root': self.harmonic_context.chord_root,
                'chord_type': self.harmonic_context.chord_type,
                'confidence': self.harmonic_context.confidence,
                'stability': self.harmonic_context.stability
            }
        
        # Add rhythmic context if available
        if self.rhythmic_context:
            result['rhythmic_context'] = {
                'current_tempo': self.rhythmic_context.current_tempo,
                'meter': self.rhythmic_context.meter,
                'beat_position': self.rhythmic_context.beat_position,
                'next_beat_time': self.rhythmic_context.next_beat_time,
                'syncopation_level': self.rhythmic_context.syncopation_level,
                'rhythmic_density': self.rhythmic_context.rhythmic_density,
                'confidence': self.rhythmic_context.confidence
            }
        
        return result

class DriftListener:
    """
    Enhanced real-time listener for Drift Engine AI system
    Based on JHS Listener with additional features for AI agent
    """
    def __init__(self,
                 ref_fn: Callable[[int], float],
                 a4_fn: Callable[[], float],
                 device: Optional[int|str]=None,
                 sr: int=44100, frame: int=2048, hop: int=256,
                 fmin: float=40.0, fmax: float=2000.0):
        self.ref_fn = ref_fn
        self.a4_fn = a4_fn
        self.device = device
        self.sr = int(sr)
        self.frame = int(frame)
        self.hop = int(hop)
        self.fmin = float(fmin)
        self.fmax = float(fmax)

        # Long buffer for AI perception (MERT/Wav2Vec needs ~1s context)
        self.long_frame = int(sr * 1.5)  # 1.5 seconds
        self._long_ring = np.zeros(self.long_frame, dtype=np.float32)
        self._long_ring_pos = 0

        self._q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=8)
        self._running = False
        self._stream = None
        self._thr = None
        self._cb: Optional[Callable] = None

        self._ring = np.zeros(self.frame, dtype=np.float32)
        self._ring_pos = 0

        # Smoothing / UI-throttle
        self._alpha = 0.12
        self._f_smooth = 0.0
        self._yin_threshold = 0.1
        self._hp_z1 = 0.0

        self.level_db_threshold = -45.0
        self._silence_hold_s = 0.12
        self._last_loud_t = 0.0

        self._avg_ms = 400.0     # rolling average/median
        self._avg_buf: "deque[tuple[float,float,float]]" = deque()
        self._use_median = False

        self._gui_ms = 30.0
        self._last_gui_t = 0.0

        # Enhanced features for AI
        self._last_onset_time = 0.0
        self._onset_threshold = 0.3
        self._last_centroid = 0.0
        self._centroid_smooth = 0.0
        self._centroid_alpha = 0.1
        
        # Instrument-aware features
        self._rolloff_smooth = 0.0
        self._zcr_smooth = 0.0
        self._bandwidth_smooth = 0.0
        self._hnr_smooth = 0.0
        self._feature_alpha = 0.1
        
        # Advanced features for better classification
        self._mfcc_smooth = np.zeros(13, dtype=np.float32)
        self._attack_time_smooth = 0.0
        self._decay_time_smooth = 0.0
        self._spectral_flux_smooth = 0.0
        self._previous_spectrum = None
        
        # Harmonic context detector (real-time chord/key detection)
        self.harmonic_detector = None
        if _HAVE_HARMONIC:
            try:
                self.harmonic_detector = RealtimeHarmonicDetector(
                    window_size=8192,
                    hop_length=2048
                )
                print("🎼 Real-time harmonic detection enabled")
            except Exception as e:
                print(f"⚠️ Failed to initialize harmonic detector: {e}")
        self._harmonic_update_interval = 0.5  # Update harmony every 500ms
        self._last_harmonic_update = 0.0
        self._harmonic_buffer = deque(maxlen=int(self.sr * 2.0 / self.hop))  # 2 seconds
        
        # Rhythmic context detector (real-time tempo/beat detection)
        self.rhythmic_detector = None
        if _HAVE_RHYTHMIC:
            try:
                self.rhythmic_detector = RealtimeRhythmicDetector(
                    update_interval=2.0  # Update tempo every 2 seconds
                )
                print("🥁 Real-time rhythmic detection enabled")
            except Exception as e:
                print(f"⚠️ Failed to initialize rhythmic detector: {e}")

    def set_responsiveness(self, percent: float) -> None:
        p = max(0.0, min(100.0, float(percent))) / 100.0
        self._alpha = 0.03 + p * (0.30 - 0.03)
        self._gui_ms = 60.0 - p * (60.0 - 18.0)
    
    def set_level_threshold_db(self, db: float) -> None:
        self.level_db_threshold = float(db)
    
    def set_avg_window_ms(self, ms: float, median: bool=False) -> None:
        self._avg_ms = max(0.0, float(ms)); self._use_median = bool(median)

    def start(self, on_update: Callable) -> None:
        if not _HAVE_SD:
            raise RuntimeError("sounddevice missing (pip install sounddevice)")
        self._cb = on_update
        self._running = True
        self._stream = sd.InputStream(channels=1, samplerate=self.sr, blocksize=self.hop,
                                      device=self.device, callback=self._audio_cb, dtype='float32')
        self._stream.start()
        self._thr = threading.Thread(target=self._worker, daemon=True); self._thr.start()

    def stop(self) -> None:
        self._running = False
        try:
            if self._stream: self._stream.stop(); self._stream.close()
        except Exception:
            pass
        self._stream = None

    @staticmethod
    def _rms_db(x: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(x*x) + 1e-12)); return 20.0*math.log10(rms + 1e-12)

    def _highpass(self, x: np.ndarray, fc=40.0) -> np.ndarray:
        a = math.exp(-2.0 * math.pi * fc / self.sr)
        y = np.empty_like(x)
        z1 = self._hp_z1; prev = 0.0
        for i in range(x.shape[0]):
            xi = float(x[i])
            yi = xi - prev + a * z1
            y[i] = yi; z1 = yi; prev = xi
        self._hp_z1 = z1; return y

    @staticmethod
    def _parabolic_min(y: np.ndarray, i: int) -> float:
        if i <= 0 or i >= len(y)-1: return float(i)
        y0,y1,y2 = y[i-1],y[i],y[i+1]
        denom = (y0 - 2*y1 + y2)
        if abs(denom) < 1e-12: return float(i)
        return float(i + 0.5*(y0 - y2)/denom)

    def _yin_pitch(self, x: np.ndarray) -> float:
        x = x.astype(np.float32, copy=False)
        N = x.shape[0]
        xw = x * np.hanning(N).astype(np.float32); xw = xw - np.mean(xw)
        L = 1 << (N*2 - 1).bit_length()
        X = np.fft.rfft(xw, n=L)
        ac = np.fft.irfft(X * np.conj(X), n=L)[:N]
        ac = ac / (float(np.max(ac)) + 1e-12)
        d_full = 2.0 - 2.0*ac
        d = d_full[:N//2].astype(np.float32, copy=False)
        cmndf = np.empty_like(d); cmndf[0] = 1.0
        csum = 0.0
        for tau in range(1, d.shape[0]):
            csum += float(d[tau]); cmndf[tau] = d[tau] * tau / (csum + 1e-9)
        thr = self._yin_threshold; tau = None
        low = max(2, int(self.sr / self.fmax))
        for i in range(low, d.shape[0]):
            if cmndf[i] < thr:
                tau = self._parabolic_min(cmndf, i); break
        if tau is None:
            i = int(np.argmin(cmndf[low:])) + low
            tau = self._parabolic_min(cmndf, i)
        f0 = self.sr / max(1e-9, float(tau))
        return float(f0) if (self.fmin <= f0 <= self.fmax) else 0.0

    def _detect_onset(self, frame: np.ndarray) -> bool:
        """Enhanced onset detection using spectral flux"""
        # Calculate spectral flux
        X = np.fft.rfft(frame)
        magnitude = np.abs(X)
        
        # Simple onset detection based on spectral flux
        if hasattr(self, '_prev_magnitude'):
            flux = np.sum(np.maximum(0, magnitude - self._prev_magnitude))
            onset = flux > self._onset_threshold
            self._prev_magnitude = magnitude
            return onset
        else:
            self._prev_magnitude = magnitude
            return False

    def _calculate_spectral_centroid(self, frame: np.ndarray) -> float:
        """Calculate spectral centroid"""
        X = np.fft.rfft(frame)
        magnitude = np.abs(X)
        freqs = np.fft.rfftfreq(len(frame), 1/self.sr)
        
        if np.sum(magnitude) > 0:
            centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
            return float(centroid)
        return 0.0

    def _calculate_spectral_rolloff(self, frame: np.ndarray) -> float:
        """Calculate spectral rolloff (85% energy cutoff)"""
        X = np.fft.rfft(frame)
        magnitude = np.abs(X)
        freqs = np.fft.rfftfreq(len(frame), 1/self.sr)
        
        if np.sum(magnitude) == 0:
            return 0.0
        
        # Calculate cumulative energy
        cumulative_energy = np.cumsum(magnitude)
        total_energy = cumulative_energy[-1]
        rolloff_threshold = 0.85 * total_energy
        
        # Find frequency where 85% of energy is contained
        rolloff_idx = np.where(cumulative_energy >= rolloff_threshold)[0]
        if len(rolloff_idx) > 0:
            return float(freqs[rolloff_idx[0]])
        return float(freqs[-1])

    def _calculate_zero_crossing_rate(self, frame: np.ndarray) -> float:
        """Calculate zero crossing rate"""
        # Count sign changes
        sign_changes = np.diff(np.sign(frame))
        zcr = np.sum(np.abs(sign_changes)) / (2.0 * len(frame))
        return float(zcr)

    def _calculate_spectral_bandwidth(self, frame: np.ndarray) -> float:
        """Calculate spectral bandwidth"""
        X = np.fft.rfft(frame)
        magnitude = np.abs(X)
        freqs = np.fft.rfftfreq(len(frame), 1/self.sr)
        
        if np.sum(magnitude) == 0:
            return 0.0
        
        # Calculate spectral centroid
        centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
        
        # Calculate bandwidth
        bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * magnitude) / np.sum(magnitude))
        return float(bandwidth)

    def _calculate_harmonic_to_noise_ratio(self, frame: np.ndarray) -> float:
        """Calculate harmonic-to-noise ratio"""
        X = np.fft.rfft(frame)
        magnitude = np.abs(X)
        
        if np.sum(magnitude) == 0:
            return 0.0
        
        # Simple HNR estimation using spectral peaks vs noise floor
        # Find peaks (local maxima)
        peaks = []
        for i in range(1, len(magnitude) - 1):
            if magnitude[i] > magnitude[i-1] and magnitude[i] > magnitude[i+1]:
                peaks.append(magnitude[i])
        
        if len(peaks) == 0:
            return 0.0
        
        # Harmonic energy (sum of peaks)
        harmonic_energy = np.sum(peaks)
        
        # Noise energy (total - harmonic)
        total_energy = np.sum(magnitude)
        noise_energy = total_energy - harmonic_energy
        
        if noise_energy <= 0:
            return 1.0
        
        hnr = harmonic_energy / noise_energy
        return float(hnr)

    def _calculate_mfcc(self, frame: np.ndarray) -> List[float]:
        """Calculate MFCC coefficients"""
        try:
            # Ensure frame has sufficient length for MFCC
            # Need at least n_fft samples for meaningful analysis
            if len(frame) < 1024:
                return [0.0] * 13
            
            # Calculate MFCC using librosa
            # Suppress warnings about small buffers
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mfcc = librosa.feature.mfcc(y=frame, sr=self.sr, n_mfcc=13, n_fft=1024, hop_length=512)
            # Take the mean across time frames
            mfcc_mean = np.mean(mfcc, axis=1)
            return mfcc_mean.tolist()
        except Exception:
            return [0.0] * 13

    def _calculate_attack_time(self, frame: np.ndarray) -> float:
        """Calculate attack time (time to reach 90% of peak amplitude)"""
        try:
            # Find peak amplitude
            peak_amplitude = np.max(np.abs(frame))
            if peak_amplitude == 0:
                return 0.0
            
            # Find time to reach 90% of peak
            threshold = 0.9 * peak_amplitude
            for i, sample in enumerate(np.abs(frame)):
                if sample >= threshold:
                    # Convert sample index to time
                    return float(i) / self.sr
            return 0.0
        except Exception:
            return 0.0

    def _calculate_decay_time(self, frame: np.ndarray) -> float:
        """Calculate decay time (time from peak to 10% of peak amplitude)"""
        try:
            # Find peak amplitude and its position
            abs_frame = np.abs(frame)
            peak_idx = np.argmax(abs_frame)
            peak_amplitude = abs_frame[peak_idx]
            
            if peak_amplitude == 0:
                return 0.0
            
            # Find time to decay to 10% of peak
            threshold = 0.1 * peak_amplitude
            for i in range(peak_idx, len(abs_frame)):
                if abs_frame[i] <= threshold:
                    # Convert sample index to time
                    return float(i - peak_idx) / self.sr
            return 0.0
        except Exception:
            return 0.0

    def _calculate_spectral_flux(self, frame: np.ndarray) -> float:
        """Calculate spectral flux (change in spectrum over time)"""
        try:
            # Calculate current spectrum
            current_spectrum = np.abs(np.fft.fft(frame))
            
            if self._previous_spectrum is None:
                self._previous_spectrum = current_spectrum
                return 0.0
            
            # Calculate difference between current and previous spectrum
            diff = current_spectrum - self._previous_spectrum
            
            # Sum only positive differences (increases in energy)
            flux = np.sum(diff[diff > 0])
            
            # Update previous spectrum
            self._previous_spectrum = current_spectrum
            
            return float(flux)
        except Exception:
            return 0.0

    def _classify_instrument(self, event_data: Dict) -> str:
        """Advanced ensemble classification using multiple features"""
        # Extract features
        centroid = event_data.get('centroid', 1000.0)
        rolloff = event_data.get('rolloff', 2000.0)
        zcr = event_data.get('zcr', 0.1)
        hnr = event_data.get('hnr', 0.5)
        f0 = event_data.get('f0', 0.0)
        mfcc = event_data.get('mfcc', [0.0] * 13)
        attack_time = event_data.get('attack_time', 0.0)
        decay_time = event_data.get('decay_time', 0.0)
        spectral_flux = event_data.get('spectral_flux', 0.0)
        
        # Debug logging (print every 10th classification to avoid spam)
        if not hasattr(self, '_classify_counter'):
            self._classify_counter = 0
        self._classify_counter += 1
        
        # Silent instrument debug (disabled for clean terminal)
        
        # Ensemble classification with weighted scoring
        scores = {'drums': 0.0, 'piano': 0.0, 'guitar': 0.0, 'bass': 0.0}
        
        # 1. Spectral features (40% weight)
        if centroid > 3000 and rolloff > 5000:
            scores['drums'] += 0.4
        elif centroid < 4000 and rolloff < 8000:
            scores['piano'] += 0.3
            scores['guitar'] += 0.2
            scores['bass'] += 0.3
        
        # 2. Temporal features (30% weight)
        if attack_time < 0.02 and decay_time < 0.2:  # Fast attack, fast decay
            scores['drums'] += 0.3
        elif attack_time > 0.005 and decay_time > 0.05:  # Slower attack, longer decay
            scores['piano'] += 0.3
            scores['guitar'] += 0.2
        
        # 3. Harmonic features (20% weight)
        if hnr < 0.6:  # Lower harmonic content
            scores['drums'] += 0.2
        elif hnr > 0.4:  # Higher harmonic content
            scores['piano'] += 0.2
            scores['guitar'] += 0.15
            scores['bass'] += 0.15
        
        # 4. Pitch features (10% weight)
        if f0 < 300:  # Low pitch
            scores['bass'] += 0.1
            scores['drums'] += 0.05
        elif f0 > 100:  # Higher pitch
            scores['piano'] += 0.1
            scores['guitar'] += 0.1
        
        
        # Find the instrument with the highest score
        best_instrument = max(scores, key=scores.get)
        best_score = scores[best_instrument]
        
        # Only classify if confidence is high enough - Lower threshold for better classification
        if best_score >= 0.3:
            return best_instrument
        else:
            return "unknown"

    def _calculate_ioi(self, current_time: float) -> float:
        """Calculate inter-onset interval"""
        if self._last_onset_time > 0:
            return current_time - self._last_onset_time
        return 0.0

    def _audio_cb(self, indata, frames, time_info, status):
        if status: pass
        try:
            self._q.put_nowait(indata[:,0].copy())
        except queue.Full:
            pass

    def _best_ref_for_freq(self, f_meas: float) -> Tuple[int, float, float]:
        if f_meas <= 0.0:
            m = 69; return m, self.ref_fn(m), 0.0
        A4 = float(self.a4_fn())
        m_rough = int(round(_freq_to_midi(f_meas, A4)))
        best = None
        for cand in (m_rough-1, m_rough, m_rough+1):
            f_ref = float(self.ref_fn(cand))
            cents = 1200.0 * math.log2(max(1e-9, f_meas) / max(1e-9, f_ref))
            score = abs(cents)
            if (best is None) or (score < best[0]):
                best = (score, cand, f_ref, cents)
        _, m_best, f_ref_best, cents_best = best
        return int(m_best), float(f_ref_best), float(cents_best)

    def _worker(self):
        buf = np.zeros(0, dtype=np.float32)
        while self._running:
            try:
                chunk = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            buf = np.concatenate((buf, chunk))
            while buf.shape[0] >= self.hop:
                hopseg = buf[:self.hop]; buf = buf[self.hop:]
                
                # 1. Update short ringbuffer (for pitch/onset)
                pos = self._ring_pos; end = pos + self.hop
                if end <= self._ring.shape[0]:
                    self._ring[pos:end] = hopseg
                else:
                    k = self._ring.shape[0] - pos
                    self._ring[pos:] = hopseg[:k]; self._ring[:self.hop-k] = hopseg[k:]
                self._ring_pos = (self._ring_pos + self.hop) % self._ring.shape[0]
                
                # 2. Update long ringbuffer (for AI perception)
                lpos = self._long_ring_pos; lend = lpos + self.hop
                if lend <= self._long_ring.shape[0]:
                    self._long_ring[lpos:lend] = hopseg
                else:
                    lk = self._long_ring.shape[0] - lpos
                    self._long_ring[lpos:] = hopseg[:lk]; self._long_ring[:self.hop-lk] = hopseg[lk:]
                self._long_ring_pos = (self._long_ring_pos + self.hop) % self._long_ring.shape[0]

                # Extract frames
                idx = (np.arange(self.frame) + self._ring_pos) % self.frame
                frame = self._ring[idx].copy()
                frame = self._highpass(frame, fc=40.0)

                # level threshold
                level_db = self._rms_db(frame); nowt = time.time()
                if level_db >= self.level_db_threshold:
                    self._last_loud_t = nowt
                elif (nowt - self._last_loud_t) > self._silence_hold_s:
                    self._f_smooth = 0.0
                    if self._cb: self._cb(None, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False)
                    continue

                f0 = self._yin_pitch(frame)
                if f0 <= 0.0: continue
                self._f_smooth = f0 if self._f_smooth == 0.0 else (1.0-self._alpha)*self._f_smooth + self._alpha*f0

                m_best, f_ref, cents = self._best_ref_for_freq(self._f_smooth)

                # Enhanced features
                onset_detected = self._detect_onset(frame)
                if onset_detected:
                    self._last_onset_time = nowt
                
                centroid = self._calculate_spectral_centroid(frame)
                self._centroid_smooth = centroid if self._centroid_smooth == 0.0 else \
                    (1.0-self._centroid_alpha)*self._centroid_smooth + self._centroid_alpha*centroid
                
                # Instrument-aware features
                rolloff = self._calculate_spectral_rolloff(frame)
                self._rolloff_smooth = rolloff if self._rolloff_smooth == 0.0 else \
                    (1.0-self._feature_alpha)*self._rolloff_smooth + self._feature_alpha*rolloff
                
                zcr = self._calculate_zero_crossing_rate(frame)
                self._zcr_smooth = zcr if self._zcr_smooth == 0.0 else \
                    (1.0-self._feature_alpha)*self._zcr_smooth + self._feature_alpha*zcr
                
                bandwidth = self._calculate_spectral_bandwidth(frame)
                self._bandwidth_smooth = bandwidth if self._bandwidth_smooth == 0.0 else \
                    (1.0-self._feature_alpha)*self._bandwidth_smooth + self._feature_alpha*bandwidth
                
                hnr = self._calculate_harmonic_to_noise_ratio(frame)
                self._hnr_smooth = hnr if self._hnr_smooth == 0.0 else \
                    (1.0-self._feature_alpha)*self._hnr_smooth + self._feature_alpha*hnr
                
                # Advanced features for better classification
                mfcc = self._calculate_mfcc(frame)
                mfcc_array = np.array(mfcc)
                self._mfcc_smooth = mfcc_array if np.all(self._mfcc_smooth == 0) else \
                    (1.0-self._feature_alpha)*self._mfcc_smooth + self._feature_alpha*mfcc_array
                
                attack_time = self._calculate_attack_time(frame)
                self._attack_time_smooth = attack_time if self._attack_time_smooth == 0.0 else \
                    (1.0-self._feature_alpha)*self._attack_time_smooth + self._feature_alpha*attack_time
                
                decay_time = self._calculate_decay_time(frame)
                self._decay_time_smooth = decay_time if self._decay_time_smooth == 0.0 else \
                    (1.0-self._feature_alpha)*self._decay_time_smooth + self._feature_alpha*decay_time
                
                spectral_flux = self._calculate_spectral_flux(frame)
                self._spectral_flux_smooth = spectral_flux if self._spectral_flux_smooth == 0.0 else \
                    (1.0-self._feature_alpha)*self._spectral_flux_smooth + self._feature_alpha*spectral_flux
                
                ioi = self._calculate_ioi(nowt)
                
                # Add frame to harmonic buffer
                if self.harmonic_detector:
                    self._harmonic_buffer.append(frame)

                # rolling average (with level as weight)
                w = max(0.2, min(1.0, (level_db - (-80.0)) / 70.0))
                self._avg_buf.append((nowt, cents, w))
                cut = nowt - (self._avg_ms / 1000.0)
                while self._avg_buf and self._avg_buf[0][0] < cut:
                    self._avg_buf.popleft()

                avg = cents
                if self._avg_ms > 0.0 and len(self._avg_buf) >= 2:
                    if self._use_median:
                        avg = float(np.median([c for _, c, _ in self._avg_buf]))
                    else:
                        sw = sum(wi for _, _, wi in self._avg_buf) + 1e-9
                        avg = sum(ci*wi for _, ci, wi in self._avg_buf) / sw
                
                # Update harmonic context periodically
                harmonic_context = None
                if self.harmonic_detector and (nowt - self._last_harmonic_update) >= self._harmonic_update_interval:
                    if len(self._harmonic_buffer) > 0:
                        # Concatenate buffer for analysis
                        audio_for_harmony = np.concatenate(list(self._harmonic_buffer))
                        try:
                            harmonic_context = self.harmonic_detector.update_from_audio(audio_for_harmony, self.sr)
                            self._last_harmonic_update = nowt
                        except Exception as e:
                            pass  # Silently fail - don't disrupt audio flow
                
                # Update rhythmic context (every event)
                rhythmic_context = None
                if self.rhythmic_detector:
                    try:
                        event_data_for_rhythm = {
                            'onset': onset_detected,
                            'ioi': ioi,
                            'rms_db': level_db,
                            't': nowt
                        }
                        rhythmic_context = self.rhythmic_detector.update_from_event(event_data_for_rhythm, nowt)
                    except Exception as e:
                        pass  # Silently fail

                nowc = time.time()
                if not self._cb: continue
                if (nowc - self._last_gui_t) * 1000.0 >= self._gui_ms:
                    # Create event data for instrument classification
                    event_data = {
                        'centroid': self._centroid_smooth,
                        'rolloff': self._rolloff_smooth,
                        'zcr': self._zcr_smooth,
                        'hnr': self._hnr_smooth,
                        'f0': float(self._f_smooth),
                        'mfcc': self._mfcc_smooth.tolist(),
                        'attack_time': self._attack_time_smooth,
                        'decay_time': self._decay_time_smooth,
                        'spectral_flux': self._spectral_flux_smooth
                    }
                    
                    # Classify instrument
                    instrument = self._classify_instrument(event_data)
                    
                    # Use cached harmonic context (updated less frequently)
                    if harmonic_context is None and self.harmonic_detector:
                        harmonic_context = self.harmonic_detector.current_context
                    
                    # Use cached rhythmic context (updated frequently)
                    if rhythmic_context is None and self.rhythmic_detector:
                        rhythmic_context = self.rhythmic_detector.last_context
                    
                    # Extract long audio buffer for AI perception
                    lidx = (np.arange(self.long_frame) + self._long_ring_pos) % self.long_frame
                    long_audio = self._long_ring[lidx].copy()

                    # Create Event object with instrument-aware, harmonic, and rhythmic features
                    event = Event(
                        t=nowc,
                        rms_db=level_db,
                        f0=float(self._f_smooth),
                        midi=m_best,
                        cents=avg,
                        centroid=self._centroid_smooth,
                        ioi=ioi,
                        onset=onset_detected,
                        rolloff=self._rolloff_smooth,
                        zcr=self._zcr_smooth,
                        bandwidth=self._bandwidth_smooth,
                        hnr=self._hnr_smooth,
                        instrument=instrument,
                        mfcc=self._mfcc_smooth.tolist(),
                        attack_time=self._attack_time_smooth,
                        decay_time=self._decay_time_smooth,
                        spectral_flux=self._spectral_flux_smooth,
                        harmonic_context=harmonic_context,  # Add harmonic awareness
                        rhythmic_context=rhythmic_context,  # Add rhythmic awareness
                        raw_audio=long_audio                # Pass long audio buffer for hybrid perception
                    )
                    
                    # Harmonic and rhythmic context (silent - shown in status bar)
                    
                    self._cb(event)
                    self._last_gui_t = nowc
