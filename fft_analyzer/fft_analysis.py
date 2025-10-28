import plotly.graph_objects as go
import numpy as np
import pyaudio
import time

# Configuration
FPS = 30
FFT_WINDOW_SECONDS = 0.25
RESOLUTION = (1920, 1080)
SCALE = 2  # 0.5=QHD, 1=HD, 2=4K

# Audio configuration
CHUNK = int(44100 * FFT_WINDOW_SECONDS)
RATE = 44100
FORMAT = pyaudio.paFloat32
CHANNELS = 1

# FFT display config
FREQ_MIN = 10
FREQ_MAX = 1000
TOP_NOTES = 3
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

class AudioFFTAnalyzer:
    def __init__(self, device_index=None):
        self.p = pyaudio.PyAudio()
        if device_index is None:
            device_index = self.p.get_default_input_device_info()['index']
        self.stream = self.p.open(
            input_device_index=device_index,
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )
        self.window = 0.5 * (1 - np.cos(np.linspace(0, 2*np.pi, CHUNK, False)))
        self.fig = go.Figure()
        self.setup_plot()

    def setup_plot(self):
        self.fig.update_layout(
            title="Frequency Spectrum",
            xaxis_title="Frequency (Hz)",
            yaxis_title="Magnitude",
            autosize=True,
            margin=dict(l=50, r=50, t=50, b=50),
            height=800
        )
        self.fig.update_xaxes(range=[FREQ_MIN, FREQ_MAX])
        self.fig.update_yaxes(range=[0, 1])

    def get_dominant_frequencies(self, fft_magnitude, xf):
        # Get top N strongest frequencies from input
        peaks = self.find_top_notes(fft_magnitude, xf)
        return [freq for freq, _, _ in peaks]

    def freq_to_number(self, f): 
        return 69 + 12 * np.log2(f/440.0)

    def note_name(self, n): 
        return NOTE_NAMES[int(n) % 12] + str(int(n/12 - 1))

    def find_top_notes(self, fft_data, xf):
        if np.max(fft_data) < 0.001:
            return []
        peaks = []
        for i in range(1, len(fft_data) - 1):
            if fft_data[i] > fft_data[i-1] and fft_data[i] > fft_data[i+1]:
                freq = xf[i]
                if FREQ_MIN <= freq <= FREQ_MAX:
                    note_num = self.freq_to_number(freq)
                    note_name = self.note_name(round(note_num))
                    peaks.append((freq, note_name, fft_data[i]))
        return sorted(peaks, key=lambda x: x[2], reverse=True)[:TOP_NOTES]

    def update_plot(self, fft_magnitude, notes):
        self.fig.data = []
        self.fig.add_trace(go.Scatter(
            x=np.linspace(0, RATE//2, len(fft_magnitude)),
            y=fft_magnitude
        ))
        for note in notes:
            self.fig.add_annotation(
                x=note[0]+10,
                y=note[2],
                text=note[1],
                showarrow=False,
                font=dict(size=48)
            )

    def process_audio(self):
        with self.fig.batch_update():
            data = np.frombuffer(self.stream.read(CHUNK), dtype=np.float32)
            level = np.max(np.abs(data))
            
            # Process FFT
            fft = np.fft.rfft(data * self.window)
            fft_magnitude = np.abs(fft) / (CHUNK/2)
            xf = np.linspace(0, RATE//2, len(fft_magnitude))
            
            # Get frequencies for both display and system interaction
            dominant_freqs = self.get_dominant_frequencies(fft_magnitude, xf)
            notes = self.find_top_notes(fft_magnitude, xf)
            
            # Format note information with fixed width fields
            note_info = " | ".join([
                f"{note[1]:>3}({note[0]:>3.0f}Hz): {note[2]:>4.2f}"
                for note in notes
            ])
            
            # Use string formatting with fixed width
            status_line = f"Level: {level:>5.2f} | Notes: {note_info:<50}"
            print(status_line, end='\r')
            
            # Update plot
            self.update_plot(fft_magnitude, notes)

    def run(self):
        self.fig.show(renderer="browser")
        while True:
            try:
                self.process_audio()
                time.sleep(0.1)
            except OSError:
                break

    def cleanup(self):
        try:
            if hasattr(self, 'stream'):
                if self.stream:
                    self.stream.stop_stream()
                    self.stream.close()
                    self.stream = None
            if hasattr(self, 'p'):
                if self.p:
                    self.p.terminate()
                    self.p = None
        except Exception as e:
            print(f"Cleanup completed with status: {str(e)}")

if __name__ == "__main__":
    temp_pyaudio = pyaudio.PyAudio()
    info = temp_pyaudio.get_host_api_info_by_index(0)
    
    print("\nAvailable Input Devices:")
    for i in range(info.get('deviceCount')):
        if temp_pyaudio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels') > 0:
            print(f"Input Device {i}: {temp_pyaudio.get_device_info_by_host_api_device_index(0, i).get('name')}")
    
    temp_pyaudio.terminate()
    
    device_choice = input("\nSelect input device by number: ")
    analyzer = AudioFFTAnalyzer(int(device_choice))
    try:
        print("Starting FFT analysis... Press Ctrl+C to exit")
        analyzer.run()
    except KeyboardInterrupt:
        print("\nStopping FFT analysis...")
    finally:
        analyzer.cleanup()

class FrequencyAnalyzer:
    def __init__(self, freq_engine):
        self.freq_engine = freq_engine
        self.freq_memory = []
        self.window_size = 30  # 3 seconds at 10Hz sampling
        
    def process_frequencies(self, frequencies):
        self.freq_memory.append(frequencies)
        if len(self.freq_memory) > self.window_size:
            self.freq_memory.pop(0)
            
        dominant_freq = np.median(self.freq_memory)
        self.freq_engine.update_base_frequency(dominant_freq)

class EnvironmentalAdapter:
    def __init__(self, base_freq=110):
        self.base_freq = base_freq
        self.adaptation_rate = 0.1
        
    def adapt_to_environment(self, detected_freq):
        # Smooth transition to new frequency
        freq_diff = detected_freq - self.base_freq
        self.base_freq += freq_diff * self.adaptation_rate
        return self.base_freq
