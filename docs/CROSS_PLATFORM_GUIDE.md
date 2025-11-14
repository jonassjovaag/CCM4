# Cross-Platform Compatibility Guide

## Overview

MusicHal 9000 is **fully cross-platform** and works on:
- ✅ **macOS** (including Apple Silicon M1/M2/M3)
- ✅ **Windows** (10/11)
- ✅ **Linux**

## GPU Support Summary

| Platform | Training GPU | Live Performance GPU | Notes |
|----------|-------------|---------------------|-------|
| **Mac (Apple Silicon)** | ✅ MPS | ❌ CPU | Training uses Metal Performance Shaders |
| **Mac (Intel)** | ❌ CPU | ❌ CPU | No GPU acceleration |
| **Windows (NVIDIA)** | ✅ CUDA | ❌ CPU | Training uses NVIDIA CUDA |
| **Windows (AMD/Intel)** | ❌ CPU | ❌ CPU | No GPU acceleration |
| **Linux (NVIDIA)** | ✅ CUDA | ❌ CPU | Training uses NVIDIA CUDA |

### Key Points:

1. **GPU is ONLY used during training** (offline process)
   - Wav2Vec 2.0 neural encoding
   - Feature extraction acceleration
   - Can be disabled with `use_gpu: false` in config

2. **Live performance does NOT use GPU**
   - Uses CPU for <50ms latency
   - Optimized for real-time response
   - No GPU required for performance

---

## GPU Usage Details

### What Uses GPU (Training Only)

#### Wav2Vec 2.0 Encoding
- **File**: `listener/wav2vec_perception.py`
- **What it does**: Extracts 768D neural audio features
- **GPU Support**:
  ```python
  # Automatically detects and uses available GPU
  if torch.backends.mps.is_available():      # Mac (Apple Silicon)
      device = "mps"
  elif torch.cuda.is_available():            # Windows/Linux (NVIDIA)
      device = "cuda"
  else:
      device = "cpu"                         # Fallback
  ```

#### Configuration
```yaml
# In config/default_config.yaml
feature_extraction:
  use_gpu: true  # Set to false to force CPU
```

### What Does NOT Use GPU

- ❌ **Live performance** - Always uses CPU
- ❌ **AudioOracle training** - CPU-based algorithm
- ❌ **MIDI I/O** - CPU-based
- ❌ **Audio listening** - CPU-based
- ❌ **Real-time processing** - Optimized for CPU latency

---

## Platform-Specific Setup

### macOS (Apple Silicon M1/M2/M3)

#### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/musichal.git
cd musichal

# Install dependencies
pip install -r requirements.txt

# Verify MPS (Metal) support
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

#### Expected Output
```
MPS available: True
```

#### Training on Mac
```bash
# GPU-accelerated training (uses MPS)
python scripts/train/train_modular.py audio.wav output.json

# You'll see:
# 🎮 Using MPS (Apple Silicon GPU) for Wav2Vec
```

#### Live Performance on Mac
```bash
# Live performance (CPU-based, no GPU needed)
python scripts/performance/MusicHal_9000.py

# MIDI Configuration for Mac:
# 1. Open Audio MIDI Setup
# 2. Window → Show MIDI Studio
# 3. Enable IAC Driver
# 4. Create IAC Bus (if not exists)
```

#### Mac-Specific Notes
- ✅ **Works perfectly on Apple Silicon**
- ✅ **MPS acceleration for training**
- ✅ **Native Python 3.10+ support**
- ⚠️ **Intel Macs**: No GPU, but everything works on CPU

---

### Windows (NVIDIA GPU)

#### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/musichal.git
cd musichal

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt

# Verify CUDA support
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

#### Expected Output
```
CUDA available: True
```

#### Training on Windows (NVIDIA)
```bash
# GPU-accelerated training (uses CUDA)
python scripts\train\train_modular.py audio.wav output.json

# You'll see:
# 🎮 Using CUDA GPU for Wav2Vec
```

#### Live Performance on Windows
```bash
# Live performance (CPU-based)
python scripts\performance\MusicHal_9000.py

# MIDI Configuration:
# 1. Install loopMIDI (virtual MIDI ports)
# 2. Create virtual port
# 3. Connect to DAW
```

---

### Windows (No NVIDIA GPU / AMD / Intel)

#### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/musichal.git
cd musichal

# Install dependencies (CPU-only)
pip install -r requirements.txt

# Verify (will show False, that's OK)
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

#### Training on CPU
```bash
# CPU training (slower but works fine)
python scripts\train\train_modular.py audio.wav output.json

# You'll see:
# 💻 GPU not available, using CPU for Wav2Vec
```

**Expected Speed**:
- GPU: ~10-20 events/sec
- CPU: ~5-10 events/sec
- Still perfectly usable, just takes longer

#### Force CPU Mode
```yaml
# In config/default_config.yaml or your custom config
feature_extraction:
  use_gpu: false  # Force CPU even if GPU available
```

---

### Linux (NVIDIA GPU)

#### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/musichal.git
cd musichal

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt

# Verify CUDA
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

Same as Windows NVIDIA setup above.

---

## Typical Workflow: Train on PC, Perform on Mac

### 1. Train Model on PC (GPU-accelerated)
```bash
# On Windows PC with NVIDIA GPU
cd musichal
python scripts/train/train_modular.py audio.wav trained_model.json

# Training output: trained_model.json (~2-3 MB)
```

### 2. Transfer Model to Mac
```bash
# Copy the trained model file
scp trained_model.json user@mac:~/musichal/models/
```

### 3. Perform Live on Mac
```bash
# On Mac (no GPU needed)
cd ~/musichal
python scripts/performance/MusicHal_9000.py --model models/trained_model.json

# Real-time performance with <50ms latency
```

---

## Performance Comparison

### Training (1000 events)

| Platform | Time | Notes |
|----------|------|-------|
| Mac M1/M2 (MPS) | ~2-3 min | Fast, energy efficient |
| Windows NVIDIA RTX | ~2-3 min | Fast |
| Windows/Mac (CPU) | ~5-10 min | Slower but works |

### Live Performance (<50ms latency)

| Platform | Latency | CPU Usage |
|----------|---------|-----------|
| Mac M1/M2 | <50ms | ~10-20% |
| Mac Intel | <50ms | ~15-25% |
| Windows (any) | <50ms | ~15-25% |
| Linux (any) | <50ms | ~15-25% |

**All platforms achieve real-time performance** ✓

---

## Dependencies

### Core Dependencies (All Platforms)
```
torch>=2.0.0          # Neural networks (CPU/GPU)
torchaudio>=2.0.0     # Audio processing
transformers>=4.30.0  # Wav2Vec 2.0
librosa>=0.10.0       # Audio analysis
numpy>=1.24.0         # Numerical computing
pydantic>=2.0.0       # Data validation
PyYAML>=6.0           # Configuration
```

### Platform-Specific

**macOS**:
- `python-rtmidi` - MIDI I/O (native CoreMIDI)
- No additional packages needed

**Windows**:
- `python-rtmidi` - MIDI I/O (native Windows MIDI)
- Optional: `loopMIDI` (virtual MIDI ports)

**Linux**:
- `python-rtmidi` - MIDI I/O (ALSA/JACK)
- `libasound2-dev` - ALSA development files

---

## Configuration for Cross-Platform

### Use Relative Paths
```yaml
# Good (works everywhere)
input_audio:
  directory: "input_audio"
output:
  directory: "output"

# Bad (platform-specific)
input_audio:
  directory: "C:\Users\You\musichal\audio"  # Windows only!
```

### Platform Detection
The code automatically detects platform:
```python
import sys
if sys.platform == "darwin":     # macOS
    # Mac-specific code
elif sys.platform == "win32":    # Windows
    # Windows-specific code
elif sys.platform.startswith("linux"):  # Linux
    # Linux-specific code
```

---

## Common Issues

### "MPS not available" on Mac M1/M2

**Solution**:
```bash
# Update PyTorch
pip install --upgrade torch torchvision torchaudio

# Verify
python -c "import torch; print(torch.backends.mps.is_available())"
```

### "CUDA not available" on Windows NVIDIA

**Solution**:
```bash
# Reinstall PyTorch with CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### "No MIDI devices found"

**macOS**: Enable IAC Driver in Audio MIDI Setup
**Windows**: Install and configure loopMIDI
**Linux**: Configure ALSA/JACK

### Slow Training on CPU

**Expected**: CPU training is 2-3x slower than GPU
**Solutions**:
- Use `--profile quick_test` for faster iteration
- Limit events with `max_events: 1000` in config
- Train on a machine with GPU (transfer model after)

---

## Testing Cross-Platform

### Quick Test
```bash
# Test package imports
python -c "from musichal.core import ConfigManager; print('✓ Package OK')"

# Test GPU detection
python -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}, CUDA: {torch.cuda.is_available()}')"

# Run quick test
python scripts/demo/quick_test.py
```

### Full Test Suite
```bash
# All platforms
pytest tests/ -v

# Should see: 57/57 tests passing
```

---

## Best Practices

### For Training
1. **Use GPU if available** (much faster)
2. **Start with `quick_test` profile** (1000 events)
3. **Monitor memory usage** (training can use 2-4 GB RAM)
4. **Save checkpoints** for long training runs

### For Live Performance
1. **Use Mac for live** (low latency, stable)
2. **Close unnecessary apps** (reduce CPU load)
3. **Use wired MIDI** (lower latency than Bluetooth)
4. **Test before performing** (verify MIDI routing)

### For Development
1. **Use same Python version** across platforms (3.10+)
2. **Keep requirements.txt updated**
3. **Test on target platform** before deploying
4. **Use configuration profiles** for different setups

---

## Summary

| Feature | Mac | Windows | Linux |
|---------|-----|---------|-------|
| **Training** | ✅ (MPS) | ✅ (CUDA) | ✅ (CUDA) |
| **Live Performance** | ✅ | ✅ | ✅ |
| **GPU Training** | ✅ M1/M2/M3 | ✅ NVIDIA | ✅ NVIDIA |
| **CPU Fallback** | ✅ | ✅ | ✅ |
| **MIDI I/O** | ✅ | ✅ | ✅ |
| **Real-time (<50ms)** | ✅ | ✅ | ✅ |
| **Package Install** | ✅ | ✅ | ✅ |

### Recommended Setup
- **Training**: Windows/Linux with NVIDIA GPU or Mac M1/M2
- **Performance**: Mac (best for live music)
- **Development**: Any platform

**Everything works everywhere, GPU just makes training faster!**
