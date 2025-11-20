# MusicHal 9000 - Quick Start Guide

## TL;DR

### Training (Offline - GPU Optional)
```bash
# Windows/Linux with NVIDIA GPU or Mac M1/M2
python scripts/train/train_modular.py your_audio.wav trained_model.json
# Uses GPU automatically if available
# Falls back to CPU if no GPU
```

### Live Performance (Real-time - CPU Only)
```bash
# Mac (recommended for live)
python scripts/performance/MusicHal_9000.py --model trained_model.json
# <50ms latency, no GPU needed
```

---

## GPU Usage: Quick Answer

**Q: Does it use GPU?**

**Training**: YES (optional, but faster)
- ✅ Mac M1/M2/M3: Uses MPS (Metal)
- ✅ Windows/Linux NVIDIA: Uses CUDA
- ✅ Automatic fallback to CPU if no GPU

**Live Performance**: NO (CPU only)
- Optimized for <50ms latency
- GPU not needed or used

---

## Cross-Platform: Quick Answer

**Q: Can I run this on Mac?**

✅ **YES - Fully supported!**

**Training on Mac M1/M2**:
```bash
pip install -r requirements.txt
python scripts/train/train_modular.py audio.wav model.json
# Will automatically use MPS (GPU)
```

**Live Performance on Mac**:
```bash
python scripts/performance/MusicHal_9000.py
# Works perfectly, <50ms latency
# Configure IAC Driver for MIDI
```

---

## Typical Workflow: Train on PC, Perform on Mac

### 1. Train on Windows PC (GPU-accelerated)
```bash
# On PC with NVIDIA GPU
python scripts/train/train_modular.py audio.wav trained_model.json
# Output: trained_model.json (~2-3 MB)
```

### 2. Copy Model to Mac
```bash
# Transfer the JSON file
scp trained_model.json user@mac:~/musichal/models/
```

### 3. Perform Live on Mac
```bash
# On Mac (no GPU needed)
python scripts/performance/MusicHal_9000.py --model models/trained_model.json
# Real-time, <50ms latency
```

---

## Installation

### Mac (Apple Silicon)
```bash
git clone https://github.com/yourusername/musichal.git
cd musichal
pip install -r requirements.txt

# Verify MPS
python -c "import torch; print('MPS:', torch.backends.mps.is_available())"
# Should print: MPS: True
```

### Windows (NVIDIA GPU)
```bash
git clone https://github.com/yourusername/musichal.git
cd musichal
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# Verify CUDA
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
# Should print: CUDA: True
```

### Any Platform (CPU-only)
```bash
git clone https://github.com/yourusername/musichal.git
cd musichal
pip install -r requirements.txt
# Works on any platform, just slower training
```

---

## Configuration Profiles

### Quick Test (Fast)
```bash
python scripts/train/train_modular.py audio.wav output.json --profile quick_test
# 1000 events, ~2-5 minutes
```

### Full Training (Production)
```bash
python scripts/train/train_modular.py audio.wav output.json --profile full_training
# 20000 events, ~10-30 minutes
```

### Force CPU (Disable GPU)
```bash
# Edit config/default_config.yaml
feature_extraction:
  use_gpu: false
```

---

## Performance Benchmarks

### Training Speed (1000 events)
- Mac M1/M2 (MPS): ~2-3 min ⚡
- Windows NVIDIA: ~2-3 min ⚡
- CPU (any): ~5-10 min 🐢

### Live Performance (all platforms)
- Latency: <50ms ✅
- CPU Usage: 10-25% ✅
- GPU: Not used ✅

---

## MIDI Setup

### macOS
1. Open **Audio MIDI Setup**
2. Window → **Show MIDI Studio**
3. Enable **IAC Driver**
4. Done! ✅

### Windows
1. Download **loopMIDI**
2. Create virtual MIDI port
3. Connect to DAW
4. Done! ✅

### Linux
1. Install ALSA/JACK
2. Configure MIDI routing
3. Done! ✅

---

## Common Commands

```bash
# Train with GPU (auto-detect)
python scripts/train/train_modular.py audio.wav output.json

# Train with specific profile
python scripts/train/train_modular.py audio.wav output.json --profile quick_test

# Live performance
python scripts/performance/MusicHal_9000.py

# Run tests
pytest tests/ -v

# Check GPU status
python -c "import torch; print('MPS:', torch.backends.mps.is_available(), 'CUDA:', torch.cuda.is_available())"
```

---

## Help & Documentation

- **Full Cross-Platform Guide**: `docs/CROSS_PLATFORM_GUIDE.md`
- **Complete Documentation**: `docs/REFACTORING_COMPLETE.md`
- **Project Structure**: `PROJECT_STRUCTURE.md`
- **README**: `README.md`

---

## Quick Troubleshooting

**"MPS not available" on Mac M1**:
```bash
pip install --upgrade torch torchvision torchaudio
```

**"CUDA not available" on Windows NVIDIA**:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Training is slow**:
- Use GPU if available (2-3x faster)
- Use `--profile quick_test` for testing
- Train on PC, perform on Mac

**No MIDI output**:
- Mac: Enable IAC Driver
- Windows: Install loopMIDI
- Linux: Configure ALSA

---

## Summary Table

| Feature | Mac | Windows | Linux |
|---------|-----|---------|-------|
| Training (GPU) | ✅ MPS | ✅ CUDA | ✅ CUDA |
| Training (CPU) | ✅ | ✅ | ✅ |
| Live Performance | ✅ | ✅ | ✅ |
| <50ms Latency | ✅ | ✅ | ✅ |
| Package Install | ✅ | ✅ | ✅ |

**Everything works everywhere!**
GPU just makes training faster (2-3x).
Live performance always uses CPU (<50ms latency).

---

**Ready to go? Start training!** 🎵
