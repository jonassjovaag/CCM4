# MusicHal 9000 - AI Musical Partner System

A real-time AI musical partner that learns from audio input and responds with MIDI output. Features **harmonic-rhythmic correlation analysis**, **hierarchical pattern recognition**, and **unified decision-making** for true musical intelligence.

## 🎯 Main Commands

### 🚀 Train Enhanced Model with Rhythmic Analysis (Best Quality)
```bash
python Chandra_trainer.py --file "input_audio/Curious_child.wav" --output "JSON/curious_child_rhythm_15k.json" --max-events 15000
```

### 🎵 Run Live Performance with Rhythmic Intelligence (Real-time)
```bash
python MusicHal_9000.py --enable-rhythmic
```

### 🎼 Run Traditional Harmonic-Only Mode
```bash
python main.py
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Live System (Real-time Performance)
```bash
python main.py
```
**This runs the live AI musical partner with <50ms latency**

## 🎵 Enhanced Capabilities (MusicHal 9000)

### 🥁 Rhythmic Intelligence
- **RhythmOracle**: Learns rhythmic patterns (sparse, moderate, dense)
- **Tempo Adaptation**: Adjusts to your playing tempo in real-time
- **Syncopation Detection**: Recognizes rhythmic complexity
- **Pattern Recognition**: Identifies familiar rhythmic contexts

### 🔗 Harmonic-Rhythmic Correlation
- **Cross-Modal Analysis**: Learns relationships between harmonic and rhythmic patterns
- **Temporal Alignment**: Understands how chord changes align with rhythmic accents
- **Unified Decision-Making**: Makes musically intelligent decisions considering both dimensions
- **Correlation Patterns**: Discovers recurring harmonic-rhythmic relationships

### 🏗️ Hierarchical Analysis
- **Multi-Timescale**: Analyzes sections, phrases, and measures
- **Perceptual Filtering**: Focuses on musically significant moments
- **Adaptive Sampling**: Intelligently samples events for training
- **Structural Understanding**: Recognizes musical form and structure

### 🧠 Music Theory Integration
- **Real Chord Detection**: Analyzes actual audio content for accurate chord progressions
- **Music Theory Transformer**: Deep musical understanding with embedded music theory knowledge
- **Key Signature Analysis**: Recognizes tonal centers and harmonic relationships
- **Scale Analysis**: Understands melodic and harmonic scales

### 3. Train Enhanced Models (Transformer + AudioOracle)
```bash
# Enhanced training with rhythmic analysis (recommended)
python Chandra_trainer.py --file "input_audio/Curious_child.wav" --output "JSON/curious_child_enhanced.json" --max-events 15000

# Enhanced training with custom sampling strategy
python Chandra_trainer.py --file "audio.mp3" --output "enhanced_model.json" --max-events 10000 --sampling-strategy perceptual

# Enhanced training without transformer (AudioOracle + Rhythmic only)
python Chandra_trainer.py --file "audio.mp3" --output "oracle_model.json" --no-transformer

# Traditional hybrid training with transformer enhancement
python train_hybrid.py --file "input_audio/Grab-a-hold.mp3" --output "enhanced_model.json" --transformer

# Efficient testing (2000 events processing, 100 events training)
python train_hybrid.py --file "your_audio.mp3" --output "test_model.json" --transformer --max-events 2000 --training-events 100
```

### 4. Train Basic Models (AudioOracle Only)
```bash
# MPS GPU acceleration
python audio_file_learning/learn_from_files_mps_complete.py --file "your_audio.mp3" --output "basic_model.json" --stats

# CPU version
python audio_file_learning/learn_from_files.py --file "your_audio.mp3" --output "basic_model.json" --stats
```

## 📁 File Structure

```
├── main.py                          # Live performance system
├── train_hybrid.py                  # Hybrid training pipeline
├── hybrid_training/                 # Transformer enhancement
│   ├── music_theory_transformer.py  # Music theory-based transformer (NEW!)
│   └── transformer_analyzer.py      # Original transformer
├── audio_file_learning/             # Basic audio training
│   ├── learn_from_files_mps_complete.py  # MPS GPU training
│   ├── learn_from_files.py          # CPU training
│   └── file_processor.py            # Feature extraction
├── memory/                          # AI learning algorithms
│   ├── audio_oracle.py              # AudioOracle algorithm
│   ├── audio_oracle_mps.py          # MPS GPU version
│   └── memory_buffer.py              # Musical memory
├── agent/                           # AI decision making
├── midi_io/                         # MIDI input/output
└── listener/                        # Audio input
```

## 🎯 Training Options

### 🚀 Hybrid Training (Transformer + AudioOracle) - RECOMMENDED
```bash
# Full hybrid training with transformer enhancement
python train_hybrid.py --file "audio.mp3" --output "enhanced_model.json" --transformer

# Efficient testing (fast)
python train_hybrid.py --file "audio.mp3" --output "test_model.json" --transformer --max-events 2000 --training-events 100

# AudioOracle only (no transformer)
python train_hybrid.py --file "audio.mp3" --output "basic_model.json" --oracle-only
```

### ⚡ Basic Training (AudioOracle Only)
```bash
# MPS GPU acceleration
python audio_file_learning/learn_from_files_mps_complete.py --file "audio.mp3" --output "model.json" --stats

# CPU version
python audio_file_learning/learn_from_files.py --file "audio.mp3" --output "model.json" --stats
```

## ⚙️ Parameters

### Hybrid Training Parameters
| Parameter | Description | Default |
|-----------|-------------|---------|
| `--file` | Audio file to process | Required |
| `--output` | Output model file | `hybrid_model.json` |
| `--transformer` | Enable transformer enhancement | False |
| `--oracle-only` | Use AudioOracle only | False |
| `--max-events` | Limit events for processing | None (all events) |
| `--training-events` | Limit events for training | None (all events) |
| `--transformer-model` | Path to transformer model | None (random init) |
| `--cpu-threshold` | CPU threshold for hybrid trainer | 5000 |

### Basic Training Parameters
| Parameter | Description | Default |
|-----------|-------------|---------|
| `--file` | Audio file to process | Required |
| `--output` | Output model file | `trained_model.json` |
| `--max-events` | Limit events for testing | None (all events) |
| `--distance-threshold` | Similarity threshold | 0.15 |
| `--distance-function` | Distance function | `euclidean` |
| `--stats` | Show detailed statistics | False |
| `--verbose` | Verbose output | False |
| `--no-mps` | Disable GPU acceleration | False |

## 🎵 Usage Examples

### 🚀 Hybrid Training (Recommended)
```bash
# Full hybrid training
python train_hybrid.py --file "input_audio/Grab-a-hold.mp3" --output "grab-a-hold_enhanced.json" --transformer

# Quick test
python train_hybrid.py --file "jazz_sample.wav" --output "jazz_test.json" --transformer --max-events 2000 --training-events 100

# AudioOracle only
python train_hybrid.py --file "jazz_sample.wav" --output "jazz_basic.json" --oracle-only
```

### ⚡ Basic Training
```bash
# MPS GPU training
python audio_file_learning/learn_from_files_mps_complete.py --file "jazz_sample.wav" --output "jazz_model.json" --stats

# CPU training
python audio_file_learning/learn_from_files.py --file "jazz_sample.wav" --output "jazz_model.json" --stats
```

### 🎵 Live Performance
```bash
# Run live AI musical partner
python main.py
```

## 🚀 Performance

| Method | Speed | GPU Usage | Best For | Accuracy |
|--------|-------|-----------|----------|----------|
| **Hybrid Training** | 10-15 events/sec | Yes | Production | **85-95%** |
| **Basic MPS** | 15-20 events/sec | Yes | Fast training | 70-85% |
| **Basic CPU** | 5-10 events/sec | No | Compatibility | 70-85% |
| **Live Performance** | <50ms latency | No | Real-time | 70-85% |

## 🔧 Troubleshooting

### MPS Not Available
```bash
# Check PyTorch installation
python -c "import torch; print(torch.backends.mps.is_available())"

# Reinstall PyTorch
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Training Too Slow
- Use `--max-events` to test with smaller datasets
- Enable MPS GPU acceleration
- Check available memory

### No MIDI Output
- Check MIDI port settings in `main.py`
- Verify IAC Driver is enabled
- Test with `python test_midi_output.py`

## 📊 Expected Results

### Hybrid Training (Music Theory Enhanced)
- **Training Time**: 10-30 minutes per audio file
- **Model Size**: ~2-3 MB for 50,000 events
- **Patterns Found**: 2000-5000+ musical patterns
- **Musical Analysis**: Real chord progressions, scales, form, tempo (music theory-based!)
- **Transformer Analysis**: 0.97s (vs 26.80s with random transformer)
- **Real-time Response**: <50ms latency

### Basic Training
- **Training Time**: 2-3 hours (MPS) vs 9+ hours (CPU)
- **Model Size**: ~1-2 MB for 50,000 events
- **Patterns Found**: 1000-2000+ musical patterns
- **Real-time Response**: <50ms latency

## 🎯 Next Steps

1. **Train enhanced model** with hybrid training
2. **Run live system** with trained model
3. **Adjust parameters** for your musical style
4. **Integrate with DAW** via MIDI

## 🎵 When to Use What

### 🚀 Use Hybrid Training When:
- You want **best accuracy** (85-95%)
- You need **musical analysis** (chords, scales, form)
- You have **time for training** (10-30 minutes)
- You want **production-quality** models

### ⚡ Use Basic Training When:
- You want **fast training** (2-3 hours)
- You need **compatibility** (CPU-only)
- You're **testing** different parameters
- You have **limited resources**

### 🎵 Use Live Performance When:
- You want **real-time interaction** (<50ms)
- You're **performing live**
- You need **immediate responses**
- You're **collaborating** with the AI

---

**Happy music making!** 🎵✨