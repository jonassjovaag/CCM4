# Wav2Vec 2.0 Integration Guide

## Overview

This system now supports **Wav2Vec 2.0** neural audio encoding as an alternative to traditional ratio-based + chroma features. Wav2Vec provides learned 768-dimensional representations of audio that can capture complex musical patterns automatically.

## Installation

```bash
# Install Wav2Vec dependencies
pip install transformers torch
```

## Architecture

### Traditional Pipeline (Ratio + Chroma)
```
Audio → Ratio Analyzer → Ratio features (10D)
      → Chroma Extractor → Chroma features (12D)
      → CONCATENATE → 22D features
      → Symbolic Quantizer → Discrete tokens
```

### Wav2Vec Pipeline
```
Audio → Wav2Vec Encoder → Neural features (768D)
      → Symbolic Quantizer → Discrete tokens
      → Pseudo-chroma extraction (for compatibility)
```

## Usage

### 1. Training with Wav2Vec

Train an AudioOracle model using Wav2Vec features:

```bash
python Chandra_trainer.py \
    --file input_audio/song.wav \
    --hybrid-perception \
    --wav2vec \
    --output training_output.json
```

With GPU acceleration (recommended):

```bash
python Chandra_trainer.py \
    --file input_audio/song.wav \
    --hybrid-perception \
    --wav2vec \
    --gpu \
    --output training_output.json
```

Using a different Wav2Vec model:

```bash
python Chandra_trainer.py \
    --file input_audio/song.wav \
    --hybrid-perception \
    --wav2vec \
    --wav2vec-model facebook/wav2vec2-large \
    --gpu \
    --output training_output.json
```

### 2. Real-time with Wav2Vec

Run MusicHal_9000 with Wav2Vec perception:

```bash
python MusicHal_9000.py \
    --hybrid-perception \
    --wav2vec \
    --gpu
```

Standard MIDI (no MPE):

```bash
python MusicHal_9000.py \
    --hybrid-perception \
    --wav2vec \
    --gpu \
    --no-mpe
```

### 3. Performance Comparison

Compare Wav2Vec vs traditional features:

```bash
# Basic comparison
python test_wav2vec_comparison.py \
    --file input_audio/test.wav

# With GPU and save results
python test_wav2vec_comparison.py \
    --file input_audio/test.wav \
    --gpu \
    --output comparison_results.json

# Extended test (more segments)
python test_wav2vec_comparison.py \
    --file input_audio/test.wav \
    --gpu \
    --segments 200 \
    --duration 60
```

## Available Wav2Vec Models

### Pre-trained on Speech (HuggingFace)

1. **facebook/wav2vec2-base** (default)
   - 768D features
   - ~95M parameters
   - Fast inference
   - Good balance of quality/speed

2. **facebook/wav2vec2-large**
   - 1024D features
   - ~317M parameters
   - Better quality, slower
   - Recommended with GPU

3. **facebook/wav2vec2-large-robust**
   - 1024D features
   - Robust to noise/artifacts
   - Good for real-world audio

### Music-Specific Models (if available)

If you have access to music-pretrained Wav2Vec models (e.g., from Ragano et al. 2023), use them for better musical understanding:

```bash
python Chandra_trainer.py \
    --file input_audio/song.wav \
    --hybrid-perception \
    --wav2vec \
    --wav2vec-model path/to/music-wav2vec2 \
    --gpu
```

## Performance Considerations

### Speed
- **CPU**: ~50-100ms per 500ms segment (10-20x real-time)
- **GPU (MPS/CUDA)**: ~5-15ms per 500ms segment (30-100x real-time)
- **Traditional (CPU)**: ~1-3ms per 500ms segment (150-500x real-time)

### Memory
- **Wav2Vec Base**: ~400MB GPU memory
- **Wav2Vec Large**: ~1.3GB GPU memory
- **Traditional**: Negligible

### Quality
- **Wav2Vec**: Learned representations capture complex patterns
- **Traditional**: Interpretable, psychoacoustically grounded
- **Best Use**: Combine both for optimal results

## When to Use Wav2Vec

### ✅ Use Wav2Vec When:
- You have GPU acceleration available
- Working with complex, polyphonic music
- Need learned representations for pattern matching
- Training on diverse musical styles
- Want automatic feature learning

### ❌ Use Traditional When:
- Real-time on CPU is critical
- Need interpretable features (consonance, ratios)
- Working with simple harmonic content
- Low-latency requirements (<10ms)
- Limited computational resources

## Hybrid Approach (Best of Both)

You can train TWO models and use both:

```bash
# Train with traditional features
python Chandra_trainer.py \
    --file input_audio/song.wav \
    --hybrid-perception \
    --output traditional_model.json

# Train with Wav2Vec features
python Chandra_trainer.py \
    --file input_audio/song.wav \
    --hybrid-perception \
    --wav2vec \
    --gpu \
    --output wav2vec_model.json
```

Then ensemble predictions in real-time.

## Troubleshooting

### "transformers library not installed"
```bash
pip install transformers torch
```

### "MPS device not available"
On Mac, ensure you have:
- macOS 12.3+
- PyTorch 1.12+ with MPS support

### "CUDA not available"
On Linux/Windows, install CUDA-enabled PyTorch:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Slow inference on CPU
- Use smaller model: `facebook/wav2vec2-base`
- Reduce segment size
- Enable GPU if available
- Consider traditional features for real-time

## References

- **Wav2Vec 2.0**: Baevski et al. (2020) - "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations"
- **Music Application**: Ragano et al. (2023) - "Wav2Vec 2.0 for Music Information Retrieval"
- **IRCAM Musical Agents**: Bujard et al. (2025) - Neural encoding for musical agents

## Integration Details

### File Structure
```
listener/
├── wav2vec_perception.py       # Wav2Vec encoder implementation
├── hybrid_perception.py         # Main perception module (supports both)
├── ratio_analyzer.py           # Traditional ratio analysis
└── harmonic_chroma.py          # Traditional chroma extraction

Chandra_trainer.py              # Training pipeline
MusicHal_9000.py                # Real-time system
test_wav2vec_comparison.py      # Performance comparison tool
```

### API Example

```python
from listener.hybrid_perception import HybridPerceptionModule

# Create Wav2Vec module
perception = HybridPerceptionModule(
    vocabulary_size=64,
    enable_wav2vec=True,
    wav2vec_model="facebook/wav2vec2-base",
    use_gpu=True
)

# Extract features
import librosa
audio, sr = librosa.load("song.wav", sr=44100)
result = perception.extract_features(audio, sr, timestamp=0.0)

print(f"Features: {result.features.shape}")  # (768,)
print(f"Consonance: {result.consonance:.3f}")
print(f"Active pitch classes: {result.active_pitch_classes}")
```

## Future Enhancements

- [ ] Fine-tune Wav2Vec on music dataset
- [ ] Multi-modal fusion (combine Wav2Vec + ratio features)
- [ ] Lightweight distillation for faster inference
- [ ] Streaming inference optimization
- [ ] Attention visualization for interpretability

---

**Status**: ✅ Fully integrated and tested
**Last Updated**: 2025-10-08





























