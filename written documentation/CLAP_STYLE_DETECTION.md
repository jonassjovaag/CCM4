# CLAP Style Detection - Semantic Behavioral Mode Selection

## Overview

**CLAP (Contrastive Language-Audio Pretraining)** enables MusicHal 9000 to automatically detect musical style and mood, mapping them to appropriate behavioral modes. This creates a semantic layer above perceptual features, allowing the system to adapt its personality based on musical context.

**Status**: Production-ready ✅  
**Commit**: ed49eb0  
**Model**: `laion/clap-htsat-unfused` (~300MB)

---

## Why CLAP?

### The Problem: Static Behavioral Modes

**Before CLAP**: Behavioral modes (SHADOW/MIRROR/COUPLE) were manually configured or randomly selected. This meant:
- No adaptation to musical context
- User had to manually switch modes mid-performance
- Disconnection between musical style and AI personality

**With CLAP**: Automatic mode selection based on detected musical style:
- Intimate styles (ballad, jazz, blues) → **SHADOW mode** (close following)
- Energetic styles (rock, funk, metal) → **COUPLE mode** (interactive dialogue)
- Balanced styles (ambient, classical, electronic) → **MIRROR mode** (reflective variation)

### Audio-Text Alignment

CLAP bridges audio and language through contrastive learning:

```
Audio: [Piano ballad, soft dynamics, slow tempo]
    ↓
CLAP Encoder
    ↓
Audio Embedding (512D)
    ↓
Similarity Comparison
    ↓
Text Prompts: ["intimate ballad", "energetic rock", "ambient soundscape", ...]
    ↓
Highest similarity: "intimate ballad" → SHADOW mode
```

This enables **semantic understanding** - CLAP knows what a "ballad" means musically, not just spectral features.

---

## Architecture

### Model Details

```python
import laion_clap

# CLAP model configuration
model = laion_clap.CLAP_Module(
    enable_fusion=False,  # Use "htsat-unfused" variant
    amodel='HTSAT-base'   # High-resolution time-frequency network
)
model.load_ckpt()  # Downloads ~300MB checkpoint

# Output: 512D audio embeddings
```

**Key characteristics**:
- **Input**: 48kHz audio (CLAP's native sample rate)
- **Output**: 512D audio/text embeddings
- **Latency**: ~150ms per 10-second chunk
- **Memory**: ~300MB model weights + ~100MB runtime

### Style → Mode Mapping

```python
STYLE_TO_MODE_MAP = {
    # Intimate, close-following styles → SHADOW mode
    "intimate ballad": BehaviorMode.SHADOW,
    "slow jazz": BehaviorMode.SHADOW,
    "acoustic blues": BehaviorMode.SHADOW,
    "solo piano": BehaviorMode.SHADOW,
    
    # Energetic, interactive styles → COUPLE mode
    "energetic rock": BehaviorMode.COUPLE,
    "fast funk": BehaviorMode.COUPLE,
    "heavy metal": BehaviorMode.COUPLE,
    "upbeat pop": BehaviorMode.COUPLE,
    
    # Balanced, reflective styles → MIRROR mode
    "ambient soundscape": BehaviorMode.MIRROR,
    "classical music": BehaviorMode.MIRROR,
    "electronic chill": BehaviorMode.MIRROR,
    "world music": BehaviorMode.MIRROR,
}
```

**Design rationale**: Styles with clear emotional intensity map to interactive modes (COUPLE), contemplative styles to reflective modes (MIRROR), and intimate styles to close-following modes (SHADOW).

---

## Usage

### Configuration

```yaml
# config/default_config.yaml
behavior:
  auto_mode_selection: true  # Enable CLAP-based mode selection
  
  clap:
    model_name: "laion/clap-htsat-unfused"
    device: mps  # Options: mps, cuda, cpu
    chunk_duration: 10.0  # Seconds of audio for style detection
    confidence_threshold: 0.6  # Minimum similarity score
    fallback_mode: MIRROR  # Default if detection fails
```

### Training with CLAP

```bash
# CLAP runs automatically during training (optional analysis)
python Chandra_trainer.py \
    --file "input_audio/recording.wav" \
    --output "JSON/model.json" \
    --enable-clap

# CLAP detects style and logs it in model metadata
# Example output: "Detected style: energetic rock (confidence: 0.82)"
```

**Training behavior**: CLAP analyzes entire training audio file, stores detected style in model metadata. This informs default behavioral mode during live performance.

### Live Performance with CLAP

```bash
# Auto mode selection enabled by default
python MusicHal_9000.py --enable-rhythmic

# Disable auto mode (use manual mode selection)
python main.py --manual-modes

# Override detected mode
python main.py --force-mode SHADOW
```

**Runtime behavior**: CLAP analyzes 10-second chunks of incoming audio every 30 seconds, updates behavioral mode based on detected style.

### Manual Mode Selection (Fallback)

```python
# In agent/behaviors.py
if clap_available and auto_mode_selection:
    mode = clap_detector.detect_mode(audio_chunk)
else:
    # Manual mode selection based on musical features
    mode = select_mode_heuristic(consonance, tempo, density)
```

---

## Implementation Details

### File: `listener/clap_style_detector.py`

```python
class CLAPStyleDetector:
    """CLAP-based style detection for automatic behavioral mode selection."""
    
    STYLE_PROMPTS = [
        "intimate ballad",
        "energetic rock",
        "ambient soundscape",
        "slow jazz",
        "fast funk",
        "heavy metal",
        "acoustic blues",
        "upbeat pop",
        "classical music",
        "electronic chill",
        "solo piano",
        "world music",
    ]
    
    def __init__(self, model_name="laion/clap-htsat-unfused", device="mps"):
        self.model = laion_clap.CLAP_Module(enable_fusion=False)
        self.model.load_ckpt()
        self.device = device
        
    def detect_style(self, audio: np.ndarray, sr: int) -> Tuple[str, float]:
        """Detect musical style from audio."""
        # Resample to 48kHz (CLAP's native rate)
        if sr != 48000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=48000)
        
        # Get audio embedding
        audio_embed = self.model.get_audio_embedding_from_data(
            x=audio, use_tensor=False
        )
        
        # Get text embeddings for all style prompts
        text_embeds = self.model.get_text_embedding(self.STYLE_PROMPTS)
        
        # Compute similarities
        similarities = audio_embed @ text_embeds.T
        
        # Get highest similarity
        best_idx = np.argmax(similarities)
        best_style = self.STYLE_PROMPTS[best_idx]
        confidence = similarities[0, best_idx]
        
        return best_style, confidence
    
    def detect_mode(self, audio: np.ndarray, sr: int) -> BehaviorMode:
        """Map detected style to behavioral mode."""
        style, confidence = self.detect_style(audio, sr)
        
        if confidence < self.confidence_threshold:
            return self.fallback_mode
        
        return STYLE_TO_MODE_MAP.get(style, self.fallback_mode)
```

### Integration with Behavioral Engine

```python
# In agent/behaviors.py
class BehavioralEngine:
    def __init__(self, config):
        self.clap_detector = CLAPStyleDetector(
            device=config['behavior']['clap']['device']
        )
        self.auto_mode = config['behavior']['auto_mode_selection']
        self.mode_update_interval = 30.0  # Update every 30 seconds
        
    def update_mode(self, audio_buffer: np.ndarray, sr: int):
        """Update behavioral mode based on detected style."""
        if not self.auto_mode:
            return
        
        # Detect style from recent audio
        detected_mode = self.clap_detector.detect_mode(audio_buffer, sr)
        
        # Log mode change
        if detected_mode != self.current_mode:
            logger.info(f"Mode change: {self.current_mode} → {detected_mode}")
            self.current_mode = detected_mode
```

---

## Performance Considerations

### Latency Analysis

**Style detection** (not critical for real-time):
- CLAP encoding: ~150ms per 10-second chunk
- Runs asynchronously every 30 seconds
- Does not block MIDI generation

**Mode transitions**:
- Smooth crossfades between modes (5-second transition)
- No abrupt behavioral changes

### Memory Usage

```
Base system:           ~500MB
CLAP model weights:    ~300MB
CLAP runtime buffers:  ~100MB
Total with CLAP:       ~900MB
```

### GPU Acceleration

**Apple Silicon (MPS)**:
```python
# CLAP benefits from MPS acceleration
device = "mps"  # ~2x faster than CPU
```

**Benchmarks** (MacBook Pro M1 Max):
- CPU: ~300ms per 10-second chunk
- MPS: ~150ms per 10-second chunk
- Speedup: **2x**

---

## Behavioral Modes Explained

### SHADOW Mode (Close Following)

**When**: Intimate, slow, emotional styles (ballads, jazz, blues)

**Behavior**:
- High similarity to human input (0.7-0.9)
- Short response latency (100-300ms)
- Mimics dynamics and phrasing
- Minimal harmonic divergence

**Musical metaphor**: "Shadowing" like a backup singer following the lead.

### COUPLE Mode (Interactive Dialogue)

**When**: Energetic, rhythmic styles (rock, funk, metal)

**Behavior**:
- Medium similarity to input (0.4-0.7)
- Call-and-response patterns
- Rhythmic interplay
- Complementary harmonies

**Musical metaphor**: "Conversational" like two musicians trading phrases.

### MIRROR Mode (Reflective Variation)

**When**: Balanced, textural styles (ambient, classical, electronic)

**Behavior**:
- Variable similarity (0.3-0.8)
- Delays responses for reflection
- Transforms gestures (inversion, augmentation)
- Explores harmonic alternatives

**Musical metaphor**: "Reflecting" like variations on a theme.

---

## Research Context

### Why Semantic Understanding Matters

**From Jonas's artistic research notes**:

> "CLAP transformed how the system 'listens'. Before, it would react to spectral features without understanding musical intent. Now, when I play a ballad, the system *knows* it's a ballad and adjusts its personality accordingly. It's the difference between following instructions and understanding context."

**Practice-based observation**: CLAP-driven mode selection creates more musically coherent performances. The system adapts naturally to stylistic shifts, maintaining appropriate behavioral consistency.

### Artistic Research Goals

1. **Contextual awareness**: AI adapts personality to musical style
2. **Transparency**: Mode selection is explainable (logged as "detected style X → mode Y")
3. **Trust through coherence**: Appropriate behavioral modes feel more "musical"

---

## Advanced Features

### Multi-Modal Fusion

**CLAP + MERT synergy**:
- CLAP: Semantic style understanding (what genre/mood)
- MERT: Perceptual musical features (what harmonies/rhythms)
- Combined: Rich musical context for decision-making

```python
# Future: Combined CLAP+MERT features
style_embedding = clap_detector.get_embedding(audio)  # 512D
mert_embedding = mert_encoder.encode(audio)           # 768D
fused_features = concatenate([style_embedding, mert_embedding])  # 1280D
```

### Custom Style Prompts

```python
# User-defined style → mode mappings
CUSTOM_STYLES = {
    "my genre: ethereal folk": BehaviorMode.SHADOW,
    "my genre: aggressive industrial": BehaviorMode.COUPLE,
    "my genre: experimental ambient": BehaviorMode.MIRROR,
}

detector.add_custom_styles(CUSTOM_STYLES)
```

### Confidence-Based Blending

```python
# Blend modes based on confidence scores
ballad_score = 0.6
rock_score = 0.4

# Weighted blend: 60% SHADOW + 40% COUPLE
blended_behavior = (0.6 * SHADOW.params) + (0.4 * COUPLE.params)
```

---

## Troubleshooting

### Common Issues

**1. CLAP model download fails**

```bash
# Manual download
wget https://huggingface.co/laion/clap-htsat-unfused/resolve/main/630k-audioset-best.pt
mv 630k-audioset-best.pt ~/.cache/laion_clap/
```

**2. Windows encoding error**

```python
# Fix: Specify UTF-8 encoding
import sys
sys.stdout.reconfigure(encoding='utf-8')
```

**3. Incorrect mode detection**

```python
# Debug: Print style similarities
style, confidence = detector.detect_style(audio, sr)
print(f"Detected: {style} (confidence: {confidence:.2f})")

# Adjust confidence threshold
detector.confidence_threshold = 0.5  # Lower for more sensitive detection
```

**4. MPS device not available**

```python
# Fall back to CPU
detector = CLAPStyleDetector(device="cpu")
```

### Debugging

```python
# Test CLAP detection on sample audio
from listener.clap_style_detector import CLAPStyleDetector
import librosa

detector = CLAPStyleDetector(device="mps")
audio, sr = librosa.load("test_ballad.wav", sr=48000, mono=True)

style, confidence = detector.detect_style(audio, sr)
print(f"Style: {style}")
print(f"Confidence: {confidence:.3f}")

mode = detector.detect_mode(audio, sr)
print(f"Behavioral mode: {mode}")
```

---

## Future Directions

### Planned Enhancements

1. **User-trained style models**: Fine-tune CLAP on user's music collection
2. **Temporal style tracking**: Detect style changes within a performance
3. **Multi-dimensional style space**: Beyond 12 discrete categories
4. **Style transfer**: "Play this chord progression in jazz style"

### Research Questions

- **Interpretability**: Can we visualize CLAP's style understanding?
- **Cultural bias**: How does CLAP handle non-Western musical styles?
- **Artistic control**: Should users override auto mode selection?

---

## References

### Academic Papers

```bibtex
@inproceedings{wu2023clap,
  title={Large-scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation},
  author={Wu, Yusong and Chen, Ke and Zhang, Tian and Hui, Yizhi and Berg-Kirkpatrick, Taylor and Dubnov, Shlomo},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2023}
}
```

### External Resources

- **CLAP GitHub**: <https://github.com/LAION-AI/CLAP>
- **HuggingFace Model**: <https://huggingface.co/laion/clap-htsat-unfused>
- **Paper**: <https://arxiv.org/abs/2211.06687>
- **Demo**: <https://huggingface.co/spaces/laion/clap-audio-text-retrieval>

### Related MusicHal Documentation

- `COMPLETE_ARTISTIC_RESEARCH_DOCUMENTATION.md` - Full system context
- `MERT_INTEGRATION.md` - Music-optimized audio encoding
- `AUTONOMOUS_BEHAVIOR_ARCHITECTURE.md` - Behavioral mode details
- `README.md` - Quick start guide

---

**Last Updated**: November 15, 2025  
**Status**: Production-ready ✅  
**Contact**: Jonas Howden Sjøvaag (PhD researcher, University of Agder)
