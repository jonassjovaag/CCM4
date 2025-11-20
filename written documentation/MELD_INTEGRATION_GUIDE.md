# Meld Synthesizer Integration Guide

## Overview

MusicHal now integrates with **Ableton Meld** bi-timbral synthesizer to expand the "Player" beyond basic MIDI note-on/off into expressive synthesis control driven by audio analysis.

### What This Adds

- **77+ feature dimensions** → CC parameter mapping
- **Probabilistic routing** for creative accidents (configurable)
- **Scale-aware filter control** from TouchOSC chord guide
- **Dual-engine architecture** (Engine A=melodic, Engine B=bass)
- **Real-time perceptual synthesis** (listening shapes timbre)

---

## Quick Start

### 1. Enable Meld Integration

```bash
python MusicHal_9000.py --enable-meld
```

This adds Meld CC control alongside existing MPE note output.

### 2. Configure Ableton Live

**For Melodic Voice:**
1. **Create MIDI track** with Meld inserted
2. **Set MIDI input** to "IAC Meld"
3. **MIDI channel**: All Channels (for MPE)
4. **Preset**: Bright/harmonic (e.g., "FM Bells", "Analog Pad")

**For Bass Voice:**
1. **Create second MIDI track** with Meld inserted
2. **Set MIDI input** to "IAC Meld Bass"
3. **MIDI channel**: All Channels (for MPE)
4. **Preset**: Dark/rhythmic (e.g., "Sub Bass", "Resonant Bass")

### 3. MIDI Learn Scale-Aware Parameters (Optional)

Meld's scale-aware filters need manual MIDI Learn for 3 CCs:

1. Open Meld → click **Plate** or **Membrane Resonator** filter
2. **Enable scale mode** → MIDI Learn → CC 26
3. **Root note** selector → MIDI Learn → CC 27
4. **Scale type** selector → MIDI Learn → CC 28

Once learned, filters will track harmonic context from TouchOSC chord guide.

---

## Architecture

### Feature Flow

```
Audio Input
    ↓
MERT-v1-95M (768D music embeddings)
    ↓
Feature Extraction (77+ dimensions)
    ↓
MeldMapper (probabilistic routing)
    ↓
MeldController (CC sending + throttling)
    ↓
Ableton Meld (synthesis)
```

### CC Mapping (20-30)

| CC | Parameter | Primary Feature | Alternative Feature |
|----|-----------|----------------|---------------------|
| 20 | Engine A Macro 1 | spectral_centroid | bandwidth |
| 21 | Engine A Macro 2 | consonance | mfcc_1 |
| 22 | Engine B Macro 1 | zcr | spectral_flatness |
| 23 | Engine B Macro 2 | spectral_rolloff | spectral_rolloff |
| 24 | AB Blend | mfcc_1 | - |
| 25 | Spread | modulation_depth | - |
| 26 | Scale Enable | (manual MIDI Learn) | - |
| 27 | Root Note | (from chord detection) | - |
| 28 | Scale Type | (from chord quality) | - |
| 29 | Filter Frequency | spectral_rolloff | - |
| 30 | Filter Resonance | **1.0 - spectral_flatness** | - |

**Critical**: Filter resonance is **inverted** (noisy → low Q, tonal → high Q).

---

## Configuration

### Probabilistic Routing

Edit `config/meld_mapping.yaml`:

```yaml
probabilistic_routing:
  enabled: true
  accident_probability: 0.2  # 20% alternative mappings
```

**Research parameter**: Adjust 0.0 (deterministic) to 1.0 (chaos) to explore coherence vs surprise.

### Throttling

Default settings prevent MIDI congestion:

```yaml
throttling:
  update_interval_ms: 50         # Max CC rate
  change_threshold: 0.05         # Min change (5%) to send
  scale_update_interval_s: 2.0   # Slower scale updates
```

**When to change**: Lower `update_interval_ms` for ultra-responsive timbre tracking (requires fast MIDI interface).

---

## Preset Recommendations

### For Melodic Voice (Engine A)

Map **Engine A Macro 1** (CC 20, spectral_centroid) to:
- Filter cutoff (brightness follows input)
- Wavetable position (timbral evolution)
- LFO rate (activity drives modulation)

Map **Engine A Macro 2** (CC 21, consonance) to:
- Oscillator detune (dissonance → wider spread)
- Reverb mix (harmonic tension → space)

### For Bass Voice (Engine B)

Map **Engine B Macro 1** (CC 22, zcr) to:
- Noise/sub-oscillator balance (percussive → noisy)
- Envelope attack time (percussive → fast)

Map **Engine B Macro 2** (CC 23, spectral_rolloff) to:
- Filter cutoff (bass brightness)
- Saturation amount (low rolloff → warmth)

### Global Parameters

**AB Blend** (CC 24, mfcc_1): Crossfade between engines based on timbral character

**Spread** (CC 25, modulation_depth): Stereo width follows rhythmic activity

---

## Scale-Aware Filter Workflow

### Without TouchOSC (Chromatic Default)

Filters default to C chromatic (all notes allowed). System still functions without scale constraints.

### With TouchOSC Chord Guide

1. **Send chord** from TouchOSC → `/chord` OSC message
2. **HarmonicContextManager** parses chord (e.g., "Dmaj7" → root=2, scale=major)
3. **MeldController** sends:
   - CC 27 = 2 (D)
   - CC 28 = 0 (major scale mapping)
4. **Meld filters** resonate at D major scale tones
5. **Override timeout** (30s default) → reverts to chromatic

**Result**: Harmonic resonance tracking without destroying MPE pitch nuance.

---

## Testing & Validation

### 1. Verify CC Traffic

Use MIDI Monitor (macOS) or similar:

```bash
# Watch for CCs 20-30 on channels 1-2
# Frequency: ~20 Hz (50ms interval)
```

### 2. Test Probabilistic Routing

Enable debug output:

```bash
python MusicHal_9000.py --enable-meld --debug-decisions
```

Watch for: "⚡ Alternative mapping used (probabilistic accident)" (~20% of events)

### 3. Validate Scale-Aware Filters

1. Send TouchOSC chord: "Cmaj7"
2. Check CC 27 = 0 (C), CC 28 = 0 (major)
3. Play dissonant input → filters should emphasize C major tones

### 4. Performance Stats

After session:

```python
stats = drift_ai.meld_controller.get_stats()
print(stats)
# {'total_cc_sent': 3421, 
#  'throttled_cc': 12054, 
#  'alternative_mappings': 684,
#  'scale_updates': 12}
```

**Healthy ratios**:
- Throttled > total (good - prevents MIDI congestion)
- Alternative ~20% of total (matches config)
- Scale updates low (chord changes are infrequent)

---

## Artistic Research Context

### Why Meld?

- **Macro-based control**: 4 macros (not 40 CCs) suits perceptual feature mapping
- **MPE optimization**: Works alongside MPE pitch bend/pressure
- **Scale-aware filters**: Harmonic tracking without pitch quantization
- **"Designed for accidents"**: Probabilistic routing aligns with Meld's philosophy
- **Bi-timbral**: Engine A/B mirrors melodic/bass voice separation

### Musical Intent

**Traditional synthesis**: Human manually sculpts parameters → sound design

**MusicHal + Meld**: AI listening actively sculpts synthesis → perceptual partner

**Result**: Timbre evolves with harmonic/spectral/rhythmic context, creating "synthesis that listens back."

### Research Questions

1. **Coherence vs Surprise**: How does `accident_probability` affect musical trust?
2. **Scale-Aware Resonance**: Do harmonic filters enhance or constrain MPE expression?
3. **Feature-Parameter Coupling**: Which features create the most musical synth control?

Adjust `config/meld_mapping.yaml` to explore these questions through performance.

---

## Troubleshooting

### No CC Messages Received

**Check**:
- MIDI port name matches: `IAC Meld`
- Meld track input = same MIDI port
- `--enable-meld` flag is set

**Fix**: Verify with `python midi_io/meld_controller.py` (test/demo mode)

### Scale-Aware Not Working

**Check**:
- CCs 26-28 MIDI Learn completed in Meld
- TouchOSC sending `/chord` messages (port 5007)
- HarmonicContextManager receiving chords (debug output)

**Fix**: Test with `python listener/harmonic_context_manager.py` standalone

### Too Much CC Traffic

**Symptom**: MIDI glitches, note timing jitter

**Fix**: Increase throttling in config:

```yaml
throttling:
  update_interval_ms: 100  # Slower (was 50)
  change_threshold: 0.10   # Larger threshold (was 0.05)
```

### Inverted Filter Behavior

**Symptom**: Noise sounds resonant, tones sound dull

**Check**: `filter.resonance.invert: true` in config

**Explanation**: High spectral_flatness (noise) + high Q = harsh ringing. Inversion prevents this.

---

## Future Enhancements

### Optional: Performance Arc Root Extraction

Currently `root_hint_frequency` in Performance Arc JSON is null. Could be populated during training:

```bash
python Chandra_trainer.py --file audio.wav --output model.json
# Enhance: Extract dominant chroma per phase → populate root_hint_frequency
```

**Benefit**: Performance Arc provides harmonic guidance when TouchOSC not active.

### Optional: LFO Rate Coupling

Add CC 31 for global LFO rate driven by `modulation_depth`:

```yaml
lfo:
  rate: {feature: modulation_depth, cc_number: 31, scaling: exponential}
```

**Benefit**: Rhythmic activity drives synthesis modulation rate.

---

## Integration Summary

**Files Created**:
- `config/meld_mapping.yaml` - Complete configuration
- `mapping/meld_mapper.py` - Feature extraction + probabilistic routing
- `midi_io/meld_controller.py` - CC sending + throttling + scale-aware

**Files Modified**:
- `scripts/performance/MusicHal_9000.py` - Integration hooks (3 note sending locations)

**Command-Line**:
- `--enable-meld` - Enable Meld integration

**Testing**:
- `python mapping/meld_mapper.py` - Test mapper
- `python midi_io/meld_controller.py` - Test controller

---

## References

**Ableton Meld**: https://www.ableton.com/en/packs/meld/
**MERT-v1-95M**: Music-aware transformer for feature extraction
**MPE Standard**: MIDI Polyphonic Expression (per-note pitch/pressure/timbre)
**TouchOSC**: Manual chord override input system

---

**Research Context**: This integration extends MusicHal's "Player" functionality from simple note triggers to perceptually-driven synthesis control, creating a system where AI listening directly sculpts timbre in real-time. Probabilistic routing introduces controlled unpredictability, aligning with artistic research goals of trust through transparency and coherent personality.
