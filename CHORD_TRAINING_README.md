# 🎓 ML Training System for Chord Detection

## Overview

This system implements **supervised learning for chord detection** using your piano playing as ground truth training data. Instead of relying on generic chord detection algorithms, we train a model specifically on your playing style and piano.

## 🎯 The Problem

The current chord detection has low accuracy (0.15-0.19 confidence) because:
- Generic algorithms don't account for your specific piano
- Room acoustics affect detection
- Playing style varies between musicians
- Different pianos have different harmonic characteristics

## 🧠 The Solution: ML Training

### How It Works

1. **Data Collection**: You play chords and label them in real-time
2. **Feature Extraction**: System captures audio features (chroma vectors, harmonic analysis)
3. **Model Training**: Machine learning model learns the mapping: `audio features → chord labels`
4. **Improved Detection**: Trained model provides much higher accuracy for your specific setup

### Training Process

```
🎹 Play C chord → Press '1' → System learns: "This audio pattern = C chord"
🎹 Play F chord → Press '6' → System learns: "This audio pattern = F chord"
🎹 Play Gm chord → Press 'i' → System learns: "This audio pattern = Gm chord"
```

## 🚀 Usage

### 1. Start Training Session

```bash
python test_interactive_chord_trainer.py
```

### 2. Label Chords

Play chords on your piano and press corresponding keys:

**Major Chords:**
- `1` = C, `2` = C#, `3` = D, `4` = D#, `5` = E, `6` = F
- `7` = F#, `8` = G, `9` = G#, `0` = A, `-` = A#, `=` = B

**Minor Chords:**
- `q` = Cm, `w` = C#m, `e` = Dm, `r` = D#m, `t` = Em, `y` = Fm
- `u` = F#m, `i` = Gm, `o` = G#m, `p` = Am, `[` = A#m, `]` = Bm

**Special:**
- `space` = Silence
- `t` = Train model
- `q` = Quit

### 3. Train Model

After collecting samples, press `t` to train the model.

### 4. Test Accuracy

The trained model will be much more accurate for your specific piano and playing style.

## 📊 Expected Results

### Before Training
- Confidence: 0.15-0.19 (unreliable)
- Wrong chord detections (A# instead of Bb)
- Inconsistent results

### After Training
- Confidence: 0.7-0.9+ (reliable)
- Accurate chord detection for your piano
- Consistent with your playing style

## 🔧 Technical Details

### Features Extracted
- **Chroma Vector**: 12-dimensional pitch class energy
- **Harmonic Features**: Confidence, stability, RMS, F0
- **Context**: Chord history, key detection

### ML Algorithm
- **Random Forest**: Robust, handles non-linear relationships
- **Feature Scaling**: Normalizes features for better training
- **Cross-validation**: Ensures model generalizes well

### Model Storage
- `models/chord_model.pkl`: Trained model
- `models/chord_scaler.pkl`: Feature scaler
- `models/chord_metadata.json`: Training metadata

## 🎵 Integration with MusicHal_9000

Once trained, the model can be integrated into `MusicHal_9000.py`:

```python
# Load trained model
model = joblib.load('models/chord_model.pkl')
scaler = joblib.load('models/chord_scaler.pkl')

# Use for chord detection
features = extract_features(audio_buffer)
features_scaled = scaler.transform([features])
predicted_chord = model.predict(features_scaled)[0]
```

## 🎯 Benefits

1. **Personalized**: Trained specifically on your piano and playing
2. **Accurate**: Much higher confidence scores
3. **Adaptive**: Learns your specific harmonic characteristics
4. **Fast**: Real-time prediction after training
5. **Improving**: Gets better with more training data

## 📈 Training Tips

1. **Play Clearly**: Hold chords for 2-3 seconds
2. **Consistent Style**: Play in your normal style
3. **Multiple Samples**: Collect 10-20 samples per chord
4. **Room Conditions**: Train in the same room you'll use
5. **Piano Settings**: Use the same piano settings

## 🔄 Iterative Improvement

The system supports iterative training:
1. Train initial model
2. Test accuracy
3. Collect more samples for problematic chords
4. Retrain model
5. Repeat until satisfied

This creates a **continuously improving chord detection system** that adapts to your specific musical environment!


