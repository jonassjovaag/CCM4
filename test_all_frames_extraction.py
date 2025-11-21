#!/usr/bin/env python3
"""
Test if extracting ALL MERT frames (not averaging) gives better token diversity
"""

import numpy as np
import librosa
import joblib
import torch
from listener.wav2vec_perception import Wav2VecMusicEncoder

# Load vocabulary
vocab = joblib.load('input_audio/General_idea_harmonic_vocab.joblib')
# Vocabulary contains: kmeans, pca, scaler, etc. (not a single quantizer object)

# Initialize MERT
encoder = Wav2VecMusicEncoder(model_name="m-a-p/MERT-v1-95M", use_gpu=False)
encoder._initialize_model()

# Load audio
audio, sr = librosa.load('input_audio/General_idea.wav', sr=44100, duration=60, offset=10)

print(f"🎵 Testing ALL FRAMES extraction (no averaging)")
print(f"Audio: {len(audio)/sr:.1f}s at {sr}Hz")
print("=" * 80)

# Segment audio (350ms, no overlap)
segment_dur = 0.35
hop = 0.35

tokens_all_frames = []

for start_sec in np.arange(0, len(audio)/sr - segment_dur, hop):
    start = int(start_sec * sr)
    end = int((start_sec + segment_dur) * sr)
    segment = audio[start:end]
    
    # Resample to 24kHz
    segment_24k = librosa.resample(segment, orig_sr=sr, target_sr=24000)
    sr_24k = 24000
    
    # Normalize
    if np.abs(segment_24k).max() > 0:
        segment_24k = segment_24k / np.abs(segment_24k).max()
    
    # Process with MERT
    inputs = encoder.processor(
        segment_24k, 
        sampling_rate=sr_24k, 
        return_tensors="pt",
        padding=True
    )
    
    # Extract features
    with torch.no_grad():
        outputs = encoder.model(**inputs)
    
    # Get hidden states - shape [batch=1, time_steps, hidden_dim=768]
    hidden_states = outputs.last_hidden_state
    
    # Extract ALL frames (not averaged!)
    # hidden_states[0] removes batch dimension → [time_steps, 768]
    all_frames = hidden_states[0].cpu().numpy()  # Shape: [time_steps, 768]
    
    # Quantize EACH frame separately
    for frame_features in all_frames:
        # Apply same preprocessing as training (L2 norm → PCA → KMeans)
        from sklearn.preprocessing import normalize
        
        # Reshape for sklearn
        features_2d = frame_features.reshape(1, -1)
        
        # L2 normalization
        features_normalized = normalize(features_2d, norm='l2', axis=1)
        
        # PCA transform
        pca = vocab['pca']
        features_pca = pca.transform(features_normalized)
        
        # KMeans predict
        kmeans = vocab['kmeans']
        token = int(kmeans.predict(features_pca)[0])
        
        tokens_all_frames.append(token)

print(f"✅ Extracted {len(tokens_all_frames)} tokens (ALL frames, no averaging)")

# Analyze diversity
from collections import Counter
token_counts = Counter(tokens_all_frames)
unique_tokens = len(token_counts)

print(f"\n📊 Results:")
print(f"Unique tokens: {unique_tokens}/64 ({100*unique_tokens/64:.1f}%)")
print(f"\nTop 10 tokens:")
for token_id, count in token_counts.most_common(10):
    print(f"  Token {token_id}: {count} times ({100*count/len(tokens_all_frames):.1f}%)")

top_3_pct = sum(c for _, c in token_counts.most_common(3)) / len(tokens_all_frames) * 100
print(f"\nTop 3 tokens: {top_3_pct:.1f}%")

print(f"\n" + "=" * 80)
print(f"📈 Comparison:")
print(f"Training:            64/64 tokens (100%), top token 3.5%")
print(f"Live (averaged):     23/64 tokens (35.9%), top token 13.5%")
print(f"Live (all frames):   {unique_tokens}/64 tokens ({100*unique_tokens/64:.1f}%), top token {100*token_counts.most_common(1)[0][1]/len(tokens_all_frames):.1f}%")

if unique_tokens > 50:
    print(f"\n✅ SUCCESS! All-frames extraction matches training diversity!")
elif unique_tokens > 40:
    print(f"\n⚠️ BETTER but not perfect. Closer to training.")
else:
    print(f"\n❌ Still poor diversity. Hypothesis incorrect.")
