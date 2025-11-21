#!/usr/bin/env python3
"""Test with dense overlapping windows to match training sampling density"""

import librosa
import numpy as np
from listener.dual_perception import DualPerceptionModule
from collections import Counter

print('TESTING WITH DENSE SAMPLING (350ms window, ~7ms hop)')
print('='*80)

perception = DualPerceptionModule(
    vocabulary_size=64,
    wav2vec_model='m-a-p/MERT-v1-95M',
    use_gpu=True,
    enable_symbolic=True,
    enable_dual_vocabulary=True,
    enable_wav2vec=True,
    gesture_window=0.0
)
perception.load_vocabulary('input_audio/General_idea_harmonic_vocab.joblib', vocabulary_type='harmonic')

audio, sr = librosa.load('input_audio/General_idea.wav', sr=44100, duration=10)

tokens = []
segment_dur = 0.35  # 350ms window
hop = 0.007  # 7ms hop (98% overlap)

print(f'Extracting with 350ms window, 7ms hop (~98% overlap)...')
print(f'This will generate ~{int(10.0/hop)} segments from 10s of audio\n')

count = 0
for start_sec in np.arange(0, len(audio)/sr - segment_dur, hop):
    start = int(start_sec * sr)
    segment = audio[start:start + int(sr * segment_dur)]
    
    result = perception.extract_features(audio=segment, sr=int(sr), timestamp=start_sec, detected_f0=None)
    if result.gesture_token is not None:
        tokens.append(result.gesture_token)
    
    count += 1
    if count >= 1000:  # Limit for speed
        break

print(f'Extracted {len(tokens)} tokens')
print(f'Unique tokens: {len(set(tokens))}/64 ({len(set(tokens))/64:.1%})\n')

token_counts = Counter(tokens)
print('Top 10 tokens:')
for token, count in token_counts.most_common(10):
    print(f'  Token {token}: {count} times ({count/len(tokens)*100:.1f}%)')

top_token_pct = max(token_counts.values()) / len(tokens) * 100

print(f'\nTraining:     64/64 tokens (100%), top token 3.5%')
print(f'Live (dense): {len(set(tokens))}/64 tokens ({len(set(tokens))/64:.1%}), top token {top_token_pct:.1f}%')

if len(set(tokens)) > 50:
    print('\n✅ DENSE SAMPLING WAS THE KEY!')
    print('   Training used 350ms windows with ~7ms hop (98% overlap)')
else:
    print(f'\n⚠️  Better but still not matching training')

print('='*80)
