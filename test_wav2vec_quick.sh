#!/bin/bash
# Quick test script for Wav2Vec integration
# Run this to verify Wav2Vec is working correctly

set -e  # Exit on error

echo "🧪 Wav2Vec Integration Quick Test"
echo "=================================="
echo ""

# Check if transformers is installed
echo "📦 Checking dependencies..."
python3 -c "import transformers; import torch; print('✅ transformers and torch installed')" || {
    echo "❌ Missing dependencies. Install with:"
    echo "   pip install transformers torch"
    exit 1
}

# Test 1: Wav2Vec module standalone
echo ""
echo "🔍 Test 1: Testing Wav2Vec module..."
python3 -c "
from listener.wav2vec_perception import is_wav2vec_available, Wav2VecMusicEncoder
import numpy as np

if not is_wav2vec_available():
    print('❌ Wav2Vec dependencies not available')
    exit(1)

print('✅ Wav2Vec available')

# Create encoder (CPU mode for quick test)
encoder = Wav2VecMusicEncoder(use_gpu=False)

# Test with synthetic audio
sr = 44100
duration = 1.0
t = np.linspace(0, duration, int(sr * duration))
audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)

print('🎵 Encoding test audio (1 second, 440Hz sine wave)...')
result = encoder.encode(audio, sr, timestamp=0.0)

if result:
    print(f'✅ Encoding successful!')
    print(f'   Feature dimension: {result.features.shape}')
    print(f'   Feature range: [{result.features.min():.3f}, {result.features.max():.3f}]')
else:
    print('❌ Encoding failed')
    exit(1)
" || exit 1

# Test 2: Hybrid perception with Wav2Vec
echo ""
echo "🔍 Test 2: Testing HybridPerceptionModule with Wav2Vec..."
python3 -c "
from listener.hybrid_perception import HybridPerceptionModule
import numpy as np

# Create module with Wav2Vec enabled
perception = HybridPerceptionModule(
    vocabulary_size=64,
    enable_ratio_analysis=False,
    enable_symbolic=False,
    enable_wav2vec=True,
    wav2vec_model='facebook/wav2vec2-base',
    use_gpu=False
)

# Test with C major chord
sr = 44100
duration = 1.0
t = np.linspace(0, duration, int(sr * duration))
audio = (
    np.sin(2 * np.pi * 261.63 * t) +  # C4
    np.sin(2 * np.pi * 329.63 * t) +  # E4
    np.sin(2 * np.pi * 392.00 * t)    # G4
) / 3.0

print('🎵 Extracting features from C major chord...')
result = perception.extract_features(audio.astype(np.float32), sr, timestamp=0.0)

if result:
    print(f'✅ Feature extraction successful!')
    print(f'   Feature dimension: {result.features.shape[0]}D')
    print(f'   Consonance: {result.consonance:.3f}')
    print(f'   Active pitch classes: {result.active_pitch_classes}')
else:
    print('❌ Feature extraction failed')
    exit(1)
" || exit 1

# Test 3: Compare with traditional features
echo ""
echo "🔍 Test 3: Comparing Wav2Vec vs Ratio+Chroma..."
python3 -c "
from listener.hybrid_perception import HybridPerceptionModule
import numpy as np
import time

# Create both modules
print('Initializing ratio+chroma module...')
ratio_module = HybridPerceptionModule(
    vocabulary_size=64,
    enable_ratio_analysis=True,
    enable_symbolic=False,
    enable_wav2vec=False
)

print('Initializing Wav2Vec module...')
wav2vec_module = HybridPerceptionModule(
    vocabulary_size=64,
    enable_ratio_analysis=False,
    enable_symbolic=False,
    enable_wav2vec=True,
    wav2vec_model='facebook/wav2vec2-base',
    use_gpu=False
)

# Test audio
sr = 44100
duration = 0.5
t = np.linspace(0, duration, int(sr * duration))
audio = (
    np.sin(2 * np.pi * 261.63 * t) +  # C4
    np.sin(2 * np.pi * 329.63 * t) +  # E4
    np.sin(2 * np.pi * 392.00 * t)    # G4
) / 3.0
audio = audio.astype(np.float32)

# Time both
print('\n⏱️  Speed comparison (500ms audio segment):')

start = time.time()
ratio_result = ratio_module.extract_features(audio, sr, timestamp=0.0)
ratio_time = time.time() - start

start = time.time()
wav2vec_result = wav2vec_module.extract_features(audio, sr, timestamp=0.0)
wav2vec_time = time.time() - start

print(f'Ratio+Chroma: {ratio_time*1000:.2f}ms ({ratio_result.features.shape[0]}D features)')
print(f'Wav2Vec:      {wav2vec_time*1000:.2f}ms ({wav2vec_result.features.shape[0]}D features)')

speedup = wav2vec_time / ratio_time
if speedup > 1:
    print(f'✅ Ratio+chroma is {speedup:.1f}x faster')
else:
    print(f'⚠️  Wav2Vec is {1/speedup:.1f}x faster')

print(f'\n📊 Feature comparison:')
print(f'Ratio consonance:   {ratio_result.consonance:.3f}')
print(f'Wav2Vec consonance: {wav2vec_result.consonance:.3f}')
print(f'Difference:         {abs(ratio_result.consonance - wav2vec_result.consonance):.3f}')
" || exit 1

echo ""
echo "=================================="
echo "✅ All tests passed!"
echo ""
echo "Next steps:"
echo "1. Train with Wav2Vec:"
echo "   python Chandra_trainer.py --file audio.wav --hybrid-perception --wav2vec --gpu"
echo ""
echo "2. Run comparison test:"
echo "   python test_wav2vec_comparison.py --file audio.wav --gpu"
echo ""
echo "3. Use in real-time:"
echo "   python MusicHal_9000.py --hybrid-perception --wav2vec --gpu"
echo ""





























