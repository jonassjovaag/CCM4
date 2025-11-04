#!/bin/bash
# Test three approaches to understand what preserves token diversity

echo "🧪 Testing Three Gesture Processing Approaches"
echo "=============================================="
echo ""

# Test 1: Raw data (no smoothing, no gestures)
echo "=== TEST 1: Raw Data (No Temporal Smoothing, No Gestures) ==="
echo "Testing baseline token diversity with raw onset events"
python Chandra_trainer.py \
    --file "input_audio/Nineteen.wav" \
    --output "JSON/test_raw_data.json" \
    --max-events 5000 \
    --no-temporal-smoothing \
    --no-musical-gestures \
    --no-hierarchical \
    --no-rhythmic \
    --no-gpt-oss 2>&1 | grep "DEBUG - Token Assignment"

echo ""
echo "---"
echo ""

# Test 2: Peak consolidation
echo "=== TEST 2: Peak Consolidation (Select Most Salient Moment) ==="
echo "Testing if selecting peak feature magnitude preserves diversity"
python Chandra_trainer.py \
    --file "input_audio/Nineteen.wav" \
    --output "JSON/test_peak_consolidation.json" \
    --max-events 5000 \
    --no-temporal-smoothing \
    --use-musical-gestures \
    --gesture-consolidation peak \
    --no-hierarchical \
    --no-rhythmic \
    --no-gpt-oss 2>&1 | grep "DEBUG - Token Assignment"

echo ""
echo "---"
echo ""

# Test 3: First consolidation
echo "=== TEST 3: First Consolidation (Select Gesture Onset) ==="
echo "Testing if selecting first event in gesture preserves diversity"
python Chandra_trainer.py \
    --file "input_audio/Nineteen.wav" \
    --output "JSON/test_first_consolidation.json" \
    --max-events 5000 \
    --no-temporal-smoothing \
    --use-musical-gestures \
    --gesture-consolidation first \
    --no-hierarchical \
    --no-rhythmic \
    --no-gpt-oss 2>&1 | grep "DEBUG - Token Assignment"

echo ""
echo "=============================================="
echo "✅ Tests complete! Analyzing results..."
echo ""

# Run analysis on each
echo "📊 Analysis Results:"
echo ""
python analyze_gesture_training_data.py
